# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import multi_token_prediction as mtp
from megatron.core.transformer.multi_token_prediction import (
    MTPLossLoggingHelper,
    process_mtp_loss,
    roll_tensor,
)
from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.mark.parametrize("sp_sharded_mask", [False, True])
@pytest.mark.parametrize("runtime_cp_override", [False, True])
def test_precomputed_mtp_mask_uses_embedding_shard_and_runtime_cp(
    monkeypatch, sp_sharded_mask, runtime_cp_override
):
    from types import SimpleNamespace

    tp_group, base_cp_group, runtime_cp_group = object(), object(), object()
    if not runtime_cp_override:
        runtime_cp_group = base_cp_group
    full_mask = torch.tensor([[False, False, False, True]], device="cpu")
    padding_mask = full_mask[:, :2].clone() if sp_sharded_mask else full_mask.clone()
    original_mask = padding_mask.clone()
    decoder_input = torch.zeros(2, 1, 4, device="cpu")
    scatter_calls, roll_calls, layer_masks = [], [], []

    def scatter(mask, group):
        assert group is tp_group and mask.shape == (4, 1)
        scatter_calls.append(mask.clone())
        return mask[:2]

    def roll(tensor, *, sp_group, cp_group, **kwargs):
        assert sp_group is tp_group and cp_group is runtime_cp_group
        assert tensor.shape[0] == decoder_input.shape[0]
        roll_calls.append(tensor.clone())
        return tensor.clone(), None

    def layer(**kwargs):
        layer_masks.append(kwargs["padding_mask"].clone())
        return (
            kwargs["hidden_states"],
            kwargs["input_ids"],
            kwargs["position_ids"],
            kwargs["padding_mask"],
            kwargs["mtp_input_mask"],
        )

    block = SimpleNamespace(
        config=SimpleNamespace(mtp_detach_heads=False, mtp_hsm=False, mtp_num_layers=2),
        cp_group=base_cp_group,
        tp_group=tp_group,
        sequence_parallel=True,
        vp_stage=None,
        pp_rank=0,
        training=True,
        mtp_use_repeated_layer=False,
        layers=[layer, layer],
    )
    monkeypatch.setattr(mtp, "resolve_cp_group", lambda group, params: runtime_cp_group)
    monkeypatch.setattr(mtp, "get_mtp_layer_offset", lambda *args, **kwargs: 0)
    monkeypatch.setattr(mtp, "scatter_to_sequence_parallel_region", scatter)
    monkeypatch.setattr(mtp, "roll_tensor_precomputed_embeddings", roll)
    result = mtp.MultiTokenPredictionBlock.forward(
        block,
        input_ids=None,
        position_ids=None,
        hidden_states=decoder_input.clone(),
        attention_mask=None,
        padding_mask=padding_mask,
        decoder_input=decoder_input,
    )
    assert len(scatter_calls) == (0 if sp_sharded_mask else 1)
    assert len(roll_calls) == 4 and len(layer_masks) == 2
    for mask in layer_masks:
        torch.testing.assert_close(mask, full_mask[:, :2])
    torch.testing.assert_close(padding_mask, original_mask)
    assert result.shape == (6, 1, 4)


@pytest.mark.parametrize("sp_sharded_mask", [False, True])
def test_mtp_embedding_roll_accepts_full_or_sp_sharded_padding_mask(monkeypatch, sp_sharded_mask):
    from types import SimpleNamespace

    tp_group = object()
    layer = SimpleNamespace(
        config=SimpleNamespace(
            sequence_parallel=True, attention_cp_layout="zigzag", mtp_detach_heads=False
        ),
        cp_group=None,
        tp_group=tp_group,
    )
    tokens = torch.arange(4, device="cpu").view(1, 4)
    full_mask = torch.tensor([[False, True, False, True]], device="cpu")
    padding_mask = full_mask[:, :2].clone() if sp_sharded_mask else full_mask.clone()
    original_mask = padding_mask.clone()
    calls = []

    def gather(mask, tensor_parallel_output_grad, group):
        assert group is tp_group and tensor_parallel_output_grad is False
        assert mask.shape == (2, 1)
        calls.append("gather")
        return full_mask.transpose(0, 1)

    def scatter(mask, group):
        assert group is tp_group and mask.shape == (4, 1)
        calls.append("scatter")
        return mask[:2]

    def roll(fields, **kwargs):
        assert kwargs["sequence_fields"] == ["input_ids", "padding_mask"]
        torch.testing.assert_close(fields[0], tokens)
        torch.testing.assert_close(fields[1], full_mask)
        return [
            torch.tensor([[1, 2, 3, 0]], device="cpu"),
            torch.tensor([[True, False, True, True]], device="cpu"),
        ]

    def embedding(input_ids, position_ids):
        return torch.zeros(2, 1, 4, device="cpu")

    embedding.add_position_embedding = False
    embedding.scatter_to_sequence_parallel = True
    monkeypatch.setattr(mtp, "resolve_cp_group", lambda group, params: None)
    monkeypatch.setattr(mtp, "gather_from_sequence_parallel_region", gather)
    monkeypatch.setattr(mtp, "scatter_to_sequence_parallel_region", scatter)
    monkeypatch.setattr(mtp, "roll_tensor", roll)
    result = mtp.MultiTokenPredictionLayer._get_embeddings(
        layer,
        input_ids=tokens,
        position_ids=tokens,
        embedding=embedding,
        hidden_states=torch.zeros(2, 1, 4, device="cpu"),
        padding_mask=padding_mask,
    )
    assert calls == (["gather", "scatter"] if sp_sharded_mask else ["scatter"])
    torch.testing.assert_close(result[2], torch.tensor([[True, False]], device="cpu"))
    torch.testing.assert_close(padding_mask, original_mask)


@pytest.mark.parametrize("axis", [0, -3])
def test_packed_first_axis_roll_respects_valid_ends_and_gradients(axis):
    values = torch.arange(12, dtype=torch.float32).view(6, 1, 2).requires_grad_()
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 2, 4], dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, 3, 6], dtype=torch.int32),
    )
    rolled, total = roll_tensor(values, dims=axis, packed_seq_params=packed)
    expected = torch.tensor(
        [[2, 3], [0, 0], [0, 0], [8, 9], [0, 0], [0, 0]], dtype=torch.float32
    ).view(6, 1, 2)
    torch.testing.assert_close(rolled, expected)
    torch.testing.assert_close(total, expected.sum())
    total.backward()
    expected_grad = torch.zeros_like(values)
    expected_grad[1] = 1
    expected_grad[4] = 1
    torch.testing.assert_close(values.grad, expected_grad)


@pytest.mark.parametrize("cleanup", ["clean_loss_in_tracker", "clean_metrics_in_tracker"])
@pytest.mark.parametrize("per_token", [False, True])
def test_mtp_cleanup_clears_both_loss_schemas(monkeypatch, cleanup, per_token):
    acceptance = torch.ones(2, 2)
    group = object()
    tracker = {
        "loss_values": torch.ones(2),
        "loss_sums": torch.ones(2),
        "values": torch.ones(2),
        "acceptance_counts": acceptance,
        "correct_values": acceptance[0],
        "total_values": acceptance[1],
        "cumulative_acceptance_counts": torch.full((2, 2), 7.0),
        "acceptance_avg_group": group,
        "calculate_per_token_loss": per_token,
    }
    if per_token:
        tracker["num_tokens"] = torch.ones(2)
    monkeypatch.setattr(MTPLossLoggingHelper, "tracker", tracker)

    getattr(MTPLossLoggingHelper, cleanup)()

    for key in ("loss_values", "loss_sums", "values", "correct_values", "total_values"):
        assert torch.count_nonzero(tracker[key]) == 0
    if per_token:
        assert torch.count_nonzero(tracker["num_tokens"]) == 0
    assert tracker["acceptance_avg_group"] is group
    assert tracker["calculate_per_token_loss"] is per_token
    assert torch.all(tracker["cumulative_acceptance_counts"] == 7)


@pytest.mark.parametrize("cp_layout", ["zigzag", "contiguous"])
@pytest.mark.parametrize("derive_labels", [False, True])
@pytest.mark.parametrize("with_input_mask", [False, True])
def test_grouped_mtp_loss_rolls_preserve_conditioning_gradients(
    monkeypatch, cp_layout, derive_labels, with_input_mask
):
    config = TransformerConfig(
        num_layers=2,
        hidden_size=1,
        num_attention_heads=1,
        mtp_num_layers=2,
        mtp_loss_scaling_factor=1.0,
        attention_cp_layout=cp_layout,
        use_cpu_initialization=True,
    )
    seq_len = 5
    hidden = torch.ones(3 * seq_len, 1, 1, requires_grad=True)
    # GPU training tests may have installed a CUDA scale in this process-global slot.
    monkeypatch.setattr(mtp.MTPLossAutoScaler, "main_loss_backward_scale", hidden.new_tensor(1.0))
    tokens = torch.arange(seq_len).unsqueeze(0)
    validity = torch.tensor([[True, False, True, True, True]]) if with_input_mask else None
    calls = []

    def capture_roll(fields, **kwargs):
        calls.append([tuple(field.shape) for field in fields])
        return roll_tensor(fields, **kwargs)

    monkeypatch.setattr(mtp, "roll_tensor", capture_roll)
    output = process_mtp_loss(
        hidden_states=hidden,
        labels=None if derive_labels else tokens,
        input_ids=tokens if derive_labels else None,
        loss_mask=torch.ones(1, seq_len),
        mtp_input_mask=validity,
        output_layer=lambda value, **_kwargs: (value, None),
        output_weight=None,
        is_training=False,
        compute_language_model_loss=lambda _labels, logits: logits.squeeze(-1).transpose(0, 1),
        config=config,
    )
    output.sum().backward()

    assert len(calls) == config.mtp_num_layers + int(derive_labels)
    mask_rows = 2 if with_input_mask and cp_layout == "zigzag" else 1
    expected_fields = [(1, seq_len), (mask_rows, seq_len)]
    if with_input_mask and cp_layout == "contiguous":
        expected_fields.append((1, seq_len))
    assert calls[int(derive_labels) :] == [expected_fields] * config.mtp_num_layers

    def next_positions(value):
        return torch.cat((value[:, 1:], torch.zeros_like(value[:, :1])), dim=-1)

    mask = torch.ones(1, seq_len)
    if derive_labels:
        mask = next_positions(mask)
    cumulative = torch.ones(1, seq_len, dtype=torch.bool)
    for depth in range(config.mtp_num_layers):
        mask = next_positions(mask)
        if validity is not None:
            validity = next_positions(validity)
            cumulative = cumulative & validity
        expected = mask * cumulative
        expected = expected / expected.sum().clamp(min=1) / config.mtp_num_layers
        actual = hidden.grad[(depth + 1) * seq_len : (depth + 2) * seq_len, 0, 0]
        torch.testing.assert_close(actual, expected.squeeze(0))


def test_mtp_padding_layout_conversion_uses_activation_tp_ownership(monkeypatch):
    from types import SimpleNamespace

    from megatron.core.transformer import multi_token_prediction as mtp

    cp_group = SimpleNamespace(size=lambda: 2)
    tp_group = object()
    tp_cp_group = object()
    plan = object()
    block = SimpleNamespace(
        config=SimpleNamespace(
            _linear_cp_layout_explicit=True,
            linear_cp_layout="contiguous",
            attention_cp_layout="zigzag",
        ),
        cp_group=cp_group,
        tp_group=tp_group,
        tp_cp_group=tp_cp_group,
        sequence_parallel=True,
    )
    target_batch = {name: None for name in ("tokens", "position_ids", "labels", "loss_mask")}
    cp_batch = SimpleNamespace(
        boundary_layout="contiguous",
        thd_plan=plan,
        get_batch=lambda layout: target_batch,
        get_packed_seq_params=lambda layout: None,
    )
    seen = []

    def scatter(mask, group):
        assert group is tp_group
        assert mask.shape == (4, 1, 1)
        seen.append("scatter")
        return mask[:2]

    def convert(value, source, target, cp, sp, tp, tp_cp, thd_plan):
        assert (source, target) == ("contiguous", "zigzag")
        assert cp is cp_group and tp is tp_group and tp_cp is tp_cp_group
        assert sp is True and thd_plan is plan
        assert value.shape[0] == 2
        if value.dtype == torch.bool:
            seen.append("convert_mask")
            return value.flip(0)
        return value

    def gather(mask, group):
        assert group is tp_group
        seen.append("gather")
        # Layout conversion operates on validity, so zero-filled padding is invalid.
        return torch.cat((mask, torch.tensor([True, False]).view(2, 1, 1)))

    monkeypatch.setattr(mtp, "scatter_to_sequence_parallel_region", scatter)
    monkeypatch.setattr(mtp, "gather_from_sequence_parallel_region", gather)
    monkeypatch.setattr(mtp, "convert_cp_layout", convert)
    result = mtp.MultiTokenPredictionBlock.prepare_cp_layout(
        block,
        input_ids=None,
        position_ids=None,
        hidden_states=torch.zeros(2, 1, 8),
        decoder_input=None,
        mhc_multistream=None,
        labels=None,
        loss_mask=None,
        mtp_input_mask=None,
        packed_seq_params=None,
        cp_batch=cp_batch,
        padding_mask=torch.tensor([[False, True, True, False]]),
    )
    assert seen == ["scatter", "convert_mask", "gather"]
    torch.testing.assert_close(result.padding_mask, torch.tensor([[True, False, False, True]]))
