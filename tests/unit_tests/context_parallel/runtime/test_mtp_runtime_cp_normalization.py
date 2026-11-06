# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MTP backward must be invariant to the DP and CP partition of a logical microbatch."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import multi_token_prediction as mtp

pytestmark = pytest.mark.launch_on_gb200


@pytest.fixture(autouse=True)
def loss_scale(monkeypatch):
    """Do not leak the pipeline's class-level backward scale between tests."""
    monkeypatch.setattr(mtp.MTPLossAutoScaler, "main_loss_backward_scale", torch.tensor(1.0))


def _config(num_layers=1, per_token=True):
    return SimpleNamespace(
        mtp_num_layers=num_layers,
        mtp_loss_scaling_factor=0.3,
        calculate_per_token_loss=per_token,
        mtp_detach_heads=False,
        context_parallel_size=1,
        attention_cp_layout="zigzag",
    )


@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("cp_size", [None, 1, 2])
@pytest.mark.parametrize("runtime_metadata", [False, True])
def test_normalization_uses_runtime_counts_without_changing_logging(
    monkeypatch, per_token, cp_size, runtime_metadata
):
    """Reduce both counts, use the runtime group, and retain local logging counts."""
    static_group = SimpleNamespace(size=lambda: 1)
    runtime_group = SimpleNamespace(size=lambda: cp_size) if cp_size else None
    params = (
        PackedSeqParams(local_cp_size=cp_size, cp_group=runtime_group)
        if cp_size and runtime_metadata
        else None
    )
    reductions = []
    logged = []

    def roll(tensor, return_sum=True, cp_group=None, **kwargs):
        assert cp_group is runtime_group
        if not return_sum:
            return tensor, None
        mask = tensor.new_tensor([[1, 1, 1, 0]])
        return mask, mask.sum()

    def reduce(counts, op=None, group=None):
        assert cp_size == 2 and group is runtime_group
        assert op == torch.distributed.ReduceOp.SUM
        torch.testing.assert_close(counts, torch.tensor([4.0, 3.0]))
        counts.add_(counts.new_tensor([4, 4]))
        reductions.append(group)

    monkeypatch.setattr(mtp, "roll_tensor", roll)
    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    monkeypatch.setattr(
        mtp.MTPLossLoggingHelper,
        "save_metrics_to_tracker",
        lambda *args, **kwargs: logged.append((args[0].detach(), kwargs["num_tokens"])),
    )
    hidden = torch.ones(8, 1, 1, requires_grad=True)
    output = mtp.process_mtp_loss(
        hidden_states=hidden,
        labels=torch.zeros(1, 4, dtype=torch.long),
        loss_mask=torch.ones(1, 4),
        output_layer=lambda value, **kwargs: (value, None),
        output_weight=None,
        is_training=True,
        compute_language_model_loss=lambda labels, logits: logits.squeeze(-1).transpose(0, 1),
        config=_config(per_token=per_token),
        cp_group=static_group if params else runtime_group,
        packed_seq_params=params,
        metric_avg_group=object(),
    )
    (output.sum() * 0).backward()
    assert len(reductions) == int(per_token and cp_size == 2)
    if per_token:
        main_tokens, mtp_tokens = (8, 7) if cp_size == 2 else (4, 3)
        torch.testing.assert_close(
            hidden.grad[4:, 0, 0] / main_tokens, torch.tensor([0.3 / mtp_tokens] * 3 + [0.0])
        )
        assert logged[0][0].item() == 3
        assert logged[0][1].item() == 3
    else:
        torch.testing.assert_close(hidden.grad[4:, 0, 0], torch.tensor([0.1] * 3 + [0.0]))
        assert logged[0][0].item() == 1
        assert logged[0][1] is None


@pytest.mark.parametrize(("cp_size", "norm_size"), [(1, None), (2, None), (1, 1), (1, 2), (2, 4)])
def test_per_token_counts_reduce_over_loss_normalization_group(monkeypatch, cp_size, norm_size):
    """Sum main/MTP counts over the gradient-normalization group, defaulting to CP."""
    cp_group = SimpleNamespace(size=lambda: cp_size)
    norm_group = SimpleNamespace(size=lambda: norm_size) if norm_size else None
    count_group = cp_group if norm_group is None else norm_group
    reductions = []

    def roll(tensor, return_sum=True, **kwargs):
        if not return_sum:
            return tensor, None
        mask = tensor.new_tensor([[1, 1, 1, 0]])
        return mask, mask.sum()

    def reduce(counts, op=None, group=None):
        assert op == torch.distributed.ReduceOp.SUM
        torch.testing.assert_close(counts, torch.tensor([4.0, 3.0]))
        # The other ranks hold four main tokens but only one MTP token.
        counts.add_(counts.new_tensor([4, 1]))
        reductions.append(group)

    monkeypatch.setattr(mtp, "roll_tensor", roll)
    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    hidden = torch.ones(8, 1, 1, requires_grad=True)
    output = mtp.process_mtp_loss(
        hidden_states=hidden,
        labels=torch.zeros(1, 4, dtype=torch.long),
        loss_mask=torch.ones(1, 4),
        output_layer=lambda value, **kwargs: (value, None),
        output_weight=None,
        is_training=False,
        compute_language_model_loss=lambda labels, logits: logits.squeeze(-1).transpose(0, 1),
        config=_config(),
        cp_group=cp_group,
        loss_normalization_group=norm_group,
    )
    (output.sum() * 0).backward()
    reduced = count_group.size() > 1
    assert reductions == ([count_group] if reduced else [])
    main_tokens, mtp_tokens = (8, 4) if reduced else (4, 3)
    torch.testing.assert_close(
        hidden.grad[4:, 0, 0], torch.tensor([0.3 * main_tokens / mtp_tokens] * 3 + [0.0])
    )


@pytest.mark.parametrize("gtp_remat", [False, True])
def test_gpt_model_normalizes_mtp_over_gradient_group(monkeypatch, gtp_remat):
    """GPTModel passes the group finalize_model_grads sums the per-token divisor over."""
    from megatron.core.models.gpt import gpt_model

    dp_cp_group, dp_cp_gtp_remat_group = object(), object()
    expected_group = dp_cp_gtp_remat_group if gtp_remat else dp_cp_group
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return kwargs["hidden_states"]

    monkeypatch.setattr(gpt_model, "process_mtp_loss", capture)
    model = SimpleNamespace(
        config=SimpleNamespace(mtp_num_layers=1, use_mup=False),
        share_embeddings_and_output_weights=False,
        post_process=True,
        output_layer=None,
        training=True,
        compute_language_model_loss=None,
        _scale_logits=None,
        tp_group=None,
        pg_collection=SimpleNamespace(
            cp=None, dp_cp=dp_cp_group, dp_cp_gtp_remat=dp_cp_gtp_remat_group if gtp_remat else None
        ),
    )
    hidden = torch.ones(2, 1, 1)
    output = gpt_model.GPTModel._postprocess(
        model,
        hidden,
        input_ids=None,
        position_ids=None,
        labels=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        output_processor=lambda **kwargs: kwargs["hidden_states"],
    )
    assert output is hidden
    assert captured["loss_normalization_group"] is expected_group
    assert captured["metric_avg_group"] is expected_group


@pytest.fixture(scope="module")
def cp_groups():
    """Create groups of 1/2/4 consecutive ranks, used as CP and DP x CP groups.

    The normal unit-test runner uses NCCL on GPUs.
    """
    if int(os.environ.get("WORLD_SIZE", "1")) < 4:
        pytest.skip("DP/CP gradient parity requires at least four distributed ranks")
    if not torch.distributed.is_initialized():
        from tests.unit_tests.test_utilities import Utils

        Utils.initialize_distributed()
    world_size = torch.distributed.get_world_size()
    assert world_size % 4 == 0
    rank = torch.distributed.get_rank()
    local_groups = {}
    for cp_size in (1, 2, 4):
        for first_rank in range(0, world_size, cp_size):
            ranks = list(range(first_rank, first_rank + cp_size))
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                local_groups[cp_size] = group
    yield local_groups
    torch.distributed.barrier()
    for group in reversed(list(local_groups.values())):
        torch.distributed.destroy_process_group(group)


def _masks(case, device):
    mask = torch.ones(1, 16, device=device)
    input_mask = None
    if case == "uneven":
        mask[0] = mask.new_tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1])
    elif case == "empty_rank":
        # CP2 rank 0 has no main tokens, but rolling brings in valid MTP targets.
        mask[0] = mask.new_tensor([0, 0, 1, 1, 1, 1, 0, 0] * 2)
    elif case == "zero_mtp":
        mask.zero_()
        mask[0, [0, 8]] = 1
    elif case == "all_zero":
        mask.zero_()
    elif case == "input_holes":
        input_mask = torch.tensor([[1, 1, 0, 1, 1, 0, 1, 1] * 2], device=device).bool()
    elif case == "zero_input":
        input_mask = torch.zeros_like(mask, dtype=torch.bool)
    return mask, input_mask


def _lane_masks(lane_cases, dp_size, device):
    """Concatenate one 16-token ``_masks`` lane per DP rank, cycling through ``lane_cases``."""
    lanes = [_masks(lane_cases[lane % len(lane_cases)], device) for lane in range(dp_size)]
    mask = torch.cat([lane_mask for lane_mask, _ in lanes], dim=1)
    if all(lane_input is None for _, lane_input in lanes):
        return mask, None
    input_mask = torch.cat(
        [
            lane_input if lane_input is not None else torch.ones_like(lane_mask, dtype=torch.bool)
            for lane_mask, lane_input in lanes
        ],
        dim=1,
    )
    return mask, input_mask


def _backward(
    cp_group,
    static_group,
    mask,
    input_mask,
    num_layers,
    derive_labels,
    runtime_metadata=True,
    dp_cp_group=None,
):
    """Use real packed rolling, cross entropy, and trainable hidden/output parameters.

    ``mask`` covers the whole batch: each DP rank owns one equal lane of packed 8-token
    sequences, split across its CP group. ``dp_cp_group`` is the gradient-normalization
    group and defaults to ``cp_group``.
    """
    device = mask.device
    norm_group = cp_group if dp_cp_group is None else dp_cp_group
    cp_size = cp_group.size()
    cp_rank = cp_group.rank()
    lane_size = mask.size(1) * cp_size // norm_group.size()
    lane = torch.arange(lane_size, device=device) + lane_size * (norm_group.rank() // cp_size)
    chunks = lane.reshape(lane_size // 8, 2 * cp_size, -1)
    indices = chunks[:, [cp_rank, 2 * cp_size - cp_rank - 1]].reshape(-1)
    cu_seqlens = torch.arange(0, lane_size + 1, 8, dtype=torch.int32, device=device)
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        local_cp_size=cp_size if runtime_metadata else None,
        cp_group=cp_group if runtime_metadata else None,
    )
    head = torch.nn.Parameter(torch.arange(20, device=device).reshape(5, 4).float() / 20)
    hidden = torch.nn.Parameter(
        torch.arange((num_layers + 1) * mask.size(1) * 4, device=device)
        .reshape(num_layers + 1, mask.size(1), 1, 4)
        .float()
        .remainder(19)
        / 19
    )
    token_ids = torch.arange(mask.size(1), device=device).remainder(5).unsqueeze(0)
    local_hidden = hidden[:, indices].reshape(-1, 1, 4)
    result = mtp.process_mtp_loss(
        hidden_states=local_hidden,
        labels=None if derive_labels else token_ids[:, indices],
        input_ids=token_ids[:, indices] if derive_labels else None,
        loss_mask=mask[:, indices],
        mtp_input_mask=input_mask[:, indices] if input_mask is not None else None,
        output_layer=lambda value, **kwargs: (F.linear(value, head), None),
        output_weight=None,
        is_training=False,
        compute_language_model_loss=lambda labels, logits: F.cross_entropy(
            logits.transpose(0, 1).reshape(-1, 5), labels.reshape(-1), reduction="none"
        ).reshape_as(labels),
        config=_config(num_layers=num_layers),
        # Exercise both a CP1-built runtime group override and the static CP path.
        cp_group=static_group if runtime_metadata else cp_group,
        packed_seq_params=params,
        loss_normalization_group=dp_cp_group,
    )
    (result.sum() * 0).backward()

    main_mask = mask
    if derive_labels:
        global_cu_seqlens = torch.arange(0, mask.size(1) + 1, 8, dtype=torch.int32, device=device)
        main_mask, _ = mtp.roll_tensor(
            mask, packed_seq_params=PackedSeqParams(cu_seqlens_q=global_cu_seqlens)
        )
    num_tokens = main_mask[:, indices].sum()
    grads = [head.grad, hidden.grad]
    # Match per-token DDP SUM followed by finalize_model_grads' main-token divisor.
    for value in [*grads, num_tokens]:
        torch.distributed.all_reduce(value, group=norm_group)
    return [grad / num_tokens.clamp(min=1) for grad in grads]


@pytest.mark.parametrize(
    "case",
    ["all_valid", "uneven", "empty_rank", "zero_mtp", "all_zero", "input_holes", "zero_input"],
)
@pytest.mark.parametrize("num_layers", [1, 3])
@pytest.mark.parametrize("derive_labels", [False, True])
@pytest.mark.parametrize("runtime_metadata", [False, True])
def test_cp2_cp4_parameter_gradients_match_cp1(
    cp_groups, case, num_layers, derive_labels, runtime_metadata
):
    """Changing runtime CP size preserves parameter gradients, including empty ranks."""
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    mask, input_mask = _masks(case, device)
    reference = _backward(cp_groups[1], cp_groups[1], mask, input_mask, num_layers, derive_labels)
    if case in ("zero_mtp", "all_zero", "zero_input"):
        for grad in reference:
            assert torch.count_nonzero(grad) == 0
    for cp_size in (2, 4):
        actual = _backward(
            cp_groups[cp_size],
            cp_groups[1],
            mask,
            input_mask,
            num_layers,
            derive_labels,
            runtime_metadata=runtime_metadata,
        )
        for expected_grad, actual_grad in zip(reference, actual):
            assert torch.isfinite(actual_grad).all()
            torch.testing.assert_close(actual_grad, expected_grad, atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize(
    "lane_cases",
    [
        ("all_valid", "uneven"),
        ("uneven", "zero_mtp"),
        ("empty_rank", "input_holes"),
        ("all_zero", "uneven"),
    ],
)
@pytest.mark.parametrize("num_layers", [1, 3])
@pytest.mark.parametrize("derive_labels", [False, True])
@pytest.mark.parametrize(
    ("dp_size", "cp_size"), [(2, 1), (4, 1), (2, 2)], ids=["dp2", "dp4", "dp2cp2"]
)
def test_dp_cp_parameter_gradients_match_single_rank(
    cp_groups, lane_cases, num_layers, derive_labels, dp_size, cp_size
):
    """DP ranks with different main/MTP token ratios keep the single-rank gradients."""
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    mask, input_mask = _lane_masks(lane_cases, dp_size, device)
    reference = _backward(cp_groups[1], cp_groups[1], mask, input_mask, num_layers, derive_labels)
    actual = _backward(
        cp_groups[cp_size],
        cp_groups[1],
        mask,
        input_mask,
        num_layers,
        derive_labels,
        dp_cp_group=cp_groups[dp_size * cp_size],
    )
    for expected_grad, actual_grad in zip(reference, actual):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-7, rtol=2e-5)
