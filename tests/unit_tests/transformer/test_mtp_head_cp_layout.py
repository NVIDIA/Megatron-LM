# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.context_parallel.utils import get_batches_on_this_cp_rank
from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer import multi_token_prediction as mtp
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data():
    """All inputs in this module are synthetic."""


@pytest.fixture(scope="module")
def model_parallel_groups(request):
    tp, cp = request.param
    if Utils.world_size < tp * cp or Utils.world_size % (tp * cp):
        pytest.skip(f"requires a world size divisible by {tp * cp}")
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
    yield (
        parallel_state.get_tensor_model_parallel_group(),
        parallel_state.get_context_parallel_group(),
        parallel_state.get_tensor_and_context_parallel_group(),
    )
    Utils.destroy_model_parallel()


def _config(tp, cp, sp, calculate_per_token_loss=True):
    return TransformerConfig(
        num_layers=2,
        hidden_size=16,
        num_attention_heads=4,
        mtp_num_layers=2,
        mtp_loss_scaling_factor=0.17,
        tensor_model_parallel_size=tp,
        context_parallel_size=cp,
        sequence_parallel=sp,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        perform_initialization=False,
        gradient_accumulation_fusion=True,
        linear_cp_layout="contiguous",
        attention_cp_layout="zigzag",
        calculate_per_token_loss=calculate_per_token_loss,
    )


def _batches(labels, mask, conditioning, cu, tp_group, cp_group, tp_cp_group, sp):
    data = {"labels": labels, "loss_mask": mask, "mtp_input_mask": conditioning}
    if cu is not None:
        data.update(cu_seqlens=cu.unsqueeze(0), max_seqlen=(cu[1:] - cu[:-1]).max().view(1))
    return get_batches_on_this_cp_rank(
        data,
        boundary_layout="contiguous",
        is_hybrid_cp=False,
        cp_group=cp_group,
        additional_layouts=("zigzag",),
        use_per_sequence_balancing=cu is not None,
        sequence_parallel=sp,
        tp_group=tp_group,
        tp_cp_group=tp_cp_group,
    )


def _head(config, vocab, tp_group):
    head = ColumnParallelLinear(
        config.hidden_size,
        vocab,
        config=config,
        init_method=lambda x: x,
        bias=False,
        gather_output=False,
        tp_group=tp_group,
        output_dtype=torch.float32,
    ).cuda()
    with torch.no_grad():
        full = torch.arange(vocab * config.hidden_size, device="cuda", dtype=torch.float32).view(
            vocab, config.hidden_size
        )
        head.weight.copy_((full.sin() * 0.1).chunk(tp_group.size(), dim=0)[tp_group.rank()])
    head.weight.main_grad = torch.zeros_like(head.weight, dtype=torch.float32)
    return head


@pytest.mark.parametrize(
    "model_parallel_groups,sp",
    [((1, 1), False), ((1, 4), False), ((2, 2), False), ((2, 2), True)],
    indirect=["model_parallel_groups"],
    scope="module",
    ids=["tp1-cp1", "tp1-cp4", "tp2-cp2", "tp2-cp2-sp"],
)
@pytest.mark.parametrize(
    "layout", ["dense", "packed_aligned", "packed_padded", "packed_without_plan"]
)
@pytest.mark.parametrize("calculate_per_token_loss", [False, True])
@pytest.mark.parametrize(
    "empty,with_mtp_input_mask",
    [(False, False), (False, True), (True, True)],
    ids=["text", "conditioning-mask", "all-masked"],
)
def test_automatic_head_compaction_matches_original(
    monkeypatch,
    model_parallel_groups,
    sp,
    layout,
    calculate_per_token_loss,
    empty,
    with_mtp_input_mask,
):
    """Preserve losses/gradients and redistribute heads only when padding can be removed."""
    tp_group, cp_group, tp_cp_group = model_parallel_groups
    tp, cp = tp_group.size(), cp_group.size()
    config = _config(tp, cp, sp, calculate_per_token_loss)
    packed = layout != "dense"
    batch, seq, vocab = (1 if packed else 2), 64, 64
    labels = torch.arange(batch * seq, device="cuda").view(batch, seq).remainder(vocab)
    mask = torch.linspace(0.2, 2.0, seq, device="cuda").repeat(batch, 1)
    mask[:, [0, 3, 7, 11, 15, 20]] = 0
    mask[:, 37:] = 0
    if empty:
        mask.zero_()
    conditioning = None
    if with_mtp_input_mask:
        conditioning = torch.ones_like(mask, dtype=torch.bool)
        conditioning[:, [2, 9, 17, 24]] = False
    cu = None
    if packed:
        boundaries = (
            [0, 16, 32, 64]
            if layout == "packed_aligned"
            else [0, 1, 3, 6, 7, 8, 9, 10, 15, 16, 17, 26, 32, 40, 48, 64]
        )
        cu = torch.tensor(boundaries, dtype=torch.int32, device="cuda")
    cp_batch = _batches(labels, mask, conditioning, cu, tp_group, cp_group, tp_cp_group, sp)
    if layout == "packed_without_plan":
        cp_batch.thd_plan = None
    local = cp_batch.get_batch("zigzag")
    local_seq = local["labels"].shape[1]
    torch.manual_seed(703 + cp_group.rank())
    base = torch.randn(3, local_seq, batch, 16, device="cuda", dtype=torch.bfloat16) * 0.1
    if sp:
        base = base.chunk(tp, dim=1)[tp_group.rank()].contiguous()
    base = base.flatten(0, 1)
    main_seq = seq // cp // (tp if sp else 1)
    main_base = torch.randn(main_seq, batch, 16, device="cuda", dtype=torch.bfloat16)
    monkeypatch.setattr(mtp.MTPLossAutoScaler, "main_loss_backward_scale", torch.tensor(0.23))
    conversions = []
    observations = []
    convert_cp_layout = mtp.convert_cp_layout

    def track_conversion(tensor, source, target, *args, **kwargs):
        conversions.append((source, target))
        return convert_cp_layout(tensor, source, target, *args, **kwargs)

    monkeypatch.setattr(mtp, "convert_cp_layout", track_conversion)
    monkeypatch.setattr(mtp, "is_observing_tensor", lambda kind: kind == "mtp_logits")
    monkeypatch.setattr(
        mtp,
        "observe_tensor",
        lambda owner, name, kind, tensor, **kwargs: observations.append((name, tensor.shape[0])),
    )

    def run(automatic_compaction):
        conversions.clear()
        observations.clear()
        hidden = base.clone().requires_grad_(True)
        main = main_base.clone().requires_grad_(True)
        head = _head(config, vocab, tp_group)
        assert head.weight.dtype == torch.bfloat16
        rows, metrics = [], []

        def check_head_input(_, args):
            assert args[0].dtype == torch.bfloat16
            rows.append(args[0].shape[0])

        hook = head.register_forward_pre_hook(check_head_input)

        def compute_loss(labels, logits):
            assert logits.dtype == torch.float32
            return fused_vocab_parallel_cross_entropy(logits, labels.T.contiguous(), tp_group).T

        def log(loss, correct, total, *args, **kwargs):
            metrics.append(
                (loss.detach().clone(), correct.detach().clone(), total.detach().clone())
            )

        monkeypatch.setattr(mtp.MTPLossLoggingHelper, "save_metrics_to_tracker", log)
        output = mtp.process_mtp_loss(
            hidden_states=hidden,
            labels=local["labels"],
            loss_mask=local["loss_mask"],
            output_layer=head,
            output_weight=None,
            runtime_gather_output=None,
            is_training=True,
            compute_language_model_loss=compute_loss,
            config=config,
            cp_group=cp_group,
            tp_group=tp_group,
            packed_seq_params=cp_batch.get_packed_seq_params("zigzag"),
            mtp_input_mask=local["mtp_input_mask"],
            main_hidden_states=main,
            metric_avg_group=cp_group,
            cp_batch=cp_batch if automatic_compaction else None,
            tp_cp_group=tp_cp_group,
        )
        output.square().sum().backward()
        hook.remove()
        should_compact = automatic_compaction and layout == "packed_padded" and cp > 1
        expected_seq = main_seq if should_compact else local_seq // (tp if sp else 1)
        assert rows == [expected_seq] * 2
        observation_suffix = ".contiguous" if should_compact else ""
        assert observations == [
            (f"mtp_logits.{depth}{observation_suffix}", expected_seq * (tp if sp else 1))
            for depth in range(config.mtp_num_layers)
        ]
        if should_compact:
            assert expected_seq < local_seq // (tp if sp else 1)
            assert ("zigzag", "contiguous") in conversions
            assert ("contiguous", "zigzag") in conversions
        else:
            # Equal row counts alone would not catch unnecessary all-to-all communication.
            assert conversions == []
        # Projection work moves between CP ranks. Their replicated weight gradient
        # is equivalent after the same CP sum used by training gradient finalization.
        weight_grad = head.weight.main_grad.clone()
        torch.distributed.all_reduce(weight_grad, group=cp_group)
        for metric in metrics:
            for index in (1, 2):
                torch.distributed.all_reduce(metric[index], group=cp_group)
        return output.detach(), hidden.grad, main.grad, weight_grad, metrics

    reference, actual = run(False), run(True)
    torch.testing.assert_close(actual, reference, rtol=0.025, atol=3e-5)
