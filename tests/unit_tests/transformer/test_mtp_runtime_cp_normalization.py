# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MTP backward must be invariant to the CP partition of a logical microbatch."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import multi_token_prediction as mtp


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
        runtime_gather_output=True,
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


@pytest.fixture(scope="module")
def cp_groups():
    """Create CP1/2/4 groups; the normal unit-test runner uses NCCL on GPUs."""
    if int(os.environ.get("WORLD_SIZE", "1")) < 4:
        pytest.skip("CP1/2/4 gradient parity requires at least four distributed ranks")
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


def _backward(cp_group, static_group, mask, input_mask, num_layers, derive_labels):
    """Use real packed rolling, cross entropy, and trainable hidden/output parameters."""
    device = mask.device
    cp_size = cp_group.size()
    cp_rank = cp_group.rank()
    chunks = torch.arange(16, device=device).reshape(2, 2 * cp_size, -1)
    indices = chunks[:, [cp_rank, 2 * cp_size - cp_rank - 1]].reshape(-1)
    cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int32, device=device)
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        local_cp_size=cp_size,
        cp_group=cp_group,
    )
    head = torch.nn.Parameter(torch.arange(20, device=device).reshape(5, 4).float() / 20)
    hidden = torch.nn.Parameter(
        torch.arange((num_layers + 1) * 16 * 4, device=device)
        .reshape(num_layers + 1, 16, 1, 4)
        .float()
        .remainder(19)
        / 19
    )
    token_ids = torch.arange(16, device=device).remainder(5).unsqueeze(0)
    local_hidden = hidden[:, indices].reshape(-1, 1, 4)
    result = mtp.process_mtp_loss(
        hidden_states=local_hidden,
        labels=None if derive_labels else token_ids[:, indices],
        input_ids=token_ids[:, indices] if derive_labels else None,
        loss_mask=mask[:, indices],
        mtp_input_mask=input_mask[:, indices] if input_mask is not None else None,
        output_layer=lambda value, **kwargs: (F.linear(value, head), None),
        output_weight=None,
        runtime_gather_output=True,
        is_training=False,
        compute_language_model_loss=lambda labels, logits: F.cross_entropy(
            logits.transpose(0, 1).reshape(-1, 5), labels.reshape(-1), reduction="none"
        ).reshape_as(labels),
        config=_config(num_layers=num_layers),
        # Deliberately pass the build-time CP1 group: packed runtime metadata wins.
        cp_group=static_group,
        packed_seq_params=params,
    )
    (result.sum() * 0).backward()

    main_mask = mask
    if derive_labels:
        main_mask, _ = mtp.roll_tensor(
            mask, packed_seq_params=PackedSeqParams(cu_seqlens_q=cu_seqlens)
        )
    num_tokens = main_mask[:, indices].sum()
    grads = [head.grad, hidden.grad]
    # Match per-token DDP SUM followed by finalize_model_grads' main-token divisor.
    for value in [*grads, num_tokens]:
        torch.distributed.all_reduce(value, group=cp_group)
    return [grad / num_tokens.clamp(min=1) for grad in grads]


@pytest.mark.parametrize(
    "case",
    ["all_valid", "uneven", "empty_rank", "zero_mtp", "all_zero", "input_holes", "zero_input"],
)
@pytest.mark.parametrize("num_layers", [1, 3])
@pytest.mark.parametrize("derive_labels", [False, True])
def test_cp2_cp4_parameter_gradients_match_cp1(cp_groups, case, num_layers, derive_labels):
    """Changing runtime CP size preserves parameter gradients, including empty ranks."""
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    mask, input_mask = _masks(case, device)
    reference = _backward(cp_groups[1], cp_groups[1], mask, input_mask, num_layers, derive_labels)
    if case in ("zero_mtp", "all_zero", "zero_input"):
        for grad in reference:
            assert torch.count_nonzero(grad) == 0
    for cp_size in (2, 4):
        actual = _backward(
            cp_groups[cp_size], cp_groups[1], mask, input_mask, num_layers, derive_labels
        )
        for expected_grad, actual_grad in zip(reference, actual):
            assert torch.isfinite(actual_grad).all()
            torch.testing.assert_close(actual_grad, expected_grad, atol=2e-7, rtol=2e-5)
