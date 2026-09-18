# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import get_context_parallel_group
from megatron.core.transformer.multi_token_prediction import (
    MTPLossAutoScaler,
    MTPLossLoggingHelper,
    process_mtp_loss,
)
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def cp_group():
    """Initialize independent two-rank CP groups within the test world."""
    if Utils.world_size < 2 or Utils.world_size % 2:
        pytest.skip("MTP CP loss tests require an even world size of at least two")
    Utils.initialize_model_parallel(context_parallel_size=2)
    yield get_context_parallel_group()
    Utils.destroy_model_parallel()


@pytest.fixture
def device():
    """Use the local GPU for the distributed test."""
    return torch.device("cuda", torch.cuda.current_device())


@pytest.fixture(autouse=True)
def restore_mtp_loss_scale():
    """Do not leave the shared MTP backward scale changed for other tests."""
    previous = MTPLossAutoScaler.main_loss_backward_scale
    yield
    MTPLossAutoScaler.set_loss_scale(previous)


def _masks(device):
    positions = torch.arange(16, device=device)
    yield torch.ones(1, 16, device=device, dtype=torch.float64)
    yield torch.zeros(1, 16, device=device, dtype=torch.float64)
    for selected in [positions < 9, positions >= 11, positions.remainder(2) == 0]:
        yield selected.unsqueeze(0).to(torch.float64)
    # Includes a rank with no LM targets receiving an MTP target from its neighbor.
    for position in range(16):
        yield (positions == position).unsqueeze(0).to(torch.float64)


def _run_loss(hidden, weight, labels, mask, config, packed, group, *, training=False):
    length = labels.size(1)
    hidden = hidden.clone().reshape(-1, 1, 4).requires_grad_()
    weight = weight.clone().requires_grad_()
    logits = []

    def output_layer(states, **kwargs):
        result = F.linear(states, weight)
        result.retain_grad()
        logits.append(result)
        return result, None

    MTPLossAutoScaler.set_loss_scale(torch.ones((), device=hidden.device))
    kwargs = {}
    if training:
        kwargs["metric_avg_group"] = group
    output = process_mtp_loss(
        hidden_states=hidden,
        labels=labels,
        loss_mask=mask,
        output_layer=output_layer,
        output_weight=None,
        runtime_gather_output=True,
        is_training=training,
        compute_language_model_loss=lambda targets, values: F.cross_entropy(
            values.permute(1, 2, 0), targets, reduction="none"
        ),
        config=config,
        cp_group=group,
        packed_seq_params=packed,
        **kwargs,
    )
    # Isolate MTP's actual custom backward from the ordinary LM gradient.
    (output.sum() * 0).backward()
    cp_size = group.size() if group is not None else 1
    count = mask.sum()
    if config.calculate_per_token_loss:
        if group is not None:
            torch.distributed.all_reduce(count, group=group)
        divisor = count.clamp(min=1)
    else:
        divisor = cp_size
    if group is not None:
        torch.distributed.all_reduce(weight.grad, group=group)
    return (
        hidden.grad.reshape(config.mtp_num_layers + 1, length, 1, 4)[1:] / divisor,
        torch.stack([value.grad for value in logits]) / divisor,
        weight.grad / divisor,
    )


def _case(depth, packed, device, cp_group):
    # Different DP replicas deliberately use different examples. Only CP peers
    # should contribute to the normalization collective.
    first_cp_rank = torch.distributed.get_process_group_ranks(cp_group)[0]
    generator = torch.Generator(device=device).manual_seed(6103 + first_cp_rank)
    hidden = torch.randn(
        depth + 1, 16, 1, 4, device=device, dtype=torch.float64, generator=generator
    )
    weight = torch.randn(7, 4, device=device, dtype=torch.float64, generator=generator)
    labels = torch.randint(0, 7, (1, 16), device=device, generator=generator)
    config = SimpleNamespace(
        mtp_num_layers=depth,
        mtp_detach_heads=False,
        mtp_loss_scaling_factor=0.3,
        calculate_per_token_loss=True,
        cross_entropy_loss_fusion=False,
    )
    boundaries = [0, 8, 16] if packed else [0, 16]
    params = (
        PackedSeqParams(qkv_format="thd", cu_seqlens_q=torch.tensor(boundaries, device=device))
        if packed
        else None
    )
    rank = cp_group.rank()
    indices = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        chunk = (end - start) // 4
        indices.extend(range(start + rank * chunk, start + (rank + 1) * chunk))
        indices.extend(range(start + (3 - rank) * chunk, start + (4 - rank) * chunk))
    indices = torch.tensor(indices, device=device)
    return hidden, weight, labels, config, params, indices


@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("per_token", [False, True])
def test_mtp_cp_gradients_match_unpartitioned(cp_group, device, depth, packed, per_token):
    """CP partitioning must preserve token, hidden-state, and shared-head gradients."""
    hidden, weight, labels, config, params, indices = _case(depth, packed, device, cp_group)
    config.calculate_per_token_loss = per_token
    for mask in _masks(device):
        replica_offset = torch.distributed.get_process_group_ranks(cp_group)[0]
        mask = mask.roll(replica_offset, dims=-1)
        expected = _run_loss(hidden, weight, labels, mask, config, params, None)
        actual = _run_loss(
            hidden[:, indices],
            weight,
            labels[:, indices],
            mask[:, indices],
            config,
            params,
            cp_group,
        )
        for local, reference in zip(actual[:2], expected[:2]):
            reconstructed = torch.zeros_like(reference)
            reconstructed[:, indices] = local
            torch.distributed.all_reduce(reconstructed, group=cp_group)
            torch.testing.assert_close(reconstructed, reference, atol=1e-11, rtol=1e-9)
        torch.testing.assert_close(actual[2], expected[2], atol=1e-11, rtol=1e-9)


def test_mtp_cp_logged_loss_uses_global_count(cp_group, device, monkeypatch):
    """Averaging the rank contributions must report the same MTP mean as CP1."""
    hidden, weight, labels, config, params, indices = _case(1, False, device, cp_group)
    captured = []
    monkeypatch.setattr(
        MTPLossLoggingHelper,
        "save_metrics_to_tracker",
        lambda loss, *args, **kwargs: captured.append(loss.detach().clone()),
    )
    for mask in _masks(device):
        captured.clear()
        _run_loss(
            hidden[:, indices],
            weight,
            labels[:, indices],
            mask[:, indices],
            config,
            params,
            cp_group,
            training=True,
        )
        actual = captured[0]
        torch.distributed.all_reduce(actual, group=cp_group)
        actual /= cp_group.size()
        losses = F.cross_entropy(
            F.linear(hidden[1, :-1], weight).permute(1, 2, 0), labels[:, 1:], reduction="none"
        )
        expected = (losses * mask[:, 1:]).sum() / mask[:, 1:].sum().clamp(min=1)
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-9)
