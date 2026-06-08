# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for combined Tensor Parallelism + Generalized Tensor Parallelism (TP+GTP).

Process group layout (world_size = tp_size x gtp_size):

    rank = gtp_rank x tp_size + tp_rank

    TP  group: all ranks that share the same gtp_rank  (size = tp_size)
    GTP group: all ranks that share the same tp_rank   (size = gtp_size)

Test groups
-----------
1.  TestTPGTPProcessGroups        - verify TP/GTP group sizes and rank assignment
2.  TestTPGTPColumnParallelLinear - column-parallel Linear: fwd/bwd correctness (weight shape verified inline)
3.  TestTPGTPRowParallelLinear    - row-parallel Linear: fwd/bwd smoke test + numerical correctness
4.  TestTPGTPLayerNormLinear      - LayerNormLinear column-parallel smoke test

Tests use (tp_size, gtp_size) = (2, 2) → world_size = 4 (runs on 4-GPU machines).

Multi-GPU tests skip automatically when ``torch.distributed.get_world_size()`` does not match
the requested combination of tp_size x gtp_size.
"""

import pytest
import torch
import torch.distributed as dist

from megatron.experimental.gtp import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.17", allow_module_level=True)

import transformer_engine.pytorch as te

from megatron.experimental.gtp import GTPShardedParam
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (
    _make_gtp_linear,
    _requires_multi_gpu,
    _run_distributed,
    _torchrun_dist_init,
    reset_fp8_state,
    reset_gtp_globals,
)


def _build_groups(rank: int, world_size: int, tp_size: int, gtp_size: int):
    """Create TP and GTP process groups for a 2D parallelism grid.

    Layout: rank = gtp_rank x tp_size + tp_rank
      TP  group: contiguous block [gtp_rank*tp_size, (gtp_rank+1)*tp_size)
      GTP group: strided set      {tp_rank, tp_rank+tp_size, tp_rank+2*tp_size, ...}

    Every rank must call new_group for ALL groups (PyTorch distributed requirement).

    Returns:
        tp_group:  this rank's TP process group
        gtp_group: this rank's GTP process group
        tp_rank:   this rank's index within its TP group
        gtp_rank:  this rank's index within its GTP group
    """
    assert tp_size * gtp_size == world_size
    tp_rank = rank % tp_size
    gtp_rank = rank // tp_size

    tp_group = None
    for er in range(gtp_size):
        ranks = list(range(er * tp_size, (er + 1) * tp_size))
        grp = dist.new_group(ranks)
        if er == gtp_rank:
            tp_group = grp

    gtp_group = None
    for tr in range(tp_size):
        ranks = list(range(tr, world_size, tp_size))
        grp = dist.new_group(ranks)
        if tr == tp_rank:
            gtp_group = grp

    return tp_group, gtp_group, tp_rank, gtp_rank


# ---------------------------------------------------------------------------
# 1. TestTPGTPProcessGroups - group sizes and rank membership
# ---------------------------------------------------------------------------


def _worker_groups(rank, world_size, port, tp_size, gtp_size):
    tp_group, gtp_group, tp_rank, gtp_rank = _build_groups(rank, world_size, tp_size, gtp_size)

    assert tp_group.size() == tp_size, f"rank {rank}: TP group size {tp_group.size()} != {tp_size}"
    assert (
        gtp_group.size() == gtp_size
    ), f"rank {rank}: GTP group size {gtp_group.size()} != {gtp_size}"
    assert (
        dist.get_rank(tp_group) == tp_rank
    ), f"rank {rank}: TP rank {dist.get_rank(tp_group)} != expected {tp_rank}"
    assert (
        dist.get_rank(gtp_group) == gtp_rank
    ), f"rank {rank}: GTP rank {dist.get_rank(gtp_group)} != expected {gtp_rank}"


class TestTPGTPProcessGroups:
    @pytest.mark.parametrize("tp_size,gtp_size", [(2, 2)])
    def test_group_sizes_and_ranks(self, tp_size, gtp_size):
        world_size = tp_size * gtp_size
        _requires_multi_gpu(world_size)
        _run_distributed(_worker_groups, world_size, tp_size, gtp_size)


# ---------------------------------------------------------------------------
# 2. TestTPGTPColumnParallelLinear
# ---------------------------------------------------------------------------


def _worker_column_correctness(rank, world_size, port, tp_size, gtp_size):
    """Column-parallel output must equal inp @ (GTP-gathered TP-local weight)^T."""
    torch.manual_seed(0)
    tp_group, gtp_group, tp_rank, gtp_rank = _build_groups(rank, world_size, tp_size, gtp_size)

    batch, in_f = 16, 64
    out_f = tp_size * gtp_size * 32  # per-rank shard = 32 rows
    dtype = torch.bfloat16

    layer = _make_gtp_linear(in_f, out_f, gtp_group, dtype, parallel_mode="column", tp_group=tp_group)

    # All-gather GTP shards → TP-local full weight [out_f/tp_size, in_f]
    shard = layer.weight.data.clone()
    all_gtp_shards = [torch.zeros_like(shard) for _ in range(gtp_size)]
    dist.all_gather(all_gtp_shards, shard, group=gtp_group)
    tp_local_weight = torch.cat(all_gtp_shards, dim=0).float()  # strip padding
    tp_local_weight = tp_local_weight[: out_f // tp_size]

    # Same full input on all ranks (column-parallel: each rank processes full input)
    inp = torch.randn(batch, in_f, dtype=dtype, device="cuda")
    dist.broadcast(inp, src=0)
    inp_te = inp.clone().requires_grad_(True)

    # TE forward: GTP all-gathers weight internally; no TP comm in column-parallel fwd
    out = layer(inp_te, is_first_microbatch=True)
    assert out.shape == (
        batch,
        out_f // tp_size,
    ), f"rank {rank}: output shape {out.shape} != ({batch}, {out_f // tp_size})"

    # Reference: this TP rank's output = inp @ tp_local_weight^T
    ref = inp.float() @ tp_local_weight.T
    ref = ref.to(dtype)
    assert torch.allclose(
        out.float(), ref.float(), atol=1e-2, rtol=1e-2
    ), f"rank {rank}: output mismatch, max_diff={(out.float() - ref.float()).abs().max():.4f}"

    # Backward: dX is all-reduced across TP group internally by TE
    grad = torch.randn_like(out)
    dist.broadcast(grad, src=0)
    # wgrad RS path always accumulates into main_grad; allocate before backward.
    layer.weight.main_grad = torch.zeros(layer.weight.shape, dtype=dtype, device="cuda")
    out.backward(grad)
    assert inp_te.grad is not None and inp_te.grad.shape == inp.shape
    assert torch.isfinite(inp_te.grad).all(), f"rank {rank}: non-finite dX"


class TestTPGTPColumnParallelLinear:
    @pytest.mark.parametrize("tp_size,gtp_size", [(2, 2)])
    def test_forward_backward_correctness(self, tp_size, gtp_size):
        world_size = tp_size * gtp_size
        _requires_multi_gpu(world_size)
        _run_distributed(_worker_column_correctness, world_size, tp_size, gtp_size)


# ---------------------------------------------------------------------------
# 3. TestTPGTPRowParallelLinear
# ---------------------------------------------------------------------------


def _worker_row_forward_backward(rank, world_size, port, tp_size, gtp_size):
    """Row-parallel: weight shape verified; output is all-reduced [batch, out_f]; backward produces finite dX."""
    torch.manual_seed(0)
    tp_group, gtp_group, tp_rank, _ = _build_groups(rank, world_size, tp_size, gtp_size)

    batch = 16
    in_f = tp_size * 64  # full in_features
    out_f = gtp_size * 64  # full out_features
    dtype = torch.bfloat16

    layer = _make_gtp_linear(in_f, out_f, gtp_group, dtype, parallel_mode="row", tp_group=tp_group)

    expected_shape = (out_f // gtp_size, in_f // tp_size)
    assert isinstance(
        layer.weight, GTPShardedParam
    ), f"rank {rank}: weight should be GTPShardedParam"
    assert (
        layer.weight.shape == expected_shape
    ), f"rank {rank}: expected {expected_shape}, got {layer.weight.shape}"

    # Row-parallel: each TP rank takes the corresponding slice of in_f
    full_inp = torch.randn(batch, in_f, dtype=dtype, device="cuda")
    dist.broadcast(full_inp, src=0)
    local_in_f = in_f // tp_size
    inp = full_inp[:, tp_rank * local_in_f : (tp_rank + 1) * local_in_f]
    inp = inp.clone().requires_grad_(True)

    # TE forward: GTP all-gathers weight, row-parallel all-reduces output across TP
    out = layer(inp, is_first_microbatch=True)
    assert out.shape == (
        batch,
        out_f,
    ), f"rank {rank}: output shape {out.shape} != ({batch}, {out_f})"
    assert torch.isfinite(out).all(), f"rank {rank}: non-finite output"

    # wgrad RS path always accumulates into main_grad; allocate before backward.
    layer.weight.main_grad = torch.zeros(layer.weight.shape, dtype=dtype, device="cuda")
    out.sum().backward()
    assert inp.grad is not None and inp.grad.shape == inp.shape
    assert torch.isfinite(inp.grad).all(), f"rank {rank}: non-finite dX"


def _worker_row_correctness(rank, world_size, port, tp_size, gtp_size):
    """Row-parallel all-reduced output must equal inp_full @ full_weight^T."""
    torch.manual_seed(0)
    tp_group, gtp_group, tp_rank, _ = _build_groups(rank, world_size, tp_size, gtp_size)

    batch = 16
    in_f = tp_size * 64
    out_f = gtp_size * 64
    dtype = torch.bfloat16

    layer = _make_gtp_linear(in_f, out_f, gtp_group, dtype, parallel_mode="row", tp_group=tp_group)

    # Reconstruct full weight: all-gather GTP shards → TP-local, then all-gather TP shards
    shard = layer.weight.data.clone()
    all_gtp_shards = [torch.zeros_like(shard) for _ in range(gtp_size)]
    dist.all_gather(all_gtp_shards, shard, group=gtp_group)
    tp_local_weight = torch.cat(all_gtp_shards, dim=0).float()  # [out_f, in_f/tp_size]

    all_tp_weights = [torch.zeros_like(tp_local_weight) for _ in range(tp_size)]
    dist.all_gather(all_tp_weights, tp_local_weight, group=tp_group)
    full_weight = torch.cat(all_tp_weights, dim=1).float()  # [out_f, in_f]

    # Full input (same on all ranks; we slice below to simulate row-parallel)
    full_inp = torch.randn(batch, in_f, dtype=dtype, device="cuda")
    dist.broadcast(full_inp, src=0)
    local_in_f = in_f // tp_size
    inp = full_inp[:, tp_rank * local_in_f : (tp_rank + 1) * local_in_f].clone()
    inp.requires_grad_(True)

    out = layer(inp, is_first_microbatch=True)

    # Reference: full input @ full weight^T — all ranks should see the same output
    ref = full_inp.float() @ full_weight.T
    ref = ref.to(dtype)
    assert torch.allclose(
        out.float(), ref.float(), atol=2e-2, rtol=1e-2
    ), f"rank {rank}: output mismatch, max_diff={(out.float() - ref.float()).abs().max():.4f}"


class TestTPGTPRowParallelLinear:
    @pytest.mark.parametrize("tp_size,gtp_size", [(2, 2)])
    def test_forward_backward(self, tp_size, gtp_size):
        world_size = tp_size * gtp_size
        _requires_multi_gpu(world_size)
        _run_distributed(_worker_row_forward_backward, world_size, tp_size, gtp_size)

    @pytest.mark.parametrize("tp_size,gtp_size", [(2, 2)])
    def test_forward_correctness(self, tp_size, gtp_size):
        world_size = tp_size * gtp_size
        _requires_multi_gpu(world_size)
        _run_distributed(_worker_row_correctness, world_size, tp_size, gtp_size)


# ---------------------------------------------------------------------------
# 4. TestTPGTPLayerNormLinear - column-parallel smoke test
# ---------------------------------------------------------------------------


def _worker_layernorm_linear(rank, world_size, port, tp_size, gtp_size):
    torch.manual_seed(0)
    tp_group, gtp_group, _, _ = _build_groups(rank, world_size, tp_size, gtp_size)

    seq, batch = 4, 2
    in_f = 64
    out_f = tp_size * gtp_size * 32
    dtype = torch.bfloat16

    layer = te.LayerNormLinear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        params_dtype=dtype,
        parallel_mode="column",
        device="cuda",
        tp_group=tp_group,
        gtp_group=gtp_group,
    )
    assert isinstance(
        layer.weight, GTPShardedParam
    ), f"rank {rank}: LayerNormLinear.weight should be GTPShardedParam"
    expected_rows = out_f // (tp_size * gtp_size)
    assert layer.weight.shape == (
        expected_rows,
        in_f,
    ), f"rank {rank}: unexpected weight shape {layer.weight.shape}"

    inp = torch.randn(seq, batch, in_f, dtype=dtype, device="cuda", requires_grad=True)
    dist.broadcast(inp, src=0)

    out = layer(inp, is_first_microbatch=True)
    assert out.shape == (seq, batch, out_f // tp_size), f"rank {rank}: output shape {out.shape}"
    assert torch.isfinite(out).all(), f"rank {rank}: non-finite output"

    # wgrad RS path always accumulates into main_grad; allocate before backward.
    layer.weight.main_grad = torch.zeros(layer.weight.shape, dtype=dtype, device="cuda")
    out.sum().backward()
    assert inp.grad is not None and inp.grad.shape == inp.shape
    assert torch.isfinite(inp.grad).all(), f"rank {rank}: non-finite dX"


class TestTPGTPLayerNormLinear:
    @pytest.mark.parametrize("tp_size,gtp_size", [(2, 2)])
    def test_forward_backward(self, tp_size, gtp_size):
        world_size = tp_size * gtp_size
        _requires_multi_gpu(world_size)
        _run_distributed(_worker_layernorm_linear, world_size, tp_size, gtp_size)
