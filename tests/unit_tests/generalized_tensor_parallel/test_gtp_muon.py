# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Parity tests for GTP + Muon Newton-Schulz orthogonalization.

``TensorParallelMuon.scaled_orthogonalize_fn_with_gtp_remat`` orthogonalizes a GTP-row-sharded
momentum in one of three modes (blockwise, duplicated, distributed) instead of all-gathering
the full matrix and running a redundant full NS on every rank. These tests verify that each
mode produces the correct per-shard result:

  1. test_gtp_only_parity   - TP1: distributed and duplicated both match full-matrix NS.
  2. test_gtp_blockwise_mode - TP1: local NS on the GTP shard, no collective.
  3. test_row_parallel      - TP on dim 1, GTP on dim 0: gather smaller group, distribute larger.
  4. test_col_parallel      - TP and GTP both on dim 0: gather smaller group, distribute larger.

The TP cases run at ``tp_size * gtp_remat_size`` ranks and cover ``tp_size != gtp_remat_size``
in both directions, since equal sizes are the one shape where a mix-up of the two dim-0 axes
would still land on self-consistent offsets. Each case skips unless the launched world size
matches, so the 4-rank shapes cover a 4-GPU dev box and the 8-rank ones cover CI.
"""

import pytest
import torch

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TE with hook registry", allow_module_level=True)

from megatron.core import parallel_state as ps
from megatron.core.optimizer.emerging_optimizers import HAVE_EMERGING_OPTIMIZERS, TensorParallelMuon

if not HAVE_EMERGING_OPTIMIZERS:
    pytest.skip("emerging_optimizers not available", allow_module_level=True)

from emerging_optimizers.orthogonalized_optimizers import get_muon_scale_factor
from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz

from megatron.core.process_groups_config import ProcessGroupCollection

# _torchrun_dist_init and reset_gtp_globals are autouse fixtures; pytest only applies them
# if the names are bound in this module.
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: F401
    _requires_multi_gpu,
    _run_distributed,
    _torchrun_dist_init,
    reset_gtp_globals,
)

# Parity is asserted at num_ns_steps=1: there the distributed Gram all-reduce equals the
# full-matrix Gram to fp32 reduction-order noise (~1e-5). At more steps the (mathematically
# identical) distributed result still matches full NS in exact arithmetic, but the aggressive
# NS coefficients are tuned beyond convergence (see newton_schulz docstring) and amplify the
# ~1e-6 fp difference (~1e-2 by step 5); that is NS conditioning, not a distribution error, so
# a one-step parity check is the meaningful correctness test. fp32-highest matmul throughout.
_M, _K = 128, 64
_NS_STEPS = 1
_ATOL, _RTOL = 1e-4, 1e-4

# Spelled out rather than left to TensorParallelMuon's defaults so the blockwise test can
# rebuild the expected value from the same primitives the production path uses.
_COEFFICIENT_TYPE = "quintic"
_SCALE_MODE = "spectral"

# GTP-only shapes (TP1); the GTP group spans the whole world.
_GTP_ONLY_WORLD_SIZES = [4, 8]
# (tp_size, gtp_remat_size) for the combined TP + GTP shapes; world size is their product.
_TP_GTP_SHAPES = [(2, 2), (2, 4), (4, 2)]


def _make_muon(pg_collection, tp_mode="distributed"):
    """A TensorParallelMuon used only for its orthogonalize helpers (never stepped)."""
    placeholder = torch.nn.Parameter(torch.zeros(1, device="cuda"))
    return TensorParallelMuon(
        params=[placeholder],
        lr=0.01,
        momentum=0.95,
        weight_decay=0.0,
        num_ns_steps=_NS_STEPS,
        coefficient_type=_COEFFICIENT_TYPE,
        scale_mode=_SCALE_MODE,
        fp32_matmul_prec="highest",
        pg_collection=pg_collection,
        tp_mode=tp_mode,
    )


def _full_weight():
    """Full [M, K] momentum, identical on every rank (rank-0 broadcast)."""
    torch.manual_seed(0)
    w = torch.randn(_M, _K, dtype=torch.float32, device="cuda")
    torch.distributed.broadcast(w, src=0)
    return w


def _reference_full_orth(opt, w, tp_group):
    """Orthogonalize the full matrix (partition_dim=None gives plain NS), same scale/coeffs."""
    return opt.scaled_orthogonalize_fn(w.clone(), tp_group, partition_dim=None)


def _world_size(group):
    return torch.distributed.get_world_size(group=group)


def _rank(group):
    return torch.distributed.get_rank(group=group)


def _init_model_parallel(tp_size, gtp_remat_size):
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        gtp_remat_size=gtp_remat_size,
    )


def _worker_gtp_only_parity(rank, world_size, port, tp_mode):
    """distributed and duplicated (TP1): the local GTP shard must match full-matrix NS.

    Both modes are mathematically identical to orthogonalizing the whole matrix, one by
    distributing the Gram all-reduce over GTP and one by all-gathering first.
    """
    _init_model_parallel(1, world_size)
    try:
        pgc = ProcessGroupCollection.use_mpu_process_groups()
        opt = _make_muon(pgc, tp_mode=tp_mode)
        w = _full_weight()
        ref = _reference_full_orth(opt, w, pgc.tp)

        gs, gr = _world_size(pgc.gtp_remat), _rank(pgc.gtp_remat)
        sp = _M // gs
        local = w[gr * sp : (gr + 1) * sp, :].clone()
        local.is_gtp_weight_remat = True

        out = opt.scaled_orthogonalize_fn_with_gtp_remat(local, local, pgc.tp, None)
        expected = ref[gr * sp : (gr + 1) * sp, :]
        torch.testing.assert_close(out, expected, atol=_ATOL, rtol=_RTOL)
    finally:
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()


def _worker_row_parallel(rank, world_size, port, tp_size, gtp_remat_size):
    """RowParallel TP, GTP (TP dim 1, GTP dim 0): gather the smaller group, distribute larger."""
    _init_model_parallel(tp_size, gtp_remat_size)
    try:
        pgc = ProcessGroupCollection.use_mpu_process_groups()
        opt = _make_muon(pgc)
        w = _full_weight()
        ref = _reference_full_orth(opt, w, pgc.tp)

        gs, gr = _world_size(pgc.gtp_remat), _rank(pgc.gtp_remat)
        ts, tr = _world_size(pgc.tp), _rank(pgc.tp)
        sp, kt = _M // gs, _K // ts  # GTP row block, TP col block
        local = w[gr * sp : (gr + 1) * sp, tr * kt : (tr + 1) * kt].clone()
        local.is_gtp_weight_remat = True

        out = opt.scaled_orthogonalize_fn_with_gtp_remat(local, local, pgc.tp, 1)
        expected = ref[gr * sp : (gr + 1) * sp, tr * kt : (tr + 1) * kt]
        torch.testing.assert_close(out, expected, atol=_ATOL, rtol=_RTOL)
    finally:
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()


def _worker_col_parallel(rank, world_size, port, tp_size, gtp_remat_size):
    """ColumnParallel TP, GTP (both dim 0): gather smaller group, distribute larger.

    dim-0 carve is TP-outer / GTP-inner (GTP slices the already-TP-sharded weight), so this
    rank owns rows ``tr*(M/ts) + gr*sp : + sp``.
    """
    _init_model_parallel(tp_size, gtp_remat_size)
    try:
        pgc = ProcessGroupCollection.use_mpu_process_groups()
        opt = _make_muon(pgc)
        w = _full_weight()
        ref = _reference_full_orth(opt, w, pgc.tp)

        gs, gr = _world_size(pgc.gtp_remat), _rank(pgc.gtp_remat)
        ts, tr = _world_size(pgc.tp), _rank(pgc.tp)
        m_tp = _M // ts
        sp = m_tp // gs
        off = tr * m_tp + gr * sp
        local = w[off : off + sp, :].clone()
        local.is_gtp_weight_remat = True

        out = opt.scaled_orthogonalize_fn_with_gtp_remat(local, local, pgc.tp, 0)
        expected = ref[off : off + sp, :]
        torch.testing.assert_close(out, expected, atol=_ATOL, rtol=_RTOL)
    finally:
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()


def _worker_gtp_blockwise(rank, world_size, port):
    """blockwise mode (TP1): local NS on the [M/gtp_size, K] shard, no GTP collective.

    The expected value is rebuilt from newton_schulz and get_muon_scale_factor rather than
    from ``scaled_orthogonalize_fn``, so the check is independent of the production wrapper
    and pins the scale factor to the shard row count instead of the full M.
    """
    _init_model_parallel(1, world_size)
    try:
        pgc = ProcessGroupCollection.use_mpu_process_groups()
        opt = _make_muon(pgc, tp_mode="blockwise")
        w = _full_weight()
        ref = _reference_full_orth(opt, w, pgc.tp)

        gs, gr = _world_size(pgc.gtp_remat), _rank(pgc.gtp_remat)
        sp = _M // gs
        local = w[gr * sp : (gr + 1) * sp, :].clone()
        local.is_gtp_weight_remat = True

        out = opt.scaled_orthogonalize_fn_with_gtp_remat(local, local, pgc.tp, None)

        raw = newton_schulz(local.clone(), steps=_NS_STEPS, coefficient_type=_COEFFICIENT_TYPE)
        expected = raw * get_muon_scale_factor(sp, _K, mode=_SCALE_MODE)
        torch.testing.assert_close(out, expected, atol=_ATOL, rtol=_RTOL)

        # NS of a row block is not the row block of NS of the full matrix, so a silent
        # regression to the all-gather path would show up here.
        assert not torch.allclose(
            out, ref[gr * sp : (gr + 1) * sp, :], atol=_ATOL, rtol=_RTOL
        ), "blockwise matched the full-matrix result; it is not orthogonalizing the shard alone"
    finally:
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()


class TestGTPMuonDistributedNS:
    """Distributed-NS orthogonalization matches full-matrix NS, per shard."""

    @pytest.mark.parametrize("tp_mode", ["distributed", "duplicated"])
    @pytest.mark.parametrize("world_size", _GTP_ONLY_WORLD_SIZES)
    def test_gtp_only_parity(self, world_size, tp_mode):
        _requires_multi_gpu(world_size)
        _run_distributed(_worker_gtp_only_parity, world_size, tp_mode)

    @pytest.mark.parametrize("tp_size,gtp_remat_size", _TP_GTP_SHAPES)
    def test_row_parallel(self, tp_size, gtp_remat_size):
        world_size = tp_size * gtp_remat_size
        _requires_multi_gpu(world_size)
        _run_distributed(_worker_row_parallel, world_size, tp_size, gtp_remat_size)

    @pytest.mark.parametrize("tp_size,gtp_remat_size", _TP_GTP_SHAPES)
    def test_col_parallel(self, tp_size, gtp_remat_size):
        world_size = tp_size * gtp_remat_size
        _requires_multi_gpu(world_size)
        _run_distributed(_worker_col_parallel, world_size, tp_size, gtp_remat_size)

    @pytest.mark.parametrize("world_size", _GTP_ONLY_WORLD_SIZES)
    def test_gtp_blockwise_mode(self, world_size):
        _requires_multi_gpu(world_size)
        _run_distributed(_worker_gtp_blockwise, world_size)
