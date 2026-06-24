# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Parity tests for GTP + Muon Newton-Schulz orthogonalization.

``TensorParallelMuon.scaled_orthogonalize_fn_with_gtp_remat`` orthogonalizes a GTP-row-sharded
momentum in one of three modes (blockwise, duplicated, distributed) instead of all-gathering
the full matrix and running a redundant full NS on every rank. These tests verify that each
mode produces the correct per-shard result:

  1. test_gtp_distributed_mode - GTP4, TP1: distribute NS over GTP (dim 0).
  2. test_row_parallel         - TP2, GTP2 (TP dim 1, GTP dim 0): gather smaller, distribute larger.
  3. test_col_parallel         - TP2, GTP2 (both dim 0): gather smaller, distribute larger.
  4. test_gtp_duplicated_mode  - GTP4, TP1: all-gather over GTP, full NS, reshard.
  5. test_gtp_blockwise_mode   - GTP4, TP1: local NS on GTP shard, no collective.

All require world_size == 4.
"""

import pytest
import torch

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.17", allow_module_level=True)

from megatron.core import parallel_state as ps
from megatron.core.optimizer.emerging_optimizers import HAVE_EMERGING_OPTIMIZERS, TensorParallelMuon

if not HAVE_EMERGING_OPTIMIZERS:
    pytest.skip("emerging_optimizers not available", allow_module_level=True)

from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (
    _torchrun_dist_init,  # autouse fixture: initializes the torchrun dist group
)
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (
    reset_gtp_globals,  # autouse fixture: resets GTP class state between tests
)
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: F401
    _requires_multi_gpu,
    _run_distributed,
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


def _make_muon(pg_collection, tp_mode="distributed"):
    """A TensorParallelMuon used only for its orthogonalize helpers (never stepped)."""
    placeholder = torch.nn.Parameter(torch.zeros(1, device="cuda"))
    return TensorParallelMuon(
        params=[placeholder],
        lr=0.01,
        momentum=0.95,
        weight_decay=0.0,
        num_ns_steps=_NS_STEPS,
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


def _worker_gtp_distributed(rank, world_size, port):
    """distributed mode (GTP4, TP1): distribute NS over the GTP group on the local dim-0 shard."""
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=world_size
    )
    try:
        pgc = ProcessGroupCollection.use_mpu_process_groups()
        opt = _make_muon(pgc)
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


def _worker_row_parallel(rank, world_size, port):
    """RowParallel TP, GTP (TP dim 1, GTP dim 0): gather the smaller group, distribute larger."""
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=2, pipeline_model_parallel_size=1, gtp_remat_size=2
    )
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


def _worker_col_parallel(rank, world_size, port):
    """ColumnParallel TP, GTP (both dim 0): gather smaller group, distribute larger.

    dim-0 carve is TP-outer / GTP-inner (GTP slices the already-TP-sharded weight), so this
    rank owns rows ``tr*(M/ts) + gr*sp : + sp``.
    """
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=2, pipeline_model_parallel_size=1, gtp_remat_size=2
    )
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


def _worker_gtp_duplicated(rank, world_size, port):
    """duplicated mode (GTP4, TP1): all-gather full matrix over GTP, whole NS, reshard.

    Mathematically identical to the full-matrix reference, so the local block must match.
    """
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=world_size
    )
    try:
        pgc = ProcessGroupCollection.use_mpu_process_groups()
        opt = _make_muon(pgc, tp_mode="duplicated")
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


def _worker_gtp_blockwise(rank, world_size, port):
    """blockwise mode (GTP4, TP1): local NS on the [M/gtp_size, K] shard, no GTP collective.

    Must equal a plain Newton-Schulz of the local shard, not the full-matrix shard.
    """
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=world_size
    )
    try:
        pgc = ProcessGroupCollection.use_mpu_process_groups()
        opt = _make_muon(pgc, tp_mode="blockwise")
        w = _full_weight()

        gs, gr = _world_size(pgc.gtp_remat), _rank(pgc.gtp_remat)
        sp = _M // gs
        local = w[gr * sp : (gr + 1) * sp, :].clone()
        local.is_gtp_weight_remat = True

        out = opt.scaled_orthogonalize_fn_with_gtp_remat(local, local, pgc.tp, None)
        # blockwise orthogonalizes the local block independently, no GTP comm.
        expected = opt.scaled_orthogonalize_fn(local.clone(), pgc.tp, None)
        torch.testing.assert_close(out, expected, atol=_ATOL, rtol=_RTOL)
    finally:
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()


class TestGTPMuonDistributedNS:
    """Distributed-NS orthogonalization matches full-matrix NS, per shard."""

    def test_gtp_distributed_mode(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_gtp_distributed, 4)

    def test_row_parallel(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_row_parallel, 4)

    def test_col_parallel(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_col_parallel, 4)

    def test_gtp_duplicated_mode(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_gtp_duplicated, 4)

    def test_gtp_blockwise_mode(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_gtp_blockwise, 4)
