# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Split-QKV must be layout-invariant: TP1 and GTP must run the SAME Muon rule.

Two properties, matching the two halves of the fix:
  1. the decision uses the GTP-unsharded row count, not this rank's shard
     (`test_local_shard_would_have_disabled` pins the shape that used to fail);
  2. the split is applied AFTER the all-gather, so the GTP result equals TP1's
     restricted to this rank's rows, and row-sharded modes fall back to whole-matrix NS.
"""

import logging
from unittest import mock

import pytest
import torch

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TE with hook registry", allow_module_level=True)

from megatron.core import parallel_state as ps
from megatron.core.optimizer import emerging_optimizers as _eo_module
from megatron.core.optimizer import qkv_rows_after_gtp_gather
from megatron.core.optimizer.emerging_optimizers import HAVE_EMERGING_OPTIMIZERS, TensorParallelMuon

if not HAVE_EMERGING_OPTIMIZERS:
    pytest.skip("emerging_optimizers not available", allow_module_level=True)

from megatron.core.process_groups_config import ProcessGroupCollection, resolve_gtp_remat_group

# _torchrun_dist_init and reset_gtp_globals are autouse fixtures; pytest only applies
# them if the names are bound in this module.
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: F401
    _requires_multi_gpu,
    _run_distributed,
    _torchrun_dist_init,
    reset_gtp_globals,
)

# Production [6144, 192, 192] scaled down 96x, keeping the ratio that makes the bug
# visible: k and v are tiny next to q, so joint orthogonalization lets q dominate them.
_SPLIT = [64, 16, 16]
_GROUP = sum(_SPLIT)  # one query group's [q|k|v]
# Two groups, not one: at groups==1 the interleaved layout coincides with a
# contiguous slab, so the interleaving would go untested.
_GROUPS = 2
_M = _GROUPS * _GROUP
_K = 64

# num_ns_steps=1 as in test_gtp_muon.py: more steps amplify fp32 reduction-order
# noise, which is NS conditioning rather than a distribution error.
_NS_STEPS = 1
_ATOL, _RTOL = 1e-4, 1e-4
_COEFFICIENT_TYPE = "quintic"
_SCALE_MODE = "spectral"

# The local shard must NOT be a multiple of _GROUP -- that is the bug shape.
# 192/2 = 96 is, so world 2 cannot reproduce it; 192/4 and 192/8 can.
_GTP_WORLD_SIZES = [4, 8]


class _FakeGroup:
    """Only ``.size()`` is read, so no real ProcessGroup is needed."""

    def __init__(self, n):
        self._n = n

    def size(self):
        return self._n


def _param(rows, pad=0, sharded=True):
    """A stand-in for a qkv weight. `sharded` mirrors is_gtp_weight_remat."""
    p = torch.nn.Parameter(torch.zeros(rows, _K), requires_grad=False)
    p.is_gtp_weight_remat = sharded
    if pad:
        p.pad_length = pad
    return p


def _rows(param, split, gtp_size=None):
    """qkv_rows_after_gtp_gather with a stand-in for the optimizer's GTP group."""
    # get_pg_size returns 1 when dist is down, which would pass these for the wrong reason.
    assert torch.distributed.is_initialized()
    return qkv_rows_after_gtp_gather(param, split, _FakeGroup(gtp_size) if gtp_size else None)


class TestQKVRowCount:
    """The is_qkv decision is taken on the GTP-unsharded shape, so it is layout-invariant."""

    @pytest.mark.parametrize("gtp_size", [1, 2, 4, 64])
    def test_decision_is_layout_invariant(self, gtp_size):
        """ONE weight of _M rows, sharded every which way: same verdict, same count.

        Asserted against constants, not against the implementation's own expression --
        restating the formula would pass for any implementation that keeps it. At
        gtp_size=64 the shard is 3 rows, so the shard-local test this replaced would
        say False here while saying True at gtp_size=1.
        """
        assert _M % gtp_size == 0, "the fixture must shard evenly to stay one weight"
        assert _rows(_param(_M // gtp_size), _SPLIT, gtp_size) == (_M, gtp_size, True)

    def test_local_shard_would_have_disabled(self):
        """The regression test: production's 102 x 64 = 6528, which the old
        ``param.shape[0] % 6528`` rule reported False on every GTP rank."""
        local, gtp = 102, 64
        split = [6144, 192, 192]
        assert local % sum(split) != 0, "shard-local test must fail for this to be a regression"
        assert _rows(_param(local), split, gtp) == (6528, 64, True)

    def test_unsharded_matches_sharded(self):
        """TP1 (no group attr) and GTP must reach the same verdict for one weight."""
        assert _rows(_param(_M), _SPLIT)[2] is True
        assert _rows(_param(_M // 4), _SPLIT, 4)[2] is True

    def test_padding_is_excluded(self):
        """GTP pads dim 0 up to a multiple of the group size; pad rows are not weight."""
        rows, _, splittable = _rows(_param(25, pad=4), _SPLIT, 4)  # 25 x 4 - 4 pad
        assert (rows, splittable) == (_GROUP, True)
        # Unsubtracted the count is 100, which is NOT splittable -- so the assert above
        # pins the subtraction, not just that 96 happens to work.
        assert 100 % _GROUP != 0

    def test_non_qkv_shape_is_rejected(self):
        """A weight whose GTP-unsharded rows do not divide is refused, not forced."""
        rows, _, splittable = _rows(_param(30), _SPLIT, 4)
        assert (rows, splittable) == (120, False)

    def test_unsharded_param_ignores_the_group(self):
        """Tagging gates on is_gtp_weight_remat, the same signal `gtp_active` uses;
        scaling regardless would hand the split a local shard. pad_length goes unused
        for the same reason -- no scaling, nothing to subtract."""
        assert _rows(_param(_GROUP, pad=4, sharded=False), _SPLIT, 4) == (_GROUP, 1, True)


def _make_muon(pg_collection, tp_mode="duplicated"):
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
    """Full [_M, _K] momentum, identical on every rank (rank-0 broadcast)."""
    torch.manual_seed(0)
    w = torch.randn(_M, _K, dtype=torch.float32, device="cuda")
    torch.distributed.broadcast(w, src=0)
    return w


def _reference_split_orth(opt, w, tp_group):
    """The TP1 path, written independently of the implementation. Layout is per query
    group -- [q0|k0|v0][q1|k1|v1] -- so `q` is a strided set of rows, not a slab."""
    out = torch.empty_like(w)
    off = 0
    for n in _SPLIT:
        rows = torch.cat(
            [torch.arange(g * _GROUP + off, g * _GROUP + off + n) for g in range(_GROUPS)]
        ).to(w.device)
        orth = opt.scaled_orthogonalize_fn(w[rows].clone(), tp_group, None)
        out[rows] = orth
        off += n
    return out


def _init_model_parallel(tp_size, gtp_remat_size):
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        gtp_remat_size=gtp_remat_size,
    )


def _worker_split_after_gather(rank, world_size, port):
    """The GTP shard of the split result must equal the TP1 split result, sliced."""
    _init_model_parallel(1, world_size)
    try:
        pgc = ProcessGroupCollection.use_mpu_process_groups()
        opt = _make_muon(pgc)
        w = _full_weight()
        ref = _reference_split_orth(opt, w, pgc.tp)

        gs = torch.distributed.get_world_size(group=pgc.gtp_remat)
        gr = torch.distributed.get_rank(group=pgc.gtp_remat)
        sp = _M // gs
        local = w[gr * sp : (gr + 1) * sp, :].clone()
        local.is_gtp_weight_remat = True

        out = opt.scaled_orthogonalize_fn_with_gtp_remat(
            local, local, pgc.tp, None, qkv_split_shapes=_SPLIT
        )
        torch.testing.assert_close(out, ref[gr * sp : (gr + 1) * sp, :], atol=_ATOL, rtol=_RTOL)
    finally:
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()


def _worker_split_differs_from_whole(rank, world_size, port):
    """Negative control: whole-matrix NS is the old GTP behaviour, so a regression
    to it would make the equality test above compare two identical things."""
    _init_model_parallel(1, world_size)
    try:
        pgc = ProcessGroupCollection.use_mpu_process_groups()
        opt = _make_muon(pgc)
        w = _full_weight()

        gs = torch.distributed.get_world_size(group=pgc.gtp_remat)
        gr = torch.distributed.get_rank(group=pgc.gtp_remat)
        sp = _M // gs
        local = w[gr * sp : (gr + 1) * sp, :].clone()
        local.is_gtp_weight_remat = True

        split = opt.scaled_orthogonalize_fn_with_gtp_remat(
            local, local, pgc.tp, None, qkv_split_shapes=_SPLIT
        )
        whole = opt.scaled_orthogonalize_fn_with_gtp_remat(local, local, pgc.tp, None)
        assert not torch.allclose(
            split, whole, atol=_ATOL, rtol=_RTOL
        ), "split-QKV produced the whole-matrix result; the split is not being applied"
    finally:
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()


def _worker_row_sharded_modes_fall_back(rank, world_size, port, mode):
    """Row-sharded modes hold no q/k/v boundary, so they must keep the pre-fix
    whole-matrix rule -- these configs trained before this fix and must still run."""
    _init_model_parallel(1, world_size)
    try:
        pgc = ProcessGroupCollection.use_mpu_process_groups()
        opt = _make_muon(pgc, tp_mode=mode)
        w = _full_weight()
        gs = torch.distributed.get_world_size(group=pgc.gtp_remat)
        gr = torch.distributed.get_rank(group=pgc.gtp_remat)
        sp = _M // gs
        local = w[gr * sp : (gr + 1) * sp, :].clone()
        local.is_gtp_weight_remat = True

        asked = opt.scaled_orthogonalize_fn_with_gtp_remat(
            local, local, pgc.tp, None, qkv_split_shapes=_SPLIT
        )
        assert opt._warned_qkv_split_disabled, "the fallback must warn"

        # Bitwise, not allclose: the fallback must be the SAME code path as a caller
        # that never asked to split, not merely a numerically similar one.
        never_asked = opt.scaled_orthogonalize_fn_with_gtp_remat(local, local, pgc.tp, None)
        assert torch.equal(
            asked, never_asked
        ), f"tp_mode={mode}: asking for split-QKV changed the result"
    finally:
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()


def _worker_warns_once(rank, world_size, port):
    """A per-step warning would print once per qkv weight per iteration, forever."""
    _init_model_parallel(1, world_size)
    try:
        pgc = ProcessGroupCollection.use_mpu_process_groups()
        opt = _make_muon(pgc, tp_mode="distributed")
        w = _full_weight()
        gs = torch.distributed.get_world_size(group=pgc.gtp_remat)
        gr = torch.distributed.get_rank(group=pgc.gtp_remat)
        sp = _M // gs
        local = w[gr * sp : (gr + 1) * sp, :].clone()
        local.is_gtp_weight_remat = True

        with mock.patch.object(_eo_module, "log_single_rank") as logged:
            for _ in range(3):
                opt.scaled_orthogonalize_fn_with_gtp_remat(
                    local, local, pgc.tp, None, qkv_split_shapes=_SPLIT
                )
        warnings = [c for c in logged.call_args_list if c.args[1] == logging.WARNING]
        assert len(warnings) == 1, f"expected exactly one warning, got {len(warnings)}"
        assert (
            "--muon-tp-mode duplicated" in warnings[0].args[2]
        ), "the warning must name the flag that restores the layout-invariant rule"
    finally:
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()


def _worker_auto_mode_forces_duplicated_for_split(rank, world_size, port):
    """tp_mode="auto" must not let the shape/group_size cost model decide whether the
    split happens: pinned to duplicated whenever qkv_split_shapes is set, so this must
    match the TP1 reference and never hit the row-sharded fallback -- regardless of what
    the cost model would otherwise pick for this shape."""
    _init_model_parallel(1, world_size)
    try:
        pgc = ProcessGroupCollection.use_mpu_process_groups()
        opt = _make_muon(pgc, tp_mode="auto")
        w = _full_weight()
        ref = _reference_split_orth(opt, w, pgc.tp)

        gs = torch.distributed.get_world_size(group=pgc.gtp_remat)
        gr = torch.distributed.get_rank(group=pgc.gtp_remat)
        sp = _M // gs
        local = w[gr * sp : (gr + 1) * sp, :].clone()
        local.is_gtp_weight_remat = True

        out = opt.scaled_orthogonalize_fn_with_gtp_remat(
            local, local, pgc.tp, None, qkv_split_shapes=_SPLIT
        )
        assert not opt._warned_qkv_split_disabled, "auto must not hit the row-sharded fallback"
        torch.testing.assert_close(out, ref[gr * sp : (gr + 1) * sp, :], atol=_ATOL, rtol=_RTOL)
    finally:
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()


def _worker_partial_pg_collection(rank, world_size, port):
    """The tagging loop and the step must resolve the SAME GTP group.

    Tagging resolves via resolve_gtp_remat_group, which falls back to the MPU group
    for a collection that never declared gtp_remat; a direct attribute read returns
    None there instead. The two then disagree about whether the weight is sharded --
    tagged splittable on the gathered row count, stepped as if unsharded -- and the
    split hits a row shard.
    """
    _init_model_parallel(1, world_size)
    try:
        full = ProcessGroupCollection.use_mpu_process_groups()
        partial = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp"])
        assert partial.gtp_remat is None, "a partial collection must not carry gtp_remat"
        assert (
            resolve_gtp_remat_group(partial, is_expert=False) is not None
        ), "tagging resolves a real group here; the step has to reach the same one"

        w = _full_weight()
        gs = torch.distributed.get_world_size(group=full.gtp_remat)
        gr = torch.distributed.get_rank(group=full.gtp_remat)
        sp = _M // gs
        local = w[gr * sp : (gr + 1) * sp, :].clone()
        local.is_gtp_weight_remat = True

        out_full = _make_muon(full).scaled_orthogonalize_fn_with_gtp_remat(
            local, local, full.tp, None, qkv_split_shapes=_SPLIT
        )
        # Before the two lookups were unified this raised "QKV split shape mismatch":
        # gtp_active came out False, so the split met an ungathered row shard.
        out_partial = _make_muon(partial).scaled_orthogonalize_fn_with_gtp_remat(
            local, local, partial.tp, None, qkv_split_shapes=_SPLIT
        )
        assert torch.equal(out_full, out_partial), (
            "a partial pg_collection changed the Muon update; the step resolved a "
            "different GTP group than the optimizer's qkv tagging did"
        )
    finally:
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()


class TestGTPMuonQKVSplit:
    """The [q|k|v] split happens after the all-gather, so GTP == TP1 per shard."""

    @pytest.mark.parametrize("world_size", _GTP_WORLD_SIZES)
    def test_split_after_gather_matches_tp1(self, world_size):
        _requires_multi_gpu(world_size)
        _run_distributed(_worker_split_after_gather, world_size)

    @pytest.mark.parametrize("world_size", _GTP_WORLD_SIZES)
    def test_gtp_split_differs_from_whole_matrix(self, world_size):
        _requires_multi_gpu(world_size)
        _run_distributed(_worker_split_differs_from_whole, world_size)

    # Parametrized, not hardcoded: _run_distributed skips rather than spawns, so a fixed
    # size never runs on a runner of a different width (2 runs neither at 4 nor at 8).
    @pytest.mark.parametrize("world_size", _GTP_WORLD_SIZES)
    @pytest.mark.parametrize("mode", ["blockwise", "distributed"])
    def test_row_sharded_modes_fall_back_to_whole_matrix(self, mode, world_size):
        _requires_multi_gpu(world_size)
        _run_distributed(_worker_row_sharded_modes_fall_back, world_size, mode)

    @pytest.mark.parametrize("world_size", _GTP_WORLD_SIZES)
    def test_auto_mode_forces_duplicated_for_split(self, world_size):
        _requires_multi_gpu(world_size)
        _run_distributed(_worker_auto_mode_forces_duplicated_for_split, world_size)

    @pytest.mark.parametrize("world_size", _GTP_WORLD_SIZES)
    def test_fallback_warns_once(self, world_size):
        _requires_multi_gpu(world_size)
        _run_distributed(_worker_warns_once, world_size)

    @pytest.mark.parametrize("world_size", _GTP_WORLD_SIZES)
    def test_partial_pg_collection_matches_full(self, world_size):
        _requires_multi_gpu(world_size)
        _run_distributed(_worker_partial_pg_collection, world_size)
