# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Split-QKV must be layout-invariant: TP1 and GTP must run the SAME Muon rule.

Two properties, matching the two halves of the fix:
  1. the decision uses the GTP-unsharded row count, not this rank's shard
     (`test_split_after_gather_matches_tp1` reproduces the shape that used to fail:
     the per-rank shard (`_M // _GTP_WORLD_SIZES`) doesn't divide evenly by `_GROUP`,
     the exact condition that used to disable the split on every GTP rank);
  2. the split is applied AFTER the all-gather, so the GTP result equals TP1's
     restricted to this rank's rows, and row-sharded modes fall back to whole-matrix NS.
"""

import contextlib
import logging
from unittest import mock

import pytest
import torch

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TE with hook registry", allow_module_level=True)

from megatron.core import parallel_state as ps
from megatron.core.optimizer import emerging_optimizers as _eo_module
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


@contextlib.contextmanager
def _gtp_case(world_size):
    """Common init/teardown for a GTP+Muon QKV-split worker at this ``world_size``."""
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=world_size
    )
    try:
        yield
    finally:
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()


def _local_shard(w, gtp_remat_group):
    """This rank's row shard of ``w``, tagged as GTP_remat-sharded."""
    gr = torch.distributed.get_rank(group=gtp_remat_group)
    sp = _M // torch.distributed.get_world_size(group=gtp_remat_group)
    local = w[gr * sp : (gr + 1) * sp, :].clone()
    local.is_gtp_weight_remat = True
    return local, gr, sp


def _worker_split_after_gather(rank, world_size, port):
    """The GTP shard of the split result must equal the TP1 split result, sliced."""
    with _gtp_case(world_size):
        pgc = ProcessGroupCollection.use_mpu_process_groups()
        opt = _make_muon(pgc)
        w = _full_weight()
        ref = _reference_split_orth(opt, w, pgc.tp)
        local, gr, sp = _local_shard(w, pgc.gtp_remat)

        out = opt.scaled_orthogonalize_fn_with_gtp_remat(
            local, local, pgc.tp, None, qkv_split_shapes=_SPLIT
        )
        torch.testing.assert_close(out, ref[gr * sp : (gr + 1) * sp, :], atol=_ATOL, rtol=_RTOL)


def _worker_split_differs_from_whole(rank, world_size, port):
    """Negative control: whole-matrix NS is the old GTP behaviour, so a regression
    to it would make the equality test above compare two identical things."""
    with _gtp_case(world_size):
        pgc = ProcessGroupCollection.use_mpu_process_groups()
        opt = _make_muon(pgc)
        w = _full_weight()
        local, _, _ = _local_shard(w, pgc.gtp_remat)

        split = opt.scaled_orthogonalize_fn_with_gtp_remat(
            local, local, pgc.tp, None, qkv_split_shapes=_SPLIT
        )
        whole = opt.scaled_orthogonalize_fn_with_gtp_remat(local, local, pgc.tp, None)
        assert not torch.allclose(
            split, whole, atol=_ATOL, rtol=_RTOL
        ), "split-QKV produced the whole-matrix result; the split is not being applied"


def _worker_row_sharded_modes_fall_back(rank, world_size, port, mode):
    """Row-sharded modes hold no q/k/v boundary, so they must keep the pre-fix
    whole-matrix rule -- these configs trained before this fix and must still run."""
    with _gtp_case(world_size):
        pgc = ProcessGroupCollection.use_mpu_process_groups()
        opt = _make_muon(pgc, tp_mode=mode)
        w = _full_weight()
        local, _, _ = _local_shard(w, pgc.gtp_remat)

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


def _worker_warns_once(rank, world_size, port):
    """A per-step warning would print once per qkv weight per iteration, forever."""
    with _gtp_case(world_size):
        pgc = ProcessGroupCollection.use_mpu_process_groups()
        opt = _make_muon(pgc, tp_mode="distributed")
        w = _full_weight()
        local, _, _ = _local_shard(w, pgc.gtp_remat)

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


def _worker_auto_mode_forces_duplicated_for_split(rank, world_size, port):
    """tp_mode="auto" must not let the shape/group_size cost model decide whether the
    split happens: pinned to duplicated whenever qkv_split_shapes is set, so this must
    match the TP1 reference and never hit the row-sharded fallback -- regardless of what
    the cost model would otherwise pick for this shape."""
    with _gtp_case(world_size):
        pgc = ProcessGroupCollection.use_mpu_process_groups()
        opt = _make_muon(pgc, tp_mode="auto")
        w = _full_weight()
        ref = _reference_split_orth(opt, w, pgc.tp)
        local, gr, sp = _local_shard(w, pgc.gtp_remat)

        out = opt.scaled_orthogonalize_fn_with_gtp_remat(
            local, local, pgc.tp, None, qkv_split_shapes=_SPLIT
        )
        assert not opt._warned_qkv_split_disabled, "auto must not hit the row-sharded fallback"
        torch.testing.assert_close(out, ref[gr * sp : (gr + 1) * sp, :], atol=_ATOL, rtol=_RTOL)


def _worker_partial_pg_collection(rank, world_size, port):
    """The tagging loop and the step must resolve the SAME GTP group.

    Tagging resolves via resolve_gtp_remat_group, which falls back to the MPU group
    for a collection that never declared gtp_remat; a direct attribute read returns
    None there instead. The two then disagree about whether the weight is sharded --
    tagged splittable on the gathered row count, stepped as if unsharded -- and the
    split hits a row shard.
    """
    with _gtp_case(world_size):
        full = ProcessGroupCollection.use_mpu_process_groups()
        partial = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp"])
        assert partial.gtp_remat is None, "a partial collection must not carry gtp_remat"
        assert (
            resolve_gtp_remat_group(partial, is_expert=False) is not None
        ), "tagging resolves a real group here; the step has to reach the same one"

        w = _full_weight()
        local, _, _ = _local_shard(w, full.gtp_remat)

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
