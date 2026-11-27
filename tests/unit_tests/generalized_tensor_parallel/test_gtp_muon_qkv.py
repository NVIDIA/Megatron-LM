# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Split-QKV must be layout-invariant: TP1 and GTP must run the SAME Muon rule.

Two properties, matching the two halves of the fix:
  1. the decision uses the GTP-unsharded row count, not this rank's shard
     (`test_split_after_gather_matches_tp1` reproduces the shape that used to fail:
     the per-rank shard (`_M // _GTP_WORLD_SIZES`) doesn't divide evenly by `_GROUP`,
     the exact condition that used to disable the split on every GTP rank);
  2. the split is applied AFTER the all-gather (duplicated), or piece by piece on the row
     shards (distributed, with GTP as the only sharding axis), so the GTP result equals TP1's
     restricted to this rank's rows; the other row-sharded cases fall back to whole-matrix NS.
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
# Seven logical query groups force real padding at both GTP4 and GTP8 when each
# rank's shard is aligned to 16 rows.
_PADDED_GROUPS = 7
_PADDED_M = _PADDED_GROUPS * _GROUP
_PAD_ALIGNMENT = 16

# num_ns_steps=1 as in test_gtp_muon.py: more steps amplify fp32 reduction-order
# noise, which is NS conditioning rather than a distribution error.
_NS_STEPS = 1
_ATOL, _RTOL = 1e-4, 1e-4
_COEFFICIENT_TYPE = "quintic"
_SCALE_MODE = "spectral"

# The local shard must NOT be a multiple of _GROUP -- that is the bug shape.
# 192/2 = 96 is, so world 2 cannot reproduce it; 192/4 and 192/8 can.
_GTP_WORLD_SIZES = [4, 8]


def _make_muon(pg_collection, tp_mode="duplicated", **kwargs):
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
        **kwargs,
    )


def _full_weight(rows=_M):
    """Full [rows, _K] momentum, identical on every rank (rank-0 broadcast)."""
    torch.manual_seed(0)
    w = torch.randn(rows, _K, dtype=torch.float32, device="cuda")
    torch.distributed.broadcast(w, src=0)
    return w


def _reference_split_orth(opt, w, tp_group):
    """The TP1 path, written independently of the implementation. Layout is per query
    group -- [q0|k0|v0][q1|k1|v1] -- so `q` is a strided set of rows, not a slab."""
    assert w.size(0) % _GROUP == 0
    num_groups = w.size(0) // _GROUP
    out = torch.empty_like(w)
    off = 0
    for n in _SPLIT:
        rows = torch.cat(
            [torch.arange(g * _GROUP + off, g * _GROUP + off + n) for g in range(num_groups)]
        ).to(w.device)
        orth = opt.scaled_orthogonalize_fn(w[rows].clone(), tp_group, None)
        out[rows] = orth
        off += n
    return out


@contextlib.contextmanager
def _gtp_case(world_size, tp_size=1):
    """Common init/teardown for a GTP+Muon QKV-split worker at this ``world_size``: GTP over
    the whole world, or TP ``tp_size`` x GTP."""
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        gtp_remat_size=world_size // tp_size,
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


def _padded_local_shard(w, gtp_group):
    """Pad fused QKV rows as GTP does, then return this rank's local shard and pad."""
    gs = torch.distributed.get_world_size(group=gtp_group)
    gr = torch.distributed.get_rank(group=gtp_group)
    pad_length = (-w.size(0)) % (_PAD_ALIGNMENT * gs)
    assert pad_length > 0, "test shape must exercise real GTP alignment padding"
    padded = torch.nn.functional.pad(w, (0, 0, 0, pad_length))
    shard_rows = padded.size(0) // gs
    local = padded[gr * shard_rows : (gr + 1) * shard_rows].clone()
    local.is_gtp_weight_remat = True
    local.pad_length = pad_length
    return local, pad_length


def _tp_gtp_local_shard(w, pgc):
    """This rank's row shard of ``w`` under TP x GTP (TP-outer / GTP-inner dim-0 carve),
    tagged as GTP_remat-sharded."""
    tp_rows = w.size(0) // torch.distributed.get_world_size(group=pgc.tp)
    sp = tp_rows // torch.distributed.get_world_size(group=pgc.gtp_remat)
    off = (
        torch.distributed.get_rank(group=pgc.tp) * tp_rows
        + torch.distributed.get_rank(group=pgc.gtp_remat) * sp
    )
    local = w[off : off + sp].clone()
    local.is_gtp_weight_remat = True
    return local


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


def _worker_padded_split_after_gather(rank, world_size, port):
    """Default duplicated mode removes fused padding before splitting Q, K, and V."""
    with _gtp_case(world_size):
        pgc = ProcessGroupCollection.use_mpu_process_groups()
        opt = _make_muon(pgc)
        w = _full_weight(_PADDED_M)
        ref = _reference_split_orth(opt, w, pgc.tp)
        local, pad_length = _padded_local_shard(w, pgc.gtp_remat)

        out = opt.scaled_orthogonalize_fn_with_gtp_remat(
            local, local, pgc.tp, None, qkv_split_shapes=_SPLIT
        )

        padded_ref = torch.nn.functional.pad(ref, (0, 0, 0, pad_length))
        gs = torch.distributed.get_world_size(group=pgc.gtp_remat)
        gr = torch.distributed.get_rank(group=pgc.gtp_remat)
        shard_rows = padded_ref.size(0) // gs
        expected = padded_ref[gr * shard_rows : (gr + 1) * shard_rows]
        torch.testing.assert_close(out, expected, atol=_ATOL, rtol=_RTOL)


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


def _worker_row_sharded_modes_fall_back(rank, world_size, port, mode, tp_size):
    """Where no q/k/v boundary can be kept -- blockwise, and distributed on a weight sharded
    by TP as well as GTP (the two-step path) -- the pre-fix whole-matrix rule must stay:
    these configs trained before this fix and must still run."""
    with _gtp_case(world_size, tp_size):
        pgc = ProcessGroupCollection.use_mpu_process_groups()
        opt = _make_muon(pgc, tp_mode=mode)
        w = _full_weight()
        if tp_size == 1:
            (local, _, _), partition_dim = _local_shard(w, pgc.gtp_remat), None
        else:
            local, partition_dim = _tp_gtp_local_shard(w, pgc), 0

        asked = opt.scaled_orthogonalize_fn_with_gtp_remat(
            local, local, pgc.tp, partition_dim, qkv_split_shapes=_SPLIT
        )
        assert opt._warned_qkv_split_disabled, "the fallback must warn"

        # Bitwise, not allclose: the fallback must be the SAME code path as a caller
        # that never asked to split, not merely a numerically similar one.
        never_asked = opt.scaled_orthogonalize_fn_with_gtp_remat(
            local, local, pgc.tp, partition_dim
        )
        assert torch.equal(
            asked, never_asked
        ), f"tp_mode={mode}: asking for split-QKV changed the result"


def _worker_warns_once(rank, world_size, port):
    """A per-step warning would print once per qkv weight per iteration, forever."""
    with _gtp_case(world_size):
        pgc = ProcessGroupCollection.use_mpu_process_groups()
        opt = _make_muon(pgc, tp_mode="blockwise")
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


def _worker_auto_mode_keeps_the_split(rank, world_size, port):
    """tp_mode="auto" must not let the shape/group_size cost model decide whether the
    split happens: it resolves every projection piece on its own, and both answers keep the
    split here (the duplicated split, or the distributed split chain), so this must match the
    TP1 reference and never hit the row-sharded fallback -- regardless of what the cost model
    picks for these shapes."""
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
    def test_padded_split_after_gather_matches_tp1(self, world_size):
        _requires_multi_gpu(world_size)
        _run_distributed(_worker_padded_split_after_gather, world_size)

    @pytest.mark.parametrize("world_size", _GTP_WORLD_SIZES)
    def test_gtp_split_differs_from_whole_matrix(self, world_size):
        _requires_multi_gpu(world_size)
        _run_distributed(_worker_split_differs_from_whole, world_size)

    # Parametrized, not hardcoded: _run_distributed skips rather than spawns, so a fixed
    # size never runs on a runner of a different width (2 runs neither at 4 nor at 8).
    @pytest.mark.parametrize("world_size", _GTP_WORLD_SIZES)
    @pytest.mark.parametrize("mode, tp_size", [("blockwise", 1), ("distributed", 2)])
    def test_row_sharded_modes_fall_back_to_whole_matrix(self, mode, tp_size, world_size):
        _requires_multi_gpu(world_size)
        _run_distributed(_worker_row_sharded_modes_fall_back, world_size, mode, tp_size)

    @pytest.mark.parametrize("world_size", _GTP_WORLD_SIZES)
    def test_auto_mode_keeps_the_split(self, world_size):
        _requires_multi_gpu(world_size)
        _run_distributed(_worker_auto_mode_keeps_the_split, world_size)

    @pytest.mark.parametrize("world_size", _GTP_WORLD_SIZES)
    def test_fallback_warns_once(self, world_size):
        _requires_multi_gpu(world_size)
        _run_distributed(_worker_warns_once, world_size)

    @pytest.mark.parametrize("world_size", _GTP_WORLD_SIZES)
    def test_partial_pg_collection_matches_full(self, world_size):
        _requires_multi_gpu(world_size)
        _run_distributed(_worker_partial_pg_collection, world_size)


# ---------------------------------------------------------------------------
# Split-QKV under the distributed mode for a weight whose only sharding axis is GTP: one mode
# per projection piece, without gathering the matrix. Under auto the tests replace the cost
# model by production's answer: Q distributed, K and V duplicated.
# Q of _SPLIT x _GROUPS is [128, 64], K and V are [32, 64].
# ---------------------------------------------------------------------------


def _muon_resolving(pg_collection, tp_mode, resolve=None):
    """A Muon in ``tp_mode``. Under auto the cost model is replaced by ``resolve`` (a piece's
    row count -> its mode), so the test does not depend on the hardware profile."""
    opt = _make_muon(pg_collection, tp_mode=tp_mode)
    if resolve is not None:
        opt._resolve_tp_mode = mock.Mock(side_effect=lambda m, n, group_size: resolve(m))
    return opt


def _split_orth(opt, local, tp_group, partition_dim=None):
    """One split-QKV orthogonalize call on this rank's shard."""
    return opt.scaled_orthogonalize_fn_with_gtp_remat(
        local, local, tp_group, partition_dim, qkv_split_shapes=_SPLIT
    )


def _shard_of(full, gtp_group):
    """This rank's rows of a [rows, ...] tensor that is already padded like GTP."""
    gs = torch.distributed.get_world_size(group=gtp_group)
    gr = torch.distributed.get_rank(group=gtp_group)
    shard_rows = full.size(0) // gs
    return full[gr * shard_rows : (gr + 1) * shard_rows]


def _worker_split_distributed_matches_tp1(rank, world_size, port, mode, num_groups, padded):
    """The distributed split against the TP1 split and the duplicated split, on every rank: the
    result is the TP1 result restricted to this rank's rows and padding rows stay zero; nothing
    is all-gathered and nothing falls back to whole-matrix Newton-Schulz. Under distributed
    every piece is one distributed Newton-Schulz step; under auto the cost model is asked once
    per piece with its logical shape, and the K and V rows are bitwise the duplicated result.

    Layouts: two query groups; seven padded groups (several runs per shard); one padded group
    (some ranks hold no Q row, the trailing ranks hold padding only)."""
    with _gtp_case(world_size):
        pgc = ProcessGroupCollection.use_mpu_process_groups()
        rows = num_groups * _GROUP
        w = _full_weight(rows)
        if padded:
            local, pad_length = _padded_local_shard(w, pgc.gtp_remat)
        else:
            (local, _, _), pad_length = _local_shard(w, pgc.gtp_remat), 0
        piece_rows = [num_groups * width for width in _SPLIT]
        q_only = mode == "auto"
        opt = _muon_resolving(
            pgc,
            mode,
            (lambda m: "distributed" if m == piece_rows[0] else "duplicated") if q_only else None,
        )

        with (
            mock.patch.object(
                torch.distributed, "all_gather", wraps=torch.distributed.all_gather
            ) as gathered,
            mock.patch.object(_eo_module, "_ChainStep", wraps=_eo_module._ChainStep) as steps,
        ):
            duplicated = _split_orth(_make_muon(pgc), local, pgc.tp)
            duplicated_gathers = gathered.call_count
            out = _split_orth(opt, local, pgc.tp)
        assert duplicated_gathers > 0, "control: the duplicated split all-gathers the matrix"
        assert gathered.call_count == duplicated_gathers, "the distributed split all-gathered"
        assert not opt._warned_qkv_split_disabled, "the distributed split fell back"
        assert [call.args[0] for call in steps.call_args_list] == (
            ["collective", "distributed_ns"] if q_only else ["distributed_ns"] * len(_SPLIT)
        )

        ref = torch.nn.functional.pad(_reference_split_orth(opt, w, pgc.tp), (0, 0, 0, pad_length))
        torch.testing.assert_close(out, _shard_of(ref, pgc.gtp_remat), atol=_ATOL, rtol=_RTOL)
        row = _shard_of(torch.arange(rows + pad_length, device="cuda"), pgc.gtp_remat)
        assert torch.count_nonzero(out[row >= rows]) == 0, "padding moved"
        if q_only:
            group_size = torch.distributed.get_world_size(group=pgc.gtp_remat)
            assert [call.args for call in opt._resolve_tp_mode.call_args_list] == [
                (piece, _K, group_size) for piece in piece_rows
            ]
            kv = (row < rows) & (row % _GROUP >= _SPLIT[0])  # [q|k|v] in every query group
            assert torch.equal(out[kv], duplicated[kv]), "K/V rows differ from duplicated mode"


def _worker_auto_every_piece_duplicated(rank, world_size, port):
    """Where auto's cost model keeps every piece duplicated, the weight gets the duplicated
    split, bitwise: the chain does not run and nothing falls back to whole-matrix
    Newton-Schulz."""
    with _gtp_case(world_size):
        pgc = ProcessGroupCollection.use_mpu_process_groups()
        local, _, _ = _local_shard(_full_weight(), pgc.gtp_remat)
        opt = _muon_resolving(pgc, "auto", lambda m: "duplicated")

        expected = _split_orth(_make_muon(pgc), local, pgc.tp)
        with mock.patch.object(_eo_module, "run_serially", wraps=_eo_module.run_serially) as chains:
            out = _split_orth(opt, local, pgc.tp)
        assert torch.equal(out, expected)
        assert chains.call_count == 0, "the distributed split chain ran"
        assert not opt._warned_qkv_split_disabled, "the split fell back to whole-matrix NS"


def _worker_split_distributed_rejects_layouts(rank, world_size, port):
    """The distributed split decides on the parameter's layout and slices the momentum's: a
    momentum shard of another shape (here 1.5x the rows, which still divides evenly) and a row
    count the query groups do not divide are both rejected before any collective, on every
    rank."""
    with _gtp_case(world_size):
        pgc = ProcessGroupCollection.use_mpu_process_groups()
        opt = _make_muon(pgc, tp_mode="distributed")
        local, _, sp = _local_shard(_full_weight(), pgc.gtp_remat)
        momentum = torch.cat([local, local[: sp // 2]])
        with pytest.raises(RuntimeError, match="layouts to match"):
            opt.scaled_orthogonalize_fn_with_gtp_remat(
                local, momentum, pgc.tp, None, qkv_split_shapes=_SPLIT
            )

        odd = local[:-1].clone()
        odd.is_gtp_weight_remat = True
        with pytest.raises(RuntimeError, match="QKV split shape mismatch"):
            _split_orth(opt, odd, pgc.tp)


class TestGTPMuonQKVSplitDistributed:
    """Split-QKV under the distributed mode: one mode per projection piece."""

    @pytest.mark.parametrize("world_size", _GTP_WORLD_SIZES)
    @pytest.mark.parametrize(
        "mode, num_groups, padded",
        [
            ("distributed", _GROUPS, False),
            ("auto", _GROUPS, False),
            ("distributed", _PADDED_GROUPS, True),
            ("auto", 1, True),
        ],
    )
    def test_matches_tp1(self, mode, num_groups, padded, world_size):
        _requires_multi_gpu(world_size)
        _run_distributed(
            _worker_split_distributed_matches_tp1, world_size, mode, num_groups, padded
        )

    @pytest.mark.parametrize("world_size", _GTP_WORLD_SIZES)
    def test_auto_with_every_piece_duplicated_is_the_duplicated_split(self, world_size):
        _requires_multi_gpu(world_size)
        _run_distributed(_worker_auto_every_piece_duplicated, world_size)

    @pytest.mark.parametrize("world_size", _GTP_WORLD_SIZES)
    def test_rejects_mismatched_layouts(self, world_size):
        _requires_multi_gpu(world_size)
        _run_distributed(_worker_split_distributed_rejects_layouts, world_size)
