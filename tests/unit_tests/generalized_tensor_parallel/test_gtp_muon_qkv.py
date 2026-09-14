# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Split-QKV must be layout-invariant across GTP-rematerialized row shards."""

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

# These fixtures are autouse only when imported into the test module.
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: F401
    _requires_multi_gpu,
    _run_distributed,
    _torchrun_dist_init,
    reset_gtp_globals,
)

# Production [6144, 192, 192] scaled down while retaining asymmetric Q/K/V sizes.
_SPLIT = [64, 16, 16]
_GROUP = sum(_SPLIT)
_GROUPS = 2
_ROWS = _GROUPS * _GROUP
_COLS = 64
_WORLD_SIZES = [4, 8]
_ATOL = 1e-4
_RTOL = 1e-4


class _FakeGroup:
    """Stand-in for the process-group ``size`` API used by row accounting."""

    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


def _param(rows, pad=0, sharded=True):
    param = torch.nn.Parameter(torch.zeros(rows, _COLS), requires_grad=False)
    param.is_gtp_weight_remat = sharded
    if pad:
        param.pad_length = pad
    return param


def _rows(param, split, gtp_size=None):
    assert torch.distributed.is_initialized()
    group = _FakeGroup(gtp_size) if gtp_size else None
    return qkv_rows_after_gtp_gather(param, split, group)


class TestQKVRowCount:
    """QKV tagging must decide from rows across GTP shards, excluding padding."""

    @pytest.mark.parametrize("gtp_size", [1, 2, 4, 64])
    def test_decision_is_layout_invariant(self, gtp_size):
        assert _ROWS % gtp_size == 0
        assert _rows(_param(_ROWS // gtp_size), _SPLIT, gtp_size) == (
            _ROWS,
            gtp_size,
            True,
        )

    def test_local_shard_would_have_disabled(self):
        local_rows, gtp_size = 102, 64
        split = [6144, 192, 192]
        assert local_rows % sum(split) != 0
        assert _rows(_param(local_rows), split, gtp_size) == (6528, 64, True)

    def test_padding_is_excluded(self):
        rows, _, splittable = _rows(_param(25, pad=4), _SPLIT, 4)
        assert (rows, splittable) == (_GROUP, True)
        assert 100 % _GROUP != 0

    def test_non_qkv_shape_is_rejected(self):
        rows, _, splittable = _rows(_param(30), _SPLIT, 4)
        assert (rows, splittable) == (120, False)

    def test_unsharded_param_ignores_group_and_padding(self):
        assert _rows(_param(_GROUP, pad=4, sharded=False), _SPLIT, 4) == (
            _GROUP,
            1,
            True,
        )


def _make_muon(pg_collection, tp_mode="duplicated"):
    placeholder = torch.nn.Parameter(torch.zeros(1, device="cuda"))
    return TensorParallelMuon(
        params=[placeholder],
        lr=0.01,
        momentum=0.95,
        weight_decay=0.0,
        split_qkv=True,
        is_qkv_fn=lambda p: getattr(p, "is_qkv", False),
        qkv_split_shapes=_SPLIT,
        num_ns_steps=1,
        coefficient_type="quintic",
        scale_mode="spectral",
        fp32_matmul_prec="highest",
        pg_collection=pg_collection,
        tp_mode=tp_mode,
    )


def _full_weight():
    torch.manual_seed(0)
    weight = torch.randn(_ROWS, _COLS, dtype=torch.float32, device="cuda")
    torch.distributed.broadcast(weight, src=0)
    return weight


def _tag_fragmented_qkv(tensor):
    """Attach the metadata produced by optimizer tagging for a GTP row shard."""
    tensor.is_qkv = True
    tensor.is_gtp_weight_remat = True
    tensor.qkv_split_shapes = _SPLIT
    tensor.qkv_split_shapes_global = _SPLIT * _GROUPS
    tensor.qkv_split_groups_are_complete = False
    tensor.qkv_gtp_pad_length = 0
    tensor.partition_dim = 0
    return tensor


def _reference_split_orth(opt, weight, tp_group):
    """Apply projection-level NS to the interleaved [q|k|v] query groups."""
    result = torch.empty_like(weight)
    offset = 0
    for split_rows in _SPLIT:
        rows = torch.cat(
            [
                torch.arange(
                    group * _GROUP + offset,
                    group * _GROUP + offset + split_rows,
                    device=weight.device,
                )
                for group in range(_GROUPS)
            ]
        )
        result[rows] = opt.scaled_orthogonalize_fn(weight[rows].clone(), tp_group, None)
        offset += split_rows
    return result


def _init_model_parallel(gtp_remat_size):
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        gtp_remat_size=gtp_remat_size,
    )


def _local_qkv_shard(weight, pg_collection):
    gtp_size = torch.distributed.get_world_size(group=pg_collection.gtp_remat)
    gtp_rank = torch.distributed.get_rank(group=pg_collection.gtp_remat)
    shard_rows = _ROWS // gtp_size
    shard = weight[gtp_rank * shard_rows : (gtp_rank + 1) * shard_rows].clone()
    return _tag_fragmented_qkv(shard), gtp_rank, shard_rows


def _worker_split_after_gather(rank, world_size, port):
    _init_model_parallel(world_size)
    try:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        opt = _make_muon(pg_collection)
        weight = _full_weight()
        expected = _reference_split_orth(opt, weight, pg_collection.tp)
        local, gtp_rank, shard_rows = _local_qkv_shard(weight, pg_collection)

        actual = opt.orthogonalize(local, local.clone())

        expected_local = expected[gtp_rank * shard_rows : (gtp_rank + 1) * shard_rows]
        torch.testing.assert_close(actual, expected_local, atol=_ATOL, rtol=_RTOL)
    finally:
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()


def _worker_split_differs_from_whole(rank, world_size, port):
    _init_model_parallel(world_size)
    try:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        opt = _make_muon(pg_collection)
        local, _, _ = _local_qkv_shard(_full_weight(), pg_collection)

        split = opt.orthogonalize(local, local.clone())
        whole = opt.scaled_orthogonalize_fn_with_gtp_remat(local, local.clone(), pg_collection.tp, None)

        assert not torch.allclose(split, whole, atol=_ATOL, rtol=_RTOL)
    finally:
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()


def _worker_distributed_fallback_warns_once(rank, world_size, port):
    _init_model_parallel(world_size)
    try:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        weight = _full_weight()
        duplicated_param, _, _ = _local_qkv_shard(weight, pg_collection)
        distributed_param = _tag_fragmented_qkv(duplicated_param.clone())
        distributed_param._debug_name = "decoder.layers.0.self_attention.linear_qkv.weight"

        duplicated = _make_muon(pg_collection, tp_mode="duplicated").orthogonalize(
            duplicated_param, duplicated_param.clone()
        )
        distributed_opt = _make_muon(pg_collection, tp_mode="distributed")
        with mock.patch.object(_eo_module, "log_single_rank") as logged:
            distributed = distributed_opt.orthogonalize(distributed_param, distributed_param.clone())
            distributed_opt.orthogonalize(distributed_param, distributed_param.clone())

        torch.testing.assert_close(distributed, duplicated, atol=_ATOL, rtol=_RTOL)
        warnings = [call for call in logged.call_args_list if call.args[1] == logging.WARNING]
        assert len(warnings) == 1
        assert "--muon-tp-mode duplicated" in warnings[0].args[2]
        assert distributed_param._debug_name in warnings[0].args[2]
    finally:
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()


def _worker_auto_mode_forces_duplicated_for_split(rank, world_size, port):
    """Reconstructed QKV under ``auto`` must use duplicated, non-TP NS."""
    _init_model_parallel(world_size)
    try:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        auto_opt = _make_muon(pg_collection, tp_mode="auto")
        weight = _full_weight()
        expected = _reference_split_orth(_make_muon(pg_collection), weight, pg_collection.tp)
        local, gtp_rank, shard_rows = _local_qkv_shard(weight, pg_collection)

        actual = auto_opt.orthogonalize(local, local.clone())

        assert not auto_opt._warned_distributed_qkv_fallback
        expected_local = expected[gtp_rank * shard_rows : (gtp_rank + 1) * shard_rows]
        torch.testing.assert_close(actual, expected_local, atol=_ATOL, rtol=_RTOL)
    finally:
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()


def _worker_partial_pg_collection(rank, world_size, port):
    _init_model_parallel(world_size)
    try:
        full = ProcessGroupCollection.use_mpu_process_groups()
        partial = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp"])
        assert partial.gtp_remat is None
        assert resolve_gtp_remat_group(partial, is_expert=False) is not None

        weight = _full_weight()
        full_param, _, _ = _local_qkv_shard(weight, full)
        partial_param = _tag_fragmented_qkv(full_param.clone())
        out_full = _make_muon(full).orthogonalize(full_param, full_param.clone())
        out_partial = _make_muon(partial).orthogonalize(partial_param, partial_param.clone())

        assert torch.equal(out_full, out_partial)
    finally:
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()


class TestGTPMuonQKVSplit:
    """The split happens after GTP reconstruction and then restores the local shard."""

    @pytest.mark.parametrize("world_size", _WORLD_SIZES)
    def test_split_after_gather_matches_tp1(self, world_size):
        _requires_multi_gpu(world_size)
        _run_distributed(_worker_split_after_gather, world_size)

    @pytest.mark.parametrize("world_size", _WORLD_SIZES)
    def test_gtp_split_differs_from_whole_matrix(self, world_size):
        _requires_multi_gpu(world_size)
        _run_distributed(_worker_split_differs_from_whole, world_size)

    @pytest.mark.parametrize("world_size", _WORLD_SIZES)
    def test_distributed_fallback_warns_once(self, world_size):
        _requires_multi_gpu(world_size)
        _run_distributed(_worker_distributed_fallback_warns_once, world_size)

    @pytest.mark.parametrize("world_size", _WORLD_SIZES)
    def test_auto_mode_forces_duplicated_for_split(self, world_size):
        _requires_multi_gpu(world_size)
        _run_distributed(_worker_auto_mode_forces_duplicated_for_split, world_size)

    @pytest.mark.parametrize("world_size", _WORLD_SIZES)
    def test_partial_pg_collection_matches_full(self, world_size):
        _requires_multi_gpu(world_size)
        _run_distributed(_worker_partial_pg_collection, world_size)
