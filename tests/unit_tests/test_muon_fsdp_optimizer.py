# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for Muon + Megatron-FSDP integration."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from packaging.version import Version

from megatron.core.optimizer import _get_mfsdp_models
from megatron.core.optimizer.clip_grads import get_grad_norm_fp32
from megatron.core.optimizer.emerging_optimizers import (
    HAVE_EMERGING_OPTIMIZERS,
    FSDPTensorParallelMuon,
    TensorParallelMuon,
)
from tests.unit_tests.test_utilities import Utils

pytestmark = [
    pytest.mark.skipif(
        Version(os.getenv("NVIDIA_PYTORCH_VERSION", "99.99")) <= Version("25.05"),
        reason="Skip Muon FSDP optimizer tests on LTS containers",
    ),
    pytest.mark.skipif(
        not HAVE_EMERGING_OPTIMIZERS, reason="Muon tests require the emerging-optimizers package"
    ),
]

WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))


class _FakeFSDP:
    def __init__(self, module):
        self.module = module


def _skip_if_single_rank():
    return pytest.mark.skipif(WORLD_SIZE <= 1, reason="Multi-rank test requires WORLD_SIZE > 1")


def _skip_if_no_dtensor():
    return pytest.mark.skipif(
        Version(torch.__version__.split("+")[0]) < Version("2.4.0"),
        reason="DTensor tests require PyTorch >= 2.4.0",
    )


def _make_fsdp_muon(params, dp_group=None, **kwargs):
    defaults = dict(
        lr=0.05,
        momentum=0.0,
        nesterov=False,
        weight_decay=0.0,
        num_ns_steps=2,
        pg_collection=None,
        tp_mode="duplicated",
        split_qkv=False,
    )
    defaults.update(kwargs)
    return FSDPTensorParallelMuon(params=params, dp_group=dp_group, **defaults)


def _reference_update(full_grad, **kwargs):
    from emerging_optimizers import utils

    defaults = dict(
        lr=0.05,
        momentum=0.0,
        nesterov=False,
        weight_decay=0.0,
        num_ns_steps=2,
        pg_collection=None,
        tp_mode="duplicated",
        split_qkv=False,
    )
    defaults.update(kwargs)
    param = nn.Parameter(torch.zeros_like(full_grad))
    optimizer = TensorParallelMuon(params=[param], **defaults)
    with utils.fp32_matmul_precision(optimizer.fp32_matmul_prec):
        return optimizer.orthogonalize(param, full_grad)


def _local_rows(plan, rank):
    return plan.get(rank, 0)


def _local_slice(full_tensor, plan, rank):
    start = sum(_local_rows(plan, r) for r in range(rank))
    rows = _local_rows(plan, rank)
    return full_tensor[start : start + rows].clone()


def _make_dtensor(local_tensor, global_shape, device_mesh):
    from torch.distributed.tensor import DTensor, Shard

    from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
        update_uneven_dtensor_chunk_metadata,
    )

    global_stride = torch.empty(
        global_shape, dtype=local_tensor.dtype, device=local_tensor.device
    ).stride()
    dtensor = DTensor.from_local(
        local_tensor=local_tensor,
        device_mesh=device_mesh,
        placements=[Shard(0)],
        shape=torch.Size(global_shape),
        stride=global_stride,
        run_check=False,
    )
    update_uneven_dtensor_chunk_metadata(dtensor)
    return dtensor


def _make_hfsdp_dtensor(local_tensor, global_shape, device_mesh):
    from torch.distributed.tensor import DTensor, Shard

    from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
        update_uneven_dtensor_chunk_metadata,
    )

    global_stride = torch.empty(
        global_shape, dtype=local_tensor.dtype, device=local_tensor.device
    ).stride()
    setattr(device_mesh, "_shard_order", [1, 0])
    dtensor = DTensor.from_local(
        local_tensor=local_tensor,
        device_mesh=device_mesh,
        placements=[Shard(0), Shard(0)],
        shape=torch.Size(global_shape),
        stride=global_stride,
        run_check=False,
    )
    update_uneven_dtensor_chunk_metadata(dtensor)
    return dtensor


def _make_hsdp_dtensor(local_tensor, global_shape, device_mesh):
    from torch.distributed.tensor import DTensor, Replicate, Shard

    from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
        update_uneven_dtensor_chunk_metadata,
    )

    global_stride = torch.empty(
        global_shape, dtype=local_tensor.dtype, device=local_tensor.device
    ).stride()
    dtensor = DTensor.from_local(
        local_tensor=local_tensor,
        device_mesh=device_mesh,
        placements=[Replicate(), Shard(0)],
        shape=torch.Size(global_shape),
        stride=global_stride,
        run_check=False,
    )
    update_uneven_dtensor_chunk_metadata(dtensor)
    return dtensor


def _make_replicated_dtensor(local_tensor, global_shape, device_mesh):
    from torch.distributed.tensor import DTensor, Replicate

    from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
        update_uneven_dtensor_chunk_metadata,
    )

    global_stride = torch.empty(
        global_shape, dtype=local_tensor.dtype, device=local_tensor.device
    ).stride()
    dtensor = DTensor.from_local(
        local_tensor=local_tensor,
        device_mesh=device_mesh,
        placements=[Replicate()],
        shape=torch.Size(global_shape),
        stride=global_stride,
        run_check=False,
    )
    update_uneven_dtensor_chunk_metadata(dtensor)
    return dtensor


def _as_float(value):
    return float(value.item()) if isinstance(value, torch.Tensor) else float(value)


def _attach_fake_mfsdp_bucket_metadata(param, bucket, item_id, offset):
    bucket.item_index_map[item_id] = SimpleNamespace(
        global_data_index=offset, size=torch.Size(param.shape).numel()
    )
    param.orig_param = SimpleNamespace(_gbuf=bucket, _item_id=item_id)


def _make_fake_mfsdp_bucket(bucket_id, dp_rank, dp_size, shard_size):
    return SimpleNamespace(
        item_index_map={},
        is_data_distributed=True,
        bucket_index=SimpleNamespace(
            bucket_id=bucket_id, global_data_index=0, size=shard_size * dp_size
        ),
        shard_bucket_index=SimpleNamespace(
            bucket_id=bucket_id, global_data_index=shard_size * dp_rank, size=shard_size
        ),
    )


class TestGetMFSDPModels:
    def test_error_on_plain_module(self):
        with pytest.raises(RuntimeError, match="Could not find any MegatronFSDP"):
            _get_mfsdp_models([nn.Module()])

    def test_extracts_inner_module(self):
        inner_module = MagicMock()
        chunk = _FakeFSDP(inner_module)

        with patch("megatron.core.optimizer.FullyShardedDataParallel", _FakeFSDP):
            assert _get_mfsdp_models([chunk]) == [inner_module]

    def test_extracts_from_multiple_chunks(self):
        chunks, inner_modules = [], []
        for _ in range(3):
            inner = MagicMock()
            chunk = _FakeFSDP(inner)
            chunks.append(chunk)
            inner_modules.append(inner)

        with patch("megatron.core.optimizer.FullyShardedDataParallel", _FakeFSDP):
            assert _get_mfsdp_models(chunks) == inner_modules


@_skip_if_single_rank()
@_skip_if_no_dtensor()
class TestFSDPTensorParallelMuon:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    def test_dtensor_grad_norm_reduces_only_fsdp_shard_dims(self):
        from torch.distributed.device_mesh import init_device_mesh

        from megatron.core import parallel_state

        world_size = torch.distributed.get_world_size()
        world_rank = torch.distributed.get_rank()
        mp_group = parallel_state.get_model_parallel_group()

        fsdp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_cp",))
        rows_per_rank = 2
        cols = 3
        local = torch.full(
            (rows_per_rank, cols), float(world_rank + 1), device="cuda", dtype=torch.float32
        )
        fsdp_grad = _make_dtensor(local, (world_size * rows_per_rank, cols), fsdp_mesh)
        local_norm = torch.linalg.vector_norm(local).item()
        expected_sq = torch.linalg.vector_norm(local).pow(2)
        torch.distributed.all_reduce(expected_sq, group=fsdp_mesh.get_group("dp_cp"))
        expected = expected_sq.sqrt().item()

        observed = _as_float(get_grad_norm_fp32([fsdp_grad], grad_stats_parallel_group=mp_group))
        observed_with_duplicate_stats_group = _as_float(
            get_grad_norm_fp32([fsdp_grad], grad_stats_parallel_group=fsdp_mesh.get_group("dp_cp"))
        )

        assert observed != pytest.approx(local_norm)
        assert observed == pytest.approx(expected, rel=1e-6, abs=1e-6)
        assert observed_with_duplicate_stats_group == pytest.approx(expected, rel=1e-6, abs=1e-6)

        if world_size < 4 or world_size % 2 != 0:
            return

        outer_size = 2
        inner_size = world_size // outer_size
        hsdp_mesh = init_device_mesh(
            "cuda", (outer_size, inner_size), mesh_dim_names=("outer_fsdp_dp", "dp_cp")
        )
        inner_group = hsdp_mesh.get_group("dp_cp")
        inner_rank = torch.distributed.get_rank(inner_group)
        hsdp_local = torch.full(
            (rows_per_rank, cols), float(inner_rank + 1), device="cuda", dtype=torch.float32
        )
        hsdp_grad = _make_hsdp_dtensor(hsdp_local, (inner_size * rows_per_rank, cols), hsdp_mesh)
        hsdp_expected_sq = torch.linalg.vector_norm(hsdp_local).pow(2)
        torch.distributed.all_reduce(hsdp_expected_sq, group=inner_group)
        hsdp_expected = hsdp_expected_sq.sqrt().item()
        hsdp_world_expected_sq = torch.linalg.vector_norm(hsdp_local).pow(2)
        torch.distributed.all_reduce(hsdp_world_expected_sq)
        hsdp_world_expected = hsdp_world_expected_sq.sqrt().item()

        hsdp_observed = _as_float(
            get_grad_norm_fp32([hsdp_grad], grad_stats_parallel_group=mp_group)
        )
        hsdp_observed_with_duplicate_stats_group = _as_float(
            get_grad_norm_fp32([hsdp_grad], grad_stats_parallel_group=inner_group)
        )

        assert hsdp_observed == pytest.approx(hsdp_expected, rel=1e-6, abs=1e-6)
        assert hsdp_observed_with_duplicate_stats_group == pytest.approx(
            hsdp_expected, rel=1e-6, abs=1e-6
        )
        assert hsdp_observed != pytest.approx(hsdp_world_expected)

        hfsdp_local = torch.full(
            (rows_per_rank, cols), float(world_rank + 1), device="cuda", dtype=torch.float32
        )
        hfsdp_grad = _make_hfsdp_dtensor(hfsdp_local, (world_size * rows_per_rank, cols), hsdp_mesh)
        hfsdp_expected_sq = torch.linalg.vector_norm(hfsdp_local).pow(2)
        torch.distributed.all_reduce(hfsdp_expected_sq)
        hfsdp_expected = hfsdp_expected_sq.sqrt().item()

        hfsdp_observed = _as_float(
            get_grad_norm_fp32([hfsdp_grad], grad_stats_parallel_group=mp_group)
        )
        hfsdp_observed_with_duplicate_stats_group = _as_float(
            get_grad_norm_fp32([hfsdp_grad], grad_stats_parallel_group=inner_group)
        )

        assert hfsdp_observed == pytest.approx(hfsdp_expected, rel=1e-6, abs=1e-6)
        assert hfsdp_observed_with_duplicate_stats_group == pytest.approx(
            hfsdp_expected, rel=1e-6, abs=1e-6
        )

    def test_step_gathers_only_split_boundary_params(self):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor

        dp_size = torch.distributed.get_world_size()
        dp_rank = torch.distributed.get_rank()
        device_mesh = init_device_mesh("cuda", (dp_size,), mesh_dim_names=("dp",))
        dp_group = device_mesh.get_group("dp")
        cols = 4

        # Param 0 is fully local on rank 0. Param 1 crosses the rank 0/1
        # boundary. Param 2 is fully local on rank 1. Other ranks hold empty
        # local shards and still participate in the boundary gather for param 1.
        plans = [{0: 4}, {0: 2, 1: 4}, {1: 5}]
        full_params = [
            torch.arange(rows * cols, device="cuda", dtype=torch.float32).view(rows, cols) / 100
            for rows in (4, 6, 5)
        ]
        full_grads = [
            torch.arange(rows * cols, device="cuda", dtype=torch.float32).view(rows, cols) / 50
            + 0.1
            for rows in (4, 6, 5)
        ]

        params = []
        initial_locals = []
        for full_param, plan in zip(full_params, plans):
            local_param = _local_slice(full_param, plan, dp_rank).contiguous()
            param = nn.Parameter(_make_dtensor(local_param, full_param.shape, device_mesh))
            params.append(param)
            initial_locals.append(local_param.clone())

        for param, full_grad, plan in zip(params, full_grads, plans):
            local_grad = _local_slice(full_grad, plan, dp_rank).contiguous()
            param.grad = _make_dtensor(local_grad, full_grad.shape, device_mesh)

        optimizer = _make_fsdp_muon(params, dp_group=dp_group, fsdp_batched_all_gather=False)
        assert optimizer._get_boundary_gather_param_indices(optimizer.param_groups[0]) == {1}

        # Pre-step: grads are DTensors, non-null, non-zero.
        for idx, (param, plan) in enumerate(zip(params, plans)):
            if _local_rows(plan, dp_rank) == 0:
                continue
            assert param.grad is not None, f"param {idx} grad should not be None"
            assert isinstance(param.grad, DTensor), f"param {idx} grad should be a DTensor"
            local_grad = param.grad.to_local()
            assert local_grad.numel() > 0, f"param {idx} local grad should be non-empty"
            assert not torch.all(local_grad == 0), f"param {idx} local grad should be non-zero"
        # Boundary param (idx=1) spans two ranks: local shape != global shape.
        if dp_rank in plans[1]:
            g1 = params[1].grad
            assert tuple(g1.to_local().shape) != tuple(
                g1.shape
            ), "boundary param grad should be unevenly sharded (local != global shape)"

        real_gather = optimizer._gather_full_uneven_local_tensor_like
        gather_calls = []

        def counting_gather(dtensor_ref, local_tensor):
            gather_calls.append(tuple(dtensor_ref.shape))
            return real_gather(dtensor_ref, local_tensor)

        with patch.object(optimizer, "_gather_full_uneven_local_tensor_like", counting_gather):
            optimizer.step()

        assert gather_calls == [(6, cols)]

        # Post-step: momentum buffers are DTensors in optimizer state.
        for idx, (param, plan) in enumerate(zip(params, plans)):
            if _local_rows(plan, dp_rank) == 0:
                continue
            assert (
                param in optimizer.state and optimizer.state[param]
            ), f"param {idx} should have optimizer state after step"
            state = optimizer.state[param]
            assert "momentum_buffer" in state, f"param {idx} missing momentum_buffer in state"
            mom = state["momentum_buffer"]
            assert isinstance(mom, DTensor), f"param {idx} momentum_buffer should be a DTensor"
            local_mom = mom.to_local()
            assert local_mom.numel() > 0, f"param {idx} local momentum should be non-empty"

        lr = optimizer.param_groups[0]["lr"]
        for idx, (param, full_param, full_grad, plan) in enumerate(
            zip(params, full_params, full_grads, plans)
        ):
            expected_update = _reference_update(full_grad)
            expected_local = _local_slice(full_param - lr * expected_update, plan, dp_rank)
            local_value = param.data.to_local()

            if local_value.numel() == 0:
                assert local_value.shape[0] == 0
                continue

            torch.testing.assert_close(
                local_value,
                expected_local,
                atol=1e-5,
                rtol=1e-4,
                msg=f"param {idx} local update mismatch on rank {dp_rank}",
            )
            assert not torch.equal(local_value, initial_locals[idx])

    def test_boundary_detector_without_mfsdp_metadata_includes_all_split_params(self):
        from torch.distributed.device_mesh import init_device_mesh

        dp_size = torch.distributed.get_world_size()
        dp_rank = torch.distributed.get_rank()
        device_mesh = init_device_mesh("cuda", (dp_size,), mesh_dim_names=("dp",))
        dp_group = device_mesh.get_group("dp")
        cols = 4

        # Without M-FSDP bucket metadata, the safe fallback is to gather every
        # globally split DTensor. Real M-FSDP optimizer params exercise the
        # exact flat-interval detection below.
        plans = [{0: 2, 1: 2}, {0: 1, 1: 3}, {0: 3, 1: 1}]

        params = []
        for plan in plans:
            full_param = torch.zeros(sum(plan.values()), cols, device="cuda")
            local_param = _local_slice(full_param, plan, dp_rank).contiguous()
            params.append(nn.Parameter(_make_dtensor(local_param, full_param.shape, device_mesh)))

        optimizer = _make_fsdp_muon(params, dp_group=dp_group, fsdp_batched_all_gather=False)

        assert optimizer._get_boundary_gather_param_indices(optimizer.param_groups[0]) == {0, 1, 2}

    def test_boundary_detector_uses_mfsdp_flat_item_intervals(self):
        from torch.distributed.device_mesh import init_device_mesh

        dp_size = torch.distributed.get_world_size()
        dp_rank = torch.distributed.get_rank()
        device_mesh = init_device_mesh("cuda", (dp_size,), mesh_dim_names=("dp",))
        dp_group = device_mesh.get_group("dp")
        cols = 4

        # Param 2 is split across the rank 0/1 flat-buffer boundary. The
        # detector should select it from M-FSDP item interval metadata without
        # relying on optimizer-group first/last parameter positions.
        plans = [{0: 4}, {1: 4}, {0: 2, 1: 2}, {0: 4}, {1: 4}]
        full_params = []
        full_grads = []
        params = []
        for idx, plan in enumerate(plans):
            rows = sum(plan.values())
            full_param = (
                torch.arange(rows * cols, device="cuda", dtype=torch.float32).view(rows, cols) / 100
                + idx
            )
            full_grad = (
                torch.arange(rows * cols, device="cuda", dtype=torch.float32).view(rows, cols) / 50
                + 0.1
                + idx
            )
            full_params.append(full_param)
            full_grads.append(full_grad)
            local_param = _local_slice(full_param, plan, dp_rank).contiguous()
            params.append(nn.Parameter(_make_dtensor(local_param, full_param.shape, device_mesh)))

        for param, full_grad, plan in zip(params, full_grads, plans):
            local_grad = _local_slice(full_grad, plan, dp_rank).contiguous()
            param.grad = _make_dtensor(local_grad, full_grad.shape, device_mesh)

        buckets = [
            _make_fake_mfsdp_bucket(0, dp_rank, dp_size, shard_size=4 * cols),
            _make_fake_mfsdp_bucket(1, dp_rank, dp_size, shard_size=2 * cols),
            _make_fake_mfsdp_bucket(2, dp_rank, dp_size, shard_size=4 * cols),
        ]
        bucket_assignments = [
            (buckets[0], 0, 0),
            (buckets[0], 1, 4 * cols),
            (buckets[1], 0, 0),
            (buckets[2], 0, 0),
            (buckets[2], 1, 4 * cols),
        ]
        for param, (bucket, item_id, offset) in zip(params, bucket_assignments):
            _attach_fake_mfsdp_bucket_metadata(param, bucket, item_id, offset)

        optimizer = _make_fsdp_muon(params, dp_group=dp_group, fsdp_batched_all_gather=False)

        assert optimizer._get_boundary_gather_param_indices(optimizer.param_groups[0]) == {2}

        real_gather = optimizer._gather_full_uneven_local_tensor_like
        gather_calls = []

        def counting_gather(dtensor_ref, local_tensor):
            gather_calls.append(tuple(dtensor_ref.shape))
            return real_gather(dtensor_ref, local_tensor)

        with patch.object(optimizer, "_gather_full_uneven_local_tensor_like", counting_gather):
            optimizer.step()

        assert gather_calls == [(4, cols)]

        if params[2].data.to_local().numel() > 0:
            lr = optimizer.param_groups[0]["lr"]
            expected_update = _reference_update(full_grads[2])
            expected_local = _local_slice(full_params[2] - lr * expected_update, plans[2], dp_rank)
            torch.testing.assert_close(
                params[2].data.to_local(),
                expected_local,
                atol=1e-5,
                rtol=1e-4,
                msg=f"middle bucket-boundary param update mismatch on rank {dp_rank}",
            )

    def test_step_no_gather_when_all_params_local(self):
        from torch.distributed.device_mesh import init_device_mesh

        dp_size = torch.distributed.get_world_size()
        dp_rank = torch.distributed.get_rank()
        device_mesh = init_device_mesh("cuda", (dp_size,), mesh_dim_names=("dp",))
        dp_group = device_mesh.get_group("dp")
        cols = 4
        rows = 4

        # Each param is fully owned by exactly one rank — no boundary splits.
        plans = [{0: rows}, {1: rows}]
        params = []
        for plan in plans:
            full_param = torch.ones(rows, cols, device="cuda") * 0.1
            local_p = _local_slice(full_param, plan, dp_rank).contiguous()
            param = nn.Parameter(_make_dtensor(local_p, full_param.shape, device_mesh))
            full_grad = torch.ones(rows, cols, device="cuda") * 0.2
            local_g = _local_slice(full_grad, plan, dp_rank).contiguous()
            param.grad = _make_dtensor(local_g, full_grad.shape, device_mesh)
            params.append(param)

        optimizer = _make_fsdp_muon(params, dp_group=dp_group)

        real_gather = optimizer._gather_full_uneven_local_tensor_like
        gather_calls = []
        batch_calls = []

        def counting_gather(dtensor_ref, local_tensor):
            gather_calls.append(tuple(dtensor_ref.shape))
            return real_gather(dtensor_ref, local_tensor)

        def counting_batch_gather(items):
            batch_calls.append([tuple(value.shape) for value, _ in items])
            return []

        with (
            patch.object(optimizer, "_gather_full_uneven_local_tensor_like", counting_gather),
            patch.object(
                optimizer, "_gather_full_uneven_local_tensors_like", counting_batch_gather
            ),
        ):
            optimizer.step()

        assert (
            gather_calls == [] and batch_calls == []
        ), f"Expected no all-gathers for fully-local params, got {gather_calls}, {batch_calls}"

    def test_step_skips_boundary_ns_on_empty_local_shards(self):
        from torch.distributed.device_mesh import init_device_mesh

        dp_size = torch.distributed.get_world_size()
        dp_rank = torch.distributed.get_rank()
        device_mesh = init_device_mesh("cuda", (dp_size,), mesh_dim_names=("dp",))
        dp_group = device_mesh.get_group("dp")
        cols = 4

        plan = {0: 2, 1: 2}
        full_param = torch.arange(4 * cols, device="cuda", dtype=torch.float32).view(4, cols) / 100
        full_grad = torch.arange(4 * cols, device="cuda", dtype=torch.float32).view(4, cols) / 50

        local_param = _local_slice(full_param, plan, dp_rank).contiguous()
        param = nn.Parameter(_make_dtensor(local_param, full_param.shape, device_mesh))
        local_grad = _local_slice(full_grad, plan, dp_rank).contiguous()
        param.grad = _make_dtensor(local_grad, full_grad.shape, device_mesh)

        optimizer = _make_fsdp_muon([param], dp_group=dp_group)
        orthogonalize_calls = []
        real_orthogonalize = TensorParallelMuon.orthogonalize

        def counting_orthogonalize(self, p, grad, **kwargs):
            orthogonalize_calls.append(tuple(grad.shape))
            return real_orthogonalize(self, p, grad, **kwargs)

        with patch.object(TensorParallelMuon, "orthogonalize", counting_orthogonalize):
            optimizer.step()

        if local_param.numel() == 0:
            assert orthogonalize_calls == []
        else:
            assert orthogonalize_calls == [tuple(full_grad.shape)]

    @pytest.mark.skipif(WORLD_SIZE < 4, reason="HFSDP gather test requires at least 4 ranks")
    def test_hfsdp_boundary_gather_matches_generic_redistribute_bitwise(self):
        from torch.distributed.device_mesh import init_device_mesh

        from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
            redistribute_uneven_dtensor_to_replicated,
        )

        dp_size = torch.distributed.get_world_size()
        dp_rank = torch.distributed.get_rank()
        outer_size = 2
        inner_size = dp_size // outer_size
        if dp_size % outer_size != 0:
            pytest.skip("HFSDP gather test requires an even world size")

        device_mesh = init_device_mesh(
            "cuda", (outer_size, inner_size), mesh_dim_names=("dp_outer", "dp")
        )
        outer_rank, inner_rank = device_mesh.get_coordinate()
        logical_rank = inner_rank * outer_size + outer_rank

        rows, cols = dp_size * 3 + 5, 4
        full_update = (
            torch.arange(rows * cols, device="cuda", dtype=torch.float32).view(rows, cols) + 13
        )
        rows_per_logical_rank = [
            rows // dp_size + (1 if rank < rows % dp_size else 0) for rank in range(dp_size)
        ]
        row_start = sum(rows_per_logical_rank[:logical_rank])
        row_count = rows_per_logical_rank[logical_rank]
        local_update = full_update[row_start : row_start + row_count].contiguous()

        dtensor = _make_hfsdp_dtensor(local_update.clone(), full_update.shape, device_mesh)
        param = nn.Parameter(dtensor)
        optimizer = _make_fsdp_muon([param], dp_group=torch.distributed.group.WORLD)

        plan = optimizer._get_uneven_gather_plan(param)
        assert plan is not None
        assert plan["shard_mesh_dims"] == (0, 1)
        assert len(plan["stages"]) == 2

        gathered = optimizer._gather_full_uneven_local_tensor_like(param, local_update)
        ref_dtensor = optimizer._dtensor_from_local_like(param, local_update)
        expected = redistribute_uneven_dtensor_to_replicated(ref_dtensor)._local_tensor

        assert torch.equal(
            expected, full_update
        ), f"generic HFSDP gather mismatch on rank {dp_rank}"
        assert torch.equal(gathered, expected), f"Muon HFSDP gather mismatch on rank {dp_rank}"

    @pytest.mark.skipif(
        WORLD_SIZE < 4, reason="HFSDP batched gather test requires at least 4 ranks"
    )
    def test_hfsdp_batched_gather_uses_multi_stage_path_bitwise(self):
        from torch.distributed.device_mesh import init_device_mesh

        from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
            redistribute_uneven_dtensor_to_replicated,
        )

        dp_size = torch.distributed.get_world_size()
        outer_size = 2
        inner_size = dp_size // outer_size
        if dp_size % outer_size != 0:
            pytest.skip("HFSDP batched gather test requires an even world size")

        device_mesh = init_device_mesh(
            "cuda", (outer_size, inner_size), mesh_dim_names=("dp_outer", "dp")
        )
        outer_rank, inner_rank = device_mesh.get_coordinate()
        logical_rank = inner_rank * outer_size + outer_rank

        rows, cols = dp_size * 3 + 5, 4
        full_update = (
            torch.arange(rows * cols, device="cuda", dtype=torch.float32).view(rows, cols) + 17
        )
        rows_per_logical_rank = [
            rows // dp_size + (1 if rank < rows % dp_size else 0) for rank in range(dp_size)
        ]
        row_start = sum(rows_per_logical_rank[:logical_rank])
        row_count = rows_per_logical_rank[logical_rank]
        local_update = full_update[row_start : row_start + row_count].contiguous()

        param = nn.Parameter(
            _make_hfsdp_dtensor(local_update.clone(), full_update.shape, device_mesh)
        )
        optimizer = _make_fsdp_muon(
            [param], dp_group=torch.distributed.group.WORLD, fsdp_batched_all_gather=True
        )
        plan = optimizer._get_uneven_gather_plan(param)
        assert plan is not None
        assert len(plan["stages"]) == 2

        def fail_scalar_gather(*_args, **_kwargs):
            raise AssertionError("HFSDP batched gather should not use the scalar fallback")

        with patch.object(optimizer, "_gather_full_uneven_local_tensor_like", fail_scalar_gather):
            gathered = optimizer._gather_full_uneven_local_tensors_like([(param, local_update)])[0]
        ref_dtensor = optimizer._dtensor_from_local_like(param, local_update)
        expected = redistribute_uneven_dtensor_to_replicated(ref_dtensor)._local_tensor

        assert torch.equal(gathered, expected), "Batched HFSDP gather produced wrong tensor"

    @pytest.mark.skipif(WORLD_SIZE < 4, reason="HFSDP cache test requires at least 4 ranks")
    def test_hfsdp_boundary_metadata_collectives_are_cached_across_steps(self):
        from torch.distributed.device_mesh import init_device_mesh

        dp_size = torch.distributed.get_world_size()
        outer_size = 2
        inner_size = dp_size // outer_size
        if dp_size % outer_size != 0:
            pytest.skip("HFSDP cache test requires an even world size")

        device_mesh = init_device_mesh(
            "cuda", (outer_size, inner_size), mesh_dim_names=("dp_outer", "dp")
        )
        outer_rank, inner_rank = device_mesh.get_coordinate()
        logical_rank = inner_rank * outer_size + outer_rank

        rows, cols = dp_size * 3 + 5, 4
        full_param = (
            torch.arange(rows * cols, device="cuda", dtype=torch.float32).view(rows, cols) / 100
        )
        full_grad = (
            torch.arange(rows * cols, device="cuda", dtype=torch.float32).view(rows, cols) / 50
            + 0.1
        )
        rows_per_logical_rank = [
            rows // dp_size + (1 if rank < rows % dp_size else 0) for rank in range(dp_size)
        ]
        row_start = sum(rows_per_logical_rank[:logical_rank])
        row_count = rows_per_logical_rank[logical_rank]

        local_param = full_param[row_start : row_start + row_count].contiguous()
        param = nn.Parameter(_make_hfsdp_dtensor(local_param, full_param.shape, device_mesh))
        local_grad = full_grad[row_start : row_start + row_count].contiguous()
        param.grad = _make_hfsdp_dtensor(local_grad, full_grad.shape, device_mesh)

        optimizer = _make_fsdp_muon([param], dp_group=torch.distributed.group.WORLD)

        with patch(
            "torch.distributed.all_gather_object", wraps=torch.distributed.all_gather_object
        ) as mocked_all_gather_object:
            optimizer.step()
            first_step_calls = mocked_all_gather_object.call_count
            optimizer.step()

        assert first_step_calls > 0
        assert mocked_all_gather_object.call_count == first_step_calls

    @pytest.mark.skipif(WORLD_SIZE < 4, reason="HFSDP boundary test requires at least 4 ranks")
    def test_hfsdp_boundary_detector_uses_local_split_union_when_flat_metadata_misses_outer_split(
        self,
    ):
        from torch.distributed.device_mesh import init_device_mesh

        dp_size = torch.distributed.get_world_size()
        outer_size = 2
        inner_size = dp_size // outer_size
        if dp_size % outer_size != 0:
            pytest.skip("HFSDP boundary test requires an even world size")

        device_mesh = init_device_mesh(
            "cuda", (outer_size, inner_size), mesh_dim_names=("dp_outer", "dp")
        )
        outer_rank, inner_rank = device_mesh.get_coordinate()
        logical_rank = inner_rank * outer_size + outer_rank

        rows, cols = dp_size * 2 + 1, 4
        full_param = torch.zeros(rows, cols, device="cuda")
        rows_per_logical_rank = [
            rows // dp_size + (1 if rank < rows % dp_size else 0) for rank in range(dp_size)
        ]
        row_start = sum(rows_per_logical_rank[:logical_rank])
        row_count = rows_per_logical_rank[logical_rank]
        local_param = full_param[row_start : row_start + row_count].contiguous()
        param = nn.Parameter(_make_hfsdp_dtensor(local_param, full_param.shape, device_mesh))

        # The fake flat bucket says the item is wholly inside one shard. HFSDP
        # still exposes a partial local DTensor shard, so Muon must gather it.
        bucket = _make_fake_mfsdp_bucket(
            bucket_id=0,
            dp_rank=torch.distributed.get_rank(),
            dp_size=dp_size,
            shard_size=full_param.numel() * 2,
        )
        _attach_fake_mfsdp_bucket_metadata(param, bucket, item_id=0, offset=0)

        optimizer = _make_fsdp_muon([param], dp_group=torch.distributed.group.WORLD)

        assert optimizer._mfsdp_param_crosses_shard_boundary(param, 0) is False
        assert optimizer._get_boundary_gather_param_indices(optimizer.param_groups[0]) == {0}

    def test_step_batches_multiple_split_boundary_params(self):
        from torch.distributed.device_mesh import init_device_mesh

        dp_size = torch.distributed.get_world_size()
        dp_rank = torch.distributed.get_rank()
        device_mesh = init_device_mesh("cuda", (dp_size,), mesh_dim_names=("dp",))
        dp_group = device_mesh.get_group("dp")
        cols = 4

        plans = [{0: 2, 1: 2}, {0: 4}, {0: 3, 1: 1}]
        full_params = [
            torch.arange(rows * cols, device="cuda", dtype=torch.float32).view(rows, cols) / 100
            for rows in (4, 4, 4)
        ]
        full_grads = [
            torch.arange(rows * cols, device="cuda", dtype=torch.float32).view(rows, cols) / 50
            + 0.1
            for rows in (4, 4, 4)
        ]

        params = []
        for full_param, plan in zip(full_params, plans):
            local_param = _local_slice(full_param, plan, dp_rank).contiguous()
            params.append(nn.Parameter(_make_dtensor(local_param, full_param.shape, device_mesh)))

        for param, full_grad, plan in zip(params, full_grads, plans):
            local_grad = _local_slice(full_grad, plan, dp_rank).contiguous()
            param.grad = _make_dtensor(local_grad, full_grad.shape, device_mesh)

        optimizer = _make_fsdp_muon(params, dp_group=dp_group, fsdp_batched_all_gather=True)
        assert optimizer._get_boundary_gather_param_indices(optimizer.param_groups[0]) == {0, 2}

        real_gather = optimizer._gather_full_uneven_local_tensors_like
        gather_batches = []

        def counting_gather(items):
            gather_batches.append([tuple(value.shape) for value, _ in items])
            return real_gather(items)

        with patch.object(optimizer, "_gather_full_uneven_local_tensors_like", counting_gather):
            optimizer.step()

        assert gather_batches == [[(4, cols), (4, cols)]]

    def test_overlap_path_starts_boundary_gather_before_local_updates(self):
        from torch.distributed.device_mesh import init_device_mesh

        dp_size = torch.distributed.get_world_size()
        dp_rank = torch.distributed.get_rank()
        device_mesh = init_device_mesh("cuda", (dp_size,), mesh_dim_names=("dp",))
        dp_group = device_mesh.get_group("dp")
        cols = 4

        small_param = torch.arange(2 * cols, device="cuda", dtype=torch.float32).view(2, cols) / 100
        small_grad = small_param + 0.2
        plan = {0: 2, 1: 4}
        boundary_param = (
            torch.arange(6 * cols, device="cuda", dtype=torch.float32).view(6, cols) / 80
        )
        boundary_grad = boundary_param + 0.3
        large_param = torch.arange(4 * cols, device="cuda", dtype=torch.float32).view(4, cols) / 70
        large_grad = large_param + 0.4

        local_boundary_param = _local_slice(boundary_param, plan, dp_rank).contiguous()
        params = [
            nn.Parameter(
                _make_replicated_dtensor(small_param.clone(), small_param.shape, device_mesh)
            ),
            nn.Parameter(_make_dtensor(local_boundary_param, boundary_param.shape, device_mesh)),
            nn.Parameter(
                _make_replicated_dtensor(large_param.clone(), large_param.shape, device_mesh)
            ),
        ]
        params[0].grad = _make_replicated_dtensor(small_grad.clone(), small_grad.shape, device_mesh)
        params[1].grad = _make_dtensor(
            _local_slice(boundary_grad, plan, dp_rank).contiguous(),
            boundary_grad.shape,
            device_mesh,
        )
        params[2].grad = _make_replicated_dtensor(large_grad.clone(), large_grad.shape, device_mesh)

        optimizer = _make_fsdp_muon(
            params,
            dp_group=dp_group,
            fsdp_batched_all_gather=True,
            fsdp_reuse_gather_scratch=True,
            fsdp_padded_all_gather=True,
            fsdp_padded_all_gather_pad_factor=1000,
            fsdp_overlap_comm_compute=True,
        )
        events = []
        real_compute = optimizer._compute_local_pre_ns_grad
        real_start = optimizer._start_gather_full_uneven_local_tensor_batch_async
        real_finish = optimizer._finish_gather_full_uneven_local_tensor_batch_async
        real_apply = optimizer._apply_precomputed_muon_update

        def recording_compute(p, group, lr):
            if p is params[1]:
                events.append("compute_boundary")
            elif p is params[2]:
                events.append("compute_large_local")
            else:
                events.append("compute_small_local")
            return real_compute(p, group, lr)

        def recording_start(*args, **kwargs):
            events.append("start")
            return real_start(*args, **kwargs)

        def recording_finish(*args, **kwargs):
            events.append("finish")
            return real_finish(*args, **kwargs)

        def recording_apply(p, pre_ns_grad, is_gathered, lr, group_kwargs):
            if is_gathered:
                events.append("apply_gathered")
            elif p is params[2]:
                events.append("apply_large_local")
            else:
                events.append("apply_small_local")
            return real_apply(p, pre_ns_grad, is_gathered, lr, group_kwargs)

        with (
            patch.object(optimizer, "_compute_local_pre_ns_grad", recording_compute),
            patch.object(
                optimizer, "_start_gather_full_uneven_local_tensor_batch_async", recording_start
            ),
            patch.object(
                optimizer, "_finish_gather_full_uneven_local_tensor_batch_async", recording_finish
            ),
            patch.object(optimizer, "_apply_precomputed_muon_update", recording_apply),
        ):
            optimizer.step()

        assert events.index("compute_boundary") < events.index("start")
        assert events.index("start") < events.index("compute_small_local")
        assert events.index("start") < events.index("compute_large_local")
        assert events.index("start") < events.index("apply_large_local") < events.index("finish")
        assert events.index("apply_large_local") < events.index("apply_small_local")
        assert events.index("finish") < events.index("apply_gathered")

    def test_padded_batch_gather_matches_uneven_batch_and_reuses_scratch(self):
        from torch.distributed.device_mesh import init_device_mesh

        dp_size = torch.distributed.get_world_size()
        dp_rank = torch.distributed.get_rank()
        device_mesh = init_device_mesh("cuda", (dp_size,), mesh_dim_names=("dp",))
        dp_group = device_mesh.get_group("dp")
        cols = 4

        plans = [{0: 2, 1: 3}, {0: 1, 1: 4}]
        full_grads = [
            torch.arange(rows * cols, device="cuda", dtype=torch.float32).view(rows, cols) + idx
            for idx, rows in enumerate([5, 5])
        ]
        full_grads_second = [
            full_grad + 1000 * (idx + 1) for idx, full_grad in enumerate(full_grads)
        ]
        params = []
        items = []
        items_second = []
        for full_grad, plan in zip(full_grads, plans):
            local_value = _local_slice(torch.zeros_like(full_grad), plan, dp_rank).contiguous()
            param = nn.Parameter(_make_dtensor(local_value, full_grad.shape, device_mesh))
            local_grad = _local_slice(full_grad, plan, dp_rank).contiguous()
            items.append((param, local_grad))
            params.append(param)
        for param, full_grad, plan in zip(params, full_grads_second, plans):
            local_grad = _local_slice(full_grad, plan, dp_rank).contiguous()
            items_second.append((param, local_grad))

        uneven_optimizer = _make_fsdp_muon(
            params,
            dp_group=dp_group,
            fsdp_batched_all_gather=True,
            fsdp_padded_all_gather=False,
            fsdp_padded_all_gather_pad_factor=1000,
        )
        uneven_results = uneven_optimizer._gather_full_uneven_local_tensors_like(items)

        padded_optimizer = _make_fsdp_muon(
            params,
            dp_group=dp_group,
            fsdp_batched_all_gather=True,
            fsdp_padded_all_gather=True,
            fsdp_padded_all_gather_pad_factor=1000,
            fsdp_reuse_gather_scratch=True,
        )
        padded_calls = []
        real_prepare_padded_buffers = padded_optimizer._prepare_padded_all_gather_buffers

        def counting_prepare_padded_buffers(*args, **kwargs):
            padded_calls.append(args[1])
            return real_prepare_padded_buffers(*args, **kwargs)

        with patch.object(
            padded_optimizer, "_prepare_padded_all_gather_buffers", counting_prepare_padded_buffers
        ):
            padded_results = padded_optimizer._gather_full_uneven_local_tensors_like(items)
            scratch_bytes_after_first = padded_optimizer._fsdp_gather_scratch_cache_bytes()
            padded_results_second = padded_optimizer._gather_full_uneven_local_tensors_like(
                items_second
            )

        assert len(padded_calls) == 2
        assert padded_optimizer._fsdp_gather_scratch_cache_bytes() == scratch_bytes_after_first

        for idx, (
            padded,
            padded_second,
            uneven,
            full_grad,
            full_grad_second,
            (_, local_grad),
        ) in enumerate(
            zip(
                padded_results,
                padded_results_second,
                uneven_results,
                full_grads,
                full_grads_second,
                items,
            )
        ):
            if local_grad.numel() == 0:
                assert padded is None
                assert padded_second is None
                assert uneven is None
                continue
            torch.testing.assert_close(padded, full_grad, atol=0, rtol=0)
            torch.testing.assert_close(padded_second, full_grad_second, atol=0, rtol=0)
            torch.testing.assert_close(padded, uneven, atol=0, rtol=0, msg=f"param {idx}")

    def test_gather_scratch_reuse_can_be_disabled(self):
        from torch.distributed.device_mesh import init_device_mesh

        dp_size = torch.distributed.get_world_size()
        dp_rank = torch.distributed.get_rank()
        device_mesh = init_device_mesh("cuda", (dp_size,), mesh_dim_names=("dp",))
        dp_group = device_mesh.get_group("dp")
        cols = 4

        plan = {0: 2, 1: 3}
        full_grad = torch.arange(5 * cols, device="cuda", dtype=torch.float32).view(5, cols)
        local_value = _local_slice(torch.zeros_like(full_grad), plan, dp_rank).contiguous()
        param = nn.Parameter(_make_dtensor(local_value, full_grad.shape, device_mesh))
        local_grad = _local_slice(full_grad, plan, dp_rank).contiguous()

        optimizer = _make_fsdp_muon(
            [param],
            dp_group=dp_group,
            fsdp_batched_all_gather=True,
            fsdp_padded_all_gather=True,
            fsdp_padded_all_gather_pad_factor=1000,
            fsdp_reuse_gather_scratch=False,
        )
        result = optimizer._gather_full_uneven_local_tensors_like([(param, local_grad)])[0]

        assert optimizer._fsdp_gather_scratch_cache_bytes() == 0
        if local_grad.numel() == 0:
            assert result is None
        else:
            torch.testing.assert_close(result, full_grad, atol=0, rtol=0)

    def test_padded_batch_gather_respects_padding_threshold(self):
        from torch.distributed.device_mesh import init_device_mesh

        dp_size = torch.distributed.get_world_size()
        dp_rank = torch.distributed.get_rank()
        device_mesh = init_device_mesh("cuda", (dp_size,), mesh_dim_names=("dp",))
        dp_group = device_mesh.get_group("dp")
        cols = 4

        plan = {0: 1, 1: 5}
        full_grad = torch.arange(6 * cols, device="cuda", dtype=torch.float32).view(6, cols)
        local_value = _local_slice(torch.zeros_like(full_grad), plan, dp_rank).contiguous()
        param = nn.Parameter(_make_dtensor(local_value, full_grad.shape, device_mesh))
        local_grad = _local_slice(full_grad, plan, dp_rank).contiguous()

        optimizer = _make_fsdp_muon(
            [param],
            dp_group=dp_group,
            fsdp_batched_all_gather=True,
            fsdp_padded_all_gather=True,
            fsdp_padded_all_gather_pad_factor=1.0,
        )

        def fail_prepare_padded_buffers(*_args, **_kwargs):
            raise AssertionError("padding threshold should select the uneven gather path")

        with patch.object(
            optimizer, "_prepare_padded_all_gather_buffers", fail_prepare_padded_buffers
        ):
            result = optimizer._gather_full_uneven_local_tensors_like([(param, local_grad)])[0]

        if local_grad.numel() == 0:
            assert result is None
        else:
            torch.testing.assert_close(result, full_grad, atol=0, rtol=0)

    def test_batched_gather_plan_none_falls_back_to_replicated_dtensor(self):
        from torch.distributed.device_mesh import init_device_mesh

        dp_size = torch.distributed.get_world_size()
        device_mesh = init_device_mesh("cuda", (dp_size,), mesh_dim_names=("dp",))
        dp_group = device_mesh.get_group("dp")
        full_grad = torch.arange(5 * 4, device="cuda", dtype=torch.float32).view(5, 4)
        param = nn.Parameter(
            _make_replicated_dtensor(full_grad.clone(), full_grad.shape, device_mesh)
        )

        optimizer = _make_fsdp_muon([param], dp_group=dp_group, fsdp_batched_all_gather=True)
        result = optimizer._gather_full_uneven_local_tensors_like([(param, full_grad.clone())])[0]

        torch.testing.assert_close(result, full_grad, atol=0, rtol=0)

    def test_batched_gather_partitions_by_dtype(self):
        from torch.distributed.device_mesh import init_device_mesh

        dp_size = torch.distributed.get_world_size()
        dp_rank = torch.distributed.get_rank()
        device_mesh = init_device_mesh("cuda", (dp_size,), mesh_dim_names=("dp",))
        dp_group = device_mesh.get_group("dp")
        cols = 4
        plan = {0: 2, 1: 3}
        dtypes = [torch.float32, torch.bfloat16]
        full_grads = [
            (torch.arange(5 * cols, device="cuda", dtype=torch.float32).view(5, cols) + idx).to(
                dtype=dtype
            )
            for idx, dtype in enumerate(dtypes)
        ]
        params = []
        items = []
        for full_grad in full_grads:
            local_value = _local_slice(torch.zeros_like(full_grad), plan, dp_rank).contiguous()
            param = nn.Parameter(_make_dtensor(local_value, full_grad.shape, device_mesh))
            local_grad = _local_slice(full_grad, plan, dp_rank).contiguous()
            params.append(param)
            items.append((param, local_grad))

        optimizer = _make_fsdp_muon(
            params,
            dp_group=dp_group,
            fsdp_batched_all_gather=True,
            fsdp_padded_all_gather=True,
            fsdp_padded_all_gather_pad_factor=1000,
        )
        batch_sizes = []
        real_batch_gather = optimizer._gather_full_uneven_local_tensor_batch

        def counting_batch_gather(items_arg, results_arg, batch):
            batch_sizes.append(len(batch["item_indices"]))
            return real_batch_gather(items_arg, results_arg, batch)

        with patch.object(
            optimizer, "_gather_full_uneven_local_tensor_batch", counting_batch_gather
        ):
            results = optimizer._gather_full_uneven_local_tensors_like(items)

        assert batch_sizes == [1, 1]
        for result, full_grad, (_, local_grad) in zip(results, full_grads, items):
            if local_grad.numel() == 0:
                assert result is None
                continue
            assert result.dtype == full_grad.dtype
            torch.testing.assert_close(result, full_grad, atol=0, rtol=0)

    def test_batched_gather_splits_on_gather_byte_budget(self):
        from torch.distributed.device_mesh import init_device_mesh

        dp_size = torch.distributed.get_world_size()
        dp_rank = torch.distributed.get_rank()
        device_mesh = init_device_mesh("cuda", (dp_size,), mesh_dim_names=("dp",))
        dp_group = device_mesh.get_group("dp")
        cols = 4

        plans = [{0: 2, 1: 3}, {0: 2, 1: 3}]
        full_grads = [
            torch.arange(5 * cols, device="cuda", dtype=torch.float32).view(5, cols) + idx
            for idx in range(2)
        ]
        params = []
        items = []
        for full_grad, plan in zip(full_grads, plans):
            local_value = _local_slice(torch.zeros_like(full_grad), plan, dp_rank).contiguous()
            param = nn.Parameter(_make_dtensor(local_value, full_grad.shape, device_mesh))
            local_grad = _local_slice(full_grad, plan, dp_rank).contiguous()
            params.append(param)
            items.append((param, local_grad))

        optimizer = _make_fsdp_muon(
            params,
            dp_group=dp_group,
            fsdp_batched_all_gather=True,
            fsdp_padded_all_gather=True,
            fsdp_padded_all_gather_pad_factor=1000,
            fsdp_batch_max_gather_bytes=64,
        )
        batch_sizes = []
        real_batch_gather = optimizer._gather_full_uneven_local_tensor_batch

        def counting_batch_gather(items_arg, results_arg, batch):
            batch_sizes.append(len(batch["item_indices"]))
            return real_batch_gather(items_arg, results_arg, batch)

        with patch.object(
            optimizer, "_gather_full_uneven_local_tensor_batch", counting_batch_gather
        ):
            results = optimizer._gather_full_uneven_local_tensors_like(items)

        assert batch_sizes == [1, 1]
        for result, full_grad, (_, local_grad) in zip(results, full_grads, items):
            if local_grad.numel() == 0:
                assert result is None
            else:
                torch.testing.assert_close(result, full_grad, atol=0, rtol=0)

    def test_boundary_metadata_collectives_are_cached_across_steps(self):
        from torch.distributed.device_mesh import init_device_mesh

        dp_size = torch.distributed.get_world_size()
        dp_rank = torch.distributed.get_rank()
        device_mesh = init_device_mesh("cuda", (dp_size,), mesh_dim_names=("dp",))
        dp_group = device_mesh.get_group("dp")
        cols = 4

        plans = [{0: 4}, {0: 2, 1: 4}, {1: 5}]
        full_params = [
            torch.arange(rows * cols, device="cuda", dtype=torch.float32).view(rows, cols) / 100
            for rows in (4, 6, 5)
        ]
        full_grads = [
            torch.arange(rows * cols, device="cuda", dtype=torch.float32).view(rows, cols) / 50
            + 0.1
            for rows in (4, 6, 5)
        ]

        params = []
        for full_param, plan in zip(full_params, plans):
            local_param = _local_slice(full_param, plan, dp_rank).contiguous()
            params.append(nn.Parameter(_make_dtensor(local_param, full_param.shape, device_mesh)))

        for param, full_grad, plan in zip(params, full_grads, plans):
            local_grad = _local_slice(full_grad, plan, dp_rank).contiguous()
            param.grad = _make_dtensor(local_grad, full_grad.shape, device_mesh)

        optimizer = _make_fsdp_muon(params, dp_group=dp_group)

        with patch(
            "torch.distributed.all_gather_object", wraps=torch.distributed.all_gather_object
        ) as mocked_all_gather_object:
            optimizer.step()
            first_step_calls = mocked_all_gather_object.call_count
            optimizer.step()

        assert first_step_calls > 0
        assert mocked_all_gather_object.call_count == first_step_calls

    def test_step_momentum_accumulates_across_steps(self):
        from torch.distributed.device_mesh import init_device_mesh

        dp_size = torch.distributed.get_world_size()
        dp_rank = torch.distributed.get_rank()
        device_mesh = init_device_mesh("cuda", (dp_size,), mesh_dim_names=("dp",))
        dp_group = device_mesh.get_group("dp")
        cols = 4
        rows = 4
        m = 0.95

        plan = {0: rows}
        full_param = torch.zeros(rows, cols, device="cuda")
        local_p = _local_slice(full_param, plan, dp_rank).contiguous()
        param = nn.Parameter(_make_dtensor(local_p, full_param.shape, device_mesh))

        optimizer = _make_fsdp_muon([param], dp_group=dp_group, momentum=m, nesterov=False)

        grad1 = torch.full((rows, cols), 0.5, device="cuda")
        grad2 = torch.full((rows, cols), 1.0, device="cuda")

        # Step 1.
        local_g1 = _local_slice(grad1, plan, dp_rank).contiguous()
        param.grad = _make_dtensor(local_g1, grad1.shape, device_mesh)
        optimizer.step()

        if local_p.numel() > 0:
            mom = optimizer.state[param]["momentum_buffer"].to_local()
            # buf = 0 * m + grad1 * (1 - m)
            torch.testing.assert_close(mom, (1 - m) * local_g1, atol=1e-6, rtol=0)

        # Step 2 with a different gradient.
        local_g2 = _local_slice(grad2, plan, dp_rank).contiguous()
        param.grad = _make_dtensor(local_g2, grad2.shape, device_mesh)
        optimizer.step()

        if local_p.numel() > 0:
            mom = optimizer.state[param]["momentum_buffer"].to_local()
            expected = m * (1 - m) * local_g1 + (1 - m) * local_g2
            torch.testing.assert_close(mom, expected, atol=1e-6, rtol=0)

    def test_step_none_grad_boundary_param_does_not_crash(self):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor

        dp_size = torch.distributed.get_world_size()
        dp_rank = torch.distributed.get_rank()
        device_mesh = init_device_mesh("cuda", (dp_size,), mesh_dim_names=("dp",))
        dp_group = device_mesh.get_group("dp")
        cols = 4

        # param 0: fully local on rank 0; param 1: boundary split across ranks 0 and 1.
        plans = [{0: 4}, {0: 2, 1: 4}]
        params = []
        for plan in plans:
            rows = sum(plan.values())
            full_param = torch.ones(rows, cols, device="cuda") * 0.1
            local_p = _local_slice(full_param, plan, dp_rank).contiguous()
            params.append(nn.Parameter(_make_dtensor(local_p, full_param.shape, device_mesh)))

        # Give param 0 a gradient; leave param 1 (the boundary param) with grad=None.
        full_grad0 = torch.ones(4, cols, device="cuda") * 0.3
        local_g0 = _local_slice(full_grad0, plans[0], dp_rank).contiguous()
        params[0].grad = _make_dtensor(local_g0, full_grad0.shape, device_mesh)
        # params[1].grad remains None

        initial_local1 = params[1].data.to_local().clone()

        # Must not raise even though the boundary param has no gradient.
        optimizer = _make_fsdp_muon(params, dp_group=dp_group)
        optimizer.step()

        # Boundary param with grad=None should not receive a meaningful weight update
        # (NS of an all-zeros update produces ~zero; the param is essentially unchanged).
        final_local1 = params[1].data.to_local()
        torch.testing.assert_close(final_local1, initial_local1, atol=1e-4, rtol=1e-3)

    @pytest.mark.parametrize("overlap_comm_compute", [False, True])
    def test_step_matches_unsharded_muon_optimizer_step_numerics(self, overlap_comm_compute):
        from torch.distributed.device_mesh import init_device_mesh

        dp_size = torch.distributed.get_world_size()
        dp_rank = torch.distributed.get_rank()
        device_mesh = init_device_mesh("cuda", (dp_size,), mesh_dim_names=("dp",))
        dp_group = device_mesh.get_group("dp")
        cols = 4

        plans = [{0: 4}, {0: 2, 1: 3}, {0: 1, 1: 5}, {1: 4}]
        row_counts = [sum(plan.values()) for plan in plans]
        full_params = [
            (
                torch.arange(rows * cols, device="cuda", dtype=torch.float32).view(rows, cols)
                + 1.3 * (idx + 1)
            )
            / (17 + idx)
            for idx, rows in enumerate(row_counts)
        ]
        full_grads_by_step = [
            [
                (
                    torch.arange(rows * cols, device="cuda", dtype=torch.float32).view(rows, cols)
                    + 0.7 * (idx + 1)
                    + step
                )
                / (23 + idx + step)
                for idx, rows in enumerate(row_counts)
            ]
            for step in range(2)
        ]
        optimizer_kwargs = dict(
            lr=0.03, momentum=0.25, nesterov=True, weight_decay=0.02, num_ns_steps=3
        )

        unsharded_params = [nn.Parameter(full_param.clone()) for full_param in full_params]
        unsharded_optimizer = TensorParallelMuon(
            params=unsharded_params,
            pg_collection=None,
            tp_mode="duplicated",
            split_qkv=False,
            **optimizer_kwargs,
        )

        sharded_params = []
        for full_param, plan in zip(full_params, plans):
            local_param = _local_slice(full_param, plan, dp_rank).contiguous()
            sharded_params.append(
                nn.Parameter(_make_dtensor(local_param, full_param.shape, device_mesh))
            )
        sharded_optimizer_kwargs = dict(optimizer_kwargs)
        if overlap_comm_compute:
            sharded_optimizer_kwargs.update(
                fsdp_batched_all_gather=True,
                fsdp_reuse_gather_scratch=True,
                fsdp_padded_all_gather=True,
                fsdp_padded_all_gather_pad_factor=1000,
                fsdp_overlap_comm_compute=True,
            )
        sharded_optimizer = _make_fsdp_muon(
            sharded_params, dp_group=dp_group, **sharded_optimizer_kwargs
        )

        for step, full_grads in enumerate(full_grads_by_step, start=1):
            for param, full_grad in zip(unsharded_params, full_grads):
                param.grad = full_grad.clone()
            for param, full_grad, plan in zip(sharded_params, full_grads, plans):
                local_grad = _local_slice(full_grad, plan, dp_rank).contiguous()
                param.grad = _make_dtensor(local_grad, full_grad.shape, device_mesh)

            unsharded_optimizer.step()
            sharded_optimizer.step()

            for idx, (sharded_param, unsharded_param, plan) in enumerate(
                zip(sharded_params, unsharded_params, plans)
            ):
                expected_local = _local_slice(unsharded_param.detach(), plan, dp_rank)
                local_value = sharded_param.data.to_local()

                if local_value.numel() == 0:
                    assert local_value.shape[0] == 0
                    continue

                torch.testing.assert_close(
                    local_value,
                    expected_local,
                    atol=1e-5,
                    rtol=1e-4,
                    msg=(
                        "Muon+M-FSDP diverged from unsharded Muon "
                        f"on step {step}, param {idx}, rank {dp_rank}"
                    ),
                )

    @pytest.mark.skipif(WORLD_SIZE < 4, reason="Hybrid topology numerics test requires 4+ ranks")
    @pytest.mark.parametrize("topology", ["hsdp", "hfsdp"])
    @pytest.mark.parametrize("overlap_comm_compute", [False, True])
    def test_hybrid_step_matches_unsharded_muon_optimizer_step_numerics(
        self, topology, overlap_comm_compute
    ):
        from torch.distributed.device_mesh import init_device_mesh

        dp_size = torch.distributed.get_world_size()
        world_rank = torch.distributed.get_rank()
        outer_size = 2
        inner_size = dp_size // outer_size
        if dp_size % outer_size != 0:
            pytest.skip("Hybrid topology numerics test requires an even world size")

        device_mesh = init_device_mesh(
            "cuda", (outer_size, inner_size), mesh_dim_names=("outer_fsdp_dp", "dp_cp")
        )
        outer_rank, inner_rank = device_mesh.get_coordinate()
        optimizer_dp_group = torch.distributed.group.WORLD
        cols = 4
        rows_by_param = [dp_size + 3, dp_size * 2 + 1, dp_size + 5]
        optimizer_kwargs = dict(
            lr=0.03, momentum=0.25, nesterov=True, weight_decay=0.02, num_ns_steps=3
        )

        full_params = [
            (
                torch.arange(rows * cols, device="cuda", dtype=torch.float32).view(rows, cols)
                + 1.7 * (idx + 1)
            )
            / (19 + idx)
            for idx, rows in enumerate(rows_by_param)
        ]
        full_grads_by_step = [
            [
                (
                    torch.arange(rows * cols, device="cuda", dtype=torch.float32).view(rows, cols)
                    + 0.9 * (idx + 1)
                    + step
                )
                / (29 + idx + step)
                for idx, rows in enumerate(rows_by_param)
            ]
            for step in range(2)
        ]

        unsharded_params = [nn.Parameter(full_param.clone()) for full_param in full_params]
        unsharded_optimizer = TensorParallelMuon(
            params=unsharded_params,
            pg_collection=None,
            tp_mode="duplicated",
            split_qkv=False,
            **optimizer_kwargs,
        )

        sharded_params = []
        local_slice_fns = []
        for full_param in full_params:
            if topology == "hsdp":
                rows_per_inner_rank = [
                    full_param.shape[0] // inner_size
                    + (1 if rank < full_param.shape[0] % inner_size else 0)
                    for rank in range(inner_size)
                ]
                row_start = sum(rows_per_inner_rank[:inner_rank])
                row_count = rows_per_inner_rank[inner_rank]
                make_dtensor = _make_hsdp_dtensor

                def local_slice(tensor, start=row_start, count=row_count):
                    return tensor[start : start + count].contiguous()

            else:
                logical_rank = inner_rank * outer_size + outer_rank
                rows_per_logical_rank = [
                    full_param.shape[0] // dp_size
                    + (1 if rank < full_param.shape[0] % dp_size else 0)
                    for rank in range(dp_size)
                ]
                row_start = sum(rows_per_logical_rank[:logical_rank])
                row_count = rows_per_logical_rank[logical_rank]
                make_dtensor = _make_hfsdp_dtensor

                def local_slice(tensor, start=row_start, count=row_count):
                    return tensor[start : start + count].contiguous()

            local_param = local_slice(full_param)
            sharded_params.append(
                nn.Parameter(make_dtensor(local_param, full_param.shape, device_mesh))
            )
            local_slice_fns.append(local_slice)

        sharded_optimizer_kwargs = dict(optimizer_kwargs)
        if overlap_comm_compute:
            sharded_optimizer_kwargs.update(
                fsdp_batched_all_gather=True,
                fsdp_reuse_gather_scratch=True,
                fsdp_padded_all_gather=True,
                fsdp_padded_all_gather_pad_factor=1000,
                fsdp_overlap_comm_compute=True,
            )
        sharded_optimizer = _make_fsdp_muon(
            sharded_params, dp_group=optimizer_dp_group, **sharded_optimizer_kwargs
        )

        for step, full_grads in enumerate(full_grads_by_step, start=1):
            for param, full_grad in zip(unsharded_params, full_grads):
                param.grad = full_grad.clone()
            for param, full_grad, local_slice, full_param in zip(
                sharded_params, full_grads, local_slice_fns, full_params
            ):
                local_grad = local_slice(full_grad)
                make_dtensor = _make_hsdp_dtensor if topology == "hsdp" else _make_hfsdp_dtensor
                param.grad = make_dtensor(local_grad, full_param.shape, device_mesh)

            unsharded_optimizer.step()
            sharded_optimizer.step()

            for idx, (sharded_param, unsharded_param, local_slice) in enumerate(
                zip(sharded_params, unsharded_params, local_slice_fns)
            ):
                expected_local = local_slice(unsharded_param.detach())
                local_value = sharded_param.data.to_local()

                if local_value.numel() == 0:
                    assert local_value.shape[0] == 0
                    continue

                torch.testing.assert_close(
                    local_value,
                    expected_local,
                    atol=1e-5,
                    rtol=1e-4,
                    msg=(
                        f"Muon+M-FSDP {topology} diverged from unsharded Muon "
                        f"on step {step}, param {idx}, rank {world_rank}"
                    ),
                )


@_skip_if_single_rank()
class TestFSDPFactoryIntegration:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    def test_muon_fsdp_gather_fast_paths_default_to_opt_in(self):
        from megatron.core.optimizer import OptimizerConfig

        config = OptimizerConfig(optimizer="muon")

        assert not config.muon_fsdp_batched_all_gather
        assert not config.muon_fsdp_reuse_gather_scratch
        assert not config.muon_fsdp_padded_all_gather
        assert config.muon_fsdp_padded_all_gather_pad_factor == 1.25
        assert config.muon_fsdp_batch_max_gather_bytes == 1024 * 1024 * 1024
        assert config.muon_fsdp_padded_all_gather_zero_pad
        assert config.muon_fsdp_fast_reconstruct
        assert not config.muon_fsdp_overlap_comm_compute

    @pytest.mark.parametrize("strategy", ["no_shard", "optim", "optim_grads", "optim_grads_params"])
    def test_factory_dispatches_correct_muon_cls(self, strategy):
        from megatron.core.optimizer import (
            DistributedOptimizer,
            OptimizerConfig,
            _build_megatron_fsdp_emerging_optimizer,
        )
        from megatron.core.optimizer.optimizer import ChainedOptimizer
        from megatron.core.process_groups_config import ProcessGroupCollection

        model_chunk = MagicMock()
        model_chunk.config.num_attention_heads = 8
        model_chunk.config.num_query_groups = 8
        model_chunk.config.kv_channels = 16

        linear_weight = nn.Parameter(torch.randn(32, 16, device="cuda"))
        linear_weight.is_embedding_or_output_parameter = False
        bias_param = nn.Parameter(torch.randn(32, device="cuda"))
        bias_param.is_embedding_or_output_parameter = False

        model_chunk.named_parameters.return_value = [
            ("layer.linear.weight", linear_weight),
            ("layer.linear.bias", bias_param),
        ]
        model_chunk.parameters.return_value = iter([linear_weight, bias_param])
        model_chunk.ddp_config.use_megatron_fsdp = True
        model_chunk.ddp_config.data_parallel_sharding_strategy = strategy

        config = OptimizerConfig(
            optimizer="muon",
            lr=0.01,
            weight_decay=0.01,
            bf16=False,
            fp16=False,
            use_distributed_optimizer=False,
            muon_momentum=0.95,
            muon_nesterov=True,
            muon_fp32_matmul_prec="medium",
            muon_num_ns_steps=5,
            muon_scale_mode="spectral",
            muon_tp_mode="duplicated",
            muon_split_qkv=False,
            muon_extra_scale_factor=1.0,
            muon_fsdp_batched_all_gather=True,
            muon_fsdp_reuse_gather_scratch=True,
            muon_fsdp_padded_all_gather=True,
            muon_fsdp_padded_all_gather_pad_factor=2.0,
            muon_fsdp_batch_max_gather_bytes=123456,
            muon_fsdp_padded_all_gather_zero_pad=False,
            muon_fsdp_fast_reconstruct=False,
            muon_fsdp_overlap_comm_compute=True,
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        def fake_get_param_groups(model_chunks, config, overrides):
            return [
                {
                    "params": [linear_weight],
                    "is_expert_parallel": False,
                    "wd_mult": 1.0,
                    "lr_mult": 1.0,
                }
            ]

        with patch("megatron.core.optimizer._get_param_groups", side_effect=fake_get_param_groups):
            with patch("megatron.core.optimizer.get_megatron_optimizer") as mock_get_opt:
                mock_adam = MagicMock()
                mock_adam_sub = MagicMock()
                mock_adam_sub.config = config
                mock_adam.chained_optimizers = [mock_adam_sub]
                mock_get_opt.return_value = mock_adam
                with patch("megatron.core.optimizer._get_mfsdp_models", return_value=[MagicMock()]):
                    result = _build_megatron_fsdp_emerging_optimizer(
                        config=config,
                        model_chunks=[model_chunk],
                        config_overrides={},
                        pg_collection=pg_collection,
                        eopt_name="muon",
                    )

        assert isinstance(result, ChainedOptimizer)

        muon_wrapper = result.chained_optimizers[0]
        assert isinstance(muon_wrapper, DistributedOptimizer)
        base_opt = muon_wrapper.optimizer
        if strategy == "no_shard":
            assert isinstance(base_opt, TensorParallelMuon)
            assert not isinstance(base_opt, FSDPTensorParallelMuon)
        else:
            assert isinstance(base_opt, FSDPTensorParallelMuon)
            assert base_opt.dp_group is not None
            assert base_opt.fsdp_batched_all_gather
            assert base_opt.fsdp_reuse_gather_scratch
            assert base_opt.fsdp_padded_all_gather
            assert base_opt.fsdp_padded_all_gather_pad_factor == 2.0
            assert base_opt.fsdp_batch_max_gather_bytes == 123456
            assert not base_opt.fsdp_padded_all_gather_zero_pad
            assert not base_opt.fsdp_fast_reconstruct
            assert base_opt.fsdp_overlap_comm_compute
