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

        optimizer = _make_fsdp_muon(params, dp_group=dp_group)
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

        optimizer = _make_fsdp_muon(params, dp_group=dp_group)

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

        optimizer = _make_fsdp_muon(params, dp_group=dp_group)

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

    def test_step_matches_unsharded_muon_optimizer_step_numerics(self):
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
        sharded_optimizer = _make_fsdp_muon(sharded_params, dp_group=dp_group, **optimizer_kwargs)

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


@_skip_if_single_rank()
class TestFSDPFactoryIntegration:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

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
