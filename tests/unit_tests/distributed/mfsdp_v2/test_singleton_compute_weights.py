# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Singleton data-parallel fast paths for MFSDP v2 compute weights."""

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard
from torch.nn import functional as F

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
    Fp8ParameterGroup,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import (
    HAVE_TE_CAST_MASTER_WEIGHTS_TO_FP8,
    HAVE_TE_MXFP8TENSOR,
    MixedPrecisionPolicy,
)


def _flat_placements() -> Placements:
    return Placements(dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)])


def _singleton_mesh(distributed_setup):
    if distributed_setup.device.type != "cuda":
        pytest.skip("MFSDP v2 requires CUDA.")
    parent_mesh = init_device_mesh(
        distributed_setup.device.type,
        (distributed_setup.world_size, 1),
        mesh_dim_names=("replica", "singleton_dp"),
    )
    return parent_mesh["singleton_dp"]


def _track_trace_pool_allocations(allocator, monkeypatch) -> list[object]:
    allocation_keys = []
    original_allocate = allocator.allocate

    def tracked_allocate(key, *args, **kwargs):
        allocation_keys.append(key)
        return original_allocate(key, *args, **kwargs)

    monkeypatch.setattr(allocator, "allocate", tracked_allocate)
    return allocation_keys


def _forbid_all_gather(*_args, **_kwargs) -> None:
    raise AssertionError("A singleton data-parallel mesh must not launch an all-gather.")


def test_singleton_flat_compute_weight_binds_persistent_storage_without_all_gather(
    distributed_setup, monkeypatch
):
    """Flat is effectively replicated on DP1 and needs no temporary gather target."""
    device = distributed_setup.device
    mesh = _singleton_mesh(distributed_setup)
    model = nn.Linear(32, 32, bias=False, device=device)

    with fully_shard_context(device=device, enable_trace_pool=True) as context:
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    (group,) = model.parameter_groups
    assert group.model_weight is not None
    allocator = context.trace_pool_allocator
    assert allocator is not None
    allocation_keys = _track_trace_pool_allocations(allocator, monkeypatch)
    monkeypatch.setattr(dist, "all_gather_into_tensor", _forbid_all_gather)

    model.pre_forward()
    try:
        (fsdp_parameter,) = group.fsdp_parameters
        assert model.weight is fsdp_parameter.unsharded
        assert (
            fsdp_parameter.unsharded.data_ptr() == group.model_weight.get_local_tensor(0).data_ptr()
        )
    finally:
        model.post_forward()

    unsharded_key = (id(group), "unsharded_model_weight")
    assert unsharded_key not in allocation_keys
    assert unsharded_key not in allocator._metadata
    assert group.model_weight.local_buffer.untyped_storage().nbytes() > 0


def test_singleton_bf16_compute_weight_refreshes_in_place_without_all_gather(
    distributed_setup, monkeypatch
):
    """DP1 refreshes a persistent BF16 compute weight from its FP32 master in place."""
    device = distributed_setup.device
    mesh = _singleton_mesh(distributed_setup)
    model = nn.Linear(32, 32, bias=False, dtype=torch.bfloat16, device=device)
    policy = MixedPrecisionPolicy(
        main_params_dtype=torch.float32,
        main_grads_dtype=torch.float32,
        grad_comm_dtype=torch.float32,
    )

    with fully_shard_context(device=device, enable_trace_pool=True) as context:
        fully_shard(model, mesh=mesh, placements=_flat_placements(), mixed_precision_policy=policy)

    (group,) = model.parameter_groups
    assert group.main_weight.dtype == torch.float32
    assert group.model_weight is not None
    assert group.model_weight.dtype == torch.bfloat16
    assert group.model_weight is not group.main_weight
    assert group._unsharded_model_weight is group.model_weight
    model_weight_data_ptr = group.model_weight.local_buffer.data_ptr()

    main_weight = group.main_weight.get_local_tensor(0)
    expected_main_weight = torch.linspace(
        -0.5, 0.5, main_weight.numel(), dtype=main_weight.dtype, device=device
    ).reshape_as(main_weight)
    main_weight.copy_(expected_main_weight)
    group.sync_model_weight_from_main_weight()

    allocator = context.trace_pool_allocator
    assert allocator is not None
    allocation_keys = _track_trace_pool_allocations(allocator, monkeypatch)
    monkeypatch.setattr(dist, "all_gather_into_tensor", _forbid_all_gather)

    inputs = torch.randn(4, 32, dtype=torch.bfloat16, device=device)
    expected_compute_weight = expected_main_weight.to(torch.bfloat16)
    model.pre_forward()
    try:
        assert group.model_weight.local_buffer.data_ptr() == model_weight_data_ptr
        torch.testing.assert_close(
            group.model_weight.get_local_tensor(0), expected_compute_weight, rtol=0, atol=0
        )
        torch.testing.assert_close(
            F.linear(inputs, model.weight),
            F.linear(inputs, expected_compute_weight),
            rtol=0,
            atol=0,
        )
    finally:
        model.post_forward()

    unsharded_key = (id(group), "unsharded_model_weight")
    assert unsharded_key not in allocation_keys
    assert unsharded_key not in allocator._metadata
    assert group.model_weight.local_buffer.untyped_storage().nbytes() > 0


def test_singleton_flat_mxfp8_payloads_alias_persistent_storage_without_all_gather(
    distributed_setup, monkeypatch
):
    """DP1 MXFP8 binds its persistent row/column payloads without gather targets."""
    device = distributed_setup.device
    mesh = _singleton_mesh(distributed_setup)
    if (
        not HAVE_TE_MXFP8TENSOR
        or not HAVE_TE_CAST_MASTER_WEIGHTS_TO_FP8
        or torch.cuda.get_device_capability(device)[0] < 10
    ):
        pytest.skip("MXFP8 requires Transformer Engine and a Blackwell-or-newer GPU.")

    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    model = nn.Linear(32, 32, bias=False, dtype=torch.bfloat16, device=device)
    quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
    model.weight = nn.Parameter(quantizer(model.weight.detach()), requires_grad=True)

    with fully_shard_context(device=device, enable_trace_pool=True) as context:
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    (group,) = model.parameter_groups
    assert isinstance(group, Fp8ParameterGroup)
    assert group._unsharded_rowwise is group._rowwise_buffer
    assert group._unsharded_colwise is group._colwise_buffer
    (fsdp_parameter,) = group.fsdp_parameters
    assert fsdp_parameter.unsharded._rowwise_data is None
    assert fsdp_parameter.unsharded._columnwise_data is None

    allocator = context.trace_pool_allocator
    assert allocator is not None
    allocation_keys = _track_trace_pool_allocations(allocator, monkeypatch)
    monkeypatch.setattr(dist, "all_gather_into_tensor", _forbid_all_gather)

    model.pre_forward()
    try:
        tensor = fsdp_parameter.unsharded
        assert (
            tensor._rowwise_data.data_ptr() == group._rowwise_buffer.get_local_tensor(0).data_ptr()
        )
        assert (
            tensor._columnwise_data.data_ptr()
            == group._colwise_buffer.get_local_tensor(0).data_ptr()
        )
    finally:
        model.post_forward()

    assert tensor._rowwise_data is None
    assert tensor._columnwise_data is None
    unsharded_keys = {(id(group), "unsharded_rowwise"), (id(group), "unsharded_colwise")}
    assert unsharded_keys.isdisjoint(allocation_keys)
    assert unsharded_keys.isdisjoint(allocator._metadata)
    assert group._rowwise_buffer.local_buffer.untyped_storage().nbytes() > 0
    assert group._colwise_buffer.local_buffer.untyped_storage().nbytes() > 0
