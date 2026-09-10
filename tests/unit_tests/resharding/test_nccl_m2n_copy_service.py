# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the native NCCL M2N ReFIT copy service."""

import importlib.util
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.resharding.copy_services.nccl_m2n_copy_service import (
    NCCLM2NCopyService,
    _M2NChannel,
    _parameter_groups,
    _stage_pairs,
    _validate_nccl_version,
    _validate_role_roster,
)
from megatron.core.resharding.transforms import MXFP8ReshardTransform
from megatron.core.resharding.utils import ReshardPlan, TensorReshardSpec
from tests.unit_tests.test_utilities import Utils


def _nccl_with_version(*release: int):
    version = SimpleNamespace(release=release)
    version_info = SimpleNamespace(libnccl=SimpleNamespace(version=version))
    return SimpleNamespace(get_version=lambda: version_info)


def _spec(
    *,
    src_ranks=(0, 1),
    dst_ranks=(2,),
    src_name=None,
    dst_name=None,
    src_shard_dim=1,
    dst_shard_dim=None,
):
    return TensorReshardSpec(
        resolved_name="decoder.layers.0.weight",
        src_ranks=src_ranks,
        dst_ranks=dst_ranks,
        global_shape=(4, 8),
        src_local_shape=(4, 4),
        dst_local_shape=(4, 8),
        dtype=torch.float32,
        src_shard_dim=src_shard_dim,
        dst_shard_dim=dst_shard_dim,
        src_param_name=src_name,
        dst_param_name=dst_name,
    )


def test_validate_nccl_version():
    _validate_nccl_version(_nccl_with_version(2, 30, 5))
    with pytest.raises(RuntimeError, match=r"NCCL >= 2\.30\.5"):
        _validate_nccl_version(_nccl_with_version(2, 30, 4))


def test_validate_role_roster_accepts_source_first_disjoint_meshes():
    topology = _validate_role_roster([(True, False), (True, False), (False, True), (False, True)])
    assert topology.src_ranks == (0, 1)
    assert topology.dst_ranks == (2, 3)


@pytest.mark.parametrize(
    ("roles", "message"),
    [
        ([(True, True), (False, True)], "non-collocated"),
        ([(True, False), (False, False), (False, True)], "idle ranks"),
        ([(True, False), (False, True), (True, False), (False, True)], "source-first"),
        ([(True, False)], "at least one source and one destination"),
    ],
)
def test_validate_role_roster_rejects_unsupported_topologies(roles, message):
    with pytest.raises(RuntimeError, match=message):
        _validate_role_roster(roles)


def test_stage_pairs_are_unique_and_sorted():
    specs = [
        _spec(src_ranks=(2, 3), dst_ranks=(5,)),
        _spec(src_ranks=(0, 1), dst_ranks=(4,)),
        _spec(src_ranks=(2, 3), dst_ranks=(5,)),
    ]

    assert _stage_pairs(specs) == (((0, 1), (4,)), ((2, 3), (5,)))


def test_parameter_groups_are_bounded_without_splitting_packed_parameters():
    first = replace(_spec(), resolved_name="first")
    second_parts = [
        replace(
            _spec(),
            resolved_name="second",
            dst_local_shape=(2, 8),
            part_index=part_index,
            part_count=2,
        )
        for part_index in range(2)
    ]
    third = replace(_spec(), resolved_name="third")

    batches = _parameter_groups([first, *second_parts, third], max_group_bytes=192)

    assert [
        [[spec.resolved_name for spec in parameter] for parameter in batch] for batch in batches
    ] == [[['first']], [['second', 'second']], [['third']]]


def test_parameter_groups_reject_noncontiguous_parameter_specs():
    first = replace(_spec(), resolved_name="first")
    second = replace(_spec(), resolved_name="second")

    with pytest.raises(RuntimeError, match="non-contiguous specs for first"):
        _parameter_groups([first, second, first], max_group_bytes=1024)


def test_model_roles_cannot_change_while_reusing_service():
    service = object.__new__(NCCLM2NCopyService)
    service._is_source = None
    service._is_destination = None

    service.set_model_roles(is_source=True, is_destination=False)
    service.set_model_roles(is_source=True, is_destination=False)

    with pytest.raises(RuntimeError, match="roles cannot change"):
        service.set_model_roles(is_source=False, is_destination=True)


def test_topology_is_collected_once(monkeypatch):
    service = object.__new__(NCCLM2NCopyService)
    service._device = torch.device("cpu")
    service._is_source = True
    service._is_destination = False
    service._topology = None
    service._max_group_bytes = 123
    service.world_size = 4
    service.group = None
    calls = 0

    def fake_all_gather(outputs, _roles, group):
        nonlocal calls
        assert group is None
        calls += 1
        roles = ((1, 0, 123), (1, 0, 123), (0, 1, 123), (0, 1, 123))
        for output, role in zip(outputs, roles):
            output.copy_(torch.tensor(role))

    monkeypatch.setattr(dist, "all_gather", fake_all_gather)

    topology = service._get_topology()
    cached_topology = service._get_topology()

    assert topology is cached_topology
    assert topology.src_ranks == (0, 1)
    assert topology.dst_ranks == (2, 3)
    assert calls == 1


def test_dense_submission_interface_is_rejected():
    service = object.__new__(NCCLM2NCopyService)
    tensor = torch.zeros(1)

    with pytest.raises(RuntimeError, match="whole-tensor ReshardPlan"):
        service.submit_send(tensor, 1)
    with pytest.raises(RuntimeError, match="whole-tensor ReshardPlan"):
        service.submit_recv(tensor, 0)
    with pytest.raises(RuntimeError, match="whole-tensor ReshardPlan"):
        service.run()


def test_plan_incompatibility_is_not_lowered_to_dense_path():
    service = object.__new__(NCCLM2NCopyService)
    service._closed = False
    service._poisoned = False
    plan = ReshardPlan(
        send_ops=[], recv_ops=[], tensor_reshard_error="partition_stride=2 is unsupported"
    )

    with pytest.raises(RuntimeError, match="partition_stride=2"):
        service.execute_plan(plan, {}, {})


def test_non_mxfp8_transform_is_rejected_before_any_collective():
    service = object.__new__(NCCLM2NCopyService)
    service._closed = False
    service._poisoned = False
    plan = ReshardPlan(send_ops=[], recv_ops=[], tensor_reshard_specs=[_spec()])

    with pytest.raises(RuntimeError, match="only supports the MXFP8 receiver-side"):
        service.execute_plan(plan, {}, {}, transform=object())


def test_sender_side_mxfp8_transform_is_rejected_before_any_collective():
    service = object.__new__(NCCLM2NCopyService)
    service._closed = False
    service._poisoned = False
    plan = ReshardPlan(send_ops=[], recv_ops=[], tensor_reshard_specs=[_spec()])
    transform = object.__new__(MXFP8ReshardTransform)
    transform.convert_on_send = True

    with pytest.raises(RuntimeError, match="does not support MXFP8 sender-side"):
        service.execute_plan(plan, {}, {}, transform=transform)


def test_local_tensor_validates_whole_shard_metadata():
    src = torch.zeros(4, 4)
    spec = _spec(src_name="weight")

    local = NCCLM2NCopyService._local_tensor(spec, 0, {"weight": src}, {})

    assert local.src.data_ptr() == src.data_ptr()
    assert local.dst is None
    with pytest.raises(RuntimeError, match="parameter shape changed"):
        NCCLM2NCopyService._local_tensor(spec, 0, {"weight": torch.zeros(4, 8)}, {})


def test_validate_specs_rejects_wrong_side_mesh():
    service = object.__new__(NCCLM2NCopyService)
    topology = _validate_role_roster([(True, False), (True, False), (False, True)])
    spec = _spec(src_ranks=(0, 2), dst_ranks=(2,))

    with pytest.raises(RuntimeError, match="non-source ranks"):
        service._validate_specs(topology, [spec])


def test_close_destroys_every_channel_after_one_failure(monkeypatch):
    calls = []

    class Resource:
        def __init__(self, name, raises=False):
            self.name = name
            self.raises = raises

        def destroy(self):
            calls.append(self.name)
            if self.raises:
                raise RuntimeError(f"{self.name} destroy failed")

    service = object.__new__(NCCLM2NCopyService)
    service._closed = False
    service._device = torch.device("cpu")
    service._handle = Resource("handle", raises=True)
    service._channels = {
        ((0,), (2,)): _M2NChannel(Resource("first"), None, None, None),
        ((1,), (3,)): _M2NChannel(Resource("second"), None, None, None),
        ((0,), (3,)): None,
    }
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: calls.append("synchronize"))

    with pytest.raises(RuntimeError, match="handle destroy failed"):
        service.close()
    service.close()

    assert calls == ["synchronize", "handle", "first", "second"]
    assert service._closed
    assert service._handle is None
    assert service._channels == {}


def _has_nccl_m2n_python_package() -> bool:
    try:
        return (
            importlib.util.find_spec("nccl.core") is not None
            and importlib.util.find_spec("nccl.m2n") is not None
        )
    except (ImportError, ModuleNotFoundError):
        return False


@pytest.mark.skipif(
    not _has_nccl_m2n_python_package(),
    reason="install NVIDIA/nccl-extensions and NCCL4Py to run the M2N integration test",
)
def test_nccl_m2n_reshards_parameter_between_tensor_dimensions():
    """Exercise direct TP shard-to-shard M2N transfer on GPUs."""
    Utils.initialize_distributed()
    world_size = dist.get_world_size()
    if world_size < 2 or world_size % 2:
        pytest.skip("NCCL M2N integration test requires an even distributed world size >= 2")

    rank = dist.get_rank()
    src_count = world_size // 2
    dst_count = world_size - src_count
    src_ranks = tuple(range(src_count))
    dst_ranks = tuple(range(src_count, world_size))
    is_source = rank in src_ranks
    rows_per_src = 3
    cols_per_dst = 5
    global_shape = (src_count * rows_per_src, dst_count * cols_per_dst)
    src_shape = (rows_per_src, global_shape[1])
    dst_shape = (global_shape[0], cols_per_dst)

    src_tensor = None
    dst_tensor = None
    if is_source:
        first_row = rank * rows_per_src
        row_ids = torch.arange(
            first_row, first_row + rows_per_src, dtype=torch.float32, device="cuda"
        ).view(-1, 1)
        col_ids = torch.arange(global_shape[1], dtype=torch.float32, device="cuda").view(1, -1)
        src_tensor = row_ids * 1000 + col_ids
    else:
        dst_tensor = torch.empty(dst_shape, dtype=torch.float32, device="cuda")

    spec = TensorReshardSpec(
        resolved_name="weight",
        src_ranks=src_ranks,
        dst_ranks=dst_ranks,
        global_shape=global_shape,
        src_local_shape=src_shape,
        dst_local_shape=dst_shape,
        dtype=torch.float32,
        src_shard_dim=0,
        dst_shard_dim=1,
        src_param_name="weight" if is_source else None,
        dst_param_name="weight" if not is_source else None,
    )
    plan = ReshardPlan(send_ops=[], recv_ops=[], tensor_reshard_specs=[spec])
    service = NCCLM2NCopyService()
    service.set_model_roles(is_source=is_source, is_destination=not is_source)

    assert service.execute_plan(
        plan,
        {"weight": src_tensor} if src_tensor is not None else {},
        {"weight": dst_tensor} if dst_tensor is not None else {},
    )
    torch.cuda.synchronize()

    local_ok = True
    if dst_tensor is not None:
        dst_index = rank - src_count
        first_col = dst_index * cols_per_dst
        row_ids = torch.arange(global_shape[0], dtype=torch.float32, device="cuda").view(-1, 1)
        col_ids = torch.arange(
            first_col, first_col + cols_per_dst, dtype=torch.float32, device="cuda"
        ).view(1, -1)
        local_ok = torch.equal(dst_tensor, row_ids * 1000 + col_ids)

    service.close()
    status = torch.tensor(int(local_ok), dtype=torch.int32, device="cuda")
    dist.all_reduce(status, op=dist.ReduceOp.MIN)
    assert status.item() == 1


@pytest.mark.skipif(
    not _has_nccl_m2n_python_package(),
    reason="install NVIDIA/nccl-extensions and NCCL4Py to run the M2N integration test",
)
def test_nccl_m2n_quantizes_packed_parameters_to_mxfp8_one_at_a_time():
    """Receive packed BF16 shards and immediately quantize each full local parameter."""
    from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

    Utils.initialize_distributed()
    if torch.cuda.get_device_properties(torch.cuda.current_device()).major < 10:
        pytest.skip("MXFP8 integration requires a Blackwell GPU")

    world_size = dist.get_world_size()
    if world_size < 2 or world_size % 2:
        pytest.skip("NCCL M2N integration test requires an even distributed world size >= 2")

    rank = dist.get_rank()
    mesh_size = world_size // 2
    src_ranks = tuple(range(mesh_size))
    dst_ranks = tuple(range(mesh_size, world_size))
    is_source = rank in src_ranks
    local_rows = 32
    columns = 128
    component_shape = (mesh_size * local_rows, columns)
    local_param_shape = (2 * local_rows, columns)

    def make_rows(first_row: int, value_offset: int = 0) -> torch.Tensor:
        rows = torch.arange(
            first_row, first_row + local_rows, dtype=torch.float32, device="cuda"
        ).view(-1, 1)
        cols = torch.arange(columns, dtype=torch.float32, device="cuda").view(1, -1)
        return (value_offset + rows + cols / columns).to(torch.bfloat16)

    parameter_names = ("first_weight", "second_weight")
    src_tensors = {}
    persistent_buffers = {}
    expected_buffers = {}
    local_mesh_rank = rank if is_source else rank - mesh_size
    for parameter_index, name in enumerate(parameter_names):
        value_offset = parameter_index * 256
        first_component = make_rows(local_mesh_rank * local_rows, value_offset)
        second_component = make_rows(
            component_shape[0] + local_mesh_rank * local_rows, value_offset
        )
        local_value = torch.cat((first_component, second_component))
        if is_source:
            src_tensors[name] = local_value
        else:
            persistent_buffers[name] = MXFP8Tensor.from_bf16(
                torch.zeros_like(local_value), backend="triton"
            )
            expected_buffers[name] = MXFP8Tensor.from_bf16(local_value, backend="triton")

    specs = []
    for name in parameter_names:
        for part_index in range(2):
            part_slice = (
                slice(part_index * local_rows, (part_index + 1) * local_rows),
                slice(None),
            )
            specs.append(
                TensorReshardSpec(
                    resolved_name=name,
                    src_ranks=src_ranks,
                    dst_ranks=dst_ranks,
                    global_shape=component_shape,
                    src_local_shape=(local_rows, columns),
                    dst_local_shape=(local_rows, columns),
                    dtype=torch.bfloat16,
                    src_shard_dim=0,
                    dst_shard_dim=0,
                    src_param_name=name if is_source else None,
                    dst_param_name=name if not is_source else None,
                    src_param_shape=local_param_shape,
                    dst_param_shape=local_param_shape,
                    src_slice=part_slice,
                    dst_slice=part_slice,
                    part_index=part_index,
                    part_count=2,
                )
            )

    class TrackingTransform(MXFP8ReshardTransform):
        def __init__(self):
            super().__init__(
                convertible_params=set(parameter_names),
                persistent_buffers=persistent_buffers,
                backend="triton",
            )
            self.active_buffers = 0
            self.max_active_buffers = 0

        def prepare_recv(self, param_name, dst_slice):
            self.active_buffers += 1
            self.max_active_buffers = max(self.max_active_buffers, self.active_buffers)
            return super().prepare_recv(param_name, dst_slice)

        def finalize_recv(self, param_name, dst_slice, recv_buffers):
            super().finalize_recv(param_name, dst_slice, recv_buffers)
            self.active_buffers -= 1

    transform = None if is_source else TrackingTransform()
    plan = ReshardPlan(send_ops=[], recv_ops=[], tensor_reshard_specs=specs)
    parameter_bytes = 2 * local_rows * columns * torch.bfloat16.itemsize
    service = NCCLM2NCopyService(max_group_bytes=parameter_bytes)
    service.set_model_roles(is_source=is_source, is_destination=not is_source)

    assert service.execute_plan(plan, src_tensors, {}, transform=transform)
    torch.cuda.synchronize()

    local_ok = True
    if transform is not None:
        local_ok = transform.active_buffers == 0 and transform.max_active_buffers == 1
        for name, expected in expected_buffers.items():
            actual = persistent_buffers[name]
            local_ok = local_ok and torch.equal(actual.data, expected.data)
            local_ok = local_ok and torch.equal(
                actual.scale.view(torch.uint8), expected.scale.view(torch.uint8)
            )

    service.close()
    status = torch.tensor(int(local_ok), dtype=torch.int32, device="cuda")
    dist.all_reduce(status, op=dist.ReduceOp.MIN)
    assert status.item() == 1
