# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the native NCCL M2N ReFIT copy service."""

import importlib.util
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from packaging.version import Version

from megatron.core.resharding.copy_services.nccl_m2n_copy_service import (
    NCCLM2NCopyService,
    _default_split_comm,
    _has_nccl_cuda_backend,
    _M2NChannel,
    _parameter_groups,
    _stage_pairs,
    _validate_nccl_version,
    _validate_role_roster,
)
from megatron.core.resharding.transforms import MXFP8ReshardTransform
from megatron.core.resharding.utils import ReshardPlan, TensorReshardSpec
from tests.unit_tests.test_utilities import Utils


def _nccl_with_version(*release: int, legacy: bool = False):
    version = Version(".".join(str(value) for value in release))
    if legacy:
        version_info = SimpleNamespace(nccl_version=version)
    else:
        version_info = SimpleNamespace(
            libnccl=SimpleNamespace(version=version), nccl_bindings=Version("2.31.0")
        )
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


@pytest.mark.parametrize("legacy", [False, True])
def test_validate_nccl_version(legacy):
    _validate_nccl_version(_nccl_with_version(2, 30, 5, legacy=legacy))
    _validate_nccl_version(_nccl_with_version(2, 31, 0, legacy=legacy))
    with pytest.raises(RuntimeError, match=r"NCCL >= 2\.30\.5, found 2\.30\.4"):
        _validate_nccl_version(_nccl_with_version(2, 30, 4, legacy=legacy))


def test_validate_nccl_version_rejects_missing_library():
    nccl_core = SimpleNamespace(
        get_version=lambda: SimpleNamespace(libnccl=None, nccl_bindings=Version("2.31.0"))
    )
    with pytest.raises(RuntimeError, match="requires a loaded NCCL library"):
        _validate_nccl_version(nccl_core)


@pytest.mark.parametrize(
    "nccl_core", [SimpleNamespace(), SimpleNamespace(get_version=lambda: SimpleNamespace())]
)
def test_validate_nccl_version_rejects_unsupported_package(nccl_core):
    with pytest.raises(RuntimeError, match="requires the current NCCL4Py package"):
        _validate_nccl_version(nccl_core)


def test_split_comm_defaults_on(monkeypatch):
    monkeypatch.delenv("NCCL_RESHARD_SPLIT_COMM", raising=False)
    _default_split_comm()
    assert os.environ["NCCL_RESHARD_SPLIT_COMM"] == "1"


@pytest.mark.parametrize("value", ["0", "1"])
def test_split_comm_respects_user_setting(monkeypatch, value):
    monkeypatch.setenv("NCCL_RESHARD_SPLIT_COMM", value)
    _default_split_comm()
    assert os.environ["NCCL_RESHARD_SPLIT_COMM"] == value


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("use_default_group", [False, True])
@pytest.mark.parametrize("backend_name", ["gloo", "undefined"])
def test_hybrid_group_uses_registered_nccl_cuda_backend(
    monkeypatch, use_default_group, backend_name
):
    cuda_backend = SimpleNamespace(_get_backend_name=lambda: "nccl")
    group = SimpleNamespace(_get_backend=lambda device: cuda_backend)
    monkeypatch.setattr(dist, "get_backend", lambda _group: backend_name)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(dist, "group", SimpleNamespace(WORLD=group))

    assert _has_nccl_cuda_backend(None if use_default_group else group)

    def get_missing_backend(_device):
        raise RuntimeError("no CUDA backend")

    gloo_only_group = SimpleNamespace(_get_backend=get_missing_backend)
    monkeypatch.setattr(dist, "group", SimpleNamespace(WORLD=gloo_only_group))
    assert not _has_nccl_cuda_backend(None if use_default_group else gloo_only_group)


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


def test_plan_overrides_local_group_limit_with_coordinated_value():
    service = object.__new__(NCCLM2NCopyService)
    service._topology = None
    service._max_group_bytes = 256
    plan = ReshardPlan([], [], execution_batch_bytes=128)

    service.set_plan(plan)

    assert service._max_group_bytes == 128


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


@pytest.fixture
def nccl_m2n_host_proxy(monkeypatch: pytest.MonkeyPatch):
    """Use host-proxy GIN for native M2N correctness tests on Blackwell.

    The dedicated GB200 CI bucket gives these tests a fresh process before NCCL
    initializes. Monkeypatch restores the environment after each test, but NCCL
    caches its transport settings, so process isolation is still required.
    """
    if torch.cuda.get_device_properties(torch.cuda.current_device()).major < 10:
        pytest.skip("Native NCCL M2N coverage runs on the Blackwell CI lane")

    monkeypatch.setenv("NCCL_GIN_TYPE", "2")
    monkeypatch.setenv("NCCL_NET_PLUGIN", "none")
    monkeypatch.setenv("NCCL_GIN_PLUGIN", "none")
    monkeypatch.setenv("NCCL_RMA_PLUGIN", "none")


@pytest.mark.skipif(
    not _has_nccl_m2n_python_package(),
    reason="install NVIDIA/nccl-extensions and NCCL4Py to run the M2N integration test",
)
@pytest.mark.usefixtures("nccl_m2n_host_proxy")
@pytest.mark.launch_on_gb200
def test_nccl_m2n_reshards_parameter_between_tensor_dimensions(monkeypatch: pytest.MonkeyPatch):
    """Exercise packed TP shard-to-shard M2N transfer on GPUs."""
    Utils.initialize_distributed()
    if torch.cuda.get_device_properties(torch.cuda.current_device()).major < 10:
        pytest.skip("Native NCCL M2N coverage runs on the GIN-enabled Blackwell CI lane")

    monkeypatch.setenv("NCCL_RESHARD_COPY_ALGORITHM", "PACK")
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
@pytest.mark.parametrize("grouped_gemm_backend", ["torch", "vllm", "flashinfer"])
@pytest.mark.usefixtures("nccl_m2n_host_proxy")
@pytest.mark.launch_on_gb200
def test_nccl_m2n_refits_selective_mxfp8_from_te(
    grouped_gemm_backend: str, monkeypatch: pytest.MonkeyPatch
):
    """Keep non-expert weights BF16 while refitting TE MXFP8 experts through M2N."""
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
    from megatron.core.inference.quantization.utils import resolve_mxfp8_backend

    Utils.initialize_distributed()
    if torch.cuda.get_device_properties(torch.cuda.current_device()).major < 10:
        pytest.skip("MXFP8 integration requires a Blackwell GPU")

    world_size = dist.get_world_size()
    if world_size < 2 or world_size % 2:
        pytest.skip("NCCL M2N integration test requires an even distributed world size >= 2")

    monkeypatch.setenv("NCCL_RESHARD_COPY_ALGORITHM", "PACK")
    rank = dist.get_rank()
    mesh_size = world_size // 2
    src_ranks = tuple(range(mesh_size))
    dst_ranks = tuple(range(mesh_size, world_size))
    is_source = rank in src_ranks
    local_mesh_rank = rank if is_source else rank - mesh_size
    local_rows, columns = 64, 128
    local_shape = (local_rows, columns)
    global_shape = (mesh_size * local_rows, columns)
    selected_names = (
        "decoder.layers.1.mlp.experts.linear_fc1.weight0",
        "decoder.layers.1.mlp.experts.linear_fc2.weight0",
    )
    bf16_name = "decoder.layers.1.self_attention.linear_qkv.weight"

    def make_weight(parameter_index: int) -> torch.Tensor:
        rows = torch.arange(
            local_mesh_rank * local_rows,
            (local_mesh_rank + 1) * local_rows,
            dtype=torch.float32,
            device="cuda",
        ).view(-1, 1)
        columns_ = torch.arange(columns, dtype=torch.float32, device="cuda").view(1, -1)
        return (parameter_index + rows / local_rows + columns_ / columns).to(torch.bfloat16)

    storage_backend = resolve_mxfp8_backend(grouped_gemm_backend)
    assert storage_backend == "triton"
    quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
    source_tensors = {}
    destination_tensors = {}
    persistent_buffers = {}
    expected_buffers = {}
    for parameter_index, name in enumerate(selected_names):
        bf16_weight = make_weight(parameter_index)
        te_weight = quantizer(bf16_weight)
        if is_source:
            source_tensors[name] = te_weight
        else:
            persistent_buffers[name] = MXFP8Tensor.from_bf16(
                torch.zeros_like(bf16_weight), backend=storage_backend
            )
            expected_buffers[name] = MXFP8Tensor.from_bf16(
                te_weight.dequantize(), backend=storage_backend
            )

    expected_bf16 = make_weight(len(selected_names))
    if is_source:
        source_tensors[bf16_name] = expected_bf16
    else:
        destination_tensors[bf16_name] = torch.zeros_like(expected_bf16)

    specs = [
        TensorReshardSpec(
            resolved_name=name,
            src_ranks=src_ranks,
            dst_ranks=dst_ranks,
            global_shape=global_shape,
            src_local_shape=local_shape,
            dst_local_shape=local_shape,
            dtype=torch.bfloat16,
            src_shard_dim=0,
            dst_shard_dim=0,
            src_param_name=name if is_source else None,
            dst_param_name=name if not is_source else None,
        )
        for name in (*selected_names, bf16_name)
    ]
    transform = (
        None
        if is_source
        else MXFP8ReshardTransform(
            convertible_params=set(selected_names),
            persistent_buffers=persistent_buffers,
            backend=storage_backend,
        )
    )
    plan = ReshardPlan(send_ops=[], recv_ops=[], tensor_reshard_specs=specs)
    service = NCCLM2NCopyService()
    service.set_model_roles(is_source=is_source, is_destination=not is_source)

    assert service.execute_plan(plan, source_tensors, destination_tensors, transform=transform)
    torch.cuda.synchronize()

    local_ok = True
    if not is_source:
        local_ok = torch.equal(destination_tensors[bf16_name], expected_bf16)
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


def _expert_meta(name, owner, ep_ranks, local_idx, num_experts, shape=(6, 4)):
    from megatron.core.resharding.utils import ParameterMetadata, assign_ep_resolved_name_inplace

    meta = ParameterMetadata(
        name=f"decoder.layers.0.mlp.experts.linear_fc1.weight{local_idx}",
        shape=shape,
        dtype=torch.bfloat16,
        element_size=2,
        is_ep=True,
        num_experts=num_experts,
        owner_rank=owner,
        tensor_parallel_group_ranks=[owner],
        expert_parallel_group_ranks=list(ep_ranks),
        data_parallel_group_ranks=[owner],
    )
    assign_ep_resolved_name_inplace(meta)
    return meta


def _expert_rosters(src_ep=(0, 1), dst_ep=(2, 3, 4, 5), num_experts=8, shape=(6, 4)):
    from megatron.core.resharding.planner import index_metadata_rosters

    world = max(src_ep + dst_ep) + 1
    pairs = [([], []) for _ in range(world)]
    for ep, side in ((src_ep, 0), (dst_ep, 1)):
        k = num_experts // len(ep)
        for owner in ep:
            pairs[owner][side].extend(
                _expert_meta(None, owner, ep, i, num_experts, shape) for i in range(k)
            )
    return index_metadata_rosters(pairs)


@pytest.mark.parametrize("stack", ["1", "0"])
def test_expert_family_becomes_one_ep_sharded_spec(monkeypatch, stack):
    from megatron.core.resharding.planner import _build_tensor_reshard_specs

    monkeypatch.setenv("MEGATRON_NCCL_M2N_STACK_EXPERTS", stack)
    dst_by_rank, src_by_name = _expert_rosters()
    for rank in range(6):
        specs, error = _build_tensor_reshard_specs(dst_by_rank, src_by_name, rank)
        assert error is None
        if stack == "0":
            assert len(specs) == 8 and all(len(s.src_ranks) == 1 for s in specs)
            continue
        assert len(specs) == 1
        (spec,) = specs
        assert spec.resolved_name == "decoder.layers.0.mlp.experts.linear_fc1.weight[experts]"
        assert (spec.src_ranks, spec.dst_ranks) == ((0, 1), (2, 3, 4, 5))
        assert spec.global_shape == (8, 6, 4)
        assert (spec.src_local_shape, spec.dst_local_shape) == ((4, 6, 4), (2, 6, 4))
        assert (spec.src_shard_dim, spec.dst_shard_dim) == (0, 0)
        local = [f"decoder.layers.0.mlp.experts.linear_fc1.weight{i}" for i in range(4)]
        assert spec.src_stack_names == (tuple(local) if rank < 2 else None)
        assert spec.dst_stack_names == (tuple(local[:2]) if rank >= 2 else None)


def test_expert_stacking_falls_back_when_an_expert_is_missing(monkeypatch):
    from megatron.core.resharding.planner import _build_tensor_reshard_specs

    monkeypatch.setenv("MEGATRON_NCCL_M2N_STACK_EXPERTS", "1")
    dst_by_rank, src_by_name = _expert_rosters()
    # Remove one source replica of expert 5: the family must keep per-expert specs.
    del src_by_name["decoder.layers.0.mlp.experts.linear_fc1.weight5"]
    specs, error = _build_tensor_reshard_specs(dst_by_rank, src_by_name, 2)
    assert specs is None and "weight5" in error


def test_adjacent_stack_view_and_staging_scatter():
    from megatron.core.resharding.copy_services.nccl_m2n_copy_service import (
        NCCLM2NCopyService,
        _adjacent_stack_view,
    )

    base = torch.arange(24, dtype=torch.float32)
    adjacent = [base[0:6].view(2, 3), base[6:12].view(2, 3)]
    view = _adjacent_stack_view(adjacent)
    assert view is not None and view.shape == (2, 2, 3)
    assert view.data_ptr() == base.data_ptr()
    separate = [torch.zeros(2, 3), torch.ones(2, 3)]
    assert _adjacent_stack_view(separate) is None

    spec = TensorReshardSpec(
        resolved_name="x.weight[experts]",
        src_ranks=(0,),
        dst_ranks=(1,),
        global_shape=(2, 2, 3),
        src_local_shape=(2, 2, 3),
        dst_local_shape=(2, 2, 3),
        dtype=torch.float32,
        src_shard_dim=0,
        dst_shard_dim=0,
        src_stack_names=("w0", "w1"),
        dst_stack_names=("w0", "w1"),
    )
    src = NCCLM2NCopyService._local_stacked_tensor(
        spec, 0, {"w0": separate[0], "w1": separate[1]}, {}
    )
    assert torch.equal(src.src, torch.stack(separate)) and src.dst is None
    dst_params = {"w0": torch.full((2, 3), -1.0), "w1": torch.full((2, 3), -1.0)}
    dst = NCCLM2NCopyService._local_stacked_tensor(spec, 1, {}, dst_params)
    assert dst.src is None and dst.dst_scatter is not None
    dst.dst.copy_(torch.stack(separate))
    NCCLM2NCopyService._scatter_stacked(dst, None)
    assert torch.equal(dst_params["w0"], separate[0]) and torch.equal(dst_params["w1"], separate[1])

    # Adjacent destination experts are received in place, without staging.
    storage = torch.full((12,), -1.0)
    adjacent_dst = {"w0": storage[0:6].view(2, 3), "w1": storage[6:12].view(2, 3)}
    direct = NCCLM2NCopyService._local_stacked_tensor(spec, 1, {}, adjacent_dst)
    assert direct.dst_scatter is None and direct.dst.data_ptr() == storage.data_ptr()


def test_stacked_experts_feed_receiver_side_transform():
    """Transformed experts go through prepare_recv/finalize_recv; the rest are copied."""
    from megatron.core.resharding.copy_services.nccl_m2n_copy_service import _adjacent_stack_view

    class _RecordingTransform:
        def __init__(self, names):
            self.names = set(names)
            self.finalized = {}

        def should_transform(self, name):
            return name in self.names

        def prepare_recv(self, name, dst_slice):
            assert dst_slice == (slice(None), slice(None))
            return [torch.empty(2, 3)]

        def finalize_recv(self, name, dst_slice, recv_buffers):
            self.finalized[name] = recv_buffers[0].clone()

    spec = TensorReshardSpec(
        resolved_name="x.weight[experts]",
        src_ranks=(0,),
        dst_ranks=(1,),
        global_shape=(2, 2, 3),
        src_local_shape=(2, 2, 3),
        dst_local_shape=(2, 2, 3),
        dtype=torch.float32,
        src_shard_dim=0,
        dst_shard_dim=0,
        src_stack_names=("w0", "w1"),
        dst_stack_names=("w0", "w1"),
    )
    storage = torch.full((12,), -1.0)
    dst_params = {"w0": storage[0:6].view(2, 3), "w1": storage[6:12].view(2, 3)}
    assert _adjacent_stack_view(list(dst_params.values())) is not None
    transform = _RecordingTransform({"w1"})
    local = NCCLM2NCopyService._local_stacked_tensor(spec, 1, {}, dst_params, frozenset({"w1"}))
    # A transformed family is always staged, even when the parameters are adjacent.
    assert local.dst.data_ptr() != storage.data_ptr()
    received = torch.arange(12, dtype=torch.float32).view(2, 2, 3)
    local.dst.copy_(received)

    NCCLM2NCopyService._scatter_stacked(local, transform)

    assert torch.equal(dst_params["w0"], received[0])
    assert torch.equal(dst_params["w1"], torch.full((2, 3), -1.0))
    assert set(transform.finalized) == {"w1"}
    assert torch.equal(transform.finalized["w1"], received[1])


@pytest.mark.skipif(
    not _has_nccl_m2n_python_package(),
    reason="install NVIDIA/nccl-extensions and NCCL4Py to run the M2N integration test",
)
def test_nccl_m2n_stacked_experts_match_per_expert_and_nccl_with_mxfp8(
    monkeypatch: pytest.MonkeyPatch,
):
    """Planner-built expert stacks, per-expert M2N and NCCL send/recv agree bitwise.

    Half of the destination experts go through the receiver-side MXFP8 transform and
    half stay BF16, so the stacked path exercises both writebacks in one call.
    """
    from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
    from megatron.core.inference.quantization.utils import resolve_mxfp8_backend
    from megatron.core.resharding.copy_services.nccl_copy_service import NCCLCopyService
    from megatron.core.resharding.planner import _build_tensor_reshard_specs

    Utils.initialize_distributed()
    if torch.cuda.get_device_properties(torch.cuda.current_device()).major < 10:
        pytest.skip("MXFP8 integration requires a Blackwell GPU")
    world_size = dist.get_world_size()
    if world_size < 4 or world_size % 2:
        pytest.skip("requires an even distributed world size >= 4")

    monkeypatch.setenv("NCCL_RESHARD_COPY_ALGORITHM", "PACK")
    rank = dist.get_rank()
    mesh_size = world_size // 2
    src_ep = tuple(range(mesh_size))
    dst_ep = tuple(range(mesh_size, world_size))
    is_source = rank in src_ep
    experts_per_rank = 2
    num_experts = mesh_size * experts_per_rank
    shape = (64, 128)
    dst_by_rank, src_by_name = _expert_rosters(src_ep, dst_ep, num_experts, shape)
    local_rank = (src_ep if is_source else dst_ep).index(rank)
    local_names = [
        f"decoder.layers.0.mlp.experts.linear_fc1.weight{i}" for i in range(experts_per_rank)
    ]
    global_index = {name: local_rank * experts_per_rank + i for i, name in enumerate(local_names)}

    def expert_weight(expert: int) -> torch.Tensor:
        values = torch.randn(shape, generator=torch.Generator().manual_seed(1234 + expert))
        return values.to(device="cuda", dtype=torch.bfloat16)

    backend = resolve_mxfp8_backend("torch")
    converted = {local_names[1]}

    def destination_state():
        tensors = {
            name: torch.zeros(shape, dtype=torch.bfloat16, device="cuda") for name in local_names
        }
        buffers = {
            name: MXFP8Tensor.from_bf16(
                torch.zeros(shape, dtype=torch.bfloat16, device="cuda"), backend=backend
            )
            for name in converted
        }
        transform = MXFP8ReshardTransform(
            convertible_params=set(converted), persistent_buffers=buffers, backend=backend
        )
        return tensors, buffers, transform

    source = {name: expert_weight(global_index[name]) for name in local_names} if is_source else {}

    results = {}
    for mode in ("stacked", "per_expert"):
        monkeypatch.setenv("MEGATRON_NCCL_M2N_STACK_EXPERTS", "1" if mode == "stacked" else "0")
        specs, error = _build_tensor_reshard_specs(dst_by_rank, src_by_name, rank)
        assert error is None
        assert len(specs) == (1 if mode == "stacked" else num_experts)
        tensors, buffers, transform = destination_state() if not is_source else ({}, {}, None)
        service = NCCLM2NCopyService()
        service.set_model_roles(is_source=is_source, is_destination=not is_source)
        plan = ReshardPlan(send_ops=[], recv_ops=[], tensor_reshard_specs=specs)
        assert service.execute_plan(plan, source, tensors, transform=transform)
        torch.cuda.synchronize()
        service.close()
        results[mode] = (tensors, buffers)

    # NCCL send/recv reference: the same transform hooks on received BF16 copies.
    tensors, buffers, transform = destination_state() if not is_source else ({}, {}, None)
    service = NCCLCopyService()
    pending = []
    for name in local_names:
        if is_source:
            service.submit_send(source[name], dst_ep[local_rank])
        else:
            full_slice = (slice(None), slice(None))
            target = (
                transform.prepare_recv(name, full_slice)[0] if name in converted else tensors[name]
            )
            service.submit_recv(target, src_ep[local_rank])
            pending.append((name, full_slice, target))
    service.run()
    torch.cuda.synchronize()
    for name, full_slice, target in pending:
        if name in converted:
            transform.finalize_recv(name, full_slice, [target])
    torch.cuda.synchronize()
    results["nccl"] = (tensors, buffers)

    local_ok = True
    if not is_source:
        for name in local_names:
            expected_bf16 = expert_weight(global_index[name])
            for mode, (tensors, buffers) in results.items():
                if name in converted:
                    expected = MXFP8Tensor.from_bf16(expected_bf16, backend=backend)
                    local_ok &= torch.equal(buffers[name].data, expected.data)
                    local_ok &= torch.equal(
                        buffers[name].scale.view(torch.uint8), expected.scale.view(torch.uint8)
                    )
                    # The BF16 parameter of a converted expert is not written.
                    local_ok &= not tensors[name].any().item()
                else:
                    local_ok &= torch.equal(tensors[name], expected_bf16)
    status = torch.tensor(int(local_ok), dtype=torch.int32, device="cuda")
    dist.all_reduce(status, op=dist.ReduceOp.MIN)
    assert status.item() == 1
