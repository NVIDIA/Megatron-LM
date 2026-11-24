# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Portions adapted from NVIDIA-NeMo/RL under Apache-2.0.
# https://github.com/NVIDIA-NeMo/RL/tree/34dd6b3b4f8f108554ec0c9312d83ba7e639da9d
"""CPU numerical checks using native MCore metadata and shard planning."""

import gc
import time
import traceback
from multiprocessing.queues import Queue
from types import SimpleNamespace

import pytest
import torch

from megatron.core.resharding import coresident
from megatron.core.resharding.coresident import plan_local_sources
from megatron.core.resharding.utils import ParameterMetadata
from megatron.core.transformer.moe.experts import _contiguous_expert_stack


def _local_sources(
    tensors: dict[str, torch.Tensor],
    sources: list[ParameterMetadata],
    destinations: list[ParameterMetadata],
) -> dict[str, coresident.RefitSource]:
    return plan_local_sources(
        tensors, sources, destinations, learner_rank=sources[0].owner_rank
    ).local_sources


def _metadata(
    name: str,
    tensor: torch.Tensor,
    *,
    rank: int,
    group: list[int],
    dim: int = 0,
    stride: int = 1,
    sizes: list[int] | None = None,
    is_tp: bool = True,
    resolved: str | None = None,
) -> ParameterMetadata:
    return ParameterMetadata(
        name=name,
        resolved_name=resolved or name,
        shape=tuple(tensor.shape),
        dtype=tensor.dtype,
        element_size=tensor.element_size(),
        is_tp=is_tp,
        partition_dim=dim,
        partition_stride=stride,
        partition_sizes=sizes,
        owner_rank=rank,
        tensor_parallel_group_ranks=group,
    )


@pytest.fixture
def cpu_copy_stream(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls = []
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda: SimpleNamespace(synchronize=lambda: calls.append("synchronized")),
    )
    return calls


@pytest.mark.parametrize("tp", [2, 4])
def test_glu_stride_two_fragments_copy_and_follow_next_publication(
    tp: int, cpu_copy_stream: list[str]
) -> None:
    full = torch.arange(16 * 12, dtype=torch.float32).reshape(16, 12)
    block = 8 // tp
    for rank in range(tp):
        expected = torch.cat(
            (
                full[rank * block : (rank + 1) * block],
                full[8 + rank * block : 8 + (rank + 1) * block],
            )
        )
        source = _metadata("weight", full, rank=rank, group=[rank], stride=2)
        destination = _metadata("weight", expected, rank=rank, group=list(range(tp)), stride=2)
        mapped = _local_sources({"weight": full}, [source], [destination])
        assert isinstance(mapped["weight"], coresident.RefitSlices)
        assert len(mapped["weight"].parts) == 2
        assert tuple(mapped["weight"].shape) == tuple(expected.shape)
        private = torch.nn.Parameter(torch.zeros_like(expected), requires_grad=False)
        assert (
            coresident.bind_shared(mapped, [("weight", private)], {id(private)})["tensors_bound"]
            == 0
        )
        copied = coresident.copy_refit(mapped, [("weight", private)])
        assert copied["bytes"] == private.numel() * private.element_size()
        assert copied["bytes_shared"] == 0
        torch.testing.assert_close(private, expected, rtol=0, atol=0)
        old = private.detach().clone()
        full.add_(100)
        torch.testing.assert_close(private, old, rtol=0, atol=0)
        coresident.copy_refit(mapped, [("weight", private)])
        torch.testing.assert_close(private, expected + 100, rtol=0, atol=0)
    assert cpu_copy_stream


@pytest.mark.parametrize("tp", [2, 4])
def test_unequal_packed_mamba_components_match_per_component_golden(
    tp: int, cpu_copy_stream: list[str]
) -> None:
    full = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8)
    component_sizes = [4, 8, 4]
    for rank in range(tp):
        expected_parts = []
        offset = 0
        for size in component_sizes:
            width = size // tp
            expected_parts.append(full[offset + rank * width : offset + (rank + 1) * width])
            offset += size
        expected = torch.cat(expected_parts)
        source = _metadata("weight", full, rank=rank, group=[rank], sizes=component_sizes)
        destination = _metadata(
            "weight",
            expected,
            rank=rank,
            group=list(range(tp)),
            sizes=[size // tp for size in component_sizes],
        )
        mapped = _local_sources({"weight": full}, [source], [destination])
        assert isinstance(mapped["weight"], coresident.RefitSlices)
        assert len(mapped["weight"].parts) == len(component_sizes)
        private = torch.nn.Parameter(torch.zeros_like(expected), requires_grad=False)
        assert (
            coresident.bind_shared(mapped, [("weight", private)], {id(private)})["tensors_bound"]
            == 0
        )
        coresident.copy_refit(mapped, [("weight", private)])
        torch.testing.assert_close(private, expected, rtol=0, atol=0)


def test_replicated_and_resolved_individual_expert_remain_unchanged() -> None:
    bias = torch.arange(12, dtype=torch.float32)
    expert = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    resolved = "decoder.layers.0.mlp.experts.linear_fc1.weight7"
    local = "decoder.layers.0.mlp.experts.linear_fc1.weight0"
    sources = {"bias": bias, resolved: expert}
    source_metadata = [
        _metadata("bias", bias, rank=3, group=[3], is_tp=False),
        _metadata(local, expert, rank=3, group=[3], is_tp=False, resolved=resolved),
    ]
    destination_metadata = [
        _metadata("engine.bias", bias, rank=3, group=[0, 1, 2, 3], is_tp=False, resolved="bias"),
        _metadata(local, expert, rank=3, group=[3], is_tp=False, resolved=resolved),
    ]
    mapped = _local_sources(sources, source_metadata, destination_metadata)
    for name, expected in [("engine.bias", bias), (local, expert)]:
        assert isinstance(mapped[name], torch.Tensor)
        assert mapped[name].data_ptr() == expected.data_ptr()
        torch.testing.assert_close(mapped[name], expected, rtol=0, atol=0)


def test_contiguous_alias_and_strided_private_copy_update_at_refit(
    cpu_copy_stream: list[str],
) -> None:
    full_row = torch.arange(96, dtype=torch.float32).reshape(8, 12)
    full_column = full_row.clone()
    expected_row = full_row[4:].clone()
    expected_column = full_column[:, 6:].clone()
    source_metadata = [
        _metadata("row", full_row, rank=1, group=[1]),
        _metadata("column", full_column, rank=1, group=[1], dim=1),
    ]
    destination_metadata = [
        _metadata("row", expected_row, rank=1, group=[0, 1]),
        _metadata("column", expected_column, rank=1, group=[0, 1], dim=1),
    ]
    mapped = _local_sources(
        {"row": full_row, "column": full_column}, source_metadata, destination_metadata
    )
    row = torch.nn.Parameter(torch.zeros_like(expected_row), requires_grad=False)
    column = torch.nn.Parameter(torch.zeros_like(expected_column), requires_grad=False)
    named = [("row", row), ("column", column)]
    assert coresident.bind_shared(mapped, named, {id(row), id(column)})["tensors_bound"] == 1
    assert row.data_ptr() == mapped["row"].data_ptr()
    assert column.data_ptr() != mapped["column"].data_ptr()
    initial = coresident.copy_refit(mapped, named)
    assert initial == {
        "bytes": float(column.numel() * column.element_size()),
        "bytes_shared": float(row.numel() * row.element_size()),
    }
    full_row.add_(100)
    full_column.add_(100)
    torch.testing.assert_close(row, expected_row + 100, rtol=0, atol=0)
    torch.testing.assert_close(column, expected_column, rtol=0, atol=0)
    assert coresident.copy_refit(mapped, named) == initial
    torch.testing.assert_close(column, expected_column + 100, rtol=0, atol=0)
    assert len(cpu_copy_stream) == 2


@pytest.mark.parametrize("stale_source", [False, True])
def test_logical_shape_or_stale_source_metadata_is_rejected(stale_source: bool) -> None:
    full = torch.arange(96, dtype=torch.float32).reshape(8, 12)
    source = _metadata("weight", full[:6] if stale_source else full, rank=0, group=[0])
    destination = _metadata(
        "weight", torch.empty(4 if stale_source else 5, 12), rank=0, group=[0, 1]
    )
    with pytest.raises((ValueError, RuntimeError), match="shape|metadata"):
        _local_sources({"weight": full}, [source], [destination])


_IPC_HANDSHAKE_TIMEOUT_S = 120


def _tp_ipc_full_weights() -> dict[str, torch.Tensor]:
    shapes = {"row": (8, 8), "column": (8, 8), "glu": (16, 8)}
    return {
        name: (((torch.arange(rows * columns).reshape(rows, columns) + offset) % 13 - 6) / 8).to(
            torch.bfloat16
        )
        for offset, (name, (rows, columns)) in enumerate(shapes.items())
    }


def _tp_ipc_golden(publication: int) -> dict[str, torch.Tensor]:
    full = _tp_ipc_full_weights()
    for weight in full.values():
        weight.add_(0.5 * publication)
    return {
        "row": full["row"][4:].clone(),
        "column": full["column"][:, 4:].clone(),
        "glu": torch.cat((full["glu"][4:8], full["glu"][12:16])),
    }


def _tp_ipc_learner(device: int, to_engine: Queue, from_engine: Queue, results: Queue) -> None:
    try:
        torch.cuda.set_device(device)
        source = {name: weight.cuda() for name, weight in _tp_ipc_full_weights().items()}
        metadata = [
            _metadata(
                name,
                weight,
                rank=1,
                group=[1],
                dim=int(name == "column"),
                stride=2 if name == "glu" else 1,
            )
            for name, weight in source.items()
        ]
        torch.cuda.synchronize()
        to_engine.put((coresident.export_refit_handles(source.items()), metadata))
        assert from_engine.get(timeout=_IPC_HANDSHAKE_TIMEOUT_S) == ("consumed", 0)
        # This handshake represents reader drain before the learner writes.
        for weight in source.values():
            weight.add_(0.5)
        torch.cuda.synchronize()
        to_engine.put(("published", 1))
        # Keep every exported allocation alive until the consumer releases it.
        assert from_engine.get(timeout=_IPC_HANDSHAKE_TIMEOUT_S) == ("consumed", 1)
        results.put({"role": "learner", "status": "PASS", "publications": 2})
    except BaseException:
        results.put({"role": "learner", "status": "FAIL", "traceback": traceback.format_exc()})
        raise


def _tp_ipc_consume(from_learner: Queue, to_learner: Queue) -> dict[str, int | bool]:
    handles, source_metadata = from_learner.get(timeout=_IPC_HANDSHAKE_TIMEOUT_S)
    source = coresident.open_refit_handles(handles)
    initial = _tp_ipc_golden(0)
    destination_metadata = [
        _metadata(
            name,
            weight,
            rank=1,
            group=[0, 1],
            dim=int(name == "column"),
            stride=2 if name == "glu" else 1,
        )
        for name, weight in initial.items()
    ]
    mapped = _local_sources(source, source_metadata, destination_metadata)
    assert isinstance(mapped["row"], torch.Tensor) and mapped["row"].is_contiguous()
    assert isinstance(mapped["column"], torch.Tensor) and not mapped["column"].is_contiguous()
    assert isinstance(mapped["glu"], coresident.RefitSlices) and len(mapped["glu"].parts) == 2
    destination = {
        name: torch.nn.Parameter(torch.zeros_like(weight, device="cuda"), requires_grad=False)
        for name, weight in initial.items()
    }
    bound = coresident.bind_shared(
        mapped, destination.items(), {id(weight) for weight in destination.values()}
    )
    assert bound["tensors_bound"] == 1
    assert destination["row"].data_ptr() == mapped["row"].data_ptr()
    assert destination["column"].data_ptr() != mapped["column"].data_ptr()
    assert (
        destination["glu"].untyped_storage().data_ptr()
        != source["glu"].untyped_storage().data_ptr()
    )
    private_before_update = {}
    previous_outputs = {}
    checks = 0
    for publication in range(2):
        if publication:
            assert from_learner.get(timeout=_IPC_HANDSHAKE_TIMEOUT_S) == ("published", 1)
            for name in ("column", "glu"):
                torch.testing.assert_close(
                    destination[name], private_before_update[name], rtol=0, atol=0
                )
        golden = {name: weight.cuda() for name, weight in _tp_ipc_golden(publication).items()}
        if publication:
            torch.testing.assert_close(destination["row"], golden["row"], rtol=0, atol=0)
            assert all(
                not torch.equal(destination[name], golden[name]) for name in ("column", "glu")
            )
        copied = coresident.copy_refit(mapped, destination.items())
        assert copied == {"bytes": 192.0, "bytes_shared": 64.0}
        for name, weight in destination.items():
            torch.testing.assert_close(weight, golden[name], rtol=0, atol=0)
            inputs = ((torch.arange(3 * weight.shape[1]).reshape(3, -1) % 5) - 2).to(
                device="cuda", dtype=torch.bfloat16
            )
            actual = torch.nn.functional.linear(inputs, weight)
            expected = torch.nn.functional.linear(inputs, golden[name])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if publication:
                assert not torch.equal(actual, previous_outputs[name])
            previous_outputs[name] = actual.clone()
            checks += 1
        torch.cuda.synchronize()
        if not publication:
            private_before_update = {
                name: destination[name].detach().clone() for name in ("column", "glu")
            }
            torch.cuda.synchronize()
            to_learner.put(("consumed", 0))
    return {
        "publications": 2,
        "linear_checks": checks,
        "contiguous_alias_only": True,
        "private_refresh": True,
    }


def _tp_ipc_engine(device: int, from_learner: Queue, to_learner: Queue, results: Queue) -> None:
    try:
        torch.cuda.set_device(device)
        report = _tp_ipc_consume(from_learner, to_learner)
        # The consumer frame and its aliasing Parameters must die before producer exit.
        torch.cuda.synchronize()
        gc.collect()
        to_learner.put(("consumed", 1))
        results.put({"role": "engine", "status": "PASS", **report})
    except BaseException:
        results.put({"role": "engine", "status": "FAIL", "traceback": traceback.format_exc()})
        raise


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA IPC and GPU linear kernels")
def test_tp_ipc_shards_two_processes_two_publications() -> None:
    """Conceptual TP2 shards on one GPU; no distributed TP or MPS performance claim."""
    device = torch.cuda.current_device()
    context = torch.multiprocessing.get_context("spawn")
    to_engine, to_learner, results = (context.Queue() for _ in range(3))
    learner = context.Process(target=_tp_ipc_learner, args=(device, to_engine, to_learner, results))
    engine = context.Process(target=_tp_ipc_engine, args=(device, to_engine, to_learner, results))
    processes = (learner, engine)
    try:
        for process in processes:
            process.start()
        reports = {}
        deadline = time.monotonic() + 300
        for _ in processes:
            report = results.get(timeout=max(1.0, deadline - time.monotonic()))
            assert report["status"] == "PASS", report
            assert report["role"] not in reports, report
            reports[report["role"]] = report
        for process in processes:
            process.join(30)
            assert process.exitcode == 0, (process.name, process.exitcode)
        assert set(reports) == {"learner", "engine"}
        assert reports["engine"]["linear_checks"] == 6
        assert reports["engine"]["contiguous_alias_only"] and reports["engine"]["private_refresh"]
    finally:
        survivors = []
        for process in processes:
            if process.pid is None:
                continue
            if process.is_alive():
                process.terminate()
            process.join(10)
            if process.is_alive():
                process.kill()
                process.join(5)
            if process.is_alive():
                survivors.append(process.pid)
        for queue in (to_engine, to_learner, results):
            queue.cancel_join_thread()
            queue.close()
        assert not survivors, f"IPC children did not exit: {survivors}"


@pytest.mark.parametrize("contiguous", [True, False])
def test_bind_shared_expert_stacks_uses_native_megatron_buffers(contiguous):
    # Use the installed class so the test cannot invent an unavailable Core API.
    from megatron.core.transformer.moe.experts import InferenceGroupedMLP

    experts = InferenceGroupedMLP.__new__(InferenceGroupedMLP)
    torch.nn.Module.__init__(experts)
    experts.num_local_experts = 2
    experts._concatenated_weights_built = False
    sources = {}
    for linear_name, shape in [("linear_fc1", (4, 3)), ("linear_fc2", (3, 4))]:
        linear = torch.nn.Module()
        packed = torch.arange(2 * shape[0] * shape[1], dtype=torch.float32).reshape(2, *shape)
        for index in range(2):
            source = packed[index] if contiguous else packed[index].clone()
            linear.register_parameter(f"weight{index}", torch.nn.Parameter(source))
            sources[f"{linear_name}.weight{index}"] = source
        experts.add_module(linear_name, linear)

    experts._build_concatenated_weights()
    for name, stack in (("linear_fc1", experts._fc1_weight), ("linear_fc2", experts._fc2_weight)):
        for i in range(2):
            before = sources[f"{name}.weight{i}"]
            torch.testing.assert_close(stack[i], before)
            assert (stack[i].data_ptr() == before.data_ptr()) == contiguous
            assert stack[i].data_ptr() == getattr(getattr(experts, name), f"weight{i}").data_ptr()


def test_flashinfer_clamp_survives_native_shared_packing():
    from unittest.mock import patch

    from megatron.core.transformer.moe import experts as core

    m = core.InferenceGroupedMLP.__new__(core.InferenceGroupedMLP)
    torch.nn.Module.__init__(m)
    m.num_local_experts = 2
    m._concatenated_weights_built = False
    m._activation_clamp_scale = 7.0
    m._flashinfer_clamp_limit = None
    m._flashinfer_activation_type = 1
    m._nvfp4_recipe = None
    m._nvfp4_weights = None
    m.ep_group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    m.inference_grouped_gemm_backend = core.InferenceGroupedGemmBackend.FLASHINFER
    src = {}
    for name, shape in (("linear_fc1", (2, 4, 3)), ("linear_fc2", (2, 3, 4))):
        lin = torch.nn.Module()
        w = torch.zeros(shape, dtype=torch.bfloat16)
        for i in range(2):
            p = torch.nn.Parameter(w[i])
            lin.register_parameter("weight" + str(i), p)
            src[f"{name}.weight{i}"] = w[i]
        setattr(m, name, lin)
    m.eval()
    captured = []

    def kernel(*args, **kwargs):
        captured.append(kwargs["clamped_relu2_limit"])
        return args[0]

    with (
        patch.object(core.InferenceMode, "is_active", return_value=True),
        patch.object(core, "HAVE_FLASHINFER", True),
        patch.object(core, "ActivationType", SimpleNamespace(ClampedRelu2=1), create=True),
        patch.object(core, "fused_moe", SimpleNamespace(cutlass_fused_moe=kernel), create=True),
    ):
        m.forward(
            torch.zeros(1, 3, dtype=torch.bfloat16),
            None,
            torch.ones(1, 2),
            torch.zeros(1, 2, dtype=torch.int),
        )
    assert captured[0].tolist() == [7.0]
    assert m._fc1_weight.data_ptr() == src["linear_fc1.weight0"].data_ptr()
    assert m._fc2_weight.data_ptr() == src["linear_fc2.weight0"].data_ptr()


@pytest.mark.parametrize("layout", ["packed", "reversed", "padded", "separate", "strided"])
def test_contiguous_expert_stack_preserves_storage_or_falls_back(layout: str) -> None:
    backing = torch.arange(3 * 8 * 4, dtype=torch.float32).reshape(3, 8, 4)
    weights = list(backing.unbind())
    if layout == "reversed":
        weights.reverse()
    elif layout == "padded":
        weights = [weight[:4] for weight in weights]
    elif layout == "separate":
        weights = [weight.clone() for weight in weights]
    elif layout == "strided":
        weights = [weight.t() for weight in weights]
    stack = _contiguous_expert_stack(weights)
    if layout != "packed":
        assert stack is None
        return
    assert stack is not None
    assert stack.is_contiguous()
    assert stack.data_ptr() == backing.data_ptr()
    backing.add_(3)
    assert torch.equal(stack, backing)


def test_contiguous_expert_stack_accepts_offset_subsets_and_rejects_mixed_shapes() -> None:
    backing = torch.randn(6, 8, 4)
    weights = list(backing[2:5].unbind())
    stack = _contiguous_expert_stack(weights)
    assert stack is not None
    assert torch.equal(stack, backing[2:5])
    assert stack.data_ptr() == weights[0].data_ptr()
    assert _contiguous_expert_stack([weights[0], weights[1][:4]]) is None
