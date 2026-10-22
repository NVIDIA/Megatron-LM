# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Transport boundary tests; real multi-rank kernel parity remains separate."""

import gc
import weakref
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.moe import replica_weight_transport as transport_module
from megatron.core.transformer.moe.replica_expert_runtime import (
    ReplicaExpertRuntime,
    ReplicaPlan,
    _WeightDirection,
)
from megatron.core.transformer.moe.replica_weight_transport import (
    ReplicaOwnership,
    ReplicaPlacement,
    ReplicaTransferHandle,
    ReplicaTransportCapabilities,
    ReplicaTransportConfig,
    ReplicaWeightLayout,
    ReplicaWeightTransport,
    create_replica_weight_transport,
)

pytestmark = pytest.mark.launch_on_gb200


class _RecordingTransport(ReplicaWeightTransport):
    capabilities = ReplicaTransportCapabilities(
        weight_formats=("bf16", "mxfp8"), grad_dtypes=(torch.float32,)
    )

    def __init__(self, config):
        super().__init__(config)
        self.prepared = []
        self.started = []
        self.waited = []

    @property
    def grad_dtype(self):
        return self.config.grad_dtype

    def projection_views(self, projection_index):
        raise AssertionError("These boundary tests do not allocate TE buffers.")

    def prepare_plan(self, placement):
        plan = super().prepare_plan(placement)
        self.prepared.append(plan)
        return plan

    def start_weight_sync(self, *, sources, plan):
        self.validate_plan(plan)
        handle = ReplicaTransferHandle(self, object(), (plan, sources))
        self.started.append(handle)
        return handle

    def wait_weight_sync(self, handle):
        self.waited.append(handle)

    def start_grad_reduce(self, *, native_grads, plan, projections):
        self.validate_plan(plan)
        return ReplicaTransferHandle(self, object(), (plan, native_grads, projections))

    def wait_grad_reduce(self, handle):
        self.waited.append(handle)


def _config():
    return ReplicaTransportConfig(
        group=None,
        device=torch.device("cpu"),
        world_size=2,
        num_local_home_experts=2,
        num_local_replica_slots=1,
        member_shapes=((4, 4), (4, 4)),
        weight_format="bf16",
        rowwise_scale_shapes=None,
        columnwise_scale_shapes=None,
        grad_dtype=torch.float32,
        num_sms=None,
    )


def _plan(table=None, version=1):
    if table is None:
        table = torch.tensor([[2], [0]], dtype=torch.int32)
    return ReplicaPlan(torch.arange(6), table, version)


def _runtime():
    # Exercise the real runtime lifecycle without TE/GTP construction.
    runtime = object.__new__(ReplicaExpertRuntime)
    runtime.transport = _RecordingTransport(_config())
    runtime.device = torch.device("cpu")
    runtime.world_size = 2
    runtime.num_local_replica_slots = 1
    runtime.ownership = ReplicaOwnership(home_experts=2)
    runtime._prepared_plans = weakref.WeakKeyDictionary()
    runtime._prefetch_plan = runtime._completed_plan = None
    runtime._prefetch_handle = runtime._completed_handle = None
    runtime.prepare_source_weights = lambda direction: None
    runtime.projections = ()
    runtime.weight_cache = None
    runtime.acquire_workspace = lambda: None
    return runtime


@pytest.mark.parametrize("backend", ["replica_hybridep"])
def test_reserved_backends_fail_before_allocating_or_importing_peer_tma(backend, monkeypatch):
    import builtins

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert "replica_peer_tma_transport" not in name
        assert "_symmetric_memory" not in name
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(NotImplementedError, match="Use replica_peer_tma"):
        create_replica_weight_transport(backend, _config())


def test_prepared_plan_rejects_invalid_metadata_and_foreign_transport():
    transport = _RecordingTransport(_config())
    placement = ReplicaPlacement(_plan().experts_to_copy, ReplicaOwnership(2), 1)
    prepared = transport.prepare_plan(placement)
    with pytest.raises(ValueError, match="different transport"):
        _RecordingTransport(_config()).validate_plan(prepared)
    with pytest.raises(ValueError, match="int32"):
        transport.prepare_plan(replace(placement, slot_to_expert=placement.slot_to_expert.long()))
    with pytest.raises(ValueError, match="home expert count"):
        transport.prepare_plan(replace(placement, ownership=ReplicaOwnership(3)))
    explicit = ReplicaOwnership(2, torch.tensor([0, 1]), torch.tensor([0, 0]))
    with pytest.raises(ValueError, match="fixed uniform"):
        transport.prepare_plan(replace(placement, ownership=explicit))
    with pytest.raises(ValueError, match="both rank and local-index"):
        ReplicaOwnership(2, owner_rank=torch.tensor([0, 1]))


def test_runtime_repeated_wait_retains_completion_and_rejects_stale_plan():
    runtime = _runtime()
    first = _plan()
    runtime.start_prefetch(first)
    with pytest.raises(RuntimeError, match="already outstanding"):
        runtime.start_prefetch(_plan())
    runtime.wait_prefetch(first)
    runtime.wait_prefetch(first)
    assert runtime.transport.waited == [runtime.transport.started[0]] * 2

    # Backward reuses scheduling metadata but has a distinct completion.
    runtime.start_backward_prefetch(first)
    runtime.wait_prefetch(first)
    assert len(runtime.transport.prepared) == 1
    assert runtime.transport.started[0] is not runtime.transport.started[1]
    assert runtime.transport.waited[-1] is runtime.transport.started[1]

    second = _plan(version=2)
    runtime.start_prefetch(second)
    with pytest.raises(RuntimeError, match="changed while outstanding"):
        runtime.wait_prefetch(first)
    runtime.wait_prefetch(second)
    with pytest.raises(RuntimeError, match="started prefetch"):
        runtime.wait_prefetch(first)


def test_fc2_wgrad_callback_reduces_only_after_a_successful_gemm():
    runtime = _runtime()
    events = []
    runtime.start_fc2_grad_reduce = lambda: events.append("reduce")
    store = runtime.make_fc2_wgrad_store()
    assert store.delay_wgrad_compute()
    assert store.context is None

    def gemm(left, right):
        events.append((left, right))

    store.put(("input", "grad_output"), gemm)
    assert events == [("input", "grad_output"), "reduce"]
    events.clear()

    def failing_gemm():
        raise RuntimeError("GEMM failed")

    with pytest.raises(RuntimeError, match="GEMM failed"):
        store.put((), failing_gemm)
    assert not events


def test_runtime_gradient_handoff_keeps_fc2_first_and_reuses_the_backward_plan():
    runtime = _runtime()
    plan = _plan()
    prepared = runtime._prepare_transport_plan(plan)
    handoffs = []

    def projection(index):
        def reduce_scatter(grads):
            handoffs.append(index)
            return grads

        return SimpleNamespace(
            native_grad=torch.full((2, 4, 4), index + 1, dtype=torch.float32),
            native_grad_bases=None,
            gtp_leader=SimpleNamespace(wgrad_reduce_scatter=reduce_scatter),
        )

    runtime.projections = (projection(0), projection(1))
    runtime._backward_plan = plan
    runtime._grad_reduce_plan = None
    runtime._grad_reduce_started = set()
    runtime._grad_reduce_handles = {}
    runtime.start_fc2_grad_reduce()
    fc2_handle = runtime._grad_reduce_handles[1]
    runtime.start_pending_grad_reduces(plan)
    assert runtime._grad_reduce_handles[1] is fc2_handle
    assert fc2_handle.keepalive[0] is prepared
    assert fc2_handle.keepalive[2] == (1,)
    with pytest.raises(RuntimeError, match="started twice"):
        runtime.start_grad_reduce(plan, 1)
    grads = runtime.wait_grad_reduce(plan)
    assert handoffs == [1, 0]
    assert len(grads) == 4
    assert [grad[0, 0].item() for grad in grads] == [1, 1, 2, 2]
    assert runtime._backward_plan is None


def test_plan_cache_does_not_reuse_recycled_slot_or_retain_finished_plan():
    runtime = _runtime()
    first = _plan()
    table = first.experts_to_copy
    first_prepared = runtime._prepare_transport_plan(first)
    first_ref = weakref.ref(first)
    del first
    gc.collect()
    assert first_ref() is None
    assert len(runtime._prepared_plans) == 0
    table.copy_(torch.tensor([[3], [-1]], dtype=torch.int32))
    second_prepared = runtime._prepare_transport_plan(_plan(table, version=2))
    assert second_prepared is not first_prepared
    assert second_prepared.placement.version == 2


@pytest.mark.parametrize(
    ("direction", "layout", "field"),
    [
        (_WeightDirection.FORWARD, ReplicaWeightLayout.ROWWISE, "rowwise"),
        (_WeightDirection.BACKWARD, ReplicaWeightLayout.COLUMNWISE, "columnwise"),
    ],
)
def test_runtime_exposes_mxfp8_direction_and_scales(direction, layout, field):
    runtime = _runtime()
    weight = SimpleNamespace(
        _rowwise_data=torch.zeros(4, dtype=torch.uint8),
        _columnwise_data=torch.ones(4, dtype=torch.uint8),
        _rowwise_scale_inv=torch.zeros(1, dtype=torch.uint8),
        _columnwise_scale_inv=torch.ones(1, dtype=torch.uint8),
    )
    binding = SimpleNamespace(source_tensors=(), data_bases=None, scale_bases=None)
    runtime.projections = (
        SimpleNamespace(
            weight_format="mxfp8", source_tensors=(weight,), binding=lambda direction: binding
        ),
    )
    source = runtime._transport_sources(direction)[0]
    assert source.layout is layout
    assert source.data[0] is getattr(weight, f"_{field}_data")
    assert source.scales[0] is getattr(weight, f"_{field}_scale_inv")
    assert source.data_bases is None


def test_finalize_only_visits_initialized_backends(monkeypatch):
    calls = []
    monkeypatch.setattr(transport_module, "_transport_finalizers", {})
    transport_module.register_replica_transport_finalizer("test", lambda: calls.append("test"))
    transport_module.finalize_replica_weight_transports()
    transport_module.finalize_replica_weight_transports()
    assert calls == ["test"]
