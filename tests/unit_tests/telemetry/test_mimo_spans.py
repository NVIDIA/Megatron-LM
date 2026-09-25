# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('nemo.lens', reason='requires the optional otel dependency')
pytest.importorskip('opentelemetry.sdk', reason='requires the optional otel dependency')

import nemo.lens.handle as lens_handle_module
import nemo.lens.state as lens_state_module
import opentelemetry.metrics._internal as metrics_module
import opentelemetry.trace as trace_module
from nemo.lens import NemoLensConfig, setup_telemetry
from opentelemetry import propagate
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._once import Once
from torch import nn

from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.pipeline_parallel.bridge_communicator import (
    BridgeCommunicator,
    CommRole,
    RankCommInfo,
)
from megatron.core.telemetry.span_groups import MegatronSpanGroup


class _LanguageModel(nn.Module):
    mtp_process = False

    def forward(self, **kwargs):
        return kwargs['decoder_input']


def _model():
    model = MimoModel.__new__(MimoModel)
    nn.Module.__init__(model)
    model.language_model = _LanguageModel()
    return model


def _sender_bridge():
    communicator = BridgeCommunicator.__new__(BridgeCommunicator)
    communicator.requires_backward = False
    communicator.current_rank = 0
    communicator.src_grid = SimpleNamespace(rank_offset=0, size=1)
    communicator.dest_grid = SimpleNamespace(rank_offset=1, size=1)
    communicator.comm_map = {0: RankCommInfo(role=CommRole.SENDER, send_to_ranks=[1])}
    communicator.src_tp_leaders = [0]
    communicator.dest_tp_leaders = [1]
    communicator.src_grid_broadcast_ranks = []
    communicator.dest_grid_broadcast_ranks = []
    communicator.src_module_name = 'vision'
    communicator.dest_module_name = 'language'
    communicator.skip_shape_exchange = True
    communicator.dest_cp_reduce_pg = None
    communicator.comm_dtype = torch.float32
    communicator._split_tensor_at_batch_dim = lambda tensor, count: [tensor] * count
    communicator._run_batched_payload_p2p = lambda *args, **kwargs: None
    return communicator


def _isolate_telemetry_globals(monkeypatch):
    monkeypatch.setattr(lens_handle_module, '_INITIALIZED', False)
    monkeypatch.setattr(trace_module, '_TRACER_PROVIDER', None)
    monkeypatch.setattr(trace_module, '_TRACER_PROVIDER_SET_ONCE', Once())
    monkeypatch.setattr(metrics_module, '_METER_PROVIDER', None)
    monkeypatch.setattr(metrics_module, '_METER_PROVIDER_SET_ONCE', Once())
    monkeypatch.setattr(lens_state_module, '_ENABLED_GROUPS', frozenset())
    monkeypatch.setattr(lens_state_module, '_PP_TRACE_CARRIER', None)


@pytest.fixture
def lens_telemetry(monkeypatch):
    handles = []
    previous_propagator = propagate.get_global_textmap()
    _isolate_telemetry_globals(monkeypatch)

    def setup(span_groups):
        if handles:
            raise RuntimeError('Lens telemetry is already set up for this test')
        exporter = InMemorySpanExporter()
        config = NemoLensConfig(
            enabled=True,
            export_strategy='all_ranks',
            traces_enabled=True,
            metrics_enabled=False,
            logs_enabled=False,
            span_groups=span_groups,
            _span_group_cls=MegatronSpanGroup,
        )
        handle = setup_telemetry(config, rank=0, world_size=1, span_exporter=exporter)
        handles.append(handle)
        return handle, exporter

    try:
        yield setup
    finally:
        for handle in handles:
            handle.shutdown()
        propagate.set_global_textmap(previous_propagator)


def test_mimo_model_and_bridge_spans_have_attributes_and_parent(lens_telemetry):
    handle, exporter = lens_telemetry('microbatch,communication')
    tensor = torch.ones(2, 3)

    with handle.tracer.start_as_current_span('test.parent') as parent:
        parent_span_id = parent.get_span_context().span_id
        _model()._run_language_model(first_stage=True, last_stage=True, decoder_input=tensor)
        _sender_bridge().send_forward(tensor, microbatch_id=7)
    handle.shutdown()

    spans = {span.name: span for span in exporter.get_finished_spans()}
    model_span = spans['megatron.mimo.language.forward']
    bridge_span = spans['megatron.mimo.bridge.send_forward']

    assert model_span.parent.span_id == parent_span_id
    assert bridge_span.parent.span_id == parent_span_id
    assert model_span.attributes['megatron.mimo.language.first_pipeline_stage'] is True
    assert bridge_span.attributes['megatron.mimo.microbatch.id'] == 7
    assert bridge_span.attributes['megatron.mimo.bridge.peer_ranks'] == (1,)
    assert bridge_span.attributes['megatron.mimo.input.bytes'] == (
        tensor.numel() * tensor.element_size()
    )


def test_mimo_spans_are_absent_when_groups_are_disabled(lens_telemetry):
    handle, exporter = lens_telemetry('default')
    tensor = torch.ones(2, 3)

    with handle.tracer.start_as_current_span('test.parent'):
        _model()._run_language_model(first_stage=True, last_stage=True, decoder_input=tensor)
        _sender_bridge().send_forward(tensor, microbatch_id=7)
    handle.shutdown()

    assert [span.name for span in exporter.get_finished_spans()] == ['test.parent']
