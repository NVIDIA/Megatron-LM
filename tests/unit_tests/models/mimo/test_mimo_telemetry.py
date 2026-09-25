# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from contextlib import contextmanager
from types import SimpleNamespace

import torch
from torch import nn

import megatron.core.models.mimo.model.base as model_module
import megatron.core.models.mimo.submodules.base as submodules_module
import megatron.core.pipeline_parallel.bridge_communicator as bridge_module
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.models.mimo.submodules.base import ModalitySubmodules
from megatron.core.pipeline_parallel.bridge_communicator import (
    BridgeCommunicator,
    CommRole,
    RankCommInfo,
)


class _SpanRecorder:
    def __init__(self):
        self.calls = []
        self.attribute_updates = []

    @contextmanager
    def managed_span(self, group, name, **attributes):
        span = object()
        self.calls.append((group, name, attributes, span))
        yield span

    def set_attributes(self, span, attributes, redact_keys=None):
        self.attribute_updates.append((span, attributes))


def _patch_telemetry(monkeypatch, module, recorder, *, enabled):
    monkeypatch.setattr(module, '_otel_sg_enabled', lambda group: enabled)
    monkeypatch.setattr(module, '_otel_managed_span', recorder.managed_span)
    monkeypatch.setattr(module, '_otel_safe_set_attrs', recorder.set_attributes)


class _TupleEncoder(nn.Module):
    def forward(self, x):
        return x + 1, {'unused': True}


class _IdentityModality(ModalitySubmodules):
    def decode(self, embeddings, data_batch):
        return embeddings


class _Embedding(nn.Module):
    def forward(self, input_ids, position_ids):
        hidden = input_ids.transpose(0, 1).unsqueeze(-1).float()
        return hidden.expand(-1, -1, 2)


class _LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = _Embedding()
        self.mtp_process = False

    def forward(self, **kwargs):
        decoder_input = kwargs.get('decoder_input')
        return decoder_input if decoder_input is not None else torch.ones(1, 1, 2)


class _PartitionAdapter:
    cfg = SimpleNamespace(use_cp=True, seq_parallel=False)

    def shard(self, *, embeddings, labels, loss_mask, packed_seq_params):
        return embeddings, labels, loss_mask, packed_seq_params


def _bare_mimo_model():
    model = MimoModel.__new__(MimoModel)
    nn.Module.__init__(model)
    model.language_model = _LanguageModel()
    model.partition_adapter = None
    model.special_token_ids = {'images': 99}
    return model


def test_encoder_and_projector_spans_are_owned_by_actual_operations(monkeypatch):
    recorder = _SpanRecorder()
    _patch_telemetry(monkeypatch, submodules_module, recorder, enabled=True)
    modality = _IdentityModality(
        encoders={'vision': _TupleEncoder()}, input_projections=[nn.Identity()]
    )
    inputs = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    encoded = modality.encode({'vision': {'x': inputs}})
    projected = modality.project_embeddings(encoded)

    torch.testing.assert_close(projected, inputs + 1)
    assert [(group, name) for group, name, _, _ in recorder.calls] == [
        ('microbatch', 'megatron.mimo.encoder.forward'),
        ('microbatch', 'megatron.mimo.projection.forward'),
    ]
    assert recorder.calls[0][2] == {
        'megatron.mimo.encoder.name': 'vision',
        'megatron.mimo.encoder.frozen': True,
        'megatron.mimo.input.shape': [2, 3],
        'megatron.mimo.input.dtype': 'torch.float32',
    }
    assert recorder.attribute_updates[0][1] == {
        'megatron.mimo.output.shape': [2, 3],
        'megatron.mimo.output.dtype': 'torch.float32',
    }
    assert recorder.calls[1][2]['megatron.mimo.projection.placement'] == 'encoder'


def test_modality_disabled_path_does_not_build_spans_or_metadata(monkeypatch):
    recorder = _SpanRecorder()
    _patch_telemetry(monkeypatch, submodules_module, recorder, enabled=False)
    modality = _IdentityModality(
        encoders={'vision': _TupleEncoder()}, input_projections=[nn.Identity()]
    )
    inputs = torch.ones(2, 3)

    projected = modality.project_embeddings(modality.encode({'vision': {'x': inputs}}))

    torch.testing.assert_close(projected, inputs + 1)
    assert recorder.calls == []
    assert recorder.attribute_updates == []


def test_model_spans_cover_embedding_projection_partition_and_language(monkeypatch):
    recorder = _SpanRecorder()
    _patch_telemetry(monkeypatch, model_module, recorder, enabled=True)
    model = _bare_mimo_model()
    input_ids = torch.tensor([[3, 99]])
    position_ids = torch.tensor([[0, 1]])

    text = model.get_text_embeddings(input_ids, position_ids, model.special_token_ids)
    fused = model.align_embeddings_by_token_positions(
        {'text': text, 'images': torch.tensor([[4.0, 4.0]])}, input_ids, model.special_token_ids
    )
    projected = model._run_language_projection(
        'images', fused, nn.ModuleDict({'images': nn.Identity()})
    )
    language_output = model._run_language_model(
        first_stage=True, last_stage=True, decoder_input=projected
    )
    model.partition_adapter = _PartitionAdapter()
    sharded = model._shard_language_inputs(language_output, None, None)

    torch.testing.assert_close(sharded[0], language_output)
    assert [name for _, name, _, _ in recorder.calls] == [
        'megatron.mimo.embedding.text',
        'megatron.mimo.embedding.fuse',
        'megatron.mimo.projection.forward',
        'megatron.mimo.language.forward',
        'megatron.mimo.input.partition',
    ]
    assert recorder.calls[2][2]['megatron.mimo.projection.placement'] == 'language'
    assert recorder.calls[3][2]['megatron.mimo.language.first_pipeline_stage'] is True


def test_partition_noop_does_not_emit_span(monkeypatch):
    recorder = _SpanRecorder()
    _patch_telemetry(monkeypatch, model_module, recorder, enabled=True)
    model = _bare_mimo_model()
    model.partition_adapter = _PartitionAdapter()
    model.partition_adapter.cfg = SimpleNamespace(use_cp=False, seq_parallel=True)

    assert model._shard_language_inputs(None, None, None) == (None, None, None, None)
    assert recorder.calls == []


def test_model_disabled_path_does_not_build_spans_or_metadata(monkeypatch):
    recorder = _SpanRecorder()
    _patch_telemetry(monkeypatch, model_module, recorder, enabled=False)
    model = _bare_mimo_model()
    tensor = torch.ones(2, 1, 2)

    projected = model._run_language_projection(
        'images', tensor, nn.ModuleDict({'images': nn.Identity()})
    )
    output = model._run_language_model(first_stage=True, last_stage=True, decoder_input=projected)
    model.partition_adapter = _PartitionAdapter()
    sharded = model._shard_language_inputs(output, None, None)

    torch.testing.assert_close(sharded[0], tensor)
    assert recorder.calls == []
    assert recorder.attribute_updates == []


def _bare_bridge(*, role, requires_backward):
    communicator = BridgeCommunicator.__new__(BridgeCommunicator)
    communicator.requires_backward = requires_backward
    communicator.current_rank = 0
    communicator.src_grid = SimpleNamespace(rank_offset=0, size=1)
    communicator.dest_grid = SimpleNamespace(rank_offset=0, size=1)
    peers = [1]
    communicator.comm_map = {
        0: RankCommInfo(
            role=role,
            send_to_ranks=peers if role is CommRole.SENDER else [],
            recv_from_ranks=peers if role is CommRole.RECEIVER else [],
        )
    }
    communicator.src_tp_leaders = [0]
    communicator.dest_tp_leaders = [1]
    communicator.src_grid_broadcast_ranks = []
    communicator.dest_grid_broadcast_ranks = []
    communicator.src_module_name = 'vision'
    communicator.dest_module_name = 'language'
    communicator.skip_shape_exchange = True
    communicator.dest_cp_reduce_pg = None
    communicator.dest_cp_size = 1
    communicator.src_local_leader_rank = 0
    communicator.dest_local_leader_rank = 0
    communicator.comm_dtype = torch.float32
    communicator._split_tensor_at_batch_dim = lambda tensor, count: [tensor] * count
    communicator._run_batched_payload_p2p = lambda *args, **kwargs: None
    return communicator


def test_bridge_noop_member_does_not_emit_span(monkeypatch):
    recorder = _SpanRecorder()
    _patch_telemetry(monkeypatch, bridge_module, recorder, enabled=True)
    communicator = _bare_bridge(role=CommRole.MEMBER, requires_backward=True)

    communicator.send_forward(torch.ones(2, 3), microbatch_id=4)

    assert recorder.calls == []
    assert recorder.attribute_updates == []


def test_bridge_sender_and_receiver_spans_include_local_metadata(monkeypatch):
    recorder = _SpanRecorder()
    _patch_telemetry(monkeypatch, bridge_module, recorder, enabled=True)
    tensor = torch.ones(2, 3)

    sender = _bare_bridge(role=CommRole.SENDER, requires_backward=False)
    sender.send_forward(tensor, microbatch_id=4)

    receiver = _bare_bridge(role=CommRole.RECEIVER, requires_backward=True)
    receiver.send_backward(tensor, microbatch_id=2)

    assert [name for _, name, _, _ in recorder.calls] == [
        'megatron.mimo.bridge.send_forward',
        'megatron.mimo.bridge.send_backward',
    ]
    sender_attributes = recorder.calls[0][2]
    assert sender_attributes['megatron.mimo.microbatch.id'] == 4
    assert sender_attributes['megatron.mimo.bridge.peer_ranks'] == [1]
    assert sender_attributes['megatron.mimo.input.bytes'] == tensor.numel() * tensor.element_size()
    assert recorder.calls[1][2]['megatron.mimo.microbatch.id'] == 2


def test_fused_bridge_span_keeps_distinct_forward_and_backward_ids(monkeypatch):
    recorder = _SpanRecorder()
    _patch_telemetry(monkeypatch, bridge_module, recorder, enabled=True)

    class Probe:
        requires_backward = True

        def _participates_in_telemetry_scope(self, participants):
            return participants == 'source'

        def _telemetry_attributes(self):
            return {}

        @bridge_module._trace_bridge_operation(
            'megatron.mimo.bridge.send_forward_recv_backward',
            participants='source',
            requires_backward=True,
        )
        def exchange(self, tensor, *, forward_microbatch_id=None, backward_microbatch_id=None):
            return tensor

    Probe().exchange(torch.ones(2, 3), forward_microbatch_id=5, backward_microbatch_id=2)

    attributes = recorder.calls[0][2]
    assert attributes['megatron.mimo.microbatch.forward_id'] == 5
    assert attributes['megatron.mimo.microbatch.backward_id'] == 2


def test_bridge_disabled_group_and_backward_do_not_emit_spans(monkeypatch):
    recorder = _SpanRecorder()
    enabled = {'value': True}
    monkeypatch.setattr(bridge_module, '_otel_sg_enabled', lambda group: enabled['value'])
    monkeypatch.setattr(bridge_module, '_otel_managed_span', recorder.managed_span)
    monkeypatch.setattr(bridge_module, '_otel_safe_set_attrs', recorder.set_attributes)
    communicator = _bare_bridge(role=CommRole.SENDER, requires_backward=False)

    assert communicator.recv_backward(microbatch_id=1) is None
    assert recorder.calls == []

    enabled['value'] = False
    communicator.send_forward(torch.ones(2, 3), microbatch_id=4)
    assert recorder.calls == []
