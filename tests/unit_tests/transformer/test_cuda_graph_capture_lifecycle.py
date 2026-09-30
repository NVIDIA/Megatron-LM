# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

import megatron.core.transformer.cuda_graphs as cuda_graphs_module
from megatron.core.transformer.cuda_graphs import _CudagraphGlobalRecord


@pytest.mark.parametrize("failing_phase", ["fwd", "bwd"])
def test_failed_capture_restores_global_state(monkeypatch, failing_phase):
    default_stream = object()
    capture_stream = object()
    state = {"te_capturing": False, "stream": default_stream}

    def set_te_capture(value):
        state["te_capturing"] = value

    def set_stream(stream):
        state["stream"] = stream

    class FailingRunner:
        base_module = torch.nn.Identity()
        gtp_remat = False
        cudagraph_created = False

        def capture(self, phase):
            assert cuda_graphs_module.is_graph_capturing()
            assert state["te_capturing"]
            set_stream(capture_stream)
            if phase == failing_phase:
                raise RuntimeError("injected capture failure")

        def create_fwd_graph(self, *args, **kwargs):
            self.capture("fwd")

        def create_bwd_graph(self):
            self.capture("bwd")

    runner = FailingRunner()
    records = [(runner, "fwd", [], {}, []), (runner, "bwd")]
    monkeypatch.setattr(_CudagraphGlobalRecord, "cudagraph_created", False)
    monkeypatch.setattr(_CudagraphGlobalRecord, "cudagraph_record", records)
    monkeypatch.setattr(cuda_graphs_module, "_IS_GRAPH_CAPTURING", False)
    monkeypatch.setattr(cuda_graphs_module, "fwd_buffer_reuse_ref_count", 0)
    monkeypatch.setattr(cuda_graphs_module, "HAVE_TE_GRAPHS", True)
    monkeypatch.setattr(
        cuda_graphs_module, "TransformerEngineBaseModule", torch.nn.Identity, raising=False
    )
    monkeypatch.setattr(
        cuda_graphs_module, "te_set_capture_start", lambda: set_te_capture(True), raising=False
    )
    monkeypatch.setattr(
        cuda_graphs_module, "te_set_capture_end", lambda: set_te_capture(False), raising=False
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda: {})
    monkeypatch.setattr(torch.cuda, "default_stream", lambda: default_stream)
    monkeypatch.setattr(torch.cuda, "set_stream", set_stream)

    with pytest.raises(RuntimeError, match="injected capture failure"):
        _CudagraphGlobalRecord.create_cudagraphs()

    assert not cuda_graphs_module.is_graph_capturing()
    assert not state["te_capturing"]
    assert state["stream"] is default_stream
    assert not _CudagraphGlobalRecord.cudagraph_created
    assert not runner.cudagraph_created
    assert _CudagraphGlobalRecord.cudagraph_record == records
