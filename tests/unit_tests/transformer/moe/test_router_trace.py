# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the RouterTracer JSONL sink, layer-name parsing, and step
bookkeeping. These test pure serialization / parsing / bookkeeping logic on CPU only.
"""

import json
from unittest.mock import MagicMock

import pytest
import torch

from megatron.core.transformer.moe.router_trace import (
    RouterTracer,
    _parse_router_module_name,
    load_hidden_states_for_record,
    load_logits_for_record,
)


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.mark.parametrize(
    "name,expected",
    [
        ("decoder.layers.3.mlp.router", ("decoder", None, 3)),
        # Stacked/hybrid MTP: the inner "layers.<M>" must not be read as a decoder layer.
        ("mtp.layers.2.mtp_model_layer.layers.5.mlp.router", ("mtp", 2, 5)),
        # Single-layer MTP: no inner stack, so inner layer defaults to 0.
        ("mtp.layers.0.mtp_model_layer.mlp.router", ("mtp", 0, 0)),
        ("embedding.word_embeddings", None),
    ],
)
def test_parse_router_module_name(name, expected):
    # Guards the (block, mtp_idx, layer) identity so MTP and decoder routers with
    # the same layer number never collide.
    assert _parse_router_module_name(name) == expected


class TestRecordIndicesSink:

    def test_stacked_tensor_roundtrip(self, tmp_path):
        """A [num_tokens, num_layers, topk] tensor (the RoutingMetadata layout) is
        serialized as one record per layer with the expected schema."""
        tracer = RouterTracer(str(tmp_path), max_steps=100, rank=0)
        indices = torch.arange(4 * 3 * 2, dtype=torch.int32).reshape(4, 3, 2)
        tracer.record_indices(indices, step=7)
        tracer.flush()

        recs = _read_jsonl(tracer.output_path)
        assert len(recs) == 3
        for layer, rec in enumerate(recs):
            assert rec["step"] == 7
            assert rec["stage"] == "pre_dispatch"
            assert rec["block"] == "decoder"
            assert rec["layer"] == layer
            assert rec["num_tokens"] == 4
            assert rec["topk"] == 2
            assert "mtp_idx" not in rec
            assert rec["top_indices"] == indices[:, layer, :].tolist()

    def test_per_layer_list_emits_mtp_fields(self, tmp_path):
        """The per-layer-list input form carries explicit layer ids and, for MTP,
        the block/mtp_idx fields that keep records collision-free."""
        tracer = RouterTracer(str(tmp_path), max_steps=100, rank=0)
        per_layer = [torch.zeros(2, 2, dtype=torch.int32), torch.ones(2, 2, dtype=torch.int32)]
        tracer.record_indices(per_layer, step=0, layer_ids=[10, 11], block="mtp", mtp_idx=0)
        tracer.flush()

        recs = _read_jsonl(tracer.output_path)
        assert [r["layer"] for r in recs] == [10, 11]
        assert all(r["block"] == "mtp" and r["mtp_idx"] == 0 for r in recs)


class TestStepBookkeeping:

    def _outputs(self):
        return (torch.zeros(2, 2), torch.zeros(2, 2, dtype=torch.int32))

    def test_hook_step_keying_distinguishes_decoder_and_mtp(self, tmp_path):
        """Inference step-boundary detection keys on (block, mtp_idx, layer), so a
        decoder layer and an MTP layer sharing a layer_number don't collide."""
        tracer = RouterTracer(str(tmp_path), max_steps=100, rank=0, training_mode=False)
        module = MagicMock()
        tracer._record(module, (), self._outputs(), identity=("decoder", None, 1))
        tracer._record(module, (), self._outputs(), identity=("mtp", 0, 1))
        assert tracer.step_id == 0 and len(tracer.records) == 2  # distinct keys, no boundary
        tracer._record(module, (), self._outputs(), identity=("decoder", None, 1))
        assert tracer.step_id == 1  # repeat of decoder layer 1 starts a new step

    def test_advance_step_stamps_caller_step_id(self, tmp_path):
        """advance_step(step_id) stamps buffered records with the caller's
        authoritative id (e.g. the training iteration)."""
        tracer = RouterTracer(str(tmp_path), max_steps=10**9, rank=0, training_mode=True)
        tracer.record_indices(torch.zeros(2, 2, 2, dtype=torch.int32))
        tracer.advance_step(5000)
        tracer.flush()
        assert all(r["step"] == 5000 for r in _read_jsonl(tracer.output_path))

    def test_max_steps_bounds_captured_steps_not_absolute_id(self, tmp_path):
        """max_steps bounds the number of captured steps, not the absolute id,
        otherwise resuming at iteration >= max_steps would stop tracing at once."""
        tracer = RouterTracer(str(tmp_path), max_steps=2, rank=0, training_mode=True)
        tracer.advance_step(5000)
        assert not tracer._stopped  # 1 captured < 2, despite the id being 5000
        tracer.advance_step(5001)
        assert tracer._stopped  # 2 captured == max_steps


@pytest.mark.parametrize("kind", ["hidden_states", "logits"])
def test_sidecar_loader_roundtrip(tmp_path, kind):
    """The bf16 sidecar loaders reconstruct the exact tensor from the .bin +
    record offsets (the byte-format contract shared with the analysis tools)."""
    tensor = torch.randn(3, 4, dtype=torch.bfloat16)
    (tmp_path / f"{kind}_rank0.bin").write_bytes(tensor.view(torch.int16).numpy().tobytes())
    nbytes = tensor.numel() * 2
    if kind == "hidden_states":
        rec = {"rank": 0, "hs_offset": 0, "hs_bytes": nbytes, "hs_shape": [3, 4]}
        out = load_hidden_states_for_record(rec, str(tmp_path))
    else:
        rec = {"rank": 0, "logit_offset": 0, "logit_bytes": nbytes, "logit_shape": [3, 4]}
        out = load_logits_for_record(rec, str(tmp_path))
    assert torch.equal(out, tensor)
