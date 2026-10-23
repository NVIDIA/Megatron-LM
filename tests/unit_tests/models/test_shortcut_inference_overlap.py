# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Minimal runtime guards and CUDA graph checks for Shortcut MoE overlap."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid import shortcut_block as shortcut_module
from megatron.core.models.hybrid.shortcut_block import ShortcutMoEBlock
from tests.unit_tests.models.test_shortcut_block import (
    _FakeCompute,
    _FakeMoE,
    _FakeNorm,
    _shortcut_config,
)


@pytest.fixture(autouse=True)
def shortcut_norm(monkeypatch):
    """Use a lightweight norm for guard tests and graph fixtures."""
    monkeypatch.setattr(
        shortcut_module,
        "TENorm",
        lambda config, hidden_size, eps, **kwargs: _FakeNorm(
            config=config, hidden_size=hidden_size, eps=eps
        ),
    )


@pytest.mark.parametrize(
    "failure,message",
    [("unsupported", "Mamba or Transformer layer"), ("dispatcher", "active NVLS dispatcher")],
)
def test_overlap_rejects_unsupported_execution_before_dispatch(monkeypatch, failure, message):
    config = _shortcut_config()
    config.inference_shortcut_moe_overlap = True
    config.transformer_impl = "inference_optimized"
    config.mlp_chunks_for_prefill = 1
    monkeypatch.setattr(torch.cuda, "Event", lambda: object())

    class Dispatcher:
        pass

    monkeypatch.setattr(shortcut_module, "NVLSAllGatherVDispatcher", Dispatcher)
    compute = _FakeCompute(config)
    if failure != "unsupported":
        monkeypatch.setattr(shortcut_module, "TransformerLayer", _FakeCompute)
    block = ShortcutMoEBlock(compute, _FakeMoE(config), overlap_a2a=False).eval()
    block.moe_layer.mlp.token_dispatcher = Dispatcher() if failure == "unsupported" else object()
    context = SimpleNamespace(
        is_dynamic_batching=lambda: True, is_decode_only=lambda: True, num_speculative_tokens=0
    )
    calls = []
    monkeypatch.setattr(block, "_launch_dispatch", lambda *args, **kwargs: calls.append(1))
    with torch.inference_mode(), InferenceMode.active(), pytest.raises(RuntimeError, match=message):
        block(
            hidden_states=torch.ones(4, 1, 8),
            attention_mask=None,
            inference_context=context,
            rotary_pos_emb=None,
            sequence_len_offset=None,
            packed_seq_params=None,
            padding_mask=None,
            quant_context_factory=None,
        )
    assert not calls


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("predecessor", ["gdp", "attention"])
def test_overlap_graph_replay_preserves_inputs_and_updates_live_cache_once(
    monkeypatch, predecessor
):
    """Real stream/event capture replays both branches with changing padded inputs.

    Lightweight tensor operations isolate the block's fork/join and cache
    ownership. Actual expert kernel coverage lives in the capped-GEMM test;
    distributed NVLS coverage uses the checkpoint replay verifier.
    """
    config = _shortcut_config()
    config.inference_shortcut_moe_overlap = True
    config.inference_shortcut_moe_expert_max_blocks = 304
    config.transformer_impl = "inference_optimized"
    config.mlp_chunks_for_prefill = 1
    rows = 8
    hidden = torch.zeros(rows, 1, config.hidden_size, device="cuda")
    padding = torch.zeros(rows, 1, dtype=torch.bool, device="cuda")
    live_rows = torch.ones(rows, dtype=torch.int32, device="cuda")
    cache = torch.zeros_like(hidden)
    updates = torch.zeros_like(live_rows)
    context = SimpleNamespace(
        is_dynamic_batching=lambda: True, is_decode_only=lambda: True, num_speculative_tokens=0
    )

    class Compute(_FakeCompute):
        def forward(self, hidden_states, **kwargs):
            projected = hidden_states + 1
            cache.add_(projected * live_rows.view(rows, 1, 1))
            updates.add_(live_rows)
            output = projected * 2 + hidden_states
            return output if predecessor == "gdp" else (output, None)

    class Dispatcher:
        routing_map = torch.ones(1, device="cuda")

    monkeypatch.setattr(
        shortcut_module, "MambaLayer", Compute if predecessor == "gdp" else type(None)
    )
    monkeypatch.setattr(
        shortcut_module, "TransformerLayer", Compute if predecessor == "attention" else type(None)
    )
    monkeypatch.setattr(shortcut_module, "NVLSAllGatherVDispatcher", Dispatcher)
    block = ShortcutMoEBlock(Compute(config), _FakeMoE(config), overlap_a2a=False).eval().cuda()
    block.moe_layer.mlp.token_dispatcher = Dispatcher()
    side = torch.cuda.Stream()
    monkeypatch.setattr(ShortcutMoEBlock, "_parallel_stream", side)
    monkeypatch.setattr(block, "_read_shortcut_hidden", lambda hidden, **kwargs: hidden)

    def route(shortcut_hidden, *, padding_mask, **kwargs):
        mask = padding_mask.unsqueeze(-1)
        return (shortcut_hidden * 3).masked_fill(mask, 0), torch.ones_like(
            shortcut_hidden
        ).masked_fill(mask, 0)

    def dispatch(hidden, probs, *, async_op=False):
        assert not async_op
        return hidden * 2, probs

    def combine(output, *, async_op=False):
        assert not async_op
        return output * 4

    monkeypatch.setattr(block, "_moe_router_preprocess", route)
    monkeypatch.setattr(block, "_launch_dispatch", dispatch)
    monkeypatch.setattr(block, "_launch_combine", combine)
    monkeypatch.setattr(
        block,
        "_moe_shared_experts",
        lambda hidden_states, **kwargs: (hidden_states * 3, None, hidden_states, None),
    )
    monkeypatch.setattr(
        block,
        "_postprocess",
        lambda combined_output, shared_expert_output, *args, **kwargs: combined_output
        + shared_expert_output,
    )

    class Experts(torch.nn.Module):
        def forward(self, hidden, tokens_per_expert, probs, *, routing_map, inference_max_blocks=0):
            assert inference_max_blocks == 304
            assert routing_map is block.moe_layer.mlp.token_dispatcher.routing_map
            return hidden + probs * 5, None

    block.moe_layer.mlp.experts = Experts()
    block.moe_layer.mlp.routed_experts_compute = lambda hidden, probs: (hidden + probs * 5, None)
    kwargs = dict(
        hidden_states=hidden,
        attention_mask=None,
        inference_context=context,
        rotary_pos_emb=None,
        sequence_len_offset=None,
        packed_seq_params=None,
        padding_mask=padding,
        quant_context_factory=lambda *args: nullcontext(),
    )
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.inference_mode(), InferenceMode.active():
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                block(**kwargs)
        torch.cuda.current_stream().wait_stream(capture_stream)
        cache.zero_()
        updates.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            captured = block(**kwargs)

        for step, count in enumerate((3, 7, 2)):
            hidden.copy_(torch.arange(hidden.numel(), device="cuda").view_as(hidden) + step)
            padding.copy_((torch.arange(rows, device="cuda") >= count).view_as(padding))
            live_rows.copy_((~padding).reshape(-1).to(torch.int32))
            saved_hidden, saved_padding = hidden.clone(), padding.clone()
            cache.zero_()
            updates.zero_()
            block.inference_overlap = False
            serial = block(**kwargs)
            serial_cache = cache.clone()
            torch.testing.assert_close(updates, live_rows)
            cache.zero_()
            updates.zero_()
            block.inference_overlap = True
            graph.replay()
            torch.testing.assert_close(captured, serial, atol=0, rtol=0)
            torch.testing.assert_close(cache, serial_cache, atol=0, rtol=0)
            torch.testing.assert_close(updates, live_rows)
            torch.testing.assert_close(hidden, saved_hidden, atol=0, rtol=0)
            torch.testing.assert_close(padding, saved_padding)
