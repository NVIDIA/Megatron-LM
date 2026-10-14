# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Cache parity and stream dependencies for the production Shortcut MoE schedule."""

from contextlib import contextmanager, nullcontext
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
    """Use a CPU norm for stream-order fixtures; graph fixtures move it to CUDA."""
    monkeypatch.setattr(
        shortcut_module,
        "TENorm",
        lambda config, hidden_size, eps, **kwargs: _FakeNorm(
            config=config, hidden_size=hidden_size, eps=eps
        ),
    )


@pytest.mark.parametrize("predecessor", ["gdp", "attention"])
@pytest.mark.parametrize("phase", ["prefill", "mixed", "decode"])
def test_overlap_cache_parity_and_stream_dependencies(monkeypatch, predecessor, phase):
    """The two branches preserve outputs, update caches once and join before writes."""
    padding, max_blocks = True, 304
    config = _shortcut_config()
    config.inference_shortcut_moe_overlap = True
    config.inference_shortcut_moe_expert_max_blocks = max_blocks
    config.transformer_impl = "inference_optimized"
    config.mlp_chunks_for_prefill = 1
    decode = phase == "decode"
    phases, waits, records, caps = [], [], [], []
    cache = torch.zeros(4, 1, config.hidden_size)
    updates = torch.zeros(4, dtype=torch.int32)

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            waits.append((self.name, "event", event.name, tuple(phases)))

        def wait_stream(self, other):
            waits.append((self.name, "stream", other.name, tuple(phases)))

    class Event:
        def __init__(self):
            self.name = "event"
            self.recorded = None

        def record(self, stream):
            self.recorded = (stream.name, tuple(phases))

    main, side = Stream("main"), Stream("moe")
    current = main

    @contextmanager
    def on_stream(stream):
        nonlocal current
        saved, current = current, stream
        try:
            yield
        finally:
            current = saved

    def note(name, expected):
        assert current is expected
        phases.append(name)

    class Compute(_FakeCompute):
        def supports_staged_inference(self, context):
            return context.is_dynamic_batching()

        supports_staged_dynamic_inference = supports_staged_inference

        def inference_project(self, hidden_states, context):
            note("project", main)
            return SimpleNamespace(projected=hidden_states + 1, residual=hidden_states)

        def forward_inference_project(self, hidden_states, attention_mask, **kwargs):
            return self.inference_project(hidden_states, kwargs["inference_context"])

        def inference_core(self, projection, context=None):
            note("core", main)
            cache.add_(projection.projected)
            updates.add_(1)
            return SimpleNamespace(output=projection.projected * 2, residual=projection.residual)

        forward_inference_core = inference_core

        def inference_post(self, projection, core):
            note("post", main)
            return core.output + core.residual

        def forward_inference_post_core(self, core):
            return self.inference_post(None, core), None

        def forward(self, hidden_states, **kwargs):
            note("atomic", main)
            cache.add_(hidden_states + 1)
            updates.add_(1)
            output = (hidden_states + 1) * 2 + hidden_states
            return output if predecessor == "gdp" else (output, None)

    class Dispatcher:
        routing_map = torch.ones(1)

    monkeypatch.setattr(
        shortcut_module, "MambaLayer", Compute if predecessor == "gdp" else type(None)
    )
    monkeypatch.setattr(shortcut_module, "NVLSAllGatherVDispatcher", Dispatcher)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: current)
    monkeypatch.setattr(torch.cuda, "stream", on_stream)
    monkeypatch.setattr(
        torch.Tensor, "record_stream", lambda tensor, stream: records.append((tensor, stream))
    )
    block = ShortcutMoEBlock(Compute(config), _FakeMoE(config), overlap_a2a=False).eval()
    block.route_ready_event.name = "route"
    block.moe_layer.mlp.token_dispatcher = Dispatcher()

    def route(shortcut_hidden, **kwargs):
        note("route", side if block.inference_overlap else main)
        return shortcut_hidden * 3, torch.ones_like(shortcut_hidden)

    def dispatch(hidden, probs, *, async_op=False):
        note("dispatch", side if block.inference_overlap else main)
        assert not async_op
        return hidden * 2, probs

    def prepare(hidden, probs, routing_map, *, max_blocks):
        note("prepare", side)
        assert routing_map is block.moe_layer.mlp.token_dispatcher.routing_map
        caps.append(max_blocks)
        return hidden, probs

    def experts(hidden, probs):
        note("experts", side if block.inference_overlap else main)
        return hidden + probs * 5, None

    def finish(output):
        note("sum", side)
        return output, None

    def combine(output, *, async_op=False):
        note("combine", side if block.inference_overlap else main)
        assert not async_op
        return output * 4

    def shared(hidden_states, **kwargs):
        note("shared", main)
        return hidden_states * 3, None, hidden_states, None

    def postprocess(combined_output, shared_expert_output, *args, **kwargs):
        note("postprocess", main)
        return combined_output + shared_expert_output

    monkeypatch.setattr(block, "_read_shortcut_hidden", lambda hidden, **kwargs: hidden)
    monkeypatch.setattr(block, "_get_a2a_overlap_stream", lambda: side)
    monkeypatch.setattr(block, "_moe_router_preprocess", route)
    monkeypatch.setattr(block, "_launch_dispatch", dispatch)
    monkeypatch.setattr(block, "_launch_combine", combine)
    monkeypatch.setattr(block, "_moe_shared_experts", shared)
    monkeypatch.setattr(block, "_postprocess", postprocess)
    block.moe_layer.mlp.routed_experts_compute = experts
    block.moe_layer.mlp.experts = SimpleNamespace(
        prepare_shortcut_overlap=prepare,
        compute_shortcut_overlap=lambda prepared: experts(*prepared)[0],
        finish_shortcut_overlap=finish,
    )
    context = SimpleNamespace(
        is_dynamic_batching=lambda: True, is_decode_only=lambda: decode, num_speculative_tokens=0
    )
    hidden = torch.arange(4 * config.hidden_size, dtype=torch.float32).reshape(
        4, 1, config.hidden_size
    )
    shortcut = hidden

    # Exercise the CUDA-only lifetime branch without moving this event-order
    # test onto the GPU; the separate graph test uses actual CUDA tensors.
    class CudaPaddingMask(torch.Tensor):
        @property
        def is_cuda(self):
            return True

    padding_mask = (
        torch.zeros(4, 1, dtype=torch.bool).as_subclass(CudaPaddingMask) if padding else None
    )
    saved_hidden, saved_shortcut = hidden.clone(), shortcut.clone()
    saved_padding = padding_mask.clone() if padding else None
    kwargs = dict(
        hidden_states=hidden,
        attention_mask=None,
        inference_context=context,
        rotary_pos_emb=None,
        sequence_len_offset=None,
        packed_seq_params=None,
        padding_mask=padding_mask,
        quant_context_factory=lambda *args: on_stream(current),
    )
    with torch.inference_mode(), InferenceMode.active():
        block.inference_overlap = False
        serial = block(**kwargs)
        serial_cache = cache.clone()
        assert torch.all(updates == 1)
        phases.clear()
        waits.clear()
        records.clear()
        cache.zero_()
        updates.zero_()
        block.inference_overlap = True
        output = block(**kwargs)
    torch.testing.assert_close(output, serial)
    torch.testing.assert_close(cache, serial_cache)
    torch.testing.assert_close(hidden, saved_hidden)
    torch.testing.assert_close(shortcut, saved_shortcut)
    if padding:
        torch.testing.assert_close(padding_mask, saved_padding)
    assert torch.all(updates == 1)
    assert caps == [max_blocks]
    assert phases == (
        [
            "route",
            "dispatch",
            "project",
            "prepare",
            "core",
            "experts",
            "sum",
            "combine",
            "post",
            "shared",
            "postprocess",
        ]
        if decode
        else [
            "route",
            "dispatch",
            "prepare",
            "experts",
            "sum",
            "combine",
            "atomic",
            "shared",
            "postprocess",
        ]
    )
    # There is no main→expert fence after the input-ready fork. The side
    # branch is independent until the join protects postprocess/residual use.
    assert [(stream, kind, other) for stream, kind, other, _ in waits] == [
        ("moe", "event", "route"),
        ("main", "stream", "moe"),
    ]
    assert waits[0][3] == ()
    assert waits[-1][3][-1] == "shared"
    assert block.route_ready_event.recorded == ("main", ())
    assert any(tensor is shortcut and stream is side for tensor, stream in records)
    assert any(stream is main for _, stream in records)
    if padding:
        assert any(tensor is padding_mask and stream is side for tensor, stream in records)


@pytest.mark.parametrize(
    "failure", ["static", "speculative", "unsupported", "dispatcher", "mlp_chunks"]
)
def test_overlap_rejects_unsupported_execution_before_dispatch(monkeypatch, failure):
    config = _shortcut_config()
    config.inference_shortcut_moe_overlap = True
    config.transformer_impl = "inference_optimized"
    config.mlp_chunks_for_prefill = 2 if failure == "mlp_chunks" else 1
    monkeypatch.setattr(torch.cuda, "Event", lambda: object())
    compute = _FakeCompute(config)
    compute.supports_staged_dynamic_inference = lambda context: failure != "unsupported"
    block = ShortcutMoEBlock(compute, _FakeMoE(config), overlap_a2a=False).eval()
    block.moe_layer.mlp.token_dispatcher = object()
    context = SimpleNamespace(
        is_dynamic_batching=lambda: failure != "static",
        is_decode_only=lambda: False,
        num_speculative_tokens=int(failure == "speculative"),
    )
    calls = []
    monkeypatch.setattr(block, "_launch_dispatch", lambda *args, **kwargs: calls.append(1))
    with torch.inference_mode(), InferenceMode.active(), pytest.raises(RuntimeError):
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
    ownership. Actual GDP/attention and NVLS kernel coverage lives in their
    dedicated tests and the end-to-end checkpoint replay verifier.
    """
    phase = "decode"
    config = _shortcut_config()
    config.inference_shortcut_moe_overlap = True
    config.inference_shortcut_moe_expert_max_blocks = 304
    config.transformer_impl = "inference_optimized"
    config.mlp_chunks_for_prefill = 1
    rows = 8
    hidden = torch.zeros(rows, 1, config.hidden_size, device="cuda")
    shortcut = hidden
    padding = torch.zeros(rows, 1, dtype=torch.bool, device="cuda")
    live_rows = torch.ones(rows, dtype=torch.int32, device="cuda")
    cache = torch.zeros_like(hidden)
    updates = torch.zeros_like(live_rows)
    context = SimpleNamespace(
        is_dynamic_batching=lambda: True,
        is_decode_only=lambda: phase == "decode",
        num_speculative_tokens=0,
    )

    class Compute(_FakeCompute):
        def supports_staged_inference(self, context):
            return True

        supports_staged_dynamic_inference = supports_staged_inference

        def inference_project(self, hidden_states, context):
            return SimpleNamespace(projected=hidden_states + 1, residual=hidden_states)

        def forward_inference_project(self, hidden_states, attention_mask, **kwargs):
            return self.inference_project(hidden_states, kwargs["inference_context"])

        def inference_core(self, projection, context=None):
            cache.add_(projection.projected * live_rows.view(rows, 1, 1))
            updates.add_(live_rows)
            return SimpleNamespace(output=projection.projected * 2, residual=projection.residual)

        forward_inference_core = inference_core

        def inference_post(self, projection, core):
            return core.output + core.residual

        def forward_inference_post_core(self, core):
            return self.inference_post(None, core), None

        def forward(self, hidden_states, **kwargs):
            projection = self.inference_project(hidden_states, kwargs["inference_context"])
            output = self.inference_post(projection, self.inference_core(projection))
            return output if predecessor == "gdp" else (output, None)

    class Dispatcher:
        routing_map = torch.ones(1, device="cuda")

    monkeypatch.setattr(
        shortcut_module, "MambaLayer", Compute if predecessor == "gdp" else type(None)
    )
    monkeypatch.setattr(shortcut_module, "NVLSAllGatherVDispatcher", Dispatcher)
    block = ShortcutMoEBlock(Compute(config), _FakeMoE(config), overlap_a2a=False).eval().cuda()
    block.moe_layer.mlp.token_dispatcher = Dispatcher()
    side = torch.cuda.Stream()
    monkeypatch.setattr(block, "_get_a2a_overlap_stream", lambda: side)
    monkeypatch.setattr(block, "_read_shortcut_hidden", lambda hidden, **kwargs: hidden)

    def route(shortcut_hidden, *, padding_mask, **kwargs):
        mask = padding_mask.unsqueeze(-1)
        return (shortcut_hidden * 3).masked_fill(mask, 0), torch.ones_like(
            shortcut_hidden
        ).masked_fill(mask, 0)

    def dispatch(hidden, probs, *, async_op=False):
        assert not async_op
        return hidden * 2, probs

    def compute(prepared):
        hidden, probs = prepared
        return hidden + probs * 5

    def prepare(hidden, probs, routing_map, *, max_blocks):
        assert max_blocks == 304
        assert routing_map is block.moe_layer.mlp.token_dispatcher.routing_map
        return hidden, probs

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
    block.moe_layer.mlp.routed_experts_compute = lambda hidden, probs: (
        compute((hidden, probs)),
        None,
    )
    block.moe_layer.mlp.experts = SimpleNamespace(
        prepare_shortcut_overlap=prepare,
        compute_shortcut_overlap=compute,
        finish_shortcut_overlap=lambda output: (output, None),
    )
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
            saved_hidden, saved_shortcut, saved_padding = (
                hidden.clone(),
                shortcut.clone(),
                padding.clone(),
            )
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
            torch.testing.assert_close(shortcut, saved_shortcut, atol=0, rtol=0)
            torch.testing.assert_close(padding, saved_padding)
