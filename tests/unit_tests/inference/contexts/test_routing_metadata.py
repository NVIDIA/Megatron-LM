# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.core.inference.config import AsyncScheduleMode
from megatron.core.inference.contexts.routing_metadata import RoutingMetadata
from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function
from megatron.core.transformer.moe.router import InferenceTopKRouter
from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction

MAX_TOKENS = 32
TOPK = 2
NUM_MOE_LAYERS = 3


def _make_context(active_token_count=10, using_cuda_graph=False):
    """Build the context fields needed by routing metadata.

    Args:
        active_token_count: Number of valid token rows.
        using_cuda_graph: Whether the current forward uses a CUDA graph.

    Returns:
        MagicMock: Context with legacy scheduling by default.
    """
    ctx = MagicMock()
    ctx.max_tokens = MAX_TOKENS
    ctx.active_token_count = active_token_count
    ctx.using_cuda_graph_this_step.return_value = using_cuda_graph
    ctx.config.async_sched_mode = AsyncScheduleMode.LEGACY
    return ctx


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="RoutingMetadata uses cuda.current_device"
)
class TestRoutingMetadata:

    def test_lazy_buffer_allocation_and_recording_lifecycle(self):
        """RoutingMetadata lazily allocates the [max_tokens, num_moe_layers, topk]
        buffer on first use (skipping when no MoE layers are registered), and
        enable/disable recording forwards to RouterReplay only when there's a
        buffer to record into."""
        with patch("megatron.core.inference.contexts.routing_metadata.RouterReplay") as fake_rr:
            # No MoE layers → no buffer allocated, no recording enabled.
            fake_rr.get_instances.return_value = []
            rm = RoutingMetadata(_make_context(), moe_router_topk=TOPK)
            rm._ensure_buffer_allocated()
            assert rm.routing_indices_buffer is None
            assert rm.num_moe_layers == 0
            rm.enable_static_buffer_recording()
            fake_rr.set_global_static_buffers.assert_not_called()

            # Re-binding global instances and allocating produces the expected shape.
            fake_rr.get_instances.return_value = [object()] * NUM_MOE_LAYERS
            rm2 = RoutingMetadata(_make_context(), moe_router_topk=TOPK)
            rm2._ensure_buffer_allocated()
            assert rm2.routing_indices_buffer.shape == (MAX_TOKENS, NUM_MOE_LAYERS, TOPK)
            assert rm2.routing_indices_buffer.dtype == torch.int32

            # Repeat call is a no-op (preserves the existing buffer identity).
            sentinel = rm2.routing_indices_buffer
            rm2._ensure_buffer_allocated()
            assert rm2.routing_indices_buffer is sentinel

            # enable/disable_static_buffer_recording forward to RouterReplay.
            fake_rr.reset_mock()
            rm2.enable_static_buffer_recording()
            fake_rr.set_global_static_buffers.assert_called_once_with(
                rm2.routing_indices_buffer,
                is_mtp_layer=False,
                buffer_index=None,
                layer_indices=None,
            )
            rm2.disable_static_buffer_recording()
            fake_rr.clear_global_static_buffers.assert_called_once_with(is_mtp_layer=False)

    @pytest.mark.parametrize(
        "using_cuda_graph,buffer_allocated,recorded_data,expected_shape",
        [
            # CUDA-graph path: no buffer → None.
            (True, False, None, None),
            # CUDA-graph path: buffer → returns view of buffer[:active_token_count].
            (True, True, None, (10, NUM_MOE_LAYERS, TOPK)),
            # Eager path: no recorded data → None (three forms of "no data").
            (False, False, None, None),
            (False, False, [], None),
            (False, False, [None], None),
            # Eager path: recorded data → stacked along dim=1 → [num_tokens, num_layers, topk].
            (False, False, "valid", (7, 2, TOPK)),
        ],
    )
    def test_get_routing_indices(
        self, using_cuda_graph, buffer_allocated, recorded_data, expected_shape
    ):
        """get_routing_indices either slices the static buffer (CUDA-graph mode)
        or stacks per-layer RouterReplay tensors (eager mode); returns None when
        no data is available."""
        ctx = _make_context(active_token_count=10, using_cuda_graph=using_cuda_graph)
        rm = RoutingMetadata(ctx, moe_router_topk=TOPK)
        if buffer_allocated:
            rm.routing_indices_buffer = torch.zeros(
                MAX_TOKENS, NUM_MOE_LAYERS, TOPK, dtype=torch.int32, device="cuda"
            )
        with patch("megatron.core.inference.contexts.routing_metadata.RouterReplay") as fake_rr:
            if recorded_data == "valid":
                fake_rr.get_recorded_data.return_value = [
                    torch.zeros(7, TOPK, dtype=torch.int32, device="cuda"),
                    torch.zeros(7, TOPK, dtype=torch.int32, device="cuda"),
                ]
            else:
                fake_rr.get_recorded_data.return_value = recorded_data
            out = rm.get_routing_indices()
            if not using_cuda_graph:
                fake_rr.get_recorded_data.assert_called_once_with(is_mtp_layer=False)
        if expected_shape is None:
            assert out is None
        else:
            assert out.shape == expected_shape

    def test_eager_collection_ignores_unrecorded_mtp_router(self, monkeypatch):
        """An unrecorded MTP router does not invalidate base-model routing."""
        monkeypatch.setattr(RouterReplay, "global_router_replay_instances", [])
        base_1 = RouterReplay()
        RouterReplay(is_mtp_layer=True)
        base_2 = RouterReplay()
        base_1_indices = torch.zeros(7, TOPK, dtype=torch.int64, device="cuda")
        base_2_indices = torch.ones(7, TOPK, dtype=torch.int64, device="cuda")
        base_1.record_indices(base_1_indices)
        base_2.record_indices(base_2_indices)

        routing = RoutingMetadata(
            _make_context(active_token_count=7), moe_router_topk=TOPK
        ).get_routing_indices()

        assert routing is not None
        assert routing.shape == (7, 2, TOPK)
        assert torch.equal(routing[:, 0], base_1_indices)
        assert torch.equal(routing[:, 1], base_2_indices)

    @pytest.mark.parametrize("execution", ["eager", "compiled", "graph"])
    @pytest.mark.parametrize("num_tokens", [1, 7])
    @torch.inference_mode()
    def test_async_banks_use_one_graph_and_exclude_mtp(self, monkeypatch, execution, num_tokens):
        """The real router alternates banks without overwriting other layers or padding.

        Args:
            monkeypatch: Fixture isolating the registered routers.
            execution: Eager, compiled, or CUDA-graph execution.
            num_tokens: Number of token rows written by the router.
        """
        monkeypatch.setattr(RouterReplay, "global_router_replay_instances", [])
        RouterReplay()
        mtp = RouterReplay(is_mtp_layer=True)
        router = RouterReplay()
        ctx = _make_context(active_token_count=num_tokens, using_cuda_graph=execution == "graph")
        ctx.config.async_sched_mode = AsyncScheduleMode.ASYNC
        topk = 6
        metadata = RoutingMetadata(ctx, topk)
        metadata.enable_static_buffer_recording()
        router.set_router_replay_action(RouterReplayAction.RECORD)
        assert mtp.static_buffer is None
        storage = metadata.routing_indices_buffer
        # Offset per-layer views can make the compiler copy both backing banks.
        assert router.static_buffer is storage
        assert router.static_buffer.is_contiguous()
        assert router.static_layer_index.storage_offset() == 0
        storage.fill_(-1)
        scores = torch.linspace(-4, 4, 128, device="cuda").repeat(num_tokens, 1)
        logits = scores.clone()
        expert_bias = torch.zeros(128, device="cuda")
        route = (
            topk_routing_with_score_function
            if execution == "eager"
            else InferenceTopKRouter._compiled_topk_routing
        )

        def forward():
            return route(
                logits,
                topk,
                use_pre_softmax=False,
                num_groups=None,
                group_topk=None,
                scaling_factor=2.5,
                score_function="sigmoid",
                expert_bias=expert_bias,
                fused=False,
                router_replay=router,
                dense_output=True,
                precomputed_indices=None,
            )

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                forward()
        torch.cuda.current_stream().wait_stream(warmup_stream)
        if execution == "graph":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                forward()
        compiled_graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        storage.fill_(-1)
        for bank, sign in [(0, 1), (1, -1), (0, -1), (1, 1)]:
            logits.copy_(scores * sign)
            expected = storage.clone()
            expected[bank, :num_tokens, 1, :] = torch.topk(
                logits.sigmoid(), topk, dim=-1, sorted=False
            ).indices.int()
            metadata.buffer_index_cuda.fill_(bank)
            if execution == "graph":
                graph.replay()
            else:
                forward()
            assert torch.equal(storage, expected)
            assert torch.equal(metadata.get_routing_indices(bank), expected[bank, :num_tokens])
        assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == compiled_graphs
        metadata.disable_static_buffer_recording()
        assert router.static_buffer is None and router.static_buffer_index is None
        assert router.static_layer_index is None
