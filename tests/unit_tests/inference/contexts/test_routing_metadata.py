# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.core.inference.contexts.routing_metadata import RoutingMetadata


def _make_context(max_tokens=64, active_token_count=10, using_cuda_graph=False):
    """Build a fake DynamicInferenceContext with the attributes RoutingMetadata uses."""
    ctx = MagicMock()
    ctx.max_tokens = max_tokens
    ctx.active_token_count = active_token_count
    ctx.using_cuda_graph_this_step.return_value = using_cuda_graph
    return ctx


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="RoutingMetadata uses cuda.current_device"
)
class TestRoutingMetadata:

    def test_init_stores_args_and_device(self):
        """__init__ records context, max_tokens, topk, and current cuda device."""
        ctx = _make_context(max_tokens=128)
        rm = RoutingMetadata(ctx, moe_router_topk=2)
        assert rm.context is ctx
        assert rm.max_tokens == 128
        assert rm.moe_router_topk == 2
        assert rm.routing_indices_buffer is None
        assert rm.num_moe_layers is None

    def test_ensure_buffer_short_circuits_when_already_allocated(self):
        """_ensure_buffer_allocated is a no-op when the buffer already exists."""
        ctx = _make_context()
        rm = RoutingMetadata(ctx, moe_router_topk=4)
        sentinel = torch.empty(1, device="cuda")
        rm.routing_indices_buffer = sentinel
        rm._ensure_buffer_allocated()
        assert rm.routing_indices_buffer is sentinel

    def test_ensure_buffer_no_allocation_when_no_moe_layers(self):
        """When RouterReplay has no global instances, the buffer stays None."""
        ctx = _make_context()
        rm = RoutingMetadata(ctx, moe_router_topk=4)
        with patch("megatron.core.inference.contexts.routing_metadata.RouterReplay") as fake_rr:
            fake_rr.global_router_replay_instances = []
            rm._ensure_buffer_allocated()
        assert rm.routing_indices_buffer is None
        assert rm.num_moe_layers == 0

    def test_ensure_buffer_allocates_with_correct_shape(self):
        """When RouterReplay has N instances, the buffer has shape [max_tokens, N, topk]."""
        ctx = _make_context(max_tokens=32)
        rm = RoutingMetadata(ctx, moe_router_topk=4)
        with patch("megatron.core.inference.contexts.routing_metadata.RouterReplay") as fake_rr:
            fake_rr.global_router_replay_instances = [object(), object(), object()]
            rm._ensure_buffer_allocated()
        assert rm.routing_indices_buffer is not None
        assert rm.routing_indices_buffer.shape == (32, 3, 4)
        assert rm.routing_indices_buffer.dtype == torch.int32
        assert rm.num_moe_layers == 3

    def test_get_routing_indices_cuda_graph_no_buffer_returns_none(self):
        """When CUDA graphs are active but no buffer is allocated, returns None."""
        ctx = _make_context(using_cuda_graph=True)
        rm = RoutingMetadata(ctx, moe_router_topk=4)
        assert rm.get_routing_indices() is None

    def test_get_routing_indices_cuda_graph_returns_buffer_slice(self):
        """When CUDA graphs are active, returns buffer[:active_token_count]."""
        ctx = _make_context(active_token_count=5, using_cuda_graph=True)
        rm = RoutingMetadata(ctx, moe_router_topk=2)
        rm.routing_indices_buffer = torch.zeros(64, 3, 2, dtype=torch.int32, device="cuda")
        out = rm.get_routing_indices()
        assert out.shape == (5, 3, 2)

    def test_get_routing_indices_eager_returns_none_when_no_recorded_data(self):
        """In eager mode, returns None if RouterReplay has no recorded data."""
        ctx = _make_context(using_cuda_graph=False)
        rm = RoutingMetadata(ctx, moe_router_topk=2)
        with patch("megatron.core.inference.contexts.routing_metadata.RouterReplay") as fake_rr:
            fake_rr.get_recorded_data.return_value = None
            assert rm.get_routing_indices() is None
            fake_rr.get_recorded_data.return_value = []
            assert rm.get_routing_indices() is None
            fake_rr.get_recorded_data.return_value = [None]
            assert rm.get_routing_indices() is None

    def test_get_routing_indices_eager_stacks_recorded_data(self):
        """In eager mode, stacks per-layer recorded data along dim=1."""
        ctx = _make_context(using_cuda_graph=False)
        rm = RoutingMetadata(ctx, moe_router_topk=2)
        layer_data = [
            torch.zeros(7, 2, dtype=torch.int32, device="cuda"),
            torch.zeros(7, 2, dtype=torch.int32, device="cuda"),
        ]
        with patch("megatron.core.inference.contexts.routing_metadata.RouterReplay") as fake_rr:
            fake_rr.get_recorded_data.return_value = layer_data
            out = rm.get_routing_indices()
        # Stacked → [num_tokens, num_layers, topk]
        assert out.shape == (7, 2, 2)

    def test_enable_static_buffer_recording_calls_router_replay(self):
        """enable_static_buffer_recording calls RouterReplay.set_global_static_buffers if buffer is allocated."""
        ctx = _make_context(max_tokens=16)
        rm = RoutingMetadata(ctx, moe_router_topk=2)
        with patch("megatron.core.inference.contexts.routing_metadata.RouterReplay") as fake_rr:
            fake_rr.global_router_replay_instances = [object(), object()]
            rm.enable_static_buffer_recording()
            fake_rr.set_global_static_buffers.assert_called_once_with(rm.routing_indices_buffer)

    def test_enable_static_buffer_recording_skips_when_no_buffer(self):
        """When no buffer is allocated (no MoE layers), enable does NOT call set_global_static_buffers."""
        ctx = _make_context()
        rm = RoutingMetadata(ctx, moe_router_topk=2)
        with patch("megatron.core.inference.contexts.routing_metadata.RouterReplay") as fake_rr:
            fake_rr.global_router_replay_instances = []
            rm.enable_static_buffer_recording()
            fake_rr.set_global_static_buffers.assert_not_called()

    def test_disable_static_buffer_recording_clears_router_replay(self):
        """disable_static_buffer_recording calls RouterReplay.clear_global_static_buffers."""
        ctx = _make_context()
        rm = RoutingMetadata(ctx, moe_router_topk=2)
        with patch("megatron.core.inference.contexts.routing_metadata.RouterReplay") as fake_rr:
            rm.disable_static_buffer_recording()
            fake_rr.clear_global_static_buffers.assert_called_once()
