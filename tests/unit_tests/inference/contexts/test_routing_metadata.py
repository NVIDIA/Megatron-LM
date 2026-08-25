# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.core.inference.contexts.routing_metadata import RoutingMetadata

MAX_TOKENS = 32
TOPK = 2
NUM_MOE_LAYERS = 3


def _make_context(active_token_count=10, using_cuda_graph=False):
    ctx = MagicMock()
    ctx.max_tokens = MAX_TOKENS
    ctx.active_token_count = active_token_count
    ctx.using_cuda_graph_this_step.return_value = using_cuda_graph
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
            fake_rr.global_router_replay_instances = []
            rm = RoutingMetadata(_make_context(), moe_router_topk=TOPK)
            rm._ensure_buffer_allocated()
            assert rm.routing_indices_buffer is None
            assert rm.num_moe_layers == 0
            rm.enable_static_buffer_recording()
            fake_rr.set_global_static_buffers.assert_not_called()

            # Re-binding global instances and allocating produces the expected shape.
            fake_rr.global_router_replay_instances = [object()] * NUM_MOE_LAYERS
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
            fake_rr.set_global_static_buffers.assert_called_once_with(rm2.routing_indices_buffer)
            rm2.disable_static_buffer_recording()
            fake_rr.clear_global_static_buffers.assert_called_once()

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
        if expected_shape is None:
            assert out is None
        else:
            assert out.shape == expected_shape
