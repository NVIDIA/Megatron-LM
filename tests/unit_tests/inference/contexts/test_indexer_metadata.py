# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.core.inference.contexts.indexer_metadata import IndexerMetadata

MAX_TOKENS = 32
TOPK = 4
NUM_DSA_LAYERS = 3


def _make_context(active_token_count=10, using_cuda_graph=False):
    ctx = MagicMock()
    ctx.max_tokens = MAX_TOKENS
    ctx.active_token_count = active_token_count
    ctx.using_cuda_graph_this_step.return_value = using_cuda_graph
    return ctx


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="IndexerMetadata uses cuda.current_device"
)
class TestIndexerMetadata:
    def test_lazy_buffer_allocation_and_recording_lifecycle(self):
        """IndexerMetadata lazily allocates the [max_tokens, num_dsa_layers, topk]
        buffer on first use (skipping when no DSA layers are registered), and
        enable/disable recording forwards to IndexerReplay only when there's a
        buffer to record into."""
        with patch(
            "megatron.core.inference.contexts.indexer_metadata.IndexerReplay"
        ) as fake_replay:
            # No DSA layers -> no buffer allocated, no recording enabled.
            fake_replay.global_indexer_replay_instances = []
            im = IndexerMetadata(_make_context(), dsa_indexer_topk=TOPK)
            im._ensure_buffer_allocated()
            assert im.indexer_indices_buffer is None
            assert im.num_dsa_layers == 0
            im.enable_static_buffer_recording()
            fake_replay.set_global_static_buffers.assert_not_called()

            # Re-binding global instances and allocating produces the expected shape.
            fake_replay.global_indexer_replay_instances = [object()] * NUM_DSA_LAYERS
            im2 = IndexerMetadata(_make_context(), dsa_indexer_topk=TOPK)
            im2._ensure_buffer_allocated()
            assert im2.indexer_indices_buffer.shape == (
                MAX_TOKENS,
                NUM_DSA_LAYERS,
                TOPK,
            )
            assert im2.indexer_indices_buffer.dtype == torch.int32

            # Repeat call is a no-op (preserves the existing buffer identity).
            sentinel = im2.indexer_indices_buffer
            im2._ensure_buffer_allocated()
            assert im2.indexer_indices_buffer is sentinel

            # enable/disable_static_buffer_recording forward to IndexerReplay.
            fake_replay.reset_mock()
            im2.enable_static_buffer_recording()
            fake_replay.set_global_static_buffers.assert_called_once_with(
                im2.indexer_indices_buffer
            )
            im2.disable_static_buffer_recording()
            fake_replay.clear_global_static_buffers.assert_called_once()

    @pytest.mark.parametrize(
        "using_cuda_graph,buffer_allocated,recorded_data,expected_shape",
        [
            (True, False, None, None),
            (True, True, None, (10, NUM_DSA_LAYERS, TOPK)),
            (False, False, None, None),
            (False, False, [], None),
            (False, False, [None], None),
            (False, False, "valid", (7, 2, TOPK)),
        ],
    )
    def test_get_indexer_indices(
        self, using_cuda_graph, buffer_allocated, recorded_data, expected_shape
    ):
        """get_indexer_indices either slices the static buffer (CUDA-graph mode)
        or stacks per-layer IndexerReplay tensors (eager mode); returns None when
        no data is available."""
        ctx = _make_context(active_token_count=10, using_cuda_graph=using_cuda_graph)
        im = IndexerMetadata(ctx, dsa_indexer_topk=TOPK)
        if buffer_allocated:
            im.indexer_indices_buffer = torch.zeros(
                MAX_TOKENS, NUM_DSA_LAYERS, TOPK, dtype=torch.int32, device="cuda"
            )
        with patch(
            "megatron.core.inference.contexts.indexer_metadata.IndexerReplay"
        ) as fake_replay:
            if recorded_data == "valid":
                fake_replay.get_recorded_data.return_value = [
                    torch.zeros(7, TOPK, dtype=torch.int32, device="cuda"),
                    torch.zeros(7, TOPK, dtype=torch.int32, device="cuda"),
                ]
            else:
                fake_replay.get_recorded_data.return_value = recorded_data
            out = im.get_indexer_indices()
        if expected_shape is None:
            assert out is None
        else:
            assert out.shape == expected_shape
