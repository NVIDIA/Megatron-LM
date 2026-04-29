# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.inference.contexts.attention_context.mha_metadata import NonGraphedMHAMetadata
from megatron.core.inference.contexts.gpu_view import ContextGPUView


def _gpu_view(slot_id: int) -> ContextGPUView:
    view = ContextGPUView(
        max_requests=4,
        max_tokens=16,
        max_kv_blocks=8,
        device=torch.cuda.current_device(),
    )
    view.current_snapshot_slot_id = slot_id
    return view


def test_mha_metadata_keeps_per_snapshot_gpu_views_distinct():
    metadata = NonGraphedMHAMetadata(
        block_count_total=8,
        max_kv_block_count=8,
        max_requests=4,
        block_size_tokens=16,
        max_seqlen=128,
    )
    first_view = _gpu_view(0)
    second_view = _gpu_view(1)

    first = metadata.set_state_data(
        padded_active_request_count=2,
        max_seqlen_q=4,
        max_seqlen_k=16,
        gpu_view=first_view,
        snapshot_slot_id=0,
    )
    second = metadata.set_state_data(
        padded_active_request_count=3,
        max_seqlen_q=8,
        max_seqlen_k=32,
        gpu_view=second_view,
        snapshot_slot_id=1,
    )

    assert first.state_data["query_lengths"].data_ptr() != second.state_data[
        "query_lengths"
    ].data_ptr()
    assert first.state_data["block_table"].data_ptr() != second.state_data["block_table"].data_ptr()

    metadata.activate_snapshot_state(0)
    assert metadata.state_data is first.state_data
    assert metadata.state_data["max_seqlen_q"] == 4

    metadata.activate_snapshot_state(1)
    assert metadata.state_data is second.state_data
    assert metadata.state_data["max_seqlen_q"] == 8
