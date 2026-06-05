# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.inference.async_txn import (
    AsyncTxnSkipReason,
    boundary_crossing_request_ids,
    classify_decode_child_launch,
)


class FakeKVAllocator:
    def __init__(self, available: int):
        self.available = available

    def is_memory_available(self, num_blocks: int) -> bool:
        return num_blocks <= self.available


class FakeContext:
    def __init__(
        self,
        *,
        decode_only: bool = True,
        paused_request_count: int = 0,
        total_request_count: int = 2,
        offsets=(0, 1),
        available_blocks: int = 10,
        chunked_prefill_request_id: int = -1,
        num_speculative_tokens: int = 0,
        block_size_tokens: int = 4,
    ):
        self._decode_only = decode_only
        self.paused_request_count = paused_request_count
        self.total_request_count = total_request_count
        self.request_last_kv_block_offset = torch.tensor(offsets, dtype=torch.int32)
        self.request_ids = torch.arange(10, 10 + len(offsets), dtype=torch.int32)
        self.kv_block_allocator = FakeKVAllocator(available_blocks)
        self.chunked_prefill_request_id = chunked_prefill_request_id
        self.num_speculative_tokens = num_speculative_tokens
        self.block_size_tokens = block_size_tokens

    def is_decode_only(self) -> bool:
        return self._decode_only


def test_boundary_crosser_requires_exactly_one_block_per_request():
    context = FakeContext(offsets=(0, 3, 2), block_size_tokens=4, total_request_count=3)

    assert boundary_crossing_request_ids(context) == (11,)
    result = classify_decode_child_launch(context, async_enabled=True)
    assert result.eligible
    assert result.boundary_request_ids == (11,)
    assert result.required_boundary_blocks == 1


def test_unavailable_kv_prevents_preparation():
    context = FakeContext(offsets=(3, 3), block_size_tokens=4, available_blocks=1)

    result = classify_decode_child_launch(context, async_enabled=True)

    assert not result.eligible
    assert result.reason == AsyncTxnSkipReason.KV_RESERVATION_UNAVAILABLE
    assert result.required_boundary_blocks == 2


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"async_enabled": False}, AsyncTxnSkipReason.ASYNC_DISABLED),
        ({"async_enabled": True, "mtp_active": True}, AsyncTxnSkipReason.MTP_ACTIVE),
        ({"async_enabled": True, "pending_admission": True}, AsyncTxnSkipReason.PENDING_ADMISSION),
        (
            {"async_enabled": True, "graph_recapture_barrier": True},
            AsyncTxnSkipReason.GRAPH_RECAPTURE_BARRIER,
        ),
        (
            {"async_enabled": True, "log_interval_barrier": True},
            AsyncTxnSkipReason.LOG_INTERVAL_BARRIER,
        ),
        ({"async_enabled": True, "resume_barrier": True}, AsyncTxnSkipReason.RESUME_BARRIER),
        ({"async_enabled": True, "evict_barrier": True}, AsyncTxnSkipReason.EVICT_BARRIER),
        (
            {"async_enabled": True, "force_pause_barrier": True},
            AsyncTxnSkipReason.FORCE_PAUSE_BARRIER,
        ),
    ],
)
def test_launch_gate_reports_concrete_skip_reasons(kwargs, reason):
    context = FakeContext()

    result = classify_decode_child_launch(context, **kwargs)

    assert not result.eligible
    assert result.reason == reason


def test_decode_only_and_chunked_prefill_are_static_gates():
    non_decode = FakeContext(decode_only=False)
    chunked = FakeContext(chunked_prefill_request_id=17)
    paused = FakeContext(paused_request_count=1, total_request_count=2)

    assert (
        classify_decode_child_launch(non_decode, async_enabled=True).reason
        == AsyncTxnSkipReason.NOT_DECODE_ONLY
    )
    assert (
        classify_decode_child_launch(chunked, async_enabled=True).reason
        == AsyncTxnSkipReason.CHUNKED_PREFILL
    )
    assert (
        classify_decode_child_launch(paused, async_enabled=True).reason
        == AsyncTxnSkipReason.PAUSED_REQUESTS
    )
