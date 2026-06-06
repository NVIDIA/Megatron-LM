# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.inference.async_txn import (
    assert_ep_phase_tag,
    broadcast_ep_accepted_counts,
    broadcast_ep_sampled_tokens,
    broadcast_ep_stop_word_finished_ids,
)


class FakeEPGroup:
    def __init__(self, *, size=2, rank=1):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def test_token_broadcast_gives_identical_survivor_set_across_ranks():
    local_tokens = torch.tensor([99, 4, 99], dtype=torch.int64)
    canonical_tokens = torch.tensor([5, 4, 7], dtype=torch.int64)
    termination_id = torch.tensor([7, 7, 7], dtype=torch.int64)

    def broadcast(tensor, src, group):
        assert src == 0
        assert group.rank() == 1
        tensor.copy_(canonical_tokens)

    broadcast_ep_sampled_tokens(local_tokens, 3, FakeEPGroup(), broadcast_fn=broadcast)

    assert local_tokens.tolist() == [5, 4, 7]
    assert (local_tokens != termination_id).tolist() == [True, True, False]


def test_stop_word_id_broadcast_gives_identical_finish_mask():
    active_request_ids = [101, 102, 103]
    canonical_finish_mask = torch.tensor([0, 1, 0], dtype=torch.int32)

    def broadcast(tensor, src, group):
        tensor.copy_(canonical_finish_mask)

    finished_ids = broadcast_ep_stop_word_finished_ids(
        active_request_ids,
        finished_request_ids=set(),
        group=FakeEPGroup(),
        broadcast_fn=broadcast,
    )

    assert finished_ids == {102}


def test_forced_argmax_tie_uses_canonical_sampled_tokens():
    local_tie_break = torch.tensor([31, 32], dtype=torch.int64)
    canonical_tie_break = torch.tensor([41, 41], dtype=torch.int64)

    def broadcast(tensor, src, group):
        tensor.copy_(canonical_tie_break)

    broadcast_ep_sampled_tokens(local_tie_break, 2, FakeEPGroup(), broadcast_fn=broadcast)

    assert local_tie_break.tolist() == [41, 41]


def test_mtp_accepted_count_broadcast_gives_identical_speculative_prefix():
    local_counts = torch.tensor([0, 2, 1, 9], dtype=torch.int64)
    canonical_counts = torch.tensor([2, 1, 0], dtype=torch.int64)

    def broadcast(tensor, src, group):
        tensor.copy_(canonical_counts)

    broadcast_ep_accepted_counts(local_counts, 3, FakeEPGroup(), broadcast_fn=broadcast)

    assert local_counts.tolist() == [2, 1, 0, 9]


def test_dummy_rank_mirrors_sync_replacement_and_mtp_phase_collectives():
    calls = []

    def broadcast(tensor, src, group):
        calls.append((src, tensor.dtype, tensor.numel()))

    broadcast_ep_sampled_tokens(
        torch.tensor([11, 12], dtype=torch.int64), 2, FakeEPGroup(), broadcast_fn=broadcast
    )
    broadcast_ep_stop_word_finished_ids(
        [101, 102], {102}, FakeEPGroup(), broadcast_fn=broadcast
    )
    broadcast_ep_accepted_counts(
        torch.tensor([1, 0], dtype=torch.int64), 2, FakeEPGroup(), broadcast_fn=broadcast
    )

    assert calls == [
        (0, torch.int64, 2),
        (0, torch.int32, 2),
        (0, torch.int64, 2),
    ]


def test_ep_protocol_does_not_exchange_layout_identifiers():
    request_ids = {101, 102}
    kv_block_ids = {909, 910}
    mamba_slots = {808, 809}
    captured = []

    def broadcast(tensor, src, group):
        captured.append(tensor.clone())

    broadcast_ep_sampled_tokens(
        torch.tensor([11, 12], dtype=torch.int64), 2, FakeEPGroup(), broadcast_fn=broadcast
    )
    broadcast_ep_stop_word_finished_ids(
        [101, 102], {102}, FakeEPGroup(), broadcast_fn=broadcast
    )
    broadcast_ep_accepted_counts(
        torch.tensor([1, 0], dtype=torch.int64), 2, FakeEPGroup(), broadcast_fn=broadcast
    )

    exchanged_values = set()
    for tensor in captured:
        exchanged_values.update(int(value) for value in tensor.tolist())

    assert exchanged_values.isdisjoint(request_ids)
    assert exchanged_values.isdisjoint(kv_block_ids)
    assert exchanged_values.isdisjoint(mamba_slots)


def test_phase_tag_mismatch_raises_explicit_error_instead_of_hanging():
    def all_gather(gathered, local, group):
        gathered[0].copy_(local)
        gathered[1].copy_(torch.tensor([local[0] + 1, local[1], local[2]], dtype=local.dtype))

    with pytest.raises(RuntimeError, match="EP async transaction phase mismatch"):
        assert_ep_phase_tag(
            "sample",
            step_id=4,
            active_request_count=2,
            group=FakeEPGroup(),
            device=torch.device("cpu"),
            all_gather_fn=all_gather,
        )
