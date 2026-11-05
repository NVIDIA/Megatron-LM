# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Multi-turn prefix-cache reuse with the MTP draft-KV plane on vs off.

Agentic traffic sends turn N+1 = turn N's prompt + turn N's output + a new tool result. Only
prompt blocks are ever hash-registered (decode-filled blocks never are, with or without MTP), so
turn N+1 can reuse at most turn N's complete prompt blocks. These tests drive the real context
(`_compute_prefix_match` / `add_request`, real allocator and hash registry) through several
turns and compare the main-model match with the draft plane on and off. The MTP trim may cost a
block at a genuine successor disagreement, but it must not stop a session's later turns from
reusing everything after that point.
"""

from dataclasses import dataclass, field
from typing import List

import pytest
import torch

from tests.unit_tests.inference.contexts.test_dynamic_prefix_caching import PrefixCachingTestBase


@dataclass
class TurnStats:
    """What one turn's admission matched, kept, and skipped."""

    kv_matched: List[int] = field(default_factory=list)  # per chunk, before the MTP trim
    inherited: List[int] = field(default_factory=list)  # per chunk, after the MTP trim
    backed_off: List[int] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    skipped_tokens: int = 0
    registered_blocks: int = 0  # complete prompt blocks findable by hash after the turn


class TestMtpMultiTurnPrefixReuse(PrefixCachingTestBase):

    BS = 32

    def _make(self, mtp: bool):
        ctx = self._ctx(
            block_size_tokens=self.BS,
            max_sequence_length=1024,
            enable_chunked_prefill=True,
            num_speculative_tokens=2 if mtp else 0,
            mtp_num_layers=1 if mtp else None,
        )
        assert ctx.enable_mtp_kv_cache == mtp
        return ctx

    @staticmethod
    def _tokens(n, offset):
        return torch.arange(offset, offset + n, device=torch.cuda.current_device())

    def _admit(self, ctx, prompt, request_id, chunks=None) -> TurnStats:
        """Admit one turn the way the engine does: chunk by chunk, carrying the boundary hidden."""
        req = self._req(ctx, prompt, request_id=request_id)
        chunks = chunks or [len(prompt)]
        assert sum(chunks) == len(prompt)
        stats = TurnStats()
        for i, n in enumerate(chunks):
            if i > 0:
                # `update_requests` hands a continuation its own row back.
                ctx.total_request_count -= 1
            ctx.active_token_count = 0
            ctx.num_prefill_requests = 0
            done = req.finished_chunk_token_count
            first = (done + self.BS - 1) // self.BS
            last = (done + n + self.BS - 1) // self.BS
            kv_matched, _ = ctx._find_kv_match_count(req, first, last)
            match = ctx._compute_prefix_match(req, n)
            ctx.add_request(req, prefill_chunk_length=n)
            stats.kv_matched.append(len(kv_matched))
            stats.inherited.append(len(match.matched_block_ids))
            stats.backed_off.append(match.backed_off_blocks)
            stats.reasons.append(match.mtp_backoff_reason)
            stats.skipped_tokens += match.prefix_skip_tokens
            if i < len(chunks) - 1:
                req.finished_chunk_token_count += n
                req.remaining_prompt_tokens = req.remaining_prompt_tokens[n:]
                if ctx.enable_mtp_kv_cache:
                    # The commit pass carries every non-final chunk's last hidden.
                    meta = ctx.mtp_metadata
                    meta.carry_chunk_boundary(
                        hidden=torch.zeros(
                            (1, 1, meta.hidden_size), device="cuda", dtype=meta.hidden_dtype
                        ),
                        req_id=req.request_id,
                        position=req.finished_chunk_token_count - 1,
                    )
        if ctx.enable_mtp_kv_cache:
            ctx.mtp_metadata.invalidate_chunk_boundary()
        # The turn finishes; its blocks stay cached for the next turn.
        row = ctx.total_request_count - 1
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([row], dtype=torch.int32))
        hashes = req.precomputed_block_hashes
        alloc = ctx.kv_block_allocator
        stats.registered_blocks = next(
            (k for k, h in enumerate(hashes) if h not in alloc.kv_hash_to_block_id), len(hashes)
        )
        return stats

    def _session(self, ctx, turn_lengths, chunking):
        """Run a session; turn t's prompt extends turn t-1's by `turn_lengths[t]` new tokens."""
        prompt = self._tokens(0, 0)
        out = []
        for t, n in enumerate(turn_lengths):
            # New tokens per turn: previous output + tool result, distinct across turns.
            prompt = torch.cat([prompt, self._tokens(n, 10_000 * (t + 1))])
            out.append(self._admit(ctx, prompt.clone(), request_id=t + 1, chunks=chunking(prompt)))
        return out

    def _compare(self, turn_lengths, chunking=lambda p: None):
        on = self._session(self._make(True), turn_lengths, chunking)
        off = self._session(self._make(False), turn_lengths, chunking)
        for t, (a, b) in enumerate(zip(on, off)):
            print(
                f"turn {t}: mtp inherited {a.inherited} (kv {a.kv_matched}, backed off "
                f"{a.backed_off} {a.reasons}, skipped {a.skipped_tokens}, registered "
                f"{a.registered_blocks}) | no-mtp inherited {b.inherited} (skipped "
                f"{b.skipped_tokens}, registered {b.registered_blocks})"
            )
        return on, off

    @pytest.mark.internal
    def test_unaligned_turns_lose_nothing(self):
        """The usual case: each prompt ends mid-block, so the last full block's successor is a
        prompt token every later turn shares. Nothing is declined."""
        on, off = self._compare([5 * self.BS + 7, 2 * self.BS + 11, 3 * self.BS + 5])
        for a, b in zip(on, off):
            assert a.inherited == b.inherited
            assert a.skipped_tokens == b.skipped_tokens
            assert sum(a.backed_off) == 0

    @pytest.mark.internal
    def test_block_aligned_turn_costs_one_block_not_the_session(self):
        """Turn 0's prompt ends exactly on a block boundary, so its last block's successor was
        never paired (-1). Turn 1 must decline that block, but turn 1's own later blocks must
        still be registered so turn 2 reuses them."""
        on, off = self._compare([5 * self.BS, 2 * self.BS + 11, 3 * self.BS + 5])
        assert on[1].backed_off == [1] and on[1].reasons == ["unwritten"]
        # Turn 2 must reuse everything turn 1 registered, apart from the one declined block.
        assert on[1].registered_blocks == off[1].registered_blocks
        assert sum(on[2].inherited) >= sum(off[2].inherited) - 1

    @pytest.mark.internal
    def test_continuation_chunk_does_not_privatize_the_rest_of_the_prompt(self):
        """A long turn admitted in chunks whose cached prefix outruns the first chunk (the
        engine sizes the first chunk to the Mamba-capped skip plus the token budget). The
        continuation chunk holds the boundary carry. It may not skip past the seam, but it
        must not make the rest of the prompt private, or every later turn re-prefills it."""

        def chunking(prompt):
            n = len(prompt)
            if n <= 6 * self.BS:
                return None
            first = 2 * self.BS
            return [first, n - first]

        on, off = self._compare(
            [5 * self.BS + 7, 2 * self.BS + 11, 3 * self.BS + 5, 2 * self.BS + 3], chunking
        )
        for t in range(len(on)):
            assert (
                on[t].registered_blocks == off[t].registered_blocks
            ), f"turn {t}: MTP left prompt blocks unregistered"
        # The last turn reuses the whole previous prompt.
        assert sum(on[-1].inherited) == sum(off[-1].inherited)


class TestMtpDeclinedBlockReplacementInvariants(PrefixCachingTestBase):
    """Replacing a declined block must keep the prefix forest valid when several requests in the
    same scheduling pass share that block.

    `_find_kv_match_count` relies on every cached block having all of its ancestors cached (block
    hashes are parent-chained, and a registered block whose parent hash is gone silently becomes a
    root). `_mtp_try_replace_declined_block` deregisters a cached block at admission time, so these
    tests admit two continuations of the same block-aligned prompt in one pass and check the
    forest, ref counts, and block tables after every admission.
    """

    BS = TestMtpMultiTurnPrefixReuse.BS
    # Borrow the turn-admission helpers without re-collecting that class's tests.
    _make = TestMtpMultiTurnPrefixReuse._make
    _tokens = staticmethod(TestMtpMultiTurnPrefixReuse._tokens)
    _admit = TestMtpMultiTurnPrefixReuse._admit

    @staticmethod
    def _check_forest(ctx, requests):
        """Assert the allocator's prefix forest and the in-flight block tables are consistent."""
        alloc = ctx.kv_block_allocator
        hashes = alloc.block_hashes
        registered = (hashes != -1).nonzero().flatten().tolist()

        for block_id in registered:
            # The hash map and the per-block hash agree.
            assert alloc.kv_hash_to_block_id[int(hashes[block_id])] == block_id
            parent = int(alloc.block_parent_id[block_id])
            if parent >= 0:
                assert int(hashes[parent]) != -1, f"block {block_id} has a deregistered parent"
            children = sum(int(alloc.block_parent_id[b]) == block_id for b in registered)
            assert int(alloc.block_child_count[block_id]) == children, (
                f"block {block_id}: child count {int(alloc.block_child_count[block_id])}, "
                f"{children} registered children"
            )

        # Prefix-closed: a request can only match block k if blocks 0..k-1 are cached, so a
        # cached block must never sit behind an uncached ancestor in any request's hash chain.
        for req in requests:
            cached = [h in alloc.kv_hash_to_block_id for h in req.precomputed_block_hashes]
            first_gap = next((k for k, c in enumerate(cached) if not c), len(cached))
            assert not any(
                cached[first_gap:]
            ), f"request {req.request_id}: cached block after uncached ancestor at {first_gap}"

        # In-flight block tables reference live blocks only, with no duplicates within a row, and
        # every block's ref count equals the number of tables holding it. A block evicted while
        # held and handed straight back out by the allocator passes the per-row checks (same
        # hash, nonzero ref count), so only the count comparison catches it.
        free = set(alloc.block_bag[: alloc.pool_avail].tolist())
        holders = {}
        for row in range(ctx.total_request_count):
            ids = [b for b in ctx.request_to_kv_block_ids[row].tolist() if b >= 0]
            assert len(ids) == len(set(ids)), f"row {row} repeats a block id: {ids}"
            for b in ids:
                assert b not in free, f"row {row} references block {b} in the free pool"
                holders[b] = holders.get(b, 0) + 1
        for b, n in holders.items():
            assert (
                int(alloc.block_ref_counts[b]) == n
            ), f"block {b}: ref count {int(alloc.block_ref_counts[b])}, held by {n} rows"

    @pytest.mark.internal
    # A 0-block tail leaves the first request's copy as a childless in-use leaf, the case only
    # the ref-count guard protects; a 2-block tail gives it registered children.
    @pytest.mark.parametrize("tail_blocks", [0, 2])
    @pytest.mark.parametrize("same_continuation", [True, False])
    @pytest.mark.parametrize("stale_checks", [False, True], ids=["engine-order", "stale-checks"])
    def test_same_pass_requests_sharing_the_declined_block(
        self, tail_blocks, same_continuation, stale_checks
    ):
        ctx = self._make(True)
        alloc = ctx.kv_block_allocator

        # Turn 0 ends exactly on a block boundary, so its last block's successor slot is never
        # written and every continuation must decline that block.
        base = self._tokens(5 * self.BS, 0)
        self._admit(ctx, base.clone(), request_id=1)
        declined_hash = self._req(ctx, base.clone(), request_id=99).precomputed_block_hashes[-1]
        declined_block = alloc.kv_hash_to_block_id[declined_hash]
        assert int(alloc.block_ref_counts[declined_block]) == 0

        tail_len = tail_blocks * self.BS + 11
        tail_a = self._tokens(tail_len, 50_000)
        tail_b = tail_a.clone() if same_continuation else self._tokens(tail_len, 70_000)
        reqs = [
            self._req(ctx, torch.cat([base, tail_a]), request_id=2),
            self._req(ctx, torch.cat([base, tail_b]), request_id=3),
        ]

        ctx.active_token_count = 0
        ctx.num_prefill_requests = 0
        if stale_checks:
            # Both availability checks see the declined block cached before either is admitted.
            assert all(all(ctx.check_availability(r)) for r in reqs)
        for req in reqs:
            if not stale_checks:
                assert all(ctx.check_availability(req))
            ctx.add_request(req)
            self._check_forest(ctx, reqs)

        # Only the first admission may replace the declined block; the second must never evict
        # the copy the first request now holds.
        assert ctx.mtp_prefix_replaced_blocks == 1
        first_copy = alloc.kv_hash_to_block_id.get(declined_hash)
        assert first_copy is not None, "the first request's copy of the block was not registered"
        assert int(alloc.block_ref_counts[first_copy]) >= 1

        # Release both and peel the whole cached forest: `evict_lru_blocks` asserts the parent
        # graph is a forest, so a dangling or cyclic chain would fail here.
        rows = torch.tensor(
            [ctx.total_request_count - 2, ctx.total_request_count - 1], dtype=torch.int32
        )
        ctx.release_memory_blocks_from_request_indexes(rows)
        self._check_forest(ctx, reqs)
        evictable = int(alloc.get_evictable_block_count())
        assert evictable > 0
        assert alloc.evict_lru_blocks(evictable)
        assert all(int(h) == -1 for h in alloc.block_hashes.tolist())
