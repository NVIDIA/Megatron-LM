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
