# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit + pipeline tests for the durable rollout bank (queued-group path).

Covers the append -> restore round trip, checksum/torn-write handling for both the
JSONL index and the binary sidecars, the consumption-marker filter, manifest +
compaction, and an end-to-end write-through/restore through the real
``_RolloutPipeline`` (reusing the mocks from ``test_grouped_rollouts``).
"""

import asyncio
import json
import os

import numpy as np
import pytest

from megatron.rl.agent.api import Rollout, RolloutGroup, TokenRollout
from megatron.rl.rollout_bank import _LEDGER, _MANIFEST, _TOKENS_BIN, RolloutBank, _segment_name

# Reuse the pipeline mocks so the integration test drives the real pipeline.
from tests.unit_tests.rl.test_grouped_rollouts import MockGenerator, MockInferenceInterface


def make_token_group(members, *, batch_id=0, index_in_batch=0):
    """Build a RolloutGroup of TokenRollout members.

    ``members`` is a list of (tokens, logprobs, mask) triples, each a per-turn
    jagged list, so the sidecar packing is exercised with multi-turn, ragged data.
    """
    rollouts = []
    for tokens, logprobs, mask in members:
        rollouts.append(
            TokenRollout(
                trajectory=tokens,
                reward=1.0,
                logprobs=logprobs,
                generation_mask=mask,
                env_id="test",
                problem_id="p",
                policy_epoch=[[(0, 0)]],
                kv_cache_epoch=[[(0, 0)]],
                num_evictions=[0],
            )
        )
    return RolloutGroup(rollouts=rollouts, batch_id=batch_id, index_in_batch=index_in_batch)


def sample_group(batch_id=0):
    return make_token_group(
        [
            ([[1, 2, 3], [4, 5]], [[-0.1, -0.2, -0.3], [-0.4, -0.5]],
             [[False, True, True], [True, True]]),
            ([[7, 8]], [[-1.5, -2.5]], [[True, True]]),
        ],
        batch_id=batch_id,
    )


def text_group():
    return RolloutGroup(
        rollouts=[
            Rollout(trajectory=["hello world"], reward=0.5, env_id="t",
                    policy_epoch=[[(0, 0)]], kv_cache_epoch=[[(0, 0)]], num_evictions=[0]),
        ],
    )


class TestRoundTrip:
    def test_token_group_round_trip(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(3)
        original = sample_group(batch_id=2)
        uid = bank.append(original)
        bank.close()

        restored = RolloutBank(str(tmp_path)).restore(trained_through=0)
        assert len(restored) == 1
        g = restored[0]
        assert g.uid == uid
        assert g.batch_id == 2
        # token ids are exact (int32)
        assert g.rollouts[0].trajectory == [[1, 2, 3], [4, 5]]
        assert g.rollouts[1].trajectory == [[7, 8]]
        # generation_mask preserved exactly
        assert g.rollouts[0].generation_mask == [[False, True, True], [True, True]]
        # logprobs recovered within fp16 tolerance
        assert np.allclose(g.rollouts[0].logprobs[0], [-0.1, -0.2, -0.3], atol=1e-3)
        assert np.allclose(g.rollouts[1].logprobs[0], [-1.5, -2.5], atol=1e-3)

    def test_text_group_round_trip(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(1)
        bank.append(text_group())
        bank.close()

        restored = RolloutBank(str(tmp_path)).restore(trained_through=0)
        assert len(restored) == 1
        assert isinstance(restored[0].rollouts[0], Rollout)
        assert restored[0].rollouts[0].trajectory == ["hello world"]

    def test_fp16_logprobs_lossy_but_close_tokens_exact(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        toks = list(range(50))
        lps = [round(-0.01 * i, 4) for i in range(50)]
        bank.append(make_token_group([([toks], [lps], [[True] * 50])]))
        bank.close()

        g = RolloutBank(str(tmp_path)).restore(0)[0]
        assert g.rollouts[0].trajectory[0] == toks  # int32 exact
        assert np.allclose(g.rollouts[0].logprobs[0], lps, atol=1e-3)


class TestDurability:
    def test_torn_final_ledger_line_dropped(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        bank.append(sample_group())
        bank.append(sample_group())
        bank.close()

        # Simulate a kill mid-append: a truncated JSON line at the end of the ledger.
        ledger = os.path.join(str(tmp_path), _segment_name(0), _LEDGER)
        with open(ledger, "a") as f:
            f.write('{"uid": "gen-000000/2", "kind": "toke')  # torn, no newline

        restored = RolloutBank(str(tmp_path)).restore(0)
        assert len(restored) == 2  # the two intact records survive

    def test_truncated_sidecar_slice_dropped(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        bank.append(sample_group())
        bank.append(sample_group())
        bank.close()

        # Chop the tail of tokens.bin so the second record's slice is short.
        tokens_bin = os.path.join(str(tmp_path), _segment_name(0), _TOKENS_BIN)
        size = os.path.getsize(tokens_bin)
        with open(tokens_bin, "r+b") as f:
            f.truncate(size - 4)

        restored = RolloutBank(str(tmp_path)).restore(0)
        assert len(restored) == 1  # only the first record's slice is intact

    def test_checksum_mismatch_dropped(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        bank.append(sample_group())
        bank.close()

        ledger = os.path.join(str(tmp_path), _segment_name(0), _LEDGER)
        with open(ledger) as f:
            rec = json.loads(f.readline())
        rec["checksum"] = "0" * 32  # tamper
        with open(ledger, "w") as f:
            f.write(json.dumps(rec) + "\n")

        assert RolloutBank(str(tmp_path)).restore(0) == []


class TestMarkerFilter:
    def test_marker_filter_rules(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(5)
        trained = bank.append(sample_group())   # consumed at 5 <= T=10 -> discard
        rolled_back = bank.append(sample_group())  # consumed at 12 > T=10 -> restore
        _never = bank.append(sample_group())     # no marker -> restore
        bank.mark_consumed(trained, 5)
        bank.mark_consumed(rolled_back, 12)
        bank.close()

        restored = RolloutBank(str(tmp_path)).restore(trained_through=10)
        uids = {g.uid for g in restored}
        assert trained not in uids
        assert rolled_back in uids
        assert _never in uids


class TestCompaction:
    def test_compaction_prunes_and_flips_manifest(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(1)
        consumed = bank.append(sample_group())
        bank.mark_consumed(consumed, 1)
        bank.set_collection(2)
        survivor = bank.append(sample_group())

        bank.checkpoint(2)  # trained_through=2: prune consumed(<=2), keep survivor

        manifest = json.loads((tmp_path / _MANIFEST).read_text())
        assert manifest["trained_through"] == 2
        assert manifest["segments"] == [_segment_name(2)]
        assert manifest["compacted_at"] == 2
        # stale segment dir removed
        assert not (tmp_path / _segment_name(1)).exists()

        restored = RolloutBank(str(tmp_path)).restore(trained_through=2)
        assert len(restored) == 1
        # the survivor's payload is intact after being rewritten by compaction
        assert restored[0].rollouts[0].trajectory == [[1, 2, 3], [4, 5]]
        assert restored[0].rollouts[1].trajectory == [[7, 8]]
        assert survivor  # uid was assigned at append time

    def test_compaction_survivor_survives_next_kill(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(2)
        bank.append(sample_group())
        bank.checkpoint(2)
        # A fresh process restores the compacted survivor.
        assert len(RolloutBank(str(tmp_path)).restore(2)) == 1


class TestPipelineIntegration:
    """Write-through + restore through the real _RolloutPipeline."""

    def _collect(self, tmp_path, num_groups=4, stop_after=None):
        async def run():
            gen = MockGenerator(parallel_generation_tasks=8)
            bank = RolloutBank(str(tmp_path))
            bank.set_collection(0)
            gen._rollout_bank = bank
            request_groups = []
            from megatron.rl.agent.api import GroupedRolloutRequest

            req = GroupedRolloutRequest(
                num_groups=num_groups,
                rollouts_per_group=2,
                inference_interface=MockInferenceInterface(),
                submission_granularity="B",
                consumption_granularity="B",
            )
            async for group in gen.get_grouped_rollouts(req):
                request_groups.append(group)
                if stop_after is not None and len(request_groups) >= stop_after:
                    break
            bank.close()
            return request_groups

        return asyncio.run(run())

    def test_write_through_then_restore(self, tmp_path):
        groups = self._collect(tmp_path, num_groups=4)
        assert len(groups) == 4
        assert all(getattr(g, "uid", None) for g in groups)

        # Fresh process (restart): no markers, T=0 -> all completed groups restored,
        # never regenerated.
        restored = RolloutBank(str(tmp_path)).restore(trained_through=0)
        assert len(restored) == 4
        assert {g.uid for g in restored} == {g.uid for g in groups}

    def test_early_exit_keeps_assembled_groups(self, tmp_path):
        # Break after the first group; write-through means at least that group is
        # already durable (assembly precedes consumption).
        groups = self._collect(tmp_path, num_groups=4, stop_after=1)
        restored = RolloutBank(str(tmp_path)).restore(trained_through=0)
        assert len(restored) >= len(groups) >= 1
