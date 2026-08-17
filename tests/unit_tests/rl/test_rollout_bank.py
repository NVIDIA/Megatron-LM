# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit + pipeline tests for the durable rollout bank (queued-group path).

Covers the append -> restore round trip, checksum/torn-write handling for both the
JSONL index and the binary sidecars, the consumption-marker filter, manifest +
compaction, and an end-to-end write-through/restore through the real
``RolloutPipeline`` (reusing the rollout-generation test mocks).
"""

import asyncio
import json
import os
from contextlib import aclosing

import numpy as np
import pytest

from megatron.core.dist_checkpointing.strategies.async_utils import AsyncRequest
from megatron.rl import rl_utils, rollout_bank
from megatron.rl.agent.api import GroupedRolloutRequest, Rollout, RolloutGroup, TokenRollout
from megatron.rl.agent.rollout_pipeline import RolloutPipeline
from megatron.rl.agent.weighted_multi_task import AgentConfig, WeightedMultiTask
from megatron.rl.rollout_bank import (
    _CONSUMED,
    _FORMAT_VERSION,
    _GENERATIONS,
    _LEDGER,
    _MANIFEST,
    _TOKENS_BIN,
    RolloutBank,
    _segment_name,
)
from megatron.rl.types import Rollout as SharedRollout
from megatron.rl.types import RolloutGroup as SharedRolloutGroup
from megatron.rl.types import TokenRollout as SharedTokenRollout
from megatron.training.checkpointing import _register_rollout_bank_compaction

# Reuse the upstream pipeline mocks so the integration test drives the real pipeline.
from tests.unit_tests.rl.test_rollout_generation import MockGenerator, MockInferenceInterface


def test_agent_api_reexports_shared_rollout_types():
    assert SharedRollout is Rollout
    assert SharedRolloutGroup is RolloutGroup
    assert SharedTokenRollout is TokenRollout


def test_rollout_reward_accepts_none():
    rollout = Rollout(trajectory=["prompt"], reward=None)

    assert rollout.reward is None


def make_token_group(members, *, batch_id=0, index_in_batch=0):
    """Build a RolloutGroup of TokenRollout members.

    ``members`` is a list of (tokens, logprobs, mask) triples, each a per-turn
    jagged list, so the sidecar packing is exercised with multi-turn, ragged data.
    """
    rollouts = []
    for member_index, (tokens, logprobs, mask) in enumerate(members):
        rollouts.append(
            TokenRollout(
                trajectory=tokens,
                reward=1.0,
                logprobs=logprobs,
                generation_mask=mask,
                env_id="test",
                problem_id="p",
                completion_ids=[
                    f"completion-{member_index}-{turn_index}" for turn_index in range(len(tokens))
                ],
            )
        )
    return RolloutGroup(rollouts=rollouts, batch_id=batch_id, index_in_batch=index_in_batch)


def sample_group(batch_id=0):
    return make_token_group(
        [
            (
                [[1, 2, 3], [4, 5]],
                [[-0.1, -0.2, -0.3], [-0.4, -0.5]],
                [[False, True, True], [True, True]],
            ),
            ([[7, 8]], [[-1.5, -2.5]], [[True, True]]),
        ],
        batch_id=batch_id,
    )


def text_group():
    return RolloutGroup(rollouts=[Rollout(trajectory=["hello world"], reward=0.5, env_id="t")])


def _manifest_sidecar_bytes(bank_dir):
    """Return the sidecar payload referenced by the published manifest."""
    manifest = json.loads((bank_dir / _MANIFEST).read_text())
    generation = bank_dir / _GENERATIONS / manifest["active_generation"]
    return sum(
        (generation / seg / name).stat().st_size
        for seg in manifest["segments"]
        for name in ("tokens.bin", "logprobs.bin", "masks.bin")
        if (generation / seg / name).exists()
    )


def active_generation_path(bank_dir):
    manifest = json.loads((bank_dir / _MANIFEST).read_text())
    return bank_dir / _GENERATIONS / manifest["active_generation"]


def active_segment_path(bank_dir, iteration):
    return active_generation_path(bank_dir) / _segment_name(iteration)


class TestRoundTrip:
    def test_manifest_and_ledger_record_current_format_version(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(3)
        bank.append(sample_group())
        bank.close()

        manifest = json.loads((tmp_path / _MANIFEST).read_text())
        ledger_path = active_segment_path(tmp_path, 3) / _LEDGER
        record = json.loads(ledger_path.read_text().splitlines()[0])

        assert manifest["format_version"] == _FORMAT_VERSION
        assert manifest["timeline"]
        assert manifest["active_generation"].startswith("generation-")
        assert active_generation_path(tmp_path).parent == tmp_path / _GENERATIONS
        assert (active_generation_path(tmp_path) / _CONSUMED).exists()
        assert record["format_version"] == _FORMAT_VERSION

    @pytest.mark.parametrize(
        "invalid_version",
        [pytest.param(None, id="missing"), pytest.param(_FORMAT_VERSION + 1, id="unsupported")],
    )
    def test_restore_rejects_incompatible_manifest_version(self, tmp_path, invalid_version):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(3)
        bank.append(sample_group())
        bank.close()

        manifest_path = tmp_path / _MANIFEST
        manifest = json.loads(manifest_path.read_text())
        if invalid_version is None:
            manifest.pop("format_version")
        else:
            manifest["format_version"] = invalid_version
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(ValueError, match="Unsupported RolloutBank format_version"):
            RolloutBank(str(tmp_path)).restore(0)

    def test_malformed_manifest_fails_closed_without_overwrite(self, tmp_path):
        manifest_path = tmp_path / _MANIFEST
        malformed = '{"format_version": 1, "segments": ['
        manifest_path.write_text(malformed)

        with pytest.raises(
            ValueError, match=r"Malformed RolloutBank manifest.*line 1, column"
        ) as exc:
            RolloutBank(str(tmp_path))

        assert str(manifest_path) in str(exc.value)
        assert manifest_path.read_text() == malformed

    def test_missing_manifest_with_bank_artifacts_fails_without_initializing(self, tmp_path):
        segment = tmp_path / _segment_name(3)
        segment.mkdir()
        ledger_path = segment / _LEDGER
        ledger_path.write_text("existing bank data\n")

        with pytest.raises(FileNotFoundError, match=r"manifest is missing.*not empty") as exc:
            RolloutBank(str(tmp_path))

        assert str(tmp_path / _MANIFEST) in str(exc.value)
        assert not (tmp_path / _MANIFEST).exists()
        assert ledger_path.read_text() == "existing bank data\n"

    def test_manifest_removed_after_initialization_fails_without_recreating(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        manifest_path = tmp_path / _MANIFEST
        manifest_path.unlink()

        with pytest.raises(FileNotFoundError, match=r"manifest is missing") as exc:
            bank.restore(0)

        assert str(manifest_path) in str(exc.value)
        assert not manifest_path.exists()

    @pytest.mark.parametrize(
        "invalid_version",
        [pytest.param(None, id="missing"), pytest.param(_FORMAT_VERSION + 1, id="unsupported")],
    )
    def test_restore_rejects_incompatible_ledger_version(self, tmp_path, invalid_version):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(3)
        bank.append(sample_group())
        bank.close()

        ledger_path = active_segment_path(tmp_path, 3) / _LEDGER
        record = json.loads(ledger_path.read_text())
        if invalid_version is None:
            record.pop("format_version")
        else:
            record["format_version"] = invalid_version
        ledger_path.write_text(json.dumps(record) + "\n")

        with pytest.raises(ValueError, match="Unsupported RolloutBank format_version"):
            RolloutBank(str(tmp_path)).restore(0)

    def test_encode_returns_named_payload(self, tmp_path):
        assert hasattr(rollout_bank, "EncodedGroup")

        bank = RolloutBank(str(tmp_path))
        bank.set_collection(3)
        encoded = bank._encode(sample_group(), "gen-000003/0")

        assert isinstance(encoded, rollout_bank.EncodedGroup)
        assert encoded.record["uid"] == "gen-000003/0"
        assert encoded.tok_bytes
        assert encoded.lp_bytes
        assert encoded.mask_bytes

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
        assert g.rollouts[0].completion_ids == []
        assert g.rollouts[1].completion_ids == []
        # logprobs recovered within fp16 tolerance
        assert np.allclose(g.rollouts[0].logprobs[0], [-0.1, -0.2, -0.3], atol=1e-3)
        assert np.allclose(g.rollouts[1].logprobs[0], [-1.5, -2.5], atol=1e-3)

    def test_empty_group_round_trip(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        uid = bank.append(RolloutGroup(rollouts=[]))
        bank.close()

        restored = RolloutBank(str(tmp_path)).restore(0)
        assert len(restored) == 1
        assert restored[0].uid == uid
        assert restored[0].rollouts == []

    def test_empty_token_trajectory_round_trip_without_sidecar_files(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        uid = bank.append(make_token_group([([], [], [])]))
        bank.close()

        segment = active_segment_path(tmp_path, 0)
        assert not (segment / _TOKENS_BIN).exists()
        record = json.loads((segment / _LEDGER).read_text())
        assert record["kind"] == "token"
        assert record["tok"]["bytes"] == 0
        assert record["lp"]["bytes"] == 0
        assert record["mask"]["bytes"] == 0
        restored = RolloutBank(str(tmp_path)).restore(0)
        assert len(restored) == 1
        assert restored[0].uid == uid
        assert isinstance(restored[0].rollouts[0], TokenRollout)
        assert restored[0].rollouts[0].trajectory == []
        assert restored[0].rollouts[0].logprobs == []
        assert restored[0].rollouts[0].generation_mask == []

    def test_token_rollout_subclass_uses_token_sidecars(self, tmp_path):
        class DerivedTokenRollout(TokenRollout):
            pass

        member = DerivedTokenRollout(
            trajectory=[[1, 2]],
            reward=1.0,
            logprobs=[[-0.1, -0.2]],
            generation_mask=[[True, True]],
            env_id="test",
            problem_id="p",
            completion_ids=["completion-derived-0"],
        )
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        uid = bank.append(RolloutGroup(rollouts=[member]))
        bank.close()

        record = json.loads((active_segment_path(tmp_path, 0) / _LEDGER).read_text())
        assert record["kind"] == "token"
        assert record["member_type"] == "TokenRollout"
        restored = RolloutBank(str(tmp_path)).restore(0)
        assert restored[0].uid == uid
        assert restored[0].rollouts[0].trajectory == [[1, 2]]

    def test_mixed_rollout_member_types_are_rejected(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        mixed_group = RolloutGroup(rollouts=[sample_group().rollouts[0], text_group().rollouts[0]])

        with pytest.raises(ValueError, match="must not mix TokenRollout and Rollout"):
            bank.append(mixed_group)

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

    @pytest.mark.parametrize("field", ["logprobs", "generation_mask"])
    def test_mixed_optional_field_presence_is_rejected(self, tmp_path, field):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        group = sample_group()
        setattr(group.rollouts[1], field, None)

        with pytest.raises(ValueError, match=f"{field} must be present for all or no rollouts"):
            bank.append(group)


class TestDurability:
    def test_generations_directory_is_fsynced_to_bank_parent(self, tmp_path, monkeypatch):
        events = []
        monkeypatch.setattr(
            rollout_bank, "_fsync_directory", lambda path: events.append(f"dir:{path}")
        )

        RolloutBank(str(tmp_path)).close()

        assert events[0] == f"dir:{tmp_path}"
        assert f"dir:{tmp_path / _GENERATIONS}" in events
        assert events[-1] == f"dir:{tmp_path}"

    def test_manifest_replace_is_followed_by_bank_directory_fsync(self, tmp_path, monkeypatch):
        bank = RolloutBank(str(tmp_path))
        events = []
        real_replace = os.replace

        def replace(src, dst):
            real_replace(src, dst)
            events.append("replace")

        monkeypatch.setattr(os, "replace", replace)
        monkeypatch.setattr(
            rollout_bank, "_fsync_directory", lambda path: events.append(f"dir:{path}")
        )

        bank._write_manifest_atomic({"trained_through": 1, "segments": [], "compacted_at": 0})

        assert events == ["replace", f"dir:{tmp_path}"]

    def test_first_append_fsyncs_new_entries_after_file_contents(self, tmp_path, monkeypatch):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        segment = active_segment_path(tmp_path, 0)
        events = []
        monkeypatch.setattr(os, "fsync", lambda fd: events.append("file"))
        monkeypatch.setattr(
            rollout_bank, "_fsync_directory", lambda path: events.append(f"dir:{path}")
        )

        bank.append(sample_group())

        assert events == ["file", f"dir:{segment}"] * 4

        events.clear()
        bank.append(sample_group())
        assert events == ["file"] * 4

    def test_new_segment_is_durable_before_manifest_publication(self, tmp_path, monkeypatch):
        bank = RolloutBank(str(tmp_path))
        events = []
        monkeypatch.setattr(
            rollout_bank, "_fsync_directory", lambda path: events.append(f"dir:{path}")
        )
        monkeypatch.setattr(
            bank,
            "_write_manifest_atomic",
            lambda manifest: events.append(f"manifest:{manifest['segments'][-1]}"),
        )

        bank.set_collection(7)

        generation = active_generation_path(tmp_path)
        assert events == [f"dir:{generation}", f"manifest:{_segment_name(7)}"]

    def test_compacted_segment_is_durable_before_manifest_publication(self, tmp_path, monkeypatch):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(1)
        bank.append(sample_group())
        events = []
        real_replace = os.replace
        real_write_manifest = bank._write_manifest_atomic

        def replace(src, dst):
            real_replace(src, dst)
            if str(src).endswith(".tmp") and "generation-" in str(src):
                events.append("generation_replace")

        def write_manifest(manifest):
            events.append(f"manifest:{manifest['trained_through']}")
            real_write_manifest(manifest)

        monkeypatch.setattr(os, "replace", replace)
        monkeypatch.setattr(
            rollout_bank, "_fsync_directory", lambda path: events.append(f"dir:{path}")
        )
        monkeypatch.setattr(bank, "_write_manifest_atomic", write_manifest)

        bank.checkpoint(2)

        replace_index = events.index("generation_replace")
        assert events[replace_index : replace_index + 3] == [
            "generation_replace",
            f"dir:{tmp_path / _GENERATIONS}",
            "manifest:2",
        ]

    def test_consumed_marker_batch_opens_and_fsyncs_once(self, tmp_path, monkeypatch):
        bank = RolloutBank(str(tmp_path))
        events = []
        real_open = open
        consumed_path = active_generation_path(tmp_path) / _CONSUMED

        def recording_open(path, mode="r", *args, **kwargs):
            if path == str(consumed_path):
                events.append(f"open:{mode}")
            return real_open(path, mode, *args, **kwargs)

        monkeypatch.setattr(rollout_bank, "open", recording_open, raising=False)
        monkeypatch.setattr(os, "fsync", lambda fd: events.append("file"))
        monkeypatch.setattr(
            rollout_bank, "_fsync_directory", lambda path: events.append(f"dir:{path}")
        )

        bank.mark_consumed_many(["gen-000000/0", "", "gen-000000/1"], 1)

        assert events == ["open:a", "file"]
        markers = [json.loads(line) for line in consumed_path.read_text().splitlines()]
        assert markers == [{"uid": "gen-000000/0", "iter": 1}, {"uid": "gen-000000/1", "iter": 1}]

        events.clear()
        bank.mark_consumed_many(["gen-000000/2", "gen-000000/3"], 2)

        assert events == ["open:a", "file"]

    def test_empty_consumed_marker_batch_does_not_touch_disk(self, tmp_path, monkeypatch):
        bank = RolloutBank(str(tmp_path))
        monkeypatch.setattr(os, "fsync", lambda fd: pytest.fail("unexpected fsync"))

        bank.mark_consumed_many([], 1)
        bank.mark_consumed("", 1)

        assert (active_generation_path(tmp_path) / _CONSUMED).read_text() == ""

    def test_startup_removes_only_unreferenced_generation_directories(self, tmp_path):
        RolloutBank(str(tmp_path)).close()
        active = active_generation_path(tmp_path)
        orphan = tmp_path / _GENERATIONS / "generation-orphan"
        staging = tmp_path / _GENERATIONS / ".generation-interrupted.tmp"
        unrelated = tmp_path / _GENERATIONS / "keep-me"
        orphan.mkdir()
        staging.mkdir()
        unrelated.mkdir()

        RolloutBank(str(tmp_path)).close()

        assert active.exists()
        assert not orphan.exists()
        assert not staging.exists()
        assert unrelated.exists()

    def test_invalid_manifest_does_not_trigger_generation_cleanup(self, tmp_path):
        RolloutBank(str(tmp_path)).close()
        active = active_generation_path(tmp_path)
        orphan = tmp_path / _GENERATIONS / "generation-orphan"
        orphan.mkdir()
        (tmp_path / _MANIFEST).write_text("{")

        with pytest.raises(ValueError, match="Malformed RolloutBank manifest"):
            RolloutBank(str(tmp_path))

        assert active.exists()
        assert orphan.exists()

    def test_torn_final_ledger_line_dropped_and_append_recovers_after_restart(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        bank.append(sample_group())
        bank.append(sample_group())
        bank.close()

        # Simulate a kill mid-append: a truncated JSON line at the end of the ledger.
        ledger = active_segment_path(tmp_path, 0) / _LEDGER
        with open(ledger, "a") as f:
            f.write('{"uid": "gen-000000/2", "kind": "toke')  # torn, no newline

        restored = RolloutBank(str(tmp_path)).restore(0)
        assert len(restored) == 2  # the two intact records survive

        restarted = RolloutBank(str(tmp_path))
        restarted.set_collection(0)
        new_uid = restarted.append(sample_group())
        restarted.close()

        restored = RolloutBank(str(tmp_path)).restore(0)
        assert len(restored) == 3
        assert new_uid == f"{_segment_name(0)}/2"
        assert new_uid in {group.uid for group in restored}

    def test_truncated_sidecar_slice_dropped(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        bank.append(sample_group())
        bank.append(sample_group())
        bank.close()

        # Chop the tail of tokens.bin so the second record's slice is short.
        tokens_bin = active_segment_path(tmp_path, 0) / _TOKENS_BIN
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

        ledger = active_segment_path(tmp_path, 0) / _LEDGER
        with open(ledger) as f:
            rec = json.loads(f.readline())
        rec["checksum"] = "0" * 32  # tamper
        with open(ledger, "w") as f:
            f.write(json.dumps(rec) + "\n")

        assert RolloutBank(str(tmp_path)).restore(0) == []


class TestMarkerFilter:
    @pytest.mark.parametrize(
        "consumed_offset, should_restore",
        [
            pytest.param(-1, False, id="ckpt_minus_one"),
            pytest.param(0, False, id="ckpt"),
            pytest.param(1, True, id="ckpt_plus_one"),
        ],
    )
    def test_marker_filter_at_checkpoint_boundary(self, tmp_path, consumed_offset, should_restore):
        checkpoint_iteration = 10
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(5)
        consumed = bank.append(sample_group())
        never_consumed = bank.append(sample_group())
        bank.mark_consumed(consumed, checkpoint_iteration + consumed_offset)
        bank.close()

        restored = RolloutBank(str(tmp_path)).restore(trained_through=checkpoint_iteration)
        restored_uids = {group.uid for group in restored}
        assert (consumed in restored_uids) is should_restore
        assert never_consumed in restored_uids


class TestCompaction:
    def test_restart_initializes_live_payload_and_enforces_cap(self, tmp_path, caplog):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(1)
        bank.append(sample_group())
        bank.close()
        live_bytes = _manifest_sidecar_bytes(tmp_path)

        with caplog.at_level("WARNING", logger="megatron.rl.rollout_bank"):
            restarted = RolloutBank(str(tmp_path), max_bytes=live_bytes - 1)

        assert restarted._bytes_written == live_bytes
        assert "exceeded --rl-rollout-bank-max-bytes" in caplog.text

    def test_staging_rewrite_does_not_count_as_live_payload(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(1)
        bank.append(sample_group())
        live_bytes = bank._bytes_written
        survivors = bank.restore(0)
        staging = tmp_path / "staging"
        staging.mkdir()

        bank._rewrite_segment(str(staging), 2, survivors)

        assert bank._bytes_written == live_bytes
        assert bank._segment_sidecar_bytes(str(staging)) == live_bytes

    def test_compaction_rebases_payload_to_survivors(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(1)
        consumed = bank.append(sample_group())
        bank.append(sample_group())
        bank.mark_consumed(consumed, 1)
        before_compaction = bank._bytes_written

        bank.checkpoint(2)

        survivor_bytes = _manifest_sidecar_bytes(tmp_path)
        assert survivor_bytes * 2 == before_compaction
        assert bank._bytes_written == survivor_bytes

        bank.set_collection(3)
        bank.append(sample_group())
        assert bank._bytes_written == _manifest_sidecar_bytes(tmp_path)
        assert bank._bytes_written == before_compaction

    def test_rollback_rebases_future_marker_into_new_timeline(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(20)
        uid = bank.append(sample_group())
        bank.checkpoint(20)
        timeline_a = json.loads((tmp_path / _MANIFEST).read_text())["timeline"]
        bank.mark_consumed(uid, 25)
        bank.close()

        restarted = RolloutBank(str(tmp_path))
        restored = restarted.recover(20)
        manifest = json.loads((tmp_path / _MANIFEST).read_text())

        assert {group.uid for group in restored} == {uid}
        assert manifest["timeline"] != timeline_a
        assert (active_generation_path(tmp_path) / _CONSUMED).read_text() == ""

        restarted.checkpoint(26)
        assert {group.uid for group in restarted.restore(26)} == {uid}

    def test_uninterrupted_future_marker_survives_async_checkpoint(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(20)
        uid = bank.append(sample_group())
        bank.mark_consumed(uid, 25)

        bank.checkpoint(20)

        assert {group.uid for group in bank.restore(20)} == {uid}
        marker_lines = (active_generation_path(tmp_path) / _CONSUMED).read_text().splitlines()
        assert [json.loads(line) for line in marker_lines] == [{"uid": uid, "iter": 25}]

        bank.checkpoint(26)
        assert bank.restore(26) == []

    def test_checkpoint_compacts_markers_to_one_per_future_survivor(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(20)
        future = bank.append(sample_group())
        trained = bank.append(sample_group())
        never = bank.append(sample_group())
        bank.mark_consumed(future, 23)
        bank.mark_consumed(future, 25)
        bank.mark_consumed(future, 24)
        bank.mark_consumed(trained, 18)
        bank.mark_consumed("orphan", 30)

        bank.checkpoint(20)

        assert {group.uid for group in bank.restore(20)} == {future, never}
        consumed = active_generation_path(tmp_path) / _CONSUMED
        assert [json.loads(line) for line in consumed.read_text().splitlines()] == [
            {"uid": future, "iter": 25}
        ]

        assert len(consumed.read_text().splitlines()) == 1
        bank.checkpoint(20)
        new_consumed = active_generation_path(tmp_path) / _CONSUMED
        assert [json.loads(line) for line in new_consumed.read_text().splitlines()] == [
            {"uid": future, "iter": 25}
        ]

    def test_failed_manifest_flip_leaves_old_generation_recoverable(self, tmp_path, monkeypatch):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(20)
        uid = bank.append(sample_group())
        old_manifest = json.loads((tmp_path / _MANIFEST).read_text())

        def fail_manifest_flip(manifest):
            raise OSError("simulated crash before manifest flip")

        monkeypatch.setattr(bank, "_write_manifest_atomic", fail_manifest_flip)
        with pytest.raises(OSError, match="simulated crash"):
            bank.checkpoint(20)

        assert json.loads((tmp_path / _MANIFEST).read_text()) == old_manifest
        restarted = RolloutBank(str(tmp_path))
        assert {group.uid for group in restarted.restore(20)} == {uid}
        assert [path.name for path in (tmp_path / _GENERATIONS).iterdir()] == [
            old_manifest["active_generation"]
        ]

    def test_async_compaction_finalize_runs_with_captured_iteration(self, tmp_path, monkeypatch):
        bank = RolloutBank(str(tmp_path))
        monkeypatch.setattr(rl_utils, "_ROLLOUT_BANK", bank)
        bank.set_collection(1)
        first = bank.append(sample_group())
        bank.mark_consumed(first, 1)

        first_save = AsyncRequest(None, (), [])
        second_save = AsyncRequest(None, (), [])
        _register_rollout_bank_compaction(first_save, 1)
        _register_rollout_bank_compaction(second_save, 2)

        bank.set_collection(2)
        second = bank.append(sample_group())
        bank.mark_consumed(second, 2)

        first_save.finalize_fns[0]()
        manifest = json.loads((tmp_path / _MANIFEST).read_text())
        assert manifest["trained_through"] == 1
        assert {group.uid for group in bank.restore(1)} == {second}

        second_save.finalize_fns[0]()
        manifest = json.loads((tmp_path / _MANIFEST).read_text())
        assert manifest["trained_through"] == 2
        assert manifest["segments"] == [_segment_name(2)]
        assert bank.restore(2) == []

    def test_marker_after_compaction_is_not_orphaned(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        old_uid = bank.append(sample_group())

        bank.checkpoint(2)
        bank.mark_consumed(old_uid, 4)

        assert bank.restore(2)[0].uid == old_uid
        assert bank.restore(4) == []

    def test_fresh_append_after_compaction_has_unique_uid(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(2)
        survivor_uid = bank.append(sample_group())

        bank.checkpoint(2)
        bank.set_collection(2)
        fresh_uid = bank.append(sample_group())

        assert fresh_uid != survivor_uid
        assert {group.uid for group in bank.restore(2)} == {survivor_uid, fresh_uid}

    def test_restore_reads_legacy_segment_marker(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(1)
        uid = bank.append(sample_group())
        bank.mark_consumed(uid, 1)
        generation = active_generation_path(tmp_path)
        os.replace(generation / _CONSUMED, generation / _segment_name(1) / _CONSUMED)

        assert RolloutBank(str(tmp_path)).restore(1) == []

    def test_compaction_prunes_and_flips_manifest(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(1)
        consumed = bank.append(sample_group())
        bank.mark_consumed(consumed, 1)
        bank.set_collection(2)
        survivor = bank.append(sample_group())
        old_generation = active_generation_path(tmp_path)

        bank.checkpoint(2)  # trained_through=2: prune consumed(<=2), keep survivor

        manifest = json.loads((tmp_path / _MANIFEST).read_text())
        assert manifest["trained_through"] == 2
        assert manifest["segments"] == [_segment_name(2)]
        assert manifest["compacted_at"] == 2
        # stale segment dir removed
        assert not old_generation.exists()

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

    def test_compaction_starts_a_new_cap_warning_epoch(self, tmp_path, monkeypatch):
        warnings = []
        monkeypatch.setattr(rollout_bank.logger, "warning", lambda *args: warnings.append(args))

        # Each sample group has 49 bytes of sidecar payload. Three groups cross
        # the 100-byte cap, while the sole survivor after compaction is below it.
        bank = RolloutBank(str(tmp_path), max_bytes=100)
        bank.set_collection(0)
        consumed = [bank.append(sample_group()) for _ in range(2)]
        bank.append(sample_group())
        assert len(warnings) == 1

        for uid in consumed:
            bank.mark_consumed(uid, 1)
        bank.checkpoint(1)
        assert len(warnings) == 1  # The staging rewrite is not a live cap crossing.

        bank.set_collection(1)
        bank.append(sample_group())
        assert len(warnings) == 1  # 98 live bytes remains below the cap.
        bank.append(sample_group())
        assert len(warnings) == 2  # A later crossing starts a new warning epoch.


class TestPipelineIntegration:
    """Write-through + restore through the real _RolloutPipeline."""

    def _collect(self, tmp_path, num_groups=4, stop_after=None):
        async def run():
            gen = MockGenerator()
            bank = RolloutBank(str(tmp_path))
            bank.set_collection(0)
            request_groups = []
            req = GroupedRolloutRequest(
                num_groups=num_groups,
                rollouts_per_group=2,
                inference_interface=MockInferenceInterface(),
                submission_granularity="B",
                consumption_granularity="B",
            )
            pipeline = RolloutPipeline(gen, req, parallel_generation_tasks=8, bank=bank)
            async with aclosing(pipeline.run()) as groups:
                async for group in groups:
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


def _env_group(env_id, problem_id="p"):
    """A minimal inline (text) RolloutGroup tagged with ``env_id``."""
    return RolloutGroup(
        rollouts=[Rollout(trajectory=["x"], reward=1.0, env_id=env_id, problem_id=problem_id)]
    )


def _weighted_agent(env_weights):
    return WeightedMultiTask(
        [
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": e}, weight=w)
            for e, w in env_weights
        ]
    )


class TestRestoreProducer:
    """Recovered groups become per-environment producers in WeightedMultiTask."""

    def test_set_restored_groups_buckets_by_env(self):
        agent = _weighted_agent([("a", 1.0), ("b", 1.0), ("c", 1.0)])
        groups = [_env_group("a"), _env_group("b"), _env_group("a")]
        assert agent.set_restored_groups(groups) == 3
        assert set(agent._restored_groups) == {"a", "b"}
        assert len(agent._restored_groups["a"]) == 2
        assert len(agent._restored_groups["b"]) == 1

    def test_set_restored_groups_rejects_env_config_drift(self):
        agent = _weighted_agent([("a", 1.0), ("b", 1.0)])
        with pytest.raises(ValueError, match="not in the current"):
            agent.set_restored_groups([_env_group("z")])

    def test_real_bank_recovery_installs_producer_queues(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        for i in range(6):
            bank.append(_env_group("a", problem_id=f"p{i}"))
        bank.close()

        restored = RolloutBank(str(tmp_path)).recover(trained_through=0)
        assert len(restored) == 6
        agent = _weighted_agent([("a", 1.0), ("b", 1.0), ("c", 1.0)])
        assert agent.set_restored_groups(restored) == 6
        assert len(agent._restored_groups["a"]) == 6

    @pytest.mark.asyncio
    async def test_restored_groups_replace_fresh_per_env(self):
        agent = _weighted_agent([("a", 1.0), ("b", 1.0)])
        restored = [_env_group("a", problem_id=f"a{i}") for i in range(2)]
        for index, group in enumerate(restored):
            group.uid = f"restored-{index}"
        assert agent.set_restored_groups(restored) == 2
        request = GroupedRolloutRequest(
            num_groups=4,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(),
        )

        pipeline = RolloutPipeline(agent, request, parallel_generation_tasks=1)
        async with aclosing(pipeline.run()) as group_stream:
            groups = [
                await asyncio.wait_for(anext(group_stream), timeout=10) for _ in range(4)
            ]

        assert sorted(group[0].env_id for group in groups) == ["a", "a", "b", "b"]
        assert sorted(group[0].problem_id for group in groups if group[0].env_id == "a") == [
            "a0",
            "a1",
        ]
        assert agent.agents[0].prepare_group_rollout_calls == 0
        assert agent.agents[1].prepare_group_rollout_calls == 2
        assert not agent._restored_groups["a"]

    @pytest.mark.asyncio
    async def test_streaming_restored_backlog_drains_before_fresh_per_env(self):
        agent = _weighted_agent([("a", 1.0), ("b", 1.0)])
        restored = [_env_group("a", problem_id=f"a{i}") for i in range(4)]
        for index, group in enumerate(restored):
            group.batch_id = 100 + index
            group.index_in_batch = 100 + index
        agent.set_restored_groups(restored)
        request = GroupedRolloutRequest(
            num_groups=4,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(),
            streaming=True,
        )
        pipeline = RolloutPipeline(agent, request, parallel_generation_tasks=1, initial_batch_id=20)

        async with aclosing(pipeline.run()) as groups:
            first_two_batches = [
                await asyncio.wait_for(anext(groups), timeout=10) for _ in range(8)
            ]
            assert [group.batch_id for group in first_two_batches] == [20] * 4 + [21] * 4
            assert [group.index_in_batch for group in first_two_batches] == [0, 1, 2, 3] * 2
            assert [group[0].env_id for group in first_two_batches].count("a") == 4
            assert [group[0].env_id for group in first_two_batches].count("b") == 4
            assert agent.agents[0].prepare_group_rollout_calls == 0
            assert agent.agents[1].prepare_group_rollout_calls == 4
            assert not agent._restored_groups["a"]

            next_batch = [await asyncio.wait_for(anext(groups), timeout=10) for _ in range(4)]
            assert [group.batch_id for group in next_batch] == [22] * 4
            assert [group.index_in_batch for group in next_batch] == [0, 1, 2, 3]
            assert [group[0].env_id for group in next_batch].count("a") == 2
            assert [group[0].env_id for group in next_batch].count("b") == 2
            assert agent.agents[0].prepare_group_rollout_calls == 2

    @pytest.mark.asyncio
    async def test_single_env_restore_routing_does_not_require_env_id(self):
        agent = _weighted_agent([("", 1.0)])
        assert agent.set_restored_groups([_env_group("", problem_id="cached")]) == 1
        request = GroupedRolloutRequest(
            num_groups=2,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(),
        )

        pipeline = RolloutPipeline(agent, request, parallel_generation_tasks=1)
        async with aclosing(pipeline.run()) as group_stream:
            groups = [
                await asyncio.wait_for(anext(group_stream), timeout=10) for _ in range(2)
            ]

        assert [group[0].env_id for group in groups] == ["", ""]
        assert agent.agents[0].prepare_group_rollout_calls == 1

    def test_multiple_envs_require_env_ids_for_restore_routing(self):
        agent = _weighted_agent([("", 1.0), ("b", 1.0)])

        with pytest.raises(ValueError, match="configuring multiple active agents"):
            agent.set_restored_groups([])
