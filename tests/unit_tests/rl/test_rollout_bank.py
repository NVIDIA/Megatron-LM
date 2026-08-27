# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Durable rollout-bank persistence, recovery, and compaction.

Format v3 stores one ledger record per rollout, so a group exists on disk only
through its members. Completeness is inferred from the member count rather than a
seal record, and a group that lost members to a kill reads back as *incomplete*,
carrying the problem state needed to regenerate the rest.
"""

import json

import numpy as np
import pytest

from megatron.rl.agent.api import Rollout, RolloutGroup, TokenRollout
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

PROBLEM = {"prompt": "Natalia sold clips to 48 friends", "golden": {"answer": "72"}}
ITER = 3


def _token(reward=1.0, tokens=((1, 2, 3), (4, 5))):
    trajectory = [list(turn) for turn in tokens]
    return TokenRollout(
        trajectory=trajectory,
        reward=reward,
        logprobs=[[-0.1 * (i + 1) for i in range(len(turn))] for turn in trajectory],
        generation_mask=[[True] * len(turn) for turn in trajectory],
        env_id="gsm8k",
        problem_id="p0",
    )


def _text(reward=0.5):
    return Rollout(trajectory=["hello world"], reward=reward, env_id="text")


def _bank(path, group_size=2, **kwargs):
    return RolloutBank(str(path), rollouts_per_group=group_size, **kwargs)


def _generation(bank_dir):
    manifest = json.loads((bank_dir / _MANIFEST).read_text())
    return bank_dir / _GENERATIONS / manifest["active_generation"]


def _segment(bank_dir, iteration=ITER):
    return _generation(bank_dir) / _segment_name(iteration)


def _records(bank_dir, iteration=ITER):
    ledger = _segment(bank_dir, iteration) / _LEDGER
    return [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]


def _write(bank, members, uid=None, problem_state=None, indices=None):
    """Persist one group's members, optionally at explicit slots."""
    uid = uid or bank.reserve_group_uid()
    if problem_state is not None:
        bank.append_problem(uid, problem_state)
    for slot, member in zip(indices or range(len(members)), members, strict=True):
        bank.append_rollout(uid, slot, member)
    return uid


def test_shared_types_are_the_same_objects_as_the_agent_api_re_exports():
    """api.py re-exports the megatron.rl.types models; drift would split the schema."""
    assert (SharedRollout, SharedRolloutGroup, SharedTokenRollout) == (
        Rollout,
        RolloutGroup,
        TokenRollout,
    )
    assert Rollout(trajectory=["prompt"], reward=None).reward is None


@pytest.mark.parametrize(
    "members, group_size",
    [
        pytest.param([_token(), _token(reward=0.0, tokens=((7, 8),))], 2, id="ragged-token"),
        pytest.param([_text()], 1, id="inline-text"),
        pytest.param([_token(tokens=())], 1, id="empty-trajectory"),
    ],
)
def test_payload_shapes_survive_a_restart(tmp_path, members, group_size):
    """Every supported member shape decodes back to what was written."""
    bank = _bank(tmp_path, group_size)
    bank.set_collection(ITER)
    uid = _write(bank, members)
    bank.close()

    records = _records(tmp_path)
    assert len(records) == group_size, "one record per member; no group record"
    assert {record["format_version"] for record in records} == {_FORMAT_VERSION}
    assert (_generation(tmp_path) / _CONSUMED).exists()

    restored = _bank(tmp_path, group_size).restore(trained_through=0)
    assert [group.uid for group in restored] == [uid]
    for actual, expected in zip(restored[0].rollouts, members, strict=True):
        assert type(actual) is type(expected)
        assert actual.trajectory == expected.trajectory
        assert actual.reward == expected.reward
        if isinstance(expected, TokenRollout):
            assert actual.generation_mask == expected.generation_mask
            for got, want in zip(actual.logprobs, expected.logprobs, strict=True):
                assert np.allclose(got, want, atol=1e-3)


@pytest.mark.parametrize(
    "members, indices, problem_state, drop_zero_variance, expect",
    [
        pytest.param([_token(0.0), _token(1.0)], None, PROBLEM, False, "complete", id="complete"),
        pytest.param([_token()], [0], PROBLEM, False, "incomplete", id="incomplete-restores"),
        pytest.param([_token()], [0], None, False, "dropped", id="incomplete-without-problem"),
        pytest.param([_token(1.0), _token(1.0)], None, None, True, "dropped", id="zero-variance"),
        pytest.param(
            [_token(1.0), _token(1.0)], None, None, False, "complete", id="zero-variance-allowed"
        ),
    ],
)
def test_restore_fold(tmp_path, members, indices, problem_state, drop_zero_variance, expect):
    """Completeness comes from the member count; there is no seal and no tombstone."""
    bank = _bank(tmp_path, 2, drop_zero_variance=drop_zero_variance)
    try:
        bank.set_collection(ITER)
        uid = _write(bank, members, problem_state=problem_state, indices=indices)
        restored = bank.restore(0)
    finally:
        bank.close()

    if expect == "dropped":
        assert restored == []
        return
    assert [group.uid for group in restored] == [uid]
    group = restored[0]
    assert group.is_complete(2) is (expect == "complete")
    if expect == "incomplete":
        assert group.problem_state == PROBLEM
        assert group.missing_indices(2) == [1]


def test_members_restore_in_slot_order_with_gaps_preserved(tmp_path):
    """Records interleave on disk, so restore must order by slot, not arrival."""
    bank = _bank(tmp_path, 4)
    try:
        bank.set_collection(ITER)
        uid = bank.reserve_group_uid()
        bank.append_problem(uid, PROBLEM)
        bank.append_rollout(uid, 2, _token(reward=1.0))
        bank.append_rollout(uid, 0, _token(reward=0.0))
        restored = bank.restore(0)
    finally:
        bank.close()

    assert restored[0].member_indices == [0, 2]
    assert [rollout.reward for rollout in restored[0].rollouts] == [0.0, 1.0]
    assert restored[0].missing_indices(4) == [1, 3]


def test_ledger_holds_only_problem_and_rollout_records(tmp_path):
    """Format v3 has exactly two record kinds."""
    bank = _bank(tmp_path, 2)
    try:
        bank.set_collection(ITER)
        uid = _write(bank, [_token(), _token()], problem_state=PROBLEM)
    finally:
        bank.close()

    records = _records(tmp_path)
    assert [record["kind"] for record in records] == ["problem", "rollout", "rollout"]
    assert records[1]["uid"] == f"{uid}#0"


def _tear_ledger(bank_dir):
    ledger = _segment(bank_dir) / _LEDGER
    lines = ledger.read_bytes().splitlines(keepends=True)
    ledger.write_bytes(b"".join(lines[:3]) + lines[3][: len(lines[3]) // 2])


def _corrupt_checksum(bank_dir):
    ledger = _segment(bank_dir) / _LEDGER
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    records[-1]["checksum"] = "0" * 32
    ledger.write_text("".join(json.dumps(record) + "\n" for record in records))


def _truncate_sidecar(bank_dir):
    tokens = _segment(bank_dir) / _TOKENS_BIN
    tokens.write_bytes(tokens.read_bytes()[:-1])


@pytest.mark.parametrize(
    "damage",
    [
        pytest.param(_tear_ledger, id="torn-final-record"),
        pytest.param(_corrupt_checksum, id="bad-checksum"),
        pytest.param(_truncate_sidecar, id="truncated-sidecar"),
    ],
)
def test_damage_drops_only_the_affected_group(tmp_path, damage):
    """A damaged record leaves its group short, which is then unrestorable.

    Losing a member can only make a group look *less* complete, never more, so
    corruption degrades into the incomplete-group case rather than corrupting.
    """
    bank = _bank(tmp_path, 2)
    bank.set_collection(ITER)
    durable = _write(bank, [_token(), _token()])
    _write(bank, [_token(), _token()])
    bank.close()

    damage(tmp_path)

    restarted = _bank(tmp_path, 2)
    restarted.set_collection(ITER)
    assert [group.uid for group in restarted.restore(trained_through=0)] == [durable]


def test_empty_group_persists_nothing(tmp_path):
    """A group exists on disk only through its members, so a memberless one cannot."""
    bank = _bank(tmp_path, 1)
    try:
        bank.set_collection(ITER)
        bank.append(RolloutGroup(rollouts=[]))
        assert (_segment(tmp_path) / _LEDGER).exists() is False
        assert bank.restore(trained_through=0) == []
    finally:
        bank.close()


def test_append_group_writes_each_member_and_keeps_its_uid(tmp_path):
    """The whole-group convenience path is the per-member path underneath."""
    bank = _bank(tmp_path, 2)
    try:
        bank.set_collection(ITER)
        group = RolloutGroup(rollouts=[_token(0.0), _token(1.0)])
        uid = bank.append(group, problem_state=PROBLEM)
        restored = bank.restore(0)
    finally:
        bank.close()

    assert [group.uid for group in restored] == [uid]
    assert restored[0].member_indices == [0, 1]
    assert restored[0].problem_state == PROBLEM


def test_uids_do_not_collide_across_bank_instances(tmp_path):
    """Uids carry a per-run nonce, so a restart cannot reuse one."""
    first, second = _bank(tmp_path / "a"), _bank(tmp_path / "b")
    try:
        uids = {first.reserve_group_uid() for _ in range(5)}
        uids |= {second.reserve_group_uid() for _ in range(5)}
        assert len(uids) == 10
    finally:
        first.close()
        second.close()


def test_rollouts_per_group_change_across_resume_is_rejected(tmp_path):
    """Completeness is inferred from the count, so the count must not shift."""
    bank = _bank(tmp_path, 4)
    try:
        bank.set_collection(ITER)
        _write(bank, [_token()], indices=[0])
    finally:
        bank.close()

    with pytest.raises(ValueError, match="rollouts_per_group"):
        _bank(tmp_path, 8)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        pytest.param({}, [0, 1], id="defaults-to-positional"),
        pytest.param({"member_indices": [0, 2]}, [0, 2], id="explicit-slots-preserved"),
    ],
)
def test_member_indices_are_always_populated(kwargs, expected):
    """One source of truth: readers never fall back to positional slots."""
    assert RolloutGroup(rollouts=[_token(), _token()], **kwargs).member_indices == expected


def test_member_indices_must_match_the_member_count():
    """A slot list that disagrees with the members is a bug, not a fallback."""
    with pytest.raises(ValueError, match="member_indices"):
        RolloutGroup(rollouts=[_token()], member_indices=[0, 1])


@pytest.mark.parametrize(
    "consumed_offset, should_restore",
    [
        pytest.param(-1, False, id="consumed-before-checkpoint"),
        pytest.param(0, False, id="consumed-at-checkpoint"),
        pytest.param(1, True, id="consumed-after-checkpoint"),
    ],
)
def test_checkpoint_compacts_at_the_consumption_boundary(tmp_path, consumed_offset, should_restore):
    """Compaction prunes trained groups and carries forward future markers."""
    checkpoint = 10
    bank = _bank(tmp_path, 2)
    bank.set_collection(5)
    consumed = _write(bank, [_token(0.0), _token(1.0)])
    kept = _write(bank, [_token(0.0), _token(1.0)])
    bank.mark_consumed(consumed, checkpoint + consumed_offset)
    old_generation = _generation(tmp_path)

    before = {group.uid for group in bank.restore(checkpoint)}
    assert (consumed in before) is should_restore
    assert kept in before

    bank.checkpoint(checkpoint)
    manifest = json.loads((tmp_path / _MANIFEST).read_text())
    assert manifest["trained_through"] == checkpoint
    assert manifest["segments"] == [_segment_name(checkpoint)]
    assert not old_generation.exists()

    restarted = _bank(tmp_path, 2)
    restored = {group.uid for group in restarted.restore(checkpoint)}
    assert restored == ({consumed, kept} if should_restore else {kept})

    markers = [
        json.loads(line) for line in (_generation(tmp_path) / _CONSUMED).read_text().splitlines()
    ]
    assert markers == ([{"uid": consumed, "iter": checkpoint + 1}] if should_restore else [])
