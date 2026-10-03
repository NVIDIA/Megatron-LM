# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Parameterized coverage for durable rollout-bank persistence and replay."""

import asyncio
import dataclasses
import json
from collections import Counter
from contextlib import aclosing

import numpy as np
import pytest

from megatron.core.inference.inference_request import (
    FinishedRequestRecord,
    OffloadedRequestPayload,
    RequestPayloadStageResult,
)
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
    _REQUEST_RECORDS,
    FinishedRequestLedgerWriter,
    LedgerPayloadStager,
    RolloutBank,
    _segment_name,
    gather_view,
    load_finished_request_records,
    request_record_backup_path,
    request_record_paths,
)
from megatron.rl.types import Rollout as SharedRollout
from megatron.rl.types import RolloutGroup as SharedRolloutGroup
from megatron.rl.types import TokenRollout as SharedTokenRollout
from tests.unit_tests.rl.test_rollout_generation import MockGenerator, MockInferenceInterface


def _token_group(batch_id=0, *, empty=False):
    """A group plus the records that rebuild it: turn tokens == prompt + generation."""
    turns_per_member = [[]] if empty else [[([1], [2, 3]), ([1, 2, 3], [5])], [([7], [8])]]
    rollouts, records = [], {}
    for member, turns in enumerate(turns_per_member):
        cids = [f"chatcmpl-{batch_id}-{member}-{turn}" for turn in range(len(turns))]
        logprobs = [[-0.125 * (i + 1) for i in range(len(generated))] for _, generated in turns]
        for cid, turn, (prompt, generated) in zip(cids, range(len(turns)), turns):
            records[cid] = FinishedRequestRecord(
                policy_epoch=[(0, 3)],
                kv_cache_epoch=[(0, 3), (2, 4)],
                num_evictions=turn,
                payload=OffloadedRequestPayload(
                    prompt_token_ids=list(prompt),
                    generated_token_ids=list(generated),
                    generated_log_probs=logprobs[turn],
                    prompt_log_probs=None,
                    routing_indices=None,
                ),
            )
        rollouts.append(
            TokenRollout(  # as a restored (or joined) rollout carries the records' turns and stamps
                trajectory=[[*prompt, *generated] for prompt, generated in turns],
                reward=1.0,
                logprobs=logprobs,
                generation_mask=[[False] * len(p) + [True] * len(g) for p, g in turns],
                env_id="test",
                problem_id="p",
                completion_ids=cids,
                policy_epochs=[[(0, 3)]] * len(turns),
                kv_cache_epochs=[[(0, 3), (2, 4)]] * len(turns),
                num_evictions=list(range(len(turns))),
            )
        )
    return RolloutGroup(rollouts=rollouts, batch_id=batch_id), records


def _text_group():
    group = RolloutGroup(rollouts=[Rollout(trajectory=["hello world"], reward=0.5, env_id="text")])
    return group, {}


def _empty_group():
    return RolloutGroup(rollouts=[]), {}


def _empty_token_group():
    return _token_group(empty=True)


def _staged_token_group():
    """The token group as the Megatron interface delivers it: the turns live only in the records."""
    group, records = _token_group()
    for rollout in group.rollouts:
        rollout.trajectory = rollout.generation_mask = rollout.logprobs = None
        rollout.policy_epochs = rollout.kv_cache_epochs = rollout.num_evictions = None
        assert not rollout.is_placeholder
    return group, records


def _active_generation(bank_dir):
    manifest = json.loads((bank_dir / _MANIFEST).read_text())
    return bank_dir / _GENERATIONS / manifest["active_generation"]


def _segment_dir(bank_dir, iteration=3):
    return _active_generation(bank_dir) / _segment_name(iteration)


def _assert_group_matches(actual, expected):
    """Field-exact: fp32 backups round-trip the engine's fp32 log probs bit-exactly."""
    assert actual.batch_id == expected.batch_id
    assert [r.model_dump() for r in actual.rollouts] == [r.model_dump() for r in expected.rollouts]


@pytest.mark.parametrize(
    "group_factory",
    [
        pytest.param(_token_group, id="ragged-token-group"),
        pytest.param(_staged_token_group, id="staged-token-group"),
        pytest.param(_text_group, id="inline-text"),
        pytest.param(_empty_group, id="empty-group"),
        pytest.param(_empty_token_group, id="empty-token-trajectory"),
    ],
)
def test_rollout_bank_round_trip(tmp_path, group_factory):
    """All supported payload shapes survive a close/restart round trip. Token groups bank as thin
    manifests that restore rebuilds from the request records, whether the turns arrived inline or
    staged with the engine; mismatched completion ids are refused at append."""
    assert (SharedRollout, SharedRolloutGroup, SharedTokenRollout) == (
        Rollout,
        RolloutGroup,
        TokenRollout,
    )
    assert Rollout(trajectory=["prompt"], reward=None).reward is None

    expected, records = group_factory()
    bank = RolloutBank(str(tmp_path))
    bank.set_collection(3)
    uid = bank.append(expected)
    if group_factory is _token_group:
        unjoinable, _ = _token_group(batch_id=1)
        unjoinable.rollouts[0].completion_ids = []
        with pytest.raises(ValueError, match="completion ids"):
            bank.append(unjoinable)
    bank.close()

    manifest = json.loads((tmp_path / _MANIFEST).read_text())
    generation = _active_generation(tmp_path)
    record = json.loads((generation / _segment_name(3) / _LEDGER).read_text())
    assert manifest["format_version"] == record["format_version"] == _FORMAT_VERSION
    assert manifest["timeline"]
    assert (generation / _CONSUMED).exists()

    restored = RolloutBank(str(tmp_path)).restore(trained_through=0, request_records=records)
    assert [group.uid for group in restored] == [uid]
    if group_factory is _staged_token_group:
        expected, _ = _token_group()  # the records rebuild what the engine staged
    _assert_group_matches(restored[0], expected)

    if group_factory not in (_text_group, _empty_group):
        # Thin manifest: the reconstructable arrays never reach the bank.
        assert record["kind"] == "token"
        for member in record["group"]["rollouts"]:
            for field in ("trajectory", "logprobs", "generation_mask", "policy_epochs"):
                assert field not in member


@pytest.mark.parametrize(
    "damage, expected_survivors",
    [
        pytest.param("torn_ledger_line", 1, id="torn-final-ledger-record"),
        pytest.param("bad_checksum", 1, id="bad-checksum"),
        pytest.param("missing_records", 1, id="missing-request-records"),
        pytest.param("no_records", 0, id="no-request-records"),
    ],
)
def test_restore_drops_only_the_damaged_groups(tmp_path, damage, expected_survivors):
    """A torn final manifest line or a bad checksum drops that record alone; a token group whose
    request records did not survive is dropped loudly rather than restored without them."""
    group_a, records_a = _token_group(batch_id=0)
    group_b, _ = _token_group(batch_id=1)
    bank = RolloutBank(str(tmp_path))
    bank.set_collection(3)
    durable_uid = bank.append(group_a)
    bank.append(group_b)
    bank.close()

    ledger = _segment_dir(tmp_path) / _LEDGER
    records = {} if damage == "no_records" else records_a  # group_b's records never reached disk
    if damage == "torn_ledger_line":
        lines = ledger.read_bytes().splitlines(keepends=True)
        ledger.write_bytes(lines[0] + lines[1][: len(lines[1]) // 2])
    elif damage == "bad_checksum":
        manifests = [json.loads(line) for line in ledger.read_text().splitlines()]
        manifests[-1]["checksum"] = "0" * 32
        ledger.write_text("".join(json.dumps(m) + "\n" for m in manifests))

    restarted = RolloutBank(str(tmp_path))
    restarted.set_collection(3)  # a torn tail is truncated before appends resume
    if damage == "torn_ledger_line":
        assert ledger.read_bytes() == lines[0]
    restored = restarted.restore(trained_through=0, request_records=records)
    assert [group.uid for group in restored] == [durable_uid][:expected_survivors]


def test_record_store_round_trip(tmp_path, monkeypatch):
    """The stager pins the payload on the engine's record on a worker thread, makes it durable and
    only then indexes it, so drain() never hands out a record without its payload. Records
    round-trip through rotate and load with exact fields, a torn tail drops only itself, an absent
    file contributes nothing, the gather view keeps just what the joins read, and a failed write
    is raised at the next rotate() or close()."""
    routing = np.arange(12, dtype=np.int64).reshape(4, 3, 1)
    record = FinishedRequestRecord(
        policy_epoch=[(0, 3)],
        kv_cache_epoch=[(0, 3), (5, 4)],
        num_evictions=2,
        num_tokens_to_generate=8,
        finish_reason="stop",
    )
    payload = OffloadedRequestPayload(
        prompt_token_ids=[1, 2, 3],
        generated_token_ids=[10, 11],
        generated_log_probs=[-0.125, -1.25],
        prompt_log_probs=[-1.0, -2.0, -3.0],
        routing_indices=routing,
    )
    plain = FinishedRequestRecord(policy_epoch=None, kv_cache_epoch=None, num_evictions=0)
    first, second = tmp_path / "a.bin", tmp_path / "b.bin"
    stager = LedgerPayloadStager(FinishedRequestLedgerWriter(str(first)))
    assert stager.drain() == {}
    staged = asyncio.run(
        asyncio.to_thread(stager.stage, "chatcmpl-0", payload, finished_metadata=record)
    )
    assert staged == RequestPayloadStageResult() and record.payload is payload
    drained = stager.drain()
    assert set(drained) == {"chatcmpl-0"} and drained["chatcmpl-0"] is record
    assert stager.drain() == {}
    with pytest.raises(AssertionError, match="duplicate uid"):
        stager.stage("chatcmpl-1", payload, finished_metadata=record)
        stager.stage("chatcmpl-1", payload, finished_metadata=record)
    stager.writer.rotate(str(second))
    stager.writer.append("chatcmpl-plain", plain)
    stager.writer.append("chatcmpl-2", record)
    stager.writer.close()

    loaded = load_finished_request_records([str(first), str(second)])
    assert set(loaded) == {"chatcmpl-0", "chatcmpl-1", "chatcmpl-plain", "chatcmpl-2"}
    durable = loaded["chatcmpl-0"]
    # Field-exact: stamps back as tuples, fp32 log probs bit-exactly, routing with its dtype.
    assert np.array_equal(durable.payload.routing_indices, routing)
    assert durable.payload.routing_indices.dtype == routing.dtype
    without_routing = dataclasses.replace(payload, routing_indices=None)
    assert dataclasses.replace(durable.payload, routing_indices=None) == without_routing
    assert dataclasses.replace(durable, payload=None) == dataclasses.replace(record, payload=None)
    assert loaded["chatcmpl-plain"] == plain

    gathered = gather_view(record)
    assert dataclasses.replace(gathered, payload=None) == dataclasses.replace(record, payload=None)
    assert gathered.payload == dataclasses.replace(without_routing, prompt_log_probs=None)
    assert payload.routing_indices is routing  # the stager's own record keeps everything

    second.write_bytes(second.read_bytes()[:-3])  # tear the final record
    loaded = load_finished_request_records([str(first), str(second), str(tmp_path / "absent.bin")])
    assert set(loaded) == {"chatcmpl-0", "chatcmpl-1", "chatcmpl-plain"}

    def broken(uid, record):
        raise OSError("disk full")

    monkeypatch.setattr(rollout_bank, "pack_finished_request_record", broken)
    writer = FinishedRequestLedgerWriter(str(tmp_path / "c.bin"))
    writer.append("chatcmpl-3", plain)
    with pytest.raises(RuntimeError, match="writer failed") as info:
        writer.rotate(str(tmp_path / "d.bin"))
    assert isinstance(info.value.__cause__, OSError)
    with pytest.raises(RuntimeError, match="writer failed"):
        writer.close()


@pytest.mark.parametrize(
    "consumed_offset, should_restore",
    [
        pytest.param(-1, False, id="consumed-before-checkpoint"),
        pytest.param(0, False, id="consumed-at-checkpoint"),
        pytest.param(1, True, id="consumed-after-checkpoint"),
    ],
)
def test_checkpoint_compacts_at_consumption_boundary(tmp_path, consumed_offset, should_restore):
    """Checkpoint compaction prunes trained groups and preserves future markers. It folds only
    the backup windows before it (the window it closes may still be open), keeps every record
    still owed to a group, and releases a record once its group is reclaimed or rank 0 retired
    it, even when the retirement precedes the record's arrival."""
    checkpoint_iteration = 10
    group_a, records_a = _token_group(batch_id=0)
    group_b, records_b = _token_group(batch_id=1)
    all_records = {**records_a, **records_b}

    def record():
        return FinishedRequestRecord(
            policy_epoch=[(0, 3)],
            kv_cache_epoch=[(0, 3)],
            num_evictions=0,
            payload=OffloadedRequestPayload(
                prompt_token_ids=[1],
                generated_token_ids=[2],
                generated_log_probs=[-0.5],
                prompt_log_probs=None,
                routing_indices=None,
            ),
        )

    def backup(window, records):
        """Back records up the way an engine does, into a closed (rotated) window file."""
        path = request_record_backup_path(str(tmp_path), rank=0, iteration=window, tag="deadbeef")
        writer = FinishedRequestLedgerWriter(path)
        for cid, rec in records.items():
            writer.append(cid, rec)
        writer.close()
        return path

    def kept():
        return set(load_finished_request_records(request_record_paths(str(tmp_path))))

    # Window 5 is closed: both groups' records, a turn of an in-flight group, a retired request.
    backup(5, {**all_records, "chatcmpl-inflight": record(), "chatcmpl-retired": record()})
    # The window the checkpoint closes is the engines' current file (an evaluation created it).
    current = backup(checkpoint_iteration, {"chatcmpl-eval": record()})

    bank = RolloutBank(str(tmp_path))
    bank.set_collection(5)
    consumed = bank.append(group_a)
    never_consumed = bank.append(group_b)
    bank.mark_consumed_many([consumed], checkpoint_iteration + consumed_offset)
    bank.retire_records(["chatcmpl-retired", "chatcmpl-later"])  # "later" has no record yet
    old_generation = _active_generation(tmp_path)

    before_checkpoint = {
        group.uid for group in bank.restore(checkpoint_iteration, request_records=all_records)
    }
    assert (consumed in before_checkpoint) is should_restore
    assert never_consumed in before_checkpoint

    bank.checkpoint(checkpoint_iteration)
    manifest = json.loads((tmp_path / _MANIFEST).read_text())
    assert manifest["trained_through"] == checkpoint_iteration
    assert manifest["segments"] == [_segment_name(checkpoint_iteration)]
    assert not old_generation.exists()
    # Window 5 was folded and pruned, the current window left alone. A reclaimed group releases
    # its records and a retired id is dropped; the in-flight turn is kept, nobody owns it yet.
    fold = _active_generation(tmp_path) / _segment_name(checkpoint_iteration) / _REQUEST_RECORDS
    assert request_record_paths(str(tmp_path)) == [current, str(fold)]
    owed = set(all_records) if should_restore else set(records_b)
    assert set(load_finished_request_records([str(fold)])) == owed | {"chatcmpl-inflight"}
    assert set(load_finished_request_records([current])) == {"chatcmpl-eval"}

    restored = RolloutBank(str(tmp_path)).restore(
        checkpoint_iteration,
        request_records=load_finished_request_records(request_record_paths(str(tmp_path))),
    )
    expected_uids = {consumed, never_consumed} if should_restore else {never_consumed}
    assert {group.uid for group in restored} == expected_uids
    for group in restored:
        _assert_group_matches(group, _token_group(batch_id=0 if group.uid == consumed else 1)[0])
    markers = [
        json.loads(line)
        for line in (_active_generation(tmp_path) / _CONSUMED).read_text().splitlines()
    ]
    assert markers == (
        [{"uid": consumed, "iter": checkpoint_iteration + 1}] if should_restore else []
    )

    # Two windows on: the current window has closed, the retired id's record arrived late, and
    # rank 0 retired the in-flight turn. Only the never-consumed group's records and the still
    # unretired evaluation record remain.
    backup(checkpoint_iteration + 1, {"chatcmpl-later": record()})
    bank.retire_records(["chatcmpl-inflight"])
    bank.checkpoint(checkpoint_iteration + 2)
    assert kept() == set(records_b) | {"chatcmpl-eval"}
    restored = bank.restore(
        checkpoint_iteration + 2,
        request_records=load_finished_request_records(request_record_paths(str(tmp_path))),
    )
    assert {group.uid for group in restored} == {never_consumed}


def _env_group(env_id, problem_id):
    return RolloutGroup(
        rollouts=[Rollout(trajectory=["cached"], reward=1.0, env_id=env_id, problem_id=problem_id)]
    )


def _weighted_agent(env_weights):
    return WeightedMultiTask(
        [
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": env_id}, weight=weight)
            for env_id, weight in env_weights
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "env_weights, restored_env, restored_count, request_groups, take_count, "
    "expected_envs, expected_fresh_calls, write_through",
    [
        pytest.param(
            [("a", 1.0), ("b", 1.0)],
            "a",
            2,
            4,
            4,
            {"a": 2, "b": 2},
            [0, 2],
            False,
            id="restored-groups-replace-fresh",
        ),
        pytest.param(
            [("a", 1.0), ("b", 1.0)],
            "a",
            4,
            4,
            12,
            {"a": 6, "b": 6},
            [2, 6],
            False,
            id="streaming-backlog-drains-before-fresh",
        ),
        pytest.param(
            [("", 1.0)], "", 1, 2, 2, {"": 2}, [1], False, id="single-environment-without-env-id"
        ),
        pytest.param(
            [("a", 1.0)], "a", 0, 4, 4, {"a": 4}, [4], True, id="fresh-groups-write-through"
        ),
    ],
)
async def test_pipeline_uses_restored_groups_before_fresh_generation(
    tmp_path,
    env_weights,
    restored_env,
    restored_count,
    request_groups,
    take_count,
    expected_envs,
    expected_fresh_calls,
    write_through,
):
    """Restored groups retain weighted routing; fresh groups remain durable."""
    agent = _weighted_agent(env_weights)
    restored = [
        _env_group(restored_env, problem_id=f"cached-{index}") for index in range(restored_count)
    ]
    assert agent.set_restored_groups(restored) == restored_count

    if len(env_weights) > 1:
        with pytest.raises(ValueError, match="not in the current"):
            agent.set_restored_groups([_env_group("unknown", "drift")])
        assert agent.set_restored_groups(restored) == restored_count

    bank = RolloutBank(str(tmp_path)) if write_through else None
    if bank is not None:
        bank.set_collection(0)
    request = GroupedRolloutRequest(
        num_groups=request_groups,
        rollouts_per_group=1,
        inference_interface=MockInferenceInterface(),
    )
    pipeline = RolloutPipeline(
        agent, request, parallel_generation_tasks=1, initial_batch_id=20, bank=bank
    )

    async with aclosing(pipeline.run()) as groups:
        produced = [await asyncio.wait_for(anext(groups), timeout=10) for _ in range(take_count)]

    assert len(produced) == take_count
    assert Counter(group[0].env_id for group in produced) == expected_envs
    assert [generator.prepare_group_rollout_calls for generator in agent.agents] == (
        expected_fresh_calls
    )
    cached_problem_ids = {f"cached-{index}" for index in range(restored_count)}
    assert cached_problem_ids <= {group[0].problem_id for group in produced}
    assert not agent._restored_groups.get(restored_env)

    assert [group.batch_id for group in produced] == [
        20 + index // request_groups for index in range(take_count)
    ]
    assert [group.index_in_batch for group in produced] == [
        index % request_groups for index in range(take_count)
    ]

    if bank is not None:
        bank.close()
        persisted = RolloutBank(str(tmp_path)).restore(trained_through=0)
        assert len(persisted) == take_count
        assert {group.uid for group in persisted} == {group.uid for group in produced}


@pytest.mark.asyncio
async def test_restored_group_metrics_are_reported_per_step_and_env(monkeypatch):
    agent = _weighted_agent([("a", 1.0), ("b", 1.0)])
    agent.set_restored_groups(
        [
            *[_env_group("a", problem_id=f"a{index}") for index in range(3)],
            _env_group("b", problem_id="b0"),
        ]
    )
    request = GroupedRolloutRequest(
        num_groups=4,
        rollouts_per_group=1,
        inference_interface=MockInferenceInterface(num_slow_calls=100),
        streaming=True,
        submission_granularity="B",
        consumption_granularity="B",
    )
    pipeline = RolloutPipeline(agent, request, parallel_generation_tasks=2)
    monkeypatch.setattr(rl_utils, "_ROLLOUT_PIPELINE", pipeline)

    empty_metrics = rl_utils._collect_rollout_pipeline_metrics()
    assert empty_metrics["a_restored_groups_percentage"] == 0.0
    assert empty_metrics["a_fresh_groups_percentage"] == 0.0

    async with aclosing(pipeline.run()) as groups:
        [await anext(groups) for _ in range(4)]
        first_step_metrics = rl_utils._collect_rollout_pipeline_metrics()

        assert first_step_metrics["rollout_pipeline_restored_count"] == 3
        assert first_step_metrics["rollout_pipeline_yielded_count"] == 4
        assert first_step_metrics["a_restored_groups"] == 2
        assert first_step_metrics["b_restored_groups"] == 1
        assert first_step_metrics["a_restored_groups_percentage"] == 100.0
        assert first_step_metrics["a_fresh_groups_percentage"] == 0.0
        assert first_step_metrics["b_restored_groups_percentage"] == 50.0
        assert first_step_metrics["b_fresh_groups_percentage"] == 50.0

        [await anext(groups) for _ in range(4)]
        second_step_metrics = rl_utils._collect_rollout_pipeline_metrics()

        assert second_step_metrics["rollout_pipeline_restored_count"] == 1
        assert second_step_metrics["rollout_pipeline_yielded_count"] == 4
        assert second_step_metrics["a_restored_groups"] == 1
        assert second_step_metrics["b_restored_groups"] == 0
        assert second_step_metrics["a_restored_groups_percentage"] == 50.0
        assert second_step_metrics["a_fresh_groups_percentage"] == 50.0
        assert second_step_metrics["b_restored_groups_percentage"] == 0.0
        assert second_step_metrics["b_fresh_groups_percentage"] == 100.0


def test_multiple_envs_require_env_ids_for_restore_routing():
    agent = _weighted_agent([("", 1.0), ("b", 1.0)])

    with pytest.raises(ValueError, match="configuring multiple active agents"):
        agent.set_restored_groups([])
