# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Parameterized coverage for durable rollout-bank persistence and replay."""

import json
from collections import Counter
from contextlib import aclosing

import numpy as np
import pytest

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
from tests.unit_tests.rl.test_rollout_generation import MockGenerator, MockInferenceInterface


def _token_group(batch_id=0, *, empty=False):
    members = (
        [([], [], [])]
        if empty
        else [
            (
                [[1, 2, 3], [4, 5]],
                [[-0.1, -0.2, -0.3], [-0.4, -0.5]],
                [[False, True, True], [True, True]],
            ),
            ([[7, 8]], [[-1.5, -2.5]], [[True, True]]),
        ]
    )
    return RolloutGroup(
        rollouts=[
            TokenRollout(
                trajectory=tokens,
                reward=1.0,
                logprobs=logprobs,
                generation_mask=mask,
                env_id="test",
                problem_id="p",
                completion_ids=[f"completion-{index}" for index in range(len(tokens))],
            )
            for tokens, logprobs, mask in members
        ],
        batch_id=batch_id,
    )


def _text_group():
    return RolloutGroup(rollouts=[Rollout(trajectory=["hello world"], reward=0.5, env_id="text")])


def _empty_group():
    return RolloutGroup(rollouts=[])


def _empty_token_group():
    return _token_group(empty=True)


def _active_generation(bank_dir):
    manifest = json.loads((bank_dir / _MANIFEST).read_text())
    return bank_dir / _GENERATIONS / manifest["active_generation"]


def _assert_group_matches(actual, expected):
    assert actual.batch_id == expected.batch_id
    assert len(actual.rollouts) == len(expected.rollouts)
    for actual_rollout, expected_rollout in zip(actual.rollouts, expected.rollouts, strict=True):
        assert type(actual_rollout) is type(expected_rollout)
        assert actual_rollout.trajectory == expected_rollout.trajectory
        assert actual_rollout.reward == expected_rollout.reward
        assert actual_rollout.env_id == expected_rollout.env_id
        if isinstance(expected_rollout, TokenRollout):
            assert actual_rollout.generation_mask == expected_rollout.generation_mask
            assert len(actual_rollout.logprobs) == len(expected_rollout.logprobs)
            for actual_turn, expected_turn in zip(
                actual_rollout.logprobs, expected_rollout.logprobs, strict=True
            ):
                assert np.allclose(actual_turn, expected_turn, atol=1e-3)


@pytest.mark.parametrize(
    "group_factory",
    [
        pytest.param(_token_group, id="ragged-token-sidecars"),
        pytest.param(_text_group, id="inline-text"),
        pytest.param(_empty_group, id="empty-group"),
        pytest.param(_empty_token_group, id="empty-token-trajectory"),
    ],
)
def test_rollout_bank_round_trip(tmp_path, group_factory):
    """All supported payload shapes survive a close/restart round trip."""
    assert (SharedRollout, SharedRolloutGroup, SharedTokenRollout) == (
        Rollout,
        RolloutGroup,
        TokenRollout,
    )
    assert Rollout(trajectory=["prompt"], reward=None).reward is None

    expected = group_factory()
    bank = RolloutBank(str(tmp_path))
    bank.set_collection(3)
    uid = bank.append(expected)
    bank.close()

    manifest = json.loads((tmp_path / _MANIFEST).read_text())
    generation = _active_generation(tmp_path)
    record = json.loads((generation / _segment_name(3) / _LEDGER).read_text())
    assert manifest["format_version"] == record["format_version"] == _FORMAT_VERSION
    assert manifest["timeline"]
    assert (generation / _CONSUMED).exists()

    restored = RolloutBank(str(tmp_path)).restore(trained_through=0)
    assert [group.uid for group in restored] == [uid]
    _assert_group_matches(restored[0], expected)

    if group_factory is _empty_token_group:
        assert record["kind"] == "token"
        assert record["tok"]["bytes"] == 0
        assert not (generation / _segment_name(3) / _TOKENS_BIN).exists()


@pytest.mark.parametrize(
    "consumed_offset, should_restore",
    [
        pytest.param(-1, False, id="consumed-before-checkpoint"),
        pytest.param(0, False, id="consumed-at-checkpoint"),
        pytest.param(1, True, id="consumed-after-checkpoint"),
    ],
)
def test_checkpoint_compacts_at_consumption_boundary(tmp_path, consumed_offset, should_restore):
    """Checkpoint compaction prunes trained groups and preserves future markers."""
    checkpoint_iteration = 10
    bank = RolloutBank(str(tmp_path))
    bank.set_collection(5)
    consumed = bank.append(_token_group())
    never_consumed = bank.append(_token_group())
    bank.mark_consumed(consumed, checkpoint_iteration + consumed_offset)
    old_generation = _active_generation(tmp_path)

    before_checkpoint = {group.uid for group in bank.restore(checkpoint_iteration)}
    assert (consumed in before_checkpoint) is should_restore
    assert never_consumed in before_checkpoint

    bank.checkpoint(checkpoint_iteration)
    manifest = json.loads((tmp_path / _MANIFEST).read_text())
    assert manifest["trained_through"] == checkpoint_iteration
    assert manifest["segments"] == [_segment_name(checkpoint_iteration)]
    assert not old_generation.exists()

    restarted = RolloutBank(str(tmp_path))
    restored = restarted.restore(checkpoint_iteration)
    restored_uids = {group.uid for group in restored}
    assert restored_uids == ({consumed, never_consumed} if should_restore else {never_consumed})
    for group in restored:
        _assert_group_matches(group, _token_group())

    markers = [
        json.loads(line)
        for line in (_active_generation(tmp_path) / _CONSUMED).read_text().splitlines()
    ]
    expected_markers = (
        [{"uid": consumed, "iter": checkpoint_iteration + 1}] if should_restore else []
    )
    assert markers == expected_markers

    if should_restore:
        restarted.checkpoint(checkpoint_iteration + 1)
        assert {group.uid for group in restarted.restore(checkpoint_iteration + 1)} == {
            never_consumed
        }


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


def test_multiple_envs_require_env_ids_for_restore_routing():
    agent = _weighted_agent([("", 1.0), ("b", 1.0)])

    with pytest.raises(ValueError, match="configuring multiple active agents"):
        agent.set_restored_groups([])
