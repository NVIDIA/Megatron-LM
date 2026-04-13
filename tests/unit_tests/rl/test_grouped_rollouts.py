# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.rl import rl_utils
from megatron.rl.agent.api import (
    GroupedRolloutGenerator,
    GroupedRolloutRequest,
    Rollout,
    RolloutGenerator,
    RolloutGroup,
    RolloutStream,
)
from megatron.rl.agent.weighted_multi_task import AgentConfig, WeightedMultiTask
from megatron.rl.inference import ReturnsRaw


def _make_group(i, rollouts_per_group=1):
    return RolloutGroup(
        rollouts=[
            Rollout(
                trajectory=[f"t{i}"],
                reward=float(i),
                policy_epoch=[[(0, 0)]],
                kv_cache_epoch=[[(0, 0)]],
                num_evictions=[0],
            )
            for _ in range(rollouts_per_group)
        ],
        batch_id=i,
        index_in_batch=0,
    )


class MockGenerator(RolloutGenerator, GroupedRolloutGenerator):
    """Mock generator with configurable per-call delays."""

    def __init__(self, env_id="test", num_slow_calls=0, **kwargs):
        super().__init__(**kwargs)
        self.env_id = env_id
        self.num_slow_calls = num_slow_calls
        self._call_count = 0

    async def rollout(self, request):
        raise NotImplementedError

    async def group_rollout(self, request):
        idx = self._call_count
        self._call_count += 1
        if idx < self.num_slow_calls:
            await asyncio.sleep(0.03)
        return [
            Rollout(
                trajectory=[f"t{idx}"],
                reward=float(idx),
                env_id=self.env_id,
                policy_epoch=[[(0, 0)]],
                kv_cache_epoch=[[(0, 0)]],
                num_evictions=[0],
            )
            for _ in range(request.rollouts_per_group)
        ]


class TestGroupedRollouts:

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "num_slow_calls, streaming, num_groups, expected_count, expected_batch_ids",
        [
            pytest.param(0, False, 1, 8, None, id="non_batched"),
            pytest.param(4, False, 2, 8, [0, 0, 1, 1, 2, 2, 3, 3], id="batched_submission_order"),
            pytest.param(0, True, 1, 10, None, id="streaming"),
        ],
    )
    async def test_get_grouped_rollouts(
        self, num_slow_calls, streaming, num_groups, expected_count, expected_batch_ids
    ):
        gen = MockGenerator(parallel_generation_tasks=8, num_slow_calls=num_slow_calls)
        request = GroupedRolloutRequest(
            num_groups=num_groups,
            rollouts_per_group=1,
            inference_interface=MagicMock(spec=ReturnsRaw),
            streaming=streaming,
            enforce_order=num_groups > 1,
        )
        groups = []
        async for group in gen.get_grouped_rollouts(request):
            groups.append(group)
            if request.streaming and len(groups) >= expected_count:
                break

        assert len(groups) == expected_count
        if expected_batch_ids is not None:
            assert [g.batch_id for g in groups] == expected_batch_ids

    @pytest.mark.asyncio
    async def test_weighted_multi_task(self):
        configs = [
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "a"}, weight=3.0),
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "b"}, weight=1.0),
        ]
        mt = WeightedMultiTask(configs)
        mt.parallel_generation_tasks = 4

        captured = []
        for agent in mt.agents:
            original = agent.get_grouped_rollouts

            async def spy(req, orig=original):
                captured.append(req)
                async for group in orig(req):
                    yield group

            agent.get_grouped_rollouts = spy

        request = GroupedRolloutRequest(
            num_groups=4,
            rollouts_per_group=1,
            inference_interface=MagicMock(spec=ReturnsRaw),
            streaming=False,
            enforce_order=False,
        )
        groups = []
        async for group in mt.get_grouped_rollouts(request):
            groups.append(group)

        assert len(groups) == 4
        # Weights 3:1 → agent "a" produces 3 groups, agent "b" produces 1.
        env_ids = [g[0].env_id for g in groups]
        assert sorted(env_ids) == ["a", "a", "a", "b"]
        for sub_req in captured:
            assert sub_req.num_groups in (1, 3)  # distributed proportionally by weight
            assert sub_req.enforce_order == request.enforce_order
            assert sub_req.streaming == request.streaming

    @pytest.mark.parametrize(
        "buffered_groups, partial_rollouts, expect_inference",
        [
            pytest.param(6, True, False, id="drain_sufficient_skips_inference"),
            pytest.param(1, True, True, id="drain_insufficient_calls_inference"),
            pytest.param(6, False, True, id="non_partial_always_calls_inference"),
        ],
    )
    def test_get_environment_rollouts(self, buffered_groups, partial_rollouts, expect_inference):
        n_prompts = 4

        async def gen():
            for i in range(buffered_groups):
                yield _make_group(i)
            # Block forever so drain sees "nothing available" after the buffered items.
            await asyncio.sleep(1000)

        def mock_nvtx(*args, **kwargs):
            return nullcontext()

        mock_args = MagicMock()
        mock_args.rl_partial_rollouts = partial_rollouts
        mock_args.curr_iteration = 1
        mock_args.langrl_env_config = "test.yaml"

        loop = asyncio.new_event_loop()
        mock_colocated = MagicMock(return_value=[_make_group(i) for i in range(n_prompts)])
        stream = RolloutStream(gen())
        try:
            with (
                patch.multiple(
                    'megatron.rl.rl_utils',
                    colocated_inference=mock_colocated,
                    get_args=MagicMock(return_value=mock_args),
                    get_nvtx_range=MagicMock(return_value=mock_nvtx),
                    get_asyncio_loop=MagicMock(return_value=loop),
                    get_attr_wrapped_model=MagicMock(return_value=MagicMock()),
                    get_pg_size=MagicMock(return_value=1),
                    _ROLLOUT_GENERATOR=stream,
                ),
                patch('torch.distributed.get_rank', return_value=0),
                patch('torch.distributed.broadcast'),
                patch('torch.distributed.broadcast_object_list'),
            ):
                # Mirror the call pattern in get_grpo_data_iterator.
                need_new, need_inference = rl_utils.need_environment_rollouts(
                    None,
                    0,
                    MagicMock(last_collection_iteration=0),
                    grpo_iterations=1,
                    grpo_prompts_per_step=n_prompts,
                    grpo_group_size=1,
                    global_batch_size=1,
                    partial_rollouts=partial_rollouts,
                )
                assert need_new
                rollouts = rl_utils.get_environment_rollouts(
                    model=[MagicMock()],
                    inference_model=None,
                    optimizer=MagicMock(),
                    n_prompts=n_prompts,
                    samples_per_group=1,
                    run_inference=need_inference,
                )
                assert mock_colocated.called == expect_inference
                assert len(rollouts) == n_prompts
        finally:
            loop.run_until_complete(stream.aclose())
            loop.close()
