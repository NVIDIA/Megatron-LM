# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
import logging
from collections import deque
from typing import Any, Optional, Type

from ..types import GroupedRollouts, GroupQueuesPerEnv, RolloutGroup
from .api import (
    AgentBaseModel,
    ContrastiveRollout,
    ContrastiveRolloutGenerator,
    EnvAllocation,
    EvaluationAgent,
    EvaluationRequest,
    EvaluationResponse,
    GroupedRolloutGenerator,
    GroupedRolloutRequest,
    GroupRolloutParams,
    Rollout,
    RolloutGenerator,
    RolloutRequest,
)
from .registry import get_agent_class

logger = logging.getLogger(__name__)


class AgentConfig(AgentBaseModel):
    """Configuration for a single agent in the weighted multi-agent setup."""

    agent_type: Type[RolloutGenerator]
    agent_args: dict
    weight: float
    evaluation_only: bool = False

    def __init__(self, **data):
        super().__init__(**data)
        if self.weight < 0:
            raise ValueError("Agent weight must be non-negative")


class WeightedMultiTask(
    RolloutGenerator, GroupedRolloutGenerator, ContrastiveRolloutGenerator, EvaluationAgent
):
    """An agent that manages multiple sub-agents and distributes rollouts according to weights."""

    def __init__(self, agent_configs: list[AgentConfig]):
        super().__init__()
        if not agent_configs:
            raise ValueError("Must provide at least one agent configuration")

        # Initialize all sub-agents
        self.agents = []
        self.agent_configs = agent_configs  # Store the configs for later use
        # Recovered rollout-bank groups are another producer for each environment.
        # ``None`` means recovery was not configured; an empty dict means recovery
        # ran but found no groups. The distinction avoids requiring env_ids when the
        # durable bank is disabled.
        self._restored_groups: GroupQueuesPerEnv | None = None

        # Calculate total weight only among non-evaluation agents
        total_weight = sum(config.weight for config in agent_configs if not config.evaluation_only)
        if total_weight <= 0:
            raise ValueError("Total weight of non-evaluation agents must be positive")

        for config in agent_configs:
            self.agents.append(config.agent_type(**config.agent_args))

        # Weight-0 entries exist so the launcher boots their servers ("weight 0 = never
        # sampled"); exclude them here so the min-one-group bump cannot revive them.
        self._rollout_agents = []
        self._rollout_env_ids = []
        self._rollout_weights = []
        self._rollout_config_indices = []
        for idx, (agent, config) in enumerate(zip(self.agents, agent_configs)):
            env_id = getattr(agent, "env_id", None) or f"agent_{idx}"
            if config.evaluation_only or config.weight <= 0.0:
                if not config.evaluation_only:
                    logger.info(
                        "WeightedMultiTask: env %s has weight 0 and is excluded "
                        "from rollout generation.",
                        env_id,
                    )
                continue
            self._rollout_agents.append(agent)
            self._rollout_env_ids.append(env_id)
            self._rollout_weights.append(config.weight / total_weight)
            self._rollout_config_indices.append(idx)
        duplicates = {
            env_id for env_id in self._rollout_env_ids if self._rollout_env_ids.count(env_id) > 1
        }
        if duplicates:
            raise ValueError(
                f"Duplicate env_ids among weighted environments: {sorted(duplicates)}; "
                "per-env layout and metrics require unique names."
            )

    @classmethod
    def from_config(cls, config: list[dict[str, Any]]) -> 'WeightedMultiTask':
        """Create a WeightedMultiTask from a config list.

        Args:
            config: List of dicts with keys:
                - agent_type: Registered agent name (see megatron.rl.agent.registry)
                - agent_args: Dict of arguments to pass to agent constructor
                - weight: Float weight for this agent

        Returns:
            A WeightedMultiTask instance
        """
        agent_configs = []
        for entry in config:
            if not all(k in entry for k in ['agent_type', 'agent_args', 'weight']):
                raise ValueError(f"Missing required keys in config entry: {entry}")
            agent_args = entry.get('agent_args', {})
            agent_type = get_agent_class(entry['agent_type'])
            agent_configs.append(
                AgentConfig(
                    agent_type=agent_type,
                    agent_args=agent_args,
                    weight=float(entry['weight']),
                    evaluation_only=entry.get('evaluation_only', False),
                )
            )

        return cls(agent_configs)

    @staticmethod
    def _round_shares(targets: list[float], total: int) -> list[int]:
        """Round fractional targets to integers, awarding the shortfall to the largest residuals."""
        counts = [int(t) for t in targets]
        by_residual = sorted(
            range(len(targets)), key=lambda i: targets[i] - counts[i], reverse=True
        )
        for i in by_residual[: total - sum(counts)]:
            counts[i] += 1
        return counts

    def _quantized_counts(self, total: int) -> list[int]:
        """Quantize weights into integer counts summing to `total`, at least one per weighted env.

        Raises ValueError when total is smaller than the number of weighted envs.
        Note that this does not operate on eval-only tasks; those are pre-filtered in `__init__`.
        """
        num_envs = len(self._rollout_weights)
        if total < num_envs:
            raise ValueError(
                f"{num_envs} weighted environments cannot fit into {total} slots; "
                "increase the batch or request size."
            )
        exact = [weight * total for weight in self._rollout_weights]
        counts = self._round_shares(exact, total)
        # Round zero shares up to one, taking from the most over-served env.
        while 0 in counts:
            zero = counts.index(0)
            donor = max(
                (i for i in range(num_envs) if counts[i] >= 2),
                key=lambda i: counts[i] - exact[i],
            )
            counts[zero] += 1
            counts[donor] -= 1
        return counts

    def _distribute_counts(self, total_count: int) -> list[int]:
        """Split a count across weighted agents by weight (largest remainder, min one each).

        Returns a per-agent list summing to total_count, 0 only for evaluation-only
        and zero-weight agents; raises when total_count < the number of weighted envs.
        """
        shares = self._quantized_counts(total_count)
        counts = [0] * len(self.agent_configs)
        for idx, share in zip(self._rollout_config_indices, shares):
            counts[idx] = share
        return counts

    def _env_ids(self) -> list[str]:
        """Return active env IDs used to route restored rollout-bank groups."""
        require_env_ids = len(self._rollout_agents) > 1
        env_ids = []
        for index, agent in enumerate(self._rollout_agents):
            env_id = getattr(agent, "env_id", None)
            if not env_id and require_env_ids:
                raise ValueError(
                    f"Active agent {index} ({type(agent).__name__}) has no env_id; it is "
                    "required to weight-balance restored rollout-bank groups by env. "
                    "Set env_id when configuring multiple active agents."
                )
            env_ids.append(env_id or "")
        return env_ids

    def set_restored_groups(self, groups: GroupedRollouts) -> int:
        """Install recovered rollout-bank groups as per-environment producers."""
        known_env_ids = set(self._env_ids())
        restored: GroupQueuesPerEnv = {}
        for group in groups:
            if not group:
                continue
            env_id = group[0].env_id
            if env_id not in known_env_ids:
                raise ValueError(
                    f"Restored rollout-bank group has env_id {env_id!r} which is not in the "
                    f"current --langrl-env-config (known: {sorted(known_env_ids)}). Changing "
                    "the environment set across a crash-resume is unsupported; resume with a "
                    "matching config or clear the rollout bank."
                )
            restored.setdefault(env_id, deque()).append(group)
        self._restored_groups = restored
        return sum(len(queue) for queue in restored.values())

    def take_restored_group(self, env_id: str) -> RolloutGroup | None:
        """Return the next recovered group for an environment, if available."""
        restored = (self._restored_groups or {}).get(env_id)
        return restored.popleft() if restored else None

    def rollout_allocations(self, num_groups: int) -> list[EnvAllocation]:
        """Constant per-batch allocation for each weighted env, in env order.

        Weights that cannot be realized as an integer split of the batch are rounded with a warning.
        """
        for agent in self._rollout_agents:
            if not isinstance(agent, GroupedRolloutGenerator):
                raise TypeError(f"Agent of type {type(agent)} does not support grouped rollouts")

        counts = self._quantized_counts(num_groups)
        env_ids = self._env_ids() if self._restored_groups is not None else self._rollout_env_ids
        exact = [weight * num_groups for weight in self._rollout_weights]
        if any(abs(count - target) > 1e-9 for count, target in zip(counts, exact)):
            logger.warning(
                "WeightedMultiTask weights changed to fit num_groups=%d: %s",
                num_groups,
                ", ".join(
                    f"{eid}: {weight:g} -> {count}/{num_groups}"
                    for eid, weight, count in zip(
                        env_ids, self._rollout_weights, counts
                    )
                ),
            )
        logger.info(
            "WeightedMultiTask layout: num_groups=%d per_agent=%s",
            num_groups,
            ", ".join(
                f"{eid}(groups={c}, weight={w:g})"
                for eid, c, w in zip(env_ids, counts, self._rollout_weights)
            ),
        )
        return [
            EnvAllocation(agent=agent, env_id=env_id, num_groups=count)
            for agent, env_id, count in zip(
                self._rollout_agents, env_ids, counts
            )
        ]

    async def prepare_group_rollout(
        self, request: GroupedRolloutRequest, *, problem_state: dict | None = None
    ) -> GroupRolloutParams:
        raise NotImplementedError(
            "WeightedMultiTask only routes; the pipeline prepares each group via the "
            "agent in the matching rollout_allocations entry."
        )

    async def get_rollout_response(self, request, inference_request):
        raise NotImplementedError(
            "WeightedMultiTask delegates to sub-agents; get_rollout_response is not used."
        )

    async def get_reward_rollouts(self, request: RolloutRequest) -> list[Rollout]:
        """Distribute rollouts across sub-agents according to weights."""
        agent_rollouts = self._distribute_counts(request.num_rollouts)

        # Create tasks for each agent with non-zero rollouts
        tasks = []
        for agent, num_rollouts in zip(self.agents, agent_rollouts):
            if num_rollouts > 0:
                agent_request = RolloutRequest(
                    num_rollouts=num_rollouts,
                    inference_interface=request.inference_interface,
                    validation=request.validation,
                    generation_args=request.generation_args,
                )
                tasks.append(agent.get_reward_rollouts(agent_request))

        # Run all tasks concurrently and gather results
        all_rollouts_lists = await asyncio.gather(*tasks)
        return [rollout for rollouts in all_rollouts_lists for rollout in rollouts]

    async def get_contrastive_rollouts(self, request: RolloutRequest) -> list[ContrastiveRollout]:
        """Distribute contrastive rollouts across sub-agents according to weights."""
        agent_rollouts = self._distribute_counts(request.num_rollouts)

        # Create tasks for each agent with non-zero rollouts
        tasks = []
        for agent, num_rollouts in zip(self.agents, agent_rollouts):
            if num_rollouts > 0:
                if not isinstance(agent, ContrastiveRolloutGenerator):
                    raise TypeError(
                        f"Agent of type {type(agent)} does not support contrastive rollouts"
                    )

                agent_request = RolloutRequest(
                    num_rollouts=num_rollouts,
                    inference_interface=request.inference_interface,
                    validation=request.validation,
                    generation_args=request.generation_args,
                )
                tasks.append(agent.get_contrastive_rollouts(agent_request))

        # Run all tasks concurrently and gather results
        all_rollouts_lists = await asyncio.gather(*tasks)
        return [rollout for rollouts in all_rollouts_lists for rollout in rollouts]

    async def run_evaluation(self, request: EvaluationRequest) -> list[EvaluationResponse]:
        """Run evaluation across all sub-agents."""
        # Create tasks for each agent
        tasks = []
        for agent in self.agents:
            if not isinstance(agent, EvaluationAgent):
                raise TypeError(f"Agent of type {type(agent)} does not support evaluation")

            agent_request = EvaluationRequest(
                num_prompts=request.num_prompts,  # For evaluation, we don't distribute prompts
                rank_info=request.rank_info,  # Pass through original rank info
                inference_interface=request.inference_interface,
                validation=request.validation,
                generation_args=request.generation_args,
            )
            tasks.append(agent.run_evaluation(agent_request))

        # Run all tasks concurrently and gather results
        all_responses = await asyncio.gather(*tasks)

        return all_responses
