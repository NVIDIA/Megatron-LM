# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
import logging
from typing import Any, Optional, Type

from .registry import get_agent_class
from .api import (
    AgentBaseModel,
    ContrastiveRollout,
    ContrastiveRolloutGenerator,
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
        self.weights = []
        self.agent_configs = agent_configs  # Store the configs for later use

        # Calculate total weight only among non-evaluation agents
        total_weight = sum(config.weight for config in agent_configs if not config.evaluation_only)
        if total_weight <= 0:
            raise ValueError("Total weight of non-evaluation agents must be positive")

        for config in agent_configs:
            # Initialize the agent with its arguments
            agent = config.agent_type(**config.agent_args)
            self.agents.append(agent)
            # Only normalize weights for non-evaluation agents
            if config.evaluation_only:
                self.weights.append(0.0)
            else:
                self.weights.append(config.weight / total_weight)

        # Weighted (non-evaluation) environment handling.
        self._rollout_agents = []
        self._rollout_env_ids = []
        self._rollout_weights = []
        for idx, (agent, config) in enumerate(zip(self.agents, agent_configs)):
            if config.evaluation_only:
                continue
            self._rollout_agents.append(agent)
            self._rollout_env_ids.append(getattr(agent, "env_id", None) or f"agent_{idx}")
            self._rollout_weights.append(self.weights[idx])

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
        counts = [max(0, int(t)) for t in targets]
        by_residual = sorted(
            range(len(targets)), key=lambda i: targets[i] - counts[i], reverse=True
        )
        for i in by_residual[: max(0, total - sum(counts))]:
            counts[i] += 1
        return counts

    def _distribute_counts(self, total_count: int) -> list[int]:
        """Split a count across non-evaluation agents by weight (largest remainder).

        Returns a per-agent list summing to total_count, 0 for evaluation-only agents.
        """
        if not self._rollout_weights:
            raise ValueError("No non-evaluation agents available for rollout generation")

        shares = iter(
            self._round_shares([total_count * w for w in self._rollout_weights], total_count)
        )
        return [0 if config.evaluation_only else next(shares) for config in self.agent_configs]

    def rollout_group_layout(self, num_groups: int) -> list[int]:
        """Constant per-batch groups for each weighted env, in env order.

        Weights that cannot be realized as an integer split of the batch are rounded with a warning.
        """
        num_envs = len(self._rollout_agents)
        if num_groups < num_envs:
            raise ValueError(
                f"{num_envs} weighted environments cannot fit into a batch of "
                f"{num_groups} groups; increase the batch size."
            )
        for agent in self._rollout_agents:
            if not isinstance(agent, GroupedRolloutGenerator):
                raise TypeError(f"Agent of type {type(agent)} does not support grouped rollouts")

        exact = [weight * num_groups for weight in self._rollout_weights]
        counts = self._round_shares(exact, num_groups)
        # Round zero shares up to one group, taking from the most over-served env.
        while 0 in counts:
            zero = counts.index(0)
            donor = max(
                (i for i in range(num_envs) if counts[i] >= 2),
                key=lambda i: counts[i] - exact[i],
            )
            counts[zero] += 1
            counts[donor] -= 1
        if any(abs(count - target) > 1e-9 for count, target in zip(counts, exact)):
            logger.warning(
                "WeightedMultiTask weights changed to fit num_groups=%d: %s",
                num_groups,
                ", ".join(
                    f"{eid}: {weight:g} -> {count}/{num_groups}"
                    for eid, weight, count in zip(
                        self._rollout_env_ids, self._rollout_weights, counts
                    )
                ),
            )
        # Snapshot for metric logging; read back by rl_utils.
        self.latest_distribution = {
            "env_ids": list(self._rollout_env_ids),
            "agent_groups": list(counts),
            "weights": list(self._rollout_weights),
            "num_groups": num_groups,
        }
        logger.info(
            "WeightedMultiTask layout: num_groups=%d per_agent=%s",
            num_groups,
            ", ".join(
                f"{eid}(groups={c}, weight={w:g})"
                for eid, c, w in zip(self._rollout_env_ids, counts, self._rollout_weights)
            ),
        )
        return counts

    async def prepare_group_rollout(
        self,
        request: GroupedRolloutRequest,
        env_index: int = 0,
    ) -> GroupRolloutParams:
        """Route the group to the sub-agent owning env slot `env_index`."""
        return await self._rollout_agents[env_index].prepare_group_rollout(request)

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
