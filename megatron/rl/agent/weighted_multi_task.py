# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
from typing import Any, AsyncGenerator, Optional, Type

import numpy as np

from .. import import_class
from .api import (
    AgentBaseModel,
    ContrastiveRollout,
    ContrastiveRolloutGenerator,
    EvaluationAgent,
    EvaluationRequest,
    EvaluationResponse,
    GroupedRolloutGenerator,
    GroupedRolloutRequest,
    Rollout,
    RolloutGenerator,
    RolloutRequest,
)


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

    @classmethod
    def from_config(cls, config: list[dict[str, Any]]) -> 'WeightedMultiTask':
        """Create a WeightedMultiTask from a config list.

        Args:
            config: List of dicts with keys:
                - agent_type: String path to agent class
                - agent_args: Dict of arguments to pass to agent constructor
                - weight: Float weight for this agent

        Returns:
            A WeightedMultiTask instance
        """
        agent_configs = []
        for entry in config:
            if not all(k in entry for k in ['agent_type', 'agent_args', 'weight']):
                raise ValueError(f"Missing required keys in config entry: {entry}")

            # Import and instantiate the agent class
            agent_type = import_class(entry['agent_type'])
            agent_configs.append(
                AgentConfig(
                    agent_type=agent_type,
                    agent_args=entry['agent_args'],
                    weight=float(entry['weight']),
                    evaluation_only=entry.get('evaluation_only', False),
                )
            )

        return cls(agent_configs)

    def _distribute_counts(self, total_count: int, distribute_remainder: bool = True) -> list[int]:
        """Helper method to distribute counts according to weights.

        This implementation ensures the most balanced distribution possible while
        maintaining the relative proportions specified by weights.

        Args:
            total_count: Total number of items to distribute
            distribute_remainder: Whether to distribute the remainder of the counts to the agents with the largest fractional parts

        Returns:
            List of counts for each agent, summing to total_count
        """
        # Filter out evaluation-only agents for rollout distribution
        rollout_weights = [
            w for w, config in zip(self.weights, self.agent_configs) if not config.evaluation_only
        ]
        if not rollout_weights:
            raise ValueError("No non-evaluation agents available for rollout generation")

        # Calculate exact fractional counts
        exact_counts = [total_count * w for w in rollout_weights]

        # Get integer part of each count
        base_counts = [int(count) for count in exact_counts]
        remaining = total_count - sum(base_counts)

        if distribute_remainder:
            # Sort indices by fractional parts to distribute remaining counts
            # to those with largest fractional parts first
            fractional_parts = [count - int(count) for count in exact_counts]
            indices = list(range(len(rollout_weights)))
            indices.sort(key=lambda i: fractional_parts[i], reverse=True)

            # Distribute remaining counts
            for i in range(remaining):
                base_counts[indices[i]] += 1

        # Map back to original indices, skipping evaluation-only agents
        final_counts = []
        rollout_idx = 0
        for config in self.agent_configs:
            if config.evaluation_only:
                final_counts.append(0)
            else:
                final_counts.append(base_counts[rollout_idx])
                rollout_idx += 1

        return final_counts

    async def group_rollout(self, request: GroupedRolloutRequest) -> list[Rollout]:
        raise NotImplementedError(
            "WeightedMultiTask is a collection of tasks and therefore doesn't implement this method directly. Use get_grouped_rollouts instead to generate grouped rollouts."
        )

    async def rollout(self, request: RolloutRequest) -> Rollout:
        raise NotImplementedError(
            "WeightedMultiTask is a collection of tasks and therefore doesn't implement this method directly. Use get_reward_rollouts instead to generate rollouts."
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

    async def get_grouped_rollouts(self, request: GroupedRolloutRequest):
        """Distribute grouped rollouts across sub-agents according to weights."""
        if request.num_groups > 0:
            agent_groups = self._distribute_counts(request.num_groups)
        else:
            agent_groups = [-1 if not agent.evaluation_only else 0 for agent in self.agent_configs]
        parallel_generation_tasks = request.num_groups if request.num_groups > 0 else 10
        agent_slots = self._distribute_counts(parallel_generation_tasks, distribute_remainder=False)
        agent_slots = np.array(agent_slots) / np.gcd.reduce(agent_slots)

        # Create tasks for each agent with non-zero groups
        generators: list[AsyncGenerator[list[Rollout], None]] = []
        for agent, num_groups in zip(self.agents, agent_groups):
            if num_groups != 0:
                if not isinstance(agent, GroupedRolloutGenerator):
                    raise TypeError(
                        f"Agent of type {type(agent)} does not support grouped rollouts"
                    )

                agent_request = GroupedRolloutRequest(
                    num_groups=num_groups,
                    rollouts_per_group=request.rollouts_per_group,
                    inference_interface=request.inference_interface,
                    validation=request.validation,
                    generation_args=request.generation_args,
                    filter_groups_with_same_reward=request.filter_groups_with_same_reward,
                )
                group_generator = agent.get_grouped_rollouts(agent_request)
                generators.append(group_generator)
            else:
                generators.append(None)

        while any(generators):
            balanced_rollouts = asyncio.Queue()

            async def get_balanced_rollouts_if_remaining(agent_id):
                generated_rollouts = 0
                while generated_rollouts < agent_slots[agent_id]:
                    if generators[agent_id] is None:
                        return
                    try:
                        await balanced_rollouts.put(await anext(generators[agent_id]))
                        generated_rollouts += 1
                    except StopAsyncIteration:
                        await balanced_rollouts.put(None)
                        generators[agent_id] = None
                        return

            tasks = [
                asyncio.create_task(get_balanced_rollouts_if_remaining(agent_id))
                for agent_id in range(len(generators))
            ]

            try:
                while balanced_rollouts.qsize() > 0 or not all(task.done() for task in tasks):
                    rollout = await balanced_rollouts.get()
                    if rollout is not None:
                        yield rollout
            finally:
                for task in tasks:
                    task.cancel()

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
