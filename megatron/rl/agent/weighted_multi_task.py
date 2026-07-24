# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional, Type

import numpy as np

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


@dataclass
class PgtRebalanceConfig:
    """Knobs for dynamic per-env inference-concurrency rebalancing."""

    ema_alpha: float = 0.2
    min_interval_s: float = 30.0
    min_samples_per_env: int = 8
    max_step_fraction: float = 0.25


class _PgtRebalancer:
    """Periodically shifts submission-gate capacity toward slower envs.

    The data mix (agent_groups / agent_slots) is untouched; only inference
    concurrency moves. Scoped to a single get_grouped_rollouts call, like
    _RolloutPipeline is for a single sub-agent."""

    def __init__(
        self,
        config: PgtRebalanceConfig,
        *,
        agent_indices: list[int],
        weights: list[float],
        current_pgts: list[int],
        min_pgts: list[int],
    ) -> None:
        self.config = config
        self.agent_indices = agent_indices
        self.weights = weights
        self.current = list(current_pgts)
        self.min_pgts = min_pgts
        self.total_pgt = sum(current_pgts)
        self.rebalance_count = 0
        self.latest_emas: list[float | None] = [None] * len(agent_indices)
        self._last_check = float("-inf")

    def maybe_rebalance(self, agents: list) -> dict | None:
        """Re-allocate gate capacity if the check interval elapsed and the
        allocation would change. Returns an event dict for logging, or None."""
        now = time.monotonic()
        if now - self._last_check < self.config.min_interval_s:
            return None
        self._last_check = now

        pipelines, durations = [], []
        for idx in self.agent_indices:
            pipeline = getattr(agents[idx], "_active_pipeline", None)
            pipelines.append(pipeline)
            if (
                pipeline is None
                or pipeline.engine_dwell_sample_count < self.config.min_samples_per_env
            ):
                durations.append(None)
            else:
                durations.append(pipeline.engine_dwell_ema)
        self.latest_emas = durations

        new = WeightedMultiTask._compute_pgt_allocation(
            self.weights,
            durations,
            self.total_pgt,
            self.current,
            min_pgts=self.min_pgts,
            max_step_fraction=self.config.max_step_fraction,
        )
        if new == self.current:
            return None
        old, self.current = self.current, new
        self.rebalance_count += 1
        for idx, pipeline, pgt in zip(self.agent_indices, pipelines, new):
            agents[idx].parallel_generation_tasks = pgt
            if pipeline is not None:
                pipeline.gate.set_capacity(pgt)
        return {"old": old, "new": new, "emas": durations}


class WeightedMultiTask(
    RolloutGenerator, GroupedRolloutGenerator, ContrastiveRolloutGenerator, EvaluationAgent
):
    """An agent that manages multiple sub-agents and distributes rollouts according to weights."""

    def __init__(
        self,
        agent_configs: list[AgentConfig],
        pgt_rebalance: PgtRebalanceConfig | None = None,
    ):
        super().__init__()
        if not agent_configs:
            raise ValueError("Must provide at least one agent configuration")
        self.pgt_rebalance = pgt_rebalance

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
    def from_config(
        cls,
        config: list[dict[str, Any]],
        *,
        parallel_generation_tasks: int | None = None,
        pgt_rebalance: PgtRebalanceConfig | None = None,
    ) -> 'WeightedMultiTask':
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
            agent_args['parallel_generation_tasks'] = parallel_generation_tasks

            agent_type = get_agent_class(entry['agent_type'])
            agent_configs.append(
                AgentConfig(
                    agent_type=agent_type,
                    agent_args=agent_args,
                    weight=float(entry['weight']),
                    evaluation_only=entry.get('evaluation_only', False),
                )
            )

        instance = cls(agent_configs, pgt_rebalance=pgt_rebalance)
        if parallel_generation_tasks is not None:
            instance.parallel_generation_tasks = parallel_generation_tasks
        return instance

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

    @staticmethod
    def _compute_pgt_allocation(
        weights: list[float],
        durations: list[float | None],
        total_pgt: int,
        current: list[int],
        *,
        min_pgts: list[int],
        max_step_fraction: float,
    ) -> list[int]:
        """Helper method to distribute parallel generation tasks by measured speed.

        The dynamic counterpart of _distribute_counts: allocates total_pgt so
        each env's share is proportional to weight_i * duration_i, equalizing
        per-env finish time for its share of the data mix. Envs without a
        duration estimate use the weighted mean of known durations; with no
        estimates at all, the current allocation is kept. Per-update movement
        is clamped to max_step_fraction of the current value (hysteresis) and
        floors are enforced; the result always sums to total_pgt.
        """
        known = [(w, d) for w, d in zip(weights, durations) if d is not None]
        if not known:
            return list(current)

        # Mean of known durations for fallback
        fallback = sum(w * d for w, d in known) / sum(w for w, _ in known)

        # Calculate shares for each agent
        shares = [w * (d if d is not None else fallback) for w, d in zip(weights, durations)]
        if (total_share := sum(shares)) <= 0:
            return list(current)

        # Calculate exact pgt for each agent
        exact = [total_pgt * s / total_share for s in shares]

        # Largest-remainder rounding (same scheme as _distribute_counts).
        target = [int(x) for x in exact]
        order = sorted(range(len(exact)), key=lambda i: exact[i] - target[i], reverse=True)
        for i in range(total_pgt - sum(target)):
            target[order[i]] += 1

        # Clamp per-update movement, then enforce floors.
        new = []
        for cur, tgt, floor in zip(current, target, min_pgts):
            step = max(1, round(max_step_fraction * cur))
            new.append(max(floor, min(max(tgt, cur - step), cur + step)))

        # Repair the sum (clamping/floors can break it); floors stay respected.
        diff = total_pgt - sum(new)
        while diff != 0:
            sign = 1 if diff > 0 else -1
            candidates = [i for i in range(len(new)) if sign > 0 or new[i] - 1 >= min_pgts[i]]
            i = max(candidates, key=lambda i: sign * (exact[i] - new[i]))
            new[i] += sign
            diff -= sign
        return new

    async def prepare_group_rollout(
        self,
        request: GroupedRolloutRequest,
    ) -> GroupRolloutParams:
        raise NotImplementedError(
            "WeightedMultiTask is a collection of tasks and therefore doesn't implement this method directly. Use get_grouped_rollouts instead to generate grouped rollouts."
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

    async def get_grouped_rollouts(self, request: GroupedRolloutRequest):
        """Distribute grouped rollouts across sub-agents according to weights."""
        agent_groups = self._distribute_counts(request.num_groups)
        if request.submission_granularity == "B":
            # In BATCH mode, pgt counts local batches in flight. agent_groups already
            # splits each batch by weight, so copy pgt to every active agent.
            agent_pgts = [
                self.parallel_generation_tasks if num_groups > 0 else 0
                for num_groups in agent_groups
            ]
        else:
            # In GROUP/ROLLOUT mode, pgt counts fine-grained work units, so split it by weight.
            agent_pgts = self._distribute_counts(self.parallel_generation_tasks)
        agent_slots = self._distribute_counts(request.num_groups, distribute_remainder=False)
        agent_slots = np.array(agent_slots) / np.gcd.reduce(agent_slots)

        # Optional dynamic rebalancing of inference concurrency (gate
        # capacity) toward slower envs. The data mix (agent_groups,
        # agent_slots) always stays at the configured weights. Not
        # applicable in B submission mode, where every active env already
        # receives the full pgt.
        rebalancer = None
        if self.pgt_rebalance is not None and request.submission_granularity != "B":
            active = [
                i
                for i, (groups, config) in enumerate(zip(agent_groups, self.agent_configs))
                if groups > 0 and not config.evaluation_only
            ]
            if len(active) >= 2:
                # G submission + B consumption releases slots only when a
                # full local batch is consumed; below agent_groups[i] slots
                # that batch can never complete, so it is the hard floor.
                needs_batch_floor = (
                    request.submission_granularity == "G"
                    and request.consumption_granularity == "B"
                )
                min_pgts = [agent_groups[i] if needs_batch_floor else 1 for i in active]
                total_active_pgt = sum(agent_pgts[i] for i in active)
                for pos, i in enumerate(active):
                    # Ceiling for this env's future growth: the total minus
                    # what the other envs can never go below. Sizes the
                    # fixed infer-worker pool in _RolloutPipeline.
                    self.agents[i].max_parallel_generation_tasks = (
                        total_active_pgt - sum(min_pgts) + min_pgts[pos]
                    )
                    self.agents[i].engine_dwell_ema_alpha = self.pgt_rebalance.ema_alpha
                rebalancer = _PgtRebalancer(
                    self.pgt_rebalance,
                    agent_indices=active,
                    weights=[self.weights[i] for i in active],
                    current_pgts=[agent_pgts[i] for i in active],
                    min_pgts=min_pgts,
                )

        # Snapshot the distribution for observability. Read back by rl_utils
        # during per-iteration metric logging.
        env_ids = [getattr(a, "env_id", f"agent_{i}") or f"agent_{i}"
                   for i, a in enumerate(self.agents)]
        self.latest_distribution = {
            "env_ids": env_ids,
            "agent_groups": list(agent_groups),
            "agent_pgts": list(agent_pgts),
            "agent_slots": agent_slots.tolist(),
            "total_pgt": int(sum(agent_pgts)),
            "num_groups": request.num_groups,
        }
        logger.info(
            "WeightedMultiTask distribution: sub=%s cons=%s num_groups=%d "
            "rollouts_per_group=%d total_pgt=%d per_agent="
            + ", ".join(
                f"{eid}(groups={g}, pgt={p}, slots={s:g})"
                for eid, g, p, s in zip(env_ids, agent_groups, agent_pgts, agent_slots)
            ),
            request.submission_granularity,
            request.consumption_granularity,
            request.num_groups,
            request.rollouts_per_group,
            int(sum(agent_pgts)),
        )

        # Create tasks for each agent with non-zero groups
        generators = []
        for agent, num_groups, pgt in zip(
            self.agents, agent_groups, agent_pgts, strict=True
        ):
            if num_groups > 0:
                if not isinstance(agent, GroupedRolloutGenerator):
                    raise TypeError(
                        f"Agent of type {type(agent)} does not support grouped rollouts"
                    )
                agent.parallel_generation_tasks = pgt
                agent_request = GroupedRolloutRequest(
                    num_groups=num_groups,
                    streaming=request.streaming,
                    rollouts_per_group=request.rollouts_per_group,
                    inference_interface=request.inference_interface,
                    validation=request.validation,
                    generation_args=request.generation_args,
                    filter_groups_with_same_reward=request.filter_groups_with_same_reward,
                    submission_granularity=request.submission_granularity,
                    consumption_granularity=request.consumption_granularity,
                )
                generators.append(agent.get_grouped_rollouts(agent_request))
            else:
                generators.append(None)

        tasks = []
        try:
            while any(generators):
                if rebalancer is not None and (event := rebalancer.maybe_rebalance(self.agents)):
                    live_pgts = list(agent_pgts)
                    duration_emas: list[float | None] = [None] * len(agent_pgts)
                    for pos, i in enumerate(rebalancer.agent_indices):
                        live_pgts[i] = rebalancer.current[pos]
                        duration_emas[i] = rebalancer.latest_emas[pos]
                    self.latest_distribution["live_pgts"] = live_pgts
                    self.latest_distribution["duration_ema_s"] = duration_emas
                    self.latest_distribution["rebalance_count"] = rebalancer.rebalance_count
                    logger.info(
                        "PGT rebalance #%d: %s -> %s (engine_dwell_ema=%s)",
                        rebalancer.rebalance_count,
                        event["old"],
                        event["new"],
                        event["emas"],
                    )
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
        finally:
            # When the consumer closes this generator early (streaming), shut
            # the sub-agent generators down too so their pipelines cancel
            # instead of leaking running tasks. The last round's puller tasks
            # must settle first: a generator with a pending anext rejects
            # aclose with "already running".
            await asyncio.gather(*tasks, return_exceptions=True)
            for generator in generators:
                if generator is not None:
                    try:
                        await generator.aclose()
                    except RuntimeError:
                        pass

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
