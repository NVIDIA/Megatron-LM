# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""RL rollout submission and consumption granularity values."""

from enum import Enum


class RLRolloutGranularity(str, Enum):
    """Granularity for RL rollout submission or consumption."""

    ROLLOUT = 'R'
    GROUP = 'G'
    BATCH = 'B'

    def __str__(self):
        return self.value


def get_rl_parallel_generation_tasks(args) -> int:
    """Return the number of generation slots implied by RL lag and submission granularity."""
    parallel_generation_tasks = args.rl_generation_lag + 1
    if args.rl_submission_granularity != RLRolloutGranularity.BATCH:
        parallel_generation_tasks *= args.grpo_prompts_per_step
    if args.rl_submission_granularity == RLRolloutGranularity.ROLLOUT:
        parallel_generation_tasks *= args.grpo_group_size
    return parallel_generation_tasks
