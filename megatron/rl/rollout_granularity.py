# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""RL rollout submission and consumption granularity values."""


def get_rl_parallel_generation_tasks(args) -> int:
    """Return the number of generation slots implied by RL lag and submission granularity."""
    parallel_generation_tasks = args.rl_generation_lag + 1
    if args.rl_submission_granularity != "B":
        parallel_generation_tasks *= args.grpo_prompts_per_step
    if args.rl_submission_granularity == "R":
        parallel_generation_tasks *= args.grpo_group_size
    return parallel_generation_tasks
