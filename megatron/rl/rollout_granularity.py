# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""RL rollout submission and consumption granularity values."""

from typing import Literal

SubmissionGranularity = Literal["R", "G", "B"]
ConsumptionGranularity = Literal["G", "B"]
ReleaseState = Literal["inferred", "assembled", "consumed"]


RELEASE_STATE_BY_SUBMISSION: dict[SubmissionGranularity, ReleaseState] = {
    "R": "inferred",
    "G": "assembled",
    "B": "consumed",
}


def get_rl_parallel_generation_tasks(args) -> int:
    """Return the number of generation slots implied by RL lag and submission granularity.

    The lag may be fractional or negative (>= -1); the slot count is rounded and clamped
    to at least one.
    """
    parallel_generation_tasks = args.rl_generation_lag + 1
    if args.rl_submission_granularity != "B":
        parallel_generation_tasks *= args.grpo_prompts_per_step
    if args.rl_submission_granularity == "R":
        parallel_generation_tasks *= args.grpo_group_size
    return max(1, round(parallel_generation_tasks))


def resolve_rl_generation_lag(args, dp_size: int, max_requests: int) -> None:
    """Resolve args.rl_generation_lag against the inference engine's request capacity.

    Autotunes the lag to fill the engine when unset; otherwise reports how the requested
    lag compares to the maximum effective lag the engine can serve.
    """
    # Import here to avoid circular imports.
    from megatron.training.utils import print_rank_0

    G = args.grpo_group_size
    P = args.grpo_prompts_per_step
    max_effective_groups = max(1, dp_size * max_requests // G)
    max_effective_lag = max_effective_groups / P - 1
    if args.rl_generation_lag is None:
        args.rl_generation_lag = max_effective_lag
        print_rank_0(
            f"Autotuned rl-generation-lag={max_effective_lag:.2f} "
            f"({get_rl_parallel_generation_tasks(args)} parallel generation tasks at "
            f"submission granularity {args.rl_submission_granularity}; "
            f"DP={dp_size}, max_requests={max_requests}, G={G}, P={P}).")
    else:
        tasks = get_rl_parallel_generation_tasks(args)
        rollouts_per_task = {"B": P * G, "G": G, "R": 1}[args.rl_submission_granularity]
        groups_in_flight = tasks * rollouts_per_task / G
        actual_lag = groups_in_flight / P - 1
        print_rank_0(
            f"Using rl-generation-lag={args.rl_generation_lag} "
            f"(actual={actual_lag:.2f}, {tasks} parallel "
            f"generation tasks at submission granularity {args.rl_submission_granularity}; "
            f"max effective lag={max_effective_lag:.2f}; "
            f"DP={dp_size}, max_requests={max_requests}, G={G}, P={P}).")
        if groups_in_flight > max_effective_groups:
            print_rank_0(
                f"WARNING: --rl-generation-lag {args.rl_generation_lag} oversubscribes the "
                f"inference engine (max effective lag is {max_effective_lag:.2f}). "
                f"Additional lag beyond that point has no benefit.")
    if max_effective_lag < 0:
        print_rank_0(
            f"WARNING: max effective lag is {max_effective_lag:.2f} (negative) — the "
            f"inference engine cannot hold even one training step's worth of rollouts "
            f"({max_effective_groups} groups < P={P}). Even fully-synchronous GRPO would "
            f"oversubscribe. Consider scaling up inference resources.")
