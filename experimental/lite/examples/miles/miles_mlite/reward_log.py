# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Rollout-side reward marker logging for the miles MLite example.

miles' built-in rollout logger (``miles.ray.rollout.metrics.log_rollout_data``)
emits response-length / perf / zero-std stats but no reward mean. This thin hook,
wired through ``--custom-rollout-log-function-path``, runs in the rollout manager
(which holds every sample, so no parallel gather is needed) and logs a reward /
pass-rate marker. Reward scoring itself stays with miles (``--rm-type``); this only
reuses miles' ``compute_pass_rate`` to surface the signal. It is reward-rule
agnostic. Returning False lets miles' built-in logger run as well.
"""

from __future__ import annotations

import logging

from miles.utils.metric_utils import compute_pass_rate

logger = logging.getLogger(__name__)


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    rewards = [float(sample.get_reward_value(args)) for sample in samples]
    if not rewards:
        return False
    passrate = compute_pass_rate(
        rewards,
        group_size=args.n_samples_per_prompt,
        num_groups=args.rollout_batch_size,
    )
    metrics = {
        "reward/score/mean": sum(rewards) / len(rewards),
        "reward/score/nonzero": sum(1 for reward in rewards if reward != 0.0),
        "reward/score/count": len(rewards),
    }
    metrics.update({f"reward/passrate/{key}": value for key, value in passrate.items()})
    logger.info("reward/score/passrate rollout %s: %s", rollout_id, metrics)
    return False
