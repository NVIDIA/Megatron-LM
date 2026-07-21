# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Smoke test keeping simulate_grouped_rollouts.py in sync with the pipeline API."""

import asyncio
from types import SimpleNamespace

import pytest

from megatron.rl.agent.weighted_multi_task import PgtRebalanceConfig
from tests.unit_tests.rl.simulate_grouped_rollouts import (
    MODES,
    EnvSpec,
    SimConfig,
    UniformLatencyInterface,
    run_mode,
    run_multi_env_mode,
    write_html_report,
)

TINY_CFG = SimConfig(
    num_groups=2,
    num_batches=2,
    rollouts_per_group=2,
    parallel_generation_tasks=2,
    latency_lo=0.001,
    latency_hi=0.002,
    seed=0,
)


class TestSimulateGroupedRollouts:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("submission, consumption", MODES)
    async def test_run_mode_smoke(self, submission, consumption):
        result = await run_mode(submission, consumption, TINY_CFG)

        expected_groups = TINY_CFG.num_batches * TINY_CFG.num_groups
        assert result.num_groups == expected_groups
        assert result.num_rollouts == expected_groups * TINY_CFG.rollouts_per_group
        assert len(result.rollout_walls) == result.num_rollouts
        assert result.total_wall > 0
        assert all(wall > 0 for wall in result.rollout_walls)
        assert len(result.sampled_latencies) >= result.num_rollouts
        assert result.counters["yielded"] >= result.num_groups
        assert (
            result.counters["prepared"]
            >= result.counters["inferred"]
            >= result.counters["assembled"] * TINY_CFG.rollouts_per_group
        )
        assert len(result.batch_done_at) == TINY_CFG.num_batches
        assert [t for _, t in result.batch_done_at] == sorted(t for _, t in result.batch_done_at)
        assert len(result.batch_inference_steps) == TINY_CFG.num_batches
        assert len(result.batch_max_seq_len) == TINY_CFG.num_batches
        lo_tokens = TINY_CFG.latency_lo / TINY_CFG.seconds_per_token
        hi_tokens = TINY_CFG.latency_hi / TINY_CFG.seconds_per_token
        assert all(lo_tokens <= max_len <= hi_tokens for max_len in result.batch_max_seq_len)
        # The first batch's longest rollout ran entirely inside the first
        # engine-active window, so its inference steps bound its max seq len.
        assert result.batch_inference_steps[0] >= result.batch_max_seq_len[0] * 0.99
        for name in ("infer_queue_dwell", "engine_dwell", "assemble_queue_dwell"):
            assert result.dwell[name], f"{name} collected no samples"
        if submission == "R":
            assert result.max_active_requests <= TINY_CFG.parallel_generation_tasks

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "submission, expected_capacity",
        [("B", 1), ("G", 2), ("R", 4)],  # lag 0: 1 batch / num_groups / num_groups*rollouts
    )
    async def test_generation_lag_capacity(self, submission, expected_capacity):
        cfg = SimConfig(
            num_groups=2,
            num_batches=2,
            rollouts_per_group=2,
            latency_lo=0.001,
            latency_hi=0.002,
            generation_lag=0,
        )
        result = await run_mode(submission, "B", cfg)
        assert result.gate["capacity"] == expected_capacity
        assert result.num_groups == cfg.num_batches * cfg.num_groups

    @pytest.mark.asyncio
    async def test_max_readahead_bounds_lag(self):
        cfg = SimConfig(
            num_groups=2,
            num_batches=6,
            rollouts_per_group=2,
            latency_lo=0.001,
            latency_hi=0.002,
            generation_lag=0,
            max_readahead_batches=2,
        )
        result = await run_mode("R", "G", cfg)
        assert result.num_groups == cfg.num_batches * cfg.num_groups
        assert max(result.readahead_batches) <= 2.0 + 1e-9
        assert max(result.batch_mean_lag) <= 2.0 + 1e-9

    @pytest.mark.asyncio
    async def test_pause_freezes_inflight_decodes(self):
        iface = UniformLatencyInterface(lo=0.02, hi=0.02, seed=0)
        request = SimpleNamespace(prompt=[SimpleNamespace(content="p0")])
        task = asyncio.create_task(iface.base_generate(request))
        await asyncio.sleep(0)  # let the decode reach its timed wait
        iface.pause()
        # Well past the 0.02s decode latency: a paused decode cannot finish
        # no matter how much wall time passes.
        await asyncio.sleep(0.1)
        assert not task.done(), "decode progressed while the engine was paused"
        iface.resume()
        response = await asyncio.wait_for(task, timeout=5)
        assert response.raw_text == "p0"
        assert iface.paused_seconds >= 0.1
        # The frozen window is excluded from the decode-step clock.
        assert iface.engine_active_time() < 0.1

    @pytest.mark.asyncio
    async def test_colocated_train_pause(self):
        cfg = SimConfig(
            num_groups=2,
            num_batches=3,
            rollouts_per_group=2,
            parallel_generation_tasks=2,
            latency_lo=0.001,
            latency_hi=0.002,
            train_time=0.05,
            colocated=True,
        )
        result = await run_mode("G", "B", cfg)
        assert result.num_groups == cfg.num_batches * cfg.num_groups
        # The consumer pauses the engine for every train step except after the
        # last batch (the loop breaks before that pause).
        expected_paused = (cfg.num_batches - 1) * cfg.train_time
        assert result.engine_paused_seconds >= expected_paused * 0.95
        # Consecutive batches are separated by at least the frozen train step.
        gaps = [t2 - t1 for (_, t1), (_, t2) in zip(result.batch_done_at, result.batch_done_at[1:])]
        assert all(gap >= cfg.train_time * 0.95 for gap in gaps)

    @pytest.mark.asyncio
    async def test_write_html_report(self, tmp_path):
        results = [await run_mode("G", "B", TINY_CFG)]
        report_path = tmp_path / "report.html"
        write_html_report(results, TINY_CFG, report_path)
        report = report_path.read_text()
        assert "<svg" in report and "G/B" in report and "data-tip" in report


MULTI_ENV_CFG = SimConfig(
    num_groups=4,
    num_batches=8,
    rollouts_per_group=2,
    latency_lo=0.001,
    latency_hi=0.002,
    generation_lag=1,
    seed=0,
)

# Same generation speed distribution shapes as the granularity runs: one env
# is ~5x slower per rollout than the other, equal data-mix weights.
FAST_SLOW = [EnvSpec(name="fast", lo=0.001, hi=0.003), EnvSpec(name="slow", lo=0.005, hi=0.010)]

REBALANCE_CFG = PgtRebalanceConfig(
    min_interval_s=0.0, min_samples_per_env=2, max_step_fraction=0.25
)


class TestMultiEnvSimulation:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("submission, consumption", [("R", "G"), ("G", "G")])
    async def test_static_split_smoke(self, submission, consumption):
        result = await run_multi_env_mode(submission, consumption, MULTI_ENV_CFG, FAST_SLOW)

        expected_groups = MULTI_ENV_CFG.num_batches * MULTI_ENV_CFG.num_groups
        assert result.num_groups == expected_groups
        # Equal weights: the data mix splits evenly regardless of env speed.
        assert result.groups_per_env == {"fast": expected_groups // 2, "slow": expected_groups // 2}
        assert len(result.batch_done_at) == MULTI_ENV_CFG.num_batches
        # No rebalancer: every sampled gate capacity stays at the static split.
        for capacities in result.pgt_trajectory.values():
            assert len(set(capacities)) == 1

    @pytest.mark.asyncio
    async def test_rebalancing_shifts_capacity_to_slow_env(self):
        result = await run_multi_env_mode(
            "R", "G", MULTI_ENV_CFG, FAST_SLOW, rebalance=REBALANCE_CFG
        )

        expected_groups = MULTI_ENV_CFG.num_batches * MULTI_ENV_CFG.num_groups
        assert result.num_groups == expected_groups
        # Rebalancing must not change the data mix.
        assert result.groups_per_env == {"fast": expected_groups // 2, "slow": expected_groups // 2}
        # The slow env's measured service time is higher...
        assert result.ema_per_env["slow"] > result.ema_per_env["fast"]
        # ...so capacity moves toward it while the total budget is conserved.
        assert result.pgt_trajectory["slow"][-1] > result.pgt_trajectory["fast"][-1]
        totals = {
            fast + slow
            for fast, slow in zip(result.pgt_trajectory["fast"], result.pgt_trajectory["slow"])
        }
        assert len(totals) == 1

    @pytest.mark.asyncio
    async def test_rebalancing_does_not_slow_the_run(self):
        static = await run_multi_env_mode("R", "G", MULTI_ENV_CFG, FAST_SLOW)
        rebalanced = await run_multi_env_mode(
            "R", "G", MULTI_ENV_CFG, FAST_SLOW, rebalance=REBALANCE_CFG
        )
        # Loose CI bound: same seed and workload, so rebalancing should be at
        # worst neutral. The decisive improvement claim lives in the manual
        # simulator A/B run, not this smoke bound.
        assert rebalanced.total_wall <= static.total_wall * 1.10
