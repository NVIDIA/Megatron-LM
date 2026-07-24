# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Standalone simulator for GroupedRolloutGenerator.get_grouped_rollouts.

Drives the real _RolloutPipeline in megatron/rl/agent/api.py for each valid
(submission, consumption) granularity combination with inference mocked as a
uniform-random sleep (no network), then reports per-mode timings: overall
get_grouped_rollouts wall time, per-rollout prepare-to-yield wall times, the
sampled generation-latency distribution, per-training-step inference steps
(engine decode steps, in tokens) versus each batch's max sequence length, and
the pipeline's own queue-dwell and gate metrics.

Run from the repo root:

    uv run python tests/unit_tests/rl/simulate_grouped_rollouts.py
"""

import argparse
import asyncio
import html
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pydantic import PrivateAttr

# Make megatron importable when this file is executed directly as a script
# (sys.path[0] is then this directory, not the repo root).
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

if __name__ == "__main__":
    print("importing megatron (takes ~10s)...", file=sys.stderr, flush=True)

from megatron.rl.agent.api import (  # noqa: E402
    GroupedRolloutGenerator,
    GroupedRolloutRequest,
    GroupRolloutParams,
    Rollout,
    RolloutGenerator,
)
from megatron.rl.agent.weighted_multi_task import (  # noqa: E402
    AgentConfig,
    PgtRebalanceConfig,
    WeightedMultiTask,
)
from megatron.rl.inference import InferenceResponse, LLMChatMessage, ReturnsRaw  # noqa: E402
from megatron.rl.rollout_granularity import get_rl_parallel_generation_tasks  # noqa: E402

# All valid (submission, consumption) combinations. B/G is rejected by
# _GranularityConfig._validate in megatron/rl/agent/api.py.
MODES = [("B", "B"), ("G", "G"), ("G", "B"), ("R", "B"), ("R", "G")]


class UniformLatencyInterface(ReturnsRaw):
    """Mocked inference backend: each call sleeps uniform(lo, hi) seconds.

    Stands in for the external HTTP inference server. Must never raise:
    pipeline coroutines are wrapped in trace_async_exceptions, which calls
    sys.exit(1) on any exception.
    """

    lo: float
    hi: float
    seed: int = 0
    # Per-env latency ranges, keyed by the "{env}:" prompt prefix that
    # SimAgent stamps on every prompt. Envs without an entry use (lo, hi).
    # All envs share this one interface — i.e. one engine.
    ranges: dict[str, tuple[float, float]] = {}
    # Fixed number of engine slots (continuous-batching rows). None = unlimited:
    # requests never queue, so slot-idle "gaps" cannot exist.
    slots: int | None = None
    # Current policy version (batches trained so far); bumped by the consumer
    # after each simulated train step. Stamped into each response's
    # policy_epoch, mirroring what the real engine records per rollout.
    policy_version: int = 0
    active_requests: int = 0
    max_active_requests: int = 0
    busy_seconds: float = 0.0
    sampled_latencies: list[float] = []
    # Sampled latencies keyed by prompt ("t{group_idx}"), so the consumer can
    # recover each consumed group's rollout lengths.
    delays_by_prompt: dict[str, list[float]] = {}
    # Cumulative wall time with >=1 active request: a continuous-batching
    # engine runs one decode step per token-time whenever any row is occupied,
    # so this is the engine's decode-step clock.
    engine_active_seconds: float = 0.0
    # True while a colocated train step holds the GPUs: in-flight decodes make
    # no progress until resume(). paused_seconds accumulates the frozen time.
    paused: bool = False
    paused_seconds: float = 0.0

    _rng: random.Random | None = PrivateAttr(default=None)
    _sem: asyncio.Semaphore | None = PrivateAttr(default=None)
    _active_since: float | None = PrivateAttr(default=None)
    _pause_event: asyncio.Event | None = PrivateAttr(default=None)
    _resume_event: asyncio.Event | None = PrivateAttr(default=None)
    _paused_since: float | None = PrivateAttr(default=None)

    def engine_active_time(self):
        """engine_active_seconds including the currently open active window."""
        active = self.engine_active_seconds
        if self._active_since is not None:
            active += time.monotonic() - self._active_since
        return active

    def _ensure_pause_events(self):
        if self._pause_event is None:
            self._pause_event = asyncio.Event()
            self._resume_event = asyncio.Event()
            self._resume_event.set()

    def pause(self):
        """Freeze the engine (colocated train step): in-flight decodes stop."""
        if self.paused:
            return
        self._ensure_pause_events()
        self.paused = True
        self._paused_since = time.monotonic()
        if self._active_since is not None:
            self.engine_active_seconds += self._paused_since - self._active_since
            self._active_since = None
        self._resume_event = asyncio.Event()
        self._pause_event.set()

    def resume(self):
        """Unfreeze the engine: frozen decodes pick up where they stopped."""
        if not self.paused:
            return
        self.paused = False
        now = time.monotonic()
        self.paused_seconds += now - self._paused_since
        self._paused_since = None
        if self.active_requests > 0:
            self._active_since = now
        self._pause_event = asyncio.Event()
        self._resume_event.set()

    async def _decode_sleep(self, delay):
        """asyncio.sleep(delay) whose clock does not advance while paused."""
        self._ensure_pause_events()
        remaining = delay
        while True:
            if self.paused:
                await self._resume_event.wait()
            started = time.monotonic()
            try:
                # Wakes early when a pause begins mid-decode; only the wall
                # time actually decoded is charged against `remaining`.
                await asyncio.wait_for(self._pause_event.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                return
            remaining -= time.monotonic() - started

    async def base_generate(self, request):
        if self._rng is None:
            self._rng = random.Random(self.seed)
        if self.slots and self._sem is None:
            self._sem = asyncio.Semaphore(self.slots)
        prompt = request.prompt[0].content
        env = prompt.split(":", 1)[0] if ":" in prompt else ""
        lo, hi = self.ranges.get(env, (self.lo, self.hi))
        delay = self._rng.uniform(lo, hi)
        self.sampled_latencies.append(delay)
        self.delays_by_prompt.setdefault(prompt, []).append(delay)
        if self._sem is not None:
            await self._sem.acquire()
        self.active_requests += 1
        if self.active_requests == 1 and not self.paused:
            self._active_since = time.monotonic()
        self.max_active_requests = max(self.max_active_requests, self.active_requests)
        try:
            await self._decode_sleep(delay)
            self.busy_seconds += delay
            version = self.policy_version
            return InferenceResponse(
                response=LLMChatMessage(role="assistant", content=prompt),
                raw_text=prompt,
                finish_reason="stop",
                policy_epoch=[(version, version)],
                kv_cache_epoch=[(0, 0)],
                num_evictions=0,
            )
        finally:
            self.active_requests -= 1
            if self.active_requests == 0 and self._active_since is not None:
                self.engine_active_seconds += time.monotonic() - self._active_since
                self._active_since = None
            if self._sem is not None:
                self._sem.release()


class SimAgent(RolloutGenerator, GroupedRolloutGenerator):
    """Agent whose only per-rollout work is one mocked inference call.

    Also nominally a RolloutGenerator so AgentConfig accepts it when the
    multi-env simulation wraps SimAgents in a real WeightedMultiTask.
    """

    def __init__(self, env_name="sim", **kwargs):
        super().__init__(**kwargs)
        self.env_id = env_name
        self.group_prepared_at = {}
        self._group_count = 0
        # Readahead window (prototype lag bound): when set, prepare may not
        # run more than this many batches ahead of consumption. Wired up by
        # run_mode from SimConfig.max_readahead_batches.
        self._readahead_window = None
        self._readahead_iface = None
        self._groups_per_batch = None

    async def get_reward_rollouts(self, request):
        raise NotImplementedError

    async def get_rollout_response(self, request, inference_request):
        return await request.inference_interface.agenerate(inference_request)

    async def prepare_group_rollout(self, request):
        if self._readahead_window is not None:
            # Awaiting here blocks stage_prepare, which is exactly the
            # submission throttle point (called once per group).
            limit = lambda: (
                (self._readahead_iface.policy_version + self._readahead_window)
                * self._groups_per_batch
            )
            while self._group_count >= limit():
                await asyncio.sleep(0.001)
        idx = self._group_count
        self._group_count += 1
        self.group_prepared_at[idx] = time.monotonic()
        inference_request = request.inference_interface.prepare_request(
            f"{self.env_id}:t{idx}", request.generation_args
        )

        async def build_rollout(response):
            # problem_id carries the group index so the consumer can look up
            # the group's prepare timestamp.
            return Rollout(
                trajectory=[response.raw_text],
                reward=0.0,
                env_id=self.env_id,
                problem_id=str(idx),
                policy_epoch=[response.policy_epoch],
                kv_cache_epoch=[response.kv_cache_epoch],
                num_evictions=[response.num_evictions],
            )

        return GroupRolloutParams(inference_request=inference_request, build_rollout=build_rollout)


@dataclass
class SimConfig:
    num_groups: int = 8  # groups per batch
    num_batches: int = 4
    rollouts_per_group: int = 4
    parallel_generation_tasks: int = 4
    latency_lo: float = 0.01
    latency_hi: float = 0.05
    seed: int = 0
    streaming: bool = True
    # Simulated trainer pause after each completed batch; generation whose gate
    # slots are already free keeps running during the pause.
    train_time: float = 0.0
    # Colocated trainer: the train step holds the GPUs, so the engine pauses
    # for train_time and in-flight decodes freeze (False = disaggregated,
    # generation keeps running through the pause).
    colocated: bool = False
    # Fixed engine slot count (None = unlimited, the pre-existing behavior).
    engine_slots: int | None = None
    # When set, gate capacity is derived per mode from the production formula
    # (get_rl_parallel_generation_tasks) instead of parallel_generation_tasks.
    generation_lag: int | None = None
    # Prototype lag bound: prepare may not run more than this many batches
    # ahead of consumption (None = current production behavior, unbounded
    # for R submission).
    max_readahead_batches: int | None = None
    # Token clock: a rollout whose mocked latency is D seconds stands for a
    # D / seconds_per_token-token generation. Converts engine-active time into
    # inference steps (decode steps) and latencies into sequence lengths.
    seconds_per_token: float = 0.001

    def gate_capacity(self, submission):
        if self.generation_lag is None:
            return self.parallel_generation_tasks
        return get_rl_parallel_generation_tasks(
            SimpleNamespace(
                rl_generation_lag=self.generation_lag,
                rl_submission_granularity=submission,
                grpo_prompts_per_step=self.num_groups,
                grpo_group_size=self.rollouts_per_group,
            )
        )


@dataclass
class ModeResult:
    mode: str
    total_wall: float
    num_groups: int
    num_rollouts: int
    rollout_walls: list[float]
    sampled_latencies: list[float]
    batch_done_at: list[tuple[int, float]]  # (batch_id, seconds since start)
    # Readahead in batches at each batch completion: (groups prepared - groups
    # yielded) / groups per batch. Proxy for policy staleness of in-flight work.
    readahead_batches: list[float]
    # Mean policy lag of each consumed batch's rollouts (policy version at
    # training minus version stamped at generation) — the simulator twin of
    # the mean_policy_staleness metric in W&B.
    batch_mean_lag: list[float]
    # Inference steps charged to each training step: engine-active time between
    # consecutive batch completions / seconds_per_token, i.e. decode steps the
    # trainer waited through (including readahead work for future batches).
    batch_inference_steps: list[float]
    # Longest rollout in each consumed batch, in tokens (max sampled latency /
    # seconds_per_token). inference_steps == max_seq_len means the trainer
    # waited only for the batch's longest rollout — the fully-parallel bound.
    batch_max_seq_len: list[float]
    engine_slots: int | None
    engine_utilization: (
        float | None
    )  # busy slot-seconds / (slots * unpaused wall); None if unlimited
    engine_paused_seconds: float  # wall time the engine spent frozen (colocated train steps)
    max_active_requests: int
    dwell: dict[str, list[float]]
    counters: dict[str, int]
    gate: dict[str, float]

    @property
    def batch_step_ratio(self):
        """Inference steps per training step relative to the batch's max seq len."""
        return [
            steps / max_len
            for steps, max_len in zip(self.batch_inference_steps, self.batch_max_seq_len)
        ]


async def run_mode(submission, consumption, cfg: SimConfig) -> ModeResult:
    """Drive get_grouped_rollouts for one granularity mode and collect timings."""
    iface = UniformLatencyInterface(
        lo=cfg.latency_lo, hi=cfg.latency_hi, seed=cfg.seed, slots=cfg.engine_slots
    )
    agent = SimAgent(parallel_generation_tasks=cfg.gate_capacity(submission))
    if cfg.max_readahead_batches is not None:
        agent._readahead_window = cfg.max_readahead_batches
        agent._readahead_iface = iface
        agent._groups_per_batch = cfg.num_groups
    request = GroupedRolloutRequest(
        num_groups=cfg.num_groups,
        rollouts_per_group=cfg.rollouts_per_group,
        inference_interface=iface,
        streaming=cfg.streaming,
        submission_granularity=submission,
        consumption_granularity=consumption,
    )
    # In streaming mode stage_prepare runs forever, so the consumer breaks
    # after num_batches batches. Non-streaming generates exactly one batch.
    target_groups = cfg.num_batches * cfg.num_groups if cfg.streaming else cfg.num_groups

    pipeline = None
    groups = []
    rollout_walls = []
    batch_group_counts = {}
    batch_done_at = []
    readahead_batches = []
    batch_gen_versions = {}
    batch_mean_lag = []
    batch_max_delay = {}
    batch_inference_steps = []
    batch_max_seq_len = []
    engine_steps_snapshot = 0.0
    agen = agent.get_grouped_rollouts(request)
    t0 = time.monotonic()
    try:
        async for group in agen:
            if pipeline is None:
                # Grab a reference before get_grouped_rollouts' finally block
                # clears agent._active_pipeline.
                pipeline = agent._active_pipeline
            now = time.monotonic()
            prepared_at = agent.group_prepared_at[int(group[0].problem_id)]
            rollout_walls.extend([now - prepared_at] * len(group))
            groups.append(group)
            batch_group_counts[group.batch_id] = batch_group_counts.get(group.batch_id, 0) + 1
            batch_gen_versions.setdefault(group.batch_id, []).extend(
                rollout.policy_epoch[0][0][0] for rollout in group
            )
            group_delays = iface.delays_by_prompt[f"{agent.env_id}:t{int(group[0].problem_id)}"]
            batch_max_delay[group.batch_id] = max(
                batch_max_delay.get(group.batch_id, 0.0), max(group_delays)
            )
            batch_completed = batch_group_counts[group.batch_id] == cfg.num_groups
            if batch_completed:
                batch_done_at.append((group.batch_id, now - t0))
                readahead_batches.append((agent._group_count - len(groups)) / cfg.num_groups)
                gen_versions = batch_gen_versions.pop(group.batch_id)
                batch_mean_lag.append(
                    sum(iface.policy_version - v for v in gen_versions) / len(gen_versions)
                )
                engine_steps_now = iface.engine_active_time()
                batch_inference_steps.append(
                    (engine_steps_now - engine_steps_snapshot) / cfg.seconds_per_token
                )
                engine_steps_snapshot = engine_steps_now
                batch_max_seq_len.append(
                    batch_max_delay.pop(group.batch_id) / cfg.seconds_per_token
                )
            if cfg.streaming and len(groups) >= target_groups:
                break
            if batch_completed:
                # The trainer runs its step. Disaggregated (default): the
                # pipeline keeps whatever generation its gate slots allow
                # running in the background. Colocated: the trainer holds the
                # GPUs, so the engine pauses and in-flight decodes freeze.
                # Afterwards the weights are refit: the policy version bumps.
                if cfg.train_time:
                    if cfg.colocated:
                        iface.pause()
                    try:
                        await asyncio.sleep(cfg.train_time)
                    finally:
                        if cfg.colocated:
                            iface.resume()
                iface.policy_version += 1
    finally:
        # Break leaves the async generator suspended; close it so the pipeline
        # tasks are cancelled inside the timed window and nothing leaks.
        await agen.aclose()
    total_wall = time.monotonic() - t0

    if pipeline is None:
        raise RuntimeError(f"{submission}/{consumption}: no groups were yielded")
    gate = pipeline.gate
    return ModeResult(
        mode=f"{submission}/{consumption}",
        total_wall=total_wall,
        num_groups=len(groups),
        num_rollouts=sum(len(group) for group in groups),
        rollout_walls=rollout_walls,
        sampled_latencies=list(iface.sampled_latencies),
        batch_done_at=batch_done_at,
        readahead_batches=readahead_batches,
        batch_mean_lag=batch_mean_lag,
        batch_inference_steps=batch_inference_steps,
        batch_max_seq_len=batch_max_seq_len,
        engine_slots=cfg.engine_slots,
        engine_utilization=(
            iface.busy_seconds / (cfg.engine_slots * (total_wall - iface.paused_seconds))
            if cfg.engine_slots
            else None
        ),
        engine_paused_seconds=iface.paused_seconds,
        max_active_requests=iface.max_active_requests,
        dwell={
            "infer_queue_dwell": list(pipeline.infer_queue_dwell),
            "engine_dwell": list(pipeline.engine_dwell),
            "assemble_queue_dwell": list(pipeline.assemble_queue_dwell),
            "output_queue_dwell": list(pipeline.output_queue_dwell),
        },
        counters={
            "prepared": pipeline.prepared_count,
            "inferred": pipeline.inferred_count,
            "assembled": pipeline.assembled_count,
            "yielded": pipeline.yielded_count,
        },
        gate={
            "capacity": gate.capacity,
            "held_at_end": gate.held,
            "prepare_blocked_seconds": gate.prepare_blocked_seconds,
            "acquire_calls": gate.acquire_calls,
            "release_calls": gate.release_calls,
        },
    )


@dataclass
class EnvSpec:
    """One simulated environment: its name, latency range, and data-mix weight."""

    name: str
    lo: float
    hi: float
    weight: float = 1.0


@dataclass
class MultiEnvResult:
    mode: str
    rebalanced: bool
    total_wall: float
    num_groups: int
    groups_per_env: dict[str, int]
    batch_done_at: list[float]  # seconds since start, one per completed mixed batch
    # Each env's gate capacity sampled at every batch completion.
    pgt_trajectory: dict[str, list[int]]
    # Each env's engine_dwell EMA at the end of the run (None = no samples).
    ema_per_env: dict[str, float | None]
    engine_utilization: float | None
    max_active_requests: int


async def run_multi_env_mode(
    submission,
    consumption,
    cfg: SimConfig,
    envs: list[EnvSpec],
    rebalance: PgtRebalanceConfig | None = None,
) -> MultiEnvResult:
    """Drive a real WeightedMultiTask over heterogeneous envs sharing one engine.

    Mirrors production wiring: one SimAgent per env under WeightedMultiTask,
    the request's num_groups is one trainer batch, and every cfg.num_groups
    yielded (mixed) groups count as one consumed batch, after which the
    policy version bumps (and the engine optionally pauses, colocated-style).
    """
    iface = UniformLatencyInterface(
        lo=cfg.latency_lo,
        hi=cfg.latency_hi,
        seed=cfg.seed,
        slots=cfg.engine_slots,
        ranges={env.name: (env.lo, env.hi) for env in envs},
    )
    multi_task = WeightedMultiTask(
        [
            AgentConfig(agent_type=SimAgent, agent_args={"env_name": env.name}, weight=env.weight)
            for env in envs
        ],
        pgt_rebalance=rebalance,
    )
    multi_task.parallel_generation_tasks = cfg.gate_capacity(submission)
    request = GroupedRolloutRequest(
        num_groups=cfg.num_groups,
        rollouts_per_group=cfg.rollouts_per_group,
        inference_interface=iface,
        streaming=cfg.streaming,
        submission_granularity=submission,
        consumption_granularity=consumption,
    )
    target_groups = cfg.num_batches * cfg.num_groups if cfg.streaming else cfg.num_groups

    groups_per_env = {env.name: 0 for env in envs}
    pgt_trajectory = {env.name: [] for env in envs}
    ema_per_env = {env.name: None for env in envs}
    batch_done_at = []
    consumed = 0
    agen = multi_task.get_grouped_rollouts(request)
    t0 = time.monotonic()
    try:
        async for group in agen:
            consumed += 1
            groups_per_env[group[0].env_id] += 1
            if consumed % cfg.num_groups == 0:
                batch_done_at.append(time.monotonic() - t0)
                for agent in multi_task.agents:
                    pipeline = getattr(agent, "_active_pipeline", None)
                    if pipeline is not None:
                        pgt_trajectory[agent.env_id].append(pipeline.gate.capacity)
                        ema_per_env[agent.env_id] = pipeline.engine_dwell_ema
                if consumed >= target_groups:
                    break
                if cfg.train_time:
                    if cfg.colocated:
                        iface.pause()
                    try:
                        await asyncio.sleep(cfg.train_time)
                    finally:
                        if cfg.colocated:
                            iface.resume()
                iface.policy_version += 1
    finally:
        await agen.aclose()
    total_wall = time.monotonic() - t0

    return MultiEnvResult(
        mode=f"{submission}/{consumption}",
        rebalanced=rebalance is not None,
        total_wall=total_wall,
        num_groups=consumed,
        groups_per_env=groups_per_env,
        batch_done_at=batch_done_at,
        pgt_trajectory=pgt_trajectory,
        ema_per_env=ema_per_env,
        engine_utilization=(
            iface.busy_seconds / (cfg.engine_slots * (total_wall - iface.paused_seconds))
            if cfg.engine_slots
            else None
        ),
        max_active_requests=iface.max_active_requests,
    )


def print_multi_env_report(result: MultiEnvResult) -> None:
    kind = "rebalanced" if result.rebalanced else "static split"
    print(f"\n=== mode {result.mode} multi-env ({kind}) ===")
    print(f"  wall time: {result.total_wall:.3f}s  groups: {result.num_groups}")
    mix = "  ".join(f"{env}={count}" for env, count in result.groups_per_env.items())
    print(f"  data mix (groups per env): {mix}")
    arrivals = "  ".join(f"b{i}={t:.2f}s" for i, t in enumerate(result.batch_done_at))
    print(f"  batch completion (since start): {arrivals}")
    for env, capacities in result.pgt_trajectory.items():
        trail = " ".join(str(capacity) for capacity in capacities)
        ema = result.ema_per_env[env]
        ema_desc = f"{ema * 1000:.1f}ms" if ema is not None else "-"
        print(f"  {env}: gate capacity per batch: {trail}  (engine_dwell EMA {ema_desc})")
    if result.engine_utilization is not None:
        print(f"  engine utilization: {result.engine_utilization:.0%}")


def stats(samples):
    """Distribution summary matching the rl_utils dwell-metric convention."""
    if not samples:
        return None
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "n": len(samples),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p50": float(np.percentile(arr, 50)),
        "p99": float(np.percentile(arr, 99)),
    }


def _stats_line(name, samples):
    summary = stats(samples)
    if summary is None:
        return f"    {name:<22} (no samples)"
    return (
        f"    {name:<22} n={summary['n']:<5d}"
        f" mean={summary['mean'] * 1000:8.2f}ms"
        f" min={summary['min'] * 1000:8.2f}ms"
        f" p50={summary['p50'] * 1000:8.2f}ms"
        f" p99={summary['p99'] * 1000:8.2f}ms"
        f" max={summary['max'] * 1000:8.2f}ms"
    )


def print_mode_report(result: ModeResult) -> None:
    counters = result.counters
    gate = result.gate
    print(f"\n=== mode {result.mode} (submission/consumption) ===")
    print(f"  get_grouped_rollouts wall time: {result.total_wall:.3f}s")
    print(
        f"  groups yielded: {result.num_groups}  rollouts: {result.num_rollouts}  "
        f"max concurrent inference calls: {result.max_active_requests}"
    )
    print(
        f"  pipeline counters: prepared={counters['prepared']} inferred={counters['inferred']} "
        f"assembled={counters['assembled']} yielded={counters['yielded']}"
    )
    print(
        f"  gate: capacity={gate['capacity']} held_at_end={gate['held_at_end']} "
        f"prepare_blocked={gate['prepare_blocked_seconds']:.3f}s "
        f"acquires={gate['acquire_calls']} releases={gate['release_calls']}"
    )
    arrivals = "  ".join(f"b{batch_id}={t:.2f}s" for batch_id, t in result.batch_done_at)
    print(f"  batch completion (since start): {arrivals}")
    if result.readahead_batches:
        trail = "  ".join(f"{r:.1f}" for r in result.readahead_batches)
        print(f"  readahead at each ship (batches, staleness proxy): {trail}")
    if result.batch_mean_lag:
        trail = "  ".join(f"{lag:.1f}" for lag in result.batch_mean_lag)
        print(f"  mean rollout lag per iteration (policy versions): {trail}")
    if result.batch_inference_steps:
        trail = "  ".join(
            f"b{batch_id}={steps:.0f}/{max_len:.0f}({steps / max_len:.1f}x)"
            for (batch_id, _), steps, max_len in zip(
                result.batch_done_at, result.batch_inference_steps, result.batch_max_seq_len
            )
        )
        total_steps = sum(result.batch_inference_steps)
        total_max = sum(result.batch_max_seq_len)
        print(
            f"  inference steps / batch max seq len (tokens): {trail}  "
            f"total={total_steps:.0f}/{total_max:.0f}({total_steps / total_max:.1f}x)"
        )
    if result.engine_utilization is not None:
        print(
            f"  engine: {result.engine_slots} slots, "
            f"utilization {result.engine_utilization:.0%} (idle slot time = white gaps)"
        )
    if result.engine_paused_seconds:
        print(f"  engine paused for colocated train steps: {result.engine_paused_seconds:.3f}s")
    print("  timings:")
    print(_stats_line("generation latency", result.sampled_latencies))
    print(_stats_line("rollout wall time", result.rollout_walls))
    for name, samples in result.dwell.items():
        print(_stats_line(name, samples))


def print_summary(results: list[ModeResult]) -> None:
    print("\n=== summary ===")
    header = (
        f"{'mode':<6} {'wall_s':>8} {'groups':>7} {'rollouts':>9} "
        f"{'rollout_wall_mean_ms':>21} {'rollout_wall_p99_ms':>20} "
        f"{'gen_lat_mean_ms':>16} {'max_conc':>9} {'util':>6} {'gate_blocked_s':>15}"
    )
    print(header)
    for result in results:
        walls = stats(result.rollout_walls)
        lats = stats(result.sampled_latencies)
        util = f"{result.engine_utilization:.0%}" if result.engine_utilization is not None else "-"
        print(
            f"{result.mode:<6} {result.total_wall:>8.3f} {result.num_groups:>7d} "
            f"{result.num_rollouts:>9d} {walls['mean'] * 1000:>21.2f} "
            f"{walls['p99'] * 1000:>20.2f} {lats['mean'] * 1000:>16.2f} "
            f"{result.max_active_requests:>9d} {util:>6} "
            f"{result.gate['prepare_blocked_seconds']:>15.3f}"
        )


# ---------------------------------------------------------------------------
# HTML report
#
# Self-contained inline-SVG charts, no plotting dependencies. Mode identity
# keeps a fixed categorical color across every chart; the pipeline-stage
# breakdown uses an ordinal single-hue ramp (stages are pipeline-ordered).
# Palettes validated with the dataviz six-checks script (light + dark).
# ---------------------------------------------------------------------------

_PLOT_W = 640
_STAGES = [
    ("infer_queue_dwell", "infer queue"),
    ("engine_dwell", "engine"),
    ("assemble_queue_dwell", "assemble queue"),
    ("output_queue_dwell", "output queue"),
]


def _fmt_ms(seconds):
    ms = seconds * 1000
    return f"{ms:,.2f} ms" if ms < 10 else f"{ms:,.1f} ms"


def _nice_ticks(vmax, n=5):
    """Round tick positions covering [0, vmax]."""
    if vmax <= 0:
        return [0.0]
    raw = vmax / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    return [i * step for i in range(int(math.ceil(vmax / step)) + 1)]


def _hbar_path(x, y, w, h, r=4):
    """Horizontal bar anchored flat at the left baseline, rounded data end."""
    r = min(r, w / 2, h / 2)
    return (
        f"M{x:.1f},{y:.1f} H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} "
        f"V{y + h - r:.1f} Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} "
        f"H{x:.1f} Z"
    )


def _vbar_path(x, y, w, h, r=4):
    """Vertical bar anchored flat at the bottom baseline, rounded top."""
    r = min(r, w / 2, h / 2)
    return (
        f"M{x:.1f},{y + h:.1f} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
        f"H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} "
        f"V{y + h:.1f} Z"
    )


def _legend(entries):
    chips = "".join(
        f'<span class="chip"><span class="swatch" style="background:{color}"></span>'
        f"{html.escape(label)}</span>"
        for label, color in entries
    )
    return f'<div class="legend">{chips}</div>'


def _svg_wall_time_chart(results):
    """Total get_grouped_rollouts wall time per mode: horizontal bars."""
    row_h, bar_h, mleft, mtop = 32, 22, 64, 8
    height = mtop + row_h * len(results) + 28
    vmax = max(result.total_wall for result in results)
    ticks = _nice_ticks(vmax)
    scale = _PLOT_W / ticks[-1]
    parts = []
    axis_y = mtop + row_h * len(results)
    for tick in ticks:
        x = mleft + tick * scale
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{mtop}" x2="{x:.1f}" y2="{axis_y}"/>')
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{axis_y + 16}" text-anchor="middle">{tick:g}s</text>'
        )
    for i, result in enumerate(results):
        y = mtop + i * row_h + (row_h - bar_h) / 2
        w = result.total_wall * scale
        tip = f"{result.mode}: {result.total_wall:.3f}s total for {result.num_rollouts} rollouts"
        parts.append(
            f'<text class="label" x="{mleft - 8}" y="{y + bar_h / 2 + 4}" text-anchor="end">{result.mode}</text>'
        )
        parts.append(
            f'<path d="{_hbar_path(mleft, y, w, bar_h)}" fill="var(--series-{i + 1})" data-tip="{tip}"/>'
        )
        parts.append(
            f'<text class="value" x="{mleft + w + 6}" y="{y + bar_h / 2 + 4}">{result.total_wall:.3f}s</text>'
        )
    parts.append(
        f'<line class="axis" x1="{mleft}" y1="{axis_y}" x2="{mleft + _PLOT_W}" y2="{axis_y}"/>'
    )
    return f'<svg viewBox="0 0 {mleft + _PLOT_W + 60} {height}">{"".join(parts)}</svg>'


def _svg_ecdf_chart(results):
    """Per-rollout prepare-to-yield wall time, cumulative distribution per mode."""
    mleft, mtop, plot_h = 48, 8, 220
    height = mtop + plot_h + 32
    vmax = max(max(result.rollout_walls) for result in results)
    ticks = _nice_ticks(vmax * 1000)
    scale = _PLOT_W / (ticks[-1] / 1000)
    parts = []
    axis_y = mtop + plot_h
    for tick in ticks:
        x = mleft + (tick / 1000) * scale
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{mtop}" x2="{x:.1f}" y2="{axis_y}"/>')
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{axis_y + 16}" text-anchor="middle">{tick:g}ms</text>'
        )
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = axis_y - frac * plot_h
        parts.append(
            f'<line class="grid" x1="{mleft}" y1="{y:.1f}" x2="{mleft + _PLOT_W}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{mleft - 6}" y="{y + 4:.1f}" text-anchor="end">{frac:.0%}</text>'
        )
    for i, result in enumerate(results):
        walls = sorted(result.rollout_walls)
        n = len(walls)
        pts = [(mleft + walls[j] * scale, axis_y - ((j + 1) / n) * plot_h) for j in range(n)]
        path = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        color = f"var(--series-{i + 1})"
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
        for pct in (50, 90, 99):
            value = walls[min(n - 1, int(round(pct / 100 * n)) - 1)]
            x, y = mleft + value * scale, axis_y - (pct / 100) * plot_h
            tip = f"{result.mode} p{pct}: {_fmt_ms(value)}"
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}" stroke="var(--surface-1)" stroke-width="2"/>'
            )
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="transparent" data-tip="{tip}"/>'
            )
    parts.append(
        f'<line class="axis" x1="{mleft}" y1="{axis_y}" x2="{mleft + _PLOT_W}" y2="{axis_y}"/>'
    )
    return f'<svg viewBox="0 0 {mleft + _PLOT_W + 24} {height}">{"".join(parts)}</svg>'


def _svg_stage_chart(results):
    """Mean time per pipeline stage, stacked horizontal bar per mode (ordinal ramp)."""
    row_h, bar_h, mleft, mtop = 32, 22, 64, 8
    height = mtop + row_h * len(results) + 28
    means = [
        [
            (sum(result.dwell[key]) / len(result.dwell[key]) if result.dwell[key] else 0.0)
            for key, _ in _STAGES
        ]
        for result in results
    ]
    vmax = max(sum(row) for row in means)
    ticks = _nice_ticks(vmax * 1000)
    scale = _PLOT_W / (ticks[-1] / 1000)
    parts = []
    axis_y = mtop + row_h * len(results)
    for tick in ticks:
        x = mleft + (tick / 1000) * scale
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{mtop}" x2="{x:.1f}" y2="{axis_y}"/>')
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{axis_y + 16}" text-anchor="middle">{tick:g}ms</text>'
        )
    for i, (result, row) in enumerate(zip(results, means)):
        y = mtop + i * row_h + (row_h - bar_h) / 2
        parts.append(
            f'<text class="label" x="{mleft - 8}" y="{y + bar_h / 2 + 4}" text-anchor="end">{result.mode}</text>'
        )
        x = float(mleft)
        breakdown = " · ".join(
            f"{label} {_fmt_ms(value)}" for (_, label), value in zip(_STAGES, row)
        )
        for j, value in enumerate(row):
            w = value * scale
            if w >= 1:  # sub-pixel stages are in the tooltip and table, not drawn
                parts.append(
                    f'<rect x="{x:.1f}" y="{y}" width="{max(w - 2, 1):.1f}" height="{bar_h}" '
                    f'fill="var(--ramp-{j + 1})" data-tip="{html.escape(f"{result.mode} — {breakdown}")}"/>'
                )
            x += w
        parts.append(
            f'<text class="value" x="{x + 6:.1f}" y="{y + bar_h / 2 + 4}">{_fmt_ms(sum(row))}</text>'
        )
    parts.append(
        f'<line class="axis" x1="{mleft}" y1="{axis_y}" x2="{mleft + _PLOT_W}" y2="{axis_y}"/>'
    )
    return f'<svg viewBox="0 0 {mleft + _PLOT_W + 70} {height}">{"".join(parts)}</svg>'


def _svg_latency_histogram(result, cfg):
    """Sampled generation latencies for one mode: should read uniform."""
    mleft, mtop, plot_h, bins = 48, 14, 180, 12
    height = mtop + plot_h + 32
    samples = result.sampled_latencies
    lo, hi = cfg.latency_lo, cfg.latency_hi
    counts = [0] * bins
    span = hi - lo
    for value in samples:
        idx = min(bins - 1, int((value - lo) / span * bins)) if span else 0
        counts[max(0, idx)] += 1
    cmax = max(counts)
    ticks = _nice_ticks(cmax, n=4)
    yscale = plot_h / ticks[-1]
    parts = []
    axis_y = mtop + plot_h
    for tick in ticks:
        y = axis_y - tick * yscale
        parts.append(
            f'<line class="grid" x1="{mleft}" y1="{y:.1f}" x2="{mleft + _PLOT_W}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{mleft - 6}" y="{y + 4:.1f}" text-anchor="end">{tick:g}</text>'
        )
    bin_w = _PLOT_W / bins
    for j, count in enumerate(counts):
        x = mleft + j * bin_w
        h = count * yscale
        b_lo, b_hi = lo + j * (hi - lo) / bins, lo + (j + 1) * (hi - lo) / bins
        tip = f"{_fmt_ms(b_lo)}–{_fmt_ms(b_hi)}: {count} calls"
        parts.append(
            f'<path d="{_vbar_path(x + 1, axis_y - h, bin_w - 2, h)}" fill="var(--seq)" data-tip="{tip}"/>'
        )
    expected_y = axis_y - (len(samples) / bins) * yscale
    parts.append(
        f'<line x1="{mleft}" y1="{expected_y:.1f}" x2="{mleft + _PLOT_W}" y2="{expected_y:.1f}" '
        'stroke="var(--ink-muted)" stroke-width="1.5" stroke-dasharray="5 4"/>'
    )
    parts.append(
        f'<text class="tick" x="{mleft + 6}" y="{expected_y - 5:.1f}" text-anchor="start">expected if uniform</text>'
    )
    for frac, label in ((0, _fmt_ms(lo)), (0.5, _fmt_ms((lo + hi) / 2)), (1, _fmt_ms(hi))):
        x = mleft + frac * _PLOT_W
        anchor = "start" if frac == 0 else ("end" if frac == 1 else "middle")
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{axis_y + 16}" text-anchor="{anchor}">{label}</text>'
        )
    parts.append(
        f'<line class="axis" x1="{mleft}" y1="{axis_y}" x2="{mleft + _PLOT_W}" y2="{axis_y}"/>'
    )
    return f'<svg viewBox="0 0 {mleft + _PLOT_W + 24} {height}">{"".join(parts)}</svg>'


def _svg_batch_timeline(results):
    """Cumulative completed batches over time, step line per mode."""
    mleft, mtop, plot_h = 48, 8, 200
    height = mtop + plot_h + 32
    tmax = max(t for result in results for _, t in result.batch_done_at)
    num_batches = max(len(result.batch_done_at) for result in results)
    ticks = _nice_ticks(tmax)
    scale = _PLOT_W / ticks[-1]
    parts = []
    axis_y = mtop + plot_h
    for tick in ticks:
        x = mleft + tick * scale
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{mtop}" x2="{x:.1f}" y2="{axis_y}"/>')
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{axis_y + 16}" text-anchor="middle">{tick:g}s</text>'
        )
    for count in range(num_batches + 1):
        y = axis_y - count / num_batches * plot_h
        parts.append(
            f'<line class="grid" x1="{mleft}" y1="{y:.1f}" x2="{mleft + _PLOT_W}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{mleft - 6}" y="{y + 4:.1f}" text-anchor="end">{count}</text>'
        )
    for i, result in enumerate(results):
        color = f"var(--series-{i + 1})"
        path = [f"M{mleft},{axis_y}"]
        for count, (batch_id, t) in enumerate(result.batch_done_at, start=1):
            x = mleft + t * scale
            y_prev = axis_y - (count - 1) / num_batches * plot_h
            y = axis_y - count / num_batches * plot_h
            path.append(f"H{x:.1f} V{y:.1f}")
        parts.append(f'<path d="{" ".join(path)}" fill="none" stroke="{color}" stroke-width="2"/>')
        for count, (batch_id, t) in enumerate(result.batch_done_at, start=1):
            x = mleft + t * scale
            y = axis_y - count / num_batches * plot_h
            tip = f"{result.mode} batch {batch_id} done at {t:.2f}s"
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}" '
                'stroke="var(--surface-1)" stroke-width="2"/>'
            )
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="transparent" data-tip="{tip}"/>'
            )
    parts.append(
        f'<line class="axis" x1="{mleft}" y1="{axis_y}" x2="{mleft + _PLOT_W}" y2="{axis_y}"/>'
    )
    return f'<svg viewBox="0 0 {mleft + _PLOT_W + 24} {height}">{"".join(parts)}</svg>'


def _svg_utilization_chart(results):
    """Engine utilization per mode: horizontal bars, 0-100% axis."""
    row_h, bar_h, mleft, mtop = 32, 22, 110, 8
    height = mtop + row_h * len(results) + 28
    scale = _PLOT_W / 1.0
    parts = []
    axis_y = mtop + row_h * len(results)
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        x = mleft + frac * scale
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{mtop}" x2="{x:.1f}" y2="{axis_y}"/>')
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{axis_y + 16}" text-anchor="middle">{frac:.0%}</text>'
        )
    for i, result in enumerate(results):
        util = result.engine_utilization or 0.0
        y = mtop + i * row_h + (row_h - bar_h) / 2
        w = util * scale
        tip = f"{result.mode}: {util:.1%} of {result.engine_slots} slots busy over the run"
        parts.append(
            f'<text class="label" x="{mleft - 8}" y="{y + bar_h / 2 + 4}" text-anchor="end">'
            f"{html.escape(result.mode)}</text>"
        )
        parts.append(
            f'<path d="{_hbar_path(mleft, y, w, bar_h)}" fill="var(--series-{i + 1})" '
            f'data-tip="{html.escape(tip)}"/>'
        )
        parts.append(
            f'<text class="value" x="{mleft + w + 6}" y="{y + bar_h / 2 + 4}">{util:.0%}</text>'
        )
    parts.append(
        f'<line class="axis" x1="{mleft}" y1="{axis_y}" x2="{mleft + _PLOT_W}" y2="{axis_y}"/>'
    )
    return f'<svg viewBox="0 0 {mleft + _PLOT_W + 60} {height}">{"".join(parts)}</svg>'


def _svg_readahead_chart(results):
    """Readahead (staleness proxy, in batches) at each batch completion, per mode."""
    return _svg_per_batch_line_chart(
        results, "readahead_batches", "{mode}: {value:.1f} batches ahead when batch {batch} shipped"
    )


def _svg_lag_chart(results):
    """Mean policy lag of each consumed batch's rollouts, per mode."""
    return _svg_per_batch_line_chart(
        results,
        "batch_mean_lag",
        "{mode}: batch {batch} trained on rollouts {value:.1f} versions old on average",
    )


def _svg_inference_steps_chart(results):
    """Inference steps per training step, relative to each batch's max seq len."""
    return _svg_per_batch_line_chart(
        results,
        "batch_step_ratio",
        "{mode}: batch {batch} took {value:.1f}x its max seq len in engine decode steps",
        ref_line=(1.0, "fully parallel bound (1.0x)"),
    )


def _svg_per_batch_line_chart(results, values_attr, tip_fmt, ref_line=None):
    """One value per completed batch over time, line + dots per mode.

    `ref_line` is an optional (value, label) pair drawn as a dashed horizontal
    reference (e.g. a theoretical bound).
    """
    mleft, mtop, plot_h = 48, 8, 200
    height = mtop + plot_h + 32
    tmax = max(t for result in results for _, t in result.batch_done_at)
    vmax = max(max(getattr(result, values_attr), default=0.0) for result in results)
    if ref_line is not None:
        vmax = max(vmax, ref_line[0])
    xticks = _nice_ticks(tmax)
    yticks = _nice_ticks(max(vmax, 1.0), n=4)
    xscale = _PLOT_W / xticks[-1]
    yscale = plot_h / yticks[-1]
    parts = []
    axis_y = mtop + plot_h
    for tick in xticks:
        x = mleft + tick * xscale
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{mtop}" x2="{x:.1f}" y2="{axis_y}"/>')
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{axis_y + 16}" text-anchor="middle">{tick:g}s</text>'
        )
    for tick in yticks:
        y = axis_y - tick * yscale
        parts.append(
            f'<line class="grid" x1="{mleft}" y1="{y:.1f}" x2="{mleft + _PLOT_W}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{mleft - 6}" y="{y + 4:.1f}" text-anchor="end">{tick:g}</text>'
        )
    if ref_line is not None:
        ref_value, ref_label = ref_line
        y = axis_y - ref_value * yscale
        parts.append(
            f'<line x1="{mleft}" y1="{y:.1f}" x2="{mleft + _PLOT_W}" y2="{y:.1f}" '
            'stroke="var(--ink-muted)" stroke-width="1.5" stroke-dasharray="5 4"/>'
        )
        parts.append(
            f'<text class="tick" x="{mleft + 6}" y="{y - 5:.1f}" text-anchor="start">'
            f"{html.escape(ref_label)}</text>"
        )
    for i, result in enumerate(results):
        color = f"var(--series-{i + 1})"
        values = getattr(result, values_attr)
        pts = [
            (mleft + t * xscale, axis_y - value * yscale)
            for (_, t), value in zip(result.batch_done_at, values)
        ]
        if len(pts) > 1:
            path = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
        for ((batch_id, t), value), (x, y) in zip(zip(result.batch_done_at, values), pts):
            tip = tip_fmt.format(mode=result.mode, batch=batch_id, value=value)
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}" '
                'stroke="var(--surface-1)" stroke-width="2"/>'
            )
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="transparent" '
                f'data-tip="{html.escape(tip)}"/>'
            )
    parts.append(
        f'<line class="axis" x1="{mleft}" y1="{axis_y}" x2="{mleft + _PLOT_W}" y2="{axis_y}"/>'
    )
    return f'<svg viewBox="0 0 {mleft + _PLOT_W + 24} {height}">{"".join(parts)}</svg>'


def _summary_table(results):
    head = (
        "<tr><th>mode</th><th>wall (s)</th><th>groups</th><th>rollouts</th>"
        "<th>rollout wall mean</th><th>p50</th><th>p99</th>"
        "<th>gen latency mean</th><th>infer steps ×</th>"
        "<th>max concurrency</th><th>gate blocked (s)</th></tr>"
    )
    rows = []
    for i, result in enumerate(results):
        walls, lats = stats(result.rollout_walls), stats(result.sampled_latencies)
        steps_ratio = (
            f"{sum(result.batch_inference_steps) / sum(result.batch_max_seq_len):.1f}×"
            if result.batch_max_seq_len
            else "-"
        )
        rows.append(
            f'<tr><td><span class="swatch" style="background:var(--series-{i + 1})"></span>{result.mode}</td>'
            f"<td>{result.total_wall:.3f}</td><td>{result.num_groups}</td><td>{result.num_rollouts}</td>"
            f"<td>{_fmt_ms(walls['mean'])}</td><td>{_fmt_ms(walls['p50'])}</td><td>{_fmt_ms(walls['p99'])}</td>"
            f"<td>{_fmt_ms(lats['mean'])}</td><td>{steps_ratio}</td><td>{result.max_active_requests}</td>"
            f"<td>{result.gate['prepare_blocked_seconds']:.3f}</td></tr>"
        )
    return f'<table>{head}{"".join(rows)}</table>'


def write_html_report(results, cfg: SimConfig, path, subtitle=None):
    """Write a self-contained HTML timing report for the simulated modes.

    `subtitle` overrides the default config line, for reports whose results
    were produced with per-mode configs rather than a single shared one.
    """
    mode_entries = [(result.mode, f"var(--series-{i + 1})") for i, result in enumerate(results)]
    stage_entries = [(label, f"var(--ramp-{j + 1})") for j, (_, label) in enumerate(_STAGES)]
    tiles = [
        ("rollouts / mode", f"{cfg.num_batches * cfg.num_groups * cfg.rollouts_per_group}"),
        ("generation latency", f"{cfg.latency_lo * 1000:g}–{cfg.latency_hi * 1000:g} ms uniform"),
        (
            "gate capacity",
            (
                f"{cfg.parallel_generation_tasks} / submission unit"
                if cfg.generation_lag is None
                else f"lag {cfg.generation_lag} (production formula)"
            ),
        ),
        ("streaming", "on" if cfg.streaming else "off"),
    ]
    tile_html = "".join(
        f'<div class="tile"><div class="tile-label">{label}</div><div class="tile-value">{value}</div></div>'
        for label, value in tiles
    )
    pause_kind = "colocated trainer pause (engine frozen)" if cfg.colocated else "trainer pause"
    train_desc = f", {cfg.train_time:g}s {pause_kind} per batch" if cfg.train_time else ""
    config_line = subtitle or (
        f"{cfg.num_batches} batch(es) × {cfg.num_groups} groups × {cfg.rollouts_per_group} rollouts per mode, "
        f"seed {cfg.seed}{train_desc} (all modes consume the identical latency sequence)"
    )
    utilization_card = ""
    if any(result.engine_utilization is not None for result in results):
        utilization_card = f"""
<div class="card">
  <h2>Engine utilization (busy slot-time / total slot-time)</h2>
  <p class="note">An idle slot is the deck's "white gap": a decode pass running below full batch size.</p>
  {_svg_utilization_chart(results)}
</div>
"""
    inference_steps_card = ""
    if any(result.batch_inference_steps for result in results):
        inference_steps_card = f"""
<div class="card">
  <h2>Inference steps per training step (× batch max seq len)</h2>
  <p class="note">Engine decode steps the trainer waited through for each batch, relative to that batch's
     longest rollout. 1.0× is the fully-parallel bound — the trainer waited only for the longest sequence.
     Higher means oversubscribed or serialized decoding; near 0× means the batch was consumed from readahead.</p>
  {_legend(mode_entries)}
  {_svg_inference_steps_chart(results)}
</div>
"""
    readahead_card = ""
    if any(result.readahead_batches for result in results):
        readahead_card = f"""
<div class="card">
  <h2>Mean rollout lag per iteration (policy versions)</h2>
  <p class="note">Average age of the rollouts each batch trained on: policy version at training minus the
     version stamped when each rollout was generated — the simulator twin of W&amp;B's mean_policy_staleness.
     Flat = the lag knob holds; climbing = generation decoupled from training.</p>
  {_legend(mode_entries)}
  {_svg_lag_chart(results)}
</div>

<div class="card">
  <h2>Readahead when each batch ships (staleness proxy)</h2>
  <p class="note">Batches of rollouts already prepared beyond what the trainer has consumed — the leading
     indicator of the lag chart above (today's readahead is a future batch's lag).</p>
  {_legend(mode_entries)}
  {_svg_readahead_chart(results)}
</div>
"""
    document = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>get_grouped_rollouts granularity timings</title>
<style>
  :root {{
    --page: #f9f9f7; --surface-1: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e;
    --ink-muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
    --series-1: #2a78d6; --series-2: #1baf7a; --series-3: #eda100; --series-4: #008300; --series-5: #4a3aa7;
    --ramp-1: #86b6ef; --ramp-2: #5598e7; --ramp-3: #2a78d6; --ramp-4: #184f95; --seq: #2a78d6;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --page: #0d0d0d; --surface-1: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
      --ink-muted: #898781; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
      --series-1: #3987e5; --series-2: #199e70; --series-3: #c98500; --series-4: #008300; --series-5: #9085e9;
      --ramp-1: #6da7ec; --ramp-2: #3987e5; --ramp-3: #256abf; --ramp-4: #184f95; --seq: #3987e5;
    }}
  }}
  body {{ background: var(--page); color: var(--ink); margin: 24px auto; max-width: 860px;
         font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; padding: 0 16px; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  h2 {{ font-size: 15px; margin: 0 0 2px; }}
  .sub {{ color: var(--ink-2); margin: 0 0 12px; }}
  .note {{ color: var(--ink-2); font-size: 12.5px; margin: 2px 0 10px; }}
  .card {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
           padding: 16px 18px; margin: 16px 0; }}
  .tiles {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0; }}
  .tile {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
           padding: 10px 16px; flex: 1 1 140px; }}
  .tile-label {{ color: var(--ink-2); font-size: 12px; }}
  .tile-value {{ font-size: 20px; font-weight: 600; margin-top: 2px; }}
  svg {{ width: 100%; height: auto; display: block; }}
  svg text {{ font: 11px system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--ink-muted); }}
  svg .label {{ fill: var(--ink-2); font-weight: 500; }}
  svg .value {{ fill: var(--ink-2); font-variant-numeric: tabular-nums; }}
  svg .tick {{ font-variant-numeric: tabular-nums; }}
  svg .grid {{ stroke: var(--grid); stroke-width: 1; }}
  svg .axis {{ stroke: var(--axis); stroke-width: 1; }}
  .legend {{ display: flex; gap: 14px; flex-wrap: wrap; margin: 6px 0 10px; color: var(--ink-2);
             font-size: 12.5px; }}
  .chip {{ display: inline-flex; align-items: center; gap: 6px; }}
  .swatch {{ width: 10px; height: 10px; border-radius: 3px; display: inline-block; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--grid);
            font-variant-numeric: tabular-nums; }}
  th {{ color: var(--ink-2); font-weight: 600; }}
  th:first-child, td:first-child {{ text-align: left; }}
  td .swatch {{ margin-right: 6px; }}
  #tip {{ position: fixed; display: none; background: var(--ink); color: var(--page);
          padding: 5px 9px; border-radius: 6px; font-size: 12px; pointer-events: none;
          z-index: 10; max-width: 340px; }}
</style></head><body>
<h1>get_grouped_rollouts granularity timings</h1>
<p class="sub">{html.escape(config_line)}</p>
<div class="tiles">{tile_html}</div>

<div class="card">
  <h2>Total wall time per mode</h2>
  <p class="note">Same workload and latency draws per mode; only the submission/consumption granularity changes.
     Gate capacity is measured in units of the submission granularity, so B submission admits the most concurrent work.</p>
  {_svg_wall_time_chart(results)}
</div>

<div class="card">
  <h2>Batches completed over time</h2>
  <p class="note">Each dot is a training batch becoming available to the trainer. Steeper is better;
     a long flat start followed by a burst means the mode front-loads generation across future batches.</p>
  {_legend(mode_entries)}
  {_svg_batch_timeline(results)}
</div>
{utilization_card}{inference_steps_card}{readahead_card}
<div class="card">
  <h2>Per-rollout wall time (prepare → yield), cumulative distribution</h2>
  <p class="note">Further left is faster. Dots mark p50 / p90 / p99 — hover them for values.</p>
  {_legend(mode_entries)}
  {_svg_ecdf_chart(results)}
</div>

<div class="card">
  <h2>Where a rollout spends its time (mean per stage)</h2>
  <p class="note">Stages in pipeline order. Queue dwells are typically sub-millisecond — segments thinner than
     a pixel are omitted from the drawing but included in the hover tooltip and table.</p>
  {_legend(stage_entries)}
  {_svg_stage_chart(results)}
</div>

<div class="card">
  <h2>Sampled generation latency ({html.escape(results[0].mode)}, n={len(results[0].sampled_latencies)})</h2>
  <p class="note">Sanity check on the mocked inference backend: draws should be uniform between the configured bounds.</p>
  {_svg_latency_histogram(results[0], cfg)}
</div>

<div class="card">
  <h2>Summary</h2>
  {_summary_table(results)}
</div>

<div id="tip"></div>
<script>
  const tip = document.getElementById('tip');
  for (const el of document.querySelectorAll('[data-tip]')) {{
    el.addEventListener('pointerenter', () => {{ tip.textContent = el.dataset.tip; tip.style.display = 'block'; }});
    el.addEventListener('pointermove', (e) => {{
      tip.style.left = Math.min(e.clientX + 12, window.innerWidth - 360) + 'px';
      tip.style.top = (e.clientY + 14) + 'px';
    }});
    el.addEventListener('pointerleave', () => {{ tip.style.display = 'none'; }});
  }}
</script>
</body></html>
"""
    Path(path).write_text(document)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--num-groups", type=int, default=8, help="groups per batch")
    parser.add_argument(
        "--num-batches", type=int, default=4, help="batches to consume (streaming mode only)"
    )
    parser.add_argument("--rollouts-per-group", type=int, default=4)
    parser.add_argument(
        "--parallel-generation-tasks",
        type=int,
        default=4,
        help=(
            "gate capacity, measured in units of the submission granularity: the same value "
            "allows N batches (B), N groups (G), or N rollouts (R) in flight at once"
        ),
    )
    parser.add_argument(
        "--latency-lo", type=float, default=0.01, help="uniform latency lower bound (s)"
    )
    parser.add_argument(
        "--latency-hi", type=float, default=0.05, help="uniform latency upper bound (s)"
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed, shared across modes")
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="use non-streaming requests (single batch; --num-batches is ignored)",
    )
    parser.add_argument(
        "--modes",
        default=",".join(f"{sub}/{cons}" for sub, cons in MODES),
        help="comma-separated submission/consumption pairs to run",
    )
    parser.add_argument(
        "--max-readahead-batches",
        type=int,
        default=None,
        metavar="W",
        help=(
            "prototype lag bound: prepare may not run more than W batches ahead of "
            "consumption (unset = current production behavior)"
        ),
    )
    parser.add_argument(
        "--engine-slots",
        type=int,
        default=None,
        metavar="N",
        help=(
            "fixed engine slot count (continuous-batching rows); submissions beyond N queue. "
            "Unset = unlimited slots, where slot-idle gaps cannot exist and R submission "
            "shows only its costs"
        ),
    )
    parser.add_argument(
        "--train-time",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help=(
            "simulated trainer pause after each completed batch; generation keeps running "
            "during the pause to the extent the submission granularity's gate allows"
        ),
    )
    parser.add_argument(
        "--colocated",
        action="store_true",
        help=(
            "trainer and engine share the GPUs: the engine pauses during each "
            "--train-time step and in-flight decodes freeze until the step ends "
            "(default: disaggregated, generation keeps running through the pause)"
        ),
    )
    parser.add_argument(
        "--generation-lag",
        type=int,
        default=None,
        metavar="LAG",
        help=(
            "derive gate capacity per mode from the production formula "
            "(get_rl_parallel_generation_tasks with --rl-generation-lag LAG) instead of "
            "--parallel-generation-tasks; lag 0 admits one training step of in-flight work"
        ),
    )
    parser.add_argument(
        "--seconds-per-token",
        type=float,
        default=0.001,
        metavar="S",
        help=(
            "token clock: a rollout whose mocked latency is D seconds counts as D/S "
            "generated tokens; engine-active time between batch completions / S is the "
            "per-training-step inference step count"
        ),
    )
    parser.add_argument(
        "--envs",
        default=None,
        metavar="NAME:LO:HI[:WEIGHT],...",
        help=(
            "run a multi-env simulation under a real WeightedMultiTask: comma-separated "
            "env specs, each name:latency_lo:latency_hi[:weight] "
            "(e.g. fast:0.01:0.03:1,slow:0.05:0.10:1). All envs share one engine."
        ),
    )
    parser.add_argument(
        "--rebalance",
        action="store_true",
        help=(
            "with --envs: also run each mode with dynamic pgt rebalancing "
            "(--rl-inference-rebalancing) and print an A/B comparison against the static split"
        ),
    )
    parser.add_argument(
        "--rebalance-interval",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="min seconds between rebalance checks (production default is 30s; 0 suits sim scale)",
    )
    parser.add_argument(
        "--rebalance-min-samples",
        type=int,
        default=4,
        metavar="N",
        help="per-env engine_dwell samples required before its EMA is trusted",
    )
    parser.add_argument(
        "--rebalance-max-step",
        type=float,
        default=0.25,
        metavar="FRACTION",
        help="max per-update allocation change as a fraction of the current value",
    )
    parser.add_argument(
        "--html",
        default=None,
        metavar="PATH",
        help="also write a self-contained HTML report with charts to PATH",
    )
    return parser.parse_args()


def parse_envs(spec: str) -> list[EnvSpec]:
    envs = []
    for entry in spec.split(","):
        parts = entry.strip().split(":")
        if len(parts) not in (3, 4):
            raise SystemExit(f"invalid env spec {entry!r}; expected name:lo:hi[:weight]")
        envs.append(
            EnvSpec(
                name=parts[0],
                lo=float(parts[1]),
                hi=float(parts[2]),
                weight=float(parts[3]) if len(parts) == 4 else 1.0,
            )
        )
    if len(envs) < 2:
        raise SystemExit("--envs needs at least two environments")
    return envs


def main():
    args = parse_args()

    valid_modes = {f"{sub}/{cons}" for sub, cons in MODES}
    modes = []
    for mode in args.modes.split(","):
        mode = mode.strip().upper()
        if mode not in valid_modes:
            raise SystemExit(f"invalid mode {mode!r}; valid modes: {sorted(valid_modes)}")
        modes.append(tuple(mode.split("/")))

    cfg = SimConfig(
        num_groups=args.num_groups,
        num_batches=1 if args.no_streaming else args.num_batches,
        rollouts_per_group=args.rollouts_per_group,
        parallel_generation_tasks=args.parallel_generation_tasks,
        latency_lo=args.latency_lo,
        latency_hi=args.latency_hi,
        seed=args.seed,
        streaming=not args.no_streaming,
        train_time=args.train_time,
        colocated=args.colocated,
        engine_slots=args.engine_slots,
        generation_lag=args.generation_lag,
        max_readahead_batches=args.max_readahead_batches,
        seconds_per_token=args.seconds_per_token,
    )
    if args.no_streaming and args.num_batches != 1:
        print("note: --no-streaming generates a single batch; --num-batches is ignored")
    if args.colocated and not args.train_time:
        print("note: --colocated has no effect without --train-time")
    total_rollouts = cfg.num_batches * cfg.num_groups * cfg.rollouts_per_group
    if cfg.generation_lag is None:
        capacity_desc = f"parallel_generation_tasks={cfg.parallel_generation_tasks}"
    else:
        per_mode = ", ".join(f"{sub}:{cfg.gate_capacity(sub)}" for sub in ("B", "G", "R"))
        capacity_desc = f"generation_lag={cfg.generation_lag} (gate capacity {per_mode})"
    print(
        f"simulating {cfg.num_batches} batch(es) x {cfg.num_groups} groups x "
        f"{cfg.rollouts_per_group} rollouts ({total_rollouts} rollouts/mode), "
        f"latency ~ U({cfg.latency_lo}s, {cfg.latency_hi}s) = "
        f"{cfg.latency_lo / cfg.seconds_per_token:g}-"
        f"{cfg.latency_hi / cfg.seconds_per_token:g} tokens, "
        f"{capacity_desc}, streaming={cfg.streaming}, "
        f"train_time={cfg.train_time}s, colocated={cfg.colocated}, seed={cfg.seed}"
    )

    if args.envs:
        envs = parse_envs(args.envs)
        rebalance_cfg = PgtRebalanceConfig(
            min_interval_s=args.rebalance_interval,
            min_samples_per_env=args.rebalance_min_samples,
            max_step_fraction=args.rebalance_max_step,
        )
        for submission, consumption in modes:
            # One event loop per run so runs are fully isolated.
            static = asyncio.run(run_multi_env_mode(submission, consumption, cfg, envs))
            print_multi_env_report(static)
            if args.rebalance:
                rebalanced = asyncio.run(
                    run_multi_env_mode(submission, consumption, cfg, envs, rebalance_cfg)
                )
                print_multi_env_report(rebalanced)
                print(
                    f"  A/B wall time ({static.mode}): static {static.total_wall:.3f}s vs "
                    f"rebalanced {rebalanced.total_wall:.3f}s "
                    f"({static.total_wall / rebalanced.total_wall:.2f}x speedup)"
                )
        return

    results = []
    for submission, consumption in modes:
        # One event loop per mode so modes are fully isolated.
        result = asyncio.run(run_mode(submission, consumption, cfg))
        print_mode_report(result)
        results.append(result)
    print_summary(results)
    if args.html:
        write_html_report(results, cfg, args.html)
        print(f"\nhtml report written to {args.html}")


if __name__ == "__main__":
    main()
