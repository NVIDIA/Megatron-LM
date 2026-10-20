# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from megatron.rl import callbacks as rl_callbacks
from megatron.rl import rl_profiling, rl_utils
from megatron.rl.callbacks import RLCheckpointCallback, RLLoggingCallback
from megatron.rl.rl_profiling import RL_LOGGABLE_TIMER_NAMES
from megatron.training.callbacks import CallbackContext
from megatron.training.config import RLConfig


@pytest.fixture
def env(monkeypatch):
    """Everything the logging callback reads from globals, and a stub for each sink it feeds."""
    env = SimpleNamespace(
        writer=MagicMock(),
        wandb_writer=MagicMock(),
        timers=MagicMock(),
        runtime_state=rl_utils.RLRuntimeState(),
        train_state=SimpleNamespace(iteration=9),
        profiles=[],
    )
    monkeypatch.setattr(
        rl_callbacks, "get_args", SimpleNamespace  # only handed on to log_rl_throughput_metrics
    )
    monkeypatch.setattr(rl_callbacks, "get_tensorboard_writer", lambda: env.writer)
    monkeypatch.setattr(rl_callbacks, "get_wandb_writer", lambda: env.wandb_writer)
    monkeypatch.setattr(rl_callbacks, "get_timers", lambda: env.timers)
    monkeypatch.setattr(rl_callbacks, "get_train_state", lambda: env.train_state)
    monkeypatch.setattr(rl_callbacks, "log_iteration_profile", lambda **kw: env.profiles.append(kw))
    monkeypatch.setattr(rl_utils, "get_rl_runtime_state", lambda: env.runtime_state)
    monkeypatch.setattr(rl_utils, "log_rl_throughput_metrics", lambda *a: " tokens/s: 10 |")
    monkeypatch.setattr(
        "megatron.rl.sequence_packing_utils.mpu.get_data_parallel_world_size", lambda: 4
    )
    monkeypatch.setattr("megatron.rl.sequence_packing_utils.get_num_microbatches", lambda: 3)
    return env


@pytest.mark.parametrize("profile", [False, True])
@pytest.mark.parametrize("packing", [False, True])
def test_logging_callback(env, packing, profile):
    """on_log extends the timers and the log line (packing fragment, throughput fragment) and
    emits the iteration profile when enabled; on_tensorboard_log adds the packing metrics and the
    GRPO collection index, all at TrainState's iteration."""
    cfg = RLConfig(
        grpo_prompts_per_step=4,
        grpo_group_size=3,
        grpo_iterations=2,
        rl_use_sequence_packing=packing,
        rl_profile=profile,
    )
    # 12 samples -> 2 batches x 2 epochs
    callback = RLLoggingCallback(cfg, micro_batch_size=2, global_batch_size=6, save_dir="/ckpt")
    env.runtime_state.consumed_train_bins = 7
    context = CallbackContext(
        model=None,
        timers_to_log=["forward-compute"],
        log_fragments=[],
        batch_size=6,
        elapsed_time_per_iteration=0.5,
        throughput=1.5,
    )

    callback.on_log(context)
    assert context.timers_to_log == ["forward-compute", *RL_LOGGABLE_TIMER_NAMES]
    packing_fragment = [" consumed bins:            7 |"] if packing else []
    assert context.log_fragments == [*packing_fragment, " tokens/s: 10 |"]
    profile_call = dict(
        iteration=9,
        timers=env.timers,
        elapsed_time_ms=500.0,
        throughput_tflops=1.5,
        global_batch_size=6,
        wandb_writer=env.wandb_writer,
        tb_writer=env.writer,
    )
    assert env.profiles == ([profile_call] if profile else [])

    callback.on_tensorboard_log(context)
    scalars = {c.args[0]: c.args[1] for c in env.writer.add_scalar.call_args_list}
    packing_metrics = {"bin-batch-size": 4 * 2 * 3, "consumed-bins": 7} if packing else {}
    assert scalars == {"grpo_collection_iteration": 9 // 4, **packing_metrics}
    assert all(c.args[1] == 9 for c in env.wandb_writer.log.call_args_list)


@pytest.mark.parametrize("profile", [False, True])
@pytest.mark.parametrize("packing", [False, True])
def test_logging_callback_lifecycle(env, monkeypatch, packing, profile):
    """on_train_start starts the profiler when enabled and on_train_end stops it;
    on_train_step_end advances the bin counter under packing (4 dp x 2 mbs x 3 microbatches)."""
    started, stopped = [], []
    monkeypatch.setattr(
        rl_callbacks,
        "maybe_initialize_rl_profiler",
        lambda cfg, save_dir, **kw: started.append((save_dir, kw)) or ("/p" if profile else None),
    )
    monkeypatch.setattr(rl_callbacks, "shutdown_rl_profiler", lambda: stopped.append(True))
    monkeypatch.setattr(rl_callbacks, "print_rank_0", lambda *_: None)
    callback = RLLoggingCallback(
        RLConfig(rl_use_sequence_packing=packing, rl_profile=profile),
        micro_batch_size=2,
        global_batch_size=6,
        save_dir="/ckpt",
    )
    env.runtime_state.consumed_train_bins = 7
    context = CallbackContext(model=None)

    callback.on_train_start(context)
    callback.on_train_step_end(context)
    callback.on_train_end(context)

    assert started == [("/ckpt", {"log_to_wandb": True, "log_to_tensorboard": True})]
    assert env.runtime_state.consumed_train_bins == (7 + 24 if packing else 7)
    assert stopped == ([True] if profile else [])


def test_maybe_initialize_rl_profiler(monkeypatch):
    calls = []
    monkeypatch.setattr(rl_profiling, "initialize_rl_profiler", lambda **kw: calls.append(kw))
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    start = lambda **kw: rl_profiling.maybe_initialize_rl_profiler(
        RLConfig(**kw), "/ckpt", log_to_wandb=False, log_to_tensorboard=True
    )

    assert start(rl_profile=False) is None and calls == []
    assert start(rl_profile=True) == "/ckpt/profiles"
    assert start(rl_profile=True, rl_profile_dir="/elsewhere") == "/elsewhere"
    assert [c["output_dir"] for c in calls] == ["/ckpt/profiles", "/elsewhere"]
    assert calls[0] == dict(
        output_dir="/ckpt/profiles",
        run_id="123",
        enabled=True,
        log_to_wandb=False,
        log_to_tensorboard=True,
    )


def test_checkpoint_callback(monkeypatch):
    """on_checkpoint_saved compacts the rollout bank at the durable iteration."""
    compacted = []
    monkeypatch.setattr(rl_utils, "maybe_compact_rollout_bank", compacted.append)
    RLCheckpointCallback().on_checkpoint_saved(CallbackContext(model=None, durable_iteration=40))
    assert compacted == [40]
