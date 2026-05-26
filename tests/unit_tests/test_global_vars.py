# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import sys
import types
from types import SimpleNamespace

import pytest

from megatron.training import global_vars


@pytest.fixture(autouse=True)
def _reset_global_vars():
    global_vars.destroy_global_vars()
    yield
    global_vars.destroy_global_vars()


def test_getters_require_initialized_globals():
    with pytest.raises(AssertionError, match="args is not initialized"):
        global_vars.get_args()
    with pytest.raises(AssertionError, match="tokenizer is not initialized"):
        global_vars.get_tokenizer()
    with pytest.raises(AssertionError, match="timers is not initialized"):
        global_vars.get_timers()
    with pytest.raises(AssertionError, match="energy monitor is not initialized"):
        global_vars.get_energy_monitor()


def test_set_args_and_destroy_global_vars():
    args = SimpleNamespace(rank=0)

    global_vars.set_args(args)
    assert global_vars.get_args() is args

    global_vars.destroy_global_vars()

    with pytest.raises(AssertionError, match="args is not initialized"):
        global_vars.get_args()


def test_rebuild_tokenizer_resets_existing_tokenizer(monkeypatch):
    built = []

    def fake_build_tokenizer(args):
        tokenizer = object()
        built.append((args, tokenizer))
        return tokenizer

    monkeypatch.setattr(global_vars, "build_tokenizer", fake_build_tokenizer)

    first_args = SimpleNamespace(name="first")
    second_args = SimpleNamespace(name="second")

    first = global_vars._build_tokenizer(first_args)
    second = global_vars.rebuild_tokenizer(second_args)

    assert first is built[0][1]
    assert second is built[1][1]
    assert global_vars.get_tokenizer() is second


def test_set_global_variables_initializes_expected_helpers(monkeypatch):
    calls = []
    args = SimpleNamespace(
        rank=0,
        rampup_batch_size=None,
        global_batch_size=8,
        micro_batch_size=2,
        data_parallel_size=4,
        decrease_batch_size_if_needed=False,
        enable_experimental=False,
        exit_signal_handler=False,
        exit_signal_handler_for_training=False,
        disable_jit_fuser=False,
    )

    monkeypatch.setattr(
        global_vars,
        "init_num_microbatches_calculator",
        lambda *items: calls.append(("microbatches", items)),
    )
    monkeypatch.setattr(global_vars, "_build_tokenizer", lambda item: calls.append(("tokenizer", item)))
    monkeypatch.setattr(global_vars, "_set_tensorboard_writer", lambda item: calls.append(("tensorboard", item)))
    monkeypatch.setattr(global_vars, "_set_wandb_writer", lambda item: calls.append(("wandb", item)))
    monkeypatch.setattr(global_vars, "_set_one_logger", lambda item: calls.append(("one_logger", item)))
    monkeypatch.setattr(global_vars, "_set_adlr_autoresume", lambda item: calls.append(("autoresume", item)))
    monkeypatch.setattr(global_vars, "_set_timers", lambda item: calls.append(("timers", item)))
    monkeypatch.setattr(global_vars, "_set_energy_monitor", lambda item: calls.append(("energy", item)))

    global_vars.set_global_variables(args)

    assert global_vars.get_args() is args
    assert [name for name, _ in calls] == [
        "microbatches",
        "tokenizer",
        "tensorboard",
        "wandb",
        "one_logger",
        "autoresume",
        "timers",
        "energy",
    ]


def test_set_global_variables_can_skip_tokenizer(monkeypatch):
    calls = []
    args = SimpleNamespace(
        rank=0,
        rampup_batch_size=None,
        global_batch_size=8,
        micro_batch_size=2,
        data_parallel_size=4,
        decrease_batch_size_if_needed=False,
        enable_experimental=False,
        exit_signal_handler=False,
        exit_signal_handler_for_training=False,
        disable_jit_fuser=False,
    )

    monkeypatch.setattr(global_vars, "init_num_microbatches_calculator", lambda *items: None)
    monkeypatch.setattr(global_vars, "_build_tokenizer", lambda item: calls.append("tokenizer"))
    monkeypatch.setattr(global_vars, "_set_tensorboard_writer", lambda item: None)
    monkeypatch.setattr(global_vars, "_set_wandb_writer", lambda item: None)
    monkeypatch.setattr(global_vars, "_set_one_logger", lambda item: None)
    monkeypatch.setattr(global_vars, "_set_adlr_autoresume", lambda item: None)
    monkeypatch.setattr(global_vars, "_set_timers", lambda item: None)
    monkeypatch.setattr(global_vars, "_set_energy_monitor", lambda item: None)

    global_vars.set_global_variables(args, build_tokenizer=False)

    assert calls == []


def test_unset_global_variables_clears_all_globals(monkeypatch):
    monkeypatch.setattr(global_vars, "unset_num_microbatches_calculator", lambda: None)
    global_vars._GLOBAL_ARGS = object()
    global_vars._GLOBAL_TOKENIZER = object()
    global_vars._GLOBAL_TENSORBOARD_WRITER = object()
    global_vars._GLOBAL_WANDB_WRITER = object()
    global_vars._GLOBAL_ONE_LOGGER = object()
    global_vars._GLOBAL_ADLR_AUTORESUME = object()
    global_vars._GLOBAL_TIMERS = object()
    global_vars._GLOBAL_ENERGY_MONITOR = object()
    global_vars._GLOBAL_SIGNAL_HANDLER = object()

    global_vars.unset_global_variables()

    assert global_vars.get_tensorboard_writer() is None
    assert global_vars.get_wandb_writer() is None
    assert global_vars.get_one_logger() is None
    assert global_vars.get_adlr_autoresume() is None
    with pytest.raises(AssertionError, match="args is not initialized"):
        global_vars.get_args()


def test_set_tensorboard_writer_success_and_non_writer_rank(monkeypatch, tmp_path):
    created = []

    class FakeSummaryWriter:
        def __init__(self, log_dir, max_queue):
            created.append((log_dir, max_queue))

    fake_tensorboard = types.ModuleType("torch.utils.tensorboard")
    fake_tensorboard.SummaryWriter = FakeSummaryWriter
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", fake_tensorboard)

    args = SimpleNamespace(
        tensorboard_dir=str(tmp_path),
        tensorboard_queue_size=17,
        rank=3,
        world_size=4,
    )
    global_vars._set_tensorboard_writer(args)

    assert created == [(str(tmp_path), 17)]
    assert isinstance(global_vars.get_tensorboard_writer(), FakeSummaryWriter)

    global_vars._GLOBAL_TENSORBOARD_WRITER = None
    args.rank = 0
    global_vars._set_tensorboard_writer(args)
    assert global_vars.get_tensorboard_writer() is None


def test_set_wandb_writer_builds_init_kwargs(monkeypatch, tmp_path):
    calls = []
    config_file = tmp_path / "kitchen.yaml"
    config_file.write_text("quant: true", encoding="utf-8")
    fake_wandb = types.ModuleType("wandb")
    fake_wandb.init = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    args = SimpleNamespace(
        wandb_project="proj",
        wandb_exp_name="run",
        wandb_save_dir=None,
        save=str(tmp_path),
        wandb_entity="team",
        kitchen_config_file=str(config_file),
        rank=1,
        world_size=2,
    )
    global_vars._set_wandb_writer(args)

    assert global_vars.get_wandb_writer() is fake_wandb
    assert calls[0]["project"] == "proj"
    assert calls[0]["name"] == "run"
    assert calls[0]["entity"] == "team"
    assert calls[0]["dir"] == str(tmp_path / "wandb")
    assert calls[0]["config"]["kitchen_config_file_contents"] == "quant: true"


def test_set_wandb_writer_requires_experiment_name():
    args = SimpleNamespace(
        wandb_project="proj",
        wandb_exp_name="",
        rank=0,
        world_size=1,
    )

    with pytest.raises(ValueError, match="wandb experiment name"):
        global_vars._set_wandb_writer(args)


def test_set_one_logger_success_and_fallback(monkeypatch):
    created = []

    class FakeOneLogger:
        def __init__(self, config):
            created.append(config)

    fake_one_logger = types.ModuleType("one_logger")
    fake_one_logger.OneLogger = FakeOneLogger
    monkeypatch.setitem(sys.modules, "one_logger", fake_one_logger)

    args = SimpleNamespace(
        enable_one_logger=True,
        one_logger_async=False,
        wandb_project="",
        one_logger_project="proj",
        one_logger_run_name="run",
        rank=0,
        world_size=1,
    )
    global_vars._set_one_logger(args)

    assert isinstance(global_vars.get_one_logger(), FakeOneLogger)
    assert created == [{"project": "proj", "name": "run", "async": False}]

    global_vars._GLOBAL_ONE_LOGGER = None
    fake_one_logger.OneLogger = lambda config: (_ for _ in ()).throw(RuntimeError("unavailable"))
    args.one_logger_async = True
    global_vars._set_one_logger(args)
    assert global_vars.get_one_logger() is None


def test_set_adlr_autoresume_success(monkeypatch):
    printed = []

    class FakeAutoResume:
        pass

    fake_userlib = types.ModuleType("userlib")
    fake_auto_resume = types.ModuleType("userlib.auto_resume")
    fake_auto_resume.AutoResume = FakeAutoResume
    monkeypatch.setitem(sys.modules, "userlib", fake_userlib)
    monkeypatch.setitem(sys.modules, "userlib.auto_resume", fake_auto_resume)
    monkeypatch.setenv("SUBMIT_SCRIPTS", "/tmp/submit_scripts")
    monkeypatch.setattr(global_vars.sys, "path", list(sys.path))
    monkeypatch.setattr("megatron.training.utils.print_rank_0", lambda message: printed.append(message))

    global_vars._set_adlr_autoresume(SimpleNamespace(adlr_autoresume=True))

    assert global_vars.get_adlr_autoresume() is FakeAutoResume
    assert printed == ["enabling autoresume ..."]


def test_set_timers_energy_and_signal_handler(monkeypatch):
    calls = []

    class FakeTimers:
        def __init__(self, log_level, log_option):
            calls.append(("timers", log_level, log_option))

    class FakeEnergyMonitor:
        def __init__(self):
            calls.append("energy")

    class FakeSignalHandler:
        def __init__(self, exit_signal):
            calls.append(("signal-init", exit_signal))

        def __enter__(self):
            calls.append("signal-enter")
            return "handler"

    monkeypatch.setattr(global_vars, "Timers", FakeTimers)
    monkeypatch.setattr(global_vars, "EnergyMonitor", FakeEnergyMonitor)
    monkeypatch.setattr(global_vars, "DistributedSignalHandler", FakeSignalHandler)

    global_vars._set_timers(SimpleNamespace(timing_log_level=2, timing_log_option="minmax"))
    global_vars._set_energy_monitor(SimpleNamespace())
    global_vars._set_signal_handler(exit_signal=15)

    assert calls == [("timers", 2, "minmax"), "energy", ("signal-init", 15), "signal-enter"]
    assert global_vars.get_timers().__class__ is FakeTimers
    assert global_vars.get_energy_monitor().__class__ is FakeEnergyMonitor
    assert global_vars.get_signal_handler() == "handler"


def test_graceful_shutdown_synchronizes_and_exits(monkeypatch):
    calls = []
    monkeypatch.setattr("megatron.training.utils.print_rank_0", lambda message: calls.append(("print", message)))
    monkeypatch.setattr(global_vars.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(global_vars.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(global_vars.torch.distributed, "barrier", lambda timeout=None: calls.append(("barrier", timeout)))
    monkeypatch.setattr(global_vars.torch.distributed, "destroy_process_group", lambda: calls.append("destroy"))

    with pytest.raises(SystemExit):
        global_vars._graceful_shutdown(None, None)

    assert any(item[0] == "print" for item in calls if isinstance(item, tuple))
    assert any(item[0] == "barrier" for item in calls if isinstance(item, tuple))
    assert "destroy" in calls
