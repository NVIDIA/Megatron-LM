# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

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
