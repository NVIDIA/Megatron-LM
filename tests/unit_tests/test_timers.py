# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import Mock

import pytest
import torch

from megatron.core import timers as timers_module
from megatron.core.timers import DummyTimer, Timer, Timers


def test_dummy_timer_rejects_elapsed_queries():
    timer = DummyTimer()

    timer.start()
    timer.stop()
    timer.reset()

    with pytest.raises(Exception, match="dummy timer should not be used"):
        timer.elapsed()

    with pytest.raises(Exception, match="active timer should not be used"):
        timer.active_time()


def test_timer_lifecycle_accumulates_elapsed_and_active_time(monkeypatch):
    monkeypatch.setattr(timers_module.cur_platform, "synchronize", Mock())
    times = iter([0.0, 10.0, 10.25])
    monkeypatch.setattr(timers_module.time, "time", lambda: next(times))

    timer = Timer("forward")
    timer.start()
    timer.stop()

    assert timer.elapsed(reset=False) == pytest.approx(0.25)
    assert timer.active_time() == pytest.approx(0.25)
    assert not timer._started

    timer.reset()
    assert timer.elapsed(reset=False) == 0.0
    assert timer.active_time() == pytest.approx(0.25)


def test_timer_elapsed_restarts_running_timer(monkeypatch):
    monkeypatch.setattr(timers_module.cur_platform, "synchronize", Mock())
    times = iter([0.0, 1.0, 1.5, 2.0])
    monkeypatch.setattr(timers_module.time, "time", lambda: next(times))

    timer = Timer("backward")
    timer.start()

    assert timer.elapsed(reset=False) == pytest.approx(0.5)
    assert timer._started
    assert timer._elapsed == pytest.approx(0.5)


def test_timer_guards_start_stop_order(monkeypatch):
    monkeypatch.setattr(timers_module.cur_platform, "synchronize", Mock())

    timer = Timer("optimizer")

    with pytest.raises(AssertionError, match="timer is not started"):
        timer.stop()

    timer.start()

    with pytest.raises(AssertionError, match="timer has already been started"):
        timer.start()


def test_timers_returns_real_or_dummy_timer_by_log_level():
    timers = Timers(log_level=1, log_option="max")

    real_timer = timers("train", log_level=1)
    dummy_timer = timers("debug", log_level=2)

    assert isinstance(real_timer, Timer)
    assert isinstance(dummy_timer, DummyTimer)
    assert timers("train") is real_timer

    with pytest.raises(AssertionError, match="does not match already existing"):
        timers("train", log_level=0)

    with pytest.raises(AssertionError, match="larger than max supported"):
        timers("too-high", log_level=3)


def test_timers_rejects_invalid_log_option():
    with pytest.raises(AssertionError, match="input log option"):
        Timers(log_level=1, log_option="median")


def test_timers_formats_max_and_minmax_strings(monkeypatch):
    sample = torch.tensor([[1.0, 0.0], [3.0, 2.0]])

    def fake_all_ranks(self, names, reset, barrier):
        return sample

    monkeypatch.setattr(Timers, "_get_elapsed_time_all_ranks", fake_all_ranks)

    max_output = Timers(log_level=1, log_option="max").get_all_timers_string(
        ["forward", "backward"], normalizer=1.0
    )
    minmax_output = Timers(log_level=1, log_option="minmax").get_all_timers_string(
        ["forward", "backward"], normalizer=1.0
    )

    assert "max time across ranks" in max_output
    assert "forward" in max_output
    assert "3000.00" in max_output
    assert "(min, max) time across ranks" in minmax_output
    assert "(1000.00, 3000.00)" in minmax_output


def test_timers_formats_all_rank_strings(monkeypatch):
    sample = torch.tensor([[1.0], [3.0]])

    monkeypatch.setattr(Timers, "_get_elapsed_time_all_ranks", lambda *args: sample)
    monkeypatch.setattr(timers_module.torch.distributed, "get_world_size", lambda: 2)

    output = Timers(log_level=1, log_option="all").get_all_timers_string(["forward"])

    assert "times across ranks" in output
    assert "rank  0: 1000.00" in output
    assert "rank  1: 3000.00" in output


def test_timers_write_records_max_times(monkeypatch):
    class FakeSummaryWriter:
        def __init__(self):
            self.calls = []

        def add_scalar(self, name, value, iteration):
            self.calls.append((name, value, iteration))

    writer = FakeSummaryWriter()
    timers = Timers(log_level=1, log_option="max")

    monkeypatch.setattr(timers_module, "SummaryWriter", FakeSummaryWriter)
    monkeypatch.setattr(
        timers,
        "_get_global_min_max_time",
        lambda names, reset, barrier, normalizer: {"forward": (1.0, 4.0)},
    )

    timers.write(["forward"], writer=writer, iteration=7)

    assert writer.calls == [("forward-time", 4.0, 7)]


def test_timers_log_defaults_to_last_rank(monkeypatch):
    timers = Timers(log_level=1, log_option="max")

    monkeypatch.setattr(timers, "get_all_timers_string", lambda *args, **kwargs: "timing output")
    monkeypatch.setattr(timers_module.torch.distributed, "get_world_size", lambda: 4)
    monkeypatch.setattr(timers_module.torch.distributed, "get_rank", lambda: 3)
    monkeypatch.setattr(timers_module.logger, "info", Mock())

    timers.log(["forward"])

    timers_module.logger.info.assert_called_once_with("timing output")
