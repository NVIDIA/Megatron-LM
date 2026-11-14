"""Disabled timer reports preserve measurements required by the training loop."""

from unittest.mock import Mock

import pytest

from megatron.core import timers as timers_mod
from megatron.core.timers import DummyTimer, Timer, Timers


@pytest.mark.parametrize("level", [-1, 0, 1, 2])
def test_timer_levels_preserve_required_measurements(level, monkeypatch):
    monkeypatch.setattr(timers_mod.torch.cuda, "synchronize", lambda: None)
    ticks = iter([0.0, 1.0, 3.0])
    monkeypatch.setattr(timers_mod.time, "time", lambda: next(ticks))
    timers = Timers(level, "max")
    interval = timers("interval-time", log_level=0)
    assert isinstance(interval, Timer)
    interval.start()
    interval.stop()
    assert interval.elapsed(reset=False) == 2.0
    assert interval.active_time() == 2.0
    monkeypatch.setattr(timers_mod.time, "time", lambda: 4.0)
    assert isinstance(timers("optimizer", log_level=1), DummyTimer) is (level < 1)
    assert isinstance(timers("forward", log_level=2), DummyTimer) is (level < 2)


def test_disabled_reports_do_not_collect_or_write_timings(monkeypatch):
    timers = Timers(-1, "max")
    collect = Mock(side_effect=AssertionError("disabled timer reports must not collect timings"))
    monkeypatch.setattr(timers, "_get_elapsed_time_all_ranks", collect)
    writer = Mock()
    timers.log(["interval-time"], barrier=True)
    timers.write(["interval-time"], writer, 1, barrier=True)
    assert timers.get_all_timers_string(["interval-time"], barrier=True) is None
    collect.assert_not_called()
    writer.add_scalar.assert_not_called()


@pytest.mark.parametrize("report", ["log", "write", "get_all_timers_string"])
@pytest.mark.parametrize("running", [False, True])
def test_disabled_reports_preserve_local_measurement_boundaries(report, running, monkeypatch):
    monkeypatch.setattr(timers_mod.torch.cuda, "synchronize", lambda: None)
    now = [0.0]
    monkeypatch.setattr(timers_mod.time, "time", lambda: now[0])
    monkeypatch.setattr(timers_mod.torch.distributed, "barrier", Mock(side_effect=AssertionError))
    timers = Timers(-1, "max")
    timer = timers("save-checkpoint", log_level=0)
    timer.start()
    now[0] = 2.0
    if not running:
        timer.stop()
    assert timer.elapsed(reset=False) == 2.0
    args = (["save-checkpoint", "missing"],)
    if report == "write":
        args += (Mock(), 1)
    getattr(timers, report)(*args, reset=False, barrier=True)
    assert timer.elapsed(reset=False) == 2.0
    getattr(timers, report)(*args, reset=True, barrier=True)
    if not running:
        timer.start()
    now[0] = 5.0
    timer.stop()
    assert timer.elapsed(reset=False) == 3.0
    assert timer.active_time() == 5.0
