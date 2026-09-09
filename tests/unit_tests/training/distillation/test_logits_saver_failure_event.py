# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import multiprocessing as mp
from collections import OrderedDict

import pytest

from megatron.training.distillation import logits_saver
from megatron.training.distillation.logits_saver import LogitsSaverHooks, check_logits_saver_failure

zstandard = pytest.importorskip("zstandard")


def test_write_batched_tar_sets_failure_event_on_exception(tmp_path, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated quarantine copy failure")

    monkeypatch.setattr(logits_saver, "quarantine_contained_tars", _boom)

    event = mp.Event()
    writes = OrderedDict({(0, 10): b"payload"})
    tar_path = str(tmp_path / "dp0__0-10.tar")

    with pytest.raises(RuntimeError, match="simulated"):
        LogitsSaverHooks._write_batched_tar(
            tar_path, writes, b"{}", msc_enabled=False, existing_tars=[], failure_event=event
        )

    assert event.is_set()


def test_write_batched_tar_does_not_set_event_on_success(tmp_path):
    event = mp.Event()
    writes = OrderedDict({(0, 10): b"payload"})
    tar_path = str(tmp_path / "dp0__0-10.tar")

    LogitsSaverHooks._write_batched_tar(
        tar_path, writes, b"{}", msc_enabled=False, existing_tars=[], failure_event=event
    )

    assert not event.is_set()
    assert (tmp_path / "dp0__0-10.tar").exists()


def test_write_batched_tar_failure_event_none_is_safe(tmp_path, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(logits_saver, "quarantine_contained_tars", _boom)

    writes = OrderedDict({(0, 10): b"payload"})
    tar_path = str(tmp_path / "dp0__0-10.tar")

    with pytest.raises(RuntimeError):
        LogitsSaverHooks._write_batched_tar(
            tar_path, writes, b"{}", msc_enabled=False, existing_tars=[], failure_event=None
        )


def test_write_batched_tar_noop_when_writes_empty_leaves_event_clear():
    event = mp.Event()
    LogitsSaverHooks._write_batched_tar(
        "unused", OrderedDict(), b"{}", msc_enabled=False, existing_tars=[], failure_event=event
    )
    assert not event.is_set()


class _FakeSaver:
    save_dir = "/fake/dir"

    def __init__(self, set_failure: bool):
        self._failure_event = mp.Event()
        if set_failure:
            self._failure_event.set()


def test_check_logits_saver_failure_raises_when_event_set(monkeypatch):
    monkeypatch.setattr(logits_saver, "_ACTIVE_LOGITS_SAVER", _FakeSaver(set_failure=True))

    with pytest.raises(RuntimeError, match="background write failed"):
        check_logits_saver_failure()


def test_check_logits_saver_failure_noop_when_event_clear(monkeypatch):
    monkeypatch.setattr(logits_saver, "_ACTIVE_LOGITS_SAVER", _FakeSaver(set_failure=False))

    check_logits_saver_failure()  # should not raise


def test_check_logits_saver_failure_noop_when_no_active_saver(monkeypatch):
    monkeypatch.setattr(logits_saver, "_ACTIVE_LOGITS_SAVER", None)

    check_logits_saver_failure()  # should not raise
