# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
import os
from types import SimpleNamespace
from typing import Any, List

import pytest

import megatron.core.datasets.object_storage_utils as object_storage_utils
from megatron.core.datasets.object_storage_utils import MSC_PREFIX, cache_index_file

REMOTE_PATH = MSC_PREFIX + "profile/bucket/dataset.idx"


class _ScriptedClient:
    """Stands in for the multistorageclient package, one scripted attempt at a time.

    Each entry of `attempts` says what the call does: raise the exception, write the file,
    or return without writing anything. The last entry repeats.
    """

    def __init__(self, attempts: List[Any]) -> None:
        self.attempts = attempts
        self.calls = 0

    def download_file(self, remote_path: str, local_path: str) -> None:
        """Mimic multistorageclient.download_file for the scripted attempt."""
        self.calls += 1
        attempt = self.attempts[min(self.calls, len(self.attempts)) - 1]
        if isinstance(attempt, Exception):
            raise attempt
        if attempt:
            with open(local_path, "w") as index_file:
                index_file.write("index")


@pytest.fixture
def slept(monkeypatch) -> List[float]:
    """Record what the retry loop would sleep for, instead of sleeping."""
    waits: List[float] = []
    monkeypatch.setattr(object_storage_utils, "time", SimpleNamespace(sleep=waits.append))
    return waits


def _use(monkeypatch, client: Any) -> None:
    """Hand the client to cache_index_file in place of the real package."""
    monkeypatch.setattr(
        object_storage_utils.MultiStorageClientFeature, "import_package", lambda: client
    )


def test_a_download_that_raises_is_retried(tmp_path, monkeypatch, slept):
    client = _ScriptedClient([RuntimeError("the store said no"), RuntimeError("again"), True])
    _use(monkeypatch, client)
    local_path = str(tmp_path / "dataset.idx")

    cache_index_file(REMOTE_PATH, local_path, sleep_duration_start=10.0, jitter=0.0)

    assert client.calls == 3
    assert os.path.exists(local_path)
    assert slept == [10.0, 20.0], "the wait doubles before each retry"


def test_a_file_that_is_not_there_yet_is_waited_for(tmp_path, monkeypatch, slept):
    # The case from the issue: the call returns but the index file is not visible yet,
    # either because another rank is still writing it or because the store has not caught up.
    client = _ScriptedClient([False, False, True])
    _use(monkeypatch, client)
    local_path = str(tmp_path / "dataset.idx")

    cache_index_file(REMOTE_PATH, local_path, sleep_duration_start=1.0, jitter=0.0)

    assert client.calls == 3
    assert os.path.exists(local_path)
    assert slept == [1.0, 2.0]


def test_each_wait_is_lengthened_by_a_fraction_of_itself(tmp_path, monkeypatch, slept):
    client = _ScriptedClient([False, False, True])
    _use(monkeypatch, client)
    # The widest draw, so the assertion pins the upper bound of the jittered wait.
    monkeypatch.setattr(
        object_storage_utils, "random", SimpleNamespace(uniform=lambda low, high: high)
    )

    cache_index_file(
        REMOTE_PATH, str(tmp_path / "dataset.idx"), sleep_duration_start=10.0, jitter=0.5
    )

    assert slept == [15.0, 30.0], "jitter scales with the backoff rather than being a constant"


def test_the_last_attempt_re_raises_the_exception(tmp_path, monkeypatch, slept):
    error = RuntimeError("the store said no")
    client = _ScriptedClient([error])
    _use(monkeypatch, client)

    with pytest.raises(RuntimeError) as exc_info:
        cache_index_file(
            REMOTE_PATH,
            str(tmp_path / "dataset.idx"),
            num_max_retries=2,
            sleep_duration_start=1.0,
            jitter=0.0,
        )

    assert exc_info.value is error, "the caller sees the store's own error, not a wrapper"
    assert client.calls == 3
    assert slept == [1.0, 2.0]


def test_a_file_that_never_appears_raises_file_not_found(tmp_path, monkeypatch, slept):
    client = _ScriptedClient([False])
    _use(monkeypatch, client)

    with pytest.raises(FileNotFoundError):
        cache_index_file(
            REMOTE_PATH,
            str(tmp_path / "dataset.idx"),
            num_max_retries=1,
            sleep_duration_start=1.0,
            jitter=0.0,
        )

    assert client.calls == 2
    assert slept == [1.0]


def test_an_unusable_path_is_not_retried(tmp_path, slept):
    with pytest.raises(ValueError):
        cache_index_file("gs://bucket/dataset.idx", str(tmp_path / "dataset.idx"))

    assert slept == []
