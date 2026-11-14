# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import errno
import multiprocessing
import os
import pickle
import sys
import types

import numpy
import pytest

from megatron.core import msc_utils
from megatron.core.datasets import indexed_dataset as indexed_dataset_module
from megatron.core.datasets.indexed_dataset import (
    IndexedDataset,
    IndexedDatasetBuilder,
    _FileBinReader,
    get_bin_path,
    get_idx_path,
)

_DTYPE = numpy.uint16


@pytest.fixture
def prefix(tmp_path):
    """Build a small IndexedDataset of variable-length documents and return its prefix."""
    prefix = str(tmp_path / "dataset")
    rng = numpy.random.default_rng(0)
    builder = IndexedDatasetBuilder(get_bin_path(prefix), dtype=_DTYPE)
    for _ in range(64):
        length = int(rng.integers(1, 300))
        builder.add_document(rng.integers(0, 50000, length, dtype=_DTYPE), [length])
    builder.finalize(get_idx_path(prefix))
    return prefix


@pytest.fixture
def no_sleep(monkeypatch):
    """Make retry backoff instant and record the sleeps."""
    sleeps = []
    monkeypatch.setattr(indexed_dataset_module.time, "sleep", sleeps.append)
    return sleeps


def _count_opens(monkeypatch, path):
    """Count os.open calls on `path`."""
    opens = []
    real_open = os.open

    def counting_open(file, *args, **kwargs):
        if file == path:
            opens.append(file)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(os, "open", counting_open)
    return opens


def _fd_is_open(fd):
    try:
        os.fstat(fd)
        return True
    except OSError as e:
        assert e.errno == errno.EBADF
        return False


def test_matches_mmap(prefix):
    file_dataset = IndexedDataset(prefix, mmap=False)
    mmap_dataset = IndexedDataset(prefix, mmap=True)
    assert isinstance(file_dataset.bin_reader, _FileBinReader)

    for idx in range(len(mmap_dataset)):
        numpy.testing.assert_array_equal(file_dataset[idx], mmap_dataset[idx])
        length = mmap_dataset.sequence_lengths[idx]
        offset, sub_length = length // 3, length // 2
        numpy.testing.assert_array_equal(
            file_dataset.get(idx, offset=offset, length=sub_length),
            mmap_dataset.get(idx, offset=offset, length=sub_length),
        )
    for file_part, mmap_part in zip(file_dataset[5:20], mmap_dataset[5:20], strict=True):
        numpy.testing.assert_array_equal(file_part, mmap_part)
    assert file_dataset.get(0, length=0).size == 0


def test_opens_once_across_reads(prefix, monkeypatch):
    opens = _count_opens(monkeypatch, get_bin_path(prefix))
    dataset = IndexedDataset(prefix, mmap=False)
    assert opens == []  # lazy: constructing the reader does not open the .bin

    for idx in range(len(dataset)):
        dataset[idx]
    assert len(opens) == 1


def test_short_reads_are_completed(prefix, monkeypatch):
    real_preadv = os.preadv

    def short_preadv(fd, buffers, offset):
        (buffer,) = buffers
        return real_preadv(fd, [memoryview(buffer)[:7]], offset)

    expected = IndexedDataset(prefix, mmap=True)
    dataset = IndexedDataset(prefix, mmap=False)
    monkeypatch.setattr(os, "preadv", short_preadv)
    for idx in range(len(dataset)):
        numpy.testing.assert_array_equal(dataset[idx], expected[idx])


def test_truncated_file_raises_without_retry(prefix, monkeypatch, no_sleep):
    dataset = IndexedDataset(prefix, mmap=False)
    os.truncate(get_bin_path(prefix), os.path.getsize(get_bin_path(prefix)) - 1)
    real_preadv = os.preadv
    eof_reads = []

    def counting_preadv(*args):
        nread = real_preadv(*args)
        if nread == 0:
            eof_reads.append(args[2])
        return nread

    monkeypatch.setattr(os, "preadv", counting_preadv)
    with pytest.raises(EOFError):
        dataset[len(dataset) - 1]
    assert len(eof_reads) == 1
    assert no_sleep == []


def test_retry_reopens_descriptor(prefix, monkeypatch, no_sleep):
    real_open, real_close, real_preadv = os.open, os.close, os.preadv
    events = []
    opened_fds = []
    read_offsets = []

    def tracking_open(path, *args, **kwargs):
        fd = real_open(path, *args, **kwargs)
        if path == get_bin_path(prefix):
            opened_fds.append(fd)
            events.append("open")
        return fd

    def tracking_close(fd):
        if fd in opened_fds:
            events.append("close")
        real_close(fd)

    def flaky_preadv(fd, buffers, offset):
        events.append("read")
        read_offsets.append(offset)
        if len(read_offsets) == 1:
            return real_preadv(fd, [buffers[0][:7]], offset)
        if len(read_offsets) == 2:
            raise OSError(errno.EIO, "injected after a partial read")
        return real_preadv(fd, buffers, offset)

    # Keep the mmap dataset alive: its returned arrays are views into its mapping.
    expected = IndexedDataset(prefix, mmap=True)
    dataset = IndexedDataset(prefix, mmap=False)
    monkeypatch.setattr(os, "open", tracking_open)
    monkeypatch.setattr(os, "close", tracking_close)
    monkeypatch.setattr(os, "preadv", flaky_preadv)
    numpy.testing.assert_array_equal(dataset[3], expected[3])
    offset = int(dataset.index.sequence_pointers[3])
    assert read_offsets == [offset, offset + 7, offset]
    assert events == ["open", "read", "read", "close", "open", "read"]
    assert len(no_sleep) == 1


@pytest.mark.parametrize("code", [errno.EMFILE, errno.ENFILE])
def test_out_of_descriptors_raises_without_retry(prefix, monkeypatch, no_sleep, code):
    attempts = []

    def exhausted_open(*args, **kwargs):
        attempts.append(args[0])
        raise OSError(code, os.strerror(code))

    dataset = IndexedDataset(prefix, mmap=False)
    monkeypatch.setattr(os, "open", exhausted_open)
    with pytest.raises(OSError) as excinfo:
        dataset[0]
    assert excinfo.value.errno == code
    assert attempts == [get_bin_path(prefix)]
    assert no_sleep == []


def test_del_closes_descriptor(prefix):
    reader = _FileBinReader(get_bin_path(prefix))
    reader.read(_DTYPE, 1, 0)
    fd = reader._fd
    assert _fd_is_open(fd)
    del reader
    assert not _fd_is_open(fd)


@pytest.mark.parametrize("duplicate", [copy.copy, lambda r: pickle.loads(pickle.dumps(r))])
def test_copies_do_not_share_descriptor(prefix, duplicate):
    reader = _FileBinReader(get_bin_path(prefix))
    expected = reader.read(_DTYPE, 4, 0)
    fd = reader._fd

    clone = duplicate(reader)
    assert clone._fd is None
    numpy.testing.assert_array_equal(clone.read(_DTYPE, 4, 0), expected)
    del clone
    assert _fd_is_open(fd)
    numpy.testing.assert_array_equal(reader.read(_DTYPE, 4, 0), expected)


def test_indexed_dataset_pickle_round_trip(prefix):
    dataset = IndexedDataset(prefix, mmap=False)
    expected = dataset[3]
    fd = dataset.bin_reader._fd

    clone = pickle.loads(pickle.dumps(dataset))
    assert clone.bin_reader is not dataset.bin_reader
    assert clone.bin_reader._fd is None
    numpy.testing.assert_array_equal(clone[3], expected)
    assert _fd_is_open(fd)
    numpy.testing.assert_array_equal(dataset[3], expected)


def test_multi_storage_client_enabled_bypasses_descriptor(prefix, monkeypatch, no_sleep):
    """With MSC enabled, each read uses msc.open rather than the local descriptor."""
    opens = []

    def msc_open(path, *args, **kwargs):
        opens.append(path)
        return open(path, *args, **kwargs)

    monkeypatch.setattr(msc_utils, "msc", types.SimpleNamespace(open=msc_open))
    monkeypatch.setattr(msc_utils.MultiStorageClientFeature, "_enabled", True)

    def forbidden_preadv(*args, **kwargs):
        raise AssertionError("the multi-storage client path must not use os.preadv")

    expected = IndexedDataset(prefix, mmap=True)
    dataset = IndexedDataset(prefix, mmap=False)
    opens.clear()
    monkeypatch.setattr(os, "preadv", forbidden_preadv)
    for idx in range(len(dataset)):
        numpy.testing.assert_array_equal(dataset[idx], expected[idx])
    assert opens == [get_bin_path(prefix)] * len(dataset)
    assert dataset.bin_reader._fd is None
    assert no_sleep == []


# Set before forking so children reach the parent's reader through inherited memory. Passing the
# dataset as a task argument would pickle it and give each child a fresh reader instead.
_FORKED_DATASET = None


def _read_all(dataset, start, stop):
    return [dataset[idx].tolist() for idx in range(start, stop)]


def _read_forked(start, stop):
    return _FORKED_DATASET.bin_reader._fd, _read_all(_FORKED_DATASET, start, stop)


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(), reason="requires fork start method"
)
def test_reads_after_fork_share_descriptor_safely(prefix, monkeypatch):
    dataset = IndexedDataset(prefix, mmap=False)
    dataset[0]  # open the descriptor in the parent so the children inherit it
    # Positional reads must leave the shared cursor untouched, regardless of worker scheduling.
    cursor = 13
    os.lseek(dataset.bin_reader._fd, cursor, os.SEEK_SET)
    monkeypatch.setattr(sys.modules[__name__], "_FORKED_DATASET", dataset)
    expected = _read_all(IndexedDataset(prefix, mmap=True), 0, len(dataset))
    half = len(dataset) // 2

    with multiprocessing.get_context("fork").Pool(2) as pool:
        # Interleave reads from two processes on one inherited descriptor.
        results = pool.starmap(_read_forked, [(0, half), (half, len(dataset))] * 4)

    for fd, _ in results:
        assert fd == dataset.bin_reader._fd
    for i, (_, part) in enumerate(results):
        assert part == (expected[:half] if i % 2 == 0 else expected[half:])
    assert os.lseek(dataset.bin_reader._fd, 0, os.SEEK_CUR) == cursor
    assert _read_all(dataset, 0, len(dataset)) == expected
    assert os.lseek(dataset.bin_reader._fd, 0, os.SEEK_CUR) == cursor
