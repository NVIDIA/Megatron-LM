# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Filesystem selection and DCP adaptation for checkpoint I/O."""

from __future__ import annotations

import importlib
import os
import stat
from contextlib import contextmanager
from pathlib import PurePosixPath
from queue import Empty, SimpleQueue
from typing import TYPE_CHECKING, BinaryIO

from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.filesystem import FileSystem

from megatron.core.msc_utils import MultiStorageClientFeature

if TYPE_CHECKING:
    from nvidia_resiliency_ext.checkpointing.filesystem import CheckpointFileSystem


def _validate_child(suffix: str) -> None:
    if not suffix or PurePosixPath(suffix).is_absolute() or ".." in suffix.split("/"):
        raise ValueError(f"Checkpoint child path must be relative: {suffix!r}")
    if "://" in suffix or "\\" in suffix:
        raise ValueError(f"Invalid checkpoint child path: {suffix!r}")


def _entry_kind(mode: int) -> str:
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISREG(mode):
        return "file"
    raise OSError("Checkpoint paths must refer to regular files or directories")


class _LocalFileSystem(FileSystem):
    def init_path(self, path):
        """Reject URIs before POSIX normalization can change their meaning."""
        if "://" in os.fspath(path):
            raise ValueError("URI checkpoint paths require an explicit filesystem or MSC")
        return super().init_path(path)

    def concat_path(self, path, suffix):
        """Join a checkpoint path with a relative child."""
        _validate_child(suffix)
        return super().concat_path(path, suffix)

    def stat(self, path):
        """Distinguish missing paths from failed filesystem operations."""
        try:
            return _entry_kind(os.stat(path).st_mode)
        except FileNotFoundError:
            return "missing"

    def listdir(self, path):
        """Return immediate child names."""
        return os.listdir(path)

    def sync(self, stream):
        """Flush Python buffers before requesting descriptor durability."""
        stream.flush()
        os.fsync(stream.fileno())


class _MultiStorageFileSystem:
    @property
    def _msc(self):
        return importlib.import_module("multistorageclient")

    def init_path(self, path):
        """Preserve MSC URI paths."""
        return os.fspath(path)

    def concat_path(self, path, suffix):
        """Join a relative child without changing the URI scheme."""
        _validate_child(suffix)
        return f"{os.fspath(path).rstrip('/')}/{suffix}"

    def create_stream(self, path, mode):
        """Open through the captured MSC backend."""
        return self._msc.open(os.fspath(path), mode)

    def mkdir(self, path):
        """Create the checkpoint directory and its parents."""
        self._msc.os.makedirs(os.fspath(path), exist_ok=True)

    def exists(self, path):
        """Check existence through MSC."""
        return self._msc.os.path.exists(os.fspath(path))

    def rename(self, path, new_path):
        """Move a file using MSC's native path implementation."""
        self._msc.Path(os.fspath(path)).rename(os.fspath(new_path))

    def rm_file(self, path):
        """Remove a checkpoint file through MSC."""
        self._msc.Path(os.fspath(path)).unlink()

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id):
        """Accept string and path-like checkpoint identifiers."""
        return isinstance(checkpoint_id, (str, os.PathLike))

    def stat(self, path):
        """Include implicit directories without hiding transport errors."""
        entry = self._msc.Path(os.fspath(path))
        if entry.is_dir(strict=False):
            return "directory"
        try:
            return _entry_kind(entry.stat().st_mode)
        except FileNotFoundError:
            return "missing"

    def listdir(self, path):
        """Return immediate names from MSC's directory listing."""
        return (entry.name for entry in self._msc.Path(os.fspath(path)).iterdir())

    def sync(self, stream):
        """Flush and use MSC's stream-specific durability operation."""
        stream.flush()
        stream.fsync()


_LOCAL_FILESYSTEM = _LocalFileSystem()
_MSC_FILESYSTEM = _MultiStorageFileSystem()


def resolve_filesystem(filesystem: CheckpointFileSystem | None = None) -> CheckpointFileSystem:
    """Resolve legacy selection only when no filesystem was supplied."""
    if filesystem is not None:
        return filesystem
    return _MSC_FILESYSTEM if MultiStorageClientFeature.is_enabled() else _LOCAL_FILESYSTEM


def is_default_filesystem(filesystem: CheckpointFileSystem) -> bool:
    """Identify adapters eligible for compatibility with older NVRx releases."""
    return filesystem is _LOCAL_FILESYSTEM or filesystem is _MSC_FILESYSTEM


def is_msc_filesystem(filesystem: CheckpointFileSystem) -> bool:
    """Identify the captured legacy MSC selection without rereading its flag."""
    return filesystem is _MSC_FILESYSTEM


class _ManagedWriteStream:
    def __init__(
        self, stream: BinaryIO, filesystem: CheckpointFileSystem, sync_files: bool
    ) -> None:
        self._stream = stream
        self._filesystem = filesystem
        self._sync_files = sync_files
        self._closed = False

    def __getattr__(self, name):
        return getattr(self._stream, name)

    @property
    def closed(self):
        """Report whether completion has been attempted."""
        return self._closed

    def close(self):
        """Synchronize before closing the underlying stream exactly once."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._sync_files:
                self._filesystem.sync(self._stream)
        finally:
            self._stream.close()


@contextmanager
def create_write_stream(filesystem: CheckpointFileSystem, path, *, sync_files: bool = True):
    """Synchronize before close, including when DCP closes the stream explicitly."""
    with filesystem.create_stream(path, "wb") as raw_stream:
        stream = _ManagedWriteStream(raw_stream, filesystem, sync_files)
        try:
            yield stream
        finally:
            stream.close()


class _WriterFileSystem:
    def __init__(self, filesystem: CheckpointFileSystem, sync_files: bool) -> None:
        self._filesystem = filesystem
        self._sync_files = sync_files
        self._errors = SimpleQueue()

    def __getattr__(self, name):
        return getattr(self._filesystem, name)

    @contextmanager
    def create_stream(self, path, mode):
        """Record stream failures even when DCP opens on a background thread."""
        try:
            if mode == "wb":
                with create_write_stream(
                    self._filesystem, path, sync_files=self._sync_files
                ) as stream:
                    yield stream
            else:
                with self._filesystem.create_stream(path, mode) as stream:
                    yield stream
        except Exception as exc:
            self._errors.put(exc)
            raise

    def raise_if_failed(self):
        """Propagate an I/O thread failure before DCP writes metadata."""
        try:
            error = self._errors.get_nowait()
        except Empty:
            return
        raise error


class _FileSystemWriter(FileSystemWriter):
    def __init__(
        self, path, filesystem: CheckpointFileSystem, *, sync_files=True, **kwargs
    ) -> None:
        super().__init__(path, sync_files=False, **kwargs)
        self.fs = _WriterFileSystem(filesystem, sync_files)
        self.path = filesystem.init_path(path)

    def write_data(self, plan, planner):
        """Use DCP serialization and verify all joined I/O workers succeeded."""
        result = super().write_data(plan, planner)
        self.fs.raise_if_failed()
        return result


def make_filesystem_writer(path, filesystem: CheckpointFileSystem, **kwargs) -> FileSystemWriter:
    """Bind DCP serialization to one filesystem and its completion policy."""
    return _FileSystemWriter(path, filesystem, **kwargs)


def make_filesystem_reader(path, filesystem: CheckpointFileSystem) -> FileSystemReader:
    """Bind a DCP reader without normalizing a URI through pathlib."""
    reader = FileSystemReader(path)
    reader.fs = filesystem
    reader.path = filesystem.init_path(path)
    return reader
