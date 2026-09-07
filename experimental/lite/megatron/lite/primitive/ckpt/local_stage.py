# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import contextlib
import os
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import IO

import torch.distributed as dist  # pyright: ignore[reportMissingImports]
from torch.distributed.checkpoint.filesystem import (  # pyright: ignore[reportMissingImports]
    FileSystem,
)


def local_stage_root() -> Path | None:
    stage_dir = os.environ.get("MLITE_DCP_LOCAL_STAGE_DIR")
    return Path(stage_dir) if stage_dir else None


def rank_stage_dir(stage_root: Path) -> Path:
    rank = dist.get_rank() if dist.is_initialized() else 0
    path = stage_root / f"rank-{rank}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def allocate_stage_path(stage_root: Path, destination_name: str) -> Path:
    descriptor, path = tempfile.mkstemp(
        prefix=f"{destination_name}.", suffix=".stage", dir=rank_stage_dir(stage_root)
    )
    os.close(descriptor)
    return Path(path)


class NodeLocalStagingFileSystem(FileSystem):
    def __init__(self, stage_root: Path):
        self._stage_root = stage_root

    @contextlib.contextmanager
    def create_stream(self, path: str, mode: str) -> Iterator[IO[bytes]]:
        if mode != "wb":
            with super().create_stream(path, mode) as stream:
                yield stream
            return

        destination = Path(path)
        stage_path = allocate_stage_path(self._stage_root, destination.name)
        try:
            with stage_path.open("wb", buffering=0) as stream:
                yield stream
                stream.flush()
                os.fsync(stream.fileno())
            publish_staged_file(stage_path, destination)
        finally:
            stage_path.unlink(missing_ok=True)


def publish_staged_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.publish.{os.getpid()}.{uuid.uuid4().hex}"
    )
    source_fd = os.open(source, os.O_RDONLY | os.O_CLOEXEC)
    destination_fd = None
    try:
        size = os.fstat(source_fd).st_size
        destination_fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
        )
        offset = 0
        while offset < size:
            written = os.sendfile(destination_fd, source_fd, offset, size - offset)
            if written <= 0:
                raise OSError(
                    f"sendfile returned {written} after {offset}/{size} bytes"
                )
            offset += written
        os.fsync(destination_fd)
        os.close(destination_fd)
        destination_fd = None
        os.close(source_fd)
        source_fd = None
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        if source_fd is not None:
            os.close(source_fd)
        temporary.unlink(missing_ok=True)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
