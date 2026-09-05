# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Helpers for safely extracting downloaded archives."""

import tarfile
import zipfile
from pathlib import Path
from typing import Iterable


def _validated_destination(destination: str | Path, member_names: Iterable[str]) -> Path:
    """Resolve archive members and require every target to stay below destination."""
    destination = Path(destination).resolve()
    for member_name in member_names:
        member_path = Path(member_name)
        if member_path.is_absolute():
            raise ValueError(f"Archive member escapes destination: {member_name!r}")
        try:
            (destination / member_path).resolve().relative_to(destination)
        except ValueError as error:
            raise ValueError(f"Archive member escapes destination: {member_name!r}") from error
    return destination


def safe_extract_zip(archive: zipfile.ZipFile, destination: str | Path) -> None:
    """Extract a ZIP archive after validating every member path."""
    destination = _validated_destination(destination, archive.namelist())
    archive.extractall(destination)


def safe_extract_tar(archive: tarfile.TarFile, destination: str | Path) -> None:
    """Extract a tar archive after validating member paths and types."""
    members = archive.getmembers()
    destination = _validated_destination(destination, (member.name for member in members))
    for member in members:
        if member.issym() or member.islnk():
            raise ValueError(f"Archive links are not allowed: {member.name!r}")
        if member.isdev():
            raise ValueError(f"Archive device entries are not allowed: {member.name!r}")
    archive.extractall(destination, members=members)
