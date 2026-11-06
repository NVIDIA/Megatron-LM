# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Random access to indexed prepared-token eligibility runs."""

import operator
import re
import struct

from megatron.energon.cache.base import FileStore
from megatron.energon.cache.file_store import ByteRangeStore
from megatron.energon.epathlib import EPath
from megatron.energon.source_info import SourceInfo

_RUN = struct.Struct("<iB3s")
_FOOTER = struct.Struct("<8sQQQQ")
_SIGNATURE = b"MASKID\x00\x01"


def _validate_runs(data: bytes) -> int:
    if len(data) % _RUN.size:
        raise ValueError("Token mask rows must contain exactly 8 bytes")
    total = 0
    for count, flag, reserved in _RUN.iter_unpack(data):
        if count <= 0 or flag not in (0, 1) or reserved != b"\x00\x00\x00":
            raise ValueError("Invalid token mask run count, flag or reserved bytes")
        total += count
    return total


def expand_token_mask(data: bytes, source_token_count: int) -> list[bool]:
    """Expand one record's RLE bytes to a Python list of built-in bool values.

    Values follow original token order; an empty record returns ``[]``.
    Validate all rows and the expected nonnegative integer token count before
    allocating the result. No token, role or packing policy is applied.
    """
    if isinstance(source_token_count, bool):
        raise ValueError("source_token_count must be a nonnegative integer")
    try:
        source_token_count = operator.index(source_token_count)
    except TypeError as exc:
        raise ValueError("source_token_count must be a nonnegative integer") from exc
    if source_token_count < 0:
        raise ValueError("source_token_count must be a nonnegative integer")
    if _validate_runs(data) != source_token_count:
        raise ValueError("Token mask count does not match source_token_count")
    result = []
    for count, flag, _ in _RUN.iter_unpack(data):
        result.extend([bool(flag)] * count)
    return result


class TokenMaskFileStore(FileStore[bytes]):
    """Read one original record's validated eligibility RLE bytes by decimal key.

    Accepts ``"1"`` or ``"1.mask"``. Construction validates the indexed mask
    layout and endpoint indexes using bounded IO; other indexes and run rows
    are validated only when requested. The first uint64 footer slot is reserved
    and ignored. Handles are owned by individual reads,
    allowing concurrent and forked callers without shared file offsets.
    """

    def __init__(self, path: EPath | str):
        self.path = EPath(path)
        self._ranges = ByteRangeStore(self.path.parent)
        size = self.path.size()
        if size < _FOOTER.size + 8:
            raise ValueError("Truncated token mask file")
        signature, _reserved, records, runs, index_offset = _FOOTER.unpack(
            self._read(size - _FOOTER.size, _FOOTER.size)
        )
        if signature != _SIGNATURE:
            raise ValueError("Invalid token mask signature")
        if (
            runs > (1 << 63) - 1
            or index_offset != runs * _RUN.size
            or index_offset + (records + 1) * 8 + _FOOTER.size != size
        ):
            raise ValueError("Invalid token mask layout")
        self.document_count = records
        self._run_count = runs
        self._index_offset = index_offset
        first = struct.unpack("<q", self._read(index_offset, 8))[0]
        last = struct.unpack("<q", self._read(index_offset + records * 8, 8))[0]
        if first != 0 or last != runs:
            raise ValueError("Invalid token mask index endpoints")

    def _read(self, offset: int, size: int) -> bytes:
        return self._ranges.read_range(self.path.name, offset, size)

    def __len__(self) -> int:
        return self.document_count

    def __getitem__(self, key: str) -> tuple[bytes, SourceInfo]:
        if not isinstance(key, str) or re.fullmatch(r"-?[0-9]+(?:\.mask)?", key) is None:
            raise ValueError(f"Invalid token mask key: {key!r}")
        index = int(key.removesuffix(".mask"))
        if not 0 <= index < self.document_count:
            raise IndexError(f"Token mask index {index} out of range")
        start, end = struct.unpack("<qq", self._read(self._index_offset + index * 8, 16))
        if not 0 <= start <= end <= self._run_count:
            raise ValueError(f"Invalid token mask index pair for record {index}")
        data = self._read(start * _RUN.size, (end - start) * _RUN.size)
        _validate_runs(data)
        return data, SourceInfo(
            dataset_path=self.path,
            index=index,
            shard_name=self.path.name,
            file_names=(f"{index}.mask",),
        )

    def get_path(self) -> str:
        """Return the companion file's stable EPath identity."""
        return str(self.path)
