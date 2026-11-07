# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Offline KD sample files: Arrow IPC with zstd-compressed buffers, plus a JSON index.

A data file holds one row per sample: the input columns first, then the teacher's
target rows (only at positions with nonzero loss weight). Rows are grouped into record
batches ("blocks") of bounded size. The settings live in the schema metadata.

The index, written after its data file is complete, lists each block's rows, byte
ranges and supervised-token count, and carries a copy of the file's footer. A reader
can therefore fetch any block, or only its input columns, with one ranged read and
never needs to touch the rest of the data file. Its existence also marks the data file
as published.
"""

import base64
import io
import json
from typing import Any

import numpy as np
import pyarrow as pa
import torch

from .contract import (
    LOGPROB_SENTINEL,
    check_targets,
    index_encoding_for_vocab,
    join_indices,
    split_indices,
)

META_KEY = b"offline_kd"
INPUT_COLUMNS = ("tokens", "labels", "position_ids", "loss_mask", "segments")
TOKEN_COLUMNS = ("tokens", "labels", "position_ids", "loss_mask")
DEFAULT_BLOCK_BYTES = 16 << 20
INDEX_VERSION = 1
_MAGIC_BYTES = 8  # "ARROW1" plus padding at the start of an IPC file.


def _numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _lists(offsets: np.ndarray, flat: np.ndarray, kind: pa.DataType) -> pa.Array:
    return pa.LargeListArray.from_arrays(pa.array(offsets, pa.int64()), pa.array(flat, kind))


def _row_lists(offsets: np.ndarray, rows: np.ndarray, kind: pa.DataType) -> pa.Array:
    inner = pa.FixedSizeListArray.from_arrays(pa.array(rows.reshape(-1), kind), rows.shape[1])
    return pa.LargeListArray.from_arrays(pa.array(offsets, pa.int64()), inner)


def samples_batch(
    inputs: dict[str, Any],
    values: torch.Tensor,
    indices: torch.Tensor,
    tail: torch.Tensor | None,
    *,
    vocab_size: int,
) -> pa.RecordBatch:
    """Build one row per sample from flattened inputs and target rows.

    ``inputs`` holds ``sample_offsets`` (``[samples + 1]``), the token-aligned
    ``tokens``, ``labels``, ``position_ids`` and ``loss_mask`` (bool, or float32
    weights), and ``segment_lengths``/``segment_counts``. Target row ``i`` belongs to the
    ``i``-th position with nonzero loss weight, in token order. ``tail`` (the float32 log
    probability outside the retained entries) is optional; readers derive it from the
    values when absent, which is exact only for float32 values.
    """
    offsets = _numpy(inputs["sample_offsets"]).astype(np.int64)
    weights = _numpy(inputs["loss_mask"])
    if weights.dtype != np.bool_:
        weights = weights.astype(np.float32)
    scored = np.concatenate(([0], np.cumsum(weights != 0)))[offsets]
    values, indices = _numpy(values), _numpy(indices).astype(np.int64)
    if values.ndim != 2 or values.shape != indices.shape or values.shape[0] != scored[-1]:
        raise ValueError(
            "Offline KD targets must be [rows, K] with one row per nonzero loss weight"
        )
    if tail is not None and tuple(tail.shape) != values.shape[:1]:
        raise ValueError("Offline KD tail log probabilities must have one value per row")
    if values.dtype not in (np.float16, np.float32):
        raise ValueError("Offline KD target values must be float16 or float32")
    low, high = split_indices(indices, values, vocab_size=vocab_size)
    counts = _numpy(inputs["segment_counts"]).astype(np.int64)
    columns = {
        **{
            key: _lists(offsets, _numpy(inputs[key]).astype(np.int32), pa.int32())
            for key in ("tokens", "labels", "position_ids")
        },
        "loss_mask": _lists(offsets, weights, pa.from_numpy_dtype(weights.dtype)),
        "segments": _lists(
            np.concatenate(([0], np.cumsum(counts))),
            _numpy(inputs["segment_lengths"]).astype(np.int32),
            pa.int32(),
        ),
        "values": _row_lists(scored, values, pa.from_numpy_dtype(values.dtype)),
        "indices_low": _row_lists(scored, low, pa.uint16()),
    }
    if high is not None:
        columns["indices_high"] = _row_lists(scored, high, pa.uint8())
    if tail is not None:
        columns["tail"] = _lists(scored, _numpy(tail).astype(np.float32), pa.float32())
    return pa.record_batch(columns)


def _common_schema(batches: list[pa.RecordBatch]) -> pa.Schema:
    """Unify per-iteration schemas: binary loss masks widen to float32 if any are fractional."""
    schema = batches[0].schema
    for batch in batches[1:]:
        if batch.schema.equals(schema):
            continue
        fields = []
        for mine, theirs in zip(schema, batch.schema):
            if mine.name != theirs.name:
                raise ValueError("Offline KD sample batches have different columns")
            if mine.type.equals(theirs.type):
                fields.append(mine)
            elif mine.name == "loss_mask":
                fields.append(pa.field("loss_mask", pa.large_list(pa.float32())))
            else:
                raise ValueError(f"Offline KD sample batches disagree on {mine.name}'s type")
        schema = pa.schema(fields)
    return schema


def _row_bytes(table: pa.Table) -> np.ndarray:
    """Approximate uncompressed bytes per row, for block sizing."""
    sizes = np.zeros(table.num_rows, dtype=np.int64)
    for column in table.itercolumns():
        lengths = column.combine_chunks().value_lengths().to_numpy(zero_copy_only=False)
        item = column.type.value_type
        width = item.list_size if pa.types.is_fixed_size_list(item) else 1
        base = item.value_type if pa.types.is_fixed_size_list(item) else item
        sizes += lengths * width * max(1, base.bit_width // 8)
    return sizes


def _block_rows(row_bytes: np.ndarray, block_bytes: int) -> list[int]:
    """Greedy row counts per block: at least one row, at most ``block_bytes`` otherwise."""
    counts, current, size = [], 0, 0
    for nbytes in row_bytes.tolist():
        if current and size + nbytes > block_bytes:
            counts.append(current)
            current, size = 0, 0
        current += 1
        size += nbytes
    if current:
        counts.append(current)
    return counts


def _options() -> pa.ipc.IpcWriteOptions:
    return pa.ipc.IpcWriteOptions(compression=pa.Codec("zstd", compression_level=3))


def _input_body_size(block: pa.RecordBatch) -> int:
    """Body size of the block's input columns alone, which prefix the full body."""
    sink = pa.BufferOutputStream()
    inputs = block.select(list(INPUT_COLUMNS))
    with pa.ipc.new_stream(sink, inputs.schema, options=_options()) as writer:
        writer.write_batch(inputs)
    reader = pa.BufferReader(sink.getvalue())
    pa.ipc.read_message(reader)  # Schema.
    return pa.ipc.read_message(reader).body.size


def write_file(
    batches: list[pa.RecordBatch], meta: dict[str, Any], *, block_bytes: int = DEFAULT_BLOCK_BYTES
) -> tuple[bytes, dict[str, Any]]:
    """Serialize sample batches into ``(data file bytes, index)``.

    Rows are regrouped into blocks of at most ``block_bytes`` uncompressed bytes (a
    larger single row gets a block of its own), independent of how they were produced.
    """
    if not batches:
        raise ValueError("Cannot write an empty offline KD file")
    schema = _common_schema(batches)
    table = pa.Table.from_batches([batch.cast(schema) for batch in batches]).combine_chunks()
    schema = schema.with_metadata({META_KEY: json.dumps(meta, sort_keys=True).encode()})
    sink = pa.BufferOutputStream()
    spans, start = [], 0
    with pa.ipc.new_file(sink, schema, options=_options()) as writer:
        for rows in _block_rows(_row_bytes(table), block_bytes):
            block = pa.RecordBatch.from_arrays(
                table.slice(start, rows).to_batches()[0].columns, schema=schema
            )
            begin = sink.tell()
            writer.write_batch(block)
            spans.append((start, rows, begin, sink.tell(), _input_body_size(block), block))
            start += rows
    data = sink.getvalue().to_pybytes()
    footer = int.from_bytes(data[-10:-6], "little") + 10
    blocks = []
    for first_row, rows, begin, end, input_body, block in spans:
        reader = pa.BufferReader(pa.py_buffer(data[begin:end]))
        if begin == 0:
            reader.seek(_MAGIC_BYTES)
        while True:  # The first batch's span also holds the schema message.
            offset = reader.tell()
            message = pa.ipc.read_message(reader)
            if message.type == "record batch":
                break
        length = end - begin - offset
        blocks.append(
            {
                "first_row": first_row,
                "rows": rows,
                "offset": begin + offset,
                "length": length,
                "input_length": length - message.body.size + input_body,
                "supervised": len(block.column("values").flatten()),
            }
        )
    index = {
        "index_version": INDEX_VERSION,
        "size": len(data),
        "footer": base64.b64encode(data[-footer:]).decode(),
        "blocks": blocks,
    }
    return data, index


class _RangeFile(io.RawIOBase):
    """A file of known size that serves only the byte ranges it was given."""

    def __init__(self, size: int, ranges: dict[int, bytes]):
        self.size, self.ranges, self.position = size, ranges, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        base = (0, self.position, self.size)[whence]
        self.position = base + offset
        return self.position

    def readinto(self, buffer: Any) -> int:
        count = min(len(buffer), self.size - self.position)
        for start, raw in self.ranges.items():
            if start <= self.position and self.position + count <= start + len(raw):
                buffer[:count] = raw[self.position - start : self.position - start + count]
                self.position += count
                return count
        raise IOError(
            f"Offline KD reader needs bytes {self.position}-{self.position + count} it did not fetch"
        )


def _open(index: dict[str, Any], ranges: dict[int, bytes], **options: Any) -> Any:
    footer = base64.b64decode(index["footer"])
    file = _RangeFile(index["size"], {index["size"] - len(footer): footer, **ranges})
    read_options = pa.ipc.IpcReadOptions(**options) if options else None
    return pa.ipc.open_file(pa.PythonFile(file, mode="r"), options=read_options)


def check_index(index: dict[str, Any], size: int | None = None) -> None:
    """Reject an index of another version, or one that belongs to a different data file."""
    if index.get("index_version") != INDEX_VERSION:
        raise ValueError("Unsupported offline KD index version")
    if size is not None and size != index["size"]:
        raise ValueError(
            f"Offline KD index describes a {index['size']}-byte data file, but it has {size} "
            "bytes; it belongs to another version of the file"
        )


def file_meta(index: dict[str, Any]) -> dict[str, Any]:
    """Return the data file's settings, read from the index's footer copy."""
    metadata = _open(index, {}).schema.metadata or {}
    if META_KEY not in metadata:
        raise ValueError("Offline KD data file carries no settings")
    return json.loads(metadata[META_KEY])


def block_range(index: dict[str, Any], block: int, *, inputs_only: bool) -> tuple[int, int]:
    """Return ``(offset, size)`` of a block, or of its input columns only."""
    entry = index["blocks"][block]
    return entry["offset"], entry["input_length" if inputs_only else "length"]


def read_block(
    index: dict[str, Any], block: int, raw: bytes, *, inputs_only: bool = False
) -> pa.RecordBatch:
    """Decode a block from the bytes :func:`block_range` names."""
    offset, _ = block_range(index, block, inputs_only=inputs_only)
    options = {"included_fields": list(range(len(INPUT_COLUMNS)))} if inputs_only else {}
    return _open(index, {offset: raw}, **options).get_batch(block)


def _tensor(array: np.ndarray) -> torch.Tensor:
    """Wrap a NumPy array as a tensor, copying Arrow's read-only buffers."""
    return torch.from_numpy(np.require(array, requirements=["C", "W"]))


def _flat(column: pa.Array) -> tuple[np.ndarray, np.ndarray]:
    """Return a large list column's ``(offsets, flat values)`` as NumPy arrays."""
    offsets = column.offsets.to_numpy().astype(np.int64)
    values = column.values
    if pa.types.is_fixed_size_list(values.type):
        width = values.type.list_size
        flat = values.flatten().to_numpy(zero_copy_only=False).reshape(-1, width)
    else:
        flat = values.to_numpy(zero_copy_only=False)
    base = offsets[0]
    return offsets - base, flat[base : offsets[-1]]


def decode_block(
    batch: pa.RecordBatch, *, vocab_size: int | None = None
) -> tuple[dict[str, torch.Tensor], tuple | None]:
    """Convert a block to flattened inputs and, when present, ``(values, indices, tail)``.

    Inputs carry ``sample_offsets``, the token-aligned columns, ``segment_lengths`` and
    ``segment_counts``. Every sample's segments must cover its tokens and its target
    rows must match its nonzero loss weights.
    """
    names = set(batch.schema.names)
    missing = [name for name in INPUT_COLUMNS if name not in names]
    if missing:
        raise ValueError(f"Offline KD block is missing columns: {', '.join(missing)}")
    offsets, tokens = _flat(batch.column("tokens"))
    inputs = {"sample_offsets": _tensor(offsets), "tokens": _tensor(tokens)}
    for key in TOKEN_COLUMNS[1:]:
        key_offsets, flat = _flat(batch.column(key))
        if not np.array_equal(key_offsets, offsets):
            raise ValueError(f"Offline KD {key} is not aligned with tokens")
        inputs[key] = _tensor(flat)
    segment_offsets, lengths = _flat(batch.column("segments"))
    counts = np.diff(segment_offsets)
    owner = np.repeat(np.arange(len(counts)), counts)
    per_sample = np.bincount(owner, weights=lengths, minlength=len(counts)).astype(np.int64)
    if (
        len(counts) != len(offsets) - 1
        or (lengths <= 0).any()
        or not np.array_equal(per_sample, np.diff(offsets))
    ):
        raise ValueError("Offline KD segments disagree with sample lengths")
    inputs["segment_lengths"] = _tensor(lengths.astype(np.int64))
    inputs["segment_counts"] = _tensor(counts.astype(np.int32))
    if "values" not in names:
        return inputs, None
    row_offsets, values = _flat(batch.column("values"))
    _, low = _flat(batch.column("indices_low"))
    high = _flat(batch.column("indices_high"))[1] if "indices_high" in names else None
    if vocab_size is not None and (high is None) != (index_encoding_for_vocab(vocab_size) == "u16"):
        raise ValueError("Offline KD target index encoding disagrees with logprob_vocab_size")
    tail = _flat(batch.column("tail"))[1] if "tail" in names else None
    scored = np.concatenate(([0], np.cumsum(_numpy(inputs["loss_mask"]) != 0)))[offsets]
    if not np.array_equal(row_offsets, scored):
        raise ValueError("Offline KD target rows disagree with the number of nonzero loss weights")
    targets = (
        _tensor(values),
        _tensor(join_indices(low, high)),
        None if tail is None else _tensor(tail),
    )
    return inputs, check_targets(*targets, vocab_size=vocab_size)


__all__ = [
    "DEFAULT_BLOCK_BYTES",
    "INPUT_COLUMNS",
    "LOGPROB_SENTINEL",
    "META_KEY",
    "block_range",
    "check_index",
    "decode_block",
    "file_meta",
    "read_block",
    "samples_batch",
    "write_file",
]
