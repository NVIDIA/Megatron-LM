# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Offline KD sample files: Arrow IPC with zstd-compressed buffers, plus a JSON index.

A data file holds one row per sample: the input columns first, then the teacher's
target rows (only at positions with nonzero loss weight). Rows are grouped into record
batches ("blocks") of bounded size. The settings live in the schema metadata.

A flush or chunk ``index.json``, written after all its data files are complete, holds
the shared settings once, each distinct Arrow schema once (base64, without settings),
and per data file its ``path``, ``schema`` number, ``size`` and blocks (rows, byte
ranges and supervised-token count). A reader can therefore fetch any block, or only its
input columns, with one ranged read and decode it with the shared schema, never
touching the rest of the data file. The data files keep their own footers, so each
stays self-describing.

Columns start with the input columns in their fixed order (input-only reads select them
by position), followed by ``values``, ``indices_low``, optionally ``indices_high``, and
``tail``. Readers select the other columns by name and ignore ones they do not know;
data that must not be ignored is declared in the contract's ``features``.
"""

import base64
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
DEFAULT_BLOCK_BYTES = 32 << 20
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
    tail: torch.Tensor,
    *,
    vocab_size: int,
) -> pa.RecordBatch:
    """Build one row per sample from flattened inputs and target rows.

    ``inputs`` holds ``sample_offsets`` (``[samples + 1]``), the token-aligned
    ``tokens``, ``labels``, ``position_ids`` and ``loss_mask`` (bool, or float32
    weights), and ``segment_lengths``/``segment_counts``. Target row ``i`` belongs to the
    ``i``-th position with nonzero loss weight, in token order. ``tail`` is each row's
    float32 log probability outside the retained entries, computed before the values
    are cast to the save dtype.
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
    if tuple(tail.shape) != values.shape[:1]:
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
    """Serialize sample batches into ``(data file bytes, part index)``.

    The part index holds the file's base64 ``schema`` (columns only; :func:`share_schemas`
    numbers the distinct ones of an index), ``size`` and ``blocks``. Rows are regrouped into blocks of at most ``block_bytes`` uncompressed bytes (a
    larger single row gets a block of its own), independent of how they were produced.
    """
    if not batches:
        raise ValueError("Cannot write an empty offline KD file")
    schema = _common_schema(batches)
    table = pa.Table.from_batches([batch.cast(schema) for batch in batches]).combine_chunks()
    columns = base64.b64encode(schema.serialize().to_pybytes()).decode()
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
    blocks = []
    for _, rows, begin, end, input_body, block in spans:
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
                "rows": rows,
                "offset": begin + offset,
                "length": length,
                "input_length": length - message.body.size + input_body,
                "supervised": len(block.column("values").flatten()),
            }
        )
    return data, {"schema": columns, "size": len(data), "blocks": blocks}


def share_schemas(parts: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    """Number the distinct schemas of an index's parts: ``(schemas, parts)``."""
    schemas: dict[str, int] = {}
    numbered = [
        {**part, "schema": schemas.setdefault(part["schema"], len(schemas))} for part in parts
    ]
    return list(schemas), numbered


def load_schema(encoded: str) -> pa.Schema:
    """Decode an index schema; it must hold the format's columns, input columns first."""
    try:
        schema = pa.ipc.read_schema(pa.py_buffer(base64.b64decode(encoded, validate=True)))
    except Exception as error:
        raise ValueError("Invalid offline KD index schema") from error
    names = tuple(schema.names)
    missing = [name for name in ("values", "indices_low", "tail") if name not in names]
    if names[: len(INPUT_COLUMNS)] != INPUT_COLUMNS or missing:
        raise ValueError(
            f"Offline KD columns {', '.join(names)} must start with "
            f"{', '.join(INPUT_COLUMNS)} and include values, indices_low and tail"
        )
    return schema


def check_parts(index: dict[str, Any], label: str) -> list[int]:
    """Validate an index's schemas and parts; return each part's row count."""
    schemas, parts = index.get("schemas"), index.get("parts")
    if not isinstance(schemas, list) or not schemas or not isinstance(parts, list) or not parts:
        raise ValueError(f"{label} needs nonempty schemas and parts")
    for schema in schemas:
        load_schema(schema)
    rows = []
    for part in parts:
        path, size, blocks = part.get("path"), part.get("size"), part.get("blocks")
        if type(part.get("schema")) is not int or not 0 <= part["schema"] < len(schemas):
            raise ValueError(f"{label} part {path} names an unknown schema")
        if type(size) is not int or not isinstance(blocks, list) or not blocks:
            raise ValueError(f"{label} part {path} needs a size and blocks")
        byte_end = _MAGIC_BYTES
        for block in blocks:
            if (
                block["rows"] <= 0
                or block["offset"] < byte_end
                or block["length"] <= 0
                or block["offset"] + block["length"] > size
                or not 0 < block["input_length"] <= block["length"]
            ):
                raise ValueError(f"Invalid block rows or byte ranges in {label} part {path}")
            byte_end = block["offset"] + block["length"]
        rows.append(sum(block["rows"] for block in blocks))
    return rows


def block_range(block: dict[str, Any], *, inputs_only: bool) -> tuple[int, int]:
    """Return ``(offset, size)`` of a block, or of its input columns only."""
    return block["offset"], block["input_length" if inputs_only else "length"]


def read_block_bytes(
    schema: pa.Schema, raw: bytes, length: int, *, inputs_only: bool = False
) -> pa.RecordBatch:
    """Decode a block from its :func:`block_range` bytes using its index schema.

    ``raw`` is the block's ``length``-byte record batch message or, for the input
    columns only, its ``input_length`` prefix.
    """
    if not inputs_only:
        if len(raw) != length:
            raise ValueError(f"Offline KD block has {len(raw)} of its {length} bytes")
        return pa.ipc.read_record_batch(pa.ipc.read_message(pa.py_buffer(raw)), schema)
    # A one-message stream whose unread target bytes are zero (untouched) pages: the
    # reader only decompresses the included input columns, which prefix the body.
    header = schema.serialize()
    stream = np.zeros(header.size + length, dtype=np.uint8)
    stream[: header.size] = np.frombuffer(header, dtype=np.uint8)
    stream[header.size : header.size + len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    options = pa.ipc.IpcReadOptions(included_fields=list(range(len(INPUT_COLUMNS))))
    return pa.ipc.open_stream(pa.py_buffer(stream), options=options).read_next_batch()


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
    """Convert a block to flattened inputs and, unless it was read inputs-only,
    ``(values, indices, tail)``.

    Inputs carry ``sample_offsets``, the token-aligned columns, ``segment_lengths`` and
    ``segment_counts``. Every sample's segments must cover its tokens and its target
    rows must match its nonzero loss weights. ``vocab_size``, the tokenizer's size,
    bounds every valid target index.
    """
    names = set(batch.schema.names)
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
        raise ValueError("Offline KD target index encoding disagrees with the tokenizer size")
    tail = _flat(batch.column("tail"))[1]
    scored = np.concatenate(([0], np.cumsum(_numpy(inputs["loss_mask"]) != 0)))[offsets]
    if not np.array_equal(row_offsets, scored):
        raise ValueError("Offline KD target rows disagree with the number of nonzero loss weights")
    targets = (_tensor(values), _tensor(join_indices(low, high)), _tensor(tail))
    return inputs, check_targets(*targets, vocab_size=vocab_size)


__all__ = [
    "DEFAULT_BLOCK_BYTES",
    "INPUT_COLUMNS",
    "LOGPROB_SENTINEL",
    "META_KEY",
    "block_range",
    "check_parts",
    "decode_block",
    "load_schema",
    "read_block_bytes",
    "samples_batch",
    "share_schemas",
    "write_file",
]
