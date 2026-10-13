# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Sample replay, collation, and token mappings for paired offline KD data."""

import bisect
import concurrent.futures
import itertools
import json
import queue
import random
import threading
from typing import Any, Callable, Iterator

import torch

from .v3_format import (
    LOGPROB_SENTINEL,
    TOKEN_FIELDS,
    decode,
    fill_input_defaults,
    mismatched_settings,
    unpack_targets,
    validate_inputs,
    validate_metadata,
)
from .v3_storage import (
    DEFAULT_CHUNK_BYTES,
    META_MEMBER,
    Storage,
    complete_ranges,
    iter_tar,
    list_tars,
    member_name,
    tar_name,
)


def merged_boundaries(rows: list[torch.Tensor], sample_lengths: list[int]) -> torch.Tensor:
    """Combine document boundaries without losing sample-relative offsets."""
    pieces = [torch.zeros(1, dtype=torch.int32)]
    offset = 0
    for row, length in zip(rows, sample_lengths):
        pieces.append(row[1:].to(torch.int32).cpu() + offset)
        offset += length
    return torch.cat(pieces)


def token_map(
    inputs: dict[str, Any],
    cp_rank: int,
    cp_size: int,
    *,
    layout: str = "zigzag",
    per_sequence: bool = False,
    hybrid_padded_zigzag: bool = False,
    tp_alignment: int = 1,
) -> torch.Tensor:
    """Return [local_sequence, model_batch] canonical token indices.

    The same map gathers teacher outputs and slices student targets. Prefix
    shortening has already happened before this mapping is constructed.
    """
    offsets = inputs["sample_offsets"]
    lengths = (offsets[1:] - offsets[:-1]).tolist()
    if len(set(lengths)) != 1:
        raise ValueError("Variable-length v3 replay is reserved for a future packing adapter")
    packed = inputs.get("cu_seqlens") is not None
    length = lengths[0]
    indices = torch.arange(int(offsets[-1]), dtype=torch.int64)
    full = (
        indices.reshape(-1, 1) if packed else indices.reshape(len(lengths), length).T.contiguous()
    )
    if cp_size == 1:
        return full
    if layout == "contiguous":
        if full.shape[0] % (2 * cp_size):
            raise ValueError("Offline KD sequence length must be divisible by 2 * student CP")
        return full.chunk(cp_size, dim=0)[cp_rank].contiguous()
    if layout != "zigzag":
        raise ValueError(f"Unsupported offline KD CP layout: {layout}")
    boundaries = None
    if packed:
        boundaries = merged_boundaries(inputs["cu_seqlens"], lengths)
        if hybrid_padded_zigzag:
            from megatron.core.context_parallel.layout import _build_thd_zigzag_metadata

            real = merged_boundaries(inputs["cu_seqlens"], lengths)
            metadata = _build_thd_zigzag_metadata(real, boundaries, cp_size, tp_alignment)
            index = metadata.rank_order_indices.reshape(cp_size, -1)[cp_rank].to(torch.int64)
            return index.reshape(-1, 1)
    if boundaries is None or per_sequence:
        boundaries = torch.tensor([0, full.shape[0]])
    pieces = []
    for a, b in zip(boundaries[:-1].tolist(), boundaries[1:].tolist()):
        if (b - a) % (2 * cp_size):
            raise ValueError("Saved offline KD document padding is incompatible with student CP")
        chunk = (b - a) // (2 * cp_size)
        pieces.extend(
            (
                full[a + cp_rank * chunk : a + (cp_rank + 1) * chunk],
                full[a + (2 * cp_size - cp_rank - 1) * chunk : a + (2 * cp_size - cp_rank) * chunk],
            )
        )
    return torch.cat(pieces, dim=0).contiguous()


def map_targets(
    values: torch.Tensor, indices: torch.Tensor, mapping: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply precisely the input token mapping to teacher top-K targets."""
    flat = mapping.reshape(-1)
    valid = flat >= 0
    selected = flat.clamp_min(0)
    shape = (*mapping.shape, values.shape[-1])
    local_values = values.index_select(0, selected).reshape(shape)
    local_indices = indices.index_select(0, selected).reshape(shape)
    if not valid.all():
        local_values = local_values.masked_fill(~valid.reshape(*mapping.shape, 1), -1e3)
        local_indices = local_indices.masked_fill(~valid.reshape(*mapping.shape, 1), 0)
    return local_values, local_indices


def layout_options(args: Any, *, hybrid: bool = False) -> dict[str, Any]:
    """Match the entrypoint's standard CP partitioning options."""
    layout = "zigzag"
    attention_layout = "zigzag"
    if hybrid:
        from megatron.training.arguments import core_transformer_config_from_args

        config = core_transformer_config_from_args(args)
        layout = config.linear_cp_layout
        attention_layout = config.attention_cp_layout
    return {
        "layout": layout,
        "per_sequence": bool(args.dataloader_inter_document_masking and not args.sft),
        "hybrid_padded_zigzag": hybrid
        and layout == "zigzag"
        and (
            attention_layout == "contiguous"
            or (args.dataloader_inter_document_masking and not args.sft)
        ),
        "tp_alignment": args.tensor_model_parallel_size if args.sequence_parallel else 1,
    }


def discover_groups(
    storage: Storage, metadata: dict
) -> tuple[list[tuple[int, int]], tuple[int, int, int] | None]:
    """List tar names once and return the contiguous published prefix.

    A flush range is published when its inputs and targets tars exist for every
    teacher DP rank. Returns ``(groups, hole)`` where ``groups`` are the published
    ``(start, end)`` sample ranges from ``first_sample`` without a gap, and
    ``hole`` is ``(gap_start, next_published_start, later_groups)`` when published
    ranges exist beyond a gap (e.g. parallel dump jobs that are still writing).
    """
    validate_metadata(metadata)
    contract = metadata["contract"]
    ranges = complete_ranges(list_tars(storage), contract["dp_size_save"])
    gbs = contract["gbs_save"]
    previous = None
    for start, end in ranges:
        if end <= start or (end - start) % gbs:
            raise ValueError(f"Offline KD v3 range {start}-{end} is not whole teacher iterations")
        if previous is not None and start < previous[1]:
            raise ValueError(f"Overlapping offline KD v3 ranges {previous} and {(start, end)}")
        previous = (start, end)
    by_start = dict(ranges)
    groups = []
    cursor = contract["first_sample"]
    while cursor in by_start:
        groups.append((cursor, by_start[cursor]))
        cursor = by_start[cursor]
    later = [span for span in ranges if span[0] > cursor]
    hole = (cursor, later[0][0], len(later)) if later else None
    return groups, hole


def describe_hole(hole: tuple[int, int, int] | None) -> str:
    """Explain a gap in the published prefix."""
    if hole is None:
        return ""
    gap_start, next_start, later = hole
    return (
        f"samples {gap_start}-{next_start} are not yet published, but {later} later range(s) "
        "are; a parallel dump job may still be writing"
    )


class _Lane:
    """Stream one teacher DP rank's tars in replay order, decoding a bounded window ahead.

    Records are consumed in non-decreasing ``(group, record)`` order, so each tar
    is read once, sequentially. Records before the first requested one are skipped
    without decoding (resuming mid-range still streams their bytes).
    """

    def __init__(self, reader: "ReplayReader", dp_rank: int, group_index: int, record_index: int):
        self.reader = reader
        self.dp_rank = dp_rank
        self.queue: queue.Queue = queue.Queue(maxsize=reader.prefetch_depth)
        self.current = None
        self.stopped = threading.Event()
        self.thread = threading.Thread(
            target=self._run,
            args=(group_index, record_index),
            name=f"offline-kd-v3-dp{dp_rank}",
            daemon=True,
        )
        self.thread.start()

    def _put(self, item: tuple) -> bool:
        while not self.stopped.is_set():
            try:
                self.queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _run(self, group_index: int, record_index: int) -> None:
        try:
            index = group_index
            while not self.stopped.is_set():
                span = self.reader.wait_for_group(index, self.stopped)
                if span is None:
                    return
                first = record_index if index == group_index else 0
                for record, future in self._stream_group(index, span, first):
                    if not self._put((index, record, future)):
                        return
                index += 1
        except BaseException as error:  # Surface any failure to the consumer.
            self._put((None, None, error))

    def _stream_group(
        self, index: int, span: tuple[int, int], first: int
    ) -> Iterator[tuple[int, concurrent.futures.Future]]:
        reader = self.reader
        start, end = span
        kinds = ("inputs", "targets") if reader.targets else ("inputs",)
        streams = [
            iter_tar(reader.storage, tar_name(self.dp_rank, start, end, kind), reader.chunk_bytes)
            for kind in kinds
        ]
        try:
            generations = set()
            for kind, stream in zip(kinds, streams):
                name, raw = next(stream, (None, None))
                if name != META_MEMBER:
                    raise ValueError(
                        f"Offline KD v3 tar for dp{self.dp_rank} {span} lacks metadata"
                    )
                generations.add(reader.check_meta(raw, kind, self.dp_rank, span))
            if len(generations) != 1:
                raise ValueError(
                    f"Offline KD v3 inputs and targets for dp{self.dp_rank} {span} "
                    "were written by different dump jobs"
                )
            generation = generations.pop()
            for record, record_start in enumerate(range(start, end, reader.gbs)):
                record_end = record_start + reader.gbs
                payloads = []
                for kind, stream in zip(kinds, streams):
                    name, raw = next(stream, (None, None))
                    if name != member_name(record_start, record_end, kind):
                        raise ValueError(
                            f"Offline KD v3 tar for dp{self.dp_rank} {span} has member {name}; "
                            f"expected {member_name(record_start, record_end, kind)}"
                        )
                    payloads.append(raw)
                if record < first:
                    continue
                yield record, reader.pool.submit(
                    reader.decode_record,
                    self.dp_rank,
                    record_start,
                    record_end,
                    generation,
                    *payloads,
                )
            for stream in streams:
                if next(stream, None) is not None:
                    raise ValueError(
                        f"Offline KD v3 tar for dp{self.dp_rank} {span} has extra members"
                    )
        finally:
            for stream in streams:
                stream.close()

    def record(self, group_index: int, record_index: int) -> tuple[dict, tuple | None]:
        """Return a decoded record, advancing the stream as needed."""
        wanted = (group_index, record_index)
        while self.current is None or self.current[:2] < wanted:
            item = self.queue.get()
            if item[0] is None:
                raise item[2]
            self.current = item
        if self.current[:2] != wanted:
            raise RuntimeError(f"Offline KD v3 replay requested {wanted} after {self.current[:2]}")
        return self.current[2].result()

    def close(self) -> None:
        """Stop streaming and release buffered records."""
        self.stopped.set()
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.current = None


class ReplayReader:
    """Read records by logical replay position, streaming each teacher DP rank's tars."""

    def __init__(
        self,
        storage: Storage,
        metadata: dict,
        groups: list[tuple[int, int]],
        *,
        targets: bool = True,
        shuffle: bool = False,
        seed: int = 0,
        decode_threads: int = 4,
        prefetch_depth: int = 2,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
        student_vocab_size: int | None = None,
    ):
        validate_metadata(metadata)
        self.storage = storage
        self.metadata = metadata
        self.contract = metadata["contract"]
        self.gbs = self.contract["gbs_save"]
        self.mbs = self.contract["mbs_save"]
        self.dp = self.contract["dp_size_save"]
        self.isolated = self.contract["attention_mode"] == "isolated_segments"
        self.student_vocab_size = student_vocab_size
        # Entries whose index the student vocabulary lacks are masked into the ghost bucket.
        self.masked_entries = 0
        self.target_entries = 0
        self._masking_warned = False
        self.groups = [tuple(group) for group in groups]
        if shuffle:
            random.Random(seed).shuffle(self.groups)
        self.targets = targets
        self.shuffle = shuffle
        self.prefetch_depth = max(1, prefetch_depth)
        self.chunk_bytes = chunk_bytes
        self.pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, decode_threads), thread_name_prefix="offline-kd-v3-decode"
        )
        self.lanes: dict[int, _Lane] = {}
        self.closed = False
        self.condition = threading.Condition()
        self._reindex()

    def _reindex(self) -> None:
        self.ends = []
        self.group_starts = []
        logical_start = 0
        for start, end in self.groups:
            self.group_starts.append(logical_start)
            logical_start += end - start
            self.ends.append(logical_start)

    @property
    def total_samples(self) -> int:
        """Number of currently available samples in replay order."""
        return self.ends[-1] if self.ends else 0

    def extend(self, groups: list[tuple[int, int]]) -> None:
        """Append newly published sequential groups (agreed collectively by all ranks)."""
        if self.shuffle and groups:
            raise RuntimeError("Shuffled offline KD replay cannot extend its group order")
        with self.condition:
            self.groups.extend(tuple(group) for group in groups)
            self._reindex()
            self.condition.notify_all()

    def wait_for_group(self, index: int, stopped: threading.Event) -> tuple[int, int] | None:
        """Block a lane until group ``index`` is known, or return None at the end."""
        with self.condition:
            while index >= len(self.groups):
                if self.closed or stopped.is_set() or self.shuffle:
                    return None
                self.condition.wait(timeout=0.5)
            return self.groups[index]

    def ensure_available(self, end: int) -> None:
        """Fail if the collectively agreed frontier does not cover ``end``; never lists."""
        if end > self.total_samples:
            raise RuntimeError(
                f"Offline KD cache exhausted at {self.total_samples} samples; requested {end}"
            )

    def check_meta(self, raw: bytes, kind: str, dp_rank: int, span: tuple[int, int]) -> str:
        """Validate a tar's ``_meta.json`` against the cache and return its job generation."""
        meta = json.loads(raw)
        validate_metadata(meta)
        mismatched = mismatched_settings(meta, self.metadata)
        if mismatched:
            raise ValueError(
                f"Offline KD v3 tar dp{dp_rank} {span[0]}-{span[1]} ({kind}) disagrees with the "
                f"cache on {', '.join(mismatched)}; it was written by an incompatible dump job"
            )
        if (meta.get("kind"), meta.get("dp_rank"), meta.get("range")) != (
            kind,
            dp_rank,
            list(span),
        ):
            raise ValueError(f"Offline KD v3 tar dp{dp_rank} {span} ({kind}) has wrong metadata")
        return meta["generation"]

    def decode_record(
        self,
        dp_rank: int,
        start: int,
        end: int,
        generation: str,
        raw_inputs: bytes,
        raw_targets: bytes | None = None,
    ) -> tuple[dict, tuple | None]:
        """Decode and validate one teacher iteration's paired members."""
        record_id = f"{generation}:dp{dp_rank}:{start}-{end}"
        inputs = decode(raw_inputs)
        validate_inputs(inputs)
        if self.isolated and inputs.get("segment_lengths") is None:
            raise ValueError("Offline KD isolated_segments requires saved segments")
        if inputs.get("record_id") != record_id:
            raise ValueError(f"Offline KD input record {inputs.get('record_id')} != {record_id}")
        if inputs["sample_offsets"].numel() - 1 != (end - start) // self.dp:
            raise ValueError(f"Offline KD record {record_id} has the wrong sample count")
        targets = None
        if raw_targets is not None:
            payload = decode(raw_targets)
            if payload.get("record_id") != record_id:
                raise ValueError("Offline KD inputs and targets belong to different records")
            targets = unpack_targets(payload, vocab_size=self.contract["logprob_vocab_size"])
            if targets[0].shape[0] != inputs["tokens"].numel():
                raise ValueError("Offline KD input/target token counts disagree")
            targets = self._mask_unknown_indices(*targets)
        inputs = fill_input_defaults(inputs)
        inputs["segment_offsets"] = torch.cat(
            (torch.zeros(1, dtype=torch.int64), torch.cumsum(inputs["segment_counts"].long(), 0))
        )
        return inputs, targets

    def _mask_unknown_indices(
        self, values: torch.Tensor, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = values != LOGPROB_SENTINEL
        self.target_entries += int(valid.sum())
        if self.student_vocab_size is None:
            return values, indices
        unknown = valid & (indices >= self.student_vocab_size)
        count = int(unknown.sum())
        if count:
            self.masked_entries += count
            values = values.masked_fill(unknown, LOGPROB_SENTINEL)
            indices = indices.masked_fill(unknown, 0)
            fraction = self.masked_entries / max(1, self.target_entries)
            if fraction > 1e-4 and not self._masking_warned:
                self._masking_warned = True
                print(
                    f"WARNING: offline KD v3 masked {self.masked_entries} teacher target entries "
                    f"({fraction:.2%}) with indices beyond the student vocabulary "
                    f"({self.student_vocab_size}); check the teacher/student vocabularies",
                    flush=True,
                )
        return values, indices

    def _lane(self, dp_rank: int, group_index: int, record_index: int) -> _Lane:
        lane = self.lanes.get(dp_rank)
        if lane is None:
            lane = self.lanes[dp_rank] = _Lane(self, dp_rank, group_index, record_index)
        return lane

    def samples(self, positions: list[int], available_end: int) -> list[dict[str, Any]]:
        """Fetch samples at logical replay positions (non-decreasing across calls)."""
        self.ensure_available(available_end)
        samples = []
        for position in positions:
            group_index = bisect.bisect_right(self.ends, position)
            start, _ = self.groups[group_index]
            sid = start + position - self.group_starts[group_index]
            record_index = (sid - start) // self.gbs
            offset = sid - (start + record_index * self.gbs)
            dp_rank = (offset // self.mbs) % self.dp
            row = (offset // (self.mbs * self.dp)) * self.mbs + offset % self.mbs
            inputs, targets = self._lane(dp_rank, group_index, record_index).record(
                group_index, record_index
            )
            a, b = inputs["sample_offsets"][row : row + 2].tolist()
            sample = {field: inputs[field][a:b] for field in TOKEN_FIELDS}
            sample["sample_id"] = sid
            s0, s1 = inputs["segment_offsets"][row : row + 2].tolist()
            sample["segments"] = inputs["segment_lengths"][s0:s1]
            if targets is not None:
                sample["teacher_values"], sample["teacher_indices"] = (
                    tensor[a:b] for tensor in targets
                )
            samples.append(sample)
        return samples

    def close(self) -> None:
        """Stop all lanes and the decode pool."""
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        for lane in self.lanes.values():
            lane.close()
        self.pool.shutdown(wait=False, cancel_futures=True)


def _layout_sample(
    sample: dict, seq_length: int, *, packed: bool, divisor: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Place a stored (padding-free) sample at the student's fixed length.

    Returns ``(source, positions, boundaries)``: for each output position, the stored
    token it comes from (``-1`` for padding) and its position id within its segment,
    plus the padded segment boundaries for packed samples. Packed segments are padded
    to ``divisor`` and the pack tail extends the last segment, as Megatron SFT does;
    non-packed samples keep their prefix and pad the tail.
    """
    length = sample["tokens"].numel()
    if not packed:
        kept = min(length, seq_length)
        source = torch.cat((torch.arange(kept), torch.full((seq_length - kept,), -1)))
        return source, torch.arange(seq_length), None
    lengths = sample["segments"].tolist()
    padded = [-(-n // divisor) * divisor for n in lengths]
    if sum(padded) > seq_length:
        raise ValueError(
            f"Offline KD packed sample needs {sum(padded)} tokens after padding segments to a "
            f"multiple of {divisor}; the student sequence length is {seq_length}"
        )
    if padded:
        padded[-1] += seq_length - sum(padded)
    else:
        padded = [seq_length]  # Fully stripped sample: one padding segment.
    pieces, positions, cursor = [], [], 0
    for real, total in zip(lengths + [0] * (len(padded) - len(lengths)), padded):
        pieces.extend((torch.arange(cursor, cursor + real), torch.full((total - real,), -1)))
        positions.append(torch.arange(total))
        cursor += real
    boundaries = torch.tensor([0, *itertools.accumulate(padded)], dtype=torch.int32)
    return torch.cat(pieces), torch.cat(positions), boundaries


def collate_samples(
    samples: list[dict],
    seq_length: int,
    *,
    packed: bool = False,
    divisor: int = 1,
    cp_size: int = 1,
    layout: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a student CPU microbatch from stored samples.

    Stored samples carry no padding: segments are padded back for the student (see
    :func:`_layout_sample`), padding positions get zero loss weight and sentinel
    targets, and teacher targets are mapped into one ``[local_sequence, batch, K]``
    shard per student CP rank with exactly the token mapping used for the inputs.
    """
    if not samples:
        raise ValueError("Cannot collate an empty offline KD microbatch")
    fields = {field: [] for field in TOKEN_FIELDS}
    boundary_rows, values, indices = [], [], []
    with_targets = "teacher_values" in samples[0]
    for sample in samples:
        source, positions, boundaries = _layout_sample(
            sample, seq_length, packed=packed, divisor=divisor
        )
        real = source >= 0
        take = source.clamp_min(0)
        for field in TOKEN_FIELDS:
            dtype = torch.float32 if field == "loss_mask" else torch.int64
            column = sample[field].to(dtype)[take] if sample["tokens"].numel() else None
            filled = (
                positions.to(dtype)
                if field == "position_ids"
                else torch.zeros(seq_length, dtype=dtype)
            )
            if column is not None:
                filled = torch.where(real, column, filled)
            fields[field].append(filled)
        if with_targets:
            k = sample["teacher_values"].shape[-1]
            row_values = torch.full(
                (seq_length, k), LOGPROB_SENTINEL, dtype=sample["teacher_values"].dtype
            )
            row_indices = torch.zeros((seq_length, k), dtype=torch.int64)
            row_values[real] = sample["teacher_values"][source[real]]
            row_indices[real] = sample["teacher_indices"][source[real]]
            values.append(row_values)
            indices.append(row_indices)
        boundary_rows.append(boundaries)
    result = {field: torch.stack(rows) for field, rows in fields.items()}
    result["cu_seqlens_padded"] = None
    if packed:
        width = max(row.numel() for row in boundary_rows)
        result["cu_seqlens"] = torch.stack(
            [torch.cat((row, row[-1:].expand(width - row.numel()))) for row in boundary_rows]
        )
        result["max_seqlen"] = torch.tensor(
            [int((row[1:] - row[:-1]).max()) for row in boundary_rows], dtype=torch.int32
        )
    else:
        result["cu_seqlens"] = result["max_seqlen"] = None
    result["_kd_sample_ids"] = [sample["sample_id"] for sample in samples]
    if with_targets:
        mapping_inputs = {
            "sample_offsets": torch.arange(len(samples) + 1, dtype=torch.int64) * seq_length,
            "cu_seqlens": boundary_rows if packed else None,
        }
        flat_values, flat_indices = torch.cat(values), torch.cat(indices)
        shards = [
            map_targets(
                flat_values,
                flat_indices,
                token_map(mapping_inputs, rank, cp_size, **(layout or {})),
            )
            for rank in range(cp_size)
        ]
        result["_kd_values"] = [shard[0] for shard in shards]
        result["_kd_indices"] = [shard[1] for shard in shards]
    return result


class ReplayIterator:
    """Consume student batches using the current ramp-up microbatch count."""

    def __init__(
        self,
        reader: ReplayReader,
        *,
        consumed: int,
        dp_rank: int,
        dp_size: int,
        micro_batch_size: int,
        seq_length: int,
        num_microbatches: Callable[[], int],
        prefetch: bool = False,
        cp_size: int = 1,
        layout: dict[str, Any] | None = None,
        divisor: int = 1,
    ):
        self.reader = reader
        self.consumed = consumed
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.mbs = micro_batch_size
        self.seq_length = seq_length
        self.num_microbatches = num_microbatches
        self.cp_size = cp_size
        self.layout = layout
        self.divisor = divisor
        self._microbatch = 0
        self._count = 0
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) if prefetch else None
        self._prefetched = None

    def __iter__(self) -> "ReplayIterator":
        return self

    def _load(self, start: int, end: int) -> dict[str, Any]:
        samples = self.reader.samples(list(range(start, start + self.mbs)), end)
        return collate_samples(
            samples,
            self.seq_length,
            packed=self.reader.isolated,
            divisor=self.divisor,
            cp_size=self.cp_size,
            layout=self.layout,
        )

    def __next__(self) -> dict[str, Any]:
        if self._microbatch == 0:
            self._count = self.num_microbatches()
        end = self.consumed + self._count * self.mbs * self.dp_size
        start = self.consumed + (self._microbatch * self.dp_size + self.dp_rank) * self.mbs
        if self._prefetched is None:
            batch = self._load(start, end)
        else:
            batch = self._prefetched.result()
            self._prefetched = None
        self._microbatch += 1
        if self._microbatch == self._count:
            self.consumed = end
            self._microbatch = 0
        elif self._executor is not None:
            # Stay within this iteration: the next iteration's ramp-up count is
            # determined only when Megatron updates its batch calculator. The
            # per-rank streams already decode ahead across iterations.
            next_start = self.consumed + (self._microbatch * self.dp_size + self.dp_rank) * self.mbs
            self._prefetched = self._executor.submit(self._load, next_start, end)
        return batch

    def close(self) -> None:
        """Drain the prefetch worker and stop the reader's streams."""
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self.reader.close()

    def __del__(self):
        if getattr(self, "_executor", None) is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
