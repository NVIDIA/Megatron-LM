# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Sample replay, collation, and token mappings for paired offline KD data."""

import bisect
import concurrent.futures
import itertools
import os
import queue
import random
import threading
from typing import Any, Callable

import numpy as np
import torch

from .format.arrow import block_range, check_index, decode_block, file_meta, read_block
from .format.contract import blend_contract
from .v3_format import LOGPROB_SENTINEL, TOKEN_FIELDS, mismatched_settings, validate_metadata
from .v3_storage import FileName, Storage, list_published, read_index, validate_published_ranges


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
    values: torch.Tensor, indices: torch.Tensor, tail: torch.Tensor, mapping: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply precisely the input token mapping to teacher top-K targets and tails."""
    flat = mapping.reshape(-1)
    valid = flat >= 0
    selected = flat.clamp_min(0)
    shape = (*mapping.shape, values.shape[-1])
    local_values = values.index_select(0, selected).reshape(shape)
    local_indices = indices.index_select(0, selected).reshape(shape)
    local_tail = tail.index_select(0, selected).reshape(mapping.shape)
    if not valid.all():
        local_values = local_values.masked_fill(~valid.reshape(*mapping.shape, 1), -1e3)
        local_indices = local_indices.masked_fill(~valid.reshape(*mapping.shape, 1), 0)
        local_tail = local_tail.masked_fill(~valid.reshape(mapping.shape), 0.0)
    return local_values, local_indices, local_tail


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
    """List published indexes once and return the contiguous published prefix.

    A flush range is published when every writer has published its index. Returns ``(groups, hole)`` where ``groups`` are the published
    ``(start, end)`` sample ranges from ``first_sample`` without a gap, and
    ``hole`` is ``(gap_start, next_published_start, later_groups)`` when published
    ranges exist beyond a gap (e.g. parallel dump jobs that are still writing).
    """
    validate_metadata(metadata)
    contract = metadata["contract"]
    ranges = validate_published_ranges(
        list_published(storage), contract["writers"], contract["writer_mbs"]
    )
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
    """Read one ordered stream of blocks, decoding a bounded window ahead.

    ``items(stopped)`` yields ``(key, (storage, data file, index, block, label))`` with
    increasing keys; consumers request keys in non-decreasing order. Each block is one
    ranged read (of its input columns only, when targets are not needed).
    """

    def __init__(self, reader: Any, name: str, items: Callable):
        self.reader = reader
        self.queue: queue.Queue = queue.Queue(maxsize=reader.prefetch_depth)
        self.current = None
        self.stopped = threading.Event()
        self.thread = threading.Thread(
            target=self._run, args=(items,), name=f"offline-kd-v3-{name}", daemon=True
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

    def _run(self, items: Callable) -> None:
        reader = self.reader
        try:
            for key, (storage, data, index, block, label) in items(self.stopped):
                offset, size = block_range(index, block, inputs_only=not reader.targets)
                raw = storage.read_range(data, offset, size)
                future = reader.pool.submit(reader.decode_rows, index, block, raw, label)
                if not self._put((key, future)):
                    return
        except BaseException as error:  # Surface any failure to the consumer.
            self._put((None, error))

    def block(self, key: tuple) -> tuple[dict, tuple | None]:
        """Return the decoded block for ``key``, advancing the stream as needed."""
        while self.current is None or self.current[0] < key:
            item = self.queue.get()
            if item[0] is None:
                raise item[1]
            self.current = item
        if self.current[0] != key:
            raise RuntimeError(f"Offline KD v3 replay requested {key} after {self.current[0]}")
        return self.current[1].result()

    def close(self) -> None:
        """Stop streaming and release buffered blocks."""
        self.stopped.set()
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.current = None


class _BlockReader:
    """Decoding, vocabulary masking and sample slicing shared by replay and corpus readers."""

    def __init__(
        self,
        contract: dict,
        *,
        targets: bool,
        decode_threads: int,
        prefetch_depth: int,
        student_vocab_size: int | None,
    ):
        self.contract = contract
        self.prefix_stable = contract["prefix_stable"]
        self.isolated = contract["attention_mode"] == "isolated_segments"
        self.targets = targets
        self.student_vocab_size = student_vocab_size
        # Entries whose index the student vocabulary lacks are masked into the ghost bucket.
        self.masked_entries = 0
        self.target_entries = 0
        self._masking_warned = False
        self.prefetch_depth = max(1, prefetch_depth)
        self.pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, decode_threads), thread_name_prefix="offline-kd-v3-decode"
        )
        self.lanes: dict[Any, _Lane] = {}

    def decode_rows(
        self, index: dict, block: int, raw: bytes, label: str
    ) -> tuple[dict, tuple | None]:
        """Decode and validate one block."""
        batch = read_block(index, block, raw, inputs_only=not self.targets)
        # Normalization domain and index encoding belong to this file, not the blend.
        inputs, targets = decode_block(
            batch, vocab_size=file_meta(index)["contract"]["logprob_vocab_size"]
        )
        if inputs["sample_offsets"].numel() - 1 != index["blocks"][block]["rows"]:
            raise ValueError(f"Offline KD v3 block {block} of {label} has wrong rows")
        inputs["segment_offsets"] = torch.cat(
            (torch.zeros(1, dtype=torch.int64), torch.cumsum(inputs["segment_counts"].long(), 0))
        )
        # Target rows exist only at positions with nonzero loss weight, in token order.
        scored = inputs["loss_mask"] != 0
        inputs["target_offsets"] = torch.cat(
            (torch.zeros(1, dtype=torch.int64), torch.cumsum(scored.long(), 0))
        )
        if targets is not None:
            targets = self._mask_unknown_indices(*targets)
        return inputs, targets

    def _mask_unknown_indices(
        self, values: torch.Tensor, indices: torch.Tensor, tail: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = values != LOGPROB_SENTINEL
        self.target_entries += int(valid.sum())
        if self.student_vocab_size is None:
            return values, indices, tail
        unknown = valid & (indices >= self.student_vocab_size)
        count = int(unknown.sum())
        if count:
            self.masked_entries += count
            # Entries the student cannot represent join the tail bucket.
            masked = values.float().masked_fill(~unknown, -torch.inf).logsumexp(-1)
            tail = torch.logaddexp(tail, masked)
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
        return values, indices, tail

    def slice_sample(self, inputs: dict, targets: tuple | None, row: int, sample_id: int) -> dict:
        """Cut one sample (and its target rows) out of a decoded block."""
        a, b = inputs["sample_offsets"][row : row + 2].tolist()
        sample = {field: inputs[field][a:b] for field in TOKEN_FIELDS}
        sample["sample_id"] = sample_id
        sample["attention_mode"] = self.contract["attention_mode"]
        sample["eod_id"] = self.contract["tokenizer"].get("eod_id")
        s0, s1 = inputs["segment_offsets"][row : row + 2].tolist()
        sample["segments"] = inputs["segment_lengths"][s0:s1]
        if targets is not None:
            ta, tb = inputs["target_offsets"][[a, b]].tolist()
            sample["teacher_values"], sample["teacher_indices"], sample["teacher_tail"] = (
                tensor[ta:tb] for tensor in targets
            )
        return sample

    def close(self) -> None:
        """Stop all lanes and the decode pool."""
        for lane in self.lanes.values():
            lane.close()
        self.pool.shutdown(wait=False, cancel_futures=True)


class ReplayReader(_BlockReader):
    """Read samples by logical replay position, streaming each writer's files."""

    def __init__(
        self,
        storage: Storage,
        metadata: dict,
        groups: list[tuple[int, int]],
        *,
        targets: bool = True,
        shuffle: bool = False,
        seed: int = 0,
        passes: int = 1,
        decode_threads: int = 4,
        prefetch_depth: int = 2,
        student_vocab_size: int | None = None,
    ):
        validate_metadata(metadata)
        super().__init__(
            metadata["contract"],
            targets=targets,
            decode_threads=decode_threads,
            prefetch_depth=prefetch_depth,
            student_vocab_size=student_vocab_size,
        )
        self.storage = storage
        self.metadata = metadata
        self.mbs = self.contract["writer_mbs"]
        self.dp = self.contract["writers"]
        # Recorded settings that differ between files are reported once each, never fatal.
        self.reported_differences: set[str] = set()
        # Each pass replays every group once; shuffled passes draw successive orders from one
        # seeded generator, so the first pass matches a single-pass shuffle.
        rng = random.Random(seed)
        self.groups = []
        for _ in range(passes):
            order = [tuple(group) for group in groups]
            if shuffle:
                rng.shuffle(order)
            self.groups.extend(order)
        # A shuffled or repeated replay covers a fixed extent; only a sequential one grows.
        self.fixed = shuffle or passes > 1
        self.indexes: dict[tuple[int, int, int], dict] = {}
        self.index_lock = threading.Lock()
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
        if self.fixed and groups:
            raise RuntimeError("Shuffled or repeated offline KD replay cannot extend its groups")
        with self.condition:
            self.groups.extend(tuple(group) for group in groups)
            self._reindex()
            self.condition.notify_all()

    def wait_for_group(self, index: int, stopped: threading.Event) -> tuple[int, int] | None:
        """Block a lane until group ``index`` is known, or return None at the end."""
        with self.condition:
            while index >= len(self.groups):
                if self.closed or stopped.is_set() or self.fixed:
                    return None
                self.condition.wait(timeout=0.5)
            return self.groups[index]

    def ensure_available(self, end: int) -> None:
        """Fail if the collectively agreed frontier does not cover ``end``; never lists."""
        if end > self.total_samples:
            raise RuntimeError(
                f"Offline KD cache exhausted at {self.total_samples} samples; requested {end}"
            )

    def index(self, dp_rank: int, span: tuple[int, int]) -> dict:
        """Load (once) and check a writer's file index for a flush range.

        The file's settings are checked against the cache before any of its blocks is
        used: required differences fail, recorded ones are reported.
        """
        key = (dp_rank, *span)
        with self.index_lock:
            cached = self.indexes.get(key)
        if cached is not None:
            return cached
        name = FileName(dp_rank, *span)
        index = read_index(self.storage, name)
        check_index(index, self.storage.size(name.data))
        self.check_meta(file_meta(index), dp_rank, span)
        rows = (span[1] - span[0]) // self.dp
        if sum(block["rows"] for block in index["blocks"]) != rows:
            raise ValueError(f"Offline KD v3 file {name.data} does not hold {rows} samples")
        index["first_rows"] = [block["first_row"] for block in index["blocks"]]
        with self.index_lock:
            return self.indexes.setdefault(key, index)

    def check_meta(self, meta: dict, dp_rank: int, span: tuple[int, int]) -> None:
        """Validate a file's settings against the cache."""
        validate_metadata(meta)
        required, recorded = mismatched_settings(meta, self.metadata)
        if required:
            raise ValueError(
                f"Offline KD v3 file dp{dp_rank} {span[0]}-{span[1]} disagrees with the cache on "
                f"{', '.join(required)}; it was written by an incompatible dump job"
            )
        new = [key for key in recorded if key not in self.reported_differences]
        if new:
            self.reported_differences.update(new)
            print(
                f"WARNING: offline KD v3 file dp{dp_rank} {span[0]}-{span[1]} differs from the "
                f"cache in recorded settings ({', '.join(new)}); it is still used",
                flush=True,
            )

    def _lane(self, dp_rank: int, group_index: int, first_block: int) -> _Lane:
        lane = self.lanes.get(dp_rank)
        if lane is None:

            def items(stopped: threading.Event) -> Any:
                group = group_index
                while not stopped.is_set():
                    span = self.wait_for_group(group, stopped)
                    if span is None:
                        return
                    name, index = FileName(dp_rank, *span), self.index(dp_rank, span)
                    label = f"dp{dp_rank} {span[0]}-{span[1]}"
                    for block in range(
                        first_block if group == group_index else 0, len(index["blocks"])
                    ):
                        yield (group, block), (self.storage, name.data, index, block, label)
                    group += 1

            lane = self.lanes[dp_rank] = _Lane(self, f"dp{dp_rank}", items)
        return lane

    def samples(self, positions: list[int], available_end: int) -> list[dict[str, Any]]:
        """Fetch samples at logical replay positions (non-decreasing across calls)."""
        self.ensure_available(available_end)
        samples = []
        for position in positions:
            group_index = bisect.bisect_right(self.ends, position)
            start, _ = self.groups[group_index]
            end = self.groups[group_index][1]
            sid = start + position - self.group_starts[group_index]
            offset = sid - start
            dp_rank = (offset // self.mbs) % self.dp
            row = (offset // (self.mbs * self.dp)) * self.mbs + offset % self.mbs
            first_rows = self.index(dp_rank, (start, end))["first_rows"]
            block = bisect.bisect_right(first_rows, row) - 1
            inputs, targets = self._lane(dp_rank, group_index, block).block((group_index, block))
            samples.append(self.slice_sample(inputs, targets, row - first_rows[block], sid))
        return samples

    def close(self) -> None:
        """Stop all lanes and the decode pool."""
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        super().close()


class CorpusReader(_BlockReader):
    """Read a planned corpus blend (see :mod:`.format.blend`) by global sample position.

    Global sample ``p`` is row ``p // L`` of lane ``p % L``. Each lane streams its
    pieces' blocks in order; a rank only creates the lanes its positions fall in.
    """

    def __init__(
        self,
        root: str,
        manifest: dict,
        *,
        targets: bool = True,
        decode_threads: int = 4,
        prefetch_depth: int = 2,
        student_vocab_size: int | None = None,
    ):
        super().__init__(
            manifest["contract"],
            targets=targets,
            decode_threads=decode_threads,
            prefetch_depth=prefetch_depth,
            student_vocab_size=student_vocab_size,
        )
        self.root = root
        self.manifest = manifest
        self.lanes_count = manifest["lanes"]
        self.total_samples = manifest["lanes"] * manifest["lane_samples"]
        segments = manifest["segments"]
        self.pieces = [segments[segments[:, 0] == lane] for lane in range(self.lanes_count)]
        self.starts = [
            np.concatenate(([0], np.cumsum(pieces[:, 4] - pieces[:, 3])))[:-1].tolist()
            for pieces in self.pieces
        ]
        self.storages: dict[str, Storage] = {}
        self.indexes: dict[int, dict] = {}
        self.index_lock = threading.Lock()

    def _file(self, file_id: int) -> tuple[Storage, FileName, dict]:
        entry = self.manifest["files"][file_id]
        storage = self.storages.setdefault(
            entry["folder"], Storage(os.path.join(self.root, entry["folder"]))
        )
        return storage, FileName(entry["dp_rank"], entry["start"], entry["end"]), entry

    def index(self, file_id: int) -> dict:
        """Load (once) and check a planned file's index against the manifest."""
        with self.index_lock:
            cached = self.indexes.get(file_id)
        if cached is not None:
            return cached
        storage, name, entry = self._file(file_id)
        index = read_index(storage, name)
        if index.get("size") != entry["size"]:
            raise ValueError(
                f"Offline KD corpus file {entry['folder']}/{name.data} changed since the blend "
                "was planned"
            )
        check_index(index, storage.size(name.data))
        metadata = file_meta(index)
        validate_metadata(metadata, corpus=True)
        contract = metadata["contract"]
        shared = blend_contract(contract)
        differing = [key for key in self.contract if shared.get(key) != self.contract[key]]
        if contract["logprob_vocab_size"] != entry["logprob_vocab_size"]:
            differing.append("logprob_vocab_size")
        if differing:
            raise ValueError(
                f"Offline KD corpus file {entry['folder']}/{name.data} differs from the blend on "
                f"{', '.join(differing)}"
            )
        with self.index_lock:
            return self.indexes.setdefault(file_id, index)

    def _lane(self, lane: int, first_piece: int) -> _Lane:
        stream = self.lanes.get(lane)
        if stream is None:

            def items(stopped: threading.Event) -> Any:
                for piece in range(first_piece, len(self.pieces[lane])):
                    if stopped.is_set():
                        return
                    file_id, block = (int(v) for v in self.pieces[lane][piece][1:3])
                    storage, name, entry = self._file(file_id)
                    label = f"{entry['folder']}/{name.data}"
                    yield (piece,), (storage, name.data, self.index(file_id), block, label)

            stream = self.lanes[lane] = _Lane(self, f"lane{lane}", items)
        return stream

    def samples(self, positions: list[int], available_end: int) -> list[dict[str, Any]]:
        """Fetch samples at global positions (non-decreasing per lane across calls)."""
        if available_end > self.total_samples:
            raise RuntimeError(
                f"Offline KD corpus blend holds {self.total_samples} samples; requested "
                f"{available_end}"
            )
        samples = []
        for position in positions:
            lane, offset = position % self.lanes_count, position // self.lanes_count
            piece = bisect.bisect_right(self.starts[lane], offset) - 1
            inputs, targets = self._lane(lane, piece).block((piece,))
            row = int(self.pieces[lane][piece][3]) + offset - self.starts[lane][piece]
            samples.append(self.slice_sample(inputs, targets, row, position))
        return samples


def _layout_sample(
    sample: dict, seq_length: int, *, packed: bool, divisor: int, prefix_stable: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Place a stored (padding-free) sample at the student's fixed length.

    Returns ``(source, positions, boundaries)``: for each output position, the stored
    token it comes from (``-1`` for padding) and its position id within its segment,
    plus the padded segment boundaries for packed samples. Packed segments are padded
    to ``divisor`` and the pack tail extends the last segment, as Megatron SFT does;
    non-packed samples keep their prefix and pad the tail.

    Lengths are checked per sample, whatever length the producer used: a sample that
    fits is used as is; a longer non-packed or single-segment sample keeps its prefix
    when the producer's targets are prefix-stable; anything else is an error.
    """
    length = sample["tokens"].numel()
    single = not packed or len(sample["segments"]) == 1
    if single and length > seq_length and not prefix_stable:
        raise ValueError(
            f"Offline KD sample {sample.get('sample_id')} has {length} tokens, more than the "
            f"student sequence length ({seq_length}), and its targets are not prefix-stable "
            "(e.g. dynamic RoPE scaling), so it cannot be shortened"
        )
    if not packed or (single and length > seq_length):
        kept = min(length, seq_length)
        source = torch.cat((torch.arange(kept), torch.full((seq_length - kept,), -1)))
        # A shortened single-segment pack stays one segment spanning the student length.
        boundaries = torch.tensor([0, seq_length], dtype=torch.int32) if packed else None
        return source, torch.arange(seq_length), boundaries
    lengths = sample["segments"].tolist()
    padded = [-(-n // divisor) * divisor for n in lengths]
    if sum(padded) > seq_length:
        raise ValueError(
            f"Offline KD packed sample {sample.get('sample_id')} needs {sum(padded)} tokens after "
            f"padding segments to a multiple of {divisor}; the student sequence length is "
            f"{seq_length}"
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
    prefix_stable: bool = True,
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
    boundary_rows, values, indices, tails = [], [], [], []
    with_targets = "teacher_values" in samples[0]
    if any(("teacher_values" in sample) != with_targets for sample in samples):
        raise ValueError("Cannot mix offline KD samples with and without targets")
    k = max(sample["teacher_values"].shape[-1] for sample in samples) if with_targets else 0
    for sample in samples:
        source, positions, boundaries = _layout_sample(
            sample, seq_length, packed=packed, divisor=divisor, prefix_stable=prefix_stable
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
            # Stored rows cover only scored tokens; other positions get no KD target.
            scored = sample["loss_mask"] != 0
            row_of = torch.cumsum(scored.long(), 0) - 1
            has_row = real.clone()
            has_row[real] = scored[source[real]]
            rows = row_of[source[has_row]]
            sample_k = sample["teacher_values"].shape[-1]
            row_values = torch.full(
                (seq_length, k), LOGPROB_SENTINEL, dtype=sample["teacher_values"].dtype
            )
            row_indices = torch.zeros((seq_length, k), dtype=torch.int64)
            row_tail = torch.zeros(seq_length, dtype=torch.float32)
            row_values[has_row, :sample_k] = sample["teacher_values"][rows]
            row_indices[has_row, :sample_k] = sample["teacher_indices"][rows]
            row_tail[has_row] = sample["teacher_tail"][rows]
            values.append(row_values)
            indices.append(row_indices)
            tails.append(row_tail)
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
        flat = (torch.cat(values), torch.cat(indices), torch.cat(tails))
        shards = [
            map_targets(*flat, token_map(mapping_inputs, rank, cp_size, **(layout or {})))
            for rank in range(cp_size)
        ]
        result["_kd_values"] = [shard[0] for shard in shards]
        result["_kd_indices"] = [shard[1] for shard in shards]
        result["_kd_tail"] = [shard[2] for shard in shards]
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
        self.unsupervised_samples = 0
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) if prefetch else None
        self._prefetched = None

    def __iter__(self) -> "ReplayIterator":
        return self

    def _load(self, start: int, end: int) -> dict[str, Any]:
        samples = self.reader.samples(list(range(start, start + self.mbs)), end)
        self._note_shortening(samples)
        return collate_samples(
            samples,
            self.seq_length,
            packed=self.reader.isolated,
            divisor=self.divisor,
            cp_size=self.cp_size,
            layout=self.layout,
            prefix_stable=self.reader.prefix_stable,
        )

    def _note_shortening(self, samples: list[dict]) -> None:
        """Report samples whose supervised tokens all lie beyond the student length."""
        for sample in samples:
            weights = sample["loss_mask"]
            if weights[: self.seq_length].any() or not weights[self.seq_length :].any():
                continue
            self.unsupervised_samples += 1
            if self.unsupervised_samples == 1:
                print(
                    f"WARNING: offline KD v3 sample {sample['sample_id']} has no supervised "
                    f"tokens within the student sequence length ({self.seq_length}); prefix "
                    "shortening leaves such samples without loss. Further cases are counted "
                    "in ReplayIterator.unsupervised_samples",
                    flush=True,
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
