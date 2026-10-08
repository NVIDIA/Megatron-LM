# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Sample-cursor replay independent of teacher iteration and batch sizes."""

import bisect
import concurrent.futures
import json
import logging
import random
import time
from collections import OrderedDict
from typing import Any, Callable

import torch

from .v3_format import (
    BOUNDARY_FIELDS,
    TOKEN_FIELDS,
    decode,
    digest,
    unpack_targets,
    validate_inputs,
)
from .v3_storage import COMPLETE_FILE, Storage, read_members

logger = logging.getLogger(__name__)


def discover_groups(storage: Storage, metadata: dict) -> list[list[dict]]:
    """Expose only complete, immutable teacher DP shard groups."""
    groups = {}
    for name in storage.list("dp*__*.tar.ready.json"):
        try:
            descriptor = storage.read_json(name)
        except FileNotFoundError:
            # A resuming teacher may retire an incomplete, unreadable group
            # between the listing and this read. Complete-prefix validation
            # still detects removal of any previously consumed group.
            continue
        if descriptor["metadata"] != metadata:
            raise ValueError("Offline KD v3 directory mixes incompatible cache generations")
        records = descriptor["records"]
        if not records:
            raise ValueError("Empty offline KD shard descriptor")
        key = (records[0]["start"], records[-1]["end"])
        dp_rank = int(descriptor["tar"].split("__")[0][2:])
        if dp_rank in groups.setdefault(key, {}):
            raise ValueError("Duplicate offline KD shard for a saved DP rank")
        groups[key][dp_rank] = descriptor
    result = []
    expected_start = metadata["first_sample"]
    for (start, end), ranks in sorted(groups.items()):
        if start < expected_start:
            raise ValueError("Overlapping offline KD v3 shard groups")
        if start != expected_start or set(ranks) != set(range(metadata["dp_size_save"])):
            # Later groups may finish first. Never consume across a hole.
            break
        group = [ranks[rank] for rank in range(metadata["dp_size_save"])]
        ranges = [(r["start"], r["end"]) for r in group[0]["records"]]
        cursor = start
        for a, b in ranges:
            if a != cursor or b <= a:
                raise ValueError("Offline KD record ranges are not contiguous")
            cursor = b
        if cursor != end or any(
            [(r["start"], r["end"]) for r in d["records"]] != ranges for d in group
        ):
            raise ValueError("Offline KD DP descriptors disagree on iteration ranges")
        for i, (a, b) in enumerate(ranges):
            ids = [sid for d in group for sid in d["records"][i]["sample_ids"]]
            if sorted(ids) != list(range(a, b)):
                raise ValueError("Offline KD shard group has missing or duplicated samples")
            mbs, dp = metadata["mbs_save"], metadata["dp_size_save"]
            if (b - a) % (mbs * dp):
                raise ValueError("Offline KD saved iteration has incomplete microbatches")
            for rank, descriptor in enumerate(group):
                expected_ids = [
                    a + (mb * dp + rank) * mbs + row
                    for mb in range((b - a) // (mbs * dp))
                    for row in range(mbs)
                ]
                if descriptor["records"][i]["sample_ids"] != expected_ids:
                    raise ValueError(
                        "Offline KD sample IDs disagree with the saved DP/microbatch layout"
                    )
        result.append(group)
        expected_start = end
    return result


def manifest_digest(groups: list[list[dict]]) -> str:
    """Fingerprint the complete replay snapshot, including member checksums."""
    return digest(json.dumps(groups, sort_keys=True, separators=(",", ":")).encode())


class ReplayReader:
    """Read records with bounded decoding and optional live-cache following."""

    def __init__(
        self,
        storage: Storage,
        metadata: dict,
        groups: list[list[dict]],
        *,
        targets: bool = True,
        shuffle: bool = False,
        seed: int = 0,
        follow: bool = False,
        timeout: float = 1800,
        poll_interval: float = 10,
        decode_threads: int = 4,
    ):
        if shuffle and follow:
            raise ValueError("Cannot shuffle a growing offline KD cache")
        self.storage = storage
        self.metadata = metadata
        self.groups = list(groups)
        if shuffle:
            random.Random(seed).shuffle(self.groups)
        self.targets = targets
        self.follow = follow
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.decode_threads = max(1, decode_threads)
        self._cache = OrderedDict()
        self._reindex()

    def _reindex(self) -> None:
        self.ends = []
        self.group_starts = []
        self.records = []
        logical_start = 0
        for group in self.groups:
            records = group[0]["records"]
            self.group_starts.append(logical_start)
            # Index iterations, not one Python reference tuple per corpus sample.
            self.records.append([record["end"] for record in records])
            logical_start += records[-1]["end"] - records[0]["start"]
            self.ends.append(logical_start)

    @property
    def total_samples(self) -> int:
        """Number of currently published samples in replay order."""
        return self.ends[-1] if self.ends else 0

    def ensure_available(self, end: int) -> None:
        """Wait for committed records, distinguishing exhaustion from delay."""
        deadline = time.monotonic() + self.timeout
        warned = False
        while end > self.total_samples:
            if not self.follow:
                raise RuntimeError(
                    f"Offline KD snapshot exhausted at {self.total_samples} samples; requested {end}"
                )
            fresh = discover_groups(self.storage, self.metadata)
            if fresh[: len(self.groups)] != self.groups:
                raise RuntimeError("Previously published offline KD data changed during following")
            self.groups = fresh
            self._reindex()
            if end <= self.total_samples:
                return
            if self.storage.exists(COMPLETE_FILE):
                complete = self.storage.read_json(COMPLETE_FILE)
                if complete["generation"] != self.metadata["generation"]:
                    raise ValueError("Offline KD completion marker belongs to another generation")
                raise RuntimeError(
                    f"Teacher completed with only {self.total_samples} replay samples; requested {end}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for offline KD samples through {end}")
            if not warned:
                logger.info("Waiting for teacher publication through replay sample %s", end)
                warned = True
            time.sleep(min(self.poll_interval, max(0, deadline - time.monotonic())))

    def _decode_record(self, descriptor: dict, record: dict) -> tuple[dict, tuple | None]:
        key = (record["start"], record["end"])
        members = read_members(self.storage, descriptor, self.targets, {key})[key]
        inputs = decode(members["inputs"], record["inputs"]["sha256"])
        validate_inputs(inputs)
        if (
            inputs.get("record_id") != record["record_id"]
            or inputs.get("sample_ids") != record["sample_ids"]
        ):
            raise ValueError("Offline KD input record identity disagrees with its descriptor")
        if inputs["sample_offsets"].numel() - 1 != len(record["sample_ids"]):
            raise ValueError("Offline KD descriptor/input sample counts disagree")
        targets = None
        if self.targets:
            payload = decode(members["targets"], record["targets"]["sha256"])
            if payload.get("record_id") != inputs["record_id"]:
                raise ValueError("Offline KD inputs and targets belong to different records")
            targets = unpack_targets(payload)
        if targets is not None and targets[0].shape[0] != inputs["tokens"].numel():
            raise ValueError("Offline KD input/target token counts disagree")
        return inputs, targets

    def samples(self, positions: list[int], available_end: int) -> list[dict[str, Any]]:
        """Fetch selected records in logical replay order, decoding once per member."""
        self.ensure_available(available_end)
        refs = []
        pending = {}
        for position in positions:
            group_index = bisect.bisect_right(self.ends, position)
            group = self.groups[group_index]
            sid = group[0]["records"][0]["start"] + position - self.group_starts[group_index]
            record_index = bisect.bisect_right(self.records[group_index], sid)
            start = group[0]["records"][record_index]["start"]
            mbs, dp = self.metadata["mbs_save"], self.metadata["dp_size_save"]
            offset = sid - start
            descriptor = group[(offset // mbs) % dp]
            record = descriptor["records"][record_index]
            row = (offset // (mbs * dp)) * mbs + offset % mbs
            key = (descriptor["tar"], record["start"])
            refs.append((key, row, record["sample_ids"][row]))
            if key not in self._cache:
                pending[key] = (descriptor, record)
        if pending:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.decode_threads) as pool:
                futures = {
                    key: pool.submit(self._decode_record, *source)
                    for key, source in pending.items()
                }
                for key, future in futures.items():
                    self._cache[key] = future.result()
        samples = []
        for key, row, sid in refs:
            inputs, targets = self._cache[key]
            self._cache.move_to_end(key)
            a, b = inputs["sample_offsets"][row : row + 2].tolist()
            sample = {field: inputs[field][a:b] for field in TOKEN_FIELDS}
            sample["sample_id"] = sid
            for field in BOUNDARY_FIELDS:
                sample[field] = None if inputs.get(field) is None else inputs[field][row]
            if targets is not None:
                sample["teacher_values"], sample["teacher_indices"] = (
                    tensor[a:b] for tensor in targets
                )
            samples.append(sample)
        # Decode at most the current batch's source members plus a small reusable
        # working set, not every payload in a multi-iteration tar group.
        while len(self._cache) > max(len(pending), self.metadata["dp_size_save"], 1):
            self._cache.popitem(last=False)
        return samples


def collate_samples(samples: list[dict], seq_length: int) -> dict[str, Any]:
    """Build a student CPU microbatch, shortening only non-packed prefixes."""
    if not samples:
        raise ValueError("Cannot collate an empty offline KD microbatch")
    result = {}
    packed = samples[0]["cu_seqlens"] is not None
    for sample in samples:
        length = sample["tokens"].numel()
        if (sample["cu_seqlens"] is not None) != packed:
            raise ValueError("Mixed packed/unpacked offline KD samples")
        if length < seq_length or (packed and length != seq_length):
            raise ValueError(
                "Offline KD sequence length mismatch; only non-packed prefix shortening is supported"
            )
    for field in TOKEN_FIELDS:
        dtype = torch.float32 if field == "loss_mask" else torch.int64
        result[field] = torch.stack([sample[field][:seq_length].to(dtype) for sample in samples])
    for field in BOUNDARY_FIELDS:
        rows = [sample[field] for sample in samples]
        if all(row is None for row in rows):
            result[field] = None
        elif any(row is None for row in rows):
            raise ValueError(f"Mixed offline KD {field} availability")
        else:
            width = max(row.numel() for row in rows)
            result[field] = torch.stack(
                [torch.cat((row, row[-1:].expand(width - row.numel()))) for row in rows]
            )
    result["max_seqlen"] = (
        torch.tensor(
            [int((s["cu_seqlens"][1:] - s["cu_seqlens"][:-1]).max()) for s in samples],
            dtype=torch.int32,
        )
        if packed
        else None
    )
    result["_kd_sample_ids"] = [sample["sample_id"] for sample in samples]
    if "teacher_values" in samples[0]:
        result["_kd_values"] = torch.stack([s["teacher_values"][:seq_length] for s in samples])
        result["_kd_indices"] = torch.stack([s["teacher_indices"][:seq_length] for s in samples])
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
    ):
        self.reader = reader
        self.consumed = consumed
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.mbs = micro_batch_size
        self.seq_length = seq_length
        self.num_microbatches = num_microbatches
        self._microbatch = 0
        self._count = 0
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) if prefetch else None
        self._prefetched = None

    def __iter__(self) -> "ReplayIterator":
        return self

    def __next__(self) -> dict[str, Any]:
        if self._microbatch == 0:
            self._count = self.num_microbatches()
        end = self.consumed + self._count * self.mbs * self.dp_size
        start = self.consumed + (self._microbatch * self.dp_size + self.dp_rank) * self.mbs
        if self._prefetched is None:
            samples = self.reader.samples(list(range(start, start + self.mbs)), end)
        else:
            samples = self._prefetched.result()
            self._prefetched = None
        batch = collate_samples(samples, self.seq_length)
        self._microbatch += 1
        if self._microbatch == self._count:
            self.consumed = end
            self._microbatch = 0
        elif self._executor is not None:
            # Stay within this iteration: the next iteration's ramp-up count
            # is determined only when Megatron updates its batch calculator.
            next_start = self.consumed + (self._microbatch * self.dp_size + self.dp_rank) * self.mbs
            self._prefetched = self._executor.submit(
                self.reader.samples, list(range(next_start, next_start + self.mbs)), end
            )
        return batch

    def close(self) -> None:
        """Drain the bounded CPU prefetch worker when its loader is discarded."""
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def __del__(self):
        if getattr(self, "_executor", None) is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
