# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pipeline-parallel bucket transport for colocated VERL weight resync.

The normal MLite HF exporter has to make every tensor visible on every PP
rank because VERL pairs each training rank with a full-layer vLLM worker.  Its
generic PP path broadcasts tensors one at a time.  This module moves that
broadcast to VERL's existing bucket sender: the owning PP stage exports only
its local layers, packs a whole layer cluster into one buffer, and broadcasts
that buffer once.  A second staging buffer overlaps production of the next
bucket with vLLM loading the current one.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

from verl_mlite.rollout.layer_cluster import resync_layer_cluster_key


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PPBroadcastContext:
    """The PP collective needed to fan one owner bucket out to local vLLM."""

    rank: int
    size: int
    global_ranks: tuple[int, ...]
    group: Any
    cpu_group: Any = None


@dataclass(frozen=True)
class PPBucketPlanEntry:
    source_pp_rank: int
    metadata: dict[str, dict[str, Any]]
    used_bytes: int


@dataclass
class PPBucketPlanCache:
    """Stable wire plan learned by the first resync and reused afterwards."""

    entries: list[PPBucketPlanEntry] = field(default_factory=list)
    ready: bool = False

    def reset(self) -> None:
        self.entries.clear()
        self.ready = False


class PPBroadcastWeightStream:
    """Marker stream consumed by the patched VERL bucket sender."""

    def __init__(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        context: PPBroadcastContext,
        plan_cache: PPBucketPlanCache,
    ) -> None:
        self.weights = weights
        self.context = context
        self.plan_cache = plan_cache

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        return iter(self.weights)


def _align_offset(offset: int, alignment: int = 8) -> int:
    return (offset + alignment - 1) // alignment * alignment


class _LocalBucketPacker:
    """Pack a local-PP HF stream without crossing a vLLM layer cluster."""

    def __init__(self, weights: Iterable[tuple[str, torch.Tensor]], bucket_size: int):
        self._weights = iter(weights)
        self._bucket_size = bucket_size
        self._pending: tuple[str, torch.Tensor] | None = None
        self._exhausted = False

    def next_bucket(
        self, staging: torch.Tensor
    ) -> tuple[dict[str, dict[str, Any]], int] | None:
        if self._exhausted:
            return None

        offset = 0
        metadata: dict[str, dict[str, Any]] = {}
        layer_key = None
        while True:
            try:
                if self._pending is None:
                    name, weight = next(self._weights)
                else:
                    name, weight = self._pending
                    self._pending = None
            except StopIteration:
                self._exhausted = True
                break

            weight = weight.detach()
            if weight.device != staging.device:
                weight = weight.to(staging.device, non_blocking=True)
            weight = weight.contiguous()
            next_layer_key = resync_layer_cluster_key(name)
            if metadata and next_layer_key != layer_key:
                self._pending = (name, weight)
                break

            aligned_offset = _align_offset(offset)
            if aligned_offset + weight.nbytes > self._bucket_size:
                if metadata:
                    self._pending = (name, weight)
                    break
                raise ValueError(
                    f"Weight {name} ({weight.nbytes} bytes) exceeds the VERL "
                    f"weight-sync bucket ({self._bucket_size} bytes). Increase "
                    "rollout.checkpoint_engine.update_weights_bucket_megabytes."
                )

            if not metadata:
                layer_key = next_layer_key
            metadata[name] = {
                "name": name,
                "shape": weight.shape,
                "dtype": weight.dtype,
                "offset": aligned_offset,
                "handle": None,
            }
            staging[aligned_offset : aligned_offset + weight.nbytes].copy_(
                weight.view(torch.uint8).reshape(-1), non_blocking=True
            )
            offset = aligned_offset + weight.nbytes
            if offset >= self._bucket_size:
                break

        if not metadata:
            return None
        return metadata, offset


def _metadata_signature(metadata: dict[str, dict[str, Any]]) -> tuple:
    return tuple(
        (name, tuple(meta["shape"]), meta["dtype"], int(meta["offset"]))
        for name, meta in metadata.items()
    )


class _PPBucketProducer:
    """Produce globally ordered PP buckets using one collective per bucket."""

    def __init__(self, stream: PPBroadcastWeightStream, bucket_size: int):
        self._context = stream.context
        self._plan = stream.plan_cache
        self._local = _LocalBucketPacker(stream.weights, bucket_size)
        self._bucket_size = bucket_size
        self._source_pp_rank = 0
        self._plan_index = 0
        self._done = False

        if self._context.size != len(self._context.global_ranks):
            raise ValueError(
                "PP weight-sync context has inconsistent size/global_ranks: "
                f"{self._context.size} vs {self._context.global_ranks}"
            )

    def _collective_group(self, staging: torch.Tensor):
        if staging.device.type == "cpu":
            if self._context.cpu_group is None:
                raise RuntimeError("CPU PP weight sync requires pp_cpu_group")
            return self._context.cpu_group
        return self._context.group

    def _broadcast_header(self, payload, source: int, staging: torch.Tensor):
        header = [payload if self._context.rank == source else None]
        kwargs = {}
        if staging.device.type != "cpu":
            kwargs["device"] = staging.device
        dist.broadcast_object_list(
            header,
            src=self._context.global_ranks[source],
            group=self._collective_group(staging),
            **kwargs,
        )
        return header[0]

    def _broadcast_buffer(self, staging: torch.Tensor, source: int, used_bytes: int):
        dist.broadcast(
            staging[:used_bytes],
            src=self._context.global_ranks[source],
            group=self._collective_group(staging),
        )

    def _next_cached(self, staging: torch.Tensor):
        if self._plan_index >= len(self._plan.entries):
            if self._local.next_bucket(staging) is not None:
                raise RuntimeError(
                    "PP export produced more buckets than its cached plan"
                )
            self._done = True
            return "eof", {}, 0, None, True

        entry = self._plan.entries[self._plan_index]
        if self._plan_index:
            previous_source = self._plan.entries[self._plan_index - 1].source_pp_rank
            if (
                previous_source != entry.source_pp_rank
                and self._context.rank == previous_source
                and self._local.next_bucket(staging) is not None
            ):
                self._plan.reset()
                raise RuntimeError(
                    "PP export produced more owner buckets than its cached plan; "
                    "the cache was invalidated for the next resync"
                )
        self._plan_index += 1
        if entry.source_pp_rank == self._context.rank:
            packed = self._local.next_bucket(staging)
            if packed is None:
                raise RuntimeError("PP export ended before its cached bucket plan")
            metadata, used_bytes = packed
            if used_bytes != entry.used_bytes or _metadata_signature(
                metadata
            ) != _metadata_signature(entry.metadata):
                self._plan.reset()
                raise RuntimeError(
                    "PP weight export layout changed after its wire plan was cached; "
                    "the cache was invalidated for the next resync"
                )
        self._broadcast_buffer(staging, entry.source_pp_rank, entry.used_bytes)
        is_last = self._plan_index == len(self._plan.entries)
        if (
            is_last
            and self._context.rank == entry.source_pp_rank
            and self._local.next_bucket(staging) is not None
        ):
            self._plan.reset()
            raise RuntimeError(
                "PP export produced more owner buckets than its cached plan; "
                "the cache was invalidated for the next resync"
            )
        ready = None
        if staging.device.type != "cpu":
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(staging.device))
        return "bucket", entry.metadata, entry.used_bytes, ready, is_last

    def next_bucket(self, staging: torch.Tensor):
        if staging.device.type != "cpu":
            torch.cuda.set_device(staging.device)
        if self._done:
            return "eof", {}, 0, None, True
        if self._plan.ready:
            return self._next_cached(staging)

        while self._source_pp_rank < self._context.size:
            source = self._source_pp_rank
            payload = None
            if self._context.rank == source:
                packed = self._local.next_bucket(staging)
                if packed is None:
                    payload = {"kind": "stage_eof"}
                else:
                    metadata, used_bytes = packed
                    payload = {
                        "kind": "bucket",
                        "metadata": metadata,
                        "used_bytes": used_bytes,
                    }
            payload = self._broadcast_header(payload, source, staging)
            if payload["kind"] == "stage_eof":
                self._source_pp_rank += 1
                continue

            metadata = payload["metadata"]
            used_bytes = int(payload["used_bytes"])
            self._broadcast_buffer(staging, source, used_bytes)
            self._plan.entries.append(PPBucketPlanEntry(source, metadata, used_bytes))
            ready = None
            if staging.device.type != "cpu":
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(staging.device))
            return "bucket", metadata, used_bytes, ready, False

        self._plan.ready = True
        self._done = True
        return "eof", {}, 0, None, True


def _install_on_sender(sender_cls: type) -> bool:
    if getattr(sender_cls, "_verl_mlite_pp_bucketed", False):
        return True

    original_async_send_weights = sender_cls.async_send_weights

    async def pp_bucketed_async_send_weights(self, weights):
        if not isinstance(weights, PPBroadcastWeightStream):
            return await original_async_send_weights(self, weights)

        started_at = time.perf_counter()
        plan_hit = weights.plan_cache.ready
        self._init_socket()
        self._init_buffer()
        stop = threading.Event()
        free_slots: queue.Queue[int] = queue.Queue(maxsize=2)
        ready_results: queue.Queue[tuple] = queue.Queue(maxsize=2)
        executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mlite-pp-export"
        )
        held_slot = None
        try:
            staging_slots = [torch.empty_like(self.buffer) for _ in range(2)]
            producer = _PPBucketProducer(weights, self.bucket_size)
            free_slots.put_nowait(0)
            free_slots.put_nowait(1)

            def put_ready(result) -> bool:
                while not stop.is_set():
                    try:
                        ready_results.put(result, timeout=0.05)
                        return True
                    except queue.Full:
                        continue
                return False

            def produce() -> None:
                slot_index = None
                try:
                    while not stop.is_set():
                        try:
                            slot_index = free_slots.get(timeout=0.05)
                        except queue.Empty:
                            continue
                        result = producer.next_bucket(staging_slots[slot_index])
                        if not put_ready((*result, slot_index)):
                            return
                        slot_index = None
                        kind, *_middle, is_last = result
                        if kind == "eof" or is_last:
                            return
                except BaseException as exc:
                    put_ready(("error", exc, 0, None, True, slot_index))

            worker = executor.submit(copy_context().run, produce)
            while True:
                try:
                    kind, metadata, used_bytes, ready, is_last, held_slot = (
                        ready_results.get(timeout=0.1)
                    )
                except queue.Empty:
                    if worker.done():
                        worker.result()
                        raise RuntimeError(
                            "MLite PP weight producer stopped without a terminal result"
                        )
                    continue

                if kind == "error":
                    raise metadata
                if kind == "eof":
                    free_slots.put_nowait(held_slot)
                    held_slot = None
                    self.socket.send_pyobj({"bucket_meta": {}, "is_last": True})
                    self.socket.recv()
                    break

                if ready is not None:
                    ready.synchronize()
                staging = staging_slots[held_slot]
                self.buffer[:used_bytes].copy_(staging[:used_bytes], non_blocking=True)
                if self.buffer.device.type != "cpu":
                    torch.cuda.synchronize(self.buffer.device)
                free_slots.put_nowait(held_slot)
                held_slot = None

                self.socket.send_pyobj({"bucket_meta": metadata, "is_last": is_last})
                self.socket.recv()
                if is_last:
                    break
            worker.result()
            if weights.context.rank == 0:
                logger.warning(
                    "MLite PP bucket sync finished: plan=%s buckets=%d time=%.2fs",
                    "hit" if plan_hit else "build",
                    len(weights.plan_cache.entries),
                    time.perf_counter() - started_at,
                )
        finally:
            stop.set()
            if held_slot is not None:
                try:
                    free_slots.put_nowait(held_slot)
                except queue.Full:
                    pass
            executor.shutdown(wait=True, cancel_futures=True)
            self._cleanup()

    sender_cls.async_send_weights = pp_bucketed_async_send_weights
    sender_cls._verl_mlite_pp_bucketed = True
    return True


def install_pp_bucketed_sender(sender_cls: type | None = None) -> bool:
    """Patch VERL's existing sender; no new transport protocol is introduced."""

    if sender_cls is None:
        try:
            from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import (
                BucketedWeightSender,
            )
        except (ImportError, AttributeError):
            return False
        sender_cls = BucketedWeightSender
    return _install_on_sender(sender_cls)


__all__ = [
    "PPBroadcastContext",
    "PPBroadcastWeightStream",
    "PPBucketPlanCache",
    "install_pp_bucketed_sender",
]
