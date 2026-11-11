# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Megatron corpus production on top of paired input/target capture."""

import json
import multiprocessing as mp
import uuid
from collections import OrderedDict
from typing import Any, Callable

import torch
import torch.distributed as dist

from megatron.training import get_args

from .format.arrow import write_file
from .format.corpus import CHUNK_VERSION, chunk_path, part_path, publish_chunk
from .v3_saver import PairedLogitsSaver
from .v3_storage import Storage


class CorpusLogitsSaver(PairedLogitsSaver):
    """Write real source rows into independent parts, then publish a chunk index."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._result_manager = mp.Manager() if self.tp_rank == self.cp_rank == 0 else None
        # Manager proxies can travel through the persistent worker's request queue.
        self._results = self._result_manager.dict() if self._result_manager else None
        self._last_flush = None

    def _prepare_record(self, inputs: dict, *targets: torch.Tensor) -> tuple:
        """Remove artificial final-batch samples before sparsifying token targets."""
        args = get_args()
        offsets = inputs["sample_offsets"]
        rows = torch.arange(len(offsets) - 1)
        mbs = args.micro_batch_size
        positions = (
            args.consumed_train_samples
            + rows.div(mbs, rounding_mode="floor") * mbs * self.dp_size
            + self.dp_rank * mbs
            + rows.remainder(mbs)
        )
        keep = positions < args.kd_corpus_samples
        lengths = offsets[1:] - offsets[:-1]
        tokens = torch.repeat_interleave(keep, lengths)
        selected = {
            "sample_offsets": torch.cat(
                (torch.zeros(1, dtype=offsets.dtype), lengths[keep].cumsum(0))
            ),
            **{
                key: inputs[key][tokens]
                for key in ("tokens", "labels", "position_ids", "loss_mask")
            },
            "cu_seqlens": (
                [row for row, real in zip(inputs["cu_seqlens"], keep.tolist()) if real]
                if inputs["cu_seqlens"] is not None
                else None
            ),
        }
        return selected, *(target[tokens] for target in targets)

    def take_pending_data(self) -> tuple:
        """Hand the part upload and a result proxy to the existing async worker."""
        args = get_args()
        writes, self._pending_writes = self._pending_writes, OrderedDict()
        task = None
        if writes:
            start = min(a for a, _ in writes) // args.global_batch_size
            end = max(b for _, b in writes) // args.global_batch_size
            task = {
                "key": uuid.uuid4().hex,
                "start": start,
                "end": end,
                "path": part_path(self.dp_rank),
                "directory": chunk_path(start),
            }
        self._last_flush = task
        return self.save_dir, writes, self._meta_bytes, task, self._results, self._failure_event

    @staticmethod
    def _write_batched_tar(
        root: str,
        writes: OrderedDict,
        metadata: bytes,
        task: dict | None,
        results: Any,
        failure_event: Any,
    ) -> None:
        """Upload one part and report its index without reading the object back."""
        if task is None:
            return
        try:
            if root.startswith("msc://"):
                from megatron.core.msc_utils import MultiStorageClientFeature

                MultiStorageClientFeature.enable()
            meta = json.loads(metadata)
            meta.pop("published_through", None)
            batches = [record["batch"] for record in writes.values() if record["batch"].num_rows]
            part = None
            if batches:
                data, index = write_file(batches, meta)
                Storage(root).write(f"{task['directory']}/{task['path']}", data)
                part = {"path": task["path"], "index": index}
            results[task["key"]] = {"start": task["start"], "end": task["end"], "part": part}
        except Exception:
            failure_event.set()
            raise

    def flush_finalize_fn(self, iteration: int) -> Callable[[], None]:
        """Collect writer success and publish the index before the progress callback.

        Every rank on the last pipeline stage calls this. Other stages enqueue a
        matching no-op request so the async queue's world synchronization still pairs.
        """
        from megatron.core import parallel_state as mpu

        from .corpus_runtime import corpus_inventory, producer_settings

        task = self._last_flush
        group = mpu.get_tensor_and_data_parallel_group(with_context_parallel=True)
        src = dist.get_process_group_ranks(group)[0] if dist.is_initialized() else 0
        settings = producer_settings(get_args())

        def finalize() -> None:
            local = None
            if self.tp_rank == self.cp_rank == 0:
                try:
                    if self._failure_event.is_set():
                        raise RuntimeError("Corpus part upload failed")
                    if task is not None:
                        local = self._results.pop(task["key"], None)
                        if local is None:
                            raise RuntimeError(
                                "Corpus worker exited without a completion descriptor"
                            )
                except Exception as error:
                    local = {"error": f"{type(error).__name__}: {error}"}
            leader = not dist.is_initialized() or dist.get_rank() == src
            gathered = (
                [None] * dist.get_world_size(group) if leader and dist.is_initialized() else [local]
            )
            if dist.is_initialized():
                dist.gather_object(local, gathered if leader else None, dst=src, group=group)
            outcome = [None]
            if leader:
                try:
                    reports = [report for report in gathered if report is not None]
                    errors = [report["error"] for report in reports if "error" in report]
                    if errors:
                        raise RuntimeError("; ".join(errors))
                    if reports:
                        if len(reports) != self.dp_size:
                            raise RuntimeError("Missing corpus writer completion descriptors")
                        starts = {report["start"] for report in reports}
                        if len(starts) != 1 or any(
                            report["end"] != iteration for report in reports
                        ):
                            raise RuntimeError("Corpus writers disagree on the flush interval")
                        start = starts.pop()
                        chunk = {
                            "chunk_version": CHUNK_VERSION,
                            "parts": sorted(
                                [
                                    report["part"]
                                    for report in reports
                                    if report["part"] is not None
                                ],
                                key=lambda part: part["path"],
                            ),
                            "producer": {
                                **settings,
                                "start_iteration": start,
                                "end_iteration": iteration,
                            },
                        }
                        rows = sum(
                            block["rows"]
                            for part in chunk["parts"]
                            for block in part["index"]["blocks"]
                        )
                        total = self.metadata_dict["contract"]["num_samples"]
                        gbs = settings["global_batch_size"]
                        if rows != min(total, iteration * gbs) - min(total, start * gbs):
                            raise RuntimeError(
                                "Corpus flush did not retain exactly its real source rows"
                            )
                        inventory = corpus_inventory()
                        directory = chunk_path(start)
                        publish_chunk(inventory.storage, directory, chunk)
                        inventory.add(directory, chunk)
                    outcome[0] = {}
                except Exception as error:
                    outcome[0] = {"error": f"{type(error).__name__}: {error}"}
            if dist.is_initialized():
                dist.broadcast_object_list(outcome, src=src, group=group)
            if "error" in outcome[0]:
                raise RuntimeError(outcome[0]["error"])

        return finalize

    def finish(self) -> None:
        """Publish a final partial flush synchronously, then record its iteration."""
        args = get_args()
        iteration = args.consumed_train_samples // args.global_batch_size
        request = self.take_pending_data()
        finalize = self.flush_finalize_fn(iteration)
        self._write_batched_tar(*request)
        finalize()
        if self.tp_rank == self.cp_rank == self.dp_rank == 0:
            Storage(args.save).write("latest_checkpointed_iteration.txt", str(iteration).encode())
