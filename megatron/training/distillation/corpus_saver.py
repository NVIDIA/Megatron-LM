# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Megatron corpus production on top of paired input/target capture."""

import json
import uuid
from collections import OrderedDict
from typing import Any

import torch

from .format.arrow import write_file
from .format.corpus import CHUNK_VERSION, chunk_path, part_path, publish_chunk
from .v3_options import get_args
from .v3_saver import PairedLogitsSaver
from .v3_storage import Storage


class CorpusLogitsSaver(PairedLogitsSaver):
    """Write real source rows into independent parts, then publish a chunk index."""

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

    def _flush_task(self, start: int, end: int, first_sample: int, end_sample: int) -> dict:
        """Name this writer's part of the chunk that starts at iteration ``start``."""
        return {
            "key": uuid.uuid4().hex,
            "start": start,
            "end": end,
            "path": part_path(self.dp_rank),
            "directory": chunk_path(start),
        }

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

    def _publish_flush(self, start: int, end: int, reports: list[dict]) -> None:
        """Publish the chunk index once every writer has uploaded its part."""
        from .corpus_runtime import corpus_inventory, producer_settings

        settings = producer_settings(get_args())
        chunk = {
            "chunk_version": CHUNK_VERSION,
            "parts": sorted(
                [report["part"] for report in reports if report["part"] is not None],
                key=lambda part: part["path"],
            ),
            "producer": {**settings, "start_iteration": start, "end_iteration": end},
        }
        rows = sum(block["rows"] for part in chunk["parts"] for block in part["index"]["blocks"])
        total = self.metadata_dict["contract"]["num_samples"]
        gbs = settings["global_batch_size"]
        if rows != min(total, end * gbs) - min(total, start * gbs):
            raise RuntimeError("Corpus flush did not retain exactly its real source rows")
        inventory = corpus_inventory()
        directory = chunk_path(start)
        publish_chunk(inventory.storage, directory, chunk)
        inventory.add(directory, chunk)
