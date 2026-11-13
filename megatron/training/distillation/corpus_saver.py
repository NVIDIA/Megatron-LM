# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Megatron corpus production on top of paired input/target capture."""

import torch

from .format.corpus import CHUNK_VERSION, publish_chunk
from .v3_options import get_args
from .v3_saver import PairedLogitsSaver
from .v3_storage import CHUNK


class CorpusLogitsSaver(PairedLogitsSaver):
    """Write real source rows into independent parts, then publish a chunk index."""

    flush_kind = CHUNK

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
        directory = reports[0]["directory"]
        publish_chunk(inventory.storage, directory, chunk)
        inventory.add(directory, chunk)
