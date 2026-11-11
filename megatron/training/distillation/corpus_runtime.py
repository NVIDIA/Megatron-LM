# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Corpus dumps: one whole source dataset, scored once by a frozen teacher.

A corpus source is generic, like the files it is derived from: complete, unweighted and
unsplit. Weights, train/validation choices and repetition belong to blends of sources.
A corpus dump captures ordinary V3 records, with independent parts published by a
consolidated chunk index:

- one unweighted source (one ``--data-path`` prefix or SFT file), written under
  ``<corpus>/<source>/``;
- all of it, exactly once, as Megatron builds it (GPT windows, SFT rows), with no
  ``--split`` partition or additional sample permutation.

The epoch is padded to whole global batches for execution. The corpus saver removes
filler samples before writing. Its progress tracker controls resume and replacement.
"""

import json
import os
from typing import Any, Callable

import torch
import torch.distributed as dist

from .format.arrow import file_meta
from .format.contract import blend_contract, mismatched_corpus_order, mismatched_settings
from .format.corpus import CorpusInventory, check_chunk
from .v3_options import runtime_options
from .v3_storage import Storage

# Placeholder budget until the epoch length is known (set once the dataset is built).
_UNTIL_EPOCH_END = 1 << 40
_INVENTORY = None
_TRACKER_PRESENT = False


def corpus_inventory() -> CorpusInventory:
    """Return the producer snapshot established collectively during checkpoint load."""
    if _INVENTORY is None:
        raise RuntimeError("Corpus inventory was not initialized during checkpoint restoration")
    return _INVENTORY


def reset_corpus_runtime() -> None:
    """Discard cached producer state on an in-process restart."""
    global _INVENTORY, _TRACKER_PRESENT
    _INVENTORY, _TRACKER_PRESENT = None, False


def producer_settings(args: Any) -> dict:
    """Megatron-only invariants for retrying an existing source's work assignment.

    A chunk's rows depend only on its iterations and the global batch size; the rank
    layout and flush interval may change between jobs of one source.
    """
    return {"name": "megatron", "global_batch_size": args.global_batch_size}


def resume_corpus_iteration(args: Any) -> int:
    """Use the output tracker as the authoritative cursor, without deleting data yet."""
    args = runtime_options(args)
    from .v3_runtime import dump_range

    global _INVENTORY, _TRACKER_PRESENT
    rank = dist.get_rank() if dist.is_initialized() else 0
    result = [None]
    if rank == 0:
        try:
            inventory = CorpusInventory(Storage(args.logits_save_dir))
            start, end = dump_range(args)
            start = start or 0
            settings = producer_settings(args)
            spans = []
            for directory, chunk in sorted(inventory.chunks.items()):
                producer = chunk.get("producer", {})
                if any(producer.get(key) != value for key, value in settings.items()):
                    raise ValueError("Corpus producer settings changed; use a new source directory")
                a, b = producer["start_iteration"], producer["end_iteration"]
                if directory != f"chunk-{a:012d}" or not 0 <= a < b:
                    raise ValueError("Invalid Megatron corpus chunk interval")
                total = inventory.metadata["contract"]["num_samples"]
                rows = sum(
                    block["rows"] for part in chunk["parts"] for block in part["index"]["blocks"]
                )
                gbs = args.global_batch_size
                if rows != min(total, b * gbs) - min(total, a * gbs):
                    raise ValueError("Corpus chunk has the wrong number of real source rows")
                spans.append((a, b))
            previous = 0
            for a, b in sorted(spans):
                if a < previous:
                    raise ValueError("Overlapping Megatron corpus chunks")
                if a < start < b or a < end < b:
                    raise ValueError("A corpus chunk crosses this job's assigned range boundary")
                previous = b
            try:
                raw = Storage(args.save).read("latest_checkpointed_iteration.txt").decode().strip()
            except FileNotFoundError:
                raw = None
            cursor = int(raw) if raw is not None else start
            if not start <= cursor <= end:
                raise ValueError(
                    f"Corpus tracker {cursor} is outside this job's range {start}-{end}"
                )
            owned = [
                path
                for path in inventory.snapshot["objects"]
                if start <= int(os.path.dirname(path).removeprefix("chunk-")) < end
            ]
            if raw is None and owned:
                raise ValueError(
                    "Existing corpus chunks have no progress tracker; create "
                    "latest_checkpointed_iteration.txt in --save at the intended resume iteration"
                )
            covered = start
            for a, b in sorted(spans):
                if a == covered and b <= cursor:
                    covered = b
            if covered != cursor:
                raise ValueError("Corpus tracker skips missing work or lies inside a chunk")
            result[0] = {
                "iteration": cursor,
                "present": raw is not None,
                "snapshot": inventory.snapshot,
            }
        except Exception as error:
            result[0] = {"error": f"{type(error).__name__}: {error}"}
    if dist.is_initialized():
        dist.broadcast_object_list(result, src=0)
    if "error" in result[0]:
        raise RuntimeError(result[0]["error"])
    _INVENTORY = CorpusInventory(Storage(args.logits_save_dir), result[0]["snapshot"])
    _TRACKER_PRESENT = result[0]["present"]
    return result[0]["iteration"]


def initialize_corpus_metadata(saver: Any, expected: dict) -> dict:
    """Validate the source, then unpublish only this job's selected replacement tail."""
    from .utils import _broadcast_without_pp
    from .v3_options import get_args
    from .v3_runtime import _report_recorded_differences, dump_range

    args = get_args()

    def initialize() -> dict:
        try:
            inventory = corpus_inventory()
            recorded_differences = set()
            for chunk in inventory.chunks.values():
                for part in chunk["parts"]:
                    meta = file_meta(part["index"])
                    differing, recorded = mismatched_settings(meta, expected)
                    recorded_differences.update(recorded)
                    differing += mismatched_corpus_order(meta, expected)
                    if differing:
                        raise ValueError(f"Corpus source settings changed: {', '.join(differing)}")
            # Preserve the existing cross-source tokenizer check. This listing is
            # performed once on initialization, never during flush or student reads.
            root = Storage(args.kd_corpus_root)
            seen = {args.kd_corpus_dump}
            for path in root.list("*/chunk-*/index.json"):
                source = path.split("/", 1)[0]
                if source in seen:
                    continue
                seen.add(source)
                chunk = json.loads(root.read(path))
                meta = check_chunk(chunk)
                if (
                    blend_contract(meta["contract"])["tokenizer"]
                    != expected["contract"]["tokenizer"]
                ):
                    raise ValueError(f"Corpus source {source} uses a different tokenizer")
                if meta.get("recorded", {}).get("teacher") != expected["recorded"]["teacher"]:
                    recorded_differences.add("teacher")
            _, end = dump_range(args)
            cursor = args.consumed_train_samples // args.global_batch_size
            if not 0 <= cursor <= min(end, args.train_iters):
                raise ValueError("Corpus tracker lies beyond the source epoch")
            # Establish the initial tracker before uploads, so an interrupted first
            # launch is distinguishable from a missing tracker on an existing source.
            if not _TRACKER_PRESENT:
                Storage(args.save).write("latest_checkpointed_iteration.txt", str(cursor).encode())
            directories = {
                os.path.dirname(path)
                for path in inventory.snapshot["objects"]
                if cursor <= int(os.path.dirname(path).removeprefix("chunk-")) < end
            }
            inventory.invalidate(directories)
            return {
                "metadata": {**expected, "published_through": args.consumed_train_samples},
                "differing": sorted(recorded_differences),
            }
        except Exception as error:
            return {"error": f"{type(error).__name__}: {error}"}

    result = _broadcast_without_pp(initialize)
    if "error" in result:
        raise RuntimeError(result["error"])
    _report_recorded_differences(result["differing"], "corpus sources and dump job")
    return result["metadata"]


def validate_corpus_dump(args: Any) -> None:
    """Check and apply ``--kd-corpus-dump`` settings (idempotent)."""
    name = getattr(args, "kd_corpus_dump", None)
    if not name:
        return
    if not args.logits_save_dir:
        raise ValueError("--kd-corpus-dump requires --logits-save-dir (the corpus directory)")
    if os.sep in name or name in ("", ".", ".."):
        raise ValueError(f"--kd-corpus-dump source name {name!r} must be a plain directory name")
    if not getattr(args, "freeze_all_layers", False):
        raise ValueError("Corpus dumps require a frozen teacher (--freeze-all-layers)")
    if getattr(args, "rampup_batch_size", None):
        raise ValueError("Corpus dumping requires a fixed global batch size")
    if getattr(args, "dataloader_type", None) not in (None, "single"):
        # Fillers are recognized by their position in Megatron's sequential order.
        raise ValueError("Corpus dumping requires --dataloader-type single")
    paths = getattr(args, "data_path", None) or []
    if not getattr(args, "mock_data", False) and len(paths) != 1:
        raise ValueError(
            "Corpus dumps take exactly one unweighted --data-path prefix (or SFT file); "
            "dump each source with its own job"
        )
    for flag in ("train_data_path", "valid_data_path", "test_data_path", "data_args_path"):
        if getattr(args, flag, None):
            raise ValueError(f"Corpus dumps do not support --{flag.replace('_', '-')}")
    if getattr(args, "per_split_data_args_path", None):
        raise ValueError("Corpus dumps do not support --per-split-data-args-path")
    if getattr(args, "phase_transition_iterations", None):
        raise ValueError("Corpus dumps do not support dataset phase transitions")
    if getattr(args, "eval_iters", 0):
        raise ValueError(
            "Corpus dumps score the whole source and no validation; use --eval-iters 0"
        )
    if getattr(args, "kd_corpus_root", None) is None:
        if getattr(args, "train_samples", None):
            raise ValueError("Corpus dumps size themselves to one epoch; omit --train-samples")
        args.kd_corpus_root = args.logits_save_dir
        args.logits_save_dir = os.path.join(args.logits_save_dir, name)
        args.train_iters = _UNTIL_EPOCH_END // args.global_batch_size
    # The whole source is the training part: a corpus source is never split.
    args.split = "100,0,0"
    args.logits_save_inputs = True


def validate_corpus_blend(args: Any) -> None:
    """Check and apply ``--kd-corpus-blend``: corpus loading reuses the replay path."""
    path = getattr(args, "kd_corpus_blend", None)
    if not path:
        return
    from .format.blend import load_config

    if getattr(args, "logits_save_inputs", False) or getattr(args, "kd_corpus_dump", None):
        raise ValueError("A run cannot both dump and load offline KD v3 data")
    if getattr(args, "logits_load_shuffle_shards", None) is not None:
        raise ValueError("A corpus blend is ordered by its plan; omit --logits-load-shuffle-shards")
    config = load_config(path)
    args.logits_load_inputs = True
    args.logits_load_dir = args.logits_load_dir or config["corpus"]


def epoch_length(dataset: Any) -> int:
    """Samples in one epoch of a Megatron dataset built with ``num_samples=None``.

    ``GPTDataset`` then builds exactly one epoch of windows; ``SFTDataset`` reports its
    requested count, so its length is set to the number of rows it indexes.
    """
    if getattr(dataset, "num_samples", 0) is None and hasattr(dataset, "indices"):
        dataset.num_samples = len(dataset.indices)
    return len(dataset)


class CorpusEpoch(torch.utils.data.Dataset):
    """Preserve a dataset's epoch order and pad it to whole global batches."""

    def __init__(self, dataset: Any, *, global_batch_size: int):
        self.dataset = dataset
        self.num_samples = epoch_length(dataset)
        if self.num_samples < 1:
            raise ValueError("Corpus dump source has no samples")
        self.padded = -(-self.num_samples // global_batch_size) * global_batch_size

    def __len__(self) -> int:
        return self.padded

    def __getitem__(self, index: int) -> Any:
        # Filler positions repeat the first samples; they are never part of the corpus.
        return self.dataset[index % self.num_samples]


def build_corpus_dump_datasets(args: Any, provider: Callable) -> tuple:
    """Build the whole source for exactly one epoch and wrap it (data-building ranks)."""
    args = runtime_options(args)
    # A single unweighted prefix, all in the training part, with size None builds one
    # full epoch of the whole source.
    dataset = provider([None, None, None])[0]
    if dataset is None:
        raise ValueError("Corpus dump source has no samples")
    epoch = CorpusEpoch(dataset, global_batch_size=args.global_batch_size)
    args.kd_corpus_samples = epoch.num_samples
    return epoch, None, None


def agree_corpus_epoch(args: Any) -> None:
    """Share the epoch length with every rank and size the run to it (collective)."""
    args = runtime_options(args)
    length = torch.tensor(
        [getattr(args, "kd_corpus_samples", 0)],
        dtype=torch.long,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    if dist.is_initialized():
        dist.all_reduce(length, op=dist.ReduceOp.MAX)
    args.kd_corpus_samples = int(length.item())
    args.train_iters = -(-args.kd_corpus_samples // args.global_batch_size)
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(
            f"Offline KD corpus dump: {args.kd_corpus_samples} samples of "
            f"{args.kd_corpus_dump} in {args.train_iters} iterations",
            flush=True,
        )
