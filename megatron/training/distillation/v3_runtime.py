# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Small integration hooks for paired dumping, replay, and checkpoint resume."""

import json
import uuid
from typing import Any

import torch
import torch.distributed as dist

from .v3_format import digest, mismatched_settings
from .v3_replay import ReplayIterator, ReplayReader, describe_hole, discover_groups, layout_options
from .v3_storage import (
    Storage,
    complete_ranges,
    contiguous_end,
    discard_unpublished,
    list_tars,
    parse_tar_name,
    read_meta,
)

_PLAN = None
_HYBRID_LAYOUT = False


def validate_options(args: Any) -> None:
    """Validate only opted-in v3 behavior; legacy flags keep their semantics."""
    saving = getattr(args, "logits_save_inputs", False)
    loading = getattr(args, "logits_load_inputs", False)
    if saving and not args.logits_save_dir:
        raise ValueError("--logits-save-inputs requires --logits-save-dir")
    if loading and not args.logits_load_dir:
        raise ValueError("--logits-load-inputs requires --logits-load-dir")
    if saving and loading:
        raise ValueError("Paired v3 dumping and replay cannot run together")
    shuffling = args.logits_load_shuffle_shards is not None
    if shuffling and not loading:
        raise ValueError("Shuffling requires --logits-load-inputs")
    if not (saving or loading):
        return
    if args.sequence_packing_scheduler or args.hybrid_context_parallel or args.use_varlen_dataset:
        raise ValueError(
            "Offline KD v3 does not yet support runtime packing, variable-length datasets, or hybrid CP"
        )
    if args.mtp_num_layers:
        raise ValueError("Offline KD v3 does not yet support MTP")
    if (
        getattr(args, "cuda_graph_impl", "none") != "none"
        or getattr(args, "enable_cuda_graph", False)
        or getattr(args, "overlap_moe_expert_parallel_comm", False)
    ):
        raise ValueError(
            "Offline KD v3 requires ordinary forward execution, without CUDA graphs or schedule-plan overlap"
        )
    if getattr(args, "allow_ambiguous_pad_tokens", False):
        raise ValueError(
            "Offline KD v3 does not support --allow-ambiguous-pad-tokens: replayed attention "
            "masks are rebuilt from tokens after pad replacement"
        )
    if saving:
        _validate_dump_options(args)
    if loading and getattr(args, "logits_load_ignore_errors", False):
        raise ValueError(
            "V3 paired input replay does not support --logits-load-ignore-errors; "
            "input/target failures must stop training"
        )
    if loading and args.dataloader_type != "single":
        raise ValueError("Offline KD v3 replay requires --dataloader-type single")
    if loading and getattr(args, "phase_transition_iterations", None):
        raise ValueError("Offline KD v3 replay does not yet support dataset phase transitions")
    if loading and getattr(args, "override_ckpt_iteration", None) is not None:
        raise ValueError("A v3 replay cursor cannot be rewound with --override-ckpt-iteration")


def _validate_dump_options(args: Any) -> None:
    if getattr(args, "recompute_granularity", None) == "full":
        raise ValueError(
            "Paired teacher dumping does not support full-layer activation recomputation"
        )
    if getattr(args, "iterations_to_skip", None):
        raise ValueError(
            "Paired teacher dumping does not support --iterations-to-skip (including iterations "
            "from --result-rejected-tracker-filename): skipped windows would leave holes"
        )
    if not getattr(args, "save", None) or not getattr(args, "save_interval", None):
        raise ValueError(
            "Paired teacher dumping requires --save and --save-interval (the flush interval)"
        )
    if args.exit_interval and args.exit_interval % args.save_interval:
        raise ValueError(
            "Paired teacher dumping requires --exit-interval to be a multiple of --save-interval"
        )
    start = getattr(args, "override_ckpt_iteration", None)
    if (
        getattr(args, "freeze_all_layers", False)
        and start is not None
        and start % args.save_interval
    ):
        raise ValueError(
            "Paired teacher dumping requires --override-ckpt-iteration to be a multiple of "
            "--save-interval"
        )


def tokenizer_identity() -> dict[str, Any]:
    """Fingerprint actual vocabulary IDs and relevant special tokens."""
    from megatron.training import get_tokenizer

    tokenizer = get_tokenizer()
    try:
        vocab = tokenizer.vocab
        if not isinstance(vocab, dict):
            raise ValueError("Offline KD v3 requires a tokenizer exposing its vocabulary mapping")
        identity = {"vocab": sorted((str(token), int(index)) for token, index in vocab.items())}
    except NotImplementedError:
        wrapped = getattr(tokenizer, "_tokenizer", tokenizer)
        if type(wrapped).__name__ != "NullTokenizer":
            raise ValueError("Offline KD v3 requires an explicit vocabulary mapping") from None
        identity = {"synthetic_integer_vocabulary": int(tokenizer.vocab_size)}
    for key in ("eod", "pad", "bos", "eos"):
        try:
            identity[key] = getattr(tokenizer, key)
        except (AttributeError, NotImplementedError):
            identity[key] = None
    return {
        "sha256": digest(json.dumps(identity, sort_keys=True).encode()),
        "vocab_size": int(tokenizer.vocab_size),
        "eod": identity["eod"],
        "pad": identity["pad"],
    }


def dump_range(args: Any) -> tuple[int | None, int]:
    """Return this dump job's ``(start, end)`` iterations.

    A frozen teacher started with ``--override-ckpt-iteration s`` owns
    ``[s, next multiple of --exit-interval)``; Megatron's own exit-interval check
    ends it there. Otherwise the job is the cache's single sequential writer and
    owns everything up to ``--train-iters`` (``start`` is None: the cache start).
    """
    start = getattr(args, "logits_save_range_start", None)
    if start is None:
        return None, args.train_iters
    if not args.exit_interval:
        return start, args.train_iters
    return start, min((start // args.exit_interval + 1) * args.exit_interval, args.train_iters)


def _published_end(storage: Storage, dp_size: int, start: int, end: int) -> int:
    return contiguous_end(complete_ranges(list_tars(storage), dp_size), start, end)


def frozen_resume_iteration(args: Any) -> int:
    """Return the iteration a frozen v3 teacher resumes from: its range's published end.

    The cache, not the progress tracker, is the source of truth, so a resubmitted
    parallel job (unchanged ``--override-ckpt-iteration``) continues inside its own
    range, and periodic ``--exit-interval`` requeues of a single writer advance.
    """
    if not hasattr(args, "logits_save_range_start"):
        # Record the user's range start once; load_checkpoint then replaces the override.
        args.logits_save_range_start = args.override_ckpt_iteration
    start_iteration, end_iteration = dump_range(args)
    gbs = args.global_batch_size

    def resolve() -> dict:
        try:
            from megatron.core import parallel_state as mpu

            storage = Storage(args.logits_save_dir)
            start = (start_iteration or 0) * gbs
            published = _published_end(
                storage, mpu.get_data_parallel_world_size(), start, end_iteration * gbs
            )
            return {"iteration": published // gbs}
        except Exception as error:
            return {"error": f"{type(error).__name__}: {error}"}

    result = [resolve() if not dist.is_initialized() or dist.get_rank() == 0 else None]
    if dist.is_initialized():
        dist.broadcast_object_list(result, src=0)
    if "error" in result[0]:
        raise RuntimeError(result[0]["error"])
    return result[0]["iteration"]


def _dump_settings(saver: Any) -> dict[str, Any]:
    """Settings every tar of a cache must share (see ``SHARED_KEYS``), minus ``first_sample``."""
    from megatron.training import get_args

    args = get_args()
    settings = {
        "format_version": 3,
        "dp_size_save": saver.dp_size,
        "mbs_save": args.micro_batch_size,
        "gbs_save": args.global_batch_size,
        "cp_size_save": saver.cp_size,
        "save_interval": args.save_interval,
        "train_budget": args.train_samples or args.train_iters * args.global_batch_size,
        "seq_length": args.seq_length,
        "sft": bool(args.sft),
        "inter_document_masking": bool(args.dataloader_inter_document_masking),
        "reset_attention_mask": bool(args.reset_attention_mask),
        "tokenizer": tokenizer_identity(),
        "padded_vocab_size": args.padded_vocab_size,
        "targets": {**saver.metadata_dict["saver"], "format_version": 3},
        "dataset_identity": saver.metadata_dict["identifiers"],
        "boundary_convention": "exact_dataset_boundaries_may_include_padding",
    }
    # Compare exactly what a JSON round trip through a tar's _meta.json preserves.
    return json.loads(json.dumps(settings))


def initialize_dump_metadata(saver: Any) -> dict:
    """Check this dump job against the cache and find where it resumes, collectively."""
    from megatron.training import get_args

    from .utils import _broadcast_without_pp

    args = get_args()
    expected = _dump_settings(saver)
    gbs = args.global_batch_size
    frozen = bool(getattr(args, "freeze_all_layers", False))

    def create():
        try:
            storage = Storage(saver.save_dir)
            if any(parse_tar_name(name) is None for name in storage.list("*.tar")):
                raise ValueError("Cannot mix legacy and v3 dumps in the same directory")
            tars = list_tars(storage)
            first_sample = 0 if frozen else args.consumed_train_samples
            if tars:
                existing = read_meta(storage, tars[0].name)
                mismatched = mismatched_settings(
                    existing, {**expected, "first_sample": existing.get("first_sample")}
                )
                if mismatched:
                    raise ValueError(
                        f"Teacher settings differ from existing v3 tars ({', '.join(mismatched)}); "
                        "use a new save directory"
                    )
                first_sample = existing["first_sample"]
            start_iteration, end_iteration = dump_range(args)
            start = first_sample if start_iteration is None else start_iteration * gbs
            end = end_iteration * gbs
            published = _published_end(storage, saver.dp_size, start, end)
            if not start <= args.consumed_train_samples <= published:
                raise ValueError(
                    f"Teacher cursor {args.consumed_train_samples} is outside this job's published "
                    f"range {start}-{published}; resume an earlier checkpoint or use a new cache "
                    "directory"
                )
            discard_unpublished(storage, published, end)
            return {
                "metadata": {
                    **expected,
                    "first_sample": first_sample,
                    "generation": uuid.uuid4().hex,
                    "teacher_checkpoint": str(args.load),
                    "published_through": published,
                    "range_end": end,
                }
            }
        except Exception as error:
            return {"error": f"{type(error).__name__}: {error}"}

    result = _broadcast_without_pp(create)
    if "error" in result:
        raise RuntimeError(result["error"])
    return result["metadata"]


def _check_student_compatibility(args: Any, metadata: dict) -> None:
    if metadata.get("format_version") != 3 or metadata["tokenizer"] != tokenizer_identity():
        raise ValueError("Offline KD v3 tokenizer/vocabulary mismatch")
    if args.padded_vocab_size != metadata["padded_vocab_size"]:
        raise ValueError(
            "V3 teacher/student padded vocabulary sizes must match; "
            "adjust vocabulary padding when changing TP"
        )
    if (
        bool(args.sft) != metadata["sft"]
        or bool(args.dataloader_inter_document_masking) != metadata["inter_document_masking"]
    ):
        raise ValueError("Student packing flags must match the saved v3 layout")
    if bool(args.reset_attention_mask) != metadata["reset_attention_mask"]:
        raise ValueError("Student attention isolation differs from the teacher cache")
    packed = metadata["sft"] or metadata["inter_document_masking"]
    if args.seq_length > metadata["seq_length"] or (
        packed and args.seq_length != metadata["seq_length"]
    ):
        raise ValueError("Only non-packed prefix shortening is supported by v3 replay")
    if args.context_parallel_size > 1 and args.seq_length % (2 * args.context_parallel_size):
        raise ValueError("Student sequence length must be divisible by 2 * CP")


def initialize_replay(args: Any) -> None:
    """Establish a common replay plan on all ranks before building datasets."""
    global _PLAN
    if not getattr(args, "logits_load_inputs", False) or _PLAN is not None:
        return
    shuffle = args.logits_load_shuffle_shards is not None
    rank = dist.get_rank() if dist.is_initialized() else 0
    result = [None]
    if rank == 0:
        try:
            storage = Storage(args.logits_load_dir)
            if any(parse_tar_name(name) is None for name in storage.list("*.tar")):
                raise ValueError("Input replay requires a v3 dump; found legacy tars")
            tars = list_tars(storage)
            if not tars:
                raise FileNotFoundError("No v3 tars found; input replay requires a v3 dump")
            metadata = read_meta(storage, tars[0].name)
            _check_student_compatibility(args, metadata)
            groups, hole = discover_groups(storage, metadata)
            replay_end = getattr(args, "logits_load_replay_end", None)
            if shuffle:
                if replay_end is None:
                    # Fix the shuffled extent at the first launch; resumes reuse it.
                    replay_end = groups[-1][1] if groups else metadata["first_sample"]
                if (groups[-1][1] if groups else metadata["first_sample"]) < replay_end:
                    raise ValueError(
                        f"Shuffled replay was planned over samples up to {replay_end}, but only "
                        f"{groups[-1][1] if groups else metadata['first_sample']} are published"
                    )
                groups = [group for group in groups if group[1] <= replay_end]
                if not groups:
                    raise ValueError("No complete teacher shard group is available to shuffle")
            available = sum(end - start for start, end in groups)
            if args.consumed_train_samples > available:
                raise ValueError(
                    f"Checkpoint sample cursor {args.consumed_train_samples} exceeds the "
                    f"published offline KD cache ({available} samples)"
                )
            result[0] = {
                "metadata": metadata,
                "groups": groups,
                "hole": hole,
                "replay_end": replay_end,
            }
        except Exception as error:
            result[0] = {"error": f"{type(error).__name__}: {error}"}
    if dist.is_initialized():
        dist.broadcast_object_list(result, src=0)
    if "error" in result[0]:
        raise RuntimeError(result[0]["error"])
    if shuffle:
        # Saved with the checkpoint's args so a resume shuffles the same groups.
        args.logits_load_replay_end = result[0]["replay_end"]
    _PLAN = {
        **result[0],
        "available": sum(end - start for start, end in result[0]["groups"]),
        "readers": [],
    }
    if rank == 0 and result[0]["hole"] is not None:
        print(f"WARNING: offline KD v3 cache: {describe_hole(result[0]['hole'])}", flush=True)


def ensure_replay_frontier(args: Any) -> None:
    """Agree on all ranks that the next iteration's samples are published.

    Collective only when the next iteration would cross the known frontier: world
    rank 0 lists once and broadcasts newly published groups, so every rank either
    continues or raises the same exhaustion error. Never sleeps or polls.
    """
    from megatron.core.num_microbatches_calculator import get_current_global_batch_size

    need = args.consumed_train_samples + get_current_global_batch_size()
    if need <= _PLAN["available"]:
        return
    if args.logits_load_shuffle_shards is None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        result = [None]
        if rank == 0:
            try:
                storage = Storage(args.logits_load_dir)
                groups, hole = discover_groups(storage, _PLAN["metadata"])
                known = _PLAN["groups"]
                if groups[: len(known)] != known:
                    raise RuntimeError("Previously published offline KD data changed during replay")
                result[0] = {"groups": groups[len(known) :], "hole": hole}
            except Exception as error:
                result[0] = {"error": f"{type(error).__name__}: {error}"}
        if dist.is_initialized():
            dist.broadcast_object_list(result, src=0)
        if "error" in result[0]:
            raise RuntimeError(result[0]["error"])
        new_groups = result[0]["groups"]
        _PLAN["groups"].extend(new_groups)
        _PLAN["available"] += sum(end - start for start, end in new_groups)
        for reader in _PLAN["readers"]:
            reader.extend(new_groups)
        if rank == 0 and result[0]["hole"] is not None and result[0]["hole"] != _PLAN["hole"]:
            print(f"WARNING: offline KD v3 cache: {describe_hole(result[0]['hole'])}", flush=True)
        _PLAN["hole"] = result[0]["hole"]
    if need > _PLAN["available"]:
        detail = describe_hole(_PLAN["hole"])
        raise RuntimeError(
            f"Offline KD cache exhausted at {_PLAN['available']} samples; requested {need}"
            + (f" ({detail})" if detail else "")
        )


def before_train_step(args: Any) -> bool:
    """Run v3 checks at the top of each training iteration; True means exit cleanly."""
    if getattr(args, "logits_load_inputs", False):
        ensure_replay_frontier(args)
    if getattr(args, "logits_save_inputs", False):
        start, end = dump_range(args)
        if start is not None and args.consumed_train_samples >= end * args.global_batch_size:
            # Megatron checks --exit-interval only after a step; a resubmitted job whose
            # range is already published must not dump into the next job's range.
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"Offline KD v3 dump range {start}-{end} is complete; exiting", flush=True)
            return True
    return False


def use_hybrid_layout() -> None:
    """Called by the hybrid entrypoint: replay must use its CP layouts."""
    global _HYBRID_LAYOUT
    _HYBRID_LAYOUT = True


class _ContextParallelReplay:
    """Serve replay batches to every CP rank while only CP rank 0 reads storage.

    CP rank 0 maps the teacher targets for every CP rank in its prefetch thread;
    this iterator scatters each rank its inputs and its own target shard.
    """

    def __init__(self, source: ReplayIterator | None, cp_rank: int, cp_size: int):
        self.source = source
        self.cp_rank = cp_rank
        self.cp_size = cp_size
        if cp_size > 1:
            from megatron.core import parallel_state as mpu

            self.group = mpu.get_context_parallel_group()
            self.src = dist.get_global_rank(self.group, 0)

    @property
    def reader(self) -> ReplayReader | None:
        """The underlying reader on CP rank 0."""
        return None if self.source is None else self.source.reader

    def __iter__(self) -> "_ContextParallelReplay":
        return self

    @staticmethod
    def _for_rank(batch: dict, rank: int) -> dict:
        local = dict(batch)
        for key in ("_kd_values", "_kd_indices"):
            if key in local:
                local[key] = local[key][rank]
        return local

    def __next__(self) -> dict:
        if self.cp_size == 1:
            return self._for_rank(next(self.source), 0)
        payload = None
        if self.cp_rank == 0:
            try:
                batch = next(self.source)
                payload = [self._for_rank(batch, rank) for rank in range(self.cp_size)]
            except Exception as error:
                # Fail every CP rank together instead of leaving peers in the collective.
                payload = [{"_kd_error": f"{type(error).__name__}: {error}"}] * self.cp_size
        received = [None]
        dist.scatter_object_list(received, payload, src=self.src, group=self.group)
        if "_kd_error" in received[0]:
            raise RuntimeError(received[0]["_kd_error"])
        return received[0]

    def close(self) -> None:
        """Stop the reader's streams."""
        if self.source is not None:
            self.source.close()


def build_replay_loader(args: Any, consumed: int) -> torch.utils.data.DataLoader | None:
    """Build already-batched replay without applying the ordinary sampler."""
    from megatron.core import parallel_state as mpu
    from megatron.core.num_microbatches_calculator import get_num_microbatches

    if mpu.get_tensor_model_parallel_rank() != 0:
        return None
    if not (
        mpu.is_pipeline_first_stage()
        or mpu.is_pipeline_last_stage()
        or args.sft
        or args.dataloader_inter_document_masking
    ):
        return None
    cp_rank = mpu.get_context_parallel_rank()
    cp_size = args.context_parallel_size
    source = None
    if cp_rank == 0:
        reader = ReplayReader(
            Storage(args.logits_load_dir),
            _PLAN["metadata"],
            _PLAN["groups"],
            targets=mpu.is_pipeline_last_stage(),
            shuffle=args.logits_load_shuffle_shards is not None,
            seed=args.logits_load_shuffle_shards or 0,
            decode_threads=args.logits_load_decode_threads,
            prefetch_depth=args.logits_load_msc_prefetch_depth,
            chunk_bytes=args.logits_load_read_chunk_mb << 20,
        )
        _PLAN["readers"].append(reader)
        source = ReplayIterator(
            reader,
            consumed=consumed,
            dp_rank=mpu.get_data_parallel_rank(),
            dp_size=mpu.get_data_parallel_world_size(),
            micro_batch_size=args.micro_batch_size,
            seq_length=args.seq_length,
            num_microbatches=get_num_microbatches,
            prefetch=True,
            cp_size=cp_size,
            layout=layout_options(args, hybrid=_HYBRID_LAYOUT),
        )
    iterator = _ContextParallelReplay(source, cp_rank, cp_size)

    class Dataset(torch.utils.data.IterableDataset):
        def __iter__(self):
            return iterator

    return torch.utils.data.DataLoader(Dataset(), batch_size=None, pin_memory=True, num_workers=0)


def validation_only_config(config: Any, sample_counts: list) -> list:
    """Disable training splits without changing validation's original split ranges."""
    from megatron.training import get_args

    args = get_args()
    if not (args.mock_data or args.data_path or args.valid_data_path or args.test_data_path):
        raise ValueError("V3 validation requires an explicit validation dataset or --mock-data")
    if config.split_matrix is not None:
        config.split_matrix = [None, *config.split_matrix[1:]]
    if config.blend_per_split is not None:
        config.blend_per_split = [None, *config.blend_per_split[1:]]
    return [0, *sample_counts[1:]]


def capture_dump_batch(batch: dict, *, hybrid: bool = False, vp_stage: int | None = None) -> None:
    """Call before the entrypoint mutates its CPU dataloader batch."""
    from megatron.core import parallel_state as mpu

    from .logits_saver import get_logits_saver

    saver = get_logits_saver()
    if (
        saver is not None
        and hasattr(saver, "capture")
        and mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)
    ):
        saver.capture(batch, hybrid=hybrid)


def prepare_replay_batch(batch: dict) -> tuple[dict, dict | None]:
    """Separate this CP rank's already-mapped targets from the model batch fields."""
    from megatron.training import get_args

    args = get_args()
    result = dict(batch)
    sidecar = None
    if "_kd_values" in batch:
        sidecar = {
            "values": batch["_kd_values"],
            "indices": batch["_kd_indices"],
            "sample_ids": batch["_kd_sample_ids"],
        }
    for key in ("_kd_values", "_kd_indices", "_kd_sample_ids"):
        result.pop(key, None)
    if args.create_attention_mask_in_dataloader:
        from megatron.core.datasets.gpt_dataset import _get_ltor_masks_and_position_ids

        eod = _PLAN["metadata"]["tokenizer"]["eod"]
        result["attention_mask"] = torch.stack(
            [
                _get_ltor_masks_and_position_ids(
                    tokens, eod, False, args.reset_attention_mask, False, True
                )[0]
                for tokens in result["tokens"]
            ]
        )
    return result, sidecar


def broadcast_targets(sidecar: dict | None, *, vp_stage: int | None = None) -> dict | None:
    """Broadcast targets only among the final pipeline stage's TP ranks."""
    from megatron.core import parallel_state as mpu

    if not mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage):
        return None
    source = mpu.get_tensor_model_parallel_src_rank()
    group = mpu.get_tensor_model_parallel_group()
    header = [
        (
            None
            if sidecar is None
            else (tuple(sidecar["values"].shape), sidecar["values"].dtype, sidecar["sample_ids"])
        )
    ]
    if dist.is_initialized():
        dist.broadcast_object_list(header, src=source, group=group)
    if header[0] is None:
        return None  # Ordinary validation data does not carry cached targets.
    shape, dtype, ids = header[0]
    device = torch.cuda.current_device()
    if sidecar is None:
        sidecar = {
            "values": torch.empty(shape, dtype=dtype, device=device),
            "indices": torch.empty(shape, dtype=torch.int64, device=device),
            "sample_ids": ids,
        }
    else:
        sidecar = dict(sidecar)
        for field in ("values", "indices"):
            tensor = sidecar[field]
            if torch.cuda.is_available() and tensor.device.type == "cpu" and not tensor.is_pinned():
                tensor = tensor.pin_memory()
            sidecar[field] = tensor.to(device, non_blocking=True)
    if dist.is_initialized():
        for field in ("values", "indices"):
            dist.broadcast(sidecar[field], src=source, group=group)
    return sidecar


def bind_student_logits(sidecar: dict | None) -> dict | None:
    """Own the current forward's differentiable logits in its loss closure."""
    if sidecar is None:
        return None
    from .cached_logits_loss import get_student_logits_capture

    return {**sidecar, "logits": get_student_logits_capture().pop()}


def finish_dump() -> None:
    """Flush the final teacher tail synchronously after draining async writes."""
    from .logits_saver import get_logits_saver

    saver = get_logits_saver()
    if saver is None or not hasattr(saver, "commit_attempt"):
        return
    saver.initialize()
    saver._write_batched_tar(*saver.take_pending_data())


def reset_runtime() -> None:
    """Clear replay state when Megatron is torn down or restarted in-process."""
    global _PLAN, _HYBRID_LAYOUT
    if _PLAN is not None:
        for reader in _PLAN.get("readers", []):
            reader.close()
    _PLAN = None
    _HYBRID_LAYOUT = False
    from . import cached_logits_loss, logits_saver

    logits_saver._ACTIVE_LOGITS_SAVER = None
    cached_logits_loss._ACTIVE_STUDENT_LOGITS_CAPTURE = None
