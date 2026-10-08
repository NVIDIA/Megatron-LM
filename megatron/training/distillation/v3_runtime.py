# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Small integration hooks for paired dumping, replay, and checkpoint resume."""

import json
import uuid
from typing import Any

import torch
import torch.distributed as dist

from .v3_format import capture_batch, digest
from .v3_replay import (
    ReplayIterator,
    ReplayReader,
    discover_groups,
    layout_options,
    map_targets,
    token_map,
)
from .v3_storage import CACHE_FILE, COMPLETE_FILE, Storage, quarantine_unpublished

_PLAN = None


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
    if saving and getattr(args, "recompute_granularity", None) == "full":
        raise ValueError(
            "Paired teacher dumping does not support full-layer activation recomputation"
        )
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


def initialize_dump_metadata(saver: Any) -> dict:
    """Create or restore one cache generation on the saver stage collectively."""
    from megatron.training import get_args

    from .utils import _broadcast_without_pp

    args = get_args()
    expected = {
        "format_version": 3,
        "dp_size_save": saver.dp_size,
        "mbs_save": args.micro_batch_size,
        "gbs_save": args.global_batch_size,
        "cp_size_save": saver.cp_size,
        "seq_length": args.seq_length,
        "sft": bool(args.sft),
        "inter_document_masking": bool(args.dataloader_inter_document_masking),
        "reset_attention_mask": bool(args.reset_attention_mask),
        "tokenizer": tokenizer_identity(),
        "padded_vocab_size": args.padded_vocab_size,
        "targets": {**saver.metadata_dict["saver"], "format_version": 3},
        "dataset_identity": saver.metadata_dict["identifiers"],
        "teacher_checkpoint": str(args.load),
        "boundary_convention": "exact_dataset_boundaries_may_include_padding",
    }

    def create():
        try:
            storage = Storage(saver.save_dir)
            if storage.exists(CACHE_FILE):
                metadata = storage.read_json(CACHE_FILE)
                comparisons = dict(expected)
                # A training teacher can resume from its output checkpoint while
                # the original checkpoint remains recorded as provenance.
                if args.consumed_train_samples > metadata["first_sample"]:
                    comparisons.pop("teacher_checkpoint")
                if any(metadata.get(key) != value for key, value in comparisons.items()):
                    raise ValueError(
                        "Teacher settings differ from existing v3 cache; use a new save directory"
                    )
            else:
                if storage.list("*.tar"):
                    raise ValueError("Cannot mix legacy and v3 dumps in the same directory")
                metadata = {
                    **expected,
                    "generation": uuid.uuid4().hex,
                    "first_sample": args.consumed_train_samples,
                }
                storage.write_json(CACHE_FILE, metadata)
            groups = discover_groups(storage, metadata)
            published = groups[-1][0]["records"][-1]["end"] if groups else metadata["first_sample"]
            if not metadata["first_sample"] <= args.consumed_train_samples <= published:
                raise ValueError(
                    "Teacher checkpoint cursor is outside the published v3 prefix; "
                    "resume an earlier checkpoint or use a new cache directory"
                )
            quarantine_unpublished(storage, published)
            if storage.exists(COMPLETE_FILE):
                # The resumed writer can extend this generation. Until it exits
                # successfully, students must not treat it as immutable.
                storage.remove(COMPLETE_FILE)
            return {"metadata": {**metadata, "published_through": published}}
        except Exception as error:
            return {"error": f"{type(error).__name__}: {error}"}

    result = _broadcast_without_pp(create)
    if "error" in result:
        raise RuntimeError(result["error"])
    return result["metadata"]


def initialize_replay(args: Any) -> None:
    """Establish a common replay plan on all ranks before building datasets."""
    global _PLAN
    if not getattr(args, "logits_load_inputs", False) or _PLAN is not None:
        return
    rank = dist.get_rank() if dist.is_initialized() else 0
    result = [None]
    if rank == 0:
        try:
            storage = Storage(args.logits_load_dir)
            if not storage.exists(CACHE_FILE):
                raise FileNotFoundError("No v3 cache header found; input replay requires a v3 dump")
            metadata = storage.read_json(CACHE_FILE)
            if metadata.get("format_version") != 3 or metadata["tokenizer"] != tokenizer_identity():
                raise ValueError("Offline KD v3 tokenizer/vocabulary mismatch")
            if args.padded_vocab_size != metadata["padded_vocab_size"]:
                raise ValueError(
                    "V3 teacher/student padded vocabulary sizes must match; "
                    "adjust vocabulary padding when changing TP"
                )
            if (
                bool(args.sft) != metadata["sft"]
                or bool(args.dataloader_inter_document_masking)
                != metadata["inter_document_masking"]
            ):
                raise ValueError("Student packing flags must match the saved v3 layout")
            if bool(args.reset_attention_mask) != metadata["reset_attention_mask"]:
                raise ValueError("Student attention isolation differs from the teacher cache")
            packed = metadata["sft"] or metadata["inter_document_masking"]
            if args.seq_length > metadata["seq_length"] or (
                packed and args.seq_length != metadata["seq_length"]
            ):
                raise ValueError("Only non-packed prefix shortening is supported by v3 replay")
            if args.context_parallel_size > 1 and args.seq_length % (
                2 * args.context_parallel_size
            ):
                raise ValueError("Student sequence length must be divisible by 2 * CP")
            groups = discover_groups(storage, metadata)
            end = groups[-1][0]["records"][-1]["end"] if groups else metadata["first_sample"]
            if args.logits_load_shuffle_shards is not None:
                if not storage.exists(COMPLETE_FILE):
                    raise ValueError("V3 shuffling requires a completed cache")
                complete = storage.read_json(COMPLETE_FILE)
                if (
                    complete["generation"] != metadata["generation"]
                    or complete["end_sample"] != end
                ):
                    raise ValueError("V3 completion marker disagrees with the published cache")
            if args.consumed_train_samples > end - metadata["first_sample"]:
                raise ValueError("Checkpoint sample cursor exceeds the published offline KD cache")
            if not groups and args.logits_load_shuffle_shards is not None:
                raise ValueError("No complete teacher DP shard group is available")
            result[0] = {"metadata": metadata, "groups": groups}
        except Exception as error:
            result[0] = {"error": f"{type(error).__name__}: {error}"}
    if dist.is_initialized():
        dist.broadcast_object_list(result, src=0)
    if "error" in result[0]:
        raise RuntimeError(result[0]["error"])
    _PLAN = result[0]


def refresh_replay_groups(storage: Storage, metadata: dict) -> list[list[dict]]:
    """Share one remote listing among TP-zero DP/CP ranks on this pipeline stage.

    This collective is entered at an iteration boundary by every participating
    reader. Decode prefetch stays within that already-available iteration and
    therefore never enters the collective from a background thread.
    """
    if not storage.remote or not dist.is_initialized():
        return discover_groups(storage, metadata)
    from megatron.core import parallel_state as mpu

    group = mpu.get_data_parallel_group(with_context_parallel=True)
    source = dist.get_process_group_ranks(group)[0]
    result = [None]
    if dist.get_rank() == source:
        try:
            result[0] = {"groups": discover_groups(storage, metadata)}
        except Exception as error:
            result[0] = {"error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(result, src=source, group=group)
    if "error" in result[0]:
        raise RuntimeError(result[0]["error"])
    return result[0]["groups"]


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
    storage = Storage(args.logits_load_dir)
    reader = ReplayReader(
        storage,
        _PLAN["metadata"],
        _PLAN["groups"],
        targets=mpu.is_pipeline_last_stage(),
        shuffle=args.logits_load_shuffle_shards is not None,
        seed=args.logits_load_shuffle_shards if args.logits_load_shuffle_shards is not None else 0,
        decode_threads=args.logits_load_decode_threads,
        refresh_groups=lambda: refresh_replay_groups(storage, _PLAN["metadata"]),
    )
    iterator = ReplayIterator(
        reader,
        consumed=consumed,
        dp_rank=mpu.get_data_parallel_rank(),
        dp_size=mpu.get_data_parallel_world_size(),
        micro_batch_size=args.micro_batch_size,
        seq_length=args.seq_length,
        num_microbatches=get_num_microbatches,
        prefetch=True,
    )

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
        if batch.get("attention_mask") is not None:
            from megatron.core.datasets.gpt_dataset import _get_ltor_masks_and_position_ids
            from megatron.training import get_args, get_tokenizer

            args = get_args()
            expected = torch.stack(
                [
                    _get_ltor_masks_and_position_ids(
                        tokens, get_tokenizer().eod, False, args.reset_attention_mask, False, True
                    )[0]
                    for tokens in batch["tokens"]
                ]
            )
            if not torch.equal(batch["attention_mask"], expected):
                raise ValueError(
                    "V3 currently supports causal/EOD attention masks only; "
                    "custom masks need a future format adapter"
                )
        saver.capture(batch, hybrid=hybrid)


def prepare_replay_batch(batch: dict, *, hybrid: bool = False) -> tuple[dict, dict | None]:
    """Keep replay tensors immutable and separate targets from model batch fields."""
    from megatron.core import parallel_state as mpu
    from megatron.training import get_args

    args = get_args()
    result = dict(batch)
    sidecar = None
    if "_kd_values" in batch:
        inputs = capture_batch(batch)
        mapping = token_map(
            inputs,
            mpu.get_context_parallel_rank(),
            args.context_parallel_size,
            **layout_options(args, hybrid=hybrid),
        )
        values = batch["_kd_values"].reshape(-1, batch["_kd_values"].shape[-1])
        indices = batch["_kd_indices"].reshape_as(values)
        values, indices = map_targets(values, indices, mapping)
        sidecar = {"values": values, "indices": indices, "sample_ids": batch["_kd_sample_ids"]}
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
            # CP index_select creates fresh CPU tensors after DataLoader's
            # pinning pass. Restore pinning for asynchronous target transfer.
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


def finish_dump(*, completed: bool) -> None:
    """Flush the final teacher tail after draining async writes, then mark completion."""
    from .logits_saver import get_logits_saver

    saver = get_logits_saver()
    if saver is None or not hasattr(saver, "commit_attempt"):
        return
    saver.initialize()
    saver._write_batched_tar(*saver.take_pending_data())
    # Synchronize only ranks on the saver stage, not the whole pipeline.
    from megatron.core import parallel_state as mpu

    if dist.is_initialized():
        dist.barrier(group=mpu.get_tensor_and_data_parallel_group(with_context_parallel=True))
    if completed and saver.tp_rank == saver.cp_rank == saver.dp_rank == 0:
        from megatron.training import get_args

        storage = Storage(saver.save_dir)
        metadata = dict(saver.metadata_dict)
        metadata.pop("published_through", None)
        groups = discover_groups(storage, metadata)
        end = groups[-1][0]["records"][-1]["end"] if groups else metadata["first_sample"]
        if end != get_args().consumed_train_samples:
            raise RuntimeError("Teacher v3 completion has unpublished sample ranges")
        storage.write_json(COMPLETE_FILE, {"generation": metadata["generation"], "end_sample": end})


def reset_runtime() -> None:
    """Clear replay state when Megatron is torn down or restarted in-process."""
    global _PLAN
    _PLAN = None
    from . import cached_logits_loss, logits_saver

    logits_saver._ACTIVE_LOGITS_SAVER = None
    cached_logits_loss._ACTIVE_STUDENT_LOGITS_CAPTURE = None
