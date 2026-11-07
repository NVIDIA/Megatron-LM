# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Small integration hooks for paired dumping, replay, and checkpoint resume."""

import json
import os
from typing import Any, Callable

import torch
import torch.distributed as dist

from .v3_format import (
    FORMAT_VERSION,
    attention_mode_for_layout,
    hf_tokenizer_identity,
    mismatched_settings,
    segment_padding_divisor,
    tokenizer_fingerprint,
    validate_metadata,
)
from .v3_replay import ReplayIterator, ReplayReader, describe_hole, discover_groups, layout_options
from .v3_storage import (
    Storage,
    complete_ranges,
    contiguous_end,
    discard_unpublished,
    has_legacy_tars,
    list_published,
    parse_name,
    read_meta,
)

_PLAN = None
_HYBRID_LAYOUT = False
# One-slot handoff of the current forward's targets from get_batch to forward_step.
_PENDING_TARGETS = None
_STAGED_TARGETS = None


def validate_options(args: Any) -> None:
    """Validate only opted-in v3 behavior; legacy flags keep their semantics."""
    from .corpus_runtime import validate_corpus_dump

    validate_corpus_dump(args)
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
    frozen = getattr(args, "freeze_all_layers", False)
    if frozen:
        # A frozen teacher still runs in train mode (the saver hooks rely on it), so any
        # stochastic or batch-dependent layer would make its targets irreproducible.
        stochastic = [
            flag
            for flag, value in (
                ("--hidden-dropout", getattr(args, "hidden_dropout", 0)),
                ("--attention-dropout", getattr(args, "attention_dropout", 0)),
                ("--moe-input-jitter-eps", getattr(args, "moe_input_jitter_eps", None)),
                ("--moe-expert-capacity-factor", getattr(args, "moe_expert_capacity_factor", None)),
                ("--moe-token-dropping", getattr(args, "moe_token_dropping", False)),
            )
            if value
        ]
        if stochastic:
            raise ValueError(
                "A frozen offline KD teacher must score deterministically; disable "
                f"{', '.join(stochastic)} (dropout defaults to 0.1)"
            )
    start = getattr(args, "override_ckpt_iteration", None)
    if frozen and start is not None and start % args.save_interval:
        raise ValueError(
            "Paired teacher dumping requires --override-ckpt-iteration to be a multiple of "
            "--save-interval"
        )


def _tokenizer_attr(tokenizer: Any, name: str) -> Any:
    try:
        return getattr(tokenizer, name)
    except (AttributeError, NotImplementedError):
        return None


def tokenizer_identity() -> dict[str, Any]:
    """Describe the tokenizer with the format's canonical fingerprint.

    Hugging Face tokenizers (wrapped by Megatron) are fingerprinted exactly as a
    non-Megatron producer would, from ``get_vocab()`` and HF special-token IDs.
    Other tokenizers use their vocabulary mapping in the same canonical form.
    """
    from megatron.training import get_tokenizer

    tokenizer = get_tokenizer()
    eod = _tokenizer_attr(tokenizer, "eod")
    inner = getattr(tokenizer, "_tokenizer", None)
    for candidate in (getattr(inner, "tokenizer", None), inner):
        if (
            candidate is not None
            and hasattr(candidate, "get_vocab")
            and hasattr(candidate, "__len__")
        ):
            return hf_tokenizer_identity(candidate, eod_id=eod)
    try:
        vocab = tokenizer.vocab
    except NotImplementedError:
        if type(inner or tokenizer).__name__ != "NullTokenizer":
            raise ValueError("Offline KD v3 requires an explicit vocabulary mapping") from None
        vocab = {str(index): index for index in range(int(tokenizer.vocab_size))}
    if not isinstance(vocab, dict):
        raise ValueError("Offline KD v3 requires a tokenizer exposing its vocabulary mapping")
    special_ids = {role: _tokenizer_attr(tokenizer, role) for role in ("bos", "eos", "pad", "unk")}
    return {
        "sha256": tokenizer_fingerprint(vocab, special_ids, len(vocab)),
        "size": len(vocab),
        "special_ids": special_ids,
        "eod_id": eod,
    }


def student_padding_divisor(args: Any) -> int:
    """Per-segment padding the student's packed (SFT) batches need for its CP/SP layout."""
    if not args.sft:
        return 1
    return segment_padding_divisor(
        context_parallel_size=args.context_parallel_size,
        sequence_parallel_tp=args.tensor_model_parallel_size * bool(args.sequence_parallel),
    )


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
    return contiguous_end(complete_ranges(list_published(storage), dp_size), start, end)


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


def _teacher_identity(args: Any) -> str:
    """The teacher checkpoint: ``--load`` plus the iteration it resolves to, if known."""
    step = getattr(args, "ckpt_step", None)
    if step is None and args.load:
        try:
            with Storage(args.load).open("latest_checkpointed_iteration.txt") as tracker:
                step = tracker.read().decode().strip()
        except Exception:
            step = None
    return f"{args.load}@{step}" if step is not None else str(args.load)


def _teacher_precision(args: Any) -> str:
    """The teacher's numerics, e.g. ``bf16`` or ``bf16+fp8:e4m3/delayed`` (provenance only)."""
    names = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}
    precision = names.get(
        getattr(args, "params_dtype", None), str(getattr(args, "params_dtype", None))
    )
    for low in ("fp8", "fp4"):
        if getattr(args, low, None):
            precision += f"+{low}:{getattr(args, low)}/{getattr(args, f'{low}_recipe', None)}"
    return precision


def _dump_settings(saver: Any) -> dict[str, Any]:
    """This dump job's metadata: required settings (minus ``first_sample``), recorded
    settings, and provenance."""
    from megatron.training import get_args

    from .utils import _blend_identifiers

    args = get_args()
    targets = saver.metadata_dict["saver"]
    settings = {
        "format_version": FORMAT_VERSION,
        "contract": {
            "tokenizer": tokenizer_identity(),
            "logprob_vocab_size": args.padded_vocab_size,
            "attention_mode": attention_mode_for_layout(
                sft=args.sft,
                inter_document_masking=args.dataloader_inter_document_masking,
                reset_attention_mask=args.reset_attention_mask,
                create_attention_mask=args.create_attention_mask_in_dataloader,
            ),
            # Megatron's position embeddings (learned, rope, yarn, mrope) depend only on
            # the position, so a causal teacher's row t never depends on later tokens.
            "prefix_stable": True,
            "writers": saver.dp_size,
            "writer_mbs": args.micro_batch_size,
        },
        "recorded": {
            "teacher": _teacher_identity(args),
            "seq_length": args.seq_length,
            "padding_divisor": student_padding_divisor(args),
            "data": _blend_identifiers(args),
            "split": getattr(args, "split", None),
            "seed": args.seed,
            "num_samples": args.train_samples or args.train_iters * args.global_batch_size,
            "builder": (
                "SFTDataset"
                if args.sft
                else "MockGPTDataset" if getattr(args, "mock_data", False) else "GPTDataset"
            ),
        },
        "provenance": {
            "producer": "megatron",
            "teacher_precision": _teacher_precision(args),
            "top_k": targets["k"],
            "top_p": targets["p"],
            "min_k": targets["min_k"],
            "save_dtype": targets["save_dtype"],
            "cp_size": saver.cp_size,
            "save_interval": args.save_interval,
        },
    }
    if getattr(args, "kd_corpus_dump", None):
        # A corpus source's extent is part of its contract: it defines completeness.
        settings["recorded"].pop("num_samples")
        settings["contract"]["num_samples"] = args.kd_corpus_samples
        settings["recorded"]["split"] = f"{settings['recorded']['split']}:{args.kd_corpus_split}"
    # Compare exactly what a JSON round trip through the files' schema metadata preserves.
    return json.loads(json.dumps(settings))


def _report_recorded_differences(differing: list[str], where: str) -> None:
    """Warn (world rank 0) about recorded settings that differ; they never block."""
    if differing and (not dist.is_initialized() or dist.get_rank() == 0):
        print(
            f"WARNING: offline KD {where} differ in recorded settings ({', '.join(differing)}); "
            "files remain usable, but check this was intended",
            flush=True,
        )


def _check_corpus_sources(root: str, own: str, expected: dict) -> list[str]:
    """Compare this source with one published file of every other source in the corpus.

    All sources of a corpus must share a tokenizer; other differences (for example the
    teacher) are returned for reporting.
    """
    storage = Storage(root)
    seen, differing = set(), set()
    for name in storage.list("*/*/dp*__*.index.json"):
        folder = os.path.dirname(name)
        if folder == own or folder in seen:
            continue
        seen.add(folder)
        meta = read_meta(Storage(os.path.join(root, folder)), parse_name(os.path.basename(name)))
        if meta["contract"]["tokenizer"] != expected["contract"]["tokenizer"]:
            raise ValueError(f"Corpus source {folder} uses a different tokenizer")
        differing.update(mismatched_settings(meta, expected)[1])
    return sorted(differing & {"teacher"})


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
            if has_legacy_tars(storage):
                raise ValueError("Cannot mix legacy and v3 dumps in the same directory")
            files = list_published(storage)
            first_sample = 0 if frozen else args.consumed_train_samples
            differing = []
            root = getattr(args, "kd_corpus_root", None)
            if root is not None:
                own = os.path.relpath(saver.save_dir, root)
                differing += _check_corpus_sources(root, own, expected)
            if files:
                existing = read_meta(storage, files[0])
                validate_metadata(existing, corpus=root is not None)
                first_sample = existing["contract"]["first_sample"]
                required, recorded = mismatched_settings(
                    existing,
                    {
                        **expected,
                        "contract": {**expected["contract"], "first_sample": first_sample},
                    },
                )
                if required:
                    raise ValueError(
                        f"Teacher settings differ from existing v3 files ({', '.join(required)}); "
                        "use a new save directory"
                    )
                differing += recorded
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
                    "contract": {**expected["contract"], "first_sample": first_sample},
                    "published_through": published,
                    "range_end": end,
                },
                "differing": differing,
            }
        except Exception as error:
            return {"error": f"{type(error).__name__}: {error}"}

    result = _broadcast_without_pp(create)
    if "error" in result:
        raise RuntimeError(result["error"])
    _report_recorded_differences(result["differing"], "dump job and existing v3 files")
    return result["metadata"]


def _check_student_compatibility(args: Any, metadata: dict) -> None:
    validate_metadata(metadata)
    contract = metadata["contract"]
    if contract["tokenizer"]["sha256"] != tokenizer_identity()["sha256"]:
        raise ValueError("Offline KD v3 tokenizer/vocabulary mismatch")
    if args.padded_vocab_size < contract["tokenizer"]["size"]:
        raise ValueError(
            f"Student vocabulary ({args.padded_vocab_size}) is smaller than the tokenizer "
            f"({contract['tokenizer']['size']})"
        )
    mode = attention_mode_for_layout(
        sft=args.sft,
        inter_document_masking=args.dataloader_inter_document_masking,
        reset_attention_mask=args.reset_attention_mask,
        create_attention_mask=args.create_attention_mask_in_dataloader,
    )
    if mode != contract["attention_mode"]:
        raise ValueError(
            f"Student attention_mode {mode} differs from the cache's {contract['attention_mode']}"
        )
    # Sample lengths are checked per sample at collation (see v3_replay._layout_sample),
    # not against the producer's recorded sequence length or padding divisor.
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
            if has_legacy_tars(storage):
                raise ValueError("Input replay requires a v3 dump; found legacy tars")
            files = list_published(storage)
            if not files:
                raise FileNotFoundError("No v3 files found; input replay requires a v3 dump")
            metadata = read_meta(storage, files[0])
            _check_student_compatibility(args, metadata)
            groups, hole = discover_groups(storage, metadata)
            replay_end = getattr(args, "logits_load_replay_end", None)
            if shuffle:
                if replay_end is None:
                    # Fix the shuffled extent at the first launch; resumes reuse it.
                    replay_end = groups[-1][1] if groups else metadata["contract"]["first_sample"]
                published_end = groups[-1][1] if groups else metadata["contract"]["first_sample"]
                if published_end < replay_end:
                    raise ValueError(
                        f"Shuffled replay was planned over samples up to {replay_end}, but only "
                        f"{published_end} are published"
                    )
                groups = [group for group in groups if group[1] <= replay_end]
                if not groups:
                    raise ValueError("No complete teacher shard group is available to shuffle")
            available = sum(end - start for start, end in groups)
            passes = 1
            if shuffle:
                # A fixed extent repeats in whole passes, each freshly shuffled, enough for the
                # student budget (and any resumed cursor); a larger budget on resume only
                # appends passes, keeping earlier ones intact.
                budget = args.train_samples or args.train_iters * args.global_batch_size
                needed = max(budget, args.consumed_train_samples)
                passes = max(1, -(-needed // available))
                available *= passes
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
                "passes": passes,
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
    passes = result[0]["passes"]
    _PLAN = {
        **result[0],
        "available": passes * sum(end - start for start, end in result[0]["groups"]),
        "readers": [],
    }
    if rank == 0 and result[0]["hole"] is not None:
        print(f"WARNING: offline KD v3 cache: {describe_hole(result[0]['hole'])}", flush=True)
    if rank == 0 and passes > 1:
        print(
            f"Offline KD v3 replay repeats its {_PLAN['available'] // passes} cached samples "
            f"{passes} times to cover the student budget",
            flush=True,
        )


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
    if args.logits_load_shuffle_shards is None:  # Only a sequential plan grows.
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
        initialize_replay(args)  # Once, collectively, before the first replay batch.
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
    this iterator scatters each rank its inputs and its own target shard. The
    reader is created on first use, after the replay plan has been agreed.
    """

    def __init__(self, source_factory: Callable[[], Any] | None, cp_rank: int, cp_size: int):
        self.source_factory = source_factory
        self._source = None
        self.cp_rank = cp_rank
        self.cp_size = cp_size
        if cp_size > 1:
            from megatron.core import parallel_state as mpu

            self.group = mpu.get_context_parallel_group()
            self.src = dist.get_global_rank(self.group, 0)

    @property
    def source(self) -> Any:
        """The replay iterator on CP rank 0, created on first use."""
        if self._source is None and self.source_factory is not None:
            self._source = self.source_factory()
        return self._source

    @property
    def reader(self) -> ReplayReader | None:
        """The underlying reader on CP rank 0."""
        return getattr(self.source, "reader", None)

    def __iter__(self) -> "_ContextParallelReplay":
        return self

    @staticmethod
    def _for_rank(batch: dict, rank: int) -> dict:
        local = dict(batch)
        for key in ("_kd_values", "_kd_indices", "_kd_tail"):
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
        if self._source is not None:
            self._source.close()


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

    def source_factory() -> ReplayIterator:
        reader = ReplayReader(
            Storage(args.logits_load_dir),
            _PLAN["metadata"],
            _PLAN["groups"],
            targets=mpu.is_pipeline_last_stage(),
            shuffle=args.logits_load_shuffle_shards is not None,
            seed=args.logits_load_shuffle_shards or 0,
            passes=_PLAN["passes"],
            decode_threads=args.logits_load_decode_threads,
            prefetch_depth=args.logits_load_msc_prefetch_depth,
            student_vocab_size=args.padded_vocab_size,
        )
        _PLAN["readers"].append(reader)
        return ReplayIterator(
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
            divisor=student_padding_divisor(args),
        )

    iterator = _ContextParallelReplay(source_factory if cp_rank == 0 else None, cp_rank, cp_size)

    class Dataset(torch.utils.data.IterableDataset):
        def __iter__(self):
            return iterator

    # Iterator creation must not consume the default CPU RNG restored from a checkpoint.
    loader_generator = torch.Generator()
    loader_generator.manual_seed(torch.initial_seed())
    return torch.utils.data.DataLoader(
        Dataset(), batch_size=None, pin_memory=True, num_workers=0, generator=loader_generator
    )


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


def on_tp0_batch(batch: dict, *, hybrid: bool = False, vp_stage: int | None = None) -> dict:
    """Entrypoint hook on TP rank 0, right after ``next(data_iterator)``.

    Dumping captures the raw batch; replay strips this CP rank's targets from the
    model batch and holds them until :func:`stage_targets`.
    """
    global _PENDING_TARGETS
    from megatron.training import get_args

    args = get_args()
    if getattr(args, "logits_save_inputs", False):
        capture_dump_batch(batch, hybrid=hybrid, vp_stage=vp_stage)
    if getattr(args, "logits_load_inputs", False) and "_kd_sample_ids" in batch:
        batch, _PENDING_TARGETS = prepare_replay_batch(batch)
    return batch


def stage_targets(*, vp_stage: int | None = None) -> None:
    """Entrypoint hook at the end of ``get_batch`` on every rank that reaches it.

    Broadcasts the pending targets to the final stage's TP ranks and stages them
    for :func:`take_targets` in the same forward step.
    """
    global _PENDING_TARGETS, _STAGED_TARGETS
    _STAGED_TARGETS = broadcast_targets(_PENDING_TARGETS, vp_stage=vp_stage)
    _PENDING_TARGETS = None


def take_targets() -> dict | None:
    """Entrypoint hook in ``forward_step`` after the model call.

    Binds the staged targets to this forward's student logits for its loss closure.
    Reruns re-read the cached batch, so targets are re-staged rather than lost.
    """
    global _STAGED_TARGETS
    staged, _STAGED_TARGETS = _STAGED_TARGETS, None
    return bind_student_logits(staged)


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
            "tail": batch["_kd_tail"],
            "sample_ids": batch["_kd_sample_ids"],
        }
    for key in ("_kd_values", "_kd_indices", "_kd_tail", "_kd_sample_ids"):
        result.pop(key, None)
    if args.create_attention_mask_in_dataloader:
        from megatron.core.datasets.gpt_dataset import _get_ltor_masks_and_position_ids

        eod = _PLAN["metadata"]["contract"]["tokenizer"]["eod_id"]
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
            "tail": torch.empty(shape[:-1], dtype=torch.float32, device=device),
            "sample_ids": ids,
        }
    else:
        sidecar = dict(sidecar)
        for field in ("values", "indices", "tail"):
            tensor = sidecar[field]
            if torch.cuda.is_available() and tensor.device.type == "cpu" and not tensor.is_pinned():
                tensor = tensor.pin_memory()
            sidecar[field] = tensor.to(device, non_blocking=True)
    if dist.is_initialized():
        for field in ("values", "indices", "tail"):
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
    global _PLAN, _HYBRID_LAYOUT, _PENDING_TARGETS, _STAGED_TARGETS
    _PENDING_TARGETS = _STAGED_TARGETS = None
    if _PLAN is not None:
        for reader in _PLAN.get("readers", []):
            reader.close()
    _PLAN = None
    _HYBRID_LAYOUT = False
    from . import cached_logits_loss, logits_saver

    logits_saver._ACTIVE_LOGITS_SAVER = None
    cached_logits_loss._ACTIVE_STUDENT_LOGITS_CAPTURE = None
