# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Configuration and deterministic table allocation for Engram."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .variants import DEEPSEEK_VARIANT_NAME, EngramVariant, resolve_variant

TOKENIZER_MAP_FORMAT = "megatron-engram-token-map"
TOKENIZER_MAP_VERSION = 1


def is_prime(value: int) -> bool:
    """Return whether ``value`` is prime using deterministic trial division."""
    if value < 2:
        return False
    if value in (2, 3):
        return True
    if value % 2 == 0 or value % 3 == 0:
        return False
    limit = math.isqrt(value)
    divisor = 5
    while divisor <= limit:
        if value % divisor == 0 or value % (divisor + 2) == 0:
            return False
        divisor += 6
    return True


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    """Return the first unused prime strictly greater than ``start``."""
    candidate = start + 1
    while not is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


_INT64_MAX = (1 << 63) - 1

# Constants of the official Qwen splitmix64 hash-constant generator.
_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_QWEN_LAYER_SEED_PRIME = 10007


def _splitmix64(value: int) -> int:
    """One official splitmix64 step over unsigned 64-bit integers."""
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def build_qwen_layer_multipliers(
    unigram_vocab_size: int, max_ngram_order: int, ple_layer_index: int, seed: int
) -> tuple[int, ...]:
    """Generate the official Qwen odd multipliers for one PLE layer.

    Multipliers are bounded by ``int64_max // unigram_vocab_size`` so ``token_id * multiplier``
    never overflows signed int64, mirroring the reference implementation exactly.
    """
    half_bound = max(1, (_INT64_MAX // max(unigram_vocab_size, 1)) // 2)
    base_seed = seed + _QWEN_LAYER_SEED_PRIME * ple_layer_index
    multipliers = []
    for index in range(max_ngram_order):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
    return tuple(multipliers)


def qwen_head_primes(vocab_size_base: int, count: int) -> list[int]:
    """Return the first ``count`` primes strictly greater than ``vocab_size_base - 1``.

    Single incremental scan, so allocating every head costs one pass rather than one pass
    per head.
    """
    primes: list[int] = []
    value = vocab_size_base - 1
    while len(primes) < count:
        value += 1
        while not is_prime(value):
            value += 1
        primes.append(value)
    return primes


def allocate_qwen_table_sizes(
    vocab_size_base: int, layer_ids: tuple[int, ...], max_ngram_order: int, num_hash_heads: int
) -> dict[int, tuple[int, ...]]:
    """Allocate the official Qwen per-head prime table sizes.

    Head ``h`` of PLE layer index ``li`` uses the ``(li * heads + h + 1)``-th prime after
    ``vocab_size_base - 1``, with heads laid out order-major (all bigram heads, then trigram).
    """
    heads_per_layer = (max_ngram_order - 1) * num_hash_heads
    primes = qwen_head_primes(vocab_size_base, heads_per_layer * len(layer_ids))
    return {
        layer_id: tuple(primes[index * heads_per_layer : (index + 1) * heads_per_layer])
        for index, layer_id in enumerate(layer_ids)
    }


def allocate_table_sizes(
    global_vocab_sizes: tuple[int, ...], layer_ids: tuple[int, ...], num_hash_heads: int
) -> dict[int, tuple[int, ...]]:
    """Allocate the official distinct prime table sizes for every layer and head."""
    seen_primes: set[int] = set()
    result: dict[int, tuple[int, ...]] = {}
    for layer_id in layer_ids:
        layer_sizes = []
        for vocab_size in global_vocab_sizes:
            search_start = vocab_size - 1
            for _ in range(num_hash_heads):
                prime = find_next_prime(search_start, seen_primes)
                seen_primes.add(prime)
                layer_sizes.append(prime)
                search_start = prime
        result[layer_id] = tuple(layer_sizes)
    return result


def _resolve_boundary_token_id(args: Any, variant: EngramVariant) -> int:
    """Return the boundary token for ``variant``, rejecting the other variant's flags.

    Each variant owns exclusive CLI flags; a mis-paired flag set means the operator expected
    different semantics, so it fails loudly instead of being silently ignored.
    """
    if variant.uses_tokenizer_artifact:
        if args.engram_eos_token_id is not None or args.engram_unigram_vocab_size is not None:
            raise ValueError(
                "--engram-eos-token-id and --engram-unigram-vocab-size do not apply to "
                f"--engram-variant {variant.name}."
            )
        return 0 if args.engram_pad_token_id is None else args.engram_pad_token_id
    if args.engram_tokenizer_map is not None:
        raise ValueError(
            f"--engram-tokenizer-map does not apply to --engram-variant {variant.name}, which "
            "hashes raw token IDs without an artifact."
        )
    if args.engram_pad_token_id is not None:
        raise ValueError(
            f"--engram-pad-token-id does not apply to --engram-variant {variant.name}, which "
            "resets n-gram windows at --engram-eos-token-id instead of padding."
        )
    if args.engram_eos_token_id is None:
        raise ValueError(f"--engram-eos-token-id is required for --engram-variant {variant.name}.")
    return args.engram_eos_token_id


def _validate_packed_sequence_args(args: Any) -> bool:
    """Return whether packed (THD) rows are in use, rejecting incompatible packing setups."""
    # Mirrors pretrain_gpt's is_packed_sequence: SBHD validation keeps the dense layout even
    # though it reads the varlen dataset, so it is not a packed (THD) run.
    packed_sequences = bool(
        getattr(args, "sft", False)
        or (
            getattr(args, "use_varlen_dataset", False)
            and not getattr(args, "varlen_sbhd_validation", False)
        )
    )
    if getattr(args, "sequence_packing_scheduler", None) is not None:
        # Scheduler microbatches are variable-length and bypass the token prefetch entirely.
        # Note --use-varlen-dataset always auto-selects a scheduler, so the only compatible
        # packed path is the --sft family (fixed-capacity packed rows).
        raise ValueError(
            "Engram does not support sequence-packing schedulers (including the one "
            "--use-varlen-dataset auto-selects); use the --sft packed path instead."
        )
    if packed_sequences and args.pipeline_model_parallel_size > 1:
        if args.max_seqlen_per_dp_cp_rank != args.seq_length:
            # The prefetch pads tokens to seq_length while pad_sequence_for_thd pads to
            # max_seqlen_per_dp_cp_rank; the identical-padding invariant needs equality.
            raise ValueError(
                "Engram with packed sequences and pipeline parallelism requires "
                f"--max-seqlen-per-dp-cp-rank ({args.max_seqlen_per_dp_cp_rank}) to equal "
                f"--seq-length ({args.seq_length})."
            )
    return packed_sequences


@dataclass
class EngramConfig:
    """Configuration for Engram / PLE n-gram memory.

    The per-variant conventions live in :mod:`megatron.core.models.engram.variants`; this
    class holds the shared shape/schedule configuration and the resolved hash constants.

    Args:
        global_vocab_sizes: Table row budget per n-gram order. Variants that allocate
            per-head primes from a single base require every entry to be equal.
        boundary_token_id: Token that fills the n-gram window before the start of a
            document. Variants that reset windows at boundaries also restart the window at
            every occurrence of this token, so it is the EOS ID for those.
        tokenizer_map_path: Versioned offline artifact, required by artifact-based variants.
        unigram_vocab_size: Vocabulary that bounds the generated hash multipliers, required
            by variants that generate their constants in process.
    """

    global_vocab_sizes: tuple[int, ...]
    layer_ids: tuple[int, ...]
    max_ngram_order: int
    num_hash_heads: int
    memory_dim: int
    kernel_size: int
    hash_seed: int
    boundary_token_id: int
    tokenizer_map_path: str = ""
    variant: str = DEEPSEEK_VARIANT_NAME
    unigram_vocab_size: int | None = None
    embedding_lr_multiplier: float = 5.0
    embedding_weight_decay: float = 0.0
    variant_spec: EngramVariant = field(init=False, repr=False)
    tokenizer_remap: torch.Tensor | None = field(init=False, repr=False, default=None)
    hash_boundary_token_id: int = field(init=False)
    layer_multipliers: dict[int, tuple[int, ...]] = field(init=False, repr=False)
    table_sizes_by_layer: dict[int, tuple[int, ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.global_vocab_sizes = tuple(self.global_vocab_sizes)
        self.layer_ids = tuple(self.layer_ids)
        self.variant_spec = resolve_variant(self.variant)
        self._validate_values()
        if self.variant_spec.uses_tokenizer_artifact:
            self._load_tokenizer_map()
            self.table_sizes_by_layer = allocate_table_sizes(
                self.global_vocab_sizes, self.layer_ids, self.num_hash_heads
            )
        else:
            self._generate_hash_constants()

    def _generate_hash_constants(self) -> None:
        """Generate the official hash constants in process (no artifact needed)."""
        # Raw token IDs are hashed directly; there is no compression remap.
        self.tokenizer_remap = None
        self.hash_boundary_token_id = self.boundary_token_id
        self.layer_multipliers = {
            layer_id: build_qwen_layer_multipliers(
                self.unigram_vocab_size, self.max_ngram_order, ple_layer_index, self.hash_seed
            )
            for ple_layer_index, layer_id in enumerate(self.layer_ids)
        }
        self.table_sizes_by_layer = allocate_qwen_table_sizes(
            self.global_vocab_sizes[0], self.layer_ids, self.max_ngram_order, self.num_hash_heads
        )

    @classmethod
    def from_args(cls, args: Any, transformer_config: Any) -> EngramConfig | None:
        """Build and validate n-gram memory configuration from Megatron CLI/YAML arguments."""
        if getattr(args, "engram_vocab_sizes", None) is None:
            return None
        variant = resolve_variant(getattr(args, "engram_variant", DEEPSEEK_VARIANT_NAME))
        config = cls(
            global_vocab_sizes=tuple(args.engram_vocab_sizes),
            layer_ids=tuple(args.engram_layer_ids or ()),
            max_ngram_order=args.engram_max_ngram_order,
            num_hash_heads=args.engram_num_hash_heads,
            memory_dim=args.engram_memory_dim,
            kernel_size=args.engram_kernel_size,
            hash_seed=args.engram_hash_seed,
            boundary_token_id=_resolve_boundary_token_id(args, variant),
            tokenizer_map_path=args.engram_tokenizer_map or "",
            variant=variant.name,
            unigram_vocab_size=getattr(args, "engram_unigram_vocab_size", None),
            embedding_lr_multiplier=args.engram_embedding_lr_multiplier,
            embedding_weight_decay=args.engram_embedding_weight_decay,
        )
        packed_sequences = _validate_packed_sequence_args(args)
        config.validate_startup(
            transformer_config,
            packed_sequences=packed_sequences,
            sequence_length=getattr(args, "seq_length", None),
            use_fsdp=bool(
                getattr(args, "use_torch_fsdp2", False) or getattr(args, "use_megatron_fsdp", False)
            ),
        )
        if getattr(args, "padded_vocab_size", None) is not None:
            config.validate_vocabulary(args.padded_vocab_size)
        # Paths that build a TransformerConfig without the argument namespace (YAML configs,
        # callers supplying an explicit config) would otherwise leave this False while the
        # modules are built, which silently changes the checkpoint layer-key layout.
        transformer_config.engram_enabled = True
        return config

    @property
    def num_tables(self) -> int:
        """Number of prime-sized tables in one Engram module."""
        return (self.max_ngram_order - 1) * self.num_hash_heads

    @property
    def head_dim(self) -> int:
        """Embedding width of one hash head."""
        return self.memory_dim // self.num_hash_heads

    @property
    def total_memory_dim(self) -> int:
        """Concatenated retrieved-memory width over all n-gram orders."""
        return (self.max_ngram_order - 1) * self.memory_dim

    def table_sizes(self, layer_id: int) -> tuple[int, ...]:
        """Return prime table sizes for a selected global layer."""
        try:
            return self.table_sizes_by_layer[layer_id]
        except KeyError as exc:
            raise ValueError(f"Engram is not configured for global layer {layer_id}.") from exc

    def multipliers(self, layer_id: int) -> tuple[int, ...]:
        """Return official hash multipliers for a selected global layer."""
        try:
            return self.layer_multipliers[layer_id]
        except KeyError as exc:
            raise ValueError(f"Tokenizer map has no multipliers for layer {layer_id}.") from exc

    def _validate_values(self) -> None:
        if self.max_ngram_order < 2:
            raise ValueError("engram_max_ngram_order must be at least 2.")
        expected_vocab_sizes = self.max_ngram_order - 1
        if len(self.global_vocab_sizes) != expected_vocab_sizes:
            raise ValueError(
                "engram_vocab_sizes must contain exactly one value for each n-gram order "
                f"2..{self.max_ngram_order}; expected {expected_vocab_sizes}, "
                f"got {len(self.global_vocab_sizes)}."
            )
        if any(size <= 0 for size in self.global_vocab_sizes):
            raise ValueError("Every engram_vocab_sizes value must be positive.")
        if not self.layer_ids:
            raise ValueError("engram_layer_ids must contain at least one 1-based layer ID.")
        if len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError("engram_layer_ids must be unique.")
        if any(layer_id < 1 for layer_id in self.layer_ids):
            raise ValueError("engram_layer_ids are 1-based and must be positive.")
        if self.num_hash_heads <= 0:
            raise ValueError("engram_num_hash_heads must be positive.")
        if self.memory_dim <= 0:
            raise ValueError("engram_memory_dim must be positive.")
        if self.memory_dim % self.num_hash_heads != 0:
            raise ValueError(
                "engram_memory_dim must be divisible by engram_num_hash_heads; "
                f"got {self.memory_dim} and {self.num_hash_heads}."
            )
        if self.kernel_size <= 0:
            raise ValueError("engram_kernel_size must be positive.")
        if self.boundary_token_id < 0:
            raise ValueError(f"{self.variant_spec.boundary_token_flag} must be nonnegative.")
        if self.embedding_lr_multiplier <= 0:
            raise ValueError("engram_embedding_lr_multiplier must be positive.")
        if self.embedding_weight_decay < 0:
            raise ValueError("engram_embedding_weight_decay must be nonnegative.")
        if self.variant_spec.uses_tokenizer_artifact:
            if not self.tokenizer_map_path:
                raise ValueError(
                    f"engram_tokenizer_map is required for the {self.variant} variant."
                )
            return
        if self.unigram_vocab_size is None or self.unigram_vocab_size <= 0:
            raise ValueError(
                f"The {self.variant} variant requires a positive engram_unigram_vocab_size "
                "(the HF config.vocab_size that bounds the hash multipliers)."
            )
        if self.boundary_token_id >= self.unigram_vocab_size:
            # The multiplier overflow bound only covers IDs below unigram_vocab_size, and an
            # out-of-vocabulary boundary token would silently never reset any window.
            raise ValueError(
                f"{self.variant_spec.boundary_token_flag} ({self.boundary_token_id}) must be "
                f"below engram_unigram_vocab_size ({self.unigram_vocab_size})."
            )
        if any(size != self.global_vocab_sizes[0] for size in self.global_vocab_sizes):
            raise ValueError(
                f"The {self.variant} variant uses one shared ngram_vocab_size_base; every "
                f"engram_vocab_sizes value must be equal, got {self.global_vocab_sizes}."
            )

    def _load_tokenizer_map(self) -> None:
        path = Path(self.tokenizer_map_path)
        if not path.is_file():
            raise ValueError(f"Engram tokenizer-map artifact does not exist: {path}")
        with path.open("r", encoding="utf-8") as artifact_file:
            artifact = json.load(artifact_file)

        if artifact.get("format") != TOKENIZER_MAP_FORMAT:
            raise ValueError(
                "Invalid Engram tokenizer-map format; expected "
                f"{TOKENIZER_MAP_FORMAT!r}, got {artifact.get('format')!r}."
            )
        if artifact.get("version") != TOKENIZER_MAP_VERSION:
            raise ValueError(
                "Unsupported Engram tokenizer-map version; expected "
                f"{TOKENIZER_MAP_VERSION}, got {artifact.get('version')!r}."
            )
        for field_name, expected in (
            ("max_ngram_order", self.max_ngram_order),
            ("hash_seed", self.hash_seed),
            ("pad_token_id", self.boundary_token_id),
        ):
            if artifact.get(field_name) != expected:
                raise ValueError(
                    f"Engram tokenizer-map {field_name} mismatch: expected {expected}, "
                    f"got {artifact.get(field_name)!r}."
                )

        artifact_layer_ids = tuple(artifact.get("layer_ids", ()))
        if artifact_layer_ids != self.layer_ids:
            raise ValueError(
                "Engram tokenizer-map layer_ids mismatch: "
                f"expected {self.layer_ids}, got {artifact_layer_ids}."
            )

        remap = artifact.get("remap")
        source_vocab_size = artifact.get("source_vocab_size")
        if not isinstance(remap, list) or len(remap) != source_vocab_size:
            raise ValueError(
                "Engram tokenizer-map remap length must equal source_vocab_size; "
                f"got {len(remap) if isinstance(remap, list) else type(remap).__name__} "
                f"and {source_vocab_size!r}."
            )
        if any(not isinstance(token_id, int) or token_id < 0 for token_id in remap):
            raise ValueError("Engram tokenizer-map remap values must be nonnegative integers.")

        compressed_vocab_size = artifact.get("compressed_vocab_size")
        if not isinstance(compressed_vocab_size, int) or compressed_vocab_size <= 0:
            raise ValueError("Engram tokenizer-map compressed_vocab_size must be positive.")
        if max(remap, default=-1) >= compressed_vocab_size:
            raise ValueError("Engram tokenizer-map remap contains an out-of-range compressed ID.")

        hash_boundary_token_id = artifact.get("compressed_pad_token_id")
        if hash_boundary_token_id != remap[self.boundary_token_id]:
            raise ValueError(
                "Engram tokenizer-map compressed_pad_token_id does not match remap[pad_token_id]."
            )

        raw_multipliers = artifact.get("layer_multipliers", {})
        layer_multipliers: dict[int, tuple[int, ...]] = {}
        for layer_id in self.layer_ids:
            values = raw_multipliers.get(str(layer_id))
            if not isinstance(values, list) or len(values) != self.max_ngram_order:
                raise ValueError(
                    "Engram tokenizer-map layer_multipliers must contain "
                    f"{self.max_ngram_order} values for layer {layer_id}."
                )
            multipliers = tuple(int(value) for value in values)
            if any(value <= 0 or value % 2 == 0 for value in multipliers):
                raise ValueError("Engram hash multipliers must be positive odd integers.")
            if max(multipliers) > _INT64_MAX // max(compressed_vocab_size - 1, 1):
                # build_ngram_hashes multiplies compressed IDs by these in int64; wrapping
                # would silently change the hash instead of failing.
                raise ValueError(
                    f"Engram hash multipliers for layer {layer_id} overflow int64 for a "
                    f"compressed vocabulary of {compressed_vocab_size}."
                )
            layer_multipliers[layer_id] = multipliers

        self.tokenizer_remap = torch.tensor(remap, dtype=torch.int64)
        self.hash_boundary_token_id = hash_boundary_token_id
        self.layer_multipliers = layer_multipliers

    def validate_startup(
        self,
        transformer_config: Any,
        expected_tokenizer_vocab_size: int | None = None,
        packed_sequences: bool = False,
        use_fsdp: bool = False,
        sequence_length: int | None = None,
    ) -> None:
        """Validate layer placement, parallelism, and feature support.

        The padded vocabulary is only known once the tokenizer has been built, which happens
        after the model config is assembled; pass it here when available, otherwise call
        :meth:`validate_vocabulary` from the model builder.
        """
        self._validate_layer_placement(transformer_config)
        if expected_tokenizer_vocab_size is not None:
            self.validate_vocabulary(expected_tokenizer_vocab_size)
        self._validate_parallelism(transformer_config, sequence_length)
        self._validate_packed_sequences(transformer_config, packed_sequences)
        self._validate_unsupported_features(transformer_config, use_fsdp)

    def _validate_layer_placement(self, transformer_config: Any) -> None:
        if any(layer_id > transformer_config.num_layers for layer_id in self.layer_ids):
            raise ValueError(
                "engram_layer_ids must fall within the model's 1-based layer range "
                f"[1, {transformer_config.num_layers}]; got {self.layer_ids}."
            )

    def validate_vocabulary(self, expected_tokenizer_vocab_size: int) -> None:
        """Check the padded vocabulary against the hash-constant bounds."""
        if self.variant_spec.uses_tokenizer_artifact:
            if self.tokenizer_remap.numel() != expected_tokenizer_vocab_size:
                raise ValueError(
                    "Engram tokenizer-map vocabulary mismatch: artifact has "
                    f"{self.tokenizer_remap.numel()} raw tokens but the model expects "
                    f"{expected_tokenizer_vocab_size}."
                )
        elif expected_tokenizer_vocab_size > self.unigram_vocab_size:
            # The multiplier bound guarantees token_id * multiplier <= int64_max only for token
            # IDs below unigram_vocab_size; padded vocabulary IDs would overflow the hash.
            raise ValueError(
                f"The {self.variant} variant requires the padded vocabulary "
                f"({expected_tokenizer_vocab_size}) to be at most engram_unigram_vocab_size "
                f"({self.unigram_vocab_size}); adjust make_vocab_size_divisible_by."
            )

    @property
    def conv_history_length(self) -> int:
        """Positions of causal left context the short convolution needs."""
        return (self.kernel_size - 1) * self.max_ngram_order

    def _validate_parallelism(self, transformer_config: Any, sequence_length: int | None) -> None:
        if transformer_config.context_parallel_size != 1:
            raise ValueError("Engram currently requires context_parallel_size == 1.")
        if (
            transformer_config.expert_tensor_parallel_size
            != transformer_config.tensor_model_parallel_size
        ):
            # Engram assumes the EP group is orthogonal to the dense TP group: tables are
            # replicated over dense TP and sharded over EP. With etp != tp, dense-TP peers sit
            # inside one EP group and own different row shards, which silently over-counts
            # table gradients (non-SP) or sums gradients of unrelated rows (SP).
            raise ValueError(
                "Engram requires expert_tensor_parallel_size == tensor_model_parallel_size; got "
                f"etp={transformer_config.expert_tensor_parallel_size} and "
                f"tp={transformer_config.tensor_model_parallel_size}."
            )
        if transformer_config.virtual_pipeline_model_parallel_size is not None:
            raise ValueError("Engram does not yet support virtual pipeline parallelism.")
        tensor_parallel_size = transformer_config.tensor_model_parallel_size
        if transformer_config.sequence_parallel and sequence_length is not None:
            # Each rank fetches its convolution history from the previous rank in one hop, so
            # a rank's own slice must be at least as long as that history.
            local_sequence_length = sequence_length // tensor_parallel_size
            if local_sequence_length < self.conv_history_length:
                raise ValueError(
                    "Engram with sequence parallelism requires each tensor-parallel rank to own "
                    f"at least {self.conv_history_length} positions, but seq_length "
                    f"{sequence_length} over {tensor_parallel_size} ranks gives "
                    f"{local_sequence_length}."
                )

    def _validate_packed_sequences(self, transformer_config: Any, packed_sequences: bool) -> None:
        if not packed_sequences:
            return
        if not self.variant_spec.supports_packed_sequences:
            raise ValueError(
                f"Packed sequences are not supported by the {self.variant} variant, whose "
                "n-gram windows do not reset at document boundaries."
            )
        if transformer_config.pipeline_model_parallel_size > 2:
            # TODO(upstream): middle pipeline stages receive no cu_seqlens/max_seqlen from the
            # TP batch broadcast, so packed training crashes there with or without Engram.
            # Reject at startup until the upstream THD pipeline gap is fixed.
            raise ValueError(
                "Packed sequences with pipeline_model_parallel_size > 2 are blocked by the "
                "upstream THD pipeline (middle stages receive no cu_seqlens/max_seqlen)."
            )
        if transformer_config.pipeline_model_parallel_size > 1:
            # Pipeline activations use the fixed sequence-capacity shape, so packed rows must
            # be padded to the capacity on every stage; the prefetched tokens are zero-padded
            # to the same capacity to stay aligned with the hidden states.
            if getattr(transformer_config, "pad_packed_seq_alignment", None) != "max":
                raise ValueError(
                    "Engram with packed sequences and pipeline parallelism requires "
                    "--pad-packed-seq-alignment max so every stage sees fixed-capacity rows."
                )

    def _validate_unsupported_features(self, transformer_config: Any, use_fsdp: bool) -> None:
        if not transformer_config.bf16:
            raise ValueError("Engram currently requires BF16 parameters.")
        if transformer_config.fp4 is not None:
            raise ValueError("Engram does not yet support FP4 training.")
        # FP8/MXFP8 recipes are permitted: Engram modules are plain torch modules outside the
        # Transformer Engine autocast regions, so tables and projections stay in BF16.
        # Multi-token prediction is permitted: MTP layers use their own local layer numbering
        # and never build Engram modules (TransformerLayer skips them via is_mtp_layer).
        if transformer_config.transformer_impl == "inference_optimized":
            raise ValueError("Engram does not yet support inference-optimized model execution.")
        if getattr(transformer_config, "overlap_moe_expert_parallel_comm", False):
            # The fine-grained attention callable does not forward input_ids, so the Engram
            # layers would fail on the first microbatch.
            raise ValueError("Engram does not yet support overlap_moe_expert_parallel_comm.")
        if transformer_config.recompute_granularity is not None:
            raise ValueError("Engram does not yet support activation recomputation.")
        if transformer_config.cuda_graph_impl != "none":
            raise ValueError("Engram does not yet support CUDA graphs.")
        if use_fsdp:
            raise ValueError("Engram does not yet support FSDP.")
