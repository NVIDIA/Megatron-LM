# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright 2026 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Parts of this file are adapted from:
# - DeepSeek Engram (engram_demo_v1.py), https://github.com/deepseek-ai/Engram at revision
#   fb7f84a21f91223715394a33a1dc24bbfb7f788e;
# - Hugging Face Transformers (src/transformers/models/qwen4_exp/modeling_qwen4_exp.py),
#   https://github.com/huggingface/transformers at revision
#   fd3cf0140de03f7619ac95b5cfed05b711ff9e22.
# Both are licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0).

"""Configuration and deterministic table allocation for Engram."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from megatron.core.transformer.cuda_graph_config import normalize_cuda_graph_modules
from megatron.core.transformer.enums import CudaGraphModule

from .variants import DEEPSEEK_VARIANT_NAME, EngramVariant, resolve_variant

TOKENIZER_MAP_FORMAT = "megatron-engram-token-map"
TOKENIZER_MAP_VERSION = 1

# Transformer Engine capture scopes that leave every Engram layer eager (see
# EngramConfig._validate_unsupported_features).
_MOE_CAPTURE_MODULES = frozenset(
    (CudaGraphModule.moe, CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess)
)


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


_DYNAMIC_CONTEXT_PARALLEL_UNSUPPORTED = (
    "Engram does not support dynamic context parallelism: it gathers and re-selects token IDs "
    "over the static context-parallel group, while dynamic CP picks the group per microbatch."
)


def _validate_packed_sequence_args(args: Any) -> bool:
    """Return whether rows are packed sequences, rejecting packing setups Engram cannot follow.

    Packed sequences are the rows that the --sft and varlen datasets pack and pad to a fixed
    capacity, as the dataset builder's ``is_packed_sequence`` defines them. Rows read with
    --dataloader-inter-document-masking also carry document boundaries (cu_seqlens), but keep
    seq_length tokens without capacity padding: they are checked here and are not packed
    sequences.
    """
    sft = bool(getattr(args, "sft", False))
    # SBHD validation reads the varlen dataset but keeps the dense layout, so it is not packed.
    # Without it, --use-varlen-dataset always selects a sequence-packing scheduler, which is
    # rejected below, so --sft is the packed path that reaches training.
    packed_sequences = sft or bool(
        getattr(args, "use_varlen_dataset", False)
        and not getattr(args, "varlen_sbhd_validation", False)
    )
    if getattr(args, "dynamic_context_parallel", False):
        raise ValueError(_DYNAMIC_CONTEXT_PARALLEL_UNSUPPORTED)
    if getattr(args, "sequence_packing_scheduler", None) is not None:
        # Scheduler microbatches are variable-length and bypass the token prefetch entirely.
        raise ValueError(
            "Engram does not support sequence-packing schedulers (including the one "
            "--use-varlen-dataset auto-selects); use the --sft packed path instead."
        )
    if (
        getattr(args, "dataloader_inter_document_masking", False)
        and not sft
        and getattr(args, "context_parallel_size", 1) > 1
    ):
        # pretrain_hybrid splits these rows over CP with a padded per-sequence zigzag
        # (use_per_sequence_balancing), while Engram restores and re-selects tokens with the
        # per-document partition whenever cu_seqlens exist, so the n-gram hashes would land on
        # the wrong positions.
        raise ValueError(
            "Engram does not support --dataloader-inter-document-masking with context "
            "parallelism unless --sft is set: those rows are split over CP with a padded "
            "per-sequence zigzag, while Engram partitions packed rows per document."
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
    """

    global_vocab_sizes: tuple[int, ...]
    """Table row budget for each n-gram order 2..max_ngram_order. Variants that allocate
    per-head primes from a single base require every entry to be equal."""

    layer_ids: tuple[int, ...]
    """1-based global layer numbers that carry an Engram memory."""

    max_ngram_order: int
    """Largest n-gram order hashed; every order from 2 up to it gets num_hash_heads tables."""

    num_hash_heads: int
    """Hash heads per n-gram order, each with its own prime-sized table."""

    memory_dim: int
    """Retrieved-memory width per n-gram order, split evenly over its hash heads."""

    kernel_size: int
    """Kernel size of the depthwise causal short convolution, dilated by max_ngram_order."""

    hash_seed: int
    """Seed of the hash multipliers: checked against the tokenizer-map artifact by
    artifact-based variants, used to generate them by the others."""

    boundary_token_id: int
    """Token that fills the n-gram window before the start of a document. Variants that reset
    windows at boundaries also restart the window at every occurrence of this token, so it is
    the EOS ID for those."""

    tokenizer_map_path: str = ""
    """Versioned offline tokenizer-map artifact, required by artifact-based variants."""

    variant: str = DEEPSEEK_VARIANT_NAME
    """Hashing and layout convention, one of the names in
    :data:`megatron.core.models.engram.variants.ENGRAM_VARIANTS`."""

    unigram_vocab_size: int | None = None
    """Vocabulary size that bounds the generated hash multipliers, required by variants that
    generate their constants in process."""

    embedding_lr_multiplier: float = 5.0
    """Multiplier on the base learning rate (and minimum learning rate) for the hash-embedding
    tables."""

    embedding_weight_decay: float = 0.0
    """Weight decay of the hash-embedding tables."""

    freeze_embedding: bool = False
    """Keep the hash-embedding tables fixed: they still serve lookups, but take no gradient,
    so they get no gradient buffer or optimizer state and the lookup's backward exchange is
    skipped. The dense half (projections, norms, short convolution) still trains."""

    variant_spec: EngramVariant = field(init=False, repr=False)
    """Conventions of the selected variant, resolved from variant."""

    tokenizer_remap: torch.Tensor | None = field(init=False, repr=False, default=None)
    """Raw-to-compressed token ID map from the artifact; None for variants that hash raw IDs."""

    hash_boundary_token_id: int = field(init=False)
    """boundary_token_id in the ID space the hashes use (compressed for artifact variants)."""

    layer_multipliers: dict[int, tuple[int, ...]] = field(init=False, repr=False)
    """Hash multipliers of every configured layer, one per position of the n-gram window."""

    table_sizes_by_layer: dict[int, tuple[int, ...]] = field(init=False, repr=False)
    """Prime table sizes of every configured layer, one per table."""

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
            freeze_embedding=getattr(args, "engram_freeze_embedding", False),
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
        # The hybrid model reads engram_enabled to route token IDs to the Engram layers. Set it
        # here too, so that a TransformerConfig built without the argument namespace (YAML
        # configs, callers supplying an explicit config) does not keep it False while the
        # Engram modules are built.
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

        Args:
            transformer_config: The model's TransformerConfig.
            expected_tokenizer_vocab_size: Padded vocabulary size, when already known.
            packed_sequences: Whether rows are packed sequences padded to a fixed capacity
                (--sft or the varlen dataset); rows that only carry document boundaries
                (--dataloader-inter-document-masking) are not.
            use_fsdp: Whether the model is wrapped with FSDP.
            sequence_length: Training sequence length, when known.
        """
        self._validate_layer_placement(transformer_config)
        if expected_tokenizer_vocab_size is not None:
            self.validate_vocabulary(expected_tokenizer_vocab_size)
        self._validate_parallelism(transformer_config, sequence_length, packed_sequences)
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

    def _validate_parallelism(
        self, transformer_config: Any, sequence_length: int | None, packed_sequences: bool = False
    ) -> None:
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
        if getattr(transformer_config, "dynamic_context_parallel", False):
            raise ValueError(_DYNAMIC_CONTEXT_PARALLEL_UNSUPPORTED)
        if transformer_config.context_parallel_size > 1:
            # Engram restores the global token row from the zigzag CP partition and selects its
            # hashes back with the same partition, so every layer boundary it can sit on must
            # stay zigzag: attention layers run in attention_cp_layout, the other layer types in
            # linear_cp_layout, and packed rows are split by cp_partition_mode. An unset
            # linear_cp_layout is treated as zigzag, which is what TransformerConfig resolves
            # it to.
            linear_cp_layout = getattr(transformer_config, "linear_cp_layout", None)
            layouts = {
                "attention_cp_layout": getattr(transformer_config, "attention_cp_layout", "zigzag"),
                "linear_cp_layout": linear_cp_layout or "zigzag",
                "cp_partition_mode": getattr(transformer_config, "cp_partition_mode", "zigzag"),
            }
            non_zigzag = [
                f"{name}={layout!r}" for name, layout in layouts.items() if layout != "zigzag"
            ]
            if non_zigzag:
                raise ValueError(
                    "Engram with context parallelism requires the zigzag CP layout for "
                    "attention_cp_layout, linear_cp_layout and cp_partition_mode; got "
                    f"{', '.join(non_zigzag)}."
                )
            if not packed_sequences and sequence_length is not None:
                # An unpacked row gives each rank two zigzag chunks, and each chunk fetches its
                # convolution history from its predecessor in one hop. Packed rows are restored
                # whole before the convolution and have no such limit.
                context_parallel_size = transformer_config.context_parallel_size
                chunk_length = sequence_length // (2 * context_parallel_size)
                if chunk_length < self.conv_history_length:
                    raise ValueError(
                        "Engram with context parallelism requires each zigzag chunk to own at "
                        f"least {self.conv_history_length} positions, but seq_length "
                        f"{sequence_length} over {context_parallel_size} context-parallel ranks "
                        f"gives {chunk_length}."
                    )
        # Virtual pipeline parallelism needs no check: the tokens a layer hashes depend on the
        # microbatch, never on the model chunk.
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
        # Context parallelism with packed rows is supported: packed rows zigzag per document,
        # and the convolution restores that layout from cu_seqlens before convolving (verified
        # at module level to 5.6e-17 against the un-sharded result at CP2 and CP4). The CP data
        # path partitions unpadded packs by cu_seqlens when cu_seqlens_padded is None.
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
        if transformer_config.fp16:
            # The hashed tables and the PLE gate are plain torch modules in params_dtype; fp16's
            # range is too small for the sqrt-gated scores. BF16 is the production dtype and
            # FP32 is needed for numerical validation against reference implementations.
            raise ValueError("Engram does not support FP16 parameters; use BF16 or FP32.")
        if transformer_config.fp4 is not None:
            raise ValueError("Engram does not yet support FP4 training.")
        # FP8/MXFP8 recipes are permitted: Engram modules are plain torch modules outside the
        # Transformer Engine autocast regions, so tables and projections stay in BF16.
        # Multi-token prediction is permitted: layer_ids name main-decoder layers, and MTP
        # layers, which number their layers locally, never build an Engram module.
        if transformer_config.transformer_impl == "inference_optimized":
            raise ValueError("Engram does not yet support inference-optimized model execution.")
        if getattr(transformer_config, "overlap_moe_expert_parallel_comm", False):
            # The fine-grained attention callable does not forward input_ids, so the Engram
            # layers would fail on the first microbatch.
            raise ValueError("Engram does not yet support overlap_moe_expert_parallel_comm.")
        if getattr(transformer_config, "wide_residual", None) is not None:
            # Engram reads its stream count from the mHC configuration and expects hidden states
            # of num_streams * hidden_size in that layout; wide-residual layers carry their own
            # stream layout through the same layer entry point.
            raise ValueError(
                "Engram does not yet support wide_residual: the memory residual is defined over "
                "the standard or mHC residual streams only."
            )
        if getattr(transformer_config, "moe_shortcut_connection", False):
            # Shortcut-connected attention/MoE pairs run both halves outside
            # TransformerLayer._forward_attention, where the Engram residual is added, so the
            # memory would be silently skipped.
            raise ValueError(
                "Engram does not yet support moe_shortcut_connection: shortcut-connected layer "
                "pairs bypass the layer entry point that applies the memory."
            )
        if getattr(transformer_config, "enable_attention_residuals", False):
            # Attention-residual hybrid entries (configs that carry the field) run through
            # their own layer entry point, which bypasses TransformerLayer._forward_attention,
            # where the Engram residual is added, so the memory would be silently skipped.
            raise ValueError(
                "Engram does not yet support enable_attention_residuals: attention-residual "
                "hybrid entries bypass the layer entry point that applies the memory."
            )
        if getattr(transformer_config, "use_mup", False):
            # muP scales the learning rate of matrix parameters, which collides with the
            # learning-rate override of the 2-D Engram tables.
            raise ValueError(
                "Engram does not yet support use_mup: muP's learning-rate overrides conflict "
                "with the Engram embedding learning-rate override."
            )
        # Activation recomputation is supported: the recompute path passes input_ids to the
        # Engram layers it replays, and the Engram collectives stay rank-symmetric under
        # replay because every rank in the EP/TP group recomputes the same layer block.
        # CUDA graphs: the Engram lookup's hashed gather and all-to-all have data-dependent
        # shapes, so no capture scope may contain an Engram layer. Transformer Engine capture
        # restricted to the MoE scopes meets that: _layer_is_graphable then captures only
        # layers whose MLP is a MoE layer, and apply_engram_to_hybrid_stack_spec rejects
        # Engram on MoE layers under this capture mode. Every other scope, whole-layer capture
        # and the 'local' and 'full_iteration' implementations can capture an Engram layer.
        if transformer_config.cuda_graph_impl not in ("none", "transformer_engine"):
            raise ValueError(
                "Engram does not yet support cuda_graph_impl="
                f"{transformer_config.cuda_graph_impl!r}: whole-layer and full-iteration "
                "capture take the Engram lookup with them, and its hashed gather and "
                "all-to-all have data-dependent shapes."
            )
        if transformer_config.cuda_graph_impl == "transformer_engine":
            modules, _, _ = normalize_cuda_graph_modules(transformer_config.cuda_graph_modules)
            if not modules or not set(modules) <= _MOE_CAPTURE_MODULES:
                names = sorted(module.name for module in modules)
                raise ValueError(
                    "Engram supports cuda_graph_impl='transformer_engine' only with "
                    "cuda_graph_modules restricted to the MoE scopes "
                    "(moe / moe_router / moe_preprocess), which leave every Engram layer "
                    f"eager; got {names or 'whole-layer capture'}. "
                    "The Engram lookup's hashed gather and all-to-all have data-dependent "
                    "shapes and cannot be captured."
                )
        if use_fsdp:
            raise ValueError("Engram does not yet support FSDP.")
