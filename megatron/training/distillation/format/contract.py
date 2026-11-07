# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Target encodings and file metadata shared by every offline KD producer and consumer.

Every file stores its inputs next to its targets, so a file is self-consistent on its
own. File decoding settings may differ between corpus sources; a blend shares token
and attention semantics. Within one corpus source, sample construction and ordering
must also remain fixed so resumed and parallel jobs cover one epoch exactly once.
"""

import hashlib
import json
from typing import Any

import numpy as np
import torch

FORMAT_VERSION = 3
MAX_VOCAB_SIZE = 1 << 24
LOGPROB_SENTINEL = -1000.0
# Allowed |retained + tail - 1| per row; generous enough for fp16-rounded values.
MASS_TOLERANCE = 0.05
ATTENTION_MODES = ("causal", "causal_eod_reset", "isolated_segments")

# Required by every reader: what token IDs mean, how target indices decode, which
# attention the inputs assume, and whether a prefix of a sample keeps its targets.
REQUIRED_KEYS = ("tokenizer", "logprob_vocab_size", "attention_mode", "prefix_stable")
# Corpus sources may use different normalization domains and index encodings.
BLEND_KEYS = ("tokenizer", "attention_mode", "prefix_stable")
# Replay dumps also need the writer layout that maps a global sample to its file and row.
REPLAY_KEYS = ("writers", "writer_mbs", "first_sample")
# A corpus source also needs its epoch size, which defines when it is complete.
CORPUS_KEYS = ("num_samples",)
# These recorded fields become invariants within a single corpus source, but may
# differ freely between sources in a blend.
CORPUS_ORDER_KEYS = ("data", "seed", "builder", "seq_length", "padding_divisor")
# Provenance fields; CORPUS_ORDER_KEYS are enforced within one corpus source.
RECORDED_KEYS = (
    "teacher",
    "seq_length",
    "padding_divisor",
    "data",
    "split",
    "seed",
    "num_samples",
    "builder",
)


def index_encoding_for_vocab(vocab_size: int) -> str:
    """Return the vocabulary-index encoding for a normalization domain size."""
    if type(vocab_size) is not int or not 1 <= vocab_size <= MAX_VOCAB_SIZE:
        raise ValueError("Offline KD vocabulary size must be an integer in [1, 2**24]")
    return "u16" if vocab_size <= 1 << 16 else "u16+u8"


def validate_metadata(
    meta: dict[str, Any], *, replay: bool = True, corpus: bool = False, blend: bool = False
) -> None:
    """Reject metadata a reader cannot interpret.

    ``meta`` holds ``format_version``, the required settings under ``contract``, and
    optional ``recorded`` and ``provenance`` dictionaries. A blend manifest carries
    shared semantics only (``blend=True, replay=False``), not file decoding settings.
    """
    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Expected offline KD format version {FORMAT_VERSION} metadata")
    contract = meta.get("contract")
    keys = (
        (BLEND_KEYS if blend else REQUIRED_KEYS)
        + (REPLAY_KEYS if replay else ())
        + (CORPUS_KEYS if corpus else ())
    )
    missing = [key for key in keys if not isinstance(contract, dict) or key not in contract]
    if missing:
        raise ValueError(f"Offline KD metadata is missing required settings: {', '.join(missing)}")
    for group in ("recorded", "provenance"):
        if not isinstance(meta.get(group, {}), dict):
            raise ValueError(f"Offline KD metadata {group} must be a dictionary")
    tokenizer = contract["tokenizer"]
    if not isinstance(tokenizer, dict) or not isinstance(tokenizer.get("sha256"), str):
        raise ValueError("Offline KD metadata tokenizer must carry a sha256 fingerprint")
    if not blend:
        index_encoding_for_vocab(contract["logprob_vocab_size"])
    if contract["attention_mode"] not in ATTENTION_MODES:
        raise ValueError(f"Unsupported offline KD attention_mode {contract['attention_mode']!r}")
    if (
        contract["attention_mode"] == "causal_eod_reset"
        and type(tokenizer.get("eod_id")) is not int
    ):
        raise ValueError("Offline KD causal_eod_reset requires an integer tokenizer eod_id")
    if type(contract["prefix_stable"]) is not bool:
        raise ValueError("Offline KD metadata prefix_stable must be a boolean")
    if replay:
        for key in ("writers", "writer_mbs"):
            if type(contract[key]) is not int or contract[key] < 1:
                raise ValueError(f"Offline KD metadata {key} must be a positive integer")
        if type(contract["first_sample"]) is not int or contract["first_sample"] < 0:
            raise ValueError("Offline KD metadata first_sample must be a nonnegative integer")
    if corpus and (type(contract["num_samples"]) is not int or contract["num_samples"] < 1):
        raise ValueError("Offline KD corpus metadata num_samples must be a positive integer")


def mismatched_settings(meta: dict[str, Any], expected: dict[str, Any]) -> tuple[list, list]:
    """Return ``(required, recorded)`` setting names whose values differ.

    Required differences make two files incompatible. Corpus callers additionally
    enforce :func:`mismatched_corpus_order` on the recorded sample construction fields.
    """
    required = sorted(
        key
        for key in set(meta.get("contract", {})) | set(expected.get("contract", {}))
        if meta.get("contract", {}).get(key) != expected.get("contract", {}).get(key)
    )
    recorded = sorted(
        key
        for key in set(meta.get("recorded", {})) | set(expected.get("recorded", {}))
        if meta.get("recorded", {}).get(key) != expected.get("recorded", {}).get(key)
    )
    return required, recorded


def blend_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Return the shared semantics of corpus sources, excluding per-file decoding.

    EOD identity matters only when it defines attention boundaries. It is deliberately
    outside the tokenizer fingerprint, and other producers may leave it unspecified.
    """
    shared = {key: contract[key] for key in BLEND_KEYS}
    shared["tokenizer"] = {
        key: value
        for key, value in contract["tokenizer"].items()
        if key != "eod_id" or contract["attention_mode"] == "causal_eod_reset"
    }
    return shared


def mismatched_corpus_order(meta: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    """Find changes that would splice different epochs into one corpus source."""
    return [
        key
        for key in CORPUS_ORDER_KEYS
        if meta.get("recorded", {}).get(key) != expected.get("recorded", {}).get(key)
    ]


def digest(data: bytes) -> str:
    """Return a stable SHA-256 fingerprint."""
    return hashlib.sha256(data).hexdigest()


def tokenizer_fingerprint(
    vocab: dict[str, int], special_ids: dict[str, int | None], size: int
) -> str:
    """Hash the ID-to-token mapping that cached inputs and targets depend on.

    The canonical object is ``{"vocab": [[token, id], ...] sorted by id,
    "special_ids": {bos, eos, pad, unk}, "size": n}`` serialized as compact,
    key-sorted UTF-8 JSON. Normalizers, pre-tokenizers, and chat templates are
    deliberately excluded: inputs are stored as token IDs.
    """
    canonical = {
        "vocab": sorted(
            ([str(token), int(index)] for token, index in vocab.items()), key=lambda item: item[1]
        ),
        "special_ids": {role: special_ids.get(role) for role in ("bos", "eos", "pad", "unk")},
        "size": int(size),
    }
    return digest(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )


def hf_tokenizer_identity(tokenizer: Any, *, eod_id: int | None = None) -> dict[str, Any]:
    """Describe a Hugging Face tokenizer for the ``tokenizer`` setting."""
    special_ids = {
        role: getattr(tokenizer, f"{role}_token_id", None) for role in ("bos", "eos", "pad", "unk")
    }
    size = len(tokenizer)
    return {
        "sha256": tokenizer_fingerprint(tokenizer.get_vocab(), special_ids, size),
        "size": size,
        "special_ids": special_ids,
        "eod_id": eod_id,
    }


def tail_logprob(values: torch.Tensor) -> torch.Tensor:
    """Log of the probability mass outside each row's retained entries.

    Writers call this on full-precision log probabilities before casting them to the
    save dtype, so rounding of the retained values does not distort the tail. Readers
    of records without a stored tail fall back to calling it on the stored values.
    Sentinel entries count as not retained; an empty tail is ``LOGPROB_SENTINEL``.
    """
    values = values.float()
    retained = torch.logsumexp(values.masked_fill(values == LOGPROB_SENTINEL, -torch.inf), -1)
    tail = torch.log(-torch.expm1(retained.clamp(max=0.0)))
    return tail.clamp(min=LOGPROB_SENTINEL)


def split_indices(
    indices: np.ndarray, values: np.ndarray, *, vocab_size: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """Encode ``[rows, K]`` vocabulary indices as uint16 low words plus, above 2**16, a
    uint8 high byte. Index -1 is allowed only on sentinel entries."""
    encoding = index_encoding_for_vocab(vocab_size)
    if indices.size and (
        (indices < -1).any()
        or (indices >= vocab_size).any()
        or ((indices == -1) & (values != LOGPROB_SENTINEL)).any()
    ):
        raise ValueError("Offline KD vocabulary index is out of range or has an unmasked sentinel")
    low = (indices & 0xFFFF).astype(np.uint16)
    high = ((indices >> 16) & 0xFF).astype(np.uint8) if encoding == "u16+u8" else None
    return low, high


def join_indices(low: np.ndarray, high: np.ndarray | None) -> np.ndarray:
    """Invert :func:`split_indices` (sentinel entries come back as arbitrary indices)."""
    indices = low.astype(np.int64)
    if high is not None:
        if high.shape != low.shape:
            raise ValueError("Malformed offline KD target indices")
        indices |= high.astype(np.int64) << 16
    return indices


def check_targets(
    values: torch.Tensor,
    indices: torch.Tensor,
    tail: torch.Tensor | None,
    *,
    vocab_size: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate decoded target rows and return them with the tail clamped to [-1000, 0].

    A stored tail plus the retained probabilities must sum to 1 within
    :data:`MASS_TOLERANCE`; without one, the tail is derived from the values.
    """
    if values.ndim != 2 or values.shape != indices.shape or not values.is_floating_point():
        raise ValueError("Malformed offline KD target values")
    if vocab_size is not None and ((indices >= vocab_size) & (values != LOGPROB_SENTINEL)).any():
        raise ValueError("Offline KD target vocabulary index is out of range")
    if tail is None:
        return values, indices, tail_logprob(values)
    if tail.dtype != torch.float32 or tail.shape != values.shape[:1] or tail.isnan().any():
        raise ValueError("Malformed offline KD tail log probabilities")
    tail = tail.clamp(min=LOGPROB_SENTINEL, max=0.0)
    retained = values.float().masked_fill(values == LOGPROB_SENTINEL, -torch.inf).exp().sum(-1)
    if values.numel() and ((retained + tail.exp() - 1).abs() > MASS_TOLERANCE).any():
        raise ValueError(
            "Offline KD retained and tail probabilities do not sum to 1; targets must be "
            "full-vocabulary log probabilities with the tail of the retained entries"
        )
    return values, indices, tail
