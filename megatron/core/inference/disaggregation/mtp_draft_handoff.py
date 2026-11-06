# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Codec for the MTP draft distributions carried across a prefill->decode handoff.

Under `mtp_verification='rejection'` the decode engine verifies an imported request's first
drafts (the `resume_tokens` proposals sampled by the prefill engine's draft loop) by chain
speculative sampling, which needs the filtered draft distributions q they were sampled from. q
travels with the handoff metadata rather than through the KV/SSM transfer: after top-k / top-p
filtering it is mostly zeros, so a sparse (index, value) encoding is typically a few KB per
request, and a dense encoding is bounded by `num_depths * vocab * 4` bytes. Values are fp32 and
round-trip bit-exactly. Arrays are base64 strings so the payload survives msgpack and JSON.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

MTP_DRAFT_Q_KEY = "mtp_draft_q"
MTP_VERIFICATION_KEY = "mtp_verification"
_VERSION = 1


def _b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def _unb64(text: Any, dtype, count: int, name: str) -> np.ndarray:
    if not isinstance(text, str):
        raise ValueError(f"MTP draft q payload field {name!r} must be a base64 string")
    raw = base64.b64decode(text.encode("ascii"), validate=True)
    array = np.frombuffer(raw, dtype=dtype)
    if array.size != count:
        raise ValueError(
            f"MTP draft q payload field {name!r} has {array.size} entries, expected {count}"
        )
    return array


def encode_mtp_draft_q(q: torch.Tensor, drafts: Sequence[int]) -> Dict[str, Any]:
    """Encode `[num_depths, vocab]` fp32 draft distributions and the drafts sampled from them.

    Uses a sparse (int32 index, fp32 value) encoding when it is smaller than the dense fp32 one.
    """
    if q.dim() != 2:
        raise ValueError(f"expected [num_depths, vocab] draft distributions, got {tuple(q.shape)}")
    q_np = q.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    num_depths, vocab = q_np.shape
    if len(drafts) != num_depths:
        raise ValueError(f"expected {num_depths} drafts, got {len(drafts)}")
    payload: Dict[str, Any] = {
        "version": _VERSION,
        "num_depths": int(num_depths),
        "vocab": int(vocab),
        "drafts": [int(token) for token in drafts],
    }
    nonzero = [np.flatnonzero(row) for row in q_np]
    total_nnz = sum(int(idx.size) for idx in nonzero)
    if total_nnz * 8 < num_depths * vocab * 4:
        payload["encoding"] = "sparse"
        payload["nnz"] = [int(idx.size) for idx in nonzero]
        payload["indices"] = _b64(np.concatenate(nonzero).astype("<i4"))
        payload["values"] = _b64(
            np.concatenate([row[idx] for row, idx in zip(q_np, nonzero)]).astype("<f4")
        )
    else:
        payload["encoding"] = "dense"
        payload["values"] = _b64(q_np.astype("<f4"))
    return payload


def decode_mtp_draft_q(
    payload: Any, *, num_depths: int, vocab: int, drafts: Sequence[int]
) -> torch.Tensor:
    """Decode and validate a payload against the local MTP configuration.

    Args:
        payload: The `encode_mtp_draft_q` dictionary from the prefill engine.
        num_depths: The local `num_mtp_depths`; must match the prefill engine's.
        vocab: The local (padded) vocabulary size; must match.
        drafts: The handoff's draft tokens (`resume_tokens[1:1 + num_depths]`); the payload
            must describe exactly these drafts.

    Returns:
        torch.Tensor: `[num_depths, vocab]` fp32 CPU tensor (pinned when CUDA is available).
    """
    if not isinstance(payload, dict) or payload.get("version") != _VERSION:
        raise ValueError("MTP draft q payload is missing or has an unsupported version")
    if payload.get("num_depths") != num_depths or payload.get("vocab") != vocab:
        raise ValueError(
            "MTP draft q payload does not match the local MTP configuration: "
            f"payload (depths={payload.get('num_depths')}, vocab={payload.get('vocab')}), "
            f"local (depths={num_depths}, vocab={vocab}); prefill and decode must serve the "
            "same model with the same number of MTP depths"
        )
    payload_drafts: List[int] = [int(token) for token in payload.get("drafts", [])]
    if payload_drafts != [int(token) for token in drafts]:
        raise ValueError("MTP draft q payload describes different drafts than the handoff")

    encoding = payload.get("encoding")
    if encoding == "dense":
        q_np = _unb64(payload.get("values"), "<f4", num_depths * vocab, "values").reshape(
            num_depths, vocab
        )
    elif encoding == "sparse":
        nnz = payload.get("nnz")
        if not isinstance(nnz, list) or len(nnz) != num_depths or any(n < 0 for n in nnz):
            raise ValueError("MTP draft q payload has an invalid sparse layout")
        total = int(sum(nnz))
        indices = _unb64(payload.get("indices"), "<i4", total, "indices")
        values = _unb64(payload.get("values"), "<f4", total, "values")
        if total and (indices.min() < 0 or indices.max() >= vocab):
            raise ValueError("MTP draft q payload has out-of-range token indices")
        q_np = np.zeros((num_depths, vocab), dtype=np.float32)
        rows = np.repeat(np.arange(num_depths), nnz)
        q_np[rows, indices] = values
    else:
        raise ValueError(f"MTP draft q payload has unknown encoding {encoding!r}")

    if not np.isfinite(q_np).all() or (q_np < 0).any():
        raise ValueError("MTP draft q payload contains negative or non-finite probabilities")
    if (q_np[np.arange(num_depths), payload_drafts] <= 0).any():
        raise ValueError("MTP draft q payload gives a draft token zero probability")
    q = torch.from_numpy(q_np.copy())
    return q.pin_memory() if torch.cuda.is_available() else q


def mtp_draft_q_payload_nbytes(payload: Optional[Dict[str, Any]]) -> int:
    """Approximate wire size of a payload (base64 text), for logging."""
    if not payload:
        return 0
    return sum(len(payload.get(key, "")) for key in ("indices", "values"))
