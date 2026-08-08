# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch


def build_causal_mask(seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Additive full-causal mask [1, 1, S, S], fill = ``torch.finfo(dtype).min``.

    Mirrors HF ``create_causal_mask`` (masking_utils.py:631-633): a float bias added
    to the attention logits, where masked positions get ``finfo(dtype).min`` (NOT
    ``-inf`` / ``-10000``) and allowed positions get ``0.0``. A query at position q
    attends keys 0..q inclusive (``kv_idx <= q_idx``).
    """
    min_value = torch.finfo(dtype).min
    q = torch.arange(seq_len, device=device)
    allowed = q[:, None] >= q[None, :]  # kv_idx <= q_idx
    mask = torch.where(allowed, 0.0, min_value).to(dtype)
    return mask.view(1, 1, seq_len, seq_len)


def build_sliding_window_causal_mask(
    seq_len: int, window: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Additive sliding-window causal mask [1, 1, S, S], fill = ``torch.finfo(dtype).min``.

    Mirrors HF ``create_sliding_window_causal_mask`` (masking_utils.py:95,130-134): a
    query at position q attends keys ``q - window < kv_idx <= q`` -> positions
    ``q-(window-1) .. q`` inclusive (``window`` keys incl. self). With window=512 the
    boundary is left-exclusive at ``q-512``.
    """
    min_value = torch.finfo(dtype).min
    q = torch.arange(seq_len, device=device)
    causal = q[:, None] >= q[None, :]  # kv_idx <= q_idx
    in_window = q[None, :] > (q[:, None] - window)  # kv_idx > q_idx - window
    allowed = causal & in_window
    mask = torch.where(allowed, 0.0, min_value).to(dtype)
    return mask.view(1, 1, seq_len, seq_len)
