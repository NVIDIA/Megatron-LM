# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Direct public FA4 backend for MLA latent CP."""

from __future__ import annotations

import importlib

import torch
from torch import Tensor

from .layout import PhaseSpec
from .utils import BackendNotQualifiedError, LatentCPError, _require


class FA4Adapter:
    """Thin lazy adapter for public flash_attn.cute.flash_attn_varlen_func."""

    def __init__(self) -> None:
        try:
            module = importlib.import_module("flash_attn.cute")
            self._function = getattr(module, "flash_attn_varlen_func")
        except (ImportError, AttributeError) as error:
            raise BackendNotQualifiedError(
                "FA4 requires public flash_attn.cute.flash_attn_varlen_func"
            ) from error

    def prepare(
        self,
        *,
        num_heads: int,
        qk_dim: int,
        v_dim: int,
        phases: tuple[PhaseSpec, ...],
        scale: float,
    ) -> None:
        """Accept phase metadata; FA4 requires no persistent plan construction."""
        del num_heads, qk_dim, v_dim, phases, scale

    @staticmethod
    def _canonical_lse(lse: Tensor, tokens: int, heads: int) -> Tensor:
        if lse.shape == (tokens, heads):
            return lse.float()
        if lse.shape == (heads, tokens):
            return lse.transpose(0, 1).contiguous().float()
        raise LatentCPError(
            f"FA4 returned unsupported LSE shape {tuple(lse.shape)}; "
            f"expected {(tokens, heads)} or {(heads, tokens)}"
        )

    def forward_phase(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        cu_q: Tensor,
        cu_kv: Tensor,
        max_q: int,
        max_kv: int,
        causal: bool,
        scale: float,
    ) -> tuple[Tensor, Tensor]:
        """Execute one phase through the public FA4 varlen API."""
        _require(
            cu_q.dtype == torch.int32 and cu_kv.dtype == torch.int32,
            "FA4 cu_seqlens must have dtype torch.int32",
        )
        _require(
            cu_q.is_contiguous() and cu_kv.is_contiguous(),
            "FA4 cu_seqlens must be contiguous",
        )
        _require(
            cu_q.device == q.device and cu_kv.device == k.device,
            "FA4 cu_seqlens must be colocated with their Q/K tensors",
        )
        result = self._function(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_kv,
            max_seqlen_q=max_q,
            max_seqlen_k=max_kv,
            softmax_scale=scale,
            causal=causal,
            return_lse=True,
        )
        if not isinstance(result, tuple) or len(result) < 2:
            raise LatentCPError("FA4 return_lse=True did not return (output, LSE)")
        output, lse = result[:2]
        _require(output.dtype == torch.bfloat16, "FA4 phase output must be BF16")
        return output.float(), self._canonical_lse(lse, q.size(0), q.size(1))
