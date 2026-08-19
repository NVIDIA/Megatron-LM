"""Autograd owners for the DS4 vLLM-visible training path."""

from .linear import (
    block_fp8_linear,
    fused_block_fp8_linear,
    gate_linear,
    visible_linear,
)
from .attention import attention_core, attach_indexer_aux_loss, visible_sparse_attention
from .mhc import mhc_head, mhc_post, mhc_pre_broadcast
from .norm import fused_qkv_rms_norm, rms_norm
from .o_proj import o_projection
from .router import fixed_route_vjp

__all__ = [
    "attention_core",
    "attach_indexer_aux_loss",
    "visible_sparse_attention",
    "block_fp8_linear",
    "fused_block_fp8_linear",
    "fused_qkv_rms_norm",
    "fixed_route_vjp",
    "gate_linear",
    "visible_linear",
    "mhc_head",
    "mhc_post",
    "mhc_pre_broadcast",
    "o_projection",
    "rms_norm",
]
