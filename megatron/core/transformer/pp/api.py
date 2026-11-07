# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Host API of the PP attention forward kernel (BF16, causal, head dim 128 or 256)."""

import math
from typing import Optional

import torch

try:
    import cutlass.cute as cute
    from cutlass.cute.typing import Float32
    from flash_attn.cute.cute_dsl_utils import to_cute_tensor

    from .prefill_kernel import PPAttentionForward

    HAVE_PP_ATTENTION = True
except ImportError:
    HAVE_PP_ATTENTION = False

TILE_KEYS = 128
MAX_ROWS = 256
HEAD_DIMS = (128, 256)


class PPAttention:
    """Causal attention whose output bytes do not depend on how the keys are split.

    Keys are visited in 128-key tiles grouped into super-blocks of
    ``super_block_tiles`` tiles at absolute key positions. Each super-block
    yields one FP32 partial (max, sum, output) and the partials are folded left
    to right. Calls on one instance share a per-SM scratch buffer and must be
    serialized on one CUDA stream.
    """

    def __init__(
        self,
        device,
        *,
        head_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        super_block_tiles: int = 8,
        softmax_scale: Optional[float] = None,
    ):
        if not HAVE_PP_ATTENTION:
            raise ImportError("PP attention needs nvidia-cutlass-dsl and flash-attn-4")
        if head_dim not in HEAD_DIMS:
            raise ValueError(f"head_dim must be one of {HEAD_DIMS}, got {head_dim}")
        if super_block_tiles < 1:
            raise ValueError("super_block_tiles must be >= 1")
        if num_q_heads % num_kv_heads != 0:
            raise ValueError("num_q_heads must be a multiple of num_kv_heads")
        self.device = torch.device(device)
        self.head_dim = int(head_dim)
        self.G = int(super_block_tiles)
        self.num_q_heads = int(num_q_heads)
        self.h_kv = int(num_kv_heads)
        self.rows = self.num_q_heads // self.h_kv
        if self.rows > MAX_ROWS:
            raise ValueError(f"query heads per kv head {self.rows} > {MAX_ROWS}")
        self.softmax_scale = (
            float(softmax_scale) if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
        )
        sm_count = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.scratch = torch.empty(
            sm_count, 2, head_dim // 4, TILE_KEYS, 4, dtype=torch.float32, device=self.device
        )
        self._compiled = {}
        self._fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    def _check(self, t, name, ndim):
        if t.dtype != torch.bfloat16 or t.shape[-1] != self.head_dim or t.ndim != ndim:
            raise ValueError(f"{name} must be rank-{ndim} bfloat16 with head_dim {self.head_dim}")
        if not t.is_cuda or not t.is_contiguous():
            raise ValueError(f"{name} must be a contiguous CUDA tensor")

    def _launch(self, key, q, k, v, out, cu, seqused, page_table, lse):
        if key not in self._compiled:
            fwd = PPAttentionForward(
                self.head_dim,
                qhead_per_kvhead=self.rows,
                is_causal=True,
                super_block_tiles=self.G,
                num_regs_other=80,
            )
            opt = lambda t, **kw: to_cute_tensor(t, **kw) if t is not None else None
            self._compiled[key] = cute.compile(
                fwd,
                to_cute_tensor(q),
                to_cute_tensor(k),
                to_cute_tensor(v),
                to_cute_tensor(out),
                Float32(self.softmax_scale),
                opt(cu, assumed_align=4, leading_dim=0),
                opt(seqused, assumed_align=4, leading_dim=0),
                opt(page_table, assumed_align=4, leading_dim=1),
                to_cute_tensor(self.scratch),
                opt(lse, assumed_align=4),
                self._fake_stream,
                options="--enable-tvm-ffi",
            )
        self._compiled[key](
            q, k, v, out, Float32(self.softmax_scale), cu, seqused, page_table, self.scratch, lse
        )

    def prefill(self, q, k, v, return_lse: bool = False):
        """Dense causal attention. q [B, L, Hq, D], k/v [B, L, Hkv, D]; lse is FP32 [B, Hq, L]."""
        for t, n in ((q, "q"), (k, "k"), (v, "v")):
            self._check(t, n, 4)
        if q.shape[:2] != k.shape[:2] or k.shape != v.shape:
            raise ValueError("q/k/v batch and sequence dims must match")
        if q.shape[2] != self.num_q_heads or k.shape[2] != self.h_kv:
            raise ValueError("q/k head counts do not match this instance")
        out = torch.empty_like(q)
        lse = (
            torch.empty(q.shape[0], q.shape[2], q.shape[1], dtype=torch.float32, device=q.device)
            if return_lse
            else None
        )
        self._launch(("prefill", return_lse), q, k, v, out, None, None, None, lse)
        return (out, lse) if return_lse else out

    def prefill_paged(
        self, q, k_cache, v_cache, cu_seqlens_q, seqused_k, page_table, return_lse: bool = False
    ):
        """Varlen causal attention over 128-token KV pages.

        q [total_q, Hq, D]; k/v_cache [pages, 128, Hkv, D]; cu_seqlens_q int32 [B+1];
        seqused_k int32 [B] (keys per sequence, query rows aligned to the end);
        page_table int32 [B, max_pages]. lse is FP32 [Hq, total_q].
        """
        self._check(q, "q", 3)
        self._check(k_cache, "k_cache", 4)
        self._check(v_cache, "v_cache", 4)
        if k_cache.shape[1] != TILE_KEYS or k_cache.shape != v_cache.shape:
            raise ValueError("paged K/V must be [pages, 128, Hkv, D]")
        if q.shape[1] != self.num_q_heads or k_cache.shape[2] != self.h_kv:
            raise ValueError("q/k head counts do not match this instance")
        for t, n in (
            (cu_seqlens_q, "cu_seqlens_q"),
            (seqused_k, "seqused_k"),
            (page_table, "page_table"),
        ):
            if t.dtype != torch.int32 or not t.is_contiguous():
                raise ValueError(f"{n} must be contiguous int32")
        batch = cu_seqlens_q.shape[0] - 1
        if seqused_k.shape[0] < batch or page_table.shape[0] < batch:
            raise ValueError("seqused_k/page_table batch dim too small")
        out = torch.empty_like(q)
        lse = (
            torch.empty(q.shape[1], q.shape[0], dtype=torch.float32, device=q.device)
            if return_lse
            else None
        )
        self._launch(
            ("paged", return_lse),
            q,
            k_cache,
            v_cache,
            out,
            cu_seqlens_q,
            seqused_k,
            page_table,
            lse,
        )
        return (out, lse) if return_lse else out


_INSTANCES = {}


def get_pp_attention(device, head_dim, num_q_heads, num_kv_heads, super_block_tiles, softmax_scale):
    """Shared instance per device and shape, so compiled kernels and scratch are reused."""
    key = (
        str(torch.device(device)),
        head_dim,
        num_q_heads,
        num_kv_heads,
        super_block_tiles,
        softmax_scale,
    )
    if key not in _INSTANCES:
        _INSTANCES[key] = PPAttention(
            device,
            head_dim=head_dim,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            super_block_tiles=super_block_tiles,
            softmax_scale=softmax_scale,
        )
    return _INSTANCES[key]
