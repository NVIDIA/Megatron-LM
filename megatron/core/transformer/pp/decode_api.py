# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Split decode and speculative verify with the bytes of the PP attention forward."""

from typing import Optional

import torch

from .api import TILE_KEYS, PPAttention

try:
    import cutlass.cute as cute
    from cutlass.cute.typing import Float32, Int32
    from flash_attn.cute.cute_dsl_utils import to_cute_tensor

    from .combine_kernel import PPAttentionCombine
    from .decode_kernel import PPAttentionDecode
except ImportError:
    pass

MAX_ITEM_ROWS = 64


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


class PPAttentionDecoder:
    """Decode and verify over a paged KV cache of 128-key pages.

    Each work item covers one super-block of one (request, kv head) and writes an FP32
    partial; the combine folds the partials left to right, so every output row has the
    bytes of the prefill row at the same position. All buffers are allocated here, so
    calls are CUDA-graph capturable after one warm-up call per shape.
    """

    def __init__(
        self, attn: PPAttention, *, max_batch: int, max_context: int, verify_tokens: int = 8
    ):
        self.attn = attn
        self.max_batch = int(max_batch)
        self.verify_tokens = int(verify_tokens)
        rows, h_kv, d = attn.rows, attn.h_kv, attn.head_dim
        self.s_max = _ceil_div(_ceil_div(int(max_context), TILE_KEYS), attn.G)
        self.tok_max = max(1, MAX_ITEM_ROWS // rows)
        self.rows_max = rows * min(self.verify_tokens, self.tok_max)
        self.flat_max = self.max_batch * h_kv * self.verify_tokens
        multi_max = self.max_batch * h_kv
        slots = self.s_max * max(multi_max * self.rows_max, self.flat_max * rows)
        dev = attn.device
        self._o_store = torch.empty(slots, d, dtype=torch.float32, device=dev)
        self._ml_store = torch.empty(slots, 2, dtype=torch.float32, device=dev)
        self.sm_count = torch.cuda.get_device_properties(dev).multi_processor_count
        self._compiled = {}

    def _split(self, q_view, k_cache, v_cache, seqused, page_table, out_view, flat, tokens):
        attn = self.attn
        rows, d = q_view.shape[1], attn.head_dim
        n = flat * self.s_max * rows
        if n > self._o_store.shape[0]:
            raise ValueError(f"{flat} items of {rows} rows exceed the decode workspace")
        splits = min(self.s_max, _ceil_div(page_table.shape[1], attn.G))
        num_ctas = min(flat * splits, self.sm_count)
        o_view = self._o_store[:n].view(flat, self.s_max, rows, d)[:, :splits]
        ml_view = self._ml_store[:n].view(flat, self.s_max, rows, 2)[:, :splits]
        if out_view.ndim == 3:
            out_view = out_view.view(out_view.shape[0], 1, 1, *out_view.shape[1:])
        scale = Float32(attn.softmax_scale)
        key = ("split", rows, tokens)
        if key not in self._compiled:
            fwd = PPAttentionDecode(
                super_block_tiles=attn.G,
                tokens_per_item=tokens,
                rows_per_item=rows,
                head_dim=attn.head_dim,
            )
            self._compiled[key] = cute.compile(
                fwd,
                to_cute_tensor(q_view),
                to_cute_tensor(k_cache),
                to_cute_tensor(v_cache),
                scale,
                to_cute_tensor(seqused, assumed_align=4, leading_dim=0),
                to_cute_tensor(page_table, assumed_align=4, leading_dim=1),
                to_cute_tensor(o_view),
                to_cute_tensor(ml_view),
                Int32(splits),
                Int32(attn.h_kv),
                Int32(num_ctas),
                attn._fake_stream,
                options="--enable-tvm-ffi",
            )
        self._compiled[key](
            q_view,
            k_cache,
            v_cache,
            scale,
            seqused,
            page_table,
            o_view,
            ml_view,
            Int32(splits),
            Int32(attn.h_kv),
            Int32(num_ctas),
        )
        key_c = ("combine", rows)
        if key_c not in self._compiled:
            self._compiled[key_c] = cute.compile(
                PPAttentionCombine(
                    attn.G, attn.head_dim, smax=max(32, 32 * _ceil_div(self.s_max, 32))
                ),
                to_cute_tensor(o_view),
                to_cute_tensor(ml_view),
                to_cute_tensor(seqused, assumed_align=4, leading_dim=0),
                to_cute_tensor(out_view),
                Int32(attn.h_kv),
                scale,
                attn._fake_stream,
                options="--enable-tvm-ffi",
            )
        self._compiled[key_c](o_view, ml_view, seqused, out_view, Int32(attn.h_kv), scale)

    def _check(self, q, k_cache, v_cache, page_table, seqused_k):
        a = self.attn
        if q.ndim != 4 or q.dtype != torch.bfloat16 or not q.is_contiguous():
            raise ValueError("q must be a contiguous bfloat16 [B, N, H_q, D] tensor")
        if q.shape[2] != a.num_q_heads or q.shape[3] != a.head_dim:
            raise ValueError(f"q must have {a.num_q_heads} heads of dim {a.head_dim}")
        if k_cache.ndim != 4 or k_cache.shape[1] != TILE_KEYS or k_cache.shape != v_cache.shape:
            raise ValueError(f"k/v cache must be [pages, {TILE_KEYS}, H_kv, D]")
        if q.shape[0] > self.max_batch:
            raise ValueError(f"batch {q.shape[0]} > max_batch {self.max_batch}")
        if page_table.dtype != torch.int32 or seqused_k.dtype != torch.int32:
            raise ValueError("page_table and seqused_k must be int32")

    def verify(
        self, q, k_cache, v_cache, page_table, seqused_k, out: Optional[torch.Tensor] = None
    ):
        """N tokens per request. q [B, N, H_q, D]; row j sees keys [0, seqused_k - N + 1 + j)."""
        self._check(q, k_cache, v_cache, page_table, seqused_k)
        a = self.attn
        batch, n, d = q.shape[0], q.shape[1], a.head_dim
        if not 1 <= n <= self.verify_tokens:
            raise ValueError(f"tokens per request must be in [1, {self.verify_tokens}], got {n}")
        out = torch.empty_like(q) if out is None else out
        if n == 1 or n * a.rows <= self.rows_max:
            q_in = q.view(batch, n, a.h_kv, a.rows, d).permute(0, 2, 1, 3, 4).contiguous()
            self._split(
                q_in.view(batch * a.h_kv, n * a.rows, 1, d),
                k_cache,
                v_cache,
                seqused_k[:batch],
                page_table[:batch],
                out.view(batch, n, a.h_kv, a.rows, d),
                batch * a.h_kv,
                n,
            )
            return out
        j = torch.arange(n, device=q.device, dtype=torch.int32).repeat(batch)
        seq_v = (seqused_k[:batch].repeat_interleave(n) - (n - 1) + j).contiguous()
        pt_v = page_table[:batch].repeat_interleave(n, dim=0)
        flat = batch * n * a.h_kv
        self._split(
            q.view(flat, a.rows, 1, d),
            k_cache,
            v_cache,
            seq_v,
            pt_v,
            out.view(flat, a.rows, d),
            flat,
            1,
        )
        return out


_DECODERS = {}


def get_pp_decoder(attn: PPAttention, max_batch: int, max_context: int, verify_tokens: int):
    """Shared decoder per attention instance and workspace bound."""
    key = (id(attn), max_batch, max_context, verify_tokens)
    if key not in _DECODERS:
        _DECODERS[key] = PPAttentionDecoder(
            attn, max_batch=max_batch, max_context=max_context, verify_tokens=verify_tokens
        )
    return _DECODERS[key]


def page_view(k_cache, v_cache, block_table):
    """View [blocks, block_size, H, D] caches as 128-key pages and map the block table."""
    block = k_cache.shape[1]
    if block % TILE_KEYS:
        raise ValueError(f"PP attention needs a KV block size divisible by {TILE_KEYS}")
    per = block // TILE_KEYS
    k_pages = k_cache.view(-1, TILE_KEYS, *k_cache.shape[2:])
    v_pages = v_cache.view(-1, TILE_KEYS, *v_cache.shape[2:])
    sub = torch.arange(per, device=block_table.device, dtype=block_table.dtype)
    pages = block_table.unsqueeze(-1) * per + sub
    pages = torch.where(block_table.unsqueeze(-1) < 0, -1, pages)
    return k_pages, v_pages, pages.view(block_table.shape[0], -1).to(torch.int32)
