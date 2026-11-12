# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DS4.1 mega attention core: the vLLM inference forward and an owned backward.

Forward is the BI=1 prefill of vLLM's ``FLASHMLA_MEGA_ATTN_DSV41`` layer
(vllm-project/vllm#62365 @ d667e410e1, ``_forward_prefill_mega``), op for op:

1. SWA KV: ``fused_deepseek_v4_kv_rope_insert`` (GPT-J RoPE on dims [448, 512),
   bf16 rounding, MXFP8 record) into a temporary paged cache;
2. compressed KV: ``rope_quant_insert`` (RoPE at ``pos // cr * cr``, NVFP4 record)
   from the per-token compressor latent, group-boundary rows only;
3. ``dequantize_and_gather_k_cache`` of both caches into one bf16 workspace,
   compressed rows at offset 0 and SWA rows at offset ``N``;
4. ``combine_swa_first_indices`` (``combine_topk_swa_indices`` for ratio 0);
5. ``fused_norm_rope_attn_rope_cast_fwd`` on the fused-layout query.

Each output row only depends on its own query, its index row and the KV rows
those index, so packing, chunking and context parallelism do not change bits.

Gradient contract (backward is ``csrc/mega_attention_bwd.cu`` plus torch glue):

* Output FP8 cast: identity STE. The upstream gradient is taken with respect to
  the dequantized output ``fp8 * 2**(sf - 127)`` in the kernel's fused layout
  ``[S, 8, 8 * 512]``; it arrives on the ``grad`` carrier of the output.
* Output inverse RoPE and query RoPE: exact rotation VJPs at the row position.
  The bf16 rounding of the rotated query is an identity STE.
* KV cache quantization (SWA MXFP8, compressed NVFP4) and the bf16 rounding of
  the rotated KV: identity STE (no clipping mask: neither record clips, its
  scale covers the block maximum). The gradient reaches the pre-RoPE ``swa_kv``
  and compressor latent through the exact KV RoPE VJP at the cache position.
* Attention: the derivative of ``softmax([s, sink]) @ K`` with ``K = V`` the
  dequantized KV the forward read, ``s = scale * rope(q) . K``, P recomputed in
  FP32 from the saved LSE (``logaddexp(lse, sink)``), and
  ``D = sum_j P_j dP_j`` recomputed from P, not from the visible FP8 output, so
  each row of dS sums to zero. The forward's bf16 rounding of P before the PV
  product is an identity STE. ``dsink = -sum_rows P_sink * D``.
* bf16 operands with FP32 accumulation; dKV accumulates with FP32 atomics, so
  dKV (and the KV input gradients) are not bitwise reproducible run to run,
  unless ``deterministic`` selects the fixed-order dKV reduction.
"""
import math
from pathlib import Path
from typing import NamedTuple

import torch

HEADS = 64
HEAD_DIM = 512
ROPE_DIM = 64
WV_GROUP = 8
QUANT_GROUP = 32
WINDOW = 128
TOP_K = 512
PAGE = 64
SWA_RECORD = 528
COMPRESSED_RECORD = 288


class MegaAttentionOutput(NamedTuple):
    """``data``/``scale`` feed ``wo_a``'s ``fp8_einsum`` as vLLM's ``_o_proj`` does.

    ``grad`` is a zero-strided bf16 placeholder of ``data``'s shape. It carries
    no values; the consumer's backward returns the gradient with respect to the
    dequantized output into it (FP8 output STE).
    """

    data: torch.Tensor
    scale: torch.Tensor
    grad: torch.Tensor


# ---- layouts ---------------------------------------------------------------


def q_from_fused(q):
    """Fused query layout ``[S, H, 512]`` (16-dim chunks across heads) to natural."""
    s, h, d = q.shape
    return q.reshape(s, d // 16, h, 16).transpose(1, 2).reshape(s, h, d)


def q_to_fused(q):
    s, h, d = q.shape
    return q.reshape(s, h, d // 16, 16).transpose(1, 2).reshape(s, h, d)


def o_from_fused(o):
    """Output layout ``[S, groups, 8 * 512]`` (32-dim chunks across 8 heads) to natural."""
    s, g, width = o.shape
    d = width // WV_GROUP
    return o.reshape(s, g, d // 32, WV_GROUP, 32).permute(0, 1, 3, 2, 4).reshape(s, g * WV_GROUP, d)


def dequantize_output(data, scale):
    """Dequantized ``[S, groups, 8 * 512]`` FP32 output of the mega kernel."""
    s, g, width = data.shape
    exponents = scale.contiguous().view(torch.uint8).reshape(s, g, width // 32).float() - 127
    return (data.float().reshape(s, g, -1, 32) * torch.exp2(exponents).unsqueeze(-1)).reshape(
        s, g, width
    )


def rope(x, cos_sin_cache, positions, inverse=False):
    """GPT-J RoPE of the last 64 dims at ``positions`` in FP32; ``inverse`` is the transpose."""
    cs = cos_sin_cache.index_select(0, positions.long()).float()
    cos, sin = cs[:, : ROPE_DIM // 2], cs[:, ROPE_DIM // 2 :]
    shape = (-1,) + (1,) * (x.dim() - 2) + (ROPE_DIM // 2,)
    cos, sin = cos.reshape(shape), sin.reshape(shape)
    if inverse:
        sin = -sin
    x = x.float()
    even, odd = x[..., -ROPE_DIM::2], x[..., -ROPE_DIM + 1 :: 2]
    out = x.clone()
    out[..., -ROPE_DIM::2] = even * cos - odd * sin
    out[..., -ROPE_DIM + 1 :: 2] = odd * cos + even * sin
    return out


# ---- request plan ----------------------------------------------------------


class _Plan(NamedTuple):
    """Request layout of one rank's rows plus every index tensor the forward and
    backward need, built once on the host and copied in one non-blocking H2D."""

    requests: int
    n: int  # compressed rows per request in the workspace
    m: int  # workspace rows per request
    swa_pages: int
    comp_pages: int
    t: dict  # device tensors


def _host_plan(cu_seqlens, row_start, row_end, ratio):
    starts, seq_lens, qsl = [], [], [0]
    for a, b in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        lo, hi = max(a, row_start), min(b, row_end)
        if lo < hi:
            starts.append(a)
            seq_lens.append(hi - a)
            qsl.append(qsl[-1] + hi - lo)
    if qsl[-1] != row_end - row_start:
        raise ValueError("cu_seqlens must cover every local query row")
    n = max(s // ratio for s in seq_lens) if ratio else 0
    m = n + max(seq_lens)
    i32, i64 = torch.int32, torch.int64
    cat = lambda parts, dtype: torch.cat(parts) if parts else torch.empty(0, dtype=dtype)

    def paged(lengths):
        pages = [max(-(-length // PAGE), 1) for length in lengths]
        base = [sum(pages[:r]) for r in range(len(lengths))]
        width = max(pages)
        table = torch.tensor(
            [[b + min(p, k - 1) for p in range(width)] for b, k in zip(base, pages)], dtype=i32
        )
        return table, base, sum(pages)

    t = {}
    t["seq_lens"] = torch.tensor(seq_lens, dtype=i32)
    t["qsl"] = torch.tensor(qsl, dtype=i32)
    t["tokens"] = cat([torch.arange(a, a + k) for a, k in zip(starts, seq_lens)], i64)
    t["token_pos"] = cat([torch.arange(k) for k in seq_lens], i64)
    t["swa_table"], swa_base, swa_pages = paged(seq_lens)
    t["swa_slots"] = cat([torch.arange(k) + b * PAGE for b, k in zip(swa_base, seq_lens)], i64)
    rows = [r * m for r in range(len(starts))]
    t["swa_rows"] = torch.stack(
        [
            cat([torch.arange(k) + row + n for row, k in zip(rows, seq_lens)], i64),
            t["tokens"],
            t["token_pos"],
        ]
    )
    comp_pages = 0
    if ratio:
        counts = [k // ratio for k in seq_lens]
        t["counts"] = torch.tensor(counts, dtype=i32)
        t["comp_table"], comp_base, comp_pages = paged(counts)
        boundary = [torch.arange(c) * ratio + ratio - 1 for c in counts]
        t["comp_pos"] = cat(boundary, i64)
        t["comp_src"] = cat([a + j for a, j in zip(starts, boundary)], i64)
        t["comp_slots"] = cat([torch.arange(c) + b * PAGE for b, c in zip(comp_base, counts)], i64)
        t["comp_rows"] = torch.stack(
            [
                cat([torch.arange(c) + row for row, c in zip(rows, counts)], i64),
                t["comp_src"],
                t["comp_pos"] - (ratio - 1),
            ]
        )
    return len(starts), n, m, swa_pages, comp_pages, t


_PLANS = {}


def _plan(cu_seqlens, total, rank, local, ratio, device):
    """Cached per (cu_seqlens object, rank, ratio): every layer of a microbatch
    shares it, so at most the first layer reads ``cu_seqlens`` on the host."""
    import weakref

    if cu_seqlens is None or not torch.is_tensor(cu_seqlens):
        values = (0, total) if cu_seqlens is None else tuple(int(x) for x in cu_seqlens)
        key, ref = ("values", values), None
    else:
        key = ("tensor", id(cu_seqlens), cu_seqlens._version)
        ref = weakref.ref(cu_seqlens)
        values = None
    key = key + (total, rank, local, ratio, str(device))
    hit = _PLANS.get(key)
    if hit is not None and (hit[0] is None or hit[0]() is cu_seqlens):
        return hit[1]
    if values is None:
        values = [int(x) for x in cu_seqlens.tolist()]  # the only host sync
    requests, n, m, swa_pages, comp_pages, host = _host_plan(
        values, rank * local, (rank + 1) * local, ratio
    )
    t = {k: v.pin_memory().to(device, non_blocking=True) for k, v in host.items()}
    plan = _Plan(requests, n, m, swa_pages, comp_pages, t)
    if len(_PLANS) > 256:
        _PLANS.clear()
    _PLANS[key] = (ref, plan)
    return plan


def _forward(q, swa_kv, latent, topk, positions, sink, cos_sin_cache, ratio, window, top_k, plan):
    """Steps 1-5 of the module docstring; returns (out data, out scale, lse, kv, indices, lens)."""
    from vllm import envs

    if not envs.VLLM_BATCH_INVARIANT:
        raise RuntimeError("mega attention replays the BI=1 rollout; set VLLM_BATCH_INVARIANT=1")
    from vllm.models.deepseek_v41.common.ops import (
        combine_topk_swa_indices,
        dequantize_and_gather_k_cache,
    )
    from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import rope_quant_insert
    from vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn import alloc_mega_attn_output

    device = q.device
    t = plan.t
    seq_lens = t["seq_lens"]
    swa_cache = torch.zeros(plan.swa_pages, PAGE, SWA_RECORD, dtype=torch.uint8, device=device)
    torch.ops._C.fused_deepseek_v4_kv_rope_insert(
        swa_kv.index_select(0, t["tokens"]).contiguous(),
        swa_cache,
        t["swa_slots"],
        t["token_pos"],
        cos_sin_cache,
        PAGE,
        None,
        True,
    )
    kv = torch.empty(plan.requests, plan.m, HEAD_DIM, dtype=torch.bfloat16, device=device)
    if ratio:
        comp_cache = torch.zeros(
            plan.comp_pages, PAGE, COMPRESSED_RECORD, dtype=torch.uint8, device=device
        )
        rope_quant_insert(
            latent.index_select(0, t["comp_src"]).contiguous(),
            t["comp_pos"],
            cos_sin_cache,
            comp_cache,
            t["comp_slots"],
            ratio,
        )
        dequantize_and_gather_k_cache(
            kv[:, : plan.n],
            comp_cache,
            seq_lens=t["counts"],
            gather_lens=None,
            block_table=t["comp_table"],
            block_size=PAGE,
            offset=0,
        )
    dequantize_and_gather_k_cache(
        kv,
        swa_cache,
        seq_lens=seq_lens,
        gather_lens=seq_lens,
        block_table=t["swa_table"],
        block_size=PAGE,
        offset=plan.n,
    )
    qsl = t["qsl"]
    if ratio:
        from vllm.model_executor.kernels.batch_invariant import combine_swa_first_indices

        indices, lens = combine_swa_first_indices(
            topk, qsl, seq_lens, seq_lens, window, ratio, top_k, plan.m, plan.n
        )
    else:
        dummy = torch.full((q.shape[0], top_k), -1, dtype=torch.int32, device=device)
        indices, lens = combine_topk_swa_indices(
            dummy, qsl, seq_lens, seq_lens, window, 0, 0, plan.m, 0
        )
    out = alloc_mega_attn_output(q.shape[0], HEADS // WV_GROUP, device)
    _, _, _, lse = torch.ops._flashmla_C.fused_norm_rope_attn_rope_cast_fwd(
        q,
        kv.view(-1, 1, HEAD_DIM),
        indices.unsqueeze(1),
        HEAD_DIM**-0.5,
        HEAD_DIM,
        sink,
        lens,
        False,  # enable_q_norm
        0.0,
        positions.to(torch.int32),
        False,  # GPT-J RoPE
        ROPE_DIM,
        cos_sin_cache,
        HEADS // WV_GROUP,
        QUANT_GROUP,
        True,
        True,
        True,
        out.data,
        out.scale,
    )
    return out.data, out.scale, lse, kv, indices, lens


_EXTENSION = None


def _extension():
    global _EXTENSION
    if _EXTENSION is None:
        from torch.utils.cpp_extension import load

        _EXTENSION = load(
            name="ds41_mega_attention_bwd",
            sources=[
                str(Path(__file__).with_name("csrc") / name)
                for name in ("mega_attention_bwd.cu", "mega_attention_bwd_sm100.cu")
            ],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a"],
            verbose=False,
        )
    return _EXTENSION


# Deterministic dKV: per-(row, slot) partials (FP32 accumulated, stored bf16),
# summed per KV row in FP32 in a fixed (row-major) order instead of FP32
# atomics. Partials are bounded by DETERMINISTIC_PARTIAL_BYTES per row chunk.
DETERMINISTIC_DKV = False
DETERMINISTIC_PARTIAL_BYTES = 1 << 30
# Backward kernel: "umma" (tcgen05 + TMA gather4) or "mma" (mma.sync), same
# contract; kept selectable for A/B.
BACKWARD_KERNEL = "umma"


def _segments(indices, lens, kv_rows, chunk_rows):
    """Stable sort of the valid (row, slot) pairs by (row chunk, KV row).

    Returns the flat ``row * width + slot`` positions in that order and, for
    every (chunk, KV row), where its run starts (``chunks * kv_rows + 1``).
    """
    rows, width = indices.shape
    slots = torch.arange(width, device=indices.device)
    valid = (slots < lens[:, None]) & (indices >= 0) & (indices < kv_rows)
    chunk = torch.arange(rows, device=indices.device) // chunk_rows
    key = torch.where(
        valid, chunk[:, None] * kv_rows + indices.long(), torch.iinfo(torch.int64).max
    ).flatten()
    keys, order = torch.sort(key, stable=True)
    chunks = -(-rows // chunk_rows)
    starts = torch.searchsorted(
        keys, torch.arange(chunks * kv_rows + 1, device=indices.device, dtype=torch.int64)
    )
    return starts, order


def attention_backward_kernel(
    q,
    grad,
    positions,
    cos_sin_cache,
    kv,
    indices,
    lens,
    lse_full,
    scale,
    deterministic=None,
    kernel=None,
):
    """Owned VJP of the mega kernel's attention, RoPE and layouts included.

    ``q`` bf16 ``[S, 64, 512]`` fused query layout (unrotated); ``grad`` the
    gradient of the dequantized output, ``[S, 8, 8 * 512]`` fused output layout;
    ``kv`` bf16 ``[M, 512]`` (rotated, dequantized); ``lse_full`` FP32 ``[S, 64]``
    natural-log LSE including the sink. Returns bf16 ``dq`` (fused layout), FP32
    ``dkv [M, 512]`` w.r.t. the rotated KV, and FP32 ``D [S, 64]``.
    """
    deterministic = DETERMINISTIC_DKV if deterministic is None else deterministic
    dq = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    dkv = torch.zeros(kv.shape, dtype=torch.float32, device=q.device)
    delta = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
    indices, lens = indices.contiguous(), lens.contiguous()
    empty = torch.empty(0, dtype=torch.int64, device=q.device)
    starts = order = empty
    chunk_rows = max(1, DETERMINISTIC_PARTIAL_BYTES // (indices.shape[1] * HEAD_DIM * 2))
    if deterministic and q.shape[0]:
        starts, order = _segments(indices, lens, kv.shape[0], chunk_rows)
    kernel = BACKWARD_KERNEL if kernel is None else kernel
    if kernel not in ("umma", "mma"):
        raise ValueError(f"unknown mega attention backward kernel {kernel!r}")
    ext = _extension()
    (ext.backward_sm100 if kernel == "umma" else ext.backward)(
        q.contiguous(),
        grad.bfloat16().contiguous(),
        positions.to(torch.int32).contiguous(),
        cos_sin_cache.float().contiguous(),
        kv.contiguous(),
        indices,
        lens,
        (lse_full * math.log2(math.e)).contiguous(),
        float(scale),
        dq,
        dkv,
        delta,
        starts,
        order,
        chunk_rows,
    )
    return dq, dkv, delta


class _MegaAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        swa_kv,
        latent,
        topk,
        positions,
        sink,
        cos_sin_cache,
        ratio,
        window,
        top_k,
        cu_seqlens,
        group,
        deterministic,
    ):
        rank, size = (group.rank(), group.size()) if group is not None else (0, 1)
        local = q.shape[0]
        swa_full, latent_full, compact = swa_kv.detach(), None, None
        if latent is not None:
            # Compact boundary rows -> the per-token latent vLLM's insert reads,
            # without a host sync: boundary token t holds compact row
            # cumsum(flag)[t] - 1, every other token the dump row N_local.
            flag = (positions + 1) % ratio == 0
            compact = torch.where(flag, flag.long().cumsum(0) - 1, latent.shape[0])
            padded = torch.cat([latent.detach(), latent.new_zeros(1, HEAD_DIM)])
            latent_full = padded.index_select(0, compact)
        if size > 1:
            swa_full = _all_gather(swa_full, group)
            latent_full = None if latent_full is None else _all_gather(latent_full, group)
        plan = _plan(cu_seqlens, local * size, rank, local, ratio, q.device)
        sink32 = sink.detach().float().contiguous()
        data, scale, lse, kv, indices, lens = _forward(
            q.detach().contiguous(),
            swa_full.bfloat16().contiguous(),
            None if latent_full is None else latent_full.bfloat16().contiguous(),
            None if topk is None else topk.to(torch.int32).contiguous(),
            positions,
            sink32,
            cos_sin_cache,
            ratio,
            window,
            top_k,
            plan,
        )
        ctx.save_for_backward(q, kv, indices, lens, lse, positions, sink32, cos_sin_cache, compact)
        ctx.meta = (
            plan,
            ratio,
            group,
            rank,
            size,
            swa_kv.dtype,
            None if latent is None else (latent.dtype, latent.shape[0]),
            sink.dtype,
            deterministic,
        )
        ctx.mark_non_differentiable(data, scale)
        carrier = torch.zeros((), dtype=torch.bfloat16, device=q.device).expand(data.shape)
        return data, scale, carrier

    @staticmethod
    def backward(ctx, _data, _scale, grad):
        q, kv, indices, lens, lse, positions, sink, cos_sin_cache, compact = ctx.saved_tensors
        plan, ratio, group, rank, size, swa_dtype, latent_meta, sink_dtype, deterministic = ctx.meta
        if grad is None:
            grad = torch.zeros(q.shape[0], HEADS // WV_GROUP, WV_GROUP * HEAD_DIM, device=q.device)
        lse_full = torch.logaddexp(lse, sink.unsqueeze(0))
        dq, dkv, delta = attention_backward_kernel(
            q,
            grad,
            positions,
            cos_sin_cache,
            kv.view(-1, HEAD_DIM),
            indices,
            lens,
            lse_full,
            HEAD_DIM**-0.5,
            deterministic=deterministic,
        )
        dq = dq.to(q.dtype)
        dsink = -(torch.exp(sink.unsqueeze(0) - lse_full) * delta).sum(0).to(sink_dtype)
        swa_rows = plan.t["swa_rows"]
        total = q.shape[0] * size
        dswa = torch.zeros(total, HEAD_DIM, dtype=torch.float32, device=q.device)
        dswa[swa_rows[1]] = rope(dkv[swa_rows[0]], cos_sin_cache, swa_rows[2], inverse=True)
        dlatent = None
        if latent_meta is not None:
            comp_rows = plan.t["comp_rows"]
            dlatent = torch.zeros(total, HEAD_DIM, dtype=torch.float32, device=q.device)
            dlatent[comp_rows[1]] = rope(
                dkv[comp_rows[0]], cos_sin_cache, comp_rows[2], inverse=True
            )
        if size > 1:
            dswa = _reduce_scatter(dswa, group)
            dlatent = None if dlatent is None else _reduce_scatter(dlatent, group)
        dswa = dswa.to(swa_dtype)
        if dlatent is not None:
            dtype, count = latent_meta
            # The boundary tokens in order: the stable sort puts compact rows
            # 0..count-1 first (no host sync, deterministic).
            rows = torch.argsort(compact, stable=True)[:count]
            dlatent = dlatent.index_select(0, rows).to(dtype)
        return dq, dswa, dlatent, None, None, dsink, None, None, None, None, None, None, None


def _all_gather(x, group):
    out = x.new_empty((x.shape[0] * group.size(),) + tuple(x.shape[1:]))
    torch.distributed.all_gather_into_tensor(out, x.contiguous(), group=group)
    return out


def _reduce_scatter(x, group):
    out = x.new_empty((x.shape[0] // group.size(),) + tuple(x.shape[1:]))
    torch.distributed.reduce_scatter_tensor(out, x.contiguous(), group=group)
    return out


def mega_attention(
    q,
    swa_kv,
    comp_latent,
    topk_local,
    positions,
    attn_sink,
    *,
    compress_ratio,
    cos_sin_cache,
    window=WINDOW,
    top_k=TOP_K,
    cu_seqlens=None,
    cp_group=None,
    deterministic=None,
):
    """Mega attention core of one DS4.1 layer (forward bitwise = vLLM BI=1 prefill).

    Args:
        q: bf16 ``[S_local, 64, 512]`` ``wq_b`` output in the fused query layout.
        swa_kv: bf16 ``[S_local, 512]`` ``kv_norm`` output, before RoPE.
        comp_latent: bf16 ``[N_local, 512]`` compressor latent of the local
            group-boundary tokens (``(position + 1) % ratio == 0``) in token
            order, i.e. those rows of vLLM's per-token ``fused_save_compress_norm``
            output; ``None`` for ratio 0.
        topk_local: int32 ``[S_local, top_k]`` document-local compressed indices,
            ascending, ``-1`` at the end; ``None`` for ratio 0.
        positions: ``[S_local]`` within-document positions of the local rows.
        attn_sink: ``[64]`` sink logits.
        compress_ratio: 0, 1 or 2.
        cos_sin_cache: FP32 ``[max_pos, 64]``, vLLM's ``rotary_emb.cos_sin_cache``.
        cu_seqlens: global document offsets over all CP ranks' rows (THD);
            ``None`` is one document.
        cp_group: context-parallel group; ranks hold contiguous equal row slices.
        deterministic: fixed-order dKV reduction instead of FP32 atomics (bitwise
            reproducible backward); ``None`` uses :data:`DETERMINISTIC_DKV`.

    Returns:
        :class:`MegaAttentionOutput`.
    """
    if q.shape[1:] != (HEADS, HEAD_DIM) or swa_kv.shape != (q.shape[0], HEAD_DIM):
        raise ValueError("mega attention expects q [S, 64, 512] and swa_kv [S, 512]")
    if (compress_ratio == 0) != (comp_latent is None) or (compress_ratio == 0) != (
        topk_local is None
    ):
        raise ValueError("ratio-0 layers take no compressed latent or top-k; others need both")
    if cp_group is not None and cp_group.size() == 1:
        cp_group = None
    return MegaAttentionOutput(
        *_MegaAttention.apply(
            q,
            swa_kv,
            comp_latent,
            topk_local,
            positions,
            attn_sink,
            cos_sin_cache,
            compress_ratio,
            window,
            top_k,
            cu_seqlens,
            cp_group,
            deterministic,
        )
    )
