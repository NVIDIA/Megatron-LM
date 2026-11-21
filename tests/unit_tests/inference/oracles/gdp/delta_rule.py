# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""References and inputs for the gated delta rule kernel oracles (`megatron/core/ssm/ops/gdp/`).

Everything here is written from the delta-rule math and the launchers' documented arguments, not
from the kernels. Each function is dtype-generic: under `ou.fp64_reference` it is the fp64 oracle;
called on fp32 copies it is the trusted implementation (plain PyTorch on the same device, with
elementwise products and sums only, so TF32 never applies).

The gated delta rule, one step on a state h (K x V) with key k, value v, query q, beta b and
log decay d (natural log):

    h <- exp(d) h
    u  = b * (v - h^T k)             (b a scalar per head, or a vector over V)
    h <- h + k u^T
    o  = scale * h^T q

with q and k optionally L2-normalised first, x / sqrt(|x|^2 + 1e-6). The Gated Delta Product
(GDP) applies M such steps per token (its M Householder copies): the decay on the first copy only,
the query on the last only, and the token's output is the last copy's. GDN is GDP with M = 1.

The chunked prefill splits each sequence into chunks of `CHUNK` tokens on two streams: the token
stream (queries, outputs, decays) and the Householder-expanded stream (keys, values, betas: M
entries per token, `cu_seqlens * M`). Its stages are defined here from that math:

* cumulative decays: G[t] = scale * sum of g over t's chunk up to t (base 2: scale = 1/ln 2);
* A = beta K K^T with decay, strictly below the diagonal, per expanded chunk;
* the inverse (I + A)^-1 per chunk;
* the WY factors u = (I + A)^-1 (beta v), w = (I + A)^-1 (beta 2^G k);
* the chunk recurrence: entering state h_c, corrected values v_new = u - w h_c, next state
  2^G_last h_c + k^T (2^(G_last - G) v_new);
* the outputs: o_t = scale (2^G_t q_t h_j + sum over s <= t in the chunk, over the copies m, of
  2^(G_t - G_s) (q_t . k_{s,m}) v_new_{s,m}).
"""

import contextlib
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from megatron.core.ssm.ops.gdp.common import CHUNK_SIZE
from tests.unit_tests.inference.oracles import oracle_utils as ou

CHUNK = CHUNK_SIZE
"""Chunk length of every chunked GDP kernel (megatron/core/ssm/ops/gdp/common.py: CHUNK_SIZE)."""

L2NORM_EPS = 1e-6
"""Epsilon of the q/k L2 norm: `l2norm_fwd`'s default and the fused recurrent kernel's constant."""

RCP_LN2 = 1.0 / math.log(2.0)
"""1/ln 2: the chunked prefill converts natural-log decays to base 2."""

GATE_CLAMP = 50.0
"""`gdp_g_clamp_value` used for the clamp-on cases: #8926's example (g >= -50)."""

# ---------------------------------------------------------------------------------------------
# The step recurrence
# ---------------------------------------------------------------------------------------------


def l2norm(x: torch.Tensor, eps: float = L2NORM_EPS) -> torch.Tensor:
    """x / sqrt(sum(x^2) + eps) over the last dimension."""
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


def expand_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    """Grouped heads: value head hv reads key/query head hv // (heads // x_heads), dim -2."""
    repeat = heads // x.shape[-2]
    return x if repeat == 1 else x.repeat_interleave(repeat, dim=-2)


def gate(a, A_log, dt_bias=None, clamp=None):
    """GDP/GDN log decay: -exp(A_log) * softplus(a + dt_bias), optionally clamped below at -clamp.

    `a` [..., H], `A_log` and `dt_bias` [H]. softplus is torch's (identity above 20), the documented
    form of both the mixers' gate and the fused kernel's in-kernel gate.
    """
    x = a if dt_bias is None else a + dt_bias
    g = -torch.exp(A_log) * F.softplus(x)
    return g if clamp is None else g.clamp_min(-clamp)


def delta_rule(q, k, v, beta, decay, h0, *, scale, l2norm_qk, record=()):
    """The gated delta rule, step by step, for N independent sequences of T steps.

    Args:
        q, k: [N, T, H, K]; v: [N, T, HV, V], HV a multiple of H (grouped heads).
        beta: [N, T, HV] (one per head) or [N, T, HV, V].
        decay: [N, T, HV] natural-log decays, or None for no decay.
        h0: [N, HV, K, V] starting states.
        scale: output scale.
        l2norm_qk: L2-normalise q and k first.
        record: step counts c (0..T) after which to keep the state.

    Returns:
        (o [N, T, HV, V] the output of every step, h after the last step, {c: state after c steps}).
    """
    heads = v.shape[2]
    q, k = expand_heads(q, heads), expand_heads(k, heads)
    if l2norm_qk:
        q, k = l2norm(q), l2norm(k)
    h = h0.clone()
    kept = {0: h.clone()} if 0 in record else {}
    outs = []
    for t in range(k.shape[1]):
        if decay is not None:
            h = h * torch.exp(decay[:, t])[:, :, None, None]
        kt = k[:, t]
        b = beta[:, t] if beta.dim() == 4 else beta[:, t, :, None]
        u = b * (v[:, t] - (h * kt[..., None]).sum(-2))
        h = h + kt[..., None] * u[..., None, :]
        outs.append((h * q[:, t][..., None]).sum(-2) * scale)
        if t + 1 in record:
            kept[t + 1] = h.clone()
    o = torch.stack(outs, 1) if outs else v.new_zeros(v.shape)
    return o, h, kept


def fold_householder(q, k, v, g, beta, num_householder):
    """GDP as plain delta-rule steps: M steps per token, decay on the first, query on the last.

    q [N, T, H, K], k [N, T*M, H, K], v [N, T*M, HV, V], g [N, T, HV] or None, beta [N, T*M, ...].
    Returns (q_steps, k, v, decay_steps, beta) on the T*M-step stream.
    """
    M = num_householder
    N, T = q.shape[:2]
    q_steps = q.new_zeros(N, T, M, *q.shape[2:])
    q_steps[:, :, M - 1] = q
    decay = None
    if g is not None:
        decay = g.new_zeros(N, T, M, *g.shape[2:])
        decay[:, :, 0] = g
        decay = decay.reshape(N, T * M, *g.shape[2:])
    return q_steps.reshape(N, T * M, *q.shape[2:]), k, v, decay, beta


def gdp_sequence(q, k, v, g, beta, h0, *, num_householder, scale, l2norm_qk, chunk=CHUNK):
    """One GDP sequence from state h0: (o [T, H, V], final state, chunk-entry states [NC, H, K, V]).

    q [T, H, K], k [T*M, H, K], v [T*M, H, V], g [T, H] natural-log decays, beta [T*M, H], h0
    [H, K, V]. Chunk c's entry state is the state after the first c * chunk tokens (c * chunk * M
    steps), for c = 0 .. ceil(T / chunk) - 1.
    """
    M = num_householder
    T = q.shape[0]
    entries = [c * chunk * M for c in range(-(-T // chunk))]
    qs, ks, vs, decay, bs = fold_householder(
        q[None], k[None], v[None], None if g is None else g[None], beta[None], M
    )
    o, h, kept = delta_rule(
        qs, ks, vs, bs, decay, h0[None], scale=scale, l2norm_qk=l2norm_qk, record=entries
    )
    o = o[0, M - 1 :: M] if T else o[0, :0]
    states = torch.stack([kept[e][0] for e in entries]) if entries else h0.new_zeros(0, *h0.shape)
    return o, h[0], states


# ---------------------------------------------------------------------------------------------
# Varlen packs and chunks
# ---------------------------------------------------------------------------------------------


def bounds(cu_seqlens) -> List[Tuple[int, int]]:
    """[(start, end)] of every sequence of a pack."""
    cu = [int(x) for x in cu_seqlens]
    return list(zip(cu[:-1], cu[1:]))


def chunks(cu_seqlens, chunk=CHUNK) -> List[Tuple[int, int, int]]:
    """[(sequence, start, end)] of every chunk: `chunk` tokens from each sequence's start."""
    return [
        (n, c, min(c + chunk, end))
        for n, (start, end) in enumerate(bounds(cu_seqlens))
        for c in range(start, end, chunk)
    ]


def chunk_offsets(cu_seqlens, chunk=CHUNK) -> List[int]:
    """Prefix sum of the sequences' chunk counts, with a leading 0."""
    out = [0]
    for start, end in bounds(cu_seqlens):
        out.append(out[-1] + -(-(end - start) // chunk))
    return out


def nan_like(x: torch.Tensor) -> torch.Tensor:
    """A NaN tensor like `x`: the reference's value for tokens no sequence owns."""
    return torch.full_like(x, float("nan"))


def local_cumsum(g, cu_seqlens, *, scale=None, reverse=False, chunk=CHUNK, sequential=False):
    """o[t] = scale * sum of g[s] over s in t's chunk with s <= t (s >= t if reverse).

    g [1, T, H] with `cu_seqlens`; positions outside every sequence are NaN. `sequential` adds the terms
    one at a time in g's dtype (the trusted implementation: a plain running sum, whose rounding is no
    better than any scan's), instead of torch's cumsum (which accumulates in double on the CPU).
    """
    out = nan_like(g)
    for _, start, end in chunks(cu_seqlens, chunk):
        seg = g[:, start:end].flip(1) if reverse else g[:, start:end]
        if sequential:
            acc = [seg[:, 0]]
            for i in range(1, seg.shape[1]):
                acc.append(acc[-1] + seg[:, i])
            acc = torch.stack(acc, 1)
        else:
            acc = seg.cumsum(1)
        acc = acc.flip(1) if reverse else acc
        out[:, start:end] = acc if scale is None else acc * scale
    return out


def scaled_dot_kkt(k, beta, G, cu_seqlens, chunk=CHUNK, *, products=None):
    """A[t, hv, j - c0] = beta[t, hv] * (k_t . k_j) * 2^(G[t, hv] - G[j, hv]) for c0 <= j < t.

    k [1, T, H, K] (value head hv reads key head hv // (HV / H)), beta [1, T, HV], G [1, T, HV]
    (within-chunk base-2 cumulative decays) or None. Zero on and above the diagonal and in the
    columns past a short chunk's end; NaN in rows outside every sequence. Returns [1, T, HV, chunk].
    `products`, if given, computes the key products [HV, L, L] from the chunk's keys [L, HV, K]
    (the trusted implementation: `tensor_core_products`), in place of an einsum in k's dtype.
    """
    heads = beta.shape[2]
    keys = expand_heads(k, heads)
    out = beta.new_full((*beta.shape, chunk), float("nan"))
    for _, start, end in chunks(cu_seqlens, chunk):
        kc = keys[0, start:end]  # [L, HV, K]
        if products is None:
            scores = torch.einsum("ihk,jhk->hij", kc, kc)
        else:
            scores = products(kc).to(out.dtype)
        if G is not None:
            gc = G[0, start:end].transpose(0, 1)  # [HV, L]
            scores = scores * torch.exp2(gc[:, :, None] - gc[:, None, :])
        scores = scores * beta[0, start:end].transpose(0, 1)[:, :, None]
        L = end - start
        block = torch.zeros(heads, L, chunk, dtype=out.dtype, device=out.device)
        block[:, :, :L] = torch.tril(scores, diagonal=-1)
        out[0, start:end] = block.permute(1, 0, 2)
    return out


def tensor_core_products(kc):
    """k_i . k_j of a chunk's keys [L, HV, K] in the activation dtype, as cuBLAS computes them on a
    GPU: a bf16/fp16 tensor-core GEMM with an fp32 result (`torch.bmm(..., out_dtype=torch.float32)`),
    the same hardware accumulation a Triton `tl.dot` of those operands uses. Without CUDA (no tensor
    cores) the fp32 product."""
    k = kc.transpose(0, 1)  # [HV, L, K]
    if k.is_cuda and k.dtype in (torch.bfloat16, torch.float16):
        return torch.bmm(k, k.transpose(1, 2), out_dtype=torch.float32)
    k = k.float()
    return (k[:, :, None, :] * k[:, None, :, :]).sum(-1)


def tril_inverse(A, cu_seqlens, chunk=CHUNK):
    """Ai = (I + A_c)^-1 per chunk c of L rows, A_c = A[c0:c0+L, hv, :L] (strictly lower).

    Zero above the diagonal and in columns >= L; NaN in rows outside every sequence.
    """
    out = nan_like(A)
    for _, start, end in chunks(cu_seqlens, chunk):
        L = end - start
        a = A[0, start:end, :, :L].permute(1, 0, 2)  # [HV, L, L]
        eye = torch.eye(L, dtype=A.dtype, device=A.device).expand_as(a)
        inv = torch.linalg.solve_triangular(eye + torch.tril(a, -1), eye, upper=False)
        block = torch.zeros(A.shape[2], L, chunk, dtype=A.dtype, device=A.device)
        block[:, :, :L] = inv
        out[0, start:end] = block.permute(1, 0, 2)
    return out


def wy_factors(k, v, beta, Ai, G, cu_seqlens, chunk=CHUNK, *, operand=None):
    """u = Ai_c (beta v), w = Ai_c (beta 2^G k) per chunk; returns (w [1, T, HV, K], u like v).

    `operand`, if given, is applied to beta v and beta 2^G k before the product: the trusted
    implementation materializes them in the input dtype, as a bf16 product would.
    """
    heads = v.shape[2]
    keys = expand_heads(k, heads)
    w = beta.new_full((*beta.shape, k.shape[-1]), float("nan"))
    u = nan_like(v).to(beta.dtype)
    for _, start, end in chunks(cu_seqlens, chunk):
        L = end - start
        ai = Ai[0, start:end, :, :L].permute(1, 0, 2)  # [HV, L, L]
        b = beta[0, start:end].transpose(0, 1)[:, :, None]  # [HV, L, 1]
        bv = b * v[0, start:end].transpose(0, 1)
        bk = b * keys[0, start:end].transpose(0, 1)
        if G is not None:
            bk = bk * torch.exp2(G[0, start:end].transpose(0, 1))[:, :, None]
        if operand is not None:
            bv, bk = operand(bv), operand(bk)
        u[0, start:end] = torch.einsum("hij,hjv->ihv", ai, bv)
        w[0, start:end] = torch.einsum("hij,hjk->ihk", ai, bk)
    return w, u


@dataclass
class ChunkStates:
    """The chunk recurrence's results for a pack."""

    entries: torch.Tensor  # [NC, H, K, V]: chunk-entry states, rows by `chunk_offsets`
    v_new: torch.Tensor  # [1, T*M, H, V], NaN outside every sequence
    final: torch.Tensor  # [N, H, K, V]


def chunk_recurrence(
    k,
    w,
    u,
    G,
    h0,
    cu_seqlens,
    *,
    num_householder,
    chunk=CHUNK,
    state_operand=None,
    update_operand=None,
):
    """The inter-chunk recurrence over each sequence's expanded chunks.

    k, w [1, T*M, H, K], u [1, T*M, H, V], G [1, T*M, H] (within-chunk base-2 cumulative decays of the
    expanded stream) or None, h0 [N, H, K, V] or None (zeros); `cu_seqlens` on the token stream. For
    expanded chunk c of a sequence (rows r): h_c enters; v_new_r = u_r - w_r h_c; h_{c+1} =
    2^G_last h_c + sum_r k_r (2^(G_last - G_r) v_new_r)^T. Expanded chunk c = j M enters token chunk
    j, whose entry state is row `chunk_offsets[n] + j` of `entries`.

    `state_operand` / `update_operand`, if given, round h_c before the w product and the decayed
    v_new before the k product (the trusted implementation's input-dtype operands).
    """
    M = num_householder
    H, K, V = k.shape[2], k.shape[3], u.shape[3]
    seqs = bounds(cu_seqlens)
    offsets = chunk_offsets(cu_seqlens, chunk)
    entries = u.new_full((offsets[-1], H, K, V), float("nan"))
    finals = u.new_zeros(len(seqs), H, K, V)
    v_new = nan_like(u)
    for n, (start, end) in enumerate(seqs):
        h = u.new_zeros(H, K, V) if h0 is None else h0[n].clone()
        for c, s in enumerate(range(start * M, end * M, chunk)):
            e = min(s + chunk, end * M)
            if c % M == 0:
                entries[offsets[n] + c // M] = h
            hc = h if state_operand is None else state_operand(h)
            kc, wc, uc = k[0, s:e], w[0, s:e], u[0, s:e]  # [L, H, *]
            vn = uc - (wc[..., None] * hc[None]).sum(-2)
            v_new[0, s:e] = vn
            if G is not None:
                gc = G[0, s:e]  # [L, H]
                last = gc[-1]
                vn = vn * torch.exp2(last[None] - gc)[..., None]
                h = h * torch.exp2(last)[:, None, None]
            if update_operand is not None:
                vn = update_operand(vn)
            h = h + (kc[..., None] * vn[:, :, None, :]).sum(0)
        finals[n] = h
    return ChunkStates(entries=entries, v_new=v_new, final=finals)


def chunk_outputs(
    q, k, v_new, h, G, cu_seqlens, *, num_householder, scale, chunk=CHUNK, score_operand=None
):
    """o_t = scale (2^G_t q_t^T h_j + sum_{c0 <= s <= t} sum_m 2^(G_t - G_s) (q_t . k_{s,m}) v_new_{s,m}).

    q [1, T, H, K], k [1, T*M, H, K], v_new [1, T*M, H, V], h [NC, H, K, V] chunk-entry states (row
    `chunk_offsets[n] + j` for chunk j of sequence n), G [1, T, H] token-stream base-2 within-chunk
    cumulative decays or None. NaN outside every sequence. `score_operand`, if given, rounds the
    masked, decayed score matrix before its product with v_new.
    """
    M = num_householder
    H, V = q.shape[2], v_new.shape[3]
    out = q.new_full((1, q.shape[1], H, V), float("nan"))
    offsets = chunk_offsets(cu_seqlens, chunk)
    for n, start, end in chunks(cu_seqlens, chunk):
        j = (start - bounds(cu_seqlens)[n][0]) // chunk
        L = end - start
        qc = q[0, start:end]  # [L, H, K]
        inter = (qc[..., None] * h[offsets[n] + j][None]).sum(-2)  # [L, H, V]
        causal = torch.ones(L, L, dtype=torch.bool, device=q.device).tril()
        if G is not None:
            gc = G[0, start:end].transpose(0, 1)  # [H, L]
            inter = inter * torch.exp2(gc.transpose(0, 1))[..., None]
            decay = torch.exp2(gc[:, :, None] - gc[:, None, :])
        intra = torch.zeros_like(inter)
        for m in range(M):
            km = k[0, start * M + m : end * M : M]  # [L, H, K]
            vm = v_new[0, start * M + m : end * M : M]
            scores = torch.einsum("ihk,jhk->hij", qc, km)
            if G is not None:
                scores = scores * torch.where(causal, decay, torch.zeros_like(decay))
            else:
                scores = scores * causal
            if score_operand is not None:
                scores = score_operand(scores)
            intra = intra + torch.einsum("hij,jhv->ihv", scores, vm)
        out[0, start:end] = (inter + intra) * scale
    return out


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------

G_REGIMES = ("production", "clamp_on", "clamp_off")
"""Log-decay regimes. production: the mixers' gate at their initialisation ranges; clamp_off: the
same gate with heavy-tailed gate inputs, so g reaches about -1e4 (decays below exp(-50), which #8926's
clamp exists for); clamp_on: those same decays clamped at -GATE_CLAMP, as the mixer does with
gdp_g_clamp_value = 50."""


def gate_parameters(heads, *, generator, device):
    """A_log and dt_bias as the GDP/GDN mixers initialise them (fp32): A in [1, 16], dt in [1e-3, 0.1]."""
    A = 1 + 15 * torch.rand(heads, generator=generator, device=device)
    dt = torch.exp(
        math.log(1e-3)
        + (math.log(0.1) - math.log(1e-3)) * torch.rand(heads, generator=generator, device=device)
    )
    dt_bias = dt + torch.log(-torch.expm1(-dt))  # softplus^-1(dt)
    return torch.log(A), dt_bias


def gate_inputs(shape, regime, *, generator, device):
    """Raw gate inputs `a` (fp32): N(0, 1), or with a heavy tail for the clamp regimes."""
    a = torch.randn(shape, generator=generator, device=device)
    if regime != "production":
        heavy = torch.rand(shape, generator=generator, device=device) < 0.5
        tail = 4 + 796 * torch.rand(shape, generator=generator, device=device)
        a = torch.where(heavy, tail, a)
    return a


def log_decays(shape, regime, *, generator, device):
    """fp32 log decays g [..., H] for a regime, from the mixers' gate (natural log, <= 0)."""
    heads = shape[-1]
    A_log, dt_bias = gate_parameters(heads, generator=generator, device=device)
    a = gate_inputs(shape, regime, generator=generator, device=device)
    clamp = GATE_CLAMP if regime == "clamp_on" else None
    return gate(a, A_log, dt_bias, clamp).to(torch.float32)


@dataclass
class Pack:
    """One varlen prefill step: real sequences, zero-length padding requests, padded tail tokens."""

    q: torch.Tensor  # [1, T, H, K]
    k: torch.Tensor  # [1, T*M, H, K]
    v: torch.Tensor  # [1, T*M, H, V]
    g: torch.Tensor  # [1, T, H] fp32
    beta: torch.Tensor  # [1, T*M, H]
    cu_seqlens: torch.Tensor  # [N+1] int32, padding requests zero-length at the end
    lens: Sequence[int]  # real and padding lengths
    num_householder: int
    tokens: int  # real tokens, cu_seqlens[-1]

    @property
    def M(self) -> int:
        return self.num_householder


def make_pack(
    lens,
    *,
    num_householder,
    heads,
    head_k,
    head_v,
    dtype=torch.bfloat16,
    beta_dtype=None,
    regime="production",
    padded_tokens=None,
    generator,
    device="cuda",
):
    """Inputs of one prefill step, distributed as the mixers produce them.

    q, k, v ~ N(0, 1) in `dtype` (post-conv activations; the kernels L2-normalise q and k); beta =
    sigmoid(N(0, 1)) in `beta_dtype` (GDP: the activation dtype; GDN: fp32); g from the mixers' gate
    in fp32 (`log_decays`). `lens` may end with zero-length padding requests; tokens past the real
    ones up to `padded_tokens` are NaN in every input (they belong to no sequence).
    """
    M = num_householder
    real = sum(lens)
    T = real if padded_tokens is None else padded_tokens
    beta_dtype = dtype if beta_dtype is None else beta_dtype

    def rand(*shape, dt=dtype):
        return torch.randn(*shape, generator=generator, device=device).to(dt)

    q = rand(1, T, heads, head_k)
    k = rand(1, T * M, heads, head_k)
    v = rand(1, T * M, heads, head_v)
    g = log_decays((1, T, heads), regime, generator=generator, device=device)
    beta = torch.sigmoid(torch.randn(1, T * M, heads, generator=generator, device=device)).to(
        beta_dtype
    )
    for x, per in ((q, 1), (k, M), (v, M), (g, 1), (beta, M)):
        ou.poison_(x[:, real * per :])
    cu = torch.tensor(
        [0, *torch.tensor(list(lens)).cumsum(0).tolist()], dtype=torch.int32, device=device
    )
    return Pack(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu,
        lens=list(lens),
        num_householder=M,
        tokens=real,
    )


def descriptors(pack: Pack, padded_tokens=None, device="cuda") -> Dict[str, torch.Tensor]:
    """The production chunk descriptors for a pack (`build_gdp_chunk_descriptors`, padded)."""
    from megatron.core.ssm.ops.gdp.metadata import build_gdp_chunk_descriptors

    cu = [int(x) for x in pack.cu_seqlens]
    padded = pack.q.shape[1] if padded_tokens is None else padded_tokens
    idx, idx_dp, offsets, n_chunks, n_chunks_dp = build_gdp_chunk_descriptors(
        cu, len(cu) - 1, pack.M, padded
    )

    def dev(values, rows):
        return torch.tensor(values, dtype=torch.int32, device=device).view(rows, -1)

    return dict(
        chunk_indices=dev(idx, n_chunks),
        chunk_indices_dp=dev(idx_dp, n_chunks_dp),
        chunk_offsets=torch.tensor(offsets, dtype=torch.int32, device=device),
    )


def round_to(dtype):
    """A function rounding a tensor to `dtype` and back (exactly representable afterwards)."""

    def fn(x):
        return x.to(dtype).to(x.dtype)

    return fn


def trusted_fp32(fn, *args, **kwargs):
    """`fn` on fp32 copies of its floating-point inputs: the trusted implementation's precision."""

    def cast(x):
        if isinstance(x, torch.Tensor) and x.is_floating_point():
            return x.to(torch.float32)
        if isinstance(x, (list, tuple)):
            return type(x)(cast(y) for y in x)
        return x

    with no_tf32():
        return fn(*cast(args), **{k: cast(v) for k, v in kwargs.items()})


@contextlib.contextmanager
def no_tf32():
    """TF32 off for matmuls and convolutions in the block (trusted fp32 implementations)."""
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


# ---------------------------------------------------------------------------------------------
# Autotune pinning
# ---------------------------------------------------------------------------------------------


def autotuner(kernel):
    """The Triton autotuner around a (heuristics-wrapped) kernel."""
    from triton.runtime.autotuner import Autotuner

    while not isinstance(kernel, Autotuner):
        kernel = kernel.fn
    return kernel


@contextlib.contextmanager
def pinned(kernel, config):
    """Leave `kernel`'s autotuner one candidate, `config`, with a cold cache."""
    tuner = autotuner(kernel)
    saved = tuner.configs, tuner.cache
    try:
        tuner.configs, tuner.cache = [config], {}
        yield
    finally:
        tuner.configs, tuner.cache = saved


def distinct_tiles(kernel, keys):
    """One autotune candidate per distinct value of the block-size `keys`: the cheapest by warps
    and stages, in list order. Warps and stages change scheduling, not the tile arithmetic."""
    seen = {}
    for config in autotuner(kernel).configs:
        tile = tuple(config.kwargs.get(key) for key in keys)
        best = seen.get(tile)
        cost = (config.num_warps, config.num_stages)
        if best is None or cost < (best.num_warps, best.num_stages):
            seen[tile] = config
    return list(seen.values())


def config_id(config) -> str:
    kw = "".join(f"{k}{v}" for k, v in sorted(config.kwargs.items()))
    return f"{kw}w{config.num_warps}s{config.num_stages}"


@contextlib.contextmanager
def pinned_autotuning(modules=("megatron.core", "fla")):
    """Deterministic-mode configuration selection for the block: every autotuner under `modules`
    runs its cheapest candidate without timing (`megatron.core.tuning`, mode "pinned"), as
    `--deterministic-mode` does. Bounds compilation to one config per kernel and shape."""
    from megatron.core.tuning import AutotunePolicy, install
    from megatron.core.tuning.interception import active_policy

    previous = active_policy()
    install(AutotunePolicy(mode="pinned", modules=tuple(modules)))
    try:
        yield
    finally:
        install(previous)


# ---------------------------------------------------------------------------------------------
# Stage inputs: the chunked pipeline's intermediate tensors, from the definitions above
# ---------------------------------------------------------------------------------------------

STAGE_PACKS = {
    # name: (sequence lengths, the last ones zero-length padding requests; M)
    "M1": ((1, 15, 16, 17, 63, 64, 65, 129, 0), 1),
    "M1b": ((31, 32, 33, 47, 48, 49, 0), 1),
    "M2": ((31, 32, 33, 64, 1, 0, 0), 2),
    "M3": ((5, 21, 22, 43, 48, 0), 3),
}
"""Packs whose chunk boundaries fall at B-1/B/B+1 of the chunk (64) on the token stream and the
expanded stream: last-chunk lengths 1, 2, 15, 16, 17, 31, 32, 33, 47, 48, 49, 62, 63, 64 (the
16-row blocks of the 64x64 inverse), 1- and 2-token sequences, sequences of 2 and 3 chunks, and
zero-length padding requests. Tokens past the real ones are padding (NaN)."""

TAIL_TOKENS = 37
"""Padded tail tokens after the real ones (no sequence owns them)."""


@dataclass
class Chain:
    """One pack's chunked-pipeline tensors, each from the fp64 definitions above, rounded to the
    dtype the pipeline hands it on in: q, k L2-normalised in the activation dtype; G, Gdp fp32
    base-2 within-chunk cumulative decays; A fp32; Ai, w, u, v_new and the chunk-entry states h in
    the activation dtype (h in `states_dtype` when given); NaN outside every sequence."""

    pack: Pack
    desc: Dict[str, torch.Tensor]
    cu: torch.Tensor
    cu_dp: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    G: torch.Tensor
    Gdp: torch.Tensor
    A: torch.Tensor
    Ai: torch.Tensor
    w: torch.Tensor
    u: torch.Tensor
    h0: Optional[torch.Tensor]
    h: torch.Tensor  # [1, NT_padded, H, K, V]: entry states at chunk_offsets rows, NaN elsewhere
    v_new: torch.Tensor


def interleave_decays(g, num_householder):
    """[1, T, H] -> [1, T*M, H]: each token's decay on its first Householder copy, 0 on the rest."""
    M = num_householder
    out = g.new_zeros(g.shape[0], g.shape[1], M, g.shape[2])
    out[:, :, 0] = g
    return out.reshape(g.shape[0], g.shape[1] * M, g.shape[2])


def build_chain(pack: Pack, *, h0=None, states_dtype=None, use_g=True, device="cuda") -> Chain:
    """The pipeline's intermediate tensors for `pack` (see `Chain`)."""
    from megatron.core.ssm.ops.gdp.common import RCP_LN2 as KERNEL_RCP_LN2

    dtype = pack.q.dtype
    M = pack.M
    desc = descriptors(pack, device=device)
    cu, cu_dp = pack.cu_seqlens, pack.cu_seqlens * M
    real = pack.tokens
    q = l2norm(pack.q.double()).to(dtype)
    k = l2norm(pack.k.double()).to(dtype)
    if use_g:
        G = local_cumsum(pack.g.double(), cu, scale=KERNEL_RCP_LN2).to(torch.float32)
        gi = interleave_decays(pack.g, M)
        ou.poison_(gi[:, real * M :])
        Gdp = local_cumsum(gi.double(), cu_dp, scale=KERNEL_RCP_LN2).to(torch.float32)
    else:
        G = Gdp = None
    A = scaled_dot_kkt(k.double(), pack.beta.double(), None if Gdp is None else Gdp.double(), cu_dp)
    A = A.to(torch.float32)
    Ai = tril_inverse(A.double(), cu_dp).to(dtype)
    w, u = wy_factors(
        k.double(),
        pack.v.double(),
        pack.beta.double(),
        Ai.double(),
        None if Gdp is None else Gdp.double(),
        cu_dp,
    )
    w, u = w.to(dtype), u.to(dtype)
    states = chunk_recurrence(
        k.double(),
        w.double(),
        u.double(),
        None if Gdp is None else Gdp.double(),
        None if h0 is None else h0.double(),
        cu,
        num_householder=M,
    )
    n_rows = desc["chunk_indices"].shape[0]
    H, K, V = k.shape[2], k.shape[3], u.shape[3]
    h = torch.full((1, n_rows, H, K, V), float("nan"), dtype=states_dtype or dtype, device=device)
    h[0, : states.entries.shape[0]] = states.entries.to(h.dtype)
    return Chain(
        pack=pack,
        desc=desc,
        cu=cu,
        cu_dp=cu_dp,
        q=q,
        k=k,
        G=G,
        Gdp=Gdp,
        A=A,
        Ai=Ai,
        w=w,
        u=u,
        h0=h0,
        h=h,
        v_new=states.v_new.to(dtype),
    )


def owned_rows(cu_seqlens, total) -> torch.Tensor:
    """Bool mask [total] of the stream positions some sequence owns."""
    mask = torch.zeros(total, dtype=torch.bool)
    for start, end in bounds(cu_seqlens):
        mask[start:end] = True
    return mask


def structural_zeros(cu_seqlens, total, chunk=CHUNK, *, diagonal=True) -> torch.Tensor:
    """Bool [total, chunk]: for each owned row, the in-chunk columns above the diagonal (and on it, with
    `diagonal`) and those past a short chunk's end: zero by definition in A (with the diagonal) and in
    (I + A)^-1 (without)."""
    mask = torch.zeros(total, chunk, dtype=torch.bool)
    col = torch.arange(chunk)[None, :]
    for _, start, end in chunks(cu_seqlens, chunk):
        L = end - start
        row = torch.arange(L)[:, None]
        mask[start:end] = (col >= row if diagonal else col > row) | (col >= L)
    return mask
