# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Experimental MLA context parallelism that exchanges the latent KV between CP ranks.

``MLAWithLatentCP`` keeps everything of ``MLASelfAttention`` except core attention. The latent KV
of the local sequence shard (``kv_lora_rank + qk_pos_emb_head_dim`` channels, gathered over the
tensor/sequence-parallel group once) travels around the CP ring in a zigzag schedule; every step
re-runs the inherited ``linear_kv_up_proj`` on the received latent, attends to it, and merges the
partial outputs in fp32. Backward re-circulates ``(latent, dlatent)`` pairs and recomputes the
up-projection per step, so no full K/V is ever saved. Core attention runs directly on the public
Transformer Engine ``fused_attn_fwd/bwd`` (cuDNN) or FlashAttention-4 kernels instead of TE's
attention wrapper. THD packed input in the zigzag CP layout, causal self-attention. RoPE is
applied as in ``MLASelfAttention``: the query's and the latent's positional channels are rotated
once before the ring; a parent that marks a layer RoPE-free (``use_rope=False``) skips it. A
contiguous CP layout for MLA depends on the input conversion of PR #7545.

``cp_comm_type`` ``a2a+p2p`` (hierarchical CP groups) and ``a2a`` add a Ulysses stage over the a2a
group: the latents of its ``A`` ranks are gathered, query tokens are exchanged for heads, and the
gathered block is reordered into the zigzag of the p2p group (natural order for pure ``a2a``), so
the ring runs unchanged with ``A``-times longer steps on ``H/A`` heads (``P/A`` steps).
"""

from __future__ import annotations

import copy
import inspect
from contextlib import contextmanager, nullcontext
from importlib.metadata import PackageNotFoundError, version
from typing import NamedTuple, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.jit import jit_fuser
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.tensor_parallel.mappings import (
    all_to_all_hp2sp,
    all_to_all_sp2hp,
    gather_from_sequence_parallel_region,
)
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.typed_torch import apply_module
from megatron.core.utils import get_pg_size

if HAVE_TE:
    from transformer_engine.pytorch import LayerNormLinear, Linear
    from transformer_engine.pytorch.cpp_extensions.fused_attn import (
        FusedAttnBackend,
        fused_attn_bwd,
        fused_attn_fwd,
    )
    from transformer_engine.pytorch.distributed import activation_recompute_forward
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager, fp8_autocast

    # Older TE builds lack the THD layout keywords; their defaults would silently mean sbhd.
    _FUSED_ATTN_API_OK = {"o_format", "cu_seqlens_q_padded"} <= set(
        inspect.signature(fused_attn_fwd).parameters
    ) and {"o_format", "do_format", "dqkv_layout", "deterministic"} <= set(
        inspect.signature(fused_attn_bwd).parameters
    )
else:
    _FUSED_ATTN_API_OK = False

# FlashAttention-4 has no public split forward/backward API; the module drives the two entry
# points its own autograd function uses, pinned to the version they were validated against.
_FA4_VERSION = "4.0.0b11"
_FA4 = None


def apply_mla_latent_cp_spec(attn_spec: ModuleSpec) -> ModuleSpec:
    """Return a copy of an MLA self-attention spec that builds ``MLAWithLatentCP``.

    The copy swaps the module class and replaces ``core_attention`` with ``IdentityOp`` (the TE
    attention wrapper is never built). The input spec is not modified.
    """
    spec = copy.deepcopy(attn_spec)
    spec.module = MLAWithLatentCP
    spec.submodules.core_attention = IdentityOp
    return spec


class _Step(NamedTuple):
    """One ring step. ``cu_*`` are actual cumulative lengths, ``off_*`` padded token offsets."""

    upper: bool  # local second chunk only (compact rows) vs. all local queries
    cu_q: torch.Tensor
    cu_kv: torch.Tensor
    off_q: torch.Tensor
    off_kv: torch.Tensor
    max_q: int
    max_kv: int
    causal: bool


class _StepPlan(NamedTuple):
    """THD metadata of the zigzag ring on one CP rank (device int32, no host synchronisation).

    Every local sequence is stored as ``[first chunk | second chunk]`` of equal padded length;
    rank ``r`` owns chunks ``r`` and ``2P-1-r`` of each packed sequence. Step ``i`` attends to the
    latent of owner ``(r - i) mod P``: all local queries against both owner chunks (step 0,
    causal), against the owner's first chunk only (owner before this rank; zero copy through the
    full offsets with half lengths), or the local second chunk against both owner chunks.
    """

    steps: list[_Step]  # forward ring order; backward walks it in reverse
    rows: Optional[torch.Tensor]  # [T/2] int64 rows of this rank's second chunks (upper steps)
    valid: Optional[torch.Tensor]  # [T] bool actual-token rows (None without padded cu_seqlens)


def _build_step_plan(packed_seq_params, cp_size: int, cp_rank: int, num_tokens: int) -> _StepPlan:
    """Derive the ring metadata from the global THD ``cu_seqlens`` (actual and padded)."""
    cu = packed_seq_params.cu_seqlens_q
    cup = packed_seq_params.cu_seqlens_q_padded
    padded = cup is not None
    if not padded:
        cup = cu
    device = cu.device
    off_local = torch.div(cup, cp_size, rounding_mode="floor")
    off_half = torch.div(off_local, 2, rounding_mode="floor")
    half = off_half[1:] - off_half[:-1]
    seqlens = cu[1:] - cu[:-1]
    owners = torch.arange(cp_size, device=device, dtype=cu.dtype).unsqueeze(1)
    # Actual tokens in chunk c of a sequence with s actual tokens: clamp(s - c * half, 0, half).
    first = torch.minimum((seqlens - owners * half).clamp_(min=0), half)
    second = torch.minimum((seqlens - (2 * cp_size - 1 - owners) * half).clamp_(min=0), half)

    def cumsum0(lengths):
        return F.pad(torch.cumsum(lengths, dim=-1, dtype=torch.int32), (1, 0))

    act_full, act_first, act_second = cumsum0(first + second), cumsum0(first), cumsum0(second)
    L = int(packed_seq_params.max_seqlen_q) // cp_size
    steps = []
    for i in range(cp_size):
        owner = (cp_rank - i) % cp_size
        if i == 0:
            step = _Step(
                False, act_full[cp_rank], act_full[owner], off_local, off_local, L, L, True
            )
        elif i <= cp_rank:
            step = _Step(
                False, act_full[cp_rank], act_first[owner], off_local, off_local, L, L // 2, False
            )
        else:
            step = _Step(
                True, act_second[cp_rank], act_full[owner], off_half, off_local, L // 2, L, False
            )
        steps.append(step)

    rows = valid = None
    if cp_rank < cp_size - 1:
        # Row index of the local second chunks in the compact second-half layout; the sum of
        # ``half`` is num_tokens // 2 for MCore THD batches (every tail is a padded sequence).
        n_half = num_tokens // 2
        reps = half.to(torch.int64)
        starts = torch.repeat_interleave(off_local[:-1] + half, reps, output_size=n_half)
        bases = torch.repeat_interleave(off_half[:-1], reps, output_size=n_half)
        rows = starts.to(torch.int64) + torch.arange(n_half, device=device) - bases.to(torch.int64)
    if padded:
        f, s = first[cp_rank], second[cp_rank]
        lens = torch.stack([f, half - f, s, half - s], dim=1).reshape(-1).to(torch.int64)
        pattern = torch.tensor([True, False, True, False], device=device).repeat(half.numel())
        valid = torch.repeat_interleave(pattern, lens, output_size=num_tokens)
    return _StepPlan(steps=steps, rows=rows, valid=valid)


class _Permute(torch.autograd.Function):
    """Row permutation by ``index_select`` in both directions (no ``index_add`` atomics)."""

    @staticmethod
    def forward(ctx, x, perm, inv):
        ctx.inv = inv
        return x.index_select(0, perm)

    @staticmethod
    def backward(ctx, grad):
        return grad.index_select(0, ctx.inv), None, None


def _a2a_permutation(packed_seq_params, cp_size: int, a2a_size: int, num_tokens: int):
    """Row index from the rank-major gather over the a2a group to the ring's sequence order.

    The gather yields ``A`` blocks of ``num_tokens`` rows, block ``a`` = source rank ``a``'s
    sequence-major ``[first | second]`` THD block. Sequence order per sequence is
    ``[first(0) … first(A-1) | second(A-1) … second(0)]`` (global chunk ids ascending in both
    halves): the zigzag of the p2p group with chunk size ``A * c`` (natural order for pure a2a).
    Returns ``(perm, inv)`` with ``x.index_select(0, perm)`` in sequence order; device int64, no
    host synchronisation.
    """
    cup = packed_seq_params.cu_seqlens_q_padded
    if cup is None:
        cup = packed_seq_params.cu_seqlens_q
    device = cup.device
    off_local = torch.div(cup, cp_size, rounding_mode="floor")  # [B+1] per-rank padded offsets
    c = torch.div(off_local[1:] - off_local[:-1], 2, rounding_mode="floor")  # [B] chunk length
    slots = torch.arange(2 * a2a_size, device=device)  # [2A] target slots within a sequence
    src_rank = torch.where(slots < a2a_size, slots, 2 * a2a_size - 1 - slots)
    src_half = (slots >= a2a_size).to(src_rank.dtype)
    # Start row of (sequence i, slot k) in the target and in the rank-major source.
    dst = (a2a_size * off_local[:-1])[:, None] + slots[None, :] * c[:, None]
    src = src_rank[None, :] * num_tokens + off_local[:-1][:, None] + src_half[None, :] * c[:, None]
    reps = c.to(torch.int64).repeat_interleave(2 * a2a_size)
    pos = torch.arange(a2a_size * num_tokens, device=device)
    shift = torch.repeat_interleave((src - dst).reshape(-1), reps, output_size=pos.numel())
    perm = pos + shift
    return perm, torch.argsort(perm)


# Backend shims. fwd(q, k, v, step, scale) -> (out bf16 [n, H, dv], lse fp32 [n, H], rng_state);
# bwd(q, k, v, out, dout, lse, rng, step, scale, deterministic) -> (dq, dk, dv).


def _lengths(cu):
    return cu[1:] - cu[:-1]


def _mask(step):
    return "padding_causal" if step.causal else "padding"


def _cudnn_fwd(q, k, v, step, scale):
    out, aux = fused_attn_fwd(
        True,
        step.max_q,
        step.max_kv,
        step.cu_q,
        step.cu_kv,
        q,
        k,
        v,
        q.dtype,
        FusedAttnBackend["F16_arbitrary_seqlen"],
        attn_scale=scale,
        dropout=0.0,
        qkv_layout="thd_thd_thd",
        o_format="thd",
        attn_mask_type=_mask(step),
        cu_seqlens_q_padded=step.off_q,
        cu_seqlens_kv_padded=step.off_kv,
    )
    lse = aux[0]
    if lse.dim() != 3:
        raise RuntimeError(
            "mla_latent_cp needs cuDNN packed THD softmax stats of shape [t, h, 1] "
            f"(cuDNN >= 9.6, not sm120); got {tuple(lse.shape)}."
        )
    return out, lse.squeeze(-1), aux[1]


def _cudnn_bwd(q, k, v, out, dout, lse, rng, step, scale, det):
    dq, dk, dv, *_ = fused_attn_bwd(
        step.max_q,
        step.max_kv,
        step.cu_q,
        step.cu_kv,
        q,
        k,
        v,
        out,
        dout,
        q.dtype,
        [lse.unsqueeze(-1).contiguous(), rng],
        FusedAttnBackend["F16_arbitrary_seqlen"],
        cu_seqlens_q_padded=step.off_q,
        cu_seqlens_kv_padded=step.off_kv,
        attn_scale=scale,
        dropout=0.0,
        qkv_layout="thd_thd_thd",
        o_format="thd",
        do_format="thd",
        dqkv_layout="thd_thd_thd",
        attn_mask_type=_mask(step),
        deterministic=det,
    )
    return dq, dk, dv


def _fa4():
    global _FA4
    if _FA4 is None:
        try:
            from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

            fa4_version = version("flash-attn-4")
        except (ImportError, PackageNotFoundError) as e:
            raise ImportError(
                "attention_backend=flash with mla_latent_cp needs the flash_attn_4 (CuTe) package."
            ) from e
        if fa4_version != _FA4_VERSION:
            raise RuntimeError(
                "mla_latent_cp drives FlashAttention-4 through its split forward/backward entry "
                f"points, validated for flash_attn_4=={_FA4_VERSION}; found {fa4_version}."
            )
        _FA4 = (_flash_attn_fwd, _flash_attn_bwd)
    return _FA4


def _fa4_kwargs(step, scale):
    return dict(
        softmax_scale=scale,
        causal=step.causal,
        cu_seqlens_q=step.off_q,
        cu_seqlens_k=step.off_kv,
        seqused_q=_lengths(step.cu_q),
        seqused_k=_lengths(step.cu_kv),
        max_seqlen_q=step.max_q,
        max_seqlen_k=step.max_kv,
    )


def _fa4_fwd(q, k, v, step, scale):
    # FA4 leaves rows it does not visit (padding) uninitialised: hand it zeroed buffers.
    out = torch.zeros(q.shape[0], q.shape[1], v.shape[-1], dtype=q.dtype, device=q.device)
    lse = torch.zeros(q.shape[1], q.shape[0], dtype=torch.float32, device=q.device)
    _fa4()[0](q, k, v, return_lse=True, out=out, lse=lse, **_fa4_kwargs(step, scale))
    return out, lse.transpose(0, 1).contiguous(), None


def _fa4_bwd(q, k, v, out, dout, lse, rng, step, scale, det):
    dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    lse = lse.transpose(0, 1).contiguous()
    kwargs = _fa4_kwargs(step, scale)
    _fa4()[1](q, k, v, out, dout, lse, dq=dq, dk=dk, dv=dv, deterministic=det, **kwargs)
    return dq, dk, dv


@jit_fuser
def _merge(out, lse, out_i, lse_i):
    """Online-softmax merge in place: fold ``(out_i, lse_i)`` into the fp32 accumulators."""
    lse_new = torch.logaddexp(lse, lse_i)
    out.mul_(torch.exp(lse - lse_new).unsqueeze(-1))
    out.addcmul_(out_i, torch.exp(lse_i - lse_new).unsqueeze(-1))
    lse.copy_(lse_new)


class _Ring(NamedTuple):
    group: dist.ProcessGroup
    rank: int
    size: int
    nxt: int  # global rank of the next peer
    prv: int  # global rank of the previous peer


def _p2p(ring, sends, recvs, forward):
    """Post one batched send/recv along the ring (forward: to r+1 from r-1; backward: reversed)."""
    dst, src = (ring.nxt, ring.prv) if forward else (ring.prv, ring.nxt)
    send_ops = [dist.P2POp(dist.isend, t, dst, ring.group) for t in sends]
    recv_ops = [dist.P2POp(dist.irecv, t, src, ring.group) for t in recvs]
    return dist.batch_isend_irecv(
        send_ops + recv_ops if ring.rank % 2 == 0 else recv_ops + send_ops
    )


_SIDE_STREAM = None


def _streams(two_stream):
    """``[current, side]`` when ``two_stream`` (opt-in, ``config.mla_latent_cp_two_stream``):
    alternating ring steps run on the two streams (TE's ``cp_stream`` pattern) so adjacent
    attention kernels overlap; otherwise ``[current, current]``, and the ring's events and
    ``record_stream`` are no-ops. The side stream is private to this module; every entry waits on
    the current stream and every exit joins it back, so tensors allocated on either stream may be
    read on the other without ``record_stream`` as long as they are freed only after an exit
    (``q``, ``out``, ``lse``, ``rng_states``, the saved latent)."""
    global _SIDE_STREAM
    current = torch.cuda.current_stream()
    if not two_stream:
        return [current, current]
    if _SIDE_STREAM is None:
        _SIDE_STREAM = torch.cuda.Stream()
    _SIDE_STREAM.wait_stream(current)
    return [current, _SIDE_STREAM]


def _te_recompute_ctx(fp8, recompute_phase):
    return activation_recompute_forward(True, recompute_phase) if fp8 else nullcontext()


@contextmanager
def _without_tp_comm(linear):
    """Run one call of a TE column-parallel linear on an already gathered input.

    TE reads ``parallel_mode`` per call; ``None`` skips the sequence-parallel input all-gather and
    the gradient reduce-scatter while the GEMM, quantization and fused weight gradient are
    unchanged. The ring gathers the latent once and reduce-scatters its gradient once instead.
    Every call of this module's ``linear_kv_up_proj`` goes through here (TE's
    ``input_quantizer.optimize_for_gemm`` stays set).
    """
    saved = linear.parallel_mode
    linear.parallel_mode = None
    try:
        yield
    finally:
        linear.parallel_mode = saved


class _Slab:
    """Rows ``[r0, r1)`` of a TE linear's weight as a leaf Parameter that the module reads as its
    weight for one call (TE takes its shapes from it), so it projects only those output rows; it
    carries the full parameter's ``main_grad`` rows and fused-wgrad flag, so TE accumulates where
    the full parameter would. A view in bf16 and TE's zero-copy ``split`` of the quantized data
    and scales for an MXFP8 parameter (padded scales are copied), built on first use by a forward
    pre-hook registered per call, i.e. after DDP's hook has waited for the weight's all-gather.
    """

    def __init__(self, r0, r1):
        self.r0, self.r1, self.param = r0, r1, None

    def __call__(self, linear, _):
        if self.param is None:
            weight, n = linear.weight, self.r1 - self.r0
            self.param = torch.nn.Parameter(
                torch.split(weight.detach(), n)[self.r0 // n], weight.requires_grad
            )
            if hasattr(weight, "main_grad"):
                self.param.main_grad = weight.main_grad[self.r0 : self.r1]
            if hasattr(weight, "grad_added_to_main_grad"):
                self.param.grad_added_to_main_grad = False
        linear._parameters["weight"] = self.param


@contextmanager
def _weight_slab(linear, slab):
    """Let one TE call read ``slab`` as the module's weight (swapped in by the slab's hook)."""
    if slab is None:
        yield
        return
    full, handle = linear.weight, linear.register_forward_pre_hook(slab)
    try:
        yield
    finally:
        handle.remove()
        linear._parameters["weight"] = full


class _BuildKV(torch.autograd.Function):
    """``K = [kv[..., :qk] | k_pos]`` (``k_pos`` broadcast over heads) and ``V = kv[..., qk:]``.

    Strided copies in both directions: ``torch.cat`` copies the stride-0 operand element by
    element, the backward of plain ``copy_`` into slices clones ``dK`` twice, and ``split``'s
    backward concatenates the strided ``dK`` slice with ``dV`` on the unvectorised path.
    """

    @staticmethod
    def forward(ctx, kv, k_pos, qk):
        ctx.qk, ctx.width = qk, kv.shape[-1]
        key = kv.new_empty(*kv.shape[:-1], qk + k_pos.shape[-1])
        key[..., :qk].copy_(kv[..., :qk])
        key[..., qk:].copy_(k_pos.unsqueeze(1).expand(-1, kv.shape[1], -1))
        return key, kv[..., qk:].contiguous()

    @staticmethod
    def backward(ctx, dk, dv):
        qk = ctx.qk
        dkv = dk.new_empty(*dk.shape[:-1], ctx.width)
        dkv[..., :qk].copy_(dk[..., :qk])
        dkv[..., qk:].copy_(dv)
        return dkv, dk[..., qk:].sum(1), None


class _LatentCPRingAttention(torch.autograd.Function):
    """Latent-KV ring attention over the zigzag schedule.

    Saves ``(q, latent received last, out, lse)``; backward recomputes the KV up-projection of
    every owner from the re-circulated latents and calls the attention backward with the final
    merged ``out``/``lse`` (the kernel then forms the global softmax probabilities of its block, so
    no per-step state and no correction are needed). The up-projection parameters are inputs of
    this function and receive their gradient once, from ``backward``.
    """

    @staticmethod
    def forward(ctx, q, z, module, plan, ring, attn_fwd, attn_bwd, scale, *up_params):
        fp8 = FP8GlobalStateManager.is_fp8_enabled()
        slab = module._slab  # this rank's rows of the up-projection weight (a2a modes), or None
        rows, valid = plan.rows, plan.valid
        q_u = q.index_select(0, rows) if rows is not None else None
        pad = None if valid is None else (~valid).unsqueeze(-1)  # [T, 1] broadcast over heads
        pad_u = pad.index_select(0, rows) if (pad is not None and rows is not None) else None
        two_stream = module.config.mla_latent_cp_two_stream
        streams, merged, up_done = _streams(two_stream), torch.cuda.Event(), torch.cuda.Event()
        out = lse = out_u = lse_u = None
        rng_states, reqs = [], []
        for i, step in enumerate(plan.steps):
            stream, z_cur = streams[i % 2], z if i == 0 else z_nxt
            with torch.cuda.stream(stream):
                for req in reqs:
                    req.wait()  # this step's latent has arrived (ordered on this stream)
                if i < ring.size - 1:
                    z_nxt = torch.empty_like(
                        z
                    )  # on the posting stream: NCCL orders its write after it
                    reqs = _p2p(ring, [z_cur], [z_nxt], forward=True)
                z_cur.record_stream(stream)  # read here, freed to the other stream's pool later
                # TE calls are serialised across the streams: one cuBLAS workspace per device
                # (get_cublas_workspace is lru_cached) and the per-microbatch fp8 weight cache.
                stream.wait_event(up_done)
                with torch.no_grad(), _te_recompute_ctx(fp8, recompute_phase=False):
                    k, v = module._expand_kv(z_cur, slab)
                stream.record_event(up_done)
                out_i, lse_i, rng_i = attn_fwd(q_u if step.upper else q, k, v, step, scale)
                del k, v
                rng_states.append(rng_i)
                if pad is not None:
                    lse_i.masked_fill_(pad_u if step.upper else pad, 0.0)
                # The in-place merges are ordered across the two streams by one event.
                stream.wait_event(merged)
                if i == 0:
                    out, lse = (out_i.float(), lse_i.clone()) if ring.size > 1 else (out_i, lse_i)
                elif not step.upper:
                    _merge(out, lse, out_i, lse_i)
                else:
                    if out_u is None:
                        out_u, lse_u = out.index_select(0, rows), lse.index_select(0, rows)
                    _merge(out_u, lse_u, out_i, lse_i)
                stream.record_event(merged)
        torch.cuda.current_stream().wait_stream(streams[1])
        if out_u is not None:
            out.index_copy_(0, rows, out_u)
            lse.index_copy_(0, rows, lse_u)
        out = out.to(q.dtype)
        ctx.save_for_backward(q, z_cur, out, lse)
        ctx.module, ctx.plan, ctx.ring, ctx.attn_bwd = module, plan, ring, attn_bwd
        ctx.scale, ctx.rng_states, ctx.fp8, ctx.params = scale, rng_states, fp8, up_params
        # The slab is the recompute graph's leaf for the up-projection weight in backward, and
        # TE's fused wgrad reads its main_grad through a weakref: ctx keeps it alive.
        ctx.slab = slab
        ctx.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe() if fp8 else None
        return out

    @staticmethod
    def backward(ctx, dout):
        q, z_last, out, lse = ctx.saved_tensors
        module, plan, ring, scale = ctx.module, ctx.plan, ctx.ring, ctx.scale
        params, slab = ctx.params, ctx.slab
        weight = module.linear_kv_up_proj.weight
        leaves = [slab.param if (slab is not None and p is weight) else p for p in params]
        det = module.config.deterministic_mode
        dout = dout.contiguous()
        rows = plan.rows
        full = (q, out, dout, lse)
        dq = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
        if rows is not None:
            upper = tuple(t.index_select(0, rows) for t in full)
            dq_u = torch.zeros(upper[0].shape, dtype=torch.float32, device=q.device)
        zb = [torch.empty_like(z_last), torch.empty_like(z_last)]
        dzb = [torch.zeros_like(z_last, dtype=torch.float32) for _ in range(2)]
        grads_p = [None] * len(params)
        reqs = []
        last = ring.size - 1
        for i in range(ring.size):
            # Latents travel backwards around the ring; the dlatent of the owner processed at this
            # step arrives one hop behind its latent and is forwarded after this rank's addition.
            z_cur = z_last if i == 0 else zb[(i - 1) % 2]
            if last > 0:
                sends, recvs = [dzb[(i - 1) % 2]], [dzb[i % 2]]
                if i < last:
                    sends, recvs = [z_cur] + sends, [zb[i % 2]] + recvs
                reqs = _p2p(ring, sends, recvs, forward=False)
            step = plan.steps[last - i]
            with torch.enable_grad(), fp8_autocast(enabled=ctx.fp8, fp8_recipe=ctx.fp8_recipe):
                with _te_recompute_ctx(ctx.fp8, recompute_phase=True):
                    z_j = z_cur.detach().requires_grad_()
                    k, v = module._expand_kv(z_j, slab)
            q_s, out_s, dout_s, lse_s = upper if step.upper else full
            rng = ctx.rng_states[last - i]
            dq_i, dk, dv = ctx.attn_bwd(q_s, k, v, out_s, dout_s, lse_s, rng, step, scale, det)
            (dq_u if step.upper else dq).add_(dq_i)
            # No AccumulateGrad fires here; TE still computes the weight gradient (fused into
            # main_grad, or returned) because the parameters require grad.
            grads = torch.autograd.grad((k, v), (z_j, *leaves), (dk, dv), allow_unused=True)
            del k, v, dk, dv, dq_i
            for n, g in enumerate(grads[1:]):
                if g is None:
                    continue
                if getattr(leaves[n], "grad_added_to_main_grad", False):
                    grads_p[n] = g  # already in main_grad; keep TE's dummy as the old path does
                elif grads_p[n] is None:
                    grads_p[n] = g.float()
                else:
                    grads_p[n].add_(g)
            for req in reqs:
                req.wait()
            dzb[i % 2].add_(grads[0])
        if rows is not None:
            dq.index_add_(0, rows, dq_u)
        dparams = []
        for p, leaf, g in zip(params, leaves, grads_p):
            if g is None:  # keep the outer AccumulateGrad (and the DDP hook) firing
                g = torch.zeros_like(p)
            elif leaf is not p:  # the slab's gradient belongs to rows [r0, r1) of the full weight
                full = torch.zeros(p.shape, dtype=g.dtype, device=g.device)
                if getattr(leaf, "grad_added_to_main_grad", False):
                    p.grad_added_to_main_grad = True  # TE accumulated into main_grad[r0:r1]
                else:
                    full[slab.r0 : slab.r1] = g
                g = full
            dparams.append(g.to(p.dtype))
        dz = dzb[last % 2]
        return (dq.to(q.dtype), dz.to(z_last.dtype), *([None] * 6), *dparams)


class MLAWithLatentCP(MLASelfAttention):
    """MLA self-attention whose context parallelism exchanges latent KV (experimental).

    Only ``get_query_key_value_tensors`` (returns the latent instead of K/V) and
    ``_run_core_attention`` (the latent ring) differ from ``MLASelfAttention``; the inherited
    ``forward`` provides the layout conversion, offload groups, output gate and projection.
    Select it with ``config.mla_latent_cp=True`` (see ``apply_mla_latent_cp_spec``).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # cp_comm_type, attention_backend and the other config constraints are validated by
        # TransformerConfig.__post_init__ when mla_latent_cp is set.
        if not self.config.mla_latent_cp:
            raise ValueError("MLAWithLatentCP requires config.mla_latent_cp=True.")
        # MLASelfAttention exposes use_rope where it supports layers without RoPE (no_rope_freq);
        # otherwise every layer rotates the positional channels, done here before the ring.
        self._use_rope = getattr(self, "use_rope", True)
        self._rope = (None, 1.0)  # rotary embedding and mscale of the current call
        if not HAVE_TE:
            raise ImportError("mla_latent_cp requires Transformer Engine.")
        if not isinstance(self.linear_kv_up_proj, (Linear, LayerNormLinear)):
            raise NotImplementedError(
                "mla_latent_cp runs linear_kv_up_proj without per-step tensor-parallel "
                "communication, which needs a Transformer Engine linear."
            )
        if self.config.attention_backend == AttnBackend.flash:
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 10:
                raise ValueError("attention_backend=flash with mla_latent_cp needs SM100 or newer.")
            _fa4()
            self._attn_fwd, self._attn_bwd = _fa4_fwd, _fa4_bwd
        else:  # fused or auto
            if not _FUSED_ATTN_API_OK:
                raise ImportError(
                    "mla_latent_cp needs a Transformer Engine whose fused_attn_fwd/bwd accept the "
                    "THD layout keywords (o_format, do_format, dqkv_layout, deterministic)."
                )
            self._attn_fwd, self._attn_bwd = _cudnn_fwd, _cudnn_bwd
        # The layer passes cp_comm_type only when CP is on; the config holds it otherwise (a list
        # is per layer, as TransformerLayer indexes it).
        mode = kwargs.get("cp_comm_type")
        if mode is None:
            mode = self.config.cp_comm_type
            if isinstance(mode, list):
                mode = mode[self.layer_number - 1]
        self.cp_comm_type = mode or "p2p"
        self._a2a_group = None
        if self.cp_comm_type == "a2a":
            self._a2a_group = self.pg_collection.cp
        elif self.cp_comm_type == "a2a+p2p":
            # hcp = [a2a group of A consecutive CP ranks, p2p group of stride A] (cp rank = p*A + a).
            cp, hcp = self.pg_collection.cp, getattr(self.pg_collection, "hcp", None)
            ok = isinstance(hcp, (list, tuple)) and len(hcp) == 2
            if ok:
                ranks, (a, p, A) = dist.get_process_group_ranks(cp), (
                    hcp[0].rank(),
                    hcp[1].rank(),
                    hcp[0].size(),
                )
                ok = (
                    A * hcp[1].size() == cp.size()
                    and dist.get_process_group_ranks(hcp[0]) == ranks[p * A : (p + 1) * A]
                    and dist.get_process_group_ranks(hcp[1]) == ranks[a::A]
                )
            if not ok:
                raise ValueError(
                    "mla_latent_cp with cp_comm_type='a2a+p2p' needs pg_collection.hcp = [a2a group "
                    "of consecutive CP ranks, p2p group] covering the CP group."
                )
            self._a2a_group = hcp[0]
        if self._a2a_group is not None and self._a2a_group.size() == 1:
            self._a2a_group = None  # degenerate a2a: plain ring
        self._slab = None  # per-call weight row slab of the a2a modes (set by _run_core_attention)

    def _expand_kv(self, latent, slab=None):
        """Up-project one gathered latent to K [T, Ha, qk] and V [T, Ha, v].

        The inherited no-RoPE path; with ``slab`` (a2a modes) the TE module projects only this
        rank's heads through the weight row slab, otherwise all local heads.
        """
        cfg, linear = self.config, self.linear_kv_up_proj
        kv_compressed, k_pos_emb = torch.split(
            latent, [cfg.kv_lora_rank, cfg.qk_pos_emb_head_dim], dim=-1
        )
        with _without_tp_comm(linear), _weight_slab(linear, slab):
            kv, _ = linear(apply_module(self.kv_layernorm)(kv_compressed))
        width = cfg.qk_head_dim + cfg.v_head_dim
        return _BuildKV.apply(kv.view(-1, kv.shape[-1] // width, width), k_pos_emb, cfg.qk_head_dim)

    def _rotate(self, t, emb, mscale, psp, which):
        """RoPE on the positional channels ``t [T, h, pos]`` of this CP rank's tokens, as the
        inherited path applies it (TE's THD kernel takes the zigzag positions from ``cu_seqlens``;
        ``which`` selects the query's or the keys' offsets of ``psp``)."""
        padded = getattr(psp, f"cu_seqlens_{which}_padded")
        cu_seqlens = padded if padded is not None else getattr(psp, f"cu_seqlens_{which}")
        if self.config.context_parallel_size == 1:
            emb = emb[: t.shape[0]]  # the inherited path's rule: a full table only with CP
        return apply_rotary_pos_emb(
            t,
            emb,
            config=self.config,
            cu_seqlens=cu_seqlens,
            mscale=mscale,
            cp_group=self.pg_collection.cp,
            mla_rotary_interleaved=True,
            max_seqlen=getattr(psp, f"max_seqlen_{which}"),
        )

    def get_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
        inference_context=None,
        *,
        inference_params=None,
    ):
        """Project ``hidden_states`` to the query and the latent-KV ring payload.

        Returns ``(query, latent, latent, q_compressed, latent)``: the latent fills both the key
        and the value slot so the inherited ``forward`` (``.contiguous()`` calls, offload release
        list) runs unchanged; ``_run_core_attention`` reads the key slot.
        """
        if inference_context is not None or inference_params is not None:
            raise NotImplementedError("mla_latent_cp supports training self-attention only.")
        if packed_seq_params is None or packed_seq_params.qkv_format != "thd":
            raise ValueError("mla_latent_cp requires THD packed input (qkv_format == 'thd').")
        q_compressed, latent = self._qkv_down_projection(hidden_states)
        if latent.size(-1) != self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim:
            raise ValueError(
                "mla_latent_cp requires the duplicated (TELinear) kv down projection; got a "
                f"tensor-parallel sharded latent of width {latent.size(-1)}."
            )
        latent = latent.squeeze(1)
        q_compressed = q_compressed.squeeze(1)
        if self.config.q_lora_rank is None:
            q_up_proj = self.linear_q_proj
        else:
            q_up_proj = self.linear_q_up_proj
            q_compressed = apply_module(self.q_layernorm)(q_compressed)
        # The rotary embedding of this call, as the inherited path builds it: the query's
        # positional channels are rotated here, the latent's once in _run_core_attention.
        emb, mscale = None, 1.0
        if self._use_rope:
            seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                None, None, hidden_states, self.config, packed_seq_params
            )
            emb = self.rotary_pos_emb(seq_len, packed_seq=True)
            emb, mscale = emb if isinstance(emb, tuple) else (emb, mscale)
        self._rope = (emb, mscale)
        heads, qk = self.num_attention_heads_per_partition, self.config.qk_head_dim

        def q_up(x, emb):  # emb is an argument so that the recompute (backward) receives it
            q, _ = q_up_proj(x)
            q = q.view(-1, heads, self.q_head_dim)
            if emb is None:
                return q
            q_pos = self._rotate(q[..., qk:], emb, mscale, packed_seq_params, "q")
            return torch.cat([q[..., :qk], q_pos], dim=-1)

        if self.recompute_up_proj:
            self.qkv_up_checkpoint = tensor_parallel.CheckpointWithoutOutput(
                fp8=self.config.fp8 or self.config.fp4
            )
            query = self.qkv_up_checkpoint.checkpoint(q_up, q_compressed, emb)
        else:
            query = q_up(q_compressed, emb)
        return query, latent, latent, q_compressed, latent

    def _run_core_attention(
        self, query, key, value, attention_mask, packed_seq_params=None, attn_mask_type=None, **_
    ):
        """Run the latent-KV ring; ``key`` is the latent payload."""
        if attention_mask is not None:
            raise ValueError("mla_latent_cp takes no explicit attention_mask (THD causal only).")
        mask_type = attn_mask_type or self.attn_mask_type
        if mask_type not in (AttnMaskType.causal, AttnMaskType.padding_causal):
            raise ValueError(f"mla_latent_cp supports causal attention only, got {mask_type}.")
        # The inherited forward already points pg_collection.cp at the dynamic-CP group.
        cp_group, a2a, tokens = self.pg_collection.cp, self._a2a_group, query.shape[0]
        heads, A = self.num_attention_heads_per_partition, 1 if a2a is None else a2a.size()
        if A > 1 and packed_seq_params.local_cp_size is not None:
            raise NotImplementedError(
                "mla_latent_cp a2a modes do not support dynamic context parallelism."
            )
        # The ring runs over the p2p group on blocks of A ranks' tokens and H/A heads.
        ring_size, a = cp_group.size() // A, 0 if A == 1 else a2a.rank()
        if ring_size > 1:
            group = cp_group if A == 1 else self.pg_collection.hcp[1]
            rank, peers = group.rank(), dist.get_process_group_ranks(group)
            ring = _Ring(
                group, rank, ring_size, peers[(rank + 1) % ring_size], peers[(rank - 1) % ring_size]
            )
        else:
            ring = _Ring(None, 0, 1, -1, -1)
        plan = _build_step_plan(packed_seq_params, ring_size, ring.rank, A * tokens)
        if get_pg_size(self.tp_group) > 1 and self.config.sequence_parallel:
            # One gather of the latent (and one reduce-scatter of its gradient, by autograd)
            # instead of a gather and a reduce-scatter inside every ring step.
            key = gather_from_sequence_parallel_region(key, group=self.tp_group)
        emb, mscale = self._rope
        if emb is not None:  # rotate the latent's positional channels once, before the ring
            kv_c, k_pos = torch.split(
                key, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
            )
            k_pos = self._rotate(k_pos.unsqueeze(1), emb, mscale, packed_seq_params, "kv")
            key = torch.cat([kv_c, k_pos.squeeze(1)], dim=-1)
        if A > 1:
            # Ulysses stage: gather the A latents (backward reduce-scatter = the head sum), exchange
            # query tokens for heads; both arrive rank-major and are permuted into the sequence
            # order of the p2p zigzag (natural order for pure a2a).
            perm, inv = _a2a_permutation(packed_seq_params, cp_group.size(), A, tokens)
            key = _Permute.apply(gather_from_sequence_parallel_region(key, group=a2a), perm, inv)
            query = all_to_all_sp2hp(query.reshape(tokens, -1), group=a2a)
            query = _Permute.apply(query.view(A * tokens, heads // A, -1), perm, inv)
        # Everything the per-step KV expansion differentiates besides the latent itself.
        up_params = [
            p
            for m in (self.kv_layernorm, self.linear_kv_up_proj)
            for p in m.parameters()
            if p.requires_grad
        ]
        # a2a modes: project only this rank's heads (their rows of the up-projection weight).
        rows = heads // A * (self.config.qk_head_dim + self.config.v_head_dim)
        self._slab = _Slab(a * rows, (a + 1) * rows) if A > 1 else None
        out = _LatentCPRingAttention.apply(
            query,
            key,
            self,
            plan,
            ring,
            self._attn_fwd,
            self._attn_bwd,
            self.softmax_scale,
            *up_params,
        )
        self._slab = None
        if A > 1:
            out = all_to_all_hp2sp(
                _Permute.apply(out, inv, perm).reshape(A * tokens, -1), group=a2a
            )
            out = out.view(tokens, heads, -1)
        return out
