# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Portions of this file follow the MIT-licensed FLA v0.5.1 KDA autograd
# contract (fla/ops/kda/chunk.py, chunk_intra.py and chunk_bwd.py; the
# autograd module and every primitive signature are unchanged in v0.5.2).
# The CP boundary pre-processing is redirected to the in-tree CuTe DSL kernels.

"""Autograd orchestration for KDA: FLA Triton kernels + in-tree CuTe DSL kernels.

Kimi Delta Attention (KDA) is the channel-wise gated delta rule. Its forward
chunked kernels live in FLA and are reused verbatim here. Two parts are
redirected in-tree:

* the two context-parallel boundary pre-processing steps, which go through
  ``cp_preprocess`` in the CuTe kernels' ``gk`` gate mode (per-K-row decay,
  pre-gated ``kg``/``qg`` operands, no per-token row gating) and fall back to
  FLA's Triton producers whenever the CuTe path refuses the call;
* on SM100 with ``K == V == 128``, the backward after the state recompute: one
  fused CuTe DSL kernel (``fused_kda_bwd_cute``) replaces FLA's dAv, dhu, WY/dqkg,
  intra and reverse dg cumsum launches. It reads the chunk states ``h`` and the
  terminal state gradient (from the CP pre-process under context parallelism).

Without a CP context and without the fused backward the public entry calls FLA.
"""

from __future__ import annotations

import inspect
import os
from typing import Any

import torch
from fla.ops.common.chunk_delta_h import (
    chunk_gated_delta_rule_bwd_dhu,
    chunk_gated_delta_rule_fwd_h,
)
from fla.ops.cp.chunk_delta_h import compress_h0, expand_h0
from fla.ops.gla.chunk import chunk_gla_fwd_o_gk
from fla.ops.kda import chunk_kda as fla_chunk_kda
from fla.ops.kda.chunk_bwd import chunk_kda_bwd_dAv, chunk_kda_bwd_wy_dqkg_fused
from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra, chunk_kda_fwd_intra
from fla.ops.kda.gate import kda_gate_bwd, kda_gate_chunk_cumsum
from fla.ops.kda.wy_fast import recompute_w_u_fwd
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.index import prepare_chunk_indices
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

from .implementation import (
    _CHUNK_SIZE,
    _backend_mode,
    _cp_backward_preprocess,
    _cp_forward_preprocess,
    _has_module,
    _validate_validated_chunk_indices,
)
from .kernels.fused_gdr_cp_cute import (
    get_cutedsl_cp_noop_count,
    get_cutedsl_fallback_reasons,
    get_cutedsl_fused_bwd_launch_count,
    get_cutedsl_fused_launch_count,
)

# Keyword arguments the FLA entry point reads from ``**kwargs`` and the adapter
# therefore understands; anything else makes the adapter defer to FLA.
_KNOWN_KWARGS = frozenset({"A_log", "dt_bias", "chunk_size", "transpose_state_layout"})


def _check_fla_kda_contract() -> None:
    """Refuse FLA releases whose KDA primitives predate the 0.5.1 recompute contract.

    The adapter copies the ``ChunkKDAFunction`` forward/backward bodies of FLA 0.5.1
    (the release Megatron pins; byte-identical in 0.5.2). It relies on
    ``chunk_kda_fwd_intra`` taking ``disable_recompute`` (so ``qg`` is only
    materialised when it is saved), on ``recompute_w_u_fwd`` taking ``q`` (so the
    backward can materialise ``qg`` for the CP pre-process), and on the gated delta
    ``fwd_h`` primitive taking ``gk`` plus ``cu_seqlens_cpu``.
    """

    def parameters(fn):
        try:
            return inspect.signature(fn).parameters
        except (TypeError, ValueError):
            # Wrapped without an introspectable signature; do not refuse a supported
            # FLA build on that account.
            return None

    intra = parameters(chunk_kda_fwd_intra)
    wy = parameters(recompute_w_u_fwd)
    fwd_h = parameters(chunk_gated_delta_rule_fwd_h)
    if (
        (intra is not None and "disable_recompute" not in intra)
        or (wy is not None and "q" not in wy)
        or (fwd_h is not None and ("gk" not in fwd_h or "cu_seqlens_cpu" not in fwd_h))
    ):
        raise ImportError(
            "The internal KDA adapter requires flash-linear-attention >= 0.5.1 (KDA chunk "
            "primitives with disable_recompute / qg materialisation); the installed release "
            "exposes an older KDA kernel signature."
        )


_check_fla_kda_contract()


def _cutedsl_progress(*, forward: bool) -> int:
    """Count CP pre-process calls the in-tree backend has answered so far."""
    launches = get_cutedsl_fused_launch_count() if forward else get_cutedsl_fused_bwd_launch_count()
    return launches + get_cutedsl_cp_noop_count()


def _require_cutedsl_progress(before: int, *, forward: bool) -> None:
    """Raise when a strict (``cute`` mode) CP pre-process silently fell back to FLA."""
    if _cutedsl_progress(forward=forward) == before:
        direction = "forward" if forward else "backward"
        raise RuntimeError(
            f"Internal KDA CP {direction} preprocess fell back to FLA Triton: "
            f"{get_cutedsl_fallback_reasons()}"
        )


def _kda_internal_support_reason(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    use_beta_sigmoid_in_kernel: bool,
    allow_neg_eigval: bool,
    safe_gate: bool,
    lower_bound: float | None,
    return_intermediate_states: bool,
    state_v_first: bool,
    kwargs: dict[str, Any],
    context_parallel: bool = True,
) -> str | None:
    """Return why the internal adapter cannot take this call, or ``None``.

    Only adapter-level restrictions live here. Device, toolchain, head-count, dtype
    and world-size requirements of the CuTe kernels are decided inside the CP
    dispatcher; when it refuses, the shims run FLA's Triton pre-process on the same
    operands, so the adapter stays numerically equal to FLA either way.
    """
    if initial_state is not None or output_final_state:
        return "context parallel KDA forbids initial or final states"
    if use_qk_l2norm_in_kernel:
        return "in-kernel QK L2 normalization is not supported"
    if use_beta_sigmoid_in_kernel or allow_neg_eigval:
        return "in-kernel beta transformation is not supported"
    if return_intermediate_states:
        return "return_intermediate_states is not supported"
    if state_v_first or kwargs.get("transpose_state_layout"):
        return "V-first state layout is not supported"
    if kwargs.get("chunk_size", _CHUNK_SIZE) != _CHUNK_SIZE:
        return f"chunk_size must be {_CHUNK_SIZE}"
    if use_gate_in_kernel and "A_log" not in kwargs:
        return "A_log is required when use_gate_in_kernel=True"
    extra_kwargs = sorted(set(kwargs) - _KNOWN_KWARGS)
    if extra_kwargs:
        return f"extra options are not supported: {extra_kwargs}"
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return "q, k, and v must use BTHD layout"
    if q.shape != k.shape:
        return "q and k shapes differ"
    if context_parallel and q.shape[0] != 1:
        return "context parallel KDA requires batch size 1"
    B, T, H, K = q.shape
    HV = v.shape[2]
    if HV != H:
        return "grouped value attention is not supported by the internal KDA adapter"
    if tuple(g.shape) != (B, T, HV, K):
        return "g must have shape [B, T, HV, K]"
    if tuple(beta.shape) != (B, T, HV):
        return "beta must have shape [B, T, HV]"
    if safe_gate and use_gate_in_kernel and (lower_bound is None or not (-5 <= lower_bound < 0)):
        return "lower_bound must be in [-5, 0) when safe_gate and use_gate_in_kernel are set"
    return None


_FUSED_BWD_ENV = "MCORE_KDA_FUSED_BWD"
_FUSED_BWD_HEAD_DIM = 128
_fused_kda_bwd_launches = 0


def get_fused_kda_bwd_launch_count() -> int:
    """Number of fused KDA backward launches in this process."""
    return _fused_kda_bwd_launches


def _fused_kda_bwd_support_reason(
    *, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor
) -> str | None:
    """Return why the fused CuTe DSL KDA backward cannot run, or ``None``.

    Decided in the forward (from the inputs alone) so the backward never changes
    its mind. ``MCORE_KDA_FUSED_BWD=0`` keeps FLA's backward kernels.
    """
    if os.environ.get(_FUSED_BWD_ENV, "1") == "0":
        return f"{_FUSED_BWD_ENV}=0"
    if not _has_module("cutlass") or not _has_module("cuda.bindings.driver"):
        return "CuTe DSL runtime is not installed"
    if not q.is_cuda or torch.cuda.get_device_capability(q.device)[0] != 10:
        return "device is not SM100"
    if not torch.version.cuda or int(torch.version.cuda.split(".", 1)[0]) < 13:
        return "CUDA 13 or newer is required"
    if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
        return "q, k and v must be bf16"
    if q.ndim != 4 or q.shape[-1] != _FUSED_BWD_HEAD_DIM:
        return f"head dimension must be {_FUSED_BWD_HEAD_DIM}"
    if k.shape != q.shape or v.shape != q.shape:
        return "q, k and v shapes differ"
    if tuple(beta.shape) != tuple(q.shape[:3]):
        return "beta must have shape [B, T, H]"
    return None


def _call_fused_kda_bwd(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    scale: float,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    """Run the fused KDA backward on FLA's residuals; returns ``dq, dk, dv, dg, dbeta``.

    ``g`` is FLA's chunk-local log2 gate cumsum and ``h`` the chunk-start states.
    With few (sequence, head) pairs the dS recurrence runs first (``kda_dhu``)
    and writes every chunk's dS and dv_new, and the gradient kernel then
    processes all chunks in parallel; otherwise one fused kernel runs the
    recurrence and the gradients per (sequence, head). The returned ``dg``
    already includes FLA's chunk-local reverse cumsum.
    """
    global _fused_kda_bwd_launches
    from .kernels.fused_kda_bwd_cute.launcher import (
        chunk_offsets_for,
        fused_kda_bwd,
        kda_dhu,
        prefer_tape,
    )

    batch, seq_len, heads, dim = q.shape
    tokens = batch * seq_len
    if cu_seqlens is None:
        cu_seqlens = torch.arange(batch + 1, dtype=torch.int32, device=q.device) * seq_len
    cu_seqlens = cu_seqlens.to(torch.int32)
    num_sequences = cu_seqlens.numel() - 1
    if dht is None:
        dht = torch.zeros(num_sequences, heads, dim, dim, dtype=torch.float32, device=q.device)

    def tokens_of(x, dtype, width=dim):
        return x.detach().reshape(1, tokens, heads, width).to(dtype).contiguous()

    q, k, v, g, Aqk, Akk, do = (
        tokens_of(q, torch.bfloat16),
        tokens_of(k, torch.bfloat16),
        tokens_of(v, torch.bfloat16),
        tokens_of(g, torch.float32),
        tokens_of(Aqk, torch.bfloat16, _CHUNK_SIZE),
        tokens_of(Akk, torch.bfloat16, _CHUNK_SIZE),
        tokens_of(do, torch.bfloat16),
    )
    beta = beta.detach().reshape(1, tokens, heads).to(torch.float32).contiguous()
    h = h.detach().reshape(1, -1, heads, dim, dim).to(torch.bfloat16).contiguous()
    dht = dht.detach().reshape(num_sequences, heads, dim, dim).to(torch.float32).contiguous()
    chunk_offsets = chunk_offsets_for(cu_seqlens)
    tape = None
    if prefer_tape(num_sequences, heads, q.device):
        dh, _dh0, dv_new = kda_dhu(
            q,
            k,
            g,
            beta,
            Aqk,
            Akk,
            do,
            dht,
            cu_seqlens,
            scale,
            chunk_offsets=chunk_offsets,
            num_chunks=h.shape[1],
        )
        tape = (dh, dv_new)
    dq, dk, dv, dg, db, _ = fused_kda_bwd(
        q,
        k,
        v,
        g,
        beta,
        Aqk,
        Akk,
        h,
        do,
        dht,
        cu_seqlens,
        scale,
        chunk_offsets=chunk_offsets,
        tape=tape,
    )
    _fused_kda_bwd_launches += 1
    shape = (batch, seq_len, heads, dim)
    return (
        dq.reshape(shape),
        dk.reshape(shape),
        dv.reshape(shape),
        dg.reshape(shape),
        db.reshape(shape[:3]),
    )


class InternalChunkKDAFunction(torch.autograd.Function):
    """FLA 0.5.1 ``ChunkKDAFunction`` with the CP pre-processing and, where supported,
    the backward redirected in-tree."""

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor | None,
        dt_bias: torch.Tensor | None,
        scale: float,
        safe_gate: bool,
        lower_bound: float | None,
        use_gate_in_kernel: bool,
        disable_recompute: bool,
        cu_seqlens: torch.LongTensor | None,
        cu_seqlens_cpu: torch.LongTensor | None,
        validated_chunk_indices: torch.LongTensor | None,
        cp_context: object | None,
        recompute_h: bool,
    ):
        strict = _backend_mode() == "cute"
        fused_bwd = _fused_kda_bwd_support_reason(q=q, k=k, v=v, beta=beta) is None
        # The fused backward reads h (and, under CP, v_new) from the forward: keeping
        # them skips the backward state recompute unless recompute_h is set.
        save_fused_bwd_state = fused_bwd and not disable_recompute and not recompute_h
        _validate_validated_chunk_indices(validated_chunk_indices, cu_seqlens)
        chunk_indices = validated_chunk_indices
        if chunk_indices is None and cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(
                cu_seqlens, _CHUNK_SIZE, cu_seqlens_cpu=cu_seqlens_cpu
            )

        # fla/ops/kda/chunk_fwd.py: gate activation + chunk-local log2 cumsum.
        g_org = g
        if use_gate_in_kernel:
            g = kda_gate_chunk_cumsum(
                g=g_org,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=RCP_LN2,
                chunk_size=_CHUNK_SIZE,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                lower_bound=lower_bound,
            )
        else:
            g = chunk_local_cumsum(
                g=g,
                scale=RCP_LN2,
                chunk_size=_CHUNK_SIZE,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )

        # qg is None unless disable_recompute.
        w, u, qg, kg, Aqk, Akk = chunk_kda_fwd_intra(
            q=q,
            k=k,
            v=v,
            gk=g,
            beta=beta,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=_CHUNK_SIZE,
            chunk_indices=chunk_indices,
            safe_gate=safe_gate,
            disable_recompute=disable_recompute,
        )

        initial_state = None
        if cp_context is not None:
            # The only redirected call: CuTe DSL gk-mode boundary state, FLA fallback.
            # ``kg`` is already gated (k * 2^(gk_last - gk)) and ``g`` is the fp32
            # per-K-row log2 cumsum, exactly what FLA's Triton producer consumes.
            before = _cutedsl_progress(forward=True)
            initial_state = _cp_forward_preprocess(
                k=kg, w=w, u=u, gk=g, cu_seqlens=cu_seqlens, initial_state=None, context=cp_context
            )
            if strict:
                _require_cutedsl_progress(before, forward=True)

        h, v_new, _final_state = chunk_gated_delta_rule_fwd_h(
            k=kg,
            w=w,
            u=u,
            gk=g,
            initial_state=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
            chunk_size=_CHUNK_SIZE,
            state_v_first=False,
        )

        if cp_context is not None:
            # Only the first local sequence can continue a cross-rank sequence; keep
            # just that slot for the backward (FLA does the same).
            initial_state = compress_h0(initial_state, context=cp_context)

        o = chunk_gla_fwd_o_gk(
            q=q,
            v=v_new,
            g=g,
            A=Aqk,
            h=h,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=_CHUNK_SIZE,
            chunk_indices=chunk_indices,
            state_v_first=False,
        )
        if save_fused_bwd_state:
            # Keep h (and v_new for the CP pre-process); the gate cumsum is cheap to
            # recompute and as large as h in fp32.
            w, u, qg, kg = None, None, None, None
            if cp_context is None:
                v_new = None
            if use_gate_in_kernel:
                g = None
        elif not disable_recompute:
            # Delete to save memory; the backward recomputes them.
            w, u, qg, kg, v_new, h = None, None, None, None, None, None
            if use_gate_in_kernel:
                g = None

        ctx.save_for_backward(
            q,
            k,
            v,
            g,
            g_org,
            beta,
            A_log,
            dt_bias,
            Aqk,
            Akk,
            w,
            u,
            qg,
            kg,
            v_new,
            h,
            initial_state,
            cu_seqlens,
            chunk_indices,
        )
        ctx.scale = scale
        ctx.safe_gate = safe_gate
        ctx.lower_bound = lower_bound
        ctx.use_gate_in_kernel = use_gate_in_kernel
        ctx.disable_recompute = disable_recompute
        ctx.cp_context = cp_context
        ctx.strict = strict
        ctx.fused_bwd = fused_bwd
        ctx.saved_fused_bwd_state = save_fused_bwd_state
        return o.type_as(q), None

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor | None):
        (
            q,
            k,
            v,
            g,
            g_org,
            beta,
            A_log,
            dt_bias,
            Aqk,
            Akk,
            w,
            u,
            qg,
            kg,
            v_new,
            h,
            initial_state,
            cu_seqlens,
            chunk_indices,
        ) = ctx.saved_tensors
        scale = ctx.scale
        cp_context = ctx.cp_context

        # fla/ops/kda/chunk_bwd.py: recompute the gate cumsum, WY tensors and states.
        # A forward that kept h for the fused backward skips the state recompute; the
        # CP pre-process still reads the WY tensors and the gated q / k.
        if not ctx.disable_recompute:
            recompute_state = not ctx.saved_fused_bwd_state
            if ctx.use_gate_in_kernel:
                g = kda_gate_chunk_cumsum(
                    g=g_org,
                    A_log=A_log,
                    dt_bias=dt_bias,
                    scale=RCP_LN2,
                    chunk_size=_CHUNK_SIZE,
                    cu_seqlens=cu_seqlens,
                    chunk_indices=chunk_indices,
                    lower_bound=ctx.lower_bound,
                )
            if recompute_state or cp_context is not None:
                # Passing q materialises qg = q * 2^gk, which the CP pre-process needs.
                w, u, qg, kg = recompute_w_u_fwd(
                    q=q,
                    k=k,
                    v=v,
                    beta=beta,
                    A=Akk,
                    gk=g,
                    cu_seqlens=cu_seqlens,
                    chunk_indices=chunk_indices,
                )
            if cp_context is not None:
                initial_state = expand_h0(initial_state, context=cp_context)
            if recompute_state:
                h, v_new, _final_state = chunk_gated_delta_rule_fwd_h(
                    k=kg,
                    w=w,
                    u=u,
                    gk=g,
                    initial_state=initial_state,
                    output_final_state=False,
                    cu_seqlens=cu_seqlens,
                    chunk_indices=chunk_indices,
                    chunk_size=_CHUNK_SIZE,
                    state_v_first=False,
                )
        elif cp_context is not None:
            initial_state = expand_h0(initial_state, context=cp_context)

        if ctx.fused_bwd:
            if cp_context is not None:
                # The CP terminal gradient reads FLA's local dv = Aqk^T do; the fused
                # kernel recomputes everything else from h and the returned dht.
                _, dv_local = chunk_kda_bwd_dAv(
                    q=q,
                    k=k,
                    v=v_new,
                    do=do,
                    A=Aqk,
                    scale=scale,
                    cu_seqlens=cu_seqlens,
                    chunk_size=_CHUNK_SIZE,
                    chunk_indices=chunk_indices,
                )
                before = _cutedsl_progress(forward=False)
                dht, _ = _cp_backward_preprocess(
                    q=qg,
                    k=kg,
                    w=w,
                    do=do,
                    dv=dv_local,
                    gk=g,
                    scale=scale,
                    cu_seqlens=cu_seqlens,
                    dht=dht,
                    initial_state=initial_state,
                    context=cp_context,
                )
                if ctx.strict:
                    _require_cutedsl_progress(before, forward=False)
            # h already carries the CP initial state (recomputed from it above).
            dq, dk, dv, dg, db = _call_fused_kda_bwd(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                Aqk=Aqk,
                Akk=Akk,
                h=h,
                do=do,
                dht=dht,
                scale=scale,
                cu_seqlens=cu_seqlens,
            )
        else:
            # dAqk = do @ v.T ; dv = A @ do (the local term the CP pre-process consumes)
            dAqk, dv = chunk_kda_bwd_dAv(
                q=q,
                k=k,
                v=v_new,
                do=do,
                A=Aqk,
                scale=scale,
                cu_seqlens=cu_seqlens,
                chunk_size=_CHUNK_SIZE,
                chunk_indices=chunk_indices,
            )

            if cp_context is not None:
                # The only redirected call: CuTe DSL gk-mode terminal gradient, FLA fallback.
                # The gk-mode kernels never gate q themselves, so ``qg`` must be passed.
                before = _cutedsl_progress(forward=False)
                dht, initial_state = _cp_backward_preprocess(
                    q=qg,
                    k=kg,
                    w=w,
                    do=do,
                    dv=dv,
                    gk=g,
                    scale=scale,
                    cu_seqlens=cu_seqlens,
                    dht=dht,
                    initial_state=initial_state,
                    context=cp_context,
                )
                if ctx.strict:
                    _require_cutedsl_progress(before, forward=False)

            dh, _dh0, dv = chunk_gated_delta_rule_bwd_dhu(
                q=qg,
                k=kg,
                w=w,
                gk=g,
                h0=initial_state,
                dht=dht,
                do=do,
                dv=dv,
                scale=scale,
                cu_seqlens=cu_seqlens,
                chunk_size=_CHUNK_SIZE,
                chunk_indices=chunk_indices,
                state_v_first=False,
            )

            dq, dk, dv, db, dg, dAkk = chunk_kda_bwd_wy_dqkg_fused(
                q=q,
                k=k,
                v=v,
                v_new=v_new,
                g=g,
                beta=beta,
                A=Akk,
                h=h,
                do=do,
                dh=dh,
                dv=dv,
                scale=scale,
                cu_seqlens=cu_seqlens,
                chunk_size=_CHUNK_SIZE,
                chunk_indices=chunk_indices,
                state_v_first=False,
            )

            dq, dk, db, dg = chunk_kda_bwd_intra(
                q=q,
                k=k,
                g=g,
                beta=beta,
                dAqk=dAqk,
                dAkk=dAkk,
                dq=dq,
                dk=dk,
                db=db,
                dg=dg,
                cu_seqlens=cu_seqlens,
                chunk_size=_CHUNK_SIZE,
                chunk_indices=chunk_indices,
                safe_gate=ctx.safe_gate,
            )

            # For GVA, reduce dq and dk from [B, T, HV, K] back to [B, T, H, K]. The
            # dispatcher only admits HV == H; kept for parity with FLA's chunk_kda_bwd.
            H, HV = q.shape[2], v.shape[2]
            if HV > H:
                G = HV // H
                dq = dq.view(*dq.shape[:2], H, G, dq.shape[-1]).sum(dim=3)
                dk = dk.view(*dk.shape[:2], H, G, dk.shape[-1]).sum(dim=3)

            dg = chunk_local_cumsum(
                dg,
                chunk_size=_CHUNK_SIZE,
                reverse=True,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )
        dA, dbias = None, None
        if ctx.use_gate_in_kernel:
            dg, dA, dbias = kda_gate_bwd(
                g=g_org, A_log=A_log, dt_bias=dt_bias, dyg=dg, lower_bound=ctx.lower_bound
            )

        # Seven gradients (q, k, v, g, beta, A_log, dt_bias) in FLA's order, then None
        # for scale, safe_gate, lower_bound, use_gate_in_kernel, disable_recompute,
        # cu_seqlens, cu_seqlens_cpu, validated_chunk_indices, cp_context and recompute_h.
        return (
            dq.to(q),
            dk.to(k),
            dv.to(v),
            dg.to(g_org),
            db.to(beta),
            dA,
            dbias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@torch.compiler.disable
def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    validated_chunk_indices: torch.LongTensor | None = None,
    cp_context: object | None = None,
    recompute_h: bool = False,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run FLA's KDA kernels with the CP pre-processing and the backward in-tree.

    When the fused backward runs, the forward keeps the chunk states ``h`` so the
    backward skips FLA's state recompute; ``recompute_h=True``
    (``gdn_gdr_recompute_h``) recomputes them instead to save activation memory.

    ``MCORE_GDN_INTERNAL_BACKEND=fla`` is FLA verbatim, and so is a call without a
    CP context that the fused backward cannot serve (see
    ``_fused_kda_bwd_support_reason``). ``auto`` uses the adapter whenever its
    restrictions allow and falls back to FLA otherwise; ``cute`` raises instead of
    falling back on CP calls, including when the CP dispatcher itself declines the
    in-tree kernels.
    """
    mode = _backend_mode()
    use_adapter = mode != "fla" and (
        cp_context is not None or _fused_kda_bwd_support_reason(q=q, k=k, v=v, beta=beta) is None
    )
    if use_adapter and cp_context is None:
        # Without CP only the fused backward is in-tree: fall back to FLA verbatim on
        # anything the adapter cannot take.
        use_adapter = (
            _kda_internal_support_reason(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                use_gate_in_kernel=use_gate_in_kernel,
                use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
                allow_neg_eigval=allow_neg_eigval,
                safe_gate=safe_gate,
                lower_bound=lower_bound,
                return_intermediate_states=return_intermediate_states,
                state_v_first=state_v_first,
                kwargs=kwargs,
                context_parallel=False,
            )
            is None
        )
    if not use_adapter:
        return fla_chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            allow_neg_eigval=allow_neg_eigval,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            disable_recompute=disable_recompute,
            return_intermediate_states=return_intermediate_states,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            cp_context=cp_context,
            **kwargs,
        )

    reason = _kda_internal_support_reason(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        return_intermediate_states=return_intermediate_states,
        state_v_first=state_v_first,
        kwargs=kwargs,
        context_parallel=cp_context is not None,
    )
    if reason is not None:
        if mode == "cute":
            raise RuntimeError(f"Internal KDA CP path is unavailable: {reason}")
        return fla_chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            allow_neg_eigval=allow_neg_eigval,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            disable_recompute=disable_recompute,
            return_intermediate_states=return_intermediate_states,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            cp_context=cp_context,
            **kwargs,
        )

    if cp_context is not None:
        # fla/ops/kda/chunk.py: the context's local offsets replace the caller's.
        cu_seqlens = getattr(cp_context, "cu_seqlens", None)
        if cu_seqlens is None:
            raise ValueError("cp_context.cu_seqlens is required for context parallel KDA")
        if getattr(cp_context, "cu_seqlens_cpu", None) is not None:
            cu_seqlens_cpu = cp_context.cu_seqlens_cpu

    A_log, dt_bias = None, None
    if use_gate_in_kernel:
        A_log, dt_bias = kwargs["A_log"], kwargs.get("dt_bias")

    if scale is None:
        scale = q.shape[-1] ** -0.5
    return InternalChunkKDAFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        scale,
        safe_gate,
        lower_bound,
        use_gate_in_kernel,
        disable_recompute,
        cu_seqlens,
        cu_seqlens_cpu,
        validated_chunk_indices,
        cp_context,
        recompute_h,
    )
