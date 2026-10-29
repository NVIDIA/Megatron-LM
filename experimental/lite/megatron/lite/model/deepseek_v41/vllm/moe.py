# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fused inference MegaMoE with an owned decoded-operand backward; the shared-first,
fixed-slot nonfused chain over routed Core A2A rows serves layers without shared experts."""
import torch

from megatron.core.tensor_parallel.mappings import all_to_all
from megatron.core.transformer.moe.moe_layer import MoELayer

from . import experts
from .linear import linear
from .routing import route


def combine(slots, indices, shared):
    result = torch.zeros_like(shared, dtype=torch.float32) + shared.float()
    for slot in range(indices.shape[1]):
        result = torch.where(
            (indices[:, slot] >= 0).unsqueeze(-1), result + slots[:, slot].float(), result
        )
    return result.bfloat16()


def dispatch_rows(x, scores, indices, encoded, local_experts, group):
    # Stable global expert order also makes contiguous expert ownership peer-major.
    valid = (indices.flatten() >= 0).nonzero().flatten()
    position = valid[torch.argsort(indices.flatten()[valid], stable=True)]
    expert = indices.flatten()[position]
    sends = torch.bincount(expert // local_experts, minlength=group.size())
    receives = all_to_all(group, sends).tolist()
    sends = sends.tolist()
    token = position // indices.shape[1]
    received_ids = all_to_all(group, expert, receives, sends)
    # Incoming chunks are source-rank-major. Stable expert sort reproduces the
    # old all-gather's expert/source/token/slot order, including padded gaps.
    order = torch.argsort(received_ids, stable=True)
    counts = torch.bincount(received_ids % local_experts, minlength=local_experts).tolist()
    live = all_to_all(group, x[token], receives, sends)[order]
    weights = all_to_all(group, scores.flatten()[position], receives, sends)[order]
    codes = all_to_all(group, encoded[0].view(torch.uint8)[token], receives, sends)[order]
    scale = all_to_all(group, encoded[1][token], receives, sends)[order]
    state = position, torch.argsort(order), sends, receives
    return live, weights, (codes.view(torch.float8_e4m3fn), scale), counts, state


def return_rows(rows, indices, state, group):
    position, inverse, sends, receives = state
    returned = all_to_all(group, rows[inverse], sends, receives)
    slots = returned.new_zeros((indices.numel(), rows.shape[-1]))
    return slots.index_copy(0, position, returned).reshape(*indices.shape, rows.shape[-1])


def actual_rows(x, scores, indices, masters, limit, group, has_shared=True):
    local = len(masters) // 2 - int(has_shared)
    codes, scale = experts.quantize_activation(x)
    shared = x * 0
    if has_shared:
        first = experts._Shared.apply(x, masters[-2], (codes, scale))
        hidden, encoded = experts.activation(first, x.new_ones(x.shape[0]), limit)
        shared = experts._Shared.apply(hidden, masters[-1], encoded)
    live, weights, encoded, splits, state = dispatch_rows(
        x, scores, indices, (codes, scale), local, group
    )
    rows = experts.expert_rows(
        live, masters[:local], masters[local : 2 * local], splits, weights, encoded, limit
    )
    return combine(return_rows(rows, indices, state, group), indices, shared)


def _mx(w):
    from megatron.lite.primitive.quantization.mxfp4 import dequantize_mxfp4, quantize_mxfp4

    return dequantize_mxfp4(*quantize_mxfp4(w.detach().bfloat16())).bfloat16()  # exact


def _fp8(w):
    from megatron.lite.primitive.quantization.block_fp8 import (
        dequantize_block_fp8,
        quantize_block_fp8,
    )

    codes, scale = quantize_block_fp8(w.detach().bfloat16(), (32, 32), scale_format="e8m0")
    return dequantize_block_fp8(codes, scale, (32, 32)).bfloat16()  # exact


def _requant_l2(h):
    """The kernel's L1-epilogue requant of the FP32 activation, decoded: 1x32 groups, UE8M0
    scale = smallest power of two with amax / scale <= 448 (DeepGEMM ``get_ue8m0_sf_exp``,
    exponent floor 2^-22), E4M3 round-to-nearest-even."""
    groups = h.detach().float().reshape(*h.shape[:-1], h.shape[-1] // 32, 32)
    bits = groups.abs().amax(-1, keepdim=True).view(torch.int32).to(torch.int64)
    exp = torch.clamp((bits + 0x7FFFFF - 0x600000) >> 23, min=113) - 8
    codes = (groups * torch.exp2((127 - exp).float())).to(torch.float8_e4m3fn)
    return (codes.float() * torch.exp2((exp - 127).float())).reshape(h.shape)


def _mlp_grads(a, score, grad, w1, w2, limit, need):
    """VJP of ``q(swiglu(bf16(a @ w1.T)) * score) @ w2.T`` over decoded operands, as the
    fused kernel computes it (GEMM1 accumulator rounded to BF16 before the clamp/SwiGLU,
    ``q`` the FP8 requant, identity STE). BF16 tensor-core GEMMs with FP32 accumulation.
    Returns FP32 (da, dscore, dw1, dw2), None where unneeded."""
    from .linear import tc_mm

    need_a, need_s, need_w1, need_w2 = need
    h = tc_mm(a, w1.T).float()
    with torch.enable_grad():
        h_ = h.requires_grad_(need_a or need_w1)
        s_ = score.detach().float().requires_grad_(need_s)
        gate, up = h_.chunk(2, -1)
        if limit is not None:
            gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
        act = torch.nn.functional.silu(gate) * up * s_.unsqueeze(-1)
    dw2 = tc_mm(grad.T, _requant_l2(act), torch.float32) if need_w2 else None
    dh = ds = None
    wanted = [t for t in (h_, s_) if t.requires_grad]
    if wanted:
        grads = iter(torch.autograd.grad(act, wanted, tc_mm(grad, w2, torch.float32)))
        dh = next(grads) if h_.requires_grad else None
        ds = next(grads) if s_.requires_grad else None
    da = tc_mm(dh, w1, torch.float32) if need_a else None
    dw1 = tc_mm(dh.T, a, torch.float32) if need_w1 else None
    return da, ds, dw1, dw2


def _all_gather(x, group):
    out = x.new_empty((group.size() * x.shape[0], *x.shape[1:]))
    torch.distributed.all_gather_into_tensor(out, x.contiguous(), group=group)
    return out


def _reduce_scatter(x, group):
    out = x.new_empty((x.shape[0] // group.size(), *x.shape[1:]))
    torch.distributed.reduce_scatter_tensor(out, x.contiguous(), group=group)
    return out


def _native_grouped_mm(left, right, counts, out_dtype=None):
    """Lite TE GroupedLinear's grouped TN/NN/NT kernels on decoded operands.

    Only tensor views are split by expert; one native grouped call issues all GEMMs.
    """
    from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm

    dtype = out_dtype or torch.bfloat16
    if right.ndim == 3:
        transposed = right.stride(-1) != 1
        weights = right.transpose(-1, -2) if transposed else right
        a = list(weights.bfloat16().unbind(0))
        b = list(left.bfloat16().split(counts))
        result = torch.empty((left.shape[0], right.shape[-1]), device=left.device, dtype=dtype)
        output, layout, single = [result], 'TN' if transposed else 'NN', True
    else:
        a = list(right.bfloat16().split(counts))
        b = list(left.T.contiguous().bfloat16().split(counts))
        result = torch.empty(
            (len(counts), left.shape[0], right.shape[1]), device=left.device, dtype=dtype
        )
        output, layout, single = list(result.unbind(0)), 'NT', False
    general_grouped_gemm(
        a,
        b,
        output,
        [None] * len(counts),
        dtype,
        layout=layout,
        m_splits=counts,
        single_output=single,
        grad=True,
    )
    return result


def _grouped_mlp_grads(a, score, grad, w1, w2, ends, limit, need):
    """One grouped GEMM per FC replay/dX/dW, over stable expert-major rows."""

    counts = ends.diff(prepend=ends.new_zeros(1)).cpu().tolist()

    def mm(left, right, out_dtype=None):
        return _native_grouped_mm(left, right, counts, out_dtype)

    need_a, need_s, need_w1, need_w2 = need
    if not a.shape[0]:
        return (
            torch.zeros_like(a, dtype=torch.float32) if need_a else None,
            torch.zeros_like(score, dtype=torch.float32) if need_s else None,
            torch.zeros_like(w1, dtype=torch.float32) if need_w1 else None,
            torch.zeros_like(w2, dtype=torch.float32) if need_w2 else None,
        )
    h = mm(a, w1.transpose(-1, -2)).float()
    with torch.enable_grad():
        h = h.requires_grad_(need_a or need_w1)
        scores = score.detach().float().requires_grad_(need_s)
        gate, up = h.chunk(2, -1)
        if limit is not None:
            gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
        activation = torch.nn.functional.silu(gate) * up * scores[:, None]
    dw2 = mm(grad.T, _requant_l2(activation), torch.float32) if need_w2 else None
    wanted = [t for t in (h, scores) if t.requires_grad]
    dh = ds = None
    if wanted:
        grads = iter(torch.autograd.grad(activation, wanted, mm(grad, w2, torch.float32)))
        dh = next(grads) if h.requires_grad else None
        ds = next(grads) if scores.requires_grad else None
    da = mm(dh, w1, torch.float32) if need_a else None
    dw1 = mm(dh.T, a, torch.float32) if need_w1 else None
    empty = ends == torch.cat((ends.new_zeros(1), ends[:-1]))
    for gradient in (dw1, dw2):
        if gradient is not None:
            gradient.masked_fill_(empty[:, None, None], 0)
    return da, ds, dw1, dw2


def _expert_weight_grads(owners, gradient, needed):
    """Batch FP32 DDP accumulation; owner iteration only sets metadata/views."""
    if gradient is None:
        return (None,) * len(owners)
    selected = [(p, g) for p, g, use in zip(owners, gradient.unbind(0), needed) if use]
    if selected and all(
        getattr(p, 'main_grad', None) is not None and p.main_grad.dtype == torch.float32
        for p, _ in selected
    ):
        from megatron.lite.primitive.parallel.linear import _dummy_wgrad

        torch._foreach_add_([p.main_grad for p, _ in selected], [g for _, g in selected])
        for p, _ in selected:
            p.grad_added_to_main_grad = True
        return tuple(_dummy_wgrad(p) if use else None for p, use in zip(owners, needed))
    rounded = gradient.to(owners[0].dtype)
    return tuple(g if use else None for g, use in zip(rounded.unbind(0), needed))


def _sum_routes(gradient, position, rows, topk):
    # Each segment follows original token/slot order; no atomic index_add.
    order = torch.argsort(position, stable=True)
    counts = torch.bincount(position // topk, minlength=rows)
    return torch.segment_reduce(gradient[order], 'sum', lengths=counts, unsafe=True)


def _exchange(fields, sends, receives, group, spare=0):
    """Launch one variable all-to-all per row field into its own contiguous buffer
    (``spare`` zero rows appended); returns the buffers and their pending works."""
    buffers, works = [], []
    total = sum(receives)
    for field in fields:
        out = field.new_empty((total + spare, *field.shape[1:]))
        if spare:
            out[total:].zero_()
        work = torch.distributed.all_to_all_single(
            out[:total], field.contiguous(), receives, sends, group=group, async_op=True
        )
        buffers.append(out)
        works.append(work)
    return buffers, works


def _decoded_experts(fc1, fc2):
    """All local experts' MXFP4 masters decoded, BF16 ``[L, N, K]`` (exact), from the
    forward's encoding."""
    from megatron.lite.primitive.quantization.mxfp4 import dequantize_mxfp4

    return [dequantize_mxfp4(*codes).bfloat16() for codes in experts.expert_codes(fc1, fc2)]


def _grouped(weights, rows, splits, layout, columns):
    """Lite's TE grouped GEMM over every local expert in one call: ``layout`` "TN" is
    ``rows @ W.T`` (forward), "NN" is ``rows @ W`` (dgrad); BF16 output."""
    from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm

    out = rows.new_empty((rows.shape[0], columns))
    general_grouped_gemm(
        list(weights),
        list(rows.split(splits)),
        [out],
        [None] * len(splits),
        out.dtype,
        layout=layout,
        m_splits=splits,
        grad=layout == "NN",
        single_output=True,
    )
    return out


def _grouped_wgrad(inputs, grads, splits, params):
    """Lite's fused grouped wgrad: each expert's ``grad.T @ input`` accumulates in FP32
    straight into its owner's ``main_grad`` (TE ``accumulate``); returns the autograd
    placeholders, or the FP32 dW in the owner dtype without DDP."""
    from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm

    from .linear import accumulate_wgrad

    fused = all(
        getattr(p, 'main_grad', None) is not None and p.main_grad.dtype == torch.float32
        for p in params
    )
    if fused:
        out = [p.main_grad for p in params]
    else:
        out = [torch.zeros(p.shape, dtype=torch.float32, device=inputs.device) for p in params]
    if inputs.shape[0]:
        general_grouped_gemm(
            list(inputs.split(splits)),
            list(grads.split(splits)),
            out,
            [None] * len(splits),
            torch.float32,
            layout="NT",
            m_splits=splits,
            grad=True,
            accumulate=True,
        )
    if fused:
        return [accumulate_wgrad((p,), None, p) for p in params]
    return [o.to(p.dtype) for p, o in zip(params, out)]


def _routed_grads(codes, scale, grad, weights, ids, fc1, fc2, limit, group, overlap):
    """Decoded-operand VJP of the routed experts over the rows actually routed.

    Rows go to the expert owner by one variable all-to-all (FP8 codes, scales, BF16 grad,
    score), every local expert runs in one grouped GEMM per product, and per-slot dX/dscore
    come back by the inverse all-to-all. Slots are scattered to unique positions and summed
    in slot order: no atomics, fixed reduction order. Expert GEMMs are lite's TE grouped
    GEMMs with the wgrad fused into ``main_grad``. Returns FP32 dx/dscore and the expert
    weight gradients (placeholders when fused), plus ``overlap()``'s result: it runs
    while the routed rows are in flight; the expert wgrads run while dX returns."""
    rows, k = ids.shape
    world, rank, local, hidden = group.size(), group.rank(), len(fc1), codes.shape[1]
    flat = ids.flatten()
    key = torch.where(flat >= 0, flat, local * world)
    position = torch.argsort(key, stable=True)
    # One host sync: this rank's per-expert sends and every peer's per-(local expert)
    # sends here give the send/receive splits and the local expert row groups.
    per_expert = torch.bincount(key, minlength=local * world + 1)[: local * world]
    counts = torch.cat((per_expert, all_to_all(group, per_expert))).cpu().view(2, world, local)
    sends, receives = counts.sum(2).tolist()
    splits = counts[1].sum(0).tolist()
    position = position[: sum(sends)]
    token = position // k
    # Expert ids are exact in FP32; score and id travel together.
    meta = torch.stack((weights.flatten()[position].float(), flat[position].float()), 1)
    fields = codes.view(torch.uint8)[token], scale[token], grad.bfloat16()[token], meta
    (r_codes, r_scale, r_grad, r_meta), works = _exchange(fields, sends, receives, group)
    shared = overlap()
    w1, w2 = _decoded_experts(fc1, fc2)
    for work in works:
        work.wait()
    order = torch.argsort(r_meta[:, 1].long() - rank * local, stable=True)
    if r_codes.shape[0]:
        a = experts._gather_decode_bf16(r_codes.view(torch.float8_e4m3fn), r_scale, order)
    else:
        a = r_grad.new_empty((0, hidden))
    g, score = r_grad.index_select(0, order), r_meta[:, 0].index_select(0, order)
    empty = not a.shape[0]
    if empty:
        h = a.new_zeros((0, w1.shape[1]), dtype=torch.float32)
    else:
        h = _grouped(w1, a, splits, "TN", w1.shape[1]).float()
    with torch.enable_grad():
        h_, s_ = h.requires_grad_(), score.detach().float().requires_grad_()
        gate, up = h_.chunk(2, -1)
        if limit is not None:
            gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
        act = torch.nn.functional.silu(gate) * up * s_.unsqueeze(-1)
    if empty:
        dact = act.detach().new_zeros(act.shape)
    else:
        dact = _grouped(w2, g, splits, "NN", w2.shape[2]).float()
    dh, ds = torch.autograd.grad(act, (h_, s_), dact)
    dh = dh.bfloat16()
    da = a.new_zeros((0, hidden)) if empty else _grouped(w1, dh, splits, "NN", hidden)
    # Back to arrival order, then home; a spare zero row stands for unrouted slots.
    inverse = torch.argsort(order)
    fields = da.index_select(0, inverse), ds.float().index_select(0, inverse)
    (back_x, back_s), works = _exchange(fields, receives, sends, group, spare=1)
    dfc2 = _grouped_wgrad(_requant_l2(act).bfloat16(), g, splits, fc2)
    dfc1 = _grouped_wgrad(a, dh, splits, fc1)
    for work in works:
        work.wait()
    source = torch.full((rows * k,), position.shape[0], dtype=torch.long, device=grad.device)
    source[position] = torch.arange(position.shape[0], device=grad.device)
    dx = back_x.index_select(0, source).view(rows, k, hidden).sum(1, dtype=torch.float32)
    dscore = back_s.index_select(0, source)
    return dx, dscore.view(rows, k), dfc1, dfc2, shared


class _FusedMoE(torch.autograd.Function):
    """Inference ``fp8_fp4_mega_moe`` forward; owned decoded-operand backward over the
    routed rows (:func:`_routed_grads`), expert dW straight into ``main_grad``."""

    @staticmethod
    def forward(ctx, x, scores, indices, options, *masters):
        group, limit, buffer, packed = options
        ctx.options = group, limit
        ctx.save_for_backward(x, scores, indices, *masters)
        return experts.launch(x, scores, indices, packed, buffer, limit, group.size())

    @staticmethod
    def backward(ctx, grad):
        from .linear import accumulate_wgrad

        x, weights, ids, *masters = ctx.saved_tensors
        group, limit = ctx.options
        local = len(masters) // 2 - 1
        need_x, need_w = ctx.needs_input_grad[:2]
        need_m = ctx.needs_input_grad[4:]
        fc1, fc2, s1, s2 = masters[:local], masters[local : 2 * local], *masters[2 * local :]
        with torch.autocast(device_type="cuda", enabled=False):
            codes, scale = experts.quantize_activation(x.detach().bfloat16())

            def shared():
                return _mlp_grads(
                    experts.dequantize_activation(codes, scale),
                    x.new_ones(x.shape[0], dtype=torch.float32),
                    grad.float(),
                    _fp8(s1),
                    _fp8(s2),
                    limit,
                    (need_x, False, need_m[-2], need_m[-1]),
                )

            routed, dscore, dfc1, dfc2, (dx, _, dw1s, dw2s) = _routed_grads(
                codes, scale, grad, weights, ids, fc1, fc2, limit, group, shared
            )
            dx = (routed if dx is None else dx + routed).to(x.dtype) if need_x else None
        return (
            dx,
            dscore.to(weights.dtype) if need_w else None,
            None,
            None,
            *dfc1,
            *dfc2,
            None if dw1s is None else accumulate_wgrad((s1,), dw1s, s1),
            None if dw2s is None else accumulate_wgrad((s2,), dw2s, s2),
        )


class MegaMoE(MoELayer):
    def forward(self, hidden_states, input_ids=None, **kwargs):
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        group, rank = self.ep_group, self.ep_group.rank()
        expected = list(range(rank * self.num_local_experts, (rank + 1) * self.num_local_experts))
        if self.local_expert_indices != expected:
            raise ValueError('MegaMoE requires the Core contiguous global expert ownership map')
        logits = linear(x, self.router.weight, fp8=False)
        weights, indices = route(logits, self.router, input_ids)
        limit = self.config.activation_func_clamp_value
        fc1 = [
            getattr(self.experts.linear_fc1, f'weight{i}') for i in range(self.num_local_experts)
        ]
        fc2 = [
            getattr(self.experts.linear_fc2, f'weight{i}') for i in range(self.num_local_experts)
        ]
        masters = (*fc1, *fc2)
        if self.shared_experts is None:
            out = actual_rows(x, weights, indices, masters, limit, group, False)
        else:
            masters += (
                self.shared_experts.linear_fc1.weight,
                self.shared_experts.linear_fc2.weight,
            )
            packed = experts.kernel_weights(fc1, fc2, *masters[-2:])
            capacity = 8192
            buffer = experts.symm_buffer(
                group,
                len(fc1) * group.size(),
                capacity,
                indices.shape[1],
                x.shape[1],
                fc1[0].shape[0] // 2,
            )
            chunks = torch.tensor(max(1, (x.shape[0] + capacity - 1) // capacity), device=x.device)
            torch.distributed.all_reduce(chunks, torch.distributed.ReduceOp.MAX, group=group)
            outputs = []
            for chunk in range(int(chunks)):
                rows = slice(chunk * capacity, (chunk + 1) * capacity)
                outputs.append(
                    _FusedMoE.apply(
                        x[rows],
                        weights[rows],
                        indices[rows],
                        (group, limit, buffer, packed),
                        *masters,
                    )
                )
            out = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
        return out.reshape_as(hidden_states), None
