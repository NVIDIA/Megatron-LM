# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared-first, fixed-slot MoE arithmetic over actual routed Core A2A rows."""
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


class _FusedMoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scores, indices, options, *masters):
        group, limit, buffer, packed = options
        ctx.options = group, limit
        ctx.save_for_backward(x, scores, indices, *masters)
        return experts.launch(x, scores, indices, packed, buffer, limit, group.size())

    @staticmethod
    def backward(ctx, grad):
        x, scores, indices, *masters = ctx.saved_tensors
        group, limit = ctx.options
        need = (*ctx.needs_input_grad[:2], *ctx.needs_input_grad[4:])
        with torch.enable_grad():
            leaves = [t.detach().requires_grad_(n) for t, n in zip((x, scores, *masters), need)]
            x, scores, *m = leaves
            local = len(m) // 2 - 1
            codes, scale = experts.quantize_activation(x)
            first = experts._Shared.apply(x, m[-2], (codes, scale))
            hidden, encoded = experts.activation(first, x.new_ones(x.shape[0]), limit)
            shared = experts._Shared.apply(hidden, m[-1], (*encoded, True))
            live, weights, encoded, counts, state = dispatch_rows(
                x, scores, indices, (codes, scale), local, group
            )
            first = experts._Grouped.apply(live, tuple(counts), encoded, *m[:local])
            hidden, encoded = experts.activation(first, weights, limit)
            routed = experts._Grouped.apply(
                hidden, tuple(counts), (*encoded, True), *m[local : 2 * local]
            )
            position, inverse, sends, receives = state
            # The combine is linear: route the incoming cotangent instead of
            # materializing/sending an unused terminal primal and its gradient.
            upstream = all_to_all(group, grad[position // indices.shape[1]], receives, sends)
            upstream = upstream[torch.argsort(inverse)]
            wanted = [t for t in leaves if t.requires_grad]
            gradients = iter(torch.autograd.grad((routed, shared), wanted, (upstream, grad)))
            result = [next(gradients) if n else None for n in need]
        return result[0], result[1], None, None, *result[2:]


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


class _DenseMoE(torch.autograd.Function):
    """#260 MegaMoE backward: per-expert decoded VJP over EP-all-gathered rows, FP32
    reduce-scatter back, expert dW straight into ``main_grad``. Same fused forward."""

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
        grad = grad.float()
        with torch.autocast(device_type="cuda", enabled=False):
            a = experts.dequantize_activation(*experts.quantize_activation(x.detach().bfloat16()))
            dx, _, dw1s, dw2s = _mlp_grads(
                a,
                a.new_ones(a.shape[0]),
                grad,
                _fp8(s1),
                _fp8(s2),
                limit,
                (need_x, False, need_m[-2], need_m[-1]),
            )
            dx = torch.zeros_like(a) if dx is None else dx
            world, rank = group.size(), group.rank()
            rows, k = ids.shape
            if world > 1:
                capacity = torch.tensor([rows], device=x.device)
                torch.distributed.all_reduce(capacity, torch.distributed.ReduceOp.MAX, group=group)
                pad = int(capacity) - rows

                def padded(t, v=0):
                    return torch.nn.functional.pad(t, (0, 0, 0, pad), value=v)

                a_all = _all_gather(padded(a.bfloat16()), group)
                g_all = _all_gather(padded(grad.bfloat16()), group)
                w_all = _all_gather(padded(weights.float()), group)
                ids_all = _all_gather(padded(ids, -1), group)
            else:
                a_all, g_all, w_all, ids_all = a, grad, weights.float(), ids
            da_all = torch.zeros(a_all.shape, dtype=torch.float32, device=a_all.device)
            dw_all = torch.zeros_like(w_all)
            flat = ids_all.flatten()
            dfc1, dfc2 = [None] * local, [None] * local
            for e in range(local):
                pos = (flat == rank * local + e).nonzero().flatten()
                if not pos.numel():
                    dfc1[e] = torch.zeros_like(fc1[e]) if need_m[e] else None
                    dfc2[e] = torch.zeros_like(fc2[e]) if need_m[local + e] else None
                    continue
                token = pos // k
                da, dsc, d1, d2 = _mlp_grads(
                    a_all[token],
                    w_all.flatten()[pos],
                    g_all[token],
                    _mx(fc1[e]),
                    _mx(fc2[e]),
                    limit,
                    (need_x, need_w, need_m[e], need_m[local + e]),
                )
                if da is not None:
                    da_all.index_add_(0, token, da)
                if dsc is not None:
                    dw_all.view(-1)[pos] = dsc
                dfc1[e] = None if d1 is None else accumulate_wgrad((fc1[e],), d1, fc1[e])
                dfc2[e] = None if d2 is None else accumulate_wgrad((fc2[e],), d2, fc2[e])
            if world > 1:
                da_all = _reduce_scatter(da_all, group)[:rows]
                dw_all = _reduce_scatter(dw_all, group)[:rows]
            dx = (dx + da_all).to(x.dtype) if need_x else None
            dweights = dw_all.to(weights.dtype) if need_w else None
        return (
            dx,
            dweights,
            None,
            None,
            *dfc1,
            *dfc2,
            None if dw1s is None else accumulate_wgrad((s1,), dw1s, s1),
            None if dw2s is None else accumulate_wgrad((s2,), dw2s, s2),
        )


class MegaMoE(MoELayer):
    # A/B of the fused-forward backward: "grouped" (variable-row EP A2A + grouped VJP)
    # or "megamoe" (#260 dense EP all-gather + per-expert VJP).
    backward_impl = 'grouped'

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
            fused = _DenseMoE if self.backward_impl == 'megamoe' else _FusedMoE
            for chunk in range(int(chunks)):
                rows = slice(chunk * capacity, (chunk + 1) * capacity)
                outputs.append(
                    fused.apply(
                        x[rows],
                        weights[rows],
                        indices[rows],
                        (group, limit, buffer, packed),
                        *masters,
                    )
                )
            out = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
        return out.reshape_as(hidden_states), None
