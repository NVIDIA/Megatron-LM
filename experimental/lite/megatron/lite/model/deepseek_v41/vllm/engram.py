# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Reuse frozen Lite rows; native inference projection/gate and owned STE VJP."""
import torch

from .linear import linear


def reference(hidden, kv, query, key, eps, mask):
    copies, width = hidden.shape[-2:]
    keys, values = kv.float().split((copies * width, width), -1)
    keys = keys.unflatten(-1, (copies, width))
    q = query + (query.bfloat16().to(query.dtype) - query).detach()
    k = key + (key.bfloat16().to(key.dtype) - key).detach()
    h = hidden.float()
    inv = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(keys.square().mean(-1) + eps)
    dot = (h * q.float() * k.float() * keys).sum(-1) * inv * width**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    if mask is not None:
        gate = gate.masked_fill(~mask.unsqueeze(-1), 0)
    return (h + gate.unsqueeze(-1) * values.unsqueeze(-2)).to(hidden.dtype)


class _Post(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, kv, query, key, eps, mask):
        import triton
        from vllm.models.deepseek_v41.common.engram import _fused_engram_post_wkv_kernel

        copies, width = hidden.shape[-2:]
        h = hidden.reshape(-1, copies, width).contiguous()
        v = kv.reshape(h.shape[0], -1).contiguous()
        q, k = query.bfloat16().contiguous(), key.bfloat16().contiguous()
        active = None if mask is None else mask.reshape(-1).contiguous()
        out = torch.empty_like(h)
        block = triton.next_power_of_2(width)
        _fused_engram_post_wkv_kernel[(h.shape[0] * copies,)](
            h,
            v,
            q,
            k,
            h if active is None else active,
            out,
            h.shape[0],
            *h.stride(),
            *v.stride(),
            *q.stride(),
            *k.stride(),
            0 if active is None else active.stride(0),
            *out.stride(),
            eps,
            1e-6,
            DIM=width,
            HC_MULT=copies,
            BLOCK_SIZE=block,
            HAS_MASK=active is not None,
            num_warps=8 if block >= 2048 else 4,
        )
        ctx.save_for_backward(hidden, kv, query, key)
        ctx.eps, ctx.mask = eps, mask
        return out.reshape_as(hidden)

    @staticmethod
    def backward(ctx, grad):
        with torch.enable_grad(), torch.autocast(device_type=grad.device.type, enabled=False):
            leaves = [x.detach().requires_grad_() for x in ctx.saved_tensors]
            output = reference(*leaves, ctx.eps, ctx.mask)
            gradients = torch.autograd.grad(output, leaves, grad)
        return *gradients, None, None


def forward(module, hidden, hash_ids, token_mask=None):
    rows = module.embed(hash_ids).flatten(-2)
    kv = linear(rows, module.wkv.weight, fp8=True).bfloat16()
    return _Post.apply(hidden, kv, module.q_weight, module.k_weight, module.eps, token_mask)
