# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Independent autograd oracle for the rounded block-factor arithmetic."""

import torch
import torch.nn.functional as F


class _Value(torch.autograd.Function):
    @staticmethod
    def forward(ctx, computed, saved):
        return saved

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def rounded(x):
    return _Value.apply(x, x.bfloat16().float())


def exp(x):
    # Match the specified FP32 exp2(x * log2(e)) rounding point, using PyTorch
    # rather than any production kernel. torch.exp uses a different primitive;
    # rare BF16 coefficient ties then evaluate gradients at different values.
    return torch.exp2(x * 1.4426950408889634)


def oracle(q, k, v, g, beta, initial, *, c=16, tape=None):
    """Differentiate the specified rounded forward, optionally at GPU tape values."""
    b, t, h, d = q.shape
    r, w = k.shape[2], v.shape[-1]
    total = t * r
    nb = (total + c - 1) // c
    queries = torch.stack([torch.zeros_like(q)] * (r - 1) + [q], dim=2)
    gates = torch.stack([g] + [torch.zeros_like(g)] * (r - 1), dim=2)

    def blocks(x):
        x = x.flatten(1, 2).transpose(1, 2).float()
        if x.ndim == 3:
            return F.pad(x, (0, nb * c - total)).reshape(b, h, nb, c)
        return F.pad(x, (0, 0, 0, nb * c - total)).reshape(b, h, nb, c, x.shape[-1])

    qs, ks, vs, gs, bs = map(blocks, (queries, k, v, gates, beta))
    state = initial.float()
    outputs = []
    strict = torch.ones((c, c), device=q.device, dtype=torch.bool).tril(-1)
    causal = torch.ones_like(strict).tril()
    for block in range(nb):
        count = min(c, total - block * c)
        query, key, val, gate, bet = (
            qs[:, :, block],
            ks[:, :, block],
            vs[:, :, block],
            gs[:, :, block],
            bs[:, :, block],
        )
        if tape is not None:
            state = _Value.apply(state, tape[0][:, block])
        sb = rounded(state)
        prefixes = []
        running = torch.zeros_like(gate[..., 0])
        for row in range(c):
            running = running + gate[..., row]
            prefixes.append(running)
        p = torch.stack(prefixes, -1)
        decay = exp(p)
        ratio = exp((p[..., None] - p[..., None, :]).masked_fill(~causal, 0))
        gram = key @ key.transpose(-1, -2)
        a = rounded((-bet[..., None] * ratio * gram).masked_fill(~strict, 0))
        factors = []
        for row in range(c):
            previous = (
                F.pad(torch.stack(factors, -2), (0, 0, 0, c - row))
                if factors
                else torch.zeros_like(a)
            )
            diagonal = torch.eye(c, device=q.device)[row].expand(b, h, -1)
            fi = rounded(diagonal + (a[..., row, :, None] * previous).sum(-2))
            if row >= count:
                fi = torch.zeros_like(fi)
            if tape is not None:
                fi = _Value.apply(fi, tape[1][:, block, :, row].float())
            factors.append(fi)
        factor = torch.stack(factors, -2)
        proj = key @ sb
        rhs = rounded((val - decay[..., None] * proj) * bet[..., None])
        if tape is not None:
            rhs = _Value.apply(rhs, tape[2][:, block].float())
        u = rounded(factor @ rhs)
        if tape is not None:
            u = _Value.apply(u, tape[3][:, block].float())
        qstate = query @ sb
        weight = rounded((ratio * (query @ key.transpose(-1, -2))).masked_fill(~causal, 0))
        output = rounded((decay[..., None] * qstate + weight @ u) * d**-0.5)
        outputs.append(output)
        if count == c:
            f = rounded(key * exp(p[..., -1, None] - p)[..., None])
            state = state * exp(p[..., -1])[..., None, None] + f.transpose(-1, -2) @ u
    output = torch.cat(outputs, -2)[..., :total, :][..., r - 1 :: r, :].transpose(1, 2)
    return output, state
