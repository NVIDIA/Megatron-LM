# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    split_sizes = [hc_mult, hc_mult, hc_mult * hc_mult]
    pre_mix, post_mix, comb_mix = mixes.split(split_sizes, dim=-1)
    base_pre, base_post, base_comb = hc_base.to(dtype=mixes.dtype, device=mixes.device).split(
        split_sizes, dim=-1
    )
    scale = hc_scale.to(dtype=mixes.dtype, device=mixes.device)
    pre = torch.sigmoid(pre_mix * scale[0] + base_pre)
    post = 2 * torch.sigmoid(post_mix * scale[1] + base_post)
    comb_logits = (comb_mix * scale[2] + base_comb).view(*comb_mix.shape[:-1], hc_mult, hc_mult)
    comb = torch.exp(comb_logits - comb_logits.max(dim=-1, keepdim=True).values)
    for _ in range(iters):
        comb = comb / comb.sum(dim=-1, keepdim=True).clamp(min=eps)
        comb = comb / comb.sum(dim=-2, keepdim=True).clamp(min=eps)
    return pre, post, comb


class HyperConnection(nn.Module):
    def __init__(self, hidden_size: int, hc_mult: int, sinkhorn_iters: int, eps: float):
        super().__init__()
        mix = (2 + hc_mult) * hc_mult
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        self.fn = nn.Parameter(torch.empty(mix, hc_mult * hidden_size, dtype=torch.float32))
        self.base = nn.Parameter(torch.zeros(mix, dtype=torch.float32))
        self.scale = nn.Parameter(torch.ones(3, dtype=torch.float32))
        nn.init.xavier_uniform_(self.fn)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() == 3:
            x = x.unsqueeze(2).expand(*x.shape[:2], self.hc_mult, x.size(-1))
        shape, dtype = x.shape, x.dtype
        xf = x.flatten(2)
        rms_inv = 1.0 / (xf.norm(dim=-1, keepdim=True) / math.sqrt(xf.shape[-1]) + self.eps)
        mixes = F.linear(xf, self.fn.to(device=x.device, dtype=dtype)) * rms_inv
        pre, post, comb = split_sinkhorn(
            mixes, self.scale, self.base, self.hc_mult, self.sinkhorn_iters, self.eps
        )
        y = torch.sum(pre.unsqueeze(-1) * xf.view(shape), dim=2)
        return y.to(dtype), post, comb

    @staticmethod
    def post(
        x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
    ) -> torch.Tensor:
        dtype = x.dtype
        placed = post.to(dtype).unsqueeze(-1) * x.unsqueeze(-2)
        mixed = torch.matmul(comb.to(dtype), residual.to(dtype))
        return placed + mixed
