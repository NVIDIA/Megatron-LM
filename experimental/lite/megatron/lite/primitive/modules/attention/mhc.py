# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadHyperConnectionHead(nn.Module):
    def __init__(self, hidden_size: int, hc_mult: int, eps: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.eps = eps
        self.hc_fn = nn.Parameter(torch.empty(hc_mult, hc_mult * hidden_size, dtype=torch.float32))
        self.hc_base = nn.Parameter(torch.zeros(hc_mult, dtype=torch.float32))
        self.hc_scale = nn.Parameter(torch.ones(1, dtype=torch.float32))
        nn.init.xavier_uniform_(self.hc_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return x
        shape, dtype = x.shape, x.dtype
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        mixes = F.linear(xf, self.hc_fn.float()) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps
        y = torch.sum(pre.unsqueeze(-1) * xf.view(shape), dim=2)
        return y.to(dtype)
