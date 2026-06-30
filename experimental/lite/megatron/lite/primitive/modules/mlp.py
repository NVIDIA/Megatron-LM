# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import torch
import torch.nn as nn

from megatron.lite.primitive.modules.experts import swiglu_with_probs


class SwiGLUMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        swiglu_limit: float = 0.0,
    ):
        super().__init__()
        self.gate_up = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.swiglu_limit = float(swiglu_limit or 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = swiglu_with_probs(self.gate_up(x), None, self.swiglu_limit)
        return self.down(y.to(dtype=x.dtype))
