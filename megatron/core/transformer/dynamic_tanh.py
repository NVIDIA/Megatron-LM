from typing import Optional

import torch
from torch import nn

from megatron.core.transformer import TransformerConfig


class DynamicTanh(nn.Module):
    def __init__(self,
        config: TransformerConfig,
        hidden_size: int,
        eps: Optional[float] = None,  # Unused. Added to match LayerNorm interface.
        init_value: Optional[float] = None,  # gamma init value.
    ):

        super().__init__()

        self.config = config
        self.hidden_size = hidden_size
        self.init_value = init_value

        self.alpha = nn.Parameter(torch.empty(1))
        self.weight = nn.Parameter(torch.empty(hidden_size))
        if self.config.dyt_bias:
            self.beta = nn.Parameter(torch.empty(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.alpha, self.config.dyt_alpha_init)
        if self.init_value is None:
            nn.init.ones_(self.weight)
        else:
            nn.init.constant_(self.weight, self.init_value)
        if self.config.dyt_bias:
            nn.init.zeros_(self.beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.weight*torch.tanh(self.alpha*x)
        if self.config.dyt_bias:
            return x + self.beta
        return x
