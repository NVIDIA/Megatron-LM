from contextlib import nullcontext

import torch

from contextlib import nullcontext

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.fp8_utils import get_fp8_context

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TERowParallelLinear,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


class ConvTokenMerge(MegatronModule):
    def __init__(self, config, size=2, use_te=True):
        super().__init__(config=config)

        # TODO: should padding?
        self.hidden_size = config.hidden_size
        self.size = size
        fp8_init_context = get_fp8_context(config, 0, is_init=True)
        with fp8_init_context:
            self.mlp = MLP(
                config,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear if use_te else ColumnParallelLinear,
                    linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
                ),
                input_size=self.hidden_size * self.size * self.size,
            )

    def create_indices(self, patch_sizes):
        assert all(
            [h % self.size == 0 and w % self.size == 0 for h, w in patch_sizes]
        ), "requires image patch sizes to be divisible by reduction size"

        offset = 0
        indices = []
        for h, w in patch_sizes:
            index = torch.arange(h * w).reshape(h, w)
            index = index.reshape(h // self.size, self.size, w // self.size, self.size)
            index = index.permute(0, 2, 1, 3).reshape(-1)
            index += offset

            indices.append(index)
            offset += int(h * w)

        return torch.cat(indices)

    def forward(self, x, patch_sizes):
        # TODO:for now, no padding and require that divisible by size
        assert all(
            [h % self.size == 0 and w % self.size == 0 for h, w in patch_sizes]
        ), "ConvTokenMerge requires image patch sizes to be divisible by 2"

        indices = self.create_indices(patch_sizes)

        new_x = x[:, indices, :]

        # Padding for fp8
        fp8_padding = 0
        pad_to_value = 16 * self.size * self.size
        if self.config.fp8 and new_x.shape[1] % pad_to_value != 0:
            fp8_padding = pad_to_value - (new_x.shape[1] % pad_to_value)
            new_x = torch.cat([new_x, torch.zeros([new_x.shape[0], fp8_padding, new_x.shape[2]]).to(new_x.device).to(new_x.dtype)], dim=1)

        fp8_context = get_fp8_context(self.config) if self.config.fp8 else nullcontext()
        with fp8_context:
            new_x, bias = self.mlp(
                new_x.view(new_x.shape[0], -1, self.hidden_size * self.size * self.size)
            )
            if bias is not None:
                new_x = new_x + bias

        if fp8_padding > 0:
            new_x = new_x[:, :-(fp8_padding // (self.size * self.size)), :]

        return new_x
