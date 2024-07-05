# Copyright (c) 2024, Tri Dao, Albert Gu.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Union

import torch
from torch import Tensor

from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class MambaLayerSubmodules:
    norm: Union[ModuleSpec, type] = IdentityOp
    mixer: Union[ModuleSpec, type] = IdentityOp


class MambaLayer(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: MambaLayerSubmodules,
        mamba_ssm_ngroups=8,
        layer_idx=None,
        residual_in_fp32=False,
    ):
        """
        Top level Mamba Layer
        """
        super().__init__(config)
        self.config = config
        self.residual_in_fp32 = residual_in_fp32
        self.mixer = build_module(
            submodules.mixer,
            self.config,
            self.config.hidden_size,
            ngroups=mamba_ssm_ngroups,
            layer_idx=layer_idx,
        )
        self.norm = build_module(submodules.norm, self.config, self.config.hidden_size)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,  # Not used in MambaLayer
        inference_params=None,
        rotary_pos_emb: Tensor = None,  # Not used in MambaLayer
    ):

        residual = hidden_states
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states + residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)
