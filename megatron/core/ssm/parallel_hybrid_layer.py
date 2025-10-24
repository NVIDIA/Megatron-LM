# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, Union, Tuple

import torch

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.core.ssm.mamba_mixer import MambaMixerSubmodules  
from megatron.core.transformer.attention import SelfAttentionSubmodules


@dataclass
class ParallelHybridLayerSubmodules:
    """Configuration class for specifying the submodules of a parallel hybrid layer."""
    mamba_mixer: Union[ModuleSpec, type] = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp
    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    parallel_hybrid_bda: Union[ModuleSpec, type] = IdentityOp


class ParallelHybridLayer(MegatronModule):
    """A parallel hybrid layer that combines Mamba and Attention components."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: ParallelHybridLayerSubmodules,
        layer_number: int = 1,
        residual_in_fp32=False,
        model_comm_pgs: ModelCommProcessGroups = None,
    ):
        super().__init__(config)
        assert model_comm_pgs is not None, "model_comm_pgs must be provided for ParallelHybridLayer"

        self.config = config
        self.layer_number = layer_number
        self.residual_in_fp32 = residual_in_fp32
        self.hidden_dropout = config.hidden_dropout

        self.input_layernorm = build_module(
            submodules.input_layernorm, 
            config=self.config, 
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        mamba_submodules = MambaMixerSubmodules(
            in_proj=submodules.mamba_mixer.submodules.in_proj,
            out_proj=submodules.mamba_mixer.submodules.out_proj,
        )

        self.mamba_mixer = build_module(
            submodules.mamba_mixer.module,
            submodules=mamba_submodules,
            config=self.config,
            layer_number=layer_number,
            d_model=self.config.hidden_size,
            model_comm_pgs=model_comm_pgs
        )

        attention_optional_kwargs = {}
        if self.config.context_parallel_size > 1 and self.config.cp_comm_type is not None:
            if isinstance(self.config.cp_comm_type, list):
                attention_optional_kwargs["cp_comm_type"] = self.config.cp_comm_type[self.layer_number]
            else:
                attention_optional_kwargs["cp_comm_type"] = self.config.cp_comm_type
        model_comm_pgs = ModelCommProcessGroups.use_mpu_process_groups()
        attention_optional_kwargs["model_comm_pgs"] = model_comm_pgs

        attention_submodules = SelfAttentionSubmodules(
            linear_qkv=submodules.self_attention.module.submodules.linear_qkv,
            core_attention=submodules.self_attention.module.submodules.core_attention,
            linear_proj=submodules.self_attention.module.submodules.linear_proj,
            q_layernorm=getattr(submodules.self_attention.module.submodules, 'q_layernorm', None),
            k_layernorm=getattr(submodules.self_attention.module.submodules, 'k_layernorm', None),
        )

        self.self_attention = build_module(
            submodules.self_attention.module,
            submodules=attention_submodules,
            config=self.config,
            layer_number=self.layer_number,
            **attention_optional_kwargs,
        )

        self.parallel_hybrid_bda = build_module(submodules.parallel_hybrid_bda)
        self.bias_dropout_add_exec_handler = torch.enable_grad

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        position_ids: Optional[torch.Tensor] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ):
        residual = hidden_states
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = hidden_states.to(dtype=self.config.params_dtype)
        hidden_states = self.input_layernorm(hidden_states)

        outputs = []
        biases = []

        mamba_output, mamba_bias = self.mamba_mixer(
            hidden_states,
            inference_context=inference_context,
        )
        outputs.append(mamba_output)
        if mamba_bias is not None:
            biases.append(mamba_bias)

        attn_output, attn_bias = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )
        outputs.append(attn_output)
        if attn_bias is not None:
            biases.append(attn_bias)

        combined_output = sum(outputs)
        combined_bias = sum(biases) if biases else None

        out_with_bias = (combined_output, combined_bias)

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.parallel_hybrid_bda(
                training=self.training, 
                fused=self.config.bias_dropout_fusion
            )(out_with_bias, residual, self.hidden_dropout)

        return hidden_states

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        """Allocate inference cache for both components."""
        caches = {}

        if self.mamba_mixer is not None:
            mamba_cache = self.mamba_mixer.allocate_inference_cache(
                batch_size, max_seqlen, dtype
            )
            caches['mamba'] = mamba_cache

        if self.self_attention is not None:
            pass

        return caches

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        from megatron.core.transformer.utils import sharded_state_dict_default

        sharded_state_dict = {}

        norm_sd = sharded_state_dict_default(
            self.input_layernorm, f'{prefix}input_layernorm.', sharded_offsets, metadata
        )
        sharded_state_dict.update(norm_sd)

        mamba_sd = sharded_state_dict_default(
            self.mamba_mixer, f'{prefix}mamba_mixer.', sharded_offsets, metadata
        )
        sharded_state_dict.update(mamba_sd)

        attn_sd = sharded_state_dict_default(
            self.self_attention, f'{prefix}self_attention.', sharded_offsets, metadata
        )
        sharded_state_dict.update(attn_sd)

        return sharded_state_dict

    