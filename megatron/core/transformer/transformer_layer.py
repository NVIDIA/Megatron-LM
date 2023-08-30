# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import re

import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.custom_layers.transformer_engine import TENorm
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_viewless_tensor


class TransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int = 1,
        self_attn_mask_type=AttnMaskType.padding,
    ):
        super().__init__(config=config)
        self.config: TransformerConfig = config

        self.layer_number = layer_number + self._get_layer_offset()

        self.self_attn_mask_type = self_attn_mask_type

        # Layernorm on the input data.
        # TODO: add pytorch only layernorm
        self.input_layernorm = IdentityOp(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
            persist_layer_norm=self.config.persist_layer_norm,
            sequence_parallel=self.config.sequence_parallel,
            zero_centered_gamma=self.config.layernorm_zero_centered_gamma,
            normalization=self.config.normalization,
        )

        # Self attention.
        self.self_attention = SelfAttention(
            config=self.config, layer_number=layer_number, attn_mask_type=self_attn_mask_type,
        )

        # Layernorm on the attention output
        self.post_self_attn_layernorm = IdentityOp(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
            persist_layer_norm=self.config.persist_layer_norm,
            sequence_parallel=self.config.sequence_parallel,
            zero_centered_gamma=self.config.layernorm_zero_centered_gamma,
            normalization=self.config.normalization,
        )

        # MLP
        self.mlp = MLP(config=self.config)

        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad

        self.bias_dropout_add_func = get_bias_dropout_add(
            self.training, self.config.bias_dropout_fusion
        )

    def _get_layer_offset(self):

        pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()

        num_layers_per_pipeline_rank = (
            self.config.num_layers // parallel_state.get_pipeline_model_parallel_world_size()
        )

        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            vp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
            vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

            total_num_layers = self.config.num_layers
            num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size
            total_virtual_chunks = total_num_layers // vp_size
            offset = vp_rank * total_virtual_chunks + (pipeline_rank * num_layers_per_virtual_rank)

        else:
            # Each stage gets a contiguous set of layers.
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:
                offset = pipeline_rank * num_layers_per_pipeline_rank
            else:
                offset = 0

        return offset

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_output=None,
        enc_dec_attn_mask=None,
        inference_params=None,
        rotary_pos_emb=None,
    ):
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output_with_bias = self.self_attention(
            layernorm_output,
            attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
        )

        # Residual connection.
        if self.config.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # bias_dropout_add fusion returning fp32 instead of bf16
        with self.bias_dropout_add_exec_handler():
            layernorm_input = self.bias_dropout_add_func(
                attention_output_with_bias, residual, self.config.hidden_dropout
            )

        # Layer norm post the self attention.
        layernorm_output = self.post_self_attn_layernorm(layernorm_input)

        # MLP.
        mlp_output_with_bias = self.mlp(layernorm_output)

        # Second residual connection.
        if self.config.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        with self.bias_dropout_add_exec_handler():
            output = self.bias_dropout_add_func(
                mlp_output_with_bias, residual, self.config.hidden_dropout
            )

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(
            inp=output, requires_grad=output.requires_grad, keep_graph=True
        )

        return output

    def sharded_state_dict(self, prefix=''):
        offset = self._get_layer_offset()
        num_layers = self.config.num_layers

        global_layer_offset = self.layer_number - 1  # self.layer_number starts at 1
        state_dict_prefix = (
            f'{prefix}{global_layer_offset - offset}.'  # module list index in TransformerBlock
        )
        sharded_pp_offset = [
            (0, global_layer_offset, num_layers)
        ]  # PP sharding offset for ShardedTensors

        attn_state_dict = self.self_attention.sharded_state_dict(
            prefix=f'{state_dict_prefix}self_attention.',
            sharded_key_prefix=f'{prefix}self_attention.',
            sharded_offsets=sharded_pp_offset,
        )

        mlp_state_dict = self.mlp.sharded_state_dict(
            prefix=f'{state_dict_prefix}mlp.',
            sharded_key_prefix=f'{prefix}mlp.',
            sharded_offsets=sharded_pp_offset,
        )

        sharded_state_dict = {**mlp_state_dict, **attn_state_dict}

        return sharded_state_dict
