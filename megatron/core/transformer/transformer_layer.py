# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Union

import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.attention import CrossAttentionSpec, SelfAttentionSpec
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_viewless_tensor


@dataclass
class TransformerLayerSpec:
    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    self_attention: SelfAttentionSpec = IdentityOp
    self_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    post_self_attn_layernorm: Union[ModuleSpec, type] = IdentityOp
    cross_attention: CrossAttentionSpec = IdentityOp
    cross_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    post_cross_attn_layernorm: Union[ModuleSpec, type] = IdentityOp
    ln_mlp: Union[ModuleSpec, type] = IdentityOp
    mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp
    post_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp


class TransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        spec: TransformerLayerSpec,
        layer_number: int = 1,
    ):
        super().__init__(config=config)
        self.config: TransformerConfig = config

        self.layer_number = layer_number + self._get_layer_offset()

        ## [Module 1: Input Layernorm] Optional Layernorm on the input data
        # TODO: add pytorch only layernorm
        self.input_layernorm = build_module(
            spec.input_layernorm,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
            persist_layer_norm=self.config.persist_layer_norm,
            sequence_parallel=self.config.sequence_parallel,
            zero_centered_gamma=self.config.layernorm_zero_centered_gamma,
            normalization=self.config.normalization,
        )

        ## [Module 2: SelfAttention]
        self.self_attention = build_module(
            spec.self_attention,
            config=self.config,
            spec=spec.self_attention,
            layer_number=layer_number,
        )

        ## [Module 3: BiasDropoutFusion]
        self.self_attn_bda = build_module(spec.self_attn_bda)

        ## [Module 4: Post SelfAttention] Optional Layernorm after self-attn
        self.post_self_attn_layernorm = build_module(
            spec.post_self_attn_layernorm,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
            persist_layer_norm=self.config.persist_layer_norm,
            sequence_parallel=self.config.sequence_parallel,
            zero_centered_gamma=self.config.layernorm_zero_centered_gamma,
            normalization=self.config.normalization,
        )

        ## [Module 5: CrossAttention]
        self.cross_attention = build_module(
            spec.cross_attention,
            config=self.config,
            spec=spec.cross_attention,
            layer_number=layer_number,
        )

        ## [Module 6: BiasDropoutFusion]
        self.cross_attn_bda = build_module(
            spec.cross_attn_bda,
            config=self.config,
            spec=spec.cross_attention,
        )

        ## [Module 7: Post Cross Attention] Optional Layernorm after cross-attn
        self.post_cross_attn_layernorm = build_module(
            spec.post_cross_attn_layernorm,
            config=self.config,
            spec=spec.cross_attention,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
            persist_layer_norm=self.config.persist_layer_norm,
            sequence_parallel=self.config.sequence_parallel,
            zero_centered_gamma=self.config.layernorm_zero_centered_gamma,
            normalization=self.config.normalization,
        )

        ## [Module 8: MLP block]
        self.ln_mlp = build_module(spec.ln_mlp, config=self.config)

        ## [Module 9: BiasDropoutFusion]
        self.mlp_bda = build_module(spec.mlp_bda)

        ## [Module 10: Post MLP] Optional Layernorm after MLP
        self.post_mlp_layernorm = build_module(
            spec.post_mlp_layernorm,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
            persist_layer_norm=self.config.persist_layer_norm,
            sequence_parallel=self.config.sequence_parallel,
            zero_centered_gamma=self.config.layernorm_zero_centered_gamma,
            normalization=self.config.normalization,
        )

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
        context=None,
        context_mask=None,
        inference_params=None,
        rotary_pos_emb=None,
        retriever_output=None,
    ):
        # hidden_states: [s, b, h]

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Residual connection.
        residual = input_layernorm_output

        # Self attention.
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
        )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.config.hidden_dropout
            )

        # Optional Layer norm after self-attention
        post_self_attn_layernorm_output = self.post_self_attn_layernorm(hidden_states)

        # Residual connection.
        residual = post_self_attn_layernorm_output

        # Cross attention.
        attention_output_with_bias = self.cross_attention(
            post_self_attn_layernorm_output, # i.e., 'x'
            attention_mask=context_mask,
            key_value_states=context,
            inference_params=inference_params,
            retriever_output=retriever_output,
        )

        if isinstance(attention_output_with_bias, dict) \
           and "retriever_output" in attention_output_with_bias:
            retriever_output = attention_output_with_bias["retriever_output"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.config.hidden_dropout
            )

        # Optional Layer norm post the cross-attention.
        post_cross_attn_layernorm_output = self.post_cross_attn_layernorm(hidden_states)

        # Residual connection.
        residual = post_cross_attn_layernorm_output

        # MLP.
        ln_mlp_output_with_bias = self.ln_mlp(post_cross_attn_layernorm_output)

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                ln_mlp_output_with_bias, residual, self.config.hidden_dropout
            )

        # Optional Layer norm post MLP
        output = self.post_mlp_layernorm(hidden_states)

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(
            inp=output, requires_grad=output.requires_grad, keep_graph=True
        )

        if retriever_output is None:
            return output
        else:
            return output, retriever_output

    def sharded_state_dict(self, prefix=''):

        # state_dict = self.state_dict(prefix=prefix, keep_vars=True)
        state_dict = self.state_dict(keep_vars=True)

        tensor_parallel_layers_axis_map = {
            'self_attention.linear_qkv.weight': 0,
            'self_attention.linear_qkv.bias': 0,
            'self_attention.linear_proj.weight': 1,
            'mlp.linear_fc1.weight': 0,
            'mlp.linear_fc1.bias': 0,
            'mlp.linear_fc2.weight': 1,
        }

        offset = self._get_layer_offset()
        num_layers = self.config.num_layers

        sharded_state_dict = {}

        for layer_name in state_dict.keys():
            tensor = state_dict[layer_name]
            global_layer_offset = self.layer_number - 1  # self.layer_number starts at 1
            layer_key = f'{prefix}{global_layer_offset - offset}.{layer_name}'  # module list index in TransformerBlock
            sharded_offsets = [(0, global_layer_offset, num_layers)]  # PP sharding

            if layer_name in tensor_parallel_layers_axis_map:
                tp_axis = tensor_parallel_layers_axis_map[layer_name]
                # TP sharding
                sharded_offsets.append(
                    [
                        tp_axis + 1,  # +1 for PP dimension
                        parallel_state.get_tensor_model_parallel_rank(),
                        parallel_state.get_tensor_model_parallel_world_size(),
                    ]
                )
                replica_id = parallel_state.get_data_parallel_rank()
            else:
                replica_id = (
                    parallel_state.get_data_parallel_rank()
                    * parallel_state.get_data_parallel_world_size()
                    + parallel_state.get_tensor_model_parallel_rank()
                )

            if layer_name.endswith('._extra_state'):
                sharded_state_dict[layer_key] = ShardedObject(
                    f'{prefix}{layer_name}',
                    tensor,
                    (num_layers,),
                    (global_layer_offset,),
                    replica_id,
                )

            else:
                sharded_state_dict[layer_key] = ShardedTensor.from_rank_offsets(
                    f'{prefix}{layer_name}',
                    tensor,
                    *sharded_offsets,
                    replica_id=replica_id,
                    prepend_axis_num=1,  # for PP sharding
                )

        return sharded_state_dict
