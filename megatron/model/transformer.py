# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer."""
import math
import torch
import torch.nn.functional as F

from megatron import get_args
from megatron import mpu
from .module import MegatronModule
from megatron.model.enums import AttnMaskType, LayerType, AttnType
from megatron.model import LayerNorm
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.fused_bias_gelu import bias_gelu_impl
from megatron.model.utils import attention_mask_func, openai_gelu, erf_gelu
from torch import distributed as dist
import deepspeed
from deepspeed.moe.layer import MoE
from deepspeed.accelerator import get_accelerator
# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
"""

class ParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(self, init_method, output_layer_init_method, moe=False, enable_expert_tensor_parallelism=False):
        super(ParallelMLP, self).__init__()
        args = get_args()

        # Project to 4h.
        self.dense_h_to_4h = mpu.ColumnParallelLinear(
            args.hidden_size,
            args.ffn_hidden_size,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            moe=moe,
            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism
            )

        self.bias_gelu_fusion = args.bias_gelu_fusion
        self.activation_func = F.gelu
        if args.openai_gelu:
            self.activation_func = openai_gelu
        elif args.onnx_safe:
            self.activation_func = erf_gelu

        # Project back to h.
        self.dense_4h_to_h = mpu.RowParallelLinear(
            args.ffn_hidden_size,
            args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            moe=moe,
            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism)

    def forward(self, hidden_states):

        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if self.bias_gelu_fusion:
             intermediate_parallel = \
                     bias_gelu_impl(intermediate_parallel, bias_parallel)
        else:
            intermediate_parallel = \
                self.activation_func(intermediate_parallel + bias_parallel)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias

class ParallelAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(self, init_method,
                 output_layer_init_method, layer_number,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.padding):
        super(ParallelAttention, self).__init__()
        args = get_args()
        self.fp16 = args.fp16
        self.bf16 = args.bf16

        self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.num_attention_heads = args.num_attention_heads
        projection_size = args.kv_channels * args.num_attention_heads

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(projection_size,
                                                    world_size)
        self.hidden_size_per_attention_head = mpu.divide(
            projection_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = mpu.divide(
            args.num_attention_heads, world_size)

        # Strided linear layer.
        if attention_type == AttnType.self_attn:
            self.query_key_value = mpu.ColumnParallelLinear(
                args.hidden_size,
                3 * projection_size,
                gather_output=False,
                init_method=init_method)
        else:
            assert attention_type == AttnType.cross_attn
            self.query = mpu.ColumnParallelLinear(
                args.hidden_size,
                projection_size,
                gather_output=False,
                init_method=init_method)

            self.key_value = mpu.ColumnParallelLinear(
                args.hidden_size,
                2 * projection_size,
                gather_output=False,
                init_method=init_method)

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16, self.bf16,
            self.attn_mask_type,
            args.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(args.attention_dropout)

        # Output.
        self.dense = mpu.RowParallelLinear(
            projection_size,
            args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True)

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint

    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False, encoder_output=None):
        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================

        if self.attention_type == AttnType.self_attn:
            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer,
             key_layer,
             value_layer) = mpu.split_tensor_along_last_dim(mixed_x_layer, 3)
        else:
            # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            mixed_kv_layer, _ = self.key_value(encoder_output)

            # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 2 * self.hidden_size_per_attention_head)
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
            (key_layer,
             value_layer) = mpu.split_tensor_along_last_dim(mixed_kv_layer, 2)

            # Attention head [sq, b, h] --> [sq, b, hp]
            query_layer, _ = self.query(hidden_states)
            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 self.hidden_size_per_attention_head)
            query_layer = query_layer.view(*new_tensor_shape)

        # ==================================
        # Adjust key and value for inference
        # ==================================

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer),
                                   key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer),
                                     value_layer), dim=0)
        if get_key_value:
            present = (key_layer, value_layer)

        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2],
                                       output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3],
                                   output_size[0] * output_size[1], -1)

        # preallocting result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0]*output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=get_accelerator().current_device_name())

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),   # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0, alpha=(1.0/self.norm_factor))

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        if get_key_value:
            with torch.no_grad():
                if layer_past is not None:
                    attention_mask = attention_mask[
                        ...,
                        attention_scores.size(3) - 1,
                        :attention_scores.size(3)].unsqueeze(2)
                else:
                    attention_mask = attention_mask[
                        ...,
                        :attention_scores.size(3),
                        :attention_scores.size(3)]

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with mpu.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1),
                       value_layer.size(2),
                       query_layer.size(0),
                       value_layer.size(3))

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0),
                                       output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        if get_key_value:
            output = [output, present]

        return output, bias


def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
    out = residual + out
    return out


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)
    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(x, bias, residual, prob):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(x, bias, residual, prob):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, False)


class ParallelTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(self, init_method, output_layer_init_method,
                 layer_number, layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding, num_experts=1):
        args = get_args()

        super(ParallelTransformerLayer, self).__init__()
        self.layer_number = layer_number
        self.layer_type = layer_type

        self.apply_residual_connection_post_layernorm \
            = args.apply_residual_connection_post_layernorm

        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon)

        # Self attention.
        self.attention = ParallelAttention(
            init_method,
            output_layer_init_method,
            layer_number,
            attention_type=AttnType.self_attn,
            attn_mask_type=self_attn_mask_type)
        self.hidden_dropout = args.hidden_dropout
        self.bias_dropout_fusion = args.bias_dropout_fusion

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon)

        if self.layer_type == LayerType.decoder:
            self.inter_attention = ParallelAttention(
                init_method,
                output_layer_init_method,
                layer_number,
                attention_type=AttnType.cross_attn)
            # Layernorm on the attention output.
            self.post_inter_attention_layernorm = LayerNorm(
                args.hidden_size,
                eps=args.layernorm_epsilon)

        self.num_experts = num_experts
        # MLP
        if self.num_experts <= 1:
            self.mlp = ParallelMLP(init_method,
                               output_layer_init_method)
        else:
            enable_expert_tensor_parallelism = args.enable_expert_tensor_parallelism
            self.mlp = MoE(args.hidden_size,
                            ParallelMLP(init_method,
                                output_layer_init_method=output_layer_init_method,
                                moe=True,
                                enable_expert_tensor_parallelism=enable_expert_tensor_parallelism),
                            num_experts=self.num_experts, 
                            ep_size=args.moe_expert_parallel_size,
                            k=args.topk,
                            use_residual=(args.mlp_type == 'residual'),
                            capacity_factor=args.moe_train_capacity_factor,
                            eval_capacity_factor=args.moe_eval_capacity_factor,
                            min_capacity=args.moe_min_capacity,
                            drop_tokens=args.moe_token_dropping, use_tutel=args.use_tutel,
                            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism) 

    def forward(self, hidden_states, attention_mask=None,
                encoder_output=None, enc_dec_attn_mask=None,
                layer_past=None, get_key_value=False):
        # hidden_states: [b, s, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, attention_bias = \
            self.attention(layernorm_output,
                                attention_mask,
                                layer_past=layer_past,
                                get_key_value=get_key_value)

        if get_key_value:
            attention_output, presents = attention_output

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # jit scripting for a nn.module (with dropout) is not
        # trigerring the fusion kernel. For now, we use two
        # different nn.functional routines to account for varying
        # dropout semantics during training and inference phases.
        if self.bias_dropout_fusion:
            if self.training:
                bias_dropout_add_func = bias_dropout_add_fused_train
            else:
                bias_dropout_add_func = bias_dropout_add_fused_inference
        else:
            bias_dropout_add_func = get_bias_dropout_add(self.training)

        # re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            layernorm_input = bias_dropout_add_func(
                attention_output,
                attention_bias.expand_as(residual),
                residual,
                self.hidden_dropout)

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        if self.layer_type == LayerType.decoder:
            attention_output, attention_bias = \
                self.inter_attention(layernorm_output,
                                     enc_dec_attn_mask,
                                     encoder_output=encoder_output)
            # residual connection
            if self.apply_residual_connection_post_layernorm:
                residual = layernorm_output
            else:
                residual = layernorm_input

            # re-enable torch grad to enable fused optimization.
            with torch.enable_grad():
                layernorm_input = bias_dropout_add_func(
                    attention_output,
                    attention_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)

            # Layer norm post the decoder attention
            layernorm_output = self.post_inter_attention_layernorm(layernorm_input)

        # MLP.
        moe_loss = torch.tensor(0.0, device=layernorm_output.device, dtype=layernorm_output.dtype)
        mlp_bias = torch.tensor(0.0, device=layernorm_output.device, dtype=layernorm_output.dtype)

        if self.num_experts == 1:
            mlp_output, mlp_bias = self.mlp(layernorm_output)
        else:
            mlp_output, moe_loss, _ = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        # re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            #if self.num_experts <= 1:
            output = bias_dropout_add_func(
                    mlp_output,
                    mlp_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)
            #else:
            #    output = mlp_output + residual

        if get_key_value:
            output = [output, presents]

        return output, moe_loss

class ParallelTransformerLayerPipe(ParallelTransformerLayer):
    """Extends ParallelTransformerLayer to forward attention_mask through the pipeline.

    Forward has two usages that affect attention mask communication:

    1) forward((input, attn_mask) , **kwargs) -> (output, mask)
       When the attention mask is provided as the second positional
       argument, typical pipeline behavior is used and both the output
       *and* mask are returned in a tuple. This tuple is then forwarded
       to the next stage in the pipeline.

       This version is useful if masks are dynamic.

    2) forward(input, **kwargs) -> output
       When the mask is static over all samples, it is advantageous to
       cache the mask and avoid communicating it.

       If no mask is provided, the module will query `self._args.attn_mask`
       for the mask and only return `super().forward(...)`
    """
    def __init__(self, isvit=False, **kwargs):
        super(ParallelTransformerLayerPipe, self).__init__(**kwargs)
        self.isvit = isvit


    def forward(self, inputs, **kwargs):
        assert torch.is_tensor(inputs) or isinstance(inputs, tuple)
        if torch.is_tensor(inputs) or len(inputs) == 1:
            # No attention mask forwarded, search for args.attn_mask
            if not hasattr(self, '_args'):
                self._args = get_args()
            if not self.isvit:
                hidden_states, attention_mask = inputs, self._args.attn_mask
            else:
                hidden_states, attention_mask = inputs, None
                # HACK: currently MoE model does not support pipeline parallel, so
            # here we just ignore the moe_loss returned by forward()
            return super().forward(hidden_states, attention_mask, **kwargs)[0]
        elif len(inputs) == 2:
            # Attention mask is an activation.
            hidden_states, attention_mask = inputs[0], inputs[1]
            # HACK: currently MoE model does not support pipeline parallel, so
            # here we just ignore the moe_loss returned by forward()
            return super().forward(*inputs, **kwargs)[0], attention_mask
        else:
            raise RuntimeError('Received more inputs than understood.')


class ParallelTransformer(MegatronModule):
    """Transformer class."""

    def __init__(self, init_method, output_layer_init_method,
                 layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 pre_process=True, post_process=True, num_experts=[1]):

        super(ParallelTransformer, self).__init__()
        args = get_args()
    
        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_tensor = None
        self.ds_inference = args.ds_inference

        # Store activation checkpoiting flag.
        self.checkpoint_activations = args.checkpoint_activations
        self.checkpoint_num_layers = args.checkpoint_num_layers

        # Number of layers.
        assert args.num_layers % mpu.get_pipeline_model_parallel_world_size() == 0, \
            'num_layers must be divisible by pipeline_model_parallel_size'
        self.num_layers = args.num_layers // mpu.get_pipeline_model_parallel_world_size()

        # Transformer layers.
        def build_layer(layer_number, n_e):
            return ParallelTransformerLayer(
                init_method,
                output_layer_init_method,
                layer_number,
                layer_type=layer_type,
                self_attn_mask_type=self_attn_mask_type,
                num_experts=n_e)

        if args.virtual_pipeline_model_parallel_size is not None:
            assert args.num_layers % args.virtual_pipeline_model_parallel_size == 0, \
                'num_layers_per_stage must be divisible by ' \
                'virtual_pipeline_model_parallel_size'
            # Number of layers in each model chunk is the number of layers in the stage,
            # divided by the number of model chunks in a stage.
            self.num_layers = self.num_layers // args.virtual_pipeline_model_parallel_size
            # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0]  [2]  [4]  [6]
            # Stage 1: [1]  [3]  [5]  [7]
            # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0, 1]  [4, 5]
            # Stage 1: [2, 3]  [6, 7]
            offset = mpu.get_virtual_pipeline_model_parallel_rank() * (
                args.num_layers // args.virtual_pipeline_model_parallel_size) + \
                (mpu.get_pipeline_model_parallel_rank() * self.num_layers)
        else:
            # Each stage gets a contiguous set of layers.
            offset = mpu.get_pipeline_model_parallel_rank() * self.num_layers
            
        assert len(num_experts) == 1 or len(num_experts) == args.num_layers // args.expert_interval, \
        'num_experts must be either a single value or a list of the same length as the number of MoE layers'

        # Create the list of MoE experts
        if len(num_experts) == 1:
            num_experts = num_experts * (args.num_layers // args.expert_interval)

        self.layers = []
        # Build the layers
        for i in range(self.num_layers):
            layer_num = i + 1 + offset
            if layer_num % args.expert_interval == 0:
                n_e = num_experts[(layer_num-1) // args.expert_interval]
            else:
                n_e = 1
            self.layers.append(build_layer(layer_num, n_e))

        self.layers = torch.nn.ModuleList(self.layers)

        if self.post_process:
            # Final layer norm before output.
            self.final_layernorm = LayerNorm(
                args.hidden_size,
                eps=args.layernorm_epsilon)

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint
    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def _checkpointed_forward(self, hidden_states, attention_mask,
                              encoder_output, enc_dec_attn_mask):
        """Forward method with activation checkpointing."""
        def custom(start, end):
            def custom_forward(*inputs):
                x_ = inputs[0]
                attention_mask = inputs[1]
                encoder_output = inputs[2]
                enc_dec_attn_mask = inputs[3]
                moe_losses = []
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_, moe_loss = layer(x_, attention_mask=attention_mask, encoder_output=encoder_output, enc_dec_attn_mask=enc_dec_attn_mask)
                    moe_losses.append(moe_loss)
                return (x_, *moe_losses)
            return custom_forward

        moe_losses = []
        # Make sure memory is freed.
        mpu.reset_checkpointed_activations_memory_buffer()
        l = 0
        while l < self.num_layers:
            hidden_states, *local_moe_losses = mpu.checkpoint(
                custom(l, l + self.checkpoint_num_layers),
                hidden_states, attention_mask, encoder_output, enc_dec_attn_mask)
            moe_losses.extend(local_moe_losses)
            l += self.checkpoint_num_layers

        return hidden_states, moe_losses

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False, encoder_output=None, enc_dec_attn_mask=None):

        # Checks.
        if layer_past is not None:
            assert get_key_value, \
                'for not None values in layer_past, ' \
                'expected get_key_value to be set'
        if get_key_value:
            assert not self.checkpoint_activations, \
                'get_key_value does not work with ' \
                'activation checkpointing'

        # Reza's note: DeepSpeed inference does not support transposes
        if not self.ds_inference:
            if self.pre_process:
                # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
                # If the input flag for fp32 residual connection is set, convert for float.
                if self.fp32_residual_connection:
                    hidden_states = hidden_states.transpose(0, 1).contiguous().float()
                # Otherwise, leave it as is.
                else:
                    hidden_states = hidden_states.transpose(0, 1).contiguous()
            else:
                # See set_input_tensor()
                hidden_states = self.input_tensor

            if encoder_output is not None:
                 encoder_output = encoder_output.transpose(0, 1).contiguous()

        moe_losses = []
        if self.checkpoint_activations:
            hidden_states, moe_losses = self._checkpointed_forward(hidden_states,
                                                       attention_mask,
                                                       encoder_output,
                                                       enc_dec_attn_mask)
        else:
            if get_key_value:
                presents = []
            for index in range(self.num_layers):
                layer = self._get_layer(index)
                past = None
                if layer_past is not None:
                    past = layer_past[index]
                hidden_states = layer(hidden_states,
                                      attention_mask=attention_mask,
                                      encoder_output=encoder_output,
                                      enc_dec_attn_mask=enc_dec_attn_mask,
                                      layer_past=past,
                                      get_key_value=get_key_value)
                if not self.ds_inference:
                    hidden_states, moe_loss = hidden_states
                    moe_losses.append(moe_loss)
                if get_key_value:
                    hidden_states, present = hidden_states
                    presents.append(present)

        # Final layer norm.
        if self.post_process:
            if not self.ds_inference:
                # Reverting data format change [s b h] --> [b s h].
                hidden_states = hidden_states.transpose(0, 1).contiguous()
            output = self.final_layernorm(hidden_states)
        else:
            output = hidden_states
        if get_key_value:
            output = [output, presents]

        return (output, *moe_losses)
