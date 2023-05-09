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

"""
Llama model.
Following implementation from huggingface, https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
"""

import math
from functools import partial

import torch
import torch.nn.functional as F

from megatron import get_args
from megatron import mpu
from megatron.model.module import MegatronModule, float16_to_fp32, fp32_to_float16
from megatron.model.enums import AttnMaskType, LayerType, AttnType
from megatron.model.utils import get_linear_layer, init_method_normal, scaled_init_method_normal, attention_mask_func, \
    openai_gelu, erf_gelu
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.language_model import Pooler

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.pipe import PipelineModule, LayerSpec


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos = cos[..., offset: q.shape[-2] + offset, :]
    sin = sin[..., offset: q.shape[-2] + offset, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# TODO not able to build apex cpp extention for Fused cuda kernel RMSNorm
# Steps performed, 1. copy https://github.com/NVIDIA/apex/blob/master/apex/normalization/fused_layer_norm.py, https://github.com/NVIDIA/apex/blob/master/csrc/layer_norm_cuda.cpp, https://github.com/NVIDIA/apex/blob/master/csrc/layer_norm_cuda_kernel.cu to ./megatron/model/fused_layer_norm.py, ./megatron/fused_kernels/layer_norm_cuda.cpp, ./megatron/fused_kernels/layer_norm_cuda_kernel.cu, and update ./megatron/fused_kernels/__init__.py accordingly 2. use below line to import MixedFusedRMSNorm
# torch.nn.LayerNorm is slower than apex.FusedLayerNorm for shapes typical in NLP models. For example: (512, 16, 1024) with normalization over the last dimension is slower using torch.nn.LayerNorm
# from megatron.model.fused_layer_norm import MixedFusedRMSNorm as RMSNorm # for cuda
class RMSNorm(torch.nn.Module):  # for cpu
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        hidden_states = self.weight * hidden_states

        return hidden_states


class LlamaLMHead(MegatronModule):
    """Causal LM head for Llama

    Arguments:
        vocab_size: size of vocabulary.
        hidden_size: hidden size
        gather_output: wether output logits being gathered or not.
        init_method: init method for weight initialization
    """

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 init_method,
                 parallel_output=True):
        super(LlamaLMHead, self).__init__()
        args = get_args()
        self.hidden_size = hidden_size
        self.init_method = init_method
        self.parallel_output = parallel_output

        self.lm_head = mpu.ColumnParallelLinear(input_size=self.hidden_size,
                                                output_size=vocab_size,
                                                bias=False,
                                                gather_output=not self.parallel_output,
                                                skip_bias_add=True,
                                                init_method=self.init_method, )

    def forward(self, inputs):
        logits, _ = self.lm_head(inputs)
        return logits


class LlamaLMHeadPipe(LlamaLMHead):

    def forward(self, inputs, **kwargs):
        assert torch.is_tensor(inputs) or isinstance(inputs, tuple)
        if isinstance(inputs, tuple):
            hidden_states = inputs[0]
        else:
            hidden_states = inputs

        if not hasattr(self, '_args'):
            self._args = get_args()

        if hasattr(self._args, 'attn_mask'):
            attention_mask = None
        else:
            attention_mask = inputs[1]

        logits = super().forward(hidden_states)

        # If cmd args has attn_mask, we don't forward it as an activation.
        if hasattr(self._args, 'attn_mask'):
            return logits
        else:
            return logits, attention_mask


class LlamaEmbedding(MegatronModule):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        init_method: weight initialization method
    """

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 init_method):
        super(LlamaEmbedding, self).__init__()
        args = get_args()

        self.hidden_size = hidden_size
        self.init_method = init_method

        # Word embeddings (parallel).
        self.word_embeddings = mpu.VocabParallelEmbedding(vocab_size, self.hidden_size,
                                                          init_method=self.init_method)

    def forward(self, input_ids):
        # Embeddings.
        embeddings = self.word_embeddings(input_ids)
        return embeddings


class LlamaEmbeddingPipe(LlamaEmbedding):

    def forward(self, inputs, **kwargs):
        assert torch.is_tensor(inputs) or isinstance(inputs, tuple)
        if isinstance(inputs, tuple):
            input_ids = inputs[0]
        else:
            input_ids = inputs

        if not hasattr(self, '_args'):
            self._args = get_args()

        if hasattr(self._args, 'attn_mask'):
            attention_mask = None
        else:
            attention_mask = inputs[1]

        embeddings = super().forward(input_ids)
        # If cmd args has attn_mask, we don't forward it as an activation.
        if hasattr(self._args, 'attn_mask'):
            return embeddings
        else:
            return embeddings, attention_mask


class LlamaParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to intermediate
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, init_method, output_layer_init_method, moe=False, enable_expert_tensor_parallelism=False):
        super(LlamaParallelMLP, self).__init__()
        args = get_args()
        self.init_method = init_method
        self.output_layer_init_method = output_layer_init_method

        # Project to intermediate.
        self.gate_proj = mpu.ColumnParallelLinear(
            args.hidden_size,
            args.ffn_hidden_size,
            gather_output=False,
            init_method=self.init_method,
            skip_bias_add=True,
            moe=moe,
            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism
        )

        self.up_proj = mpu.ColumnParallelLinear(
            args.hidden_size,
            args.ffn_hidden_size,
            gather_output=False,
            init_method=self.init_method,
            skip_bias_add=True,
            moe=moe,
            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism
        )

        self.activation_func = F.silu

        # Project back to h.
        self.down_proj = mpu.RowParallelLinear(
            args.ffn_hidden_size,
            args.hidden_size,
            input_is_parallel=True,
            init_method=self.output_layer_init_method,
            skip_bias_add=True,
            moe=moe,
            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism)

    def forward(self, hidden_states):
        intermediate_parallel = self.gate_proj(hidden_states)[0] * self.up_proj(hidden_states)[0]

        intermediate_parallel = self.activation_func(intermediate_parallel)

        output, _ = self.down_proj(intermediate_parallel)
        return output


class LlamaParallelAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(self, init_method,
                 output_layer_init_method, layer_number,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.causal):
        super(LlamaParallelAttention, self).__init__()

        assert attention_type == AttnType.self_attn
        assert attn_mask_type == AttnMaskType.causal

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
        self.init_method = init_method
        self.output_layer_init_method = output_layer_init_method

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
                init_method=self.init_method)

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

        ## Rotary Position Embedding
        self.rotary_emb = RotaryEmbedding(self.hidden_size_per_attention_head)

        # Output.
        self.dense = mpu.RowParallelLinear(
            projection_size,
            args.hidden_size,
            input_is_parallel=True,
            init_method=self.output_layer_init_method,
            skip_bias_add=True)

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint

    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False):
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
        # ==================================
        # Rotary Position Embedding
        # ==================================
        # [sq, b, np, hn] --> [b, np, sq, hn] TODO optimize the permute of dimension back and forth
        query_layer = query_layer.permute(1, 2, 0, 3)
        key_layer = key_layer.permute(1, 2, 0, 3)
        value_layer = value_layer.permute(1, 2, 0, 3)

        cos, sin = self.rotary_emb(value_layer, seq_len=new_tensor_shape[0])
        query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin, offset=0)

        # [b, np, sq, hn] --> [sq, b, np, hn] TODO optimize the permute of dimension back and forth
        query_layer = query_layer.permute(2, 0, 1, 3).contiguous()
        key_layer = key_layer.permute(2, 0, 1, 3).contiguous()
        value_layer = value_layer.permute(2, 0, 1, 3).contiguous()

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
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=get_accelerator().current_device_name())

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0, alpha=(1.0 / self.norm_factor))

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

        output, _ = self.dense(context_layer)

        if get_key_value:
            output = [output, present]

        return output


class LlamaParallelTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(self, init_method, output_layer_init_method,
                 layer_number,
                 self_attn_mask_type=AttnMaskType.causal):
        args = get_args()

        super(LlamaParallelTransformerLayer, self).__init__()
        self.layer_number = layer_number
        assert self_attn_mask_type == AttnMaskType.causal

        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection
        self.init_method = init_method
        self.output_layer_init_method = output_layer_init_method

        # Layernorm on the input data.
        self.input_layernorm = RMSNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon)

        # Self attention.
        self.attention = LlamaParallelAttention(
            self.init_method,
            self.output_layer_init_method,
            layer_number,
            attn_mask_type=self_attn_mask_type)

        # Layernorm on the attention output
        self.post_attention_layernorm = RMSNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon)

        # MLP
        self.mlp = LlamaParallelMLP(self.init_method, self.output_layer_init_method)

    def forward(self, hidden_states, attention_mask=None,
                layer_past=None, get_key_value=False):
        # hidden_states: [b, s, h]
        residual = hidden_states
        # Layer norm at the beginning of the transformer layer.
        hidden_states = self.input_layernorm(hidden_states)
        # Self attention.
        hidden_states = self.attention(hidden_states,
                                       attention_mask,
                                       layer_past=layer_past,
                                       get_key_value=get_key_value)

        if get_key_value:
            hidden_states, presents = hidden_states

        # Residual connection.
        hidden_states = hidden_states + residual
        residual = hidden_states

        # Layer norm post the self attention.
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP.
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        if get_key_value:
            hidden_states = [hidden_states, presents]
        return hidden_states


class LlamaParallelTransformerLayerPipe(LlamaParallelTransformerLayer):
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

    def forward(self, inputs, **kwargs):
        assert torch.is_tensor(inputs) or isinstance(inputs, tuple)
        if torch.is_tensor(inputs) or len(inputs) == 1:
            # No attention mask forwarded, search for args.attn_mask
            if not hasattr(self, '_args'):
                self._args = get_args()
            hidden_states, attention_mask = inputs, self._args.attn_mask
            return super().forward(hidden_states, attention_mask, **kwargs)
        elif len(inputs) == 2:
            # Attention mask is an activation.
            hidden_states, attention_mask = inputs[0], inputs[1]
            return super().forward(*inputs, **kwargs), attention_mask
        else:
            raise RuntimeError('Received more inputs than understood.')


class LlamaParallelTransformer(MegatronModule):
    """Transformer class."""

    def __init__(self, init_method, output_layer_init_method,
                 self_attn_mask_type=AttnMaskType.causal,
                 pre_process=True, post_process=True):

        super(LlamaParallelTransformer, self).__init__()
        args = get_args()
        assert self_attn_mask_type == AttnMaskType.causal

        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_tensor = None
        self.ds_inference = args.ds_inference
        self.init_method = init_method
        self.output_layer_init_method = output_layer_init_method

        # Store activation checkpoiting flag.
        self.checkpoint_activations = args.checkpoint_activations
        self.checkpoint_num_layers = args.checkpoint_num_layers

        # Number of layers.
        assert args.num_layers % mpu.get_pipeline_model_parallel_world_size() == 0, \
            'num_layers must be divisible by pipeline_model_parallel_size'
        self.num_layers = args.num_layers // mpu.get_pipeline_model_parallel_world_size()

        # Transformer layers.
        def build_layer(layer_number):
            return LlamaParallelTransformerLayer(
                self.init_method,
                self.output_layer_init_method,
                layer_number)

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

        self.layers = []
        # Build the layers
        for i in range(self.num_layers):
            layer_num = i + 1 + offset
            self.layers.append(build_layer(layer_num))

        self.layers = torch.nn.ModuleList(self.layers)

        if self.post_process:
            # Final layer norm before output.
            self.final_layernorm = RMSNorm(
                args.hidden_size,
                eps=args.layernorm_epsilon)

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def _checkpointed_forward(self, hidden_states, attention_mask):
        """Forward method with activation checkpointing."""

        def custom(start, end):
            def custom_forward(*inputs):
                x_ = inputs[0]
                attention_mask = inputs[1]
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_ = layer(x_, attention_mask=attention_mask)
                return x_

            return custom_forward

        # Make sure memory is freed.
        mpu.reset_checkpointed_activations_memory_buffer()
        l = 0
        while l < self.num_layers:
            hidden_states = mpu.checkpoint(
                custom(l, l + self.checkpoint_num_layers),
                hidden_states, attention_mask)
            l += self.checkpoint_num_layers

        return hidden_states

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(self, hidden_states, attention_mask, layer_past=None, get_key_value=False):

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

        if self.checkpoint_activations:
            hidden_states = self._checkpointed_forward(hidden_states, attention_mask)
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
                                      layer_past=past,
                                      get_key_value=get_key_value)
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

        return output


def CrossEntropy(output, labels):
    labels, loss_mask = labels[0], labels[1]

    args = get_args()
    losses = mpu.vocab_parallel_cross_entropy(output.contiguous().float(), labels)
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    return loss


class LlamaModelPipe(PipelineModule, MegatronModule):
    """llama Language model."""

    def __init__(self, parallel_output=True):
        args = get_args()

        self.init_method = init_method_normal(args.init_method_std)
        self.output_layer_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)
        self.parallel_output = parallel_output

        self.specs = []

        def _to_float16(inputs):
            if args.fp16:
                return fp32_to_float16(inputs, lambda v: v.half())
            elif args.bf16:
                return fp32_to_float16(inputs, lambda v: v.bfloat16())
            else:
                return inputs

        self.specs.append(_to_float16)

        # Embedding layer
        self.specs.append(LayerSpec(LlamaEmbeddingPipe, hidden_size=args.hidden_size, vocab_size=args.padded_vocab_size,
                                    init_method=self.init_method, ))

        if args.fp32_residual_connection:
            self.specs.append(lambda x: x.transpose(0, 1).contiguous().float())
        else:
            self.specs.append(lambda x: x.transpose(0, 1).contiguous())

        for layer_idx in range(args.num_layers):
            self.specs.append(
                LayerSpec(LlamaParallelTransformerLayerPipe
                          , init_method=self.init_method
                          , output_layer_init_method=self.output_layer_init_method
                          , layer_number=layer_idx))

        # Undo data format change
        self.specs.append(lambda x: x.transpose(0, 1).contiguous())

        # Final layernorm after transformer layers
        self.specs.append(LayerSpec(RMSNorm, args.hidden_size, eps=args.layernorm_epsilon))

        self.specs.append(
            LayerSpec(LlamaLMHeadPipe, hidden_size=args.hidden_size, vocab_size=padded_vocab_size,
                      init_method=self.init_method, parallel_output=self.parallel_output)
        )

        # Convert to fp32 if needed
        if args.fp16 or args.bf16:
            self.specs.append(float16_to_fp32)

        if args.checkpoint_activations:
            interval = args.checkpoint_num_layers
        else:
            interval = 0

        from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
        topo = PipeModelDataParallelTopology(num_pp=mpu.get_pipeline_model_parallel_world_size(),
                                             num_mp=mpu.get_tensor_model_parallel_world_size(),
                                             num_dp=mpu.get_data_parallel_world_size())

        super().__init__(layers=self.specs,
                         loss_fn=CrossEntropy,
                         topology=topo,
                         activation_checkpoint_interval=interval,
                         partition_method='type:transformer')


class LlamaModel(MegatronModule):
    """llama Language model."""

    def __init__(self, pre_process, post_process, parallel_output=True, add_pooler=False):
        super(LlamaModel, self).__init__()
        args = get_args()
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.hidden_size = args.hidden_size
        self.pre_process = pre_process
        self.post_process = post_process
        self.parallel_output = parallel_output
        self.add_pooler = add_pooler
        self.init_method = init_method_normal(args.init_method_std)
        self.output_layer_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)
        self.self_attn_mask_type = AttnMaskType.causal
        self.padded_vocab_size = args.padded_vocab_size

        if self.pre_process:
            self.embedding = LlamaEmbedding(hidden_size=args.hidden_size,
                                            init_method=self.init_method,
                                            vocab_size=self.padded_vocab_size)

        # Transformer.
        self.transformer = LlamaParallelTransformer(
            self.init_method,
            self.output_layer_init_method,
            self_attn_mask_type=self.self_attn_mask_type,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

        if self.post_process:
            # Pooler.
            if self.add_pooler:
                self.pooler = Pooler(self.hidden_size, self.init_method)

            self.lm_head = LlamaLMHead(hidden_size=args.hidden_size,
                                       vocab_size=self.padded_vocab_size,
                                       init_method=self.init_method,
                                       parallel_output=self.parallel_output)

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.transformer.set_input_tensor(input_tensor)

    def forward(self, input_ids, attention_mask, labels=None, layer_past=None, get_key_value=False):
        args = get_args()

        if self.pre_process:
            hidden_states = self.embedding(input_ids)
        else:
            hidden_states = input_ids

        # decoder
        hidden_states = self.transformer(hidden_states, attention_mask, layer_past=layer_past,
                                         get_key_value=get_key_value)

        if self.post_process:
            if get_key_value:
                hidden_states, presents = hidden_states

            if self.add_pooler:
                hidden_states = self.pooler(hidden_states, pooling_sequence_index)

            hidden_states = self.lm_head(hidden_states)

            if labels is None:
                if get_key_value:
                    return [hidden_states, presents]
                return hidden_states

            else:
                if self.fp16_lm_cross_entropy:
                    assert hidden_states.dtype == torch.half
                    loss = mpu.vocab_parallel_cross_entropy(hidden_states, labels)
                else:
                    loss = mpu.vocab_parallel_cross_entropy(hidden_states.float(), labels)
                return loss

        return hidden_states
