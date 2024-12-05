# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Transformer."""
from contextlib import nullcontext
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing import Optional
import deepspeed.comm as dist

from megatron import get_timers, get_args, get_retro_args, core, get_num_microbatches
from .module import MegatronModule
from megatron.core import parallel_state, tensor_parallel, mpu
from megatron.core.enums import ModelType
from megatron.model import LayerNorm, RMSNorm
from megatron.model.enums import AttnMaskType, LayerType, AttnType
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.fused_bias_gelu import bias_gelu_impl
from megatron.model.rotary_pos_embedding import apply_rotary_pos_emb
from megatron.model.utils import attention_mask_func, openai_gelu, erf_gelu
import deepspeed
from deepspeed.moe.layer import MoE
from deepspeed.accelerator import get_accelerator
from deepspeed.sequence.fpdt_layer import FPDT_FFN, FPDT_Attention

try:
    from deepspeed.sequence.layer import DistributedAttention
    dist_attn_supported = True
except ImportError:
    dist_attn_supported = False

try:
    from einops import rearrange
except ImportError:
    rearrange = None

try:
    # FlashAttention (1.x)
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func
except ImportError:
    flash_attn_unpadded_func = None

try:
    from flash_attn.flash_attn_triton import flash_attn_func
except ImportError:
    flash_attn_func = None

try:
    # FlashAttention-2
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None

FlashAttentionBuilder = get_accelerator().get_op_builder("FlashAttentionBuilder")
flash_attn_builder = None


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

class DropPath(MegatronModule):
    """Drop paths (Stochastic Depth) per sample
    (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_state):
        if self.drop_prob == 0. or not self.training:
            return hidden_state
        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        # hidden_state: [s, b, h]
        shape = (1,) + (hidden_state.shape[1],) + (1,) * (hidden_state.ndim - 2)
        random_tensor = keep_prob + \
            torch.rand(shape, dtype=hidden_state.dtype, device=hidden_state.device)
        random_tensor.floor_()  # binarize
        output = hidden_state.div(keep_prob) * random_tensor
        return output

class ParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config, moe=False, enable_expert_tensor_parallelism=False):
        super(ParallelMLP, self).__init__()
        args = get_args()

        self.add_bias = config.add_bias_linear

        ffn_hidden_size = config.ffn_hidden_size
        if config.gated_linear_unit:
            ffn_hidden_size *= 2

        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            ffn_hidden_size,
            config=config,
            init_method=config.init_method,
            bias=self.add_bias,
            gather_output=False,
            skip_bias_add=True,
            moe=moe,
            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism
        )

        self.bias_gelu_fusion = False
        self.activation_func = None
        self.swiglu = args.swiglu

        if args.openai_gelu:
            self.activation_func = openai_gelu
        elif args.onnx_safe:
            self.activation_func = erf_gelu
        elif args.swiglu:
            def swiglu(x):
                x = torch.chunk(x, 2, dim=-1)
                return F.silu(x[0]) * x[1]
            self.activation_func = swiglu
        elif args.squared_relu:
            def squared_relu(x):
                return torch.pow(F.relu(x), 2)
            self.activation_func = squared_relu
        else:
            self.bias_gelu_fusion = args.bias_gelu_fusion
            self.activation_func = F.gelu

        # Project back to h.
        self.dense_4h_to_h = tensor_parallel.RowParallelLinear(
            config.ffn_hidden_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=self.add_bias,
            input_is_parallel=True,
            moe=moe,
            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism
        )
        
        self.ds_sequence_parallel_fpdt = args.ds_sequence_parallel_fpdt
        if self.ds_sequence_parallel_fpdt:
            self.fpdt_FFN_chunk_size = int(args.ds_sequence_parallel_fpdt_chunk_size // mpu.get_sequence_parallel_world_size() // 2)

    def forward(self, hidden_states):

        if self.ds_sequence_parallel_fpdt:
            output, output_bias = FPDT_FFN.apply(hidden_states, self.dense_h_to_4h.weight, self.dense_h_to_4h.bias, self.dense_4h_to_h.weight, self.dense_4h_to_h.bias, self.add_bias, self.fpdt_FFN_chunk_size)
        else:
            # [s, b, 4hp]
            intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

            if self.bias_gelu_fusion:
                assert self.add_bias is True
                # DeepSpeed FLOPS profiler temporarily substitues functions like F.gelu to calculate the throughput
                assert hasattr(self, "__flops__") or self.activation_func == F.gelu
                intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
            else:
                if bias_parallel is not None:
                    intermediate_parallel = intermediate_parallel + bias_parallel
                intermediate_parallel = self.activation_func(intermediate_parallel)

            # [s, b, h]
            output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias

class SwitchMLP(MegatronModule):
    """
    Routes input to one of N MLP "experts"
    """
    def __init__(self, config):
        super(SwitchMLP, self).__init__()
        args = get_args()
        self.router = torch.nn.Linear(config.hidden_size, args.num_experts_switch)
        self.experts = torch.nn.ModuleList()
        for i in range(args.num_experts_switch):
            self.experts.append(ParallelMLP(config))

    def forward(self, hidden_states):
        # hidden_states: [s, b, h]
        s = hidden_states.size(0)
        b = hidden_states.size(1)
        h = hidden_states.size(2)
        route = self.router(hidden_states)
        route = torch.nn.functional.softmax(route, dim=2)
        max_prob, max_ind = torch.max(route, dim=2)
        max_prob = torch.unsqueeze(max_prob, 2) # [s b 1]

        # TODO (rprenger) TODO this could be made easier to read
        # Converting [s, b, h] to [s*b, h].
        # Each vector could be routed differently
        hidden_states = hidden_states.view(-1, hidden_states.size(2)) # [s*b h]
        max_prob = max_prob.view(-1, max_prob.size(2)) # [s*b 1]
        max_ind = max_ind.view(-1) # [s*b]

        output_total = torch.empty_like(hidden_states)
        output_bias_total = torch.empty_like(hidden_states)
        #TODO (rprenger) This does each expert in serial, but it could be parallelized

        for expert_num, expert in enumerate(self.experts):
            local_indices = (max_ind == expert_num).nonzero()
            hidden = hidden_states[local_indices,:]
            output, output_bias = expert(hidden)
            output_bias = output_bias.expand_as(output)
            output_total[local_indices,:] = output
            output_bias_total[local_indices,:] = output_bias

        output_total = output_total*max_prob
        output_bias_total = output_bias_total*max_prob
        output_total = output_total.view(s, b, h)
        output_bias_total = output_bias_total.view(s, b, h)

        return output_total, output_bias_total


class CoreAttention(MegatronModule):

    def __init__(self, layer_number, config,
                 attn_mask_type=AttnMaskType.padding):
        super(CoreAttention, self).__init__()
        self.fp16 = config.fp16
        self.bf16 = config.bf16

        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.sequence_parallel = config.sequence_parallel

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        seq_parallel_world_size = 1
        if parallel_state.sequence_parallel_is_initialized():
            seq_parallel_world_size = parallel_state.get_sequence_parallel_world_size()
        world_size = seq_parallel_world_size if seq_parallel_world_size > 1 else parallel_state.get_tensor_model_parallel_world_size()

        self.hidden_size_per_partition = core.utils.divide(projection_size,
                                                           world_size)
        self.hidden_size_per_attention_head = core.utils.divide(
            projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = core.utils.divide(
            config.num_attention_heads, world_size)

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16, self.bf16,
            self.attn_mask_type,
            config.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(config.attention_dropout)

    def forward(self, query_layer, key_layer,
                value_layer, attention_mask):

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

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(
            (output_size[0]*output_size[1], output_size[2], output_size[3]),
            query_layer.dtype, "mpu")

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer.transpose(0, 1),   # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0, alpha=(1.0/self.norm_factor))

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        if not self.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
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

        return context_layer


class FlashSelfAttention(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 device=None, dtype=None):
        super().__init__()
        assert flash_attn_unpadded_func is not None or flash_attn_varlen_func is not None or flash_attn_builder is not None, \
            ('Please install FlashAttention first, e.g., with pip install flash-attn or implement your own flash attention')
        assert rearrange is not None, 'Please install einops first, e.g., with pip install einops'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

        # Use FlashAttention-2 when args.use_flash_attn_v2 is True
        args = get_args()
        self.use_flash_attn_builder_v1 = False
        self.use_flash_attn_builder_v2 = False
        self.use_flash_attn = False
        if args.use_flash_attn_builder:
            if hasattr(flash_attn_builder, 'flash_attn_func'):
                self.flash_attn_func = flash_attn_builder.flash_attn_func
                self.use_flash_attn_builder_v1 = True
            else:
                self.flash_attn_func = flash_attn_builder.flash_attn_func_v2
                self.use_flash_attn_builder_v2 = True
        else:
            self.flash_attn_func = flash_attn_varlen_func if args.use_flash_attn_v2 else flash_attn_unpadded_func
            self.use_flash_attn = True

    def forward(self, q, k, v):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """

        assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q,k,v)))
        assert all((get_accelerator().on_accelerator(i) for i in (q, k, v)))

        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = k.shape[1]

        if self.use_flash_attn:
            q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                        device=q.device)
        elif self.use_flash_attn_builder_v1:
            q, k, v = [rearrange(x, 'b s h d -> b h s d').contiguous() for x in [q, k, v]]
        else:
            # use_flash_attn_builder_v2
            q, k, v = [rearrange(x, 'b s h d -> b h s d') for x in [q, k, v]]

        if self.training:
            # during training q,k,v always have same seqlen
            assert seqlen_k == seqlen_q

            is_causal = self.causal
            cu_seqlens_k = cu_seqlens_q if get_accelerator().device_name() == 'cuda' else None
            dropout_p = self.dropout_p
        else:
            # turn off FA causal mask after first inference autoregressive iteration
            # only on first autoregressive step q,k,v have same seqlen
            is_causal = seqlen_q == seqlen_k
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                        device=q.device) if get_accelerator().device_name() == 'cuda' else None
            dropout_p = 0

        if self.use_flash_attn:
            output = self.flash_attn_func(
                q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k,
                dropout_p,
                softmax_scale=self.softmax_scale, causal=is_causal
            )
        else:
            # use_flash_attn_builder
            output = self.flash_attn_func(
                q, k, v, self.dropout_p, self.softmax_scale, is_causal
            )

        if self.use_flash_attn:
            output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        elif self.use_flash_attn_builder_v1:
            output = rearrange(output, 'b h s d -> b s h d').contiguous()
        else:
            # use_flash_attn_builder_v2:
            output = rearrange(output, 'b h s d -> b s h d')

        return output

class FlashSelfAttentionTriton(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 device=None, dtype=None):
        super().__init__()
        assert flash_attn_func is not None, ('Triton version of FlashAttention is not installed.')
        assert rearrange is not None, 'Please install einops first, e.g., with pip install einops'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, q, k, v):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """

        assert q.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda
        q, k, v = [rearrange(x, 's b ... -> b s ...').contiguous()
                       for x in (q, k, v)]
        
        output = flash_attn_func(q, k, v, None, self.causal)
        output = rearrange(output, 'b s h d -> s b (h d)').contiguous()
        return output

class ParallelAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, config, layer_number,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.padding):
        super(ParallelAttention, self).__init__()
        args = get_args()
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.params_dtype = config.params_dtype
        self.sequence_parallel = config.sequence_parallel
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.use_gqa = (self.num_attention_heads != self.num_key_value_heads)

        self.use_flash_attn = (args.use_flash_attn_v1 or args.use_flash_attn_triton or args.use_flash_attn_v2 or \
            args.use_flash_attn_builder) \
            and attention_type == AttnType.self_attn \
            and self.attn_mask_type == AttnMaskType.causal
        self.use_flash_attn_triton = args.use_flash_attn_triton
        if self.use_flash_attn:
            global flash_attn_builder
            try:
                flash_attn_builder = FlashAttentionBuilder().load()
            except TypeError:
                flash_attn_builder = None

            if args.use_flash_attn_v1:
                assert flash_attn_unpadded_func != None, "Cannot import FlashAttention v1 "
            if args.use_flash_attn_v2:
                assert flash_attn_varlen_func != None, "Cannot import FlashAttention v2 "
            if args.use_flash_attn_triton:
                assert flash_attn_func != None, "Cannot import FlashAttention triton "
            if args.use_flash_attn_builder:
                assert flash_attn_builder != None, "Cannot find FlashAttention op builder "

            assert attention_type == AttnType.self_attn, ('FlashAttention code path only supports '
                                                          'self-attention for now')
            assert self.attn_mask_type == AttnMaskType.causal, ('FlashAttention code path only '
                                                                'supports causal mask for now')
            if rearrange is None:
                raise ImportError('einops is not installed, please install with pip install einops')

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = core.utils.divide(
            projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = core.utils.divide(
            config.num_attention_heads, world_size)

        # Per GQA head and per partition values
        self.num_key_value_heads_per_partition = core.utils.divide(
            config.num_key_value_heads, world_size)
        self.num_key_value_groups = core.utils.divide(
            config.num_attention_heads, config.num_key_value_heads)
        kv_projection_size = config.kv_channels * config.num_key_value_heads
        assert self.hidden_size_per_attention_head == core.utils.divide(
            kv_projection_size, config.num_key_value_heads)

        self.projection_size = projection_size
        self.kv_projection_size = kv_projection_size
        self.head_dim = config.kv_channels

        # Strided linear layer.
        if attention_type == AttnType.self_attn:
            self.query_key_value = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                projection_size + 2 * kv_projection_size,
                config=config,
                init_method=config.init_method,
                bias=args.add_bias_linear,
                gather_output=False)
        else:
            assert attention_type == AttnType.cross_attn
            self.query = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                projection_size,
                config=config,
                init_method=config.init_method,
                bias=config.add_bias_linear,
                gather_output=False)


            self.key_value = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                2 * projection_size,
                config=config,
                init_method=config.init_method,
                bias=config.add_bias_linear,
                gather_output=False)

        # Currently FlashAttention only works with causal mask
        if self.use_flash_attn_triton:
            local_attn = FlashSelfAttentionTriton(causal=True, attention_dropout=args.attention_dropout)
        elif self.use_flash_attn:
            local_attn = FlashSelfAttention(causal=True, attention_dropout=config.attention_dropout)
        else:
            local_attn = CoreAttention(self.layer_number, config, self.attn_mask_type)

        self.enable_ds_sequence_parallel = parallel_state.get_sequence_parallel_world_size() > 1 \
                                           or args.force_ds_sequence_parallel
        if self.enable_ds_sequence_parallel:
            assert dist_attn_supported, 'Distributed attention is not supported in this DeepSpeed version'
            assert args.num_attention_heads % parallel_state.get_sequence_parallel_world_size() == 0
            self.dist_attn = DistributedAttention(
                local_attn, 
                parallel_state.get_sequence_parallel_group(), 
                gather_idx=1 if args.use_flash_attn_v1 or args.use_flash_attn_v2 else 0) 
            # flash_attn_cuda assumes [b, s, nh, hd] layout, we need to make sure all2all gathers into the correct sequence dimension.
        else:
            if self.use_flash_attn:
                self.core_attention_flash = local_attn
            else:
                self.core_attention = local_attn
                self.checkpoint_core_attention = config.recompute_granularity == 'selective'

        # Output.
        self.dense = tensor_parallel.RowParallelLinear(
            projection_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=args.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True)


    def _checkpointed_attention_forward(self, query_layer, key_layer,
                                        value_layer, attention_mask,
                                        rotary_pos_emb=None):
        """Forward method with activation checkpointing."""
        def custom_forward(*inputs):
            query_layer = inputs[0]
            key_layer = inputs[1]
            value_layer = inputs[2]
            attention_mask = inputs[3]
            output_ = self.core_attention(query_layer, key_layer,
                                          value_layer, attention_mask)
            return output_

        q_pos_emb, k_pos_emb = (None, None) if rotary_pos_emb is None \
            else rotary_pos_emb

        hidden_states = tensor_parallel.checkpoint(
            custom_forward,
            False, query_layer, key_layer, value_layer, attention_mask,
            q_pos_emb, k_pos_emb)

        return hidden_states

    def _allocate_memory(self, inference_max_sequence_len, batch_size):
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
            dtype=self.params_dtype,
            device=get_accelerator().current_device_name())

    def repeat_kv(self, hidden_states, n_rep):
        slen, batch, num_key_value_heads_per_partition, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        elif num_key_value_heads_per_partition == 1:
            # If no of KV heads is 1 then just perform expand operation
            # instead of unsqueeze, expand and reshape to match query states.
            return hidden_states.expand(slen, batch, n_rep, head_dim)
        else:
            hidden_states = hidden_states[:, :, :, None, :].expand(
                slen, batch, num_key_value_heads_per_partition, n_rep, head_dim)
            return hidden_states.reshape(slen, batch,
                                         num_key_value_heads_per_partition * n_rep,
                                         head_dim)
                                     
    def split_tensor(self, mixed_x_layer):
        query_layer, key_layer, value_layer = torch.split(mixed_x_layer, [self.num_key_value_groups, 1, 1], dim=-2)
        query_layer = query_layer.reshape(mixed_x_layer.shape[:2] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head))
        key_layer = torch.squeeze(key_layer, -2)
        value_layer = torch.squeeze(value_layer, -2)

        return query_layer, key_layer, value_layer

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, inference_params=None,
                rotary_pos_emb=None):
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        is_first_step = False
        if inference_params:
            if self.layer_number not in inference_params.key_value_memory_dict:
                inf_max_seq_len = inference_params.max_sequence_len
                inf_max_batch_size = inference_params.max_batch_size
                inference_key_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size)
                inference_value_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size)
                inference_params.key_value_memory_dict[self.layer_number] = (
                    inference_key_memory, inference_value_memory)
                is_first_step = True
            else:
                inference_key_memory, inference_value_memory = \
                    inference_params.key_value_memory_dict[self.layer_number]

        # =====================
        # Query, Key, and Value
        # =====================
        if self.attention_type == AttnType.self_attn:
            # Attention heads [sq, b, h] --> [sq, b, ((nq + 2 * nkv) * hn)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)
            
            if self.enable_ds_sequence_parallel:
                assert self.projection_size == self.kv_projection_size
                seq_len, bs = mixed_x_layer.shape[0], mixed_x_layer.shape[1]
                query_layer = mixed_x_layer[:, :, :self.projection_size].reshape(seq_len, bs, -1, self.head_dim)
                key_layer = mixed_x_layer[:, :, self.projection_size:self.projection_size+self.kv_projection_size].reshape(seq_len, bs, -1, self.head_dim)
                value_layer = mixed_x_layer[:, :, self.projection_size+self.kv_projection_size:].reshape(seq_len, bs, -1, self.head_dim)
            if self.sequence_parallel or not self.enable_ds_sequence_parallel:
                seq_len, bs = mixed_x_layer.shape[0], mixed_x_layer.shape[1]
                each_hidden_size = mixed_x_layer.shape[-1] // 3
                query_layer = mixed_x_layer[:, :, :each_hidden_size].reshape(seq_len, bs, -1, self.head_dim)
                key_layer = mixed_x_layer[:, :, each_hidden_size:each_hidden_size+each_hidden_size].reshape(seq_len, bs, -1, self.head_dim)
                value_layer = mixed_x_layer[:, :, each_hidden_size+each_hidden_size:].reshape(seq_len, bs, -1, self.head_dim)

            # Repeat kv
            if self.use_gqa:
                key_layer = self.repeat_kv(key_layer, self.num_key_value_groups)
                value_layer = self.repeat_kv(value_layer,
                                             self.num_key_value_groups)
        else:
            assert not self.use_gqa, 'GQA + cross-attn not tested yet'

            # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            mixed_kv_layer, _ = self.key_value(encoder_output)

            # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 2 * self.hidden_size_per_attention_head)
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
            (key_layer,
             value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_kv_layer, 2)

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

        # duplicate the pos_emb for self attention
        if rotary_pos_emb is not None:
            if isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = rotary_pos_emb
            else:
                rotary_pos_emb = ((rotary_pos_emb,) * 2)

        if inference_params:
            batch_start = inference_params.batch_size_offset
            batch_end = batch_start + key_layer.size(1)
            assert batch_end <= inference_key_memory.size(1)
            sequence_start = inference_params.sequence_len_offset
            sequence_end = sequence_start + key_layer.size(0)
            assert sequence_end <= inference_key_memory.size(0)
            # Copy key and values.
            inference_key_memory[sequence_start:sequence_end,
                                 batch_start:batch_end, ...] = key_layer
            inference_value_memory[sequence_start:sequence_end,
                                   batch_start:batch_end, ...] = value_layer
            key_layer = inference_key_memory[
                :sequence_end, batch_start:batch_end, ...]
            value_layer = inference_value_memory[
                :sequence_end, batch_start:batch_end, ...]


            # adjust the key rotary positional embedding
            if rotary_pos_emb is not None:
                q_pos_emb, k_pos_emb = rotary_pos_emb
                # need to cross check this condition during inference
                # if not set_inference_key_value_memory:
                if not is_first_step:
                    # In inference, we compute one token at a time.
                    # Select the correct positional embedding
                    # (only the last token in the sequence)
                    q_pos_emb = q_pos_emb[sequence_end - 1 : sequence_end]
                else:
                    # In the first forward pass of inference,
                    # we use the entire provided prefix.
                    # q_pos_emb here has the rope embeddings of the entire
                    # prefix + to-be-generated output so
                    # we slice to just the prefix.
                    q_pos_emb = q_pos_emb[:sequence_end, :, :, :]
                k_pos_emb = k_pos_emb[:sequence_end, :, :, :]
                rotary_pos_emb = (q_pos_emb, k_pos_emb)


        # ==================================
        # core attention computation
        # ==================================

        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            if not self.enable_ds_sequence_parallel:
                q_pos_emb, k_pos_emb = rotary_pos_emb
                query_layer = apply_rotary_pos_emb(query_layer, q_pos_emb)
                key_layer = apply_rotary_pos_emb(key_layer, k_pos_emb)
            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

        if self.enable_ds_sequence_parallel:
            batch_dim_idx = 1
            if self.use_flash_attn:
                if not self.use_flash_attn_triton:
                    query_layer, key_layer, value_layer = [rearrange(x, 's b ... -> b s ...').contiguous()
                            for x in (query_layer, key_layer, value_layer)]
                    batch_dim_idx = 0

                context_layer = self.dist_attn(query_layer, key_layer, value_layer, batch_dim_idx, rotary_pos_emb=rotary_pos_emb)

                if not self.use_flash_attn_triton:
                    context_layer = rearrange(context_layer, 'b s h d -> s b (h d)').contiguous()
            else:
                context_layer = self.dist_attn(query_layer, key_layer, value_layer, attention_mask)
        else:
            if self.use_flash_attn:
                if not self.use_flash_attn_triton:
                    query_layer, key_layer, value_layer = [rearrange(x, 's b ... -> b s ...').contiguous()
                            for x in (query_layer, key_layer, value_layer)]

                if self.sequence_parallel:
                    context_layer = self.core_attention_flash(query_layer, key_layer, value_layer)
                else:
                    with tensor_parallel.get_cuda_rng_tracker().fork():
                        context_layer = self.core_attention_flash(query_layer, key_layer, value_layer)

                if not self.use_flash_attn_triton:
                    context_layer = rearrange(context_layer, 'b s h d -> s b (h d)').contiguous()
            else:
                if self.checkpoint_core_attention:
                    context_layer = self._checkpointed_attention_forward(
                        query_layer, key_layer, value_layer, attention_mask)
                else:
                    context_layer = self.core_attention(
                        query_layer, key_layer, value_layer, attention_mask)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        return output, bias


def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Optional[Tensor], Tensor, float, bool) -> Tensor
    if bias is not None:
        x = x + bias
    out = torch.nn.functional.dropout(x, p=prob, training=training)
    out = residual + out
    return out


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)
    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(x: torch.Tensor,
                                 bias: Optional[torch.Tensor],
                                 residual: torch.Tensor,
                                 prob: float) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(x: torch.Tensor,
                                     bias: Optional[torch.Tensor],
                                     residual: torch.Tensor,
                                     prob: float) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, False)


def launch_chunk_attention(args, config):
    assert args.num_attention_heads % parallel_state.get_sequence_parallel_world_size() == 0
            
    projection_size = config.kv_channels * config.num_attention_heads
    kv_projection_size = config.kv_channels * config.num_key_value_heads

    qkv_linear_weight = Parameter(torch.empty(
        projection_size + 2 * kv_projection_size, config.hidden_size,
        device=get_accelerator().current_device_name(), dtype=config.params_dtype))
    if config.perform_initialization:
        tensor_parallel.layers._initialize_affine_weight_gpu(qkv_linear_weight, config.init_method,
                                        partition_dim=0)
    qkv_linear_bias = None
    if args.add_bias_linear:
        qkv_linear_bias = Parameter(torch.empty(
                projection_size + 2 * kv_projection_size,
                device=get_accelerator().current_device_name(),
                dtype=config.params_dtype))
        if config.perform_initialization:
            # Always initialize bias to zero.
            with torch.no_grad():
                qkv_linear_bias.zero_()

    qkv_dense_weight = Parameter(torch.empty(
        config.hidden_size, projection_size,
        device=get_accelerator().current_device_name(), dtype=config.params_dtype))
    if config.perform_initialization:
        tensor_parallel.layers._initialize_affine_weight_gpu(qkv_dense_weight, config.output_layer_init_method,
                                        partition_dim=0)
    qkv_dense_bias = None
    if args.add_bias_linear:
        qkv_dense_bias = Parameter(torch.empty(
                config.hidden_size,
                device=get_accelerator().current_device_name(),
                dtype=config.params_dtype))
        if config.perform_initialization:
            # Always initialize bias to zero.
            with torch.no_grad():
                qkv_dense_bias.zero_()
    
    return FPDT_Attention(config,
                            qkv_linear_weight,
                            qkv_linear_bias, 
                            qkv_dense_weight,
                            qkv_dense_bias,
                            sequence_process_group=parallel_state.get_sequence_parallel_group(), 
                            gather_idx=1 if args.use_flash_attn_v1 or args.use_flash_attn_v2 else 0,
                            return_bias=True,
                            chunk_size=args.ds_sequence_parallel_fpdt_chunk_size,
                            enable_offloading=args.ds_sequence_parallel_fpdt_offloading)


class ParallelTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config,
                 layer_number, layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 drop_path_rate=0., num_experts=1):
        # retriever=None):
        args = get_args()

        super(ParallelTransformerLayer, self).__init__()
        self.layer_number = layer_number
        self.layer_type = layer_type

        self.apply_residual_connection_post_layernorm \
            = config.apply_residual_connection_post_layernorm

        self.bf16 = config.bf16
        self.fp32_residual_connection = config.fp32_residual_connection

        # Layernorm on the input data.
        if args.normalization == 'layernorm':
            if get_accelerator().device_name() == 'cuda':
                self.input_layernorm = LayerNorm(
                    config.hidden_size,
                    eps=config.layernorm_epsilon,
                    no_persist_layer_norm=args.no_persist_layer_norm,
                    sequence_parallel=config.sequence_parallel,
                    apply_layernorm_1p=args.apply_layernorm_1p,
                    mem_efficient_ln=args.mem_efficient_ln)
            else:
                self.input_layernorm = LayerNorm(
                    config.hidden_size,
                    eps=config.layernorm_epsilon)
        else:
            self.input_layernorm = RMSNorm(config.hidden_size, config.layernorm_epsilon,
                                           sequence_parallel=config.sequence_parallel)
        # Self attention.
        self.ds_sequence_parallel_fpdt = args.ds_sequence_parallel_fpdt
        self.ds_sequence_parallel_fpdt_offloading = args.ds_sequence_parallel_fpdt_offloading
        if args.ds_sequence_parallel_fpdt:            
            self.self_attention = launch_chunk_attention(args, config)
        else:
            self.self_attention = ParallelAttention(
                config,
                layer_number,
                attention_type=AttnType.self_attn,
                attn_mask_type=self_attn_mask_type)
            
        self.hidden_dropout = config.hidden_dropout
        self.bias_dropout_fusion = config.bias_dropout_fusion
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else None

        # Layernorm on the attention output
        if args.normalization == 'layernorm':
            if get_accelerator().device_name() == 'cuda':
                self.post_attention_layernorm = LayerNorm(
                    config.hidden_size,
                    eps=config.layernorm_epsilon,
                    no_persist_layer_norm=not config.persist_layer_norm,
                    sequence_parallel=config.sequence_parallel,
                    apply_layernorm_1p=args.apply_layernorm_1p,
                    mem_efficient_ln=args.mem_efficient_ln)
            else:
                self.post_attention_layernorm = LayerNorm(
                    config.hidden_size,
                    eps=config.layernorm_epsilon)
        else:
            self.post_attention_layernorm = RMSNorm(config.hidden_size, config.layernorm_epsilon,
                                                    sequence_parallel=config.sequence_parallel)
            # Cross attention.
        if self.layer_type in (LayerType.decoder,
                               LayerType.retro_decoder,
                               LayerType.retro_decoder_with_retriever,
                               LayerType.retro_encoder):
            self.inter_attention = ParallelAttention(
                config,
                layer_number,
                attention_type=AttnType.cross_attn)
            # Layernorm on the attention output.
            if args.normalization == 'layernorm':
                self.post_inter_attention_layernorm = LayerNorm(
                    config.hidden_size,
                    eps=config.layernorm_epsilon,
                    no_persist_layer_norm=not config.persist_layer_norm,
                    sequence_parallel=config.sequence_parallel,
                    apply_layernorm_1p=args.apply_layernorm_1p,
                    mem_efficient_ln=args.mem_efficient_ln)
            else:
                self.post_inter_attention_layernorm = RMSNorm(config.hidden_size,
                                                              config.layernorm_epsilon,
                                                              sequence_parallel=config.sequence_parallel)

        # MLP
        self.num_experts = num_experts
        if args.num_experts_switch is not None:
            self.mlp = SwitchMLP(config) # Megatron-LM's MoE
        else:
            if self.num_experts <= 1: # dense, not MoE
                self.mlp = ParallelMLP(config)
            else: # DeepSpeed's MoE
                enable_expert_tensor_parallelism = args.enable_expert_tensor_parallelism
                self.mlp = MoE(args.hidden_size,
                               ParallelMLP(config,
                                           moe=True,
                                           enable_expert_tensor_parallelism=enable_expert_tensor_parallelism),
                               num_experts=self.num_experts,
                               ep_size=args.moe_expert_parallel_size,
                               k=args.topk,
                               use_residual=(args.mlp_type == 'residual'),
                               capacity_factor=args.moe_train_capacity_factor,
                               eval_capacity_factor=args.moe_eval_capacity_factor,
                               min_capacity=args.moe_min_capacity,
                               drop_tokens=args.moe_token_dropping,
                               use_tutel=args.use_tutel,
                               enable_expert_tensor_parallelism=enable_expert_tensor_parallelism,
                               top2_2nd_expert_sampling=args.moe_top2_2nd_expert_sampling)

        # Set bias+dropout+add fusion grad_enable execution handler.
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        self.bias_dropout_add_exec_handler = \
                nullcontext if use_nvfuser else torch.enable_grad

        if args.retro_add_retriever:
            retro_args = get_retro_args()
            self.retro_num_neighbors = args.retro_num_neighbors
            self.retro_chunk_length = retro_args.retro_gpt_chunk_length
            self.retro_retrieved_length = retro_args.retro_gpt_retrieved_length

        # Retriever (bi-directional transformer with cross attention)
        if layer_type == LayerType.retro_decoder_with_retriever:
            self.retriever = ParallelTransformer(
                init_method,
                output_layer_init_method,
                model_type=ModelType.retro_encoder,
                self_attn_mask_type=AttnMaskType.padding,
                pre_process=True,
                post_process=False,
            )
            self._retriever_key = 'retriever'
        else:
            self.retriever = None

    def default_decoder_cross_attention(self,
                                        encoder_output,
                                        enc_dec_attn_mask,
                                        layernorm_input,
                                        layernorm_output,
                                        bias_dropout_add_func):
        '''Cross attention for a standard encoder-decoder model.'''

        # Attention.
        attention_output, attention_bias = \
            self.inter_attention(layernorm_output,
                                 enc_dec_attn_mask,
                                 encoder_output=encoder_output)

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        if attention_bias is not None:
            attention_bias = attention_bias.expand_as(residual)

        # Bias-dropout-add.
        with self.bias_dropout_add_exec_handler():
            layernorm_input = bias_dropout_add_func(
                attention_output,
                attention_bias,
                residual,
                self.hidden_dropout)

        # Layer norm.
        layernorm_output = self.post_inter_attention_layernorm(layernorm_input)

        return layernorm_input, layernorm_output

    def retro_encoder_cross_attention(self,
                                      retriever_output,
                                      layernorm_input,
                                      layernorm_output,
                                      bias_dropout_add_func):
        """Cross attention for Retro encoder.

        Notation:
            ns : Sequence length.
            bs : Batch size.
            d  : Hidden size.
            l  : Number of chunks per sample (i.e., seq_length/chunk_length).
            k  : Number of neighbors.
            r  : Number of retrieved tokens (neighbors + continuation).
        """

        ns, bs, d = layernorm_output.shape # [r, bs * l * k, d]

        # Divide sequence dimension into chunks.
        chunked_outputs = layernorm_output.reshape(self.retro_retrieved_length,
                                                   -1,
                                                   self.retro_num_neighbors,
                                                   d)
        chunked_outputs_before_layer_norm = \
            layernorm_input.reshape(self.retro_retrieved_length, -1,
                                    self.retro_num_neighbors, d) # [r, bs*l, k, d]

        # Per-chunk attention.
        layernorm_inputs = []
        layernorm_outputs = []
        for k in range(self.retro_num_neighbors):

            # Attention.
            chunked_output = chunked_outputs[:,:,k].contiguous()
            attention_output, attention_bias = \
                self.inter_attention(
                    chunked_output, # Q (neighbor embedding)
                    None,
                    encoder_output=retriever_output) # K, V (hidden act)

            # Residual connection.
            if self.apply_residual_connection_post_layernorm:
                residual = chunked_output
            else:
                residual = chunked_outputs_before_layer_norm[:,:,k]

            # Re-enable torch grad to enable fused optimization.
            with torch.enable_grad():
                layernorm_input = bias_dropout_add_func(
                    attention_output,
                    None if attention_bias is None else attention_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)
                layernorm_inputs.append(layernorm_input)

            # Layer norm.
            layernorm_output = \
                self.post_inter_attention_layernorm(layernorm_input)
            layernorm_outputs.append(layernorm_output)

        # Concatenate layer norms.
        # layernorm_input : [r, k * bs * l, d]
        # layernorm_output : [r, k * bs * l, d]
        layernorm_input = \
            torch.stack(layernorm_inputs, dim=1).reshape(ns, bs, d)
        layernorm_output = \
            torch.stack(layernorm_outputs, dim=1).reshape(ns, bs, d)

        return layernorm_input, layernorm_output

    def retro_decoder_cross_attention(self,
                                      retriever_input,
                                      retriever_output,
                                      retriever_attn_mask,
                                      layernorm_input,
                                      layernorm_output,
                                      inference_params,
                                      bias_dropout_add_func):
        """Cross attention for Retro decoder.

        Notation:
            ns : Sequence length.
            bs : Batch size.
            d  : Hidden size.
            l  : Number of chunks per sample (i.e., seq_length/chunk_length).
            m  : Number of tokens per chunk.
            k  : Number of neighbors.
            r  : Number of retrieved tokens (neighbors + continuation).
        """

        ns, bs, d = layernorm_output.shape
        l = int(np.ceil(ns / self.retro_chunk_length))

        # Retrieve neighbors.
        if self.layer_type == LayerType.retro_decoder_with_retriever:
            first_ns = ns % self.retro_chunk_length
            if first_ns > 0:
                raise Exception("test this case.")
                first_chunk, rest_chunk = \
                    layernorm_output[:first_ns], layernorm_output[first_ns:]
                first_chunk = torch.nn.functional.pad(
                    first_chunk,
                    (0, 0, 0, 0, 0, self.retro_chunk_length - first_ns),
                    'constant',
                    0)
                chunked_output = \
                    torch.cat((first_chunk, rest_chunk), dim=0) # [l * m, bs, d]
            else:
                chunked_output = layernorm_output # [l * m, bs, d]
            chunked_output = chunked_output \
                .reshape(l, self.retro_chunk_length, bs, d) \
                .permute(1, 2, 0, 3) \
                .reshape(self.retro_chunk_length, bs * l, d) \
                .contiguous()

            # Get Encoder Output
            retriever_output = self.retriever(
                hidden_states=retriever_input,
                attention_mask=retriever_attn_mask,
                retriever_output=chunked_output,
                retriever_attn_mask=retriever_attn_mask,
                inference_params=inference_params) # [r, k * bs * l , d]
            retriever_output = retriever_output.reshape(
                self.retro_retrieved_length * self.retro_num_neighbors, bs * l, d) # [r * k, bs * l, d]

        # Chunks.
        pad = (ns - 1) % self.retro_chunk_length
        attending_chunks = layernorm_output[pad:]
        padded_chunks = torch.nn.functional.pad(
            attending_chunks,
            (0, 0, 0, 0, 0, self.retro_chunk_length - 1),
            'constant', 0)
        padded_chunked_output = padded_chunks \
            .reshape(l, self.retro_chunk_length, bs, d) \
            .permute(1, 2, 0, 3)
        padded_chunked_output = padded_chunked_output.reshape(
            self.retro_chunk_length, bs * l, d).contiguous()

        # Encoder output.
        attention_output, attention_bias = \
            self.inter_attention(padded_chunked_output,
                                 None,
                                 encoder_output=retriever_output)

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        # Re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            layernorm_input = bias_dropout_add_func(
                attention_output,
                None if attention_bias is None else attention_bias.expand_as(attention_output),
                torch.zeros_like(attention_output),
                self.hidden_dropout)
            layernorm_input = layernorm_input \
                .reshape(self.retro_chunk_length, bs, l, d) \
                .permute(2, 0, 1, 3) # [l, m, bs, d]
            layernorm_input = layernorm_input.reshape(self.retro_chunk_length * l, bs, d)
            layernorm_input = torch.nn.functional.pad(
                layernorm_input,
                (0, 0, 0, 0, pad, 0),
                'constant', 0)[:ns] # [ns, b, d]
            layernorm_input = layernorm_input + residual

        # Layer norm post the decoder attention
        layernorm_output = self.post_inter_attention_layernorm(layernorm_input)

        return retriever_output, layernorm_input, layernorm_output

    def forward(self, hidden_states, attention_mask=None,
                encoder_output=None, enc_dec_attn_mask=None,
                retriever_input=None,
                retriever_output=None,
                retriever_attn_mask=None,
                inference_params=None,
                rotary_pos_emb=None,
                aggregated_moe_loss=None):
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        try:
            attention_output, attention_bias = \
                self.self_attention(
                    layernorm_output,
                    attention_mask,
                    inference_params=inference_params,
                    rotary_pos_emb=rotary_pos_emb,
                    cpu_offloading=self.ds_sequence_parallel_fpdt_offloading)
        except TypeError:
            attention_output, attention_bias = \
                self.self_attention(
                    layernorm_output,
                    attention_mask,
                    inference_params=inference_params,
                    rotary_pos_emb=rotary_pos_emb)
            
        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # FPDT does not support dropout yet.
        if not self.ds_sequence_parallel_fpdt:
            if self.drop_path is None:
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

                if attention_bias is not None:
                    attention_bias = attention_bias.expand_as(residual)
                with self.bias_dropout_add_exec_handler():
                    layernorm_input = bias_dropout_add_func(
                        attention_output,
                        attention_bias,
                        residual,
                        self.hidden_dropout)
            else:
                out = torch.nn.functional.dropout(attention_output + attention_bias,
                                                p=self.hidden_dropout,
                                                training=self.training)
                layernorm_input = residual + self.drop_path(out)
        else:
            if attention_bias is not None:
                attention_output = attention_output + attention_bias
            layernorm_input = attention_output + residual

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # Cross attention.
        if self.layer_type == LayerType.encoder:
            pass
        elif self.layer_type == LayerType.decoder:
            layernorm_input, layernorm_output = \
                self.default_decoder_cross_attention(
                    encoder_output,
                    enc_dec_attn_mask,
                    layernorm_input,
                    layernorm_output,
                    bias_dropout_add_func)
        elif self.layer_type == LayerType.retro_encoder:
            layernorm_input, layernorm_output = \
                self.retro_encoder_cross_attention(
                    retriever_output,
                    layernorm_input,
                    layernorm_output,
                    bias_dropout_add_func)
        elif self.layer_type in (LayerType.retro_decoder,
                                 LayerType.retro_decoder_with_retriever):
            retriever_output, layernorm_input, layernorm_output = \
                self.retro_decoder_cross_attention(
                    retriever_input,
                    retriever_output,
                    retriever_attn_mask,
                    layernorm_input,
                    layernorm_output,
                    inference_params,
                    bias_dropout_add_func)
        else:
            raise Exception("Unsupported layer type, '%s'." %
                            self.layer_type.name)

        # MLP.
        moe_loss = torch.tensor(0.0, device=layernorm_output.device, dtype=layernorm_output.dtype)
        mlp_bias = torch.tensor(0.0, device=layernorm_output.device, dtype=layernorm_output.dtype)

        if self.num_experts == 1:
            mlp_output, mlp_bias = self.mlp(layernorm_output)
        else:
            mlp_output, moe_loss, _ = self.mlp(layernorm_output)

        # when aggregated_moe_loss received, returned moe_loss is the aggregated moe loss
        if aggregated_moe_loss is not None:
            moe_loss += aggregated_moe_loss

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        # FPDT does not support dropout yet.
        if not self.ds_sequence_parallel_fpdt:
            if self.drop_path is None:
                if mlp_bias is not None:
                    mlp_bias = mlp_bias.expand_as(residual)
                with self.bias_dropout_add_exec_handler():
                    output = bias_dropout_add_func(
                        mlp_output,
                        mlp_bias,
                        residual,
                        self.hidden_dropout)

                # Jit compiled function creates 'view' tensor. This tensor
                # potentially gets saved in the MPU checkpoint function context,
                # which rejects view tensors. While making a viewless tensor here
                # won't result in memory savings (like the data loader, or
                # p2p_communication), it serves to document the origin of this
                # 'view' tensor.
                output = core.utils.make_viewless_tensor(inp = output,
                                                        requires_grad = output.requires_grad,
                                                        keep_graph = True)

            else:
                if mlp_bias is not None:
                    mlp_output = mlp_output + mlp_bias
                out = torch.nn.functional.dropout(mlp_output,
                                                p=self.hidden_dropout,
                                                training=self.training)
                output = residual + self.drop_path(out)
        else:
            if mlp_bias is not None:
                mlp_output = mlp_output + mlp_bias
            output = mlp_output + residual

        if self.layer_type == LayerType.retro_decoder_with_retriever:
            return output, retriever_output, moe_loss
        else:
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
    def __init__(self, config,
                 layer_number, layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 drop_path_rate=0., num_experts=1,
                 input_aggregated_moe_loss=False, return_aggregated_moe_loss=False):
        self.input_aggregated_moe_loss = input_aggregated_moe_loss
        self.return_aggregated_moe_loss = return_aggregated_moe_loss
        super().__init__(config, layer_number, layer_type, self_attn_mask_type, drop_path_rate, num_experts)

    def forward(self, inputs, **kwargs):
        assert torch.is_tensor(inputs) or isinstance(inputs, tuple)
        if not hasattr(self, '_args'):
            self._args = get_args()
        rotary_pos_emb = self._args.rotary_pos_emb if self._args.use_rotary_position_embeddings else None
        if torch.is_tensor(inputs) or len(inputs) == 1:
            assert not self.input_aggregated_moe_loss, f'Expecting an input tuple of size >= 2'
            # No attention mask forwarded, search for args.attn_mask
            hidden_states, attention_mask = inputs, self._args.attn_mask
            output, moe_loss = super().forward(hidden_states, attention_mask, **kwargs, rotary_pos_emb=rotary_pos_emb)
            return (output, moe_loss) if self.return_aggregated_moe_loss else output
        elif len(inputs) in (2, 3):
            # Attention mask and aggregated_moe can both be activations.
            return_attention_mask = False
            if len(inputs) == 2:
                if self.input_aggregated_moe_loss:
                    hidden_states, aggregated_moe_loss = inputs[0], inputs[1]
                    attention_mask = self._args.attn_mask
                else:
                    hidden_states, attention_mask = inputs[0], inputs[1]
                    return_attention_mask = True
            else:
                hidden_states, attention_mask, aggregated_moe_loss = inputs[0], inputs[1], inputs[2]

            # Forward aggregated_moe_loss to ParallelTransformerLayer for further accumulation
            if self.input_aggregated_moe_loss:
                kwargs.update({'aggregated_moe_loss': aggregated_moe_loss})

            output, moe_loss = super().forward(hidden_states, attention_mask, **kwargs, rotary_pos_emb=rotary_pos_emb)

            ret = (output, )
            if return_attention_mask:
                ret += (attention_mask, )
            if self.return_aggregated_moe_loss:
                ret += (moe_loss, )
            return ret
        else:
            raise RuntimeError('Received more inputs than understood.')


class NoopTransformerLayer(MegatronModule):
    """A single 'no-op' transformer layer.

    The sole purpose of this layer is for when a standalone embedding layer
    is used (i.e., args.standalone_embedding_stage == True). In this case,
    zero transformer layers are assigned when pipeline rank == 0. Additionally,
    when virtual pipeline rank >= 1, zero total model parameters are created
    (virtual rank 0 contains the input embedding). This results in the model's
    input and output tensors being the same, which causes an error when
    performing certain memory optimiations on the output tensor (e.g.,
    deallocating it). Thus, this layer disconnects the input from the output
    via a clone. Since ranks containing a no-op layer are generally under-
    utilized (both compute and memory), there's no worry of any performance
    degredation.
    """

    def __init__(self, layer_number):
        super().__init__()
        self.layer_number = layer_number

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                inference_params=None):
        return hidden_states.clone()


def _get_num_layers(args, model_type, is_decoder=False):
    """Compute the number of transformer layers resident on the current rank."""
    is_encoder_and_decoder_model = (model_type == ModelType.encoder_and_decoder)
    if model_type == ModelType.retro_encoder:
        num_layers = args.retro_encoder_layers
    elif parallel_state.get_pipeline_model_parallel_world_size() > 1:
        if is_encoder_and_decoder_model:
            assert args.pipeline_model_parallel_split_rank is not None

            # When a standalone embedding stage is used, a rank is taken from
            # the encoder's ranks, to be used for the encoder's embedding
            # layer. This way, the rank referenced by the 'split rank' remains
            # the same whether or not a standalone embedding stage is used.
            num_ranks_in_encoder = (
                args.pipeline_model_parallel_split_rank - 1
                if args.standalone_embedding_stage else
                args.pipeline_model_parallel_split_rank
            )
            num_ranks_in_decoder = args.transformer_pipeline_model_parallel_size - num_ranks_in_encoder
            assert args.encoder_num_layers % num_ranks_in_encoder == 0, \
                    'encoder_num_layers (%d) must be divisible by number of ranks given to encoder (%d)' % (args.encoder_num_layers, num_ranks_in_encoder)
            assert args.decoder_num_layers % num_ranks_in_decoder == 0, \
                    'decoder_num_layers (%d) must be divisible by number of ranks given to decoder (%d)' % (args.decoder_num_layers, num_ranks_in_decoder)
            if parallel_state.is_pipeline_stage_before_split():
                num_layers = (
                    0
                    if args.standalone_embedding_stage
                    and parallel_state.get_pipeline_model_parallel_rank() == 0 else
                    args.encoder_num_layers // num_ranks_in_encoder
                )
            else:
                num_layers = args.decoder_num_layers // num_ranks_in_decoder
        else:
            assert args.num_layers == args.encoder_num_layers
            assert args.num_layers % args.transformer_pipeline_model_parallel_size == 0, \
                'num_layers must be divisible by transformer_pipeline_model_parallel_size'

            # When a standalone embedding stage is used, all transformer layers
            # are divided among pipeline rank >= 1, while on pipeline rank 0,
            # ranks either contain the input embedding layer (virtual pp rank 0),
            # or no layers at all (virtual pp rank >= 1).
            num_layers = (
                0
                if args.standalone_embedding_stage
                and parallel_state.get_pipeline_model_parallel_rank() == 0 else
                args.num_layers // args.transformer_pipeline_model_parallel_size
            )
    else:
        if not is_decoder:
            num_layers = args.encoder_num_layers
        else:
            num_layers = args.decoder_num_layers
    return num_layers


def _get_layer_type(model_type, default_layer_type, retro_layer_numbers,
                    layer_number):
    args = get_args()
    if args.retro_add_retriever and layer_number in retro_layer_numbers:
        if model_type == ModelType.retro_decoder:
            return LayerType.retro_decoder_with_retriever \
                if layer_number == retro_layer_numbers[0] \
                   else LayerType.retro_decoder
        elif model_type == ModelType.retro_encoder:
            return LayerType.retro_encoder
        else:
            raise Exception("Unsupported model type, '%s'." % model_type)
    else:
        return default_layer_type


def get_num_experts_per_layer(num_experts: list, num_layers: int, expert_interval: int, offset: int = 0) -> list:
    assert len(num_experts) == 1 or len(num_experts) == num_layers // expert_interval, \
        'num_experts must be either a single value or a list of the same length as the number of MoE layers'
    if len(num_experts) == 1:
        num_experts = num_experts * (num_layers // expert_interval)
    experts_per_layer = []
    for i in range(num_layers):
        layer_num = i + 1 + offset
        n_e = num_experts[(layer_num-1) // expert_interval] if layer_num % expert_interval == 0 else 1
        experts_per_layer.append(n_e)
    return experts_per_layer


class ParallelTransformer(MegatronModule):
    """Transformer class."""

    def __init__(self, config,
                 model_type, layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 post_layer_norm=True,
                 pre_process=True,
                 post_process=True,
                 drop_path_rate=0.0,
                 num_experts=[1]):
        super(ParallelTransformer, self).__init__()
        args = get_args()

        self.layer_type = layer_type
        self.model_type = model_type
        self.bf16 = config.bf16
        self.fp32_residual_connection = config.fp32_residual_connection
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_tensor = None
        self.drop_path_rate = drop_path_rate
        self.transformer_impl = args.transformer_impl
        self.retro_add_retriever = args.retro_add_retriever
        self.ds_inference = args.ds_inference

        # Store activation checkpoiting flag.
        self.checkpoint_activations = args.checkpoint_activations
        self.checkpoint_num_layers = args.checkpoint_num_layers
        self.recompute_granularity = config.recompute_granularity
        self.recompute_method = config.recompute_method
        self.recompute_num_layers = config.recompute_num_layers
        self.distribute_saved_activations = \
            config.distribute_saved_activations and not config.sequence_parallel

        self.sequence_parallel = config.sequence_parallel

        # Transformer Engine Init.
        self.transformer_engine_rope_available = False
        if self.transformer_impl == 'transformer_engine':
            global transformer_engine
            import transformer_engine
            from importlib.metadata import version
            from pkg_resources import packaging

            te_version = packaging.version.Version(version("transformer-engine"))
            if te_version >= packaging.version.Version("0.10.0"):
                self.transformer_engine_rope_available = True

            del version, packaging

        self.use_fp8 = args.fp8_e4m3 or args.fp8_hybrid
        self.fp8_recipe = None
        self.fp8_group = None
        if self.use_fp8:
            self.fp8_group = parallel_state.get_data_parallel_group()
            if args.fp8_e4m3:
                fp8_format = transformer_engine.common.recipe.Format.E4M3
            elif args.fp8_hybrid:
                fp8_format = transformer_engine.common.recipe.Format.HYBRID
            self.fp8_recipe = transformer_engine.common.recipe.DelayedScaling(
                margin=args.fp8_margin,
                interval=args.fp8_interval,
                fp8_format=fp8_format,
                amax_history_len=args.fp8_amax_history_len,
                amax_compute_algo=args.fp8_amax_compute_algo,
                override_linear_precision=(False, False, not args.fp8_wgrad),
            )

        self.num_microbatches_in_previous_step = -1
        self.microbatch_count = 0
        self.checkpoint_core_attention = config.recompute_granularity == 'selective'

        # Number of layers.
        self.num_layers = _get_num_layers(args, model_type,
                                          layer_type==LayerType.decoder)

        self.drop_path_rates = [
            rate.item() for rate in
            torch.linspace(0, self.drop_path_rate, config.num_layers)]

        self.retro_layer_numbers = None
        if model_type == ModelType.retro_decoder:
            retro_layer_start = 6 if config.num_layers <= 15 else 9
            self.retro_layer_numbers = \
                np.arange(retro_layer_start, args.num_layers + 1, 3).tolist()
        if model_type == ModelType.retro_encoder:
            self.retro_layer_numbers = [1]

        # Transformer layers.
        if args.retro_add_retriever:
            assert self.recompute_granularity != 'full', \
                "Full recompute not supported for Retro."
            assert args.transformer_impl == 'local', \
                "Transformer engine does not support Retro layers."
        def build_layer(layer_number, n_e):
            if args.transformer_impl == 'local':
                current_layer_type = _get_layer_type(
                    model_type, layer_type, self.retro_layer_numbers,
                    layer_number)
                return ParallelTransformerLayer(
                    config,
                    layer_number,
                    layer_type=current_layer_type,
                    self_attn_mask_type=self_attn_mask_type,
                    drop_path_rate=self.drop_path_rates[layer_number - 1],
                    num_experts=n_e)
            else:
                assert config.num_attention_heads == config.num_key_value_heads, \
                        'Transformer_engine does not support GQA'
                return transformer_engine.pytorch.TransformerLayer(
                    config.hidden_size,
                    config.ffn_hidden_size,
                    config.num_attention_heads,
                    layernorm_epsilon=config.layernorm_epsilon,
                    hidden_dropout=config.hidden_dropout,
                    attention_dropout=config.attention_dropout,
                    init_method=config.init_method,
                    output_layer_init_method=config.output_layer_init_method,
                    layer_number=layer_number,
                    kv_channels=config.kv_channels,
                    self_attn_mask_type=self_attn_mask_type.name,
                    tp_group=parallel_state.get_tensor_model_parallel_group(),
                    get_rng_state_tracker=tensor_parallel.get_cuda_rng_tracker,
                    fuse_wgrad_accumulation=config.gradient_accumulation_fusion,
                    apply_query_key_layer_scaling=config.apply_query_key_layer_scaling,
                    attention_softmax_in_fp32=config.attention_softmax_in_fp32,
                    seq_length=args.seq_length,
                    micro_batch_size=args.micro_batch_size,
                    sequence_parallel=config.sequence_parallel,
                    params_dtype=config.params_dtype,
                    apply_residual_connection_post_layernorm=config.apply_residual_connection_post_layernorm,
                    output_layernorm=False,
                    layer_type="encoder",
                    drop_path_rate=self.drop_path_rates[layer_number - 1],
                    set_parallel_mode=True,
                    fuse_qkv_params=True)

        if config.virtual_pipeline_model_parallel_size is not None:
            assert config.num_layers % config.virtual_pipeline_model_parallel_size == 0, \
                'num_layers_per_stage must be divisible by ' \
                'virtual_pipeline_model_parallel_size'
            assert args.model_type != ModelType.encoder_and_decoder
            # Number of layers in each model chunk is the number of layers in the stage,
            # divided by the number of model chunks in a stage.
            self.num_layers = self.num_layers // config.virtual_pipeline_model_parallel_size
            # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0]  [2]  [4]  [6]
            # Stage 1: [1]  [3]  [5]  [7]
            # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0, 1]  [4, 5]
            # Stage 1: [2, 3]  [6, 7]
            offset = parallel_state.get_virtual_pipeline_model_parallel_rank() * (
                config.num_layers // config.virtual_pipeline_model_parallel_size) + \
                (parallel_state.get_pipeline_model_parallel_rank() * self.num_layers)
        else:
            # Each stage gets a contiguous set of layers.
            if args.model_type == ModelType.encoder_and_decoder and \
                    parallel_state.get_pipeline_model_parallel_world_size() > 1:
                pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()
                if layer_type == LayerType.encoder:
                    offset = pipeline_rank * self.num_layers
                else:
                    num_ranks_in_enc = args.pipeline_model_parallel_split_rank
                    offset = (pipeline_rank - num_ranks_in_enc) * self.num_layers
            else:
                offset = parallel_state.get_pipeline_model_parallel_rank() * self.num_layers

        if self.num_layers == 0:
            # When a standalone embedding stage is used (e.g.,
            # args.standalone_embedding_stage == True), virtual pipeline ranks
            # on pipeline rank 0 will have zero transformer layers assigned to
            # them. This results in the model's input and output tensors to be
            # the same, which will cause failure for certain output tensor
            # optimizations (e.g., pipeline output deallocation). To remedy
            # this, we assign a 'no-op' layer on these ranks, which will
            # disconnect the input tensor from the output tensor.
            self.num_layers = 1
            self.layers = torch.nn.ModuleList([ NoopTransformerLayer(1) ])
        else:
            # Build the layers
            self.layers = []
            experts_per_layer = get_num_experts_per_layer(num_experts, self.num_layers, args.expert_interval, offset)
            for i in range(self.num_layers):
                layer_num = i + 1 + offset
                n_e = experts_per_layer[i]
                self.layers.append(build_layer(layer_num, n_e))
            self.layers = torch.nn.ModuleList(self.layers)

            # Update dropout rate for Retro encoder.
            if model_type == ModelType.retro_encoder:
                for layer in self.layers:
                    if layer.self_attention.use_flash_attn:
                        layer.self_attention.core_attention_flash.dropout_p = \
                            torch.nn.Dropout(args.retro_encoder_attention_dropout)
                    else:
                        layer.self_attention.core_attention.attention_dropout.p =\
                            args.retro_encoder_attention_dropout
                    layer.hidden_dropout = args.retro_encoder_hidden_dropout

        if self.post_process and self.post_layer_norm:
            # Final layer norm before output.
            if args.normalization == 'layernorm':
                if get_accelerator().device_name() == 'cuda':
                    self.final_layernorm = LayerNorm(
                        config.hidden_size,
                        eps=config.layernorm_epsilon,
                        no_persist_layer_norm=args.no_persist_layer_norm,
                        sequence_parallel=config.sequence_parallel,
                        apply_layernorm_1p=args.apply_layernorm_1p,
                        mem_efficient_ln=args.mem_efficient_ln)
                else:
                    self.final_layernorm = LayerNorm(
                        config.hidden_size,
                        eps=config.layernorm_epsilon)
            else:
                self.final_layernorm = RMSNorm(config.hidden_size, config.layernorm_epsilon,
                                               sequence_parallel=config.sequence_parallel)

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def _checkpointed_forward(self, hidden_states, attention_mask,
                              encoder_output, enc_dec_attn_mask,
                              rotary_pos_emb, is_first_microbatch):
        args = get_args()

        """Forward method with activation checkpointing."""
        def custom(start, end):
            def custom_forward(*args, **kwargs):
                x_, *args = args
                moe_losses = []
                for index in range(start, end):
                    layer = self._get_layer(index)
                    output = layer(x_, *args, **kwargs)
                    if isinstance(output, tuple):
                        x_, moe_loss = output
                    else:
                        x_ = output
                        moe_loss = torch.tensor(0.0, device=x_.device, dtype=x_.dtype, requires_grad=True)
                    moe_losses.append(moe_loss)
                return (x_, *moe_losses)
            return custom_forward
        
        if args.deepspeed and args.deepspeed_activation_checkpointing:
            moe_losses = []
            # Make sure memory is freed.
            tensor_parallel.reset_checkpointed_activations_memory_buffer()
            l = 0
            while l < self.num_layers:
                hidden_states, *local_moe_losses = tensor_parallel.checkpoint(
                    custom(l, l + self.checkpoint_num_layers), False,
                    hidden_states, attention_mask, encoder_output, enc_dec_attn_mask,
                    None, None, None, None, rotary_pos_emb)
                moe_losses.extend(local_moe_losses)
                l += self.checkpoint_num_layers

            return hidden_states, moe_losses
        else:
            moe_losses = []
            te_forward_kwargs = {}
            if self.transformer_impl == 'transformer_engine':
                te_forward_kwargs['is_first_microbatch'] = is_first_microbatch
                if self.transformer_engine_rope_available:
                    te_forward_kwargs['rotary_pos_emb'] = rotary_pos_emb

            if self.recompute_method == 'uniform':
                # Uniformly divide the total number of Transformer layers and
                # checkpoint the input activation of each divided chunk.
                # A method to further reduce memory usage reducing checkpoints.
                l = 0
                while l < self.num_layers:
                    if self.transformer_impl == 'transformer_engine':
                        hidden_states, *local_moe_losses = transformer_engine.pytorch.distributed.checkpoint(
                            custom(l, l + self.recompute_num_layers),
                            self.distribute_saved_activations,
                            tensor_parallel.get_cuda_rng_tracker,
                            mpu.get_tensor_model_parallel_group(),
                            hidden_states, attention_mask, encoder_output,
                            enc_dec_attn_mask, **te_forward_kwargs)
                    else:
                        hidden_states, *local_moe_losses = tensor_parallel.checkpoint(
                            custom(l, l + self.recompute_num_layers),
                            self.distribute_saved_activations,
                            hidden_states, attention_mask,
                            encoder_output, enc_dec_attn_mask,
                            None, None, None, None, rotary_pos_emb)
                    moe_losses.extend(local_moe_losses)
                    l += self.recompute_num_layers
            elif self.recompute_method == 'block':
                # Checkpoint the input activation of only a set number of individual
                # Transformer layers and skip the rest.
                # A method fully use the device memory removing redundant re-computation.
                for l in range(self.num_layers):
                    if l < self.recompute_num_layers:
                        if self.transformer_impl == 'transformer_engine':
                            hidden_states, *local_moe_losses = transformer_engine.pytorch.distributed.checkpoint(
                                custom(l, l + 1),
                                self.distribute_saved_activations,
                                tensor_parallel.get_cuda_rng_tracker,
                                mpu.get_tensor_model_parallel_group(),
                                hidden_states, attention_mask, encoder_output,
                                enc_dec_attn_mask, **te_forward_kwargs)
                        else:
                            hidden_states, *local_moe_losses = tensor_parallel.checkpoint(
                                custom(l, l + 1),
                                self.distribute_saved_activations,
                                hidden_states, attention_mask,
                                encoder_output, enc_dec_attn_mask,
                                None, None, None, None, rotary_pos_emb)
                    else:
                        if self.transformer_impl == 'transformer_engine':
                            hidden_states, *local_moe_losses = custom(l, l + 1)(
                                hidden_states, attention_mask, encoder_output,
                                enc_dec_attn_mask, **te_forward_kwargs)
                        else:
                            hidden_states, *local_moe_losses = custom(l, l + 1)(
                                hidden_states, attention_mask,
                                encoder_output, enc_dec_attn_mask,
                                None, None, None, None, rotary_pos_emb)
                            
                    moe_losses.extend(local_moe_losses)
            else:
                raise ValueError("Invalid activation recompute method.")
            return hidden_states, moe_losses

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                retriever_input=None,
                retriever_output=None,
                retriever_attn_mask=None,
                inference_params=None,
                rotary_pos_emb=None):
        # hidden_states: [s, b, h]

        # Checks.
        if inference_params:
            assert self.recompute_granularity is None, \
                'inference does not work with activation checkpointing'

        # TODO: Below old DeepSpeed code are commented because it's unsure whether
        # it is still relevant.
        # # Reza's note: DeepSpeed inference does not support transposes
        # if not self.ds_inference:
        #     if self.pre_process:
        #         # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        #         # If the input flag for fp32 residual connection is set, convert for float.
        #         if self.fp32_residual_connection:
        #             hidden_states = hidden_states.transpose(0, 1).contiguous().float()
        #         # Otherwise, leave it as is.
        #         else:
        #             hidden_states = hidden_states.transpose(0, 1).contiguous()
        #     else:
        #         # See set_input_tensor()
        #         hidden_states = self.input_tensor
        #     if encoder_output is not None:
        #          encoder_output = encoder_output.transpose(0, 1).contiguous()

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = core.utils.make_viewless_tensor(
            hidden_states,
            requires_grad=True,
            keep_graph=True,
        )

        # RNG context.
        if self.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # Forward layers.
        with rng_context:
            # The fp8_autocast context manager is a no-op when enabled=True
            # The if...else serves to short circuit name resolution for fp8_autocast
            with transformer_engine.pytorch.fp8_autocast(
                enabled=self.use_fp8,
                fp8_recipe=self.fp8_recipe,
                fp8_group=self.fp8_group
            ) if self.use_fp8 else nullcontext():
                # Determine if the current iteration is first microbatch
                if self.num_microbatches_in_previous_step != get_num_microbatches():
                    self.microbatch_count = 0 # Reset count on new batch size rampup interval
                self.num_microbatches_in_previous_step = get_num_microbatches()
                is_first_microbatch = self.microbatch_count % get_num_microbatches() == 0

                # Forward pass.
                moe_losses = []
                if self.checkpoint_activations:
                    hidden_states, moe_losses = self._checkpointed_forward(hidden_states,
                                                               attention_mask,
                                                               encoder_output,
                                                               enc_dec_attn_mask,
                                                               rotary_pos_emb,
                                                               is_first_microbatch)
                elif self.recompute_granularity == 'full':
                    hidden_states, moe_losses = self._checkpointed_forward(hidden_states,
                                                               attention_mask,
                                                               encoder_output,
                                                               enc_dec_attn_mask,
                                                               rotary_pos_emb,
                                                               is_first_microbatch)
                else:
                    forward_kwargs = {
                        'encoder_output': encoder_output,
                        'enc_dec_attn_mask': enc_dec_attn_mask,
                        'inference_params': inference_params,
                    }

                    if self.transformer_impl == 'transformer_engine':
                        forward_kwargs['is_first_microbatch'] = is_first_microbatch
                        forward_kwargs['checkpoint_core_attention'] = self.checkpoint_core_attention
                        if self.transformer_engine_rope_available:
                            forward_kwargs['rotary_pos_emb'] = rotary_pos_emb
                    else:
                        forward_kwargs['rotary_pos_emb'] = rotary_pos_emb
                        forward_kwargs['retriever_input'] = retriever_input
                        forward_kwargs['retriever_output'] = retriever_output
                        forward_kwargs['retriever_attn_mask'] = retriever_attn_mask

                    for index in range(self.num_layers):
                        layer = self._get_layer(index)

                        hidden_states = layer(
                            hidden_states,
                            attention_mask,
                            **forward_kwargs)

                        # First Retro decoder layer returns both hidden_states
                        # and retriever_output. Make retriever_output available
                        # to subsequence Retro layers.
                        if isinstance(hidden_states, tuple):
                            assert (len(hidden_states) == 2 or len(hidden_states) == 3)
                            if len(hidden_states) == 2:
                                if not self.ds_inference:
                                    hidden_states, moe_loss = hidden_states
                                    moe_losses.append(moe_loss)
                            else:
                                forward_kwargs["retriever_output"] = hidden_states[1]
                                if not self.ds_inference:
                                    hidden_states, _, moe_loss = hidden_states
                                    moe_losses.append(moe_loss)

                # Skip counter update for eval and activation checkpointing
                if torch.is_grad_enabled() and self.training:
                    self.microbatch_count += 1

        # Final layer norm.
        if self.post_process and self.post_layer_norm:
            # TODO: Below old DeepSpeed code are commented because it's unsure whether
            # it is still relevant.
            # if not self.ds_inference:
            #     # Reverting data format change [s b h] --> [b s h].
            #     hidden_states = hidden_states.transpose(0, 1).contiguous()
            hidden_states = self.final_layernorm(hidden_states)

        return (hidden_states, *moe_losses)

class LMHeadPipe(MegatronModule):
    """
    Arguments:
        vocab_size: size of vocabulary.
        hidden_size: hidden size
        gather_output: wether output logits being gathered or not.
        init_method: init method for weight initialization
        config:
    """

    def __init__(self, hidden_size, vocab_size, config):
        args = get_args()
        super(LMHeadPipe, self).__init__()
        self.lm_head = tensor_parallel.ColumnParallelLinear(input_size=hidden_size,
                                                            output_size=vocab_size,
                                                            bias=False,
                                                            config=config,
                                                            init_method=config.init_method,)

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

        logits, _ = self.lm_head(hidden_states)

        # If cmd args has attn_mask, we don't forward it as an activation.
        if hasattr(self._args, 'attn_mask'):
            return logits
        else:
            return logits, attention_mask
