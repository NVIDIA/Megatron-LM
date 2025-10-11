# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


import math
from typing import Optional, Tuple

import torch
import einops
from torch import Tensor
from torch.nn import functional as F

from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import (
    attention_mask_func,
    is_layer_window_attention,
    make_sharded_tensors_for_checkpoint,
)
from megatron.core.utils import divide


@torch.no_grad
def eager_attn_fwd(q, k, v, attn_bias, sinks, scale, dropout):
    # 1.
    b, sq, h, d = q.shape
    sk = k.shape[1]
    assert dropout == 0.
    _q = einops.rearrange(q, 'b s h d -> b h s d')
    _k = einops.rearrange(k, 'b s h d -> b h d s')
    _v = einops.rearrange(v, 'b s h d -> b h s d')

    # 2.
    attn_w = torch.matmul(_q, _k) * scale
    attn_w = attn_w + attn_bias

    # 3.
    if sinks is None:
        logits = attn_w
    else:
        _sinks = sinks.reshape(1, h, 1, 1).expand(b, -1, sq, 1)
        logits = torch.cat([attn_w, _sinks], dim=-1)

    # 4.
    # logits = logits - logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(logits, dim=-1, dtype=logits.dtype)
    if sinks is None:
        attn_w = probs
    else:
        attn_w = probs[..., :-1]

    # 5.
    attn_output = torch.matmul(attn_w, _v)
    attn_output = einops.rearrange(attn_output, 'b h s d -> b s h d')
    attn_output = attn_output.contiguous()

    return attn_output, probs


# @torch.compile
@torch.no_grad
def eager_attn_bwd(q, k, v, attn_bias, sinks, scale, dropout, attn_output, probs, grad_output):
    b, sq, h, d = q.shape
    sk = k.shape[1]
    _q_T = einops.rearrange(q, 'b s h d -> b h d s')
    _k_T = einops.rearrange(k, 'b s h d -> b h s d')
    _v_T = einops.rearrange(v, ' b s h d -> b h d s')

    if sinks is None:
        attn_w = probs
    else:
        attn_w = probs[..., :-1]

    # 5.
    grad_output = einops.rearrange(grad_output, 'b s h d -> b h s d')
    attn_w_T = einops.rearrange(attn_w, ' b h sq sk -> b h sk sq')
    grad__v = torch.matmul(attn_w_T, grad_output)
    grad_attn_w = torch.matmul(grad_output, _v_T)

    # 4.
    if sinks is None:
        grad_probs = grad_attn_w
    else:
        dummy = torch.zeros((b, h, sq, 1), device=q.device, dtype=q.dtype)
        grad_probs = torch.cat([grad_attn_w, dummy], dim=3)
    del grad_attn_w
    grad_logits = torch._softmax_backward_data(grad_probs, probs, -1, probs.dtype)  # [b, h, sq, sk+1]

    # 3.
    if sinks is None:
        grad_sinks = None
        grad_attn_w = grad_logits
    else:
        grad__sinks = grad_logits[:, :, :, -1]  # [b, h, sq]
        grad_sinks = einops.rearrange(grad__sinks, 'b h s -> h (b s)').sum(-1)
        grad_attn_w = grad_logits[:, :, :, :-1].contiguous()  # [b, h, sq, sk]

    # 2.
    grad_attn_w *= scale
    # C = A @ B
    # gA = gC @ B.T
    # gB = A.T @ gC
    grad__q = torch.matmul(grad_attn_w, _k_T)
    grad__k = torch.matmul(_q_T, grad_attn_w)

    # 1.
    grad_v = einops.rearrange(grad__v, 'b h s d -> b s h d')
    grad_k = einops.rearrange(grad__k, 'b h d s -> b s h d')
    grad_q = einops.rearrange(grad__q, 'b h s d -> b s h d')
    return grad_q, grad_k, grad_v, grad_sinks


class AllGatherComm:
    def __init__(self, group=None) -> None:
        self.group = group
        self.handles = []

    def all_gather(self, output_tensor: torch.Tensor, input_tensor: torch.Tensor):
        if self.group is None:
            output_tensor.copy_(input_tensor)
        else:
            handle = torch.distributed.all_gather_into_tensor(
                output_tensor, input_tensor, group=self.group, async_op=True
            )
            self.handles.append(handle)

    def wait(self):
        if self.group is not None:
            for handle in self.handles:
                handle.wait()
            self.handles = []


def to_zz_mask_attn_bias(attention_mask, cp_size, q, nheads, nheads_k, heads_k_stride):
    if cp_size == 1:
        zz_mask = attention_mask
    else:
        chunked = attention_mask.chunk(dim=3, chunks=cp_size * 2)
        zz_mask = [_x for _p in zip(chunked[:cp_size], reversed(chunked[cp_size:])) for _x in _p]
        zz_mask = torch.cat(zz_mask, dim=3)
    attn_bias = torch.zeros(zz_mask.shape, device=q.device, dtype=q.dtype)
    attn_bias.masked_fill_(zz_mask, float('-inf'))
    attn_bias = attn_bias.expand(-1, heads_k_stride * (nheads // nheads_k), -1, -1)
    return attn_bias


class AttnFuncWithCp(torch.autograd.Function):
    # Adapted and simplified from
    # > https://github.com/zhuzilin/ring-flash-attention/blob/main/ring_flash_attn/llama3_flash_attn_varlen.py

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        attention_mask,
        attention_dropout,
        softmax_scale,
        pg,
    ):
        cp_size = 1
        if pg is not None:
            cp_size = torch.distributed.get_world_size(pg)
        comm = AllGatherComm(group=pg)

        nheads = q.shape[2]
        nheads_k = k.shape[2]
        heads_k_stride = 1
        assert nheads % nheads_k == 0 and nheads_k % heads_k_stride == 0
        outs = []
        probs = []

        kv_buffer = torch.empty(
            (2, k.shape[0] * cp_size, k.shape[1], heads_k_stride, k.shape[3]),
            dtype=k.dtype,
            device=k.device,
        )
        kv_buffer_copy = torch.empty_like(kv_buffer)

        k_0 = k[:, :, :heads_k_stride].contiguous()
        v_0 = v[:, :, :heads_k_stride].contiguous()
        comm.all_gather(kv_buffer_copy[0], k_0)
        comm.all_gather(kv_buffer_copy[1], v_0)

        attn_bias = to_zz_mask_attn_bias(attention_mask, cp_size, q, nheads, nheads_k, heads_k_stride)

        for i in range(0, nheads_k, heads_k_stride):
            comm.wait()
            kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer
            if i < nheads_k - heads_k_stride:
                kvsl = i + heads_k_stride
                kvsr = kvsl + heads_k_stride
                send_k = k[:, :, kvsl:kvsr].contiguous()
                send_v = v[:, :, kvsl:kvsr].contiguous()
                comm.all_gather(kv_buffer_copy[0], send_k)
                comm.all_gather(kv_buffer_copy[1], send_v)

            q_i = q[:, :, i * nheads // nheads_k:(i + heads_k_stride) * nheads // nheads_k]
            k_i = kv_buffer[0]
            v_i = kv_buffer[1]

            q_i = einops.rearrange(q_i, 's b h d -> b s h d')
            k_i = einops.rearrange(k_i, 's b h d -> b s h d')
            v_i = einops.rearrange(v_i, 's b h d -> b s h d')

            out_i, probs_i = eager_attn_fwd(q_i, k_i, v_i, attn_bias, None, softmax_scale, 0.)
            outs.append(out_i)
            probs.append(probs_i)

        out = torch.cat(outs, dim=2)
        out = einops.rearrange(out, 'b s h d -> s b h d')

        ctx.save_for_backward(q, k, v, attention_mask, *outs, *probs)
        ctx.dropout = attention_dropout
        ctx.scale = softmax_scale
        ctx.heads_k_stride = heads_k_stride # TODO make it configurable
        ctx.pg = pg

        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, attention_mask, *rest = ctx.saved_tensors
        nheads = q.shape[2]
        nheads_k = k.shape[2]
        heads_k_stride = ctx.heads_k_stride
        assert nheads_k % heads_k_stride == 0
        outs = rest[:nheads_k // heads_k_stride]
        probs = rest[nheads_k // heads_k_stride:]

        pg = ctx.pg
        cp_size = 1
        if pg is not None:
            cp_size = torch.distributed.get_world_size(pg)
        comm = AllGatherComm(group=pg)

        kv_buffer = torch.empty(
            (2, k.shape[0] * cp_size, k.shape[1], heads_k_stride, k.shape[3]),
            dtype=k.dtype,
            device=k.device,
        )
        kv_buffer_copy = torch.empty_like(kv_buffer)

        dq = []
        dk = []
        dv = []
        k_0 = k[:, :, :heads_k_stride].contiguous()
        v_0 = v[:, :, :heads_k_stride].contiguous()
        comm.all_gather(kv_buffer_copy[0], k_0)
        comm.all_gather(kv_buffer_copy[1], v_0)

        attn_bias = to_zz_mask_attn_bias(attention_mask, cp_size, q, nheads, nheads_k, heads_k_stride)

        for i in range(0, nheads_k, heads_k_stride):
            q_slice = slice(i * nheads // nheads_k, (i + heads_k_stride) * nheads // nheads_k)
            q_i = q[:, :, q_slice]
            dout_i = dout[:, :, q_slice]

            comm.wait()
            kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer

            if i < nheads_k - heads_k_stride:
                kvsl = i + heads_k_stride
                kvsr = kvsl + heads_k_stride
                send_k = k[:, :, kvsl:kvsr].contiguous()
                send_v = v[:, :, kvsl:kvsr].contiguous()
                comm.all_gather(kv_buffer_copy[0], send_k)
                comm.all_gather(kv_buffer_copy[1], send_v)

            k_i = kv_buffer[0]
            v_i = kv_buffer[1]

            q_i = einops.rearrange(q_i, 's b h d -> b s h d')
            k_i = einops.rearrange(k_i, 's b h d -> b s h d')
            v_i = einops.rearrange(v_i, 's b h d -> b s h d')
            dout_i = einops.rearrange(dout_i, 's b h d -> b s h d')

            dq_i, _dk_i, _dv_i, _ = eager_attn_bwd(
                q_i, k_i, v_i, attn_bias, None, ctx.scale, ctx.dropout, outs[i], probs[i], dout_i
            )

            dq_i = einops.rearrange(dq_i, 'b s h d -> s b h d')
            _dk_i = einops.rearrange(_dk_i, 'b s h d -> s b h d')
            _dv_i = einops.rearrange(_dv_i, 'b s h d -> s b h d')
            if pg is None:
                dk_i = _dk_i
                dv_i = _dv_i
            else:
                dk_i = torch.zeros(
                    (k_i.shape[1] // cp_size, k_i.shape[0], k_i.shape[2], k_i.shape[3]),
                    device=k_i.device,
                    dtype=k_i.dtype,
                )
                dv_i = torch.zeros(
                    (v_i.shape[1] // cp_size, v_i.shape[0], v_i.shape[2], v_i.shape[3]),
                    device=v_i.device,
                    dtype=v_i.dtype,
                )
                torch.distributed.reduce_scatter_tensor(dk_i, _dk_i, group=pg)
                torch.distributed.reduce_scatter_tensor(dv_i, _dv_i, group=pg)

            dq.append(dq_i)
            dk.append(dk_i)
            dv.append(dv_i)

        dq = torch.cat(dq, dim=2)
        dk = torch.cat(dk, dim=2)
        dv = torch.cat(dv, dim=2)
        return dq, dk, dv, None, None, None, None


class DotProductAttention(MegatronModule):
    """
    Region where selective activation recomputation is applied.
    This region is memory intensive but less compute intensive which
    makes activation checkpointing more efficient for LLMs (20B+).
    See Reducing Activation Recomputation in Large Transformer Models:
    https://arxiv.org/abs/2205.05198 for more details.

    We use the following notation:
     h: hidden size
     n: number of attention heads
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
        softmax_scale: float = None,
        cp_comm_type: str = None,
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config

        if self.config.context_parallel_size > 1:
            assert (
                attention_dropout is None and self.config.attention_dropout == 0.
            ), f'context with cp > does not support {attention_dropout=} {self.config.attention_dropout=}'

        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type  # unused for now

        projection_size = self.config.kv_channels * self.config.num_attention_heads

        # Per attention head and per partition values.
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp'])
        else:
            assert hasattr(
                pg_collection, 'tp'
            ), "DotProductAttention pg_collection must have tp process group"

        world_size = pg_collection.tp.size()
        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

        coeff = None
        if softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head)
        else:
            self.softmax_scale = softmax_scale

        if self.config.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.softmax_scale /= coeff

        if is_layer_window_attention(
            self.config.window_size, self.config.window_attn_skip_freq, layer_number
        ):
            window_size = self.config.window_size
        else:
            window_size = None

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            input_in_fp16=self.config.fp16,
            input_in_bf16=self.config.bf16,
            attn_mask_type=self.attn_mask_type,
            scaled_masked_softmax_fusion=self.config.masked_softmax_fusion,
            mask_func=attention_mask_func,
            softmax_in_fp32=self.config.attention_softmax_in_fp32,
            scale=coeff,
            window_size=window_size,
        )

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(
            self.config.attention_dropout if attention_dropout is None else attention_dropout
        )

        if self.config.softmax_type == "vanilla":
            self.softmax_offset = None
        elif self.config.softmax_type == "off-by-one":
            self.softmax_offset = torch.zeros(
                self.num_attention_heads_per_partition,
                device=torch.cuda.current_device(),
                dtype=self.config.params_dtype,
            )
        elif self.config.softmax_type == "learnable":
            self.register_parameter(
                "softmax_offset",
                torch.nn.Parameter(
                    torch.empty(
                        self.num_attention_heads_per_partition,
                        device=torch.cuda.current_device(),
                        dtype=self.config.params_dtype,
                    )
                ),
            )
            if config.perform_initialization:
                self.softmax_offset = config.init_method(self.softmax_offset)
        else:
            raise ValueError("Softmax type not supported")

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        attention_bias: Tensor = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """Forward."""
        assert packed_seq_params is None, (
            "Packed sequence is not supported by DotProductAttention."
            "Please use TEDotProductAttention instead."
        )
        assert attention_bias is None, "Attention bias is not supported for DotProductAttention."

        # ===================================
        # Raw attention scores. [b, n/p, s, s]
        # ===================================

        # expand the key and value [sk, b, ng, hn] -> [sk, b, np, hn]
        # This is a noop for normal attention where ng == np. When using group query attention this
        # creates a view that has the keys and values virtually repeated along their dimension to
        # match the number of queries.

        # attn_mask_type is not used.
        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
            key = key.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )
            value = value.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )

        if self.config.context_parallel_size > 1:
            output = AttnFuncWithCp.apply(
                query,
                key,
                value,
                attention_mask,
                self.config.attention_dropout,
                self.softmax_scale,
                parallel_state.get_context_parallel_group(),
            )
            output = output.view(query.shape[0], query.shape[1], self.hidden_size_per_partition)
            return output

        # [b, np, sq, sk]
        output_size = (query.size(1), query.size(2), query.size(0), key.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        # This will be a simple view when doing normal attention, but in group query attention
        # the key and value tensors are repeated to match the queries so you can't use
        # simple strides to extract the queries.
        query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key = key.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(
            (output_size[0] * output_size[1], output_size[2], output_size[3]), query.dtype, "mpu"
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query.transpose(0, 1),  # [b * np, sq, hn]
            key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=self.softmax_scale,
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs: Tensor = self.scale_mask_softmax(
            attention_scores, attention_mask, self.softmax_offset
        )
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        if not self.config.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value.size(1), value.size(2), query.size(0), value.size(3))

        # change view [sk, b * np, hn]
        value = value.view(value.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context = torch.bmm(attention_probs, value.transpose(0, 1))

        # change view [b, np, sq, hn]
        context = context.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context = context.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        context = context.view(*new_context_shape)

        return context

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Sharded state dict for the learnable softmax offset parameter"""
        if self.config.softmax_type == "learnable":
            state_dict = self.state_dict(prefix="", keep_vars=True)
        else:
            state_dict = {}
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {'softmax_offset': 0}, sharded_offsets
        )
