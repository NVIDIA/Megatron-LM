# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Some of this code was adopted from https://github.com/zhuzilin/ring-flash-attention/
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn import functional as F

try:
    import einops

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False


@torch.no_grad
def eager_attn_fwd(q, k, v, attn_bias, sinks, scale, dropout):
    """Forward pass for eager attention"""

    # Rearrange query, key, value to (b, h, s, d)
    b, sq, h, d = q.shape
    sk = k.shape[1]
    _q = einops.rearrange(q, 'b s h d -> b h s d')
    _k = einops.rearrange(k, 'b s h d -> b h d s')
    _v = einops.rearrange(v, 'b s h d -> b h s d')

    # Compute attention weights
    attn_w = torch.matmul(_q, _k) * scale
    attn_w = attn_w + attn_bias

    # Add sinks to attention weights
    if sinks is None:
        logits = attn_w
    else:
        _sinks = sinks.reshape(1, h, 1, 1).expand(b, -1, sq, 1)
        logits = torch.cat([attn_w, _sinks], dim=-1)

    # Compute attention scores
    probs = F.softmax(logits, dim=-1, dtype=logits.dtype)
    if sinks is None:
        attn_w = probs
    else:
        attn_w = probs[..., :-1]  # Drop the sink

    # Compute attention output
    attn_output = torch.matmul(attn_w, _v)
    attn_output = einops.rearrange(attn_output, 'b h s d -> b s h d')
    attn_output = attn_output.contiguous()

    return attn_output, probs


@torch.no_grad
def eager_attn_bwd(q, k, v, attn_bias, sinks, scale, dropout, attn_output, probs, grad_output):
    """Backward pass for eager attention"""

    # Rearrange query, key, value to (b, h, s, d)
    b, sq, h, d = q.shape
    sk = k.shape[1]
    _q_T = einops.rearrange(q, 'b s h d -> b h d s')
    _k_T = einops.rearrange(k, 'b s h d -> b h s d')
    _v_T = einops.rearrange(v, ' b s h d -> b h d s')

    # Backward pass for score @ value
    if sinks is None:
        attn_w = probs
    else:
        attn_w = probs[..., :-1]  # Drop the sink
    grad_output = einops.rearrange(grad_output, 'b s h d -> b h s d')
    attn_w_T = einops.rearrange(attn_w, ' b h sq sk -> b h sk sq')
    grad__v = torch.matmul(attn_w_T, grad_output)
    grad_attn_w = torch.matmul(grad_output, _v_T)

    # Backward pass for softmax
    if sinks is None:
        grad_probs = grad_attn_w
    else:
        dummy = torch.zeros((b, h, sq, 1), device=q.device, dtype=q.dtype)
        grad_probs = torch.cat([grad_attn_w, dummy], dim=3)
    del grad_attn_w
    grad_logits = torch._softmax_backward_data(
        grad_probs, probs, -1, probs.dtype
    )  # [b, h, sq, sk+1]

    # Backward pass for adding sinks
    if sinks is None:
        grad_sinks = None
        grad_attn_w = grad_logits
    else:
        grad__sinks = grad_logits[:, :, :, -1]  # [b, h, sq]
        grad_sinks = einops.rearrange(grad__sinks, 'b h s -> h (b s)').sum(-1)
        grad_attn_w = grad_logits[:, :, :, :-1].contiguous()  # [b, h, sq, sk]

    # Backward pass for q @ K^T
    grad_attn_w *= scale
    grad__q = torch.matmul(grad_attn_w, _k_T)
    grad__k = torch.matmul(_q_T, grad_attn_w)

    # Rearrange grads to (b, s, h, d)
    grad_v = einops.rearrange(grad__v, 'b h s d -> b s h d')
    grad_k = einops.rearrange(grad__k, 'b h d s -> b s h d')
    grad_q = einops.rearrange(grad__q, 'b h s d -> b s h d')
    return grad_q, grad_k, grad_v, grad_sinks


class AllGatherComm:
    """All gather communication with async operations"""

    def __init__(self, group=None) -> None:
        self.group = group
        self.handles = []

    def all_gather(self, output_tensor: torch.Tensor, input_tensor: torch.Tensor):
        '''All gather the input tensor to the output tensor'''

        if self.group is None:
            output_tensor.copy_(input_tensor)
        else:
            handle = torch.distributed.all_gather_into_tensor(
                output_tensor, input_tensor, group=self.group, async_op=True
            )
            self.handles.append(handle)

    def wait(self):
        '''Wait for all gather operations to complete'''

        if self.group is not None:
            for handle in self.handles:
                handle.wait()
            self.handles = []


def to_zz_mask_attn_bias(attention_mask, cp_size, nheads, nheads_k, heads_k_stride, device, dtype):
    '''Convert the attention mask to the attention bias'''

    if cp_size == 1:
        zz_mask = attention_mask
    else:
        chunked = attention_mask.chunk(dim=3, chunks=cp_size * 2)
        zz_mask = [_x for _p in zip(chunked[:cp_size], reversed(chunked[cp_size:])) for _x in _p]
        zz_mask = torch.cat(zz_mask, dim=3)
    attn_bias = torch.zeros(zz_mask.shape, device=device, dtype=dtype)
    attn_bias.masked_fill_(zz_mask, float('-inf'))
    attn_bias = attn_bias.expand(-1, heads_k_stride * (nheads // nheads_k), -1, -1)
    return attn_bias


class AttentionFuncionWithContextParallel(torch.autograd.Function):
    """Native attention function with context parallelism."""

    @staticmethod
    def forward(ctx, q, k, v, attention_mask, attention_dropout, softmax_scale, pg):
        '''Forward pass for the native attention function with context parallelism'''

        # Assert einops exists
        if not HAVE_EINOPS:
            raise ImportError("einops is required by the attention CP but cannot be imported.")

        # Initialize communication group and constants
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

        # Initialize KV buffers
        kv_buffer = torch.empty(
            (2, k.shape[0] * cp_size, k.shape[1], heads_k_stride, k.shape[3]),
            dtype=k.dtype,
            device=k.device,
        )
        kv_buffer_copy = torch.empty_like(kv_buffer)

        # All-gather first chunk of KV buffers
        k_0 = k[:, :, :heads_k_stride].contiguous()
        v_0 = v[:, :, :heads_k_stride].contiguous()
        comm.all_gather(kv_buffer_copy[0], k_0)
        comm.all_gather(kv_buffer_copy[1], v_0)

        # Prepare attention bias
        attn_bias = to_zz_mask_attn_bias(
            attention_mask, cp_size, nheads, nheads_k, heads_k_stride, q.device, q.dtype
        )

        # Iterate over heads
        for i in range(0, nheads_k, heads_k_stride):
            # Wait for previous all-gather to complete
            comm.wait()
            kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer
            # All-gather the next portion of KV buffers if not the last iteration
            if i < nheads_k - heads_k_stride:
                kvsl = i + heads_k_stride
                kvsr = kvsl + heads_k_stride
                send_k = k[:, :, kvsl:kvsr].contiguous()
                send_v = v[:, :, kvsl:kvsr].contiguous()
                comm.all_gather(kv_buffer_copy[0], send_k)
                comm.all_gather(kv_buffer_copy[1], send_v)

            # Prepare query, key, value for attention
            q_i = q[:, :, i * nheads // nheads_k : (i + heads_k_stride) * nheads // nheads_k]
            k_i = kv_buffer[0]
            v_i = kv_buffer[1]

            # Rearrange query, key, value to (b, s, h, d)
            q_i = einops.rearrange(q_i, 's b h d -> b s h d')
            k_i = einops.rearrange(k_i, 's b h d -> b s h d')
            v_i = einops.rearrange(v_i, 's b h d -> b s h d')

            # Forward pass
            out_i, probs_i = eager_attn_fwd(
                q_i, k_i, v_i, attn_bias, None, softmax_scale, attention_dropout
            )
            outs.append(out_i)
            probs.append(probs_i)

        # Concatenate outputs and rearrange to (s, b, h, d)
        out = torch.cat(outs, dim=2)
        out = einops.rearrange(out, 'b s h d -> s b h d')

        # Save contexts for backward pass
        ctx.save_for_backward(q, k, v, attention_mask, *outs, *probs)
        ctx.dropout = attention_dropout
        ctx.scale = softmax_scale
        ctx.heads_k_stride = heads_k_stride  # TODO make it configurable
        ctx.pg = pg

        return out

    @staticmethod
    def backward(ctx, dout):
        '''Backward pass for the native attention function with context parallelism'''

        # Initialize or resume constants and communication group
        q, k, v, attention_mask, *rest = ctx.saved_tensors
        nheads = q.shape[2]
        nheads_k = k.shape[2]
        heads_k_stride = ctx.heads_k_stride
        assert nheads_k % heads_k_stride == 0
        outs = rest[: nheads_k // heads_k_stride]
        probs = rest[nheads_k // heads_k_stride :]
        pg = ctx.pg
        cp_size = 1
        if pg is not None:
            cp_size = torch.distributed.get_world_size(pg)
        comm = AllGatherComm(group=pg)

        # Initialize KV buffers
        kv_buffer = torch.empty(
            (2, k.shape[0] * cp_size, k.shape[1], heads_k_stride, k.shape[3]),
            dtype=k.dtype,
            device=k.device,
        )
        kv_buffer_copy = torch.empty_like(kv_buffer)

        # All-gather first chunk of KV buffers
        dq = []
        dk = []
        dv = []
        k_0 = k[:, :, :heads_k_stride].contiguous()
        v_0 = v[:, :, :heads_k_stride].contiguous()
        comm.all_gather(kv_buffer_copy[0], k_0)
        comm.all_gather(kv_buffer_copy[1], v_0)

        # Prepare attention bias
        attn_bias = to_zz_mask_attn_bias(
            attention_mask, cp_size, nheads, nheads_k, heads_k_stride, q.device, q.dtype
        )

        # Iterate over heads
        for i in range(0, nheads_k, heads_k_stride):
            # Slice query and output for this iteration
            q_slice = slice(i * nheads // nheads_k, (i + heads_k_stride) * nheads // nheads_k)
            q_i = q[:, :, q_slice]
            dout_i = dout[:, :, q_slice]

            # Wait for previous all-gather to complete
            comm.wait()
            kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer

            # All-gather the next portion of KV buffers if not the last iteration
            if i < nheads_k - heads_k_stride:
                kvsl = i + heads_k_stride
                kvsr = kvsl + heads_k_stride
                send_k = k[:, :, kvsl:kvsr].contiguous()
                send_v = v[:, :, kvsl:kvsr].contiguous()
                comm.all_gather(kv_buffer_copy[0], send_k)
                comm.all_gather(kv_buffer_copy[1], send_v)

            # Prepare key, value for attention
            k_i = kv_buffer[0]
            v_i = kv_buffer[1]

            # Rearrange query, key, value to (b, s, h, d)
            q_i = einops.rearrange(q_i, 's b h d -> b s h d')
            k_i = einops.rearrange(k_i, 's b h d -> b s h d')
            v_i = einops.rearrange(v_i, 's b h d -> b s h d')
            dout_i = einops.rearrange(dout_i, 's b h d -> b s h d')

            # Backward pass
            dq_i, _dk_i, _dv_i, _ = eager_attn_bwd(
                q_i, k_i, v_i, attn_bias, None, ctx.scale, ctx.dropout, outs[i], probs[i], dout_i
            )

            # Rearrange gradients to (s, b, h, d)
            dq_i = einops.rearrange(dq_i, 'b s h d -> s b h d')
            _dk_i = einops.rearrange(_dk_i, 'b s h d -> s b h d')
            _dv_i = einops.rearrange(_dv_i, 'b s h d -> s b h d')
            if pg is None:
                dk_i = _dk_i
                dv_i = _dv_i
            else:
                # Reduce-scatter gradients if CP > 1
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

            # Collect gradients
            dq.append(dq_i)
            dk.append(dk_i)
            dv.append(dv_i)

        # Concatenate gradients and return
        dq = torch.cat(dq, dim=2)
        dk = torch.cat(dk, dim=2)
        dv = torch.cat(dv, dim=2)
        return dq, dk, dv, None, None, None, None
