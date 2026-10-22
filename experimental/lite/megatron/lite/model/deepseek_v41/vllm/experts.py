# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""R1's nonfused MegaMoE chain, using shared codecs and native BI activation."""
import torch

from megatron.lite.primitive.quantization.mxfp4 import dequantize_mxfp4, quantize_mxfp4


def packed_scales(scale):
    raw = (scale.contiguous().view(torch.int32) >> 23).to(torch.uint8).view(torch.int32)
    rows, groups = raw.shape
    return torch.empty_strided(
        (rows, groups), (1, (rows + 3) // 4 * 4), device=scale.device, dtype=torch.int32
    ).copy_(raw)


def unpack_scales(packed, groups):
    rows = packed.shape[0]
    width = 4 * packed.shape[1]
    raw = packed.contiguous().view(torch.uint8).reshape(rows, width)[:, :groups]
    return (raw.to(torch.int32) << 23).view(torch.float32)


def dequantize_activation(codes, scale):
    width = codes.shape[-1] // scale.shape[-1]
    return (codes.float().reshape(*scale.shape, width) * scale.unsqueeze(-1)).reshape(codes.shape)


@torch.compile(fullgraph=True, dynamic=True)
def _decode_bf16(codes, scale):
    return dequantize_activation(codes, scale).bfloat16()


def quantize_activation(x):
    if not x.is_cuda:
        raise RuntimeError("Deployment experts require CUDA")
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

    if not x.shape[0]:
        return x.to(torch.float8_e4m3fn), x.new_empty((0, x.shape[1] // 32), dtype=torch.float32)
    return per_token_group_quant_fp8(x.contiguous(), 32, eps=1e-4, use_ue8m0=True)


def _activation_surrogate(first, scores, limit):
    gate, up = first.float().chunk(2, -1)
    if limit is not None:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    return torch.nn.functional.silu(gate) * up * scores.float().unsqueeze(-1)


class _Activation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, first, scores, limit):
        ctx.save_for_backward(first, scores)
        ctx.limit = limit
        return _activation_surrogate(first, scores, limit)

    @staticmethod
    def backward(ctx, grad):
        first, scores = ctx.saved_tensors
        with torch.enable_grad():
            first, scores = first.detach().requires_grad_(), scores.detach().requires_grad_()
            output = _activation_surrogate(first, scores, ctx.limit)
            df, ds = torch.autograd.grad(output, (first, scores), grad)
        return df, ds, None


def activation(fc1, scores, limit):
    if not fc1.is_cuda:
        raise RuntimeError("Deployment expert activation requires CUDA")
    from vllm.model_executor.kernels.batch_invariant import swiglu_apply_weight_to_fp8

    hidden = _Activation.apply(fc1, scores, limit)
    if not fc1.shape[0]:
        return hidden, (
            hidden.to(torch.float8_e4m3fn),
            hidden.new_empty((0, hidden.shape[1] // 32)),
        )
    codes, packed = swiglu_apply_weight_to_fp8(
        fc1.detach(), scores.detach().float(), clamp_limit=limit, fast_math=True
    )
    return hidden, (codes, unpack_scales(packed, hidden.shape[1] // 32))


def _grouped(codes, scale, weights, weight_scales, counts):
    from vllm.utils.deep_gemm import _import_deep_gemm

    deep_gemm = _import_deep_gemm()
    if deep_gemm is None:
        raise RuntimeError("Pinned DeepGEMM is required for MegaMoE numerics")
    alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
    positions, ends, offset = [], [], 0
    for count in counts:
        positions.append(torch.arange(offset, offset + count, device=codes.device))
        ends.append(offset + count)
        offset = (offset + count + alignment - 1) // alignment * alignment
    positions = torch.cat(positions)
    capacity = max(offset, alignment)
    padded = torch.zeros((capacity, codes.shape[1]), dtype=torch.uint8, device=codes.device)
    padded.index_copy_(0, positions, codes.view(torch.uint8))
    sf = scale.new_ones((capacity, scale.shape[1])).index_copy_(0, positions, scale)
    exponent = (weight_scales.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)
    weight_sf = deep_gemm.transform_sf_into_required_layout(
        sf=exponent,
        mn=weights.shape[1],
        k=codes.shape[1],
        recipe=(1, 1, 32),
        num_groups=weights.shape[0],
        is_sfa=False,
        disable_ue8m0_cast=False,
    )
    output = torch.zeros((capacity, weights.shape[1]), dtype=torch.bfloat16, device=codes.device)
    deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
        (padded.view(torch.float8_e4m3fn), packed_scales(sf)),
        (weights, weight_sf),
        output,
        torch.tensor(ends, dtype=torch.int32, device=codes.device),
        use_psum_layout=True,
        recipe=(1, 1, 32),
    )
    return output.index_select(0, positions)


class _Grouped(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, counts, encoded, *weights):
        if not x.is_cuda:
            raise RuntimeError("Deployment grouped experts require CUDA")
        if len(counts) != len(weights) or sum(counts) != x.shape[0]:
            raise ValueError("One contiguous row count is required per native local expert")
        codes, scale = encoded
        packed = [quantize_mxfp4(weight.detach()) for weight in weights]
        weight_codes = torch.stack([v[0] for v in packed])
        weight_scales = torch.stack([v[1] for v in packed])
        ctx.save_for_backward(codes, scale, weight_codes, weight_scales)
        ctx.counts, ctx.dtypes, ctx.input_dtype = counts, [w.dtype for w in weights], x.dtype
        return _grouped(codes, scale, weight_codes, weight_scales, counts)

    @staticmethod
    def backward(ctx, grad):
        codes, scale, weight_codes, weight_scales = ctx.saved_tensors
        x = (
            _decode_bf16(codes, scale)
            if codes.is_cuda and codes.numel() and grad.dtype == torch.bfloat16
            else dequantize_activation(codes, scale).to(grad.dtype)
        )
        ends = torch.tensor(ctx.counts, device=grad.device, dtype=torch.int32).cumsum(
            0, dtype=torch.int32
        )
        weight = dequantize_mxfp4(weight_codes, weight_scales).to(grad.dtype)
        with torch.autocast(device_type=grad.device.type, enabled=False):
            dx = (
                torch._grouped_mm(grad.contiguous(), weight, offs=ends)
                if grad.is_cuda
                else torch.cat([g @ w for g, w in zip(grad.split(ctx.counts), weight.unbind(0))])
            )
            if grad.is_cuda:
                if not grad.shape[0]:
                    dw = torch.zeros_like(weight)
                else:
                    dw = torch._grouped_mm(grad.contiguous().T, x.contiguous(), offs=ends)
                    empty = ends == torch.cat((ends.new_zeros(1), ends[:-1]))
                    dw.masked_fill_(empty[:, None, None], 0)
            else:
                dw = torch.stack(
                    [
                        g.float().T @ a.float()
                        for g, a in zip(grad.split(ctx.counts), x.split(ctx.counts))
                    ]
                )
        return (
            dx.to(ctx.input_dtype),
            None,
            None,
            *(value.to(dtype) for value, dtype in zip(dw.unbind(0), ctx.dtypes)),
        )


def expert_rows(x, fc1, fc2, counts, scores, encoded, limit):
    first = _Grouped.apply(x, tuple(counts), encoded, *fc1)
    hidden, encoded = activation(first, scores, limit)
    return _Grouped.apply(hidden, tuple(counts), encoded, *fc2)


class _Shared(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, encoded):
        from vllm.utils.deep_gemm import _import_deep_gemm

        from megatron.lite.primitive.quantization.block_fp8 import quantize_block_fp8

        deep_gemm = _import_deep_gemm()
        if deep_gemm is None:
            raise RuntimeError("Pinned DeepGEMM is required for shared MegaMoE numerics")
        a, scale = encoded
        codes, sf = quantize_block_fp8(weight, (32, 32), scale_format='e8m0')
        ctx.save_for_backward(a, scale, codes, sf)
        ctx.weight_dtype, ctx.input_shape, ctx.input_dtype = weight.dtype, x.shape, x.dtype
        output = x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.bfloat16)
        if x.shape[0]:
            rows = (sf.view(torch.uint8).repeat_interleave(32, 0).to(torch.int32) << 23).view(
                torch.float32
            )
            deep_gemm.fp8_gemm_nt(
                (a, packed_scales(scale)),
                (codes, packed_scales(rows)),
                output,
                recipe=(1, 1, 32),
                disable_ue8m0_cast=True,
            )
        return output

    @staticmethod
    def backward(ctx, grad):
        from megatron.lite.primitive.quantization.block_fp8 import dequantize_block_fp8

        from .linear import linear_vjp

        a, scale, codes, sf = ctx.saved_tensors
        activation = dequantize_activation(a, scale).to(ctx.input_dtype)
        weight = dequantize_block_fp8(codes, sf, (32, 32)).to(ctx.input_dtype)
        return linear_vjp(ctx, grad, activation, weight)


_BUFFERS = {}


def _layout():
    from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import DeepGemmMegaMoEExperts

    return DeepGemmMegaMoEExperts


def _write(dst, src, shard, is_scale):
    layout = _layout()
    rows = dst.shape[0] // (1 if shard == "w2" else 2)
    src = layout._pack_scale(src, rows, dst.shape[1] * 4) if is_scale else src.view(dst.dtype)
    dst, src = layout._kernel_layout_shard(dst, src, shard, is_scale)
    dst.copy_(src)


def _empty_weight(outer, n, k, values_per_element, device):
    weight = torch.zeros(
        *outer,
        n,
        k // values_per_element,
        dtype=torch.int8 if values_per_element == 2 else torch.float8_e4m3fn,
        device=device,
    )
    scale = torch.zeros(*outer, k // 128, n, dtype=torch.int32, device=device).transpose(-1, -2)
    return weight, scale


def _write_experts(dst, src, shard, is_scale):
    """:func:`_write` for every local expert at once: vLLM's ``_pack_scale`` and
    ``_kernel_layout_shard`` index maps with an explicit leading expert dimension."""
    experts, k = dst.shape[0], src.shape[-1]
    if is_scale:
        rows = dst.shape[1] // (1 if shard == "w2" else 2)
        src = src.view(torch.uint8)
        src = src.repeat_interleave(rows // src.shape[1], 1)
        src = src.repeat_interleave(dst.shape[2] * 4 // src.shape[2], 2)
        src = src.contiguous().view(torch.int32)
        k = src.shape[-1]
    else:
        src = src.view(dst.dtype)
    if shard == "w2":
        if is_scale:
            dst = dst.view(experts, -1, 32, 4, k).transpose(2, 3)
            src = src.reshape(experts, -1, 4, 32, k)
    else:
        half = 0 if shard == "w1" else 1
        if is_scale:
            dst = dst.view(experts, -1, 2, 2, 8, 4, k)[:, :, :, half].permute(0, 1, 4, 2, 3, 5)
            src = src.reshape(experts, -1, 4, 2, 8, k)
        else:
            dst = dst.view(experts, -1, 2, 8, k)[:, :, half]
            src = src.reshape(experts, -1, 8, k)
    dst.copy_(src)


@torch.no_grad()
def kernel_weights(fc1, fc2, shared_fc1, shared_fc2):
    """Pack fresh masters into native MegaMoE kernel layouts, all local experts at once."""
    from megatron.lite.primitive.quantization.block_fp8 import quantize_block_fp8
    from megatron.lite.primitive.quantization.mxfp4 import quantize_mxfp4

    device = fc1[0].device
    inter, hidden = fc1[0].shape[0] // 2, fc1[0].shape[1]
    w13, s13 = _empty_weight((len(fc1),), 2 * inter, hidden, 2, device)
    w2, s2 = _empty_weight((len(fc2),), hidden, inter, 2, device)
    up = torch.stack([w.detach() for w in fc1])
    for shard, part in (("w1", up[:, :inter]), ("w3", up[:, inter:])):
        codes, scale = quantize_mxfp4(part)
        _write_experts(w13, codes, shard, False)
        _write_experts(s13, scale, shard, True)
    codes, scale = quantize_mxfp4(torch.stack([w.detach() for w in fc2]))
    _write_experts(w2, codes, "w2", False)
    _write_experts(s2, scale, "w2", True)
    s_inter = shared_fc1.shape[0] // 2
    l1, l1s = _empty_weight((), 2 * s_inter, hidden, 1, device)
    l2, l2s = _empty_weight((), hidden, s_inter, 1, device)
    for shard, part, (w, s) in (
        ("w1", shared_fc1[:s_inter], (l1, l1s)),
        ("w3", shared_fc1[s_inter:], (l1, l1s)),
        ("w2", shared_fc2, (l2, l2s)),
    ):
        codes, scale = quantize_block_fp8(part.detach(), (32, 32), scale_format="e8m0")
        _write(w, codes, shard, False)
        _write(s, scale, shard, True)
    return (w13, s13), (w2, s2), (l1, l1s), (l2, l2s)


def symm_buffer(group, num_experts, max_tokens, top_k, hidden, inter, shared_experts=1):
    from vllm.utils.deep_gemm import _import_deep_gemm

    deep_gemm = _import_deep_gemm()
    key = (id(group), torch.cuda.current_device(), num_experts, max_tokens, top_k, hidden, inter)
    if key not in _BUFFERS:
        torch.cuda.empty_cache()
        _BUFFERS[key] = deep_gemm.get_symm_buffer_for_mega_moe(
            group,
            num_experts,
            max_tokens,
            top_k,
            hidden,
            inter,
            num_shared_experts=shared_experts,
            mma_type="fp8xfp4",
            activation="swiglu",
        )
        torch.cuda.synchronize()
        torch.distributed.barrier(group=group)
    return _BUFFERS[key]


def launch(x, weights, ids, packed, buffer, limit, ep_size):
    """The inference launch over prepacked kernel weights. Returns bf16 [T, H]."""
    from vllm.model_executor.layers.fused_moe.prepare_megamoe import prepare_megamoe_inputs
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
    from vllm.utils.deep_gemm import _import_deep_gemm

    deep_gemm = _import_deep_gemm()
    tokens = x.shape[0]
    if tokens > buffer.num_max_tokens_per_rank:
        raise ValueError(
            f"MegaMoE buffer holds {buffer.num_max_tokens_per_rank} tokens, got {tokens}"
        )
    l1, l2, shared_l1, shared_l2 = packed
    num_experts = l1[0].shape[0] * ep_size
    block_m = deep_gemm.get_block_m_for_mega_moe(
        ep_size, num_experts, buffer.num_max_tokens_per_rank, tokens, ids.shape[1], "fp8xfp4"
    )
    prepare_megamoe_inputs(
        x.contiguous(),
        weights.contiguous(),
        ids.contiguous(),
        buffer.x[:tokens],
        buffer.x_sf[:tokens],
        buffer.topk_idx[:tokens],
        buffer.topk_weights[:tokens],
        is_padding=None,
        shared_x_sf=buffer.shared_l1_acts_sf,
        shared_block_m=block_m,
        hidden_quant=kMxfp8Dynamic,
    )
    y = torch.empty_like(x, dtype=torch.bfloat16)
    deep_gemm.fp8_fp4_mega_moe(
        y,
        l1,
        l2,
        buffer,
        fast_math=True,
        activation_clamp=limit,
        shared_l1_weights=shared_l1,
        shared_l2_weights=shared_l2,
    )
    return y
