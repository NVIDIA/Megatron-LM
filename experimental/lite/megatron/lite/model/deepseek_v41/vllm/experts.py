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

    # Only the owned VJP uses this surrogate; native codes drive GEMM2.
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
        x = dequantize_activation(codes, scale).to(grad.dtype)
        # Ragged M/K grouping is owned by Torch's grouped GEMM. Decode the
        # stacked expert operand once; no per-expert CUDA launch sequence.
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
                from .backward_kernels import grouped_weight_vjp

                dw = torch.empty(weight.shape, device=grad.device, dtype=torch.float32)
                grouped_weight_vjp[
                    (len(ctx.counts), (grad.shape[1] + 31) // 32, (x.shape[1] + 63) // 64)
                ](grad.contiguous(), x.contiguous(), ends, dw, grad.shape[1], x.shape[1])
            else:
                # CPU-only derivative controls have no grouped CUDA backend.
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


def shared_rows(x, module, encoded, limit):
    first = _Shared.apply(x, module.linear_fc1.weight, encoded)
    hidden, encoded = activation(first, x.new_ones(x.shape[0]), limit)
    return _Shared.apply(hidden, module.linear_fc2.weight, encoded)
