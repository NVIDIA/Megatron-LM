# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from typing import Callable, Optional, Union

import torch
import torch.nn as nn

from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.utils import get_default_causal_mask


class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
    """
    Fused operation which performs following three operations in sequence
    1. Scale the tensor.
    2. Apply upper triangular mask (typically used in gpt models).
    3. Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs, scale):
        import scaled_upper_triang_masked_softmax_cuda

        scale_t = torch.tensor([scale])
        softmax_results = scaled_upper_triang_masked_softmax_cuda.forward(inputs, scale_t[0])

        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        import scaled_upper_triang_masked_softmax_cuda

        softmax_results, scale_t = ctx.saved_tensors
        input_grads = scaled_upper_triang_masked_softmax_cuda.backward(
            output_grads, softmax_results, scale_t[0]
        )

        return input_grads, None


class ScaledMaskedSoftmax(torch.autograd.Function):
    """
    Fused operation which performs following three operations in sequence
    1. Scale the tensor.
    2. Apply the mask.
    3. Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs, mask, scale):
        import scaled_masked_softmax_cuda

        scale_t = torch.tensor([scale])

        softmax_results = scaled_masked_softmax_cuda.forward(inputs, mask, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        import scaled_masked_softmax_cuda

        softmax_results, scale_t = ctx.saved_tensors

        input_grads = scaled_masked_softmax_cuda.backward(output_grads, softmax_results, scale_t[0])
        return input_grads, None, None


class ScaledSoftmax(torch.autograd.Function):
    """
    Fused operation which performs following two operations in sequence
    1. Scale the tensor.
    2. Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs, scale):
        import scaled_softmax_cuda

        scale_t = torch.tensor([scale])

        softmax_results = scaled_softmax_cuda.forward(inputs, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        import scaled_softmax_cuda

        softmax_results, scale_t = ctx.saved_tensors

        input_grads = scaled_softmax_cuda.backward(output_grads, softmax_results, scale_t[0])
        return input_grads, None, None


class SoftmaxOne(nn.Module):
    r"""
    Softmax-off-by-one function as introduced in https://www.evanmiller.org/attention-is-off-by-one.html
    Supports fixed or learnable offset
    """

    def __init__(
        self, dim: Optional[int] = None, denominator_offset: Union[torch.Tensor, float] = 1.0
    ) -> None:
        super().__init__()
        self.dim = dim
        self.denominator_offset = denominator_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # subtract the max for stability
        exp_x = torch.exp(x - x.max(dim=self.dim, keepdim=True).values)
        # add denominator offset
        return exp_x / (self.denominator_offset + exp_x.sum(dim=self.dim, keepdim=True))


class FusedScaleMaskSoftmax(nn.Module):
    """
    fused operation: scaling + mask + softmax

    Args:
        config: transformer config
        attn_mask_type: attention mask type (pad or causal)
        mask_func: mask function to be applied.
        scale: scaling factor used in input tensor scaling.
    """

    def __init__(
        self,
        config: TransformerConfig,
        attn_mask_type: AttnMaskType,
        mask_func: Callable,
        scale: Optional[float] = None,
    ):
        super(FusedScaleMaskSoftmax, self).__init__()
        self.input_in_fp16 = config.fp16
        self.input_in_bf16 = config.bf16
        assert not (
            self.input_in_fp16 and self.input_in_bf16
        ), "both fp16 and bf16 flags cannot be active at the same time."
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16
        self.attn_mask_type = attn_mask_type
        self.scaled_masked_softmax_fusion = config.masked_softmax_fusion
        self.mask_func = mask_func
        self.softmax_in_fp32 = config.attention_softmax_in_fp32
        self.scale = scale
        if config.attention_softmax_denominator_offset == 'learnable':
            self.softmax_denominator_weight = nn.Parameter(
                torch.empty(config.num_attention_heads // config.tensor_model_parallel_size)[
                    None, :, None, None
                ]
            )
        elif config.attention_softmax_denominator_offset is not None:
            self.softmax_denominator_weight = config.attention_softmax_denominator_offset
        else:
            self.softmax_denominator_weight = None

        assert self.scale is None or self.softmax_in_fp32, "softmax should be in fp32 when scaled"

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor]):
        """Forward pass of softmax with masked input.

        In case attn_mask_type is causal the mask is generated and None can be passed.
        A user-defined mask is only needed when attn_mask_type is not causal.
        """
        # [b, np, sq, sk]
        assert input.dim() == 4

        if (
            self.is_kernel_available(mask, *input.size())
            and self.softmax_denominator_weight is None
        ):
            return self.forward_fused_softmax(input, mask)
        else:
            return self.forward_torch_softmax(input, mask)

    def is_kernel_available(self, mask, b, np, sq, sk):
        attn_batches = b * np

        if (
            self.scaled_masked_softmax_fusion  # user want to fuse
            and self.input_in_float16  # input must be fp16
            and 16 < sk <= 4096  # sk must be 16 ~ 2048
            and sq % 4 == 0  # sq must be divisor of 4
            and sk % 4 == 0  # sk must be divisor of 4
            and attn_batches % 4 == 0  # np * b must be divisor of 4
        ):
            if 0 <= sk <= 4096:
                batch_per_block = self.get_batch_per_block(sq, sk, b, np)

                if self.attn_mask_type == AttnMaskType.causal:
                    if attn_batches % batch_per_block == 0:
                        return True
                else:
                    if sq % batch_per_block == 0:
                        return True
        return False

    def forward_fused_softmax(self, input, mask):
        b, np, sq, sk = input.size()
        scale = self.scale if self.scale is not None else 1.0

        if self.attn_mask_type == AttnMaskType.causal:
            assert sq == sk, "causal mask is only for self attention"

            # input is 3D tensor (attn_batches, sq, sk)
            input = input.view(-1, sq, sk)
            probs = ScaledUpperTriangMaskedSoftmax.apply(input, scale)
            return probs.view(b, np, sq, sk)
        else:
            # input is 4D tensor (b, np, sq, sk)
            if mask is not None:
                return ScaledMaskedSoftmax.apply(input, mask, scale)
            else:
                return ScaledSoftmax.apply(input, scale)

    def forward_torch_softmax(self, input, mask):
        if self.input_in_float16 and self.softmax_in_fp32:
            input = input.float()

        if self.scale is not None:
            input = input * self.scale

        # Generate causal mask if not given
        sq, sk = input.size(2), input.size(3)
        if self.attn_mask_type == AttnMaskType.causal and mask is None and sq > 1:
            # If sq == 1 then either KV cache is used or one-element context is passed
            # so keeping mask=None in this case; subsequent code should handle it
            assert sq == sk, "causal mask is only for self attention"
            mask = get_default_causal_mask(sq)

        mask_output = self.mask_func(input, mask) if mask is not None else input
        if self.softmax_denominator_weight is None:
            softmax_fn = torch.nn.Softmax(dim=-1)
        else:
            softmax_fn = SoftmaxOne(-1, self.softmax_denominator_weight)

        probs = softmax_fn(mask_output)

        if self.input_in_float16 and self.softmax_in_fp32:
            if self.input_in_fp16:
                probs = probs.half()
            else:
                probs = probs.bfloat16()

        return probs

    @staticmethod
    def get_batch_per_block(sq, sk, b, np):
        import scaled_masked_softmax_cuda

        return scaled_masked_softmax_cuda.get_batch_per_block(sq, sk, b, np)
