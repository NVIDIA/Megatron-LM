# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""TE batched projection with the matrix-shaped parameter used by DSv4 optimizers."""

import torch

try:
    from transformer_engine.pytorch.ops import BatchedLinear
    from transformer_engine.pytorch.ops._common import is_quantized_tensor
    from transformer_engine.pytorch.tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage
except ImportError:
    BatchedLinear = None


if BatchedLinear is not None:

    class DSv4BatchedLinear(BatchedLinear):
        """Keep one [groups * out_features, in_features] parameter for the optimizer.

        TE's strided GEMMs receive explicit matrix dimensions and strides, so their
        physical storage is unchanged. Keeping the legacy 2D logical shape also
        preserves Muon's matrix classification and whole-matrix update semantics.
        """

        def __init__(self, num_gemms: int, in_features: int, out_features: int, **kwargs) -> None:
            if kwargs.get('bias', False):
                raise ValueError('DSv4 grouped output projection does not support bias.')
            kwargs['bias'] = False
            super().__init__(num_gemms, in_features, out_features, **kwargs)
            # Meta-device construction defers reset_parameters until materialization.
            self._flatten_weight()

        def _flatten_weight(self) -> None:
            weight = self.weight
            if weight.ndim == 2:
                return
            flat_shape = (self.num_gemms * self.out_features, self.in_features)
            self.weight = torch.nn.Parameter(
                weight.reshape(flat_shape), requires_grad=weight.requires_grad
            )
            # Preserve the high-precision initialization used to create optimizer masters.
            if hasattr(weight, '_high_precision_init_val'):
                initial_value = weight._high_precision_init_val
                self.weight._high_precision_init_val = (
                    initial_value.reshape(flat_shape) if initial_value is not None else None
                )

        def reset_parameters(self) -> None:
            """Initialize using the existing TE recipe, then expose a single matrix."""
            super().reset_parameters()
            self._flatten_weight()

        def _validate_parameters(self) -> None:
            expected = (self.num_gemms * self.out_features, self.in_features)
            if tuple(self.weight.shape) != expected:
                raise ValueError(f'DSv4 grouped weight must have shape {expected}.')
            if self.weight.device.type != 'cuda' or not self.weight.is_contiguous():
                raise ValueError('DSv4 grouped weight must be a contiguous CUDA tensor.')
            if is_quantized_tensor(self.weight):
                if not isinstance(self.weight, MXFP8TensorStorage):
                    raise ValueError('DSv4 BatchedLinear supports only MXFP8 quantized weights.')
                if self.weight._with_gemm_swizzled_scales:
                    raise ValueError('DSv4 BatchedLinear requires compact MXFP8 scales.')

        def _op_backward(self, ctx, grad_output, grad_returned_bias):
            grad_input, (grad_weight,) = super()._op_backward(ctx, grad_output, grad_returned_bias)
            if grad_weight is not None:
                grad_weight = grad_weight.reshape(self.weight.shape)
            return grad_input, [grad_weight]

else:
    DSv4BatchedLinear = None
