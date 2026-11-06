# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""TE batched projection with the matrix-shaped parameter used by DSv4 optimizers."""

import torch

try:
    from transformer_engine.pytorch.ops import BatchedLinear
    from transformer_engine.pytorch.ops._common import is_quantized_tensor
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager
    from transformer_engine.pytorch.tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage
except ImportError:
    BatchedLinear = None


if BatchedLinear is not None:

    class _GroupedWeightView:
        """Give TE a grouped storage view without replacing its optimizer parameter.

        All operation methods stay bound to the real owner, including quantizer
        state, validation and main-gradient bookkeeping. Only the forward GEMM's
        weight read needs the [groups, out_features, in_features] geometry.
        """

        def __init__(self, operation) -> None:
            self.operation = operation

        @property
        def weight(self):
            operation = self.operation
            weight = operation.weight
            grouped = weight.view(
                operation.batch_size, operation.out_features, operation.in_features
            )
            # MXFP8's custom view creates a new storage wrapper. Under the
            # fuser's no-grad forward it does not inherit requires_grad, which
            # TE uses to decide whether to compute this parameter's wgrad.
            if grouped.requires_grad != weight.requires_grad:
                grouped.requires_grad_(weight.requires_grad)
            return grouped

        def __getattr__(self, name):
            return getattr(self.operation, name)

    class DSv4BatchedLinear(BatchedLinear):
        """Keep one [groups * out_features, in_features] parameter for the optimizer.

        A grouped view supplies TE's batched GEMM geometry without changing the
        parameter or its quantized storage. The 2D parameter preserves Muon's
        matrix classification and whole-matrix update semantics.
        """

        def __init__(self, batch_size: int, in_features: int, out_features: int, **kwargs) -> None:
            if kwargs.get('bias', False):
                raise ValueError('DSv4 grouped output projection does not support bias.')
            kwargs['bias'] = False
            # Deferred meta initialization may occur after quantized_model_init exits.
            self._preserve_high_precision_init_val = (
                FP8GlobalStateManager.with_high_precision_init_val()
            )
            super().__init__(batch_size, in_features, out_features, **kwargs)
            # Meta-device construction defers reset_parameters until materialization.
            self._flatten_weight()

        def _flatten_weight(self) -> None:
            weight = self.weight
            if weight.ndim == 2:
                return
            flat_shape = (self.batch_size * self.out_features, self.in_features)
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
            init_method = self._init_method
            initial_value = None
            preserve_initial_value = (
                self._with_quantized_weight and self._preserve_high_precision_init_val
            )

            def initialize(weight):
                nonlocal initial_value
                result = init_method(weight)
                # TE's operation currently omits the high-precision initializer
                # retained by its Linear modules for optimizer master weights.
                initial_value = weight.detach().clone()
                return result

            if preserve_initial_value:
                self._init_method = initialize
            try:
                super().reset_parameters()
            finally:
                self._init_method = init_method
            self._flatten_weight()
            if initial_value is not None:
                self.weight._high_precision_init_val = initial_value.reshape(self.weight.shape)

        def _validate_parameters(self) -> None:
            expected = (self.batch_size * self.out_features, self.in_features)
            if tuple(self.weight.shape) != expected:
                raise ValueError(f'DSv4 grouped weight must have shape {expected}.')
            if self.weight.device.type != 'cuda' or not self.weight.is_contiguous():
                raise ValueError('DSv4 grouped weight must be a contiguous CUDA tensor.')
            if is_quantized_tensor(self.weight):
                if not isinstance(self.weight, MXFP8TensorStorage):
                    raise ValueError('DSv4 BatchedLinear supports only MXFP8 quantized weights.')
                if self.weight._with_gemm_swizzled_scales:
                    raise ValueError('DSv4 BatchedLinear requires compact MXFP8 scales.')

        def op_forward(self, ctx, input_, **kwargs):
            """Use TE's native quantization and capture path with a zero-copy 3D view."""
            return BatchedLinear.op_forward(_GroupedWeightView(self), ctx, input_, **kwargs)

        def op_backward(self, ctx, grad_output):
            """Return flat gradients while TE accumulates directly into flat main_grad."""
            grad_input, (grad_weight,) = super().op_backward(ctx, grad_output)
            if grad_weight is not None:
                grad_weight = grad_weight.reshape(self.weight.shape)
            return grad_input, [grad_weight]

else:
    DSv4BatchedLinear = None
