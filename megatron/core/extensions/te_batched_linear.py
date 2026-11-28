# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DSv4's matrix-parameter contract around Transformer Engine's batched operation."""

import copy
from types import MethodType
from typing import Callable, Optional

import torch

from megatron.core.quantization.quant_config import QuantizationConfig
from megatron.core.quantization.utils import get_quant_config_or_none
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    from transformer_engine.pytorch.ops import BatchedLinear
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager
    from transformer_engine.pytorch.tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage

    from megatron.core.extensions.transformer_engine import (
        TEQuantizationParams,
        _get_fp8_autocast_for_quant_params,
        _get_fp8_model_init_for_quant_params,
    )
except ImportError:
    BatchedLinear = None


def _get_high_precision_init_val(param):
    """Return the unquantized value consumed by MCore's master-weight initialization."""
    return param._high_precision_init_val


def _clear_high_precision_init_val(param):
    """Release the temporary initializer after the optimizer has consumed it."""
    param._high_precision_init_val = None


TEBatchedLinear: type | None = None

if BatchedLinear is not None:

    class _TEBatchedLinear(BatchedLinear):
        """Keep one 2D optimizer parameter while TE computes with 3D tensor views.

        DSv4 historically updates the whole [groups * out_features, in_features]
        matrix with Muon. A 3D parameter instead selects the scalar optimizer.
        The operation view shares storage, but never replaces the registered
        parameter during forward/backward or changes its shape in place.

        Training with MFSDPv2 phase-sharded MXFP8 primary weights is unsupported:
        TE saves a weight view that needs both orientations available in forward.
        BF16 primary weights and replicated MXFP8 compute weights are unaffected.
        """

        weight: torch.nn.Parameter
        _init_method: Callable[[torch.Tensor], object]

        def __init__(self, *args, config: Optional[TransformerConfig] = None, **kwargs) -> None:
            if kwargs.get("bias", False):
                raise ValueError("DSv4 batched output projection does not support bias.")
            kwargs["bias"] = False
            # Like TELinear: resolve the per-module override from config.quant_recipe and
            # allocate the weight under its storage choice. TE reads the model-init state in
            # __init__, so a later finish_init() can only change the forward precision.
            self.config = config
            self.te_quant_params: Optional[TEQuantizationParams] = None
            if config is not None:
                self.finish_init(get_quant_config_or_none(kwargs.get("name"), config.quant_recipe))
            init_quant_context = _get_fp8_model_init_for_quant_params(
                self.te_quant_params, torch.is_grad_enabled()
            )
            with init_quant_context:
                self._preserve_initial_value = FP8GlobalStateManager.with_high_precision_init_val()
                super().__init__(*args, **kwargs)
            self._flatten_weight()

        def finish_init(self, quantization_config: Optional[QuantizationConfig]) -> None:
            """Post-init of quantization override"""
            if quantization_config is None:
                self.te_quant_params = None
            else:
                self.te_quant_params = TEQuantizationParams.parse_from_config(
                    quantization_config, model_config=self.config
                )

        def forward(self, *args, **kwargs):
            """Run under the per-module training or evaluation recipe, as TELinear does."""
            with _get_fp8_autocast_for_quant_params(self.te_quant_params, self.training):
                return super().forward(*args, **kwargs)

        def _flatten_weight(self) -> None:
            # Initialization/materialization only, before optimizer/fuser ownership.
            # Later resets already have the canonical shape and retain the object.
            if self.weight.ndim != 2:
                self.weight = torch.nn.Parameter(
                    self.weight.reshape(-1, self.in_features),
                    requires_grad=self.weight.requires_grad,
                )

        def reset_parameters(self) -> None:
            """Preserve the requested unquantized initializer for the FP32 master."""
            initializer = self._init_method
            initial_value = None
            cpu_weight = (
                self.weight
                if self._with_quantized_weight and self.weight.device.type == "cpu"
                else None
            )

            def initialize(weight):
                nonlocal initial_value
                initializer(weight)
                if self._with_quantized_weight and self._preserve_initial_value:
                    initial_value = (
                        weight.detach().reshape(-1, self.in_features).to("cpu", copy=True)
                    )

            self._init_method = initialize
            try:
                super().reset_parameters()
                if cpu_weight is not None:
                    # TE quantization can read CPU storage asynchronously. Keep the
                    # initializer alive until that read completes before releasing it.
                    torch.cuda.current_stream(self.weight.device).synchronize()
            finally:
                self._init_method = initializer
            self._flatten_weight()
            if initial_value is not None:
                self.weight._high_precision_init_val = initial_value
                self.weight.get_high_precision_init_val = MethodType(
                    _get_high_precision_init_val, self.weight
                )
                self.weight.clear_high_precision_init_val = MethodType(
                    _clear_high_precision_init_val, self.weight
                )

        def _grouped_operation(self):
            """Make an operation-local view; keep the parameter and optimizer state 2D."""
            operation = copy.copy(self)
            operation._parameters = self._parameters.copy()
            weight = self.weight.view(self.batch_size, self.out_features, self.in_features)
            # TE quantized views use an autograd Function. Inside the fuser's
            # no-grad forward that view loses requires_grad, but TE still needs
            # this flag to decide whether to save inputs and compute weight grads.
            weight.requires_grad_(self.weight.requires_grad)
            # TE reads these fields from its operation weight, including FSDP's
            # lazy main-grad accessor. Refresh them for every backward, after zeroing
            # or buffer relocation; do not cache a view of an old main-grad buffer.
            for name in (
                "main_grad",
                "get_main_grad",
                "__fsdp_param__",
                "overwrite_main_grad",
                "grad_added_to_main_grad",
                "zero_out_wgrad",
                "zero_tensor_dummy_wgrad",
            ):
                if hasattr(self.weight, name):
                    setattr(weight, name, getattr(self.weight, name))
            operation._parameters["weight"] = weight
            return operation

        def _validate_parameters(self) -> None:
            if self.weight.ndim == 3:
                return super()._validate_parameters()
            return BatchedLinear._validate_parameters(self._grouped_operation())

        def op_forward(self, ctx, input_, **kwargs):
            """Delegate arithmetic and quantization to TE using the grouped view."""
            if (
                ctx.requires_grad
                and getattr(self.weight, "_mfsdp_parameter_group", None) is not None
                and isinstance(self.weight, MXFP8TensorStorage)
                and (
                    self.weight._columnwise_data is None
                    or self.weight._columnwise_scale_inv is None
                )
            ):
                raise NotImplementedError(
                    "DSv4's native batched projection does not support MFSDPv2 "
                    "phase-sharded MXFP8 primary weights. Use BF16 primary weights "
                    "(disable fp8_param) or replicated compute weights. Both MXFP8 "
                    "orientations must be available before the training forward."
                )
            return BatchedLinear.op_forward(self._grouped_operation(), ctx, input_, **kwargs)

        def op_backward(self, ctx, grad_output):
            """Return matrix gradients and publish fused accumulation on the real owner."""
            operation = self._grouped_operation()
            grad_input, grad_params = BatchedLinear.op_backward(operation, ctx, grad_output)
            if self._accumulate_into_main_grad:
                for name in ("main_grad", "grad_added_to_main_grad"):
                    if hasattr(operation.weight, name):
                        setattr(self.weight, name, getattr(operation.weight, name))
            grad_weight = list(grad_params)[0]
            if grad_weight is not None:
                grad_weight = grad_weight.reshape(self.weight.shape)
            return grad_input, [grad_weight]

    TEBatchedLinear = _TEBatchedLinear
