# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Native BAGEL linear kernels for exact vision-connector alignment."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.utils import get_pg_size


def _validate_native_connector_linear(module: torch.nn.Module) -> None:
    """Fail fast when the native connector's TP=1 contract is not satisfied."""

    if get_pg_size(module.tp_group) != 1:
        raise ValueError("Native BAGEL connector alignment requires tensor parallel size 1")
    if getattr(module.config, "fp8", None):
        raise ValueError("Native BAGEL connector alignment does not support FP8")


class BagelNativeColumnParallelLinear(ColumnParallelLinear):
    """TP=1 linear matching native BAGEL's ``torch.nn.Linear`` call boundary.

    The native connector passes the bias directly to ``F.linear``.  MCore's
    projector normally uses a Transformer-Engine GEMM with ``skip_bias_add``
    and adds the bias afterward in ``MLP``.  Those paths are mathematically
    equivalent but not bitwise equivalent in BF16.  This class preserves the
    existing parameter layout, state-dict keys, and MLP interface while using
    the native operation exactly.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        _validate_native_connector_linear(self)
        if self.gather_output:
            raise ValueError("Native BAGEL connector alignment does not gather TP output")
        if self.weight is None:
            raise ValueError("Native BAGEL connector alignment requires an owned weight")

    def forward(
        self,
        input_: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if runtime_gather_output:
            raise ValueError("Native BAGEL connector alignment does not gather TP output")
        if weight is not None and weight is not self.weight:
            raise ValueError("Native BAGEL connector alignment does not accept a runtime weight")

        # Fuse the bias into F.linear even though MLP constructed this module
        # with skip_bias_add=True.  Returning None prevents MLP from adding it
        # for a second time and exactly mirrors native nn.Linear.
        return F.linear(input_, self.weight, self.bias), None


class BagelNativeRowParallelLinear(RowParallelLinear):
    """TP=1 output linear matching native BAGEL's second ``nn.Linear``."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        _validate_native_connector_linear(self)
        if not self.input_is_parallel:
            raise ValueError("Native BAGEL connector alignment expects input_is_parallel=True")

    def forward(self, input_: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # TP=1 means no scatter/reduce is needed.  Keeping one direct F.linear
        # also preserves native BAGEL's autograd GEMM boundaries.
        return F.linear(input_, self.weight, self.bias), None
