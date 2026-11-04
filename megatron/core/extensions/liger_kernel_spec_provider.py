# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Local modules with opt-in Liger RMSNorm."""

from megatron.core.models.backends import LocalSpecProvider
from megatron.core.ops._backends import require
from megatron.core.ops.norm.torch_norm import LayerNormBuilder


class LigerSpecProvider(LocalSpecProvider):
    """Replace local RMSNorm without changing any other provider slots."""

    def layer_norm(
        self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False
    ) -> LayerNormBuilder:
        """Select Liger only for RMSNorm; retain the local LayerNorm implementation."""
        if not rms_norm:
            return super().layer_norm(rms_norm=rms_norm, for_qk=for_qk, has_residual=has_residual)
        require("liger_kernel.megatron", "LigerMegatronRMSNorm", needed_by="Liger RMSNorm")
        from megatron.core.ops.norm.liger_norm import LigerRMSNorm

        return LigerRMSNorm
