# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Which backend a Transformer Engine run uses for each operation.

Every entry is one line naming a class in :mod:`megatron.core.ops`, so this file is the whole
answer to "which implementation does a TE run get". The choices themselves -- when TE cannot
serve query/key norm, when a residual can be fused -- belong to those classes, not here.
"""

from __future__ import annotations

from typing import Optional

from megatron.core.models.backends import BackendSpecProvider
from megatron.core.ops import require
from megatron.core.ops.attention import AttentionTE
from megatron.core.ops.linear import LinearTE
from megatron.core.ops.loss import LossMegatron, LossMegatronFused, VocabParallelCrossEntropy
from megatron.core.ops.mlp import MlpMegatron, MlpTEOpFuser
from megatron.core.ops.moe import MoeTE
from megatron.core.ops.norm import NormTE
from megatron.core.transformer.mlp import TEActivationFunctionBuilder
from megatron.core.transformer.moe.moe_layer import ExpertsBuilder
from megatron.core.transformer.torch_norm import LayerNormBuilder


class TESpecProvider(BackendSpecProvider):
    """Every backend a Transformer Engine run uses."""

    def __init__(
        self, use_te_op_fuser: bool = False, cross_entropy_loss_fusion: bool = False
    ) -> None:
        require("transformer_engine")
        self._norm = NormTE()
        self._linear = LinearTE()
        self._attention = AttentionTE()
        # The operation fuser folds the linears it is handed into TE operations and can only
        # convert TE ones, which is why it is offered here and not by the local provider.
        self._mlp = MlpTEOpFuser() if use_te_op_fuser else MlpMegatron()
        self._moe = MoeTE()
        self._loss = LossMegatronFused() if cross_entropy_loss_fusion else LossMegatron()

    def linear(self) -> type:
        """Which non-parallel linear module the backend uses."""
        return self._linear.linear()

    def column_parallel_linear(self) -> type:
        """Which column parallel linear module the backend uses."""
        return self._linear.column_parallel_linear()

    def row_parallel_linear(self) -> type:
        """Which row parallel linear module the backend uses."""
        return self._linear.row_parallel_linear()

    def fuse_layernorm_and_linear(self) -> bool:
        """Transformer Engine chooses a single module for layernorm and linear."""
        return True

    def column_parallel_layer_norm_linear(self) -> Optional[type]:
        """Which module for sequential layernorm and linear."""
        return self._linear.column_parallel_layer_norm_linear()

    def layer_norm(
        self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False
    ) -> LayerNormBuilder:
        """Which module to use for layer norm."""
        return self._norm.layer_norm(rms_norm=rms_norm, for_qk=for_qk, has_residual=has_residual)

    def core_attention(self) -> type:
        """Which module to use for attention."""
        return self._attention.core_attention()

    def grouped_mlp_modules(self, moe_use_grouped_gemm: bool) -> ExpertsBuilder:
        """Which module and submodules to use for grouped mlp."""
        return self._moe.grouped_mlp_modules(moe_use_grouped_gemm)

    def activation_func(self) -> TEActivationFunctionBuilder | None:
        """Which module to use for activation function."""
        return self._moe.activation_func()

    def mlp_module(self, grouped: bool = False) -> type:
        """Which module to use for the dense MLP block."""
        return self._mlp.mlp_module(grouped=grouped)

    def moe_router(self) -> Optional[type]:
        """Which MoE router to use, or None to keep the MoESubmodules default."""
        return self._moe.moe_router()

    def vocab_parallel_cross_entropy(self) -> VocabParallelCrossEntropy:
        """Which vocab-parallel cross entropy to use."""
        return self._loss.vocab_parallel_cross_entropy()
