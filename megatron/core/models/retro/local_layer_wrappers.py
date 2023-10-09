# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

? ? ?

from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
# from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
# from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer import MegatronModule, TransformerConfig


class LocalLayerNorm(MegatronModule):

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        sequence_parallel: bool = False,
        normalization: str = "LayerNorm",
        **kwargs
    ):
        super().__init__(config=config)

        # >>>
        # config: TransformerConfig=None, # included for build_module interface
        # normalization: str=None, # included to match TE interface
        # <<<

        assert normalization == "LayerNorm"

        self.norm = FusedLayerNorm(
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
            persist_layer_norm=self.config.persist_layer_norm,
            sequence_parallel=self.config.sequence_parallel,
            zero_centered_gamma=self.config.layernorm_zero_centered_gamma,
            # normalization=self.config.normalization,
        )

# class LocalDotProductAttention(DotProductAttention):
#     """Wrapper for the local `DotProductAttention` layer."""

#     def __init__(
#         self,
#         config: TransformerConfig,
#         layer_number: int = 1,
#         attn_mask_type: AttnMaskType = AttnMaskType.padding,
#         attention_dropout: float = None,
#         **kwargs
#     ):
