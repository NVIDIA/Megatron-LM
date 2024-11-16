# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.ssm.mamba_block import MambaStack, MambaStackSubmodules
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.ssm.mamba_mixer import MambaMixer, MambaMixerSubmodules
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules

try:
    from megatron.core.extensions.transformer_engine import (
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TERowParallelLinear,
    )
    HAVE_TE = True

except ImportError:
    from megatron.core.transformer.dot_product_attention import DotProductAttention
    from megatron.core.transformer.torch_layer_norm import WrappedTorchLayerNorm
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
    import warnings

    if torch.cuda.is_available():
        warnings.warn('Transformer Engine is not installed. Falling back to Megatron Local')
    
    HAVE_TE = False

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

if HAVE_TE:
    mamba_stack_spec = ModuleSpec(
        module=MambaStack,
        submodules=MambaStackSubmodules(
            mamba_layer=ModuleSpec(
                module=MambaLayer,
                submodules=MambaLayerSubmodules(
                    mixer=ModuleSpec(
                        module=MambaMixer,
                        submodules=MambaMixerSubmodules(
                            in_proj=TELayerNormColumnParallelLinear,
                            out_proj=TERowParallelLinear,
                        ),
                    ),
                    mamba_bda=get_bias_dropout_add,
                ),
            ),
            # Started with spec from gpt_layer_specs.py (with MLP removed)
            # Using the TE spec because we had problems getting the non-TE spec
            # working
            attention_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    self_attention=ModuleSpec(
                        module=SelfAttention,
                        params={"attn_mask_type": AttnMaskType.causal},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=TELayerNormColumnParallelLinear,
                            core_attention=TEDotProductAttention,
                            linear_proj=TERowParallelLinear,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                ),
            ),
            # Started with spec from gpt_layer_specs.py
            # Using the TE spec because we had problems getting the non-TE spec
            # working
            mlp_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    mlp=ModuleSpec(
                        module=MLP,
                        submodules=MLPSubmodules(
                            linear_fc1=TELayerNormColumnParallelLinear,
                            linear_fc2=TERowParallelLinear,
                        ),
                    ),
                    mlp_bda=get_bias_dropout_add,
                ),
            ),
        ),
    )
else:
    mamba_stack_spec = ModuleSpec(
        module=MambaStack,
        submodules=MambaStackSubmodules(
            mamba_layer=ModuleSpec(
                module=MambaLayer,
                submodules=MambaLayerSubmodules(
                    norm=WrappedTorchLayerNorm,
                    mixer=ModuleSpec(
                        module=MambaMixer,
                        submodules=MambaMixerSubmodules(
                            in_proj=ColumnParallelLinear,
                            out_proj=RowParallelLinear,
                        ),
                    ),
                    mamba_bda=get_bias_dropout_add,
                ),
            ),
            # Started with spec from gpt_layer_specs.py (with MLP removed)
            # Using the TE spec because we had problems getting the non-TE spec
            # working
            attention_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=WrappedTorchLayerNorm,
                    self_attention=ModuleSpec(
                        module=SelfAttention,
                        params={"attn_mask_type": AttnMaskType.causal},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=ColumnParallelLinear,
                            core_attention=DotProductAttention,
                            linear_proj=RowParallelLinear,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                ),
            ),
            # Started with spec from gpt_layer_specs.py
            # Using the TE spec because we had problems getting the non-TE spec
            # working
            mlp_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=WrappedTorchLayerNorm,
                    mlp=ModuleSpec(
                        module=MLP,
                        submodules=MLPSubmodules(
                            linear_fc1=ColumnParallelLinear,
                            linear_fc2=RowParallelLinear,
                        ),
                    ),
                    mlp_bda=get_bias_dropout_add,
                ),
            ),
        ),
    )
