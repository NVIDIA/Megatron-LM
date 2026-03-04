# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""
Layer specifications for the Engram-augmented GPT model.

Provides factory functions that produce ModuleSpec objects pointing to
EngramTransformerLayer (instead of the standard TransformerLayer) while
reusing the standard GPT submodule wiring for attention and MLP.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.backends import LocalSpecProvider
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules

from megatron.core.models.engram.engram_layer import EngramTransformerLayer
from megatron.core.models.engram.engram_module import EngramConfig

try:
    import apex  # type: ignore
    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
    LNImpl = FusedLayerNorm
except ImportError:
    from megatron.core.transformer.torch_norm import WrappedTorchNorm
    LNImpl = WrappedTorchNorm


def get_engram_layer_local_spec(
    engram_config: EngramConfig,
    vocab_size_across_layers: Dict[int, List[List[int]]],
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    normalization: Optional[str] = None,
    qk_l2_norm: Optional[bool] = False,
) -> ModuleSpec:
    """Build a ModuleSpec for EngramTransformerLayer using Megatron-Core-only modules.

    This mirrors ``get_gpt_layer_local_spec`` but substitutes EngramTransformerLayer
    and passes the engram configuration via the spec's ``params`` dict so that
    ``build_module`` can forward them to the layer constructor.

    Args:
        engram_config: Engram-specific configuration.
        vocab_size_across_layers: Mapping from layer_id to per-ngram-level head
            vocab sizes, as computed by NgramHashMapping.
        Other args match those of ``get_gpt_layer_local_spec``.

    Returns:
        ModuleSpec targeting EngramTransformerLayer.
    """
    backend = LocalSpecProvider()

    if normalization == "RMSNorm":
        layer_norm = backend.layer_norm(rms_norm=True, for_qk=False)
        qk_norm = backend.layer_norm(rms_norm=True, for_qk=True)
    else:
        layer_norm = backend.layer_norm(rms_norm=False, for_qk=False)
        qk_norm = backend.layer_norm(rms_norm=False, for_qk=True)

    mlp = _get_mlp_spec(backend, num_experts, moe_grouped_gemm)

    submodules = TransformerLayerSubmodules(
        input_layernorm=layer_norm,
        self_attention=ModuleSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=backend.column_parallel_linear(),
                core_attention=backend.core_attention(),
                linear_proj=backend.row_parallel_linear(),
                q_layernorm=qk_norm if qk_layernorm else IdentityOp,
                k_layernorm=qk_norm if qk_layernorm else IdentityOp,
            ),
        ),
        self_attn_bda=get_bias_dropout_add,
        pre_mlp_layernorm=layer_norm,
        mlp=mlp,
        mlp_bda=get_bias_dropout_add,
    )

    return ModuleSpec(
        module=EngramTransformerLayer,
        submodules=submodules,
        params={
            "engram_config": engram_config,
            "engram_vocab_size_across_layers": vocab_size_across_layers,
        },
    )


def _get_mlp_spec(
    backend: LocalSpecProvider,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
) -> ModuleSpec:
    """Build a dense MLP spec using the backend provider."""
    if num_experts is None:
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=backend.column_parallel_linear(),
                linear_fc2=backend.row_parallel_linear(),
            ),
        )
    else:
        from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec_for_backend
        return get_mlp_module_spec_for_backend(
            backend=backend,
            num_experts=num_experts,
            moe_grouped_gemm=moe_grouped_gemm,
        )
