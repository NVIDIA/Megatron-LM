# Copyright (c) 2026, ETH Zurich / Swiss AI Initiative.
#
# Mamba 3 stack spec: hybrid_stack_spec with the Mamba 2 mixer swapped for our
# Mamba3Mixer wrapper. Layer pattern (M-M-M-*-) and other layer types
# (attention, MLP, MoE) are inherited unchanged from hybrid_stack_spec.

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.ssm.mamba3_mixer import Mamba3Mixer, Mamba3MixerSubmodules
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec


# Mamba 3 mamba-layer slot. Mixer points to Mamba3Mixer (wraps state-spaces'
# Mamba3); everything else (BDA, attention layers, MLP layers) is inherited
# from hybrid_stack_spec.
_mamba3_layer_spec = ModuleSpec(
    module=MambaLayer,
    submodules=MambaLayerSubmodules(
        mixer=ModuleSpec(
            module=Mamba3Mixer,
            submodules=Mamba3MixerSubmodules(),
        ),
        mamba_bda=get_bias_dropout_add,
    ),
)


# Build a new stack spec that reuses everything from hybrid_stack_spec except
# the mamba_layer slot.
_h = hybrid_stack_spec.submodules
mamba3_stack_spec = ModuleSpec(
    module=HybridStack,
    submodules=HybridStackSubmodules(
        mamba_layer=_mamba3_layer_spec,
        gdn_layer=_h.gdn_layer,
        attention_layer=_h.attention_layer,
        dsa_layer=_h.dsa_layer,
        mlp_layer=_h.mlp_layer,
        moe_layer=_h.moe_layer,
        mtp_block_spec=_h.mtp_block_spec,
    ),
)
