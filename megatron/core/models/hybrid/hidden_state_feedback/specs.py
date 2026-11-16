# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Module specs for hidden-state fusion."""

from megatron.core.extensions.transformer_engine import TELinear, TENorm
from megatron.core.models.hybrid.hidden_state_feedback.fusion import (
    AdditiveHiddenStateFusion,
    AdditiveHiddenStateFusionSubmodules,
    GatedHiddenStateFusion,
    GatedHiddenStateFusionSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec

additive_hidden_state_fusion_spec = ModuleSpec(
    module=AdditiveHiddenStateFusion,
    submodules=AdditiveHiddenStateFusionSubmodules(
        hidden_state_norm=TENorm, hidden_state_projection=TELinear
    ),
)

gated_hidden_state_fusion_spec = ModuleSpec(
    module=GatedHiddenStateFusion,
    submodules=GatedHiddenStateFusionSubmodules(
        token_embedding_norm=TENorm,
        output_norm=TENorm,
        hidden_state_projection=TELinear,
        gate_projection=TELinear,
    ),
)
