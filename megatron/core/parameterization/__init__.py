# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from .model_policy import ResolvedModelPolicy, build_resolved_model_policy
from .roles import (
    PARAMETERIZATION_ROLE_ATTR,
    PARAMETERIZATION_SHARED_GROUP_ATTR,
    PARAMETERIZATION_TAGS_ATTR,
    ROLE_BLOCK_OUT_PROJ,
    ROLE_EMBEDDING,
    ROLE_HIDDEN_MATRIX,
    ROLE_MUON_MANAGED_MATRIX,
    ROLE_OUTPUT,
    ROLE_SHARED_EMBEDDING_OUTPUT,
    ROLE_VECTOR_LIKE,
    set_parameterization_metadata,
)
from .spec import (
    CanonicalScalingSpec,
    ResolvedScalingContext,
    SCALING_RECIPE_MUP,
    SCALING_RECIPE_NONE,
    ScalingReferences,
    ScalingUserConfig,
    build_resolved_scaling_context,
    build_scaling_user_config,
    canonicalize_scaling_user_config,
    sync_legacy_mup_fields,
)
from .training_policy import ResolvedTrainingPolicy, build_resolved_training_policy

__all__ = [
    'CanonicalScalingSpec',
    'PARAMETERIZATION_ROLE_ATTR',
    'PARAMETERIZATION_SHARED_GROUP_ATTR',
    'PARAMETERIZATION_TAGS_ATTR',
    'ROLE_BLOCK_OUT_PROJ',
    'ROLE_EMBEDDING',
    'ROLE_HIDDEN_MATRIX',
    'ROLE_MUON_MANAGED_MATRIX',
    'ROLE_OUTPUT',
    'ROLE_SHARED_EMBEDDING_OUTPUT',
    'ROLE_VECTOR_LIKE',
    'ResolvedModelPolicy',
    'ResolvedScalingContext',
    'ResolvedTrainingPolicy',
    'SCALING_RECIPE_MUP',
    'SCALING_RECIPE_NONE',
    'ScalingReferences',
    'ScalingUserConfig',
    'build_resolved_model_policy',
    'build_resolved_scaling_context',
    'build_resolved_training_policy',
    'build_scaling_user_config',
    'canonicalize_scaling_user_config',
    'set_parameterization_metadata',
    'sync_legacy_mup_fields',
]
