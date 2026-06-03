# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from .model_policy import ModelScalingPolicy, build_model_scaling_policy
from .roles import (
    IS_OUTPUT_PARAMETER_ATTR,
    PARAMETERIZATION_ROLE_ATTR,
    PARAMETERIZATION_SHARED_GROUP_ATTR,
    PARAMETERIZATION_TAGS_ATTR,
    ROLE_EMBEDDING,
    ROLE_HIDDEN_MATRIX,
    ROLE_HIDDEN_VECTOR,
    ROLE_MUON_MANAGED_MATRIX,
    ROLE_OUTPUT,
    ROLE_SHARED_EMBEDDING_OUTPUT,
    get_parameterization_role,
    is_embedding_class_parameter,
    is_embedding_or_output_parameter,
    is_hidden_matrix_parameter,
    is_muon_managed_matrix_parameter,
    is_output_parameter,
    is_vector_like_parameter,
    set_parameterization_metadata,
)
from .spec import ScalingContext, build_scaling_context
from .training_policy import TrainingScalingPolicy, build_legacy_mup_training_policy

__all__ = [
    'IS_OUTPUT_PARAMETER_ATTR',
    'PARAMETERIZATION_ROLE_ATTR',
    'PARAMETERIZATION_SHARED_GROUP_ATTR',
    'PARAMETERIZATION_TAGS_ATTR',
    'ROLE_EMBEDDING',
    'ROLE_HIDDEN_MATRIX',
    'ROLE_HIDDEN_VECTOR',
    'ROLE_MUON_MANAGED_MATRIX',
    'ROLE_OUTPUT',
    'ROLE_SHARED_EMBEDDING_OUTPUT',
    'ModelScalingPolicy',
    'ScalingContext',
    'TrainingScalingPolicy',
    'build_legacy_mup_training_policy',
    'build_model_scaling_policy',
    'build_scaling_context',
    'get_parameterization_role',
    'is_embedding_class_parameter',
    'is_embedding_or_output_parameter',
    'is_hidden_matrix_parameter',
    'is_muon_managed_matrix_parameter',
    'is_output_parameter',
    'is_vector_like_parameter',
    'set_parameterization_metadata',
]
