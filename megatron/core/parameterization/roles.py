# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from typing import Any, Iterable, Optional

PARAMETERIZATION_ROLE_ATTR = 'parameterization_role'
PARAMETERIZATION_SHARED_GROUP_ATTR = 'parameterization_shared_group'
PARAMETERIZATION_TAGS_ATTR = 'parameterization_tags'
IS_OUTPUT_PARAMETER_ATTR = 'is_output_parameter'

ROLE_EMBEDDING = 'embedding'
ROLE_OUTPUT = 'output'
ROLE_SHARED_EMBEDDING_OUTPUT = 'shared_embedding_output'
ROLE_BLOCK_OUT_PROJ = 'block_out_proj'
ROLE_HIDDEN_MATRIX = 'hidden_matrix'
ROLE_HIDDEN_VECTOR = 'hidden_vector'
ROLE_HIDDEN_BIAS = 'hidden_bias'
ROLE_NORM_SCALE = 'norm_scale'
ROLE_NORM_BIAS = 'norm_bias'
ROLE_QK_NORM_SCALE = 'qk_norm_scale'
ROLE_HIDDEN_VECTOR_OTHER = 'hidden_vector_other'
ROLE_VECTOR_LIKE = 'vector_like'
ROLE_MUON_MANAGED_MATRIX = 'muon_managed_matrix'

_EMBEDDING_CLASS_ROLES = frozenset((ROLE_EMBEDDING, ROLE_OUTPUT, ROLE_SHARED_EMBEDDING_OUTPUT))
_OUTPUT_ROLES = frozenset((ROLE_OUTPUT, ROLE_SHARED_EMBEDDING_OUTPUT))
_HIDDEN_VECTOR_ROLES = frozenset(
    (
        ROLE_HIDDEN_VECTOR,
        ROLE_HIDDEN_BIAS,
        ROLE_NORM_SCALE,
        ROLE_NORM_BIAS,
        ROLE_QK_NORM_SCALE,
        ROLE_HIDDEN_VECTOR_OTHER,
    )
)
_NORM_ROLES = frozenset((ROLE_NORM_SCALE, ROLE_NORM_BIAS, ROLE_QK_NORM_SCALE))


def set_parameterization_metadata(
    param: Any, *, role: str, shared_group: Optional[str] = None, tags: Iterable[str] = ()
) -> None:
    setattr(param, PARAMETERIZATION_ROLE_ATTR, role)
    if shared_group is not None:
        setattr(param, PARAMETERIZATION_SHARED_GROUP_ATTR, shared_group)
    if tags:
        setattr(param, PARAMETERIZATION_TAGS_ATTR, tuple(tags))


def get_parameterization_role(param: Any) -> Optional[str]:
    return getattr(param, PARAMETERIZATION_ROLE_ATTR, None)


def is_output_parameter(param: Any) -> bool:
    if hasattr(param, IS_OUTPUT_PARAMETER_ATTR):
        return bool(getattr(param, IS_OUTPUT_PARAMETER_ATTR))
    return get_parameterization_role(param) in _OUTPUT_ROLES


def is_embedding_or_output_parameter(param: Any) -> bool:
    if hasattr(param, 'is_embedding_or_output_parameter'):
        return bool(param.is_embedding_or_output_parameter)
    return get_parameterization_role(param) in _EMBEDDING_CLASS_ROLES


def is_embedding_class_parameter(param: Any, param_name: Optional[str] = None) -> bool:
    if getattr(param, 'shared_embedding', False):
        return True
    if hasattr(param, 'is_embedding_parameter'):
        return bool(param.is_embedding_parameter)
    if get_parameterization_role(param) in _EMBEDDING_CLASS_ROLES:
        return True
    # Compatibility-only fallback for older unannotated parameters. The scaling
    # recipes added in this branch are intended to rely on explicit metadata.
    return bool(param_name and 'embedding' in param_name.lower())


def is_vector_like_parameter(param: Any, param_name: Optional[str] = None) -> bool:
    if is_embedding_class_parameter(param, param_name):
        return True
    return param.dim() <= 1


def _lower_name(param_name: Optional[str]) -> str:
    return param_name.lower() if param_name else ''


def is_qk_norm_parameter(param: Any, param_name: Optional[str] = None) -> bool:
    role = get_parameterization_role(param)
    if role == ROLE_QK_NORM_SCALE:
        return True
    name = _lower_name(param_name)
    return param.dim() <= 1 and ('q_layernorm.' in name or 'k_layernorm.' in name)


def is_norm_parameter(param: Any, param_name: Optional[str] = None) -> bool:
    role = get_parameterization_role(param)
    if role in _NORM_ROLES:
        return True
    if param.dim() > 1:
        return False
    name = _lower_name(param_name)
    return 'layernorm' in name or 'layer_norm' in name or 'rmsnorm' in name or '.norm.' in name


def is_hidden_bias_parameter(param: Any, param_name: Optional[str] = None) -> bool:
    role = get_parameterization_role(param)
    if role == ROLE_HIDDEN_BIAS:
        return True
    if role in _EMBEDDING_CLASS_ROLES or role in _NORM_ROLES:
        return False
    name = _lower_name(param_name)
    return param.dim() <= 1 and name.endswith('.bias') and not is_norm_parameter(param, param_name)


def should_skip_depth_mup_vector_weight_decay(
    param: Any, param_name: Optional[str] = None, *, apply_wd_to_qk_layernorm: bool = False
) -> bool:
    """Return true for vector-like params outside the AdamW hidden-bias table row.

    The spectral width-depth table gives base weight decay to hidden biases, but
    Megatron's 1-D tensors also include normalization scale parameters. Keep
    hidden linear/MLP/attention biases on base weight decay and keep normalization
    or otherwise unknown 1-D tensors on the conservative no-WD path unless the
    user explicitly asks to apply WD to q/k layernorm.
    """
    if is_embedding_class_parameter(param, param_name):
        return False
    if is_hidden_bias_parameter(param, param_name):
        return False
    if apply_wd_to_qk_layernorm and is_qk_norm_parameter(param, param_name):
        return False
    if is_norm_parameter(param, param_name):
        return True
    return param.dim() <= 1


def is_hidden_vector_parameter(param: Any, param_name: Optional[str] = None) -> bool:
    role = get_parameterization_role(param)
    if role in _HIDDEN_VECTOR_ROLES:
        return True
    if role in _EMBEDDING_CLASS_ROLES:
        return False
    return param.dim() <= 1 and not is_embedding_class_parameter(param, param_name)


def is_hidden_matrix_parameter(param: Any, param_name: Optional[str] = None) -> bool:
    role = get_parameterization_role(param)
    if role == ROLE_HIDDEN_MATRIX:
        return True
    if role in _HIDDEN_VECTOR_ROLES or role in _EMBEDDING_CLASS_ROLES:
        return False
    return param.dim() > 1 and not is_embedding_class_parameter(param, param_name)


def is_muon_managed_matrix_parameter(param: Any, *, optimizer_type: str) -> bool:
    if 'muon' not in optimizer_type.lower():
        return False
    return param.dim() == 2 and not is_embedding_or_output_parameter(param)
