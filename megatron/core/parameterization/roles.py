# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from typing import Any, Iterable, Optional

PARAMETERIZATION_ROLE_ATTR = 'parameterization_role'
PARAMETERIZATION_SHARED_GROUP_ATTR = 'parameterization_shared_group'
PARAMETERIZATION_TAGS_ATTR = 'parameterization_tags'

ROLE_EMBEDDING = 'embedding'
ROLE_OUTPUT = 'output'
ROLE_SHARED_EMBEDDING_OUTPUT = 'shared_embedding_output'
ROLE_BLOCK_OUT_PROJ = 'block_out_proj'
ROLE_HIDDEN_MATRIX = 'hidden_matrix'
ROLE_VECTOR_LIKE = 'vector_like'
ROLE_MUON_MANAGED_MATRIX = 'muon_managed_matrix'


def set_parameterization_metadata(
    param: Any,
    *,
    role: str,
    shared_group: Optional[str] = None,
    tags: Iterable[str] = (),
) -> None:
    setattr(param, PARAMETERIZATION_ROLE_ATTR, role)
    if shared_group is not None:
        setattr(param, PARAMETERIZATION_SHARED_GROUP_ATTR, shared_group)
    if tags:
        setattr(param, PARAMETERIZATION_TAGS_ATTR, tuple(tags))
