# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Songlin Yang, Jan Kautz, Ali Hatamizadeh.

# Some of this code was adopted from https://github.com/huggingface/transformers
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

"""Backward-compatibility shim.

The Gated Delta Net implementation moved to the ``megatron.core.ssm.gdn`` package.
Import from ``megatron.core.ssm.gdn`` instead;
this module re-exports the public names for existing users.
"""

# pylint: disable=unused-import

from megatron.core.ssm.gdn.common import (
    HAVE_FLA,
    GatedDeltaNetSubmodules,
    _build_head_perm_for_split_sections,
    _build_thd_cp_a2a_perm,
    get_parameter_local_cp,
    tensor_a2a_cp2hp,
    tensor_a2a_hp2cp,
    torch_chunk_gated_delta_rule,
)
from megatron.core.ssm.gdn.gated_delta_net import GatedDeltaNet
from megatron.core.ssm.utils import _split_tensor_factory

__all__ = [
    "GatedDeltaNet",
    "GatedDeltaNetSubmodules",
    "get_parameter_local_cp",
    "tensor_a2a_cp2hp",
    "tensor_a2a_hp2cp",
    "torch_chunk_gated_delta_rule",
]
