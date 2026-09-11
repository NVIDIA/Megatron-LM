# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Concrete GDN/GDN2 operation modules and their construction submodules.

Import this module explicitly when constructing a mixer. The family namespace
stays lightweight and does not eagerly load these implementations.
"""

from megatron.core.ops.ssm.gated_delta.common import (
    GatedDeltaNetSubmodules,
    get_parameter_local_cp,
    tensor_a2a_cp2hp,
    tensor_a2a_hp2cp,
)
from megatron.core.ops.ssm.gated_delta.gdn import GatedDeltaNet, torch_chunk_gated_delta_rule
from megatron.core.ops.ssm.gated_delta.gdn2 import GatedDeltaNet2, torch_chunk_gdn2

__all__ = [
    "GatedDeltaNet",
    "GatedDeltaNet2",
    "GatedDeltaNetSubmodules",
    "get_parameter_local_cp",
    "tensor_a2a_cp2hp",
    "tensor_a2a_hp2cp",
    "torch_chunk_gated_delta_rule",
    "torch_chunk_gdn2",
]
