# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from copy import deepcopy

from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
from megatron.core.models.hybrid.hybrid_layer_specs import (
    hybrid_stack_spec as default_hybrid_stack_spec,
)
from megatron.core.tensor_parallel import ColumnParallelLinear

LING_V3_TINY_HYBRID_LAYER_PATTERN = "K-KEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+E/+E"


# Ling-V3 Tiny uses tensor-parallel projections for the KDA beta path and the
# MLA latent down projections and output gate. Keep the default hybrid spec
# unchanged and expose this model recipe through the standard --spec hook.
hybrid_stack_spec = deepcopy(default_hybrid_stack_spec)

_kda_submodules = hybrid_stack_spec.submodules.kda_layer.submodules.self_attention.submodules
_kda_submodules.beta_proj = ColumnParallelLinear

_mla_submodules = hybrid_stack_spec.submodules.mla_layer.submodules.self_attention.submodules
_mla_submodules.linear_q_down_proj = TEColumnParallelLinear
_mla_submodules.linear_kv_down_proj = TEColumnParallelLinear
_mla_submodules.linear_gate = ColumnParallelLinear
