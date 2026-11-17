# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/utils/activation_map.py
# Source license: Apache-2.0.

"""Shared bidirectional mapping between activation function names and callables.

Used by:
- ``omegaconf_utils`` for CLI override serialization/deserialization
- ``ModelBridge`` for HuggingFace <-> Megatron activation conversion
"""

import logging
from typing import Callable

import torch
import torch.nn.functional as F

from megatron.core.activations import fast_gelu, squared_relu

try:
    from megatron.core.fusions.fused_bias_geglu import quick_gelu
except ImportError:
    # Fallback if fused_bias_geglu is not available in the pinned MCore version
    quick_gelu = F.gelu


logger = logging.getLogger(__name__)

# Canonical name -> callable.
# Short names come first; fully-qualified aliases are added below.
ACTIVATION_FUNC_MAP: dict[str, Callable] = {
    "gelu": F.gelu,
    "relu": F.relu,
    "silu": F.silu,
    "sigmoid": F.sigmoid,
    "tanh": torch.tanh,
    "relu2": squared_relu,  # alias; canonical is squared_relu (below)
    "squared_relu": squared_relu,
    "gelu_pytorch_tanh": fast_gelu,  # alias; canonical is fast_gelu (below)
    "fast_gelu": fast_gelu,
    "quick_gelu": quick_gelu,
}

# Add fully-qualified torch.nn.functional aliases
ACTIVATION_FUNC_MAP.update(
    {
        "torch.nn.functional.gelu": F.gelu,
        "torch.nn.functional.relu": F.relu,
        "torch.nn.functional.silu": F.silu,
        "torch.nn.functional.sigmoid": F.sigmoid,
        "torch.tanh": torch.tanh,
    }
)

# Reverse map: callable id -> canonical short name (used during serialization).
# Only short names (no dots) are used as canonical representations.
_ACTIVATION_FUNC_TO_STR: dict[int, str] = {
    id(fn): name for name, fn in ACTIVATION_FUNC_MAP.items() if "." not in name
}


def callable_to_str(fn: Callable) -> str | None:
    """Convert a known activation callable to its short string name.

    Returns ``None`` if the callable is not in the registry.
    """
    return _ACTIVATION_FUNC_TO_STR.get(id(fn))


def str_to_callable(name: str) -> Callable:
    """Resolve an activation function name to its callable.

    Accepts registered short names (``"silu"``) and registered fully-qualified
    aliases (``"torch.nn.functional.silu"``).

    Raises:
        ValueError: If the name cannot be resolved.
    """
    if name in ACTIVATION_FUNC_MAP:
        return ACTIVATION_FUNC_MAP[name]

    known_names = sorted(n for n in ACTIVATION_FUNC_MAP if "." not in n)
    raise ValueError(f"Unknown activation function: '{name}'. Known names: {known_names}")


# Short name -> torch.dtype
DTYPE_MAP: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "32": torch.float32,
    "32-true": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "16": torch.float16,
    "16-mixed": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "bf16-mixed": torch.bfloat16,
    "fp8": torch.float8_e4m3fn,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}

# Add "torch.<name>" aliases for canonical (non-numeric, non-mixed) names
DTYPE_MAP.update(
    {
        f"torch.{k}": v
        for k, v in DTYPE_MAP.items()
        if "." not in k and not k[0].isdigit() and "-" not in k
    }
)


def str_to_dtype(name: str) -> torch.dtype:
    """Resolve a dtype string to a ``torch.dtype``.

    Accepts short names (``"bf16"``), canonical names (``"bfloat16"``), or
    fully-qualified names (``"torch.bfloat16"``).

    Raises:
        ValueError: If the name cannot be resolved.
    """
    if name in DTYPE_MAP:
        return DTYPE_MAP[name]
    known = sorted(n for n in DTYPE_MAP if "." not in n)
    raise ValueError(f"Unknown dtype: '{name}'. Known names: {known}")
