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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/conversion/transformers_compat.py
# Source license: Apache-2.0.

"""Compatibility utilities for HuggingFace transformers 5.0+ configs."""

import transformers.utils.import_utils as _hf_import_utils

# Shim for is_torch_fx_available, removed in transformers 5.x but still
# referenced by some custom model repos (e.g. Kimi-K2's modeling_deepseek.py).
# torch.fx has been stable since PyTorch 1.10, so always return True.
if not hasattr(_hf_import_utils, "is_torch_fx_available"):
    _hf_import_utils.is_torch_fx_available = lambda: True


def rope_theta_from_hf(config) -> float:
    """Extract rope_theta from a HuggingFace config.

    This utility method handles the extraction of rope_theta (rotary position
    embedding base frequency) from HuggingFace configs, supporting both the
    legacy format (direct rope_theta attribute) and the new transformers 5.0+
    format (rope_parameters dictionary).

    Args:
        config: HuggingFace configuration object.

    Returns:
        float: The rope_theta value for rotary embeddings.

    Raises:
        ValueError: If rope_theta is not found in either format.
    """
    # Check for direct attribute (transformers <5.0)
    if hasattr(config, "rope_theta"):
        rope_theta = config.rope_theta
        if rope_theta is not None:
            return rope_theta

    # Check rope_parameters (transformers >=5.0)
    if hasattr(config, "rope_parameters") and config.rope_parameters:
        # Flat structure: rope_parameters["rope_theta"]
        if "rope_theta" in config.rope_parameters:
            rope_theta = config.rope_parameters["rope_theta"]
            if rope_theta is not None:
                return rope_theta
        # Nested structure for Gemma3 in transformers 5.0+: rope_parameters["global"]["base"]
        if "global" in config.rope_parameters:
            global_params = config.rope_parameters["global"]
            if isinstance(global_params, dict) and "base" in global_params:
                rope_theta = global_params["base"]
                if rope_theta is not None:
                    return rope_theta
        # Gemma3 transformers 5.0+ uses "full_attention" key with "rope_theta"
        if "full_attention" in config.rope_parameters:
            full_attn_params = config.rope_parameters["full_attention"]
            if isinstance(full_attn_params, dict) and "rope_theta" in full_attn_params:
                rope_theta = full_attn_params["rope_theta"]
                if rope_theta is not None:
                    return rope_theta

    # Fallback to default_theta (transformers 5.0+)
    if hasattr(config, "default_theta") and config.default_theta:
        # default_theta can be a plain float (e.g. NemotronH) or a dict (e.g. Gemma3)
        if isinstance(config.default_theta, (int, float)):
            return float(config.default_theta)
        if isinstance(config.default_theta, dict) and "global" in config.default_theta:
            rope_theta = config.default_theta["global"]
            if rope_theta is not None:
                return rope_theta

    raise ValueError(
        "rope_theta not found in config. Expected either 'rope_theta' attribute "
        "(transformers <5.0), 'rope_parameters[\"rope_theta\"]', "
        '\'rope_parameters["global"]["base"]\', \'rope_parameters["full_attention"]["rope_theta"]\', '
        "or 'default_theta[\"global\"]' (transformers >=5.0)."
    )


def partial_rotary_factor_from_hf(config) -> float:
    """Extract partial_rotary_factor from a HuggingFace config.

    Transformers 5.19 and earlier expose it as a ``partial_rotary_factor``
    attribute (and also in ``rope_parameters``). Transformers 5.20+ keeps it
    only in the flat ``rope_parameters`` dictionary.

    Args:
        config: HuggingFace configuration object.

    Returns:
        float: The fraction of each attention head dimension that gets rotary embeddings.

    Raises:
        ValueError: If partial_rotary_factor is not found in either format.
    """
    partial_rotary_factor = getattr(config, "partial_rotary_factor", None)
    if partial_rotary_factor is not None:
        return partial_rotary_factor

    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict):
        partial_rotary_factor = rope_parameters.get("partial_rotary_factor")
        if partial_rotary_factor is not None:
            return partial_rotary_factor

    raise ValueError(
        "partial_rotary_factor not found in config. Expected either a 'partial_rotary_factor' attribute "
        "(transformers <5.20) or 'rope_parameters[\"partial_rotary_factor\"]' (transformers >=5.20)."
    )


def rope_local_base_freq_from_hf(config) -> float:
    """Extract rope_local_base_freq from a HuggingFace config.

    Similar to rope_theta_from_hf but for the local base frequency parameter
    used by some models (e.g., Gemma3).

    Args:
        config: HuggingFace configuration object.

    Returns:
        float: The rope_local_base_freq value.

    Raises:
        ValueError: If rope_local_base_freq is not found in either format.
    """
    # Check for direct attribute (transformers <5.0)
    if hasattr(config, "rope_local_base_freq"):
        rope_local_base_freq = config.rope_local_base_freq
        if rope_local_base_freq is not None:
            return rope_local_base_freq

    # Check rope_parameters (transformers >=5.0)
    if hasattr(config, "rope_parameters") and config.rope_parameters:
        # Flat structure: rope_parameters["rope_local_base_freq"]
        if "rope_local_base_freq" in config.rope_parameters:
            rope_local_base_freq = config.rope_parameters["rope_local_base_freq"]
            if rope_local_base_freq is not None:
                return rope_local_base_freq
        # Nested structure for Gemma3 in transformers 5.0+: rope_parameters["local"]["base"]
        if "local" in config.rope_parameters:
            local_params = config.rope_parameters["local"]
            if isinstance(local_params, dict) and "base" in local_params:
                rope_local_base_freq = local_params["base"]
                if rope_local_base_freq is not None:
                    return rope_local_base_freq
        # Gemma3 transformers 5.0+ uses "sliding_attention" key with "rope_theta"
        if "sliding_attention" in config.rope_parameters:
            sliding_attn_params = config.rope_parameters["sliding_attention"]
            if isinstance(sliding_attn_params, dict) and "rope_theta" in sliding_attn_params:
                rope_local_base_freq = sliding_attn_params["rope_theta"]
                if rope_local_base_freq is not None:
                    return rope_local_base_freq

    # Check rope_scaling as a fallback
    if hasattr(config, "rope_scaling") and config.rope_scaling:
        if "rope_local_base_freq" in config.rope_scaling:
            rope_local_base_freq = config.rope_scaling["rope_local_base_freq"]
            if rope_local_base_freq is not None:
                return rope_local_base_freq

    # Fallback to default_theta (transformers 5.0+)
    if hasattr(config, "default_theta") and config.default_theta:
        if isinstance(config.default_theta, dict) and "local" in config.default_theta:
            rope_local_base_freq = config.default_theta["local"]
            if rope_local_base_freq is not None:
                return rope_local_base_freq

    raise ValueError(
        "rope_local_base_freq not found in config. Expected either 'rope_local_base_freq' attribute "
        "(transformers <5.0), 'rope_parameters[\"rope_local_base_freq\"]', "
        '\'rope_parameters["local"]["base"]\', \'rope_parameters["sliding_attention"]["rope_theta"]\', '
        "'rope_scaling[\"rope_local_base_freq\"]', or 'default_theta[\"local\"]' (transformers >=5.0)."
    )


def rope_scaling_factor_from_hf(config, default: float = 1.0) -> float:
    """Extract rope scaling factor from a HuggingFace config.

    This utility method handles the extraction of the rope scaling factor from
    HuggingFace configs, supporting both the legacy format (rope_scaling dict)
    and the new transformers 5.0+ format (rope_parameters dictionary).

    Args:
        config: HuggingFace configuration object.
        default: Default value to return if no scaling factor is found.

    Returns:
        float: The rope scaling factor value, or default if not found.
    """
    # Check rope_scaling (transformers <5.0 and some 5.0+ models)
    if hasattr(config, "rope_scaling") and config.rope_scaling:
        if isinstance(config.rope_scaling, dict) and "factor" in config.rope_scaling:
            factor = config.rope_scaling["factor"]
            if factor is not None:
                return factor

    # Check rope_parameters (transformers >=5.0)
    if hasattr(config, "rope_parameters") and config.rope_parameters:
        # Check for nested structure with layer types (Gemma3 style)
        for layer_type in ["full_attention", "global"]:
            if layer_type in config.rope_parameters:
                layer_params = config.rope_parameters[layer_type]
                if isinstance(layer_params, dict) and "factor" in layer_params:
                    factor = layer_params["factor"]
                    if factor is not None:
                        return factor
        # Check flat structure
        if "factor" in config.rope_parameters:
            factor = config.rope_parameters["factor"]
            if factor is not None:
                return factor

    # Return default if no scaling factor found
    return default


def full_attention_interval_from_hf(config, default: int = 4) -> int | list[int]:
    """Extract the full-attention layout from a Qwen3-Next-style HuggingFace config.

    In transformers <5.5 the interval was stored directly as
    ``config.full_attention_interval``. In transformers >=5.5 the field was
    removed; the kwarg is consumed in ``__post_init__`` and converted into
    ``config.layer_types`` (a list whose ``i``-th entry is ``"linear_attention"``
    or ``"full_attention"`` according to ``(i + 1) % interval``). This helper
    handles both layouts.

    Args:
        config: HuggingFace configuration object (e.g. ``Qwen3NextConfig``).
        default: Value to return if neither layout is present.

    Returns:
        int | list[int]: The legacy interval or the exact per-layer pattern,
        where 1 is linear attention and 0 is full attention.

    Raises:
        ValueError: If an explicit layer type is unsupported.
    """
    layer_types = getattr(config, "layer_types", None)
    if isinstance(layer_types, list) and layer_types:
        supported_layer_types = {"linear_attention": 1, "full_attention": 0}
        try:
            return [supported_layer_types[layer_type] for layer_type in layer_types]
        except KeyError as error:
            raise ValueError(f"Unsupported attention layer type: {error.args[0]}") from error

    interval = getattr(config, "full_attention_interval", None)
    if interval is not None:
        return interval

    return default
