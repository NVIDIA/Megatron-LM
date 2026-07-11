# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GPT/Hybrid state-dict transformations.

Converts state dictionaries in either direction, maps layer indices and keys,
and initializes fresh SSM ('M') tensors when converting GPT to Hybrid.

SSM tensors are initialized once on rank 0 and persisted to a temporary dist
checkpoint that every rank then reads back with DCP-driven TP/PP/FSDP
resharding, so the freshly initialized tensors are single-sourced and
consistent by construction — no per-rank RNG coordination needed.
"""

import logging
import math
import re
from collections import OrderedDict

import torch

from megatron.core.models.hybrid.conversion.compatibility import build_layer_index_mapping

logger = logging.getLogger(__name__)


def get_layer_num_from_key(key):
    """Extract the layer number from a state-dict key."""
    match = re.search(r'decoder\.layers\.(\d+)\.', key)
    return int(match.group(1)) if match else None


def replace_layer_num(key, old_num, new_num):
    """Replace the layer number in a state-dict key."""
    return key.replace(f'decoder.layers.{old_num}.', f'decoder.layers.{new_num}.', 1)


def is_attention_param(key):
    """Return whether a key belongs to an attention sub-layer."""
    return 'self_attention.' in key or 'input_layernorm.' in key


def is_mlp_param(key):
    """Return whether a key belongs to an MLP sub-layer."""
    return ('mlp.' in key or 'pre_mlp_layernorm.' in key) and 'self_attention' not in key


def is_ssm_param(key):
    """Return whether a key belongs to a Mamba SSM mixer sub-layer."""
    ssm_markers = (
        'mixer.in_proj',
        'mixer.conv1d',
        'mixer.A_log',
        'mixer.D',
        'mixer.dt_bias',
        'mixer.norm',
        'mixer.out_proj',
        'mixer.x_proj',
        'mixer.dt_proj',
    )
    return any(marker in key for marker in ssm_markers)


def is_layer_norm_for_ssm(key):
    """Return whether a key is the input layer norm for an SSM layer."""
    return 'in_proj.layer_norm_weight' in key


def initialize_ssm_layer_params(
    layer_idx,
    d_model,
    mamba_d_inner,
    mamba_d_state,
    mamba2_n_groups,
    mamba2_n_heads,
    mamba_head_dim,
    d_conv=4,
    dt_min=0.001,
    dt_max=0.1,
    dt_init_floor=1e-4,
    A_init_range=(1, 16),
    init_method_std=0.02,
    dtype=torch.float32,
):
    """Initialize parameters for a single Mamba SSM layer from scratch.

    Follows the initialization logic from MambaMixer.__init__:
        - A_log:          log(uniform(A_init_range))
        - dt_bias:        inverse_softplus(log_uniform(dt_min, dt_max))
        - D:              ones(nheads)
        - conv1d.weight:  kaiming_uniform(a=sqrt(5))
        - conv1d.bias:    zeros
        - in_proj.weight: kaiming_uniform(a=sqrt(5))
        - in_proj.layer_norm_weight: ones(d_model)
        - out_proj.weight: kaiming_uniform(a=sqrt(5)) or normal(0, std)
        - norm.weight:    ones(d_inner)

    Returns:
        dict: {param_suffix: tensor} for one SSM layer
    """
    prefix = f"decoder.layers.{layer_idx}.mixer."

    nheads = mamba2_n_heads
    conv_dim = mamba_d_inner + 2 * mamba2_n_groups * mamba_d_state
    in_proj_out_dim = 2 * mamba_d_inner + 2 * mamba2_n_groups * mamba_d_state + nheads

    params = OrderedDict()

    # in_proj (ColumnParallelLinear)
    in_proj_weight = torch.empty(in_proj_out_dim, d_model, dtype=dtype)
    torch.nn.init.kaiming_uniform_(in_proj_weight, a=math.sqrt(5))
    params[prefix + "in_proj.weight"] = in_proj_weight

    # in_proj layer norm weight (fused into ColumnParallelLinear in TE)
    params[prefix + "in_proj.layer_norm_weight"] = torch.ones(d_model, dtype=dtype)

    # conv1d
    conv_weight = torch.empty(conv_dim, 1, d_conv, dtype=dtype)
    torch.nn.init.kaiming_uniform_(conv_weight, a=math.sqrt(5))
    params[prefix + "conv1d.weight"] = conv_weight
    params[prefix + "conv1d.bias"] = torch.zeros(conv_dim, dtype=dtype)

    # A_log (kept in fp32)
    A = torch.empty(nheads, dtype=torch.float32)
    A.uniform_(*A_init_range)
    params[prefix + "A_log"] = torch.log(A)

    # D (kept in fp32)
    params[prefix + "D"] = torch.ones(nheads, dtype=torch.float32)

    # dt_bias
    dt = torch.exp(
        torch.rand(nheads, dtype=dtype) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
    ).clamp(min=dt_init_floor)
    inv_dt = dt + torch.log(-torch.expm1(-dt))
    params[prefix + "dt_bias"] = inv_dt

    # norm (RMSNorm)
    params[prefix + "norm.weight"] = torch.ones(mamba_d_inner, dtype=dtype)

    # out_proj (RowParallelLinear)
    out_proj_weight = torch.empty(d_model, mamba_d_inner, dtype=dtype)
    torch.nn.init.kaiming_uniform_(out_proj_weight, a=math.sqrt(5))
    params[prefix + "out_proj.weight"] = out_proj_weight

    return params


def convert_gpt_to_hybrid(full_model, layer_types, args):
    """Convert a GPT state dict to a Hybrid state dict.

    Args:
        full_model: OrderedDict with globally-indexed GPT state dict keys.
        layer_types: list of layer type chars from hybrid_layer_pattern.
        args: Parsed CLI arguments.

    Returns:
        OrderedDict: Hybrid state dict with globally-indexed keys.
    """
    attn_map, mlp_map, ssm_indices = build_layer_index_mapping(layer_types, "gpt-to-hybrid")
    num_gpt_layers = len(attn_map)

    # Validate GPT layer count
    gpt_layer_nums = set()
    for key in full_model:
        lnum = get_layer_num_from_key(key)
        if lnum is not None:
            gpt_layer_nums.add(lnum)

    if len(gpt_layer_nums) != num_gpt_layers:
        raise ValueError(
            f"GPT checkpoint has {len(gpt_layer_nums)} layers, but the pattern "
            f"has {num_gpt_layers} attention ('*') and {num_gpt_layers} MLP ('-') "
            f"layers. These must match."
        )

    target = OrderedDict()
    dtype = None

    # Copy / rename non-layer params
    for key, tensor in full_model.items():
        if dtype is None and tensor.dtype in (torch.float16, torch.bfloat16, torch.float32):
            dtype = tensor.dtype

        if "decoder.layers." in key:
            continue

        # Rename final_layernorm -> final_norm
        if "decoder.final_layernorm" in key:
            new_key = key.replace("decoder.final_layernorm", "decoder.final_norm")
            target[new_key] = tensor
        else:
            target[key] = tensor

    if dtype is None:
        dtype = torch.float32

    # Map attention and MLP params
    for key, tensor in full_model.items():
        lnum = get_layer_num_from_key(key)
        if lnum is None:
            continue

        if is_attention_param(key):
            target_layer = attn_map[lnum]
            new_key = replace_layer_num(key, lnum, target_layer)
            target[new_key] = tensor

        elif is_mlp_param(key):
            target_layer = mlp_map[lnum]
            new_key = replace_layer_num(key, lnum, target_layer)
            target[new_key] = tensor

        # (any other layer params get copied as-is with their own mapping,
        #  but for pure GPT there should only be attention + MLP)

    # Initialize SSM layers from scratch
    logger.info("Initializing %d SSM layers from scratch...", len(ssm_indices))
    for layer_idx in ssm_indices:
        ssm_params = initialize_ssm_layer_params(
            layer_idx=layer_idx,
            d_model=args.d_model,
            mamba_d_inner=args.mamba_d_inner,
            mamba_d_state=args.mamba_d_state,
            mamba2_n_groups=args.mamba2_n_groups,
            mamba2_n_heads=args.mamba2_n_heads,
            mamba_head_dim=args.mamba2_head_dim,
            d_conv=getattr(args, "d_conv", 4),
            init_method_std=getattr(args, "init_method_std", 0.02),
            dtype=dtype,
        )
        target.update(ssm_params)

    # Sort by layer index for consistent ordering
    target = _sort_state_dict(target)

    return target


def _sort_state_dict(state_dict):
    """Sort state dict keys so that layer-indexed keys are in order."""

    def sort_key(item):
        key = item[0]
        # Extract layer number if present
        match = re.search(r"decoder\.layers\.(\d+)\.", key)
        if match:
            return (1, int(match.group(1)), key)
        # Non-layer keys: embeddings first, output_layer last
        if "embedding" in key:
            return (0, 0, key)
        if "output_layer" in key:
            return (2, 0, key)
        if "decoder.final" in key:
            return (1, 999999, key)
        return (0, 1, key)

    return OrderedDict(sorted(state_dict.items(), key=sort_key))


def convert_hybrid_to_gpt(full_model, layer_types, args):
    """Convert a Hybrid state dict to a GPT state dict.

    Args:
        full_model: State dict with globally-indexed Hybrid keys.
        layer_types: Layer symbols parsed from ``hybrid_layer_pattern``.
        args: Parsed CLI arguments. Reserved for conversion options.

    Returns:
        OrderedDict: GPT state dict with globally-indexed keys.
    """
    del args
    attn_map, mlp_map, ssm_indices = build_layer_index_mapping(layer_types, 'hybrid-to-gpt')

    target = OrderedDict()
    discarded_ssm_keys = []

    for key, tensor in full_model.items():
        if 'decoder.layers.' in key:
            continue
        if 'decoder.final_norm' in key:
            target[key.replace('decoder.final_norm', 'decoder.final_layernorm')] = tensor
        else:
            target[key] = tensor

    for key, tensor in full_model.items():
        layer_num = get_layer_num_from_key(key)
        if layer_num is None:
            continue

        if (is_ssm_param(key) or is_layer_norm_for_ssm(key)) and layer_num in ssm_indices:
            discarded_ssm_keys.append(key)
            continue

        if is_attention_param(key) and layer_num in attn_map:
            target_layer = attn_map[layer_num]
            target[replace_layer_num(key, layer_num, target_layer)] = tensor
        elif is_mlp_param(key) and layer_num in mlp_map:
            target_layer = mlp_map[layer_num]
            target[replace_layer_num(key, layer_num, target_layer)] = tensor
        elif layer_num in ssm_indices:
            discarded_ssm_keys.append(key)

    if discarded_ssm_keys:
        logger.warning(
            "Discarded %d SSM parameter tensors from %d SSM layers (no GPT equivalent). "
            "First few discarded keys: %s",
            len(discarded_ssm_keys),
            len(ssm_indices),
            discarded_ssm_keys[:5],
        )

    return _sort_state_dict(target)
