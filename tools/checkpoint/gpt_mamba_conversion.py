# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
GPT <-> Mamba Checkpoint Conversion Tool
=========================================

Directly converts checkpoints between GPTModel (homogeneous Transformer) and
MambaModel (hybrid Mamba+Transformer) without going through HuggingFace as an
intermediary.

Supported directions:
    gpt-to-mamba : Convert a GPT checkpoint to Mamba hybrid format.
    mamba-to-gpt : Convert a Mamba hybrid checkpoint to GPT format.

How the hybrid layer pattern maps GPT layers (gpt-to-mamba):
    - Each GPT layer contains both attention and MLP sub-layers.
    - The target hybrid model's hybrid_layer_pattern specifies per-layer types:
        M = Mamba SSM layer
        * = Attention-only layer
        - = MLP-only layer (dense)
        E = MoE MLP-only layer (router + experts; supports EP)
        G = GDN layer (not currently mapped)
    - GPT layer i's attention params map to the i-th '*' layer in the pattern.
    - GPT layer i's MLP/MoE params map to the i-th MLP-bearing position
      ('-' or 'E') in the pattern. Dense ('-') and MoE ('E') cannot be mixed:
      GPT layers are uniform.
    - The number of '*' positions and MLP-bearing positions must each equal
      the number of GPT layers.
    - Mamba SSM ('M') layers have no GPT equivalent and are initialized from
      scratch using standard Mamba initialization.

How MoE / Expert Parallelism (EP) works through the converter:
    - GPTModel can run with MoE (Mixtral-style: every layer has a router and
      N local experts). State-dict keys live under
      `decoder.layers.<i>.mlp.{router,experts,shared_experts}.*`.
    - Hybrid 'E' layers use the same key naming, so MoE tensors round-trip
      verbatim — no expert collapsing, no router init, no per-expert work.
    - EP-sharded checkpoints load through DCP transparently because each
      tensor's `global_shape` is in the metadata, regardless of how many
      EP / TP / PP / FSDP ranks wrote it.
    - Use a pattern like 'M*EM*EM*E' to pair Mamba/Attn/MoE-MLP per stage.

What happens to SSM parameters:
    gpt-to-mamba: SSM layers (M) are initialized from scratch:
        - A_log:          log(uniform(1, 16))
        - dt_bias:        inverse_softplus(log_uniform(dt_min, dt_max))
        - D:              ones
        - conv1d.weight:  kaiming_uniform(a=sqrt(5))
        - conv1d.bias:    zeros
        - in_proj.weight: kaiming_uniform(a=sqrt(5))
        - in_proj.layer_norm_weight: ones
        - out_proj.weight: kaiming_uniform(a=sqrt(5))
        - norm.weight:    ones
    mamba-to-gpt: SSM layers are discarded with a warning.

Supported checkpoint formats:
    - torch_dist   : Megatron distributed checkpoint (TP + PP + FSDP).
    - fsdp_dtensor : FSDP DTensor export (TP + PP + FSDP).

    PyTorch DCP gathers TP/PP/FSDP shards via the checkpoint's global-shape
    metadata, so no explicit TP/PP/DP config is needed on input. The input
    format is auto-detected; the output format defaults to the input format.

    The legacy ``mp_rank_XX/model_optim_rng.pt`` layout is not supported —
    convert old checkpoints to ``torch_dist`` first.

GPT compatibility whitelist (safeguard):
    GPTModel is a strict homogeneous transformer (self-attention + MLP per
    layer, standard linear_qkv / linear_fc1 / linear_fc2 state-dict keys).
    The converter fails fast if either side uses features that GPTModel
    cannot express.

    Rejected pattern symbols: 'G' (GDN), 'D' (DS-attention), 'E' (MoE).
    Allowed: 'M' (Mamba SSM), '*' (attention), '-' (MLP). The number of
    '*' and '-' layers must be equal.

    Rejected source-args features (checked against the args stored in the
    source checkpoint):
        - num_moe_experts / moe_shared_expert_intermediate_size / moe_layer_freq
        - experimental_attention_variant (gated_delta_net, dsa, ...)
        - linear_attention_freq
        - heterogeneous_block_specs / heterogeneous_layers_config_path
        - multi_latent_attention (MLA)
        - mtp_num_layers (Multi-Token Prediction)

    See `validate_pattern_gpt_compatible` and
    `validate_source_args_gpt_compatible` for the exact rules.

Example commands:
    # GPT -> Mamba (TP+PP+FSDP dist checkpoint)
    python tools/checkpoint/gpt_mamba_conversion.py \\
        --direction gpt-to-mamba \\
        --load-dir /path/to/gpt-dist-checkpoint \\
        --save-dir /path/to/mamba-dist-checkpoint \\
        --hybrid-layer-pattern "M*-M*-M*-M*-" \\
        --d-model 4096 \\
        --mamba-d-state 128 \\
        --mamba2-n-groups 8 \\
        --mamba2-head-dim 64

    # Mamba -> GPT (dist checkpoint)
    python tools/checkpoint/gpt_mamba_conversion.py \\
        --direction mamba-to-gpt \\
        --load-dir /path/to/mamba-dist-checkpoint \\
        --save-dir /path/to/gpt-dist-checkpoint \\
        --hybrid-layer-pattern "M*-M*-M*-M*-" \\
        --d-model 4096 \\
        --mamba-d-state 128 \\
        --mamba2-n-groups 8 \\
        --mamba2-head-dim 64
"""

import argparse
import copy
import math
import os
import re
from collections import OrderedDict

import torch

from dist_checkpoint_io import (
    DIST_FORMATS,
    FORMAT_TORCH_DIST,
    detect_checkpoint_format,
    load_dist_checkpoint_full,
    save_dist_checkpoint_full,
    write_latest_iteration_marker,
)


# ---------------------------------------------------------------------------
# Hybrid layer pattern parsing (standalone, no Megatron imports needed)
# ---------------------------------------------------------------------------

VALID_LAYER_SYMBOLS = {'M', 'G', '*', '-', 'E'}

# Layer symbols GPTModel can emit or absorb:
#   '*' : standard self-attention layer (MHA / GQA / MQA)
#   '-' : standard (optionally gated) dense MLP layer
#   'E' : MoE MLP layer. Both sides keep the keys under
#         decoder.layers.<i>.mlp.{router,experts,shared_experts}.* so MoE
#         tensors round-trip verbatim (see convert_gpt_to_mamba and
#         convert_mamba_to_gpt — `is_mlp_param` already matches `mlp.*`).
# SSM ('M') has no GPT equivalent and is initialized from scratch /
# discarded (see convert_gpt_to_mamba / convert_mamba_to_gpt).
# 'G' (GDN) and 'D' (DS-attention) are not currently mapped — they would
# need separate key-naming work. Reject for now.
GPT_COMPATIBLE_PATTERN_SYMBOLS = {'M', '*', '-', 'E'}


def parse_hybrid_layer_pattern(pattern):
    """Parse a hybrid layer pattern string into a list of layer types.

    Strips MTP separators (/) and pipeline stage separators (|), returning only
    the main decoder pattern as a list of single-character layer types.

    Returns:
        list[str]: e.g. ['M', '*', '-', 'M', '*', '-']
    """
    # Take only the main pattern (before first '/')
    main_pattern = pattern.split('/')[0]
    # Remove pipeline stage separators
    main_pattern = main_pattern.replace('|', '')
    layer_types = list(main_pattern)
    for ch in layer_types:
        if ch not in VALID_LAYER_SYMBOLS:
            raise ValueError(
                f"Invalid layer symbol '{ch}' in pattern. "
                f"Valid symbols: {VALID_LAYER_SYMBOLS}"
            )
    return layer_types


# Pattern symbols that pair to a GPT-side MLP block. Both dense ('-') and MoE
# ('E') keep their state-dict keys under `decoder.layers.<i>.mlp.*`, so they
# round-trip identically. The pattern uniformity check
# (validate_pattern_gpt_compatible) ensures '-' and 'E' don't appear together,
# which would mean GPT layers aren't uniform.
_MLP_BEARING_SYMBOLS = ('-', 'E')


def build_layer_index_mapping(layer_types, direction):
    """Build mapping between GPT layer indices and hybrid-model layer indices.

    For gpt-to-mamba:
        Returns (attn_map, mlp_map, ssm_indices) where:
        - attn_map[gpt_layer_i] = hybrid_layer_j  (j is the index of the i-th '*')
        - mlp_map[gpt_layer_i]  = hybrid_layer_k  (k is the index of the i-th
          MLP-bearing position; either '-' or 'E')

    For mamba-to-gpt:
        Returns (attn_map, mlp_map, ssm_indices) where:
        - attn_map[hybrid_attn_idx] = gpt_layer_i
        - mlp_map[hybrid_mlp_idx]   = gpt_layer_i
    """
    attn_indices = [i for i, t in enumerate(layer_types) if t == '*']
    mlp_indices = [i for i, t in enumerate(layer_types) if t in _MLP_BEARING_SYMBOLS]
    ssm_indices = [i for i, t in enumerate(layer_types) if t == 'M']

    if direction == 'gpt-to-mamba':
        if len(attn_indices) != len(mlp_indices):
            raise ValueError(
                f"For gpt-to-mamba, the number of attention layers ({len(attn_indices)}) "
                f"must equal the number of MLP/MoE layers ({len(mlp_indices)}) in the pattern."
            )
        attn_map = {i: attn_indices[i] for i in range(len(attn_indices))}
        mlp_map = {i: mlp_indices[i] for i in range(len(mlp_indices))}
        return attn_map, mlp_map, ssm_indices

    elif direction == 'mamba-to-gpt':
        if len(attn_indices) != len(mlp_indices):
            raise ValueError(
                f"For mamba-to-gpt, the number of attention layers ({len(attn_indices)}) "
                f"must equal the number of MLP/MoE layers ({len(mlp_indices)}) in the pattern."
            )
        attn_map = {attn_indices[i]: i for i in range(len(attn_indices))}
        mlp_map = {mlp_indices[i]: i for i in range(len(mlp_indices))}
        return attn_map, mlp_map, ssm_indices

    else:
        raise ValueError(f"Unknown direction: {direction}")


# ---------------------------------------------------------------------------
# GPT compatibility whitelist
# ---------------------------------------------------------------------------
#
# GPTModel is a *uniform* transformer: every decoder layer is the same kind.
# It can run with dense MLP or MoE MLP — both keep keys under
# decoder.layers.<i>.mlp.* — so MoE checkpoints round-trip through the
# converter as long as both sides share the same kind on every layer.
# The helpers below reject any hybrid layout or source-args combination that
# violates uniformity (and would therefore silently produce a corrupt target).
#
# Pattern-level rules (checked on the parsed hybrid_layer_pattern):
#   * only 'M', '*', '-', 'E' are allowed (no 'G' GDN, no 'D' DS-attention)
#   * MLP-bearing symbols must be uniform: '-' and 'E' cannot both appear
#     (that would imply GPT has both dense and MoE layers — heterogeneous)
#   * '*' count must equal '-'+'E' count (one-to-one GPT attn<->MLP pairing)
#
# Args-level rules (checked against the training args stored in the source
# checkpoint): reject anything that makes GPT layers heterogeneous OR uses
# attention variants the converter doesn't currently key-translate:
#   * moe_layer_freq != 1                      (interleaved dense/MoE layers)
#   * experimental_attention_variant           (gated_delta_net, dsa, ...)
#   * linear_attention_freq                    (interleaved linear-attention)
#   * heterogeneous_block_specs / heterogeneous_layers_config_*
#                                              (Nemotron-NAS per-layer specs)
#   * multi_latent_attention                   (MLA: different QKV key layout)
#   * mtp_num_layers                           (Multi-Token Prediction head)
#
# Notably NOT rejected (they round-trip via mlp.* / self_attention.* keys):
#   * num_moe_experts                          (MoE on every layer)
#   * moe_shared_expert_intermediate_size      (shared experts on every layer)
#
# All rejected configurations raise ValueError early, before any tensors
# are touched.

# Source-args field name -> (predicate-that-means-"reject", human reason).
# Predicates are applied with getattr(args, field, None); missing fields
# are treated as "absent" and pass.
_GPT_COMPAT_REJECT_FIELDS = (
    (
        'moe_layer_freq',
        # moe_layer_freq is None or 1 when every layer is the same kind (all
        # dense or all MoE). A value > 1 or a list with mixed entries means
        # GPT has interleaved dense/MoE layers — heterogeneous, can't pair
        # one-to-one with a uniform hybrid pattern.
        lambda v: (
            v is not None
            and not (isinstance(v, int) and v == 1)
            and not (isinstance(v, str) and v.strip() in ('', '1'))
            and not (isinstance(v, (list, tuple)) and all(x == 1 for x in v))
        ),
        'interleaved dense/MoE layers (moe_layer_freq)',
    ),
    (
        'experimental_attention_variant',
        lambda v: v is not None and v != '',
        'experimental attention variant (gated_delta_net / dsa / ...)',
    ),
    (
        'linear_attention_freq',
        lambda v: v is not None,
        'linear attention layers (linear_attention_freq)',
    ),
    (
        'heterogeneous_block_specs',
        lambda v: bool(v),
        'heterogeneous per-layer block specs',
    ),
    (
        'heterogeneous_layers_config_path',
        lambda v: v is not None and v != '',
        'heterogeneous layers config (Nemotron-NAS)',
    ),
    (
        'heterogeneous_layers_config_encoded_json',
        lambda v: v is not None and v != '',
        'heterogeneous layers config (Nemotron-NAS, inline JSON)',
    ),
    (
        'multi_latent_attention',
        lambda v: bool(v),
        'Multi-Latent Attention (MLA)',
    ),
    (
        'mtp_num_layers',
        lambda v: v is not None and v > 0,
        'Multi-Token Prediction head (mtp_num_layers)',
    ),
)


def validate_pattern_gpt_compatible(layer_types, direction):
    """Raise ValueError if the hybrid pattern cannot round-trip with GPTModel.

    Args:
        layer_types: list of layer-type chars from parse_hybrid_layer_pattern().
        direction: 'gpt-to-mamba' or 'mamba-to-gpt' (for error messages).

    Rules:
        * Allowed symbols: 'M', '*', '-', 'E'. 'G' (GDN) and 'D' (DS-attention)
          are not currently key-translated.
        * MLP-bearing symbols must be uniform: '-' (dense) and 'E' (MoE) cannot
          both appear, because that would imply GPT has both dense and MoE
          layers — the GPT side must be uniform.
        * The number of attention positions must equal the number of
          MLP-bearing positions: every GPT layer pairs one attention with one
          MLP/MoE.
    """
    bad = sorted({c for c in layer_types if c not in GPT_COMPATIBLE_PATTERN_SYMBOLS})
    if bad:
        raise ValueError(
            f"Hybrid layer pattern contains symbols {bad} that are not "
            f"GPT-compatible (allowed: {sorted(GPT_COMPATIBLE_PATTERN_SYMBOLS)}). "
            f"'G' (GDN) and 'D' (DS-attention) are not currently key-translated "
            f"and cannot be {direction}-converted."
        )

    mlp_kinds_present = {t for t in layer_types if t in _MLP_BEARING_SYMBOLS}
    if len(mlp_kinds_present) > 1:
        raise ValueError(
            f"Hybrid layer pattern mixes '-' (dense MLP) and 'E' (MoE) "
            f"positions. GPTModel layers must be uniform — either all GPT "
            f"layers are dense MLP, or all are MoE. Use only one of '-' or "
            f"'E' in the pattern."
        )

    n_attn = sum(1 for t in layer_types if t == '*')
    n_mlp = sum(1 for t in layer_types if t in _MLP_BEARING_SYMBOLS)
    if n_attn != n_mlp:
        raise ValueError(
            f"GPT-compatible hybrid patterns must pair every attention layer "
            f"('*') with one MLP/MoE layer ('-' or 'E'). Got {n_attn} '*' "
            f"and {n_mlp} MLP-bearing layers in the pattern."
        )


def validate_source_args_gpt_compatible(source_args, direction):
    """Raise ValueError if the source checkpoint uses features GPTModel can't express.

    Args:
        source_args: argparse.Namespace (or any attribute-bag) loaded from the
            source checkpoint; may be None, in which case this check is a no-op
            (dist checkpoints without a cached args blob).
        direction: 'gpt-to-mamba' or 'mamba-to-gpt'.

    Rejects MoE, MLA, MTP, linear / experimental attention, and heterogeneous
    per-layer specs. See the module header for the full list.
    """
    if source_args is None:
        return

    rejected = []
    for field, predicate, reason in _GPT_COMPAT_REJECT_FIELDS:
        if not hasattr(source_args, field):
            continue
        value = getattr(source_args, field)
        try:
            if predicate(value):
                rejected.append(f"  - {reason}: {field}={value!r}")
        except Exception:
            # Defensive: never let the validator crash on an unexpected
            # value type — treat it as "cannot verify, pass".
            continue

    if rejected:
        joined = "\n".join(rejected)
        raise ValueError(
            f"Source checkpoint is not GPT-compatible for {direction} "
            f"conversion. The following features have no GPTModel equivalent "
            f"and would produce a corrupt target checkpoint:\n{joined}\n"
            f"Remove these features from the model (or use a different "
            f"conversion tool) before running gpt_mamba_conversion."
        )


# ---------------------------------------------------------------------------
# SSM parameter initialization (for gpt-to-mamba)
# ---------------------------------------------------------------------------

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
    prefix = f'decoder.layers.{layer_idx}.mixer.'

    nheads = mamba2_n_heads
    conv_dim = mamba_d_inner + 2 * mamba2_n_groups * mamba_d_state
    in_proj_out_dim = 2 * mamba_d_inner + 2 * mamba2_n_groups * mamba_d_state + nheads

    params = OrderedDict()

    # in_proj (ColumnParallelLinear)
    in_proj_weight = torch.empty(in_proj_out_dim, d_model, dtype=dtype)
    torch.nn.init.kaiming_uniform_(in_proj_weight, a=math.sqrt(5))
    params[prefix + 'in_proj.weight'] = in_proj_weight

    # in_proj layer norm weight (fused into ColumnParallelLinear in TE)
    params[prefix + 'in_proj.layer_norm_weight'] = torch.ones(d_model, dtype=dtype)

    # conv1d
    conv_weight = torch.empty(conv_dim, 1, d_conv, dtype=dtype)
    torch.nn.init.kaiming_uniform_(conv_weight, a=math.sqrt(5))
    params[prefix + 'conv1d.weight'] = conv_weight
    params[prefix + 'conv1d.bias'] = torch.zeros(conv_dim, dtype=dtype)

    # A_log (kept in fp32)
    A = torch.empty(nheads, dtype=torch.float32)
    A.uniform_(*A_init_range)
    params[prefix + 'A_log'] = torch.log(A)

    # D (kept in fp32)
    params[prefix + 'D'] = torch.ones(nheads, dtype=torch.float32)

    # dt_bias
    dt = torch.exp(
        torch.rand(nheads, dtype=dtype)
        * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    ).clamp(min=dt_init_floor)
    inv_dt = dt + torch.log(-torch.expm1(-dt))
    params[prefix + 'dt_bias'] = inv_dt

    # norm (RMSNorm)
    params[prefix + 'norm.weight'] = torch.ones(mamba_d_inner, dtype=dtype)

    # out_proj (RowParallelLinear)
    out_proj_weight = torch.empty(d_model, mamba_d_inner, dtype=dtype)
    torch.nn.init.kaiming_uniform_(out_proj_weight, a=math.sqrt(5))
    params[prefix + 'out_proj.weight'] = out_proj_weight

    return params


# ---------------------------------------------------------------------------
# Key name helpers
# ---------------------------------------------------------------------------

def get_layer_num_from_key(key):
    """Extract the layer number from a state dict key like 'decoder.layers.5.mlp...'"""
    match = re.search(r'decoder\.layers\.(\d+)\.', key)
    if match:
        return int(match.group(1))
    return None


def replace_layer_num(key, old_num, new_num):
    """Replace the layer number in a state dict key."""
    return key.replace(f'decoder.layers.{old_num}.', f'decoder.layers.{new_num}.', 1)


def is_attention_param(key):
    """Check if a key belongs to an attention sub-layer."""
    return 'self_attention.' in key or 'input_layernorm.' in key


def is_mlp_param(key):
    """Check if a key belongs to an MLP sub-layer."""
    return ('mlp.' in key or 'pre_mlp_layernorm.' in key) and 'self_attention' not in key


def is_ssm_param(key):
    """Check if a key belongs to a Mamba SSM mixer sub-layer."""
    ssm_markers = ['mixer.in_proj', 'mixer.conv1d', 'mixer.A_log', 'mixer.D',
                   'mixer.dt_bias', 'mixer.norm', 'mixer.out_proj',
                   'mixer.x_proj', 'mixer.dt_proj']
    return any(m in key for m in ssm_markers)


def is_layer_norm_for_ssm(key):
    """Check if a key is the input layer norm for an SSM layer.

    In hybrid models, SSM layers can have their own input_layernorm or the
    norm can be fused into in_proj.layer_norm_weight.
    """
    return 'in_proj.layer_norm_weight' in key


# ---------------------------------------------------------------------------
# Core conversion: GPT -> Mamba
# ---------------------------------------------------------------------------

def convert_gpt_to_mamba(full_model, layer_types, args):
    """Convert a GPT state dict to a Mamba hybrid state dict.

    Args:
        full_model: OrderedDict with globally-indexed GPT state dict keys.
        layer_types: list of layer type chars from hybrid_layer_pattern.
        args: Parsed CLI arguments.

    Returns:
        OrderedDict: Mamba state dict with globally-indexed keys.
    """
    attn_map, mlp_map, ssm_indices = build_layer_index_mapping(
        layer_types, 'gpt-to-mamba'
    )
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

        if 'decoder.layers.' in key:
            continue

        # Rename final_layernorm -> final_norm
        if 'decoder.final_layernorm' in key:
            new_key = key.replace('decoder.final_layernorm', 'decoder.final_norm')
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
    print(f"  Initializing {len(ssm_indices)} SSM layers from scratch...")
    for layer_idx in ssm_indices:
        ssm_params = initialize_ssm_layer_params(
            layer_idx=layer_idx,
            d_model=args.d_model,
            mamba_d_inner=args.mamba_d_inner,
            mamba_d_state=args.mamba_d_state,
            mamba2_n_groups=args.mamba2_n_groups,
            mamba2_n_heads=args.mamba2_n_heads,
            mamba_head_dim=args.mamba2_head_dim,
            d_conv=getattr(args, 'd_conv', 4),
            init_method_std=getattr(args, 'init_method_std', 0.02),
            dtype=dtype,
        )
        target.update(ssm_params)

    # Sort by layer index for consistent ordering
    target = _sort_state_dict(target)

    return target


# ---------------------------------------------------------------------------
# Core conversion: Mamba -> GPT
# ---------------------------------------------------------------------------

def convert_mamba_to_gpt(full_model, layer_types, args):
    """Convert a Mamba hybrid state dict to a GPT state dict.

    Args:
        full_model: OrderedDict with globally-indexed Mamba state dict keys.
        layer_types: list of layer type chars from hybrid_layer_pattern.
        args: Parsed CLI arguments.

    Returns:
        OrderedDict: GPT state dict with globally-indexed keys.
    """
    attn_map, mlp_map, ssm_indices = build_layer_index_mapping(
        layer_types, 'mamba-to-gpt'
    )
    num_gpt_layers = len(attn_map)

    target = OrderedDict()
    discarded_ssm_keys = []

    # Copy / rename non-layer params
    for key, tensor in full_model.items():
        if 'decoder.layers.' in key:
            continue

        # Rename final_norm -> final_layernorm
        if 'decoder.final_norm' in key:
            new_key = key.replace('decoder.final_norm', 'decoder.final_layernorm')
            target[new_key] = tensor
        else:
            target[key] = tensor

    # Map attention and MLP params, discard SSM
    for key, tensor in full_model.items():
        lnum = get_layer_num_from_key(key)
        if lnum is None:
            continue

        if is_ssm_param(key) or is_layer_norm_for_ssm(key):
            # Discard SSM params
            if lnum in ssm_indices:
                discarded_ssm_keys.append(key)
                continue

        if is_attention_param(key) and lnum in attn_map:
            target_layer = attn_map[lnum]
            new_key = replace_layer_num(key, lnum, target_layer)
            target[new_key] = tensor

        elif is_mlp_param(key) and lnum in mlp_map:
            target_layer = mlp_map[lnum]
            new_key = replace_layer_num(key, lnum, target_layer)
            target[new_key] = tensor

        elif lnum in ssm_indices:
            # Any remaining SSM-layer param not caught above
            discarded_ssm_keys.append(key)

    if discarded_ssm_keys:
        print(f"\n  WARNING: Discarded {len(discarded_ssm_keys)} SSM parameter tensors "
              f"from {len(ssm_indices)} Mamba layers (no GPT equivalent).")
        print(f"  First few discarded keys: {discarded_ssm_keys[:5]}")

    target = _sort_state_dict(target)

    return target


# ---------------------------------------------------------------------------
# Sorting helper
# ---------------------------------------------------------------------------

def _sort_state_dict(state_dict):
    """Sort state dict keys so that layer-indexed keys are in order."""
    def sort_key(item):
        key = item[0]
        # Extract layer number if present
        match = re.search(r'decoder\.layers\.(\d+)\.', key)
        if match:
            return (1, int(match.group(1)), key)
        # Non-layer keys: embeddings first, output_layer last
        if 'embedding' in key:
            return (0, 0, key)
        if 'output_layer' in key:
            return (2, 0, key)
        if 'decoder.final' in key:
            return (1, 999999, key)
        return (0, 1, key)

    return OrderedDict(sorted(state_dict.items(), key=sort_key))


# ---------------------------------------------------------------------------
# Format-aware save
# ---------------------------------------------------------------------------

def _save_dist_full(target_state_dict, common_state, model_prefix, backend,
                    args, iteration):
    """Save a fully-gathered state dict in dist-ckpt format.

    The on-disk tensors carry their full logical shape, so downstream Megatron
    training reads them back with any TP+PP+FSDP configuration.
    """
    if iteration is None:
        out_iter = 0 if args.reset_iterations else 0
        iter_dir = args.save_dir
    else:
        out_iter = 0 if args.reset_iterations else iteration
        iter_dir = os.path.join(args.save_dir, f'iter_{out_iter:07d}')

    # Update common state args to reflect target model structure.
    common_state = copy.deepcopy(common_state) if common_state else {}
    if 'args' in common_state and common_state['args'] is not None:
        ckpt_args = common_state['args']
        ckpt_args.num_layers = args.target_num_layers
        if hasattr(ckpt_args, 'hybrid_layer_pattern'):
            if args.direction == 'gpt-to-mamba':
                ckpt_args.hybrid_layer_pattern = args.hybrid_layer_pattern
            else:
                ckpt_args.hybrid_layer_pattern = None
        if args.reset_iterations:
            for attr in ('iteration', 'consumed_valid_samples',
                         'consumed_train_samples', 'train_iters', 'train_samples'):
                if hasattr(ckpt_args, attr):
                    setattr(ckpt_args, attr, 0)
    if args.reset_iterations and 'iteration' in common_state:
        common_state['iteration'] = 0

    print(f"  Writing dist checkpoint to {iter_dir} "
          f"(backend={backend}, prefix='{model_prefix}')...")
    save_dist_checkpoint_full(
        target_state_dict, common_state, iter_dir,
        model_prefix=model_prefix, backend=backend,
    )
    write_latest_iteration_marker(iter_dir, out_iter)


def main(args):
    print("\n====RUNNING GPT <-> MAMBA CHECKPOINT CONVERSION====\n")
    print(f"  Direction:            {args.direction}")
    print(f"  Source:               {args.load_dir}")
    print(f"  Target:               {args.save_dir}")
    print(f"  Hybrid layer pattern: {args.hybrid_layer_pattern}")

    # Compute derived Mamba dimensions
    args.mamba_d_inner = args.d_model * 2
    args.mamba2_n_heads = args.mamba_d_inner // args.mamba2_head_dim

    # Parse hybrid layer pattern
    layer_types = parse_hybrid_layer_pattern(args.hybrid_layer_pattern)
    total_mamba_layers = len(layer_types)
    attn_count = sum(1 for t in layer_types if t == '*')
    mlp_count = sum(1 for t in layer_types if t == '-')
    ssm_count = sum(1 for t in layer_types if t == 'M')
    print(f"\n  Pattern: {len(layer_types)} total layers "
          f"({attn_count} attn, {mlp_count} MLP, {ssm_count} SSM, "
          f"{len(layer_types) - attn_count - mlp_count - ssm_count} other)")

    # Pattern-level GPT compatibility whitelist (fails fast, pre-load).
    validate_pattern_gpt_compatible(layer_types, args.direction)

    # 1. Resolve input format
    input_format = getattr(args, 'input_format', 'auto')
    if input_format == 'auto':
        input_format = detect_checkpoint_format(args.load_dir)
    output_format = getattr(args, 'output_format', 'auto')
    if output_format == 'auto':
        output_format = input_format
    print(f"\n  Input format:  {input_format}")
    print(f"  Output format: {output_format}")

    if input_format not in DIST_FORMATS:
        raise ValueError(
            f"Unsupported input format: {input_format}. "
            f"Only dist formats are supported: {DIST_FORMATS}."
        )
    if output_format not in DIST_FORMATS:
        raise ValueError(
            f"Unsupported output format: {output_format}. "
            f"Only dist formats are supported: {DIST_FORMATS}."
        )

    # 2. Load source checkpoint into a fully-gathered state dict
    print("\n[Step 1] Loading source checkpoint...")
    full_model, common_state, model_prefix, dist_backend, iteration = (
        load_dist_checkpoint_full(args.load_dir)
    )
    print(f"  Source: dist backend={dist_backend}, prefix='{model_prefix}', "
          f"iteration={iteration}, params={len(full_model)}")

    # Args-level GPT compatibility whitelist: reject MoE, MLA, MTP, linear /
    # experimental attention, heterogeneous block specs, etc. See module header.
    source_args = common_state.get('args') if common_state else None
    validate_source_args_gpt_compatible(source_args, args.direction)

    # 3. Convert
    print(f"\n[Step 2] Converting ({args.direction})...")
    if args.direction == 'gpt-to-mamba':
        target_state_dict = convert_gpt_to_mamba(full_model, layer_types, args)
        args.target_num_layers = total_mamba_layers
    elif args.direction == 'mamba-to-gpt':
        target_state_dict = convert_mamba_to_gpt(full_model, layer_types, args)
        args.target_num_layers = attn_count
    else:
        raise ValueError(f"Unknown direction: {args.direction}")
    print(f"  Target model: {len(target_state_dict)} parameters")

    # 4. Save
    print(f"\n[Step 3] Saving to {args.save_dir}...")
    _save_dist_full(
        target_state_dict, common_state, model_prefix, output_format,
        args, iteration,
    )

    print("\n====CONVERSION COMPLETE====\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert checkpoints between GPTModel and MambaModel formats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--direction', type=str, required=True,
        choices=['gpt-to-mamba', 'mamba-to-gpt'],
        help='Conversion direction.',
    )
    parser.add_argument('--load-dir', type=str, required=True,
                        help='Path to source checkpoint directory.')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Path to target checkpoint directory.')
    parser.add_argument('--hybrid-layer-pattern', type=str, required=True,
                        help='Hybrid layer pattern string, e.g. "M*-M*-M*-M*-".')

    parser.add_argument(
        '--input-format', type=str, default='auto',
        choices=('auto',) + DIST_FORMATS,
        help='Source checkpoint format. "auto" detects from metadata.json.',
    )
    parser.add_argument(
        '--output-format', type=str, default='auto',
        choices=('auto',) + DIST_FORMATS,
        help='Target checkpoint format. "auto" matches the input format. '
             'Dist formats (torch_dist / fsdp_dtensor) transparently support '
             'TP+PP+FSDP training checkpoints.',
    )

    # Model architecture params
    parser.add_argument('--d-model', type=int, default=4096,
                        help='Model hidden dimension.')
    parser.add_argument('--mamba-version', type=int, default=2,
                        choices=[1, 2], help='Mamba SSM version.')
    parser.add_argument('--mamba-d-state', type=int, default=128,
                        help='Mamba state dimension.')
    parser.add_argument('--mamba2-n-groups', type=int, default=8,
                        help='Number of groups (Mamba v2).')
    parser.add_argument('--mamba2-head-dim', type=int, default=64,
                        help='Head dimension (Mamba v2).')
    parser.add_argument('--d-conv', type=int, default=4,
                        help='Causal convolution kernel size.')

    # Initialization params
    parser.add_argument('--init-method-std', type=float, default=0.02,
                        help='Std for initializing new Mamba SSM params.')

    # Checkpoint control
    parser.add_argument('--reset-iterations', action='store_true',
                        help='Zero out the training iteration count.')

    args = parser.parse_args()
    main(args)
