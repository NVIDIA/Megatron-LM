# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GPT/Hybrid pattern parsing, index mapping, and compatibility validation.

Standalone: no torch, no Megatron imports. Callable from both the offline
CLI tool and the in-memory auto-conversion hook in
``megatron/training/checkpointing.py``.
"""

VALID_LAYER_SYMBOLS = {'M', 'G', '*', '-', 'E'}

# Layer symbols GPTModel can emit or absorb:
#   '*' : standard self-attention layer (MHA / GQA / MQA)
#   '-' : standard (optionally gated) dense MLP layer
#   'E' : MoE MLP layer. Both sides keep the keys under
#         decoder.layers.<i>.mlp.{router,experts,shared_experts}.* so MoE
#         tensors round-trip verbatim (see convert_gpt_to_hybrid and
#         convert_hybrid_to_gpt — `is_mlp_param` already matches `mlp.*`).
# SSM ('M') has no GPT equivalent and is initialized from scratch /
# discarded (see convert_gpt_to_hybrid / convert_hybrid_to_gpt).
# 'G' (GDN) and 'D' (DS-attention) are not currently mapped — they would
# need separate key-naming work. Reject for now.
GPT_COMPATIBLE_PATTERN_SYMBOLS = {'M', '*', '-', 'E'}


# Pattern symbols that pair to a GPT-side MLP block. Both dense ('-') and MoE
# ('E') keep their state-dict keys under `decoder.layers.<i>.mlp.*`, so they
# round-trip identically. The pattern uniformity check
# (validate_pattern_gpt_compatible) ensures '-' and 'E' don't appear together,
# which would mean GPT layers aren't uniform.
_MLP_BEARING_SYMBOLS = ('-', 'E')


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
                f"Invalid layer symbol '{ch}' in pattern. " f"Valid symbols: {VALID_LAYER_SYMBOLS}"
            )
    return layer_types


def build_layer_index_mapping(layer_types, direction):
    """Build mapping between GPT layer indices and hybrid-model layer indices.

    For gpt-to-hybrid:
        Returns (attn_map, mlp_map, ssm_indices) where:
        - attn_map[gpt_layer_i] = hybrid_layer_j  (j is the index of the i-th '*')
        - mlp_map[gpt_layer_i]  = hybrid_layer_k  (k is the index of the i-th
          MLP-bearing position; either '-' or 'E')

    For hybrid-to-gpt:
        Returns (attn_map, mlp_map, ssm_indices) where:
        - attn_map[hybrid_attn_idx] = gpt_layer_i
        - mlp_map[hybrid_mlp_idx]   = gpt_layer_i
    """
    attn_indices = [i for i, t in enumerate(layer_types) if t == '*']
    mlp_indices = [i for i, t in enumerate(layer_types) if t in _MLP_BEARING_SYMBOLS]
    ssm_indices = [i for i, t in enumerate(layer_types) if t == 'M']

    if direction == 'gpt-to-hybrid':
        if len(attn_indices) != len(mlp_indices):
            raise ValueError(
                f"For gpt-to-hybrid, the number of attention layers ({len(attn_indices)}) "
                f"must equal the number of MLP/MoE layers ({len(mlp_indices)}) in the pattern."
            )
        attn_map = {i: attn_indices[i] for i in range(len(attn_indices))}
        mlp_map = {i: mlp_indices[i] for i in range(len(mlp_indices))}
        return attn_map, mlp_map, ssm_indices

    elif direction == 'hybrid-to-gpt':
        if len(attn_indices) != len(mlp_indices):
            raise ValueError(
                f"For hybrid-to-gpt, the number of attention layers ({len(attn_indices)}) "
                f"must equal the number of MLP/MoE layers ({len(mlp_indices)}) in the pattern."
            )
        attn_map = {attn_indices[i]: i for i in range(len(attn_indices))}
        mlp_map = {mlp_indices[i]: i for i in range(len(mlp_indices))}
        return attn_map, mlp_map, ssm_indices

    else:
        raise ValueError(f"Unknown direction: {direction}")


_GPT_COMPAT_REJECT_FIELDS = (
    (
        'moe_layer_freq',
        lambda value: (
            value is not None
            and not (isinstance(value, int) and value == 1)
            and not (isinstance(value, str) and value.strip() in ('', '1'))
            and not (isinstance(value, (list, tuple)) and all(item == 1 for item in value))
        ),
        'interleaved dense/MoE layers (moe_layer_freq)',
    ),
    (
        'experimental_attention_variant',
        lambda value: value is not None and value != '',
        'experimental attention variant (gated_delta_net / dsa / ...)',
    ),
    (
        'linear_attention_freq',
        lambda value: value is not None,
        'linear attention layers (linear_attention_freq)',
    ),
    ('heterogeneous_block_specs', lambda value: bool(value), 'heterogeneous per-layer block specs'),
    (
        'heterogeneous_layers_config_path',
        lambda value: value is not None and value != '',
        'heterogeneous layers config (Nemotron-NAS)',
    ),
    (
        'heterogeneous_layers_config_encoded_json',
        lambda value: value is not None and value != '',
        'heterogeneous layers config (Nemotron-NAS, inline JSON)',
    ),
    ('multi_latent_attention', lambda value: bool(value), 'Multi-Latent Attention (MLA)'),
    (
        'mtp_num_layers',
        lambda value: value is not None and value > 0,
        'Multi-Token Prediction head (mtp_num_layers)',
    ),
)


def validate_pattern_gpt_compatible(layer_types, direction):
    """Raise ``ValueError`` if a hybrid pattern cannot round-trip with GPTModel."""
    bad_symbols = sorted(
        {symbol for symbol in layer_types if symbol not in GPT_COMPATIBLE_PATTERN_SYMBOLS}
    )
    if bad_symbols:
        raise ValueError(
            f"Hybrid layer pattern contains symbols {bad_symbols} that are not "
            f"GPT-compatible (allowed: {sorted(GPT_COMPATIBLE_PATTERN_SYMBOLS)}). "
            f"'G' (GDN) and 'D' (DS-attention) are not currently key-translated "
            f"and cannot be {direction}-converted."
        )

    mlp_kinds = {symbol for symbol in layer_types if symbol in _MLP_BEARING_SYMBOLS}
    if len(mlp_kinds) > 1:
        raise ValueError(
            "Hybrid layer pattern mixes '-' (dense MLP) and 'E' (MoE) "
            "positions. GPTModel layers must be uniform — either all GPT "
            "layers are dense MLP, or all are MoE. Use only one of '-' or "
            "'E' in the pattern."
        )

    num_attention = sum(1 for symbol in layer_types if symbol == '*')
    num_mlp = sum(1 for symbol in layer_types if symbol in _MLP_BEARING_SYMBOLS)
    if num_attention != num_mlp:
        raise ValueError(
            "GPT-compatible hybrid patterns must pair every attention layer "
            f"('*') with one MLP/MoE layer ('-' or 'E'). Got {num_attention} '*' "
            f"and {num_mlp} MLP-bearing layers in the pattern."
        )


def validate_source_args_gpt_compatible(source_args, direction):
    """Raise ``ValueError`` if source arguments describe an unsupported GPT model."""
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
            continue

    if rejected:
        joined = "\n".join(rejected)
        raise ValueError(
            f"Source checkpoint is not GPT-compatible for {direction} "
            f"conversion. The following features have no GPTModel equivalent "
            f"and would produce a corrupt target checkpoint:\n{joined}\n"
            f"Remove these features from the model (or use a different "
            f"conversion tool) before running gpt_hybrid_conversion."
        )
