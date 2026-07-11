# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
GPT <-> Hybrid Checkpoint Conversion Tool
=========================================

Directly converts checkpoints between GPTModel (homogeneous Transformer) and
HybridModel (hybrid Mamba+Transformer) without going through HuggingFace as an
intermediary.

Supported directions:
    gpt-to-hybrid : Convert a GPT checkpoint to Hybrid format.
    hybrid-to-gpt : Convert a Hybrid checkpoint to GPT format.

The pure state-dict transformation logic lives in
``megatron.core.models.hybrid.conversion``. This file is the offline CLI
wrapper: it handles argparse, on-disk dist-checkpoint I/O, and the small bit
of common-state bookkeeping needed to make the output directory a fully
valid Megatron distributed checkpoint.

How the hybrid layer pattern maps GPT layers (gpt-to-hybrid):
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
    gpt-to-hybrid: SSM layers (M) are initialized from scratch:
        - A_log:          log(uniform(1, 16))
        - dt_bias:        inverse_softplus(log_uniform(dt_min, dt_max))
        - D:              ones
        - conv1d.weight:  kaiming_uniform(a=sqrt(5))
        - conv1d.bias:    zeros
        - in_proj.weight: kaiming_uniform(a=sqrt(5))
        - in_proj.layer_norm_weight: ones
        - out_proj.weight: kaiming_uniform(a=sqrt(5))
        - norm.weight:    ones
    hybrid-to-gpt: SSM layers are discarded with a warning.

Supported checkpoint formats:
    - torch_dist   : Megatron distributed checkpoint (TP + PP + FSDP).
    - fsdp_dtensor : FSDP DTensor export (TP + PP + FSDP).

    PyTorch DCP gathers TP/PP/FSDP shards via the checkpoint's global-shape
    metadata, so no explicit TP/PP/DP config is needed on input. The input
    format is auto-detected; the output format defaults to the input format.

    The legacy ``mp_rank_XX/model_optim_rng.pt`` layout is not supported —
    convert old checkpoints to ``torch_dist`` first.

Example commands:
    # GPT -> Hybrid (TP+PP+FSDP dist checkpoint)
    python tools/checkpoint/gpt_hybrid_conversion.py \\
        --direction gpt-to-hybrid \\
        --load-dir /path/to/gpt-dist-checkpoint \\
        --save-dir /path/to/hybrid-dist-checkpoint \\
        --hybrid-layer-pattern "M*-M*-M*-M*-" \\
        --d-model 4096 \\
        --mamba-d-state 128 \\
        --mamba2-n-groups 8 \\
        --mamba2-head-dim 64

    # Hybrid -> GPT (dist checkpoint)
    python tools/checkpoint/gpt_hybrid_conversion.py \\
        --direction hybrid-to-gpt \\
        --load-dir /path/to/hybrid-dist-checkpoint \\
        --save-dir /path/to/gpt-dist-checkpoint \\
        --hybrid-layer-pattern "M*-M*-M*-M*-" \\
        --d-model 4096 \\
        --mamba-d-state 128 \\
        --mamba2-n-groups 8 \\
        --mamba2-head-dim 64
"""

import argparse
import copy
import os
import sys

# Ensure the in-repo Megatron package is importable when this script is
# invoked directly (``python tools/checkpoint/gpt_hybrid_conversion.py``)
# from a checkout that hasn't been installed as a package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dist_checkpoint_io import (
    DIST_FORMATS,
    detect_checkpoint_format,
    load_dist_checkpoint_full,
    save_dist_checkpoint_full,
    write_latest_iteration_marker,
)
from megatron.core.models.hybrid.conversion import (
    convert_gpt_to_hybrid,
    convert_hybrid_to_gpt,
    parse_hybrid_layer_pattern,
    validate_pattern_gpt_compatible,
    validate_source_args_gpt_compatible,
)

# ---------------------------------------------------------------------------
# Format-aware save
# ---------------------------------------------------------------------------

def _save_dist_full(target_state_dict, common_state, model_prefix, backend,
                    args, iteration):
    """Save a fully-gathered state dict in dist-ckpt format.

    The on-disk tensors carry their full logical shape, so downstream Megatron
    training reads them back with any TP+PP+FSDP configuration.
    """
    # Always write into an iter_XXXXXXX/ subdirectory so
    # write_latest_iteration_marker can leave a discoverable tracker file next
    # to it. Reset requests and unknown iteration both fall back to iter 0.
    out_iter = 0 if (args.reset_iterations or iteration is None) else iteration
    iter_dir = os.path.join(args.save_dir, f'iter_{out_iter:07d}')

    # Update common state args to reflect target model structure.
    common_state = copy.deepcopy(common_state) if common_state else {}
    if 'args' in common_state and common_state['args'] is not None:
        ckpt_args = common_state['args']
        ckpt_args.num_layers = args.target_num_layers
        if hasattr(ckpt_args, 'hybrid_layer_pattern'):
            if args.direction == 'gpt-to-hybrid':
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
    print("\n====RUNNING GPT <-> Hybrid CHECKPOINT CONVERSION====\n")
    print(f"  Direction:            {args.direction}")
    print(f"  Source:               {args.load_dir}")
    print(f"  Target:               {args.save_dir}")
    print(f"  Hybrid layer pattern: {args.hybrid_layer_pattern}")

    # Compute derived Mamba dimensions
    args.mamba_d_inner = args.d_model * 2
    args.mamba2_n_heads = args.mamba_d_inner // args.mamba2_head_dim

    # Parse hybrid layer pattern
    layer_types = parse_hybrid_layer_pattern(args.hybrid_layer_pattern)
    total_hybrid_layers = len(layer_types)
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
    if args.direction == 'gpt-to-hybrid':
        target_state_dict = convert_gpt_to_hybrid(full_model, layer_types, args)
        args.target_num_layers = total_hybrid_layers
    elif args.direction == 'hybrid-to-gpt':
        target_state_dict = convert_hybrid_to_gpt(full_model, layer_types, args)
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
        description="Convert checkpoints between GPTModel and HybridModel formats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--direction', type=str, required=True,
        choices=['gpt-to-hybrid', 'hybrid-to-gpt'],
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
