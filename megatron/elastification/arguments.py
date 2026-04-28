# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import math


def convert_per_lists_to_int_lists(config):
    """Convert all *_per_list attributes to *_int_list using model dimensions.

    Called once after model dimensions are known so downstream code can always
    use the int-list path without branching on which list type is active.
    After this call every *_per_list is None and every *_int_list is set.
    """
    conversions = [
        ('emb_per_list', 'emb_int_list', config.hidden_size),
        ('mlp_per_list', 'mlp_int_list', config.ffn_hidden_size),
        ('head_per_list', 'head_int_list', config.num_attention_heads),
        ('mamba_per_list', 'mamba_int_list', config.mamba_num_heads),
        ('moe_expert_per_list', 'moe_expert_int_list', config.num_moe_experts),
    ]
    for per_attr, int_attr, ref_dim in conversions:
        per_val = getattr(config, per_attr, None)
        if per_val is not None:
            setattr(config, int_attr, [math.floor(x * ref_dim) for x in per_val])
            setattr(config, per_attr, None)


def validate_flextron_per_int_lists(args):
    """
    Enforce mutual exclusion between ratio per-lists and integer choice lists.

    For each module, at most one of (*_per_list, *_int_list) may be set. If neither
    is set, *_per_list defaults to [1.0]. Skips when flextron-related args were not
    registered on the parser.
    """
    pairs = (
        ('mamba', 'mamba_per_list', 'mamba_int_list'),
        ('mlp', 'mlp_per_list', 'mlp_int_list'),
        ('emb', 'emb_per_list', 'emb_int_list'),
        ('head', 'head_per_list', 'head_int_list'),
        ('moe-expert', 'moe_expert_per_list', 'moe_expert_int_list'),
    )
    for cli_name, per_attr, int_attr in pairs:
        per_val = getattr(args, per_attr)
        int_val = getattr(args, int_attr)
        per_set = per_val is not None
        int_set = int_val is not None
        if per_set:
            for x in per_val:
                assert 0.0 <= x <= 1.0, f'--{cli_name}-per-list values must be in [0, 1], got {x}.'
        assert not (
            per_set and int_set
        ), f'Use either --{cli_name}-per-list or --{cli_name}-int-list for {cli_name}, not both.'
        if not per_set and not int_set:
            setattr(args, per_attr, [1.0])


def add_flextron_args(parser):
    group = parser.add_argument_group(title='flextron')
    # Distillation flags
    group.add_argument('--distillation', action='store_true', help='Enable self-distillation.')
    group.add_argument('--distill-coeff', type=float, default=0.0, help='Distillation coefficient.')
    group.add_argument('--distill-only', action='store_true', help='Distillation only.')
    # Basic Flextron flags
    group.add_argument('--flextron', action='store_true', help='Enable Flextron.')
    group.add_argument('--binary-mask', action='store_true', help='Use binary mask in Flextron.')
    group.add_argument('--slice', action='store_true', help='Use slice in Flextron.')
    group.add_argument('--enable-router', action='store_true', help='Enable router in Flextron.')
    group.add_argument(
        '--add-skipping', action='store_true', help='Add layer skipping in Flextron.'
    )
    group.add_argument('--no-attn-skip', action='store_true', help='No attn skip in Flextron.')
    group.add_argument(
        '--lr-mult-router',
        type=float,
        default=1.0,
        help='Learning rate multiplier for router in Flextron.',
    )
    group.add_argument('--flex-strict', action='store_true', help='Strict loading of Flextron.')
    group.add_argument('--is-flex-eval', action='store_true', help='Is Flextron evaluation.')
    group.add_argument('--freeze-router', action='store_true', help='Freeze router in Flextron.')
    group.add_argument('--freeze-model', action='store_true', help='Freeze model in Flextron.')
    group.add_argument(
        '--flex-hetero-ffn', action='store_true', help='Use flex hetero FFN in Flextron.'
    )
    group.add_argument(
        '--flex-hetero-mamba', action='store_true', help='Use flex hetero Mamba in Flextron.'
    )
    group.add_argument(
        '--flex-hetero-head',
        action='store_true',
        help='Use flex hetero attention head in Flextron.',
    )
    group.add_argument(
        '--flex-hetero-moe-expert',
        action='store_true',
        help='Use flex hetero MoE expert in Flextron.',
    )
    group.add_argument(
        '--router-std', type=float, default=0.1, help='Router init std for Flextron.'
    )
    group.add_argument(
        '--normalize-router-logits',
        action='store_true',
        help='Normalize router logits in Flextron.',
    )
    group.add_argument('--soft-mask', action='store_true', help='Soft mask in Flextron.')

    # Flextron hyperparameters
    group.add_argument(
        '--budget-probs',
        nargs='+',
        type=float,
        default=None,
        help='List of budget probabilities for Flextron.',
    )
    group.add_argument(
        '--prefill-chunk-size', type=int, default=16384, help='Prefill chunk size for Flextron.'
    )
    group.add_argument(
        '--mem-infer-seq-len',
        type=int,
        default=131072,
        help='Memory infer sequence length for Flextron.',
    )
    group.add_argument(
        '--mem-batch-size', type=int, default=1, help='Memory batch size for Flextron.'
    )
    group.add_argument(
        '--original-model-sample-prob',
        type=float,
        default=0.33,
        help='Probability of sampling the original model in Flextron.',
    )
    group.add_argument(
        '--force-router-skip',
        nargs='+',
        type=int,
        default=None,
        help='Force router skip for Flextron router.',
    )
    group.add_argument(
        '--force-mlp', nargs='+', type=float, default=None, help='Force MLP for Flextron router.'
    )
    group.add_argument(
        '--force-head',
        nargs='+',
        type=float,
        default=None,
        help='Force Attn head for Flextron router.',
    )
    group.add_argument(
        '--force-mamba',
        nargs='+',
        type=float,
        default=None,
        help='Force Mamba for Flextron router.',
    )
    group.add_argument(
        '--force-emb',
        nargs='+',
        type=float,
        default=None,
        help='Force Embedding for Flextron router.',
    )
    group.add_argument(
        '--skip-num-attn-layer-constraint',
        type=int,
        default=None,
        help='Skip number of attention layer constraint for Flextron router.',
    )
    group.add_argument(
        '--skip-total-layer-constraint',
        type=int,
        default=None,
        help='Skip total layer constraint for Flextron router.',
    )
    group.add_argument(
        '--disable-budget', action='store_true', help='Disable budget for Flextron router.'
    )
    group.add_argument(
        '--curr-iteration', type=int, default=None, help='Current iteration for Flextron router.'
    )
    group.add_argument(
        '--hard-sample-th',
        type=float,
        default=0.996,
        help='Hard sample threshold for Flextron router.',
    )
    group.add_argument(
        '--router-beta', type=float, default=1.0, help='Beta value for Flextron router.'
    )
    group.add_argument(
        '--loss-alpha', type=float, default=1.0, help='Alpha coefficient for Flextron loss.'
    )
    group.add_argument('--tau-init', type=float, default=1.0, help='Tau init for Flextron router.')
    group.add_argument(
        '--tau-decay', type=float, default=0.9999, help='Tau decay for Flextron router.'
    )
    group.add_argument(
        '--router-inter-dim',
        type=int,
        default=128,
        help='Intermediate dimension for Flextron router.',
    )
    group.add_argument(
        '--linear-scaler-start',
        type=float,
        default=1.0,
        help='Linear scaler start for Flextron router.',
    )
    group.add_argument(
        '--linear-scaler-end',
        type=float,
        default=10.0,
        help='Linear scaler end for Flextron router.',
    )
    group.add_argument(
        '--override-selected-budget',
        nargs='+',
        type=float,
        default=None,
        help='Override selected budget for Flextron router.',
    )
    group.add_argument('--router-gbs', type=int, default=32, help='Router gbs for Flextron router.')
    # Model configuration lists
    group.add_argument(
        '--budget-list',
        nargs='+',
        type=float,
        default=[1.0],
        help='List of budget values for Flextron.',
    )
    group.add_argument(
        '--mamba-per-list',
        nargs='+',
        type=float,
        default=None,
        help='List of Mamba percentage values for Flextron (mutually exclusive with --mamba-int-list).',
    )
    group.add_argument(
        '--mlp-per-list',
        nargs='+',
        type=float,
        default=None,
        help='List of MLP percentage values for Flextron (mutually exclusive with --mlp-int-list).',
    )
    group.add_argument(
        '--emb-per-list',
        nargs='+',
        type=float,
        default=None,
        help='List of embedding percentage values for Flextron (mutually exclusive with --emb-int-list).',
    )
    group.add_argument(
        '--head-per-list',
        nargs='+',
        type=float,
        default=None,
        help='List of head percentage values for Flextron (mutually exclusive with --head-int-list).',
    )
    group.add_argument(
        '--moe-expert-per-list',
        nargs='+',
        type=float,
        default=None,
        help='List of MoE expert percentage values for Flextron (mutually exclusive with --moe-expert-int-list).',
    )
    group.add_argument(
        '--mamba-int-list',
        nargs='+',
        type=int,
        default=None,
        help='List of Mamba integer router choices for Flextron (mutually exclusive with --mamba-per-list).',
    )
    group.add_argument(
        '--mlp-int-list',
        nargs='+',
        type=int,
        default=None,
        help='List of MLP integer router choices for Flextron (mutually exclusive with --mlp-per-list).',
    )
    group.add_argument(
        '--emb-int-list',
        nargs='+',
        type=int,
        default=None,
        help='List of embedding integer router choices for Flextron (mutually exclusive with --emb-per-list).',
    )
    group.add_argument(
        '--head-int-list',
        nargs='+',
        type=int,
        default=None,
        help='List of head integer router choices for Flextron (mutually exclusive with --head-per-list).',
    )
    group.add_argument(
        '--moe-expert-int-list',
        nargs='+',
        type=int,
        default=None,
        help='List of MoE expert integer router choices for Flextron (mutually exclusive with --moe-expert-per-list).',
    )
    group.add_argument(
        '--budget-type', type=str, default='param', choices=['param', 'mem'], help='Type of budget.'
    )
    # Memory quantization profile
    group.add_argument(
        '--memory-profile',
        type=str,
        default='bf16',
        help='Named memory quantization preset from memory_profiles.yaml '
        '(e.g. bf16, fp8_kv, fp8_all, int8).  '
        'Individual --bpe-* overrides take priority.',
    )
    group.add_argument(
        '--memory-profile-path',
        type=str,
        default=None,
        help='Path to a custom memory_profiles.yaml.  '
        'Defaults to the bundled megatron/elastification/memory_profiles.yaml.',
    )
    group.add_argument(
        '--bpe-params',
        type=float,
        default=None,
        help='Override bytes-per-element for model parameters ' '(2=BF16, 1=FP8/INT8, 0.5625=FP4).',
    )
    group.add_argument(
        '--bpe-kv-cache', type=float, default=None, help='Override bytes-per-element for KV cache.'
    )
    group.add_argument(
        '--bpe-ssm-cache',
        type=float,
        default=None,
        help='Override bytes-per-element for Mamba SSM state cache.',
    )
    group.add_argument(
        '--bpe-max-buffer',
        type=float,
        default=None,
        help='Override bytes-per-element for MoE dispatch buffer.',
    )
    group.add_argument(
        '--param-budget-target',
        type=str,
        default=None,
        choices=['active', 'total'],
        help='Whether param budget loss supervises on active params '
        '(top-k experts only) or total params.  '
        'Overrides the preset value from --memory-profile.',
    )
    group.add_argument(
        '--layer-ranking-list', nargs='+', type=int, default=None, help='List of layer ranking.'
    )
    group.add_argument(
        '--log-budgets',
        nargs='+',
        type=str,
        default=["all"],
        help='Budget values to log distillation loss for (space-separated list or "all").',
    )
    # Additional parameters

    group.add_argument(
        '--basemodel-type',
        type=str,
        default='nemotronh_8b',
        choices=['nemotronh_8b'],
        help='Base model type for parameter loss calculation.',
    )

    # Budget configuration
    group.add_argument(
        '--flextron-config-file',
        type=str,
        default=None,
        help='Configuration file for Flextron budget settings.',
    )

    return parser
