# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Export a GPTModel."""
import functools
import os
import sys
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import modelopt.torch.export as mtex
import torch

from megatron.post_training.arguments import add_modelopt_args
from megatron.post_training.checkpointing import load_modelopt_checkpoint
from megatron.post_training.model_provider import model_provider
from megatron.training import get_args, get_model
from megatron.training.initialize import initialize_megatron
from megatron.training.utils import unwrap_model

warnings.filterwarnings('ignore')


def add_modelopt_export_args(parser):
    """Add additional arguments for ModelOpt hf-like export."""
    group = parser.add_argument_group(title='ModelOpt hf-like export')
    group.add_argument(
        "--export-extra-modules",
        action="store_true",
        help="Export extra modules such as Medusa, EAGLE, or MTP.",
    )
    group.add_argument(
        "--pretrained-model-name",
        type=str,
        help="A pretrained model hosted inside a model repo on huggingface.co.",
    )
    group.add_argument("--export-dir", type=str, help="The target export path.")
    add_modelopt_args(parser)
    return parser


if __name__ == "__main__":
    initialize_megatron(
        extra_args_provider=add_modelopt_export_args,
        args_defaults={
            'tokenizer_type': 'HuggingFaceTokenizer',
            'no_load_rng': True,
            'no_load_optim': True,
        },
    )

    args = get_args()

    model = get_model(functools.partial(model_provider, parallel_output=True), wrap_with_ddp=False)

    if args.load is not None:
        _ = load_modelopt_checkpoint(model)

    unwrapped_model = unwrap_model(model)[0]

    mtex.export_mcore_gpt_to_hf(
        unwrapped_model,
        args.pretrained_model_name,
        export_extra_modules=args.export_extra_modules,
        dtype=torch.bfloat16,
        export_dir=args.export_dir,
    )
