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
from megatron.post_training.model_builder import modelopt_gpt_mamba_builder
from megatron.training import get_args, get_model
from megatron.training.initialize import initialize_megatron
from megatron.training.utils import unwrap_model
from model_provider import model_provider

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

    # Meta device initialization for ParallelLinear only works if using cpu initialization.
    # Meta device initialization is used such that models can be materialized in low-precision
    # directly when ModelOpt real quant is used. Otherwise, the model is first initialized
    # as BF16 in memory which may result in OOM and defeat the purpose of real quant.
    args.use_cpu_initialization = True
    if not args.init_model_with_meta_device:
        warnings.warn(
            "--init-model-with-meta-device is not set. If you would like to resume the "
            "model in low-bit directly (low-memory initialization and skipping 16-bit), "
            "--init-model-with-meta-device must be set.",
            UserWarning,
        )

    model = get_model(functools.partial(model_provider, modelopt_gpt_mamba_builder), wrap_with_ddp=False)

    # Materialize the model from meta device to cpu before loading the checkpoint.
    unwrapped_model = unwrap_model(model)[0]
    unwrapped_model.to_empty(device="cpu")

    if args.load is not None:
        _ = load_modelopt_checkpoint(model)

    # Decide whether we are exporting only the extra_modules (e.g. EAGLE3).
    # Only the last pp stage may have extra_modules, hence broadcast from the last rank.
    export_extra_modules = hasattr(unwrapped_model, "eagle_module") or hasattr(unwrapped_model, "medusa_heads")
    torch.distributed.broadcast_object_list(
        [export_extra_modules],
        src=torch.distributed.get_world_size() - 1,
    )

    mtex.export_mcore_gpt_to_hf(
        unwrapped_model,
        args.pretrained_model_name,
        export_extra_modules=export_extra_modules,
        dtype=torch.bfloat16,
        export_dir=args.export_dir,
    )
