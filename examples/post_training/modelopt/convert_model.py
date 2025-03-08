# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Convert a GPTModel."""
import functools
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import modelopt
import modelopt.torch.speculative as mtsp
import torch
from modelopt.torch.export import import_mcore_gpt_from_hf

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.post_training.arguments import add_modelopt_args
from megatron.post_training.checkpointing import load_modelopt_checkpoint
from megatron.post_training.model_provider import model_provider
from megatron.training import get_args  # , get_model
from megatron.training.checkpointing import save_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training.utils import print_rank_0, unwrap_model


def add_convert_args(parser):
    """Add additional arguments for ModelOpt checkpoint convertion."""
    group = parser.add_argument_group(title='ModelOpt MCore checkpoint convertion')
    group.add_argument(
        "--pretrained-model-path", type=str, default=None, help="HuggingFace pretrained model"
    )
    add_modelopt_args(parser)
    return parser


def get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
    """Build the model."""
    args = get_args()
    args.model_type = model_type
    pre_process = mpu.is_pipeline_first_stage()
    post_process = mpu.is_pipeline_last_stage()
    model = model_provider_func(pre_process=pre_process, post_process=post_process)
    model.model_type = model_type
    return [model]


def check_arguments():
    """Checking user arguments."""
    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print_rank_0("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()

    if hasattr(args, 'moe_grouped_gemm') and args.moe_grouped_gemm == True:
        print_rank_0("WARNING: Forcing moe_grouped_gemm to False for PTQ and export.")
        args.moe_grouped_gemm = False


if __name__ == "__main__":
    initialize_megatron(
        extra_args_provider=add_convert_args,
        args_defaults={
            'tokenizer_type': 'HuggingFaceTokenizer',
            'no_load_rng': True,
            'no_load_optim': True,
        },
    )
    check_arguments()

    args = get_args()

    model = get_model(functools.partial(model_provider, parallel_output=True), wrap_with_ddp=False)

    unwrapped_model = unwrap_model(model)[0]

    if args.pretrained_model_path is not None:
        unwrapped_model = unwrap_model(model)[0]
        workspace_dir = os.environ.get("MLM_WORK_DIR", "/tmp")
        import_mcore_gpt_from_hf(unwrapped_model, args.pretrained_model_path, workspace_dir)
    elif args.load is not None:
        _ = load_modelopt_checkpoint(model)

    if args.export_num_eagle_layers > 0:
        # raise ValueError("EAGLE GPTModel is not yet supported!")
        config = {"eagle_num_layers": args.export_num_eagle_layers}
        unwrapped_model = mtsp.convert(unwrapped_model, [("eagle", config)])

    if args.export_num_medusa_heads > 0:
        config = {"medusa_num_heads": args.export_num_medusa_heads, "medusa_num_layers": 1}
        unwrapped_model = mtsp.convert(unwrapped_model, [("medusa", config)])

    print_rank_0(f"Converted Model:\n {model}")
    torch.distributed.barrier()

    save_checkpoint(1, model, None, None, 0)
