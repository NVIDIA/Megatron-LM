# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Attach a speculative-decoding draft module (EAGLE3) to a Megatron-Core checkpoint."""
import functools
import json
import os
import sys
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import modelopt.torch.speculative as mtsp
import torch

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.parallel_state import destroy_model_parallel
from megatron.core.utils import unwrap_model
from megatron.post_training.arguments import add_modelopt_args
from megatron.post_training.model_builder import modelopt_gpt_hybrid_builder
from megatron.post_training.utils import report_current_memory_info, to_empty_if_meta
from megatron.training import get_args
from megatron.training.argument_utils import inference_cfg_container_from_args
from megatron.training.arguments import parse_and_validate_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.global_vars import set_run_config
from megatron.training.initialize import initialize_megatron
from megatron.training.utils import print_rank_0
from model_provider import model_provider

ALGO_TO_CONFIG = {
    "eagle3": mtsp.config.EAGLE3_DEFAULT_CFG,
    "eagle-mtp": mtsp.config.EAGLE_MTP_DEFAULT_CFG,
}


def add_convert_args(parser):
    """Add additional arguments for ModelOpt checkpoint convertion."""
    group = parser.add_argument_group(title='ModelOpt MCore checkpoint convertion')
    group.add_argument(
        "--extra-model-path", type=str, default=None, help="Extra module weights to load"
    )
    group.add_argument(
        '--algorithm',
        type=str,
        choices=["eagle3"],
        default="eagle3",
        help='Speculative decoding algorithm of the draft module to attach.',
    )
    group.add_argument(
        "--eagle-config",
        type=str,
        default=None,
        help="EAGLE architecture config. If not given, "
        "a default config will be use. If provided, it will overwrite the default config.",
    )
    group.add_argument(
        "--mix-hidden-states",
        type=bool,
        default=False,
        help="Whether to mix hidden states from previous TTT step.",
    )

    add_modelopt_args(parser)
    return parser


def get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
    """Build the model."""
    args = get_args()
    args.model_type = model_type
    pre_process = mpu.is_pipeline_first_stage()
    post_process = mpu.is_pipeline_last_stage()

    if args.init_model_with_meta_device:
        with torch.device("meta"):
            model = model_provider_func(pre_process=pre_process, post_process=post_process)
        to_empty_if_meta(model, device="cuda")
    else:
        model = model_provider_func(pre_process=pre_process, post_process=post_process)

    model.model_type = model_type
    return [model]


def check_arguments():
    """Checking user arguments."""
    args = get_args()
    if args.load is None:
        raise ValueError(
            "--load is required. Import Hugging Face checkpoints with Megatron-Bridge first."
        )
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print_rank_0("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()

    if hasattr(args, 'moe_grouped_gemm') and args.moe_grouped_gemm == True:
        if not getattr(args, 'export_default_te_spec', False):
            print_rank_0("WARNING: Forcing moe_grouped_gemm to False for PTQ and export.")
            args.moe_grouped_gemm = False


if __name__ == "__main__":
    args = parse_and_validate_args(extra_args_provider=add_convert_args, args_defaults={
            'tokenizer_type': 'HuggingFaceTokenizer',
            'no_load_rng': True,
            'no_load_optim': True,
        })
    # Temporary args/config duplication during the training-loop refactor:
    # migrated settings use config; remaining settings still use legacy args.
    cfg = inference_cfg_container_from_args(args, build_model_config=False)
    set_run_config(cfg)
    initialize_megatron(training=False)
    check_arguments()

    args = get_args()

    # Meta device initialization for ParallelLinear only works if using cpu initialization.
    # Meta device initialization is used such that models can be materialized in low-precision
    # directly when ModelOpt real quant is used. Otherwise, the model is first initialized
    # as BF16 in memory which may result in OOM and defeat the purpose of real quant.
    if args.init_model_with_meta_device:
        args.use_cpu_initialization = True
    else:
        warnings.warn(
            "--init-model-with-meta-device is not set. If you would like to resume the "
            "model in low-bit directly (low-memory initialization and skipping 16-bit), "
            "--init-model-with-meta-device must be set.",
            UserWarning,
        )

    model = get_model(
        functools.partial(model_provider, modelopt_gpt_hybrid_builder), wrap_with_ddp=False
    )
    report_current_memory_info()

    unwrapped_model = unwrap_model(model)[0]

    load_checkpoint(model, None, None)

    if args.algorithm == "eagle3":
        mtsp_config = ALGO_TO_CONFIG[args.algorithm]
        if args.eagle_config:
            with open(args.eagle_config) as f:
                eagle_config = json.load(f)
            mtsp_config["config"]["eagle_architecture_config"].update(eagle_config)

        if args.export_offline_model:
            mtsp_config["config"]["eagle_offline"] = True
        if args.mix_hidden_states:
            mtsp_config["config"]["eagle_mix_hidden_states"] = True

        unwrapped_model = mtsp.convert(unwrapped_model, mtsp_config)

        if args.extra_model_path is not None:
            eagle_module = getattr(unwrapped_model, "eagle_module", None)
            if eagle_module is not None:
                mcore_eagle_state_dict = torch.load(args.extra_model_path)
                eagle_module.load_state_dict(mcore_eagle_state_dict, strict=False)

    print_rank_0(f"Converted Model:\n {model}")
    torch.distributed.barrier()

    save_checkpoint(1, model, None, None, 0, release=True)

    destroy_model_parallel()
