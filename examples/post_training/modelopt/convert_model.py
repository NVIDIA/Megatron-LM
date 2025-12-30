# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Convert a GPTModel."""
import functools
import json
import os
import sys
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import modelopt.torch.speculative as mtsp
import torch
from modelopt.torch.export import import_mcore_gpt_from_hf

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.parallel_state import destroy_model_parallel
from megatron.post_training.arguments import add_modelopt_args
from megatron.post_training.checkpointing import load_modelopt_checkpoint
from megatron.post_training.model_builder import modelopt_gpt_mamba_builder
from megatron.post_training.utils import report_current_memory_info, to_empty_if_meta
from megatron.training import get_args, get_tokenizer
from megatron.training.checkpointing import save_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training.utils import print_rank_0, unwrap_model
from model_provider import model_provider

ALGO_TO_CONFIG = {
    "eagle1": mtsp.config.EAGLE1_DEFAULT_CFG,
    "eagle3": mtsp.config.EAGLE3_DEFAULT_CFG,
    "eagle-mtp": mtsp.config.EAGLE_MTP_DEFAULT_CFG,
}


def add_convert_args(parser):
    """Add additional arguments for ModelOpt checkpoint convertion."""
    group = parser.add_argument_group(title='ModelOpt MCore checkpoint convertion')
    group.add_argument(
        "--pretrained-model-path", type=str, default=None, help="HuggingFace pretrained model"
    )
    group.add_argument(
        "--extra-model-path", type=str, default=None, help="Extra module weights to load"
    )
    group.add_argument(
        '--algorithm',
        type=str,
        choices=["medusa", "eagle1", "eagle3", "None"],
        default="None",
        help='Chosing between different speculative decoding algorithms. Default is None.',
    )
    group.add_argument(
        '--export-num-medusa-heads',
        type=int,
        default=0,
        help='Number of Medusa heads for speculative decoding.',
    )
    group.add_argument(
        "--eagle-config", type=str, default=None, help="EAGLE architecture config. If not given, " \
        "a default config will be use. If provided, it will overwrite the default config."
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

    model = get_model(functools.partial(model_provider, modelopt_gpt_mamba_builder), wrap_with_ddp=False)
    report_current_memory_info()

    unwrapped_model = unwrap_model(model)[0]

    if args.pretrained_model_path is not None:
        import_dtype = torch.float16 if args.fp16 else torch.bfloat16
        unwrapped_model = unwrap_model(model)[0]
        workspace_dir = os.environ.get("MLM_WORK_DIR", "/tmp")
        print_rank_0("Import model from Hugging Face checkpoint in dtype {}.".format(str(import_dtype)))
        import_mcore_gpt_from_hf(
            unwrapped_model,
            args.pretrained_model_path,
            workspace_dir,
            dtype = import_dtype,
        )
    elif args.load is not None:
        _ = load_modelopt_checkpoint(model)

    if args.algorithm in ("eagle1", "eagle3"):
        mtsp_config = ALGO_TO_CONFIG[args.algorithm]
        if args.eagle_config:
            with open(args.eagle_config)as f:
                eagle_config = json.load(f)
            mtsp_config["config"]["eagle_architecture_config"].update(eagle_config)
        
        if args.export_offline_model:
            mtsp_config["config"]["eagle_offline"] = True

        unwrapped_model = mtsp.convert(unwrapped_model, mtsp_config)

        if args.extra_model_path is not None:
            eagle_module = getattr(unwrapped_model, "eagle_module", None)
            if eagle_module is not None:
                mcore_eagle_state_dict = torch.load(args.extra_model_path)
                eagle_module.load_state_dict(mcore_eagle_state_dict, strict=False)
                
    elif args.algorithm == "medusa":
        config = {"medusa_num_heads": args.export_num_medusa_heads, "medusa_num_layers": 1}
        unwrapped_model = mtsp.convert(unwrapped_model, [("medusa", config)])


    print_rank_0(f"Converted Model:\n {model}")
    torch.distributed.barrier()

    save_checkpoint(1, model, None, None, 0, release=True)

    destroy_model_parallel()
