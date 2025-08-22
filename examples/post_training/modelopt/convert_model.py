# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Convert a GPTModel."""
import functools
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import modelopt.torch.speculative as mtsp
import torch
from modelopt.torch.export import import_mcore_gpt_from_hf

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.parallel_state import destroy_model_parallel
from megatron.post_training.arguments import add_modelopt_args
from megatron.post_training.checkpointing import load_modelopt_checkpoint
from megatron.post_training.model_provider import model_provider
from megatron.training import get_args, get_tokenizer
from megatron.training.checkpointing import save_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training.utils import print_rank_0, unwrap_model

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
        '--export-num-medusa-heads',
        type=int,
        default=0,
        help='Number of Medusa heads for speculative decoding.',
    )
    group.add_argument(
        '--export-eagle-algorithm',
        type=str,
        choices=['eagle1', 'eagle3', 'eagle-mtp'],
        default="eagle-mtp",
        help='Chosing the between different flavors of EAGLE algorithms.',
    )
    group.add_argument(
        '--export-num-eagle-layers',
        type=int,
        default=0,
        help='Number of EAGLE layers for speculative decoding.',
    )
    group.add_argument(
        '--export-draft-vocab-size',
        type=int,
        default=0,
        help='The reduced vocabulary size of the draft model.',
    )
    group.add_argument(
        '--export-eagle-ffn-hidden-size',
        type=int,
        default=0,
        help='ffn_hidden_size of the eagle module. Using base model ffn_hidden_size is set to 0.',
    )

    group.add_argument(
        '--export-num-mtp',
        type=int,
        default=0,
        help='Number of MTP modules for speculative decoding.',
    )
    group.add_argument(
        '--export-freeze-mtp',
        type=int,
        nargs="*",
        default=[],
        help='Index of MTP that will be frozen in training.',
    )
    group.add_argument(
        '--export-parallel-draft-step',
        type=int,
        default=1,
        help='The number of tokens generated in parallel draft. If set to 1, draft is not in parallel mode.',
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
        mtsp_config = ALGO_TO_CONFIG[args.export_eagle_algorithm]
        mtsp_config["config"]["eagle_num_layers"] = args.export_num_eagle_layers
        mtsp_config["config"]["draft_vocab_size"] = args.export_draft_vocab_size
        mtsp_config["config"]["ffn_hidden_size"] = args.export_eagle_ffn_hidden_size
        mtsp_config["config"]["parallel_draft_step"] = args.export_parallel_draft_step

        unwrapped_model = mtsp.convert(unwrapped_model, mtsp_config)

        if args.extra_model_path is not None:
            eagle_module = getattr(unwrapped_model, "eagle_module", None)
            if eagle_module is not None:
                mcore_eagle_state_dict = torch.load(args.extra_model_path)
                eagle_module.load_state_dict(mcore_eagle_state_dict, strict=False)

        # Add mask tokens for parallel draft
        if args.export_parallel_draft_step > 1:
            assert args.export_parallel_draft_step <= 4, "Parallel draft only supports steps less than or equal to 4."
            tokenizer = get_tokenizer()
            for i in range(args.export_parallel_draft_step - 1):
                mask_token = "[MASK_{}]".format(i)
                tokenizer._tokenizer.add_tokens([mask_token], special_tokens=True) 
                token_id = tokenizer._tokenizer.convert_tokens_to_ids(mask_token)
                setattr(unwrapped_model, "mask_token_{}".format(i), torch.tensor(token_id))
                

    if args.export_num_medusa_heads > 0:
        config = {"medusa_num_heads": args.export_num_medusa_heads, "medusa_num_layers": 1}
        unwrapped_model = mtsp.convert(unwrapped_model, [("medusa", config)])

    if args.export_num_mtp > 0:
        config = {
            "mtp_num_module": args.export_num_mtp,
            "mtp_num_layers": 1,
            "mtp_freeze_list": args.export_freeze_mtp,
            "use_last_layernorm": False,
        }
        unwrapped_model = mtsp.convert(unwrapped_model, [("mtp", config)])

    print_rank_0(f"Converted Model:\n {model}")
    torch.distributed.barrier()

    save_checkpoint(1, model, None, None, 0)

    destroy_model_parallel()
