# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Sample Generate GPT."""
import functools
import json
import os
import sys
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import torch
from modelopt.torch.speculative.plugins.megatron_eagle import MegatronARValidation

from megatron.post_training.arguments import add_modelopt_args
from megatron.post_training.checkpointing import load_modelopt_checkpoint
from megatron.post_training.model_builder import modelopt_gpt_mamba_builder
from megatron.post_training.utils import get_mtbench_chat_data
from megatron.training import get_args, get_model, get_tokenizer, initialize_megatron
from megatron.training.utils import print_rank_0, unwrap_model
from model_provider import model_provider

warnings.filterwarnings('ignore')



def add_ar_validation_args(parser):
    """Add additional arguments for ModelOpt acceptance rate validation."""
    group = parser.add_argument_group(title='ModelOpt ar validation')
    group.add_argument(
        "--osl", type=int, default=64, help="Output sequence length."
    )
    parser.add_argument(
        "--prompts-path",
        type=str,
        default=None,
        help="Path to the prompts json file. If not provided, MTBench will be used.",
    )
    parser.add_argument(
        "--ground-truth-path",
        type=str,
        default=None,
        help="Path to the ground truth pt file.",
    )
    parser.add_argument(
        "--steps", type=int, default=1, help="Only used in EAGLE."
    )
    parser.add_argument(
        "--save-ground-truth-path",
        type=str,
        default=None,
        help="Save path for the ground truth pt file.",
    )

    add_modelopt_args(parser)
    return parser


def check_arguments():
    """Checking user arguments."""
    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print_rank_0("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()

    if hasattr(args, 'moe_grouped_gemm') and args.moe_grouped_gemm == True:
        print_rank_0("WARNING: Forcing moe_grouped_gemm to False for PTQ and export.")
        args.moe_grouped_gemm = False


def get_current_memory_info():
    remaining_mem, total_mem = torch.cuda.mem_get_info()
    info = "rank {:02}  memory remaining {:03}% ({}/{} MB) ".format(
        torch.distributed.get_rank(),
        int(remaining_mem * 100 / total_mem),
        remaining_mem // 1048576,
        total_mem // 1048576,
    )
    return info


def report_current_memory_info():
    """Report current memory usage."""
    print(get_current_memory_info(), flush=True)
    torch.distributed.barrier()




if __name__ == "__main__":
    initialize_megatron(
        extra_args_provider=add_ar_validation_args,
        args_defaults={
            'tokenizer_type': 'HuggingFaceTokenizer',
            'no_load_rng': True,
            'no_load_optim': True,
        },
    )

    check_arguments()

    args = get_args()

    if not args.prompts_path:
        dataset = get_mtbench_chat_data()
        prompts = [[sample["conversations"][0]] for sample in dataset]
    else:
        with open(args.prompts_path, "r") as f:
            prompts = [json.loads(line) for line in f]

    if args.ground_truth_path is not None:
        ground_truth = torch.load(args.ground_truth_path)
        ground_truth = [gt.to(torch.cuda.current_device()) for gt in ground_truth]
    else:
        ground_truth = [None for _ in range(len(prompts))]

    tokenizer = get_tokenizer()._tokenizer
    model = get_model(functools.partial(model_provider, modelopt_gpt_mamba_builder), wrap_with_ddp=False)

    report_current_memory_info()

    if args.load is not None:
        load_modelopt_checkpoint(model, strict=not args.untie_embeddings_and_output_weights)
        print_rank_0("Done loading checkpoint")


    unwrapped_model = unwrap_model(model)[0]
    unwrapped_model.eval()

    validator = MegatronARValidation(unwrapped_model, tokenizer)
    gt = []
    ar = []
    for prompt, truth in zip(prompts, ground_truth):
        output = validator.validate(args.osl, prompt, ground_truth=truth, steps=args.steps)
        gt.append(output[0])
        ar.append(output[1])
    print_rank_0("Acceptance Rate: " + str(ar))
    print_rank_0("Average: " + str(sum(ar)/len(ar)))

    if args.save_ground_truth_path is not None:
        torch.save(gt, args.save_ground_truth_path)
