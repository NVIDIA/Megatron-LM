# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Sample Generate GPT."""
import functools
import os
import sys
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import torch
from datasets import load_dataset

from megatron.post_training.arguments import add_modelopt_args
from megatron.post_training.checkpointing import load_modelopt_checkpoint
from megatron.post_training.generate import simple_generate
from megatron.post_training.model_provider import model_provider
from megatron.post_training.utils import report_current_memory_info
from megatron.training import get_args, get_model, get_tokenizer, initialize_megatron
from megatron.training.utils import print_rank_0, unwrap_model

warnings.filterwarnings('ignore')


def add_generate_args(parser):
    """Add additional arguments for ModelOpt acceptance rate validation."""
    group = parser.add_argument_group(title='ModelOpt ar validation')
    group.add_argument("--osl", type=int, default=128, help="Output sequence length.")
    group.add_argument("--draft-length", type=int, default=0, help="Only used in EAGLE.")
    group.add_argument("--draft-topk", type=int, default=1, help="Only used in EAGLE.")
    group.add_argument("--disable-tqdm", action="store_true", help="Disable tqdm.")
    group.add_argument("--percentage", type=float, default=1.0)

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


def mtbench_to_oai_chat(example):
    """Convert MTBench data to OpenAI chat completion format."""
    conversations = []
    for prompt in example["prompt"]:
        conversations.append({"role": "user", "content": prompt})
    example["conversations"] = conversations
    return example


def get_conversations(example):
    """Extract the input for tokenizer.apply_chat_template."""
    conversations = example.get("conversations", None)
    if conversations is None:
        conversations = example.get("messages", None)
    if conversations is None:
        raise ValueError(
            "The data must either have conversations or messages field, but got {}".format(example)
        )
    return conversations


if __name__ == "__main__":
    initialize_megatron(
        extra_args_provider=add_generate_args,
        args_defaults={
            'tokenizer_type': 'HuggingFaceTokenizer',
            'no_load_rng': True,
            'no_load_optim': True,
        },
    )

    check_arguments()

    args = get_args()

    default_conversations = [
        {
            "role": "user",
            "content": "Write an email to a wine expert, requesting a guest "
            "article contribution for your wine blog.",
        }
    ]

    if args.finetune_hf_dataset is None:
        if args.draft_length > 0:
            dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
            dataset = dataset.map(mtbench_to_oai_chat)
        else:
            dataset = [{"conversations": default_conversations}]
    else:
        dataset = load_dataset(args.finetune_hf_dataset, split=args.finetune_data_split)

    tokenizer = get_tokenizer()._tokenizer
    model = get_model(functools.partial(model_provider, parallel_output=True), wrap_with_ddp=False)

    report_current_memory_info()

    if args.load is not None:
        load_modelopt_checkpoint(model, strict=not args.untie_embeddings_and_output_weights)
        print_rank_0("Done loading checkpoint")

    unwrapped_model = unwrap_model(model)[0]
    unwrapped_model.eval()

    for idx, example in enumerate(dataset):
        if idx > args.percentage * len(dataset):
            break
        ref_conversations = get_conversations(example)
        new_conversations = []

        for message in ref_conversations:
            ground_truth = None
            if message["role"] == "assistant":
                ground_truth = message["content"]
            if message["role"] == "user":
                new_conversations.append(message)
                print_rank_0(
                    "{}".format(
                        tokenizer.apply_chat_template(
                            new_conversations, tokenize=False, add_generation_prompt=True
                        )
                    )
                )
                input_ids = tokenizer.apply_chat_template(
                    new_conversations, return_tensors="pt", add_generation_prompt=True
                )
                output_ids = simple_generate(
                    unwrapped_model, input_ids.cuda(), osl=args.osl, disable_tqdm=args.disable_tqdm
                )
                output_texts = tokenizer.batch_decode(output_ids)[0]
                print_rank_0("{}".format(output_texts))
                new_conversations.append({"role": "assistant", "content": output_texts})

    torch.distributed.barrier()
