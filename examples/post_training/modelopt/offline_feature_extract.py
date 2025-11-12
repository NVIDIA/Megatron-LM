# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Supervised Finetuning GPT."""
import functools
import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))


from megatron.core import mpu
from megatron.post_training.arguments import add_modelopt_args
from megatron.post_training.checkpointing import load_modelopt_checkpoint
from megatron.post_training.model_provider import model_provider
from megatron.training import get_args, get_model, get_tokenizer, initialize_megatron
from megatron.training.utils import print_rank_0, unwrap_model

from examples.post_training.modelopt.finetune import SFTDataset

def add_extract_args(parser):
    """Add additional arguments for feature extraction."""
    group = parser.add_argument_group(title='Feature extraction')
    group.add_argument("--num-samples", type=int, default=128000, help="Number of samples.")
    group.add_argument("--output-dir", type=str, help="Path to the output directory.")

    add_modelopt_args(parser)
    return parser

def extract_feature(dataset, model, output_dir, idx_start, idx_end):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(idx_start + mpu.get_expert_data_parallel_rank(), idx_end, mpu.get_expert_data_parallel_world_size()):
        file_name = "{:08d}.pt".format(i - idx_start)
        file_path = os.path.join(output_dir, file_name)
        if not os.path.exists(file_path):
            input_ids = dataset[i]["input_ids"][:dataset.seq_length].unsqueeze(0).to(torch.cuda.current_device())
            output = model(input_ids, return_eagle_inputs=True)
            if mpu.get_tensor_model_parallel_rank() == 0 and mpu.get_expert_model_parallel_rank() == 0:
                torch.save(output, file_path)
            torch.distributed.barrier()

if __name__ == "__main__":
    initialize_megatron(
        extra_args_provider=add_extract_args,
        args_defaults={
            'tokenizer_type': 'HuggingFaceTokenizer',
            'no_load_rng': True,
            'no_load_optim': True,
        },
    )

    args = get_args()
    tokenizer = get_tokenizer()
    model = get_model(functools.partial(model_provider, parallel_output=True), wrap_with_ddp=False)

    load_modelopt_checkpoint(model, strict=not args.untie_embeddings_and_output_weights)
    print_rank_0("Done loading checkpoint")

    unwrapped_model = unwrap_model(model)[0]
    unwrapped_model.eval()

    kwargs = {
        "tokenizer": tokenizer._tokenizer,
        "seq_length": args.seq_length,
        # Optional kwargs
        "hf_dataset": args.finetune_hf_dataset,
        "num_shards": mpu.get_expert_data_parallel_world_size(),
        "shard_index": mpu.get_expert_data_parallel_rank(),
    }
    sft_dataset = SFTDataset(args.num_samples, None, **kwargs)
    
    extract_feature(sft_dataset, unwrapped_model, os.path.join(args.output_dir, "train"), 0, int(args.num_samples * 0.98))
    extract_feature(sft_dataset, unwrapped_model, os.path.join(args.output_dir, "valid"), int(args.num_samples * 0.98), int(args.num_samples * 0.99))
    extract_feature(sft_dataset, unwrapped_model, os.path.join(args.output_dir, "test"), int(args.num_samples * 0.99), args.num_samples)


