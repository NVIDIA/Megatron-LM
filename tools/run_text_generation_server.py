# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Sample Generate GPT"""
import os
import socket
import torch
from argparse import ArgumentParser
from megatron import print_rank_0
from megatron.core import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import GPTModel
from megatron.training import get_model
from megatron.text_generation_server import MegatronServer
from megatron.text_generation import generate_and_post_process, beam_search_and_post_process

def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    print_rank_0('building GPT model ...')
    model = GPTModel(num_tokentypes=0, parallel_output=False, pre_process=pre_process, post_process=post_process)
    return model

def add_text_generate_args(parser):
    parser.add_argument("--temperature", type=float, default=1.0, help='Sampling temperature.')
    parser.add_argument("--top_p", type=float, default=0.0, help='Top p sampling.')
    parser.add_argument("--top_k", type=int, default=0, help='Top k sampling.')
    parser.add_argument("--out-seq-length", type=int, default=1024, help='Size of the output generated text.')
    return parser

if __name__ == "__main__":
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer',
                                       'no_load_rng': True,
                                       'no_load_optim': True})

    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()
    
    # Set up model and load checkpoint
    model = get_model(model_provider, wrap_with_ddp=False)

    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]
    if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        server = MegatronServer(model)
        server.run("0.0.0.0")

    while True:
        choice = torch.cuda.LongTensor(1)
        torch.distributed.broadcast(choice, 0)
        try:
            if choice[0].item() == 0:
                generate_and_post_process(model)
            elif choice[0].item() == 1:
                beam_search_and_post_process(model)
        except ValueError:
            pass
