# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.


"""Sample Generate GPT"""
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))
import torch
from megatron import get_args
from megatron import get_tokenizer
from megatron import print_rank_0
from megatron.checkpointing import load_checkpoint
from megatron.core import mpu
from megatron.initialize import initialize_megatron
from megatron.model import GPTModel
from megatron.training import get_model
from megatron.text_generation import generate_and_post_process


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    model = GPTModel(num_tokentypes=0, parallel_output=False,
                     pre_process=pre_process, post_process=post_process)

    return model

def add_text_generate_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='text generation')

    group.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    group.add_argument("--greedy", action='store_true', default=False,
                       help='Use greedy sampling.')
    group.add_argument("--top_p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--top_k", type=int, default=0,
                       help='Top k sampling.')
    group.add_argument("--out-seq-length", type=int, default=1024,
                       help='Size of the output generated text.')
    group.add_argument("--sample-input-file", type=str, default=None,
                       help='Get input from file instead of interactive mode, '
                       'each line is an input.')
    group.add_argument("--sample-output-file", type=str, default=None,
                       help='Output file got from --sample-input-file')
    group.add_argument("--num-samples", type=int, default=0,
                       help='Number of samples to generate unconditionally, '
                       'defaults to 0 and interactive conditional sampling')
    group.add_argument("--genfile", type=str,
                       help='Output file when generating unconditionally')
    return parser

def generate_samples_unconditional(model):
    args = get_args()

    if torch.distributed.get_rank() == 0:
        cnt = 0
        num_samples = args.num_samples
        from tqdm import tqdm
        pbar = tqdm(total=num_samples)

    while True:
        if torch.distributed.get_rank() == 0:
            sentences = [''] * args.global_batch_size
            print("global batch size", args.global_batch_size)
            max_len = args.out_seq_length
            resp_sentences, resp_sentences_seg, output_logits, \
            tokens = generate_and_post_process(model, prompts=sentences,
                                               tokens_to_generate=max_len,
                                               return_output_log_probs=False,
                                               top_k_sampling=args.top_k,
                                               top_p_sampling=args.top_p,
                                               add_BOS=True,
                                               temperature=1.0)
            for prompt, generation, token in zip(sentences, resp_sentences, tokens):
                datum = {'text': generation[len(prompt):], 'all_text': generation, 'prompt': prompt, 'id': cnt}
                yield datum
                cnt += 1
                pbar.update()
                if cnt >= num_samples:
                    break

            if cnt >= num_samples:
                pbar.close()
                break
        else:
            generate_and_post_process(model)


def generate_samples_conditional(model):
    args = get_args()

    if torch.distributed.get_rank() == 0:
        num_samples = args.num_samples
        cnt = 0
        from tqdm import tqdm
        pbar = tqdm(total=num_samples)

        fname = open(args.sample_input_file, "r")
        lines = fname.readlines()
        all_raw_text = [json.loads(line)['prompt']['text'] for line in lines]
        input_count = len(all_raw_text)
        input_pos = 0

    while True:
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            sentences = []
            print("global batch size", args.global_batch_size)
            for _ in range(args.global_batch_size):
                if input_pos >= input_count:
                    print(f"input pos: {input_pos}, input count: {input_count}")
                    raw_text = "EMPTY TEXT"
                else:
                    raw_text = all_raw_text[input_pos]
                input_pos += 1
                sentences.append(raw_text)

            max_len = args.out_seq_length
            resp_sentences, resp_sentences_seg, output_logits, \
            tokens = generate_and_post_process(model, prompts=sentences,
                                               tokens_to_generate=max_len,
                                               return_output_log_probs=False,
                                               top_k_sampling=args.top_k,
                                               top_p_sampling=args.top_p,
                                               add_BOS=False,
                                               temperature=1.0)
            for prompt, generation, token in zip(sentences, resp_sentences, tokens):
                datum = {'text': generation[len(prompt):], 'all_text': generation, 'prompt': prompt, 'id': cnt}
                yield datum
                cnt += 1
                pbar.update()
                if cnt >= num_samples:
                    break

            if cnt >= num_samples:
                pbar.close()
                break
        else:
            generate_and_post_process(model)


def generate_and_write_samples_unconditional(model):
    args = get_args()
    assert args.genfile is not None
    with open(args.genfile, 'w') as f:
        for datum in generate_samples_unconditional(model):
            if torch.distributed.get_rank() == 0:
                f.write(json.dumps(datum) + '\n')


def generate_and_write_samples_conditional(model):
    args = get_args()
    if args.sample_output_file is None:
        sample_output_file = args.sample_input_file + ".out"
        print('`sample-output-file` not specified, setting '
              'it to {}'.format(sample_output_file))
    else:
        sample_output_file = args.sample_output_file
    with open(sample_output_file, 'w') as f:
        for datum in generate_samples_conditional(model):
            if torch.distributed.get_rank() == 0:
                f.write(json.dumps(datum) + '\n')


def main():
    """Main program."""

    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer',
                                       'no_load_rng': True,
                                       'no_load_optim': True,
                                       'seq_length': 2048})

    # Set up model and load checkpoint
    model = get_model(model_provider, wrap_with_ddp=False)

    args = get_args()

    if args.load is not None:
        _ = load_checkpoint(model, None, None)
    model = model[0]

    # Generate samples.
    if args.sample_input_file != None:
        print(f"{args.sample_input_file}")
        generate_and_write_samples_conditional(model)
    else:
        generate_and_write_samples_unconditional(model)


if __name__ == "__main__":

    main()
