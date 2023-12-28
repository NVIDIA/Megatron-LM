# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Sample Generate GPT"""
import torch
import os
import sys
from typing import Union

sys.path.append(os.path.abspath(os.path.join(
    os.path.join(os.path.dirname(__file__), "../../../"))))
from megatron import get_args, get_retro_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.core.models.gpt import GPTModel
from megatron.training import get_model
from tools.retro.text_generation.retro_api import retro_generate_and_post_process
from tools.retro.sft.sft_retro import get_tasks_args
from tools.retro.sft.dataset_conv import reformat_prompt, preprocess, reformat_prompt_short
import numpy as np
import time
import megatron.model
from megatron.arguments import core_transformer_config_from_args



def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.model.GPTModel]:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.model.GPTModel]: The returned model
    """
    print_rank_0('building GPT model ...')
    config = core_transformer_config_from_args(get_args())

    # not support core model yet
    model = megatron.model.GPTModel(
        config,
        num_tokentypes=0,
        parallel_output=False,
        pre_process=pre_process,
        post_process=post_process
    )

    return model


def pad_neighbours_for_query_only(args, nb_tokens, pad_id, ft_neighbours):
    # take top k neighbours and padding
    neighbours_tokens = []
    retro_args = get_retro_args()
    r = retro_args.retro_gpt_retrieved_length

    if args.reuse_top:
        valid_nb_tokens = nb_tokens[:args.retro_num_neighbors]
    else:
        valid_nb_tokens = nb_tokens[ft_neighbours:args.retro_num_neighbors + ft_neighbours]

    for nb_token in valid_nb_tokens:
        if len(nb_token) >= r:
            nb_token = nb_token[:r]
        else:
            nb_token = nb_token + [pad_id] * (r - len(nb_token))
        neighbours_tokens.append(nb_token)
    print("len(nb_tokens)", len(nb_tokens))
    print("len(neighbours_tokens)", len(neighbours_tokens))
    print("args.retro_num_neighbors", args.retro_num_neighbors)

    if len(neighbours_tokens) < args.retro_num_neighbors:
        assert ValueError("neighbours are not enough, add empty ones and create mask for those empty ones")
    neighbours_tokens = np.array(neighbours_tokens)
    return neighbours_tokens


def add_text_generate_args(parser):
    """Text generation arguments."""

    parser = get_tasks_args(parser)
    group = parser.add_argument_group(title='text generation')

    group.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    group.add_argument("--greedy", action='store_true', default=False,
                       help='Use greedy sampling.')
    group.add_argument("--top_p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--top_k", type=int, default=0,
                       help='Top k sampling.')
    group.add_argument("--out-seq-length", type=int, default=256,
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
    group.add_argument("--recompute", action='store_true',
                       help='During generation recompute all attention '
                            'instead of using previously computed keys/values.')
    group.add_argument("--epsilon", type=float, default=0.01,
                       help="Minimum factor by which each probability is multiplied")
    group.add_argument("--debug-gen", action='store_true',
                       help="If set, additional debugging output is printed to stdout")
    group.add_argument('--length-penalty', type=float, default=1.0,
                       help='length penalty')
    group.add_argument('--gen-start-idx', type=int, default=0,
                       help='project size for adapters')
    group.add_argument('--num-gen', type=int, default=-1,
                       help='project size for adapters')
    group.add_argument('--ckpt-step', type=int, default=None,
                       help='setting ckpt step manually')
    group.add_argument("--short-format", action='store_true',
                       help='Use short format QA')
    group.add_argument("--use-retrieved-neighbours", action='store_true', default=False,
                       help='Use retrieved neighbours')
    group.add_argument('--template-id', type=int, default=0,
                       help='template id for generation,')
    return parser


def generate_samples_conditional(model):
    args = get_args()
    start = time.time()
    avg_time = []
    tokenizer = get_tokenizer()
    model.eval()
    if torch.distributed.get_rank() == 0:

        data = preprocess(args.sample_input_file, inference_only=True,
                          retrieved_neighbours=args.use_retrieved_neighbours)
        print("total rows {}".format(len(data)))
        all_data = data[args.gen_start_idx:]  # start from gen_start_idx
        if args.num_gen > 0:
            all_data = all_data[:args.num_gen]
        input_count = len(all_data)
        input_pos = 0

    terminate_runs = 0
    while True:
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            sentences = []
            n_arrays = []
            print("global batch size", args.global_batch_size)
            for _ in range(args.global_batch_size):
                print(input_pos)
                if input_pos >= input_count:
                    print("reach the last row")
                    break
                else:
                    sample = all_data[input_pos]
                input_pos += 1

                if True:
                    max_target_len = args.out_seq_length
                    query, _, neighbours = sample

                    neighbours_array = pad_neighbours_for_query_only(args,
                                                                     [tokenizer.tokenize(neighbour) for neighbour in
                                                                      neighbours], tokenizer.eod, args.ft_neighbours)
                    print("neighbours_array.shape", neighbours_array.shape)
                    tokenizer = get_tokenizer()

                    if args.short_format:
                        input_tokens = reformat_prompt_short(query, neighbours, args.task, args.ft_neighbours,
                                                             max_target_len,
                                                             tokenizer, args.seq_length)
                    else:
                        input_tokens = reformat_prompt(query, neighbours, args.task, args.ft_neighbours, max_target_len,
                                                       tokenizer, args.seq_length, template_id=args.template_id)
                    raw_text = tokenizer.detokenize(input_tokens)
                    print(raw_text)
                else:
                    raise ValueError("invalid arg for task")
                sentences.append(raw_text)
            retro_args = get_retro_args()

            resp_sentences, resp_sentences_seg, scores, \
                tokens = retro_generate_and_post_process(model, prompts=sentences,
                                                         neighbours_array=neighbours_array,
                                                         tokens_to_generate=args.seq_length - retro_args.retro_gpt_chunk_length,
                                                         return_output_log_probs=False,
                                                         top_k_sampling=args.top_k,
                                                         top_p_sampling=args.top_p,
                                                         add_BOS=False,
                                                         temperature=1.0)
            print("len of resp_sentences", len(resp_sentences))
            for prompt, generation in zip(sentences, resp_sentences):
                datum = generation[len(prompt):]
                print("prompt:", generation[:len(prompt)])
                if "<|endoftext|>" in datum:
                    datum = datum[:datum.find("<|endoftext|>")].strip()
                datum = datum.replace("\n", " ")
                print("cont:", datum)
                yield datum
            avg_time.append((time.time() - start) / args.global_batch_size)
            print("avg time for each sample: ", sum(avg_time) / len(avg_time))
            start = time.time()
            if input_pos >= input_count:
                print("finish all lines")
                terminate_runs = 1
        else:
            retro_generate_and_post_process(model)

        terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
        torch.distributed.broadcast(terminate_runs_tensor, 0)
        terminate_runs = terminate_runs_tensor[0].item()

        if terminate_runs == 1:
            return


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
                f.write(datum + '\n')


def main():
    """Main program."""

    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'no_load_rng': True,
                                       'no_load_optim': True})

    # Set up model and load checkpoint
    model = get_model(model_provider, wrap_with_ddp=False)
    print(model)
    args = get_args()

    if args.load is not None:
        _ = load_checkpoint(model, None, None)
    model = model[0]

    # Generate samples.
    if args.sample_input_file is not None:
        print(f"{args.sample_input_file}")
        generate_and_write_samples_conditional(model)


if __name__ == "__main__":
    main()
