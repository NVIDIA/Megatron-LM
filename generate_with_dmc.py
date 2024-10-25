# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Run a simple inference with a Dynamic Memory Compression model."""

import datetime
import itertools
import json
import os
import sys
import time
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import numpy as np
import torch

from megatron.contrib.dmc import add_dmc_layer
from megatron.contrib.dmc.arguments import add_dmc_args

add_dmc_args()

from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.inference.text_generation.forward_step import ForwardStep
from megatron.inference.text_generation.generation import _build_attention_mask_and_position_ids
from megatron.inference.text_generation.sampling import sample
from megatron.training import get_args, get_model, get_tokenizer
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training.training import build_train_valid_test_data_iterators
from pretrain_gpt import train_valid_test_datasets_provider


class Timer:
    def __init__(self, name='', enabled=True):
        self.name = name + ': ' if name != '' else name
        self.enabled = enabled
        if enabled:
            torch.cuda.synchronize()
            self.tik = time.time()

    def end(self):
        if self.enabled:
            torch.cuda.synchronize()
            return time.time() - self.tik


def print_rank_0(message, end="\n"):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True, end=end)
    else:
        print(message, flush=True, end=end)


def model_provider(pre_process=True, post_process=True):
    args = get_args()
    config = core_transformer_config_from_args(get_args())

    assert args.spec is None
    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        args.num_experts, args.moe_grouped_gemm
    )

    if args.generate_dmc:
        add_dmc_layer(transformer_layer_spec)

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
    )

    return model


def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')
    group.add_argument('--generate-prompt-phase', action='store_true')
    group.add_argument('--generate-len', type=int, default=2048)
    group.add_argument('--generate-batch', type=int, default=16)
    group.add_argument('--generate-top-k', type=int, default=16)
    group.add_argument('--generate-iters', type=int, default=32)
    group.add_argument('--generate-print', action='store_true')
    group.add_argument('--generate-reps', type=int, default=1)
    group.add_argument('--generate-save', action='store_true', help='Save benchmarking results')
    group.add_argument('--generate-bmark-tokens', type=int, default=32, help='Measure latency of that many last tokens')
    group.add_argument('--generate-dmc', action='store_true')
    group.add_argument('--generate-context-len', type=int, default=64,
        help='If supplied a dataset through --data-path, use that many tokens for context')
    group.add_argument('--generate-prompt-file', type=str, default=None,
        help='When specified it will be used as input prompt instead of a sample from the dataset')
    return parser


import torch.distributed.elastic.multiprocessing.errors
@torch.distributed.elastic.multiprocessing.errors.record
def generate_tokens_probs_and_return_on_first_stage(model):
    args = get_args()
    tokenizer = get_tokenizer()

    tp_rank = parallel_state.get_tensor_model_parallel_rank()

    torch.cuda.manual_seed(args.seed)

    args.iteration = 0
    args.train_iters = 0
    args.skip_train = True

    def detok(t):
        return tokenizer.detokenize(t.cpu().tolist())

    if args.generate_prompt_file is not None:
        with open(args.generate_prompt_file) as f:
            prompt = f.read()
        tokens = torch.tensor(tokenizer.tokenize(prompt), dtype=torch.int64, device=torch.device("cuda"))
        args.generate_context_len = tokens.size(0)
        tokens =  torch.nn.functional.pad(tokens, (0, args.generate_len + 1))
        data = {"tokens": tokens[None, :].repeat(args.generate_batch, 1)}
    else:
        args.micro_batch_size = args.generate_batch
        _, valid_data_iterator, test_data_iterator = build_train_valid_test_data_iterators(
            train_valid_test_datasets_provider)

        assert valid_data_iterator is not None
        data = next(valid_data_iterator)
    tokens = tensor_parallel.broadcast_data(['tokens'], data, torch.int64)['tokens'].cuda()

    batch_size = tokens.size(0)
    context_len = args.generate_context_len
    total_len = context_len + args.generate_len
    with_poolsum = not args.use_flash_attn

    all_times = []
    bmark_times = []
    num_tokens_for_average = args.generate_bmark_tokens  # From every rep

    for rep in range(args.generate_reps):

        times = []
        forward_step = ForwardStep(model, batch_size, total_len)

        with torch.no_grad():
            attention_mask, position_ids = _build_attention_mask_and_position_ids(tokens)
            prev_num_tokens = 0

            if args.generate_prompt_phase:
                ctx = args.generate_context_len
                steps = itertools.chain([ctx], range(ctx + 1, total_len + 1))
            else:
                steps = range(1, total_len + 1)

            num_flushed_chars = 0

            for num_tokens in steps:

                if with_poolsum:
                    # Reset KV Cache and take the entire context
                    prev_num_tokens = 0
                    forward_step = ForwardStep(model, batch_size, total_len)

                # Pick the slice that we need to pass through the network.
                tokens2use = tokens[:, prev_num_tokens:num_tokens]

                positions2use = position_ids[:, prev_num_tokens:num_tokens]
                attention_mask2use = attention_mask[..., prev_num_tokens:num_tokens, :num_tokens]

                timer = Timer()
                logits = forward_step(tokens2use, positions2use, attention_mask2use)
                elapsed = timer.end()

                times.append(elapsed)

                torch.manual_seed(num_tokens)
                torch.cuda.manual_seed(num_tokens)

                new_sample = sample(
                    logits[:, -1, :],
                    top_k=args.generate_top_k,
                    top_p=0.0,
                    temperature=1.0,
                    vocab_size=tokenizer.vocab_size
                )

                out = (
                    f"{rep}: {num_tokens: >5} tokens | {1000*elapsed: >6.2f} ms | "
                    f"{1/elapsed*args.generate_batch: >5.0f} tok/s"
                )

                if num_tokens >= context_len:
                    tokens[:, num_tokens] = new_sample
                    if args.generate_print:
                        out_text = (
                            detok(tokens[0][:context_len]) + " <END_OF_PROMPT> "
                            + detok(tokens[0][context_len:num_tokens + 1])
                        )
                else:
                    if args.generate_print:
                        out_text = detok(tokens[0][:num_tokens + 1])

                if args.generate_print:
                    out_text = out_text.replace("\n", "\\n")[num_flushed_chars:]
                    line_len = 160
                    while len(out_text) > line_len:
                        print_rank_0(out + " | " + out_text[:line_len])
                        num_flushed_chars += line_len
                        out_text = out_text[line_len:]
                    print_rank_0(out + " | " + out_text, end="\r")
                else:
                    print_rank_0(out)

                prev_num_tokens = num_tokens

        all_times.append(times)
        bmark_times.append(times[-num_tokens_for_average:])

    t = np.array(bmark_times)
    # Drop the worst/best 10%
    t = np.sort(t, axis=1)[:, round(t.shape[1]*0.05):-round(t.shape[1]*0.05)]

    print_rank_0(f'\n{1000*t.mean():.2f} ms, std {1000*t.std():.2f}')
    print_rank_0(f'{1/t.mean()*args.generate_batch:.0f} tok/s')
    print_rank_0(t.shape)

    if args.generate_save and tp_rank == 0:

        results = {
            'model': args.load,
            'infrence_poolsum': args.inference_poolsum,
            'latency_mean': t.mean(),
            'latency_std': t.std(),
            'tput': 1/t.mean() * args.generate_batch,
            'reps': args.generate_reps,
            'batch': args.generate_batch,
            'context_len': args.generate_context_len,
            'generated_len': args.generate_len,
            'num_tokens_for_average': num_tokens_for_average,
            'all_times': all_times,
        }

        fname = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.json")
        Path('bmarks').mkdir(parents=False, exist_ok=True)
        with open(Path('bmarks', fname), 'w') as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'no_load_rng': True,
                                       'no_load_optim': True})

    args = get_args()
    model = get_model(model_provider, wrap_with_ddp=False)

    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    model = model[0].eval()
    generate_tokens_probs_and_return_on_first_stage(model=model)
