# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for generating text."""

import copy
import json
import os
import time

import torch
import torch.nn.functional as F

from megatron import get_args
from megatron import get_tokenizer
from megatron import mpu
from megatron.utils import get_ltor_masks_and_position_ids, unwrap_model
from megatron.p2p_communication import recv_forward, send_forward

# These are needed to unwrap the model, would be nice to put these in megatron.utils if possible?
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module

def get_batch(context_tokens):
    """Generate batch from context tokens."""
    args = get_args()
    tokenizer = get_tokenizer()

    # Move to GPU.
    tokens = context_tokens.view(args.micro_batch_size, -1).contiguous().cuda()
    # Get the attention mask and postition ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, attention_mask, position_ids


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ This function has been mostly taken from huggingface conversational
     ai code at
         https://medium.com/huggingface/how-to-build-a-state-of-the-art-
              conversational-ai-with-transfer-learning-2d818ac26313 """

    if top_k > 0:
        # Remove all tokens with a probability less than the
        # last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Cconvert to 1D
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1),
                                        dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] \
            = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for i in range(sorted_indices.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value

    return logits


def generate_samples_input_from_file(model):

    args = get_args()
    tokenizer = get_tokenizer()

    # Read the sample file and open the output file.
    assert args.sample_input_file is not None, \
        'sample input file is not provided.'
    if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        fname = open(args.sample_input_file, "r")
        all_raw_text = fname.readlines()
        input_count = len(all_raw_text)
        input_pos = 0
        if args.sample_output_file is None:
            sample_output_file = args.sample_input_file + ".out"
            print('`sample-output-file` not specified, setting '
                  'it to {}'.format(sample_output_file))
        else:
            sample_output_file = args.sample_output_file
        fname_out = open(sample_output_file, "w+")

    context_count = 0
    model.eval()
    with torch.no_grad():
        while True:
            terminate_runs = 0
            raw_text_len = 0

            if mpu.is_pipeline_first_stage() \
               and mpu.get_tensor_model_parallel_rank() == 0:
                raw_text = all_raw_text[input_pos]
                input_pos += 1
                if input_pos == input_count:
                    raw_text = "stop"
                raw_text_len = len(raw_text)

                if "stop" in raw_text:
                    terminate_runs = 1
                else:
                    context_tokens = tokenizer.tokenize(raw_text)
                    context_length = len(context_tokens)

                    if context_length >= (args.seq_length // 2):
                        print("\nContext length", context_length,
                              "\nPlease give smaller context (half of the "
                              "sequence length)!", flush=True)
                        continue
            else:
                context_tokens = tokenizer.tokenize("EMPTY TEXT")
                context_length = 0

            input_info = [terminate_runs, raw_text_len, context_length]
            input_info_tensor = torch.cuda.LongTensor(input_info)
            torch.distributed.all_reduce(input_info_tensor,
                                         group=mpu.get_model_parallel_group())
            terminate_runs = input_info_tensor[0].item()
            raw_text_len = input_info_tensor[1].item()
            context_length = input_info_tensor[2].item()

            if terminate_runs == 1:
                return

            # For pipeline parallel we send context tokens to other stages
            # so they get the lengths correct
            if mpu.get_tensor_model_parallel_rank() == 0 \
               and args.pipeline_model_parallel_size > 1:
                if mpu.is_pipeline_first_stage():
                    src = mpu.get_pipeline_model_parallel_first_rank()
                    group = mpu.get_pipeline_model_parallel_group()
                    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
                    torch.distributed.broadcast(context_tokens_tensor, src, group)
                else:
                    src = mpu.get_pipeline_model_parallel_first_rank()
                    group = mpu.get_pipeline_model_parallel_group()
                    context_tokens_tensor = torch.empty(context_length,
                                                        dtype=torch.int64,
                                                        device=torch.device("cuda"))
                    torch.distributed.broadcast(context_tokens_tensor, src, group)
                    context_tokens = context_tokens_tensor.cpu().numpy().tolist()

            token_stream = get_token_stream(model, [context_tokens])
            for _, decode_tokens in enumerate(token_stream):
                pass

            if mpu.get_tensor_model_parallel_rank() == 0:
                if mpu.is_pipeline_first_stage():
                    os.system('clear')
                    print("\nContext:", raw_text, flush=True)

                    fname_out.write("\nContext:")
                    fname_out.write(raw_text)

                    decode_tokens, _ = decode_tokens
                    decode_tokens = decode_tokens[0].cpu().numpy().tolist()
                    trim_decode_tokens = tokenizer.detokenize(
                        decode_tokens)[raw_text_len:]
                    print("\nMegatron-LM:", trim_decode_tokens, flush=True)

                    fname_out.write("\n\nMegatron-LM:")
                    fname_out.write(trim_decode_tokens)
                    fname_out.write("\n")

            raw_text = None
            context_count += 1

# We added this function to support the tasks evaluation such as squad
# and drop in the https://github.com/EleutherAI/lm-evaluation-harness 
# codebase. The lm-evaluation-harness code can now call this function
# similar to their current generate function call used for gpt style models.
def generate_samples_eval(model, context, max_gen_length, eos_token_id):
    # Generate samples for lm evaluation
    # NEED TO THINK ABOUT eos token

    args = get_args()
    tokenizer = get_tokenizer()

    raw_text_len = len(context)
    model.eval()

    context_tokens = tokenizer.tokenize(context)
    args.out_seq_length = max_gen_length + len(context_tokens)
    args.eos_id = eos_token_id

    with torch.no_grad():
        token_stream = get_token_stream(model, [context_tokens])
        for counter, decode_tokens in enumerate(token_stream):
            if counter == args.out_seq_length:
                break

    decode_tokens, _ = decode_tokens
    decode_tokens = decode_tokens[0].cpu().numpy().tolist()
    trim_decode_tokens = tokenizer.detokenize(
        decode_tokens)[raw_text_len:]
 
    return trim_decode_tokens


def generate_samples_interactive(model, print_frequency=24):

    args = get_args()
    tokenizer = get_tokenizer()

    context_count = 0
    model.eval()
    with torch.no_grad():
        while True:
            terminate_runs = 0
            raw_text_len = 0

            if mpu.is_pipeline_first_stage() \
               and mpu.get_tensor_model_parallel_rank() == 0:
                os.system('clear')
                raw_text = input("\nContext prompt (stop to exit) >>> ")
                while not raw_text:
                    print('Prompt should not be empty!')
                    raw_text = input("\nContext prompt (stop to exit) >>> ")
                raw_text_len = len(raw_text)

                if "stop" in raw_text:
                    terminate_runs = 1
                else:
                    context_tokens = tokenizer.tokenize(raw_text)
                    context_length = len(context_tokens)

                    if context_length >= (args.seq_length // 2):
                        print("\nContext length", context_length,
                              "\nPlease give smaller context (half of the "
                              "sequence length)!", flush=True)
                        continue
            else:
                context_tokens = tokenizer.tokenize("EMPTY TEXT")
                context_length = 0

            input_info = [terminate_runs, raw_text_len, context_length]
            input_info_tensor = torch.cuda.LongTensor(input_info)
            torch.distributed.all_reduce(input_info_tensor,
                                         group=mpu.get_model_parallel_group())
            terminate_runs = input_info_tensor[0].item()
            raw_text_len = input_info_tensor[1].item()
            context_length = input_info_tensor[2].item()

            if terminate_runs == 1:
                return

            # For pipeline parallel we send context tokens to other stages
            # so they get the lengths correct
            if mpu.get_tensor_model_parallel_rank() == 0 \
               and args.pipeline_model_parallel_size > 1:
                if mpu.is_pipeline_first_stage():
                    src = mpu.get_pipeline_model_parallel_first_rank()
                    group = mpu.get_pipeline_model_parallel_group()
                    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
                    torch.distributed.broadcast(context_tokens_tensor, src, group)
                else:
                    src = mpu.get_pipeline_model_parallel_first_rank()
                    group = mpu.get_pipeline_model_parallel_group()
                    context_tokens_tensor = torch.empty(context_length,
                                                        dtype=torch.int64,
                                                        device=torch.device("cuda"))
                    torch.distributed.broadcast(context_tokens_tensor, src, group)
                    context_tokens = context_tokens_tensor.cpu().numpy().tolist()

            token_stream = get_token_stream(model, [context_tokens])

            for counter, decode_tokens in enumerate(token_stream):
                if counter % print_frequency != 0 \
                   or mpu.get_tensor_model_parallel_rank() != 0 \
                   or not mpu.is_pipeline_first_stage():
                    continue

                os.system('clear')
                print("\nContext:", raw_text, flush=True)

                decode_tokens, _ = decode_tokens
                decode_tokens = decode_tokens[0].cpu().numpy().tolist()
                trim_decode_tokens = tokenizer.detokenize(
                    decode_tokens)[raw_text_len:]
                print("\nMegatron-LM:", trim_decode_tokens, flush=True)

            if mpu.is_pipeline_first_stage() \
               and mpu.get_tensor_model_parallel_rank() == 0:
                os.system('clear')
                print("\nContext:", raw_text, flush=True)

                if not isinstance(decode_tokens, list):
                    decode_tokens, _ = decode_tokens
                    decode_tokens = decode_tokens[0].cpu().numpy().tolist()
                trim_decode_tokens = tokenizer.detokenize(
                    decode_tokens)[raw_text_len:]
                print("\nMegatron-LM:", trim_decode_tokens, flush=True)

                input("\nPress Enter to continue >>>")

            raw_text = None
            context_count += 1



def generate_samples_unconditional(model):

    args = get_args()
    tokenizer = get_tokenizer()

    num_samples = args.num_samples
    context_tokens = [[tokenizer.eod]
                      for _ in range(args.micro_batch_size)]
    ctr = 0
    while True:
        start_time = time.time()
        for token_stream in get_token_stream(model,
                                             copy.deepcopy(context_tokens)):
            pass
        if mpu.is_pipeline_last_stage() and \
           mpu.get_tensor_model_parallel_rank() == 0:
            if ctr % args.log_interval == 0:
                print('Avg s/batch:',
                      (time.time() - start_time) / min(args.log_interval, ctr + 1))
                start_time = time.time()
            length = len(token_stream)
            token_batch = token_stream[0].cpu().numpy().tolist()
            length_batch = token_stream[1].cpu().numpy().tolist()
            assert len(length_batch) == args.micro_batch_size
            for tokens, length in zip(token_batch, length_batch):
                tokens = tokens[1:length - 1]
                text = tokenizer.detokenize(tokens)
                is_finished = length < args.seq_length - 1
                datum = {'text': text, 'length': length - 1, 'finished': is_finished}
                yield datum
                ctr += 1
                if ctr >= num_samples:
                    break
        else:
            for _ in range(args.micro_batch_size):
                yield None
                ctr += 1
                if ctr >= num_samples:
                    break
        if ctr >= num_samples:
            break


def generate_and_write_samples_unconditional(model):

    args = get_args()
    assert args.genfile is not None
    with open(args.genfile, 'w') as f:
        for datum in generate_samples_unconditional(model):
            if mpu.is_pipeline_last_stage() and \
               mpu.get_tensor_model_parallel_rank() == 0:
                f.write(json.dumps(datum) + '\n')


def pad_batch(batch, pad_id, args):

    context_lengths = []
    for tokens in batch:
        context_length = len(tokens)
        if context_length < args.seq_length:
            tokens.extend([pad_id] * (args.seq_length - context_length))
        context_lengths.append(context_length)
    return batch, context_lengths


def get_token_stream(model, context_tokens):

    args = get_args()
    tokenizer = get_tokenizer()

    context_tokens, context_lengths = pad_batch(context_tokens,
                                                tokenizer.eod, args)

    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    context_length_tensor = torch.cuda.LongTensor(context_lengths)

    torch.distributed.broadcast(context_length_tensor,
                                mpu.get_tensor_model_parallel_src_rank(),
                                group=mpu.get_tensor_model_parallel_group())
    torch.distributed.broadcast(context_tokens_tensor,
                                mpu.get_tensor_model_parallel_src_rank(),
                                group=mpu.get_tensor_model_parallel_group())

    context_length = context_length_tensor.min().item()
    tokens, attention_mask, position_ids = get_batch(context_tokens_tensor)

    batch_token_iterator = sample_sequence_batch(model, context_tokens_tensor,
                                                 context_length_tensor,
                                                 attention_mask, position_ids)
    for tokens, lengths in batch_token_iterator:
        context_length += 1
        if tokens is not None:
            yield tokens[:, :context_length], lengths
        else:
            yield None, None


def switch(val1, val2, boolean):

    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2


def forward_step(model, tokens, position_ids, attention_mask, tokentype_ids,
                 layer_past=None, get_key_value=None,
                 forward_method_parallel_output=None):

    # Hidden size changes when not using recompute, need to tell p2p_communicate
    # functions the correct size
    args = get_args()
    orig_seq_length = args.seq_length
    args.seq_length = tokens.shape[1]

    input_tensor = recv_forward()

    # Forward pass through the model.
    unwrapped_model = unwrap_model(
        model, (torchDDP, LocalDDP, Float16Module))
    unwrapped_model.set_input_tensor(input_tensor)
    output_tensor = model(tokens, position_ids, attention_mask,
                          tokentype_ids=tokentype_ids,
                          layer_past=layer_past,
                          get_key_value=get_key_value,
                          forward_method_parallel_output=forward_method_parallel_output)

    if get_key_value:
        output_tensor, layer_past = output_tensor

    send_forward(output_tensor)

    args.seq_length = orig_seq_length
    if get_key_value:
        return output_tensor, layer_past
    return output_tensor


def sample_sequence_batch(model, context_tokens, context_lengths,
                          attention_mask, position_ids,
                          maxlen=None, type_ids=None):

    args = get_args()
    tokenizer = get_tokenizer()

    model.eval()
    with torch.no_grad():
        context_length = context_lengths.min().item()

        # added eos_id to support the function generate_samples_eval that passes
        # eos_id as an argument and needs termination when that id id found.
        if hasattr(args, 'eos_id'):
            eos_id = args.eos_id
        else:
            eos_id = tokenizer.eod

        counter = 0
        org_context_length = context_length

        layer_past = None
        batch_size = context_tokens.size(0)
        is_done = torch.zeros([batch_size]).byte().cuda()
        tokens = context_tokens
        if maxlen is None:
            maxlen = args.seq_length - 1
            if maxlen > (org_context_length + args.out_seq_length):
                maxlen = org_context_length + args.out_seq_length

        lengths = torch.ones([batch_size]).long().cuda() * maxlen

        while context_length <= (maxlen):
            if args.recompute:
                output = forward_step(model, tokens,
                                      position_ids,
                                      attention_mask,
                                      tokentype_ids=type_ids,
                                      forward_method_parallel_output=False)
                if mpu.is_pipeline_last_stage():
                    assert output is not None
                    logits = output[:, context_length - 1, :]
            else:
                types2use = None
                if counter == 0:
                    tokens2use = tokens[:, :context_length]
                    positions2use = position_ids[:, :context_length]
                    if type_ids is not None:
                        types2use = type_ids[:, :context_length]
                else:
                    tokens2use = tokens[:, context_length - 1].view(
                        batch_size, -1)
                    positions2use = position_ids[:, context_length - 1].view(
                        batch_size, -1)
                    if type_ids is not None:
                        types2use = type_ids[:, context_length - 1].view(
                            batch_size, -1)
                output, layer_past = forward_step(model, tokens2use,
                                                  positions2use,
                                                  attention_mask,
                                                  layer_past=layer_past,
                                                  get_key_value=True,
                                                  tokentype_ids=types2use,
                                                  forward_method_parallel_output=False)
                if mpu.is_pipeline_last_stage():
                    assert output is not None
                    logits = output[:, -1].view(batch_size, -1).contiguous()

            if mpu.is_pipeline_last_stage():
                if args.greedy:
                    prev = torch.argmax(logits, dim=-1).view(-1)
                else:
                    logits = logits.float()
                    logits /= args.temperature
                    logits = top_k_logits(logits, top_k=args.top_k,
                                          top_p=args.top_p)
                    log_probs = F.softmax(logits, dim=-1)
                    prev = torch.multinomial(log_probs, num_samples=1).view(-1)

                started = context_lengths <= context_length

                new_tokens = switch(
                    tokens[:, context_length].view(-1), prev, started)
                tokens[:, context_length] = new_tokens
                src = mpu.get_pipeline_model_parallel_last_rank()
                group = mpu.get_embedding_group()
                torch.distributed.broadcast(new_tokens, src, group)

                done_token = (prev == eos_id).byte() & started.byte()
                just_finished = (done_token & ~is_done).bool()
                lengths[just_finished.view(-1)] = context_length
                is_done = is_done | done_token

                done = torch.all(is_done)
                src = mpu.get_pipeline_model_parallel_last_rank()
                group = mpu.get_pipeline_model_parallel_group()
                torch.distributed.broadcast(done, src, group)
                yield tokens, lengths

            else:
                if mpu.is_pipeline_first_stage():
                    src = mpu.get_pipeline_model_parallel_last_rank()
                    group = mpu.get_embedding_group()
                    new_tokens = torch.empty_like(tokens[:, context_length])
                    torch.distributed.broadcast(new_tokens, src, group)
                    tokens[:, context_length] = new_tokens
                    yield tokens, None
                else:
                    yield None, None

                done = torch.cuda.ByteTensor([0])
                src = mpu.get_pipeline_model_parallel_last_rank()
                group = mpu.get_pipeline_model_parallel_group()
                torch.distributed.broadcast(done, src, group)

            context_length += 1
            counter += 1
            if done:
                break
