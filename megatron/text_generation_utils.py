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
    tokens = context_tokens.contiguous().cuda()
    
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

def pad_batch(batch, pad_id, max_len):
    context_lengths = []
    max_context_length = max([len(tokens) for tokens in batch])
    for tokens in batch:
        context_length = len(tokens)
        if context_length < max_context_length + max_len:
            tokens.extend([pad_id] * (max_context_length + max_len - context_length))
        context_lengths.append(context_length)
    return batch, context_lengths

def tokenize_batch(sentences, max_len, add_BOS):
    args = get_args()
    tokenizer = get_tokenizer()
    if add_BOS:
        context_tokens = [[tokenizer.eod] + tokenizer.tokenize(s) for s in sentences]
    else:
        context_tokens = [tokenizer.tokenize(s) for s in sentences]
    context_tokens, context_lengths = pad_batch(context_tokens,
                                                tokenizer.eod, max_len)
    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    context_length_tensor = torch.cuda.LongTensor(context_lengths)
    return context_tokens_tensor, context_length_tensor 

def send_generate_info(context_tokens_tensor, context_length_tensor, tokens_to_generate, all_probs):
    """
    Needs to be synced up with receive_generate_info
    """
    # Send the sizes of the tensors
    input_info = [context_tokens_tensor.size(0), context_tokens_tensor.size(1), tokens_to_generate, all_probs]
    input_info_tensor = torch.cuda.LongTensor(input_info)
    torch.distributed.broadcast(input_info_tensor, 0)

    # Send variables to all ranks 
    torch.distributed.broadcast(context_length_tensor, 0)
    torch.distributed.broadcast(context_tokens_tensor, 0)

def receive_generate_info():
    """
    Needs to be synced up with send_generate_info
    """
    input_info_tensor = torch.empty(4, dtype=torch.int64, device=torch.cuda.current_device())
    torch.distributed.broadcast(input_info_tensor, 0)
    batch_size = input_info_tensor[0].item()
    seq_len = input_info_tensor[1].item()
    tokens_to_generate = input_info_tensor[2].item()
    all_probs = input_info_tensor[3].item()
    
    context_length_tensor = torch.empty(batch_size, dtype=torch.int64, device=torch.cuda.current_device())
    context_tokens_tensor = torch.empty(batch_size, seq_len, dtype=torch.int64, device=torch.cuda.current_device())
    
    # Send variables to all ranks 
    torch.distributed.broadcast(context_length_tensor, 0)
    torch.distributed.broadcast(context_tokens_tensor, 0)
    
    return context_length_tensor, context_tokens_tensor, tokens_to_generate, all_probs

def synced_generate(model, context_tokens_tensor, context_length_tensor, tokens_to_generate, all_probs, temperature):
    context_length = context_length_tensor.min().item()
    tokens, attention_mask, position_ids = get_batch(context_tokens_tensor)
    batch_token_iterator = sample_sequence_batch(model, context_tokens_tensor,
                                                 context_length_tensor,
                                                 attention_mask, position_ids,
                                                 tokens_to_generate,
                                                 all_probs,
                                                 temperature=temperature)
    for tokens, lengths, output_logits, full_logits in batch_token_iterator:
        context_length += 1
                
    if mpu.is_pipeline_last_stage():
        src = mpu.get_pipeline_model_parallel_last_rank()
        group = mpu.get_embedding_group()
        torch.distributed.broadcast(output_logits, src, group)
        if all_probs:
            src = mpu.get_pipeline_model_parallel_last_rank()
            group = mpu.get_embedding_group()
            torch.distributed.broadcast(full_logits, src, group)

    else:
        if mpu.is_pipeline_first_stage():
            src = mpu.get_pipeline_model_parallel_last_rank()
            group = mpu.get_embedding_group()
            output_logits = torch.empty(tokens.size(0), context_length-1, dtype=torch.float32, device=torch.device("cuda"))
            torch.distributed.broadcast(output_logits, src, group)
            
            if all_probs:
                args = get_args()
                src = mpu.get_pipeline_model_parallel_last_rank()
                group = mpu.get_embedding_group()
                full_logits = torch.empty(tokens.size(0), context_length, args.padded_vocab_size, dtype=torch.float32, device=torch.device("cuda"))
                torch.distributed.broadcast(full_logits, src, group)
    if tokens is not None:
        return tokens[:, :context_length], output_logits, full_logits 

def generate(model, sentences=None, tokens_to_generate=0, all_probs=False, temperature=1.0, add_BOS=False):
    model.eval()
    if torch.distributed.get_rank() == 0:
        context_tokens_tensor, context_length_tensor = tokenize_batch(sentences, tokens_to_generate, add_BOS)
        send_generate_info(context_tokens_tensor, context_length_tensor, tokens_to_generate, all_probs)
    else:
        context_length_tensor, context_tokens_tensor, tokens_to_generate, all_probs = receive_generate_info()

    output = synced_generate(model, context_tokens_tensor, context_length_tensor, tokens_to_generate, all_probs, temperature)
    if output is not None:
        decode_tokens, output_logits, full_logits = output
        
        args = get_args()
        tokenizer = get_tokenizer()
        resp_sentences = []
        resp_sentences_seg = []
        
        decode_tokens = decode_tokens.cpu().numpy().tolist()
        for decode_token in decode_tokens:
            resp_sentences.append(tokenizer.detokenize(decode_token))
            words = []
            for token in decode_token:
                word = tokenizer.tokenizer.decoder[token]
                word = bytearray([tokenizer.tokenizer.byte_decoder[c] for c in word]).decode('utf-8', errors='replace')
                words.append(word)
            resp_sentences_seg.append(words)

        output_logits = output_logits.cpu().numpy().tolist()
        if all_probs:
            full_logits = full_logits.cpu().numpy().tolist()
       
        return resp_sentences, resp_sentences_seg, output_logits, full_logits, decode_tokens 

def generate_samples_eval(model, context, max_gen_length, eos_token_id):
    """
    This function is here to provide an a matching API for a legacy task
    This implementation hasn't been tested yet to make sure it matches
    """
    #assert False, "Implementation untested"
    args = get_args()
    args.eos_id = eos_token_id
    raw_text_len = len(context)
    resp_sentences = generate(model, [context], max_gen_length)
    if resp_sentences:
        return resp_sentences[0][raw_text_len:]

def switch(val1, val2, boolean):
    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2


def forward_step(model, tokens, position_ids, attention_mask, tokentype_ids,
                 set_inference_key_value_memory=False,
                 inference_max_sequence_len=None):

    # Hidden size changes when not using recompute, need to tell p2p_communicate
    # functions the correct size
    args = get_args()
    orig_seq_length = args.seq_length
    args.seq_length = tokens.shape[1]
    args.micro_batch_size = tokens.shape[0]

    input_tensor = recv_forward()

    # Forward pass through the model.
    unwrapped_model = unwrap_model(
        model, (torchDDP, LocalDDP, Float16Module))
    unwrapped_model.set_input_tensor(input_tensor)
    output_tensor = model(
        tokens, position_ids, attention_mask,
        tokentype_ids=tokentype_ids,
        set_inference_key_value_memory=set_inference_key_value_memory,
        inference_max_sequence_len=inference_max_sequence_len)

    send_forward(output_tensor)

    args.seq_length = orig_seq_length

    return output_tensor


def sample_sequence_batch(model, context_tokens, context_lengths,
                          attention_mask, position_ids,
                          tokens_to_generate, all_probs=False, type_ids=None, temperature=None):
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

        batch_size = context_tokens.size(0)
        is_done = torch.zeros([batch_size]).byte().cuda()
        tokens = context_tokens
        output_logits = None
       
        # Generate enough tokens for the longest sequence
        maxlen = tokens_to_generate + context_lengths.max().item() 
       
        if maxlen > args.seq_length:
            maxlen = args.seq_length
        
        lengths = torch.ones([batch_size]).long().cuda() * maxlen

        while context_length < maxlen:
            types2use = None
            if counter == 0:
                # Allocate memory for the entire context.
                set_inference_key_value_memory = True
                tokens2use = tokens[:, :context_length]
                positions2use = position_ids[:, :context_length]
                if type_ids is not None:
                    types2use = type_ids[:, :context_length]
            else:
                # Set this to false so the memory is not reallocated.
                set_inference_key_value_memory = False
                tokens2use = tokens[:, context_length - 1].view(
                    batch_size, -1)
                positions2use = position_ids[:, context_length - 1].view(
                    batch_size, -1)
                if type_ids is not None:
                    types2use = type_ids[:, context_length - 1].view(
                        batch_size, -1)
            
            output = forward_step(
                model, tokens2use,
                positions2use,
                attention_mask,
                set_inference_key_value_memory=set_inference_key_value_memory,
                inference_max_sequence_len=maxlen,
                tokentype_ids=types2use)

            if mpu.is_pipeline_last_stage():
                assert output is not None
                output = output.float()
                logits = output[:, -1].view(batch_size, -1).contiguous()

                if args.greedy:
                    prev = torch.argmax(logits, dim=-1).view(-1)
                else:
                    logits = logits.float()
                    logits /= temperature
                    logits = top_k_logits(logits, top_k=args.top_k,
                                          top_p=args.top_p)
                    log_probs = F.softmax(logits, dim=-1)
                    prev = torch.multinomial(log_probs, num_samples=1).view(-1)
                started = context_lengths <= context_length

                # Clamp the out of vocabulary tokens.
                tokenizer = get_tokenizer()
                prev = torch.clamp(prev, max=tokenizer.vocab_size - 1)

                new_tokens = switch(
                    tokens[:, context_length].view(-1), prev, started)
                tokens[:, context_length] = new_tokens
                
                if output_logits is None:
                    output_context = F.log_softmax(output[:, :context_length, :], 2)
                    indices = torch.unsqueeze(tokens[:, 1:context_length+1],2)
                    output_logits = torch.gather(output_context, 2, indices).squeeze(2)
                    if all_probs:
                        full_logits = output_context
                else:
                    output_context = F.log_softmax(output, 2)
                    indices = torch.unsqueeze(new_tokens,1).unsqueeze(2)
                    new_output_logits = torch.gather(output_context, 2, indices).squeeze(2)
                    
                    # TODO(rprenger) we're copying output_logits every time.  Should pre-allocate
                    output_logits = torch.cat([output_logits, new_output_logits],1)
                    if all_probs:
                        full_logits = torch.cat([full_logits, output_context], 1)
                
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
                if all_probs:
                    yield tokens, lengths, output_logits, full_logits
                else:
                    yield tokens, lengths, output_logits, None

            else:
                if mpu.is_pipeline_first_stage():
                    src = mpu.get_pipeline_model_parallel_last_rank()
                    group = mpu.get_embedding_group()
                    new_tokens = torch.empty_like(tokens[:, context_length])
                    torch.distributed.broadcast(new_tokens, src, group)
                    tokens[:, context_length] = new_tokens
                    yield tokens, None, None, None
                else:
                    yield None, None, None, None

                done = torch.cuda.ByteTensor([0])
                src = mpu.get_pipeline_model_parallel_last_rank()
                group = mpu.get_pipeline_model_parallel_group()
                torch.distributed.broadcast(done, src, group)

            context_length += 1
            counter += 1
            if done:
                break
