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

"""Generation utilities."""
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from megatron import get_args, get_tokenizer
from megatron import get_retro_args
from megatron.core import mpu
from megatron.utils import get_ltor_masks_and_position_ids, unwrap_model
from megatron.text_generation.forward_step import ForwardStep, InferenceParams
from megatron.text_generation.communication import (
    copy_from_last_to_first_pipeline_stage,
    broadcast_from_last_pipeline_stage,
    broadcast_from_last_to_first_pipeline_stage, send_to_next_pipeline_rank, broadcast_int_list, broadcast_tensor)
from megatron.text_generation.generation import _build_attention_mask_and_position_ids
from megatron.text_generation.sampling import sample
from megatron.text_generation.beam_utils import BeamHypotheses
from megatron.model import Float16Module


def _forward_step_helper(model, tokens, position_ids, attention_mask,
                         inference_params, recv_buffer=None):
    """Single forward step. Update the allocate memory flag so
    only the first time the memory is allocated."""
    # Forward pass through the model.
    model.set_input_tensor(recv_buffer)
    output_tensor = model(tokens, position_ids, attention_mask,
                          inference_params=None)

    # Send output to the next stage.
    send_to_next_pipeline_rank(output_tensor)

    return output_tensor


def _no_pipelining_forward_step(model, tokens, position_ids, attention_mask,
                                inference_params, recv_buffer=None):
    """If recv_buffer is none, we will allocate one on the fly."""
    # Run a simple forward pass.
    output_tensor = _forward_step_helper(model, tokens, position_ids,
                                         attention_mask, None,
                                         recv_buffer=None)
    logits = None
    if mpu.is_pipeline_last_stage():
        logits = output_tensor

    return logits


def _with_pipelining_forward_step(model, tokens, position_ids, attention_mask,
                                  inference_params, micro_batch_size):
    """No interleaving is supported."""
    sequence_length = tokens.size(1)
    batch_size = tokens.size(0)

    # Divide the batch dimension into micro batches.
    num_micro_batches, last_chunk = divmod(batch_size,
                                           micro_batch_size)
    if last_chunk > 0:
        num_micro_batches += 1

    # Preallocate memory for output logits.
    logits = None
    if mpu.is_pipeline_last_stage():
        args = get_args()
        logits = torch.empty(
            (batch_size, sequence_length, args.padded_vocab_size),
            dtype=torch.float32, device=torch.cuda.current_device())

    for micro_batch_index in range(num_micro_batches):
        # Slice among the batch dimenion.
        start = micro_batch_index * micro_batch_size
        end = min(start + micro_batch_size, batch_size)
        this_micro_batch_size = end - start
        tokens2use = tokens[start:end, ...]
        position_ids2use = position_ids[start:end, ...]

        # Run a simple forward pass.
        if this_micro_batch_size != micro_batch_size:
            recv_buffer = None
        output = _forward_step_helper(model, tokens2use, position_ids2use,
                                      attention_mask, None,
                                      recv_buffer=None)

        # Copy logits.
        if mpu.is_pipeline_last_stage():
            logits[start:end, ...] = output

    return logits

class ForwardStep:
    """Forward step function with all the communications.
    We use a class here to hide the inference parameters
    from the outside caller."""

    def __init__(self, model, max_batch_size, max_sequence_len):
        """Set values so we don't need to do it multiple times."""
        # Make sure model is in eval mode.
        assert not isinstance(model, Iterable), \
            'interleaving schedule is not supported for inference'
        model.eval()
        self.model = model
        # Initialize inference parameters.
        self.inference_params = InferenceParams(max_batch_size,
                                                max_sequence_len)
        # Pipelining arguments.
        args = get_args()
        self.pipeline_size_larger_than_one = (
            args.pipeline_model_parallel_size > 1)
        # Threshold of pipelining.
        self.pipelining_batch_x_seqlen = \
            args.inference_batch_times_seqlen_threshold


    def __call__(self, tokens, position_ids, attention_mask):
        """Invocation of the forward methods. Note that self.inference_params
        is being modified by the forward step."""
        # Pipelining case.
        if self.pipeline_size_larger_than_one:
            current_batch_x_seqlen = tokens.size(0) * tokens.size(1)
            if current_batch_x_seqlen >= self.pipelining_batch_x_seqlen:
                micro_batch_size = \
                    max(1, self.pipelining_batch_x_seqlen // tokens.size(1))
                return _with_pipelining_forward_step(self.model,
                                                     tokens,
                                                     position_ids,
                                                     attention_mask,
                                                     self.inference_params,
                                                     micro_batch_size)

        return _no_pipelining_forward_step(self.model,
                                           tokens,
                                           position_ids,
                                           attention_mask,
                                           self.inference_params)


def get_tokens_from_tensors(tokens):
    # split tokens
    args = get_args()
    tokenizer = get_tokenizer()
    tokens_list = []
    for token in tokens:
        token_len = len(token)
        remainder = len(token) % args.m
        token_list = []
        if remainder > 0:
            token_list.append(tokenizer.detokenize(token[:remainder].cpu().numpy().tolist()))
        for i in range(remainder, token_len, args.m):
            token_list.append(tokenizer.detokenize(token[i:i+args.m].cpu().numpy().tolist()))
        tokens_list.append(token_list)
    return tokens_list



def get_features_from_tokens(tokens):
    args = get_args()
    bert = args.bert
    embeddings = bert(tokens)
    embeddings = np.array(embeddings)
    print(embeddings.shape)
    print(embeddings.dtype)
    return embeddings

def query_neighbors_from_features(features):
    args = get_args()
    k = args.retro_num_neighbors
    retriever = args.retriever
    shape = features.shape
    flattened_features = features.reshape((-1, shape[-1]))
    D, I = retriever.search(flattened_features, k)  # [-1, k]
    I = I.reshape(shape[0], shape[1], k)
    print(I.shape)
    return I

def get_tokens_from_neighbors(neighbors):
    args = get_args()
    retro_args = get_retro_args()

    database = args.database
    shape = neighbors.shape
    flatten_neighbors = np.reshape(neighbors, (-1, 1))
    continuations = (flatten_neighbors + 1) % len(database['chunks'])
    neighbors = np.hstack((flatten_neighbors, continuations)).flatten()

    neighbor_tokens = np.array([database['chunks'][neighbor] for neighbor in neighbors], dtype='int64')
    neighbor_tokens = neighbor_tokens.reshape((shape[0], shape[1], shape[2], retro_args.retro_gpt_retrieved_length))
    # print(neighbor_tokens)
    print(neighbor_tokens.shape)
    tokenizer = get_tokenizer()
    print(tokenizer.detokenize(neighbor_tokens[0][0][0]))
    return neighbor_tokens

def retro_generate_tokens_probs_and_return_on_first_stage(
        model, tokens, lengths, neighbours_array=None,
        return_output_log_probs=False,
        top_k=0, top_p=0.0,
        temperature=1.0,
        use_eod_token_for_early_termination=True,
        stop_on_double_eol=False,
        stop_on_eol=False,
        logits_mask = None):
    """Main token generation function.
    Arguments:
        model: no interleaving is supported.
        tokens: prompt tokens extended to be of size [b, max-sequence-length]
        lengths: original prompt length, size: [b]
        neighbours_array: neighbours array of size [b, l, k, r]
        return_output_log_probs: flag to calculate the log probability of
            the generated tokens. Note that the log probability is the one
            from the original logit.
        top_k, top_p: top-k and top-p sampling parameters.
            Note that top-k = 1 is gready. Also, these paramters are
            exclusive meaning that:
                if top-k > 0 then we expect top-p=0.
                if top-p > 0 then we check for top-k=0.
        temperature: sampling temperature.
        use_eod_token_for_early_termination: if True, do early termination if
            all the sequences have reached this token.
    Note: Outside of model, other parameters only need to be available on
          rank 0.
    Outputs: Note that is size is adjusted to a lower value than
             max-sequence-length if generation is terminated early.
        tokens: prompt and generated tokens. size: [b, :]
        generated_sequence_lengths: total length (including prompt) of
            the generated sequence. size: [b]
        output_log_probs: log probability of the selected tokens. size: [b, s]
    """

    args = get_args()
    retro_args = get_retro_args()

    tokenizer = get_tokenizer()

    batch_size = tokens.size(0)
    min_prompt_length = lengths.min().item()
    max_sequence_length = tokens.size(1)
    print("max_sequence_length", max_sequence_length)
    print("min_prompt_length", min_prompt_length)
    max_sequence_length = min(max_sequence_length, args.max_position_embeddings)

    # If the context is too big, this happens
    if min_prompt_length >= max_sequence_length:
        raise ValueError("context length + tokens_to_generate too large")

    # forward step.
    # forward_step = ForwardStep(model, batch_size, max_sequence_length)
    # inference_params = InferenceParams(batch_size, max_sequence_length)
    # from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
    # from megatron.model import DistributedDataParallel as LocalDDP
    unwrapped_model = unwrap_model(
        model)
    unwrapped_model.language_model.seq_length = max_sequence_length

    # Added termination_id to support the case that we want to terminate the
    # generation once that id is generated.
    if hasattr(args, 'eos_id'):
        termination_id = args.eos_id
    else:
        termination_id = tokenizer.eod

    # ===================
    # Pre-allocate memory
    # ===================

    # Log probability of the sequence (prompt + generated tokens).
    output_log_probs = None
    output_log_probs_size = (batch_size, max_sequence_length - 1)
    # Lengths of generated seuquence including including prompts.
    generated_sequence_lengths = None
    if mpu.is_pipeline_last_stage():
        if return_output_log_probs:
            output_log_probs = torch.empty(output_log_probs_size,
                                           dtype=torch.float32,
                                           device=torch.cuda.current_device())
        generated_sequence_lengths = torch.ones(
                batch_size, dtype=torch.int64,
                device=torch.cuda.current_device()) * max_sequence_length

    # Whether we have reached a termination id.
    is_generation_done = torch.zeros(batch_size, dtype=torch.uint8,
                                     device=torch.cuda.current_device())

    # =============
    # Run infernece
    # =============

    with torch.no_grad():
        attention_mask, position_ids = _build_attention_mask_and_position_ids(
            tokens)
        print(min_prompt_length, max_sequence_length)
        for context_length in range(min_prompt_length, max_sequence_length):
            prev_context_length = 0
            sizes_list = None
            neighbor_tokens_cuda_long_tensor = None

            # get the chunks for retrieval
            if torch.distributed.get_rank() == 0:
                if getattr(args, 'task', None) is None:
                    tokens2query = get_tokens_from_tensors(tokens[:, prev_context_length:context_length])
                    print(tokens2query)
                    features = get_features_from_tokens(tokens2query)
                    neighbors = query_neighbors_from_features(features)
                    neighbor_tokens = get_tokens_from_neighbors(neighbors)
                else:
                    neighbor_tokens = neighbours_array
                neighbor_tokens_cuda_long_tensor = torch.cuda.LongTensor(neighbor_tokens.reshape((-1, retro_args.retro_gpt_retrieved_length)))
                sizes_list = [neighbor_tokens_cuda_long_tensor.size(0),  # Batch size
                          neighbor_tokens_cuda_long_tensor.size(1)]  # Sequence lenght
            sizes_tensor = broadcast_int_list(2, int_list=sizes_list)
            sizes = sizes_tensor.tolist()
            neighbor_tokens_cuda_long_tensor = broadcast_tensor(
                sizes, torch.int64, tensor=neighbor_tokens_cuda_long_tensor)

            _, _, neighbor_position_ids = get_ltor_masks_and_position_ids(
                neighbor_tokens_cuda_long_tensor,
                tokenizer.eod,
                args.reset_position_ids,
                args.reset_attention_mask,
                args.eod_mask_loss)
            neighbor_attention_mask = None

            # Pick the slice that we need to pass through the network.
            tokens2use = tokens[:, prev_context_length:4096]
            positions2use = position_ids[:, prev_context_length:4096]
            attention_mask2use = attention_mask[
                ..., prev_context_length:4096, :4096]

            # logits will be meanigful only in the last pipeline stage.
            # logits = forward_step(tokens2use, positions2use, attention_mask2use)


            logits = model(tokens2use, positions2use, attention_mask2use, retriever_input_ids=neighbor_tokens_cuda_long_tensor,
                                  retriever_position_ids=neighbor_position_ids, retriever_attn_mask=neighbor_attention_mask,
                           )

            if mpu.is_pipeline_last_stage():
                # Always the last stage should have an output.
                assert logits is not None

                # Sample.
                last_token_logits = logits[:, context_length-1, :]
                # last_token_logits = logits[:, -1, :]

                # word banning
                if logits_mask is not None:
                    last_token_logits[:, logits_mask] = float('-Inf')

                new_sample = sample(last_token_logits,
                                    top_k=top_k,
                                    top_p=top_p,
                                    temperature=temperature,
                                    vocab_size=tokenizer.vocab_size)

                # If a prompt length is smaller or equal th current context
                # length, it means we have started generating tokens
                started = lengths <= context_length
                # Update the tokens.
                tokens[started, context_length] = new_sample[started]

                # Calculate the log probabilities.
                if return_output_log_probs:
                    log_probs = F.log_softmax(logits, dim=2)
                    if return_output_log_probs:
                        # Pick the tokens that we need to get the log
                        # probabilities for. Note that next input token is
                        # the token which we selected in the current logits,
                        # so shift by 1.
                        indices = torch.unsqueeze(
                            tokens[
                                :,
                                (prev_context_length + 1):(context_length + 1)],
                            2)
                        output_log_probs[:,
                                         prev_context_length:context_length] = \
                            torch.gather(log_probs, 2, indices).squeeze(2)

            # Update the tokens on the first stage so the next input to
            # the network is correct.
            copy_from_last_to_first_pipeline_stage(batch_size, torch.int64,
                                                   tokens[:, context_length])

            # Update the context length for the next token generation.
            prev_context_length = context_length

            # Check if all the sequences have hit the termination_id.
            done = None
            if mpu.is_pipeline_last_stage():
                # TODO(rprenger) These stopping methods are tokenizer dependent
                # instead tokenization should be in the inference loop so stop sequences can be used
                if stop_on_double_eol:
                    hit_double_eol = (new_sample == 628).byte() & started.byte()
                    hit_two_eols = (new_sample == 198).byte() & (tokens[:, context_length-1] == 198).byte() & started.byte()
                    done_token = hit_double_eol | hit_two_eols
                elif stop_on_eol:
                    hit_double_eol = (new_sample == 628).byte() & started.byte()
                    hit_eol = (new_sample == 198).byte() & started.byte()
                    done_token = hit_double_eol | hit_eol
                elif context_length > min_prompt_length + 64:  # previous retrov1 limitations
                    done_token = 1
                else:
                    done_token = (new_sample == termination_id).byte() & \
                        started.byte()

                just_finished = (done_token & ~is_generation_done).bool()
                generated_sequence_lengths[just_finished.view(-1)] = \
                    context_length + 1
                is_generation_done = is_generation_done | done_token
                done = torch.all(is_generation_done)
            done = broadcast_from_last_pipeline_stage(1, torch.uint8,
                                                      tensor=done)
            if use_eod_token_for_early_termination and done:
                break

    # ===================================================
    # Update the length of based on max generated length.
    # ===================================================

    tokens = tokens[:, :(context_length + 1)]
    if mpu.is_pipeline_last_stage():
        if return_output_log_probs:
            output_log_probs = output_log_probs[:, :context_length]

    # ======================================
    # Broadcast to the first pipeline stage.
    # ======================================

    generated_sequence_lengths = broadcast_from_last_to_first_pipeline_stage(
        batch_size, torch.int64, generated_sequence_lengths)
    if return_output_log_probs:
        output_log_probs_size = (batch_size, context_length)
        output_log_probs = broadcast_from_last_to_first_pipeline_stage(
            output_log_probs_size, torch.float32, output_log_probs)

    return tokens, generated_sequence_lengths, output_log_probs


def retro_beam_search_and_return_on_first_stage(model, neighbours_array, tokens, lengths, beam_size, stop_token, num_return_gen, length_penalty):
    args = get_args()
    retro_args = get_retro_args()
    tokenizer = get_tokenizer()

    batch_size = tokens.size(0)
    assert(batch_size == 1)
    prompt_length = lengths.item()
    final_sequence_length = tokens.size(1)
    final_sequence_length = min(final_sequence_length, args.max_position_embeddings)
    
    # If the context is too big, this happens
    if prompt_length >= final_sequence_length:
        raise ValueError("context length + tokens_to_generate too large")

    # forward step.
    forward_step = ForwardStep(model, beam_size, final_sequence_length)

    beam_hyp = BeamHypotheses(beam_size, length_penalty)
    best_batches = None
    done = torch.zeros(1, dtype=torch.uint8, device=torch.cuda.current_device())
    scores = torch.zeros(beam_size,
                         dtype=torch.float32,
                         device=torch.cuda.current_device()).unsqueeze(1)
    scores_size_tensor, tokens_size_tensor = None, None
    # =============
    # Run infernece
    # =============
    with torch.no_grad():
        tokens = tokens.repeat(beam_size, 1)
        attention_mask, position_ids = _build_attention_mask_and_position_ids(tokens)
        prev_context_length = 0
        print(prompt_length, final_sequence_length)
        for context_length in range(prompt_length, final_sequence_length):
            prev_context_length = 0
            sizes_list = None
            neighbor_tokens_cuda_long_tensor = None

            # get the chunks for retrieval
            if torch.distributed.get_rank() == 0:
                if getattr(args, 'task', None) is None:
                    tokens2query = get_tokens_from_tensors(tokens[:, prev_context_length:context_length])
                    print(tokens2query)
                    features = get_features_from_tokens(tokens2query)
                    neighbors = query_neighbors_from_features(features)
                    neighbor_tokens = get_tokens_from_neighbors(neighbors)
                else:
                    neighbor_tokens = neighbours_array
                neighbor_tokens_cuda_long_tensor = torch.cuda.LongTensor(neighbor_tokens.reshape((-1, retro_args.retro_gpt_retrieved_length)))
                sizes_list = [neighbor_tokens_cuda_long_tensor.size(0),  # Batch size
                          neighbor_tokens_cuda_long_tensor.size(1)]  # Sequence lenght
            sizes_tensor = broadcast_int_list(2, int_list=sizes_list)
            sizes = sizes_tensor.tolist()
            neighbor_tokens_cuda_long_tensor = broadcast_tensor(
                sizes, torch.int64, tensor=neighbor_tokens_cuda_long_tensor)

            _, _, neighbor_position_ids = get_ltor_masks_and_position_ids(
                neighbor_tokens_cuda_long_tensor,
                tokenizer.eod,
                args.reset_position_ids,
                args.reset_attention_mask,
                args.eod_mask_loss)
            neighbor_attention_mask = None

            # Pick the slice that we need to pass through the network.
            tokens2use = tokens[:, prev_context_length:2048]
            positions2use = position_ids[:, prev_context_length:2048]
            attention_mask2use = attention_mask[
                ..., prev_context_length:2048, :2048]

            # logits will be meanigful only in the last pipeline stage.
            logits = model(tokens2use, positions2use, attention_mask2use, ret_int_ids=neighbor_tokens_cuda_long_tensor,
                                  ret_position_ids=neighbor_position_ids, ret_attn_mask=neighbor_attention_mask)

            if mpu.is_pipeline_last_stage():
                vocab_size = logits.size(2)
                log_probs = F.log_softmax(logits, dim=2)
                new_scores = log_probs[:, context_length-1, :] + scores

                if context_length == prompt_length:  # if this is the first one
                    sorted_scores, indices = torch.sort(new_scores[0,:], descending=True)
                else:
                    sorted_scores, indices = torch.sort(new_scores.view(-1), descending=True)

                best_beam_ids = torch.div(indices[: 2 * beam_size], vocab_size).trunc().long()
                best_words = indices[:2 * beam_size] % vocab_size
                best_scores = sorted_scores[: 2 * beam_size]

                next_beams = []
                for beam_token_rank, (token_id, beam_score, beam_id) in enumerate(
                    zip(best_words, best_scores, best_beam_ids)
                ):
                    if token_id.item() == stop_token:
                        # if beam_token does not belong to top num_beams tokens, it should not be added
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= beam_size
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        beam_hyp.add(
                            tokens[beam_id].clone(),
                            beam_score,
                            context_length + 1 - prompt_length
                        )
                    else:
                        # add next predicted token since it is not eos_token
                        next_beams.append((token_id, beam_score, beam_id))

                    if len(next_beams) == beam_size:
                        break

                if beam_hyp.is_done(best_scores.max().item(), context_length + 1 - prompt_length):
                    done = torch.ones(1, dtype=torch.uint8, device=torch.cuda.current_device())
            
                best_batches = tokens.new([item[2] for item in next_beams])
                tokens = tokens[best_batches,:]
                tokens[:, context_length] = tokens.new([item[0] for item in next_beams])
                scores = scores.new([item[1] for item in next_beams]).unsqueeze(1)
          
            # torch.distributed.barrier()
            done = broadcast_from_last_pipeline_stage(1, torch.uint8, done)
            if done:
                break

            # Update the tokens on the first stage so the next input to
            # the network is correct.
            copy_from_last_to_first_pipeline_stage(tokens.size(), torch.int64,
                                                   tokens)

            # set inference key values to make it consistent with best beam index
            # best_batches = broadcast_from_last_pipeline_stage(beam_size, torch.int64, best_batches)
            # forward_step.inference_params.swap_key_value_dict(best_batches)

            # Update the context length for the next token generation.
            # prev_context_length = context_length

        if mpu.is_pipeline_last_stage():
            # if cannot find stop token, add open beams to hyps
            if not done:
                for beam_id in range(beam_size):
                    beam_hyp.add(tokens[beam_id].clone(), scores[beam_id].squeeze(), context_length + 1 - prompt_length)

            # rank based on scores
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0], reverse=True)
            num_return_gen = min(num_return_gen, len(sorted_hyps))
            scores = [sorted_hyps[i][0] for i in range(num_return_gen)]
            tokens = [sorted_hyps[i][1] for i in range(num_return_gen)]
            scores = torch.stack(scores, dim=0)
            tokens = torch.stack(tokens, dim=0)
            scores_size_tensor = torch.tensor(scores.shape, dtype=torch.int64, device=torch.cuda.current_device())
            tokens_size_tensor = torch.tensor(tokens.shape, dtype=torch.int64, device=torch.cuda.current_device())

        scores_size_tensor = broadcast_from_last_pipeline_stage(1, torch.int64, scores_size_tensor)
        tokens_size_tensor = broadcast_from_last_pipeline_stage(2, torch.int64, tokens_size_tensor)

        scores = broadcast_from_last_to_first_pipeline_stage(tuple(scores_size_tensor), torch.float32, scores)
        tokens = broadcast_from_last_to_first_pipeline_stage(tuple(tokens_size_tensor), torch.int64, tokens)

    return tokens, scores
