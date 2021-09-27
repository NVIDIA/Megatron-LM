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


import torch
import torch.nn.functional as F

from megatron import get_args, get_tokenizer
from megatron import mpu
from megatron.utils import get_ltor_masks_and_position_ids
from .communication import (
    copy_from_last_to_first_pipeline_stage,
    broadcast_from_last_pipeline_stage)
from .forward_step import forward_step
from .sampling import sample


def generate_tokens(model, tokens, lengths, return_all_probs=False,
                    temperature=1.0):
    """Main token generation function."""

    args = get_args()
    tokenizer = get_tokenizer()

    batch_size = tokens.size(0)
    min_prompt_length = lengths.min().item()
    max_sequence_length = tokens.size(1)
    max_sequence_length = min(max_sequence_length, args.max_position_embeddings)

    # Added termination_id to support the case that we want to terminate the
    # generation once that id is generated.
    if hasattr(args, 'eos_id'):
        termination_id = args.eos_id
    else:
        termination_id = tokenizer.eod

    # ===================
    # Pre-allocate memory
    # ===================

    # Log probability of the sequence (prompt + generated tokens)
    output_log_probs = torch.empty(batch_size, max_sequence_length - 1,
                                   dtype=torch.float32,
                                   device=torch.cuda.current_device())
    # Lengths of generated seuquence including including prompts.
    generated_sequence_lengths = torch.ones(
        batch_size, dtype=torch.int64,
        device=torch.cuda.current_device()) * max_sequence_length
    # Whether we have reached a termination id.
    is_generation_done = torch.zeros(batch_size, dtype=torch.uint8,
                                     device=torch.cuda.current_device())

    attention_mask, position_ids = _build_attention_mask_and_position_ids(
        tokens)

    model.eval()
    with torch.no_grad():
        prev_context_length = 0
        for context_length in range(min_prompt_length, max_sequence_length):

            # If we are starting from scratch, allocate memory for the entire
            # context, otherwise  set this to false so the memory is not
            # reallocated.
            set_inference_key_value_memory = (prev_context_length == 0)

            # Pick the slice that we need to pass through the network.
            tokens2use = tokens[:, prev_context_length:context_length]
            positions2use = position_ids[:, prev_context_length:context_length]
            attention_mask2use = attention_mask[
                ..., prev_context_length:context_length, :context_length]

            # logits will be meanigful only in the last pipeline stage.
            logits = forward_step(
                model, tokens2use, positions2use, attention_mask2use,
                set_inference_key_value_memory=set_inference_key_value_memory,
                inference_max_sequence_len=max_sequence_length)

            if mpu.is_pipeline_last_stage():
                # Always the last stage should have an output.
                assert logits is not None

                # Sample.
                last_token_logits = logits[:, -1, :]
                new_sample, updated_last_token_logits = sample(
                    last_token_logits,
                    greedy=args.greedy,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    temperature=temperature,
                    vocab_size=tokenizer.vocab_size)
                # Now that we have the sample and updated logits,
                # update the main logits and input tokens.
                # If a prompt length is smaller or equal th current context
                # length, it means we have started generating tokens
                started = lengths <= context_length
                # Update the logits
                last_token_logits.masked_scatter_(
                    started.unsqueeze(1), updated_last_token_logits[started])
                # and the tokens.
                tokens[started, context_length] = new_sample[started]

                # Calculate the log probabilities.
                log_probs = F.log_softmax(logits, dim=2)
                # Pick the tokens that we need to get the log probabilities for.
                # Note that next input token is the token which we selected in
                # the current logits, so shift by 1.
                indices = torch.unsqueeze(
                    tokens[:, (prev_context_length + 1):(context_length + 1)],
                    2)
                output_log_probs[:, prev_context_length:context_length] = \
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
                done_token = (new_sample == termination_id).byte() & \
                    started.byte()
                just_finished = (done_token & ~is_generation_done).bool()
                generated_sequence_lengths[just_finished.view(-1)] = \
                    context_length + 1
                is_generation_done = is_generation_done | done_token
                done = torch.all(is_generation_done)
            done = broadcast_from_last_pipeline_stage(1, torch.uint8,
                                                      tensor=done)
            if done:
                break

        if mpu.is_pipeline_last_stage():
            if return_all_probs:
                full_logits = None
                return tokens, generated_sequence_lengths, output_log_probs, \
                    full_logits, context_length + 1
            return tokens, generated_sequence_lengths, output_log_probs, \
                None, context_length + 1

        if mpu.is_pipeline_first_stage():
            return tokens, None, None, None, context_length + 1
        return None, None, None, None, context_length + 1


def _build_attention_mask_and_position_ids(tokens):
    """Build the attention mask and postition ids for the input tokens."""

    # Since we are not interested in loss-mask and reset attention/position
    # is also False, eod_token is not used so it is safe to set it to None.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=None,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False)

    return attention_mask, position_ids
