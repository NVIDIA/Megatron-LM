# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


"""Generation utilities."""
import torch
import torch.nn.functional as F
from megatron import get_args, get_tokenizer
from megatron import get_retro_args
from megatron.core import mpu
from megatron.utils import get_ltor_masks_and_position_ids, unwrap_model
from megatron.text_generation.communication import (
    copy_from_last_to_first_pipeline_stage,
    broadcast_from_last_pipeline_stage,
    broadcast_from_last_to_first_pipeline_stage, broadcast_int_list, broadcast_tensor)
from megatron.text_generation.generation import _build_attention_mask_and_position_ids
from megatron.text_generation.sampling import sample



def retro_generate_tokens_probs_and_return_on_first_stage(
        model, tokens, lengths, neighbours_array=None,
        return_output_log_probs=False,
        top_k=0, top_p=0.0,
        temperature=1.0,
        use_eod_token_for_early_termination=True,
        stop_on_double_eol=False,
        stop_on_eol=False,
        logits_mask=None):
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
        for context_length in range(min_prompt_length, max_sequence_length):
            prev_context_length = 0
            sizes_list = None
            neighbor_tokens_cuda_long_tensor = None

            # get the chunks for retrieval
            if torch.distributed.get_rank() == 0:
                neighbor_tokens = neighbours_array
                neighbor_tokens_cuda_long_tensor = torch.cuda.LongTensor(
                    neighbor_tokens.reshape((-1, retro_args.retro_gpt_retrieved_length)))
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

            logits = model(tokens2use, positions2use, attention_mask2use,
                           retriever_input_ids=neighbor_tokens_cuda_long_tensor,
                           retriever_position_ids=neighbor_position_ids, retriever_attn_mask=neighbor_attention_mask,
                           )

            if mpu.is_pipeline_last_stage():
                # Always the last stage should have an output.
                assert logits is not None

                # Sample.
                last_token_logits = logits[:, context_length - 1, :]
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
                    hit_two_eols = (new_sample == 198).byte() & (
                            tokens[:, context_length - 1] == 198).byte() & started.byte()
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
