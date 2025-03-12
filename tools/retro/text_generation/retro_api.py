# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


"""Inference API."""
import numpy as np
import torch
from megatron.core import mpu
from megatron.training import print_rank_0, get_retro_args, get_args, get_tokenizer
from megatron.inference.text_generation.communication import broadcast_float_list, broadcast_tensor, broadcast_int_list
from megatron.inference.text_generation.generation import (
    score_and_return_on_first_stage)
from tools.retro.text_generation.retro_generation import (
    retro_generate_tokens_probs_and_return_on_first_stage)
from megatron.inference.text_generation.tokenization import (
    detokenize_generations)


def tokenize_prompts(prompts=None, tokens_to_generate=None,
                     add_BOS=None, rank=0):
    """Tokenize prompts and make them avaiable on all ranks."""

    # On all ranks set to None so we can pass them to functions
    sizes_list = None
    prompts_tokens_cuda_long_tensor = None
    prompts_length_cuda_long_tensor = None

    # On the specified rank, build the above.
    if torch.distributed.get_rank() == rank:
        assert prompts is not None
        assert tokens_to_generate is not None
        # Tensor of tokens padded and their unpadded length.
        prompts_tokens_cuda_long_tensor, prompts_length_cuda_long_tensor = \
            _tokenize_prompts_and_batch(prompts, tokens_to_generate, add_BOS)
        # We need the sizes of these tensors for the boradcast
        sizes_list = [prompts_tokens_cuda_long_tensor.size(0), # Batch size
                      prompts_tokens_cuda_long_tensor.size(1)] # Sequence lenght

    # First, broadcast the sizes.
    sizes_tensor = broadcast_int_list(2, int_list=sizes_list, rank=rank)

    # Now that we have the sizes, we can boradcast the tokens
    # and length tensors.
    sizes = sizes_tensor.tolist()
    prompts_tokens_cuda_long_tensor = broadcast_tensor(
        sizes, torch.int64, tensor=prompts_tokens_cuda_long_tensor, rank=rank)
    prompts_length_cuda_long_tensor = broadcast_tensor(
        sizes[0], torch.int64, tensor=prompts_length_cuda_long_tensor,
        rank=rank)

    return prompts_tokens_cuda_long_tensor, prompts_length_cuda_long_tensor


def _tokenize_prompts_and_batch(prompts, tokens_to_generate, add_BOS):
    """Given a set of prompts and number of tokens to generate:
        - tokenize prompts
        - set the sequence length to be the max of length of prompts
          plus the number of tokens we would like to generate
        - pad all the sequences to this length so we can convert them
          into a 2D tensor.
    """

    # Tokenize all the prompts.
    tokenizer = get_tokenizer()
    if add_BOS:
        prompts_tokens = [[tokenizer.eod] + tokenizer.tokenize(prompt)
                          for prompt in prompts]
    else:
        prompts_tokens = [tokenizer.tokenize(prompt) for prompt in prompts]

    # Now we have a list of list of tokens which each list has a different
    # size. We want to extend this list to:
    #   - incorporate the tokens that need to be generated
    #   - make all the sequences equal length.
    # Get the prompts length.
    prompts_length = [len(prompt_tokens) for prompt_tokens in prompts_tokens]
    # Get the max prompts length.
    max_prompt_len = max(prompts_length)
    # Set the tokens to generate to the max prompts length for Retro
    args = get_args()
    if args.retro_add_retriever:
        tokens_to_generate = max_prompt_len
    # Number of tokens in the each sample of the batch.
    samples_length = max_prompt_len + tokens_to_generate
    # Now update the list of list to be of the same size: samples_length.
    for prompt_tokens, prompt_length in zip(prompts_tokens, prompts_length):
        padding_size = samples_length - prompt_length
        prompt_tokens.extend([tokenizer.eod] * padding_size)

    # Now we are in a structured format, we can convert to tensors.
    prompts_tokens_tensor = torch.cuda.LongTensor(prompts_tokens)
    prompts_length_tensor = torch.cuda.LongTensor(prompts_length)

    return prompts_tokens_tensor, prompts_length_tensor


def retro_generate_and_post_process(model,
                              prompts=None,
                              neighbours_array=None,
                              tokens_to_generate=0,
                              return_output_log_probs=False,
                              top_k_sampling=0,
                              top_p_sampling=0.0,
                              temperature=1.0,
                              add_BOS=False,
                              use_eod_token_for_early_termination=True,
                              random_seed=-1,
                              logits_mask=None):
    """Run inference and post-process outputs, i.e., detokenize,
    move to cpu and convert to list."""

    # Main inference.
    tokens, lengths, output_log_probs = retro_generate(
        model,
        prompts=prompts,
        neighbours_array=neighbours_array,
        tokens_to_generate=tokens_to_generate,
        return_output_log_probs=return_output_log_probs,
        top_k_sampling=top_k_sampling,
        top_p_sampling=top_p_sampling,
        temperature=temperature,
        add_BOS=add_BOS,
        use_eod_token_for_early_termination=use_eod_token_for_early_termination,
        random_seed=random_seed,
        logits_mask=logits_mask)

    # Only post-process on first stage.
    if mpu.is_pipeline_first_stage():
        tokens, prompts_plus_generations, prompts_plus_generations_segments = \
            detokenize_generations(tokens, lengths, True)

        if return_output_log_probs:
            output_log_probs = output_log_probs.cpu().numpy().tolist()
            for i, (prob, seg) in enumerate(zip(output_log_probs, prompts_plus_generations_segments)):
                output_log_probs[i] = prob[:len(seg) - 1]

        return prompts_plus_generations, prompts_plus_generations_segments, \
               output_log_probs, tokens

    return None


def retro_generate(model,
             prompts=None,
             neighbours_array=None,
             tokens_to_generate=0,
             return_output_log_probs=False,
             top_k_sampling=0,
             top_p_sampling=0.0,
             temperature=1.0,
             add_BOS=False,
             use_eod_token_for_early_termination=True,
             stop_on_double_eol=False,
             stop_on_eol=False,
             random_seed=-1,
             logits_mask=None):
    """Given prompts and input parameters, run inference and return:
       tokens: prompts plus the generated tokens.
       lengths: length of the prompt + generations. Note that we can
           discard tokens in the tokens tensor that are after the
           corresponding length.
       output_log_probs: log probs of the tokens.
    """

    # Make sure input params are avaialble to all ranks.
    values = [tokens_to_generate,
              return_output_log_probs,
              top_k_sampling, top_p_sampling,
              temperature, add_BOS, use_eod_token_for_early_termination,
              stop_on_double_eol,
              stop_on_eol,
              random_seed]
    values_float_tensor = broadcast_float_list(10, float_list=values)
    tokens_to_generate = int(values_float_tensor[0].item())
    return_output_log_probs = bool(values_float_tensor[1].item())
    top_k_sampling = int(values_float_tensor[2].item())
    top_p_sampling = values_float_tensor[3].item()
    temperature = values_float_tensor[4].item()
    add_BOS = bool(values_float_tensor[5].item())
    use_eod_token_for_early_termination = bool(values_float_tensor[6].item())
    stop_on_double_eol = bool(values_float_tensor[7].item())
    stop_on_eol = bool(values_float_tensor[8].item())
    random_seed = int(values_float_tensor[9].item())

    if random_seed != -1:
        torch.random.manual_seed(random_seed)

    # Tokenize prompts and get the batch.
    # Note that these tensors are broadcaseted to all ranks.
    if torch.distributed.get_rank() == 0:
        assert prompts is not None

    context_tokens_tensor, context_length_tensor = tokenize_prompts(
        prompts=prompts, tokens_to_generate=tokens_to_generate, add_BOS=add_BOS)

    retro_args = get_retro_args()
    retro_args.retro_gpt_chunk_length = context_length_tensor.item()

    retro_args = get_retro_args()
    args = get_args()
    r = retro_args.retro_gpt_retrieved_length
    l = int(np.ceil(min(args.max_position_embeddings, context_tokens_tensor.size(1)) / retro_args.retro_gpt_chunk_length))
    if torch.distributed.get_rank() == 0:
        neighbours_array = neighbours_array.reshape(1, args.retro_num_neighbors, r).repeat(l, axis=0)  ## dim (l, k, r)

    if tokens_to_generate == 0:
        return score_and_return_on_first_stage(
            model, context_tokens_tensor, context_length_tensor)

    # Main inference function.
    # Note that the outputs are available on the first stage.
    return retro_generate_tokens_probs_and_return_on_first_stage(
        model, context_tokens_tensor, context_length_tensor,
        neighbours_array=neighbours_array,
        return_output_log_probs=return_output_log_probs,
        top_k=top_k_sampling,
        top_p=top_p_sampling,
        temperature=temperature,
        use_eod_token_for_early_termination=use_eod_token_for_early_termination,
        stop_on_double_eol=stop_on_double_eol,
        stop_on_eol=stop_on_eol,
        logits_mask=logits_mask)