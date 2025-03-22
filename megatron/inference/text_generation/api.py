# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Inference API."""


import torch

from megatron.core import mpu
from .communication import broadcast_float_list
from .generation import (
        generate_tokens_probs_and_return_on_first_stage,
        score_and_return_on_first_stage,
        beam_search_and_return_on_first_stage)
from .tokenization import (
    tokenize_prompts,
    detokenize_generations)
from .forward_step import ForwardStep

def generate_and_post_process(model,
                              forward_step=ForwardStep,
                              prompts=None,
                              tokens_to_generate=0,
                              return_output_log_probs=False,
                              top_k_sampling=0,
                              top_p_sampling=0.0,
                              top_p_decay=0.0,
                              top_p_bound=0.0,
                              temperature=1.0,
                              add_BOS=False,
                              use_eod_token_for_early_termination=True,
                              stop_on_double_eol=False,
                              stop_on_eol=False,
                              prevent_newline_after_colon=False,
                              random_seed=-1,
                              detokenize_segments=True,
                              data_parallel=False,
                              return_topk_logprobs=0):
    """Run inference and post-process outputs, i.e., detokenize,
    move to cpu and convert to list.

    Args:
        data_parallel (bool): Enable data parallel text generation. Note: Caller must ensure
            that 1) different data parallel model replicas are provided different prompts and
            2) outputs from the different model replicas are gathered.
    """

    # Main inference.
    tokens, lengths, output_log_probs, logprobs_topk = generate(
        model,
        forward_step=forward_step,
        prompts=prompts,
        tokens_to_generate=tokens_to_generate,
        return_output_log_probs=return_output_log_probs,
        top_k_sampling=top_k_sampling,
        top_p_sampling=top_p_sampling,
        top_p_decay=top_p_decay,
        top_p_bound=top_p_bound,
        temperature=temperature,
        add_BOS=add_BOS,
        use_eod_token_for_early_termination=use_eod_token_for_early_termination,
        stop_on_double_eol=stop_on_double_eol,
        stop_on_eol=stop_on_eol,
        prevent_newline_after_colon=prevent_newline_after_colon,
        random_seed=random_seed,
        data_parallel=data_parallel)

    # Only post-process on first stage.
    if mpu.is_pipeline_first_stage():
        tokens, prompts_plus_generations, prompts_plus_generations_segments = \
            detokenize_generations(tokens, lengths, detokenize_segments)

        if return_output_log_probs:
            output_log_probs = output_log_probs.cpu().numpy().tolist()
            for i, (prob, seg) in enumerate(zip(output_log_probs, prompts_plus_generations_segments)):
                output_log_probs[i] = prob[:len(seg)-1]

        if return_topk_logprobs > 0:
            assert tokens_to_generate == 0
            return prompts_plus_generations, prompts_plus_generations_segments, \
            output_log_probs, tokens, logprobs_topk
        else:
            return prompts_plus_generations, prompts_plus_generations_segments, \
            output_log_probs, tokens

    return None

def generate(model,
             forward_step=None,
             prompts=None,
             tokens_to_generate=0,
             return_output_log_probs=False,
             top_k_sampling=0,
             top_p_sampling=0.0,
             top_p_decay=0.0,
             top_p_bound=0.0,
             temperature=1.0,
             add_BOS=False,
             use_eod_token_for_early_termination=True,
             stop_on_double_eol=False,
             stop_on_eol=False,
             prevent_newline_after_colon=False,
             random_seed=-1,
             data_parallel=False):
    """Given prompts and input parameters, run inference.

    Args:
        data_parallel (bool): Enable data parallel text generation.

    Returns:
       tokens: prompts plus the generated tokens.
       lengths: length of the prompt + generations. Note that we can
           discard tokens in the tokens tensor that are after the
           corresponding length.
       output_log_probs: log probs of the tokens.
    """
    # Make sure input params are avaialble to all ranks.
    values = [tokens_to_generate,
              return_output_log_probs,
              top_k_sampling, top_p_sampling, top_p_decay, top_p_bound,
              temperature, add_BOS, use_eod_token_for_early_termination,
              stop_on_double_eol,
              stop_on_eol,
              prevent_newline_after_colon,
              random_seed]

    values_float_tensor = broadcast_float_list(len(values), float_list=values, data_parallel=data_parallel)
    tokens_to_generate = int(values_float_tensor[0].item())
    return_output_log_probs = bool(values_float_tensor[1].item())
    top_k_sampling = int(values_float_tensor[2].item())
    top_p_sampling = values_float_tensor[3].item()
    top_p_decay = values_float_tensor[4].item()
    top_p_bound = values_float_tensor[5].item()
    temperature = values_float_tensor[6].item()
    add_BOS = bool(values_float_tensor[7].item())
    use_eod_token_for_early_termination = bool(values_float_tensor[8].item())
    stop_on_double_eol = bool(values_float_tensor[9].item())
    stop_on_eol = bool(values_float_tensor[10].item())
    prevent_newline_after_colon = bool(values_float_tensor[11].item())
    random_seed = int(values_float_tensor[12].item())

    if random_seed != -1:
        torch.random.manual_seed(random_seed)

    # Tokenize prompts and get the batch.
    # Note that these tensors are broadcasted to all ranks.
    if torch.distributed.get_rank() == 0:
        assert prompts is not None

    context_tokens_tensor, context_length_tensor = tokenize_prompts(
        prompts=prompts, tokens_to_generate=tokens_to_generate, add_BOS=add_BOS,
        data_parallel=data_parallel)

    if tokens_to_generate == 0:
        return score_and_return_on_first_stage(
            model, context_tokens_tensor, context_length_tensor)

    # Main inference function.
    # Note that the outputs are available on the first stage.
    return generate_tokens_probs_and_return_on_first_stage(
        model, forward_step, context_tokens_tensor, context_length_tensor,
        return_output_log_probs=return_output_log_probs,
        top_k=top_k_sampling,
        top_p=top_p_sampling,
        top_p_decay=top_p_decay,
        top_p_bound=top_p_bound,
        temperature=temperature,
        use_eod_token_for_early_termination=use_eod_token_for_early_termination,
        stop_on_double_eol=stop_on_double_eol,
        stop_on_eol=stop_on_eol,
        prevent_newline_after_colon=prevent_newline_after_colon,
        data_parallel=data_parallel)

def beam_search_and_post_process(model,
                                 forward_step=ForwardStep,
                                 prompts=None,
                                 tokens_to_generate=0,
                                 beam_size=0,
                                 add_BOS=False,
                                 stop_token=50256,
                                 num_return_gen=1,
                                 length_penalty=1,
                                 prevent_newline_after_colon=False,
                                 detokenize_segments=True):
    """Run beam search and post-process outputs, i.e., detokenize,
    move to cpu and convert to list."""

    # Main inference.
    tokens, scores = beam_search(model,
                                 forward_step=forward_step,
                                 prompts=prompts,
                                 tokens_to_generate=tokens_to_generate,
                                 beam_size=beam_size,
                                 add_BOS=add_BOS,
                                 stop_token=stop_token,
                                 num_return_gen=num_return_gen,
                                 length_penalty=length_penalty,
                                 prevent_newline_after_colon=prevent_newline_after_colon)
    # Only post-process on first stage.
    if mpu.is_pipeline_first_stage():
        lengths = tokens.size(1)*torch.ones(beam_size, dtype=torch.int64, device=torch.cuda.current_device())
        tokens, prompts_plus_generations, prompts_plus_generations_segments = detokenize_generations(tokens, lengths, detokenize_segments)
        scores = scores.cpu().numpy().tolist()
        return prompts_plus_generations, prompts_plus_generations_segments, scores

    return None

def beam_search(model, forward_step, prompts=None, tokens_to_generate=0, beam_size=0, add_BOS=False, stop_token=50256, num_return_gen=1, length_penalty=1, prevent_newline_after_colon=False):
    # Make sure input params are avaialble to all ranks.
    values = [tokens_to_generate,
              beam_size,
              add_BOS,
              stop_token,
              num_return_gen,
              length_penalty,
              prevent_newline_after_colon]
    values_float_tensor = broadcast_float_list(len(values), float_list=values)
    tokens_to_generate = int(values_float_tensor[0].item())
    beam_size = int(values_float_tensor[1].item())
    add_BOS = bool(values_float_tensor[2].item())
    stop_token = int(values_float_tensor[3].item())
    num_return_gen = int(values_float_tensor[4].item())
    length_penalty = values_float_tensor[5].item()
    prevent_newline_after_colon = values_float_tensor[6].item()

    context_tokens_tensor, context_length_tensor = tokenize_prompts(
        prompts=prompts, tokens_to_generate=tokens_to_generate, add_BOS=add_BOS)

    return beam_search_and_return_on_first_stage(model, forward_step, context_tokens_tensor, context_length_tensor,
            beam_size, stop_token=stop_token, num_return_gen=num_return_gen, length_penalty=length_penalty,
            prevent_newline_after_colon=prevent_newline_after_colon)
