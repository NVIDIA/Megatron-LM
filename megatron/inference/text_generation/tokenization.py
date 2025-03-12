# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Tokenization utilities."""


import torch


from megatron.core import parallel_state
from megatron.training import get_args, get_tokenizer
from .communication import broadcast_int_list, broadcast_tensor


def detokenize_generations(tokens_gpu_tensor,
                           lengths_gpu_tensor,
                           detokenize_segments):
    """Detokenize the generated tokens."""

    tokenizer = get_tokenizer()
    prompts_plus_generations = []
    prompts_plus_generations_segments = []

    tokens = tokens_gpu_tensor.cpu().numpy().tolist()
    lengths = lengths_gpu_tensor.cpu().numpy().tolist()

    for sequence_tokens, length in zip(tokens, lengths):
        sequence_tokens = sequence_tokens[:length]
        detok_str = tokenizer.detokenize(sequence_tokens)
        prompts_plus_generations.append(detok_str)
        if detokenize_segments:
            try:
                offsets = tokenizer.offsets(sequence_tokens, detok_str)
                words = [
                    detok_str[start:end]
                    for start, end in zip(offsets, offsets[1:] + [len(detok_str)])
                ]
            except NotImplementedError:
                words = []
                for token in sequence_tokens:
                    word = tokenizer.tokenizer.decoder[token]
                    word = bytearray([tokenizer.tokenizer.byte_decoder[c] for c in word]).decode(
                        "utf-8", errors="replace"
                    )
                    words.append(word)

            prompts_plus_generations_segments.append(words)

    return tokens, prompts_plus_generations, prompts_plus_generations_segments


def tokenize_prompts(prompts=None, tokens_to_generate=None,
                     add_BOS=None, rank=0, data_parallel=False):
    """Tokenize prompts and make them avaiable on all ranks.

    Args:
        data_parallel (bool): Broadcast tokens across a single data parallel model replica.
    """

    # On all ranks set to None so we can pass them to functions
    sizes_list = None
    prompts_tokens_cuda_long_tensor = None
    prompts_length_cuda_long_tensor = None

    # On the specified rank, build the above.
    src_rank = torch.distributed.get_rank()
    if data_parallel:
        src_rank = parallel_state.get_data_parallel_src_rank()

    if src_rank == rank:
        assert prompts is not None
        assert tokens_to_generate is not None
        # Tensor of tokens padded and their unpadded length.
        prompts_tokens_cuda_long_tensor, prompts_length_cuda_long_tensor = \
            _tokenize_prompts_and_batch(prompts, tokens_to_generate, add_BOS)
        # We need the sizes of these tensors for the boradcast
        sizes_list = [prompts_tokens_cuda_long_tensor.size(0), # Batch size
                      prompts_tokens_cuda_long_tensor.size(1)] # Sequence lenght

    # First, broadcast the sizes.
    sizes_tensor = broadcast_int_list(2, int_list=sizes_list, rank=rank, data_parallel=data_parallel)

    # Now that we have the sizes, we can boradcast the tokens
    # and length tensors.
    sizes = sizes_tensor.tolist()
    prompts_tokens_cuda_long_tensor = broadcast_tensor(
        sizes, torch.int64, tensor=prompts_tokens_cuda_long_tensor, rank=rank, data_parallel=data_parallel)
    prompts_length_cuda_long_tensor = broadcast_tensor(
        sizes[0], torch.int64, tensor=prompts_length_cuda_long_tensor,
        rank=rank, data_parallel=data_parallel)

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
    if hasattr(tokenizer, 'eod'):
        eod_token = tokenizer.eod
    elif hasattr(tokenizer, 'eos_id'):
        eod_token = tokenizer.eos_id
    else:
        raise AttributeError('No eod token found in Tokenizer')
    if add_BOS:
        prompts_tokens = [[eod_token] + tokenizer.tokenize(prompt)
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
    # Number of tokens in the each sample of the batch.
    samples_length = max_prompt_len + tokens_to_generate
    # Now update the list of list to be of the same size: samples_length.
    for prompt_tokens, prompt_length in zip(prompts_tokens, prompts_length):
        padding_size = samples_length - prompt_length
        prompt_tokens.extend([eod_token] * padding_size)

    # Now we are in a structured format, we can convert to tensors.
    prompts_tokens_tensor = torch.tensor(prompts_tokens, dtype=torch.long, device='cuda')
    prompts_length_tensor = torch.tensor(prompts_length, dtype=torch.long, device='cuda')

    return prompts_tokens_tensor, prompts_length_tensor
