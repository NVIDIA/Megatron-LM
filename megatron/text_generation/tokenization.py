# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Tokenization utilities."""


import torch


from megatron import get_tokenizer, get_args
from .communication import broadcast_int_list, broadcast_tensor


def detokenize_generations(tokens_gpu_tensor,
                           lengths_gpu_tensor,
                           return_segments):
    """Detokenize the generated tokens."""

    tokenizer = get_tokenizer()
    args = get_args()
    prompts_plus_generations = []
    if return_segments:
        prompts_plus_generations_segments = []

    tokens = tokens_gpu_tensor.cpu().numpy().tolist()
    lengths = lengths_gpu_tensor.cpu().numpy().tolist()
    for sequence_tokens, length in zip(tokens, lengths):
        sequence_tokens = sequence_tokens[:length]
        prompts_plus_generations.append(
            tokenizer.detokenize(sequence_tokens))
        if return_segments:
            words = []
            for token in sequence_tokens:
                if args.tokenizer_type in ['SentencePieceTokenizer', 
                        'GPTSentencePieceTokenizer']:
                    word = tokenizer.decoder[token]
                elif args.tokenizer_type == 'NullTokenizer':
                    word = str(token)
                else:
                    word = tokenizer.tokenizer.decoder[token]
                    word = bytearray(
                        [tokenizer.tokenizer.byte_decoder[c] for c in word]).decode(
                            'utf-8', errors='replace')
                words.append(word)
            prompts_plus_generations_segments.append(words)

    if return_segments:
        return tokens, prompts_plus_generations, \
            prompts_plus_generations_segments

    return tokens, prompts_plus_generations


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
