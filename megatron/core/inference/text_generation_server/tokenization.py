# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Tokenization utilities."""

from typing import Optional

import torch

from megatron.core.inference.communication_utils import broadcast_int_list, broadcast_tensor
from megatron.core.process_groups_config import ProcessGroupCollection, resolve_process_groups


def tokenize_prompts(
    tokenizer,
    prompts=None,
    tokens_to_generate=None,
    add_BOS=None,
    rank=0,
    data_parallel=False,
    pg_collection: Optional[ProcessGroupCollection] = None,
):
    """Tokenize prompts and make them avaiable on all ranks.

    Args:
        data_parallel (bool): Broadcast tokens across a single data parallel model replica.
        pg_collection: Process groups of the model, used with `data_parallel`: the ranks whose
            `dp` group starts at global rank `rank` tokenize, and each `mp` group broadcasts the
            tokens from its first rank. If None, the global groups are used; that fallback is
            deprecated.
    """

    # On all ranks set to None so we can pass them to functions
    sizes_list = None
    prompts_tokens_cuda_long_tensor = None
    prompts_length_cuda_long_tensor = None

    # On the specified rank, build the above.
    src_rank = torch.distributed.get_rank()
    mp_group = None
    if data_parallel:
        pg_collection = resolve_process_groups(
            pg_collection,
            owner="tokenize_prompts",
            required=("dp", "mp"),
            deprecated_in="0.21",
            removed_in="0.23",
        )
        dp_group, mp_group = pg_collection.dp, pg_collection.mp
        # torch.distributed treats a None group as the whole world.
        if dp_group is None or mp_group is None:
            raise ValueError(
                "tokenize_prompts with data_parallel=True broadcasts within the dp and mp "
                "groups; pg_collection must not set them to None."
            )
        src_rank = torch.distributed.get_global_rank(dp_group, 0)

    if src_rank == rank:
        assert prompts is not None
        assert tokens_to_generate is not None
        # Tensor of tokens padded and their unpadded length.
        prompts_tokens_cuda_long_tensor, prompts_length_cuda_long_tensor = (
            _tokenize_prompts_and_batch(tokenizer, prompts, tokens_to_generate, add_BOS)
        )
        # We need the sizes of these tensors for the boradcast
        sizes_list = [
            prompts_tokens_cuda_long_tensor.size(0),  # Batch size
            prompts_tokens_cuda_long_tensor.size(1),
        ]  # Sequence lenght

    # First, broadcast the sizes.
    sizes_tensor = broadcast_int_list(
        2, int_list=sizes_list, rank=rank, data_parallel=data_parallel, mp_group=mp_group
    )

    # Now that we have the sizes, we can boradcast the tokens
    # and length tensors.
    sizes = sizes_tensor.tolist()
    prompts_tokens_cuda_long_tensor = broadcast_tensor(
        sizes,
        torch.int64,
        tensor=prompts_tokens_cuda_long_tensor,
        rank=rank,
        data_parallel=data_parallel,
        mp_group=mp_group,
    )
    prompts_length_cuda_long_tensor = broadcast_tensor(
        sizes[0],
        torch.int64,
        tensor=prompts_length_cuda_long_tensor,
        rank=rank,
        data_parallel=data_parallel,
        mp_group=mp_group,
    )

    return prompts_tokens_cuda_long_tensor, prompts_length_cuda_long_tensor


def _tokenize_prompts_and_batch(tokenizer, prompts, tokens_to_generate, add_BOS):
    """Given a set of prompts and number of tokens to generate:
    - tokenize prompts
    - set the sequence length to be the max of length of prompts
      plus the number of tokens we would like to generate
    - pad all the sequences to this length so we can convert them
      into a 2D tensor.
    """

    # Tokenize all the prompts.
    if hasattr(tokenizer, 'eod'):
        eod_token = tokenizer.eod
    elif hasattr(tokenizer, 'eos_id'):
        eod_token = tokenizer.eos_id
    else:
        raise AttributeError('No eod token found in Tokenizer')
    if add_BOS:
        prompts_tokens = [[eod_token] + tokenizer.tokenize(prompt) for prompt in prompts]
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
