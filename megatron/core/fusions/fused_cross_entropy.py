# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Tuple

import torch

from megatron.core.jit import jit_fuser
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.tensor_parallel.cross_entropy import VocabParallelCrossEntropy
from megatron.core.tensor_parallel.utils import VocabUtility


@jit_fuser
def calculate_logits_max(vocab_parallel_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    vocab_parallel_logits, logits_max = VocabParallelCrossEntropy.calculate_logits_max(
        vocab_parallel_logits
    )

    return vocab_parallel_logits, logits_max


@jit_fuser
def calculate_predicted_logits(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    logits_max: torch.Tensor,
    vocab_start_index: int,
    vocab_end_index: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    (
        target_mask,
        masked_target_1d,
        predicted_logits,
        sum_exp_logits,
        exp_logits,
    ) = VocabParallelCrossEntropy.calculate_predicted_logits(
        vocab_parallel_logits, target, logits_max, vocab_start_index, vocab_end_index
    )

    predicted_logits_sum_exp_logits = torch.cat((predicted_logits, sum_exp_logits))

    return target_mask, masked_target_1d, predicted_logits_sum_exp_logits, exp_logits


@jit_fuser
def calculate_cross_entropy_loss(
    exp_logits: torch.Tensor, predicted_logits_sum_exp_logits: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    split_val = predicted_logits_sum_exp_logits.size()[0] // 2
    predicted_logits, sum_exp_logits = torch.split(predicted_logits_sum_exp_logits, split_val)

    exp_logits, loss = VocabParallelCrossEntropy.calculate_cross_entropy_loss(
        exp_logits, predicted_logits, sum_exp_logits
    )

    return exp_logits, loss


@jit_fuser
def calculate_gradients(
    softmax: torch.Tensor,
    grad_output: torch.Tensor,
    target_mask: torch.Tensor,
    masked_target_1d: torch.Tensor,
) -> torch.Tensor:

    (
        grad_2d,
        arange_1d,
        softmax_update,
        grad_input,
    ) = VocabParallelCrossEntropy.prepare_gradient_calculation_operands(softmax, target_mask)

    grad_input = VocabParallelCrossEntropy.calculate_gradients(
        grad_2d, arange_1d, masked_target_1d, softmax_update, grad_input, grad_output
    )

    grad_input = grad_input.to(torch.bfloat16)

    return grad_input


class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target):

        vocab_parallel_logits, logits_max = calculate_logits_max(vocab_parallel_logits)
        torch.distributed.all_reduce(
            logits_max, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group()
        )

        # Get the partition's vocab indices
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)

        (
            target_mask,
            masked_target_1d,
            predicted_logits_sum_exp_logits,
            exp_logits,
        ) = calculate_predicted_logits(
            vocab_parallel_logits, target, logits_max, vocab_start_index, vocab_end_index
        )

        # All reduce is needed to get the chunks from other GPUs.
        # In the fused case, tensors are batches to invoke a single
        # AllReduce call
        torch.distributed.all_reduce(
            predicted_logits_sum_exp_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group(),
        )

        exp_logits, loss = calculate_cross_entropy_loss(exp_logits, predicted_logits_sum_exp_logits)

        # Store softmax, target-mask and masked-target for backward pass.
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss

    @staticmethod
    def backward(ctx, grad_output):

        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        grad_input = calculate_gradients(softmax, grad_output, target_mask, masked_target_1d)

        return grad_input, None


def fused_vocab_parallel_cross_entropy(vocab_parallel_logits, target):
    """
    Performs cross entropy loss when logits are split across tensor parallel ranks

    Args:
        vocab_parallel_logits: logits split across tensor parallel ranks
                               dimension is [sequence_length, batch_size, hidden_size]

        target: correct vocab ids of dimseion [sequence_length, micro_batch_size]

    """
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target)
