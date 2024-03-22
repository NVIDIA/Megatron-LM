
import torch

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank
)


class _VocabParallelMaxZ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits):
        # Maximum value along vocab dimension across all GPUs.
        global_logits_values = torch.max(vocab_parallel_logits, dim=-1)[0]

        torch.distributed.all_reduce(
            global_logits_values, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group()
        )

        # cited from https://arxiv.org/pdf/2309.10305.pdf Lmax-z = 2e-4 * z^2
        z_loss_weight = 2e-4

        loss = z_loss_weight * (global_logits_values ** 2)

        ctx.save_for_backward(vocab_parallel_logits, global_logits_values)
        ctx.z_loss_weight = z_loss_weight

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        vocab_parallel_logits, global_logits_values = ctx.saved_tensors
        z_loss_weight = ctx.z_loss_weight
        resize_global_values = global_logits_values.unsqueeze(-1)
        grad_input = (vocab_parallel_logits == resize_global_values) * 2 * z_loss_weight * resize_global_values
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input


def vocab_parallel_max_z(vocab_parallel_logits):
    """
    Performs max-z loss when logits are split across tensor parallel ranks

    Arguments:
        vocab_parallel_logits: logits split across tensor parallel ranks
                               dimension is [sequence_length, batch_size, hidden_size]

    """
    return _VocabParallelMaxZ.apply(vocab_parallel_logits)
