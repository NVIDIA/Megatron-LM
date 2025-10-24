"""
Linear Cross Entropy API
Fuse cross entropy with linear layer.
"""

import typing
import torch

class LinearCrossEntropy(torch.autograd.Function):
    """
    This class implements a custom autograd function for linear and cross entropy, whose equivalent logic in PyTorch is:
        ```python
        def torch_entropy(hidden, weight, labels):
            logits = torch.matmul(hidden, weight)
            logprobs = torch.nn.functional.cross_entropy(logits, labels)
            return logprobs
        ```
    """

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        reduction: typing.Optional[str] = "mean",
        dist_process_group: typing.Optional[torch.distributed.ProcessGroup] = None,
        ignore_index: typing.Optional[int] = -100,
    ) -> torch.Tensor:
        """
        The forward pass of the Linear Cross Entropy.
        If dist_process_group is passed for distributed loss calculation,
        the weight tensor to each distributed rank should be (*, vocab_size / world_size).
        Note that each of the ranks should get equal shards along the vocab_size dimension.

        Args:
            hidden (torch.Tensor): The input tensor of shape (num_tokens, hidden_size).
            weight (torch.Tensor): The weight tensor of shape (hidden_size, vocab_size).
            labels (torch.Tensor): The labels tensor of shape (num_tokens,).
            reduction (str, optional): The reduction method. Defaults to "mean", and can be
                one of "none", "sum", "mean".
        Returns:
            logprobs (torch.Tensor): The cross entropy.
        """
        with torch.cuda.nvtx.range("LinearCrossEntropy-forward"):
            logprobs = torch.empty(
                hidden.view(-1, hidden.shape[-1]).shape[0], 
                device=hidden.device, 
                dtype=torch.float32)

        return logprobs

    @staticmethod
    def backward(ctx, dlogprobs: torch.Tensor) -> typing.List[torch.Tensor]:
        """
        The backward pass of the Linear Cross Entropy.
        Args:
            dlogprobs (torch.Tensor): The gradient of the cross entropy.
        Returns:
            dhidden (torch.Tensor): The gradient of the hidden.
            dweight (torch.Tensor): The gradient of the weight.
        """
        with torch.cuda.nvtx.range("LinearCrossEntropy-backward"):
            d_hidden = torch.empty(hidden.shape, device=hidden.device, dtype=hidden.dtype)
            d_weight = torch.empty(weight.shape, device=weight.device, dtype=weight.dtype)
        return d_hidden, d_weight, None, None, None, None


linear_cross_entropy = LinearCrossEntropy.apply

__all__ = [
    "linear_cross_entropy",
    "LinearCrossEntropy",
]
