"""
Linear Cross Entropy API
Fuse cross entropy with linear layer.
"""

import typing
import torch

def _setup_platform():
    """
    Setup the platform for the Linear Cross Entropy.
    """
    assert torch.cuda.is_available(), "CUDA is not available"
    device = torch.cuda.current_device()
    cc = torch.cuda.get_device_capability(device)
    
    global forward_func, backward_func
    if cc[0] == 10:
        # from linear_cross_entropy.blackwell import entry as platform
        from .linear_cross_entropy.blackwell import entry as platform
        forward_func = platform.forward
        backward_func = platform.backward
    else:
        raise ValueError(f"Unsupported architecture: {cc[0]}")
_setup_platform()

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
        tp_group: typing.Optional[torch.distributed.ProcessGroup] = None,
        reduction: typing.Optional[str] = "mean",
        ignore_index: typing.Optional[int] = -100,
        sequence_parallel: typing.Optional[bool] = False,
    ) -> torch.Tensor:
        """
        The forward pass of the Linear Cross Entropy.
        If tp_group is not None, the weight tensor to each TP rank should be (global_vocab_size // world_size, dim).
        Note that each of the ranks should get equal shards along the vocab_size dimension.

        Args:
            @param hidden: the input tensor with shape (num_tokens, dim)
            @param weight: the lm_head weight tensor with shape (local_vocab_size, dim)
            @param labels: the labels tensor with shape (num_tokens,)
            @param tp_group: the distributed process group for TP.
            @param reduction: Default to "mean", and can be one of "none", "sum", "mean".
            @param ignore_index: The index to ignore. Default to -100.
            @param sequence_parallel: Whether to use sequence parallel. Default to False.
        Returns:
            @return: logprobs with shape
                - either (num_tokens,) when reduction is "none"
                - or (1,) when reduction is "mean" or "sum"

        tp_group is None ----------------------------------> DP
                B
            A   C
        tp_group is not None & sequence_parallel is False -> TP
                B0  B1
            A   C0  C1
        tp_group is not None & sequence_parallel is True --> SP
                B0  B1
            A0  C0  XX
            A1  XX  C1

        When tp_group is not None, the weight tensor will be split along the vocab_size dimension, 
        which means each rank will get equal shards along the global_vocab_size dimension.
        Specifically, the weight tensor to each rank will be (local_vocab_size, dim). 
        And there is an assumption that each rank will get the same local_vocab_size.

        When sequence_parallel is True, the hidden tensor will be split along the sequence length dimension,
        which means each rank will get equal shards along the sequence length dimension.
        Specifically, the hidden tensor to each rank will be (local_num_tokens, dim).
        And there is an assumption that each rank will get the same local_num_tokens.

        In TP forward pass, the hidden tensor and label tensor shall be identical among all TP ranks,
        and it's user's responsibility to ensure the hidden tensor is identical among all TP ranks.
        Then this operation will produce identical logprobs among all TP ranks.

        In TP backward pass, the gradient of the logprobs shall be identical among all TP ranks,
        and it's user's responsibility to ensure the gradient of the logprobs is identical among all TP ranks.
        Then this operation will produce distinct gradients for the local weight tensor,
        and identical gradients for the hidden tensor. 

        ```python
        # ------------ forward pass ------------ #
        hidden = tp_group.broadcast(hidden, src=0) # handled by framework
        labels = tp_group.broadcast(labels, src=0) # handled by framework
        logprobs = linear_cross_entropy(...)
        # each rank will get the same logprobs

        # ------------ backward pass ------------ #
        g_logprobs = tp_group.broadcast(g_logprobs, src=0) # handled by framework
        d_hidden, d_weight = torch.autograd.grad(...)
        # each rank will get the same d_hidden, 
        # and distinct d_weight for local weight shard
        ```

        In SP forward pass, the hidden tensor shall be split along the sequence length dimension, 
        and the label tensor shall be identical among all TP ranks.
        Then this operation will produce identical logprobs among all TP ranks.

        In SP backward pass, the gradient of the logprobs shall be identical among all TP ranks,
        Then this operation will produce distinct gradients for the local hidden tensor and weight tensor.
        ```python
        # ------------ forward pass ------------ #
        hidden = global_hidden[tp_rank] # handled by framework
        labels = tp_group.broadcast(labels, src=0) # handled by framework
        logprobs = linear_cross_entropy(...)
        # each rank will get the same logprobs

        # ------------ backward pass ------------ #
        g_logprobs = tp_group.broadcast(g_logprobs, src=0) # handled by framework
        d_hidden, d_weight = torch.autograd.grad(...)
        # each rank will get distinct local d_hidden and d_weight
        ```
        """
        with torch.cuda.nvtx.range("LinearCrossEntropy-forward"):
            logprobs, _maximum, _acc, _num_valid_tokens, tp_rank, tp_world_size, global_hidden = (
                forward_func(
                    hidden, weight, labels,
                    tp_group, 
                    reduction,
                    ignore_index,
                    sequence_parallel,
                )
            )
            ctx.save_for_backward(
                global_hidden, weight, labels,
                _maximum, _acc, _num_valid_tokens,
            )
            ctx.tp_group = tp_group
            ctx.ignore_index = ignore_index
            ctx.reduction = reduction
            ctx.tp_rank = tp_rank
            ctx.tp_world_size = tp_world_size
            ctx.sequence_parallel = sequence_parallel

        return logprobs
            

    @staticmethod
    def backward(
        ctx,
        dlogprobs: torch.Tensor
    ) -> typing.List[torch.Tensor]:
        """
        The backward pass of the Linear Cross Entropy.
        Args:
            dlogprobs (torch.Tensor): The gradient of the cross entropy, with shape
                - either (num_tokens,) when reduction is "none"
                - or (1,) when reduction is "mean" or "sum"
        Returns:
            dhidden (torch.Tensor): The gradient of the hidden.
            dweight (torch.Tensor): The gradient of the weight.
        """
        with torch.cuda.nvtx.range("LinearCrossEntropy-backward"):
            (global_hidden, weight, labels, _maximum, _accu, _num_valid_tokens) = ctx.saved_tensors

            tp_group = ctx.tp_group
            ignore_index = ctx.ignore_index
            reduction = ctx.reduction
            tp_rank = ctx.tp_rank
            tp_world_size = ctx.tp_world_size
            sequence_parallel = ctx.sequence_parallel

            d_hidden, d_weight = backward_func(
                dlogprobs,
                global_hidden,
                weight,
                labels,
                _maximum,
                _accu,
                _num_valid_tokens,
                reduction,
                ignore_index,
                tp_group,
                tp_rank,
                tp_world_size,
                sequence_parallel,
            )

        return d_hidden, d_weight, None, None, None, None, None


def linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    tp_group: typing.Optional[torch.distributed.ProcessGroup] = None,
    reduction: typing.Optional[str] = "mean",
    ignore_index: typing.Optional[int] = -100,
    sequence_parallel: typing.Optional[bool] = False,
) -> torch.Tensor:
    """
    helper function for linear cross entropy.
    """
    _impl = LinearCrossEntropy.apply
    return _impl(hidden, weight, labels, tp_group, reduction, ignore_index, sequence_parallel)

__all__ = [
    "linear_cross_entropy",
    "LinearCrossEntropy",
]


# FIXME: move this unit-test to other place
if __name__ == "__main__":
    def test_dp():
        # batch = 4
        # seqlen = 2035
        # vocab_size = 152063
        # dim = 4096
        batch = 1
        seqlen = 80
        vocab_size = 125
        dim = 64
        dtype = torch.float16
        reduction = "none"

        hidden = (
            torch.empty((batch, seqlen, dim), device="cuda", dtype=dtype)
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        weight = (
            torch.empty((vocab_size, dim), device="cuda", dtype=dtype)
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )

        labels = torch.randint(0, vocab_size, (batch, seqlen), device="cuda", dtype=torch.long)

        logits = hidden @ weight.T
        # print(logits)

        _logits = logits.to(torch.float32)
        _logits_view = _logits.view(-1, _logits.shape[-1])
        maximum = _logits_view.max(dim=-1, keepdim=False).values
        accu = torch.exp(_logits_view - maximum.unsqueeze(-1)).sum(dim=-1)
        
        logprobs = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
            reduction=reduction,
        )
        
        custom_logprobs = linear_cross_entropy(
            hidden, weight, labels, 
            reduction=reduction,
        )

        print(custom_logprobs)
        print(logprobs)

        # backward
        g_logprobs = torch.rand_like(logprobs, dtype=dtype, device="cuda")

        (d_torch_hidden, d_torch_weight) = torch.autograd.grad(
            (logprobs,), 
            (hidden, weight),
            (g_logprobs,),
            retain_graph=False
        )

        # first way to do backward
        if reduction == "mean":
            _g_logprobs = torch.broadcast_to(g_logprobs / (batch * seqlen), (batch * seqlen,))
        elif reduction == "sum":
            _g_logprobs = torch.broadcast_to(g_logprobs, (batch * seqlen,))
        else:
            _g_logprobs = g_logprobs

        intermediate = _logits_view - maximum.unsqueeze(-1)
        exp_logits = torch.exp(intermediate)
        d_logits = exp_logits / accu.unsqueeze(-1)
        d_logits *= _g_logprobs.unsqueeze(-1)
        # mask = torch.arange(vocab_size, dtype=torch.long, device="cuda")
        # mask = torch.broadcast_to(mask, (batch * seqlen, vocab_size))
        # mask = (labels.view(-1).unsqueeze(-1) == mask)

        one_hot = torch.zeros_like(_logits_view)
        one_hot.scatter_(1, labels.view(-1).unsqueeze(-1), 1)

        d_logits += one_hot * -_g_logprobs.unsqueeze(-1)
        d_logits = d_logits.to(hidden.dtype)
        # print(d_logits)
        
        d_hidden = d_logits @ weight
        d_weight = d_logits.T @ hidden.view(-1, dim)

        # print("first way to do backward")
        # print(d_hidden.view(hidden.shape))
        # print(d_torch_hidden)
        # print(d_weight)
        # print(d_torch_weight)
        # print(d_logits)

        (d_custom_hidden, d_custom_weight) = torch.autograd.grad(
            (custom_logprobs,),
            (hidden, weight),
            (g_logprobs,),
            retain_graph=False
        )
        # print(d_torch_hidden)
        # print(d_custom_hidden)
        print(d_torch_weight)
        print(d_custom_weight)

    torch.manual_seed(42)

    test_dp()