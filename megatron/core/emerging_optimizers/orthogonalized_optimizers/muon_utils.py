import torch
from absl import logging

__all__ = ["newton_schulz"]

_COEFFICIENT_SETS = {
    "simple": [
        (3.4445, -4.7750, 2.0315),
    ],
    "quintic": [
        # optimized for a quintic iteration.
        # Source: https://leloykun.github.io/ponder/muon-opt-coeffs/#how-do-we-optimize-the-coefficients
        # Numbers from: https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt_medium.py#L44
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ],
    "polar_express": [
        # Polar Express iteration from: https://arxiv.org/abs/2505.16932
        (7.2086, -15.5131, 9.0178),
        (3.9623, -2.5813, 0.4542),
        (3.9466, -2.5765, 0.4544),
        (3.8991, -2.5671, 0.4566),
        (3.7186, -2.5308, 0.4653),
        (3.1390, -2.3073, 0.4733),
        (2.1715, -1.5246, 0.3885),
        (1.8648, -1.2224, 0.3577),
    ],
}


def newton_schulz(
    x: torch.Tensor,
    steps: int,
    coefficient_type: str = "quintic",
    custom_coefficient_sets: list[tuple[float, float, float]] | None = None,
    tp_group: torch.distributed.ProcessGroup | None = None,
    tp_mode: str = "blockwise",
    partition_dim: int | None = None,
) -> torch.Tensor:
    """Use Newton-Schulz iteration to compute the zeroth power / orthogonalization of x.

    Newton-Schulz iteration to compute the zeroth power / orthogonalization of x. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero and minimize variance.
    For the purpose of minimizing steps, it turns out to be empirically effective to keep increasing the
    slope at zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce :math:`UV^T` but rather something like :math:`US'V^T`
    where :math:`S'` is diagonal with noisy values around 1, which turns out not to hurt model performance
    at all relative to :math:`UV^T`, where :math:`USV^T = G` is the SVD.

    Arguments:
        x: The tensor to be orthogonalized.
        steps: Number of Newton-Schulz iterations.
        coefficient_type: Type of coefficient set to use for the Newton-Schulz iteration.
            - "simple": Default coefficient set.
            - "quintic": Quintic iteration with optimized coefficients.
            - "polar_express": Polar Express iteration with optimized coefficients.
            - "custom": Custom coefficient sets.
        custom_coefficient_sets: Custom coefficient sets to use for the Newton-Schulz iteration.
            - If coefficient_type is "custom", custom_coefficient_sets must be provided.
            - If coefficient_type is not "custom", custom_coefficient_sets is ignored.
        tp_group: The process group for communication if input is distributed and global NS is required.
        tp_mode: how NS calcuation is handled when tensor-parallel is used. 3 available modes are:
            - "blockwise": Default, NS is calculated by local grad and not communicated between TP ranks
            - "global": allgather to recreate global grad then do duplicate NS same as TP1 on every rank
            - "global_dist": distributed NS calculation
        partition_dim: TP weight split dimension

    Returns:
        The orthogonalization of x.
    """
    # Muon is not for 1d parameters
    if x.ndim < 2:
        raise ValueError("Input tensor x must have at least 2 dimensions since Muon is not for 1d parameters.")
    if x.dtype != torch.float32:
        raise ValueError(f"Input tensor x must be in float32, got {x.dtype}")
    # fallback tp_group/tb_mode when not needed to simplify code
    if (tp_group and tp_group.size() == 1) or 'global' not in tp_mode:
        tp_group = None
        tp_mode = "blockwise"
    # TODO: properly support > 2D tensor
    if tp_group and (partition_dim is None or partition_dim < 0):
        raise ValueError(f"Choosing global NS for muon but parameter.partition_dim is {partition_dim}.")
    if x.ndim > 2 and tp_mode == 'gloabl_dist':
        raise ValueError("muon global_dist mode does not support >2D tensor, try global mode.")

    # All gather grad shards in global mode, result grad should be equivalent to tp == 1
    if tp_mode == "global":
       x_shards = [torch.empty_like(x) for _ in range(tp_group.size())]
       torch.distributed.all_gather(x_shards, x, tp_group)
       x = torch.cat(x_shards, dim=partition_dim)

    # transpose tensor to perform whitening on the smaller dimension
    needs_transpose = x.size(-2) > x.size(-1)
    # overwrites transpose flag since in dist NS we need k-dim be same as tp partition dim
    if tp_mode == "global_dist":
        needs_transpose = partition_dim == 0

    if needs_transpose:
        x = x.mT

    # Ensure spectral norm is at most 1
    if tp_mode == "global_dist":
        x_sq_sum = (x * x).sum(dim=(-2, -1))
        torch.distributed.all_reduce(x_sq_sum, op=torch.distributed.ReduceOp.SUM, group=tp_group)
        X = x / (torch.sqrt(x_sq_sum) + 1e-7)
    else:
        X = torch.nn.functional.normalize(x, p=2, dim=(-2, -1), eps=1e-7)

    if coefficient_type in _COEFFICIENT_SETS:
        coefficient_sets = _COEFFICIENT_SETS[coefficient_type]
    elif coefficient_type == "custom":
        if custom_coefficient_sets is None:
            raise ValueError("custom_coefficient_sets must be provided when coefficient_type is 'custom'.")
        coefficient_sets = custom_coefficient_sets
    else:
        raise ValueError(f"Invalid coefficient type: {coefficient_type}")

    if steps % len(coefficient_sets) != 0:
        raise ValueError(f"steps ({steps}) must be multiple of len(coefficient_sets) ({len(coefficient_sets)}).")

    # Perform the NS iterations
    if torch.get_float32_matmul_precision() == "medium":
        # PyTorch doesn't really have FP32 I/O BF16 compute kernels for precision "medium"
        # We explicitly convert to BF16 and back to FP32.
        # NOTE: There is a small difference to calling FP32 I/O BF16 compute kernels because the final result
        # is converted to BF16 before converting back to FP32. The rest should be the same as long as epilogue
        # is always in FP32.
        X = X.to(torch.bfloat16)
        logging.log_first_n(logging.INFO, "Using BF16 I/O kernels for Newton-Schulz iteration.", 1)

    for i in range(steps):
        a, b, c = coefficient_sets[i % len(coefficient_sets)]
        A = X @ X.mT
        if tp_mode == "global_dist":
            torch.distributed.all_reduce(A, op=torch.distributed.ReduceOp.SUM, group=tp_group)
        B = torch.addmm(A, A, A, beta=b, alpha=c)
        X = torch.addmm(X, B, X, beta=a, alpha=1.0)

    # Convert back to FP32. This is a noop if X is already in FP32.
    X = X.to(torch.float32)

    # undo transpose if necessary
    if needs_transpose:
        X = X.mT

    # get local result from all gather mode
    if tp_mode == "global":
        X = X.chunk(tp_group.size(), dim=partition_dim)[tp_group.rank()]

    return X
