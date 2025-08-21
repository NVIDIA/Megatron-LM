from typing import Any, Callable, override

from absl import logging

import torch
import torch.optim as optim
from torch.optim.optimizer import ParamsT

from llm_shower import utils


class OrthogonalizedOptimizer(optim.Optimizer):
    """Base class for orthogonalized optimizers.

    This class is a wrapper around a base optimizer that performs orthogonalization on the updates.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: The learning rate used by the internal SGD.
        momentum_beta: The momentum used by the internal SGD.
        use_nesterov: Whether to use Nesterov-style momentum in the internal SGD.
        weight_decay: The weight decay used by the optimizer, default to be decoupled weight decay.
            See Decoupled Weight Decay Regularization: https://arxiv.org/abs/1711.05101
        use_decoupled_weight_decay: Whether to use decoupled weight decay, default to be True.
        split_qkv: Whether parameter is fused attention parameters (QKV, GQA, etc.), default to be False.
        qkv_split_shapes: For grouped attention parameters (QKV, GQA, etc.), specify the shapes as a tuple of 3 integers
            representing the sizes of Q, K, V components along the first dimension.
        fp32_matmul_prec: Precision of the matmul operations in optimizer states GEMM operations.
        orthogonalize_fn: Function to orthogonalize the updates.
        scale_factor_fn: Function to compute the scale factor for the update.
        **kwargs: Arguments passed through to the base optimizer.

    Notes:
        Keyword arguments passed through are not checked here. Optimizer inherited from this class should check them.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float,
        momentum_beta: float,
        use_nesterov: bool,
        weight_decay: float,
        use_decoupled_weight_decay: bool,
        split_qkv: bool,
        qkv_split_shapes: tuple[int, int, int] | None,
        fp32_matmul_prec: str,
        orthogonalize_fn: Callable | None = None,
        scale_factor_fn: Callable | None = None,
        **kwargs: Any,
    ):
        if orthogonalize_fn is None:
            logging.warning("orthogonalize_fn not provided. Using noop")
            orthogonalize_fn = torch.nn.Identity()

        if scale_factor_fn is None:
            logging.warning("scale_factor_fn not provided. Using default scale_factor_fn.")

            def return_one(*args, **kwargs):  # type: ignore[no-untyped-def]
                return 1.0

            scale_factor_fn = return_one

        if qkv_split_shapes is not None:
            if len(qkv_split_shapes) != 3:
                raise ValueError(
                    f"qkv_split_shapes must be a tuple of 3 integers, got {len(qkv_split_shapes)} elements"
                )
            if not all(isinstance(s, int) for s in qkv_split_shapes):
                raise ValueError(f"All elements in qkv_split_shapes must be integers, got {qkv_split_shapes}")
            if any(s <= 0 for s in qkv_split_shapes):
                raise ValueError(f"All elements in qkv_split_shapes must be positive, got {qkv_split_shapes}")

        self.fp32_matmul_prec = fp32_matmul_prec
        default_args_dict = dict(
            lr=lr,
            momentum_beta=momentum_beta,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            use_decoupled_weight_decay=use_decoupled_weight_decay,
            split_qkv=split_qkv,
            qkv_split_shapes=qkv_split_shapes,
            **kwargs,
        )

        super().__init__(params, default_args_dict)
        self.orthogonalize_fn = orthogonalize_fn
        self.scale_factor_fn = scale_factor_fn

    @torch.no_grad()  # type: ignore[misc]
    @override
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        if closure is None:
            loss = None
        else:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                state = self.state[p]

                # initialize momentum buffer
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                # Subsequent update to exp_avg are all inplace, so it is not assigned back to state.
                exp_avg = state["momentum_buffer"]

                # Apply weight decay
                if group["weight_decay"] > 0.0:
                    if group["use_decoupled_weight_decay"]:
                        # Apply decoupled weight decay
                        p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
                    else:
                        # add l2 regularization before preconditioning (i.e. adding a squared loss term)
                        grad += group["weight_decay"] * p

                # update momentum buffer with EMA of gradient
                exp_avg.lerp_(grad, 1 - group["momentum_beta"])

                # include nesterov momentum
                if group["use_nesterov"]:
                    grad = grad.lerp(exp_avg, group["momentum_beta"])
                else:
                    grad = exp_avg

                with utils.fp32_matmul_precision(self.fp32_matmul_prec):
                    if grad.dim() == 1:
                        # for 1D parameters, skip Newton-Schulz iteration, optimizer defaults to AdamW update
                        raise ValueError("OrthogonalizedOptimizer does not support 1D parameters")
                    elif group["split_qkv"]:
                        # split grouped attention parameters (e.g., QKV, GQA, etc.)
                        qkv_shapes = _get_qkv_split_dim_0(grad, group["qkv_split_shapes"])
                        # Split gradient for Q, K, V
                        qkv_grads = torch.split(grad, qkv_shapes, dim=0)
                        # Apply Newton-Schulz to each component
                        qkv_whitened = [self.orthogonalize_fn(g) for g in qkv_grads]
                        qkv_scales = [self.scale_factor_fn(size, g.size(1)) for size, g in zip(qkv_shapes, qkv_grads)]
                        # Apply individual scales to each component and concatenate
                        grad = torch.cat([whitened * scale for whitened, scale in zip(qkv_whitened, qkv_scales)])
                        scale = 1.0  # no additional scaling needed as we have already applied scale to each component
                    else:
                        grad = self.orthogonalize_fn(grad)
                        scale = self.scale_factor_fn(grad.size(0), grad.size(1))

                # perform weight update
                # scale is applied to have update RMS == 1
                p.add_(grad, alpha=-group["lr"] * scale)

        return loss


def _get_qkv_split_dim_0(grad: torch.Tensor, qkv_split_shapes: tuple[int, int, int]) -> tuple[int, int, int]:
    """Get the split shapes for the QKV parameters.

    Args:
        grad: The gradient tensor.
        qkv_split_shapes: The split shapes tuple for the QKV parameters.

    Returns:
        The split shapes tuple for the QKV parameters.
    """
    q_size, k_size, v_size = qkv_split_shapes

    # Validate that shapes add up to gradient size
    expected_size = q_size + k_size + v_size
    if grad.size(0) != expected_size:
        raise ValueError(
            f"QKV split shapes {qkv_split_shapes} sum to {expected_size}, "
            f"but gradient has size {grad.size(0)} in first dimension"
        )

    return q_size, k_size, v_size
