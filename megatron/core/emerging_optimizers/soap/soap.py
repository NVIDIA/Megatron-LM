from itertools import chain
from typing import Iterable, Callable, Tuple, List, Optional, Union, override

import torch
import torch.optim as optim

from absl import logging

from megatron.core.emerging_optimizers.soap.soap_utils import (
    get_eigenbasis_eigh,
    get_eigenbasis_qr,
)

from megatron.core.emerging_optimizers import utils
from megatron.core.emerging_optimizers.scalar_optimizers import calculate_adam_update

__all__ = [
    "SOAP",
    "precondition",
    "init_kronecker_factors",
    "update_kronecker_factors",
    "update_eigenbasis_and_momentum",
]


class SOAP(optim.Optimizer):
    """Implements a variant of SOAP (ShampoO with Adam in the Preconditioner eigenbasis) algorithm.

    SOAP (https://arxiv.org/abs/2409.11321) is a preconditioned optimizer that combines the benefits of Shampoo's
    non-diagonal preconditioning with Adam's adaptive learning rates. It uses
    gradient correlation matrix eigenbasis-based preconditioning to adapt to the local geometry of the
    optimization landscape.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: The learning rate to use
        betas: Inner Adam's betas parameters (b1, b2)
        shampoo_beta: Beta for the kronecker factor matrices (L and R in paper) moving average
            instead of betas[1] if >= 0
        eps: Inner Adam's epsilon for numerical stability
        weight_decay: Weight decay coefficient
        use_decoupled_weight_decay: Whether to use decoupled weight decay, see Decoupled Weight Decay Regularization:
            https://arxiv.org/abs/1711.05101.
        use_nesterov: uses Nesterov momentum in Adam (https://cs229.stanford.edu/proj2015/054_report.pdf,
            https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ)
        precondition_frequency: How often to update the preconditioner. Can be an integer for fixed frequency
            or a callable function that takes the current step as input and returns the frequency.
        precondition_warmup_steps: How many steps to warm up the preconditioner (i.e. update every step)
        adam_warmup_steps: How many steps to skip preconditioning in the beginning (i.e. use standard AdamW updates)
        precondition_1d: Whether to precondition 1D gradients (like biases).
        max_precond_dim: Maximum dimension of the preconditioner matrices. Skips preconditioning if any tensor dimension exceeds.
        trace_normalization: Whether to normalize update by the trace of the kronecker factor matrix
        normalize_preconditioned_grads: Whether to normalize preconditioned gradients per layer
        correct_bias: Whether to use bias correction in Inner Adam and Kronecker factor matrices EMA
        fp32_matmul_prec: Precision of the matmul operations in optimizer states GEMM operations
        use_eigh: Whether to use full symmetric eigendecomposition (eigh) to compute the eigenbasis.
            If False, use orthogonal iteration to compute the eigenbasis.
        qr_fp32_matmul_prec: Precision of the matmul operations in QR decomposition.
        use_adaptive_criteria: Whether to use criteria to determine if eigenbasis update is needed
        adaptive_update_criteria_tolerance: Tolerance threshold for the update criteria.
            Only used if use_adaptive_criteria is True.
        power_iter_steps: Number of power iteration steps to perform before QR decomposition.
            More steps can lead to better convergence but increased computation time.
        max_update_rms: Clip the update RMS to this value (0 means no clipping).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 3e-3,
        betas: Optional[Tuple[float, float]] = None,
        shampoo_beta: float = -1,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        use_nesterov: bool = False,
        precondition_frequency: Union[int, Callable[[int], int]] = 10,
        precondition_warmup_steps: int = 0,
        adam_warmup_steps: int = 1,
        precondition_1d: bool = False,
        max_precond_dim: int = 8192,
        trace_normalization: bool = False,
        normalize_preconditioned_grads: bool = False,
        correct_bias: bool = True,
        fp32_matmul_prec: str = "high",
        use_eigh: bool = False,
        qr_fp32_matmul_prec: str = "high",
        use_adaptive_criteria: bool = False,
        adaptive_update_criteria_tolerance: Optional[float] = None,
        power_iter_steps: int = 1,
        max_update_rms: float = 0.0,
    ) -> None:

        # Check for betas.
        if betas is None:
            betas = (0.95, 0.95)
            logging.debug("betas not provided. Setting betas equal to " f"betas = {betas} by default.")

        # Check for update criteria
        if use_adaptive_criteria:
            if adaptive_update_criteria_tolerance is None:
                adaptive_update_criteria_tolerance = 1e-30
                logging.info(
                    "adaptive_update_criteria_tolerance not provided. Setting adaptive_update_criteria_tolerance equal to "
                    f"eps = {adaptive_update_criteria_tolerance} by default."
                )

        # Check for adam_warmup_steps since <1 will cause key errors in update_eigenbasis_and_momentum step
        if adam_warmup_steps < 1:
            adam_warmup_steps = 1
            logging.info("adam_warmup_steps is less than 1. Setting adam_warmup_steps to 1 by default.")

        # Check for precondition warmup steps and adam warmup steps
        if adam_warmup_steps >= precondition_warmup_steps and precondition_warmup_steps > 0:
            original_adam_warmup_steps = adam_warmup_steps
            adam_warmup_steps = max(1, precondition_warmup_steps - 1)
            logging.info(
                f"adam_warmup_steps ({original_adam_warmup_steps}) should be less than precondition_warmup_steps ({precondition_warmup_steps}). "
                f"Setting adam_warmup_steps to {adam_warmup_steps} by default."
            )

        defaults = {
            "lr": lr,
            "betas": betas,
            "shampoo_beta": shampoo_beta,
            "eps": eps,
            "weight_decay": weight_decay,
            "precondition_frequency": precondition_frequency,
            "precondition_warmup_steps": precondition_warmup_steps,
            "adam_warmup_steps": adam_warmup_steps,
            "precondition_1d": precondition_1d,
            "max_precond_dim": max_precond_dim,
            "trace_normalization": trace_normalization,
            "normalize_preconditioned_grads": normalize_preconditioned_grads,
            "use_nesterov": use_nesterov,
            "correct_bias": correct_bias,
            "use_decoupled_weight_decay": use_decoupled_weight_decay,
            "fp32_matmul_prec": fp32_matmul_prec,
            "use_eigh": use_eigh,
            "qr_fp32_matmul_prec": qr_fp32_matmul_prec,
            "use_adaptive_criteria": use_adaptive_criteria,
            "adaptive_update_criteria_tolerance": adaptive_update_criteria_tolerance,
            "power_iter_steps": power_iter_steps,
            "max_update_rms": max_update_rms,
        }
        super().__init__(params, defaults)

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
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                # State initialization
                # (TODO @mkhona): Better way to check state initialization - use state initializer?
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                # Initialize kronecker factor matrices
                if "GG" not in state:
                    state["GG"] = init_kronecker_factors(
                        grad,
                        precondition_1d=group["precondition_1d"],
                        max_precond_dim=group["max_precond_dim"],
                    )

                    # Update preconditioner matrices with gradient statistics, do not use shampoo_beta for EMA at first step
                    with utils.fp32_matmul_precision(group["fp32_matmul_prec"]):
                        update_kronecker_factors(
                            kronecker_factor_list=state["GG"],
                            grad=grad,
                            shampoo_beta=0.0,
                            precondition_1d=group["precondition_1d"],
                            max_precond_dim=group["max_precond_dim"],
                        )

                # Increment step counter
                state["step"] += 1

                # Apply weight decay
                if group["weight_decay"] > 0.0:
                    if group["use_decoupled_weight_decay"]:
                        # Apply decoupled weight decay
                        p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
                    else:
                        # add l2 regularization before preconditioning (i.e. like adding a squared loss term)
                        grad += group["weight_decay"] * p

                # Projecting gradients to the eigenbases of Shampoo's preconditioner
                torch.cuda.nvtx.range_push("precondition")
                with utils.fp32_matmul_precision(group["fp32_matmul_prec"]):
                    grad_projected = precondition(
                        grad=grad,
                        eigenbasis_list=state.get("Q"),
                        dims=[[0], [0]],
                    )
                torch.cuda.nvtx.range_pop()

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                # Calculate the Adam update for the projected gradient tensor
                torch.cuda.nvtx.range_push("calculate_adam_update")
                adam_update = calculate_adam_update(
                    grad_projected,
                    exp_avg,
                    exp_avg_sq,
                    group["betas"],
                    group["correct_bias"],
                    group["use_nesterov"],
                    state["step"],
                    group["eps"],
                )
                step_size = group["lr"]
                torch.cuda.nvtx.range_pop()

                # Projecting back the preconditioned (by ADAM) exponential moving average of gradients
                torch.cuda.nvtx.range_push("precondition")
                with utils.fp32_matmul_precision(group["fp32_matmul_prec"]):
                    norm_precond_grad = precondition(
                        grad=adam_update,
                        eigenbasis_list=state.get("Q"),
                        dims=[[0], [1]],
                    )
                torch.cuda.nvtx.range_pop()

                if group["trace_normalization"]:
                    if state["GG"][0].numel() > 0:
                        trace_normalization = 1 / torch.sqrt(torch.trace(state["GG"][0]))
                        norm_precond_grad = norm_precond_grad / trace_normalization

                if group["normalize_preconditioned_grads"]:
                    norm_precond_grad = norm_precond_grad / (1e-30 + torch.mean(norm_precond_grad**2) ** 0.5)

                # Clip the update RMS to a maximum value
                _clip_update_rms_in_place(norm_precond_grad, group["max_update_rms"])

                torch.cuda.nvtx.range_push("weight update")
                p.add_(norm_precond_grad, alpha=-step_size)
                torch.cuda.nvtx.range_pop()

                # Update kronecker factor matrices with gradient statistics
                shampoo_beta = group["shampoo_beta"] if group["shampoo_beta"] >= 0 else group["betas"][1]
                if group["correct_bias"]:
                    # step size correction for shampoo kronecker factors EMA
                    shampoo_beta = 1 - (1 - shampoo_beta) / (1 - shampoo_beta ** (state["step"] + 1))

                torch.cuda.nvtx.range_push("update_kronecker_factors")
                with utils.fp32_matmul_precision(group["fp32_matmul_prec"]):
                    update_kronecker_factors(
                        kronecker_factor_list=state["GG"],
                        grad=grad,
                        shampoo_beta=shampoo_beta,
                        precondition_1d=group["precondition_1d"],
                        max_precond_dim=group["max_precond_dim"],
                    )
                torch.cuda.nvtx.range_pop()

                # If current step is the last step to skip preconditioning, initialize eigenbases and end first order warmup
                if state["step"] == group["adam_warmup_steps"]:
                    # Obtain kronecker factor eigenbases from kronecker factor matrices using eigendecomposition
                    state["Q"] = get_eigenbasis_eigh(state["GG"])
                    # rotate momentum to the new eigenbasis
                    with utils.fp32_matmul_precision(group["fp32_matmul_prec"]):
                        state["exp_avg"] = precondition(
                            grad=state["exp_avg"],
                            eigenbasis_list=state["Q"],
                            dims=[[0], [0]],
                        )
                    continue

                # Update eigenbases at precondition_frequency steps or until precondition_warmup_steps is done,
                # but only after the adam_warmup_steps are completed.
                torch.cuda.nvtx.range_push("Update eigen basis")
                if _is_eigenbasis_update_step(
                    state["step"],
                    group["adam_warmup_steps"],
                    group["precondition_warmup_steps"],
                    group["precondition_frequency"],
                ):
                    with utils.fp32_matmul_precision(group["qr_fp32_matmul_prec"]):
                        state["Q"], state["exp_avg"], state["exp_avg_sq"] = update_eigenbasis_and_momentum(
                            kronecker_factor_list=state["GG"],
                            eigenbasis_list=state["Q"],
                            exp_avg_sq=state["exp_avg_sq"],
                            momentum=state["exp_avg"],
                            use_eigh=group["use_eigh"],
                            use_adaptive_criteria=group["use_adaptive_criteria"],
                            adaptive_update_criteria_tolerance=group["adaptive_update_criteria_tolerance"],
                            power_iter_steps=group["power_iter_steps"],
                        )
                torch.cuda.nvtx.range_pop()

        return loss


@torch.no_grad()  # type: ignore[misc]
def init_kronecker_factors(
    grad: torch.Tensor,
    precondition_1d: bool = False,
    max_precond_dim: int = 8192,
) -> List[torch.Tensor]:
    """Initializes the kronecker factor matrices for the SOAP optimizer.

    This function creates the initial Kronecker factor matrices (L and R) used for
    preconditioning. For 1D tensors (like biases), it can either skip preconditioning
    or create a single square kronecker factor matrix. For higher dimensional tensors,
    it creates a square kronecker factor matrix for each dimension.

    When precondition_1d is:
        * False (default):
            - 1D tensors (like biases) will skip SOAP preconditioning entirely
            - These parameters will use standard Adam-style updates
            - This is often desirable as biases typically have fewer parameters and simpler optimization landscapes
            - Can improve performance and reduce memory usage
        * True:
            - All parameters, including 1D tensors, will use SOAP preconditioning
            - May be beneficial for certain architectures or training scenarios

    Args:
        grad: Gradient tensor used to initialize the kronecker factor matrices.
            The shape of this tensor determines the size of the kronecker factor matrices.
        precondition_1d: Whether to create kronecker factor matrices for 1D tensors
            (like biases). If False, 1D tensors will skip preconditioning.
        max_precond_dim: Maximum dimension of the preconditioner matrices.
            Skips preconditioning if any tensor dimension exceeds.

    Returns:
        List[torch.Tensor]: List of kronecker factor matrices (L and R in paper).
            - For 1D tensors with precondition_1d=False: List containing an empty tensor
            - For 1D tensors with precondition_1d=True: List containing a square matrix
            - For higher dimensional tensors: List of square matrices, one per dimension

    Example:
        >>> # For a 1D tensor (bias)
        >>> grad_1d = torch.randn(10)
        >>> precond_1d = init_kronecker_factors(grad_1d, precondition_1d=True)
        >>> print(len(precond_1d))  # 1
        >>> print(precond_1d[0].shape)  # (10, 10)

        >>> # For a 2D tensor (weight matrix)
        >>> grad_2d = torch.randn(10, 20)
        >>> precond_2d = init_kronecker_factors(grad_2d)
        >>> print(len(precond_2d))  # 2
        >>> print(precond_2d[0].shape)  # (10, 10)
        >>> print(precond_2d[1].shape)  # (20, 20)

    """
    kronecker_factor_list: List[torch.Tensor] = []

    if grad.dim() == 1:
        if not precondition_1d:
            # Skip preconditioning for 1D tensors
            kronecker_factor_list.append(torch.empty(0, device=grad.device))
        else:
            # Create a square preconditioner matrix for 1D tensors
            size = grad.shape[0]
            if size > max_precond_dim:
                # if tensor dimension is larger than max_precond_dim, skip preconditioning this dimension
                # append empty tensor to kronecker_factor_list so that subsequent check that use numel() to check if preconditioner is initialized will not fail
                kronecker_factor_list.append(torch.empty(0, device=grad.device))
            else:
                kronecker_factor_list.append(torch.zeros(size, size, device=grad.device))
    else:
        # Create a square kronecker factor matrix for each dimension
        for size in grad.shape:
            if size > max_precond_dim:
                # append empty tensor to kronecker_factor_list so that subsequent check that use numel() to check if preconditioner is initialized will not fail
                # skip preconditioning this dimension
                kronecker_factor_list.append(torch.empty(0, device=grad.device))
            else:
                kronecker_factor_list.append(torch.zeros(size, size, device=grad.device))

    return kronecker_factor_list


@torch.no_grad()  # type: ignore[misc]
def update_kronecker_factors(
    kronecker_factor_list: List[torch.Tensor],
    grad: torch.Tensor,
    shampoo_beta: float,
    precondition_1d: bool = False,
    max_precond_dim: int = 8192,
) -> None:
    """Updates the preconditioner matrices using gradient outer products.

    This function updates the Kronecker factor matrices (L and R) used for preconditioning
    by computing and accumulating gradient outer products. For 1D tensors (like biases),
    it can optionally skip preconditioning or use a special 1D preconditioning strategy.
    It modifies the kronecker_factor_list in place.

    Args:
        kronecker_factor_list: List of preconditioner matrices (L and R) to update.
            Each matrix should be square and match the corresponding dimension of grad.
        grad: Gradient tensor of the parameter being optimized
        shampoo_beta: Momentum coefficient for updating preconditioners.
            Controls how much weight to give to new vs old gradient statistics.
        precondition_1d: Whether to apply preconditioning to 1D tensors (like biases).
            If False, 1D tensors will skip preconditioning.
        max_precond_dim: Maximum dimension of the preconditioner matrices.
            Skips preconditioning if any tensor dimension exceeds.

    Example:
        >>> grad = torch.randn(10, 20)
        >>> L = torch.zeros(10, 10)
        >>> R = torch.zeros(20, 20)
        >>> update_preconditioner([L, R], grad, shampoo_beta=0.95)

    """
    if grad.dim() == 1:
        if precondition_1d:
            # For 1D tensors, compute outer product directly
            outer_product = grad.unsqueeze(1) @ grad.unsqueeze(0)
            kronecker_factor_list[0].lerp_(outer_product, 1 - shampoo_beta)
        else:
            # For 1D tensors, skip preconditioning
            return
    else:
        # For higher dimensional tensors, compute outer products for each dimension
        for idx, dim_size in enumerate(grad.shape):
            if dim_size <= max_precond_dim:
                # Compute outer product by contracting all dimensions except idx
                contract_dims = [*chain(range(idx), range(idx + 1, grad.dim()))]
                outer_product = torch.tensordot(
                    grad,
                    grad,
                    dims=[contract_dims] * 2,
                )
                # Update the corresponding Kronecker factor
                kronecker_factor_list[idx].lerp_(outer_product, 1 - shampoo_beta)


@torch.no_grad()  # type: ignore[misc]
def update_eigenbasis_and_momentum(
    kronecker_factor_list: List[torch.Tensor],
    eigenbasis_list: List[torch.Tensor],
    exp_avg_sq: torch.Tensor,
    momentum: torch.Tensor,
    use_eigh: bool = False,
    use_adaptive_criteria: bool = False,
    adaptive_update_criteria_tolerance: Optional[float] = None,
    power_iter_steps: int = 1,
    convert_to_float: bool = True,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Updates the eigenbases using QR decomposition and power iteration or eigh.

    This function performs an update of the eigenbases (QL and QR)
    used for preconditioning. It follows these steps:

    1. Projects momentum back to the original basis
    2. Updates the eigenbases using QR decomposition and power iteration
    3. Projects momentum back to the new eigenbasis

    Args:
        kronecker_factor_list: List of preconditioner matrices (L and R) that define
            the optimization landscape. These are updated with gradient statistics.
        eigenbasis_list: List of current eigenbases (QL and QR)
            used for preconditioning. These will be updated by this function.
        exp_avg_sq: Inner Adam's second moment tensor, used for scaling the preconditioner updates.
            This tensor is modified in-place.
        momentum: Inner Adam's first moment tensor, used for tracking gradient momentum.
            This tensor is modified in-place.
        use_eigh: Whether to use full symmetric eigendecomposition (eigh) to compute the eigenbasis.
            If False, use orthogonal iteration to compute the eigenbasis.
        use_adaptive_criteria: Whether to use criteria to determine if eigenbasis update is needed
        adaptive_update_criteria_tolerance: Tolerance threshold for the update criteria.
            Only used if use_adaptive_criteria is True.
        power_iter_steps: Number of power iteration steps to perform before QR decomposition.
            More steps can lead to better convergence but increased computation time.
        convert_to_float: Whether to convert the preconditioner matrices and their corresponding
            orthonormal matrices to float for amortized computation. Otherwise, they are left in their original type.

    Returns:
        Tuple[List[torch.Tensor], torch.Tensor]: A tuple containing:
            - List[torch.Tensor]: Updated list of eigenbases (QL and QR)
            - torch.Tensor: Updated momentum tensor projected to the new eigenbasis

    Example:
        >>> L = torch.randn(10, 10)
        >>> R = torch.randn(20, 20)
        >>> QL = torch.randn(10, 10)
        >>> QR = torch.randn(20, 20)
        >>> exp_avg_sq = torch.randn(10, 20)
        >>> momentum = torch.randn(10, 20)
        >>> updated_eigenbases = update_eigenbasis(
        ...     [L, R], [QL, QR], exp_avg_sq, momentum)

    """
    # Step 1: Project momentum back to the original basis
    torch.cuda.nvtx.range_push("eigenbasis update step 1: precondition")
    momentum = precondition(
        momentum,
        eigenbasis_list,
        dims=[[0], [1]],  # Project back to original space
    )
    torch.cuda.nvtx.range_pop()

    # Step 2: Update eigenbases
    torch.cuda.nvtx.range_push("eigenbasis update step 2: update Q")
    if use_eigh:
        updated_eigenbasis_list = get_eigenbasis_eigh(
            kronecker_factor_list,
            convert_to_float,
        )
    else:
        # Use QR decomposition and power iteration (orthogonal iteration)
        updated_eigenbasis_list, exp_avg_sq = get_eigenbasis_qr(
            kronecker_factor_list,
            eigenbasis_list,
            exp_avg_sq,
            convert_to_float,
            use_adaptive_criteria,
            adaptive_update_criteria_tolerance,
            power_iter_steps,
        )
    torch.cuda.nvtx.range_pop()

    # Step 3: Project momentum to the new eigenbasis using the updated eigenbases
    torch.cuda.nvtx.range_push("eigenbasis update step 3: project momentum")
    momentum = precondition(
        momentum,
        updated_eigenbasis_list,  # Use the new eigenbases
        dims=[[0], [0]],  # Project to new eigenbasis
    )
    torch.cuda.nvtx.range_pop()

    return updated_eigenbasis_list, momentum, exp_avg_sq


@torch.no_grad()  # type: ignore[misc]
@torch.compile  # type: ignore[misc]
def precondition(
    grad: torch.Tensor,
    eigenbasis_list: Optional[List[torch.Tensor]],
    dims: Optional[List[List[int]]] = None,
) -> torch.Tensor:
    """Projects the gradient to and from the eigenbases of the kronecker factor matrices.

    This function performs tensor contractions between the input gradient
    and kronecker factor eigenbases.


    Args:
        grad: Input tensor to be preconditioned
        eigenbasis_list: List of eigenbases for preconditioning.
            Each matrix should be a square matrix of eigenvectors.
        dims: Dimensions for tensor contraction. Default is [[0], [0]] which contracts
            the first dimension of grad with the first dimension of each eigenbasis matrix,
            for projecting into the eigenbasis. Use [[0], [1]] for projecting back to original space.

    Example:
        >>> grad = torch.randn(10, 20)
        >>> Q = torch.randn(10, 10)
        >>> precondition(grad, [Q], dims=[[0], [0]])
    """
    if dims is None:
        # Pick contraction dims to project to the eigenbasis
        dims = [[0], [0]]

    if not eigenbasis_list:
        # If eigenbases are not provided, return the gradient without any preconditioning
        return grad

    for Q in eigenbasis_list:
        if Q.numel() > 0:
            # Perform in-place contraction
            grad = torch.tensordot(
                grad,
                Q,
                dims=dims,
            )
        else:
            # Permute gradient dimensions to process the next dimension in the following iteration
            # when preconditioning for the current dimension is skipped (Q is empty), in the case of one-sided preconditioning.
            permute_order = list(range(1, grad.dim())) + [0]
            grad = grad.permute(permute_order)

    return grad


def _get_precondition_frequency(precondition_frequency: Union[int, Callable[[int], int]], step: int) -> int:
    """Get the current precondition frequency based on the schedule or fixed value.

    Args:
        precondition_frequency: Either an integer for fixed frequency or a callable that takes step and returns frequency
        step: Current optimization step

    Returns:
        The precondition frequency for the current step
    """
    if callable(precondition_frequency):
        return precondition_frequency(step)
    else:
        return precondition_frequency


def _is_eigenbasis_update_step(
    step: int,
    adam_warmup_steps: int,
    precondition_warmup_steps: int,
    precondition_frequency: Union[int, Callable[[int], int]],
) -> bool:
    """Checks if amortized computation of the eigenbasis should be recomputed.

    Args:
        step: Current step of the optimizer
        adam_warmup_steps: Number of steps to skip preconditioning in the beginning (i.e. use standard AdamW updates)
        precondition_warmup_steps: How many steps to warm up the preconditioner (i.e. update every step)
        precondition_frequency: How often to update the preconditioner. Can be an integer for fixed frequency
            or a callable function that takes the current step as input and returns the frequency.
    """
    if step <= adam_warmup_steps:
        return False

    # During warmup period, update every step
    if step <= precondition_warmup_steps:
        return True

    # After warmup, use the scheduled frequency
    current_frequency = _get_precondition_frequency(precondition_frequency, step)
    return step % current_frequency == 0


@torch.compile  # type: ignore[misc]
def _clip_update_rms_in_place(u: torch.Tensor, max_rms: float = 1.0, eps: float = 1e-12) -> None:
    """Clip the update root mean square (RMS) to a maximum value, in place.

    Do not clip if max_rms is 0.
    Inspired by Adafactor (https://arxiv.org/abs/1804.04235) and RMS_t (https://arxiv.org/abs/2304.13013)

    Args:
        u: The update tensor.
        max_rms: The maximum RMS value.
        eps: The epsilon value to prevent division by zero.
    """
    if max_rms == 0:
        return
    # compute current update RMS
    rms = u.square().mean().sqrt()
    # compute scale factor = min(1.0, max_rms/(rms + eps))
    scale = (max_rms / (rms + eps)).clamp(max=1.0)
    # in‐place scale
    u.mul_(scale)
