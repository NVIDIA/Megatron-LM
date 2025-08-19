import torch
from typing import Optional

__all__ = [
    "calculate_lion_update",
]


@torch.compile  # type: ignore[misc]
@torch.no_grad()  # type: ignore[misc]
def calculate_lion_update(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    momentum_beta: float,
    momentum_beta2: Optional[float] = None,
) -> torch.Tensor:
    """Performs the Lion update.

    This function performs the computation of 1 step of Lion update.

    The update rule is as follows:

    .. math::
        \\text{update} = \\text{sign}(\\beta_1 m_{t-1} + (1 - \\beta_1) g_t) \\\\
        m_t = \\beta_2 m_{t-1} + (1 - \\beta_2) g_t

    Args:
        grad: The gradient tensor.
        exp_avg: The accumulated first moment of the gradient.
        momentum_beta: The EMA beta coefficients for the momentum update (beta1 in Lion).
        momentum_beta2: The second EMA beta coefficient for Lion momentum update.

    Returns:
        The Lion update.
    """

    # Lion update: interpolate before sign, update momentum after
    if momentum_beta2 is None:
        momentum_beta2 = momentum_beta

    # Compute update using interpolation (like Lion's beta1)
    update_momentum = momentum_beta * exp_avg + (1 - momentum_beta) * grad

    # Update the momentum state (Lion's beta2)
    exp_avg.lerp_(grad, 1 - momentum_beta2)

    # Return signed update (no shape scaling for Lion)
    return torch.sign(update_momentum)
