import torch
from typing import Tuple


__all__ = [
    "calculate_laprop_update",
]


@torch.compile  # type: ignore[misc]
@torch.no_grad()  # type: ignore[misc]
def calculate_laprop_update(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    correct_bias: bool,
    betas: Tuple[float, float],
    step: int,
    eps: float,
) -> torch.Tensor:
    """Performs the LAProp/Normalized SGD with momentum update.

    LAProp can be seen as RMSProp with a momentum term, or normalized SGD with momentum.
    Based on https://github.com/Z-T-WANG/LaProp-Optimizer/blob/master/laprop.py
    and https://arxiv.org/abs/2002.04839.

    The update rule is as follows:

    .. math::
        v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2 \\\\
        \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t} \\\\
        g'_t = \\frac{g_t}{\\sqrt{\\hat{v}_t} + \\epsilon} \\\\
        m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g'_t \\\\
        \\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t} \\\\
        \\text{update} = \\hat{m}_t

    Args:
        grad: The gradient tensor.
        exp_avg: The exponential moving average of the gradient.
        exp_avg_sq: The exponential moving average of the gradient squared.
        correct_bias: Whether to correct the bias of the Adam update.
        betas: The betas for the exponential moving average.
        step: The current step.
        eps: The epsilon for the second moment update.

    Returns:
        The LAProp update.
    """
    beta1, beta2 = betas

    # Decay the second moment running average coefficient
    exp_avg_sq.lerp_(grad.square(), 1 - beta2)

    # step size correction for optimizer states EMA
    bias_correction1 = 1.0
    bias_correction2 = 1.0
    if correct_bias:
        # step size correction for ADAM moments EMA
        bias_correction1 = 1.0 - beta1 ** (step)
        bias_correction2 = 1.0 - beta2 ** (step)

    # construct the denominator of the inner ADAM optimizer
    second_moment = exp_avg_sq / bias_correction2
    second_moment = second_moment.sqrt() + eps

    normalized_grad = grad / second_moment

    # update the exponential moving average of the gradient
    exp_avg.lerp_(normalized_grad, 1 - beta1)

    # return the LAProp update
    return exp_avg / bias_correction1
