# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""AngularMuOwn with sharded state and owner-compute orthogonalization for MFSDP v2.

AngularMuOwn writes every row of a 2-D weight as ``W_i = g_i * U_i`` with a signed row scale
``g_i`` and a unit row direction ``U_i``. Adam updates the row scales. The directions take a
Muon step on the product of unit spheres, followed by row normalization, so each step rotates
every row by an angle of about ``lr * kappa_t``.

Only the Newton-Schulz orthogonalization needs whole matrices. Everything else works on
complete local rows, which every MFSDP v2 placement keeps on one rank. ``FsdpAngularMuOwn``
therefore reuses ``FsdpMuon``'s owner gather, Newton-Schulz, and scatter unchanged and replaces
only the local work before the gather and after the scatter.

References:
    - Hübler, Lion, Orvieto, and He, *Muown Implicitly Performs Angular Step-size Decay*,
      arXiv:2606.23637 (2026).
    - Lion, Hübler, Li, Orvieto, and He, *Muown: Row-Norm Control for Muon Optimization*,
      arXiv:2605.10797 (2026).
"""

from collections.abc import Callable
from typing import Any, cast

import torch
from torch.distributed.tensor import DTensor

from .dbuffer import DBuffer
from .muon import FsdpMuon
from .parameter_group import FsdpParameterGroup

__all__ = ["FsdpAngularMuOwn", "angular_multiplier"]


def angular_multiplier(
    step: int,
    *,
    warmup_steps: int,
    horizon_steps: int,
    decay_scale: float,
    decay_degree: float,
    plateau_degree: float,
) -> float:
    """Return the angular step-size multiplier ``kappa_t``.

    ``kappa_t = (1 + C * u**r) ** (-p / r)`` with progress ``u = (t - t_w) / T``. The
    multiplier is 1 through warmup and when the decay is disabled by ``C = 0``, ``p = 0``, or
    ``T = 0``.

    Args:
        step: One-based optimizer step ``t``.
        warmup_steps: Steps ``t_w`` that keep ``kappa_t = 1``.
        horizon_steps: Steps ``T`` that normalize the progress.
        decay_scale: Dimensionless decay scale ``C``.
        decay_degree: Decay degree ``p``.
        plateau_degree: Plateau degree ``r >= 1``. ``r > 1`` flattens the start of the decay.
    """
    if step <= warmup_steps or horizon_steps <= 0 or decay_scale == 0 or decay_degree == 0:
        return 1.0
    progress = (step - warmup_steps) / horizon_steps
    return (1.0 + decay_scale * progress**plateau_degree) ** (-decay_degree / plateau_degree)


def _safe_row_scale(row_scale: torch.Tensor, norm_eps: float) -> torch.Tensor:
    """Clamp row scales away from zero while keeping their signs, with +0 mapped to +eps."""
    return torch.copysign(row_scale.abs().clamp_min(norm_eps), row_scale)


def _prepare_direction_update(
    weight: torch.Tensor,
    grad: torch.Tensor,
    row_scale: torch.Tensor,
    row_scale_exp_avg: torch.Tensor,
    row_scale_exp_avg_sq: torch.Tensor,
    momentum: torch.Tensor,
    out: torch.Tensor,
    *,
    momentum_coefficient: float,
    beta1: float,
    beta2: float,
    nesterov: bool,
    norm_eps: float,
) -> None:
    """Split local rows' gradients, update the moments, and write the Newton-Schulz input.

    With ``U = W / g``, the radial gradient ``d = <G, U>`` (per row) drives the row scale and the
    tangential gradient ``g * (G - d * U)`` drives the direction. Adam's moments for ``g`` are
    updated here so that nothing but the optimizer state carries over to the update step.

    Args:
        weight: Local rows of the FP32 main weight, ``(k, n)``.
        grad: Local rows of the gradient in FP32, ``(k, n)``.
        row_scale: Signed row scales ``g``, ``(k, 1)``.
        row_scale_exp_avg: Adam first moment of ``g``, updated in place.
        row_scale_exp_avg_sq: Adam second moment of ``g``, updated in place.
        momentum: Raw-sum direction momentum ``M``, updated in place.
        out: Receives the Newton-Schulz input, ``(k, n)``.
        momentum_coefficient: Momentum coefficient ``beta``.
        beta1: Adam beta1 for ``g``.
        beta2: Adam beta2 for ``g``.
        nesterov: Whether the Newton-Schulz input is ``dL/dU + beta * M`` instead of ``M``.
        norm_eps: Floor for ``|g|`` when recovering ``U``.
    """
    direction = weight / _safe_row_scale(row_scale, norm_eps)
    radial_grad = (grad * direction).sum(dim=1, keepdim=True)
    # out = g * (G - d * U), the direction gradient on the product of unit spheres.
    torch.addcmul(grad, direction, radial_grad, value=-1.0, out=out)
    out.mul_(row_scale)

    row_scale_exp_avg.lerp_(radial_grad, 1 - beta1)
    row_scale_exp_avg_sq.mul_(beta2).addcmul_(radial_grad, radial_grad, value=1 - beta2)

    # Raw-sum momentum. Newton-Schulz normalizes its input, so this matches Muon's EMA momentum
    # up to the positive factor 1 / (1 - beta).
    momentum.mul_(momentum_coefficient).add_(out)
    if nesterov:
        out.add_(momentum, alpha=momentum_coefficient)
    else:
        out.copy_(momentum)


def _apply_direction_update(
    weight: torch.Tensor,
    update: torch.Tensor,
    row_scale: torch.Tensor,
    row_scale_exp_avg: torch.Tensor,
    row_scale_exp_avg_sq: torch.Tensor,
    *,
    lr: float,
    angular_lr: float,
    weight_decay: float,
    step: int,
    beta1: float,
    beta2: float,
    adam_eps: float,
    norm_eps: float,
) -> None:
    """Rotate local unit rows, retract them, update the row scales, and rewrite the weight.

    Args:
        weight: Local rows of the FP32 main weight, ``(k, n)``, rewritten in place.
        update: Local rows of the scaled orthogonalized update ``O``, ``(k, n)``.
        row_scale: Signed row scales ``g``, ``(k, 1)``, updated in place.
        row_scale_exp_avg: Adam first moment of ``g``, already updated for this step.
        row_scale_exp_avg_sq: Adam second moment of ``g``, already updated for this step.
        lr: Learning rate of the row-scale Adam step and the decoupled row-scale decay.
        angular_lr: Direction step ``lr * kappa_t``.
        weight_decay: Decoupled weight decay of the row scales.
        step: One-based step for Adam's bias corrections.
        beta1: Adam beta1 for ``g``.
        beta2: Adam beta2 for ``g``.
        adam_eps: Adam epsilon for ``g``.
        norm_eps: Floor for ``|g|`` and for the retraction denominator.
    """
    direction = weight / _safe_row_scale(row_scale, norm_eps)
    direction.add_(update, alpha=-angular_lr)
    row_norm = torch.linalg.vector_norm(direction, dim=1, keepdim=True)
    denominator = row_norm.clamp_min(norm_eps)

    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    adam_update = (row_scale_exp_avg / bias_correction1) / (
        (row_scale_exp_avg_sq / bias_correction2).sqrt() + adam_eps
    )
    if weight_decay != 0:
        row_scale.mul_(1 - lr * weight_decay)
    row_scale.sub_(adam_update, alpha=lr)

    torch.mul(direction, row_scale / denominator, out=weight)
    # A clamped retraction leaves rows shorter than |g|; rescale g so |g_i| = ||W_i|| holds.
    row_scale.mul_(row_norm / denominator)


def _row_state(param: DTensor, local: torch.Tensor) -> DTensor:
    """Wrap per-row local values ``(k, 1)`` as an ``(m, 1)`` DTensor sharded like ``param``."""
    return DTensor.from_local(
        local,
        device_mesh=param.device_mesh,
        placements=param.placements,
        run_check=False,
        shape=torch.Size((param.shape[0], 1)),
        stride=(1, 1),
    )


class FsdpAngularMuOwn(FsdpMuon):
    """AngularMuOwn for 2-D parameters with sharded optimizer state on MFSDP v2.

    One step for each matrix, with ``U = W / g``:

    1. Locally, before the gather: split the gradient into the radial part ``d = <G, U>`` and
       the direction gradient ``g * (G - d * U)``, update Adam's moments of ``g`` and the raw-sum
       momentum ``M``, and write the (Nesterov) momentum into the Newton-Schulz input.
    2. ``FsdpMuon``: gather the input to the owner, orthogonalize it with
       ``scale_mode="shape_scaling"``, and scatter ``O`` back.
    3. Locally, after the scatter: ``U <- rownorm(U - lr * kappa_t * O)``,
       ``g <- (1 - lr * weight_decay * weight_decay_mult) * g - lr * Adam(d)``, and
       ``W <- g * U``.

    Optimizer state per parameter: ``momentum_buffer`` (``m x n``) and ``row_scale``,
    ``row_scale_exp_avg``, and ``row_scale_exp_avg_sq`` (``m x 1``), all FP32 DTensors sharded
    like the parameter. ``row_scale`` starts at the row norms of the weight; zero rows stay zero.
    Each parameter group counts its steps in ``muown_step``, which drives ``kappa_t`` and Adam's
    bias corrections and round-trips with the group hyperparameters.

    Args:
        *args: Positional arguments passed unchanged to ``FsdpMuon``.
        angular_decay_scale: Dimensionless scale ``C`` of ``kappa_t``.
        angular_decay_degree: Degree ``p`` of ``kappa_t``. Zero disables the decay.
        angular_plateau_degree: Plateau degree ``r >= 1`` of ``kappa_t``.
        angular_warmup_steps: Steps ``t_w`` that keep ``kappa_t = 1``.
        angular_horizon_steps: Steps ``T`` that normalize the progress of ``kappa_t``, usually
            the number of training iterations. Zero disables the decay.
        row_scale_betas: Adam betas for the row scales.
        row_scale_eps: Adam epsilon for the row scales.
        norm_eps: Floor for ``|g|`` when recovering ``U`` and for the retraction denominator.
        weight_decay_mult: Multiplier of each group's ``weight_decay`` for the decoupled decay of
            the row scales. The directions are never decayed.
        **kwargs: Keyword arguments passed to ``FsdpMuon``. ``scale_mode`` defaults to, and must
            be, ``"shape_scaling"``, so that ``kappa_t`` means the same rotation in every layer.
    """

    def __init__(
        self,
        *args: Any,
        angular_decay_scale: float = 8.9,
        angular_decay_degree: float = 1.0,
        angular_plateau_degree: float = 1.0,
        angular_warmup_steps: int = 0,
        angular_horizon_steps: int = 0,
        row_scale_betas: tuple[float, float] = (0.9, 0.95),
        row_scale_eps: float = 1e-8,
        norm_eps: float = 1e-12,
        weight_decay_mult: float = 1.0,
        **kwargs: Any,
    ) -> None:
        scale_mode = kwargs.setdefault("scale_mode", "shape_scaling")
        if scale_mode != "shape_scaling":
            raise ValueError(
                "FsdpAngularMuOwn requires scale_mode='shape_scaling' so that the angular "
                f"multiplier gives every layer the same rotation, got {scale_mode!r}."
            )
        weight_decay_method = kwargs.get("weight_decay_method", "decoupled")
        if weight_decay_method != "decoupled":
            raise ValueError(
                "FsdpAngularMuOwn decays the row scales with decoupled weight decay only, got "
                f"weight_decay_method={weight_decay_method!r}."
            )
        if angular_decay_scale < 0 or angular_decay_degree < 0:
            raise ValueError("angular_decay_scale and angular_decay_degree must be non-negative.")
        if angular_plateau_degree < 1:
            raise ValueError(f"angular_plateau_degree must be >= 1, got {angular_plateau_degree}.")
        if angular_warmup_steps < 0 or angular_horizon_steps < 0:
            raise ValueError("angular_warmup_steps and angular_horizon_steps must be non-negative.")
        if not all(0 <= beta < 1 for beta in row_scale_betas):
            raise ValueError(f"row_scale_betas must be in [0, 1), got {row_scale_betas}.")
        if row_scale_eps < 0 or norm_eps <= 0:
            raise ValueError("row_scale_eps must be non-negative and norm_eps positive.")
        super().__init__(*args, **kwargs)

        self.angular_decay_scale = angular_decay_scale
        self.angular_decay_degree = angular_decay_degree
        self.angular_plateau_degree = angular_plateau_degree
        self.angular_warmup_steps = angular_warmup_steps
        self.angular_horizon_steps = angular_horizon_steps
        self.row_scale_betas = row_scale_betas
        self.row_scale_eps = row_scale_eps
        self.norm_eps = norm_eps
        self.weight_decay_mult = weight_decay_mult
        # The most recent kappa_t, for logging.
        self.last_angular_multiplier = 1.0
        for group in self.param_groups:
            group.setdefault("muown_step", 0)
        self._active_group_indices: set[int] = set()

    def angular_multiplier(self, step: int) -> float:
        """Return ``kappa_t`` for the one-based ``step`` with this optimizer's schedule."""
        return angular_multiplier(
            step,
            warmup_steps=self.angular_warmup_steps,
            horizon_steps=self.angular_horizon_steps,
            decay_scale=self.angular_decay_scale,
            decay_degree=self.angular_decay_degree,
            plateau_degree=self.angular_plateau_degree,
        )

    @torch.no_grad()
    def _init_state(self, param: DTensor, state: dict[str, Any]) -> None:
        """Allocate the momentum and row states, with ``g`` set to the current row norms."""
        state["momentum_buffer"] = torch.zeros_like(param)
        row_norm = torch.linalg.vector_norm(param.to_local().float(), dim=1, keepdim=True)
        state["row_scale"] = _row_state(param, row_norm)
        state["row_scale_exp_avg"] = _row_state(param, torch.zeros_like(row_norm))
        state["row_scale_exp_avg_sq"] = _row_state(param, torch.zeros_like(row_norm))

    @torch.no_grad()
    def _init_group(self, group: dict, skip_non_grad_params: bool = True) -> None:
        """Lazily initialize the state of a parameter group's matrices."""
        for param in group["params"]:
            if param.ndim != 2 or (skip_non_grad_params and param.grad is None):
                continue
            state = self.state[param]
            if not state:
                self._init_state(cast(DTensor, param), state)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> None:
        """Update the sharded main weights and advance the active groups' step counters."""
        self._active_group_indices = set()
        super().step(closure)
        for group_index in sorted(self._active_group_indices):
            self.param_groups[group_index]["muown_step"] += 1

    def _compute_momentum_inputs(
        self, fsdp_group: FsdpParameterGroup, tensor_to_optimizer_group_index: dict[int, int]
    ) -> DBuffer:
        """Split gradients into radial and direction parts and build Newton-Schulz inputs."""
        buffer: DBuffer = fsdp_group.main_weight
        momentum_inputs: DBuffer = DBuffer(
            buffer.mesh, buffer.placements, buffer.layout, buffer.dtype, buffer.device
        )
        beta1, beta2 = self.row_scale_betas
        for i, group_index in tensor_to_optimizer_group_index.items():
            group: dict[str, Any] = self.param_groups[group_index]
            # Every rank sees the same active tensors, even ranks without local rows, so this
            # records the same kappa_t everywhere.
            self._active_group_indices.add(group_index)
            self.last_angular_multiplier = self.angular_multiplier(group["muown_step"] + 1)
            param: DTensor = cast(DTensor, fsdp_group.fsdp_parameters[i].sharded)
            assert isinstance(param.grad, DTensor)
            state: dict[str, Any] = self.state[param]
            if not state:
                self._init_state(param, state)
            momentum: torch.Tensor = state["momentum_buffer"].to_local()
            if momentum.numel() == 0:
                continue
            _prepare_direction_update(
                param.to_local(),
                param.grad.to_local().to(momentum.dtype).reshape(momentum.shape),
                state["row_scale"].to_local(),
                state["row_scale_exp_avg"].to_local(),
                state["row_scale_exp_avg_sq"].to_local(),
                momentum,
                momentum_inputs.get_tensor_view(i).view_as(momentum),
                momentum_coefficient=group["momentum"],
                beta1=beta1,
                beta2=beta2,
                nesterov=self.nesterov,
                norm_eps=self.norm_eps,
            )
        return momentum_inputs

    def _apply_local_updates(
        self,
        fsdp_group: FsdpParameterGroup,
        tensor_to_optimizer_group_index: dict[int, int],
        full_updates: dict[int, torch.Tensor],
        local_updates: DBuffer,
    ) -> None:
        """Rotate, retract, and rescale local rows with owned slices and received updates."""
        buffer: DBuffer = fsdp_group.main_weight
        beta1, beta2 = self.row_scale_betas
        for i, group_index in tensor_to_optimizer_group_index.items():
            param: DTensor = cast(DTensor, fsdp_group.fsdp_parameters[i].sharded)
            weight: torch.Tensor = param.to_local()
            if weight.numel() == 0:
                continue
            # Select this rank's rows of O as FsdpMuon._apply_local_updates does.
            if i in full_updates:
                offset: int = max(buffer.offset - buffer.layout.get_tensor_range(i).start, 0)
                update: torch.Tensor = full_updates[i].view(-1).narrow(0, offset, weight.numel())
            else:
                update = local_updates.get_tensor_view(i)
            group: dict[str, Any] = self.param_groups[group_index]
            state: dict[str, Any] = self.state[param]
            step: int = group["muown_step"] + 1
            lr: float = group["lr"]
            kappa: float = self.angular_multiplier(step)
            _apply_direction_update(
                weight,
                update.reshape(weight.shape).to(weight.dtype),
                state["row_scale"].to_local(),
                state["row_scale_exp_avg"].to_local(),
                state["row_scale_exp_avg_sq"].to_local(),
                lr=lr,
                angular_lr=lr * kappa,
                weight_decay=group["weight_decay"] * self.weight_decay_mult,
                step=step,
                beta1=beta1,
                beta2=beta2,
                adam_eps=self.row_scale_eps,
                norm_eps=self.norm_eps,
            )
