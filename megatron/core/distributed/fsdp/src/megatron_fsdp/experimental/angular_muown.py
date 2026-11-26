# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""AngularMuOwn with sharded state and owner-compute orthogonalization for MFSDP v2.

AngularMuOwn writes every row of a 2-D weight as ``W_i = g_i * U_i`` with a signed row scale
``g_i`` and a unit row direction ``U_i``. Adam updates the row scales. The directions take a
Muon step on the product of unit spheres, followed by row normalization, so each step rotates
every row by an angle of about ``lr * kappa_t``.

Only the Newton-Schulz orthogonalization needs whole matrices. Everything else works on
complete local rows, which every MFSDP v2 placement keeps on one rank. ``FsdpAngularMuOwn``
therefore reuses ``FsdpMuon``'s owner gather, Newton-Schulz, and scatter unchanged and overrides
only the local work before the gather and after the scatter. The update follows the reference
AngularMuown implementation; the orthogonalization is Emerging Optimizers' Newton-Schulz, chosen
by the ``FsdpMuon`` arguments.

References:
    - Hübler, Lion, Orvieto, and He, *Muown Implicitly Performs Angular Step-size Decay*,
      arXiv:2606.23637 (2026).
    - Lion, Hübler, Li, Orvieto, and He, *Muown: Row-Norm Control for Muon Optimization*,
      arXiv:2605.10797 (2026).
"""

import math
from collections.abc import Callable
from typing import Any, cast

import torch
from torch.distributed.tensor import DTensor

from .dbuffer import DBuffer
from .muon import FsdpMuon, _FsdpGroupAssignment
from .parameter_group import FsdpParameterGroup

__all__ = ["FsdpAngularMuOwn"]

# Floor of |g| and of the retraction denominator, as in the reference AngularMuown.
EPS = 1e-7
# Row scale given to all-zero rows, so that their first direction step creates a unit row.
ZERO_ROW_SCALE = 0.33**0.5


class FsdpAngularMuOwn(FsdpMuon):
    """AngularMuOwn for 2-D parameters with sharded optimizer state on MFSDP v2.

    One step for each matrix, with ``U = W / g``:

    1. ``_compute_momentum_inputs``, locally before the gather: split the gradient into the
       radial part ``d = <G, U>`` and the direction gradient ``g * (G - d * U)``, update Adam's
       moments of ``g`` and the raw-sum momentum ``M``, and write the (Nesterov) momentum into
       the Newton-Schulz input.
    2. ``FsdpMuon``: gather the input to the owner, orthogonalize it with
       ``scale_mode="shape_scaling"``, and scatter ``O`` back.
    3. ``_apply_local_updates``, locally after the scatter: ``U+ = U - lr * kappa_t * O``,
       ``g <- g - lr * Adam(d)`` with ``|g|`` floored at ``EPS``, and ``W <- g * U+ / ||U+||``.

    ``kappa_t = (1 + c * max(0, t - t_w))^(-p)``, where ``t`` counts the optimizer steps of the
    parameter group from zero. Adam's bias corrections use each parameter's own step count. The
    row scales are not weight-decayed.

    Optimizer state per parameter: ``momentum_buffer`` (``m x n``); ``row_scale``,
    ``row_scale_exp_avg``, and ``row_scale_exp_avg_sq`` (``m x 1``), all FP32 DTensors sharded
    like the parameter; and its step count ``step``. ``row_scale`` starts at the row norms of the
    weight, and at ``ZERO_ROW_SCALE`` for all-zero rows. Each parameter group counts its steps in
    ``angular_step``.

    Args:
        *args: Positional arguments passed unchanged to ``FsdpMuon``.
        angular_warmup_steps: Steps ``t_w`` before the angular decay starts.
        angular_decay_scale: Scale ``c`` of ``kappa_t``.
        angular_decay_degree: Degree ``p`` of ``kappa_t``. Zero disables the decay.
        row_scale_betas: Adam betas for the row scales.
        row_scale_eps: Adam epsilon for the row scales.
        **kwargs: Keyword arguments passed to ``FsdpMuon``. ``scale_mode`` defaults to, and must
            be, ``"shape_scaling"``, so that ``kappa_t`` means the same rotation in every layer.
    """

    def __init__(
        self,
        *args: Any,
        angular_warmup_steps: int = 0,
        angular_decay_scale: float = 0.001,
        angular_decay_degree: float = 1.0,
        row_scale_betas: tuple[float, float] = (0.9, 0.95),
        row_scale_eps: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        scale_mode = kwargs.setdefault("scale_mode", "shape_scaling")
        if scale_mode != "shape_scaling":
            raise ValueError(
                "FsdpAngularMuOwn requires scale_mode='shape_scaling' so that the angular "
                f"multiplier gives every layer the same rotation, got {scale_mode!r}."
            )
        if angular_warmup_steps < 0:
            raise ValueError(
                f"angular_warmup_steps must be non-negative, got {angular_warmup_steps}."
            )
        if not math.isfinite(angular_decay_scale) or angular_decay_scale <= 0:
            raise ValueError(f"angular_decay_scale must be positive, got {angular_decay_scale}.")
        if not math.isfinite(angular_decay_degree) or angular_decay_degree < 0:
            raise ValueError(
                f"angular_decay_degree must be non-negative, got {angular_decay_degree}."
            )
        if not all(0 <= beta < 1 for beta in row_scale_betas):
            raise ValueError(f"row_scale_betas must be in [0, 1), got {row_scale_betas}.")
        if row_scale_eps < 0:
            raise ValueError(f"row_scale_eps must be non-negative, got {row_scale_eps}.")
        super().__init__(*args, **kwargs)
        self.angular_warmup_steps = angular_warmup_steps
        self.angular_decay_scale = angular_decay_scale
        self.angular_decay_degree = angular_decay_degree
        self.row_scale_betas = row_scale_betas
        self.row_scale_eps = row_scale_eps
        for group in self.param_groups:
            group.setdefault("angular_step", 0)

    @torch.no_grad()
    def _init_state(self, param: DTensor, state: dict[str, Any]) -> None:
        """Allocate the momentum and row states, with ``g`` set to the current row norms."""
        state["momentum_buffer"] = torch.zeros_like(param)
        # g_i = ||W_i||, so that U_i = W_i / g_i starts on the unit sphere. All-zero rows have no
        # direction; their g lets the first direction step create a unit row.
        row_norm = torch.linalg.vector_norm(param.to_local().float(), dim=1, keepdim=True)
        row_norm = torch.where(row_norm <= EPS, ZERO_ROW_SCALE, row_norm)
        for key, local in (
            ("row_scale", row_norm),
            ("row_scale_exp_avg", torch.zeros_like(row_norm)),
            ("row_scale_exp_avg_sq", torch.zeros_like(row_norm)),
        ):
            # Local (k, 1) rows of an (m, 1) DTensor sharded like the parameter's rows.
            state[key] = DTensor.from_local(
                local,
                device_mesh=param.device_mesh,
                placements=param.placements,
                run_check=False,
                shape=torch.Size((param.shape[0], 1)),
                stride=(1, 1),
            )
        # A tensor, like torch Adam's step, so that checkpoint loads restore it in place.
        state["step"] = torch.zeros((), dtype=torch.int64)

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
        """Update the sharded main weights and advance every group's angular step."""
        super().step(closure)
        for group in self.param_groups:
            group["angular_step"] += 1

    def _compute_momentum_inputs(
        self, fsdp_group: FsdpParameterGroup, tensor_to_optimizer_group_index: dict[int, int]
    ) -> DBuffer:
        """Split local rows' gradients, update the moments, and write the Newton-Schulz inputs."""
        buffer: DBuffer = fsdp_group.main_weight
        momentum_inputs: DBuffer = DBuffer(
            buffer.mesh, buffer.placements, buffer.layout, buffer.dtype, buffer.device
        )
        beta1, beta2 = self.row_scale_betas
        for i, group_index in tensor_to_optimizer_group_index.items():
            group: dict[str, Any] = self.param_groups[group_index]
            param: DTensor = cast(DTensor, fsdp_group.fsdp_parameters[i].sharded)
            assert isinstance(param.grad, DTensor)
            state: dict[str, Any] = self.state[param]
            if not state:
                self._init_state(param, state)
            momentum: torch.Tensor = state["momentum_buffer"].to_local()
            if momentum.numel() == 0:
                # This rank holds no rows of the tensor.
                continue
            beta: float = group["momentum"]
            # Local rows: W and G are (k, n); g and Adam's moments of g are (k, 1).
            weight: torch.Tensor = param.to_local()
            grad: torch.Tensor = param.grad.to_local().to(momentum.dtype).reshape(momentum.shape)
            row_scale: torch.Tensor = state["row_scale"].to_local()
            row_scale_exp_avg: torch.Tensor = state["row_scale_exp_avg"].to_local()
            row_scale_exp_avg_sq: torch.Tensor = state["row_scale_exp_avg_sq"].to_local()
            # This rank's rows of the Newton-Schulz input, filled in place below.
            ns_input: torch.Tensor = momentum_inputs.get_tensor_view(i).view_as(momentum)

            # U = W / g, with |g| floored at EPS. copysign keeps g's sign and maps +0 to +EPS.
            direction = weight / torch.copysign(row_scale.abs().clamp_min(EPS), row_scale)
            # d_i = <G_i, U_i>: the radial gradient, which is dL/dg_i.
            radial_grad = (grad * direction).sum(dim=1, keepdim=True)
            # dL/dU_i = g_i * (G_i - d_i * U_i): the direction gradient, tangent to the unit sphere.
            torch.addcmul(grad, direction, radial_grad, value=-1.0, out=ns_input)
            ns_input.mul_(row_scale)

            # Adam moments of g:  m_g <- beta1 * m_g + (1 - beta1) * d,
            #                     v_g <- beta2 * v_g + (1 - beta2) * d^2.
            row_scale_exp_avg.lerp_(radial_grad, 1 - beta1)
            row_scale_exp_avg_sq.mul_(beta2).addcmul_(radial_grad, radial_grad, value=1 - beta2)

            # Raw-sum momentum:  M <- beta * M + dL/dU.
            momentum.mul_(beta).add_(ns_input)
            # Newton-Schulz input:  M~ = dL/dU + beta * M with Nesterov, otherwise M~ = M.
            if self.nesterov:
                ns_input.add_(momentum, alpha=beta)
            else:
                ns_input.copy_(momentum)
        return momentum_inputs

    def _apply_local_updates(
        self,
        assignment: _FsdpGroupAssignment,
        full_updates: dict[int, torch.Tensor],
        local_updates: DBuffer,
    ) -> None:
        """Rotate and retract local rows, update their scales, and rewrite the weights."""
        fsdp_group = assignment.fsdp_group
        buffer: DBuffer = fsdp_group.main_weight
        beta1, beta2 = self.row_scale_betas
        for i, group_index in assignment.tensor_to_optimizer_group_index.items():
            param: DTensor = cast(DTensor, fsdp_group.fsdp_parameters[i].sharded)
            if param.grad is None:
                continue
            group: dict[str, Any] = self.param_groups[group_index]
            lr: float = group["lr"]
            state: dict[str, Any] = self.state[param]
            # Every rank counts the step, including ranks without rows of the tensor.
            state["step"] += 1
            step: int = int(state["step"])
            weight: torch.Tensor = param.to_local()
            if weight.numel() == 0:
                # This rank holds no rows of the tensor.
                continue
            # kappa_t = (1 + c * max(0, t - t_w))^(-p), with t the group's completed steps.
            kappa = (
                1
                + self.angular_decay_scale
                * max(0, group["angular_step"] - self.angular_warmup_steps)
            ) ** (-self.angular_decay_degree)
            # O: this rank's rows of the scaled orthogonalized update, selected as in FsdpMuon.
            # Scatter writes only remote-owned shards; owners keep their full result.
            if i in full_updates:
                offset: int = max(buffer.offset - buffer.layout.get_tensor_range(i).start, 0)
                update: torch.Tensor = full_updates[i].view(-1).narrow(0, offset, weight.numel())
            else:
                update = local_updates.get_tensor_view(i)
            update = update.reshape(weight.shape).to(weight.dtype)
            row_scale: torch.Tensor = state["row_scale"].to_local()
            row_scale_exp_avg: torch.Tensor = state["row_scale_exp_avg"].to_local()
            row_scale_exp_avg_sq: torch.Tensor = state["row_scale_exp_avg_sq"].to_local()

            # U+ = U - lr * kappa_t * O, with U = W / g recovered before g changes.
            direction = weight / torch.copysign(row_scale.abs().clamp_min(EPS), row_scale)
            direction.add_(update, alpha=-lr * kappa)

            # Adam on g:  g <- g - lr * m_hat / (sqrt(v_hat) + eps),
            #   m_hat = m_g / (1 - beta1^t),  v_hat = v_g / (1 - beta2^t), t the parameter's steps,
            # then |g| floored at EPS with g's sign.
            row_scale.sub_(
                (row_scale_exp_avg / (1 - beta1**step))
                / ((row_scale_exp_avg_sq / (1 - beta2**step)).sqrt() + self.row_scale_eps),
                alpha=lr,
            )
            row_scale.copy_(torch.copysign(row_scale.abs().clamp_min(EPS), row_scale))

            # W = g * U+ / ||U+||.
            row_norm = torch.linalg.vector_norm(direction, dim=1, keepdim=True).clamp_min(EPS)
            torch.mul(direction, row_scale / row_norm, out=weight)
