# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""AngularMuOwn with sharded state and owner-compute orthogonalization for MFSDP v2.

AngularMuOwn writes every row of a 2-D weight as ``W_i = g_i * U_i`` with a signed row scale
``g_i`` and a unit row direction ``U_i``. Adam updates the row scales. The directions take a
Muon step on the product of unit spheres, followed by row normalization, so each step rotates
every row by an angle of about ``lr * kappa_t``.

Only the Newton-Schulz orthogonalization needs whole matrices. Everything else works on
complete local rows, which every MFSDP v2 placement keeps on one rank. ``FsdpAngularMuOwn``
therefore reuses ``FsdpMuon``'s owner gather, Newton-Schulz, and scatter unchanged and overrides
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
from .muon import FsdpMuon, _FsdpGroupAssignment
from .parameter_group import FsdpParameterGroup

try:
    from megatron.core.fusions.fused_angular_muown import (
        fused_angular_muown_apply_update,
        fused_angular_muown_momentum_inputs,
    )
except ImportError:
    # Standalone Megatron-FSDP has only the torch path.
    fused_angular_muown_apply_update = fused_angular_muown_momentum_inputs = None

__all__ = ["FsdpAngularMuOwn"]


class FsdpAngularMuOwn(FsdpMuon):
    """AngularMuOwn for 2-D parameters with sharded optimizer state on MFSDP v2.

    One step for each matrix, with ``U = W / g``:

    1. ``_compute_momentum_inputs``, locally before the gather: split the gradient into the
       radial part ``d = <G, U>`` and the direction gradient ``g * (G - d * U)``, update Adam's
       moments of ``g`` and the raw-sum momentum ``M``, and write the (Nesterov) momentum into
       the Newton-Schulz input.
    2. ``FsdpMuon``: gather the input to the owner, orthogonalize it with
       ``scale_mode="shape_scaling"``, and scatter ``O`` back.
    3. ``_apply_local_updates``, locally after the scatter: ``U <- rownorm(U - lr * kappa_t * O)``,
       ``g <- (1 - lr * weight_decay * weight_decay_mult) * g - lr * Adam(d)``, and
       ``W <- g * U``.

    The two local stages share only the optimizer state and ``O``: step 1 folds ``d`` into Adam's
    moments, and step 3 recomputes ``U`` from ``W`` and the not yet updated ``g``.

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
        use_fused_kernels: Run the local work before the gather and after the scatter as two
            fused Triton kernels, one launch per tensor, instead of torch ops.
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
        use_fused_kernels: bool = False,
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

        # C in kappa_t = (1 + C * u^r)^(-p / r).
        self.angular_decay_scale = angular_decay_scale
        # p in kappa_t = (1 + C * u^r)^(-p / r).
        self.angular_decay_degree = angular_decay_degree
        # r in kappa_t = (1 + C * u^r)^(-p / r).
        self.angular_plateau_degree = angular_plateau_degree
        # t_w in u = (t - t_w) / T; kappa_t = 1 while t <= t_w.
        self.angular_warmup_steps = angular_warmup_steps
        # T in u = (t - t_w) / T.
        self.angular_horizon_steps = angular_horizon_steps
        # (beta1, beta2) in m_g <- beta1 * m_g + (1 - beta1) * d,
        #                   v_g <- beta2 * v_g + (1 - beta2) * d^2,
        # and in m_hat = m_g / (1 - beta1^t), v_hat = v_g / (1 - beta2^t).
        self.row_scale_betas = row_scale_betas
        # eps in g <- g - lr * m_hat / (sqrt(v_hat) + eps).
        self.row_scale_eps = row_scale_eps
        # eps_norm in U = W / copysign(max(|g|, eps_norm), g) and rho_den = max(||U+_i||, eps_norm).
        self.norm_eps = norm_eps
        # mult in lambda' = weight_decay * mult, used in g <- (1 - lr * lambda') * g.
        self.weight_decay_mult = weight_decay_mult
        if use_fused_kernels and fused_angular_muown_apply_update is None:
            raise ImportError("use_fused_kernels needs Megatron-Core's fused AngularMuOwn kernels.")
        self.use_fused_kernels = use_fused_kernels
        # kappa_t of the most recent step, for logging only; no formula reads it.
        self.last_angular_multiplier = 1.0
        # muown_step = t - 1, the completed steps of each group. Step t uses t = muown_step + 1 in
        # u = (t - t_w) / T and in the bias corrections 1 - beta1^t and 1 - beta2^t.
        for group in self.param_groups:
            group.setdefault("muown_step", 0)
        # Groups with gradients in the current step; step() adds 1 to their muown_step at its end.
        # Bookkeeping only; no formula reads it.
        self._active_group_indices: set[int] = set()

    @torch.no_grad()
    def _init_state(self, param: DTensor, state: dict[str, Any]) -> None:
        """Allocate the momentum and row states, with ``g`` set to the current row norms."""
        state["momentum_buffer"] = torch.zeros_like(param)
        # g_i = ||W_i||, so that U_i = W_i / g_i starts on the unit sphere.
        row_norm = torch.linalg.vector_norm(param.to_local().float(), dim=1, keepdim=True)
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
        # t <- t + 1 once per step, after every tensor of the step has used the same t.
        for group_index in sorted(self._active_group_indices):
            self.param_groups[group_index]["muown_step"] += 1

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
            self._active_group_indices.add(group_index)
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
            grad: torch.Tensor = param.grad.to_local().reshape(momentum.shape)
            row_scale: torch.Tensor = state["row_scale"].to_local()
            row_scale_exp_avg: torch.Tensor = state["row_scale_exp_avg"].to_local()
            row_scale_exp_avg_sq: torch.Tensor = state["row_scale_exp_avg_sq"].to_local()
            # This rank's rows of the Newton-Schulz input, filled in place below.
            ns_input: torch.Tensor = momentum_inputs.get_tensor_view(i).view_as(momentum)
            if self.use_fused_kernels:
                # The steps below as one kernel.
                fused_angular_muown_momentum_inputs(
                    weight,
                    grad,
                    momentum,
                    ns_input,
                    row_scale,
                    row_scale_exp_avg,
                    row_scale_exp_avg_sq,
                    beta=beta,
                    betas=self.row_scale_betas,
                    norm_eps=self.norm_eps,
                    nesterov=self.nesterov,
                )
                continue
            grad = grad.to(momentum.dtype)

            # U = W / g, with |g| floored at norm_eps. copysign keeps g's sign and maps +0 to +eps.
            direction = weight / torch.copysign(row_scale.abs().clamp_min(self.norm_eps), row_scale)
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
            # Newton-Schulz normalizes its input, so this matches Muon's EMA momentum up to the
            # positive factor 1 / (1 - beta).
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
            if fsdp_group.fsdp_parameters[i].sharded.grad is None:
                continue
            group: dict[str, Any] = self.param_groups[group_index]
            lr: float = group["lr"]
            step: int = group["muown_step"] + 1

            # kappa_t = (1 + C * u^r)^(-p / r) with progress u = (t - t_w) / T, and kappa_t = 1
            # during warmup or when C, p, or T is zero. Every rank computes it, including ranks
            # without local rows, so the logged value agrees across ranks.
            if (
                step <= self.angular_warmup_steps
                or self.angular_horizon_steps <= 0
                or self.angular_decay_scale == 0
                or self.angular_decay_degree == 0
            ):
                kappa = 1.0
            else:
                progress = (step - self.angular_warmup_steps) / self.angular_horizon_steps
                kappa = (
                    1.0 + self.angular_decay_scale * progress**self.angular_plateau_degree
                ) ** (-self.angular_decay_degree / self.angular_plateau_degree)
            self.last_angular_multiplier = kappa

            param: DTensor = cast(DTensor, fsdp_group.fsdp_parameters[i].sharded)
            weight: torch.Tensor = param.to_local()
            if weight.numel() == 0:
                # This rank holds no rows of the tensor.
                continue
            # O: this rank's rows of the scaled orthogonalized update, selected as in FsdpMuon.
            # Scatter writes only remote-owned shards; owners keep their full result.
            if i in full_updates:
                offset: int = max(buffer.offset - buffer.layout.get_tensor_range(i).start, 0)
                update: torch.Tensor = full_updates[i].view(-1).narrow(0, offset, weight.numel())
            else:
                update = local_updates.get_tensor_view(i)
            update = update.reshape(weight.shape).to(weight.dtype)
            state: dict[str, Any] = self.state[param]
            row_scale: torch.Tensor = state["row_scale"].to_local()
            row_scale_exp_avg: torch.Tensor = state["row_scale_exp_avg"].to_local()
            row_scale_exp_avg_sq: torch.Tensor = state["row_scale_exp_avg_sq"].to_local()
            if self.use_fused_kernels:
                # The steps below as one kernel.
                fused_angular_muown_apply_update(
                    weight,
                    update,
                    row_scale,
                    row_scale_exp_avg,
                    row_scale_exp_avg_sq,
                    direction_lr=lr * kappa,
                    lr=lr,
                    decay=1 - lr * group["weight_decay"] * self.weight_decay_mult,
                    bias_corrections=(1 - beta1**step, 1 - beta2**step),
                    adam_eps=self.row_scale_eps,
                    norm_eps=self.norm_eps,
                )
                continue

            # U = W / g. g is not updated yet, so this is the U that split the gradient.
            direction = weight / torch.copysign(row_scale.abs().clamp_min(self.norm_eps), row_scale)
            # U+ = U - lr * kappa_t * O
            direction.add_(update, alpha=-lr * kappa)
            # Retraction denominators:  rho_i = ||U+_i||,  rho_den_i = max(rho_i, norm_eps).
            row_norm = torch.linalg.vector_norm(direction, dim=1, keepdim=True)
            denominator = row_norm.clamp_min(self.norm_eps)

            # Adam on g with decoupled decay:
            #   g <- (1 - lr * lambda') * g - lr * m_hat / (sqrt(v_hat) + eps),
            #   m_hat = m_g / (1 - beta1^t),  v_hat = v_g / (1 - beta2^t),
            #   lambda' = weight_decay * weight_decay_mult.
            weight_decay: float = group["weight_decay"] * self.weight_decay_mult
            if weight_decay != 0:
                row_scale.mul_(1 - lr * weight_decay)
            row_scale.sub_(
                (row_scale_exp_avg / (1 - beta1**step))
                / ((row_scale_exp_avg_sq / (1 - beta2**step)).sqrt() + self.row_scale_eps),
                alpha=lr,
            )

            # W = g * U_new with the retracted unit rows U_new = U+ / rho_den. |g| is floored at
            # norm_eps as for U = W / g, so a row whose g lands on zero keeps its direction.
            floored_scale = torch.copysign(row_scale.abs().clamp_min(self.norm_eps), row_scale)
            torch.mul(direction, floored_scale / denominator, out=weight)
            # g <- g * rho / rho_den: a no-op unless the clamp fired, where it keeps
            # |g_i| = ||W_i||.
            row_scale.mul_(row_norm / denominator)
