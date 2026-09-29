# Adapted from facebookresearch/schedule_free (Copyright (c) Meta Platforms, Inc. and
# affiliates), licensed under the Apache License, Version 2.0.

"""ScheduleFree+ (AdamC + Schedule-Free + Polyak step size) for Megatron.

Port of ``AdamCScheduleFreePlusPaper`` from facebookresearch/schedule_free
(schedulefree/adamc_schedulefree_plus_paper.py), described in Defazio, "ScheduleFree+:
Scaling Learning-Rate-Free & Schedule-Free Learning to Large Language Models" (2026),
Algorithm 1. Differences from the reference:

- Parameters are the fp32 main-param shards owned by Megatron's (distributed) optimizer,
  so the global gradient L1 norm and the <g, z - x> correction are summed over an explicit
  process group (``reduce_group``) instead of relying on DTensor detection.
- The loss is supplied by the training loop through ``set_function_value`` (already
  averaged over the data-parallel group), so ``step()`` keeps the standard signature.
- The gradient-L1 EMA stores its raw value and bias-corrects a copy. The reference writes
  the corrected value back into the EMA, which double-corrects when ``polyak_beta > 0``.
- Beta annealing uses the 1-indexed step, as in the paper's Algorithm 1; the reference code
  is one step behind.
- ``y`` is not stored: the parameter tensor itself is ``y`` in train mode, and is rebuilt
  bit-exactly from ``x`` and ``z`` when switching back from eval mode.
- Gradient clipping does not change the Polyak step. Megatron clips the gradients before
  ``step()``; the gradient statistics (||g||_1 and <g, z - x>) are linear in g, so they are
  divided by the clip coefficient (``set_grad_clip_coeff``) to get their unclipped values.
  Otherwise the denominator shrinks with the clip coefficient while the loss does not, and
  the step grows with the grad norm. The Adam moments still see the clipped gradients.

In Megatron, evaluation and checkpoint saving run with ``x`` swapped into the weights
(``eval()``), so checkpoints hold ``x`` as the model weights and fp32 main params; ``train()``
after loading rebuilds ``y`` (see ``sfplus_eval_weights`` in megatron/training/training.py).

The group ``lr`` set by Megatron's LR scheduler acts as a multiplier on the Polyak step
(use it for warmup, with a constant peak of 1.0). Weight decay is AdamC-style,
``z -= lr_t**2 * weight_decay * y``, so values are much larger than AdamW's (e.g. 5-50).
"""

import math
from typing import Optional

import torch

# Per-parameter tensor state, in the order the distributed optimizer checkpoints it.
SFPLUS_STATE_KEYS = ("exp_avg", "exp_avg_sq", "z", "x")


class ScheduleFreePlusAdamC(torch.optim.Optimizer):
    """AdamC + Schedule-Free + Polyak step size (see module docstring).

    Args:
        params: parameters or parameter groups.
        lr: multiplier on the Polyak step size; the LR scheduler overwrites it per step.
        betas: Adam (beta1, beta2) for the inner AdamC update of ``z``.
        eps: Adam epsilon.
        weight_decay: AdamC decoupled weight decay, scaled by ``lr_t**2``.
        sf_beta: Schedule-Free interpolation ``y = sf_beta * x + (1 - sf_beta) * z``.
        sf_beta_max: final value of ``sf_beta`` when annealing.
        sf_beta_anneal_steps: steps over which ``1 - sf_beta`` is log-linearly annealed to
            ``1 - sf_beta_max``; 0 keeps ``sf_beta`` fixed.
        r: polynomial power of the step index in the averaging weights.
        weight_lr_power: power of the running max LR in the averaging weights.
        c_warmup: number of initial steps during which ``x`` tracks ``z`` exactly.
        polyak_beta: EMA coefficient for the gradient L1 norm in the Polyak denominator.
    """

    def __init__(
        self,
        params,
        lr: float = 1.0,
        betas=(0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        sf_beta: float = 0.9,
        sf_beta_max: float = 0.965,
        sf_beta_anneal_steps: int = 0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        c_warmup: int = 0,
        polyak_beta: float = 0.9,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            sf_beta=sf_beta,
            sf_beta_max=sf_beta_max,
            sf_beta_anneal_steps=sf_beta_anneal_steps,
            r=r,
            weight_lr_power=weight_lr_power,
            c_warmup=c_warmup,
            polyak_beta=polyak_beta,
            # Run state. Plain Python scalars, identical in every group, so they travel
            # with the param_groups in Megatron's (non-sharded) optimizer checkpoint.
            k=0,
            weight_sum=0.0,
            lr_max=eps,
            grad_l1_ema=0.0,
            y_weight=0.0,
            train_mode=True,
        )
        super().__init__(params, defaults)
        self.reduce_group: Optional[torch.distributed.ProcessGroup] = None
        self.function_value: Optional[float] = None
        self.grad_clip_coeff: float = 1.0
        self.last_stats: dict = {}

    def set_function_value(self, value: float):
        """Set the loss at the current ``y`` (global mean over data-parallel ranks)."""
        self.function_value = float(value)

    def set_grad_clip_coeff(self, coeff: float):
        """Set the factor (<= 1) the gradients were scaled by in clipping before this step."""
        self.grad_clip_coeff = float(coeff)

    @torch.no_grad()
    def init_state(self):
        """Allocate per-parameter state; ``z`` and ``x`` start at the current weights."""
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['z'] = p.detach().clone()
                    state['x'] = p.detach().clone()

    @torch.no_grad()
    def eval(self):
        """Put ``x`` (the averaged weights) into the parameters."""
        for group in self.param_groups:
            if group['train_mode']:
                for p in group['params']:
                    if 'x' in self.state[p]:
                        p.copy_(self.state[p]['x'])
                group['train_mode'] = False

    @torch.no_grad()
    def train(self):
        """Put ``y`` back into the parameters, recomputed exactly as the last step did."""
        for group in self.param_groups:
            if not group['train_mode']:
                for p in group['params']:
                    state = self.state[p]
                    if 'x' in state:
                        p.copy_(state['x']).lerp_(state['z'], group['y_weight'])
                group['train_mode'] = True

    def _sf_beta(self, group, k):
        anneal_steps = group['sf_beta_anneal_steps']
        if anneal_steps <= 0:
            return group['sf_beta']
        # Algorithm 1 uses the 1-indexed step t = k + 1 (the reference code uses k).
        progress = min((k + 1) / anneal_steps, 1.0)
        return 1 - math.exp(
            (1 - progress) * math.log(1 - group['sf_beta'])
            + progress * math.log(1 - group['sf_beta_max'])
        )

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is None, "ScheduleFreePlusAdamC does not support closures"
        if not all(group['train_mode'] for group in self.param_groups):
            raise RuntimeError("ScheduleFreePlusAdamC.step() called in eval mode; call train()")
        if self.function_value is None:
            raise RuntimeError(
                "ScheduleFreePlusAdamC needs the loss: call set_function_value() before step()"
            )
        self.init_state()

        group0 = self.param_groups[0]
        k = group0['k']
        sf_beta = self._sf_beta(group0, k)

        # Global statistics: ||g||_1 and sf_beta * <g, z - x>, summed over all shards.
        params = [p for group in self.param_groups for p in group['params']]
        device = params[0].device if params else torch.device('cuda', torch.cuda.current_device())
        stats = torch.zeros(2, dtype=torch.float64, device=device)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                stats[0] += g.abs().sum()
                stats[1] += torch.dot(g.view(-1), (state['z'] - state['x']).view(-1))
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats, group=self.reduce_group)
        # Undo gradient clipping in the statistics (both are linear in g).
        grad_clip_coeff = self.grad_clip_coeff
        grad_l1, ip = (value / grad_clip_coeff for value in stats.tolist())
        ip_term = sf_beta * ip

        # Polyak step size, with the denominator estimated as sqrt(pi/2) * EMA(||g||_1).
        polyak_beta = group0['polyak_beta']
        grad_l1_ema = polyak_beta * group0['grad_l1_ema'] + (1 - polyak_beta) * grad_l1 * math.sqrt(
            math.pi / 2
        )
        grad_l1_ema_corr = grad_l1_ema / (1 - polyak_beta ** (k + 1))
        numerator = self.function_value + ip_term
        polyak_lr = max(0.0, numerator) / max(grad_l1_ema_corr, 1e-30)

        # Squared L2 norms of x and z after this step, for logging (y is Megatron's params-norm).
        norms_sq = torch.zeros(2, dtype=torch.float64, device=device)
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            decay = group['weight_decay']
            lr = max(group['lr'], eps) * polyak_lr

            group['grad_l1_ema'] = grad_l1_ema
            group['lr_max'] = lr_max = max(lr, group['lr_max'])
            if k < group['c_warmup']:
                ckp1 = 1.0
            else:
                weight = ((k + 1) ** group['r']) * (lr_max ** group['weight_lr_power'])
                group['weight_sum'] = group['weight_sum'] + weight
                ckp1 = weight / group['weight_sum']
            y_weight = 1 - sf_beta
            group['y_weight'] = y_weight

            bias_correction1 = 1 - beta1 ** (k + 1)
            bias_correction2 = 1 - beta2 ** (k + 1)

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                z, x = state['z'], state['x']

                # AdamC weight decay on z, applied at y (the current parameter value).
                if decay != 0:
                    z.add_(p, alpha=-lr * lr * decay)

                exp_avg.lerp_(g, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                denom = (exp_avg_sq / bias_correction2).sqrt_().add_(eps)
                z.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)
                del denom

                x.lerp_(z, ckp1)
                # y = sf_beta * x + (1 - sf_beta) * z
                p.copy_(x).lerp_(z, y_weight)
                norms_sq[0] += torch.dot(x.view(-1), x.view(-1))
                norms_sq[1] += torch.dot(z.view(-1), z.view(-1))

            group['k'] = k + 1

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(norms_sq, group=self.reduce_group)
        x_norm_sq, z_norm_sq = norms_sq.tolist()
        sqrt_pi_2 = math.sqrt(math.pi / 2)
        self.last_stats = {
            'effective_lr': max(group0['lr'], group0['eps']) * polyak_lr,
            'polyak_lr': polyak_lr,
            'lr_multiplier': group0['lr'],
            'function_value': self.function_value,
            'ip_term': ip_term,
            # Raw and smoothed ||g||_1 on the same scale; the denominator adds sqrt(pi/2).
            'grad_l1': grad_l1,
            'grad_l1_ema': grad_l1_ema_corr / sqrt_pi_2,
            'polyak_denominator': grad_l1_ema_corr,
            'x_norm': math.sqrt(x_norm_sq),
            'z_norm': math.sqrt(z_norm_sq),
            'sf_beta': sf_beta,
            'ckp1': ckp1 if self.param_groups else 1.0,
            'grad_clip_coeff': grad_clip_coeff,
        }
        self.function_value = None
        self.grad_clip_coeff = 1.0
        return None
