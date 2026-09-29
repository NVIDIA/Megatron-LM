"""ScheduleFree+ with Muon as the base optimizer for hidden weight matrices.

Same Schedule-Free wrapper as ``ScheduleFreePlusAdamC`` (y/z/x iterates, c-warmup, beta
annealing, r-weighted averaging, Polyak step size, fully decoupled AdamC-style weight decay),
but the inner step on ``z`` for 2D hidden matrices is Muon's: momentum, orthogonalized by a
Newton-Schulz iteration and scaled to a fixed per-element RMS. Embeddings, the output layer
and 1D parameters (norm gains) keep the AdamC step. Not from the ScheduleFree+ paper, which
only mentions Muon as a possible base optimizer; the two pieces derived there for Adam are
re-derived as follows.

Polyak denominator. The Polyak step for ``z -= lr * d`` has ``<g, d>`` in its denominator.
For Adam the paper uses the momentum-free direction ``g / sqrt(v)``, whose per-element RMS is
about 1, and approximates ``sum g^2 / sqrt(v)`` by ``sqrt(pi/2) * ||g||_1`` (Eq. 9). The Muon
counterpart is the momentum-free direction ``orth(G)`` rescaled to exactly unit RMS,
``d = sqrt(m n) * orth(G) / ||orth(G)||_F``, computed with one extra Newton-Schulz pass on the
raw gradient. For a full-rank G, ``||orth(G)||_F = sqrt(min(m, n))`` and ``<G, d>`` is
``sqrt(max(m, n)) * ||G||_*`` (the nuclear norm, dual of the spectral norm Muon descends in).
For a low-rank G, as early in training, Newton-Schulz leaves the near-zero singular values
near zero, so ``sqrt(max(m, n)) * orth(G)`` has RMS well below 1 (~0.1 at rank 4 of 768) and
the plain nuclear-norm term would make the step several times too large; the rescaling keeps
the term on the scale of the AdamC one (for rank 1: ``sigma * sqrt(m n)`` against
``~0.8 * sigma * sqrt(m n)``). The logged ``muon_polyak_dir_rms`` is the RMS before rescaling.
The denominator is the sum of the AdamC and Muon parts, smoothed by the same EMA.

The actual Muon update is scaled to a smaller RMS (``muon_rms``, 0.2), just as Adam's actual
update ``m / sqrt(v)`` is smaller than ``g / sqrt(v)`` once momentum averages out gradient
noise. Using the update's scale in the denominator as well makes the step size ~5x too large
and diverges.

Weight decay. A Muon update has a fixed Frobenius norm, so ``z -= lr**2 * wd * y`` gives a
weight-norm equilibrium independent of the step size, as AdamC does for Adam. The level of that
equilibrium is not shared with AdamC, though: per element, roughly
``RMS_eq ~ r * sqrt(C / (2 * wd))`` for update RMS ``r`` and momentum correlation
``C ~ (1 + beta) / (1 - beta)``. Muon's ``r`` is always ``muon_rms`` and its momentum is higher,
so for the same decay its matrices settle 2-3x larger than AdamC's (measured at wd 5: weight RMS
~0.28 vs ~0.11). ``muon_weight_decay`` sets the Muon matrices' decay separately; the decay also
sets how fast the norm reaches equilibrium, ~1 / (2 * lr**2 * wd) steps.

Muon needs whole matrices, so this optimizer requires the non-distributed optimizer (every
data-parallel rank holds all fp32 params and gradients and runs the same step).
"""

import math
from typing import Optional, Sequence

import torch

from .schedulefree_plus import ScheduleFreePlusAdamC

# Newton-Schulz coefficient sets, as in emerging_optimizers (the package behind Megatron's
# --optimizer muon), cycled over the iterations. Both are fast rather than exact: singular
# values end up near 1, not at 1.
NS_COEFFICIENTS = {
    # Keller Jordan's original fixed coefficients.
    "simple": [(3.4445, -4.7750, 2.0315)],
    # Per-iteration coefficients from modded-nanogpt; Megatron's default.
    "quintic": [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ],
}


def newton_schulz(g: torch.Tensor, steps: int, coefficient_type: str = "quintic") -> torch.Tensor:
    """Approximate ``U V^T`` of a matrix ``[m, n]``.

    Matches emerging_optimizers' ``newton_schulz`` with fp32 matmul precision "medium":
    normalize in fp32, iterate in bf16, on the side with the smaller dimension.
    """
    coefficients = NS_COEFFICIENTS[coefficient_type]
    x = g.float()
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT
    x = torch.nn.functional.normalize(x, p=2, dim=(-2, -1), eps=1e-7).bfloat16()
    for i in range(steps):
        a, b, c = coefficients[i % len(coefficients)]
        # x = a x + (b A + c A^2) x with A = x x^T, fused as in emerging_optimizers.
        gram = x @ x.mT
        poly = torch.addmm(gram, gram, gram, alpha=c, beta=b)
        x = torch.addmm(x, poly, x, alpha=1.0, beta=a)
    x = x.float()
    if transposed:
        x = x.mT
    return x


class ScheduleFreePlusMuon(ScheduleFreePlusAdamC):
    """ScheduleFree+ with a Muon inner step for hidden matrices (see module docstring).

    Args (on top of ``ScheduleFreePlusAdamC``'s):
        muon_momentum: momentum of the Muon step.
        muon_nesterov: use Nesterov momentum in the Muon step.
        muon_ns_steps: Newton-Schulz iterations.
        muon_coefficient_type: Newton-Schulz coefficient set, "quintic" or "simple".
        muon_rms: per-element RMS of the Muon update before the step size, i.e. the update
            is ``muon_rms * sqrt(max(m, n)) * orth(M)``. 0.2 matches AdamW's typical update
            RMS (Moonshot, "Muon is Scalable", 2025), so one Polyak step size serves both
            the Muon and the AdamC parameters.
        muon_weight_decay: AdamC-form weight decay of the Muon matrices; None uses the
            group's ``weight_decay``. Groups with ``weight_decay`` 0 stay undecayed.

    Which parameters use Muon is decided from the model parameters passed in: 2D and not
    tagged ``is_embedding_or_output_parameter``. A parameter with an ``sfplus_qkv_split``
    attribute (row counts of one query group's q, k and v blocks) is orthogonalized as three
    matrices, all q rows, all k rows and all v rows, as Megatron's Muon does with split_qkv.
    """

    def __init__(
        self,
        params,
        muon_momentum: float = 0.95,
        muon_nesterov: bool = False,
        muon_ns_steps: int = 5,
        muon_coefficient_type: str = "quintic",
        muon_rms: float = 0.2,
        muon_weight_decay: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self.muon_momentum = muon_momentum
        self.muon_nesterov = muon_nesterov
        self.muon_ns_steps = muon_ns_steps
        if muon_coefficient_type not in NS_COEFFICIENTS:
            raise ValueError(
                f"ScheduleFreePlusMuon supports coefficient types {sorted(NS_COEFFICIENTS)}, "
                f"got {muon_coefficient_type!r}"
            )
        self.muon_coefficient_type = muon_coefficient_type
        self.muon_rms = muon_rms
        self.muon_weight_decay = muon_weight_decay
        # Row splits per parameter position, or None for AdamC parameters. Kept by position
        # because Megatron swaps the model params in param_groups for fp32 main params (which
        # lose the attributes) in place, and loading a checkpoint keeps the group order.
        self.muon_splits = [
            [self._muon_split(p) for p in group['params']] for group in self.param_groups
        ]

    @staticmethod
    def _muon_split(p) -> Optional[Sequence[int]]:
        if p.dim() != 2 or getattr(p, 'is_embedding_or_output_parameter', False):
            return None
        return tuple(getattr(p, 'sfplus_qkv_split', None) or (p.shape[0],))

    def _orth_blocks(self, t: torch.Tensor, splits: Sequence[int]):
        """``orth(W)`` in fp32 for each matrix ``W`` in ``t``, as [groups, rows, cols] blocks.

        ``t`` stacks row blocks of sizes ``splits`` (one per matrix) repeated over groups, e.g.
        [q, k, v] per query group; each matrix gathers its block from every group.
        """
        cols = t.shape[-1]
        num_groups = t.shape[0] // sum(splits)
        blocks = t.view(num_groups, sum(splits), cols).split(list(splits), dim=1)
        return [
            newton_schulz(
                block.reshape(-1, cols), self.muon_ns_steps, self.muon_coefficient_type
            ).view(num_groups, -1, cols)
            for block in blocks
        ]

    def _muon_direction(self, t: torch.Tensor, splits: Sequence[int], rms: float) -> torch.Tensor:
        """The Muon update ``rms * sqrt(max(m, n)) * orth(W)`` for each matrix ``W`` in ``t``."""
        out = []
        for orth in self._orth_blocks(t, splits):
            rows, cols = orth.shape[0] * orth.shape[1], orth.shape[2]
            out.append(orth * (rms * math.sqrt(max(rows, cols))))
        return torch.cat(out, dim=1).view_as(t)

    def _polyak_direction(self, t: torch.Tensor, splits: Sequence[int]):
        """``orth(W)`` rescaled to per-element RMS 1 for each matrix ``W`` in ``t``.

        Also returns the sum of squares of ``sqrt(max(m, n)) * orth(W)``, the unrescaled
        direction, whose RMS shows how far the gradient is from full rank.
        """
        out = []
        unscaled_sq = 0.0
        for orth in self._orth_blocks(t, splits):
            rows, cols = orth.shape[0] * orth.shape[1], orth.shape[2]
            norm = orth.norm()
            unscaled_sq = unscaled_sq + norm.square() * max(rows, cols)
            out.append(orth * (math.sqrt(rows * cols) / norm.clamp_min(1e-30)))
        return torch.cat(out, dim=1).view_as(t), unscaled_sq

    def _params_with_splits(self):
        for group, splits in zip(self.param_groups, self.muon_splits):
            for p, split in zip(group['params'], splits):
                if p.grad is not None:
                    yield group, p, split

    @torch.no_grad()
    def init_state(self):
        """Muon parameters keep a momentum buffer, AdamC ones both moments; all keep z, x."""
        for group, splits in zip(self.param_groups, self.muon_splits):
            for p, split in zip(group['params'], splits):
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    if split is None:
                        state['exp_avg_sq'] = torch.zeros_like(p)
                    state['z'] = p.detach().clone()
                    state['x'] = p.detach().clone()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is None, "ScheduleFreePlusMuon does not support closures"
        if not all(group['train_mode'] for group in self.param_groups):
            raise RuntimeError("ScheduleFreePlusMuon.step() called in eval mode; call train()")
        if self.function_value is None:
            raise RuntimeError(
                "ScheduleFreePlusMuon needs the loss: call set_function_value() before step()"
            )
        self.init_state()

        group0 = self.param_groups[0]
        k = group0['k']
        sf_beta = self._sf_beta(group0, k)

        # Global statistics: ||g||_1 over AdamC params, sum of <G, unit-RMS orth(G)> over
        # Muon params, <g, z - x> over all params, and (for logging) the squared norm and
        # element count of the Muon Polyak direction before its rescaling to unit RMS.
        params = [p for group in self.param_groups for p in group['params']]
        device = params[0].device if params else torch.device('cuda', torch.cuda.current_device())
        stats = torch.zeros(5, dtype=torch.float64, device=device)
        for _, p, split in self._params_with_splits():
            g = p.grad
            state = self.state[p]
            if split is None:
                stats[0] += g.abs().sum()
            else:
                direction, unscaled_sq = self._polyak_direction(g, split)
                stats[1] += torch.dot(g.view(-1), direction.view(-1))
                stats[3] += unscaled_sq
                stats[4] += g.numel()
            stats[2] += torch.dot(g.view(-1), (state['z'] - state['x']).view(-1))
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats, group=self.reduce_group)
        grad_l1, muon_denominator, ip, muon_dir_sq, muon_numel = stats.tolist()
        # Undo gradient clipping in the first three statistics (all linear in g); the
        # direction's RMS does not depend on the gradient's scale.
        grad_clip_coeff = self.grad_clip_coeff
        grad_l1, muon_denominator, ip = (
            value / grad_clip_coeff for value in (grad_l1, muon_denominator, ip)
        )
        adam_denominator = grad_l1 * math.sqrt(math.pi / 2)
        ip_term = sf_beta * ip

        # Polyak step size. 'grad_l1_ema' holds the EMA of the whole denominator, as it does
        # for ScheduleFreePlusAdamC (where the denominator is the AdamC part alone).
        polyak_beta = group0['polyak_beta']
        denominator_ema = polyak_beta * group0['grad_l1_ema'] + (1 - polyak_beta) * (
            adam_denominator + muon_denominator
        )
        denominator_ema_corr = denominator_ema / (1 - polyak_beta ** (k + 1))
        numerator = self.function_value + ip_term
        polyak_lr = max(0.0, numerator) / max(denominator_ema_corr, 1e-30)

        # Squared L2 norms of x and z after this step, split into AdamC and Muon params:
        # [adam x, adam z, muon x, muon z].
        norms_sq = torch.zeros(4, dtype=torch.float64, device=device)
        for group, splits in zip(self.param_groups, self.muon_splits):
            beta1, beta2 = group['betas']
            eps = group['eps']
            decay = group['weight_decay']
            # Megatron's no-decay groups (weight_decay 0) stay undecayed for Muon too.
            muon_decay = (
                decay if decay == 0 or self.muon_weight_decay is None else self.muon_weight_decay
            )
            lr = max(group['lr'], eps) * polyak_lr

            group['grad_l1_ema'] = denominator_ema
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

            for p, split in zip(group['params'], splits):
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                exp_avg = state['exp_avg']
                z, x = state['z'], state['x']

                # AdamC weight decay on z, applied at y (the current parameter value).
                param_decay = decay if split is None else muon_decay
                if param_decay != 0:
                    z.add_(p, alpha=-lr * lr * param_decay)

                if split is None:
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg.lerp_(g, 1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    denom = (exp_avg_sq / bias_correction2).sqrt_().add_(eps)
                    z.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)
                    del denom
                else:
                    exp_avg.lerp_(g, 1 - self.muon_momentum)
                    update = g.lerp(exp_avg, self.muon_momentum) if self.muon_nesterov else exp_avg
                    z.add_(self._muon_direction(update, split, self.muon_rms), alpha=-lr)

                x.lerp_(z, ckp1)
                # y = sf_beta * x + (1 - sf_beta) * z
                p.copy_(x).lerp_(z, y_weight)
                offset = 0 if split is None else 2
                norms_sq[offset] += torch.dot(x.view(-1), x.view(-1))
                norms_sq[offset + 1] += torch.dot(z.view(-1), z.view(-1))

            group['k'] = k + 1

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(norms_sq, group=self.reduce_group)
        adam_x_sq, adam_z_sq, muon_x_sq, muon_z_sq = norms_sq.tolist()
        denominator = adam_denominator + muon_denominator
        self.last_stats = {
            'effective_lr': max(group0['lr'], group0['eps']) * polyak_lr,
            'polyak_lr': polyak_lr,
            'lr_multiplier': group0['lr'],
            'function_value': self.function_value,
            'ip_term': ip_term,
            'grad_l1': grad_l1,
            'adam_denominator': adam_denominator,
            'muon_denominator': muon_denominator,
            # Fraction of the (unsmoothed) Polyak denominator from the Muon matrices, i.e. how
            # much they, rather than the AdamC params, set the shared step size.
            'muon_denominator_share': muon_denominator / denominator if denominator > 0 else 0.0,
            'polyak_denominator': denominator_ema_corr,
            'x_norm': math.sqrt(adam_x_sq + muon_x_sq),
            'z_norm': math.sqrt(adam_z_sq + muon_z_sq),
            # Per-part norms: whether AdamC-form weight decay holds the Muon matrices at a
            # steady norm (the decay was calibrated for Adam, not Muon).
            'muon_x_norm': math.sqrt(muon_x_sq),
            'muon_z_norm': math.sqrt(muon_z_sq),
            'adam_x_norm': math.sqrt(adam_x_sq),
            'adam_z_norm': math.sqrt(adam_z_sq),
            # RMS of sqrt(max(m, n)) * orth(G) before the rescaling to 1: 1 for full-rank
            # gradients, below 1 by as much as the unrescaled nuclear-norm term would have
            # underestimated the denominator.
            'muon_polyak_dir_rms': math.sqrt(muon_dir_sq / muon_numel) if muon_numel else 0.0,
            'sf_beta': sf_beta,
            'ckp1': ckp1 if self.param_groups else 1.0,
            'grad_clip_coeff': grad_clip_coeff,
        }
        self.function_value = None
        self.grad_clip_coeff = 1.0
        return None
