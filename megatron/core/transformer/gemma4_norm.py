# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch


class Gemma4RMSNorm(torch.nn.Module):
    """Bitwise-faithful port of HF ``Gemma4RMSNorm`` (modeling_gemma4.py:193-211).

    The compute is intentionally an exact mirror of HF: cast to fp32, normalize
    with ``pow(2).mean(-1) + eps`` and ``torch.pow(., -0.5)`` (NOT ``rsqrt`` — HF
    uses ``pow`` to match JAX), multiply by ``weight.float()`` (plain weight, no
    ``+1``), then cast back to the input dtype only at the end.

    ``with_scale=False`` gives the weightless variant used for the scaleless
    v_norm; it has no parameter and is a pure RMS normalization.
    """

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if self.with_scale:
            self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        mean_squared = x.pow(2).mean(-1, keepdim=True) + self.eps
        return x * torch.pow(mean_squared, -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self._norm(x.float())
        if self.with_scale:
            normed = normed * self.weight.float()
        return normed.type_as(x)


def gemma4_rms_norm_builder(*, config, hidden_size: int, eps: float, **kwargs) -> Gemma4RMSNorm:
    """:class:`LayerNormBuilder` for the weighted :class:`Gemma4RMSNorm` (local spec)."""
    norm = Gemma4RMSNorm(hidden_size, eps=eps, with_scale=True)
    # Under sequence parallelism the norm weight is replicated across TP but each rank only
    # sees a partition of the activations (the sequence shard for the hidden-size sandwich
    # norms; the head shard for the q/k norms), so its gradient is a partial sum. Tagging
    # the weight ``sequence_parallel`` makes grad finalization all-reduce it across TP
    # (megatron/core/distributed/finalize_model_grads.py), recovering the full gradient.
    # This mirrors TENorm (extensions/transformer_engine.py), which passes
    # ``sequence_parallel=config.sequence_parallel`` for the TE spec. No-op when SP is off.
    norm.weight.sequence_parallel = config.sequence_parallel
    return norm


def gemma4_rms_norm_scaleless_builder(
    *, config, hidden_size: int, eps: float, **kwargs
) -> Gemma4RMSNorm:
    """:class:`LayerNormBuilder` for the scaleless v_norm :class:`Gemma4RMSNorm`."""
    return Gemma4RMSNorm(hidden_size, eps=eps, with_scale=False)
