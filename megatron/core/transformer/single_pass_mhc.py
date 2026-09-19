# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Single-pass mHC (DeepSeek-V4.1 report, Eq. 6).

Unlike ordinary mHC, a sublayer consumes the *preceding* sublayer's input
mix. The caller carries that tensor explicitly, including across recomputation;
no module retains activations belonging to a microbatch.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core.transformer.hyper_connection import HyperConnectionModule


def contract_streams(hidden_states: Tensor, pre_mix: Tensor | None, n: int) -> Tensor:
    """Contract flattened [sequence, batch, n * hidden] streams in FP32.

    A missing predecessor at model entry selects stream zero. At model exit,
    use the last feed-forward sublayer's mix, without a separate learned head.
    """
    if n < 1 or hidden_states.ndim != 3 or hidden_states.shape[-1] % n:
        raise ValueError("Expected [sequence, batch, n * hidden] residual streams")
    streams = hidden_states.unflatten(-1, (n, -1))
    if pre_mix is None:
        return streams[..., 0, :].contiguous()
    if pre_mix.shape != hidden_states.shape[:2] + (n,):
        raise ValueError("The preceding mHC mix must have shape [sequence, batch, n]")
    return (streams.float() * pre_mix.float().unsqueeze(-1)).sum(-2).to(hidden_states.dtype)


class SinglePassHyperConnection(HyperConnectionModule):
    """FP32 coefficient prediction and residual mixing with an explicit mix handoff.

    Parameter names match ordinary mHC so its checkpoint sharding and parameter
    precision rules apply. This reference path intentionally preserves the FP32
    coefficients used by DeepSeek rather than rounding them to the residual dtype.
    """

    def __init__(self, config, layer_number: int, epsilon: float = 1e-6) -> None:
        super().__init__(config, layer_number)
        self.norm_eps = config.layernorm_epsilon
        self.compute_h_eps = self.sinkhorn_eps = epsilon

    def compute_mappings(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict coefficients before mixing, retaining full precision and gradients."""
        x = hidden_states.float()
        # V4's fused coefficient projection adds epsilon after sqrt. V4.1 adds
        # it inside sqrt, so keep that projection explicit and fuse the other
        # operations without changing the zero/small-input gradient.
        with torch.autocast(device_type=x.device.type, enabled=False):
            projected = F.linear(x, self.mapping_proj.weight.float())
        projected = projected * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        pre = (projected[..., : self.n] * self.alpha_pre + self.bias[: self.n]).sigmoid()
        post = (
            projected[..., self.n : 2 * self.n] * self.alpha_post + self.bias[self.n : 2 * self.n]
        ).sigmoid()
        residual = (
            projected[..., 2 * self.n :] * self.alpha_res + self.bias[2 * self.n :]
        ).unflatten(-1, (self.n, self.n))
        if self.config.use_fused_mhc:
            residual = self._sinkhorn_op(residual, self.sinkhorn_iterations, self.sinkhorn_eps)
            return pre + self.compute_h_eps, 2 * post, residual
        residual = residual.softmax(-1) + self.sinkhorn_eps
        residual = residual / (residual.sum(-2, keepdim=True) + self.sinkhorn_eps)
        for _ in range(self.sinkhorn_iterations - 1):
            residual = residual / (residual.sum(-1, keepdim=True) + self.sinkhorn_eps)
            residual = residual / (residual.sum(-2, keepdim=True) + self.sinkhorn_eps)
        return pre + self.compute_h_eps, 2 * post, residual

    def forward(
        self, hidden_states: Tensor, previous_mix: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return branch input, next mix, output mix, and residual mix."""
        next_mix, post, residual = self.compute_mappings(hidden_states)
        if self.config.use_fused_mhc and previous_mix is not None:
            branch = self._h_aggregate_op(
                hidden_states.unflatten(-1, (self.n, self.hidden_size)), previous_mix
            )
        else:
            branch = contract_streams(hidden_states, previous_mix, self.n)
        return branch.to(hidden_states.dtype), next_mix, post, residual

    def combine(
        self, branch_output: Tensor, hidden_states: Tensor, post: Tensor, residual: Tensor
    ) -> Tensor:
        """Apply H_res transpose and H_post in FP32, then cast once to activation dtype."""
        if self.config.use_fused_mhc:
            streams = hidden_states.float().unflatten(-1, (self.n, self.hidden_size))
            return (
                self._h_post_bda_op(residual, streams, post, branch_output.float(), None)
                .flatten(-2)
                .to(branch_output.dtype)
            )
        streams = hidden_states.float().unflatten(-1, (self.n, self.hidden_size))
        mixed = torch.einsum("sbij,sbic->sbjc", residual.float(), streams)
        expanded = post.float().unsqueeze(-1) * branch_output.float().unsqueeze(-2)
        return (mixed + expanded).flatten(-2).to(branch_output.dtype)
