# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Gated Residual: the Qwen4-Exp hyper-connection variant.

Implements the 4-stream residual mechanism of Qwen4-Exp (HF ``model_type:
qwen4_exp``): a low-rank per-channel read gate over group-RMS-normalized
streams, a per-stream scalar write gate, and an identity residual (no
cross-stream mixing matrix).

The module shares :class:`~megatron.core.transformer.hyper_connection.HyperConnectionModule`'s
4-tuple API — ``forward`` returns ``(mixed, h_res, h_post, residual)`` with
``h_res`` always ``None`` and ``h_post`` carrying the write gate — so the
per-sublayer call sites in ``transformer_layer.py`` / ``hybrid_block.py`` need
no changes. The write-back is ``residual + h_post ⊙ (x + bias)``, i.e. mHC's
``native_h_post_bda`` with the ``bmm(H_res, x)`` term replaced by the residual
itself.

Reference: ``Qwen4ExpTextGatedResidual`` in huggingface/transformers
(``src/transformers/models/qwen4_exp/``), Apache-2.0.
"""

from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import nvtx_decorator

if TYPE_CHECKING:
    from megatron.core.tensor_parallel.random import CheckpointWithoutOutputManager


@torch.compile
def grouped_rms_norm(
    x: Tensor, weight: Tensor, group_size: int, eps: float, zero_centered_gamma: bool
) -> Tensor:
    """RMS-normalize ``[..., n*C]`` in independent groups of ``group_size``.

    Runs in fp32 and returns fp32 (the caller decides when to downcast); the
    learnable gamma is applied as ``out * (1 + weight)`` when
    ``zero_centered_gamma`` to match the Qwen4-Exp reference exactly.
    """
    x = x.float()
    grouped = x.view(*x.shape[:-1], -1, group_size)
    normed = grouped * torch.rsqrt(grouped.pow(2).mean(-1, keepdim=True) + eps)
    normed = normed.reshape_as(x)
    weight = weight.float()
    if zero_centered_gamma:
        return normed * (1.0 + weight)
    return normed * weight


class GroupedRMSNorm(nn.Module):
    """RMSNorm over ``[..., dim]`` applied in independent groups of ``group_size``.

    Equivalent to ``dim // group_size`` independent ``RMSNorm(group_size)``
    instances sharing one ``[dim]`` weight, matching the HF tensor layout
    (``Qwen4ExpTextRMSNorm(dim, group_size=...)``). Normalization runs in fp32
    and the output is cast back to the input dtype.
    """

    def __init__(
        self, dim: int, group_size: int, eps: float = 1e-6, zero_centered_gamma: bool = True
    ):
        super().__init__()
        if dim % group_size != 0:
            raise ValueError(f"dim ({dim}) must be divisible by group_size ({group_size})")
        self.dim = dim
        self.group_size = group_size
        self.eps = eps
        self.zero_centered_gamma = zero_centered_gamma
        init = torch.zeros(dim) if zero_centered_gamma else torch.ones(dim)
        self.weight = nn.Parameter(init)

    def forward(self, x: Tensor) -> Tensor:
        """Normalize and cast back to the input dtype."""
        return grouped_rms_norm(
            x, self.weight, self.group_size, self.eps, self.zero_centered_gamma
        ).type_as(x)

    def extra_repr(self) -> str:
        """Shape/eps summary for module printouts."""
        return (
            f"{self.dim}, group_size={self.group_size}, eps={self.eps}, "
            f"zero_centered_gamma={self.zero_centered_gamma}"
        )


# The three fused building blocks of the native path. Kept as free functions so
# torch.compile caches one graph per shape family rather than per module
# instance, mirroring the native mHC kernels in hyper_connection.py.


@torch.compile
def _native_read_gate_mix(
    x_norm_f32: Tensor, x_norm_gemm: Tensor, down_weight: Tensor, up_weight: Tensor, n: int, C: int
) -> Tensor:
    """The whole read-gate chain: down GEMM, silu, up GEMM, sigmoid, gated mean.

    ``x_norm_gemm`` is the normed input pre-cast to the gate-projection weight
    dtype, so both GEMMs run in the parameter dtype while silu, sigmoid and the
    mean aggregation run in fp32 (the gate multiplies every channel of every
    stream, so quantization noise here propagates into the whole residual path).
    """
    down = F.linear(x_norm_gemm, down_weight)
    u = F.silu(down.float() / n).to(x_norm_gemm.dtype)
    g_read = torch.sigmoid(F.linear(u, up_weight).float())
    shape = x_norm_f32.shape[:-1]
    mixed = (g_read.view(*shape, n, C) * x_norm_f32.view(*shape, n, C)).mean(dim=-2)
    # Downcast inside the compiled region so inductor fuses it into the mean
    # epilogue instead of a separate cast kernel per module.
    return mixed.to(x_norm_gemm.dtype)


@torch.compile
def _native_write_gate(
    x_norm_f32: Tensor, inject_weight: Tensor, n: int, out_dtype: torch.dtype
) -> Tensor:
    """Per-stream scalar write gate ``2*sigmoid(inject(x̃)/n)``, computed in fp32
    and downcast to ``out_dtype`` inside the compiled region."""
    gate = 2.0 * torch.sigmoid(F.linear(x_norm_f32, inject_weight.float()) / n)
    return gate.to(out_dtype)


@torch.compile
def _native_gr_write_back(
    residual: Tensor, h_post: Tensor, x: Tensor, bias: Optional[Tensor]
) -> Tensor:
    """``residual + h_post ⊙ (x [+ bias])`` broadcast over the n streams.

    Args:
        residual: [s, b, n*C] - the unnormalized n-stream residual
        h_post: [s, b, n] - write gate
        x: [s, b, C] - sublayer output
        bias: [C] or None
    """
    s, b, nC = residual.shape
    n = h_post.shape[-1]
    C = nC // n
    if bias is not None:
        x = x + bias
    injection = h_post.unsqueeze(-1) * x.unsqueeze(2)  # [s, b, n, C]
    return residual + injection.view(s, b, nC)


class GatedResidualModule(MegatronModule):
    """Qwen4-Exp style hyper-connection: low-rank per-channel read gate +
    per-stream scalar write gate + identity residual.

    Shares the 4-tuple ``forward`` / ``fused_h_res_h_post_bda`` contract with
    :class:`HyperConnectionModule`, so ``TransformerLayer`` / hybrid call sites
    do not distinguish the implementations. Differences from mHC:

    - the ``h_res`` slot of the 4-tuple is always ``None`` (identity residual,
      no Sinkhorn/bmm);
    - the read gate is per-channel ``[s, b, n*C]`` through a low-rank
      bottleneck (``hc_lowrank``), aggregated with ``mean`` rather than ``sum``;
    - normalization is an explicit :class:`GroupedRMSNorm` with learnable
      gamma instead of mHC's implicit RMS scaling.

    dtype policy (see the design discussion): the norm weight and the write
    gate run in fp32 (``mark_keep_in_fp32``); the two low-rank GEMMs run in the
    parameter dtype (bf16) because one of their dimensions is only
    ``hc_lowrank`` wide; nonlinearities and the mean aggregation run in fp32;
    the residual streams themselves are never upcast.

    Args:
        config: TransformerConfig with hyper-connection fields
            (``mhc_num_residual_streams``, ``hc_lowrank``).
        layer_number: Current layer index (kept for parity with
            ``HyperConnectionModule``'s constructor signature).
        use_combine: When True (per-sublayer instance) build the write gate and
            return the 4-tuple. When False (block-exit contract instance) the
            module only computes ``mixed`` — the n→1 contraction.
    """

    def __init__(self, config: TransformerConfig, layer_number: int, use_combine: bool = True):
        super().__init__(config)
        self.config = config
        self.layer_number = layer_number
        self.n = config.mhc_num_residual_streams
        self.hidden_size = config.hidden_size
        self.lowrank = config.hc_lowrank
        self.use_combine = use_combine
        nC = self.n * self.hidden_size

        self.hc_norm = GroupedRMSNorm(
            nC,
            group_size=self.hidden_size,
            eps=config.layernorm_epsilon,
            zero_centered_gamma=config.layernorm_zero_centered_gamma,
        )
        self.input_mix_weight_down = nn.Linear(nC, self.lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(self.lowrank, nC, bias=False)
        self.block_inject_weight = nn.Linear(nC, self.n, bias=False) if use_combine else None

        # The norm gamma and the write gate generate bounded mixing
        # coefficients; keep them fp32 like mHC's mapping path. The two large
        # low-rank GEMMs stay in the parameter dtype.
        mark_keep_in_fp32(self.hc_norm.weight)
        if self.block_inject_weight is not None:
            mark_keep_in_fp32(self.block_inject_weight.weight)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize gate projections and mark params for SP grad all-reduce."""
        self.config.init_method(self.input_mix_weight_down.weight)
        self.config.init_method(self.input_mix_weight_up.weight)
        if self.block_inject_weight is not None:
            self.config.init_method(self.block_inject_weight.weight)

        # Gated-residual weights are replicated across TP ranks (non-TP-aware
        # nn.Linear); with sequence parallelism each rank sees s/TP of the
        # sequence, so their gradients must be marked for the TP all-reduce.
        if self.config.sequence_parallel:
            for param in self.parameters():
                setattr(param, 'sequence_parallel', True)

    # ==================== gate computation ====================

    def _compute_mixed_and_gwrite(self, hidden_states: Tensor) -> Tuple[Tensor, ...]:
        """Compute ``mixed`` (and the write gate when ``use_combine``) from x.

        Everything here is recomputable from ``hidden_states`` alone, which is
        what makes the single-checkpoint recompute path below exact.
        """
        # The compute dtype is the gate-projection *weight* dtype (params dtype),
        # not the activation dtype: with fp32_residual_connection the n-stream
        # hidden states arrive in fp32 while the gate weights are bf16, and
        # F.linear (and the downstream TE sublayer GEMMs consuming ``mixed``)
        # require matching dtypes. In the common case both dtypes agree and this
        # is a no-op.
        gemm_dtype = self.input_mix_weight_down.weight.dtype
        x_norm_f32 = grouped_rms_norm(
            hidden_states,
            self.hc_norm.weight,
            self.hidden_size,
            self.hc_norm.eps,
            self.hc_norm.zero_centered_gamma,
        )
        mixed = _native_read_gate_mix(
            x_norm_f32,
            x_norm_f32.to(gemm_dtype),
            self.input_mix_weight_down.weight,
            self.input_mix_weight_up.weight,
            self.n,
            self.hidden_size,
        )
        if not self.use_combine:
            return (mixed,)
        g_write = _native_write_gate(
            x_norm_f32, self.block_inject_weight.weight, self.n, gemm_dtype
        )
        return mixed, g_write

    @nvtx_decorator(message="GatedResidual::forward")
    def forward(
        self,
        hidden_states: Tensor,
        mhc_recompute_manager: Optional['CheckpointWithoutOutputManager'] = None,
        return_residual: bool = True,
    ) -> Union[Tensor, Tuple[Tensor, None, Tensor, Tensor]]:
        """Compute the read-gated mix and (for sublayer instances) the 4-tuple.

        Returns ``(mixed, None, g_write, residual)`` aligned with
        ``HyperConnectionModule.forward``'s ``(aggregated, h_res, h_post,
        residual)``. Block-exit instances (``use_combine=False``) return just
        ``mixed`` — the n→1 output contraction.

        Args:
            hidden_states: [s, b, n*C] - n-stream hidden states
            mhc_recompute_manager: When provided, the gate computation is
                wrapped in ``CheckpointWithoutOutput`` so only ``mixed`` /
                ``g_write`` survive the forward (recomputed from the residual,
                which is alive anyway).
            return_residual: Accepted for signature compatibility with
                ``HyperConnectionModule.forward``. Sublayer instances always return the
                residual as the 4-tuple's last element, so ``False`` has no meaning here.
        """
        if self.use_combine and not return_residual:
            raise ValueError(
                "GatedResidualModule always returns the residual with the 4-tuple; "
                "return_residual=False is not supported."
            )

        if mhc_recompute_manager is not None:
            from megatron.core.tensor_parallel.random import CheckpointWithoutOutput

            outputs = CheckpointWithoutOutput(ckpt_manager=mhc_recompute_manager).checkpoint(
                self._compute_mixed_and_gwrite, hidden_states
            )
        else:
            outputs = self._compute_mixed_and_gwrite(hidden_states)

        if not self.use_combine:
            return outputs[0]
        mixed, g_write = outputs
        return mixed, None, g_write, hidden_states

    # ==================== write-back ====================

    @nvtx_decorator(message="GatedResidual::fused_h_res_h_post_bda")
    def fused_h_res_h_post_bda(
        self,
        h_res: Optional[Tensor],
        original_residual: Tensor,
        h_post: Tensor,
        layer_output_with_bias: Tuple[Tensor, Optional[Tensor]],
        dropout_prob: float,
        training: bool,
        fused: bool,
        manager: Optional['CheckpointWithoutOutputManager'] = None,
    ) -> Tensor:
        """Write-back: ``residual + h_post ⊙ (x + bias)`` (identity residual).

        Signature-compatible with ``HyperConnectionModule.fused_h_res_h_post_bda``;
        ``h_res`` must be ``None`` (the gated-residual variant has no
        cross-stream mixing).
        """
        # An explicit raise (not an assert, which python -O strips): silently
        # ignoring a real h_res would drop cross-stream mixing a caller asked for.
        if h_res is not None:
            raise TypeError("GatedResidualModule write-back takes no h_res (got a tensor)")
        x, bias = layer_output_with_bias

        if dropout_prob == 0.0 or not training:
            if manager is not None:
                from megatron.core.tensor_parallel.random import CheckpointWithoutOutput

                # Positional args must all be Tensors; optional bias is handled
                # by varargs presence, never by passing None.
                def _wrapper(residual, h_post, x, *optional_bias):
                    b_arg = optional_bias[0] if optional_bias else None
                    return _native_gr_write_back(residual, h_post, x, b_arg)

                ckpt = CheckpointWithoutOutput(ckpt_manager=manager)
                if bias is not None:
                    return ckpt.checkpoint(_wrapper, original_residual, h_post, x, bias)
                return ckpt.checkpoint(_wrapper, original_residual, h_post, x)
            return _native_gr_write_back(original_residual, h_post, x, bias)

        # Dropout path: dropout applies to the injected delta, the residual is
        # added unscaled — matching mHC's structure with the bmm-mixed residual
        # replaced by the identity residual.
        from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add

        bda_func = get_bias_dropout_add(training, fused)
        has_bias = bias is not None

        def _dropout_write_back(residual, h_post, x, *optional_bias):
            s, b, nC = residual.shape
            x_expanded = (h_post.unsqueeze(-1) * x.unsqueeze(2)).view(s, b, nC)
            if optional_bias:
                bias_expanded = (h_post.unsqueeze(-1) * optional_bias[0].view(1, 1, 1, -1)).view(
                    s, b, nC
                )
            else:
                bias_expanded = None
            return bda_func((x_expanded, bias_expanded), residual, dropout_prob)

        if manager is not None:
            from megatron.core.tensor_parallel.random import CheckpointWithoutOutput

            ckpt = CheckpointWithoutOutput(ckpt_manager=manager)
            if has_bias:
                return ckpt.checkpoint(_dropout_write_back, original_residual, h_post, x, bias)
            return ckpt.checkpoint(_dropout_write_back, original_residual, h_post, x)
        if has_bias:
            return _dropout_write_back(original_residual, h_post, x, bias)
        return _dropout_write_back(original_residual, h_post, x)

    # Block-level stream expansion is replication, identical to mHC (and to
    # Qwen4-Exp's ``repeat(1, 1, hc_count)``); the block-entry call sites use
    # ``HyperConnectionModule.input_expand`` directly because expansion is
    # variant-invariant, so no alias is exposed here.
