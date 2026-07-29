# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Songlin Yang, Jan Kautz, Ali Hatamizadeh.

# Some of this code was adopted from https://github.com/huggingface/transformers
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from functools import partial

import torch
import torch.nn.functional as F

from megatron.core import tensor_parallel
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.jit import jit_fuser
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm.gated_delta_net.common import (
    _GDNBase,
    a2a_cp_to_hp,
    causal_conv1d,
    get_parameter_local_cp,
)
from megatron.core.utils import deprecate_inference_params, nvtx_range_pop, nvtx_range_push

try:
    # The GDN2 kernel is only available in flash-linear-attention >= 0.5.1.
    from fla.ops.gdn2.chunk import chunk_gdn2
except ImportError:
    chunk_gdn2 = None

logger = logging.getLogger(__name__)


class GatedDeltaNet2(_GDNBase):
    """GDN2 (Gated DeltaNet-2) layer class.

    GDN2 replaces GDN's per-head scalar decay and write strength with channel-wise
    gates, decoupling erase and write:

        S_t = (I - k_t (b_t * k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t * v_t)^T

    where ``g_t`` is a per-key-channel log-decay, ``b_t`` (in R^{d_k}) is the
    channel-wise erase gate, and ``w_t`` (in R^{d_v}) is the channel-wise write gate.
    Reference: "Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention"
    (https://github.com/NVlabs/GatedDeltaNet-2).

    Note: unlike the GDN2 reference implementation, which uses low-rank decay and
    output-gate projections, all GDN2 projections are fused full-rank into the single
    column-parallel in_proj for TP/CP/SP simplicity.

    The layer takes input with size [s, b, h] and returns output of the same size.
    """

    def _setup_variant_attrs(self):
        """Set the GDN2 in_proj sizing, split tables, gate parameter dims, and kernel."""
        assert (
            chunk_gdn2 is not None
        ), "GDN2 requires flash-linear-attention >= 0.5.1 with the fla.ops.gdn2 kernel."
        assert (
            not self.config.deterministic_mode
        ), "GDN2 has no torch-native implementation for deterministic mode."

        # f (decay pre-activation), b (erase gate), w (write gate), on top of the
        # q/k/v/z sections the base class already accounts for.
        # TODO: for now, output gate is forced for GDN2.
        # We may remove this restriction in the future.
        self.in_proj_extra_dim = self.qk_dim * 2 + self.v_dim

        # Per-section sizes (and names) of the in_proj output, local to this TP rank.
        # Used for the CP head permutation (pre-a2a), for splitting the projection
        # output (post-a2a), and for the sharded checkpoint split of in_proj.weight.
        self.in_proj_split_names = ["query", "key", "value", "z", "f", "b", "w"]
        self.in_proj_split_sections = (
            self.qk_dim_local_tp,  # q
            self.qk_dim_local_tp,  # k
            self.v_dim_local_tp,  # v
            self.v_dim_local_tp,  # gate (z)
            self.qk_dim_local_tp,  # f (decay pre-activation)
            self.qk_dim_local_tp,  # b (erase gate)
            self.v_dim_local_tp,  # w (write gate)
        )
        self.feat_dim_split = (
            (self.qk_dim_local_tp * 2 + self.v_dim_local_tp) // self.cp_size,  # qkv
            self.v_dim_local_tp // self.cp_size,  # gate (z)
            self.qk_dim_local_tp // self.cp_size,  # f
            self.qk_dim_local_tp // self.cp_size,  # b
            self.v_dim_local_tp // self.cp_size,  # w
        )

        # Time step projection (discretization): per-key-channel dt_bias and
        # per-key-head A_log, following the GDN2 reference implementation.
        self.dt_bias_dim = self.qk_dim_local_tp
        self.a_log_dim = self.num_k_heads_local_tp

        self.gated_delta_rule = chunk_gdn2

    def _reset_dt_bias(self):
        """Softplus-inverse init of dt_bias.

        Initializes so the initial per-channel step size lands in [1e-3, 0.1],
        following the GDN2 reference implementation.
        """
        dt = torch.exp(
            torch.rand(
                self.dt_bias.shape[0], dtype=torch.float32, device=torch.cuda.current_device()
            )
            * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias.data.copy_(inv_dt)

    @jit_fuser
    def _compute_gates(
        self,
        A_log_local_cp: torch.Tensor,
        dt_bias_local_cp: torch.Tensor,
        batch: int,
        seq_len: int,
        *gate_feats: tuple[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the per-channel log-decay g and the erase/write gates b/w."""
        f, b, w = gate_feats
        # Channel-wise log-decay, computed in fp32 for numerical stability. A_log is a
        # per-key-head rate broadcast over the head's key channels; dt_bias is per-channel.
        g = -A_log_local_cp.float().exp().repeat_interleave(self.key_head_dim) * F.softplus(
            f.float() + dt_bias_local_cp
        )
        g = g.reshape(batch, seq_len, -1, self.key_head_dim)

        # Channel-wise erase (key axis) and write (value axis) gates, squashed to [0, 1]
        b = b.sigmoid().reshape(batch, seq_len, -1, self.key_head_dim)
        w = w.sigmoid().reshape(batch, seq_len, -1, self.value_head_dim)

        # Expand key-side gates across value-head groups (grouped value attention)
        repeat_factor = self.num_value_heads // self.num_key_heads
        if repeat_factor > 1:
            g = g.repeat_interleave(repeat_factor, dim=2)
            b = b.repeat_interleave(repeat_factor, dim=2)

        return g, {"b": b.contiguous(), "w": w.contiguous()}

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        inference_context: BaseInferenceContext | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: int | None = None,
        *,
        inference_params: BaseInferenceContext | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Perform a forward pass through the GDN2 module.

        Return:
            (tuple[torch.Tensor, torch.Tensor]) GDN2 output and bias.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        seq_len, batch, _ = hidden_states.shape
        seq_len = seq_len * self.sp_size * self.cp_size

        if inference_context is not None:
            assert (
                inference_context.is_static_batching()
            ), "GDN2 does not currently support dynamic inference batching."
            assert not self.config.sequence_parallel
            # TODO: support inference
            raise NotImplementedError("GDN2 does not support inference for now.")

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            assert batch == 1, "Packed sequence expects batch dimension to be 1"

            # Resolve cu_seqlens with alignment padding handling.
            cu_seqlens_q = self._resolve_cu_seqlens(
                packed_seq_params.cu_seqlens_q_padded,
                packed_seq_params.cu_seqlens_q,
                seq_len,
                "cu_seqlens_q",
                cp_size=self.cp_size,
            )
            cu_seqlens_kv = self._resolve_cu_seqlens(
                packed_seq_params.cu_seqlens_kv_padded,
                packed_seq_params.cu_seqlens_kv,
                seq_len,
                "cu_seqlens_kv",
                cp_size=self.cp_size,
            )
            assert torch.equal(cu_seqlens_q, cu_seqlens_kv), (
                "Currently only support cu_seqlens_q equals to cu_seqlens_kv, "
                f"but got {cu_seqlens_q=} and {cu_seqlens_kv=}"
            )
            num_packed_seqs = cu_seqlens_q.shape[0] - 1
            assert num_packed_seqs > 0, (
                "Number of packed sequences must be greater than 0, "
                f"but got {cu_seqlens_q=} and {cu_seqlens_kv=}"
            )
        else:
            cu_seqlens_q = None
            cu_seqlens_kv = None

        # Input projection
        nvtx_range_push(suffix="in_proj")
        qkvzfbw, _ = self.in_proj(hidden_states)
        nvtx_range_pop(suffix="in_proj")

        qkvzfbw, thd_cp_a2a_inv = a2a_cp_to_hp(
            qkvzfbw,
            self.in_proj_split_sections,
            self.cp_size,
            self.pg_collection.cp,
            cu_seqlens_q,
            seq_len,
            packed_seq_params,
        )

        # Transpose: s b x --> b s x
        # From sbhd to bshd format
        qkvzfbw = qkvzfbw.transpose(0, 1)

        # Split the tensor into q/k/v, gate (z), and the GDN2 gate features f, b, w
        qkv, gate, f, b, w = torch.split(qkvzfbw, self.feat_dim_split, dim=-1)
        gate = gate.reshape(batch, seq_len, -1, self.value_head_dim)

        # Convolution on qkv
        nvtx_range_push(suffix="conv1d")
        seq_len = qkv.shape[1]
        qkv_channels_split_sections = [
            self.qk_dim_local_tp,
            self.qk_dim_local_tp,
            self.v_dim_local_tp,
        ]
        conv1d_weight = get_parameter_local_cp(
            self.conv1d.weight,
            dim=0,
            cp_group=self.pg_collection.cp,
            split_sections=qkv_channels_split_sections,
        )
        conv1d_bias = (
            get_parameter_local_cp(
                self.conv1d.bias,
                dim=0,
                cp_group=self.pg_collection.cp,
                split_sections=qkv_channels_split_sections,
            )
            if self.conv_bias
            else None
        )
        assert self.activation in ["silu", "swish"]
        qkv, _ = causal_conv1d(
            x=qkv,  # FLA conv1d accepts [b, s, d] format input
            weight=conv1d_weight.squeeze(1),  # d, 1, w -> d, w
            bias=conv1d_bias,
            activation=self.activation,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=cu_seqlens_q,
        )
        nvtx_range_pop(suffix="conv1d")

        A_log_local_cp = get_parameter_local_cp(self.A_log, dim=0, cp_group=self.pg_collection.cp)
        dt_bias_local_cp = get_parameter_local_cp(
            self.dt_bias, dim=0, cp_group=self.pg_collection.cp
        )

        # Prepare all kernel inputs (split, reshape, L2 norm, gates, contiguous)
        nvtx_range_push(suffix="prepare_input_for_gated_delta_rule")
        kernel_inputs = self._prepare_input_for_gated_delta_rule(
            qkv, gate, A_log_local_cp, dt_bias_local_cp, batch, seq_len, f, b, w
        )
        gate = kernel_inputs.pop("gate")
        nvtx_range_pop(suffix="prepare_input_for_gated_delta_rule")

        nvtx_range_push(suffix="gated_delta_rule")
        core_attn_out, _ = self.gated_delta_rule(
            **kernel_inputs,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=cu_seqlens_q,
        )
        nvtx_range_pop(suffix="gated_delta_rule")

        if self.recompute_norm_out:
            self.norm_out_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            norm_func = partial(
                self._gated_norm_and_a2a,
                thd_cp_a2a_inv=thd_cp_a2a_inv,
                batch=batch,
                seq_len=seq_len,
                packed_seq_params=packed_seq_params,
            )
            norm_out = self.norm_out_checkpoint.checkpoint(norm_func, core_attn_out, gate)
        else:
            norm_out = self._gated_norm_and_a2a(
                core_attn_out, gate, thd_cp_a2a_inv, batch, seq_len, packed_seq_params
            )

        # Output projection
        nvtx_range_push(suffix="out_proj")
        out, out_bias = self.out_proj(norm_out)
        nvtx_range_pop(suffix="out_proj")

        if self.recompute_norm_out:
            self.norm_out_checkpoint.discard_output_and_register_recompute(out)

        return out, out_bias
