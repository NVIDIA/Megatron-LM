# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Songlin Yang, Jan Kautz, Ali Hatamizadeh.

# Some of this code was adopted from https://github.com/huggingface/transformers
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Optional

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
    chunk_gated_delta_rule,
    get_parameter_local_cp,
    torch_chunk_gated_delta_rule,
)
from megatron.core.utils import deprecate_inference_params, nvtx_range_pop, nvtx_range_push


class GatedDeltaNet(_GDNBase):
    # pylint: disable=missing-class-docstring
    def _setup_variant_attrs(self):
        """Set the GDN in_proj sizing, split tables, gate parameter dims, and kernel."""
        # alpha, beta
        self.in_proj_extra_dim = self.num_value_heads * 2

        # Per-section sizes (and names) of the in_proj output, local to this TP rank.
        # Used for the CP head permutation (pre-a2a), for splitting the projection
        # output (post-a2a), and for the sharded checkpoint split of in_proj.weight.
        self.in_proj_split_names = ["query", "key", "value", "z", "beta", "alpha"]
        self.in_proj_split_sections = (
            self.qk_dim_local_tp,  # q
            self.qk_dim_local_tp,  # k
            self.v_dim_local_tp,  # v
            self.v_dim_local_tp,  # gate (z)
            self.num_value_heads // self.tp_size,  # beta
            self.num_value_heads // self.tp_size,  # alpha
        )
        self.feat_dim_split = (
            (self.qk_dim_local_tp * 2 + self.v_dim_local_tp) // self.cp_size,  # qkv
            self.v_dim_local_tp // self.cp_size,  # gate (z)
            self.num_value_heads // self.tp_size // self.cp_size,  # beta
            self.num_value_heads // self.tp_size // self.cp_size,  # alpha
        )

        self.dt_bias_dim = self.num_v_heads_local_tp
        self.a_log_dim = self.num_v_heads_local_tp

        if self.config.deterministic_mode:
            self.gated_delta_rule = torch_chunk_gated_delta_rule
        else:
            self.gated_delta_rule = chunk_gated_delta_rule

    @jit_fuser
    def _compute_gates(
        self,
        A_log_local_cp: torch.Tensor,
        dt_bias_local_cp: torch.Tensor,
        batch: int,
        seq_len: int,
        *gate_feats: tuple[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the per-head log-decay g and the write strength beta."""
        # ``gate_feats`` arrives in ``in_proj_split_names`` order: beta, then alpha.
        beta, alpha = gate_feats
        g = -A_log_local_cp.exp() * F.softplus(alpha.float() + dt_bias_local_cp)  # In fp32
        beta = beta.sigmoid()
        return g, {"beta": beta.contiguous()}

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Perform a forward pass through the GDN module.

        Return:
            (tuple[torch.Tensor, torch.Tensor]) GDN output and bias.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        seq_len, batch, _ = hidden_states.shape
        seq_len = seq_len * self.sp_size * self.cp_size

        if inference_context is not None:
            assert (
                inference_context.is_static_batching()
            ), "GDN does not currently support dynamic inference batching."
            assert not self.config.sequence_parallel
            # TODO: support inference
            raise NotImplementedError("GDN does not support inference for now.")

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            assert batch == 1, "Packed sequence expects batch dimension to be 1"
            assert (
                not self.config.deterministic_mode
            ), "Packed sequence does not support deterministic mode."

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
        qkvzba, _ = self.in_proj(hidden_states)
        nvtx_range_pop(suffix="in_proj")

        qkvzba, thd_cp_a2a_inv = a2a_cp_to_hp(
            qkvzba,
            self.in_proj_split_sections,
            self.cp_size,
            self.pg_collection.cp,
            cu_seqlens_q,
            seq_len,
            packed_seq_params,
        )

        # Transpose: s b x --> b s x
        # From sbhd to bshd format
        qkvzba = qkvzba.transpose(0, 1)

        # Split the tensor into q, k, v, gate (z), and the variant-specific gate features
        # (beta, alpha for GDN; f, b, w for GDN2)
        qkv, gate, beta, alpha = torch.split(qkvzba, self.feat_dim_split, dim=-1)
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
        if self.config.deterministic_mode:
            qkv = qkv.transpose(1, 2).contiguous()  # b, s, d -> b, d, s
            conv_out = F.conv1d(
                input=qkv,  # Torch-native only accept [b, d, s] format input
                weight=conv1d_weight,
                bias=conv1d_bias,
                stride=self.conv1d.stride,
                padding=self.conv1d.padding,
                dilation=self.conv1d.dilation,
                groups=self.conv_dim_local_tp // self.cp_size,
            )
            qkv = self.act_fn(conv_out[..., :seq_len])
            qkv = qkv.transpose(1, 2)  # b, d, s -> b, s, d
        else:
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
            qkv, gate, A_log_local_cp, dt_bias_local_cp, batch, seq_len, beta, alpha
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
