# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Songlin Yang, Jan Kautz, Ali Hatamizadeh.

# Some of this code was adopted from https://github.com/huggingface/transformers
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Optional

import torch

from megatron.core import tensor_parallel
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm.gated_delta_net.common import (
    _build_thd_cp_a2a_perm,
    _GDNBase,
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

        thd_cp_a2a_idx, thd_cp_a2a_inv = None, None
        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            if self.cp_size > 1:
                # Permute at the seq dim so that a single unsectioned a2a
                # is equivalent to per-sequence a2a.
                # This also folds the ``_undo_attention_load_balancing`` step.
                thd_cp_a2a_idx, thd_cp_a2a_inv = _build_thd_cp_a2a_perm(
                    cu_seqlens_q, self.cp_size, seq_len
                )

        in_proj_conv_func = partial(
            self._in_proj_conv,
            thd_cp_a2a_idx=thd_cp_a2a_idx,
            batch=batch,
            seq_len=seq_len,
            cu_seqlens_q=cu_seqlens_q,
            packed_seq_params=packed_seq_params,
        )

        if self.recompute_in_proj_conv:
            qkv, gate, beta, alpha = tensor_parallel.checkpoint(
                in_proj_conv_func, False, hidden_states
            )
        else:
            qkv, gate, beta, alpha = in_proj_conv_func(hidden_states)

        A_log_local_cp = get_parameter_local_cp(self.A_log, dim=0, cp_group=self.pg_collection.cp)
        dt_bias_local_cp = get_parameter_local_cp(
            self.dt_bias, dim=0, cp_group=self.pg_collection.cp
        )

        # Prepare QKV tensors (split, reshape, L2 norm, repeat_interleave, contiguous)
        nvtx_range_push(suffix="prepare_input_for_gated_delta_rule")
        query, key, value, gate, beta, alpha = self._prepare_input_for_gated_delta_rule(
            qkv, gate, batch, seq_len, beta, alpha
        )
        nvtx_range_pop(suffix="prepare_input_for_gated_delta_rule")

        nvtx_range_push(suffix="g_and_beta")
        g, beta = self._compute_g_and_beta(A_log_local_cp, dt_bias_local_cp, alpha, beta)
        nvtx_range_pop(suffix="g_and_beta")

        nvtx_range_push(suffix="gated_delta_rule")
        core_attn_out, _ = self.gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
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
