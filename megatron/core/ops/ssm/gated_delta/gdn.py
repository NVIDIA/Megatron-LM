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
from megatron.core.inference.contexts import BaseInferenceContext, DynamicInferenceContext
from megatron.core.inference.contexts.attention_context.triton.tensor_ops import (
    tensor_masked_update,
)
from megatron.core.jit import jit_fuser
from megatron.core.ops.kernel_metadata import KernelMetadata
from megatron.core.ops.ssm.common.inference import SSMDynamicInferenceMixin
from megatron.core.ops.ssm.gated_delta.common import _GDNBase, a2a_cp_to_hp, get_parameter_local_cp
from megatron.core.ops.ssm.gated_delta.kernel_metadata import (
    FLA_CONV_UPDATE,
    GDN_FLA,
    GDN_RECURRENT,
)
from megatron.core.ops.ssm.gated_delta.reference import (
    torch_chunk_gated_delta_rule as torch_chunk_gated_delta_rule,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import deprecate_inference_params, nvtx_range_pop, nvtx_range_push

__all__ = ["GatedDeltaNet", "torch_chunk_gated_delta_rule"]


class GatedDeltaNet(SSMDynamicInferenceMixin, _GDNBase):
    # pylint: disable=missing-class-docstring
    def get_inference_kernel_metadata(self) -> tuple[KernelMetadata, ...]:
        """Declare the additional FLA targets selected by dynamic inference."""
        return (FLA_CONV_UPDATE, GDN_FLA, GDN_RECURRENT)

    def _setup_variant_attrs(self, kernel_backend=None):
        """Set the GDN in_proj sizing, split tables, gate parameter dims, and kernel."""
        from megatron.core.models.backends import backend_slot, get_backend_from_config
        from megatron.core.ops.ssm.gated_delta.backends import select_gated_delta_rule

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

        self.gated_delta_rule = backend_slot(
            backend=(
                kernel_backend
                if kernel_backend is not None
                else get_backend_from_config(self.config)
            ),
            name="gated_delta_rule",
            default=lambda: select_gated_delta_rule("gdn", self.config.deterministic_mode),
            variant="gdn",
            deterministic=self.config.deterministic_mode,
        )
        self.chunk_size = 64

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
            if inference_context.is_dynamic_batching():
                assert (
                    not self.config.deterministic_mode
                ), "GDN dynamic inference requires the FLA recurrent kernels."
                assert (
                    not self.config.batch_invariant_mode
                ), "GDN dynamic inference does not support batch-invariant mode."
                assert (
                    self.cp_size == 1
                ), "Context parallelism is not supported for GDN dynamic inference."
                assert (
                    inference_context.num_speculative_tokens == 0
                ), "GDN dynamic inference does not support speculative decoding."
                assert (
                    not inference_context.enable_prefix_caching
                ), "GDN dynamic inference does not support prefix caching."
                return self.ssm_dynamic_inference(hidden_states, inference_context)
            assert inference_context.is_static_batching()
            assert not self.config.sequence_parallel
            raise NotImplementedError("GDN static-batching inference is not supported.")

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
        qkv, gate, beta, alpha = self._split_projection(qkvzba, batch, seq_len)

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
            qkv, _ = self.causal_conv1d(
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

    def _split_projection(
        self, projected: torch.Tensor, batch: int, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the fused projection into qkv, output gate, beta, and alpha."""
        qkv, gate, beta, alpha = torch.split(projected, self.feat_dim_split, dim=-1)
        gate = gate.reshape(batch, seq_len, -1, self.value_head_dim)
        return qkv, gate, beta, alpha

    def _prepare_inference_inputs(
        self, qkv: torch.Tensor, beta: torch.Tensor, alpha: torch.Tensor, batch: int, seq_len: int
    ) -> dict[str, torch.Tensor]:
        """Prepare raw FLA inputs while leaving normalization and gates fused in-kernel."""
        query_key, value = torch.split(qkv, [2 * self.qk_dim_local_tp, self.v_dim_local_tp], dim=-1)
        query_key = query_key.reshape(batch, seq_len, -1, self.key_head_dim)
        query, key = torch.chunk(query_key, 2, dim=2)
        value = value.reshape(batch, seq_len, -1, self.value_head_dim)
        return {
            "q": query.contiguous(),
            "k": key.contiguous(),
            "v": value.contiguous(),
            "g": alpha.contiguous(),
            "beta": beta.contiguous(),
        }

    def mamba_state_shapes_per_request(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return the TP-local convolution and delta-rule cache shapes."""
        return (
            (self.conv_dim_local_tp, self.conv_kernel_dim),
            (self.num_v_heads_local_tp, self.key_head_dim, self.value_head_dim),
        )

    def ssm_decode(
        self,
        projected: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        batch_indices: torch.Tensor,
        intermediate_conv_state: torch.Tensor | None = None,
        intermediate_ssm_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one CUDA-graph-compatible GDN decode token per request."""
        batch, seq_len, _ = projected.shape
        assert seq_len == 1, "GDN speculative decoding is not supported."
        assert (
            intermediate_conv_state is None and intermediate_ssm_state is None
        ), "GDN speculative decoding state capture is not supported."
        from fla.modules.convolution import causal_conv1d_update
        from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

        qkv, gate, beta, alpha = self._split_projection(projected, batch, seq_len)
        read_indices = batch_indices.clamp(min=0)

        active_conv_state = conv_state[read_indices].contiguous()
        qkv_dtype = qkv.dtype
        qkv, active_conv_state = causal_conv1d_update(
            x=qkv.to(conv_state.dtype),
            cache=active_conv_state,
            weight=self.conv1d.weight.squeeze(1).to(conv_state.dtype),
            bias=self.conv1d.bias.to(conv_state.dtype) if self.conv1d.bias is not None else None,
            activation=self.activation,
        )
        qkv = qkv.to(qkv_dtype)
        tensor_masked_update(conv_state, batch_indices, active_conv_state)

        kernel_inputs = self._prepare_inference_inputs(qkv, beta, alpha, batch, seq_len)
        active_ssm_state = ssm_state[read_indices].contiguous()
        core_attn_out, final_ssm_state = fused_recurrent_gated_delta_rule(
            **kernel_inputs,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            initial_state=active_ssm_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=self.use_qk_l2norm,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
        )
        tensor_masked_update(ssm_state, batch_indices, final_ssm_state)
        return self._apply_gated_norm(core_attn_out, gate).reshape(batch, seq_len, -1)

    def ssm_prefill(
        self,
        projected: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        context: DynamicInferenceContext,
    ) -> torch.Tensor:
        """Run packed variable-length GDN prefill and populate request states."""
        assert (
            not context.is_chunked_prefill_enabled()
        ), "GDN dynamic inference does not support chunked prefill."
        metadata = context.mamba_metadata
        cu_seqlens = metadata.cu_seqlens
        batch_indices = metadata.batch_indices_prefill
        token_count = projected.shape[0]

        projected = projected.transpose(0, 1).contiguous()
        qkv, gate, beta, alpha = self._split_projection(projected, 1, token_count)
        read_indices = batch_indices.clamp(min=0)

        qkv_dtype = qkv.dtype
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        qkv, final_conv_state = self.causal_conv1d(
            x=qkv.to(conv_state.dtype),
            weight=self.conv1d.weight.squeeze(1).to(conv_state.dtype),
            bias=self.conv1d.bias.to(conv_state.dtype) if self.conv1d.bias is not None else None,
            activation=self.activation,
            initial_state=conv_state[read_indices].contiguous(),
            output_final_state=True,
            cu_seqlens=cu_seqlens,
        )
        qkv = qkv.to(qkv_dtype)
        tensor_masked_update(conv_state, batch_indices, final_conv_state)

        kernel_inputs = self._prepare_inference_inputs(qkv, beta, alpha, 1, token_count)
        core_attn_out, final_ssm_state = chunk_gated_delta_rule(
            **kernel_inputs,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            initial_state=ssm_state[read_indices].contiguous(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=self.use_qk_l2norm,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
        tensor_masked_update(ssm_state, batch_indices, final_ssm_state)
        y = self._apply_gated_norm(core_attn_out, gate)
        return y.reshape(1, token_count, -1).transpose(0, 1).contiguous()
