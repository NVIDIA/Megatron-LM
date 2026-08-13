# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025, Songlin Yang, Jan Kautz, Ali Hatamizadeh.

# Some of this code was adopted from https://github.com/huggingface/transformers
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F

from megatron.core import tensor_parallel
from megatron.core.context_parallel_layout import convert_module_input_tensors_cp_partition_mode
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.jit import jit_fuser
from megatron.core.packed_seq_params import PackedSeqParams, resolve_cp_group
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net.common import (
    _GDNBase,
    a2a_cp_to_hp,
    a2a_hp_to_cp,
    build_cp_context,
    causal_conv1d,
    chunk_gated_delta_rule,
    get_parameter_local_cp,
    l2norm,
)
from megatron.core.ssm.gated_delta_net.internal_gdn_backend import (
    chunk_gated_delta_rule as internal_chunk_gated_delta_rule,
)
from megatron.core.utils import deprecate_inference_params, nvtx_range_pop, nvtx_range_push


class GatedDeltaNet(_GDNBase):
    # pylint: disable=missing-class-docstring
    def _setup_variant_attrs(self):
        """Set the GDN in_proj sizing, split tables, gate parameter dims, and kernel."""
        self.gdn_pre_gated_delta_rule_fusion = self.config.gdn_pre_gated_delta_rule_fusion
        if self.config.deterministic_mode and self.gdn_pre_gated_delta_rule_fusion:
            raise ValueError(
                "Pre-GDR fusion is non-deterministic, but deterministic_mode=True. "
                "Disable gdn_pre_gated_delta_rule_fusion or deterministic_mode."
            )

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
        self.dt_bias_dim = self.num_v_heads_local_tp
        self.a_log_dim = self.num_v_heads_local_tp

        if self.config.gdn_gdr_backend == "torch":
            self.gated_delta_rule = torch_chunk_gated_delta_rule
        elif self.config.gdn_gdr_backend == "fla":
            self.gated_delta_rule = chunk_gated_delta_rule
        elif self.config.gdn_gdr_backend == "internal":
            self.gated_delta_rule = internal_chunk_gated_delta_rule
        else:
            raise ValueError(f"Unsupported GDN GDR backend: {self.config.gdn_gdr_backend!r}.")

    def _get_feat_dim_split(self, cp_size_headwise: int) -> tuple[int, int, int, int]:
        """Return GDN1 qkv/z/beta/alpha split sizes for a runtime headwise CP size."""
        return (
            (self.qk_dim_local_tp * 2 + self.v_dim_local_tp) // cp_size_headwise,
            self.v_dim_local_tp // cp_size_headwise,
            self.num_value_heads // self.tp_size // cp_size_headwise,
            self.num_value_heads // self.tp_size // cp_size_headwise,
        )

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
        pg_collection: Optional[ProcessGroupCollection] = None,
        inference_params: Optional[BaseInferenceContext] = None,
        **kwargs,
    ):
        """
        Perform a forward pass through the GDN module.

        Return:
            (tuple[torch.Tensor, torch.Tensor]) GDN output and bias.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        active_pg_collection = pg_collection if pg_collection is not None else self.pg_collection
        base_cp_group = active_pg_collection.cp
        cp_group = resolve_cp_group(base_cp_group, packed_seq_params)
        if self.config.linear_cp_mode == "chunkwise":
            cp_group_chunkwise = cp_group
            cp_group_headwise = None
        elif self.config.linear_cp_mode == "headwise":
            cp_group_chunkwise = None
            cp_group_headwise = cp_group
        elif cp_group.size() == 1:
            cp_group_chunkwise = None
            cp_group_headwise = None
        else:
            raise ValueError(
                f"Unsupported linear_cp_mode {self.config.linear_cp_mode!r}; "
                "expected 'headwise' or 'chunkwise'."
            )
        cp_size_chunkwise = cp_group_chunkwise.size() if cp_group_chunkwise is not None else 1
        cp_size_headwise = cp_group_headwise.size() if cp_group_headwise is not None else 1
        cp_size_runtime = cp_group.size()
        back_to_input_converter = None
        if self.config.linear_cp_mode == "chunkwise":
            hidden_states, back_to_input_converter = convert_module_input_tensors_cp_partition_mode(
                hidden_states=hidden_states,
                packed_seq_params=packed_seq_params,
                cp_group=cp_group_chunkwise,
                tp_group=self.tp_group,
                tp_cp_group=getattr(active_pg_collection, "tp_cp", None),
                target_partition_mode="contiguous",
                sequence_parallel=self.config.sequence_parallel,
                config=self.config,
            )

        seq_len_local, batch, _ = hidden_states.shape
        seq_len_post_headwise = seq_len_local * self.sp_size * cp_size_headwise
        seq_len_global = seq_len_post_headwise * cp_size_chunkwise

        if inference_context is not None:
            assert (
                inference_context.is_static_batching()
            ), "GDN does not currently support dynamic inference batching."
            assert not self.config.sequence_parallel
            # TODO: support inference
            raise NotImplementedError("GDN does not support inference for now.")

        if cp_size_headwise > 1 and (
            (
                packed_seq_params is not None
                and packed_seq_params.qkv_format == "thd"
                and packed_seq_params.cp_partition_mode != "zigzag"
            )
            or (
                (packed_seq_params is None or packed_seq_params.qkv_format != "thd")
                and self.config.cp_partition_mode != "zigzag"
            )
        ):
            raise ValueError(
                "GatedDeltaNet with headwise CP requires zigzag layout. CP partition "
                "conversion must be handled before calling GatedDeltaNet."
            )

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            assert batch == 1, "Packed sequence expects batch dimension to be 1"
            assert (
                not self.config.deterministic_mode
            ), "Packed sequence does not support deterministic mode."

            # Resolve cu_seqlens with alignment padding handling.
            cu_seqlens_q = self._resolve_cu_seqlens(
                packed_seq_params.cu_seqlens_q_padded,
                packed_seq_params.cu_seqlens_q,
                seq_len_global,
                "cu_seqlens_q",
                cp_size=cp_size_runtime,
            )
            cu_seqlens_kv = self._resolve_cu_seqlens(
                packed_seq_params.cu_seqlens_kv_padded,
                packed_seq_params.cu_seqlens_kv,
                seq_len_global,
                "cu_seqlens_kv",
                cp_size=cp_size_runtime,
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

        if cp_size_chunkwise > 1:
            if cu_seqlens_q is None:
                cache_key = (seq_len_global, batch)
                cached = self._chunkwise_cp_context_cache.get(cache_key)
                if cached is None:
                    cached_cu_seqlens = (
                        torch.arange(
                            batch + 1, device=torch.cuda.current_device(), dtype=torch.long
                        )
                        * seq_len_global
                    )
                    cached_ctx = build_cp_context(
                        cu_seqlens=cached_cu_seqlens,
                        group=cp_group_chunkwise,
                        conv1d_kernel_size=self.conv_kernel_dim,
                    )
                    cached = (cached_cu_seqlens, cached_ctx)
                    self._chunkwise_cp_context_cache[cache_key] = cached
                cu_seqlens_q, chunkwise_cp_context = cached
            else:
                chunkwise_cp_context = build_cp_context(
                    cu_seqlens=cu_seqlens_q,
                    group=cp_group_chunkwise,
                    conv1d_kernel_size=self.conv_kernel_dim,
                )
        else:
            chunkwise_cp_context = None

        if self.recompute_gdn and self.training:

            def _checkpointed_compute(hidden_states):
                return self._forward_compute(
                    hidden_states,
                    batch,
                    seq_len_post_headwise,
                    cp_size_headwise,
                    cp_group_headwise,
                    cp_size_chunkwise,
                    cp_group_chunkwise,
                    cu_seqlens_q,
                    packed_seq_params,
                    chunkwise_cp_context,
                )

            out, out_bias = tensor_parallel.checkpoint(_checkpointed_compute, False, hidden_states)
        else:
            out, out_bias = self._forward_compute(
                hidden_states,
                batch,
                seq_len_post_headwise,
                cp_size_headwise,
                cp_group_headwise,
                cp_size_chunkwise,
                cp_group_chunkwise,
                cu_seqlens_q,
                packed_seq_params,
                chunkwise_cp_context,
            )

        if back_to_input_converter is not None:
            out = back_to_input_converter.convert(
                out, seq_dim=0, sequence_parallel=self.config.sequence_parallel
            )

        return out, out_bias

    def _forward_compute(
        self,
        hidden_states,
        batch,
        seq_len_post_headwise,
        cp_size_headwise,
        cp_group_headwise,
        cp_size_chunkwise,
        cp_group_chunkwise,
        cu_seqlens_q,
        packed_seq_params,
        chunkwise_cp_context,
    ):
        """Core GDN computation (in_proj -> conv1d -> gated_delta_rule -> norm -> out_proj)."""
        # Input projection
        nvtx_range_push(suffix="in_proj")
        qkvzba, _ = self.in_proj(hidden_states)
        nvtx_range_pop(suffix="in_proj")

        qkvzba, thd_cp_a2a_inv = a2a_cp_to_hp(
            qkvzba,
            self.in_proj_split_sections,
            cp_size_headwise,
            cp_group_headwise,
            cu_seqlens_q,
            seq_len_post_headwise,
            packed_seq_params,
        )

        if self.gdn_pre_gated_delta_rule_fusion:
            if cp_size_chunkwise > 1 and batch > 1:
                raise ValueError(
                    "GDN chunkwise CP with SBHD inputs currently requires micro_batch_size == 1 "
                    "when cp_context is used. Use packed THD input or micro_batch_size=1."
                )
            if cp_size_chunkwise > 1 and self.config.gdn_conv_pad_alignment is not None:
                raise ValueError(
                    "gdn_conv_pad_alignment is incompatible with GDN chunkwise CP. Padding "
                    "chunk-local causal-conv inputs can change later chunk numerics."
                )
            nvtx_range_push(suffix="fused_streamed_pre_gated_delta_rule")
            seq_idx = (
                packed_seq_params.seq_idx
                if packed_seq_params is not None
                and packed_seq_params.qkv_format == 'thd'
                and cp_size_chunkwise == 1
                else None
            )
            fused_cu_seqlens_q = (
                cu_seqlens_q
                if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
                else None
            )
            query, key, value, gate, beta, g = self._fused_streamed_pre_gated_delta_rule(
                qkvzba,
                cu_seqlens_q=fused_cu_seqlens_q,
                seq_idx=seq_idx,
                cp_group=cp_group_chunkwise if cp_size_chunkwise > 1 else None,
                cp_group_headwise=cp_group_headwise,
            )
            kernel_inputs = {"q": query, "k": key, "v": value, "g": g, "beta": beta}
            nvtx_range_pop(suffix="fused_streamed_pre_gated_delta_rule")
        else:
            nvtx_range_push(suffix="pre_gated_delta_rule")
            if cp_size_chunkwise > 1 and packed_seq_params is None and batch > 1:
                # TODO: If additional gated delta rule backends are added, handle this
                # SBHD + chunkwise CP + batch>1 case per backend instead of
                # unconditionally rejecting it.
                raise ValueError(
                    "GDN chunkwise CP with SBHD inputs currently requires micro_batch_size == 1 "
                    "when cp_context is used. Use packed THD input or micro_batch_size=1."
                )
            if cp_size_chunkwise > 1 and self.config.gdn_conv_pad_alignment is not None:
                raise ValueError(
                    "gdn_conv_pad_alignment is incompatible with GDN chunkwise CP. Padding "
                    "chunk-local causal-conv inputs can change later chunk numerics."
                )
            query, key, value, gate, beta, g = self.pre_gated_delta_rule(
                qkvzba,
                batch,
                seq_len_post_headwise,
                cp_size_headwise,
                cp_group_headwise,
                cu_seqlens_q,
                chunkwise_cp_context,
                packed_seq_params=packed_seq_params,
            )
            kernel_inputs = {"q": query, "k": key, "v": value, "g": g, "beta": beta}
            nvtx_range_pop(suffix="pre_gated_delta_rule")

        nvtx_range_push(suffix="gated_delta_rule")
        core_attn_out, _ = self.gated_delta_rule(
            **kernel_inputs,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=cu_seqlens_q,
            cp_context=chunkwise_cp_context,
        )
        nvtx_range_pop(suffix="gated_delta_rule")

        if self.recompute_norm_out and self.training:
            self.norm_out_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            norm_func = partial(
                self._gated_norm_and_layout_restore,
                thd_cp_a2a_inv=thd_cp_a2a_inv,
                batch=batch,
                seq_len=seq_len_post_headwise,
                packed_seq_params=packed_seq_params,
                cp_size_headwise=cp_size_headwise,
                cp_group_headwise=cp_group_headwise,
                cp_size_chunkwise=cp_size_chunkwise,
                cp_group_chunkwise=cp_group_chunkwise,
                cu_seqlens_q=cu_seqlens_q,
            )
            norm_out = self.norm_out_checkpoint.checkpoint(norm_func, core_attn_out, gate)
        else:
            norm_out = self._gated_norm_and_layout_restore(
                core_attn_out,
                gate,
                thd_cp_a2a_inv,
                batch,
                seq_len_post_headwise,
                packed_seq_params,
                cp_size_headwise,
                cp_group_headwise,
                cp_size_chunkwise,
                cp_group_chunkwise,
                cu_seqlens_q,
            )

        # Output projection
        nvtx_range_push(suffix="out_proj")
        out, out_bias = self.out_proj(norm_out)
        nvtx_range_pop(suffix="out_proj")

        if self.recompute_norm_out and self.training:
            self.norm_out_checkpoint.discard_output_and_register_recompute(out)

        return out, out_bias

    def _gated_norm_and_layout_restore(
        self,
        core_attn_out: torch.Tensor,
        gate: torch.Tensor,
        thd_cp_a2a_inv: torch.Tensor | None,
        batch: int,
        seq_len: int,
        packed_seq_params: PackedSeqParams | None,
        cp_size_headwise: int,
        cp_group_headwise: torch.distributed.ProcessGroup | None,
        cp_size_chunkwise: int,
        cp_group_chunkwise: torch.distributed.ProcessGroup | None,
        cu_seqlens_q: torch.Tensor | None,
    ) -> torch.Tensor:
        nvtx_range_push(suffix="gated_norm")
        norm_out = self._apply_gated_norm(core_attn_out, gate)
        nvtx_range_pop(suffix="gated_norm")

        norm_out = norm_out.reshape(batch, seq_len, -1)
        norm_out = norm_out.transpose(0, 1).contiguous()

        return a2a_hp_to_cp(
            norm_out, cp_size_headwise, cp_group_headwise, packed_seq_params, thd_cp_a2a_inv
        )

    def pre_gated_delta_rule(
        self,
        qkvzba,
        batch,
        seq_len,
        cp_size_headwise,
        cp_group_headwise,
        cu_seqlens_q=None,
        chunkwise_cp_context=None,
        packed_seq_params=None,
    ):
        """Prepare QKV, gate, beta, and decay tensors before the gated delta rule."""

        qkvzba = qkvzba.transpose(0, 1)
        qkv, gate, beta, alpha = torch.split(
            qkvzba, self._get_feat_dim_split(cp_size_headwise), dim=-1
        )
        gate = gate.reshape(batch, seq_len, -1, self.value_head_dim)

        nvtx_range_push(suffix="conv1d")
        kernel_seq_len = qkv.shape[1]
        qkv_channels_split_sections = [
            self.qk_dim_local_tp,
            self.qk_dim_local_tp,
            self.v_dim_local_tp,
        ]
        conv1d_weight = get_parameter_local_cp(
            self.conv1d.weight,
            dim=0,
            cp_group=cp_group_headwise,
            split_sections=qkv_channels_split_sections,
        )
        conv1d_bias = (
            get_parameter_local_cp(
                self.conv1d.bias,
                dim=0,
                cp_group=cp_group_headwise,
                split_sections=qkv_channels_split_sections,
            )
            if self.conv_bias
            else None
        )
        if self.config.deterministic_mode:
            qkv = qkv.transpose(1, 2).contiguous()
            conv_out = F.conv1d(
                input=qkv,
                weight=conv1d_weight,
                bias=conv1d_bias,
                stride=self.conv1d.stride,
                padding=self.conv1d.padding,
                dilation=self.conv1d.dilation,
                groups=self.conv_dim_local_tp // cp_size_headwise,
            )
            qkv = self.act_fn(conv_out[..., :kernel_seq_len])
            qkv = qkv.transpose(1, 2)
        else:
            assert self.activation in ["silu", "swish"]
            orig_seq = qkv.shape[1]
            pad_n = 0
            conv_input = qkv.contiguous()
            conv_cu_seqlens = cu_seqlens_q
            conv_cp_context = chunkwise_cp_context
            if self.config.gdn_conv_pad_alignment is not None:
                if packed_seq_params is None or cu_seqlens_q is None:
                    raise ValueError(
                        "gdn_conv_pad_alignment is only supported with packed sequence "
                        "parameters in THD format. SBHD inputs do not need causal-conv padding."
                    )
                if chunkwise_cp_context is not None:
                    raise ValueError(
                        "gdn_conv_pad_alignment is incompatible with GDN chunkwise CP. Padding "
                        "chunk-local causal-conv inputs can change later chunk numerics."
                    )
                pad_n = -orig_seq % self.config.gdn_conv_pad_alignment
            if pad_n > 0:
                conv_input = torch.nn.functional.pad(conv_input, (0, 0, 0, pad_n))
                conv_cu_seqlens = cu_seqlens_q.clone()
                conv_cu_seqlens[-1] += pad_n
            qkv, _ = causal_conv1d(
                x=conv_input,
                weight=conv1d_weight.squeeze(1),
                bias=conv1d_bias,
                activation=self.activation,
                initial_state=None,
                output_final_state=False,
                cu_seqlens=conv_cu_seqlens,
                cp_context=conv_cp_context,
            )
            if pad_n > 0:
                qkv = qkv[:, :orig_seq, :]
        nvtx_range_pop(suffix="conv1d")

        A_log_local_cp = get_parameter_local_cp(self.A_log, dim=0, cp_group=cp_group_headwise)
        dt_bias_local_cp = get_parameter_local_cp(self.dt_bias, dim=0, cp_group=cp_group_headwise)

        nvtx_range_push(suffix="prepare_input_for_gated_delta_rule")
        kernel_inputs = self._prepare_input_for_gated_delta_rule(
            qkv,
            gate,
            A_log_local_cp,
            dt_bias_local_cp,
            batch,
            kernel_seq_len,
            beta,
            alpha,
            cp_size_headwise=cp_size_headwise,
        )
        nvtx_range_pop(suffix="prepare_input_for_gated_delta_rule")

        gate = kernel_inputs.pop("gate")

        return (
            kernel_inputs["q"],
            kernel_inputs["k"],
            kernel_inputs["v"],
            gate,
            kernel_inputs["beta"],
            kernel_inputs["g"],
        )

    def _fused_streamed_pre_gated_delta_rule(
        self, qkvzba, cu_seqlens_q=None, seq_idx=None, cp_group=None, cp_group_headwise=None
    ):
        """Call the streamed fused pre-GDR wrapper."""

        try:
            from megatron.core.fusions.fused_pre_gated_delta_rule import (
                fused_streamed_pre_gated_delta_rule,
            )
        except ImportError as exc:
            raise ImportError(
                "gdn_pre_gated_delta_rule_fusion requires the streamed pre-GDR fusion "
                "dependencies, including causal-conv1d."
            ) from exc

        qkv_channels_split_sections = [
            self.qk_dim_local_tp,
            self.qk_dim_local_tp,
            self.v_dim_local_tp,
        ]
        conv1d_weight = get_parameter_local_cp(
            self.conv1d.weight,
            dim=0,
            cp_group=cp_group_headwise,
            split_sections=qkv_channels_split_sections,
        )
        conv1d_bias = (
            get_parameter_local_cp(
                self.conv1d.bias,
                dim=0,
                cp_group=cp_group_headwise,
                split_sections=qkv_channels_split_sections,
            )
            if self.conv_bias
            else None
        )
        A_log = get_parameter_local_cp(self.A_log, dim=0, cp_group=cp_group_headwise)
        dt_bias = get_parameter_local_cp(self.dt_bias, dim=0, cp_group=cp_group_headwise)
        num_value_heads = A_log.numel()
        num_key_heads = (conv1d_weight.shape[0] - num_value_heads * self.value_head_dim) // (
            2 * self.key_head_dim
        )

        return fused_streamed_pre_gated_delta_rule(
            qkvzba,
            conv1d_weight,
            conv1d_bias,
            A_log,
            dt_bias,
            num_key_heads=num_key_heads,
            num_value_heads=num_value_heads,
            key_head_dim=self.key_head_dim,
            value_head_dim=self.value_head_dim,
            use_qk_l2norm=self.use_qk_l2norm,
            cu_seqlens=cu_seqlens_q,
            seq_idx=seq_idx,
            cp_group=cp_group,
        )


####################
# Torch native gated delta rule
####################
def torch_chunk_gated_delta_rule(
    q,
    k,
    v,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
    cp_context=None,
    scale=None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # pylint: disable=line-too-long
    '''
    Torch-native implementation of chunked gated delta rule for deterministic mode.
    Need this because FLA is not deterministic.

    ``scale`` defaults to ``1 / sqrt(K)``, matching the FLA kernel.

    Reference: https://github.com/huggingface/transformers/blob/144c8ce2809a2e21914017652700e1ecb450501e/src/transformers/models/qwen3_next/modeling_qwen3_next.py#L470-L547
    '''

    assert (
        cu_seqlens is None
    ), "cu_seqlens is not supported for torch_chunk_gated_delta_rule for now."
    assert (
        cp_context is None
    ), "cp_context is not supported for torch_chunk_gated_delta_rule for now."

    initial_dtype = q.dtype
    if use_qk_l2norm_in_kernel:
        q = l2norm(q, dim=-1, eps=1e-6)
        k = l2norm(k, dim=-1, eps=1e-6)
    q, k, v, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (q, k, v, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = k.shape
    v_head_dim = v.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    q = F.pad(q, (0, 0, 0, pad_size))
    k = F.pad(k, (0, 0, 0, pad_size))
    v = F.pad(v, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    q = q * scale

    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)
    # reshape to chunks
    q, k, v, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (q, k, v, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0
    )

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ k.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    v = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(v)
        if initial_state is None
        else initial_state.to(v)
    )
    core_attn_out = torch.zeros_like(v)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1
    )

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state
