# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Kimi Delta Attention, a channel-wise Gated DeltaNet variant."""

import math
from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F

from megatron.core import tensor_parallel
from megatron.core.context_parallel import contiguous_to_zigzag, zigzag_to_contiguous
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.jit import jit_fuser
from megatron.core.packed_seq_params import PackedSeqParams, resolve_cp_group
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net.common import (
    HAVE_FLA,
    GatedDeltaNetSubmodules,
    _GDNBase,
    a2a_cp_to_hp,
    a2a_hp_to_cp,
    build_cp_context,
    causal_conv1d,
    get_parameter_local_cp,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import mark_keep_in_fp32
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import deprecate_inference_params, nvtx_range_pop, nvtx_range_push

try:
    from fla.ops.kda import chunk_kda

    # KDA also relies on the shared FLA convolution, normalization, and CP helpers.
    HAVE_FLA_KDA = HAVE_FLA
except ImportError:  # pragma: no cover
    chunk_kda = None
    HAVE_FLA_KDA = False


@dataclass
class KimiDeltaAttentionSubmodules(GatedDeltaNetSubmodules):
    """Submodules used by Kimi Delta Attention."""

    beta_proj: ModuleSpec | type = IdentityOp


class KimiDeltaAttention(_GDNBase):
    """Channel-wise Gated DeltaNet variant with direct Q/K/V/F/G projections.

    The initial implementation supports training with equal query/key and value
    head layouts. Optional low-rank F/B/G projections and recurrent inference
    are intentionally outside this implementation.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: KimiDeltaAttentionSubmodules,
        layer_number: int | None = None,
        bias: bool = False,
        conv_bias: bool = False,
        conv_init: float | None = None,
        use_qk_l2norm: bool = True,
        A_init_range: tuple[float, float] = (1, 16),
        pg_collection: ProcessGroupCollection | None = None,
        *,
        name: str | None = None,
        cp_comm_type: str | None = None,
        pp_layer_offset: Optional[int] = None,
        is_mtp_layer: bool = False,
    ) -> None:
        if not HAVE_FLA or not HAVE_FLA_KDA:  # pragma: no cover
            raise ImportError(
                "FLA KDA is not installed. Install flash-linear-attention with KDA support."
            )

        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            bias=bias,
            conv_bias=conv_bias,
            conv_init=conv_init,
            use_qk_l2norm=use_qk_l2norm,
            A_init_range=A_init_range,
            pg_collection=pg_collection,
            name=name,
            cp_comm_type=cp_comm_type,
            pp_layer_offset=pp_layer_offset,
            is_mtp_layer=is_mtp_layer,
        )

        # KDA keeps beta in a separate projection so its checkpoint layout remains
        # independent from the direct Q/K/V/F/G projection.
        self.beta_proj = build_module(
            submodules.beta_proj,
            self.hidden_size,
            self.num_key_heads,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="beta_proj",
            tp_group=self.pg_collection.tp,
            name=(name + ".beta_proj") if name is not None else None,
        )

        # These kernel parameters participate in FP32 gate math and must remain
        # FP32 through model casting, optimizer construction, and checkpointing.
        mark_keep_in_fp32(self.dt_bias)
        mark_keep_in_fp32(self.A_log)

        # Pass raw g to FLA and let the KDA kernel apply A_log and dt_bias.
        self.use_gate_in_kernel = True

    def _setup_variant_attrs(self) -> None:
        """Set KDA dimensions, projection checkpoint metadata, and kernel callable."""

        self.gdn_pre_gated_delta_rule_fusion = self.config.gdn_pre_gated_delta_rule_fusion

        # Channel-wise raw memory-decay gate g.
        self.in_proj_extra_dim = self.qk_dim

        # Per-section sizes (and names) of the in_proj output, local to this TP rank.
        # Used for the CP head permutation, post-a2a split, and sharded checkpoint split.
        self.in_proj_split_names = ["query", "key", "value", "g", "gate"]
        self.in_proj_split_sections = (
            self.qk_dim_local_tp,
            self.qk_dim_local_tp,
            self.v_dim_local_tp,
            self.qk_dim_local_tp,
            self.v_dim_local_tp,
        )
        self.dt_bias_dim = self.qk_dim_local_tp
        self.a_log_dim = self.num_k_heads_local_tp
        self.gate_params_dtype = torch.float32
        self.gated_delta_rule = chunk_kda

    def _get_feat_dim_split(self, cp_size_headwise: int) -> tuple[int, int, int]:
        """Return KDA qkv/raw-g/output-gate split sizes for runtime headwise CP."""

        return (
            (self.qk_dim_local_tp * 2 + self.v_dim_local_tp) // cp_size_headwise,
            self.qk_dim_local_tp // cp_size_headwise,
            self.v_dim_local_tp // cp_size_headwise,
        )

    def _reset_dt_bias(self) -> None:
        """Initialize the KDA channel-wise step-size bias in inverse-softplus space."""

        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(
            torch.rand(self.dt_bias_dim, device=self.dt_bias.device, dtype=self.dt_bias.dtype)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        self.dt_bias.data.copy_(dt + torch.log(-torch.expm1(-dt)))

    @jit_fuser
    def _compute_gates(
        self,
        A_log_local_cp: torch.Tensor,
        dt_bias_local_cp: torch.Tensor,
        batch: int,
        seq_len: int,
        *gate_feats: tuple[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Shape the raw channel-wise decay gate and activate the write strength."""

        # ``gate_feats`` follows the KDA pre-GDR order: raw g, then separate beta.
        raw_g, beta = gate_feats
        num_key_heads = A_log_local_cp.numel()
        raw_g = raw_g.reshape(batch, seq_len, num_key_heads, self.key_head_dim)
        beta = beta.reshape(batch, seq_len, num_key_heads).float().sigmoid()
        return raw_g, {"beta": beta.contiguous()}

    @jit_fuser
    def _apply_gated_norm(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Apply per-head RMSNorm followed by KDA's sigmoid output gate."""

        x_dtype = x.dtype
        x = x.reshape(-1, self.value_head_dim)
        x = self.out_norm(x)
        gate = gate.reshape(-1, self.value_head_dim)
        return (x * torch.sigmoid(gate.float())).to(x_dtype)

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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the direct-projection KDA training path."""

        del attention_mask, sequence_len_offset, kwargs
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
        convert_from_zigzag = (
            self.config.linear_cp_mode == "chunkwise"
            and cp_size_chunkwise > 1
            and self.config.cp_partition_mode == "zigzag"
        )
        if convert_from_zigzag:
            hidden_states = zigzag_to_contiguous(
                hidden_states,
                cp_group_chunkwise,
                self.config.sequence_parallel,
                self.tp_group,
                getattr(active_pg_collection, "tp_cp", None),
            )

        seq_len_local, batch, _ = hidden_states.shape
        seq_len_post_headwise = seq_len_local * self.sp_size * cp_size_headwise
        seq_len_global = seq_len_post_headwise * cp_size_chunkwise

        if inference_context is not None:
            assert (
                inference_context.is_static_batching()
            ), "KimiDeltaAttention does not currently support dynamic inference batching."
            assert not self.config.sequence_parallel
            raise NotImplementedError("KimiDeltaAttention does not support inference for now.")

        if cp_size_headwise > 1 and (
            (
                packed_seq_params is not None
                and packed_seq_params.qkv_format == "thd"
                and getattr(packed_seq_params, "cp_partition_mode", self.config.cp_partition_mode)
                != "zigzag"
            )
            or (
                (packed_seq_params is None or packed_seq_params.qkv_format != "thd")
                and self.config.cp_partition_mode != "zigzag"
            )
        ):
            raise ValueError(
                "KimiDeltaAttention with headwise CP requires zigzag layout. CP partition "
                "conversion must be handled before calling KimiDeltaAttention."
            )

        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            if batch != 1:
                raise ValueError("Packed KDA expects batch dimension to be 1.")
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
            self._validate_packed_cu_seqlens(cu_seqlens_q, cu_seqlens_kv)
        else:
            cu_seqlens_q = None

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

        if convert_from_zigzag:
            out = contiguous_to_zigzag(
                out,
                cp_group_chunkwise,
                self.config.sequence_parallel,
                self.tp_group,
                getattr(active_pg_collection, "tp_cp", None),
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
        """Core KDA computation (in_proj -> conv1d -> gated_delta_rule -> norm -> out_proj)."""

        # Input projections. Beta intentionally remains a separate matrix.
        nvtx_range_push(suffix="in_proj")
        qkvfg, _ = self.in_proj(hidden_states)
        beta, _ = self.beta_proj(hidden_states)
        nvtx_range_pop(suffix="in_proj")

        qkvfg, thd_cp_a2a_inv = a2a_cp_to_hp(
            qkvfg,
            self.in_proj_split_sections,
            cp_size_headwise,
            cp_group_headwise,
            cu_seqlens_q,
            seq_len_post_headwise,
            packed_seq_params,
        )
        beta, _ = a2a_cp_to_hp(
            beta,
            (self.num_k_heads_local_tp,),
            cp_size_headwise,
            cp_group_headwise,
            cu_seqlens_q,
            seq_len_post_headwise,
            packed_seq_params,
        )

        if self.gdn_pre_gated_delta_rule_fusion:
            raise NotImplementedError(
                "gdn_pre_gated_delta_rule_fusion is not implemented for KDA yet."
            )

        if cp_size_chunkwise > 1 and packed_seq_params is None and batch > 1:
            raise ValueError(
                "KDA chunkwise CP with SBHD inputs currently requires micro_batch_size == 1 "
                "when cp_context is used. Use packed THD input or micro_batch_size=1."
            )
        if cp_size_chunkwise > 1 and self.config.gdn_conv_pad_alignment is not None:
            raise ValueError(
                "gdn_conv_pad_alignment is incompatible with KDA chunkwise CP. Padding "
                "chunk-local causal-conv inputs can change later chunk numerics."
            )

        nvtx_range_push(suffix="pre_gated_delta_rule")
        query, key, value, gate, beta, raw_g, A_log, dt_bias = self.pre_gated_delta_rule(
            qkvfg,
            beta,
            batch,
            seq_len_post_headwise,
            cp_size_headwise,
            cp_group_headwise,
            cu_seqlens_q,
            chunkwise_cp_context,
            packed_seq_params=packed_seq_params,
        )
        kernel_inputs = {
            "q": query,
            "k": key,
            "v": value,
            "g": raw_g,
            "beta": beta,
            "A_log": A_log,
            "dt_bias": dt_bias,
        }
        nvtx_range_pop(suffix="pre_gated_delta_rule")

        nvtx_range_push(suffix="gated_delta_rule")
        core_attn_out, _ = self.gated_delta_rule(
            **kernel_inputs,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
            use_gate_in_kernel=self.use_gate_in_kernel,
            safe_gate=self.config.kda_safe_gate,
            lower_bound=self.config.kda_lower_bound,
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

        # Output projection.
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
        """Apply KDA's gated output norm and restore its context-parallel layout."""

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
        qkvfg,
        beta,
        batch,
        seq_len,
        cp_size_headwise,
        cp_group_headwise,
        cu_seqlens_q=None,
        chunkwise_cp_context=None,
        packed_seq_params=None,
    ):
        """Prepare QKV, output gate, beta, and raw decay tensors before KDA."""

        qkvfg = qkvfg.transpose(0, 1)
        beta = beta.transpose(0, 1)
        qkv, raw_g, gate = torch.split(qkvfg, self._get_feat_dim_split(cp_size_headwise), dim=-1)
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
        if self.config.deterministic_mode or causal_conv1d is None:
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
            qkv = qkv.transpose(1, 2).contiguous()
        else:
            if self.activation not in ("silu", "swish"):
                raise ValueError(f"FLA causal convolution requires SiLU, got {self.activation}.")
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
                        "gdn_conv_pad_alignment is incompatible with KDA chunkwise CP. Padding "
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
            raw_g,
            beta,
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
            A_log_local_cp,
            dt_bias_local_cp,
        )

    @staticmethod
    def _validate_packed_cu_seqlens(
        cu_seqlens_q: torch.Tensor, cu_seqlens_kv: torch.Tensor
    ) -> None:
        """Validate the self-attention boundary contract for packed KDA."""

        if cu_seqlens_q.numel() < 2 or cu_seqlens_kv.numel() < 2:
            raise ValueError(
                "Packed KDA requires at least one sequence in both Q and KV boundaries."
            )
        if cu_seqlens_q.shape != cu_seqlens_kv.shape or not torch.equal(
            cu_seqlens_q, cu_seqlens_kv
        ):
            raise ValueError(
                "Packed KDA requires cu_seqlens_q to equal cu_seqlens_kv, "
                f"but got shapes {tuple(cu_seqlens_q.shape)} and "
                f"{tuple(cu_seqlens_kv.shape)}."
            )

    def backward_dw(self) -> None:
        """Execute weight-gradient computation for all KDA projections."""

        super().backward_dw()
        self.beta_proj.backward_dw()
