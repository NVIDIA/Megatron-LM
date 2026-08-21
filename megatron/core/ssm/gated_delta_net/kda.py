# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Songlin Yang, Jan Kautz, Ali Hatamizadeh.

"""Kimi Delta Attention, a channel-wise Gated DeltaNet variant."""

import math
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F

from megatron.core import tensor_parallel
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.jit import jit_fuser
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net.common import (
    HAVE_FLA,
    GatedDeltaNetSubmodules,
    _GDNBase,
    a2a_cp_to_hp,
    a2a_hp_to_cp,
    causal_conv1d,
    get_parameter_local_cp,
    l2norm,
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
    ) -> None:
        self._validate_config(config)
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
        )

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
        self.dt_bias._no_reinit = True
        self.dt_bias._no_weight_decay = True
        self.A_log._no_weight_decay = True

        # Pass raw g to FLA and let the KDA kernel apply A_log and dt_bias.
        self.use_gate_in_kernel = True

    @staticmethod
    def _validate_config(config: TransformerConfig) -> None:
        """Validate the direct-projection KDA layout supported by this implementation."""

        if config.kda_safe_gate:
            if config.kda_lower_bound is None:
                raise ValueError(
                    "KimiDeltaAttention requires kda_lower_bound when kda_safe_gate=True."
                )
            if not (-5.0 <= config.kda_lower_bound < 0.0):
                raise ValueError(
                    "KimiDeltaAttention requires kda_lower_bound to be in [-5, 0) "
                    "when kda_safe_gate=True."
                )

        if config.linear_num_key_heads != config.linear_num_value_heads:
            raise ValueError(
                "KimiDeltaAttention currently requires equal key and value head counts."
            )
        if config.linear_key_head_dim != config.linear_value_head_dim:
            raise ValueError(
                "KimiDeltaAttention currently requires equal key and value head dimensions."
            )
        tp_cp_size = config.tensor_model_parallel_size * config.context_parallel_size
        if config.linear_num_key_heads % tp_cp_size != 0:
            raise ValueError(
                "KimiDeltaAttention key heads must be divisible by tensor parallel size times "
                "context parallel size."
            )

    def _setup_variant_attrs(self) -> None:
        """Set KDA dimensions and projection checkpoint metadata."""

        self.in_proj_extra_dim = self.qk_dim
        self.in_proj_split_names = ["query", "key", "value", "g", "gate"]
        self.in_proj_split_sections = (
            self.qk_dim_local_tp,
            self.qk_dim_local_tp,
            self.v_dim_local_tp,
            self.qk_dim_local_tp,
            self.v_dim_local_tp,
        )
        self.feat_dim_split = (
            (2 * self.qk_dim_local_tp + self.v_dim_local_tp) // self.cp_size,
            self.qk_dim_local_tp // self.cp_size,
            self.v_dim_local_tp // self.cp_size,
        )
        self.dt_bias_dim = self.qk_dim_local_tp
        self.a_log_dim = self.num_k_heads_local_tp
        self.gate_params_dtype = torch.float32
        self.gated_delta_rule = chunk_kda

    def reset_parameters(self) -> None:
        """Initialize convolution and KDA gate parameters."""

        if not self.config.perform_initialization:
            return

        with tensor_parallel.get_cuda_rng_tracker().fork():
            if self.conv_init is not None:
                torch.nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

            dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
            dt = torch.exp(
                torch.rand(self.dt_bias_dim, device=self.dt_bias.device, dtype=torch.float32)
                * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            self.dt_bias.data.copy_(dt + torch.log(-torch.expm1(-dt)))

            A = torch.empty(self.a_log_dim, device=self.A_log.device, dtype=torch.float32).uniform_(
                *self.A_init_range
            )
            self.A_log.data.copy_(torch.log(A))

    @jit_fuser
    def _apply_gated_norm(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Apply per-head RMSNorm followed by KDA's sigmoid output gate."""

        x_dtype = x.dtype
        x = x.reshape(-1, self.value_head_dim)
        x = self.out_norm(x)
        gate = gate.reshape(-1, self.value_head_dim)
        return (x * torch.sigmoid(gate.float())).to(x_dtype)

    def _prepare_kda_inputs(
        self,
        qkv: torch.Tensor,
        raw_g: torch.Tensor,
        beta: torch.Tensor,
        output_gate: torch.Tensor,
        batch: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, ...]:
        """Split, normalize, and reshape tensors for the FLA KDA kernel."""

        query_key, value = torch.split(
            qkv,
            [2 * self.qk_dim_local_tp // self.cp_size, self.v_dim_local_tp // self.cp_size],
            dim=-1,
        )
        query_key = query_key.reshape(batch, seq_len, -1, self.key_head_dim)
        value = value.reshape(batch, seq_len, -1, self.value_head_dim)
        if self.use_qk_l2norm:
            query_key = l2norm(query_key.contiguous())

        num_key_heads = self.num_k_heads_local_tp // self.cp_size
        query, key = torch.split(query_key, [num_key_heads, num_key_heads], dim=2)
        raw_g = raw_g.reshape(batch, seq_len, num_key_heads, self.key_head_dim)
        beta = beta.reshape(batch, seq_len, num_key_heads).float().sigmoid()
        output_gate = output_gate.reshape(
            batch, seq_len, self.num_v_heads_local_tp // self.cp_size, self.value_head_dim
        )
        return (
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            raw_g.contiguous(),
            beta.contiguous(),
            output_gate.contiguous(),
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

    def _gated_norm_and_a2a(
        self,
        core_attn_out: torch.Tensor,
        gate: torch.Tensor,
        thd_cp_a2a_inv: torch.Tensor | None,
        batch: int,
        seq_len: int,
        packed_seq_params: PackedSeqParams | None = None,
    ) -> torch.Tensor:
        """Apply KDA's gated output norm and restore its context-parallel layout."""

        nvtx_range_push(suffix="gated_norm")
        norm_out_hp = self._apply_gated_norm(core_attn_out, gate)
        nvtx_range_pop(suffix="gated_norm")

        norm_out_hp = norm_out_hp.reshape(batch, seq_len, -1)
        norm_out_hp = norm_out_hp.transpose(0, 1).contiguous()

        return a2a_hp_to_cp(
            norm_out_hp, self.cp_size, self.pg_collection.cp, packed_seq_params, thd_cp_a2a_inv
        )

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
        """Run the direct-projection KDA training path."""

        del attention_mask, sequence_len_offset, kwargs
        inference_context = deprecate_inference_params(inference_context, inference_params)
        if inference_context is not None:  # pragma: no cover
            raise NotImplementedError("KDA recurrent inference is not implemented.")

        seq_len, batch, _ = hidden_states.shape
        total_seq_len = seq_len * self.sp_size * self.cp_size

        cu_seqlens_q = None
        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            if batch != 1:
                raise ValueError("Packed KDA expects batch dimension to be 1.")
            cu_seqlens_q = self._resolve_cu_seqlens(
                packed_seq_params.cu_seqlens_q_padded,
                packed_seq_params.cu_seqlens_q,
                total_seq_len,
                "cu_seqlens_q",
                cp_size=self.cp_size,
            )
            cu_seqlens_kv = self._resolve_cu_seqlens(
                packed_seq_params.cu_seqlens_kv_padded,
                packed_seq_params.cu_seqlens_kv,
                total_seq_len,
                "cu_seqlens_kv",
                cp_size=self.cp_size,
            )
            self._validate_packed_cu_seqlens(cu_seqlens_q, cu_seqlens_kv)

        nvtx_range_push(suffix="kda_in_proj")
        qkvfg, _ = self.in_proj(hidden_states)
        beta, _ = self.beta_proj(hidden_states)
        nvtx_range_pop(suffix="kda_in_proj")

        qkvfg, thd_cp_a2a_inv = a2a_cp_to_hp(
            qkvfg,
            self.in_proj_split_sections,
            self.cp_size,
            self.pg_collection.cp,
            cu_seqlens_q,
            total_seq_len,
            packed_seq_params,
        )
        beta, _ = a2a_cp_to_hp(
            beta,
            (self.num_k_heads_local_tp,),
            self.cp_size,
            self.pg_collection.cp,
            cu_seqlens_q,
            total_seq_len,
            packed_seq_params,
        )

        qkvfg = qkvfg.transpose(0, 1)
        beta = beta.transpose(0, 1)
        seq_len = qkvfg.shape[1]
        qkv, raw_g, output_gate = torch.split(qkvfg, self.feat_dim_split, dim=-1)

        nvtx_range_push(suffix="kda_conv1d")
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
        if self.config.deterministic_mode or causal_conv1d is None:
            qkv = qkv.transpose(1, 2).contiguous()
            conv_out = F.conv1d(
                qkv,
                conv1d_weight,
                bias=conv1d_bias,
                stride=self.conv1d.stride,
                padding=self.conv1d.padding,
                dilation=self.conv1d.dilation,
                groups=self.conv_dim_local_tp // self.cp_size,
            )
            qkv = self.act_fn(conv_out[..., :seq_len]).transpose(1, 2).contiguous()
        else:
            if self.activation not in ("silu", "swish"):
                raise ValueError(f"FLA causal convolution requires SiLU, got {self.activation}.")
            qkv, _ = causal_conv1d(
                x=qkv,
                weight=conv1d_weight.squeeze(1),
                bias=conv1d_bias,
                activation=self.activation,
                initial_state=None,
                output_final_state=False,
                cu_seqlens=cu_seqlens_q,
            )
        nvtx_range_pop(suffix="kda_conv1d")

        query, key, value, raw_g, beta, output_gate = self._prepare_kda_inputs(
            qkv, raw_g, beta, output_gate, batch, seq_len
        )
        A_log_local_cp = get_parameter_local_cp(self.A_log, dim=0, cp_group=self.pg_collection.cp)
        dt_bias_local_cp = get_parameter_local_cp(
            self.dt_bias, dim=0, cp_group=self.pg_collection.cp
        )

        nvtx_range_push(suffix="kda_chunk")
        core_attn_out, _ = self.gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=raw_g,
            beta=beta,
            A_log=A_log_local_cp,
            dt_bias=dt_bias_local_cp,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
            use_gate_in_kernel=self.use_gate_in_kernel,
            safe_gate=self.config.kda_safe_gate,
            lower_bound=self.config.kda_lower_bound,
            cu_seqlens=cu_seqlens_q,
        )
        nvtx_range_pop(suffix="kda_chunk")

        if self.recompute_norm_out:
            self.norm_out_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            norm_func = partial(
                self._gated_norm_and_a2a,
                thd_cp_a2a_inv=thd_cp_a2a_inv,
                batch=batch,
                seq_len=seq_len,
                packed_seq_params=packed_seq_params,
            )
            norm_out = self.norm_out_checkpoint.checkpoint(norm_func, core_attn_out, output_gate)
        else:
            norm_out = self._gated_norm_and_a2a(
                core_attn_out, output_gate, thd_cp_a2a_inv, batch, seq_len, packed_seq_params
            )

        nvtx_range_push(suffix="kda_out_proj")
        out, out_bias = self.out_proj(norm_out)
        nvtx_range_pop(suffix="kda_out_proj")

        if self.recompute_norm_out:
            self.norm_out_checkpoint.discard_output_and_register_recompute(out)

        return out, out_bias

    def backward_dw(self) -> None:
        """Execute weight-gradient computation for all KDA projections."""

        super().backward_dw()
        self.beta_proj.backward_dw()
