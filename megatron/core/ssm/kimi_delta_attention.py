# Copyright (c) 2026, ETH Zurich / Swiss AI Initiative.
#
# Kimi Delta Attention (KDA) layer for Megatron-LM.
#
# Why we don't drop in MoonshotAI's Kimi-Linear release directly
# --------------------------------------------------------------
# The official Kimi-Linear repo (github.com/MoonshotAI/Kimi-Linear) ships a
# clean reference implementation, but it targets the HuggingFace Transformers
# stack: layers extend nn.Module / PreTrainedModel, projections are plain
# nn.Linear (no Megatron ColumnParallel/RowParallel), tensors flow in
# [b, s, h] format, there is no context-parallel all-to-all, and there is no
# sharded-state-dict logic for distributed checkpointing. Megatron requires
# all of those: pg_collection, TP-aware projections, CP all-to-all, sbhd
# layout, and ShardedTensor-based checkpoints. Plumbing the upstream module
# into Megatron would mean rewriting it anyway, so we wrap the FLA `chunk_kda`
# kernel directly here, mirroring the structure of `gated_delta_net.py`.
#
# This keeps the kernel (and its FlashKDA/CUTLASS path inside FLA 0.5.0)
# exactly as released by Moonshot, while letting the rest of the layer reuse
# Megatron's distributed conventions.

import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from megatron.core.jit import jit_fuser
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.ssm.gated_delta_net import (
    GatedDeltaNet,
    GatedDeltaNetSubmodules,
)

try:
    from fla.ops.kda import chunk_kda

    HAVE_KDA = True
except ImportError:
    chunk_kda = None
    HAVE_KDA = False

logger = logging.getLogger(__name__)


# Reuse GatedDeltaNetSubmodules for structural parity. KDA has the same
# in/out projections; the differences live inside the layer.
KimiDeltaAttentionSubmodules = GatedDeltaNetSubmodules


class KimiDeltaAttention(GatedDeltaNet):
    """Kimi Delta Attention (KDA) — channel-wise diagonal decay variant of GDN.

    Diff vs GatedDeltaNet:
      - decay g is a vector in R^{key_head_dim} per head (vs scalar per head),
        so the alpha slot in the input projection is num_value_heads*key_head_dim
        instead of num_value_heads.
      - the chunkwise op is `chunk_kda` (FLA 0.5.0+).
      - gates: KDA keeps GDN's output gate; the in-kernel `use_gate_in_kernel`
        path is left off here so the gate stays in the explicit
        `_apply_gated_norm` branch (matches the rest of the codebase).

    Reference: Kimi Team, "Kimi Linear: An Expressive, Efficient Attention
    Architecture", arXiv:2510.26692; FLA op `fla.ops.kda.chunk_kda`.

    The class subclasses `GatedDeltaNet` and replaces three things:
      1. `__init__` resizes `self.in_proj`, `self.A_log`, `self.dt_bias` to the
         vector-alpha layout, then rebuilds the fla op binding.
      2. `_compute_g_and_beta` returns a vector g of shape [b, s, h, d_k].
      3. `forward` overrides only the splitting + reshape of `qkvzba` (alpha
         is wider) and the FLA-op call (chunk_kda instead of
         chunk_gated_delta_rule). All other plumbing (conv1d, CP/TP all-to-all,
         output projection, gated norm) is inherited unchanged.

    Currently TP=1 / CP=1 only — sharded_state_dict from the parent class
    will report wrong shapes for the alpha slot under TP>1. Distributed
    checkpointing for KDA is a TODO; for the 1-node ablation runs in this
    repo it is unused.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: KimiDeltaAttentionSubmodules,
        layer_number: int = None,
        bias: bool = False,
        conv_bias: bool = False,
        conv_init: Optional[float] = None,
        use_qk_l2norm: bool = True,
        A_init_range: Tuple[float, float] = (1, 16),
        pg_collection: ProcessGroupCollection = None,
    ):
        if not HAVE_KDA:
            raise ImportError(
                "FLA's chunk_kda op is required for Kimi Delta Attention. "
                "Install flash-linear-attention >= 0.5.0."
            )

        # Initialize via the GatedDeltaNet path. This builds in_proj/A_log/
        # dt_bias with the GDN scalar-alpha layout. We resize them below.
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
        )

        # KDA in_proj layout: qkv (qk*2 + v), gate (v), beta (num_v_heads),
        # alpha (num_v_heads * key_head_dim). Differs from GDN's alpha slot
        # of num_value_heads.
        self.alpha_dim = self.num_value_heads * self.key_head_dim
        self.in_proj_dim = (
            self.qk_dim * 2 + self.v_dim * 2 + self.num_value_heads + self.alpha_dim
        )

        # Rebuild in_proj with the new output dim. Uses the same submodule spec
        # as GDN; the parent's instance is replaced.
        self.in_proj = build_module(
            submodules.in_proj,
            self.hidden_size,
            self.in_proj_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="fc1",
            tp_group=self.pg_collection.tp,
        )

        # A_log + dt_bias are now per-channel: shape [num_v_heads, key_head_dim].
        self.num_v_heads_local_tp = self.num_value_heads // self.tp_size
        self.dt_bias = nn.Parameter(
            torch.empty(
                self.num_v_heads_local_tp,
                self.key_head_dim,
                dtype=config.params_dtype,
                device=torch.cuda.current_device(),
            )
        )
        setattr(self.dt_bias, "tensor_model_parallel", True)
        setattr(self.dt_bias, "partition_dim", 0)
        self.A_log = nn.Parameter(
            torch.empty(
                self.num_v_heads_local_tp,
                self.key_head_dim,
                dtype=config.params_dtype,
                device=torch.cuda.current_device(),
            )
        )
        setattr(self.A_log, "tensor_model_parallel", True)
        setattr(self.A_log, "partition_dim", 0)

        # Bind the FLA op.
        self.gated_delta_rule = chunk_kda

        # Re-init A_log + dt_bias for the new shape.
        self._reset_kda_decay_params(A_init_range)

    def _reset_kda_decay_params(self, A_init_range: Tuple[float, float]) -> None:
        if not self.config.perform_initialization:
            return
        with get_cuda_rng_tracker().fork():
            torch.ones(
                self.num_v_heads_local_tp,
                self.key_head_dim,
                out=self.dt_bias.data,
                dtype=self.config.params_dtype,
                device=torch.cuda.current_device(),
            )
            A = torch.empty(
                self.num_v_heads_local_tp,
                self.key_head_dim,
                dtype=self.config.params_dtype,
                device=torch.cuda.current_device(),
            ).uniform_(*A_init_range)
            self.A_log.data.copy_(torch.log(A))

    @jit_fuser
    def _compute_g_and_beta(self, A_log_local_cp, dt_bias_local_cp, alpha, beta):
        """Vector decay variant: g has shape [b, s, h, d_k] instead of [b, s, h].

        alpha here is the post-projection tensor reshaped to [b, s, h, d_k].
        A_log/dt_bias have shape [h, d_k] (broadcast over batch and seq).
        """
        # alpha: fp32 to match the precision used by FLA decay path.
        g = -A_log_local_cp.exp() * F.softplus(alpha.float() + dt_bias_local_cp)
        beta = beta.sigmoid()
        if self.config.linear_attention_allow_neg_eigval:
            beta = beta * 2.0
        return g, beta

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        inference_context=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
        **kwargs,
    ):
        """KDA forward. Mirrors GDN.forward, but the alpha slot is wider
        (num_v_heads * key_head_dim vs num_v_heads). KDA has no DeltaProduct
        variant, so n_hh is forced to 1 and erase logic is skipped.
        """
        import os
        from megatron.core.ssm.gated_delta_net import (
            tensor_a2a_cp2hp, tensor_a2a_hp2cp, get_parameter_local_cp,
            nvtx_range_push, nvtx_range_pop, causal_conv1d,
        )
        from megatron.core.utils import deprecate_inference_params
        from fla.modules.l2norm import l2norm

        inference_context = deprecate_inference_params(inference_context, inference_params)

        seq_len, batch, _ = hidden_states.shape
        seq_len = seq_len * self.sp_size * self.cp_size

        if inference_context is not None:
            raise NotImplementedError("KDA does not support inference for now.")
        if packed_seq_params is not None:
            raise NotImplementedError("KDA does not support packed sequence for now.")
        assert self.n_hh == 1, "KDA does not have a DeltaProduct (n_householder>1) variant."

        # Input projection
        nvtx_range_push(suffix="in_proj")
        qkvzba, _ = self.in_proj(hidden_states)
        nvtx_range_pop(suffix="in_proj")

        num_v_heads_tp = self.num_value_heads // self.tp_size

        # CP All to All: CP to HP. Alpha slot is wider than GDN's:
        # alpha_dim = num_value_heads * key_head_dim (vs num_value_heads for GDN).
        qkvzba = tensor_a2a_cp2hp(
            qkvzba,
            seq_dim=0,
            head_dim=-1,
            cp_group=self.pg_collection.cp,
            split_sections=[
                self.qk_dim_local_tp,           # Q
                self.qk_dim_local_tp,           # K
                self.v_dim_local_tp,            # V
                self.v_dim_local_tp,            # gate
                num_v_heads_tp,                 # beta (per-head scalar, sigmoid'd)
                self.alpha_dim // self.tp_size, # alpha (per-channel: num_v_heads*key_head_dim)
            ],
        )

        # Transpose: s b x --> b s x
        qkvzba = qkvzba.transpose(0, 1)

        # Split into qkv (Q+K+V), gate, beta, alpha. The alpha slot is wider here.
        alpha_local = self.alpha_dim // self.tp_size // self.cp_size
        qkv, gate, beta, alpha = torch.split(
            qkvzba,
            [
                (2 * self.qk_dim_local_tp + self.v_dim_local_tp) // self.cp_size,
                self.v_dim_local_tp // self.cp_size,
                num_v_heads_tp // self.cp_size,
                alpha_local,
            ],
            dim=-1,
        )
        gate = gate.reshape(batch, seq_len, -1, self.value_head_dim)
        beta = beta.reshape(batch, seq_len, -1)

        # Convolution on qkv (Q, K, V all per-token for KDA — no DeltaProduct).
        nvtx_range_push(suffix="conv1d")
        seq_len = qkv.shape[1]
        qkv_channels_split_sections = [
            self.qk_dim_local_tp, self.qk_dim_local_tp, self.v_dim_local_tp,
        ]
        conv1d_weight = get_parameter_local_cp(
            self.conv1d.weight, dim=0, cp_group=self.pg_collection.cp,
            split_sections=qkv_channels_split_sections,
        )
        conv1d_bias = (
            get_parameter_local_cp(
                self.conv1d.bias, dim=0, cp_group=self.pg_collection.cp,
                split_sections=qkv_channels_split_sections,
            ) if self.conv_bias else None
        )
        if self.config.deterministic_mode:
            qkv = qkv.transpose(1, 2).contiguous()
            conv_out = F.conv1d(
                input=qkv, weight=conv1d_weight, bias=conv1d_bias,
                stride=self.conv1d.stride, padding=self.conv1d.padding,
                dilation=self.conv1d.dilation,
                groups=self.conv_dim_local_tp // self.cp_size,
            )
            qkv = self.act_fn(conv_out[..., :seq_len])
            qkv = qkv.transpose(1, 2)
        else:
            assert self.activation in ["silu", "swish"]
            qkv, _ = causal_conv1d(
                x=qkv, weight=conv1d_weight.squeeze(1), bias=conv1d_bias,
                activation=self.activation, initial_state=None, output_final_state=False,
            )
        nvtx_range_pop(suffix="conv1d")

        # Q/K/V split + reshape + alpha reshape (overridden helper handles the wider alpha).
        nvtx_range_push(suffix="prepare_qkv_for_kda")
        query, key, value, gate, beta, alpha = self._prepare_qkv_for_gated_delta_rule(
            qkv, gate, beta, alpha, batch, seq_len
        )
        nvtx_range_pop(suffix="prepare_qkv_for_kda")

        # Compute vector g and sigmoid'd beta (overridden helper).
        nvtx_range_push(suffix="g_and_beta")
        A_log_local_cp = get_parameter_local_cp(self.A_log, dim=0, cp_group=self.pg_collection.cp)
        dt_bias_local_cp = get_parameter_local_cp(self.dt_bias, dim=0, cp_group=self.pg_collection.cp)
        g, beta = self._compute_g_and_beta(A_log_local_cp, dt_bias_local_cp, alpha, beta)
        nvtx_range_pop(suffix="g_and_beta")

        # Initial state: learnable param > carried full-batch state > None.
        if self.initial_state_param is not None:
            initial_state = self.initial_state_param.unsqueeze(0).expand(batch, -1, -1, -1).contiguous()
        elif self._carry_enabled:
            num_v_heads_local = self.num_value_heads // self.tp_size
            need_init = (
                self._carried_state is None
                or self._carried_state.shape[0] != batch
                or self._carried_state.shape[1] != num_v_heads_local
            )
            if need_init:
                self._carried_state = torch.zeros(
                    batch, num_v_heads_local, self.key_head_dim, self.value_head_dim,
                    dtype=hidden_states.dtype, device=hidden_states.device,
                )
            initial_state = self._carried_state.detach()
        else:
            initial_state = None

        log_state_stats = os.environ.get("APERTUS_LOG_STATE_STATS", "0") == "1"
        need_final_state = self._carry_enabled or log_state_stats

        # chunk_kda requires initial_state in float32 (asserted inside FLA).
        initial_state_f32 = initial_state.float() if initial_state is not None else None

        nvtx_range_push(suffix="chunk_kda")
        core_attn_out, last_recurrent_state = self.gated_delta_rule(
            query, key, value,
            g=g, beta=beta,
            initial_state=initial_state_f32,
            output_final_state=need_final_state,
            use_qk_l2norm_in_kernel=False,
            use_gate_in_kernel=False,
        )
        nvtx_range_pop(suffix="chunk_kda")

        # Carry-over (matches GDN; only active if linear_attention_carry_state=True).
        if self._carry_enabled and last_recurrent_state is not None:
            with torch.no_grad():
                new_state = last_recurrent_state.detach().to(self._carried_state.dtype)
                cap = self.config.linear_attention_carried_state_max_frob
                if cap > 0.0:
                    flat = new_state.reshape(new_state.shape[0], -1).float()
                    norms = flat.norm(dim=-1, keepdim=True)
                    scale = torch.clamp(cap / norms.clamp_min(1e-12), max=1.0)
                    flat = flat * scale
                    new_state = flat.reshape_as(new_state).to(new_state.dtype)
                self._carried_state = new_state

        if log_state_stats and last_recurrent_state is not None:
            with torch.no_grad():
                s = last_recurrent_state.detach().float()
                stats = {
                    "frob": s.norm(), "amax": s.abs().amax(),
                    "abs_mean": s.abs().mean(), "std": s.std(),
                }
                if initial_state is not None:
                    init_f = initial_state.detach().float().norm()
                    stats["init_frob"] = init_f
                    stats["delta_frob"] = stats["frob"] - init_f
                else:
                    stats["init_frob"] = torch.zeros((), device=s.device)
                    stats["delta_frob"] = stats["frob"].clone()
                self._last_state_stats = stats
        else:
            self._last_state_stats = None

        # RMSNorm + (optional) output gate
        nvtx_range_push(suffix="gated_norm")
        norm_out = self._apply_gated_norm(core_attn_out, gate)
        nvtx_range_pop(suffix="gated_norm")

        # Transpose back to sbhd, CP a2a HP→CP, output projection.
        norm_out = norm_out.reshape(batch, seq_len, -1)
        norm_out = norm_out.transpose(0, 1).contiguous()
        norm_out = tensor_a2a_hp2cp(norm_out, seq_dim=0, head_dim=-1, cp_group=self.pg_collection.cp)
        nvtx_range_push(suffix="out_proj")
        out, out_bias = self.out_proj(norm_out)
        nvtx_range_pop(suffix="out_proj")
        return out, out_bias

    def _prepare_qkv_for_gated_delta_rule(self, qkv, gate, beta, alpha, batch, seq_len):
        """Reshape alpha from a flat [b, s, h*d_k] slice to [b, s, h, d_k].

        Otherwise identical to the GDN helper (Q,K l2/rmsnorm, GQA expansion).
        """
        # Same Q,K,V split as GDN
        query_key, value = torch.split(
            qkv,
            [2 * self.qk_dim_local_tp // self.cp_size, self.v_dim_local_tp // self.cp_size],
            dim=-1,
        )
        query_key = query_key.reshape(batch, seq_len, -1, self.key_head_dim)
        value = value.reshape(batch, seq_len, -1, self.value_head_dim)

        # Q,K normalization (l2norm by default; rmsnorm if configured)
        if self.use_qk_l2norm:
            query_key = query_key.contiguous()
            if self.qk_rmsnorm is not None:
                query_key = self.qk_rmsnorm(query_key)
            else:
                from fla.modules.l2norm import l2norm
                query_key = l2norm(query_key)

        split_size = self.qk_dim_local_tp // self.key_head_dim // self.cp_size
        query, key = torch.split(query_key, [split_size, split_size], dim=2)

        if self.num_value_heads // self.num_key_heads > 1:
            repeat_factor = self.num_value_heads // self.num_key_heads
            query = query.repeat_interleave(repeat_factor, dim=2)
            key = key.repeat_interleave(repeat_factor, dim=2)

        # alpha: [b, s, num_v_heads, key_head_dim]
        alpha = alpha.reshape(batch, seq_len, self.num_v_heads_local_tp, self.key_head_dim)

        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        gate = gate.contiguous()
        beta = beta.contiguous()
        alpha = alpha.contiguous()

        return query, key, value, gate, beta, alpha


def get_kimi_delta_attention_module_spec(
    config: TransformerConfig, backend=None,
) -> ModuleSpec:
    """Module spec for KDA. Mirrors the GDN spec but instantiates KimiDeltaAttention.

    Reuses the projections + output norm structure of GDN since the KDA
    layer's external interface is identical to GDN's.
    """
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_gated_delta_net_module_spec,
    )
    spec = get_gated_delta_net_module_spec(config=config, backend=backend)
    spec.module = KimiDeltaAttention
    return spec
