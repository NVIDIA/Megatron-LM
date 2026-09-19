"""Shared MC-derived ops for the Qwen3.5 lite path.

MCoreFullAttn / MCoreGDN / MCoreMoELayer + their build helpers live here so
that lite keeps the mbridge-derived code in one place.

Vision helpers (_build_vision_model, _resolve_hf_vision_cls) are also here so
that the vision_model DDP bucket anchor is registered consistently.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]
from mbridge import AutoBridge  # pyright: ignore[reportMissingImports,reportAttributeAccessIssue]
from mbridge.models.qwen3_5.attention import (  # pyright: ignore[reportMissingImports]
    Qwen3_5VLSelfAttention,
)
from mbridge.models.qwen3_5.rope_utils import (  # pyright: ignore[reportMissingImports]
    Qwen3VLMultimodalRotaryEmbedding,
)
from megatron.core import (
    parallel_state as mcore_parallel_state,  # pyright: ignore[reportMissingImports]
)
from megatron.core.process_groups_config import (
    ProcessGroupCollection,  # pyright: ignore[reportMissingImports]
)
from megatron.core.transformer.enums import (  # pyright: ignore[reportMissingImports]
    AttnMaskType,
)
from megatron.core.transformer.moe.moe_layer import (
    MoELayer as MCoreBaseMoELayer,  # pyright: ignore[reportMissingImports]
)

from megatron.lite.model.qwen3_5.lite._mbridge_glue import (
    _hook_fp32_rotary_emb_verbatim,
    _hook_vision_params_avg_grad_across_tp_verbatim,
)

# ---------------------------------------------------------------------------
# MC parallel-state helpers
# ---------------------------------------------------------------------------


def _effective_etp(train_cfg) -> int:
    return train_cfg.etp if train_cfg.etp is not None else 1


def _current_parallel_sizes() -> tuple[int, int, int, int, int]:
    return (
        int(mcore_parallel_state.get_tensor_model_parallel_world_size()),
        int(mcore_parallel_state.get_expert_model_parallel_world_size()),
        int(mcore_parallel_state.get_expert_tensor_parallel_world_size() or 1),
        int(mcore_parallel_state.get_pipeline_model_parallel_world_size()),
        int(mcore_parallel_state.get_context_parallel_world_size()),
    )


def ensure_mcore_parallel_state(train_cfg) -> None:
    expected = (
        train_cfg.tp,
        train_cfg.ep,
        _effective_etp(train_cfg),
        train_cfg.pp,
        train_cfg.cp,
    )
    if mcore_parallel_state.is_initialized():
        if _current_parallel_sizes() != expected:
            raise RuntimeError(
                "Hybrid runtime found an incompatible existing Megatron-Core parallel state."
            )
        return

    mcore_parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=train_cfg.tp,
        pipeline_model_parallel_size=train_cfg.pp,
        virtual_pipeline_model_parallel_size=train_cfg.vpp,
        context_parallel_size=train_cfg.cp,
        expert_model_parallel_size=train_cfg.ep,
        expert_tensor_parallel_size=_effective_etp(train_cfg),
        create_gloo_process_groups=bool(getattr(train_cfg, "deterministic", False)),
    )
    # GatedDeltaNet.reset_parameters() requires model-parallel-rng in the
    # cuda rng tracker; initialize_model_parallel does not add it.
    from megatron.core.tensor_parallel.random import (  # pyright: ignore[reportMissingImports]
        model_parallel_cuda_manual_seed,
    )
    model_parallel_cuda_manual_seed(torch.initial_seed() % (2**31))


def _set_mcore_virtual_pipeline_rank(vp_stage: int | None) -> None:
    if vp_stage is None or not mcore_parallel_state.is_initialized():
        return
    vpp_size = mcore_parallel_state.get_virtual_pipeline_model_parallel_world_size()
    if vpp_size is not None and vpp_size > 1:
        mcore_parallel_state.set_virtual_pipeline_model_parallel_rank(vp_stage)


# ---------------------------------------------------------------------------
# mbridge bridge factory
# ---------------------------------------------------------------------------


def _build_mbridge_bridge(
    model_cfg, train_cfg, *, hf_path: str = "", attention_backend_override: str | None = None
):
    import os

    bridge = AutoBridge.from_pretrained(
        hf_path,
        trust_remote_code=True,
    )
    hf_text = getattr(bridge.hf_config, "text_config", bridge.hf_config)
    hf_text.moe_intermediate_size = model_cfg.moe_intermediate_size
    hf_text.router_aux_loss_coef = model_cfg.router_aux_loss_coef
    hf_text.rope_theta = model_cfg.rope_theta
    bridge.config = bridge._build_config()
    from megatron.lite.model.qwen3_5.common import (
        _apply_moe_hack_to_bridge,
        apply_qwen3_5_bridge_config,
    )
    apply_qwen3_5_bridge_config(bridge, deterministic=getattr(train_cfg, "deterministic", True))
    bridge.set_extra_args(sequence_parallel=train_cfg.tp > 1)
    # Match the bridge runtime default. Precision validation compares bridge and
    # lite MC layerwise, so the attention backend must not silently diverge.
    backend_name = attention_backend_override or getattr(model_cfg, "attention_backend", None) or "flash"
    bridge.set_extra_args(attention_backend=backend_name)
    cp_comm_type_override = os.environ.get("BUMBLEBEE_Q35_CP_COMM_TYPE_OVERRIDE")
    if cp_comm_type_override:
        bridge.set_extra_args(cp_comm_type=cp_comm_type_override)
    _apply_moe_hack_to_bridge(bridge, model_cfg)
    bridge.config = bridge._build_config()
    return bridge


# ---------------------------------------------------------------------------
# MCoreContext
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MCoreContext:
    config: Any
    pg_collection: Any
    full_attn_submodules: Any
    gdn_submodules: Any
    moe_submodules: Any
    rotary_base: float
    bridge: Any


def build_mcore_context(
    model_cfg,
    train_cfg,
    *,
    hf_path: str = "",
    attention_backend_override: str | None = None,
    vp_stage: int | None = None,
) -> MCoreContext:
    ensure_mcore_parallel_state(train_cfg)
    _set_mcore_virtual_pipeline_rank(vp_stage)
    bridge = _build_mbridge_bridge(
        model_cfg, train_cfg, hf_path=hf_path,
        attention_backend_override=attention_backend_override,
    )
    block_spec = bridge._get_transformer_layer_spec(vp_stage=vp_stage)
    layer_specs = getattr(block_spec, "layer_specs", None)
    if not layer_specs:
        raise RuntimeError("Hybrid runtime could not extract layer specs from mbridge.")

    def _split_specs(specs):
        full = None
        gdn = None
        for spec in specs:
            submodules = spec.submodules.self_attention.submodules
            if hasattr(submodules, "core_attention"):
                full = full or spec
            else:
                gdn = gdn or spec
        return full, gdn

    full_spec, gdn_spec = _split_specs(layer_specs)
    if full_spec is None or gdn_spec is None:
        for stage in range(max(int(getattr(train_cfg, "vpp", 1) or 1), 1)):
            stage_block_spec = bridge._get_transformer_layer_spec(vp_stage=stage)
            stage_layer_specs = getattr(stage_block_spec, "layer_specs", None)
            if not stage_layer_specs:
                continue
            stage_full_spec, stage_gdn_spec = _split_specs(stage_layer_specs)
            full_spec = full_spec or stage_full_spec
            gdn_spec = gdn_spec or stage_gdn_spec
            if full_spec is not None and gdn_spec is not None:
                break
    if full_spec is None or gdn_spec is None:
        raise RuntimeError("Hybrid runtime could not identify both full-attention and GDN layer specs.")

    full_attn_submodules = full_spec.submodules.self_attention.submodules
    gdn_submodules = gdn_spec.submodules.self_attention.submodules
    moe_submodules = gdn_spec.submodules.mlp.submodules

    return MCoreContext(
        config=bridge.config,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        full_attn_submodules=full_attn_submodules,
        gdn_submodules=gdn_submodules,
        moe_submodules=moe_submodules,
        rotary_base=bridge._get_gptmodel_args()["rotary_base"],
        bridge=bridge,
    )


# ---------------------------------------------------------------------------
# MCoreFullAttn
# ---------------------------------------------------------------------------


class MCoreFullAttn(Qwen3_5VLSelfAttention):
    """MCore full-attention with an ML-compatible forward signature."""

    def __init__(
        self,
        model_cfg,
        train_cfg,
        ps,
        layer_idx: int,
        mcore_context: MCoreContext,
        *,
        layer_number: int | None = None,
    ):
        super().__init__(
            config=mcore_context.config,
            submodules=mcore_context.full_attn_submodules,
            layer_number=layer_number if layer_number is not None else layer_idx + 1,
            attn_mask_type=AttnMaskType.causal,
            pg_collection=mcore_context.pg_collection,
        )
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.ps = ps
        self.layer_idx = layer_idx
        self.rotary_embedding = Qwen3VLMultimodalRotaryEmbedding(
            kv_channels=mcore_context.config.kv_channels,
            rotary_percent=mcore_context.config.rotary_percent,
            rotary_interleaved=mcore_context.config.rotary_interleaved,
            seq_len_interpolation_factor=None,
            rotary_base=int(mcore_context.rotary_base),
        )
        self.mrope_section = mcore_context.config.mrope_section

    @property
    def qkv(self):
        return self.linear_qkv

    @property
    def core_attn(self):
        return self.core_attention

    @property
    def proj(self):
        return self.linear_proj

    @property
    def q_norm(self):
        return self.q_layernorm

    @property
    def k_norm(self):
        return self.k_layernorm

    def _build_rotary_pos_emb(self, position_ids: torch.Tensor):
        return self.rotary_embedding(position_ids, self.mrope_section)

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        if position_ids is None:
            raise ValueError(
                "Hybrid MCore full_attention requires position_ids (shape (3,b,s) for mrope)."
            )
        rotary_pos_emb = self._build_rotary_pos_emb(position_ids)
        output, bias = super().forward(
            hidden_states=x,
            attention_mask=None,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
        )
        if bias is not None:
            output = output + bias
        return output


# ---------------------------------------------------------------------------
# MCoreGDN
# ---------------------------------------------------------------------------


class MCoreGDN(nn.Module):
    """MC GatedDeltaNet wrapper for linear_attention layers."""

    def __init__(
        self,
        model_cfg,
        train_cfg,
        ps,
        layer_idx: int,
        mcore_context: MCoreContext,
        *,
        layer_number: int | None = None,
    ):
        super().__init__()
        from megatron.core.ssm.gated_delta_net import GatedDeltaNet  # pyright: ignore

        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.ps = ps
        self.layer_idx = layer_idx
        self.layer_number = layer_number if layer_number is not None else layer_idx + 1
        self.gdn = GatedDeltaNet(
            config=mcore_context.config,
            submodules=mcore_context.gdn_submodules,
            layer_number=self.layer_number,
            pg_collection=mcore_context.pg_collection,
        )
        assert self.gdn.pg_collection.cp.size() == mcore_parallel_state.get_context_parallel_world_size(), (
            f"MCoreGDN cp group size {self.gdn.pg_collection.cp.size()} != "
            f"mcore CP world size {mcore_parallel_state.get_context_parallel_world_size()}"
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        out, out_bias = self.gdn(hidden_states=x, attention_mask=None, packed_seq_params=packed_seq_params)
        if out_bias is not None:
            out = out + out_bias
        return out


# ---------------------------------------------------------------------------
# MCoreMoELayer
# ---------------------------------------------------------------------------


class MCoreMoELayer(MCoreBaseMoELayer):
    """MCore MoE with an ML-compatible forward signature."""

    def __init__(
        self,
        model_cfg,
        train_cfg,
        ps,
        layer_idx: int,
        mcore_context: MCoreContext,
        *,
        layer_number: int | None = None,
        is_mtp_layer: bool = False,
    ):
        super().__init__(
            config=mcore_context.config,
            submodules=mcore_context.moe_submodules,
            layer_number=layer_number if layer_number is not None else layer_idx + 1,
            pg_collection=mcore_context.pg_collection,
            is_mtp_layer=is_mtp_layer,
        )
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.ps = ps
        self.layer_idx = layer_idx

    @property
    def dispatcher(self):
        return self.token_dispatcher

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        output, mlp_bias = cast(
            tuple[torch.Tensor, torch.Tensor | None],
            super().forward(hidden_states, padding_mask=None),
        )
        if mlp_bias is not None:
            output = output + mlp_bias
        return output.to(hidden_states.dtype)


# ---------------------------------------------------------------------------
# Vision model helpers
# ---------------------------------------------------------------------------


def _resolve_hf_vision_cls(bridge) -> type:
    cls = getattr(type(bridge), "HfVisionClass", None)
    if cls is None:
        raise RuntimeError("Bridge does not expose HfVisionClass; cannot build vision_model.")
    return cls


def _build_vision_model(bridge) -> nn.Module:
    """Instantiate HF vision encoder and apply mbridge VL hooks.

    vision_model is NOT called in text-only forward; its parameters exist
    solely to make the DDP bucket layout match the mbridge Qwen3_5VLModel.
    """
    hf_vision_cls = _resolve_hf_vision_cls(bridge)
    vision = hf_vision_cls._from_config(bridge.hf_config.vision_config)
    _hook_fp32_rotary_emb_verbatim(vision)
    _hook_vision_params_avg_grad_across_tp_verbatim(vision)
    return vision.to(torch.bfloat16)


# ---------------------------------------------------------------------------

__all__ = [
    "MCoreContext",
    "MCoreFullAttn",
    "MCoreGDN",
    "MCoreMoELayer",
    "_build_mbridge_bridge",
    "_build_vision_model",
    "_current_parallel_sizes",
    "_effective_etp",
    "_resolve_hf_vision_cls",
    "build_mcore_context",
    "ensure_mcore_parallel_state",
]
