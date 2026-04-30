# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
FlextronConfig — all Flextron/elastification config fields in one place.

Previously these lived as fields on TransformerConfig (megatron/core).
They are now injected onto the model config at runtime via inject_flextron_config
so that megatron/core stays clean.
"""

import dataclasses
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FlextronConfig:
    # ── Core flags ────────────────────────────────────────────────────────────
    flextron: bool = False
    binary_mask: bool = False
    add_skipping: bool = False
    no_attn_skip: bool = False
    slice: bool = False
    soft_mask: bool = False

    # ── Router ────────────────────────────────────────────────────────────────
    enable_router: bool = False
    router_inter_dim: int = 128
    hard_sample_th: float = 0.996
    router_beta: float = 1.0
    loss_alpha: float = 1.0
    tau_init: float = 1.0
    tau_decay: float = 0.9999
    router_std: float = 0.1
    router_gbs: int = 32
    normalize_router_logits: bool = False
    linear_scaler_start: Optional[float] = None
    linear_scaler_end: Optional[float] = None

    # ── Budget ────────────────────────────────────────────────────────────────
    budget_probs: Optional[List[float]] = None
    budget_list: Optional[List[float]] = None
    budget_type: str = 'param'
    disable_budget: bool = False

    # ── Training / eval control ───────────────────────────────────────────────
    basemodel_type: str = 'nemotronh_8b'
    is_flex_eval: bool = False
    freeze_router: bool = False
    freeze_model: bool = False
    curr_iteration: Optional[int] = None
    original_model_sample_prob: float = 0.33
    override_selected_budget: Optional[List[float]] = None

    # ── Layer-skip constraints ────────────────────────────────────────────────
    skip_num_attn_layer_constraint: Optional[int] = None
    skip_total_layer_constraint: Optional[int] = None
    layer_ranking_list: Optional[List[int]] = None

    # ── Force overrides (eval / frozen-router mode) ───────────────────────────
    force_router_skip: Optional[List[int]] = None
    force_mlp: Optional[List[float]] = None
    force_mamba: Optional[List[float]] = None
    force_emb: Optional[List[float]] = None

    # ── Choice lists (converted to int at model-setup time) ───────────────────
    mamba_per_list: Optional[List[float]] = None
    mlp_per_list: Optional[List[float]] = None
    emb_per_list: Optional[List[float]] = None
    moe_expert_per_list: Optional[List[float]] = None
    mamba_int_list: Optional[List[int]] = None
    mlp_int_list: Optional[List[int]] = None
    emb_int_list: Optional[List[int]] = None
    moe_expert_int_list: Optional[List[int]] = None

    # ── Heterogeneous per-layer routing ───────────────────────────────────────
    flex_hetero_ffn: bool = False
    flex_hetero_mamba: bool = False
    flex_hetero_moe_expert: bool = False

    # ── Memory / inference sizing ─────────────────────────────────────────────
    prefill_chunk_size: int = 16384
    mem_infer_seq_len: int = 131072
    mem_batch_size: int = 1

    # ── Distillation ──────────────────────────────────────────────────────────
    distillation: bool = False
    distill_coeff: float = 0.0
    distill_only: bool = False


def inject_flextron_config(args, config) -> None:
    """Copy all FlextronConfig fields from parsed args onto an existing config object.

    Safe to call even when flextron args were not registered on the parser —
    falls back to FlextronConfig defaults via getattr.  After this call every
    FlextronConfig field is accessible directly as config.<field>.
    """
    # Validate per-list/int-list mutual exclusion and apply default fallbacks
    # before copying onto config so downstream code sees the resolved state.
    from megatron.elastification.arguments import validate_flextron_per_int_lists

    validate_flextron_per_int_lists(args)

    defaults = FlextronConfig()
    for f in dataclasses.fields(defaults):
        value = getattr(args, f.name, getattr(defaults, f.name))
        setattr(config, f.name, value)
