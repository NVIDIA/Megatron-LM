# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
memory_config.py — MemoryConfig dataclass and loader for Flextron budget calculations.

Usage
-----
From CLI args (in training/eval scripts):
    cfg = load_memory_config(args)
    total_gb = get_memory_footprint(..., memory_config=cfg)

Directly (in tests or notebooks):
    cfg = MemoryConfig(bpe_kv_cache=1, param_budget_target='active')
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import yaml

# Path to the bundled presets file (same directory as this module).
_DEFAULT_PROFILES_PATH = os.path.join(os.path.dirname(__file__), "memory_profiles.yaml")


@dataclass
class MemoryConfig:
    """
    Bytes-per-element for each memory component and param budget supervision target.

    Attributes
    ----------
    bpe_params : float
        Bytes per weight parameter element (2 = BF16, 1 = FP8/INT8, 0.5625 = FP4).
    bpe_kv_cache : float
        Bytes per KV-cache element.
    bpe_ssm_cache : float
        Bytes per SSM-state element (covers both conv_state and ssm_state).
    bpe_max_buffer : float
        Bytes per MoE dispatch buffer element.
    param_budget_target : str
        Whether the param-budget loss supervises on ``'active'`` (top-k experts only)
        or ``'total'`` (all parameters including non-active experts) parameter count.
    """
    bpe_params:          float = 2.0
    bpe_kv_cache:        float = 2.0
    bpe_ssm_cache:       float = 2.0
    bpe_max_buffer:      float = 2.0
    param_budget_target: str   = "active"   # "active" | "total"

    def __post_init__(self):
        valid_targets = {"active", "total"}
        if self.param_budget_target not in valid_targets:
            raise ValueError(
                f"param_budget_target must be one of {valid_targets}, "
                f"got '{self.param_budget_target}'"
            )


def load_memory_config(args) -> MemoryConfig:
    """
    Build a MemoryConfig from parsed CLI args.

    Resolution order (highest wins):
      1. Individual override args (--bpe-params, --bpe-kv-cache, …)
      2. Named preset from YAML (--memory-profile <name>)
      3. Built-in defaults (BF16 everywhere, active param target)

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.  Relevant attributes (all optional):
          memory_profile        str   — preset name  (default: 'bf16')
          memory_profile_path   str   — path to YAML profiles file
          bpe_params            float — override
          bpe_kv_cache          float — override
          bpe_ssm_cache         float — override
          bpe_max_buffer        float — override
          param_budget_target   str   — override ('active' | 'total')
    """
    cfg = MemoryConfig()

    # ── Load preset from YAML ──────────────────────────────────────────────
    profile_name = getattr(args, "memory_profile", "bf16") or "bf16"
    profile_path = getattr(args, "memory_profile_path", None) or _DEFAULT_PROFILES_PATH
    print(f"[memory_config] profile='{profile_name}'  path={profile_path}")

    if not os.path.isfile(profile_path):
        raise FileNotFoundError(
            f"Memory profiles file not found: {profile_path}"
        )

    with open(profile_path) as f:
        profiles = yaml.safe_load(f)

    presets = profiles.get("presets", {})
    if profile_name not in presets:
        available = list(presets.keys())
        raise ValueError(
            f"Memory profile '{profile_name}' not found in {profile_path}. "
            f"Available: {available}"
        )

    preset = presets[profile_name]
    cfg.bpe_params          = float(preset.get("params",           cfg.bpe_params))
    cfg.bpe_kv_cache        = float(preset.get("kv_cache",         cfg.bpe_kv_cache))
    cfg.bpe_ssm_cache       = float(preset.get("ssm_cache",        cfg.bpe_ssm_cache))
    cfg.bpe_max_buffer      = float(preset.get("max_buffer",       cfg.bpe_max_buffer))
    cfg.param_budget_target =       preset.get("param_budget_target", cfg.param_budget_target)
    print(f"[memory_config] after preset : bpe_params={cfg.bpe_params} bpe_kv_cache={cfg.bpe_kv_cache} "
          f"bpe_ssm_cache={cfg.bpe_ssm_cache} bpe_max_buffer={cfg.bpe_max_buffer} "
          f"param_budget_target={cfg.param_budget_target}")

    # ── Apply individual CLI overrides (take priority over preset) ─────────
    if getattr(args, "bpe_params", None) is not None:
        print(f"[memory_config] override bpe_params: {cfg.bpe_params} -> {args.bpe_params}")
        cfg.bpe_params = float(args.bpe_params)
    if getattr(args, "bpe_kv_cache", None) is not None:
        print(f"[memory_config] override bpe_kv_cache: {cfg.bpe_kv_cache} -> {args.bpe_kv_cache}")
        cfg.bpe_kv_cache = float(args.bpe_kv_cache)
    if getattr(args, "bpe_ssm_cache", None) is not None:
        print(f"[memory_config] override bpe_ssm_cache: {cfg.bpe_ssm_cache} -> {args.bpe_ssm_cache}")
        cfg.bpe_ssm_cache = float(args.bpe_ssm_cache)
    if getattr(args, "bpe_max_buffer", None) is not None:
        print(f"[memory_config] override bpe_max_buffer: {cfg.bpe_max_buffer} -> {args.bpe_max_buffer}")
        cfg.bpe_max_buffer = float(args.bpe_max_buffer)
    if getattr(args, "param_budget_target", None) is not None:
        print(f"[memory_config] override param_budget_target: {cfg.param_budget_target} -> {args.param_budget_target}")
        cfg.param_budget_target = args.param_budget_target

    print(f"[memory_config] final       : {cfg}")
    return cfg
