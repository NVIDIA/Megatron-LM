"""Qwen3.5 bridge config contract — single maintenance point (Fix-M2)."""
from __future__ import annotations


def is_expert_param(name: str) -> bool:
    return "experts" in name and "router" not in name and "shared" not in name


def apply_qwen3_5_bridge_config(bridge, *, deterministic: bool = True) -> None:
    """Register required TransformerConfig flags via set_extra_args (Fix-M2).

    Must be called after AutoBridge.from_pretrained() AND before bridge_post_init
    (or any _apply_moe_hack call). Uses set_extra_args so flags survive every
    subsequent _build_config() rebuild via mbridge's extra_args highest-priority
    override channel (llm_bridge.py::_build_base_config last update).

    deterministic: when True, set MC deterministic_mode for bitwise runs.
      MUST be False for THD / packed_seq_params runs (GDN hard assert).
    fused_single_qkv_rope=False → full_attn output_gate layout is incompatible
                               with fused QKV+RoPE fusion.
    """
    bridge.set_extra_args(
        deterministic_mode=deterministic,
        fused_single_qkv_rope=False,
    )


def _apply_moe_hack_to_bridge(bridge, model_cfg) -> None:
    """Mirror model_cfg truncations into bridge.hf_config + install router monkey-patch.

    Handles Qwen3.5's nested text_config structure. Must be called AFTER
    apply_qwen3_5_bridge_config so extra_args flags survive the final _build_config.
    Only mutates when cfg < bridge (truncation); no-ops otherwise.
    """
    hf_text = getattr(bridge.hf_config, "text_config", bridge.hf_config)

    bridge_num_experts = getattr(hf_text, "num_experts", None)
    bridge_num_layers = getattr(hf_text, "num_hidden_layers", None)
    cfg_num_experts = getattr(model_cfg, "num_experts", None)
    cfg_num_layers = getattr(model_cfg, "num_hidden_layers", None)

    needs_rebuild = False

    # ── expert truncation ────────────────────────────────────────────
    if bridge_num_experts is not None and cfg_num_experts is not None:
        if cfg_num_experts < bridge_num_experts:
            hf_text.num_experts = cfg_num_experts
            hf_text.num_experts_per_tok = model_cfg.num_experts_per_tok
            needs_rebuild = True
            keep = cfg_num_experts
            orig_fn = bridge._weight_to_mcore_format

            def patched(name: str, hf_weights: list, _k=keep, _f=orig_fn):
                if "mlp.router.weight" in name and len(hf_weights) == 1:
                    hf_weights = [hf_weights[0][:_k].contiguous()]
                return _f(name, hf_weights)

            bridge._weight_to_mcore_format = patched

    # ── layer truncation ─────────────────────────────────────────────
    if bridge_num_layers is not None and cfg_num_layers is not None:
        if cfg_num_layers < bridge_num_layers:
            hf_text.num_hidden_layers = cfg_num_layers
            needs_rebuild = True

    if needs_rebuild:
        bridge.config = bridge._build_config()
