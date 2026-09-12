"""Qwen3.5: HF safetensors load/save, weight mapping, DCP spec.

Post-Step-3: load_hf_weights uses mbridge bridge dispatch (MC naming).
export_weights / save_hf_weights use bridge._weight_to_hf_format for reverse-map.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import save_file as safe_save
from torch.distributed.tensor import Replicate, Shard

from megatron.lite.model.qwen3_5.config import Qwen35Config
from megatron.lite.primitive.ckpt.hf_bridge import SafeTensorReader as _HFSafeTensorReader
from megatron.lite.primitive.ckpt.hf_bridge import unwrap_model
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.utils import ensure_divisible, log_rank0

# ── DCP Placement & Expert Classifier ─────────────────────────────────


def PLACEMENT_FN(param_name: str) -> list:
    """DTensor placement for DCP checkpoint save/load."""
    if "experts" in param_name and "router" not in param_name and "shared" not in param_name:
        if "fc1" in param_name:
            return [Replicate(), Replicate(), Shard(0), Shard(0)]
        elif "fc2" in param_name:
            return [Replicate(), Replicate(), Shard(0), Shard(1)]
        else:
            return [Replicate(), Replicate(), Replicate(), Replicate()]
    else:
        if ("in_proj" in param_name or "qkv" in param_name) and "layer_norm" not in param_name:
            return [Replicate(), Replicate(), Replicate(), Shard(0)]
        elif ("proj" in param_name or "o_proj" in param_name) and "attn" in param_name:
            return [Replicate(), Replicate(), Replicate(), Shard(1)]
        elif "gate_up" in param_name and "shared" in param_name:
            return [Replicate(), Replicate(), Replicate(), Shard(0)]
        elif "down" in param_name and "shared" in param_name:
            return [Replicate(), Replicate(), Replicate(), Shard(1)]
        elif "eh_proj" in param_name:
            return [Replicate(), Replicate(), Replicate(), Shard(0)]
        elif "embed" in param_name or "head" in param_name:
            return [Replicate(), Replicate(), Replicate(), Shard(0)]
        elif "conv1d" in param_name or "dt_bias" in param_name or "A_log" in param_name:
            return [Replicate(), Replicate(), Replicate(), Shard(0)]
        else:
            return [Replicate(), Replicate(), Replicate(), Replicate()]


def EXPERT_CLASSIFIER(name: str) -> bool:
    """Which params are expert params (for EP-aware DDP)."""
    return "experts" in name and "router" not in name and "shared" not in name


# ── Bridge-based loading (MC naming) ───────────────────────────────────

_GLOBAL_MBRIDGE = {
    "embed.embedding.weight": "embedding.word_embeddings.weight",
    "mtp_embed.embedding.weight": "embedding.word_embeddings.weight",
    "norm.weight": "decoder.final_layernorm.weight",
    "head.col.linear.weight": "output_layer.weight",
}

_FULL_ATTN_MBRIDGE = {
    "self_attention.linear_qkv.layer_norm_weight": "self_attention.linear_qkv.layer_norm_weight",
    "self_attention.linear_qkv.weight": "self_attention.linear_qkv.weight",
    "self_attention.q_layernorm.weight": "self_attention.q_layernorm.weight",
    "self_attention.k_layernorm.weight": "self_attention.k_layernorm.weight",
    "self_attention.linear_proj.weight": "self_attention.linear_proj.weight",
}

_GDN_MBRIDGE = {
    "self_attention.gdn.in_proj.weight": "self_attention.in_proj.weight",
    "self_attention.gdn.in_proj.layer_norm_weight": "self_attention.in_proj.layer_norm_weight",
    "self_attention.gdn.conv1d.weight": "self_attention.conv1d.weight",
    "self_attention.gdn.dt_bias": "self_attention.dt_bias",
    "self_attention.gdn.A_log": "self_attention.A_log",
    "self_attention.gdn.out_norm.weight": "self_attention.out_norm.weight",
    "self_attention.gdn.out_proj.weight": "self_attention.out_proj.weight",
}

_MOE_MBRIDGE = {
    "pre_mlp_layernorm.weight": "pre_mlp_layernorm.weight",
    "mlp.router.weight": "mlp.router.weight",
    "mlp.shared_experts.linear_fc1.weight": "mlp.shared_experts.linear_fc1.weight",
    "mlp.shared_experts.linear_fc2.weight": "mlp.shared_experts.linear_fc2.weight",
    "mlp.shared_experts.gate_weight": "mlp.shared_experts.gate_weight",
}


def _lite_to_mbridge_name(local_name: str, global_idx_map: dict[int, int]) -> str | None:
    def _map() -> str | None:
        if local_name in _GLOBAL_MBRIDGE:
            return _GLOBAL_MBRIDGE[local_name]
        if not local_name.startswith("layers."):
            return None
        parts = local_name.split(".", 2)
        if len(parts) < 3:
            return None
        try:
            local_i = int(parts[1])
        except ValueError:
            return None
        g = global_idx_map.get(local_i)
        if g is None:
            return None
        suffix = parts[2]
        if suffix in _FULL_ATTN_MBRIDGE:
            return f"decoder.layers.{g}.{_FULL_ATTN_MBRIDGE[suffix]}"
        if suffix in _GDN_MBRIDGE:
            return f"decoder.layers.{g}.{_GDN_MBRIDGE[suffix]}"
        if suffix in _MOE_MBRIDGE:
            return f"decoder.layers.{g}.{_MOE_MBRIDGE[suffix]}"
        if suffix.startswith("mlp.experts.linear_fc1.weight") or suffix.startswith(
            "mlp.experts.linear_fc2.weight"
        ):
            return f"decoder.layers.{g}.{suffix}"
        return None

    mc = _map()
    return f"language_model.{mc}" if mc is not None else None


def _is_mtp_param(local_name: str) -> bool:
    return local_name.startswith("mtp.layers.")


def _is_mtp_embedding_param(local_name: str) -> bool:
    return local_name == "mtp_embed.embedding.weight"


def _mtp_layer_parts(local_name: str) -> tuple[int, str] | None:
    parts = local_name.split(".", 3)
    if len(parts) != 4 or parts[0] != "mtp" or parts[1] != "layers":
        return None
    try:
        return int(parts[2]), parts[3]
    except ValueError:
        return None


def _mtp_direct_hf_name(local_name: str) -> str | None:
    parts = _mtp_layer_parts(local_name)
    if parts is None:
        return None
    _mtp_idx, suffix = parts
    direct = {
        "enorm.weight": "mtp.pre_fc_norm_embedding.weight",
        "hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
        "eh_proj.linear.weight": "mtp.fc.weight",
        "final_layernorm.weight": "mtp.norm.weight",
    }
    return direct.get(suffix)


def _mtp_proxy_layer_idx(config: Qwen35Config, mtp_idx: int) -> int:
    mtp_type = config.layer_type_at(config.num_hidden_layers + mtp_idx)
    for layer_idx in range(config.num_hidden_layers):
        if config.layer_type_at(layer_idx) == mtp_type:
            return layer_idx
    return 0


def _mtp_transformer_mbridge_name(local_name: str, config: Qwen35Config) -> tuple[str, int, int] | None:
    parts = _mtp_layer_parts(local_name)
    if parts is None:
        return None
    mtp_idx, suffix = parts
    if not suffix.startswith("transformer_layer."):
        return None
    inner = suffix.removeprefix("transformer_layer.")
    proxy_idx = _mtp_proxy_layer_idx(config, mtp_idx)
    mc_name = _lite_to_mbridge_name(f"layers.{proxy_idx}.{inner}", {proxy_idx: proxy_idx})
    if mc_name is None:
        return None
    return mc_name, proxy_idx, mtp_idx


def _mtp_rewrite_proxy_hf_name(hf_name: str, proxy_idx: int, mtp_idx: int) -> str:
    hf_name = str(hf_name)
    marker = "language_mtp.layers."
    if marker in hf_name:
        return "mtp.layers." + hf_name.split(marker, 1)[1]
    src_language_mtp = f"model.language_mtp.layers.{mtp_idx}."
    dst = f"mtp.layers.{mtp_idx}."
    if hf_name.startswith(src_language_mtp):
        return dst + hf_name[len(src_language_mtp):]
    src = f"model.layers.{proxy_idx}."
    if hf_name.startswith(src):
        return dst + hf_name[len(src):]
    if src in hf_name:
        return hf_name.replace(src, dst, 1)
    return hf_name


def _mtp_expert_info(local_name: str) -> tuple[int, str, str] | None:
    parts = _mtp_layer_parts(local_name)
    if parts is None:
        return None
    mtp_idx, suffix = parts
    fc1_match = re.fullmatch(r"transformer_layer\.mlp\.experts\.linear_fc1\.weight(\d+)", suffix)
    if fc1_match is not None:
        return mtp_idx, "linear_fc1", fc1_match.group(1)
    fc2_match = re.fullmatch(r"transformer_layer\.mlp\.experts\.linear_fc2\.weight(\d+)", suffix)
    if fc2_match is not None:
        return mtp_idx, "linear_fc2", fc2_match.group(1)
    return None


def _mtp_global_expert_id(local_expert_id: str, config: Qwen35Config, ps: ParallelState) -> int:
    num_local = ensure_divisible(config.num_experts, max(ps.ep_size, 1))
    return ps.ep_rank * num_local + int(local_expert_id)


def _mtp_expert_hf_names(local_name: str, config: Qwen35Config, ps: ParallelState) -> list[str] | None:
    info = _mtp_expert_info(local_name)
    if info is None:
        return None
    mtp_idx, kind, local_expert_id = info
    global_expert_id = _mtp_global_expert_id(local_expert_id, config, ps)
    prefix = f"mtp.layers.{mtp_idx}.mlp.experts.{global_expert_id}."
    if kind == "linear_fc1":
        return [f"{prefix}gate_proj.weight", f"{prefix}up_proj.weight"]
    return [f"{prefix}down_proj.weight"]


def _mtp_expert_tensor_from_hf(local_name: str, hf_tensors: list[torch.Tensor]) -> torch.Tensor | None:
    info = _mtp_expert_info(local_name)
    if info is None:
        return None
    _mtp_idx, kind, _expert_id = info
    if kind == "linear_fc1":
        return torch.cat(hf_tensors, dim=0)
    return hf_tensors[0]


def _split_mtp_expert_tensor_for_rank(
    local_name: str,
    tensor: torch.Tensor,
    ps: ParallelState,
) -> torch.Tensor:
    info = _mtp_expert_info(local_name)
    if info is None or ps.etp_size <= 1:
        return tensor
    _mtp_idx, kind, _expert_id = info
    if kind == "linear_fc1":
        gate, up = tensor.chunk(2, dim=0)
        ensure_divisible(gate.shape[0], ps.etp_size)
        ensure_divisible(up.shape[0], ps.etp_size)
        return torch.cat(
            [
                gate.chunk(ps.etp_size, dim=0)[ps.etp_rank],
                up.chunk(ps.etp_size, dim=0)[ps.etp_rank],
            ],
            dim=0,
        )
    ensure_divisible(tensor.shape[1], ps.etp_size)
    return tensor.chunk(ps.etp_size, dim=1)[ps.etp_rank]


def _mtp_expert_agg_info(local_name: str) -> tuple[int, str] | None:
    parts = _mtp_layer_parts(local_name)
    if parts is None:
        return None
    mtp_idx, suffix = parts
    if suffix == "transformer_layer.mlp.experts.linear_fc1.weight":
        return mtp_idx, "linear_fc1"
    if suffix == "transformer_layer.mlp.experts.linear_fc2.weight":
        return mtp_idx, "linear_fc2"
    return None


def _mtp_expert_export_items(local_name: str, tensor: torch.Tensor) -> dict[str, torch.Tensor] | None:
    info = _mtp_expert_agg_info(local_name)
    if info is None:
        return None
    mtp_idx, kind = info
    out: dict[str, torch.Tensor] = {}
    for expert_id, expert_tensor in enumerate(tensor):
        prefix = f"mtp.layers.{mtp_idx}.mlp.experts.{expert_id}."
        if kind == "linear_fc1":
            gate, up = expert_tensor.chunk(2, dim=0)
            out[f"{prefix}gate_proj.weight"] = gate
            out[f"{prefix}up_proj.weight"] = up
        else:
            out[f"{prefix}down_proj.weight"] = expert_tensor
    return out


def _mtp_manual_transformer_hf_names(
    local_name: str,
    mtp_idx: int,
    config: Qwen35Config | None = None,
    ps: ParallelState | None = None,
) -> list[str] | None:
    parts = _mtp_layer_parts(local_name)
    if parts is None:
        return None
    _, suffix = parts
    if not suffix.startswith("transformer_layer."):
        return None
    inner = suffix.removeprefix("transformer_layer.")
    prefix = f"mtp.layers.{mtp_idx}."

    direct = {
        "self_attention.linear_proj.weight": [f"{prefix}self_attn.o_proj.weight"],
        "self_attention.linear_qkv.layer_norm_weight": [f"{prefix}input_layernorm.weight"],
        "self_attention.linear_qkv.weight": [
            f"{prefix}self_attn.q_proj.weight",
            f"{prefix}self_attn.k_proj.weight",
            f"{prefix}self_attn.v_proj.weight",
        ],
        "self_attention.q_layernorm.weight": [f"{prefix}self_attn.q_norm.weight"],
        "self_attention.k_layernorm.weight": [f"{prefix}self_attn.k_norm.weight"],
        "pre_mlp_layernorm.weight": [f"{prefix}post_attention_layernorm.weight"],
        "mlp.router.weight": [f"{prefix}mlp.gate.weight"],
        "mlp.shared_experts.gate_weight": [f"{prefix}mlp.shared_expert_gate.weight"],
        "mlp.shared_experts.linear_fc1.weight": [
            f"{prefix}mlp.shared_expert.gate_proj.weight",
            f"{prefix}mlp.shared_expert.up_proj.weight",
        ],
        "mlp.shared_experts.linear_fc2.weight": [f"{prefix}mlp.shared_expert.down_proj.weight"],
    }
    if inner in direct:
        return direct[inner]
    if config is not None and ps is not None:
        expert_hf_names = _mtp_expert_hf_names(local_name, config, ps)
        if expert_hf_names is not None:
            return expert_hf_names
    return None


def _mtp_transformer_hf_names(
    local_name: str,
    config: Qwen35Config,
    bridge,
    ps: ParallelState,
) -> tuple[str, list[str]] | None:
    mapped = _mtp_transformer_mbridge_name(local_name, config)
    if mapped is None:
        return None
    mc_name, proxy_idx, mtp_idx = mapped
    manual_hf_names = _mtp_manual_transformer_hf_names(local_name, mtp_idx, config, ps)
    if manual_hf_names is not None:
        return mc_name, manual_hf_names
    hf_names = bridge._weight_name_mapping_mcore_to_hf(mc_name)
    return mc_name, [_mtp_rewrite_proxy_hf_name(n, proxy_idx, mtp_idx) for n in hf_names]


def _split_mtp_direct_tensor(
    local_name: str,
    tensor: torch.Tensor,
    param: torch.nn.Parameter,
    ps: ParallelState,
) -> torch.Tensor:
    if local_name.endswith(".eh_proj.linear.weight") and ps.tp_size > 1 and tensor.shape != param.data.shape:
        ensure_divisible(tensor.shape[0], ps.tp_size)
        start = ps.tp_rank * (tensor.shape[0] // ps.tp_size)
        stop = (ps.tp_rank + 1) * (tensor.shape[0] // ps.tp_size)
        tensor = tensor[start:stop]
    return tensor


def _mtp_direct_export_items(local_name: str, tensor: torch.Tensor) -> dict[str, torch.Tensor] | None:
    hf_name = _mtp_direct_hf_name(local_name)
    if hf_name is None:
        return None
    return {hf_name: tensor}


def _mtp_transformer_export_items(
    local_name: str,
    tensor: torch.Tensor,
    config: Qwen35Config,
    bridge,
) -> dict[str, torch.Tensor] | None:
    mapped = _mtp_transformer_mbridge_name(local_name, config)
    if mapped is None:
        return None
    mc_name, proxy_idx, mtp_idx = mapped
    hf_names, hf_tensors = bridge._weight_to_hf_format(mc_name, tensor)
    manual_hf_names = _mtp_manual_transformer_hf_names(local_name, mtp_idx)
    if manual_hf_names is not None and len(manual_hf_names) == len(hf_tensors):
        hf_names = manual_hf_names
    return {
        _mtp_rewrite_proxy_hf_name(hn, proxy_idx, mtp_idx): ht
        for hn, ht in zip(hf_names, hf_tensors, strict=True)
    }


# ══════════════════════════════════════════════════════════════════════
# Main load entry point
# ══════════════════════════════════════════════════════════════════════


def load_hf_weights(
    model: nn.Module,
    path: str,
    config: Qwen35Config,
    ps: ParallelState,
    *,
    num_layers_override: int | None = None,  # noqa: ARG001 — model.layer_indices already correct
) -> None:
    """Load HF safetensors into a Qwen35Model via mbridge bridge dispatch."""
    from megatron.lite.model.qwen3_5.lite._mbridge_glue import _weight_split_across_tp

    base_model = unwrap_model(model)
    bridge = getattr(base_model, "mbridge_bridge", None)
    if bridge is None:
        raise RuntimeError("Qwen35Model (lite) requires mbridge_bridge; pass hf_path to build_model.")

    reader = _HFSafeTensorReader(path)
    loaded: dict[str, torch.Tensor] = {}
    global_idx_map = {i: g for i, g in enumerate(base_model.layer_indices)}

    # D1: track failure reason for every param → greppable [lite/WARN-NOTLOADED] dump
    _fail_reason: dict[str, tuple[str, list[str]]] = {}  # name → (mc_name, hf_keys_tried)

    for local_name, param in base_model.named_parameters():
        if _is_mtp_param(local_name):
            direct_hf_name = _mtp_direct_hf_name(local_name)
            if direct_hf_name is not None:
                try:
                    my_slice = _split_mtp_direct_tensor(
                        local_name,
                        reader.get_tensor(direct_hf_name),
                        param,
                        ps,
                    )
                except Exception as _e:
                    _fail_reason[local_name] = (direct_hf_name, [f"<reader_error:{_e}>"])
                    continue
                if my_slice.shape != param.data.shape:
                    _fail_reason[local_name] = (direct_hf_name, [direct_hf_name])
                    log_rank0(
                        f"[lite/WARN-NOTLOADED] name={local_name} hf_name={direct_hf_name} "
                        f"shape_mismatch loaded={tuple(my_slice.shape)} param={tuple(param.data.shape)}"
                    )
                    continue
                loaded[local_name] = my_slice.to(dtype=param.data.dtype)
                continue

            expert_hf_names = _mtp_expert_hf_names(local_name, config, ps)
            if expert_hf_names is not None:
                try:
                    hf_tensors = [reader.get_tensor(n) for n in expert_hf_names]
                except Exception as _e:
                    _fail_reason[local_name] = ("mtp_expert_direct", expert_hf_names)
                    log_rank0(
                        f"[lite/WARN-NOTLOADED] name={local_name} mc_name=mtp_expert_direct "
                        f"hf_keys={expert_hf_names} error={_e}"
                    )
                    continue
                mcore_weight = _mtp_expert_tensor_from_hf(local_name, hf_tensors)
                if mcore_weight is None:
                    _fail_reason[local_name] = ("mtp_expert_direct", expert_hf_names)
                    continue
                my_slice = _split_mtp_expert_tensor_for_rank(local_name, mcore_weight, ps)
                if my_slice.shape != param.data.shape:
                    _fail_reason[local_name] = ("mtp_expert_direct", expert_hf_names)
                    log_rank0(
                        f"[lite/WARN-NOTLOADED] name={local_name} mc_name=mtp_expert_direct "
                        f"shape_mismatch loaded={tuple(my_slice.shape)} param={tuple(param.data.shape)}"
                    )
                    continue
                loaded[local_name] = my_slice.to(dtype=param.data.dtype)
                continue

            mapped = _mtp_transformer_hf_names(local_name, config, bridge, ps)
            if mapped is None:
                _fail_reason[local_name] = ("no_mtp_mapping", [])
                continue
            mc_name, hf_names = mapped
            try:
                hf_tensors = [reader.get_tensor(n) for n in hf_names]
            except Exception as _e:
                _fail_reason[local_name] = (mc_name, hf_names)
                log_rank0(f"[lite/WARN-NOTLOADED] name={local_name} mc_name={mc_name} hf_keys={hf_names} error={_e}")
                continue
            mcore_weight = bridge._weight_to_mcore_format(mc_name, hf_tensors)
            is_expert = "mlp.experts.linear_fc" in mc_name
            tp_size = ps.etp_size if is_expert else ps.tp_size
            tp_rank = ps.etp_rank if is_expert else ps.tp_rank
            splits = _weight_split_across_tp(bridge, mc_name, mcore_weight, param, tp_size)
            my_slice = splits[tp_rank]
            if my_slice.shape != param.data.shape:
                _fail_reason[local_name] = (mc_name, hf_names)
                log_rank0(
                    f"[lite/WARN-NOTLOADED] name={local_name} mc_name={mc_name} "
                    f"shape_mismatch loaded={tuple(my_slice.shape)} param={tuple(param.data.shape)}"
                )
                continue
            loaded[local_name] = my_slice.to(dtype=param.data.dtype)
            continue

        mc_name = _lite_to_mbridge_name(local_name, global_idx_map)
        if mc_name is None:
            _fail_reason[local_name] = ("no_mbridge_mapping", [])
            continue
        try:
            hf_names = bridge._weight_name_mapping_mcore_to_hf(mc_name)
        except (KeyError, NotImplementedError) as _e:
            _fail_reason[local_name] = (mc_name, [f"<bridge_error:{_e}>"])
            continue
        try:
            hf_tensors = [reader.get_tensor(n) for n in hf_names]
        except Exception as _e:
            _fail_reason[local_name] = (mc_name, hf_names)
            log_rank0(f"[lite/WARN-NOTLOADED] name={local_name} mc_name={mc_name} hf_keys={hf_names} error={_e}")
            continue
        mcore_weight = bridge._weight_to_mcore_format(mc_name, hf_tensors)
        is_expert = "mlp.experts.linear_fc" in mc_name
        tp_size = ps.etp_size if is_expert else ps.tp_size
        tp_rank = ps.etp_rank if is_expert else ps.tp_rank
        splits = _weight_split_across_tp(bridge, mc_name, mcore_weight, param, tp_size)
        my_slice = splits[tp_rank]
        if my_slice.shape != param.data.shape:
            _fail_reason[local_name] = (mc_name, hf_names)
            log_rank0(
                f"[lite/WARN-NOTLOADED] name={local_name} mc_name={mc_name} "
                f"shape_mismatch loaded={tuple(my_slice.shape)} param={tuple(param.data.shape)}"
            )
            continue
        loaded[local_name] = my_slice.to(dtype=torch.bfloat16)

    if getattr(base_model, "vision_model", None) is not None:
        for vname, _vparam in base_model.vision_model.named_parameters():
            mcore_name = f"vision_model.{vname}"
            try:
                hf_names = bridge._weight_name_mapping_mcore_to_hf(mcore_name)
            except (KeyError, NotImplementedError):
                log_rank0(f"WARNING: vision param {mcore_name} has no HF mapping, skipping")
                continue
            if len(hf_names) != 1:
                log_rank0(f"WARNING: vision param {mcore_name} maps to {len(hf_names)} HF names, skipping")
                continue
            loaded[f"vision_model.{vname}"] = reader.get_tensor(hf_names[0]).to(dtype=torch.bfloat16)

    for name, param in base_model.named_parameters():
        if name in loaded:
            param.data.copy_(loaded[name])
        else:
            mc_name_fail, hf_keys_fail = _fail_reason.get(name, ("?", []))
            log_rank0(
                f"[lite/WARN-NOTLOADED] name={name} mc_name={mc_name_fail} hf_keys={hf_keys_fail}"
            )


# ══════════════════════════════════════════════════════════════════════
# Export weights (gather shards → full tensors → HF format)
# ══════════════════════════════════════════════════════════════════════


def export_weights(
    model: nn.Module | list[nn.Module],
    config: Qwen35Config,
    ps: ParallelState,
    *,
    include_mtp_only: bool = False,
    include_local_prefixes: tuple[str, ...] | list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Gather distributed shards → HF flat dict.

    Rank-0 returns full {hf_key: tensor} dict; other ranks return {}.
    """
    chunks = model if isinstance(model, list | nn.ModuleList) else [model]
    base = unwrap_model(chunks[0])
    bridge = getattr(base, "mbridge_bridge", None)
    if bridge is None:
        raise RuntimeError(
            "export_weights requires mbridge_bridge; "
            "ensure load_hf_weights was called first (attaches bridge)."
        )

    # Stage 1: gather distributed shards → full MC tensors keyed by global param name
    mc: dict[str, torch.Tensor] = {}
    identity_map = {g: g for g in range(config.num_hidden_layers)}
    for chunk in chunks:
        chunk_base = unwrap_model(chunk)
        layer_map = (
            {i: chunk_base.layer_indices[i] for i in range(len(chunk_base.layer_indices))}
            if hasattr(chunk_base, "layer_indices") else {}
        )
        for name, param in chunk_base.named_parameters():
            gname = _to_global_layer_name(name, layer_map)
            if _is_mtp_embedding_param(gname):
                continue
            if include_mtp_only and not _is_mtp_param(gname):
                continue
            if include_local_prefixes and not any(
                gname.startswith(prefix) for prefix in include_local_prefixes
            ):
                continue
            if not _is_mtp_param(gname) and _lite_to_mbridge_name(gname, identity_map) is None:
                continue
            if _is_expert_weight(gname):
                _gather_expert_param(gname, param.data, config, ps, mc)
            else:
                mc[gname] = _gather_dense_param(gname, param.data, ps, config)

    # PP gather only on the canonical TP/CP/DP lane.  The per-param TP/EP
    # gathers above already materialize full tensors on every lane, so gathering
    # every PP group duplicates a full-model object on all ranks and OOMs large
    # checkpoints.  Use the CPU PP group when available because object collectives
    # over NCCL serialize through CUDA buffers.
    if ps.pp_size > 1:
        export_lane = ps.tp_rank == 0 and ps.cp_rank == 0 and ps.dp_rank == 0
        if export_lane:
            if ps.pp_global_ranks is None:
                raise RuntimeError("Pipeline ranks were not initialized.")
            dst_rank = ps.pp_global_ranks[0]
            group = ps.pp_cpu_group or ps.pp_group
            if group is None:
                raise RuntimeError("Pipeline process group was not initialized.")
            all_states: list[dict | None] | None
            all_states = [None] * ps.pp_size if dist.get_rank() == dst_rank else None
            dist.gather_object(mc, all_states, dst=dst_rank, group=group)
            if dist.get_rank() == dst_rank:
                merged: dict[str, torch.Tensor] = {}
                for s in all_states or []:
                    if s is not None:
                        merged.update(s)
                mc = merged
            else:
                mc = {}
        else:
            mc = {}

    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return {}

    # Aggregate per-expert tensors into stacked [E, ...] tensors for bridge reverse-map
    expert_fc1: dict[str, dict[int, torch.Tensor]] = {}
    expert_fc2: dict[str, dict[int, torch.Tensor]] = {}
    non_expert: dict[str, torch.Tensor] = {}
    for name, tensor in mc.items():
        if _is_expert_weight(name) and _is_mtp_param(name):
            ei = _parse_expert_idx(name)
            base_name = re.sub(r"weight\d+$", "weight", name)
            if "fc1" in name:
                expert_fc1.setdefault(base_name, {})[ei] = tensor
            else:
                expert_fc2.setdefault(base_name, {})[ei] = tensor
        else:
            non_expert[name] = tensor

    mc_agg: dict[str, torch.Tensor] = dict(non_expert)
    for base_name, experts in expert_fc1.items():
        mc_agg[base_name] = torch.stack([experts[i] for i in sorted(experts)])
    for base_name, experts in expert_fc2.items():
        mc_agg[base_name] = torch.stack([experts[i] for i in sorted(experts)])

    # Stage 2 (rank-0 only): reverse-map MC → HF via mbridge
    hf_flat: dict[str, torch.Tensor] = {}
    for name, tensor in mc_agg.items():
        if _is_mtp_param(name):
            expert = _mtp_expert_export_items(name, tensor)
            if expert is not None:
                hf_flat.update(expert)
                continue
            direct = _mtp_direct_export_items(name, tensor)
            if direct is not None:
                hf_flat.update(direct)
                continue
            transformer = _mtp_transformer_export_items(name, tensor, config, bridge)
            if transformer is None:
                log_rank0(f"[export/SKIP] no MTP mapping for {name}")
                continue
            hf_flat.update(transformer)
            continue
        mbridge_mc_name = _lite_to_mbridge_name(name, identity_map)
        if mbridge_mc_name is None:
            log_rank0(f"[export/SKIP] no mbridge mapping for {name}")
            continue
        try:
            hf_names, hf_tensors = bridge._weight_to_hf_format(mbridge_mc_name, tensor)
        except (KeyError, NotImplementedError) as e:
            log_rank0(f"[export/SKIP] _weight_to_hf_format failed for {mbridge_mc_name}: {e}")
            continue
        for hn, ht in zip(hf_names, hf_tensors, strict=True):
            hf_flat[hn] = ht

    return hf_flat


# ══════════════════════════════════════════════════════════════════════
# Save HF weights (sharded safetensors + index.json)
# ══════════════════════════════════════════════════════════════════════


def save_hf_weights(
    model: nn.Module | list[nn.Module],
    path: str,
    config: Qwen35Config,
    ps: ParallelState,
    shard_size_bytes: int = 5 * 1024**3,  # 5 GB per shard
) -> None:
    """Export weights and write sharded HF safetensors + index.json to path."""
    hf_flat = export_weights(model, config, ps)
    if not hf_flat:  # non-rank-0
        return

    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)

    keys_sorted = sorted(hf_flat.keys())

    def _tensor_bytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    # Greedy bin-pack: deterministic across runs for same model config
    shards: list[dict[str, torch.Tensor]] = [{}]
    shard_bytes = [0]
    weight_map: dict[str, str] = {}

    for k in keys_sorted:
        t = hf_flat[k]
        tb = _tensor_bytes(t)
        if shard_bytes[-1] + tb > shard_size_bytes and shard_bytes[-1] > 0:
            shards.append({})
            shard_bytes.append(0)
        shards[-1][k] = t
        shard_bytes[-1] += tb

    total = len(shards)
    total_size = sum(shard_bytes)
    for i, sh in enumerate(shards, 1):
        fname = f"model-{i:05d}-of-{total:05d}.safetensors"
        safe_save(sh, str(out / fname), metadata={"format": "pt"})
        for k in sh:
            weight_map[k] = fname

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(out / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    log_rank0(f"[save_hf_weights] {len(hf_flat)} tensors → {path} ({total} shards, {total_size/1e9:.1f} GB)")


# ══════════════════════════════════════════════════════════════════════
# Gather helpers (used by export_weights)
# ══════════════════════════════════════════════════════════════════════


def _to_global_layer_name(name: str, layer_map: dict[int, int]) -> str:
    if name.startswith("mtp.layers."):
        return name
    if not layer_map:
        return name

    def _replace(m: re.Match) -> str:
        return f"layers.{layer_map.get(int(m.group(1)), int(m.group(1)))}."
    return re.sub(r"layers\.(\d+)\.", _replace, name)


def _is_expert_weight(name: str) -> bool:
    return "experts" in name and "router" not in name and "shared" not in name


def _gather_dense_param(
    name: str, tensor: torch.Tensor, ps: ParallelState,
    config: Qwen35Config,
) -> torch.Tensor:
    t = tensor.detach()
    if ps.tp_size > 1:
        # Post-Step-3 MC naming: col-parallel (dim=0) and row-parallel (dim=1)
        tp_col_keys = [
            "linear_qkv.weight",          # full_attn QKV
            "in_proj.weight",             # GDN in_proj
            "conv1d.weight",              # GDN conv1d
            "dt_bias",                    # GDN dt_bias
            "A_log",                      # GDN A_log
            "shared_experts.linear_fc1.weight",  # shared expert gate_up
            "eh_proj.linear.weight",
            "eh_proj.weight",
        ]
        tp_row_keys = [
            "linear_proj.weight",         # full_attn output proj
            "out_proj.weight",            # GDN out_proj
            "shared_experts.linear_fc2.weight",  # shared expert down
        ]

        if any(k in name for k in tp_col_keys):
            t = _allgather_concat(t, ps.tp_size, ps.tp_group, dim=0)
        elif any(k in name for k in tp_row_keys):
            t = _allgather_concat(t, ps.tp_size, ps.tp_group, dim=1)
        elif "embed" in name or "head" in name:
            t = _allgather_concat(t, ps.tp_size, ps.tp_group, dim=0)
    return t.contiguous().cpu()


def _gather_expert_param(
    name: str,
    tensor: torch.Tensor,
    config: Qwen35Config,
    ps: ParallelState,
    out: dict[str, torch.Tensor],
) -> None:
    t = tensor.detach()
    is_fc1 = "fc1" in name

    if ps.etp_size > 1 and ps.etp_group is not None:
        if is_fc1:
            t = _gather_gate_up_expert(t, ps.etp_size, ps.etp_group)
        else:
            t = _allgather_concat(t, ps.etp_size, ps.etp_group, dim=1)

    n_local = ensure_divisible(config.num_experts, max(ps.ep_size, 1))
    local_idx = _parse_expert_idx(name)

    if ps.ep_size > 1 and ps.ep_group is not None:
        ep_gathered = [torch.empty_like(t) for _ in range(ps.ep_size)]
        dist.all_gather(ep_gathered, t.contiguous(), group=ps.ep_group)
        for ep_rank, ep_tensor in enumerate(ep_gathered):
            global_idx = ep_rank * n_local + local_idx
            out[_set_expert_idx(name, global_idx)] = ep_tensor.contiguous().cpu()
    else:
        out[name] = t.contiguous().cpu()


def _allgather_concat(
    tensor: torch.Tensor, world_size: int, group, dim: int,
) -> torch.Tensor:
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.contiguous(), group=group)
    return torch.cat(gathered, dim=dim)


def _gather_gate_up_expert(
    tensor: torch.Tensor, world_size: int, group,
) -> torch.Tensor:
    ffn_local = tensor.shape[0] // 2
    gate_local = tensor[:ffn_local]
    up_local = tensor[ffn_local:]
    gate_full = _allgather_concat(gate_local, world_size, group, dim=0)
    up_full = _allgather_concat(up_local, world_size, group, dim=0)
    return torch.cat([gate_full, up_full], dim=0)


def _parse_expert_idx(name: str) -> int:
    m = re.search(r"weight(\d+)$", name)
    return int(m.group(1)) if m else 0


def _set_expert_idx(name: str, idx: int) -> str:
    return re.sub(r"weight\d+$", f"weight{idx}", name)
