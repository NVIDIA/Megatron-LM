# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Kimi K2 lite native checkpoint mapping."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import Replicate, Shard

from megatron.lite.model.kimi_k2.config import KimiK2Config
from megatron.lite.primitive.ckpt.hf_weights import (
    SafeTensorReader,
    _cast_export_tensor,
    _resolve_export_dtype,
    parse_expert_idx,
    to_global_layer_name,
    unwrap_model,
)
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.utils import ensure_divisible, log_rank0


def EXPERT_CLASSIFIER(name: str) -> bool:
    return "experts" in name and "router" not in name and "shared" not in name


def PLACEMENT_FN(param_name: str) -> list:
    if "experts" in param_name and "router" not in param_name and "shared" not in param_name:
        if "fc1" in param_name:
            return [Replicate(), Replicate(), Shard(0), Shard(0)]
        if "fc2" in param_name:
            return [Replicate(), Replicate(), Shard(0), Shard(1)]
        return [Replicate(), Replicate(), Replicate(), Replicate()]
    if "eh_proj.linear.weight" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    if "linear_q_up_proj.linear.weight" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    if "linear_kv_up_proj.linear.weight" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    if "linear_proj.linear.weight" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(1)]
    if "gate_up" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    if "down" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(1)]
    if "embed" in param_name or "head" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    return [Replicate(), Replicate(), Replicate(), Replicate()]


def _tp(tensor: torch.Tensor, rank: int, size: int, dim: int = 0) -> torch.Tensor:
    return tensor if size <= 1 else tensor.chunk(size, dim=dim)[rank].contiguous()


def _split_gate_up(tensor: torch.Tensor, rank: int, size: int) -> torch.Tensor:
    if size <= 1:
        return tensor
    ffn = tensor.shape[0] // 2
    gate = tensor[:ffn].chunk(size, dim=0)[rank]
    up = tensor[ffn:].chunk(size, dim=0)[rank]
    return torch.cat([gate, up], dim=0).contiguous()


def _has(reader: SafeTensorReader, name: str) -> bool:
    if reader.index:
        return name in reader.index
    try:
        reader.get_tensor(name)
    except Exception:
        return False
    return True


def _dequant_fp8_weight(reader: SafeTensorReader, name: str, weight: torch.Tensor) -> torch.Tensor:
    scale_name = f"{name}_scale_inv"
    if weight.dim() != 2 or weight.element_size() != 1 or not _has(reader, scale_name):
        return weight
    scale = reader.get_tensor(scale_name).float()
    rows, cols = weight.shape
    expanded = scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    expanded = expanded[:rows, :cols]
    return weight.float() * expanded


def _unpack_int4_from_int32(packed: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if packed.dtype != torch.int32:
        raise ValueError(f"Expected packed int4 tensor to be int32, got {packed.dtype}.")
    pack_factor = 8
    mask = 0xF
    rows, cols = int(shape[0]), int(shape[1])
    unpacked = torch.empty((packed.shape[0], packed.shape[1] * pack_factor), dtype=torch.int32)
    for offset in range(pack_factor):
        unpacked[:, offset::pack_factor] = (packed >> (4 * offset)) & mask
    return (unpacked[:rows, :cols] - 8).to(torch.int8)


def _dequant_int4_weight(reader: SafeTensorReader, name: str) -> torch.Tensor:
    packed = reader.get_tensor(f"{name}_packed")
    scale = reader.get_tensor(f"{name}_scale")
    shape_tensor = reader.get_tensor(f"{name}_shape")
    shape = torch.Size(int(x) for x in shape_tensor.tolist())

    unpacked = _unpack_int4_from_int32(packed, shape).to(scale.dtype)
    if scale.dim() != 2 or unpacked.dim() != 2:
        raise ValueError(
            f"Expected groupwise int4 tensors for {name}, got "
            f"weight={tuple(unpacked.shape)} scale={tuple(scale.shape)}."
        )
    if scale.shape[0] not in (1, unpacked.shape[0]):
        raise ValueError(
            f"Unsupported int4 scale rows for {name}: "
            f"weight={tuple(unpacked.shape)} scale={tuple(scale.shape)}."
        )
    if unpacked.shape[1] % scale.shape[1] != 0:
        raise ValueError(
            f"Unsupported int4 group layout for {name}: "
            f"weight={tuple(unpacked.shape)} scale={tuple(scale.shape)}."
        )

    group_size = unpacked.shape[1] // scale.shape[1]
    return (unpacked.unflatten(-1, (scale.shape[1], group_size)) * scale.unsqueeze(-1)).flatten(
        start_dim=-2
    )


def _get(reader: SafeTensorReader, name: str) -> torch.Tensor:
    if not _has(reader, name) and _has(reader, f"{name}_packed"):
        return _dequant_int4_weight(reader, name)
    tensor = reader.get_tensor(name)
    return _dequant_fp8_weight(reader, name, tensor)


def _text_prefix(reader: SafeTensorReader) -> str:
    for prefix in ("model", "language_model.model", "model.language_model"):
        if _has(reader, f"{prefix}.embed_tokens.weight"):
            return prefix
    raise KeyError("Could not find Kimi K2 text model prefix in HF checkpoint.")


def _lm_head_name(reader: SafeTensorReader, text_prefix: str) -> str:
    candidates = [
        "lm_head.weight",
        "language_model.lm_head.weight",
        f"{text_prefix}.lm_head.weight",
    ]
    for name in candidates:
        if _has(reader, name):
            return name
    raise KeyError("Could not find Kimi K2 lm_head.weight in HF checkpoint.")


def _load_vocab(
    reader: SafeTensorReader, name: str, cfg: KimiK2Config, ps: ParallelState
) -> torch.Tensor:
    from megatron.lite.primitive.parallel import pad_vocab_for_tp

    tensor = _get(reader, name)
    padded = pad_vocab_for_tp(cfg.vocab_size, ps.tp_size)
    if tensor.size(0) < padded:
        pad = torch.zeros(padded - tensor.size(0), tensor.size(1), dtype=tensor.dtype)
        tensor = torch.cat([tensor, pad], dim=0)
    return _tp(tensor, ps.tp_rank, ps.tp_size)


def _load_attention(
    out: dict[str, torch.Tensor],
    *,
    local_prefix: str,
    hf_prefix: str,
    reader: SafeTensorReader,
    ps: ParallelState,
) -> None:
    out[f"{local_prefix}.self_attention.linear_q_down_proj.weight"] = _get(
        reader,
        f"{hf_prefix}.q_a_proj.weight",
    )
    out[f"{local_prefix}.self_attention.linear_q_up_proj.linear.layer_norm_weight"] = _get(
        reader,
        f"{hf_prefix}.q_a_layernorm.weight",
    )
    out[f"{local_prefix}.self_attention.linear_q_up_proj.linear.weight"] = _tp(
        _get(reader, f"{hf_prefix}.q_b_proj.weight"),
        ps.tp_rank,
        ps.tp_size,
    )
    out[f"{local_prefix}.self_attention.linear_kv_down_proj.weight"] = _get(
        reader,
        f"{hf_prefix}.kv_a_proj_with_mqa.weight",
    )
    out[f"{local_prefix}.self_attention.linear_kv_up_proj.linear.layer_norm_weight"] = _get(
        reader,
        f"{hf_prefix}.kv_a_layernorm.weight",
    )
    out[f"{local_prefix}.self_attention.linear_kv_up_proj.linear.weight"] = _tp(
        _get(reader, f"{hf_prefix}.kv_b_proj.weight"),
        ps.tp_rank,
        ps.tp_size,
    )
    out[f"{local_prefix}.self_attention.linear_proj.linear.weight"] = _tp(
        _get(reader, f"{hf_prefix}.o_proj.weight"),
        ps.tp_rank,
        ps.tp_size,
        dim=1,
    )


def _load_dense_mlp(
    out: dict[str, torch.Tensor],
    *,
    local_prefix: str,
    hf_mlp_prefix: str,
    hf_layer_prefix: str,
    reader: SafeTensorReader,
    ps: ParallelState,
) -> None:
    out[f"{local_prefix}.mlp.gate_up.linear.layer_norm_weight"] = _get(
        reader,
        f"{hf_layer_prefix}.post_attention_layernorm.weight",
    )
    gate_up = torch.cat(
        [
            _get(reader, f"{hf_mlp_prefix}.gate_proj.weight"),
            _get(reader, f"{hf_mlp_prefix}.up_proj.weight"),
        ],
        dim=0,
    )
    out[f"{local_prefix}.mlp.gate_up.linear.weight"] = _split_gate_up(
        gate_up,
        ps.tp_rank,
        ps.tp_size,
    )
    out[f"{local_prefix}.mlp.down.linear.weight"] = _tp(
        _get(reader, f"{hf_mlp_prefix}.down_proj.weight"),
        ps.tp_rank,
        ps.tp_size,
        dim=1,
    )


def _load_shared_expert(
    out: dict[str, torch.Tensor],
    *,
    local_prefix: str,
    hf_mlp_prefix: str,
    reader: SafeTensorReader,
    ps: ParallelState,
) -> None:
    prefixes = [f"{hf_mlp_prefix}.shared_experts", f"{hf_mlp_prefix}.shared_expert"]
    shared = next(prefix for prefix in prefixes if _has(reader, f"{prefix}.down_proj.weight"))
    gate_up = torch.cat(
        [
            _get(reader, f"{shared}.gate_proj.weight"),
            _get(reader, f"{shared}.up_proj.weight"),
        ],
        dim=0,
    )
    out[f"{local_prefix}.moe.shared_expert.gate_up.linear.weight"] = _split_gate_up(
        gate_up,
        ps.tp_rank,
        ps.tp_size,
    )
    out[f"{local_prefix}.moe.shared_expert.down.linear.weight"] = _tp(
        _get(reader, f"{shared}.down_proj.weight"),
        ps.tp_rank,
        ps.tp_size,
        dim=1,
    )


def _load_experts(
    out: dict[str, torch.Tensor],
    *,
    local_prefix: str,
    hf_mlp_prefix: str,
    cfg: KimiK2Config,
    ps: ParallelState,
    reader: SafeTensorReader,
) -> None:
    num_local = ensure_divisible(cfg.num_experts, ps.ep_size)
    local_start = ps.ep_rank * num_local
    for local_idx in range(num_local):
        global_idx = local_start + local_idx
        ep = f"{hf_mlp_prefix}.experts.{global_idx}"
        fc1 = torch.cat(
            [
                _get(reader, f"{ep}.gate_proj.weight"),
                _get(reader, f"{ep}.up_proj.weight"),
            ],
            dim=0,
        )
        fc2 = _get(reader, f"{ep}.down_proj.weight")
        if ps.etp_size > 1:
            fc1 = _split_gate_up(fc1, ps.etp_rank, ps.etp_size)
            fc2 = _tp(fc2, ps.etp_rank, ps.etp_size, dim=1)
        out[f"{local_prefix}.moe.experts.fc1.weight{local_idx}"] = fc1
        out[f"{local_prefix}.moe.experts.fc2.weight{local_idx}"] = fc2


def _copy_loaded_state(model: nn.Module, loaded: dict[str, torch.Tensor]) -> None:
    state = model.state_dict()
    resolved: dict[str, torch.Tensor] = {}
    for name, tensor in loaded.items():
        actual = name if name in state else None
        if actual is None:
            for key in state:
                if name in key:
                    actual = key
                    break
        if actual is not None:
            resolved[actual] = tensor
        else:
            log_rank0(f"WARNING: kimi_k2 checkpoint tensor has no target param: {name}")

    for name, target in model.named_parameters():
        if name not in resolved:
            log_rank0(f"WARNING: {name} not loaded from checkpoint")
            continue
        tensor = resolved[name].to(device=target.device)
        target.data.copy_(tensor.to(dtype=target.dtype))

    for name, target in model.named_buffers():
        if name not in resolved:
            continue
        tensor = resolved[name].to(device=target.device)
        target.data.copy_(tensor.to(dtype=target.dtype) if target.is_floating_point() else tensor)


class KimiK2WeightSpec:
    """Export Kimi K2 lite weights to HF DeepSeekV3/Kimi-style names."""

    def __init__(self, config: KimiK2Config):
        self.config = config

    @property
    def num_experts(self) -> int:
        return self.config.num_experts

    def weight_map(self) -> dict[str, list[str]]:
        return {}

    def hf_to_native(self, native_name: str, hf_tensors: list[torch.Tensor]) -> torch.Tensor:
        del native_name
        return hf_tensors[0]

    def native_to_hf(
        self, native_name: str, tensor: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        if native_name == "mtp_embed.embedding.weight":
            return []
        if native_name.startswith("mtp.layers."):
            parts = native_name.split(".")
            mtp_idx = int(parts[2])
            hf_layer_idx = self.config.num_hidden_layers + mtp_idx
            hp = f"model.layers.{hf_layer_idx}"
            if native_name.endswith(".enorm.weight"):
                return [(f"{hp}.enorm.weight", tensor)]
            if native_name.endswith(".hnorm.weight"):
                return [(f"{hp}.hnorm.weight", tensor)]
            if native_name.endswith(".eh_proj.linear.weight"):
                return [(f"{hp}.eh_proj.weight", tensor)]
            if native_name.endswith(".final_layernorm.weight"):
                return [(f"{hp}.shared_head.norm.weight", tensor)]
            proxy = native_name.replace(
                f"mtp.layers.{mtp_idx}.transformer_layer",
                f"layers.{hf_layer_idx}",
            )
            return self.native_to_hf(proxy, tensor)
        if native_name == "embed.embedding.weight":
            return [("model.embed_tokens.weight", tensor)]
        if native_name == "norm.weight":
            return [("model.norm.weight", tensor)]
        if native_name == "head.col.linear.weight":
            return [("lm_head.weight", tensor)]

        parts = native_name.split(".")
        if len(parts) < 3 or parts[0] != "layers":
            return []
        layer_idx = int(parts[1])
        suffix = ".".join(parts[2:])
        hp = f"model.layers.{layer_idx}"
        ap = f"{hp}.self_attn"
        mp = f"{hp}.mlp"

        if suffix == "input_layernorm.weight":
            return [(f"{hp}.input_layernorm.weight", tensor)]
        if suffix == "self_attention.linear_q_down_proj.weight":
            return [(f"{ap}.q_a_proj.weight", tensor)]
        if suffix == "self_attention.linear_q_up_proj.linear.layer_norm_weight":
            return [(f"{ap}.q_a_layernorm.weight", tensor)]
        if suffix == "self_attention.linear_q_up_proj.linear.weight":
            return [(f"{ap}.q_b_proj.weight", tensor)]
        if suffix == "self_attention.linear_kv_down_proj.weight":
            return [(f"{ap}.kv_a_proj_with_mqa.weight", tensor)]
        if suffix == "self_attention.linear_kv_up_proj.linear.layer_norm_weight":
            return [(f"{ap}.kv_a_layernorm.weight", tensor)]
        if suffix == "self_attention.linear_kv_up_proj.linear.weight":
            return [(f"{ap}.kv_b_proj.weight", tensor)]
        if suffix == "self_attention.linear_proj.linear.weight":
            return [(f"{ap}.o_proj.weight", tensor)]

        if suffix == "mlp.gate_up.linear.layer_norm_weight":
            return [(f"{hp}.post_attention_layernorm.weight", tensor)]
        if suffix == "mlp.gate_up.linear.weight":
            gate, up = tensor.chunk(2, dim=0)
            return [
                (f"{mp}.gate_proj.weight", gate.contiguous()),
                (f"{mp}.up_proj.weight", up.contiguous()),
            ]
        if suffix == "mlp.down.linear.weight":
            return [(f"{mp}.down_proj.weight", tensor)]

        if suffix == "mlp_norm.weight":
            return [(f"{hp}.post_attention_layernorm.weight", tensor)]
        if suffix == "moe.router.gate.weight":
            return [(f"{mp}.gate.weight", tensor)]
        if suffix == "moe.router.expert_bias":
            return [(f"{mp}.gate.e_score_correction_bias", tensor.float())]
        if suffix == "moe.shared_expert.gate_up.linear.weight":
            gate, up = tensor.chunk(2, dim=0)
            return [
                (f"{mp}.shared_experts.gate_proj.weight", gate.contiguous()),
                (f"{mp}.shared_experts.up_proj.weight", up.contiguous()),
            ]
        if suffix == "moe.shared_expert.down.linear.weight":
            return [(f"{mp}.shared_experts.down_proj.weight", tensor)]

        if ".moe.experts.fc1.weight" in native_name:
            expert_idx = parse_expert_idx(native_name)
            gate, up = tensor.chunk(2, dim=0)
            return [
                (f"{mp}.experts.{expert_idx}.gate_proj.weight", gate.contiguous()),
                (f"{mp}.experts.{expert_idx}.up_proj.weight", up.contiguous()),
            ]
        if ".moe.experts.fc2.weight" in native_name:
            expert_idx = parse_expert_idx(native_name)
            return [(f"{mp}.experts.{expert_idx}.down_proj.weight", tensor)]

        return []

    def qkv_spec(self, native_name: str) -> tuple[int, int, int] | None:
        del native_name
        return None

    def tp_spec(self, native_name: str) -> tuple[int, int] | None:
        if native_name.startswith("mtp.layers.") and ".transformer_layer." in native_name:
            proxy = native_name.replace(".transformer_layer.", ".")
            return self.tp_spec(proxy)
        if native_name.endswith(".eh_proj.linear.weight"):
            return (0, 0)
        if self.is_expert(native_name):
            if ".fc1." in native_name:
                return (0, 1)
            if ".fc2." in native_name:
                return (1, 1)
            return None
        if native_name in {"embed.embedding.weight", "head.col.linear.weight"}:
            return (0, 0)
        if native_name.endswith(".self_attention.linear_q_up_proj.linear.weight"):
            return (0, 0)
        if native_name.endswith(".self_attention.linear_kv_up_proj.linear.weight"):
            return (0, 0)
        if native_name.endswith(".self_attention.linear_proj.linear.weight"):
            return (1, 0)
        if native_name.endswith(".mlp.gate_up.linear.weight"):
            return (0, 0)
        if native_name.endswith(".mlp.down.linear.weight"):
            return (1, 0)
        if native_name.endswith(".moe.shared_expert.gate_up.linear.weight"):
            return (0, 0)
        if native_name.endswith(".moe.shared_expert.down.linear.weight"):
            return (1, 0)
        return None

    def is_expert(self, native_name: str) -> bool:
        return ".moe.experts." in native_name and ".router." not in native_name

    def expert_global_id(self, native_name: str) -> int | None:
        if self.is_expert(native_name):
            return parse_expert_idx(native_name)
        return None

    def expert_local_name(self, native_name: str, local_idx: int) -> str:
        prefix = native_name.rsplit(".weight", 1)[0]
        return f"{prefix}.weight{local_idx}"


def load_hf_weights(model: nn.Module, path: str, config: KimiK2Config, ps: ParallelState) -> None:
    base_model = unwrap_model(model)
    reader = SafeTensorReader(path)
    out: dict[str, torch.Tensor] = {}

    prefix = _text_prefix(reader)

    if getattr(base_model, "embed", None) is not None:
        out["embed.embedding.weight"] = _load_vocab(
            reader, f"{prefix}.embed_tokens.weight", config, ps
        )
    if getattr(base_model, "mtp_embed", None) is not None:
        out["mtp_embed.embedding.weight"] = _load_vocab(
            reader,
            f"{prefix}.embed_tokens.weight",
            config,
            ps,
        )
    if getattr(base_model, "norm", None) is not None:
        out["norm.weight"] = _get(reader, f"{prefix}.norm.weight")
    if getattr(base_model, "head", None) is not None:
        out["head.col.linear.weight"] = _load_vocab(
            reader, _lm_head_name(reader, prefix), config, ps
        )

    for local_idx, global_idx in enumerate(base_model.layer_indices):
        lp = f"layers.{local_idx}"
        hp = f"{prefix}.layers.{global_idx}"
        out[f"{lp}.input_layernorm.weight"] = _get(reader, f"{hp}.input_layernorm.weight")
        _load_attention(
            out,
            local_prefix=lp,
            hf_prefix=f"{hp}.self_attn",
            reader=reader,
            ps=ps,
        )
        if config.is_moe_layer(global_idx):
            out[f"{lp}.mlp_norm.weight"] = _get(reader, f"{hp}.post_attention_layernorm.weight")
            out[f"{lp}.moe.router.gate.weight"] = _get(reader, f"{hp}.mlp.gate.weight")
            bias_name = f"{hp}.mlp.gate.e_score_correction_bias"
            if _has(reader, bias_name):
                out[f"{lp}.moe.router.expert_bias"] = _get(reader, bias_name).float()
            _load_shared_expert(
                out, local_prefix=lp, hf_mlp_prefix=f"{hp}.mlp", reader=reader, ps=ps
            )
            _load_experts(
                out,
                local_prefix=lp,
                hf_mlp_prefix=f"{hp}.mlp",
                cfg=config,
                ps=ps,
                reader=reader,
            )
        else:
            _load_dense_mlp(
                out,
                local_prefix=lp,
                hf_mlp_prefix=f"{hp}.mlp",
                hf_layer_prefix=hp,
                reader=reader,
                ps=ps,
            )

    mtp = getattr(base_model, "mtp", None)
    if mtp is not None:
        for local_idx, _mtp_layer in enumerate(mtp.layers):
            global_idx = config.num_hidden_layers + local_idx
            lp = f"mtp.layers.{local_idx}"
            hp = f"{prefix}.layers.{global_idx}"
            tlp = f"{lp}.transformer_layer"
            out[f"{lp}.enorm.weight"] = _get(reader, f"{hp}.enorm.weight")
            out[f"{lp}.hnorm.weight"] = _get(reader, f"{hp}.hnorm.weight")
            out[f"{lp}.eh_proj.linear.weight"] = _tp(
                _get(reader, f"{hp}.eh_proj.weight"),
                ps.tp_rank,
                ps.tp_size,
            )
            shared_head_norm = f"{hp}.shared_head.norm.weight"
            final_norm = (
                shared_head_norm
                if _has(reader, shared_head_norm)
                else f"{hp}.final_layernorm.weight"
            )
            out[f"{lp}.final_layernorm.weight"] = _get(reader, final_norm)
            out[f"{tlp}.input_layernorm.weight"] = _get(reader, f"{hp}.input_layernorm.weight")
            _load_attention(
                out,
                local_prefix=tlp,
                hf_prefix=f"{hp}.self_attn",
                reader=reader,
                ps=ps,
            )
            if config.is_moe_layer(global_idx):
                out[f"{tlp}.mlp_norm.weight"] = _get(
                    reader, f"{hp}.post_attention_layernorm.weight"
                )
                out[f"{tlp}.moe.router.gate.weight"] = _get(reader, f"{hp}.mlp.gate.weight")
                bias_name = f"{hp}.mlp.gate.e_score_correction_bias"
                if _has(reader, bias_name):
                    out[f"{tlp}.moe.router.expert_bias"] = _get(reader, bias_name).float()
                _load_shared_expert(
                    out,
                    local_prefix=tlp,
                    hf_mlp_prefix=f"{hp}.mlp",
                    reader=reader,
                    ps=ps,
                )
                _load_experts(
                    out,
                    local_prefix=tlp,
                    hf_mlp_prefix=f"{hp}.mlp",
                    cfg=config,
                    ps=ps,
                    reader=reader,
                )
            else:
                _load_dense_mlp(
                    out,
                    local_prefix=tlp,
                    hf_mlp_prefix=f"{hp}.mlp",
                    hf_layer_prefix=hp,
                    reader=reader,
                    ps=ps,
                )

    _copy_loaded_state(base_model, out)


def export_hf_weights(model, config: KimiK2Config, ps: ParallelState, **kwargs):
    from megatron.lite.primitive.ckpt.hf_weights import export_hf_weights as _export

    spec = KimiK2WeightSpec(config)
    rank0_only = bool(kwargs.get("rank0_only", False))
    export_dtype = _resolve_export_dtype(kwargs.get("export_dtype"))
    yield from _export(model, spec, ps, vocab_size=config.vocab_size, **kwargs)
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank0_only and rank != 0:
        return
    chunks = list(model) if isinstance(model, list | nn.ModuleList) else [model]
    for chunk in chunks:
        base_chunk = unwrap_model(chunk)
        layer_map = (
            {i: base_chunk.layer_indices[i] for i in range(len(base_chunk.layer_indices))}
            if hasattr(base_chunk, "layer_indices")
            else {}
        )
        for name, buffer in base_chunk.named_buffers():
            if not name.endswith(".moe.router.expert_bias"):
                continue
            global_name = to_global_layer_name(name, layer_map)
            for hf_name, hf_tensor in spec.native_to_hf(global_name, buffer.detach().cpu()):
                yield hf_name, _cast_export_tensor(hf_tensor, export_dtype)


def save_hf_weights(model, path: str, config: KimiK2Config, ps: ParallelState) -> None:
    from megatron.lite.primitive.ckpt.hf_weights import save_safetensors

    rank = dist.get_rank() if dist.is_initialized() else 0
    out = dict(export_hf_weights(model, config, ps, rank0_only=True))
    if rank == 0 and out:
        save_safetensors(out, path)
    if dist.is_initialized():
        dist.barrier()


__all__ = [
    "EXPERT_CLASSIFIER",
    "KimiK2WeightSpec",
    "PLACEMENT_FN",
    "_dequant_int4_weight",
    "_dequant_fp8_weight",
    "export_hf_weights",
    "load_hf_weights",
    "save_hf_weights",
]
