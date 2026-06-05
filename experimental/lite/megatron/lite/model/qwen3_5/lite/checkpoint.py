"""Qwen3.5 lite native checkpoint mapping.

The loader reads HF safetensors directly into lite's native module names.
It intentionally does not require wrapper-specific state on the model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributed.tensor import Replicate, Shard

from megatron.lite.model.qwen3_5.config import Qwen35Config
from megatron.lite.primitive.ckpt.hf_bridge import SafeTensorReader, unwrap_model
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
    if "in_proj" in param_name and "layer_norm" not in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    if "qkv" in param_name and "layer_norm" not in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    if ("proj" in param_name or "o_proj" in param_name) and (
        "full_attn" in param_name or "linear_attn" in param_name
    ):
        return [Replicate(), Replicate(), Replicate(), Shard(1)]
    if "gate_up" in param_name and "shared" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    if "down" in param_name and "shared" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(1)]
    if "embed" in param_name or "head" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    if "conv1d" in param_name or "dt_bias" in param_name or "A_log" in param_name:
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


def _zero_centered_gamma_from_hf(tensor: torch.Tensor) -> torch.Tensor:
    return tensor - 1


def _get(reader: SafeTensorReader, name: str) -> torch.Tensor:
    return reader.get_tensor(name)


def _has(reader: SafeTensorReader, name: str) -> bool:
    if reader.index:
        return name in reader.index
    try:
        reader.get_tensor(name)
    except Exception:
        return False
    return True


def _load_vocab(reader: SafeTensorReader, name: str, cfg: Qwen35Config, ps: ParallelState) -> torch.Tensor:
    from megatron.lite.primitive.parallel import pad_vocab_for_tp

    tensor = _get(reader, name)
    padded = pad_vocab_for_tp(cfg.vocab_size, ps.tp_size)
    if tensor.size(0) < padded:
        pad = torch.zeros(padded - tensor.size(0), tensor.size(1), dtype=tensor.dtype)
        tensor = torch.cat([tensor, pad], dim=0)
    return _tp(tensor, ps.tp_rank, ps.tp_size)


def _load_full_attn(
    out: dict[str, torch.Tensor],
    *,
    local_prefix: str,
    hf_prefix: str,
    input_ln: torch.Tensor,
    cfg: Qwen35Config,
    ps: ParallelState,
    reader: SafeTensorReader,
) -> None:
    out[f"{local_prefix}.full_attn.qkv.linear.layer_norm_weight"] = input_ln
    q = _get(reader, f"{hf_prefix}.q_proj.weight")
    k = _get(reader, f"{hf_prefix}.k_proj.weight")
    v = _get(reader, f"{hf_prefix}.v_proj.weight")
    out[f"{local_prefix}.full_attn.qkv.linear.weight"] = _tp(
        _merge_full_attn_qkvg(q, k, v, cfg=cfg),
        ps.tp_rank,
        ps.tp_size,
    )
    out[f"{local_prefix}.full_attn.q_norm.weight"] = _get(reader, f"{hf_prefix}.q_norm.weight")
    out[f"{local_prefix}.full_attn.k_norm.weight"] = _get(reader, f"{hf_prefix}.k_norm.weight")
    out[f"{local_prefix}.full_attn.proj.linear.weight"] = _tp(
        _get(reader, f"{hf_prefix}.o_proj.weight"),
        ps.tp_rank,
        ps.tp_size,
        dim=1,
    )


def _merge_full_attn_qkvg(
    q_gate: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cfg: Qwen35Config,
) -> torch.Tensor:
    kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    hidden = q_gate.shape[1]
    q_gate = q_gate.reshape(cfg.num_attention_heads, 2 * head_dim, hidden)
    query = q_gate.narrow(1, 0, head_dim).reshape(cfg.num_attention_heads * head_dim, hidden)
    gate = q_gate.narrow(1, head_dim, head_dim).reshape(cfg.num_attention_heads * head_dim, hidden)
    q_heads_per_group = ensure_divisible(cfg.num_attention_heads, cfg.num_key_value_heads)
    q_group_width = q_heads_per_group * head_dim
    query = query.reshape(kv_heads, q_group_width, hidden)
    gate = gate.reshape(kv_heads, q_group_width, hidden)
    key = key.reshape(kv_heads, head_dim, hidden)
    value = value.reshape(kv_heads, head_dim, hidden)
    return torch.cat([query, gate, key, value], dim=1).reshape(-1, hidden).contiguous()


def _load_linear_attn(
    out: dict[str, torch.Tensor],
    *,
    local_prefix: str,
    hf_prefix: str,
    input_ln: torch.Tensor,
    cfg: Qwen35Config,
    ps: ParallelState,
    reader: SafeTensorReader,
) -> None:
    dk, dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    nk, nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
    qk_dim, v_dim = nk * dk, nv * dv
    qkv = _get(reader, f"{hf_prefix}.in_proj_qkv.weight")
    q, k, v = qkv.split([qk_dim, qk_dim, v_dim], dim=0)
    z = _get(reader, f"{hf_prefix}.in_proj_z.weight")
    b = _get(reader, f"{hf_prefix}.in_proj_b.weight")
    a = _get(reader, f"{hf_prefix}.in_proj_a.weight")

    # lite GDN splits by contiguous Q/K/V/Z/B/A sections.
    qkvzba = torch.cat([q, k, v, z, b, a], dim=0)
    out[f"{local_prefix}.linear_attn.in_proj.linear.weight"] = _tp(
        qkvzba,
        ps.tp_rank,
        ps.tp_size,
        dim=0,
    )
    out[f"{local_prefix}.linear_attn.in_proj.linear.layer_norm_weight"] = input_ln
    out[f"{local_prefix}.linear_attn.conv1d.weight"] = _tp(
        _get(reader, f"{hf_prefix}.conv1d.weight"),
        ps.tp_rank,
        ps.tp_size,
    )
    out[f"{local_prefix}.linear_attn.dt_bias"] = _tp(_get(reader, f"{hf_prefix}.dt_bias"), ps.tp_rank, ps.tp_size)
    out[f"{local_prefix}.linear_attn.A_log"] = _tp(_get(reader, f"{hf_prefix}.A_log"), ps.tp_rank, ps.tp_size)
    out[f"{local_prefix}.linear_attn.norm.weight"] = _zero_centered_gamma_from_hf(
        _get(reader, f"{hf_prefix}.norm.weight")
    )
    out[f"{local_prefix}.linear_attn.o_proj.linear.weight"] = _tp(
        _get(reader, f"{hf_prefix}.out_proj.weight"),
        ps.tp_rank,
        ps.tp_size,
        dim=1,
    )


def _load_shared_expert(
    out: dict[str, torch.Tensor],
    *,
    local_prefix: str,
    hf_mlp_prefix: str,
    ps: ParallelState,
    reader: SafeTensorReader,
) -> None:
    shared = f"{hf_mlp_prefix}.shared_expert"
    gate_up = torch.cat(
        [
            _get(reader, f"{shared}.gate_proj.weight"),
            _get(reader, f"{shared}.up_proj.weight"),
        ],
        dim=0,
    )
    out[f"{local_prefix}.moe.shared_expert.gate_up.linear.weight"] = _tp(
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
    out[f"{local_prefix}.moe.shared_expert.shared_gate.weight"] = _get(
        reader,
        f"{hf_mlp_prefix}.shared_expert_gate.weight",
    )


def _load_experts(
    out: dict[str, torch.Tensor],
    *,
    local_prefix: str,
    hf_mlp_prefix: str,
    cfg: Qwen35Config,
    ps: ParallelState,
    reader: SafeTensorReader,
) -> None:
    num_local = ensure_divisible(cfg.num_experts, ps.ep_size)
    local_start = ps.ep_rank * num_local
    packed_gate_up = f"{hf_mlp_prefix}.experts.gate_up_proj"
    packed_down = f"{hf_mlp_prefix}.experts.down_proj"
    if _has(reader, packed_gate_up) and _has(reader, packed_down):
        gate_up_all = _get(reader, packed_gate_up)
        down_all = _get(reader, packed_down)
        for local_idx in range(num_local):
            global_idx = local_start + local_idx
            fc1 = gate_up_all[global_idx]
            fc2 = down_all[global_idx]
            if ps.etp_size > 1:
                fc1 = _split_gate_up(fc1, ps.etp_rank, ps.etp_size)
                fc2 = _tp(fc2, ps.etp_rank, ps.etp_size, dim=1)
            out[f"{local_prefix}.moe.experts.fc1.weight{local_idx}"] = fc1
            out[f"{local_prefix}.moe.experts.fc2.weight{local_idx}"] = fc2
        return

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
            log_rank0(f"WARNING: lite checkpoint tensor has no target param: {name}")

    fp32_names = ("A_log", "dt_bias")
    for name, param in model.named_parameters():
        if name not in resolved:
            log_rank0(f"WARNING: {name} not loaded from checkpoint")
            continue
        tensor = resolved[name].to(device=param.device)
        if any(k in name for k in fp32_names):
            param.data.copy_(tensor.float())
        else:
            param.data.copy_(tensor.to(dtype=param.dtype))


def load_hf_weights(model: nn.Module, path: str, config: Qwen35Config, ps: ParallelState) -> None:
    base_model = unwrap_model(model)
    reader = SafeTensorReader(path)
    out: dict[str, torch.Tensor] = {}

    prefix = "model.language_model"
    if getattr(base_model, "embed", None) is not None:
        out["embed.embedding.weight"] = _load_vocab(reader, f"{prefix}.embed_tokens.weight", config, ps)
    if getattr(base_model, "norm", None) is not None:
        out["norm.weight"] = _get(reader, f"{prefix}.norm.weight")
    if getattr(base_model, "head", None) is not None:
        out["head.col.linear.weight"] = _load_vocab(reader, "lm_head.weight", config, ps)

    for local_idx, global_idx in enumerate(base_model.layer_indices):
        lp = f"layers.{local_idx}"
        hp = f"{prefix}.layers.{global_idx}"
        input_ln = _get(reader, f"{hp}.input_layernorm.weight")
        if config.layer_type_at(global_idx) == "full_attention":
            _load_full_attn(
                out,
                local_prefix=lp,
                hf_prefix=f"{hp}.self_attn",
                input_ln=input_ln,
                cfg=config,
                ps=ps,
                reader=reader,
            )
        else:
            _load_linear_attn(
                out,
                local_prefix=lp,
                hf_prefix=f"{hp}.linear_attn",
                input_ln=input_ln,
                cfg=config,
                ps=ps,
                reader=reader,
            )
        out[f"{lp}.mlp_norm.weight"] = _get(reader, f"{hp}.post_attention_layernorm.weight")
        out[f"{lp}.moe.router.gate.weight"] = _get(reader, f"{hp}.mlp.gate.weight")[: config.num_experts]
        _load_shared_expert(out, local_prefix=lp, hf_mlp_prefix=f"{hp}.mlp", ps=ps, reader=reader)
        _load_experts(out, local_prefix=lp, hf_mlp_prefix=f"{hp}.mlp", cfg=config, ps=ps, reader=reader)

    _copy_loaded_state(base_model, out)

__all__ = [
    "EXPERT_CLASSIFIER",
    "PLACEMENT_FN",
    "load_hf_weights",
]
