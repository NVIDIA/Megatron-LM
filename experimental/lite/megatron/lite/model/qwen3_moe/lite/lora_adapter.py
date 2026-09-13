# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""PEFT adapter import/export for Qwen3-MoE lite native LoRA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
from megatron.lite.primitive.modules.lora import LoraConfig, normalize_lora_config
from megatron.lite.primitive.parallel import ParallelState

_PEFT_PREFIX = "base_model.model.model"


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _world_size(group=None) -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size(group)


def _unwrap_model(module: nn.Module) -> nn.Module:
    current = module
    seen: set[int] = set()
    while hasattr(current, "module") and id(current) not in seen:
        seen.add(id(current))
        inner = getattr(current, "module")
        if isinstance(inner, list):
            if len(inner) != 1:
                break
            inner = inner[0]
        if inner is current or not isinstance(inner, nn.Module):
            break
        current = inner
    return current


def _iter_qwen_chunks(chunks: list[nn.Module] | tuple[nn.Module, ...]) -> list[nn.Module]:
    return [_unwrap_model(chunk) for chunk in chunks]


def _all_gather_cat(tensor: torch.Tensor, group, dim: int) -> torch.Tensor:
    if _world_size(group) == 1:
        return tensor.contiguous()
    gathered = [torch.empty_like(tensor) for _ in range(_world_size(group))]
    dist.all_gather(gathered, tensor.contiguous(), group=group)
    return torch.cat(gathered, dim=dim).contiguous()


def _select_tp_replicated(tensor: torch.Tensor, ps: ParallelState) -> torch.Tensor:
    if _world_size(ps.tp_group) == 1:
        return tensor.contiguous()
    gathered = [torch.empty_like(tensor) for _ in range(_world_size(ps.tp_group))]
    dist.all_gather(gathered, tensor.contiguous(), group=ps.tp_group)
    return gathered[0].contiguous()


def _is_rank_partitioned_lora_a(lora: Any, ps: ParallelState) -> bool:
    if getattr(lora, "rank_partitioned_a", False):
        return True
    rank = int(getattr(lora, "rank", lora.lora_b.shape[1]))
    return ps.tp_size > 1 and lora.lora_a.shape[0] != rank


def _gather_lora_rank_partition(tensor: torch.Tensor, ps: ParallelState) -> torch.Tensor:
    return _all_gather_cat(tensor, ps.tp_group, dim=0)


def _slice_lora_rank_partition(tensor: torch.Tensor, ps: ParallelState) -> torch.Tensor:
    if ps.tp_size == 1:
        return tensor.contiguous()
    if tensor.shape[0] % ps.tp_size != 0:
        raise ValueError(f"Cannot shard LoRA rank dim {tensor.shape[0]} over TP={ps.tp_size}.")
    local_rank = tensor.shape[0] // ps.tp_size
    start = ps.tp_rank * local_rank
    return tensor[start : start + local_rank].contiguous()


def _is_output_partitioned_lora_b(lora: Any, ps: ParallelState) -> bool:
    if getattr(lora, "output_partitioned_b", False):
        return True
    return ps.tp_size > 1 and lora.lora_b.shape[0] * ps.tp_size == getattr(lora, "out_features", -1)


def _expert_lora_is_shared(lora: Any) -> bool:
    return bool(getattr(lora, "shared_across_experts", False)) or lora.lora_a.dim() == 2


def _expand_shared_expert_lora(
    lora: Any, num_local_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if _expert_lora_is_shared(lora):
        return (
            lora.lora_a.detach().unsqueeze(0).expand(num_local_experts, -1, -1).contiguous(),
            lora.lora_b.detach().unsqueeze(0).expand(num_local_experts, -1, -1).contiguous(),
        )
    return lora.lora_a.detach(), lora.lora_b.detach()


def _split_local_mcore_qkv_b(
    qkv_b: torch.Tensor, *, num_heads_local: int, num_kv_heads_local: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_per_group = num_heads_local // num_kv_heads_local
    group_width = (q_per_group + 2) * head_dim
    packed = qkv_b.view(num_kv_heads_local, group_width, -1)
    q_end = q_per_group * head_dim
    k_end = q_end + head_dim
    q = packed[:, :q_end].reshape(num_heads_local * head_dim, -1)
    k = packed[:, q_end:k_end].reshape(num_kv_heads_local * head_dim, -1)
    v = packed[:, k_end:].reshape(num_kv_heads_local * head_dim, -1)
    return q.contiguous(), k.contiguous(), v.contiguous()


def _pack_local_mcore_qkv_b(
    q_b: torch.Tensor,
    k_b: torch.Tensor,
    v_b: torch.Tensor,
    *,
    num_heads_local: int,
    num_kv_heads_local: int,
    head_dim: int,
) -> torch.Tensor:
    q_per_group = num_heads_local // num_kv_heads_local
    q = q_b.view(num_kv_heads_local, q_per_group * head_dim, -1)
    k = k_b.view(num_kv_heads_local, head_dim, -1)
    v = v_b.view(num_kv_heads_local, head_dim, -1)
    return torch.cat([q, k, v], dim=1).reshape(-1, q_b.shape[-1]).contiguous()


def _layer_prefix(layer_idx: int) -> str:
    return f"{_PEFT_PREFIX}.layers.{layer_idx}"


def _attn_key(layer_idx: int, module: str, suffix: str) -> str:
    return f"{_layer_prefix(layer_idx)}.self_attn.{module}.{suffix}.weight"


def _expert_key(layer_idx: int, expert_idx: int, module: str, suffix: str) -> str:
    return f"{_layer_prefix(layer_idx)}.mlp.experts.{expert_idx}.{module}.{suffix}.weight"


def _target_modules_from_lora_config(lora_config: LoraConfig) -> list[str]:
    targets = lora_config.targets()
    out: list[str] = []
    if "linear_qkv" in targets:
        out += ["q_proj", "k_proj", "v_proj"]
    if "linear_proj" in targets:
        out.append("o_proj")
    if "linear_fc1" in targets:
        out += ["gate_proj", "up_proj"]
    if "linear_fc2" in targets:
        out.append("down_proj")
    return out


def _state_target_modules(state: dict[str, torch.Tensor]) -> set[str]:
    out: set[str] = set()
    for key in state:
        if ".q_proj." in key:
            out.add("q_proj")
        elif ".k_proj." in key:
            out.add("k_proj")
        elif ".v_proj." in key:
            out.add("v_proj")
        elif ".o_proj." in key:
            out.add("o_proj")
        elif ".gate_proj." in key:
            out.add("gate_proj")
        elif ".up_proj." in key:
            out.add("up_proj")
        elif ".down_proj." in key:
            out.add("down_proj")
    return out


def _effective_lora_alpha(lora_config: LoraConfig) -> int:
    return lora_config.rank if lora_config.alpha is None else int(lora_config.alpha)


def _peft_target_set(value: Any) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    return {str(item) for item in value}


def _infer_state_rank(state: dict[str, torch.Tensor]) -> int | None:
    ranks = {
        int(tensor.shape[0])
        for key, tensor in state.items()
        if key.endswith(".lora_A.weight") and tensor.ndim >= 2
    }
    if not ranks:
        return None
    if len(ranks) != 1:
        raise ValueError(f"Adapter contains inconsistent LoRA ranks: {sorted(ranks)}.")
    return next(iter(ranks))


def _iter_native_lora_modules(chunks: list[nn.Module] | tuple[nn.Module, ...]):
    for chunk in _iter_qwen_chunks(list(chunks)):
        for layer in chunk.layers:
            attn = layer.attn
            if attn.qkv_lora is not None:
                yield attn.qkv_lora
            if attn.proj_lora is not None:
                yield attn.proj_lora
            experts = layer.moe.experts
            if experts.fc1_lora is not None:
                yield experts.fc1_lora
            if experts.fc2_lora is not None:
                yield experts.fc2_lora


def _infer_native_alpha(chunks: list[nn.Module] | tuple[nn.Module, ...]) -> int | None:
    alphas: set[int] = set()
    for module in _iter_native_lora_modules(chunks):
        rank = getattr(module, "rank", None)
        scale = getattr(module, "scale", None)
        if rank is None or scale is None:
            continue
        alphas.add(int(round(float(scale) * int(rank))))
    if not alphas:
        return None
    if len(alphas) != 1:
        raise ValueError(
            f"Native LoRA modules have inconsistent effective alpha values: {sorted(alphas)}."
        )
    return next(iter(alphas))


def _validate_adapter_config(
    chunks: list[nn.Module] | tuple[nn.Module, ...],
    state: dict[str, torch.Tensor],
    adapter_config: dict[str, Any],
    *,
    lora_config: LoraConfig | dict[str, Any] | None = None,
) -> None:
    peft_type = adapter_config.get("peft_type")
    if peft_type is not None and str(peft_type).upper() != "LORA":
        raise ValueError(f"Expected PEFT adapter_config peft_type='LORA', got {peft_type!r}.")

    state_rank = _infer_state_rank(state)
    config_rank = adapter_config.get("r")
    if config_rank is not None and state_rank is not None and int(config_rank) != state_rank:
        raise ValueError(
            f"Adapter config rank r={config_rank} does not match tensor rank {state_rank}."
        )

    config_targets = _peft_target_set(adapter_config.get("target_modules"))
    state_targets = _state_target_modules(state)
    if config_targets is not None and config_targets != state_targets:
        raise ValueError(
            "Adapter config target_modules do not match adapter tensors: "
            f"config={sorted(config_targets)}, tensors={sorted(state_targets)}."
        )

    config_alpha = adapter_config.get("lora_alpha")
    native_alpha = _infer_native_alpha(chunks)
    if config_alpha is not None and native_alpha is not None and int(config_alpha) != native_alpha:
        raise ValueError(
            f"Adapter config lora_alpha={config_alpha} does not match native model alpha={native_alpha}."
        )

    if lora_config is not None:
        expected = normalize_lora_config(lora_config)
        expected_targets = set(_target_modules_from_lora_config(expected))
        if config_rank is not None and int(config_rank) != expected.rank:
            raise ValueError(
                f"Adapter config rank r={config_rank} does not match expected rank {expected.rank}."
            )
        if config_alpha is not None and int(config_alpha) != _effective_lora_alpha(expected):
            raise ValueError(
                "Adapter config lora_alpha="
                f"{config_alpha} does not match expected alpha {_effective_lora_alpha(expected)}."
            )
        if config_targets is not None and config_targets != expected_targets:
            raise ValueError(
                "Adapter config target_modules do not match expected LoRA config: "
                f"config={sorted(config_targets)}, expected={sorted(expected_targets)}."
            )


def _validate_attention_tp(model_cfg: Qwen3MoEConfig, ps: ParallelState) -> None:
    if ps.tp_size <= 0:
        raise ValueError(f"TP size must be positive, got {ps.tp_size}.")
    if model_cfg.num_attention_heads % ps.tp_size != 0:
        raise ValueError(
            "LoRA adapter import/export requires num_attention_heads "
            f"({model_cfg.num_attention_heads}) to be divisible by TP={ps.tp_size}."
        )
    if model_cfg.num_key_value_heads % ps.tp_size != 0:
        raise ValueError(
            "LoRA adapter import/export requires num_key_value_heads "
            f"({model_cfg.num_key_value_heads}) to be divisible by TP={ps.tp_size}."
        )
    q_heads_local = model_cfg.num_attention_heads // ps.tp_size
    kv_heads_local = model_cfg.num_key_value_heads // ps.tp_size
    if kv_heads_local <= 0:
        raise ValueError(
            "LoRA adapter import/export requires at least one local KV head; "
            f"got num_key_value_heads={model_cfg.num_key_value_heads}, TP={ps.tp_size}."
        )
    if q_heads_local % kv_heads_local != 0:
        raise ValueError(
            "LoRA adapter import/export requires local query heads to be divisible "
            f"by local KV heads, got q={q_heads_local}, kv={kv_heads_local}."
        )


def export_lora_adapter_state(
    chunks: list[nn.Module] | tuple[nn.Module, ...],
    model_cfg: Qwen3MoEConfig,
    ps: ParallelState,
    *,
    cpu: bool = True,
) -> dict[str, torch.Tensor]:
    """Export native LoRA tensors to a full PEFT-style adapter state dict.

    All distributed ranks must call this function. Rank 0 returns the full
    adapter state; other ranks return an empty dict.
    """

    if ps.pp_size != 1:
        raise NotImplementedError("LoRA adapter export currently supports pp=1.")
    if ps.etp_size != 1:
        raise NotImplementedError("LoRA adapter export currently supports etp=1.")

    _validate_attention_tp(model_cfg, ps)
    state: dict[str, torch.Tensor] = {}
    q_heads_local = model_cfg.num_attention_heads // ps.tp_size
    kv_heads_local = model_cfg.num_key_value_heads // ps.tp_size

    for chunk in _iter_qwen_chunks(list(chunks)):
        for layer in chunk.layers:
            layer_idx = int(layer.layer_idx)
            attn = layer.attn

            if attn.qkv_lora is not None:
                if _is_rank_partitioned_lora_a(attn.qkv_lora, ps):
                    qkv_a = _gather_lora_rank_partition(attn.qkv_lora.lora_a.detach(), ps)
                else:
                    qkv_a = _select_tp_replicated(attn.qkv_lora.lora_a.detach(), ps)
                q_b_local, k_b_local, v_b_local = _split_local_mcore_qkv_b(
                    attn.qkv_lora.lora_b.detach(),
                    num_heads_local=q_heads_local,
                    num_kv_heads_local=kv_heads_local,
                    head_dim=model_cfg.head_dim,
                )
                q_b = _all_gather_cat(q_b_local, ps.tp_group, dim=0)
                k_b = _all_gather_cat(k_b_local, ps.tp_group, dim=0)
                v_b = _all_gather_cat(v_b_local, ps.tp_group, dim=0)
                if _rank() == 0:
                    state[_attn_key(layer_idx, "q_proj", "lora_A")] = qkv_a
                    state[_attn_key(layer_idx, "q_proj", "lora_B")] = q_b
                    state[_attn_key(layer_idx, "k_proj", "lora_A")] = qkv_a.clone()
                    state[_attn_key(layer_idx, "k_proj", "lora_B")] = k_b
                    state[_attn_key(layer_idx, "v_proj", "lora_A")] = qkv_a.clone()
                    state[_attn_key(layer_idx, "v_proj", "lora_B")] = v_b

            if attn.proj_lora is not None:
                proj_a = _all_gather_cat(attn.proj_lora.lora_a.detach(), ps.tp_group, dim=1)
                if _is_output_partitioned_lora_b(attn.proj_lora, ps):
                    proj_b = _all_gather_cat(attn.proj_lora.lora_b.detach(), ps.tp_group, dim=0)
                else:
                    proj_b = _select_tp_replicated(attn.proj_lora.lora_b.detach(), ps)
                if _rank() == 0:
                    state[_attn_key(layer_idx, "o_proj", "lora_A")] = proj_a
                    state[_attn_key(layer_idx, "o_proj", "lora_B")] = proj_b

            experts = layer.moe.experts
            if experts.fc1_lora is not None:
                fc1_a_local, fc1_b_local = _expand_shared_expert_lora(
                    experts.fc1_lora, experts.num_local_experts
                )
                fc1_a = _all_gather_cat(fc1_a_local, ps.ep_group, dim=0)
                fc1_b = _all_gather_cat(fc1_b_local, ps.ep_group, dim=0)
                gate_b, up_b = fc1_b.chunk(2, dim=1)
                if _rank() == 0:
                    for expert_idx in range(model_cfg.num_experts):
                        state[_expert_key(layer_idx, expert_idx, "gate_proj", "lora_A")] = fc1_a[
                            expert_idx
                        ]
                        state[_expert_key(layer_idx, expert_idx, "gate_proj", "lora_B")] = gate_b[
                            expert_idx
                        ]
                        state[_expert_key(layer_idx, expert_idx, "up_proj", "lora_A")] = fc1_a[
                            expert_idx
                        ].clone()
                        state[_expert_key(layer_idx, expert_idx, "up_proj", "lora_B")] = up_b[
                            expert_idx
                        ]

            if experts.fc2_lora is not None:
                fc2_a_local, fc2_b_local = _expand_shared_expert_lora(
                    experts.fc2_lora, experts.num_local_experts
                )
                fc2_a = _all_gather_cat(fc2_a_local, ps.ep_group, dim=0)
                fc2_b = _all_gather_cat(fc2_b_local, ps.ep_group, dim=0)
                if _rank() == 0:
                    for expert_idx in range(model_cfg.num_experts):
                        state[_expert_key(layer_idx, expert_idx, "down_proj", "lora_A")] = fc2_a[
                            expert_idx
                        ]
                        state[_expert_key(layer_idx, expert_idx, "down_proj", "lora_B")] = fc2_b[
                            expert_idx
                        ]

    if _rank() != 0:
        return {}
    if cpu:
        return {name: tensor.detach().cpu().contiguous() for name, tensor in state.items()}
    return {name: tensor.detach().contiguous() for name, tensor in state.items()}


def save_lora_adapter(
    chunks: list[nn.Module] | tuple[nn.Module, ...],
    model_cfg: Qwen3MoEConfig,
    ps: ParallelState,
    output_dir: str | Path,
    *,
    base_model_name_or_path: str = "",
    lora_config: LoraConfig | dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save a PEFT/Mint-compatible LoRA adapter directory."""

    from safetensors.torch import save_file

    if lora_config is None:
        raise ValueError("save_lora_adapter requires the LoRA config used to build the model.")
    lora = normalize_lora_config(lora_config)
    if not lora.enabled:
        raise ValueError("save_lora_adapter requires an enabled LoRA config.")
    state = export_lora_adapter_state(chunks, model_cfg, ps, cpu=True)
    output = Path(output_dir)
    if _rank() == 0:
        output.mkdir(parents=True, exist_ok=True)
        save_file(state, str(output / "adapter_model.safetensors"))
        config = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": base_model_name_or_path,
            "inference_mode": False,
            "r": lora.rank,
            "lora_alpha": _effective_lora_alpha(lora),
            "lora_dropout": lora.dropout,
            "target_modules": _target_modules_from_lora_config(lora),
            "bias": "none",
            "fan_in_fan_out": False,
            "init_lora_weights": True,
            "modules_to_save": None,
        }
        (output / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
        meta = {
            "format": "megatron.lite_qwen3_moe_lora_peft_v1",
            "expert_lora_representation": (
                "shared_local_expert_group"
                if any(
                    _expert_lora_is_shared(getattr(layer.moe.experts, attr))
                    for chunk in _iter_qwen_chunks(list(chunks))
                    for layer in chunk.layers
                    for attr in ("fc1_lora", "fc2_lora")
                    if getattr(layer.moe.experts, attr) is not None
                )
                else "per_expert"
            ),
            "num_tensors": len(state),
            "num_parameters": int(sum(t.numel() for t in state.values())),
            "parallel": {"tp": ps.tp_size, "ep": ps.ep_size, "etp": ps.etp_size, "pp": ps.pp_size},
            "model": {
                "num_hidden_layers": model_cfg.num_hidden_layers,
                "hidden_size": model_cfg.hidden_size,
                "num_attention_heads": model_cfg.num_attention_heads,
                "num_key_value_heads": model_cfg.num_key_value_heads,
                "head_dim": model_cfg.head_dim,
                "num_experts": model_cfg.num_experts,
                "moe_intermediate_size": model_cfg.moe_intermediate_size,
            },
            "metadata": metadata or {},
        }
        (output / "megatron.lite_adapter_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        result = {
            "path": str(output),
            "adapter_model": str(output / "adapter_model.safetensors"),
            "adapter_config": str(output / "adapter_config.json"),
            **meta,
        }
    else:
        result = {}
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return result


def _require_tensor(state: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    try:
        return state[key]
    except KeyError as exc:
        raise KeyError(f"Missing adapter tensor {key!r}") from exc


def _slice_tp_output(tensor: torch.Tensor, local_width: int, ps: ParallelState) -> torch.Tensor:
    start = ps.tp_rank * local_width
    return tensor[start : start + local_width].contiguous()


def _slice_tp_input(tensor: torch.Tensor, local_width: int, ps: ParallelState) -> torch.Tensor:
    start = ps.tp_rank * local_width
    return tensor[:, start : start + local_width].contiguous()


def load_lora_adapter_state(
    chunks: list[nn.Module] | tuple[nn.Module, ...],
    state: dict[str, torch.Tensor],
    model_cfg: Qwen3MoEConfig,
    ps: ParallelState,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Load a PEFT adapter state into this rank's native LoRA shards."""

    if ps.pp_size != 1:
        raise NotImplementedError("LoRA adapter import currently supports pp=1.")
    if ps.etp_size != 1:
        raise NotImplementedError("LoRA adapter import currently supports etp=1.")

    _validate_attention_tp(model_cfg, ps)
    q_heads_local = model_cfg.num_attention_heads // ps.tp_size
    kv_heads_local = model_cfg.num_key_value_heads // ps.tp_size
    q_width_local = q_heads_local * model_cfg.head_dim
    kv_width_local = kv_heads_local * model_cfg.head_dim
    attn_in_width_local = q_width_local

    loaded = 0
    for chunk in _iter_qwen_chunks(list(chunks)):
        for layer in chunk.layers:
            layer_idx = int(layer.layer_idx)
            attn = layer.attn
            if attn.qkv_lora is not None:
                q_a = _require_tensor(state, _attn_key(layer_idx, "q_proj", "lora_A")).to(
                    device=attn.qkv_lora.lora_a.device, dtype=attn.qkv_lora.lora_a.dtype
                )
                k_a = _require_tensor(state, _attn_key(layer_idx, "k_proj", "lora_A")).to(q_a)
                v_a = _require_tensor(state, _attn_key(layer_idx, "v_proj", "lora_A")).to(q_a)
                if strict and (not torch.equal(q_a, k_a) or not torch.equal(q_a, v_a)):
                    raise ValueError(
                        "Megatron Lite fused qkv_lora requires q/k/v lora_A tensors to match."
                    )
                q_a_local = (
                    _slice_lora_rank_partition(q_a, ps)
                    if _is_rank_partitioned_lora_a(attn.qkv_lora, ps)
                    else q_a.contiguous()
                )
                q_b = _slice_tp_output(
                    _require_tensor(state, _attn_key(layer_idx, "q_proj", "lora_B")).to(
                        device=attn.qkv_lora.lora_b.device, dtype=attn.qkv_lora.lora_b.dtype
                    ),
                    q_width_local,
                    ps,
                )
                k_b = _slice_tp_output(
                    _require_tensor(state, _attn_key(layer_idx, "k_proj", "lora_B")).to(
                        device=attn.qkv_lora.lora_b.device, dtype=attn.qkv_lora.lora_b.dtype
                    ),
                    kv_width_local,
                    ps,
                )
                v_b = _slice_tp_output(
                    _require_tensor(state, _attn_key(layer_idx, "v_proj", "lora_B")).to(
                        device=attn.qkv_lora.lora_b.device, dtype=attn.qkv_lora.lora_b.dtype
                    ),
                    kv_width_local,
                    ps,
                )
                attn.qkv_lora.lora_a.data.copy_(q_a_local)
                attn.qkv_lora.lora_b.data.copy_(
                    _pack_local_mcore_qkv_b(
                        q_b,
                        k_b,
                        v_b,
                        num_heads_local=q_heads_local,
                        num_kv_heads_local=kv_heads_local,
                        head_dim=model_cfg.head_dim,
                    )
                )
                loaded += 2

            if attn.proj_lora is not None:
                proj_a = _slice_tp_input(
                    _require_tensor(state, _attn_key(layer_idx, "o_proj", "lora_A")).to(
                        device=attn.proj_lora.lora_a.device, dtype=attn.proj_lora.lora_a.dtype
                    ),
                    attn_in_width_local,
                    ps,
                )
                proj_b = _require_tensor(state, _attn_key(layer_idx, "o_proj", "lora_B")).to(
                    device=attn.proj_lora.lora_b.device, dtype=attn.proj_lora.lora_b.dtype
                )
                if _is_output_partitioned_lora_b(attn.proj_lora, ps):
                    proj_b = _slice_tp_output(proj_b, attn.proj_lora.lora_b.shape[0], ps)
                attn.proj_lora.lora_a.data.copy_(proj_a)
                attn.proj_lora.lora_b.data.copy_(proj_b)
                loaded += 2

            experts = layer.moe.experts
            expert_start = ps.ep_rank * experts.num_local_experts
            expert_stop = expert_start + experts.num_local_experts
            if experts.fc1_lora is not None:
                if _expert_lora_is_shared(experts.fc1_lora):
                    local_gate_a = []
                    local_gate_b = []
                    local_up_b = []
                    for expert_idx in range(expert_start, expert_stop):
                        gate_a = _require_tensor(
                            state, _expert_key(layer_idx, expert_idx, "gate_proj", "lora_A")
                        ).to(
                            device=experts.fc1_lora.lora_a.device,
                            dtype=experts.fc1_lora.lora_a.dtype,
                        )
                        up_a = _require_tensor(
                            state, _expert_key(layer_idx, expert_idx, "up_proj", "lora_A")
                        ).to(gate_a)
                        if strict and not torch.equal(gate_a, up_a):
                            raise ValueError(
                                "Megatron Lite fused fc1_lora requires gate/up lora_A tensors to match."
                            )
                        local_gate_a.append(gate_a)
                        local_gate_b.append(
                            _require_tensor(
                                state, _expert_key(layer_idx, expert_idx, "gate_proj", "lora_B")
                            ).to(
                                device=experts.fc1_lora.lora_b.device,
                                dtype=experts.fc1_lora.lora_b.dtype,
                            )
                        )
                        local_up_b.append(
                            _require_tensor(
                                state, _expert_key(layer_idx, expert_idx, "up_proj", "lora_B")
                            ).to(
                                device=experts.fc1_lora.lora_b.device,
                                dtype=experts.fc1_lora.lora_b.dtype,
                            )
                        )
                    if strict:
                        if any(
                            not torch.equal(local_gate_a[0], value) for value in local_gate_a[1:]
                        ):
                            raise ValueError(
                                "Megatron Lite shared expert fc1_lora can only import PEFT adapters "
                                "whose local expert gate lora_A tensors are identical."
                            )
                        if any(
                            not torch.equal(local_gate_b[0], value) for value in local_gate_b[1:]
                        ):
                            raise ValueError(
                                "Megatron Lite shared expert fc1_lora can only import PEFT adapters "
                                "whose local expert gate lora_B tensors are identical."
                            )
                        if any(not torch.equal(local_up_b[0], value) for value in local_up_b[1:]):
                            raise ValueError(
                                "Megatron Lite shared expert fc1_lora can only import PEFT adapters "
                                "whose local expert up lora_B tensors are identical."
                            )
                    experts.fc1_lora.lora_a.data.copy_(local_gate_a[0])
                    experts.fc1_lora.lora_b.data.copy_(
                        torch.cat([local_gate_b[0], local_up_b[0]], dim=0)
                    )
                    loaded += 2
                else:
                    for local_idx, expert_idx in enumerate(range(expert_start, expert_stop)):
                        gate_a = _require_tensor(
                            state, _expert_key(layer_idx, expert_idx, "gate_proj", "lora_A")
                        ).to(
                            device=experts.fc1_lora.lora_a.device,
                            dtype=experts.fc1_lora.lora_a.dtype,
                        )
                        up_a = _require_tensor(
                            state, _expert_key(layer_idx, expert_idx, "up_proj", "lora_A")
                        ).to(gate_a)
                        if strict and not torch.equal(gate_a, up_a):
                            raise ValueError(
                                "Megatron Lite fused fc1_lora requires gate/up lora_A tensors to match."
                            )
                        gate_b = _require_tensor(
                            state, _expert_key(layer_idx, expert_idx, "gate_proj", "lora_B")
                        ).to(
                            device=experts.fc1_lora.lora_b.device,
                            dtype=experts.fc1_lora.lora_b.dtype,
                        )
                        up_b = _require_tensor(
                            state, _expert_key(layer_idx, expert_idx, "up_proj", "lora_B")
                        ).to(
                            device=experts.fc1_lora.lora_b.device,
                            dtype=experts.fc1_lora.lora_b.dtype,
                        )
                        experts.fc1_lora.lora_a.data[local_idx].copy_(gate_a)
                        experts.fc1_lora.lora_b.data[local_idx].copy_(
                            torch.cat([gate_b, up_b], dim=0)
                        )
                        loaded += 2

            if experts.fc2_lora is not None:
                if _expert_lora_is_shared(experts.fc2_lora):
                    local_a = []
                    local_b = []
                    for expert_idx in range(expert_start, expert_stop):
                        local_a.append(
                            _require_tensor(
                                state, _expert_key(layer_idx, expert_idx, "down_proj", "lora_A")
                            ).to(
                                device=experts.fc2_lora.lora_a.device,
                                dtype=experts.fc2_lora.lora_a.dtype,
                            )
                        )
                        local_b.append(
                            _require_tensor(
                                state, _expert_key(layer_idx, expert_idx, "down_proj", "lora_B")
                            ).to(
                                device=experts.fc2_lora.lora_b.device,
                                dtype=experts.fc2_lora.lora_b.dtype,
                            )
                        )
                    if strict:
                        if any(not torch.equal(local_a[0], value) for value in local_a[1:]):
                            raise ValueError(
                                "Megatron Lite shared expert fc2_lora can only import PEFT adapters "
                                "whose local expert down lora_A tensors are identical."
                            )
                        if any(not torch.equal(local_b[0], value) for value in local_b[1:]):
                            raise ValueError(
                                "Megatron Lite shared expert fc2_lora can only import PEFT adapters "
                                "whose local expert down lora_B tensors are identical."
                            )
                    experts.fc2_lora.lora_a.data.copy_(local_a[0])
                    experts.fc2_lora.lora_b.data.copy_(local_b[0])
                    loaded += 2
                else:
                    for local_idx, expert_idx in enumerate(range(expert_start, expert_stop)):
                        experts.fc2_lora.lora_a.data[local_idx].copy_(
                            _require_tensor(
                                state, _expert_key(layer_idx, expert_idx, "down_proj", "lora_A")
                            ).to(
                                device=experts.fc2_lora.lora_a.device,
                                dtype=experts.fc2_lora.lora_a.dtype,
                            )
                        )
                        experts.fc2_lora.lora_b.data[local_idx].copy_(
                            _require_tensor(
                                state, _expert_key(layer_idx, expert_idx, "down_proj", "lora_B")
                            ).to(
                                device=experts.fc2_lora.lora_b.device,
                                dtype=experts.fc2_lora.lora_b.dtype,
                            )
                        )
                        loaded += 2

    return {"loaded_tensors": loaded}


def load_lora_adapter(
    chunks: list[nn.Module] | tuple[nn.Module, ...],
    adapter_dir: str | Path,
    model_cfg: Qwen3MoEConfig,
    ps: ParallelState,
    *,
    strict: bool = True,
    lora_config: LoraConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    from safetensors.torch import load_file

    path = Path(adapter_dir) / "adapter_model.safetensors"
    state = load_file(str(path), device="cpu")
    config_path = Path(adapter_dir) / "adapter_config.json"
    if config_path.exists():
        adapter_config = json.loads(config_path.read_text())
        _validate_adapter_config(chunks, state, adapter_config, lora_config=lora_config)
    elif lora_config is not None:
        expected = normalize_lora_config(lora_config)
        state_rank = _infer_state_rank(state)
        if state_rank is not None and state_rank != expected.rank:
            raise ValueError(
                f"Adapter tensor rank {state_rank} does not match expected rank {expected.rank}."
            )
    return load_lora_adapter_state(chunks, state, model_cfg, ps, strict=strict)


__all__ = [
    "export_lora_adapter_state",
    "load_lora_adapter",
    "load_lora_adapter_state",
    "save_lora_adapter",
]
