# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek V4 (ds4flash) lite native <-> HF checkpoint mapping.

Like kimi_k2 / glm5: ``DeepseekV4WeightSpec`` encodes the per-param native -> HF
name (+ TP/EP shard spec); export/save route through the shared
``primitive/ckpt/hf_weights.py`` exporter (its PP ``all_gather_object`` is
reached by all ranks before any ``rank0_only`` filter, so PP>1 export doesn't
desync).  Native names are bare ``DeepseekV4Model`` keys; ``self.layers`` is a
ModuleDict keyed by GLOBAL layer index (kimi uses a local ModuleList), so the
exporter's local->global remap is an identity here.

HF targets are canonical HF DeepSeek (``model.embed_tokens.weight`` /
``model.norm.weight`` / ``lm_head.weight`` / ``model.layers.<i>.self_attn.*`` /
``...mlp.experts.<id>.{gate,up,down}_proj.weight``).  DS4 extras:
  * CSA: ``self_attn.*`` incl. ``compressor.*`` / ``indexer.*``; ``sinks`` ->
    ``self_attn.attn_sink``.
  * mHC: ``attn_hc`` / ``ffn_hc`` -> ``...self_attn.hc_*`` / ``...mlp.hc_*``;
    model-wide ``hc_head`` -> ``model.hc_head.*`` (no HF analogue, kept
    model.-rooted; fidelity vs Megatron's latest mHC is a TODO).
  * MTP: folded into the decoder namespace at ``model.layers.<num_hidden+i>``.

CSA is not TP-capable: DS4 runs TP=ETP=1 (only EP shards experts), like GLM-5.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn
from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.utils import ensure_divisible
from megatron.lite.runtime.contracts.weights import ResyncFormat

from megatron.lite.primitive.ckpt.hf_weights import (  # isort: skip
    parse_expert_idx,
    to_global_layer_name,
)

_QUANTIZED_RESYNC_TARGETS = {ResyncFormat.BLOCK_FP8.value, ResyncFormat.MXFP4.value}


def EXPERT_CLASSIFIER(name: str) -> bool:
    return ".experts." in name and ".shared_experts." not in name


def PLACEMENT_FN(param_name: str) -> list:
    # distckpt sharded placement (TP=ETP=1 for ds4; shares kimi/glm5's
    # Experts/SwiGLUMLP/VocabParallel structure). EP-sharded experts must carry
    # an explicit placement or the dist-opt checkpoint won't restore them
    # bit-exactly. The CSA/mHC/MTP-norm params fall through to all-Replicate.
    from torch.distributed.tensor import Replicate, Shard

    if ".experts." in param_name and ".shared_experts." not in param_name:
        if "fc1" in param_name:
            return [Replicate(), Replicate(), Shard(0), Shard(0)]
        if "fc2" in param_name:
            return [Replicate(), Replicate(), Shard(0), Shard(1)]
        return [Replicate(), Replicate(), Replicate(), Replicate()]
    if "eh_proj.linear.weight" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    if "gate_up" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    if "down" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(1)]
    if "embed" in param_name or "head" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    return [Replicate(), Replicate(), Replicate(), Replicate()]


# Native <-> HF name mapping (shared by export spec and load path).  Native
# names are bare DeepseekV4Model state_dict keys with GLOBAL layer indices.
_BLOCK_KEY_RE = re.compile(r"^(layers|mtp)\.(\d+)\.(.+)$")
_GROUPED_EXPERT_RE = re.compile(r"^mlp\.experts\.fc([12])\.weight(\d+)$")
# Native top-level params -> real DeepSeek-V4-Flash release names (NOT DeepSeek-V3 HF
# names; the V4 release uses bare `embed.weight` / `head.weight` / `norm.weight` and a
# `layers.N.attn.* / ffn.* / hc_*` layout). This same mapping drives both the load path
# and the export spec, so MLite round-trips against the real release / vLLM ds4 format.
_TOP_LEVEL = {
    "embed_tokens.embedding.weight": "embed.weight",
    "norm.weight": "norm.weight",
    "hc_head.hc_fn": "hc_head_fn",
    "hc_head.hc_base": "hc_head_base",
    "hc_head.hc_scale": "hc_head_scale",
    "lm_head.col.linear.weight": "head.weight",
}


def _map_block_attr(attr: str, block: str) -> str | tuple[str, ...] | None:
    """Map a native per-block attr -> real V4-Flash suffix (relative to layers.N / mtp.N)."""
    if attr == "input_layernorm.weight":
        return "attn_norm.weight"
    if attr == "post_attention_layernorm.weight":
        return "ffn_norm.weight"
    # CSA attention: native `self_attn.self_attn.*` (the SBHD wrapper adds one extra
    # `self_attn` level) -> real `attn.*`. Covers compressor.* / indexer.* / wq_a / wkv / ...
    sub = None
    if attr.startswith("self_attn.self_attn."):
        sub = attr.removeprefix("self_attn.self_attn.")
    elif attr.startswith("self_attn."):
        sub = attr.removeprefix("self_attn.")
    if sub is not None:
        return "attn.attn_sink" if sub == "sinks" else f"attn.{sub}"
    if attr.startswith("mlp.gate."):
        suffix = attr.removeprefix("mlp.gate.")
        return "ffn.gate." + {
            "gate.weight": "weight",
            "weight": "weight",
            "expert_bias": "bias",
            "e_score_correction_bias": "bias",
            "tid2eid": "tid2eid",
        }.get(suffix, suffix)
    if attr.startswith("mlp.shared_experts."):
        proj = attr.removeprefix("mlp.shared_experts.").removesuffix(".weight")
        if proj == "gate_up":
            return "ffn.shared_experts.w1.weight", "ffn.shared_experts.w3.weight"
        if proj == "down":
            return "ffn.shared_experts.w2.weight"
        return f"ffn.shared_experts.{proj}.weight"
    # mHC (hyper-connections): native attn_hc/ffn_hc.{base,fn,scale} -> hc_attn_*/hc_ffn_*;
    # mtp carries its own hc_head.hc_* -> hc_head_*.
    for prefix, target in (("attn_hc", "hc_attn"), ("ffn_hc", "hc_ffn")):
        if attr.startswith(f"{prefix}."):
            return f"{target}_{attr.rsplit('.', 1)[-1]}"
    if attr.startswith("hc_head."):
        return f"hc_head_{attr.rsplit('.', 1)[-1].removeprefix('hc_')}"
    if block == "mtp" and attr in {
        "e_proj.weight",
        "h_proj.weight",
        "enorm.weight",
        "hnorm.weight",
        "norm.weight",
    }:
        return attr
    return None


def _global_expert_idx_from_local(
    local_idx: int, config: DeepseekV4Config, ps: ParallelState
) -> int:
    num_local = ensure_divisible(config.n_routed_experts, ps.ep_size)
    return ps.ep_rank * num_local + local_idx


def _router_buffer_matches_layer_kind(name: str, config: DeepseekV4Config) -> bool:
    """Keep only the router buffer serialized by the corresponding HF layer."""
    match = _BLOCK_KEY_RE.match(name)
    if match is None:
        return False
    block, index_text, _ = match.groups()
    is_hash_layer = block == "layers" and int(index_text) < config.num_hash_layers
    if name.endswith(".mlp.gate.tid2eid"):
        return is_hash_layer
    if name.endswith(".mlp.gate.expert_bias"):
        return not is_hash_layer
    return False


def _hf_names_for_state_key(name: str, config: DeepseekV4Config) -> list[str]:
    """Map a bare DS4 native key (global layer idx, global expert id) to HF name(s).

    Callers (load path + shared exporter) supply global expert ids first.
    """
    mapped = _TOP_LEVEL.get(name)
    if mapped is not None:
        return [mapped]
    match = _BLOCK_KEY_RE.match(name)
    if match is None:
        return []
    block, index, attr = match.groups()
    # Real V4-Flash keeps decoder layers under ``layers.{i}`` and the MTP block under its
    # own ``mtp.{i}`` namespace (no ``model.`` prefix, no continued global index).
    prefix = f"layers.{index}" if block == "layers" else f"mtp.{index}"
    mapped = _map_block_attr(attr, block)
    if mapped is not None:
        if isinstance(mapped, tuple):
            return [f"{prefix}.{part}" for part in mapped]
        return [f"{prefix}.{mapped}"]
    expert = _GROUPED_EXPERT_RE.match(attr)
    if expert is None:
        return []
    fc, expert_id = expert.groups()
    # native fused gate_up (fc1) -> real w1 (gate) + w3 (up); fc2 -> w2 (down).
    expert_prefix = f"{prefix}.ffn.experts.{int(expert_id)}"
    if fc == "1":
        return [f"{expert_prefix}.w1.weight", f"{expert_prefix}.w3.weight"]
    return [f"{expert_prefix}.w2.weight"]


# ======================================================================
# FP4 / scaled-tensor dequant helpers (load path).
# ======================================================================


def _is_native_metadata_key(name: str) -> bool:
    return name.endswith("._extra_state")


def load_hf_weights(
    model: nn.Module, path: str, config: DeepseekV4Config, ps: ParallelState
) -> None:
    from megatron.lite.primitive.ckpt.hf_weights import (  # isort: skip
        load_hf_weights as _load,
    )

    _load(model, path, DeepseekV4WeightSpec(config), ps, vocab_size=config.vocab_size)


def _to_global_expert_name(
    name: str, config: DeepseekV4Config, ps: ParallelState
) -> str:
    """Rewrite an EP-local expert ``weight<local>`` suffix to its global id.

    The native ``state_dict`` carries the EP-local expert index; the HF target
    name uses the global expert id.  Non-expert names pass through unchanged.
    """
    match = _BLOCK_KEY_RE.match(name)
    if match is None:
        return name
    block, index, attr = match.groups()
    expert = _GROUPED_EXPERT_RE.match(attr)
    if expert is None:
        return name
    fc, local_idx = expert.groups()
    global_idx = _global_expert_idx_from_local(int(local_idx), config, ps)
    return f"{block}.{index}.mlp.experts.fc{fc}.weight{global_idx}"


# ======================================================================
# Export: shared TP/ETP/EP/PP gather via DeepseekV4WeightSpec.
# ======================================================================


class DeepseekV4WeightSpec:
    """Export DS4 lite weights to HF DeepSeek-V4 names (CSA / mHC / MTP / MoE).

    Mirrors ``KimiK2WeightSpec`` / ``Glm5WeightSpec`` on DS4's bare native names
    with global layer indices.  The shared exporter rewrites EP-local expert
    ``weight<local>`` ids to global before calling ``native_to_hf``.
    """

    # HF names whose released dtype is fp32.  These tensors are declared fp32
    # at construction (see ``mhc.py`` / attention sinks / compressor.ape) but
    # the protocol's module-wide ``.to(bfloat16)`` downcasts them; the export
    # path re-materializes them as fp32 so the saved checkpoint matches the
    # DeepSeek-V4-Flash release byte-for-byte in dtype.
    _FP32_HF_SUFFIXES: tuple[str, ...] = (
        ".attn_sink",
        ".compressor.ape",
        ".indexer.compressor.ape",
        ".ffn.gate.bias",
    )
    _FP32_HF_INFIXES: tuple[str, ...] = ("hc_head_", ".hc_attn_", ".hc_ffn_")

    def __init__(self, config: DeepseekV4Config):
        self.config = config

    def hf_export_dtype_override(self, hf_name: str) -> torch.dtype | None:
        """Return an explicit export dtype for ``hf_name``, or ``None`` when
        the shared exporter's default (usually the training dtype) is fine."""
        if hf_name.endswith(self._FP32_HF_SUFFIXES):
            return torch.float32
        if any(marker in hf_name for marker in self._FP32_HF_INFIXES):
            return torch.float32
        return None

    @property
    def num_experts(self) -> int:
        return self.config.n_routed_experts

    def weight_map(self) -> dict[str, list[str]]:
        return {}

    def validate_load(self, ps: ParallelState) -> None:
        if (ps.tp_size, ps.etp_size) != (1, 1):
            raise NotImplementedError(
                "DeepSeek V4 direct HF load currently supports only TP=ETP=1."
            )

    def load_weight_map(
        self,
        base_model: nn.Module,
        ps: ParallelState,
        logical_state_keys: tuple[str, ...],
    ) -> dict[str, list[str]]:
        layer_map = (
            {
                local_idx: base_model.layer_indices[local_idx]
                for local_idx in range(len(base_model.layer_indices))
            }
            if hasattr(base_model, "layer_indices")
            else {}
        )
        weight_map: dict[str, list[str]] = {}
        for name in logical_state_keys:
            if ".parametrizations." in name or _is_native_metadata_key(name):
                continue
            global_name = to_global_layer_name(name, layer_map)
            mapped_name = _to_global_expert_name(global_name, self.config, ps)
            hf_names = _hf_names_for_state_key(mapped_name, self.config)
            if hf_names:
                weight_map[mapped_name] = hf_names
        return weight_map

    def hf_to_native(
        self, native_name: str, hf_tensors: list[torch.Tensor]
    ) -> torch.Tensor:
        del native_name
        return torch.cat(hf_tensors, dim=0) if len(hf_tensors) == 2 else hf_tensors[0]

    def hf_target_shape(
        self, native_name: str, source_index: int, target_shape: torch.Size
    ) -> torch.Size:
        del source_index
        if (
            len(_hf_names_for_state_key(native_name, self.config)) == 2
            and target_shape
            and target_shape[0] % 2 == 0
        ):
            return torch.Size((target_shape[0] // 2, *target_shape[1:]))
        return target_shape

    @staticmethod
    def replica_group_for_load(native_name: str, ps: ParallelState):
        if EXPERT_CLASSIFIER(native_name):
            return getattr(ps, "ep_dp_group", None)
        return getattr(ps, "dp_cp_group", None)

    @staticmethod
    def replica_source_rank_for_load(native_name: str, ps: ParallelState) -> int:
        if not EXPERT_CLASSIFIER(native_name):
            return 0
        expert_dp_size = int(getattr(ps, "expert_dp_size", 1))
        if expert_dp_size < 1:
            raise ValueError(f"Invalid expert_dp_size={expert_dp_size}")
        return int(getattr(ps, "ep_rank", 0)) % expert_dp_size

    def native_to_hf(
        self, native_name: str, tensor: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        # ``native_name`` is the global native name; experts already carry the
        # global expert id (shared exporter rewrote weight<local> -> weight<gid>).
        hf_names = _hf_names_for_state_key(native_name, self.config)
        if not hf_names:
            return []
        if len(hf_names) == 1:
            return [(hf_names[0], tensor)]
        if len(hf_names) == 2:
            # 2 targets == fused gate/up split into (w1, w3) for shared/routed
            # experts; split the leading dim exactly as the bespoke export did.
            first, second = tensor.chunk(2, dim=0)
            return [
                (hf_names[0], first.contiguous()),
                (hf_names[1], second.contiguous()),
            ]
        raise AssertionError(
            f"Unexpected HF name fan-out for {native_name}: {hf_names}"
        )

    def qkv_spec(self, native_name: str) -> tuple[int, int, int] | None:
        del native_name
        return None

    def tp_spec(self, native_name: str) -> tuple[int, int] | None:
        # DS4 is TP=ETP=1 (CSA is not TP-capable); only EP shards experts.  The
        # expert (split_dim, ETP) entries are declared so the shared ETP path
        # would be correct if ETP were ever enabled; embed/head/eh_proj carry
        # the vocab split-dim spec for completeness (no-op at TP=1).
        if self.is_expert(native_name):
            if ".fc1." in native_name:
                return (0, 1)
            if ".fc2." in native_name:
                return (1, 1)
            return None
        if native_name.endswith(".eh_proj.linear.weight"):
            return (0, 0)
        if native_name in {
            "embed_tokens.embedding.weight",
            "lm_head.col.linear.weight",
        }:
            return (0, 0)
        return None

    def is_expert(self, native_name: str) -> bool:
        return ".mlp.experts." in native_name and ".shared_experts." not in native_name

    def expert_global_id(self, native_name: str) -> int | None:
        if self.is_expert(native_name):
            return parse_expert_idx(native_name)
        return None

    def expert_local_name(self, native_name: str, local_idx: int) -> str:
        prefix = native_name.rsplit(".weight", 1)[0]
        return f"{prefix}.weight{local_idx}"


def _export_unquantized_weights(
    model, config: DeepseekV4Config, ps: ParallelState, **kwargs
):
    """Export DS4 parameters and mapped persistent buffers through one plan."""
    from megatron.lite.primitive.ckpt.hf_weights import (  # isort: skip
        export_hf_weights as _export,
    )

    spec = DeepseekV4WeightSpec(config)
    for name, tensor in _export(
        model, spec, ps, vocab_size=config.vocab_size, **kwargs
    ):
        if tensor.is_floating_point():
            override = spec.hf_export_dtype_override(name)
            if override is not None and tensor.dtype != override:
                tensor = tensor.to(override)
        yield name, tensor


def export_hf_weights(model, config: DeepseekV4Config, ps: ParallelState, **kwargs):
    """Export DS4 weights as HF or serialized vLLM-checkpoint pairs.

    The default remains the ordinary HF/BF16 stream. ``block_fp8`` and
    ``mxfp4`` are model-owned adapters over that gathered stream; the runtime
    and veRL engine do not classify individual DS4 tensors. ``mxfp4`` describes
    the routed-expert representation; dense quantized weights remain block FP8.
    """
    target = kwargs.pop("target", "hf")
    resync_config = kwargs.pop("resync_config", None)
    if target not in {"hf", ResyncFormat.BF16.value, *_QUANTIZED_RESYNC_TARGETS}:
        raise ValueError(f"Unsupported DeepSeek-V4 export target: {target!r}")
    weights = _export_unquantized_weights(model, config, ps, **kwargs)
    if target in _QUANTIZED_RESYNC_TARGETS:
        from megatron.lite.model.deepseek_v4.lite.resync import (  # isort: skip
            export_resync_weights,
        )

        if target == ResyncFormat.MXFP4.value:
            resync_config = dict(resync_config or {})
            configured_dtype = resync_config.get("expert_dtype")
            if configured_dtype not in {None, "fp4"}:
                raise ValueError(
                    "DeepSeek-V4 target='mxfp4' requires expert_dtype='fp4', "
                    f"got {configured_dtype!r}"
                )
            resync_config["expert_dtype"] = "fp4"
        yield from export_resync_weights(weights, config, resync_config=resync_config)
    else:
        if resync_config:
            raise ValueError(
                "DeepSeek-V4 resync_config requires a quantized export target"
            )
        yield from weights


def save_hf_weights(
    model, path: str, config: DeepseekV4Config, ps: ParallelState, **kwargs
) -> None:
    """Export + write sharded safetensors via ``stream_export_to_shards``."""
    from megatron.lite.primitive.ckpt.hf_weights import stream_export_to_shards

    kwargs.pop("cpu", None)
    shard_size_bytes = int(kwargs.pop("shard_size_bytes", 5 * 1024**3))
    stream_export_to_shards(
        export_hf_weights(model, config, ps, rank0_only=True, cpu=True, **kwargs),
        path,
        shard_size_bytes=shard_size_bytes,
    )


__all__ = [
    "EXPERT_CLASSIFIER",
    "DeepseekV4WeightSpec",
    "PLACEMENT_FN",
    "export_hf_weights",
    "load_hf_weights",
    "save_hf_weights",
]
