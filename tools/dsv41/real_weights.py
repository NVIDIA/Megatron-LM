# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Released DeepSeek-V4.1-Flash weights for the alignment tool (milestone M2).

Reads the first ``n_layers`` model layers (plus embedding, final norm and head) straight from
the Hugging Face safetensors shards, dequantises them to bf16 and distributes them to the
official ``inference/model.py`` model (which shards over ``world_size`` like ``convert.py``)
and to the Megatron model (expert-parallel experts, row-sharded Engram table, everything
else replicated at TP 1).

Quantised formats of the release (``config.json`` ``quantization_config``):

* dense linears and the Engram projection: ``float8_e4m3fn`` weights with one ``e8m0`` scale
  per 32 x 32 block (``scale.shape == ceil(weight.shape / 32)``);
* routed experts: FP4 E2M1 packed two values per byte (``int8`` ``[out, in / 2]``) with one
  ``e8m0`` scale per row and 32 input elements;
* Engram tables: ``float8_e4m3fn`` rows with one ``e8m0`` scale per row and 32 columns;
* everything else (embedding, head, norms, router, compressor, indexer projections, sinks,
  hyper-connection tensors) is stored in bf16 or fp32 as is.

The mapping of official names to Megatron names is the contract of
``align_with_reference.copy_weights`` and of the checkpoint importer. Independent
implementation (official ``convert.py`` / ``kernel.py`` read for the formats).
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch

_FP4_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


# ---------------------------------------------------------------------------------------------
# Dequantisation
# ---------------------------------------------------------------------------------------------


def e8m0_to_float(scale: torch.Tensor) -> torch.Tensor:
    """``e8m0`` (or its ``uint8`` byte form) scales as fp32 powers of two."""
    if scale.dtype == torch.uint8:
        return torch.ldexp(torch.ones_like(scale, dtype=torch.float32), scale.to(torch.int32) - 127)
    return scale.to(torch.float32)


def _block_size(dim: int, n_scales: int) -> int:
    """Block width of a block-scaled axis. The scale count is ``ceil(dim / block)``; a partial
    last block makes ``ceil(dim / n_scales)`` too small, so prefer the power-of-two width that
    reproduces the scale count exactly (the released tensors use 32 or 128)."""
    for block in (32, 128, 64, 16, 256, 8):
        if -(-dim // block) == n_scales:
            return block
    return -(-dim // n_scales)


def dequant_fp8_blocks(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP8 weight ``[M, N]`` with one scale per ``bm x bn`` block (``ceil(M/bm) x ceil(N/bn)``
    scales) -> bf16."""
    m, n = weight.shape
    sm, sn = scale.shape
    bm, bn = _block_size(m, sm), _block_size(n, sn)
    s = e8m0_to_float(scale)
    s = s.repeat_interleave(bm, dim=0)[:m].repeat_interleave(bn, dim=1)[:, :n]
    return (weight.to(torch.float32) * s).to(torch.bfloat16)


def dequant_fp8_rows(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP8 rows ``[R, D]`` with one scale per row and ``D / scale.shape[1]`` columns -> bf16
    (Engram tables)."""
    block = weight.shape[1] // scale.shape[1]
    s = e8m0_to_float(scale).repeat_interleave(block, dim=1)
    return (weight.to(torch.float32) * s).to(torch.bfloat16)


def dequant_fp4_packed(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP4 E2M1 values packed two per byte ``[out, in / 2]`` (low nibble first) with one
    ``e8m0`` scale per row and 32 input elements -> bf16 ``[out, in]``."""
    u8 = packed.view(torch.uint8)
    table = _FP4_E2M1.to(packed.device)
    values = torch.stack([table[(u8 & 0x0F).long()], table[(u8 >> 4).long()]], dim=-1)
    values = values.reshape(u8.shape[0], -1)
    block = values.shape[1] // scale.shape[1]
    s = e8m0_to_float(scale).repeat_interleave(block, dim=1)
    return (values * s).to(torch.bfloat16)


# ---------------------------------------------------------------------------------------------
# Snapshot access
# ---------------------------------------------------------------------------------------------


@dataclass
class Snapshot:
    """Lazy reader over the safetensors shards of one Hugging Face snapshot."""

    root: str
    weight_map: Dict[str, str]
    text_config: dict

    @classmethod
    def open(cls, root: str) -> "Snapshot":
        index = json.load(open(os.path.join(root, "model.safetensors.index.json")))
        config = json.load(open(os.path.join(root, "config.json")))
        return cls(root, index["weight_map"], config.get("text_config", config))

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def names(self, pattern: str) -> List[str]:
        rx = re.compile(pattern)
        return sorted(k for k in self.weight_map if rx.fullmatch(k))

    def tensor(self, name: str, rows: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Load one tensor (optionally a row range) on CPU."""
        from safetensors import safe_open

        with safe_open(
            os.path.join(self.root, self.weight_map[name]), framework="pt", device="cpu"
        ) as f:
            if rows is None:
                return f.get_tensor(name)
            r0, r1 = rows
            return f.get_slice(name)[r0:r1]

    def dequant(self, name: str, rows: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """``<name>`` (a ``.weight``) dequantised to bf16 when it carries a ``.scale`` sibling;
        other tensors are returned unchanged (row range applies to both)."""
        weight = self.tensor(name, rows)
        scale_name = name[: -len(".weight")] + ".scale" if name.endswith(".weight") else None
        if scale_name is None or not self.has(scale_name):
            return weight
        if weight.dtype == torch.int8:
            return dequant_fp4_packed(weight, self.tensor(scale_name, rows))
        if weight.dtype == torch.float8_e4m3fn:
            scale = self.tensor(scale_name, rows)
            if scale.shape[0] == weight.shape[0]:
                return dequant_fp8_rows(weight, scale)
            if rows is not None:
                raise ValueError(f"{name}: block-scaled tensors cannot be row-sliced")
            return dequant_fp8_blocks(weight, scale)
        return weight


# ---------------------------------------------------------------------------------------------
# Truncated configuration
# ---------------------------------------------------------------------------------------------


def truncated_text_config(text_config: dict, n_layers: int) -> dict:
    """The released text configuration restricted to its first ``n_layers`` model layers.

    Source lists keep only layers below ``n_layers``; a candidate source at or beyond the cut
    disables candidate blocks; Engram tables beyond the cut are dropped (with their sizes).
    """
    c = dict(text_config)
    c["num_hidden_layers"] = n_layers
    c["compress_ratios"] = list(text_config["compress_ratios"][:n_layers])
    c["kv_source_layer_ids"] = [i for i in text_config["kv_source_layer_ids"] if i < n_layers]
    c["index_source_layer_ids"] = [i for i in text_config["index_source_layer_ids"] if i < n_layers]
    if text_config.get("candidate_source_layer_id", -1) >= n_layers:
        c["candidate_source_layer_id"] = -1
    keep = [k for k, lid in enumerate(text_config["engram_layer_ids"]) if lid < n_layers]
    c["engram_layer_ids"] = [text_config["engram_layer_ids"][k] for k in keep]
    c["engram_num_embeddings"] = [text_config["engram_num_embeddings"][k] for k in keep]
    c["num_nextn_predict_layers"] = 0
    return c


# ---------------------------------------------------------------------------------------------
# Distribution rules
# ---------------------------------------------------------------------------------------------

# Official model: parameters sharded over ``world_size`` (dimension), as in ``convert.py``.
_OFFICIAL_SHARD_DIM = {
    "embed.weight": 0,
    "head.weight": 0,
    "wq_b.weight": 0,
    "wo_a.weight": 0,
    "wo_b.weight": 1,
    "attn_sink": 0,
    "weights_proj.weight": 0,
}


def official_shard(
    name: str, tensor: torch.Tensor, rank: int, world: int
) -> Optional[torch.Tensor]:
    """The slice of ``tensor`` that official rank ``rank`` of ``world`` holds, or None when the
    rank holds nothing of it (routed experts of other ranks)."""
    if world == 1:
        return tensor
    if re.search(r"\.ffn\.experts\.(\d+)\.", name):
        raise RuntimeError("route expert tensors through official_expert_owner")
    if ".engram.embed." in name:
        rows = -(-tensor.shape[0] // world)
        part = tensor[rank * rows : (rank + 1) * rows]
        if part.shape[0] < rows:
            pad_value = 1 if name.endswith(".scale") else 0
            pad = tensor.new_full((rows - part.shape[0], *tensor.shape[1:]), pad_value)
            part = torch.cat([part, pad])
        return part.contiguous()
    for suffix, dim in _OFFICIAL_SHARD_DIM.items():
        if name == suffix or name.endswith("." + suffix):
            size = tensor.shape[dim] // world
            return tensor.narrow(dim, rank * size, size).contiguous()
    return tensor


def official_expert_owner(expert: int, n_experts: int, world: int) -> Tuple[int, int]:
    """``(rank, local index)`` of a routed expert in the official sharding."""
    n_local = n_experts // world
    return expert // n_local, expert % n_local


# ---------------------------------------------------------------------------------------------
# Loading into the two models
# ---------------------------------------------------------------------------------------------


def _assign(param: torch.Tensor, value: torch.Tensor, what: str) -> None:
    if tuple(param.shape) != tuple(value.shape):
        raise ValueError(f"{what}: parameter {tuple(param.shape)} vs source {tuple(value.shape)}")
    with torch.no_grad():
        param.copy_(value.to(device=param.device, dtype=param.dtype))


@torch.no_grad()
def load_official(model, snap: Snapshot, n_layers: int, rank: int, world: int, log=print) -> None:
    """Fill the official model (already built for ``n_layers`` on ``world`` ranks)."""
    params = dict(model.named_parameters())
    params.update(dict(model.named_buffers()))
    n_experts = int(snap.text_config["n_routed_experts"])
    seen = set()

    def put(name: str, value: torch.Tensor) -> None:
        _assign(params[name], value, name)
        seen.add(name)

    for name in ("embed.weight", "head.weight", "norm.weight"):
        put(name, official_shard(name, snap.dequant(name), rank, world))
    for layer in range(n_layers):
        prefix = f"layers.{layer}."
        for name in snap.names(re.escape(prefix) + r".*"):
            if ".ffn.experts." in name:
                expert = int(re.search(r"\.ffn\.experts\.(\d+)\.", name).group(1))
                owner, local = official_expert_owner(expert, n_experts, world)
                if owner != rank or name.endswith(".scale"):
                    continue
                put(name, snap.dequant(name))
            elif ".engram.embed." in name:
                # kept fp8 + e8m0 in the official model: raw shards
                put(name, official_shard(name, snap.tensor(name), rank, world))
            elif name.endswith(".scale"):
                continue  # consumed by dequant of the matching .weight
            elif name.endswith("gate.bias_vl"):
                continue  # vision routing bias, not part of the text model
            else:
                put(name, official_shard(name, snap.dequant(name), rank, world))
    missing = sorted(k for k in params if k not in seen and not _official_transient(k))
    if missing:
        raise RuntimeError(f"official parameters not loaded: {missing[:20]}")
    log(f"official rank {rank}: {len(seen)} tensors loaded")


def _official_transient(name: str) -> bool:
    """Buffers the official model derives itself (caches, RoPE tables, hash constants)."""
    return any(
        s in name
        for s in ("freqs_cis", "_cache", "kv_state", "score_state", "engram_hash.", "cache")
    )


@torch.no_grad()
def load_megatron(
    mg_model, snap: Snapshot, n_layers: int, ep_rank: int, ep_size: int, log=print
) -> None:
    """Fill the Megatron HybridModel (TP 1, EP ``ep_size``): experts by expert-parallel rank,
    Engram rows by shard, everything else replicated."""
    dst = dict(mg_model.named_parameters())
    dst.update(dict(mg_model.named_buffers()))
    used = set()
    n_experts = int(snap.text_config["n_routed_experts"])
    n_local = n_experts // ep_size

    def put(name: str, value: torch.Tensor) -> None:
        if name not in dst:
            import difflib

            close = difflib.get_close_matches(name, list(dst), n=5, cutoff=0.6)
            raise KeyError(f"Megatron parameter missing: {name}; closest: {close}")
        _assign(dst[name], value, name)
        used.add(name)

    put("embedding.word_embeddings.weight", snap.dequant("embed.weight"))
    put("output_layer.weight", snap.dequant("head.weight"))
    put("decoder.final_norm.weight", snap.dequant("norm.weight"))

    for i in range(n_layers):
        attn, ffn = f"decoder.layers.{2 * i}", f"decoder.layers.{2 * i + 1}"
        inner_a, inner_f = f"{attn}.inner_layer", f"{ffn}.inner_layer"
        for prefix, mg in ((f"layers.{i}.hc_attn", attn), (f"layers.{i}.hc_ffn", ffn)):
            put(f"{mg}.hyper_connection.mapping_proj.weight", snap.dequant(f"{prefix}_fn"))
            put(f"{mg}.hyper_connection.bias", snap.dequant(f"{prefix}_base"))
            scale = snap.dequant(f"{prefix}_scale")
            put(f"{mg}.hyper_connection.alpha_pre", scale[0:1])
            put(f"{mg}.hyper_connection.alpha_post", scale[1:2])
            put(f"{mg}.hyper_connection.alpha_res", scale[2:3])
        put(f"{inner_a}.input_layernorm.weight", snap.dequant(f"layers.{i}.attn_norm.weight"))
        put(f"{inner_f}.pre_mlp_layernorm.weight", snap.dequant(f"layers.{i}.ffn_norm.weight"))

        sa = f"{inner_a}.self_attention"
        a = f"layers.{i}.attn"
        put(f"{sa}.linear_q_down_proj.weight", snap.dequant(f"{a}.wq_a.weight"))
        put(f"{sa}.q_layernorm.weight", snap.dequant(f"{a}.q_norm.weight"))
        put(f"{sa}.linear_q_up_proj.weight", snap.dequant(f"{a}.wq_b.weight"))
        put(f"{sa}.linear_kv_proj.weight", snap.dequant(f"{a}.wkv.weight"))
        put(f"{sa}.kv_layernorm.weight", snap.dequant(f"{a}.kv_norm.weight"))
        put(f"{sa}.linear_o_group_proj", snap.dequant(f"{a}.wo_a.weight"))
        put(f"{sa}.linear_proj.weight", snap.dequant(f"{a}.wo_b.weight"))
        core = f"{sa}.core_attention"
        put(f"{core}.attn_sink", snap.dequant(f"{a}.attn_sink"))
        if snap.has(f"{a}.compressor.wkv.weight"):
            put(f"{core}.compressor.linear_wkv.weight", snap.dequant(f"{a}.compressor.wkv.weight"))
            put(f"{core}.compressor.norm.weight", snap.dequant(f"{a}.compressor.norm.weight"))
            if snap.has(f"{a}.compressor.wgate.weight"):
                put(
                    f"{core}.compressor.linear_wgate.weight",
                    snap.dequant(f"{a}.compressor.wgate.weight"),
                )
        if snap.has(f"{a}.indexer.wq_b.weight"):
            put(f"{core}.indexer.linear_wq_b.weight", snap.dequant(f"{a}.indexer.wq_b.weight"))
            put(
                f"{core}.indexer.linear_weights_proj.weight",
                snap.dequant(f"{a}.indexer.weights_proj.weight"),
            )
            if snap.has(f"{a}.indexer.wk.weight"):
                put(f"{core}.indexer.linear_wk.weight", snap.dequant(f"{a}.indexer.wk.weight"))
                put(f"{core}.indexer.k_norm.weight", snap.dequant(f"{a}.indexer.k_norm.weight"))

        mlp = f"{inner_f}.mlp"
        f_ = f"layers.{i}.ffn"
        put(f"{mlp}.router.weight", snap.dequant(f"{f_}.gate.weight"))
        put(f"{mlp}.router.expert_bias", snap.dequant(f"{f_}.gate.bias"))
        grouped = f"{mlp}.experts.linear_fc1.weight0" in dst
        for local in range(n_local):
            e = ep_rank * n_local + local
            w1 = snap.dequant(f"{f_}.experts.{e}.w1.weight")
            w3 = snap.dequant(f"{f_}.experts.{e}.w3.weight")
            w2 = snap.dequant(f"{f_}.experts.{e}.w2.weight")
            if grouped:
                put(f"{mlp}.experts.linear_fc1.weight{local}", torch.cat([w1, w3], dim=0))
                put(f"{mlp}.experts.linear_fc2.weight{local}", w2)
            else:
                put(
                    f"{mlp}.experts.local_experts.{local}.linear_fc1.weight",
                    torch.cat([w1, w3], dim=0),
                )
                put(f"{mlp}.experts.local_experts.{local}.linear_fc2.weight", w2)
        w1 = snap.dequant(f"{f_}.shared_experts.w1.weight")
        w3 = snap.dequant(f"{f_}.shared_experts.w3.weight")
        put(f"{mlp}.shared_experts.linear_fc1.weight", torch.cat([w1, w3], dim=0))
        put(
            f"{mlp}.shared_experts.linear_fc2.weight",
            snap.dequant(f"{f_}.shared_experts.w2.weight"),
        )

        if snap.has(f"layers.{i}.engram.wkv.weight"):
            eg = f"{attn}.engram"
            target = dst[f"{eg}.embedding_rows"]
            rows_per_shard = target.shape[0]
            r0 = ep_rank * rows_per_shard
            # The stored table has ``total`` rows; the shard may extend past it (zero padding).
            weight_slice = snap.tensor(
                f"layers.{i}.engram.embed.weight", rows=(r0, r0 + rows_per_shard)
            )
            scale_slice = snap.tensor(
                f"layers.{i}.engram.embed.scale", rows=(r0, r0 + rows_per_shard)
            )
            table = torch.zeros(rows_per_shard, target.shape[1], dtype=torch.bfloat16)
            chunk = 8_000_000
            for c0 in range(0, weight_slice.shape[0], chunk):
                w = weight_slice[c0 : c0 + chunk].to(target.device)
                s = scale_slice[c0 : c0 + chunk].to(target.device)
                table[c0 : c0 + w.shape[0]] = dequant_fp8_rows(w, s).cpu()
            put(f"{eg}.embedding_rows", table)
            put(f"{eg}.linear_wkv.weight", snap.dequant(f"layers.{i}.engram.wkv.weight"))
            put(f"{eg}.q_weight", snap.dequant(f"layers.{i}.engram.q_weight"))
            put(f"{eg}.k_weight", snap.dequant(f"layers.{i}.engram.k_weight"))

    params = dict(mg_model.named_parameters())
    missing = [k for k in params if k not in used]
    if missing:
        raise RuntimeError(f"Megatron parameters not covered: {missing[:20]}")
    unmatched_buffers = [
        k for k in dst if k not in used and k not in params and not _megatron_transient(k)
    ]
    if unmatched_buffers:
        log(f"megatron buffers left at their defaults: {unmatched_buffers[:20]}")
    log(f"megatron ep rank {ep_rank}: {len(used)} tensors loaded")


def _megatron_transient(name: str) -> bool:
    """Buffers Megatron derives itself (hash constants, RoPE caches, router statistics)."""
    return any(
        s in name
        for s in (
            "engram_hasher.",
            "inv_freq",
            "local_tokens_per_expert",
            "expert_bias_",
            "_extra_state",
            "rotary_pos_emb",
        )
    ) or name.endswith("router.local_tokens_per_expert")
