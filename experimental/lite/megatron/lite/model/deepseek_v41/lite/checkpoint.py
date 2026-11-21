# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Strict release-to-Core mapping; quantized storage is dequantized to BF16."""
import re
from pathlib import Path

import torch

from megatron.lite.primitive.ckpt.row_reader import RowSliceReader
from megatron.lite.primitive.ckpt.row_state import local_tensors

_ATTENTION = {
    "linear_q_down_proj.weight": "wq_a.weight",
    "q_layernorm.weight": "q_norm.weight",
    "linear_q_up_proj.weight": "wq_b.weight",
    "linear_kv_proj.weight": "wkv.weight",
    "kv_layernorm.weight": "kv_norm.weight",
    "linear_o_group_proj": "wo_a.weight",
    "linear_proj.weight": "wo_b.weight",
}


def source_name(name):
    """Return release keys, without local/global expert-number ambiguity."""
    if name in {"embed.weight", "head.weight", "norm.weight"}:
        return name
    match = re.fullmatch(r"layers\.(\d+)\.(.*)", name)
    if match is None:
        raise KeyError(f"Unmapped DS4.1 tensor {name}")
    layer, suffix = match.groups()
    prefix = f"layers.{layer}."
    if suffix in {"attn_norm.weight", "ffn_norm.weight"}:
        return prefix + suffix
    if suffix.startswith("attention."):
        key = suffix.removeprefix("attention.")
        if key in _ATTENTION:
            return prefix + "attn." + _ATTENTION[key]
        key = key.removeprefix("core_attention.")
        if key.startswith("native."):
            key = key.removeprefix("native.")
        if key == "attn_sink":
            return prefix + "attn.attn_sink"
        replacements = {
            "compressor.linear_wkv.weight": "compressor.wkv.weight",
            "compressor.linear_wgate.weight": "compressor.wgate.weight",
            "compressor.norm.weight": "compressor.norm.weight",
            "indexer.linear_wq_b.linear.weight": "indexer.wq_b.weight",
            "indexer.linear_weights_proj.linear.weight": "indexer.weights_proj.weight",
            "indexer.compressor.wk.weight": "indexer.wk.weight",
            "indexer.compressor.norm.weight": "indexer.k_norm.weight",
            "main_projection.weight": "compressor.wkv.weight",
            "main_norm.weight": "compressor.norm.weight",
            "index_key.0.weight": "indexer.wk.weight",
            "index_key.1.weight": "indexer.k_norm.weight",
            "index_q.linear.weight": "indexer.wq_b.weight",
            "index_weights.linear.weight": "indexer.weights_proj.weight",
        }
        if key in replacements:
            return prefix + "attn." + replacements[key]
    if suffix.startswith("ffn.router."):
        return (
            prefix
            + "ffn.gate."
            + {"weight": "weight", "expert_bias": "bias", "tid2eid": "tid2eid"}[
                suffix.removeprefix("ffn.router.")
            ]
        )
    for group in ("attn", "ffn"):
        key = suffix.removeprefix(group + "_mixes.")
        if key != suffix:
            mapped = {
                "mapping_proj.weight": "fn",
                "bias": "base",
                "alpha_pre": "scale",
                "alpha_post": "scale",
                "alpha_res": "scale",
            }[key]
            return f"{prefix}hc_{group}_{mapped}"
    raise KeyError(f"Unmapped DS4.1 tensor {name}")


def _read(reader, key, shape):
    value = reader.get_tensor(key, target_shape=torch.Size(shape), target_dtype=torch.bfloat16)
    if tuple(value.shape) != tuple(shape):
        raise ValueError(
            f"DS4.1 release shape mismatch for {key}: {tuple(value.shape)} != {tuple(shape)}"
        )
    return value


def _expert_sources(name, cfg, ps):
    match = re.fullmatch(
        r"layers\.(\d+)\.ffn\.(experts|shared_experts)\.linear_fc([12])\.weight(\d*)", name
    )
    if match is None:
        return None
    layer, kind, fc, local = match.groups()
    prefix = f"layers.{layer}.ffn."
    if kind == "experts":
        if not local:
            raise ValueError("DS4.1 release loader requires Core per-expert grouped weights")
        expert = int(local) + ps.ep_rank * (cfg.n_routed_experts // ps.ep_size)
        prefix += f"experts.{expert}."
    else:
        prefix += "shared_experts."
    return [prefix + part + ".weight" for part in (("w1", "w3") if fc == "1" else ("w2",))]


def _experts(reader, name, tensor, cfg, ps):
    keys = _expert_sources(name, cfg, ps)
    if keys is None:
        return None
    if len(keys) == 1:
        return _read(reader, keys[0], tensor.shape)
    shape = (tensor.shape[0] // 2, tensor.shape[1])
    return torch.cat([_read(reader, key, shape) for key in keys])


def prefetch_engram(reader, table, layer):
    """Fill final native host buffers while subsequent layers/TE modules build."""
    start = table.embedding.lookup.boundaries[table.group.rank()]
    stop = min(start + table.embedding.weight.shape[0], table.rows)
    reader.prefetch_row_copies(
        [
            (f"layers.{layer}.engram.embed.{kind}", start, stop, getattr(table.embedding, kind))
            for kind in ("weight", "scale")
        ]
    )


def _prefetch_keys(state, cfg, ps):
    for name, _ in state:
        if name.endswith(".ape"):
            continue
        keys = _expert_sources(name, cfg, ps)
        if keys is not None:
            yield from keys
        elif name.startswith("engrams."):
            _, layer, suffix = name.split(".", 2)
            if not suffix.startswith("embed.embedding."):
                yield f"layers.{layer}.engram.{suffix}"
        else:
            yield source_name(name)


def _engram(reader, name, tensor, model):
    match = re.fullmatch(r"engrams\.(\d+)\.(.*)", name)
    if match is None:
        return None
    layer, suffix = match.groups()
    prefix = f"layers.{layer}.engram."
    if suffix in {"embed.embedding.weight", "embed.embedding.scale"}:
        table = model.engrams[layer].embed
        start = table.embedding.lookup.boundaries[table.group.rank()]
        stop = min(start + tensor.shape[0], table.rows)
        kind = suffix.rsplit(".", 1)[1]
        if not hasattr(reader, "_engram_rows"):
            reader._engram_rows = {}
        if layer not in reader._engram_rows:
            requests = [(prefix + "embed." + k, start, stop) for k in ("weight", "scale")]
            reader._engram_rows[layer] = reader.get_row_tensors(requests)
        raw = reader._engram_rows[layer].pop(prefix + "embed." + kind)
        if not reader._engram_rows[layer]:
            del reader._engram_rows[layer]
        expected = torch.float8_e4m3fn if kind == "weight" else torch.float8_e8m0fnu
        if raw.dtype != expected or tensor.dtype != expected or raw.shape[1] != tensor.shape[1]:
            raise ValueError("Engram release storage must retain FP8 weight and E8M0/32 scales")
        if raw is not tensor:
            tensor[stop - start :].zero_()
            tensor[: stop - start].copy_(raw)
        return tensor
    if suffix not in {"wkv.weight", "q_weight", "k_weight"}:
        raise KeyError(f"Unmapped DS4.1 Engram tensor {name}")
    value = _read(reader, prefix + suffix, tensor.shape)
    return value


def _record_reads(reader):
    """Record every tensor the reader materializes, quantization sidecars included."""
    keys, read = set(), reader._get_raw_tensor

    def recording(name, device):
        keys.add(name)
        return read(name, device)

    row_read = reader.get_row_tensors

    def row_recording(requests):
        values = row_read(requests)
        keys.update(values)
        return values

    reader._get_raw_tensor, reader.get_row_tensors = recording, row_recording
    return keys


def _stage_keys(index, model, cfg, ps):
    """Release keys this pipeline stage and EP rank must consume."""
    start, end = model.local_layer_range
    local = cfg.n_routed_experts // ps.ep_size
    top = {"embed.weight"} if model.layout.has_embed else set()
    top |= {"head.weight", "norm.weight"} if model.layout.has_head else set()
    keys = set()
    for key in index:
        if re.fullmatch(r"layers\.\d+\.ffn\.gate\.bias_vl", key):
            continue  # Release vision-only router bias; text routing uses ffn.gate.bias.
        layer = re.match(r"layers\.(\d+)\.(?:ffn\.experts\.(\d+)\.)?", key)
        if layer is None:
            keys |= {key} & top
        elif start <= int(layer[1]) < end and (
            layer[2] is None or 0 <= int(layer[2]) - ps.ep_rank * local < local
        ):
            keys.add(key)
    return keys


@torch.no_grad()
def load_hf_weights(chunk, hf_path, model_cfg, ps):
    model = chunk
    while hasattr(model, "module"):
        model = model.module
    state = list(model.named_parameters())
    # Core router bias and frozen Engram tables are persistent training state.
    state += [
        (n, v)
        for n, v in model.named_buffers()
        if n.endswith("expert_bias")
        or n.endswith(("embed.embedding.weight", "embed.embedding.scale"))
        or n.endswith("tid2eid")
    ]
    loaded = []
    reader = getattr(model, "_load_reader", None)
    model._load_reader = None
    if reader is not None and reader.path.resolve() != Path(hf_path).resolve():
        reader.close()
        raise ValueError("Prefetch source differs from the requested release directory")
    with reader or RowSliceReader(hf_path, device=next(model.parameters()).device) as reader:
        read = _record_reads(reader)
        if getattr(getattr(model, "impl", None), "prefetch_weights", False):
            reader.prefetch_tensors(_prefetch_keys(state, model_cfg, ps))
        for name, tensor in state:
            if name.endswith(".ape"):
                if torch.count_nonzero(tensor):
                    raise ValueError("DS4.1 compressor APE must stay zero")
                continue
            value = _experts(reader, name, tensor, model_cfg, ps)
            if value is None:
                value = _engram(reader, name, tensor, model)
            if value is None:
                key = source_name(name)
                if name.endswith((".alpha_pre", ".alpha_post", ".alpha_res")):
                    index = ("alpha_pre", "alpha_post", "alpha_res").index(name.rsplit(".", 1)[1])
                    value = _read(reader, key, (3,))[index : index + 1]
                else:
                    value = _read(reader, key, tensor.shape)
            tensor.copy_(value.to(device=tensor.device, dtype=tensor.dtype))
            loaded.append(name)
        unread = sorted(_stage_keys(reader.index, model, model_cfg, ps) - read)
        unloaded = sorted(
            n
            for m, module in model.named_modules()
            for n in (f"{m}.{b}".lstrip(".") for b in local_tensors(module))
            if n not in loaded and not n.endswith(".ape")
        )
        if unread or unloaded:
            raise ValueError(f"DS4.1 release load: unread {unread[:8]}, unloaded {unloaded[:8]}")
    model.ds41_loaded_tensors = tuple(loaded)
