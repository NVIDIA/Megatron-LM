# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Strict release-to-Core mapping; quantized storage is dequantized to BF16."""
import re

import torch

from megatron.lite.primitive.ckpt.hf_weights import SafeTensorReader

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
            return (
                prefix
                + "hc_"
                + group
                + "_"
                + {
                    "mapping_proj.weight": "fn",
                    "bias": "base",
                    "alpha_pre": "scale",
                    "alpha_post": "scale",
                    "alpha_res": "scale",
                }[key]
            )
    raise KeyError(f"Unmapped DS4.1 tensor {name}")


def _read(reader, key, shape):
    value = reader.get_tensor(key, target_shape=torch.Size(shape), target_dtype=torch.bfloat16)
    if tuple(value.shape) != tuple(shape):
        raise ValueError(
            f"DS4.1 release shape mismatch for {key}: {tuple(value.shape)} != {tuple(shape)}"
        )
    return value


def _experts(reader, name, tensor, cfg, ps):
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
    if fc == "1":
        shape = (tensor.shape[0] // 2, tensor.shape[1])
        return torch.cat([_read(reader, prefix + p + ".weight", shape) for p in ("w1", "w3")])
    return _read(reader, prefix + "w2.weight", tensor.shape)


def _engram(reader, name, tensor, model):
    match = re.fullmatch(r"engrams\.(\d+)\.(.*)", name)
    if match is None:
        return None
    layer, suffix = match.groups()
    prefix = f"layers.{layer}.engram."
    if suffix == "embed.embedding.weight":
        table = model.engrams[layer].embed
        start = table.embedding.lookup.boundaries[table.group.rank()]
        stop = min(start + tensor.shape[0], table.rows)
        raw = reader._get_raw_tensor(prefix + "embed.weight", torch.device("cpu"))
        scales = (
            reader._get_raw_tensor(prefix + "embed.scale", torch.device("cpu"))
            if reader.has_tensor(prefix + "embed.scale")
            else None
        )
        tensor.zero_()
        for begin in range(start, stop, 4096):
            end = min(begin + 4096, stop)
            value = raw[begin:end].float()
            if scales is not None:
                value = (
                    value.unflatten(-1, (-1, 32)) * scales[begin:end].float().unsqueeze(-1)
                ).flatten(-2)
            tensor[begin - start : end - start].copy_(value.to(torch.bfloat16))
        return tensor
    key = {"wkv.weight": "wkv.weight", "q_weight": "q_weight", "k_weight": "k_weight"}.get(suffix)
    if key is None:
        raise KeyError(f"Unmapped DS4.1 Engram tensor {name}")
    value = _read(reader, prefix + key, tensor.shape)
    return value


def _record_reads(reader):
    """Record every tensor the reader materializes, quantization sidecars included."""
    keys, read = set(), reader._get_raw_tensor

    def recording(name, device):
        keys.add(name)
        return read(name, device)

    reader._get_raw_tensor = recording
    return keys


def _stage_keys(index, model, cfg, ps):
    """Release keys this pipeline stage and EP rank must consume."""
    start, end = model.local_layer_range
    local = cfg.n_routed_experts // ps.ep_size
    top = {"embed.weight"} if model.layout.has_embed else set()
    top |= {"head.weight", "norm.weight"} if model.layout.has_head else set()
    keys = set()
    for key in index:
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
        or n.endswith("embed.embedding.weight")
        or n.endswith("tid2eid")
    ]
    loaded = []
    with SafeTensorReader(hf_path) as reader:
        read = _record_reads(reader)
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
        # No silently unmapped state on either side.
        unread = sorted(_stage_keys(reader.index, model, model_cfg, ps) - read)
        unloaded = sorted(
            n
            for m, module in model.named_modules()
            for n in (f"{m}.{b}".lstrip(".") for b in _local_tensors(module))
            if n not in loaded and not n.endswith(".ape")
        )
        if unread or unloaded:
            raise ValueError(f"DS4.1 release load: unread {unread[:8]}, unloaded {unloaded[:8]}")
    model.ds41_loaded_tensors = tuple(loaded)


def _local_tensors(module):
    """Parameters and persistent buffers owned directly by ``module``."""
    tensors = dict(module.named_parameters(recurse=False))
    tensors.update(
        (n, b)
        for n, b in module.named_buffers(recurse=False)
        if n not in module._non_persistent_buffers_set
    )
    return tensors


def _sharded_state(module, prefix, offsets, ps, metadata):
    """Core sharded state for the DS4.1 tree.

    Native Core modules describe themselves (expert offsets, TP axes). A module
    whose rows are sharded over ``lookup.group`` (frozen Engram table) keeps each
    rank's rows at their global offset. Other lite modules are replicated.
    """
    from megatron.core.dist_checkpointing.mapping import ShardedTensor
    from megatron.core.transformer.module import MegatronModule
    from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

    lookup = getattr(module, "lookup", None)
    if getattr(lookup, "group", None) is not None:
        size, rank = lookup.group.size(), lookup.group.rank()
        shard = lookup.boundaries[1]
        if tuple(lookup.boundaries) != tuple(i * shard for i in range(size + 1)):
            raise ValueError("DS4.1 row-sharded checkpoint requires equal Core row shards")
        axis = len(offsets)  # rows follow any prepended (pipeline) axes
        return {
            prefix
            + name: ShardedTensor.from_rank_offsets(
                prefix + name,
                tensor,
                *offsets,
                (axis, rank, size),
                replica_id=(0, 0, ps.dp_cp_rank // size),
                prepend_axis_num=axis,
            )
            for name, tensor in _local_tensors(module).items()
        }
    rows = [m for m in module.modules() if getattr(getattr(m, "lookup", None), "group", None)]
    if isinstance(module, MegatronModule) and not rows:
        return module.sharded_state_dict(prefix, offsets, metadata)
    state = make_sharded_tensors_for_checkpoint(
        _local_tensors(module), prefix, {}, offsets, dp_cp_group=ps.dp_cp_group
    )
    for name, child in module.named_children():
        state.update(_sharded_state(child, f"{prefix}{name}.", offsets, ps, metadata))
    return state


def attach_checkpoint_metadata(chunks, ps):
    """Opt DS4.1 chunks into the Core dist-opt checkpoint (model + optimizer state)."""
    from types import MethodType

    for chunk in chunks:
        model = chunk
        while hasattr(model, "module"):
            model = model.module

        def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None, _m=model):
            # Explicit groups: Core's defaults would consult the global mpu state.
            metadata = dict(metadata or {}, dp_cp_group=ps.dp_cp_group)
            state = {}
            for name, child in _m.named_children():
                state.update(
                    _sharded_state(child, f"{prefix}{name}.", sharded_offsets, ps, metadata)
                )
            return state

        chunk.sharded_state_dict = MethodType(sharded_state_dict, chunk)
        chunk._mlite_dist_opt_sharded_state_dict = True
        chunk._mlite_dist_opt_parallel_state = ps
