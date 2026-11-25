# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""HF/BF16 and native release export, shared by training and deployment protocols.

Native targets use block-FP8 32x32 UE8M0 and MXFP4 experts. The paired vLLM
target keeps official BF16 wo_a unquantized, matching the training forward. Frozen tables
always retain original row-sharded FP8/E8M0 bytes. With include_frozen=True,
export streams TensorRowChunk objects for these tables; save_hf_weights drains
them directly to safetensors. Online sync can set include_frozen=False.
"""
import re

import torch

from megatron.lite.model.deepseek_v41.lite.checkpoint import source_name
from megatron.lite.primitive.ckpt.hf_weights import parse_expert_idx, unwrap_model
from megatron.lite.primitive.ckpt.row_export import iter_row_shards, save_row_shards
from megatron.lite.primitive.quantization.block_fp8 import quantize_block_fp8
from megatron.lite.primitive.quantization.mxfp4 import quantize_mxfp4

FP8_BLOCK = (32, 32)
_FP8_BLOCK_SUFFIXES = (
    ".attn.wq_a.weight",
    ".attn.wq_b.weight",
    ".attn.wkv.weight",
    ".attn.wo_a.weight",
    ".attn.wo_b.weight",
    ".attn.indexer.wq_b.weight",
    ".ffn.shared_experts.w1.weight",
    ".ffn.shared_experts.w2.weight",
    ".ffn.shared_experts.w3.weight",
    ".engram.wkv.weight",
)
_MXFP4 = re.compile(r"\.ffn\.experts\.\d+\.w[123]\.weight$")
_FP32 = re.compile(r"(\.hc_(attn|ffn)_(fn|base|scale)|\.attn\.attn_sink|\.ffn\.gate\.bias)$")
_EXPERT = re.compile(
    r"^layers\.(\d+)\.ffn\.(experts|shared_experts)\.linear_fc([12])\.weight(\d*)$"
)
_ENGRAM = re.compile(r"^engrams\.(\d+)\.(q_weight|k_weight|wkv\.weight)$")
_ALPHA = ("alpha_pre", "alpha_post", "alpha_res")


def release_format(key):
    """``"mxfp4"``, ``"fp8_block"`` or the release dtype of a release key."""
    if key.endswith(".tid2eid"):
        return torch.int64
    if _MXFP4.search(key):
        return "mxfp4"
    if key.endswith(_FP8_BLOCK_SUFFIXES):
        return "fp8_block"
    return torch.float32 if _FP32.search(key) else torch.bfloat16


def release_names(name):
    """Release key(s) for a global native name; ``#slot`` marks one of three mHC scales."""
    if name.endswith(".ape") or name.startswith("hash."):
        return []
    match = _EXPERT.fullmatch(name)
    if match:
        layer, kind, fc, expert = match.groups()
        prefix = f"layers.{layer}.ffn.{kind}." + (f"{expert}." if kind == "experts" else "")
        return [prefix + "w1.weight", prefix + "w3.weight"] if fc == "1" else [prefix + "w2.weight"]
    match = _ENGRAM.fullmatch(name)
    if match:
        return [f"layers.{match[1]}.engram.{match[2]}"]
    if name.startswith("engrams."):
        return []  # frozen tables
    if name.endswith(_ALPHA):
        return [f"{source_name(name)}#{_ALPHA.index(name.rsplit('.', 1)[1])}"]
    return [source_name(name)]


class DeepseekV41WeightSpec:
    """Shared-exporter spec: EP/PP gather of native tensors under release names."""

    def __init__(self, config):
        self.config = config

    @property
    def num_experts(self):
        return self.config.n_routed_experts

    def weight_map(self):
        return {}

    def load_weight_map(self, base_model, ps, logical_state_keys):
        del base_model, ps
        # Only persistent buffers consult this map: the router bias is exported,
        # Engram tables and hash constants are not.
        return {
            name: release_names(name)
            for name in logical_state_keys
            if name.endswith((".ffn.router.expert_bias", ".ffn.router.tid2eid"))
        }

    def native_to_hf(self, native_name, tensor):
        keys = release_names(native_name)
        if len(keys) == 2:
            return list(zip(keys, (t.contiguous() for t in tensor.chunk(2, dim=0))))
        return [(keys[0], tensor)] if keys else []

    def tp_spec(self, native_name):
        return None

    def qkv_spec(self, native_name):
        return None

    def is_expert(self, native_name):
        return ".ffn.experts." in native_name

    def expert_global_id(self, native_name):
        return parse_expert_idx(native_name) if self.is_expert(native_name) else None

    def expert_local_name(self, native_name, local_idx):
        return f"{native_name.rsplit('.weight', 1)[0]}.weight{local_idx}"


def to_release(key, tensor):
    """Serialize one release tensor (+ its scale for quantized storage)."""
    kind = release_format(key)
    if kind in ("mxfp4", "fp8_block"):
        from megatron.lite.primitive.ckpt.hf_weights import _dequantize_block_scaled_tensor

        quantize = (
            quantize_mxfp4
            if kind == "mxfp4"
            else lambda x: quantize_block_fp8(x, FP8_BLOCK, scale_format="e8m0")
        )
        codes, scale = quantize(tensor)
        # Canonicalize equivalent scale/code pairs: reloading BF16 then exporting
        # must not choose a different power-of-two scale at a rounding boundary.
        for _ in range(3):
            decoded = _dequantize_block_scaled_tensor(codes, scale, tensor.shape).bfloat16()
            again, descale = quantize(decoded)
            if torch.equal(scale.view(torch.uint8), descale.view(torch.uint8)) and torch.equal(
                codes.view(torch.uint8), again.view(torch.uint8)
            ):
                return [(key, codes), (key[: -len("weight")] + "scale", scale)]
            codes, scale = again, descale
        raise RuntimeError(f"Non-idempotent release encoding for {key}")
    return [(key, tensor.to(kind).contiguous())]


def export_hf_weights(chunks, model_cfg, ps, **kwargs):
    from megatron.lite.primitive.ckpt.hf_weights import export_hf_weights as gather

    target = kwargs.pop("target", "hf")
    if target not in ("release", "hf", "bf16", "vllm"):
        raise ValueError(f"Unknown DS4.1 export target, got {target!r}")
    if kwargs.pop("resync_config", None) or kwargs.get("export_dtype"):
        raise ValueError("DS4.1 release export owns dtypes and quantization; no overrides")
    kwargs.pop("export_dtype", None)
    include_frozen = kwargs.pop("include_frozen", True)
    row_bytes = kwargs.get("buffer_max_size_bytes", 64 * 1024**2)
    scales = {}
    for key, tensor in gather(
        chunks, DeepseekV41WeightSpec(model_cfg), ps, vocab_size=model_cfg.vocab_size, **kwargs
    ):
        if "#" in key:
            key, slot = key.split("#")
            parts = scales.setdefault(key, [None] * 3)
            parts[int(slot)] = tensor.reshape(1)
            if any(p is None for p in parts):
                continue
            tensor = torch.cat(scales.pop(key))
        yield from (
            [
                (
                    key,
                    tensor.to(
                        release_format(key)
                        if isinstance(release_format(key), torch.dtype)
                        else torch.bfloat16
                    ).contiguous(),
                )
            ]
            if target in ("hf", "bf16") or (target == "vllm" and key.endswith(".attn.wo_a.weight"))
            else to_release(key, tensor)
        )
    if scales:
        raise RuntimeError(f"Incomplete mHC scale export: {sorted(scales)[:4]}")
    if include_frozen:
        yield from _frozen_rows(chunks, row_bytes, kwargs.get("rank0_only", False))


def _frozen_rows(chunks, row_bytes, rank0_only):
    rows = []
    for chunk in chunks if isinstance(chunks, (list, torch.nn.ModuleList)) else [chunks]:
        for layer, module in unwrap_model(chunk).engrams.items():
            table = module.embed
            start = table.embedding.lookup.boundaries[table.group.rank()]
            for kind in ("weight", "scale"):
                value = getattr(table.embedding, kind)[: max(0, table.rows - start)]
                rows.append((f"layers.{layer}.engram.embed.{kind}", value, start, table.rows))
    yield from iter_row_shards(rows, max_bytes=row_bytes, rank0_only=rank0_only)


def save_hf_weights(chunks, path, model_cfg, ps, **kwargs):
    from megatron.lite.primitive.ckpt.hf_weights import stream_export_to_shards

    for chunk in chunks if isinstance(chunks, (list, torch.nn.ModuleList)) else [chunks]:
        sync = getattr(chunk, "start_param_sync", None)
        if callable(sync):
            sync(force_sync=True)
    kwargs.pop("cpu", None)
    include_frozen = kwargs.pop("include_frozen", True)
    shard = int(kwargs.pop("shard_size_bytes", 5 * 1024**3))
    stream_export_to_shards(
        export_hf_weights(
            chunks, model_cfg, ps, rank0_only=True, cpu=True, include_frozen=False, **kwargs
        ),
        path,
        shard_size_bytes=shard,
    )
    if include_frozen:
        save_row_shards(
            _frozen_rows(chunks, kwargs.get("buffer_max_size_bytes", 64 * 1024**2), True), path
        )
