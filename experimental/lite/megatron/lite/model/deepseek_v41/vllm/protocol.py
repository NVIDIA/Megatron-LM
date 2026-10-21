# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native Lite lifecycle/DistOpt; deployment operator model selection only."""
from dataclasses import dataclass

from megatron.lite.model.deepseek_v41.lite import protocol as native
from megatron.lite.model.deepseek_v41.lite.protocol import (
    build_model_config,
    forward_step,
    load_hf_weights,
    unpack_forward_output,
)


@dataclass(frozen=True)
class ImplConfig(native.ImplConfig):
    attention_core: str = "flashmla"
    head_dtype: str = "float32"


def validate_impl_cfg_keys(values):
    native.validate_impl_cfg_keys(
        {k: v for k, v in values.items() if k not in ("attention_core", "head_dtype")}
    )


def release_token_map(hf_path):
    """Engram compressed token map, built by vLLM's own rule from the release tokenizer."""
    from transformers import AutoTokenizer
    from vllm.models.deepseek_v41.common.engram import build_compressed_token_map

    return build_compressed_token_map(AutoTokenizer.from_pretrained(hf_path))[0]


def build_model(model_cfg, *, impl_cfg):
    from dataclasses import replace
    from pathlib import Path

    from .model import DeepseekV41Model

    if getattr(impl_cfg, "attention_core", "flashmla") not in ("flashmla", "mega"):
        raise ValueError("attention_core must be 'flashmla' or 'mega'")

    if (
        impl_cfg.token_map is None
        and impl_cfg.hf_path
        and model_cfg.engram_layer_ids
        and not (Path(impl_cfg.hf_path) / "token_map.pt").is_file()
    ):
        impl_cfg = replace(impl_cfg, token_map=release_token_map(impl_cfg.hf_path))
    return native.build_model(model_cfg, impl_cfg=impl_cfg, _model_class=DeepseekV41Model)


def export_hf_weights(chunks, model_cfg, ps, **kwargs):
    if kwargs.get("target") == "vllm":
        kwargs["target"] = "release"  # FP8 wo_a, matching the deployment forward.
    yield from native.export_hf_weights(chunks, model_cfg, ps, **kwargs)


def save_hf_weights(chunks, path, model_cfg, ps, **kwargs):
    if kwargs.get("target") == "vllm":
        kwargs["target"] = "release"
    return native.save_hf_weights(chunks, path, model_cfg, ps, **kwargs)
