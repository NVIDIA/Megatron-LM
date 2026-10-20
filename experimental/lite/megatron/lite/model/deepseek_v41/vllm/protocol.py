# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native Lite lifecycle/DistOpt; deployment operator model selection only."""
from megatron.lite.model.deepseek_v41.lite import protocol as native
from megatron.lite.model.deepseek_v41.lite.protocol import (
    ImplConfig,
    build_model_config,
    forward_step,
    load_hf_weights,
    unpack_forward_output,
    validate_impl_cfg_keys,
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

    if (
        impl_cfg.token_map is None
        and impl_cfg.hf_path
        and model_cfg.engram_layer_ids
        and not (Path(impl_cfg.hf_path) / "token_map.pt").is_file()
    ):
        impl_cfg = replace(impl_cfg, token_map=release_token_map(impl_cfg.hf_path))
    return native.build_model(model_cfg, impl_cfg=impl_cfg, _model_class=DeepseekV41Model)


def export_hf_weights(chunks, model_cfg, ps, **kwargs):
    yield from native.export_hf_weights(chunks, model_cfg, ps, **kwargs)


def save_hf_weights(chunks, path, model_cfg, ps, **kwargs):
    return native.save_hf_weights(chunks, path, model_cfg, ps, **kwargs)
