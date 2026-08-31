"""Pin fp32-on-export whitelist for DeepSeek-V4 HF checkpoints."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[3]
    / "megatron"
    / "lite"
    / "model"
    / "deepseek_v4"
    / "lite"
    / "checkpoint.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "deepseek_v4_checkpoint_under_test", _CHECKPOINT_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
DeepseekV4WeightSpec = _MODULE.DeepseekV4WeightSpec


# ``hf_export_dtype_override`` inspects only the HF name; config is unused, so
# ``None`` is enough to exercise the whitelist logic without dragging in the
# full ``DeepseekV4Config`` (and its transitive Megatron-core deps).
_SPEC_UNDER_TEST = DeepseekV4WeightSpec.__new__(DeepseekV4WeightSpec)


@pytest.mark.parametrize(
    "hf_name",
    [
        "model.layers.0.self_attn.attn_sink",
        "model.layers.3.self_attn.compressor.ape",
        "model.layers.3.self_attn.indexer.compressor.ape",
        "model.layers.5.mlp.ffn.gate.bias",
        "model.hc_head_q",
        "model.layers.2.self_attn.hc_attn_k",
        "model.layers.2.mlp.hc_ffn_gate",
    ],
)
def test_fp32_whitelist_forces_float32(hf_name: str) -> None:
    assert _SPEC_UNDER_TEST.hf_export_dtype_override(hf_name) is torch.float32


@pytest.mark.parametrize(
    "hf_name",
    [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.mlp.shared_experts.up_proj.weight",
    ],
)
def test_non_whitelisted_names_default_to_none(hf_name: str) -> None:
    assert _SPEC_UNDER_TEST.hf_export_dtype_override(hf_name) is None
