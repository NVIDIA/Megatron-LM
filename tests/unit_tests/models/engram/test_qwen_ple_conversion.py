# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Round-trip test for the HF PLE -> Megatron Engram weight converter."""

import importlib.util
import sys
from pathlib import Path

import torch

from megatron.core.models.engram.config import allocate_qwen_table_sizes
from megatron.core.models.engram.engram import Engram

from ._test_utils import make_module_config, make_pg_collection
from .test_qwen_variant import _EOS, _qwen_config

_TOOL_PATH = Path(__file__).resolve().parents[4] / "tools" / "engram" / "convert_qwen_ple.py"
_spec = importlib.util.spec_from_file_location("convert_qwen_ple", _TOOL_PATH)
convert_qwen_ple = importlib.util.module_from_spec(_spec)
sys.modules["convert_qwen_ple"] = convert_qwen_ple
_spec.loader.exec_module(convert_qwen_ple)


def _fake_hf_state(module: Engram, hf_layer_index: int, pad_to: int = 8):
    """Serialize an Engram module into the HF PLE key layout (fused padded table)."""
    prefix = f"model.layers.{hf_layer_index}.ple."
    state = {
        f"{prefix}key_proj.weight": module.key_projection.weight.detach().clone(),
        f"{prefix}value_proj.weight": module.value_projection.weight.detach().clone(),
        f"{prefix}norm_key.weight": module.key_norm.weight.detach().clone(),
        f"{prefix}norm_query.weight": module.query_norm.weight.detach().clone(),
        f"{prefix}norm_conv.weight": module.conv_norm.weight.detach().clone(),
        f"{prefix}conv1d.weight": module.short_conv.weight.detach().clone(),
    }
    full = torch.cat([table.weight.detach() for table in module.embedding.tables], dim=0)
    padded_rows = -(-full.shape[0] // pad_to) * pad_to
    padding = full.new_zeros((padded_rows - full.shape[0], full.shape[1]))
    full_padded = torch.cat([full, padding], dim=0)
    # Split the table into uneven dim-0 shards, as HF checkpoints store it.
    split = full_padded.shape[0] // 2
    state[f"{prefix}ple_embedding.ngram_embedding.weight.part_0"] = full_padded[:split]
    state[f"{prefix}ple_embedding.ngram_embedding.weight.part_1"] = full_padded[split:]
    return state


def _build_module(seed: int) -> Engram:
    torch.manual_seed(seed)
    return Engram(
        config=make_module_config(num_streams=4),
        engram_config=_qwen_config(),
        layer_number=1,
        pg_collection=make_pg_collection(),
    )


def test_hf_bundle_round_trip_restores_module_exactly():
    source = _build_module(seed=5)
    with torch.no_grad():
        source.short_conv.weight.normal_(std=0.05)
        source.key_norm.weight.normal_(std=0.02)
    state = _fake_hf_state(source, hf_layer_index=2)

    bundle = convert_qwen_ple.convert_hf_ple_layer(
        state,
        state.__getitem__,
        hf_layer_index=2,
        ple_layer_index=0,
        table_sizes=source.engram_config.table_sizes(1),
    )
    # Cross-check against an independent allocation, not the value we just passed in.
    assert bundle["table_sizes"] == list(allocate_qwen_table_sizes(17, (1,), 3, 2)[1])

    target = _build_module(seed=11)  # different init; must be fully overwritten
    convert_qwen_ple.load_bundle_into_engram(target, bundle)

    for (name, source_param), target_param in zip(source.named_parameters(), target.parameters()):
        torch.testing.assert_close(target_param, source_param, msg=name)

    tokens = torch.tensor([[3, 9, _EOS, 4, 2, 6, 1, 5, 8, 2, 11, 4]], dtype=torch.int64)
    hidden = torch.randn(tokens.shape[1], 1, 32, dtype=torch.float64)
    torch.testing.assert_close(target(hidden, tokens), source(hidden, tokens))
