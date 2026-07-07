# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek-V4 HF load must lift LOCAL pipeline layer indices to GLOBAL.

Under PP the model's ``self.layers`` ModuleDict is keyed by LOCAL pipeline
position, so a non-first stage's native ``state_dict`` keys carry local indices
(``layers.0`` ...). The HF release is keyed by GLOBAL layer index, so -- exactly
like the exporter -- ``load_hf_weights`` must map local->global via
``to_global_layer_name(name, layer_map)`` before resolving HF names. Without it a
non-first stage reads the wrong global layer's weights.

This is a CPU unit test: a minimal stand-in stage (no GPU/TE) whose layers are
keyed locally but carry ``layer_indices`` = the global ids it owns, plus a tiny
on-disk safetensors keyed by GLOBAL names. ``load_hf_weights`` must copy each
local layer the GLOBAL layer's tensor; pre-fix it resolves local names, finds
nothing, and leaves the params untouched.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.mlite


class _LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))


class _Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.input_layernorm = _LayerNorm(dim)


class _Stage(nn.Module):
    """A non-first PP stage: layers keyed by LOCAL position, ``layer_indices``
    gives the GLOBAL ids it owns (e.g. [4, 5] for the 3rd stage of pp)."""

    def __init__(self, global_ids: list[int], dim: int):
        super().__init__()
        self.layer_indices = list(global_ids)
        self.layers = nn.ModuleDict({str(i): _Block(dim) for i in range(len(global_ids))})


def test_ds4_load_hf_resolves_local_pp_layer_to_global(tmp_path):
    from safetensors.torch import save_file

    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
    from megatron.lite.model.deepseek_v4.lite import checkpoint as ckpt

    dim = 4
    global_ids = [4, 5]  # this stage owns global layers 4 and 5, keyed local 0 and 1
    model = _Stage(global_ids, dim)
    cfg = DeepseekV4Config(num_hidden_layers=8, n_routed_experts=8)
    ps = SimpleNamespace(tp_size=1, etp_size=1, ep_size=1, ep_rank=0)

    # Real-release layout is keyed by GLOBAL layer index; input_layernorm maps to
    # the bare V4-Flash ``attn_norm.weight``.
    save_file(
        {f"layers.{g}.attn_norm.weight": torch.full((dim,), float(g)) for g in global_ids},
        str(tmp_path / "model.safetensors"),
    )

    ckpt.load_hf_weights(model, str(tmp_path), cfg, ps)

    # local layer 0 -> global 4 -> filled with 4.0; local 1 -> global 5 -> 5.0.
    # Pre-fix, load resolved layers.0/layers.1 (local), found nothing, left zeros.
    torch.testing.assert_close(
        model.layers["0"].input_layernorm.weight.detach(), torch.full((dim,), 4.0)
    )
    torch.testing.assert_close(
        model.layers["1"].input_layernorm.weight.detach(), torch.full((dim,), 5.0)
    )
