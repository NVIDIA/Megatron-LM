# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DSA modules selecting their indexer top-k through bindings (optional, Blackwell GPU).

The tests bind the matched-precision reference selector (``backend="reference"``,
``precision="fast"``: DeepGEMM scores, cuDNN radix top-k) and need no plugin. They feed the
dispatch point operands whose scores are far apart and exactly representable in both the BF16
upstream selector and the FP8 reference selector, so every selector must return the same set
per row: the binding's result is compared with the module's upstream selector and with a torch
oracle. The cuDNN frontend 1.27 is needed, for example through the runner's ``PYTHONPATH``.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.gpus(1, min_architecture="blackwell"), pytest.mark.optional]


def _levels(count: int, per_binade: int, generator: torch.Generator) -> torch.Tensor:
    """``count`` distinct positive magnitudes in random order, exact in BF16 and in the
    selector's format: ``per_binade`` = 128 (BF16 mantissas; FP8 rows of one nonzero value are
    exact)."""
    index = torch.arange(count, dtype=torch.float64)
    binade, step = torch.div(index, per_binade, rounding_mode="floor"), index % per_binade
    values = torch.exp2(binade) * (1 + step / per_binade)
    return values[torch.randperm(count, generator=generator)]


def _separated(rows: int, heads: int, key_values: torch.Tensor, device) -> tuple:
    """Queries, keys and weights whose row scores order the keys by ``key_values``."""
    q = torch.zeros((rows, heads, 128), dtype=torch.bfloat16, device=device)
    q[..., 0] = 1
    k = torch.zeros((key_values.numel(), 128), dtype=torch.bfloat16, device=device)
    k[:, 0] = key_values.to(device=device, dtype=torch.bfloat16)
    weights = torch.ones((rows, heads), dtype=torch.bfloat16, device=device)
    return q, k, weights


def _oracle(key_values: torch.Tensor, visible: list[tuple[int, int]], topk: int) -> torch.Tensor:
    """Per row: the ``topk`` largest of the keys ``[start, start + count)``, ids relative to
    ``start``, ascending, -1 padded."""
    out = torch.full((len(visible), topk), -1, dtype=torch.int32)
    for row, (start, count) in enumerate(visible):
        if count:
            chosen = key_values[start : start + count].topk(min(topk, count)).indices
            out[row, : chosen.numel()] = chosen.sort().values.to(torch.int32)
    return out


def _ascending(indices: torch.Tensor) -> torch.Tensor:
    from megatron.lite.primitive.kernels.indexer_topk import sort_topk_rows_

    return sort_topk_rows_(indices.detach().to(torch.int32).clone()).cpu()


def _bind(module, native_format: str):
    from megatron.lite.primitive.modules.attention.indexer_topk import configure_indexer_topk

    installation = configure_indexer_topk(
        [module], {"backend": "reference", "precision": "fast"}, native_format=native_format
    )
    (binding,) = installation.bindings
    return binding


def _dsa_module(device, *, index_topk: int):
    from megatron.lite.primitive.modules.attention.dsa import DynamicSparseAttention

    module = DynamicSparseAttention(
        hidden_size=64,
        num_attention_heads=2,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=16,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=index_topk,
        rms_norm_eps=1e-5,
    )
    return module.to(device).eval()


def test_dsa_reference_binding_matches_upstream_on_separated_scores():
    from megatron.lite.primitive.modules.attention import dsa

    device = torch.device("cuda")
    tokens, topk = 2048, 64
    module = _dsa_module(device, index_topk=topk)
    binding = _bind(module, "fp8")
    key_values = _levels(tokens, 128, torch.Generator().manual_seed(1))
    q, k, weights = _separated(tokens, 32, key_values, device)
    q, k, weights = q.unsqueeze(1), k.unsqueeze(1), weights.unsqueeze(1)  # [s, b=1, ...]
    with torch.no_grad():
        selected = module._select_full_prompt_topk(binding, q, k, weights, topk)
        upstream, _ = dsa._dsa_kernels.indexer_topk(
            q, k, weights, topk, 1, indexer_softmax_scale=module.indexer_softmax_scale
        )
    assert selected.shape == upstream.shape == (1, tokens, topk)
    expected = _oracle(key_values, [(0, row + 1) for row in range(tokens)], topk)
    assert torch.equal(_ascending(selected[0]), expected)
    assert torch.equal(_ascending(upstream[0]), expected)
    assert binding.stats.reference_rows == tokens and binding.stats.calls == 1
