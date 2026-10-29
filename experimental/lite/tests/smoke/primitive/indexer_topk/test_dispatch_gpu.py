# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DSA and CSA modules selecting their indexer top-k through bindings (optional, Blackwell GPU).

The tests bind the matched-precision reference selector (``backend="reference"``,
``precision="fast"``: DeepGEMM scores, cuDNN radix top-k) and need no plugin. They feed each
dispatch point operands whose scores are far apart and exactly representable in both the BF16
upstream selector and the FP8/MXFP4 reference selector, so every selector must return the same
set per row: the binding's result is compared with the module's upstream selector and with a
torch oracle, including the CSA rows that see no compressed key, through
``build_attention_indices``. A C4 CSA layer must select the same top-k through its THD and its
BSHD dispatch point. The cuDNN frontend 1.27 is needed (as for the upstream CSA THD tests), for
example through the runner's ``PYTHONPATH``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.gpus(1, min_architecture="blackwell"), pytest.mark.optional]


def _levels(count: int, per_binade: int, generator: torch.Generator) -> torch.Tensor:
    """``count`` distinct positive magnitudes in random order, exact in BF16 and in the
    selector's format: ``per_binade`` = 128 (BF16 mantissas; FP8 rows of one nonzero value are
    exact) or 2 (1 and 1.5 times a power of two: exact in MXFP4)."""
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


def _csa_config(**overrides):
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config

    fields = dict(
        hidden_size=64,
        num_attention_heads=64,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=8,
        compress_ratios=[4],
        sliding_window=8,
        index_head_dim=128,
        index_n_heads=64,
        index_topk=16,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        num_hidden_layers=1,
        num_nextn_predict_layers=1,
    )
    fields.update(overrides)
    return DeepseekV4Config(**fields)


def _init_single_rank_group(init_file: Path) -> bool:
    """Start a one-rank NCCL group unless one exists (the THD path gathers over its CP group)."""
    import torch.distributed as dist

    if dist.is_initialized():
        assert dist.get_world_size() == 1
        return False
    dist.init_process_group("nccl", init_method=f"file://{init_file}", rank=0, world_size=1)
    return True


@pytest.fixture
def single_rank_group(tmp_path):
    import torch.distributed as dist

    created = _init_single_rank_group(tmp_path / "process-group")
    try:
        yield dist.group.WORLD
    finally:
        if created and dist.is_initialized():
            dist.destroy_process_group()


def _csa_module(device, config=None, group=None):
    from megatron.lite.primitive.modules.attention.csa import CompressedSparseAttention

    ps = SimpleNamespace(cp_size=1, cp_rank=0, cp_group=group)
    module = CompressedSparseAttention(config or _csa_config(), layer_idx=0, ps=ps)
    return module.to(device=device, dtype=torch.bfloat16).eval()


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


def test_csa_bshd_reference_binding_matches_upstream_on_separated_scores():
    from megatron.lite.primitive.kernels import dsa_kernels

    device = torch.device("cuda")
    tokens, topk = 512, 16
    module = _csa_module(device)
    binding = _bind(module, "mxfp4")
    key_values = _levels(tokens // 4, 2, torch.Generator().manual_seed(2))
    q, k, weights = (x.unsqueeze(1) for x in _separated(tokens, 64, key_values, device))
    with torch.no_grad():
        selected = module._select_bshd_indexer_topk(binding, q, k, weights, topk)
        upstream, _ = dsa_kernels.indexer_topk(
            q, k, weights, topk, 4, indexer_softmax_scale=module.indexer.softmax_scale
        )
    expected = _oracle(key_values, [(0, (row + 1) // 4) for row in range(tokens)], topk)
    assert torch.equal(_ascending(selected[0]), expected)
    assert torch.equal(_ascending(upstream[0]), expected)
    assert (expected[:3] == -1).all()  # rows 0-2 see no compressed key


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_csa_thd_reference_binding_matches_upstream_on_separated_scores(cp_rank):
    """Packed sequences on one of two context-parallel ranks, through the attention indices."""
    from megatron.core.transformer.experimental_attention_variant.csa_utils import (
        cp_utils,
        thd_layout_kernels,
    )

    device = torch.device("cuda")
    cp_size, l_local, topk, window = 2, 512, 16, 8
    offsets = [0, 200, 520, 1024]  # rank 1 starts 312 tokens into the second sequence
    lengths = [end - start for start, end in zip(offsets, offsets[1:])]
    counts = [length // 4 for length in lengths]
    key_starts = [sum(counts[:index]) for index in range(len(counts) + 1)]
    generator = torch.Generator().manual_seed(3)
    key_values = torch.cat([_levels(count, 2, generator) for count in counts])
    module = _csa_module(device, _csa_config(index_topk=topk, sliding_window=window))
    binding = _bind(module, "mxfp4")
    global_start = cp_rank * l_local
    q, k, weights = _separated(l_local, 64, key_values, device)
    cu = torch.tensor(offsets, dtype=torch.int32, device=device)
    cu_compressed = torch.tensor(key_starts, dtype=torch.int32, device=device)
    with torch.no_grad():
        selected = module._select_thd_indexer_topk(
            binding,
            q,
            weights,
            k,
            cu,
            cu_compressed,
            global_start=global_start,
            cp_size=cp_size,
            max_seqlen_q=max(lengths),
        )
        upstream = {
            fused: cp_utils.compute_cp_indexer_topk(
                q,
                weights,
                k,
                cu,
                cu_compressed,
                global_start,
                4,
                topk,
                module.indexer.softmax_scale,
                max_seqlen_q=max(lengths),
                use_fused=fused,
            )[0]
            for fused in (True, False)
        }
    visible = []
    for row in range(global_start, global_start + l_local):
        sequence = max(index for index in range(len(lengths)) if offsets[index] <= row)
        position = row - offsets[sequence]
        visible.append((key_starts[sequence], min(counts[sequence], (position + 1) // 4)))
    expected = _oracle(key_values, visible, topk)
    assert torch.equal(_ascending(selected), expected)
    for fused, ids in upstream.items():
        assert torch.equal(_ascending(ids), expected), f"use_fused={fused}"
    # Positions 0-2 of a sequence see no compressed key: rows 0-2 and 200-202 on rank 0, and
    # the first three rows of the third sequence on rank 1.
    no_keys = (expected == -1).all(dim=1)
    assert no_keys.nonzero().flatten().tolist() == ([0, 1, 2, 200, 201, 202], [8, 9, 10])[cp_rank]

    # Rows without a compressed key, and every other row, lower to the same attention indices.
    def attention_indices(compressed_topk):
        indices, lengths_out, _, _ = thd_layout_kernels.build_attention_indices(
            cu,
            global_start,
            l_local,
            window,
            window,
            4,
            topk,
            compressed_topk.contiguous(),
            cu_seqlens_compressed=cu_compressed,
            compressed_rows=k.shape[0],
            compressed_is_sequence_major=True,
        )
        return _ascending(indices), lengths_out.cpu()

    ours, ours_lengths = attention_indices(selected)
    theirs, theirs_lengths = attention_indices(upstream[True])
    assert torch.equal(ours, theirs) and torch.equal(ours_lengths, theirs_lengths)
    total_rows = window + l_local + k.shape[0]
    assert int(ours.max()) < total_rows and bool(((ours >= 0) | (ours == -1)).all())
    # Those rows keep only their window entries.
    assert bool((ours_lengths[no_keys] <= window).all())
    assert bool((ours[no_keys] < window + l_local).all())


def test_cp1_thd_equals_bshd_fused_with_reference_binding(single_rank_group):
    """The C4 CSA layer selects the same top-k through its THD and BSHD dispatch points."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    tokens = 256
    module = _csa_module(device, group=single_rank_group)
    module.attention_backend = "flash"
    module.apply_dsa_kernel_fusion = True
    binding = _bind(module, "mxfp4")
    selections = []
    select = binding.select

    def recording_select(*args, **kwargs):
        selections.append(select(*args, **kwargs))
        return selections[-1]

    binding.select = recording_select
    x = torch.randn(1, tokens, module.config.hidden_size, device=device, dtype=torch.bfloat16)
    positions = torch.arange(tokens, device=device).unsqueeze(0)
    cu = torch.tensor([0, tokens], dtype=torch.int32, device=device)
    packed = SimpleNamespace(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_q_padded=None,
        cu_seqlens_kv=cu,
        cu_seqlens_kv_padded=None,
        max_seqlen_q=tokens,
        max_seqlen_kv=tokens,
    )
    with torch.no_grad():
        out_bshd = module(x, position_ids=positions)
        out_thd = module(x, position_ids=positions, packed_seq_params=packed)
    assert len(selections) == 2 and binding.stats.calls == 2
    assert torch.equal(selections[0], selections[1])
    assert int((selections[0] >= 0).sum(dim=1).max()) == 16
    assert out_thd.shape == out_bshd.shape == (1, tokens, module.config.hidden_size)
    torch.testing.assert_close(out_thd, out_bshd, rtol=2e-2, atol=2e-2)
