# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU tests for DeepSeek-V4.1 CSA2 role resolution and configuration validation."""

import pytest

from megatron.core.transformer.experimental_attention_variant.csa2.roles import (
    SYMBOLS_PER_MODEL_LAYER,
    CSA2LayerMode,
    is_attention_position,
    model_layer_id_from_layer_number,
    model_layer_ratios_from_pattern_ratios,
    pattern_ratios_from_model_layer_ratios,
    resolve_csa2_plan,
)

# Released DeepSeek-V4.1-Flash text configuration (40 backbone layers, MTP stripped).
V41_RATIOS = [0, 0] + [2] * 18 + [1] * 20
V41_KV_SOURCES = [2, 8, 14, 20]
V41_INDEX_SOURCES = [2, 8, 14, 20, 24, 28, 32, 36]


def _v41_plan(**overrides):
    kwargs = dict(
        compress_ratios=V41_RATIOS,
        kv_source_layers=V41_KV_SOURCES,
        index_source_layers=V41_INDEX_SOURCES,
        candidate_source_layer=20,
        candidate_topk_blocks=2048,
        candidate_block_size=8,
    )
    kwargs.update(overrides)
    return resolve_csa2_plan(**kwargs)


class TestReleasedLayout:
    def test_modes(self):
        plan = _v41_plan()
        modes = [layer.mode for layer in plan.layers]
        assert modes[0] == modes[1] == CSA2LayerMode.WINDOW
        for src in V41_KV_SOURCES:
            assert modes[src] == CSA2LayerMode.FULL
        for src in (24, 28, 32, 36):
            assert modes[src] == CSA2LayerMode.REINDEX
        for layer_id in (3, 7, 9, 13, 15, 19, 21, 23, 25, 39):
            assert modes[layer_id] == CSA2LayerMode.REUSE
        assert sum(m == CSA2LayerMode.FULL for m in modes) == 4
        assert sum(m == CSA2LayerMode.REINDEX for m in modes) == 4
        assert sum(m == CSA2LayerMode.WINDOW for m in modes) == 2
        assert sum(m == CSA2LayerMode.REUSE for m in modes) == 30

    def test_sources(self):
        plan = _v41_plan()
        assert plan[5].kv_source == 2 and plan[5].index_source == 2
        assert plan[13].kv_source == 8 and plan[13].index_source == 8
        assert plan[19].kv_source == 14 and plan[19].index_source == 14
        assert plan[23].kv_source == 20 and plan[23].index_source == 20
        assert plan[27].kv_source == 20 and plan[27].index_source == 24
        assert plan[39].kv_source == 20 and plan[39].index_source == 36
        assert plan[20].kv_source == 20 and plan[20].index_source == 20

    def test_candidates(self):
        plan = _v41_plan()
        assert plan.uses_candidate_blocks
        assert plan[20].is_candidate_source and not plan[20].uses_candidates
        for src in (24, 28, 32, 36):
            assert plan[src].uses_candidates and not plan[src].is_candidate_source
        # Reuse layers never touch candidates; earlier index sources neither.
        assert not plan[25].uses_candidates
        assert not plan[14].uses_candidates

    def test_properties(self):
        plan = _v41_plan()
        assert plan[2].runs_compressor and plan[2].runs_indexer
        assert not plan[24].runs_compressor and plan[24].runs_indexer
        assert not plan[25].runs_compressor and not plan[25].runs_indexer
        assert not plan[0].has_compressed_path and plan[2].has_compressed_path
        assert len(plan) == 40

    def test_candidates_disabled(self):
        plan = _v41_plan(
            candidate_source_layer=None, candidate_topk_blocks=0, candidate_block_size=0
        )
        assert not plan.uses_candidate_blocks
        assert not plan[24].uses_candidates
        plan = _v41_plan(candidate_source_layer=-1, candidate_topk_blocks=0, candidate_block_size=0)
        assert not plan.uses_candidate_blocks


class TestRejections:
    def test_kv_source_must_be_index_source(self):
        with pytest.raises(ValueError, match="csa2_index_source_layers"):
            _v41_plan(index_source_layers=[2, 8, 20, 24, 28, 32, 36])

    def test_sources_strictly_increasing(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            _v41_plan(kv_source_layers=[2, 14, 8, 20])
        with pytest.raises(ValueError, match="strictly increasing"):
            _v41_plan(index_source_layers=[2, 8, 8, 14, 20, 24, 28, 32, 36])

    def test_source_out_of_range(self):
        with pytest.raises(ValueError, match="outside"):
            _v41_plan(kv_source_layers=[2, 8, 14, 40], candidate_source_layer=40)

    def test_window_layer_cannot_be_source(self):
        with pytest.raises(ValueError, match="window-only"):
            _v41_plan(
                kv_source_layers=[0, 8, 14, 20], index_source_layers=[0, 8, 14, 20, 24, 28, 32, 36]
            )

    def test_first_compressing_layer_needs_source(self):
        with pytest.raises(ValueError, match="no KV source layer at or before"):
            _v41_plan(kv_source_layers=[8, 14, 20], index_source_layers=[8, 14, 20, 24, 28, 32, 36])

    def test_shared_kv_needs_same_ratio(self):
        ratios = list(V41_RATIOS)
        ratios[5] = 1  # layer 5 would read ratio-2 KV from layer 2
        with pytest.raises(ValueError, match="same compress ratio"):
            _v41_plan(compress_ratios=ratios)

    def test_candidate_must_be_kv_source(self):
        with pytest.raises(ValueError, match="must be a KV source"):
            _v41_plan(candidate_source_layer=24)

    def test_candidate_must_be_last_kv_source(self):
        with pytest.raises(ValueError, match="last KV source"):
            _v41_plan(candidate_source_layer=14)

    def test_candidate_sizes_required(self):
        with pytest.raises(ValueError, match="csa2_candidate_topk_blocks"):
            _v41_plan(candidate_topk_blocks=0)
        with pytest.raises(ValueError, match="csa2_candidate_source_layer is not"):
            _v41_plan(candidate_source_layer=None, candidate_topk_blocks=8)

    def test_negative_ratio(self):
        ratios = list(V41_RATIOS)
        ratios[3] = -1
        with pytest.raises(ValueError, match="non-negative"):
            _v41_plan(compress_ratios=ratios)

    def test_empty(self):
        with pytest.raises(ValueError, match="not be empty"):
            resolve_csa2_plan([], [], [])


class TestPatternMapping:
    def test_layer_number_mapping(self):
        assert SYMBOLS_PER_MODEL_LAYER == 2
        assert model_layer_id_from_layer_number(1) == 0
        assert model_layer_id_from_layer_number(2) == 0
        assert model_layer_id_from_layer_number(41) == 20
        assert is_attention_position(1) and not is_attention_position(2)
        assert is_attention_position(41) and not is_attention_position(42)
        with pytest.raises(ValueError):
            model_layer_id_from_layer_number(0)

    def test_pattern_ratio_roundtrip(self):
        full = pattern_ratios_from_model_layer_ratios(V41_RATIOS)
        assert len(full) == 80
        assert full[1::2] == [0] * 40
        assert model_layer_ratios_from_pattern_ratios(full) == V41_RATIOS

    def test_pattern_ratio_rejects_bad_shapes(self):
        with pytest.raises(ValueError, match="multiple"):
            model_layer_ratios_from_pattern_ratios([0, 0, 2])
        with pytest.raises(ValueError, match="MoE pattern positions"):
            model_layer_ratios_from_pattern_ratios([0, 2, 2, 0])


class TestSharedStateTransport:
    def test_export_load_roundtrip(self):
        import torch

        from megatron.core.transformer.experimental_attention_variant.csa2.state import (
            CompressedKVRecord,
            DSv41SharedState,
        )

        state = DSv41SharedState()
        kv, ik = torch.randn(6, 2, 8), torch.randn(6, 2, 4)
        state.publish_compressed(CompressedKVRecord(2, 2, 6, kv, ik))
        topk = torch.randint(-1, 6, (12, 2, 3), dtype=torch.int32)
        state.publish_topk(2, topk)
        cand = torch.randint(-1, 3, (12, 2, 2), dtype=torch.int32)
        state.publish_candidates(2, cand)
        h_pre = torch.rand(12, 2, 4)
        state.publish_h_pre(5, h_pre)
        keys = state.keys()
        assert keys == [("kv", 2), ("ik", 2), ("topk", 2), ("cand", 2), ("hpre", 5)]
        tensors = state.export(keys)

        rebuilt = DSv41SharedState()
        rebuilt.load(keys, tensors, ratios={2: 2})
        rec = rebuilt.get_compressed(2)
        assert rec.compress_ratio == 2 and rec.n_compressed == 6
        assert torch.equal(rec.kv, kv) and torch.equal(rec.index_keys, ik)
        assert torch.equal(rebuilt.get_topk(2), topk)
        assert torch.equal(rebuilt.get_candidates(2), cand)
        assert torch.equal(rebuilt.get_h_pre_for(6), h_pre)
        assert rebuilt.get_h_pre_for(5) is None
