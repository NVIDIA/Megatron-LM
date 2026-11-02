# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest

from tools import build_sequences_per_dataset as metadata_tool


def _args(**overrides):
    values = dict(
        data_path=None,
        data_args_path=None,
        train_data_path=None,
        valid_data_path=None,
        test_data_path=None,
        per_split_data_args_path=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_source_paths_accept_train_triples_and_deduplicate():
    assert metadata_tool.get_paths_from_blend(
        None, [(["source-a", "source-b"], [0.4, 0.6], [0.25, 1.0]), (["source-a"], None), None]
    ) == ["source-a", "source-b"]
    assert metadata_tool.get_paths_from_blend((["source-a"], None), None) == ["source-a"]


def test_downsampling_metadata_counts_full_sources(monkeypatch):
    opened = []

    def read_index(path, multimodal):
        assert not multimodal
        opened.append(path)
        return SimpleNamespace(sequence_count=101, document_count=29)

    monkeypatch.setattr(metadata_tool, "_IndexReader", read_index)
    result = metadata_tool.build_sequences_per_dataset(
        _args(train_data_path=["1", "source-a", "0.25"], valid_data_path=["source-a", "source-b"])
    )

    assert result == {"source-a": (101, 29), "source-b": (101, 29)}
    assert opened == ["source-a.idx", "source-b.idx"]


def test_metadata_tool_rejects_validation_downsampling():
    with pytest.raises(ValueError, match="not enabled"):
        metadata_tool.build_sequences_per_dataset(_args(valid_data_path=["1", "source-a", "0.25"]))
