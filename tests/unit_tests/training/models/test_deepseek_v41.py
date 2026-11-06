# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json

import pytest

from megatron.core.models.deepseek_v41.config import EngramConfig
from megatron.training.models import deepseek_v41

_ENGRAM_FIELDS = dict(
    layer_ids=[0, 1],
    num_embeddings=[96, 96],
    max_ngram_size=2,
    vocab_size=17,
    n_heads=2,
    head_dim=4,
    pad_token_id=2,
    compressed_vocab_size=128,
)


def test_engram_config_is_parsed_once_per_path(tmp_path):
    """The argument setup and the per-iteration FLOPs estimate share one parsed EngramConfig
    per JSON path, so training does not reopen the file; a failed read is not cached."""
    path = tmp_path / "engram.json"
    path.write_text(json.dumps(_ENGRAM_FIELDS), encoding="utf-8")
    deepseek_v41._load_engram_config.cache_clear()
    try:
        first = deepseek_v41._load_engram_config(str(path))
        assert first == EngramConfig(**_ENGRAM_FIELDS)
        assert deepseek_v41._load_engram_config(str(path)) is first
        info = deepseek_v41._load_engram_config.cache_info()
        assert (info.hits, info.misses, info.currsize) == (1, 1, 1)

        missing = str(tmp_path / "missing.json")
        with pytest.raises(FileNotFoundError):
            deepseek_v41._load_engram_config(missing)
        assert deepseek_v41._load_engram_config.cache_info().currsize == 1
    finally:
        deepseek_v41._load_engram_config.cache_clear()
