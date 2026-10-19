# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Release source ownership and unsupported candidate publishers."""
from dataclasses import replace

import pytest

from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
from megatron.lite.model.deepseek_v41.topology import layer_sources


def test_release_sources_and_candidate_publisher():
    release = DeepseekV41Config()  # 40 layers: kv (2,8,14,20), index (...,36), candidate 20
    sources = layer_sources(release)
    assert [(s.kv, s.index, s.candidate) for s in (sources[19], sources[20], sources[23])] == [
        (14, 14, False),
        (20, 20, True),
        (20, 20, True),
    ]
    assert (sources[24].index, sources[39].kv, sources[39].index) == (24, 20, 36)
    for candidate in (21, 14):  # a reader of index owner 20; a ratio-two index owner
        with pytest.raises(ValueError, match="candidate publisher"):
            layer_sources(replace(release, candidate_source_layer_id=candidate))
