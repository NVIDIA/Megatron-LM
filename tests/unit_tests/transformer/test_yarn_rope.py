# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import YarnRotaryEmbedding


class TestYarnRotaryEmbedding:
    def test_cpu_initialization_keeps_cache_on_cpu(self):
        rope = YarnRotaryEmbedding(
            kv_channels=8, use_cpu_initialization=True, original_max_position_embeddings=64
        )

        assert rope.inv_freq_extra.device.type == 'cpu'
        assert rope.inv_freq_inter.device.type == 'cpu'
        assert rope.cos_cached.device.type == 'cpu'
        assert rope.sin_cached.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cpu_initialized_frequencies_follow_migrated_cache(self):
        rope = YarnRotaryEmbedding(
            kv_channels=8, use_cpu_initialization=True, original_max_position_embeddings=64
        ).cuda()

        rope.get_cached_cos_sin(128)

        assert rope.inv_freq_extra.device.type == 'cuda'
        assert rope.inv_freq_inter.device.type == 'cuda'
        assert rope.cos_cached.device.type == 'cuda'
        assert rope.sin_cached.device.type == 'cuda'
