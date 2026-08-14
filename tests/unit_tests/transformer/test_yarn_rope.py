# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import YarnRotaryEmbedding


class TestYarnRotaryEmbedding:
    def test_cpu_initialization_keeps_cache_on_cpu(self, monkeypatch):
        monkeypatch.setattr(
            torch.cuda,
            "current_device",
            lambda: pytest.fail("CPU-initialized YARN must not access CUDA"),
        )
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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_cache_is_invalidated_after_device_migration(self):
        rope = YarnRotaryEmbedding(
            kv_channels=8, use_cpu_initialization=True, original_max_position_embeddings=64
        )

        cpu_embedding, cpu_mscale = rope(64)
        assert cpu_embedding.device.type == 'cpu'

        rope.cuda()
        cuda_embedding, cuda_mscale = rope(64)

        assert cuda_embedding.device.type == 'cuda'
        assert rope.inv_freq_extra.device.type == 'cuda'
        assert rope.inv_freq_inter.device.type == 'cuda'
        assert cuda_mscale == cpu_mscale
        torch.testing.assert_close(cuda_embedding.cpu(), cpu_embedding)
