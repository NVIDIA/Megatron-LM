# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
import torch.nn.init as init

from megatron.core.models.common.embeddings.relative_pos_embedding import RelativePositionEmbedding
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.test_utilities import Utils


class TestRelativePositionEmbedding:
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.num_heads = 12
        self.relative_pos_emb = RelativePositionEmbedding(
            bidirectional=True,
            init_method=init.normal_,
            num_attention_heads=self.num_heads,
            relative_attention_num_buckets=32,
            relative_attention_max_distance=128,
        )

    def teardown_method(self, method):
        del self.relative_pos_emb
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.relative_pos_emb, RelativePositionEmbedding)

    def test_forward(self):
        self.query_seq_length = 512
        output = self.relative_pos_emb(self.query_seq_length, self.query_seq_length)
        assert output.shape[0] == 1
        assert output.shape[1] == self.num_heads
        assert output.shape[2] == self.query_seq_length
        assert output.shape[3] == self.query_seq_length
