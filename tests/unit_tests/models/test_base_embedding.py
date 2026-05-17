# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestBaseEmbedding:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True
        )
        self.base_embedding = LanguageModelEmbedding(
            config=transformer_config,
            vocab_size=100,
            max_sequence_length=4,
            position_embedding_type='learned_absolute',
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.base_embedding, LanguageModelEmbedding)
        num_weights = sum([p.numel() for p in self.base_embedding.parameters()])
        assert num_weights == 1248

    def test_zero_parameters(self):
        sum_weights = sum([p.sum() for p in self.base_embedding.parameters()])
        assert sum_weights != 0
        self.base_embedding.zero_parameters()
        sum_weights = sum([p.sum() for p in self.base_embedding.parameters()])
        assert sum_weights == 0

    def test_cpu_forward(self):
        input_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64).repeat((2, 1))
        position_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64).repeat((2, 1))
        embeddings = self.base_embedding(input_ids, position_ids)
        assert embeddings.device.type == 'cpu'
        assert embeddings.shape[0] == self.base_embedding.max_sequence_length
        assert embeddings.shape[1] == input_ids.shape[0]
        assert embeddings.shape[2] == self.base_embedding.config.hidden_size

    def test_gpu_forward(self):
        self.base_embedding.cuda()
        input_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64).repeat((2, 1)).cuda()
        position_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64).repeat((2, 1)).cuda()
        embeddings = self.base_embedding(input_ids, position_ids)
        assert embeddings.device.type == 'cuda'
        assert embeddings.shape[0] == self.base_embedding.max_sequence_length
        assert embeddings.shape[1] == input_ids.shape[0]
        assert embeddings.shape[2] == self.base_embedding.config.hidden_size


@pytest.mark.skipif(Utils.world_size < 4, reason="requires at least 4 ranks for tp=2 and cp=2")
def test_embedding_dropout_uses_context_parallel_tracker():
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, context_parallel_size=2)
    try:
        config = TransformerConfig(
            num_layers=2,
            hidden_size=32,
            num_attention_heads=4,
            hidden_dropout=0.5,
            use_cpu_initialization=False,
        )
        model_parallel_cuda_manual_seed(321, force_reset_rng=True)
        embedding = LanguageModelEmbedding(
            config=config,
            vocab_size=128,
            max_sequence_length=16,
            position_embedding_type='learned_absolute',
        ).cuda()
        embedding.train()

        with torch.no_grad():
            embedding.word_embeddings.weight.fill_(1.0)
            embedding.position_embeddings.weight.fill_(2.0)

        batch_size = 4
        seq_len = embedding.max_sequence_length
        input_ids = torch.arange(seq_len, device='cuda', dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
        position_ids = input_ids.clone()
        output = embedding(input_ids, position_ids)
        dropout_mask = output.eq(0).cpu()

        assert dropout_mask.any()
        assert (~dropout_mask).any()

        payload = {
            "cp_rank": parallel_state.get_context_parallel_rank(),
            "mask": dropout_mask,
        }
        gathered = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered, payload)

        masks_by_cp_rank = {}
        for item in gathered:
            masks_by_cp_rank.setdefault(item["cp_rank"], []).append(item["mask"])

        for masks in masks_by_cp_rank.values():
            reference = masks[0]
            for mask in masks[1:]:
                assert torch.equal(reference, mask)

        cp_ranks = sorted(masks_by_cp_rank)
        assert len(cp_ranks) == 2
        assert not torch.equal(
            masks_by_cp_rank[cp_ranks[0]][0],
            masks_by_cp_rank[cp_ranks[1]][0],
        )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(Utils.world_size < 4, reason="requires at least 4 ranks for tp=2 and cp=2")
def test_embedding_dropout_uses_model_and_context_parallel_tracker_when_sequence_parallel():
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, context_parallel_size=2)
    try:
        config = TransformerConfig(
            num_layers=2,
            hidden_size=32,
            num_attention_heads=4,
            hidden_dropout=0.5,
            sequence_parallel=True,
            use_cpu_initialization=False,
        )
        model_parallel_cuda_manual_seed(654, force_reset_rng=True)
        embedding = LanguageModelEmbedding(
            config=config,
            vocab_size=128,
            max_sequence_length=16,
            position_embedding_type='learned_absolute',
        ).cuda()
        embedding.train()

        with torch.no_grad():
            embedding.word_embeddings.weight.fill_(1.0)
            embedding.position_embeddings.weight.fill_(2.0)

        batch_size = 4
        seq_len = embedding.max_sequence_length
        input_ids = torch.arange(seq_len, device='cuda', dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
        position_ids = input_ids.clone()
        output = embedding(input_ids, position_ids)
        dropout_mask = output.eq(0).cpu()

        assert dropout_mask.any()
        assert (~dropout_mask).any()

        payload = {
            "cp_rank": parallel_state.get_context_parallel_rank(),
            "tp_rank": parallel_state.get_tensor_model_parallel_rank(),
            "mask": dropout_mask,
        }
        gathered = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered, payload)

        masks_by_rank = {(item["cp_rank"], item["tp_rank"]): item["mask"] for item in gathered}
        assert len(masks_by_rank) == 4

        for cp_rank in range(2):
            assert not torch.equal(masks_by_rank[(cp_rank, 0)], masks_by_rank[(cp_rank, 1)])

        for tp_rank in range(2):
            assert not torch.equal(masks_by_rank[(0, tp_rank)], masks_by_rank[(1, tp_rank)])
    finally:
        Utils.destroy_model_parallel()
