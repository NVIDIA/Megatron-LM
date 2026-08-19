# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
import torch.nn as nn

import megatron.core.parallel_state as ps
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.models.dist_utils import (
    prepare_existing_model_chunks_for_distributed_training,
)
from tests.unit_tests.test_utilities import Utils


class Test:

    transformer_config = TransformerConfig(
        num_layers=1, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True
    )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_muon_ht_init_runs_before_distributed_wrapping(self):
        """The model initialization lifecycle should enforce a configured Hyperball radius."""
        Utils.initialize_model_parallel(1, 1)
        config = TransformerConfig(
            num_layers=1, hidden_size=12, num_attention_heads=4, muon_ht_radius=2.5
        )
        model = nn.Linear(12, 12, bias=True, device='cuda')

        result = prepare_existing_model_chunks_for_distributed_training(
            [model],
            config,
            ProcessGroupCollection.use_mpu_process_groups(),
            wrap_with_ddp=False,
            mixed_precision_wrapper=None,
        )

        torch.testing.assert_close(result[0].weight.norm(), torch.tensor(2.5, device='cuda'))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_muon_ht_init_respects_disabled_parameter_initialization(self):
        """Checkpoint-oriented model construction should not modify uninitialized parameters."""
        Utils.initialize_model_parallel(1, 1)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=12,
            num_attention_heads=4,
            muon_ht_radius=2.5,
            perform_initialization=False,
        )
        model = nn.Linear(12, 12, bias=False, device='cuda')
        weight_before = model.weight.detach().clone()

        result = prepare_existing_model_chunks_for_distributed_training(
            [model],
            config,
            ProcessGroupCollection.use_mpu_process_groups(),
            wrap_with_ddp=False,
            mixed_precision_wrapper=None,
        )

        torch.testing.assert_close(result[0].weight, weight_before, rtol=0.0, atol=0.0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_embedding_init(self):

        Utils.initialize_model_parallel(1, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)

        tp1 = VocabParallelEmbedding(
            num_embeddings=16,
            embedding_dim=4,
            init_method=self.transformer_config.init_method,
            config=self.transformer_config,
        ).weight
        Utils.destroy_model_parallel()

        Utils.initialize_model_parallel(4, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(41)  # intentionally different.
        tp4 = VocabParallelEmbedding(
            num_embeddings=16,
            embedding_dim=4,
            init_method=self.transformer_config.init_method,
            config=self.transformer_config,
        ).weight

        rank = ps.get_tensor_model_parallel_rank()
        assert tp4.shape[0] * 4 == tp1.shape[0]
        assert torch.equal(tp1[rank * 4 : (rank + 1) * 4], tp4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_row_init(self):

        Utils.initialize_model_parallel(1, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)

        tp1 = RowParallelLinear(
            input_size=16,
            output_size=16,
            init_method=self.transformer_config.init_method,
            bias=True,
            input_is_parallel=False,
            config=self.transformer_config,
            skip_bias_add=False,
        ).weight
        Utils.destroy_model_parallel()

        Utils.initialize_model_parallel(4, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(41)  # intentionally different.
        tp4 = RowParallelLinear(
            input_size=16,
            output_size=16,
            init_method=self.transformer_config.init_method,
            bias=True,
            input_is_parallel=False,
            config=self.transformer_config,
            skip_bias_add=False,
        ).weight

        rank = ps.get_tensor_model_parallel_rank()
        assert tp4.shape[1] * 4 == tp1.shape[1]
        assert torch.equal(tp1[:, rank * 4 : (rank + 1) * 4], tp4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_col_init(self):

        Utils.initialize_model_parallel(1, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)

        tp1 = ColumnParallelLinear(
            input_size=16,
            output_size=16,
            init_method=self.transformer_config.init_method,
            bias=True,
            config=self.transformer_config,
            skip_bias_add=False,
        ).weight
        Utils.destroy_model_parallel()

        Utils.initialize_model_parallel(4, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(41)  # intentionally different.
        tp4 = ColumnParallelLinear(
            input_size=16,
            output_size=16,
            init_method=self.transformer_config.init_method,
            bias=True,
            config=self.transformer_config,
            skip_bias_add=False,
        ).weight

        rank = ps.get_tensor_model_parallel_rank()
        assert tp4.shape[0] * 4 == tp1.shape[0]
        assert torch.equal(tp1[rank * 4 : (rank + 1) * 4], tp4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.timeout(100)
    def test_te_col_init(self):

        Utils.initialize_model_parallel(1, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)

        tp1 = TEColumnParallelLinear(
            input_size=16,
            output_size=16,
            init_method=self.transformer_config.init_method,
            bias=True,
            config=self.transformer_config,
            skip_bias_add=False,
            gather_output=False,
            is_expert=False,
        ).weight
        Utils.destroy_model_parallel()

        Utils.initialize_model_parallel(4, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(41)  # intentionally different.
        tp4 = TEColumnParallelLinear(
            input_size=16,
            output_size=16,
            init_method=self.transformer_config.init_method,
            bias=True,
            config=self.transformer_config,
            skip_bias_add=False,
            gather_output=False,
            is_expert=False,
        ).weight

        if torch.distributed.get_rank() == 0:
            assert tp4.shape[0] * 4 == tp1.shape[0]
            assert torch.allclose(tp1[:4], tp4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.timeout(100)
    def test_te_row_init(self):

        Utils.initialize_model_parallel(1, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)

        tp1 = TERowParallelLinear(
            input_size=16,
            output_size=16,
            init_method=self.transformer_config.init_method,
            bias=True,
            input_is_parallel=True,
            config=self.transformer_config,
            skip_bias_add=False,
            is_expert=False,
        ).weight
        Utils.destroy_model_parallel()

        Utils.initialize_model_parallel(4, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(41)  # intentionally different.
        tp4 = TERowParallelLinear(
            input_size=16,
            output_size=16,
            init_method=self.transformer_config.init_method,
            bias=True,
            input_is_parallel=True,
            config=self.transformer_config,
            skip_bias_add=False,
            is_expert=False,
        ).weight

        if torch.distributed.get_rank() == 0:
            assert tp4.shape[1] * 4 == tp1.shape[1]
            assert torch.allclose(tp1[:, :4], tp4)
