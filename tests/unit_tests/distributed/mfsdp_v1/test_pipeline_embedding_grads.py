# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Shared embedding gradients must reduce across PP, including empty local shards."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import _allreduce_word_embedding_grads
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallelV1
from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    uneven_dtensor_to_full_tensor,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module
from tests.unit_tests.test_utilities import Utils


class StageEmbedding(torch.nn.Linear):
    def __init__(self, value, rows):
        self.value = value
        super().__init__(16, rows, bias=False, device='cuda', dtype=torch.bfloat16)
        self.weight.shared_embedding = True
        self.weight._tensor_parallel_mode = 'column'

    def reset_parameters(self):
        torch.nn.init.constant_(self.weight, self.value)


class EmbeddingStage(torch.nn.Module):
    def __init__(self, sharing, stage, rows):
        super().__init__()
        self.embedding = StageEmbedding(
            (stage + 1) * 10 + parallel_state.get_tensor_model_parallel_rank(), rows
        )
        self.share_embeddings_and_output_weights = sharing == 'tied'
        self.embd_group = parallel_state.get_embedding_group()
        self.pre_process = stage == 0

    def shared_embedding_or_output_weight(self):
        return self.embedding.weight

    def forward(self, value):
        return self.embedding(value)


@pytest.mark.parametrize('rows', [1, 17])
@pytest.mark.parametrize('pp', [1, 2])
@pytest.mark.parametrize('tp', [1, 2])
@pytest.mark.parametrize('sharing', ['tied', 'mtp', 'none'])
@pytest.mark.parametrize('wrapped', [False, True])
@pytest.mark.parametrize('decoupled', [False, True])
def test_shared_embedding_gradient_reduction(pp, tp, sharing, wrapped, decoupled, rows):
    """Exercise ordinary and decoupled gradients through real v1 BF16 backward."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp)
    try:
        model_parallel_cuda_manual_seed(123)
        stage = parallel_state.get_pipeline_model_parallel_rank()
        config = TransformerConfig(
            num_layers=pp,
            hidden_size=16,
            num_attention_heads=4,
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            pipeline_dtype=torch.bfloat16,
            init_model_with_meta_device=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            mtp_num_layers=1 if sharing == 'mtp' else None,
        )
        module = EmbeddingStage(sharing, stage, rows)
        if wrapped:
            module = Float16Module(config=config, module=module)
        model = FullyShardedDataParallelV1(
            config=config,
            ddp_config=DistributedDataParallelConfig(
                use_megatron_fsdp=True,
                data_parallel_sharding_strategy='optim_grads_params',
                overlap_grad_reduce=True,
                overlap_param_gather=True,
                megatron_fsdp_main_params_dtype=torch.float32,
                megatron_fsdp_use_decoupled_grad=decoupled,
            ),
            module=module,
            fsdp_unit_modules=[StageEmbedding],
        )
        expected = (stage + 1) * 10
        name = 'module.embedding.weight' if wrapped else 'embedding.weight'
        output = model(torch.ones(2, 16, device='cuda', dtype=torch.bfloat16))
        assert torch.all(
            output == (expected + parallel_state.get_tensor_model_parallel_rank()) * 16
        )
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        row_weights = torch.arange(1 + tp_rank * rows, 1 + (tp_rank + 1) * rows, device='cuda')
        ((output.float() * row_weights).sum() * (stage + 1)).backward()
        model.finish_grad_sync()
        _allreduce_word_embedding_grads(
            [model],
            config,
            embd_group=parallel_state.get_embedding_group(),
            pp_group=parallel_state.get_pipeline_model_parallel_group(),
        )
        gradient = model.param_and_grad_buffer.dist_main_grad[name]
        if rows == 1:
            empty_shards = torch.tensor(int(gradient._local_tensor.numel() == 0), device='cuda')
            torch.distributed.all_reduce(empty_shards)
            assert empty_shards.item() > 0
        expected_grad = 6 if pp == 2 and sharing != 'none' else 2 * (stage + 1)
        reference_grad = torch.arange(1, tp * rows + 1, device='cuda').view(-1, 1) * expected_grad
        assert torch.all(uneven_dtensor_to_full_tensor(gradient) == reference_grad)
    finally:
        Utils.destroy_model_parallel()
