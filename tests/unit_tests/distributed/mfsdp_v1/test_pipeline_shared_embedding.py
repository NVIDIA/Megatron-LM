# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Meta initialization must preserve shared embedding identity across PP stages."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallelV1
from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    uneven_dtensor_to_full_tensor,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module
from tests.unit_tests.test_utilities import Utils


class StageEmbedding(torch.nn.Linear):
    def __init__(self, value):
        self.value = value
        super().__init__(16, 17, bias=False, device='meta', dtype=torch.bfloat16)
        self.weight.shared_embedding = True
        self.weight._tensor_parallel_mode = 'column'

    def reset_parameters(self):
        torch.nn.init.constant_(self.weight, self.value)


class EmbeddingStage(torch.nn.Module):
    def __init__(self, sharing, stage):
        super().__init__()
        self.embedding = StageEmbedding(
            (stage + 1) * 10 + parallel_state.get_tensor_model_parallel_rank()
        )
        self.share_embeddings_and_output_weights = sharing == 'tied'
        self.embd_group = parallel_state.get_embedding_group()

    def shared_embedding_or_output_weight(self):
        return self.embedding.weight

    def forward(self, value):
        return self.embedding(value)


@pytest.mark.parametrize('pp', [1, 2])
@pytest.mark.parametrize('tp', [1, 2])
@pytest.mark.parametrize('sharing', ['tied', 'mtp', 'none'])
@pytest.mark.parametrize('wrapped', [False, True])
def test_materialized_embedding_replicas_and_compute_weights(pp, tp, sharing, wrapped):
    """Use the real v1 constructor, master buffers and BF16 forward hooks."""
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
            init_model_with_meta_device=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            mtp_num_layers=1 if sharing == 'mtp' else None,
        )
        module = EmbeddingStage(sharing, stage)
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
            ),
            module=module,
            fsdp_unit_modules=[StageEmbedding],
        )
        expected = (stage + 1) * 10 if sharing == 'none' else 10
        name = 'module.embedding.weight' if wrapped else 'embedding.weight'
        master = model.param_and_grad_buffer.dist_main_weight[name]
        full = uneven_dtensor_to_full_tensor(master)
        reference = torch.cat(
            [torch.full((17, 16), expected + i, device='cuda') for i in range(tp)]
        )
        torch.testing.assert_close(full, reference, rtol=0, atol=0, check_dtype=False)
        output = model(torch.ones(2, 16, device='cuda', dtype=torch.bfloat16))
        assert torch.all(
            output == (expected + parallel_state.get_tensor_model_parallel_rank()) * 16
        )
    finally:
        Utils.destroy_model_parallel()
