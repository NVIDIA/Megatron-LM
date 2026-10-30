# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compare mHC loss with and without MFSDP v2."""

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.distributed.mfsdp_v2.mcore_utils import destroy_model_parallel
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def process_group_collection(distributed_setup):
    """Initialize TP=PP=1 so FSDP shards over the data-parallel ranks."""
    Utils.initialize_model_parallel(1, 1)
    torch.manual_seed(1234)  # mHC also initializes parameters on the CPU.
    model_parallel_cuda_manual_seed(1234)
    yield ProcessGroupCollection.use_mpu_process_groups()
    destroy_model_parallel()


@pytest.mark.parametrize("recompute_mhc", [False, True], ids=["no-recompute", "recompute"])
def test_mhc_loss_matches_reference(process_group_collection, recompute_mhc):
    """FSDP preserves the reference loss with mHC recompute on or off."""
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=128,
        add_bias_linear=False,
        params_dtype=torch.float32,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        gradient_accumulation_fusion=False,
        attention_backend=AttnBackend.unfused,
        enable_mhc_connections=True,
        mhc_num_residual_streams=2,
        mhc_sinkhorn_iterations=3,
        use_fused_mhc=True,
        recompute_granularity="selective" if recompute_mhc else None,
        recompute_modules=["mhc"] if recompute_mhc else None,
        mhc_recompute_layer_num=2 if recompute_mhc else None,
    )
    reference, model = [
        HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=128,
            max_sequence_length=8,
            hybrid_layer_pattern="**",
            pg_collection=process_group_collection,
        ).cuda()
        for _ in range(2)
    ]
    model.load_state_dict(reference.state_dict())
    model = FullyShardedDataParallel(
        config=config,
        ddp_config=DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            megatron_fsdp_version=2,
            use_distributed_optimizer=False,
            data_parallel_sharding_strategy="optim_grads_params",
        ),
        module=model,
        pg_collection=process_group_collection,
    )

    input_ids = torch.arange(16, device="cuda").reshape(2, 8)
    position_ids = torch.arange(8, device="cuda").repeat(2, 1)
    losses = []
    for current_model in (reference, model):
        output = current_model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
        loss = output.square().mean()
        loss.backward()  # Exercise mHC replay, which runs during backward.
        losses.append(loss.detach())
    model.finish_grad_sync()
    torch.testing.assert_close(losses[0], losses[1])
