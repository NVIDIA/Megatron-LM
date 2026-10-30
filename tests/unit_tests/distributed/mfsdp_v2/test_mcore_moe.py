# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MoE training and gradient clipping tests for the MFSDP v2 adapter."""

import logging
from dataclasses import replace

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.fully_sharded_optimizer import FullyShardedOptimizer
from megatron.core.optimizer.optimizer import ChainedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import MoETransformerLayer
from tests.unit_tests.distributed.mfsdp_v2.mcore_training_utils import (
    run_forward_backward_on_microbatches,
)
from tests.unit_tests.distributed.mfsdp_v2.mcore_utils import destroy_model_parallel
from tests.unit_tests.test_utilities import Utils

logger = logging.getLogger(__name__)


@pytest.fixture
def process_group_collections(distributed_setup):
    """Yield EP=2 process groups and singleton reference groups, then clean them up."""
    world_size = distributed_setup.world_size
    if world_size < 2 or world_size % 2:
        pytest.skip("MFSDP v2 EP adapter test requires an even world size of at least two.")
    Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=2)
    process_group_collection = ProcessGroupCollection.use_mpu_process_groups()
    assert process_group_collection.ep.size() == 2
    assert process_group_collection.expt_dp.size() == world_size // 2
    reference_group = torch.distributed.new_group(
        [torch.distributed.get_rank()], use_local_synchronization=True
    )
    reference_process_group_collection = ProcessGroupCollection(
        tp=reference_group,
        expt_tp=reference_group,
        cp=reference_group,
        pp=reference_group,
        tp_cp=reference_group,
        tp_dp_cp=reference_group,
        ep=reference_group,
        tp_ep=reference_group,
        expt_dp=reference_group,
        dp=reference_group,
        dp_cp=reference_group,
        embd=None,
        pos_embd=None,
        mp=reference_group,
        tp_ep_pp=reference_group,
    )
    model_parallel_cuda_manual_seed(1234)
    yield process_group_collection, reference_process_group_collection
    destroy_model_parallel()


@pytest.mark.parametrize(
    ("optimizer_name", "layer_pattern"), [("adam", "*E"), ("muon", "*E"), ("muon", "*-*E")]
)
def test_build_train_step_and_clip(
    process_group_collections, distributed_setup, optimizer_name, layer_pattern
):
    """Shard experts over expert-DP and clip their combined gradients."""
    process_group_collection, reference_process_group_collection = process_group_collections
    world_size = distributed_setup.world_size
    # The in-process EP=1 reference needs rank-invariant initialization. GPU expert
    # initialization instead uses the globally configured EP=2 rank in its RNG seed.
    config = TransformerConfig(
        num_layers=len(layer_pattern),
        hidden_size=64,
        num_attention_heads=4,
        num_moe_experts=4,
        expert_model_parallel_size=2,
        moe_layer_freq=[int(layer == "E") for layer in layer_pattern],
        moe_token_dispatcher_type="alltoall",
        moe_router_topk=2,
        moe_grouped_gemm=True,
        moe_ffn_hidden_size=128,
        add_bias_linear=False,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        gradient_accumulation_fusion=False,
        attention_backend=AttnBackend.unfused,
    )
    # Pair CPU initialization with an explicit common seed for the reference and EP model.
    torch.manual_seed(123)
    reference_config = replace(config, expert_model_parallel_size=1)
    reference_model = HybridModel(
        config=reference_config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=128,
        max_sequence_length=8,
        hybrid_layer_pattern=layer_pattern,
        pg_collection=reference_process_group_collection,
    ).cuda()
    model = HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=128,
        max_sequence_length=8,
        hybrid_layer_pattern=layer_pattern,
        pg_collection=process_group_collection,
    ).cuda()
    model.load_state_dict(reference_model.state_dict(), strict=False)
    for model_layer, reference_layer in zip(model.decoder.layers, reference_model.decoder.layers):
        if not isinstance(model_layer, MoETransformerLayer):
            continue
        for fc in ("linear_fc1", "linear_fc2"):
            model_fc = getattr(model_layer.mlp.experts, fc)
            reference_fc = getattr(reference_layer.mlp.experts, fc)
            for local, global_ in enumerate(model_layer.mlp.local_expert_indices):
                for parameter_name in ("weight", "bias"):
                    model_parameter = getattr(model_fc, f"{parameter_name}{local}", None)
                    reference_parameter = getattr(reference_fc, f"{parameter_name}{global_}", None)
                    if model_parameter is not None:
                        model_parameter.data.copy_(reference_parameter.data)
    reference_model.ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
    model = FullyShardedDataParallel(
        config=config,
        ddp_config=DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            megatron_fsdp_version=2,
            use_distributed_optimizer=False,
            data_parallel_sharding_strategy="optim_grads_params",
            fsdp_all_gather_in_start_param_sync=False,
        ),
        module=model,
        pg_collection=process_group_collection,
    )
    assert isinstance(model.module, FsdpModule)
    assert isinstance(model.module.decoder.layers[-1].mlp.experts, FsdpModule)

    optimizer_config = OptimizerConfig(
        optimizer=optimizer_name,
        lr=1.0e-3,
        weight_decay=0.0,
        use_distributed_optimizer=False,
        clip_grad=1.0e-4,
        muon_split_qkv=False,
        # FP32 Newton-Schulz avoids BF16 rounding differences between EP and the reference.
        muon_fp32_matmul_prec="highest",
    )
    reference_optimizer = get_megatron_optimizer(
        optimizer_config,
        [reference_model],
        pg_collection=reference_process_group_collection,
        use_gloo_process_groups=False,
    )
    optimizer = get_megatron_optimizer(
        optimizer_config,
        [model],
        pg_collection=process_group_collection,
        use_gloo_process_groups=False,
    )
    if optimizer_name == "muon":
        assert isinstance(optimizer, ChainedOptimizer)
    else:
        assert isinstance(optimizer, FullyShardedOptimizer)
    optimizer.reload_model_params()

    local_batch_size = 2
    torch.manual_seed(4321)
    input_ids = torch.randint(0, 128, (world_size * local_batch_size, 8), device="cuda")
    position_ids = torch.arange(8, device="cuda").repeat(world_size * local_batch_size, 1)
    targets = torch.randn(world_size * local_batch_size, 8, 128, device="cuda")
    input_slice = slice(
        torch.distributed.get_rank() * local_batch_size,
        (torch.distributed.get_rank() + 1) * local_batch_size,
    )

    def reference_forward():
        output = reference_model(
            input_ids=input_ids, position_ids=position_ids, attention_mask=None
        )
        return torch.nn.functional.mse_loss(output, targets)

    reference_losses = []
    reference_norms = []
    for _ in range(5):
        reference_optimizer.zero_grad(set_to_none=True)
        (reference_loss,) = run_forward_backward_on_microbatches(reference_model, reference_forward)
        reference_success, reference_pre_clip_norm, _ = reference_optimizer.step()
        assert reference_success
        assert (
            reference_pre_clip_norm > optimizer_config.clip_grad
        ), "Reference gradients must exceed the clipping threshold to exercise clipping."
        reference_losses.append(reference_loss)
        reference_norms.append(float(reference_pre_clip_norm))

    def forward():
        output = model(
            input_ids=input_ids[input_slice],
            position_ids=position_ids[input_slice],
            attention_mask=None,
        )
        return torch.nn.functional.mse_loss(output, targets[input_slice])

    losses = []
    norms = []
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        (loss,) = run_forward_backward_on_microbatches(model, forward)
        success, norm, _ = optimizer.step()
        assert success
        torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
        losses.append(loss)
        norms.append(float(norm))

    losses = torch.stack(losses)
    reference_losses = torch.stack(reference_losses)
    if torch.distributed.get_rank() == 0:
        logger.info("MFSDP v2 EP loss curve: %s", losses.tolist())
        logger.info("MFSDP v2 EP reference loss curve: %s", reference_losses.tolist())
    assert torch.isfinite(losses).all()
    assert torch.isfinite(reference_losses).all()
    assert losses[-1] < losses[0]
    torch.testing.assert_close(losses, reference_losses)
    torch.testing.assert_close(torch.tensor(norms), torch.tensor(reference_norms))
