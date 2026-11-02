# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MoE training and gradient clipping tests for the MFSDP v2 adapter."""

import logging
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
from megatron.core.models.hybrid.hybrid_layer_specs import (
    hybrid_stack_spec,
    wide_residual_hybrid_stack_spec,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.fully_sharded_optimizer import FullyShardedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.moe.experts import SequentialMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig, WideResidualConfig
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
    )
    model_parallel_cuda_manual_seed(1234)
    yield process_group_collection, reference_process_group_collection
    destroy_model_parallel()


@pytest.mark.parametrize("grouped_gemm", [True, False], ids=["grouped", "sequential"])
def test_standard_moe_training_and_clipping(
    process_group_collections, distributed_setup, grouped_gemm
):
    """Compare ordinary MoE training and clipping against an EP=1 reference."""
    process_group_collection, reference_process_group_collection = process_group_collections
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_moe_experts=4,
        expert_model_parallel_size=2,
        moe_layer_freq=[0, 1],
        moe_token_dispatcher_type="alltoall",
        moe_router_topk=2,
        moe_grouped_gemm=grouped_gemm,
        moe_ffn_hidden_size=128,
        moe_shared_expert_intermediate_size=128,
        add_bias_linear=False,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        gradient_accumulation_fusion=False,
        attention_backend=AttnBackend.unfused,
    )
    stack_spec = deepcopy(hybrid_stack_spec)
    stack_spec.submodules.moe_layer.submodules.mlp = get_moe_module_spec(
        use_te=True, num_experts=config.num_moe_experts, moe_grouped_gemm=grouped_gemm
    )
    torch.manual_seed(123)
    reference_model = HybridModel(
        config=replace(config, expert_model_parallel_size=1),
        hybrid_stack_spec=stack_spec,
        vocab_size=128,
        max_sequence_length=8,
        hybrid_layer_pattern="*E",
        pg_collection=reference_process_group_collection,
    ).cuda()
    model = HybridModel(
        config=config,
        hybrid_stack_spec=stack_spec,
        vocab_size=128,
        max_sequence_length=8,
        hybrid_layer_pattern="*E",
        pg_collection=process_group_collection,
    ).cuda()
    _sync_model_parameters(reference_model, model)
    reference_model.ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
    model = FullyShardedDataParallel(
        config=model.config,
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

    layer = model.module.decoder.layers[1]
    assert isinstance(layer, FsdpModule)
    assert not isinstance(layer.mlp, FsdpModule)
    assert not isinstance(layer.mlp.router, FsdpModule)
    assert not isinstance(layer.mlp.shared_experts, FsdpModule)
    assert isinstance(layer.mlp.experts, FsdpModule)

    optimizer_config = OptimizerConfig(
        lr=1.0e-3, weight_decay=0.0, use_distributed_optimizer=False, clip_grad=1.0e-4
    )
    reference_optimizer = get_megatron_optimizer(optimizer_config, [reference_model])
    optimizer = get_megatron_optimizer(optimizer_config, [model])
    assert isinstance(optimizer, FullyShardedOptimizer)
    optimizer.reload_model_params()

    world_size = distributed_setup.world_size
    local_batch_size = 2
    torch.manual_seed(4321)
    input_ids = torch.randint(0, 128, (world_size * local_batch_size, 8), device="cuda")
    position_ids = torch.arange(8, device="cuda").repeat(world_size * local_batch_size, 1)
    targets = torch.randn(world_size * local_batch_size, 8, 128, device="cuda")
    input_slice = slice(
        torch.distributed.get_rank() * local_batch_size,
        (torch.distributed.get_rank() + 1) * local_batch_size,
    )

    reference_losses = _train(
        reference_model, reference_optimizer, input_ids, position_ids, targets
    )
    losses = _train(
        model, optimizer, input_ids[input_slice], position_ids[input_slice], targets[input_slice]
    )
    torch.distributed.all_reduce(losses, op=torch.distributed.ReduceOp.AVG)
    if torch.distributed.get_rank() == 0:
        logger.info("MFSDP v2 EP loss curve: %s", losses.tolist())
        logger.info("MFSDP v2 EP reference loss curve: %s", reference_losses.tolist())
    assert torch.isfinite(losses).all()
    assert torch.isfinite(reference_losses).all()
    assert losses[-1] < losses[0]
    torch.testing.assert_close(losses, reference_losses)


@pytest.mark.parametrize("overlap", [False, True], ids=["no-overlap", "overlap"])
def test_shortcut_moe_training_and_clipping(process_group_collections, distributed_setup, overlap):
    """Keep shortcut parameters together while matching reference training with wide residuals."""
    process_group_collection, reference_process_group_collection = process_group_collections
    config = TransformerConfig(
        num_layers=3,
        hidden_size=64,
        num_attention_heads=4,
        num_moe_experts=4,
        expert_model_parallel_size=2,
        moe_layer_freq=[0, 1, 0],
        moe_token_dispatcher_type="alltoall",
        moe_router_topk=2,
        moe_grouped_gemm=True,
        moe_ffn_hidden_size=128,
        moe_shortcut_connection=True,
        moe_shortcut_parallel=overlap,
        moe_shortcut_post_norm=True,
        moe_shared_expert_intermediate_size=128,
        add_bias_linear=False,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        gradient_accumulation_fusion=False,
        attention_backend=AttnBackend.unfused,
        wide_residual=WideResidualConfig(num_streams=3, learned_retention=True),
        recompute_granularity="selective",
        recompute_modules=["moe_act"],
    )
    torch.manual_seed(123)
    reference_model = HybridModel(
        config=replace(config, expert_model_parallel_size=1),
        hybrid_stack_spec=wide_residual_hybrid_stack_spec,
        vocab_size=128,
        max_sequence_length=8,
        hybrid_layer_pattern="*E*",
        pg_collection=reference_process_group_collection,
    ).cuda()
    model = HybridModel(
        config=config,
        hybrid_stack_spec=wide_residual_hybrid_stack_spec,
        vocab_size=128,
        max_sequence_length=8,
        hybrid_layer_pattern="*E*",
        pg_collection=process_group_collection,
    ).cuda()
    _sync_model_parameters(reference_model, model)
    reference_model.ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
    model = FullyShardedDataParallel(
        config=model.config,
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

    block = model.module.decoder.layers[0]
    assert isinstance(block, FsdpModule)
    assert not isinstance(block.compute_layer, FsdpModule)
    assert not isinstance(block.moe_layer, FsdpModule)
    assert not isinstance(block.moe_layer.mlp, FsdpModule)
    assert isinstance(block.moe_layer.mlp.experts, FsdpModule)
    assert isinstance(model.module.decoder.layers[-1], FsdpModule)

    optimizer_config = OptimizerConfig(
        lr=1.0e-3, weight_decay=0.0, use_distributed_optimizer=False, clip_grad=1.0e-4
    )
    reference_optimizer = get_megatron_optimizer(optimizer_config, [reference_model])
    optimizer = get_megatron_optimizer(optimizer_config, [model])
    assert isinstance(optimizer, FullyShardedOptimizer)
    optimizer.reload_model_params()

    world_size = distributed_setup.world_size
    local_batch_size = 2
    torch.manual_seed(4321)
    input_ids = torch.randint(0, 128, (world_size * local_batch_size, 8), device="cuda")
    position_ids = torch.arange(8, device="cuda").repeat(world_size * local_batch_size, 1)
    targets = torch.randn(world_size * local_batch_size, 8, 128, device="cuda")
    input_slice = slice(
        torch.distributed.get_rank() * local_batch_size,
        (torch.distributed.get_rank() + 1) * local_batch_size,
    )

    reference_losses = _train(
        reference_model, reference_optimizer, input_ids, position_ids, targets
    )
    losses = _train(
        model, optimizer, input_ids[input_slice], position_ids[input_slice], targets[input_slice]
    )
    torch.distributed.all_reduce(losses, op=torch.distributed.ReduceOp.AVG)
    if torch.distributed.get_rank() == 0:
        logger.info("MFSDP v2 EP loss curve: %s", losses.tolist())
        logger.info("MFSDP v2 EP reference loss curve: %s", reference_losses.tolist())
    assert torch.isfinite(losses).all()
    assert torch.isfinite(reference_losses).all()
    assert losses[-1] < losses[0]
    torch.testing.assert_close(losses, reference_losses)


def _sync_model_parameters(reference_model, model):
    """Copy reference parameters, mapping global expert indices to local experts."""
    # CPU initialization gives each rank the same reference weights. Remap the
    # EP=1 reference's global expert indices to the EP=2 model's local indices.
    model.load_state_dict(reference_model.state_dict(), strict=False)
    for name, moe in model.named_modules():
        if not isinstance(moe, MoELayer):
            continue
        reference_moe = reference_model.get_submodule(name)
        if isinstance(moe.experts, SequentialMLP):
            for local, global_ in enumerate(moe.local_expert_indices):
                moe.experts.local_experts[local].load_state_dict(
                    reference_moe.experts.local_experts[global_].state_dict()
                )
            continue
        for fc in ("linear_fc1", "linear_fc2"):
            model_fc = getattr(moe.experts, fc)
            reference_fc = getattr(reference_moe.experts, fc)
            for local, global_ in enumerate(moe.local_expert_indices):
                for parameter_name in ("weight", "bias"):
                    model_parameter = getattr(model_fc, f"{parameter_name}{local}", None)
                    reference_parameter = getattr(reference_fc, f"{parameter_name}{global_}", None)
                    if model_parameter is not None:
                        model_parameter.data.copy_(reference_parameter.data)


def _train(model, optimizer, input_ids, position_ids, targets):
    """Train one model for five steps and return its stacked losses."""

    def forward():
        output = model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
        return torch.nn.functional.mse_loss(output, targets)

    losses = []
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        (loss,) = run_forward_backward_on_microbatches(model, forward)
        success, pre_clip_norm, _ = optimizer.step()
        assert success
        assert (
            pre_clip_norm > optimizer.config.clip_grad
        ), "Gradients must exceed the clipping threshold to exercise clipping."
        losses.append(loss)
    return torch.stack(losses)
