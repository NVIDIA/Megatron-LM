# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Hybrid data-parallel placement and training tests for the MFSDP v2 adapter."""

import os

import pytest
import torch
from torch.distributed.tensor import Replicate, Shard

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import RowAtomic
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.distributed.mfsdp_v2.mcore_training_utils import (
    run_forward_backward_on_microbatches,
)
from tests.unit_tests.distributed.mfsdp_v2.mcore_utils import destroy_model_parallel
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel_cleanup(distributed_setup):
    """Initialize distributed state and clean up the topology created by each test."""
    yield
    destroy_model_parallel()


def _config() -> TransformerConfig:
    return TransformerConfig(
        num_layers=2,
        hidden_size=16,
        num_attention_heads=4,
        ffn_hidden_size=32,
        bf16=True,
        params_dtype=torch.bfloat16,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )


def _build_model_and_optimizer(
    config, outer_size, outer_strategy, inner_strategy="optim_grads_params"
):
    """Build a model and optimizer over the already-initialized DP topology.

    ``outer_size`` must match what initialize_model_parallel was given: it selects the
    adapter's mesh, while the process groups it maps onto come from the caller.
    """
    process_group_collection = ProcessGroupCollection.use_mpu_process_groups()
    model_parallel_cuda_manual_seed(1234)
    model = FullyShardedDataParallel(
        config=config,
        ddp_config=DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            megatron_fsdp_version=2,
            use_distributed_optimizer=False,
            data_parallel_sharding_strategy=inner_strategy,
            num_distributed_optimizer_instances=outer_size,
            outer_dp_sharding_strategy=outer_strategy,
        ),
        module=TransformerBlock(config=config, spec=get_gpt_layer_local_spec()).to(
            device="cuda", dtype=config.params_dtype
        ),
        pg_collection=process_group_collection,
    )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="sgd",
            lr=1.0e-2,
            weight_decay=0.0,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=False,
            clip_grad=0.0,
        ),
        [model],
    )
    return model, optimizer


def _train(model, optimizer, steps=3, microbatches=1):
    """Train an existing model and return per-step losses.

    With ``microbatches`` > 1 each step accumulates gradients, and every microbatch
    but the last runs inside ``no_sync`` -- which is how MCore's schedules tell a
    data-parallel wrapper which backward finalizes gradients.
    """
    losses = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        batches = []
        for index in range(microbatches):
            # Rank-dependent but step-deterministic input, so every configuration
            # sees the same global batch however the domain is split. Microbatches
            # differ so that dropping any of them changes the result.
            hidden = torch.arange(
                1, model.config.hidden_size + 1, device="cuda", dtype=torch.bfloat16
            ).view(1, 1, -1).expand(8, 2, -1) * (torch.distributed.get_rank() + 1 + step + index)
            batches.append(hidden)
        batch_iterator = iter(batches)

        def forward():
            output = model(hidden_states=next(batch_iterator), attention_mask=None)
            return torch.nn.functional.mse_loss(
                output.float(), torch.zeros_like(output, dtype=torch.float32)
            )

        step_losses = run_forward_backward_on_microbatches(
            model, forward, num_microbatches=microbatches
        )
        success, _, _ = optimizer.step()
        assert success
        # No update happens until optimizer.step(), so every microbatch in a step sees
        # the same parameters; averaging them matches what train_step reports.
        losses.append(torch.stack(step_losses).float().mean())
    return torch.stack(losses)


@pytest.mark.parametrize("inner_strategy", ["optim_grads", "optim_grads_params"])
@pytest.mark.parametrize("outer_strategy", ["no_shard", "optim"], ids=["hsdp", "hfsdp"])
def test_hybrid_placements(model_parallel_cleanup, outer_strategy, inner_strategy):
    """Parameters and gradients expose the configured hybrid optimizer placements."""
    Utils.initialize_model_parallel(1, 1, num_distributed_optimizer_instances=2)
    process_group_collection = ProcessGroupCollection.use_mpu_process_groups()
    model_parallel_cuda_manual_seed(1234)
    config = _config()
    model = FullyShardedDataParallel(
        config=config,
        ddp_config=DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            megatron_fsdp_version=2,
            use_distributed_optimizer=False,
            data_parallel_sharding_strategy=inner_strategy,
            num_distributed_optimizer_instances=2,
            outer_dp_sharding_strategy=outer_strategy,
        ),
        module=TransformerBlock(config=config, spec=get_gpt_layer_local_spec()).to(
            device="cuda", dtype=config.params_dtype
        ),
        pg_collection=process_group_collection,
    )
    expected_outer = Replicate() if outer_strategy == "no_shard" else Shard(0)
    for parameter in model.parameters():
        assert parameter.data.placements == (expected_outer, Shard(0))

    hidden = torch.randn(8, 2, config.hidden_size, device="cuda", dtype=torch.bfloat16)

    def forward():
        output = model(hidden_states=hidden, attention_mask=None)
        return torch.nn.functional.mse_loss(
            output.float(), torch.zeros_like(output, dtype=torch.float32), reduction="sum"
        )

    run_forward_backward_on_microbatches(model, forward)

    graded = [p for p in model.parameters() if p.grad is not None]
    assert graded, "no gradients to inspect"
    for parameter in graded:
        assert parameter.grad.device_mesh.mesh_dim_names == ("dp_outer", "dp_shard")
        assert parameter.grad.placements == (expected_outer, Shard(0))


@pytest.mark.parametrize("microbatches", [1, 2])
@pytest.mark.parametrize("outer_strategy", ["no_shard", "optim"])
@pytest.mark.parametrize("inner_strategy", ["optim_grads", "optim_grads_params"])
def test_hybrid_matches_zero3_reference(
    model_parallel_cleanup, outer_strategy, inner_strategy, microbatches
):
    """Compare losses against ZeRO-3 sharded across all DP ranks.

    With eight ranks, the reference uses a 1D mesh of size 8 and the hybrid
    run uses a 2D mesh of shape (2 outer, 4 inner).
    Both runs use the same global batches, including no_sync accumulation.
    """
    config = _config()
    Utils.initialize_model_parallel(1, 1, num_distributed_optimizer_instances=1)
    model, optimizer = _build_model_and_optimizer(config, 1, "no_shard")
    reference = _train(model, optimizer, microbatches=microbatches)
    del model, optimizer
    destroy_model_parallel()

    outer_size = 2
    Utils.initialize_model_parallel(1, 1, num_distributed_optimizer_instances=outer_size)
    model, optimizer = _build_model_and_optimizer(
        config, outer_size, outer_strategy, inner_strategy
    )
    actual = _train(model, optimizer, microbatches=microbatches)
    assert torch.isfinite(reference).all()
    torch.testing.assert_close(actual, reference, rtol=1e-2, atol=0)


@pytest.mark.parametrize("ep_size", [1, 2])
@pytest.mark.parametrize("dense_outer_strategy", ["optim", "no_shard"])
@pytest.mark.parametrize("expert_outer_strategy", ["optim", "no_shard"])
def test_moe_with_independent_hybrid_placements(
    model_parallel_cleanup, dense_outer_strategy, expert_outer_strategy, ep_size
):
    """Check dense/expert placements and finite losses across accumulated optimizer steps."""
    outer_size = 2
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size % (outer_size * ep_size):
        pytest.skip("MoE + hybrid needs a world size divisible by EP * 2 (outer DP size).")

    Utils.initialize_model_parallel(
        1, 1, expert_model_parallel_size=ep_size, num_distributed_optimizer_instances=outer_size
    )
    process_group_collection = ProcessGroupCollection.use_mpu_process_groups()
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(1234)
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_moe_experts=4,
        expert_model_parallel_size=ep_size,
        moe_layer_freq=[0, 1],
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
    model = FullyShardedDataParallel(
        config=config,
        ddp_config=DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            megatron_fsdp_version=2,
            use_distributed_optimizer=False,
            data_parallel_sharding_strategy="optim_grads",
            expert_data_parallel_sharding_strategy="optim_grads_params",
            num_distributed_optimizer_instances=outer_size,
            outer_dp_sharding_strategy=dense_outer_strategy,
            expert_outer_dp_sharding_strategy=expert_outer_strategy,
        ),
        module=HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=128,
            max_sequence_length=8,
            hybrid_layer_pattern="*E",
            pg_collection=process_group_collection,
        ).cuda(),
        pg_collection=process_group_collection,
    )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="sgd",
            lr=1.0e-3,
            weight_decay=0.0,
            use_distributed_optimizer=False,
            clip_grad=0.0,
        ),
        [model],
    )

    for name, submodule in model.named_modules():
        if isinstance(submodule, FsdpModule):
            inner = RowAtomic() if "experts" in name else Replicate()
            for group in submodule.parameter_groups:
                assert group.model_weight.placements == (Replicate(), inner), name

    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)

        def forward():
            input_ids = torch.randint(0, 128, (2, 8), device="cuda")
            position_ids = torch.arange(8, device="cuda").repeat(2, 1)
            output = model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
            return torch.nn.functional.mse_loss(
                output.float(), torch.zeros_like(output, dtype=torch.float32)
            )

        losses = run_forward_backward_on_microbatches(
            model, forward, num_microbatches=2, loss_scale=0.5
        )
        assert torch.isfinite(torch.stack(losses)).all()
        success, _, _ = optimizer.step()
        assert success

    dense_outer = Replicate() if dense_outer_strategy == "no_shard" else Shard(0)
    expert_outer = Replicate() if expert_outer_strategy == "no_shard" else Shard(0)
    seen_expert = set()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            # Router weights are dense; expert weights live under mlp.experts.
            is_expert = "experts" in name
            seen_expert.add(is_expert)
            outer = expert_outer if is_expert else dense_outer
            assert parameter.grad.device_mesh.mesh_dim_names == ("dp_outer", "dp_shard")
            assert parameter.grad.placements == (outer, Shard(0)), name
    assert seen_expert == {False, True}
