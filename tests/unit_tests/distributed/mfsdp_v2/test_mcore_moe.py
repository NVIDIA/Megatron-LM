# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MoE training and gradient clipping tests for the MFSDP v2 adapter."""

import logging
from dataclasses import replace

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
    get_containing_parameter_group,
)
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.fully_sharded_optimizer import FullyShardedOptimizer
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


def _assert_shard_matches_reference(parameter, reference, *, gradient=False):
    """Compare the owned packed-buffer slice, including empty parameter shards."""
    group = get_containing_parameter_group(parameter)
    index = next(i for i, item in enumerate(group.fsdp_parameters) if item.sharded is parameter)
    buffer = group.pre_optimizer_main_grad if gradient else group.main_weight
    actual = parameter.grad if gradient else parameter.detach()
    owned = buffer._get_owned_range(index)
    local = actual.to_local().reshape(-1)
    if owned is None:
        assert local.numel() == 0
    else:
        expected = (
            reference.detach()
            .contiguous()
            .view(-1)
            .narrow(0, owned.tensor_relative_offset, owned.numel)
        )
        torch.testing.assert_close(
            local, expected, msg=lambda message: f"{group.fsdp_parameters[index].fqns}: {message}"
        )


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


@pytest.mark.parametrize("enable_mhc", [False, True])
def test_build_train_step_and_clip(
    process_group_collections, distributed_setup, enable_mhc, monkeypatch
):
    """Match dense training with expert-DP sharding, including mHC branch calls."""
    process_group_collection, reference_process_group_collection = process_group_collections
    # mHC's native mapping matmuls must use FP32 for different local/global batch shapes.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    world_size = distributed_setup.world_size
    # The in-process EP=1 reference needs rank-invariant initialization. GPU expert
    # initialization instead uses the globally configured EP=2 rank in its RNG seed.
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_moe_experts=4,
        expert_model_parallel_size=2,
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
        enable_mhc_connections=enable_mhc,
        mhc_num_residual_streams=2,
        normalization="RMSNorm",
    )
    # Pair CPU initialization with an explicit common seed for the reference and EP model.
    torch.manual_seed(123)
    reference_config = replace(config, expert_model_parallel_size=1)
    reference_model = HybridModel(
        config=reference_config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=128,
        max_sequence_length=8,
        hybrid_layer_pattern="*E",
        pg_collection=reference_process_group_collection,
    ).cuda()
    model = HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=128,
        max_sequence_length=8,
        hybrid_layer_pattern="*E",
        pg_collection=process_group_collection,
    ).cuda()
    model.load_state_dict(reference_model.state_dict(), strict=False)
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    reference_parameters = dict(reference_model.named_parameters())
    reference_names = {id(parameter): name for name, parameter in reference_parameters.items()}
    parameter_mapping = {name: name for name in parameter_names.values()}
    for model_layer, reference_layer in zip(model.decoder.layers, reference_model.decoder.layers):
        if enable_mhc:
            model_layer = model_layer.inner_layer
            reference_layer = reference_layer.inner_layer
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
                    if isinstance(model_parameter, torch.nn.Parameter):
                        parameter_mapping[parameter_names[id(model_parameter)]] = reference_names[
                            id(reference_parameter)
                        ]
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
    moe_layer = model.module.decoder.layers[1]
    if enable_mhc:
        assert all(isinstance(layer, FsdpModule) for layer in model.module.decoder.layers)
        assert not isinstance(moe_layer.inner_layer, FsdpModule)
        moe_layer = moe_layer.inner_layer
    assert isinstance(moe_layer.mlp.experts, FsdpModule)

    optimizer_config = OptimizerConfig(
        lr=1.0e-3, weight_decay=0.0, use_distributed_optimizer=False, clip_grad=1.0e-4
    )
    reference_optimizer = get_megatron_optimizer(optimizer_config, [reference_model])
    optimizer = get_megatron_optimizer(optimizer_config, [model])
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

    def forward():
        output = model(
            input_ids=input_ids[input_slice],
            position_ids=position_ids[input_slice],
            attention_mask=None,
        )
        return torch.nn.functional.mse_loss(output, targets[input_slice])

    parameters = dict(model.module.named_parameters())
    reference_losses, losses = [], []
    for _ in range(5):
        reference_optimizer.zero_grad(set_to_none=True)
        reference_loss, _ = run_forward_backward_on_microbatches(
            reference_model, reference_forward, num_microbatches=2, loss_scale=0.5
        )
        optimizer.zero_grad(set_to_none=True)
        loss, _ = run_forward_backward_on_microbatches(
            model, forward, num_microbatches=2, loss_scale=0.5
        )
        for name, reference_name in parameter_mapping.items():
            actual, expected = parameters[name].grad, reference_parameters[reference_name].grad
            assert (actual is None) == (expected is None), name
            if expected is not None:
                assert not any(placement.is_partial() for placement in actual.placements), name
                _assert_shard_matches_reference(parameters[name], expected, gradient=True)
        # A plain reference model has no DDP hooks maintaining main_grad aliases.
        for parameter in reference_parameters.values():
            if hasattr(parameter, "main_grad"):
                parameter.main_grad = parameter.grad
        reference_success, reference_pre_clip_norm, _ = reference_optimizer.step()
        assert reference_success
        assert (
            reference_pre_clip_norm > optimizer_config.clip_grad
        ), "Reference gradients must exceed the clipping threshold to exercise clipping."
        success, _, _ = optimizer.step()
        assert success
        for name, reference_name in parameter_mapping.items():
            _assert_shard_matches_reference(parameters[name], reference_parameters[reference_name])
        torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
        losses.append(loss)
        reference_losses.append(reference_loss)

    losses = torch.stack(losses)
    reference_losses = torch.stack(reference_losses)
    if torch.distributed.get_rank() == 0:
        logger.info("MFSDP v2 EP loss curve: %s", losses.tolist())
        logger.info("MFSDP v2 EP reference loss curve: %s", reference_losses.tolist())
    assert torch.isfinite(losses).all()
    assert torch.isfinite(reference_losses).all()
    assert losses[-1] < losses[0]
    torch.testing.assert_close(losses, reference_losses)
