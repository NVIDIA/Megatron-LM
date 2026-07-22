# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MCore adapter and optimizer integration tests for experimental MFSDP v2."""

import os
from copy import copy
from dataclasses import replace

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.fully_sharded_optimizer import FullyShardedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_layer import BaseMoELayer
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils


def _build_layer(config: TransformerConfig) -> TransformerLayer:
    return TransformerLayer(
        config=config,
        submodules=get_gpt_layer_local_spec().submodules,
        layer_number=1,
        add_layer_offset=False,
    )


def _build_block(config: TransformerConfig) -> TransformerBlock:
    return TransformerBlock(config=config, spec=get_gpt_layer_local_spec()).to(
        device="cuda", dtype=config.params_dtype
    )


def _mfsdp_v2_config() -> DistributedDataParallelConfig:
    return DistributedDataParallelConfig(
        use_megatron_fsdp=True,
        megatron_fsdp_version=2,
        use_distributed_optimizer=True,
        data_parallel_sharding_strategy="optim_grads_params",
        megatron_fsdp_main_params_dtype=torch.float32,
        megatron_fsdp_main_grads_dtype=torch.float32,
        fsdp_all_gather_in_start_param_sync=False,
    )


class _ToyMoELayer(BaseMoELayer):
    """Minimal ownership-only MoE layer with routed and dense subtrees."""

    def __init__(self, hidden_size: int):
        torch.nn.Module.__init__(self)
        self.experts = torch.nn.Sequential(torch.nn.Linear(hidden_size, hidden_size, bias=False))
        for parameter in self.experts.parameters():
            parameter.allreduce = False
        self.router = torch.nn.Linear(hidden_size, 2, bias=False)
        self.shared_experts = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.experts(hidden_states) + self.shared_experts(hidden_states)


class TestMcoreAdapter:
    """Exercise a dense MCore transformer block over two data-parallel ranks."""

    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        if torch.distributed.get_world_size() < 2:
            pytest.skip("MFSDP v2 MCore integration test requires at least two ranks.")
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        model_parallel_cuda_manual_seed(1234)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def test_wraps_fsdp_unit_modules_before_root(self):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=4,
            ffn_hidden_size=32,
            bf16=True,
            params_dtype=torch.bfloat16,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )
        layer = _build_layer(config)
        model = torch.nn.Sequential(layer, torch.nn.Linear(config.hidden_size, config.hidden_size))
        model = model.to(device="cuda", dtype=config.params_dtype)

        wrapped = FullyShardedDataParallel(
            config=config,
            ddp_config=DistributedDataParallelConfig(
                use_megatron_fsdp=True,
                megatron_fsdp_version=2,
                use_distributed_optimizer=True,
                data_parallel_sharding_strategy="optim_grads_params",
                megatron_fsdp_main_params_dtype=torch.float32,
                megatron_fsdp_main_grads_dtype=torch.float32,
                fsdp_all_gather_in_start_param_sync=False,
            ),
            module=model,
            fsdp_unit_modules=[TransformerLayer],
            pg_collection=self.pg_collection,
        )

        assert isinstance(wrapped.module, FsdpModule)
        assert isinstance(wrapped.module[0], FsdpModule)

        # Post-order wrapping gives the selected TransformerLayer its own parameter group;
        # the root FSDP unit should own only the parameters of the remaining Linear module.
        child_parameter_names = {
            name for group in wrapped.module[0].parameter_groups for name in group.parameter_names
        }
        root_parameter_names = {
            name for group in wrapped.module.parameter_groups for name in group.parameter_names
        }
        assert child_parameter_names
        assert root_parameter_names == {"1.weight", "1.bias"}

    def test_build_train_and_step(self):
        config = TransformerConfig(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=4,
            ffn_hidden_size=32,
            bf16=True,
            params_dtype=torch.bfloat16,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )
        reference_model = _build_block(config)
        model = _build_block(config)
        model.load_state_dict(reference_model.state_dict())
        # get_megatron_optimizer expects every model chunk to expose ddp_config. The
        # reference model remains unwrapped/unsharded, so it cannot use the
        # DistributedOptimizer path that expects DDP/FSDP buffer metadata.
        reference_model.ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)

        model = FullyShardedDataParallel(
            config=config,
            ddp_config=DistributedDataParallelConfig(
                use_megatron_fsdp=True,
                megatron_fsdp_version=2,
                use_distributed_optimizer=True,
                data_parallel_sharding_strategy="optim_grads_params",
                megatron_fsdp_main_params_dtype=torch.float32,
                megatron_fsdp_main_grads_dtype=torch.bfloat16,
                fsdp_all_gather_in_start_param_sync=False,
            ),
            module=model,
            pg_collection=self.pg_collection,
        )

        reference_optimizer_config = OptimizerConfig(
            optimizer="adam",
            lr=1.0e-3,
            weight_decay=0.0,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=False,
            clip_grad=0.0,
        )
        optimizer_config = replace(reference_optimizer_config, use_distributed_optimizer=True)
        reference_optimizer = get_megatron_optimizer(
            reference_optimizer_config, [reference_model], use_gloo_process_groups=False
        )
        optimizer = get_megatron_optimizer(optimizer_config, [model], use_gloo_process_groups=False)
        assert isinstance(optimizer, FullyShardedOptimizer)
        optimizer.reload_model_params()

        steps = [
            [
                torch.randn(8, 2, config.hidden_size, device="cuda", dtype=torch.bfloat16)
                for _ in range(2)
            ]
            for _ in range(3)
        ]

        reference_losses = []
        for microbatches in steps:
            reference_optimizer.zero_grad(set_to_none=True)
            microbatch_losses = []
            for batch in microbatches:
                reference_output = reference_model(hidden_states=batch, attention_mask=None)
                reference_loss = reference_output.square().mean()
                (reference_loss / len(microbatches)).backward()
                microbatch_losses.append(reference_loss.detach())
            reference_success, _, _ = reference_optimizer.step()
            assert reference_success
            reference_losses.append(torch.stack(microbatch_losses).mean())

        losses = []
        for microbatches in steps:
            model.zero_grad_buffer()
            optimizer.zero_grad(set_to_none=True)
            microbatch_losses = []
            for batch in microbatches:
                output = model(hidden_states=batch, attention_mask=None)
                loss = output.square().mean()
                (loss / len(microbatches)).backward()
                microbatch_losses.append(loss.detach())
            success, _, _ = optimizer.step()
            assert success
            losses.append(torch.stack(microbatch_losses).mean())

        losses = torch.stack(losses)
        reference_losses = torch.stack(reference_losses)
        assert torch.isfinite(losses).all()
        assert torch.isfinite(reference_losses).all()
        torch.testing.assert_close(losses, reference_losses, rtol=1e-3, atol=0)


class TestMcoreAdapterExpertParallel:
    """Exercise separate dense and routed-expert ownership with EP=world size."""

    def setup_method(self):
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size < 2:
            pytest.skip("MFSDP v2 + EP integration tests require at least two ranks.")
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=world_size)
        self.world_size = world_size
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def _config(self) -> TransformerConfig:
        return TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=4,
            ffn_hidden_size=32,
            num_moe_experts=self.world_size,
            expert_model_parallel_size=self.world_size,
            bf16=True,
            params_dtype=torch.bfloat16,
            gradient_accumulation_fusion=False,
        )

    def test_routes_experts_to_singleton_mesh_and_preserves_marker(self):
        model = torch.nn.Sequential(_ToyMoELayer(16), torch.nn.Linear(16, 16, bias=False)).to(
            device="cuda", dtype=torch.bfloat16
        )

        wrapped = FullyShardedDataParallel(
            config=self._config(),
            ddp_config=_mfsdp_v2_config(),
            module=model,
            pg_collection=self.pg_collection,
        )

        expert_groups = wrapped.module[0].experts.parameter_groups
        assert expert_groups
        assert all(group.mesh.size() == 1 for group in expert_groups)
        assert all(
            getattr(parameter, "allreduce", True) is False
            for group in expert_groups
            for parameter in group.sharded_parameters
        )

        dense_groups = [
            group
            for fsdp_module in wrapped.module.modules()
            if isinstance(fsdp_module, FsdpModule) and fsdp_module is not wrapped.module[0].experts
            for group in fsdp_module.parameter_groups
        ]
        assert dense_groups
        assert all(group.mesh.size() == self.world_size for group in dense_groups)

    def test_rejects_replicated_experts(self):
        model = _ToyMoELayer(16).to(device="cuda", dtype=torch.bfloat16)
        replicated_expert_pg_collection = copy(self.pg_collection)
        replicated_expert_pg_collection.expt_dp = self.pg_collection.dp_cp

        with pytest.raises(ValueError, match=r"MFSDP v2 \+ EP currently requires expert-DP size 1"):
            FullyShardedDataParallel(
                config=self._config(),
                ddp_config=_mfsdp_v2_config(),
                module=model,
                pg_collection=replicated_expert_pg_collection,
            )

    def test_rejects_expert_marker_outside_routed_container(self):
        model = torch.nn.Linear(16, 16, bias=False).to(device="cuda", dtype=torch.bfloat16)
        model.weight.allreduce = False

        with pytest.raises(ValueError, match="outside recognized routed-expert containers"):
            FullyShardedDataParallel(
                config=self._config(),
                ddp_config=_mfsdp_v2_config(),
                module=model,
                pg_collection=self.pg_collection,
            )
