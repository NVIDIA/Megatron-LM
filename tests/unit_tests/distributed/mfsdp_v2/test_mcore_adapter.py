# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MCore adapter and optimizer integration tests for experimental MFSDP v2."""

import pytest
import torch
from torch.distributed.tensor import DTensor

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils


def _process_groups() -> ProcessGroupCollection:
    return ProcessGroupCollection.use_mpu_process_groups(
        required_pgs=["tp", "pp", "mp", "cp", "ep", "tp_ep_pp", "dp", "dp_cp", "expt_dp"]
    )


def _transformer_config(**overrides) -> TransformerConfig:
    values = dict(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        ffn_hidden_size=32,
        bf16=True,
        params_dtype=torch.bfloat16,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )
    values.update(overrides)
    return TransformerConfig(**values)


def _ddp_config(**overrides) -> DistributedDataParallelConfig:
    values = dict(
        use_megatron_fsdp=True,
        use_distributed_optimizer=False,
        data_parallel_sharding_strategy="optim_grads_params",
        megatron_fsdp_main_params_dtype=torch.float32,
        megatron_fsdp_main_grads_dtype=torch.float32,
    )
    values.update(overrides)
    return DistributedDataParallelConfig(**values)


def _build_layer(
    config: TransformerConfig, pg_collection: ProcessGroupCollection
) -> TransformerLayer:
    layer = TransformerLayer(
        config=config,
        submodules=get_gpt_layer_local_spec().submodules,
        layer_number=1,
        pg_collection=pg_collection,
        add_layer_offset=False,
    )
    return layer.to(dtype=config.params_dtype)


class TestMcoreAdapter:
    """Exercise a dense MCore transformer block over two data-parallel ranks."""

    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        if torch.distributed.get_world_size() != 2:
            pytest.skip("MFSDP v2 MCore integration test requires exactly two ranks.")
        model_parallel_cuda_manual_seed(1234)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def test_wraps_fsdp_unit_modules_before_root(self):
        config = _transformer_config()
        pg_collection = _process_groups()
        layer = _build_layer(config, pg_collection)
        model = torch.nn.Sequential(layer, torch.nn.Linear(config.hidden_size, config.hidden_size))
        model = model.to(device="cuda", dtype=config.params_dtype)

        wrapped = FullyShardedDataParallel(
            config=config,
            ddp_config=_ddp_config(),
            module=model,
            fsdp_unit_modules=[TransformerLayer],
            pg_collection=pg_collection,
        )

        assert isinstance(wrapped.module, FsdpModule)
        assert isinstance(wrapped.module[0], FsdpModule)

        # Post-order wrapping gives the selected TransformerLayer its own parameter group;
        # the root FSDP unit should own only the parameters of the remaining Linear module.
        child_parameter_names = {
            name for group in wrapped.module[0].parameter_groups() for name in group.parameter_names
        }
        root_parameter_names = {
            name for group in wrapped.module.parameter_groups() for name in group.parameter_names
        }
        assert child_parameter_names
        assert root_parameter_names == {"1.weight", "1.bias"}

    def test_build_train_and_step(self):
        config = _transformer_config(num_layers=2)
        pg_collection = _process_groups()
        model = FullyShardedDataParallel(
            config=config,
            ddp_config=_ddp_config(),
            module=TransformerBlock(
                config=config, spec=get_gpt_layer_local_spec(), pg_collection=pg_collection
            ).to(device="cuda", dtype=config.params_dtype),
        )

        assert isinstance(model.module, TransformerBlock)
        assert isinstance(model.module, FsdpModule)
        assert callable(model.module.forward)
        assert callable(model.module.sharded_state_dict)
        assert all(isinstance(parameter, DTensor) for parameter in model.module.parameters())
        fsdp_modules = [
            module for module in model.module.modules() if isinstance(module, FsdpModule)
        ]
        assert len([module for module in fsdp_modules if isinstance(module, TransformerLayer)]) == 2
        parameter_groups = [
            parameter_group
            for fsdp_module in fsdp_modules
            for parameter_group in fsdp_module.parameter_groups()
        ]
        for parameter_group in parameter_groups:
            assert parameter_group.main_weight.dtype == torch.float32
            assert parameter_group.main_grad is not None
            assert parameter_group.main_grad.dtype == torch.float32

        optimizer = get_megatron_optimizer(
            OptimizerConfig(
                optimizer="adam",
                lr=1.0e-3,
                bf16=True,
                params_dtype=torch.bfloat16,
                use_distributed_optimizer=False,
                clip_grad=0.0,
            ),
            [model],
            use_gloo_process_groups=False,
            pg_collection=pg_collection,
        )
        assert optimizer.config.use_distributed_optimizer is False

        initial_buffers = [group.main_weight.local_buffer.clone() for group in parameter_groups]
        losses = []
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            hidden_states = torch.randn(
                8, 2, config.hidden_size, device="cuda", dtype=torch.bfloat16
            )
            output = model(hidden_states=hidden_states, attention_mask=None)
            loss = output.float().square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            success, _, _ = optimizer.step()
            assert success
            losses.append(loss.detach())

        gathered_losses = [torch.empty_like(torch.stack(losses)) for _ in range(2)]
        torch.distributed.all_gather(gathered_losses, torch.stack(losses))
        torch.testing.assert_close(gathered_losses[0], gathered_losses[1])
        assert all(isinstance(parameter, DTensor) for parameter in model.parameters())
        buffer_changes = [
            (initial - group.main_weight.local_buffer).abs().max().item()
            for initial, group in zip(initial_buffers, parameter_groups)
        ]
        assert any(change > 0 for change in buffer_changes)
