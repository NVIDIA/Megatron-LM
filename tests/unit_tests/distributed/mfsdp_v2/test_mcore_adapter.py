# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MCore adapter and optimizer integration tests for experimental MFSDP v2."""

from dataclasses import replace

import pytest
import torch

import megatron.core.distributed.fsdp.mcore_fsdp_adapter as mcore_fsdp_adapter
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.utils import find_megatron_fsdp
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.fully_sharded_optimizer import FullyShardedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
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

    def test_wraps_fsdp_unit_modules_before_root(self, monkeypatch):
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
                use_distributed_optimizer=False,
                data_parallel_sharding_strategy="optim_grads_params",
                megatron_fsdp_main_params_dtype=torch.float32,
                megatron_fsdp_main_grads_dtype=torch.float32,
            ),
            module=model,
            fsdp_unit_modules=[TransformerLayer],
            pg_collection=ProcessGroupCollection(dp_cp=self.pg_collection.dp_cp),
        )

        assert isinstance(wrapped.module, FsdpModule)
        assert isinstance(wrapped.module[0], FsdpModule)
        assert all(
            getattr(parameter, "__fsdp_param__", False) for parameter in wrapped.parameters()
        )

        # Post-order wrapping gives the selected TransformerLayer its own parameter group;
        # the root FSDP unit should own only the parameters of the remaining Linear module.
        child_parameter_names = {
            name
            for group in wrapped.module[0].parameter_groups
            for parameter in group.fsdp_parameters
            for name in parameter.fqns
        }
        root_parameter_names = {
            name
            for group in wrapped.module.parameter_groups
            for parameter in group.fsdp_parameters
            for name in parameter.fqns
        }
        assert child_parameter_names
        assert root_parameter_names == {"1.weight", "1.bias"}

    def test_nccl_ub_enables_symmetric_memory(self, monkeypatch):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=4,
            ffn_hidden_size=32,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        model = torch.nn.Linear(config.hidden_size, config.hidden_size).to(
            device="cuda", dtype=config.params_dtype
        )
        fully_shard_context_calls = []
        original_fully_shard_context = mcore_fsdp_adapter.fully_shard_context

        def record_fully_shard_context(*args, **kwargs):
            fully_shard_context_calls.append(kwargs["use_symmetric_memory"])
            return original_fully_shard_context(*args, **kwargs)

        monkeypatch.setattr(mcore_fsdp_adapter, "fully_shard_context", record_fully_shard_context)
        FullyShardedDataParallel(
            config=config,
            ddp_config=DistributedDataParallelConfig(
                use_megatron_fsdp=True,
                megatron_fsdp_version=2,
                data_parallel_sharding_strategy="optim_grads_params",
                nccl_ub=True,
            ),
            module=model,
            pg_collection=self.pg_collection,
        )

        assert fully_shard_context_calls == [True]

    def test_build_train_and_step(self):
        root_calls = []
        monkeypatch.setattr(
            wrapped.module,
            "pre_backward",
            lambda *, register_final_callback: root_calls.append(
                ("pre_backward", register_final_callback)
            ),
        )
        monkeypatch.setattr(
            wrapped.module,
            "post_backward",
            lambda *, finalize_context: root_calls.append(("post_backward", finalize_context)),
        )
        wrapped._setup_1f1b_overlap_interface()
        assert find_megatron_fsdp(wrapped) is wrapped
        assert find_megatron_fsdp(wrapped.module) is wrapped.module

        # The overlap schedule initializes the root before calling child modules directly.
        assert wrapped.module._context is None
        wrapped._replace_param_with_raw_if_needed()
        assert wrapped.module.is_root()
        assert wrapped.module[0].context is wrapped.module.context

        # no_sync() is entered before the first forward in the non-pipeline schedule.
        with wrapped.no_sync():
            assert not wrapped.module.context.is_last_microbatch
        assert wrapped.module.context.is_last_microbatch

        release_calls = []
        monkeypatch.setattr(
            wrapped.module[0], "reshard_parameters", lambda: release_calls.append("reshard")
        )
        monkeypatch.setattr(
            wrapped.module[0], "reduce_grad", lambda: release_calls.append("reduce_grad")
        )
        wrapped.post_forward_release_module(wrapped.module[0])
        wrapped.post_backward_release_module(wrapped.module[0])
        assert release_calls == ["reshard", "reshard", "reduce_grad"]
        wrapped.pre_backward()
        wrapped.post_backward()
        assert root_calls == [("pre_backward", False), ("post_backward", True)]

    def test_build_train_and_step(self, monkeypatch):
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
                use_distributed_optimizer=False,
                data_parallel_sharding_strategy="optim_grads_params",
                megatron_fsdp_main_params_dtype=torch.float32,
                megatron_fsdp_main_grads_dtype=torch.bfloat16,
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
        reference_optimizer = get_megatron_optimizer(
            reference_optimizer_config, [reference_model], use_gloo_process_groups=False
        )
        optimizer_config = replace(reference_optimizer_config)
        with pytest.raises(
            ValueError, match="MFSDP v2 currently requires use_distributed_optimizer=False"
        ):
            get_megatron_optimizer(
                replace(reference_optimizer_config, use_distributed_optimizer=True),
                [model],
                use_gloo_process_groups=False,
            )
        optimizer = get_megatron_optimizer(optimizer_config, [model], use_gloo_process_groups=False)
        assert isinstance(optimizer, FullyShardedOptimizer)
        optimizer.reload_model_params()

        parameter_groups = [
            parameter_group
            for module in model.modules()
            if isinstance(module, FsdpModule)
            for parameter_group in module.parameter_groups
        ]
        assert parameter_groups
        sync_counts = {parameter_group: 0 for parameter_group in parameter_groups}
        for parameter_group in parameter_groups:
            sync_model_weight = parameter_group.sync_model_weight_from_main_weight

            def count_sync(parameter_group=parameter_group, sync_model_weight=sync_model_weight):
                sync_counts[parameter_group] += 1
                sync_model_weight()

            monkeypatch.setattr(parameter_group, "sync_model_weight_from_main_weight", count_sync)

        steps = [
            [
                torch.randn(8, 2, config.hidden_size, device="cuda", dtype=torch.bfloat16)
                for _ in range(2)
            ]
            for _ in range(10)
        ]

        reference_losses = []
        for microbatches in steps:
            reference_optimizer.zero_grad(set_to_none=True)
            microbatch_losses = []
            for batch in microbatches:
                reference_output = reference_model(hidden_states=batch, attention_mask=None)
                reference_loss = reference_output.float().square().mean()
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
                loss = output.float().square().mean()
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
