# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MoE training and gradient clipping tests for the MFSDP v2 adapter."""

import logging
import os
from dataclasses import replace

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.models.hybrid.hybrid_layer_specs import (
    hybrid_stack_spec,
    wide_residual_hybrid_stack_spec,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.hybrid.shortcut_block import ShortcutMoEBlock
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.fully_sharded_optimizer import FullyShardedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig, WideResidualConfig
from megatron.core.transformer.transformer_layer import MoETransformerLayer, TransformerLayer
from tests.unit_tests.distributed.mfsdp_v2.mcore_training_utils import (
    run_forward_backward_on_microbatches,
)
from tests.unit_tests.distributed.mfsdp_v2.mcore_utils import destroy_model_parallel
from tests.unit_tests.test_utilities import Utils

logger = logging.getLogger(__name__)


@pytest.mark.usefixtures("_adapter_setup")
class TestMcoreMoE:
    """Exercise the MFSDP v2 adapter over an MoE model with EP=2."""

    @pytest.fixture
    def _adapter_setup(self, distributed_setup):
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if self.world_size < 2 or self.world_size % 2:
            pytest.skip("MFSDP v2 EP adapter test requires an even world size of at least two.")
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=2)
        self.process_group_collection = ProcessGroupCollection.use_mpu_process_groups()
        assert self.process_group_collection.ep.size() == 2
        assert self.process_group_collection.expt_dp.size() == self.world_size // 2
        self.reference_group = torch.distributed.new_group(
            [torch.distributed.get_rank()], use_local_synchronization=True
        )
        self.reference_process_group_collection = ProcessGroupCollection(
            tp=self.reference_group,
            expt_tp=self.reference_group,
            cp=self.reference_group,
            pp=self.reference_group,
            tp_cp=self.reference_group,
            tp_dp_cp=self.reference_group,
            ep=self.reference_group,
            tp_ep=self.reference_group,
            expt_dp=self.reference_group,
            dp=self.reference_group,
            dp_cp=self.reference_group,
            embd=None,
            pos_embd=None,
        )
        model_parallel_cuda_manual_seed(1234)

        yield
        destroy_model_parallel()

    @pytest.mark.parametrize(
        "shortcut, custom_units, overlap, recompute_norms, wide_residual",
        [
            (False, False, False, False, False),
            (False, True, False, False, False),
            (True, False, False, False, False),
            (True, True, False, False, False),
            (True, True, True, False, False),
            (True, True, False, True, False),
            (True, True, True, True, False),
            (False, False, False, False, True),
            (True, False, False, False, True),
            (True, False, True, False, True),
        ],
        ids=[
            "standard",
            "standard-custom-units",
            "shortcut",
            "shortcut-custom-units",
            "shortcut-custom-units-overlap",
            "shortcut-custom-units-recompute",
            "shortcut-custom-units-overlap-recompute",
            "wide-residual-moe-act",
            "wide-residual-shortcut-moe-act",
            "wide-residual-shortcut-overlap-moe-act",
        ],
    )
    def test_build_train_step_and_clip(
        self, shortcut, custom_units, overlap, recompute_norms, wide_residual
    ):
        """Shard experts over expert-DP and clip their combined gradients."""
        # The in-process EP=1 reference needs rank-invariant initialization. GPU expert
        # initialization instead uses the globally configured EP=2 rank in its RNG seed.
        config = TransformerConfig(
            num_layers=4 if wide_residual else 2,
            hidden_size=64,
            num_attention_heads=4,
            num_moe_experts=4,
            expert_model_parallel_size=2,
            moe_layer_freq=[0, 1, 0, 1] if wide_residual else [0, 1],
            moe_token_dispatcher_type="alltoall",
            moe_router_topk=2,
            moe_grouped_gemm=True,
            moe_ffn_hidden_size=128,
            moe_shortcut_connection=shortcut,
            moe_shortcut_parallel=overlap,
            moe_shortcut_post_norm=shortcut,
            moe_shared_expert_intermediate_size=128 if shortcut else None,
            add_bias_linear=False,
            use_cpu_initialization=True,
            params_dtype=torch.float32,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            gradient_accumulation_fusion=False,
            attention_backend=AttnBackend.unfused,
            wide_residual=(
                WideResidualConfig(num_streams=3, learned_retention=True) if wide_residual else None
            ),
            recompute_granularity="selective" if wide_residual or recompute_norms else None,
            recompute_modules=(
                ["moe_act"]
                if wide_residual
                else ["layernorm", "shortcut_pre_mlp_layernorm"] if recompute_norms else None
            ),
        )
        # Pair CPU initialization with an explicit common seed for the reference and EP model.
        torch.manual_seed(123)
        reference_config = replace(config, expert_model_parallel_size=1)
        reference_model = HybridModel(
            config=reference_config,
            hybrid_stack_spec=(
                wide_residual_hybrid_stack_spec if wide_residual else hybrid_stack_spec
            ),
            vocab_size=128,
            max_sequence_length=8,
            hybrid_layer_pattern="*E*E" if wide_residual else "*E",
            pg_collection=self.reference_process_group_collection,
        ).cuda()
        model = HybridModel(
            config=config,
            hybrid_stack_spec=(
                wide_residual_hybrid_stack_spec if wide_residual else hybrid_stack_spec
            ),
            vocab_size=128,
            max_sequence_length=8,
            hybrid_layer_pattern="*E*E" if wide_residual else "*E",
            pg_collection=self.process_group_collection,
        ).cuda()
        model.load_state_dict(reference_model.state_dict(), strict=False)
        for model_layer, reference_layer in zip(
            model.decoder.layers, reference_model.decoder.layers
        ):
            if isinstance(model_layer, ShortcutMoEBlock):
                model_layer = model_layer.moe_layer
                reference_layer = reference_layer.moe_layer
            if not isinstance(model_layer.mlp, MoELayer):
                continue
            for fc in ("linear_fc1", "linear_fc2"):
                model_fc = getattr(model_layer.mlp.experts, fc)
                reference_fc = getattr(reference_layer.mlp.experts, fc)
                for local, global_ in enumerate(model_layer.mlp.local_expert_indices):
                    for parameter_name in ("weight", "bias"):
                        model_parameter = getattr(model_fc, f"{parameter_name}{local}", None)
                        reference_parameter = getattr(
                            reference_fc, f"{parameter_name}{global_}", None
                        )
                        if model_parameter is not None:
                            model_parameter.data.copy_(reference_parameter.data)
        reference_model.ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        fsdp_unit_modules = None
        if custom_units:
            fsdp_unit_modules = [ShortcutMoEBlock, TransformerLayer, MoETransformerLayer, MoELayer]
            if shortcut:
                fsdp_unit_modules.append(type(model.decoder.layers[0].shortcut_pre_mlp_layernorm))
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
            fsdp_unit_modules=fsdp_unit_modules,
            pg_collection=self.process_group_collection,
        )
        assert isinstance(model.module, FsdpModule)
        if shortcut:
            block = model.module.decoder.layers[0]
            assert isinstance(block, ShortcutMoEBlock)
            assert isinstance(block, FsdpModule)
            assert not isinstance(block.compute_layer, FsdpModule)
            assert not isinstance(block.moe_layer, FsdpModule)
            assert not isinstance(block.moe_layer.mlp, FsdpModule)
            assert isinstance(block.shortcut_pre_mlp_layernorm, FsdpModule) == custom_units
            experts = block.moe_layer.mlp.experts
        else:
            experts = model.module.decoder.layers[1].mlp.experts
        assert isinstance(experts, FsdpModule)
        for parameter in experts.parameters():
            assert (
                parameter.device_mesh.get_group("expert_dp")
                == self.process_group_collection.expt_dp
            )

        optimizer_config = OptimizerConfig(
            lr=1.0e-3, weight_decay=0.0, use_distributed_optimizer=False, clip_grad=1.0e-4
        )
        reference_optimizer = get_megatron_optimizer(optimizer_config, [reference_model])
        optimizer = get_megatron_optimizer(optimizer_config, [model])
        assert isinstance(optimizer, FullyShardedOptimizer)
        optimizer.reload_model_params()

        local_batch_size = 2
        torch.manual_seed(4321)
        input_ids = torch.randint(0, 128, (self.world_size * local_batch_size, 8), device="cuda")
        position_ids = torch.arange(8, device="cuda").repeat(self.world_size * local_batch_size, 1)
        targets = torch.randn(self.world_size * local_batch_size, 8, 128, device="cuda")
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
        for _ in range(5):
            reference_optimizer.zero_grad(set_to_none=True)
            (reference_loss,) = run_forward_backward_on_microbatches(
                reference_model, reference_forward
            )
            reference_success, reference_pre_clip_norm, _ = reference_optimizer.step()
            assert reference_success
            assert (
                reference_pre_clip_norm > optimizer_config.clip_grad
            ), "Reference gradients must exceed the clipping threshold to exercise clipping."
            reference_losses.append(reference_loss)

        def forward():
            output = model(
                input_ids=input_ids[input_slice],
                position_ids=position_ids[input_slice],
                attention_mask=None,
            )
            return torch.nn.functional.mse_loss(output, targets[input_slice])

        losses = []
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            (loss,) = run_forward_backward_on_microbatches(model, forward)
            assert all(
                unit._unshard_event is None
                for unit in model.module.modules()
                if isinstance(unit, FsdpModule)
            ), "Static prefetch must not leave unused full weights after backward."
            success, _, _ = optimizer.step()
            assert success
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
            losses.append(loss)

        losses = torch.stack(losses)
        reference_losses = torch.stack(reference_losses)
        if torch.distributed.get_rank() == 0:
            logger.info("MFSDP v2 EP loss curve: %s", losses.tolist())
            logger.info("MFSDP v2 EP reference loss curve: %s", reference_losses.tolist())
        assert torch.isfinite(losses).all()
        assert torch.isfinite(reference_losses).all()
        assert losses[-1] < losses[0]
        torch.testing.assert_close(losses, reference_losses)

    @pytest.mark.parametrize(
        "unit_modules, shard_shortcut",
        [
            (None, True),
            ([ShortcutMoEBlock], True),
            ([TransformerLayer, MoETransformerLayer, MoELayer], False),
            ([], False),
        ],
        ids=["default", "explicit-shortcut", "other-types", "root-only"],
    )
    def test_shortcut_ownership_and_weight_lifetime(self, unit_modules, shard_shortcut):
        """Skip helper-only containers inside shortcuts, but shard ordinary sibling layers."""
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
        )
        model = HybridModel(
            config=config,
            hybrid_stack_spec=wide_residual_hybrid_stack_spec,
            vocab_size=128,
            max_sequence_length=8,
            hybrid_layer_pattern="*E*",
            pg_collection=self.process_group_collection,
        ).cuda()
        block = model.decoder.layers[0]
        names_before = set(model.state_dict())
        assert any("compute_layer.residual_read_self_attn.read_map" in key for key in names_before)
        assert any("moe_layer.mlp.experts" in key for key in names_before)
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
            fsdp_unit_modules=unit_modules,
            pg_collection=self.process_group_collection,
        )
        assert set(model.module.state_dict()) == names_before
        assert isinstance(block, FsdpModule) == shard_shortcut
        assert not isinstance(block.compute_layer, FsdpModule)
        assert not isinstance(block.moe_layer, FsdpModule)
        assert not isinstance(block.moe_layer.mlp, FsdpModule)
        assert isinstance(block.moe_layer.mlp.experts, FsdpModule)
        shard_ordinary_layer = unit_modules is None or TransformerLayer in unit_modules
        assert isinstance(model.module.decoder.layers[1], FsdpModule) == shard_ordinary_layer
        for child in (
            block.compute_layer.residual_read_self_attn,
            block.compute_layer.residual_write_self_attn,
            block.moe_layer.residual_read_mlp,
            block.moe_layer.residual_write_mlp,
            block.shortcut_pre_mlp_layernorm,
            block.shortcut_post_norm,
        ):
            assert not isinstance(child, FsdpModule)
        expected = [module for module in model.module.modules() if isinstance(module, FsdpModule)]
        assert list(model.module._context.forward_order) == expected
        calls = []
        handles = []
        for module in expected:

            def record(module, args):
                calls.append(module)

            handles.append(module.register_forward_pre_hook(record, prepend=True))
        optimizer = get_megatron_optimizer(
            OptimizerConfig(lr=1.0e-3, use_distributed_optimizer=False), [model]
        )
        optimizer.reload_model_params()
        for _ in range(3):
            calls.clear()
            optimizer.zero_grad(set_to_none=True)
            x = torch.randint(0, 128, (2, 8), device="cuda")
            positions = torch.arange(8, device="cuda").repeat(2, 1)

            def forward():
                output = model(input_ids=x, position_ids=positions, attention_mask=None)
                assert calls == expected
                assert all(
                    group._unsharded_model_weight.local_buffer.untyped_storage().nbytes() == 0
                    for group in block.moe_layer.mlp.experts.parameter_groups
                )
                loss = output.square().mean()
                assert torch.isfinite(loss)
                return loss

            run_forward_backward_on_microbatches(model, forward)
            assert all(module._unshard_event is None for module in expected)
            success, _, _ = optimizer.step()
            assert success
        for handle in handles:
            handle.remove()
