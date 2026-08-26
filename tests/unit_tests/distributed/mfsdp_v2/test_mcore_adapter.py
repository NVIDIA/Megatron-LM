# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MCore adapter and optimizer integration tests for experimental MFSDP v2."""

import logging
import os
from dataclasses import replace

import pytest
import torch

import megatron.core.distributed.fsdp.mcore_fsdp_adapter as mcore_fsdp_adapter
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.fully_sharded_optimizer import FullyShardedOptimizer
from megatron.core.optimizer.optimizer_cuda_graph import OptimizerCudaGraphWrapper
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import MoETransformerLayer, TransformerLayer
from tests.unit_tests.test_utilities import Utils

logger = logging.getLogger(__name__)


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


class TestMcoreAdapterDense:
    """Exercise a dense MCore transformer block over two data-parallel ranks."""

    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        model_parallel_cuda_manual_seed(1234)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def test_init_model_with_meta_device_initializes_fsdp_v2_parameters(self):
        """init_model_with_meta_device should materialize FSDP v2 parameters with configured values."""

        def initialize_to_constant(weight):
            return torch.nn.init.constant_(weight, 0.25)

        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=4,
            init_method=initialize_to_constant,
            output_layer_init_method=initialize_to_constant,
            use_cpu_initialization=True,
            # The FSDP adapter uses this flag to materialize and initialize meta parameters.
            init_model_with_meta_device=True,
        )
        # MCore local linear layers otherwise allocate directly on CUDA. With CPU
        # initialization enabled, their allocations inherit this meta device context.
        with torch.device("meta"):
            meta_layer = _build_layer(config)
        meta_parameters = list(meta_layer.parameters())
        assert meta_parameters
        assert all(parameter.is_meta for parameter in meta_parameters)

        wrapped = FullyShardedDataParallel(
            config=config,
            ddp_config=DistributedDataParallelConfig(
                use_megatron_fsdp=True,
                megatron_fsdp_version=2,
                use_distributed_optimizer=False,
                data_parallel_sharding_strategy="optim_grads_params",
            ),
            module=meta_layer,
            fsdp_unit_modules=[TransformerLayer],
            pg_collection=self.pg_collection,
        )

        assert isinstance(wrapped.module, FsdpModule)

        parameters = dict(wrapped.module.named_parameters())
        assert parameters
        for name, parameter in parameters.items():
            local_parameter = parameter.to_local()

            if name.endswith("bias"):
                expected_value = 0.0
            elif "layernorm" in name:
                expected_value = 1.0
            else:
                expected_value = 0.25
            torch.testing.assert_close(
                local_parameter,
                torch.full_like(local_parameter, expected_value),
                rtol=0,
                atol=0,
                msg=f"{name} was not initialized to {expected_value}",
            )

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
                use_distributed_optimizer=False,
                data_parallel_sharding_strategy="optim_grads_params",
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

    @pytest.mark.parametrize("optimizer_cuda_graph", [False, True], ids=["eager", "cuda_graph"])
    def test_build_train_and_step(self, optimizer_cuda_graph):
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

        ddp_config = DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            megatron_fsdp_version=2,
            use_distributed_optimizer=False,
            data_parallel_sharding_strategy="optim_grads_params",
        )
        if optimizer_cuda_graph:
            # Capturable TE FusedAdam requires the main-gradient and main-weight dtypes
            # to match (https://github.com/NVIDIA/TransformerEngine/issues/3358).
            ddp_config.megatron_fsdp_main_grads_dtype = ddp_config.megatron_fsdp_main_params_dtype

        model = FullyShardedDataParallel(
            config=config, ddp_config=ddp_config, module=model, pg_collection=self.pg_collection
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
        optimizer_config = replace(
            reference_optimizer_config,
            optimizer_cuda_graph=optimizer_cuda_graph,
            use_precision_aware_optimizer=True,
            exp_avg_dtype=torch.bfloat16,
            exp_avg_sq_dtype=torch.bfloat16,
        )
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
        if optimizer_cuda_graph:
            optimizer.step = OptimizerCudaGraphWrapper(optimizer.step, cuda_graph_warmup_steps=1)

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
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
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

        if optimizer_cuda_graph:
            # The first step is eager; capture replays once and every subsequent step replays.
            graph_launches = sum(event.name == "cudaGraphLaunch" for event in prof.events())
            assert graph_launches == len(steps) - 1

        for state in optimizer.optimizer.state.values():
            assert state["exp_avg"].dtype == torch.bfloat16
            assert state["exp_avg_sq"].dtype == torch.bfloat16

        losses = torch.stack(losses)
        reference_losses = torch.stack(reference_losses)
        assert torch.isfinite(losses).all()
        assert torch.isfinite(reference_losses).all()
        torch.testing.assert_close(losses, reference_losses, rtol=1e-3, atol=0)

    def test_fused_sgd_casts_mismatched_grads(self):
        """FusedSGD steps after MCore casts V2's BF16 gradients to FP32."""
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
            module=_build_block(config),
            pg_collection=self.pg_collection,
        )
        optimizer_config = OptimizerConfig(
            optimizer="sgd",
            lr=1.0e-3,
            weight_decay=0.0,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=False,
            clip_grad=0.0,
        )
        optimizer = get_megatron_optimizer(
            optimizer_config,
            [model],
            use_gloo_process_groups=False,
            pg_collection=self.pg_collection,
        )

        optimizer.zero_grad(set_to_none=True)
        output = model(
            hidden_states=torch.randn(
                8, 2, config.hidden_size, device="cuda", dtype=torch.bfloat16
            ),
            attention_mask=None,
        )
        output.float().square().mean().backward()

        success, _, _ = optimizer.step()
        assert success


class TestMcoreAdapterExpertParallel:
    """Exercise the MFSDP v2 adapter over an MoE model with EP=2."""

    def setup_method(self):
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if self.world_size < 2 or self.world_size % 2:
            pytest.skip("MFSDP v2 EP adapter test requires an even world size of at least two.")
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=2)
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        assert self.pg_collection.ep.size() == 2
        assert self.pg_collection.expt_dp.size() == self.world_size // 2
        self.reference_group = torch.distributed.new_group(
            [torch.distributed.get_rank()], use_local_synchronization=True
        )
        self.reference_pg_collection = ProcessGroupCollection(
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

    def teardown_method(self):
        torch.distributed.destroy_process_group(self.reference_group)
        Utils.destroy_model_parallel()

    def test_build_train_and_step(self):
        """Shard experts over expert-DP and dense parameters over full DP."""
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
            pg_collection=self.reference_pg_collection,
        ).cuda()
        model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=128,
            max_sequence_length=8,
            hybrid_layer_pattern="*E",
            pg_collection=self.pg_collection,
        ).cuda()
        model.load_state_dict(reference_model.state_dict(), strict=False)
        for model_layer, reference_layer in zip(
            model.decoder.layers, reference_model.decoder.layers
        ):
            if not isinstance(model_layer, MoETransformerLayer):
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
            pg_collection=self.pg_collection,
        )
        assert isinstance(model.module, FsdpModule)
        assert isinstance(model.module.decoder.layers[1].mlp.experts, FsdpModule)

        optimizer_config = OptimizerConfig(
            lr=1.0e-3, weight_decay=0.0, use_distributed_optimizer=False, clip_grad=0.0
        )
        reference_optimizer = get_megatron_optimizer(
            optimizer_config, [reference_model], use_gloo_process_groups=False
        )
        optimizer = get_megatron_optimizer(
            replace(optimizer_config), [model], use_gloo_process_groups=False
        )
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
        reference_losses = []
        for _ in range(5):
            reference_optimizer.zero_grad(set_to_none=True)
            reference_loss = torch.nn.functional.mse_loss(
                reference_model(
                    input_ids=input_ids, position_ids=position_ids, attention_mask=None
                ),
                targets,
            )
            reference_loss.backward()
            reference_success, _, _ = reference_optimizer.step()
            assert reference_success
            reference_losses.append(reference_loss.detach())

        losses = []
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(
                model(
                    input_ids=input_ids[input_slice],
                    position_ids=position_ids[input_slice],
                    attention_mask=None,
                ),
                targets[input_slice],
            )
            loss.backward()
            success, _, _ = optimizer.step()
            assert success
            loss = loss.detach()
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
