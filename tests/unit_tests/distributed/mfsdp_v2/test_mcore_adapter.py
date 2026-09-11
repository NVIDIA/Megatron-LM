# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MCore adapter and optimizer integration tests for experimental MFSDP v2."""

import logging
import os
from dataclasses import replace

import pytest
import torch
from torch.distributed.distributed_c10d import _world
from torch.distributed.tensor import DTensor, Replicate, Shard

import megatron.core.distributed.fsdp.mcore_fsdp_adapter as mcore_fsdp_adapter
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.full_cuda_graph import FullCudaGraphWrapper, StaticBufferLoader
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
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


def _destroy_model_parallel():
    """Utils.destroy_model_parallel, plus the groups it leaves behind.

    It clears Megatron's references but frees only a few of the groups; c10d holds its own
    reference to the rest, so their NCCL communicators -- and, with NVLS enabled, their
    multicast reservations -- would outlive the test. See #6897.
    """
    Utils.destroy_model_parallel()
    for group in list(_world.pg_map):
        if group is not torch.distributed.group.WORLD:
            torch.distributed.destroy_process_group(group)


class TestMcoreAdapterDense:
    """Exercise a dense MCore transformer block over two data-parallel ranks."""

    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        model_parallel_cuda_manual_seed(1234)

    def teardown_method(self):
        _destroy_model_parallel()

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
            # The FSDP adapter uses this flag to materialize and initialize meta parameters.
            init_model_with_meta_device=True,
        )
        with torch.device("meta"):
            meta_layer = TransformerLayer(
                config=config,
                submodules=get_gpt_layer_with_transformer_engine_spec().submodules,
                layer_number=1,
                add_layer_offset=False,
            )
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

            # Some parameters use module-specific initializers, so only check those
            # initialized by the configured init method.
            if name.endswith("bias") or "layernorm" in name or "layer_norm" in name:
                continue
            torch.testing.assert_close(
                local_parameter,
                torch.full_like(local_parameter, 0.25),
                rtol=0,
                atol=0,
                msg=f"{name} was not initialized to 0.25",
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

    def test_build_train_and_step(self):
        """Match eager training against an MFSDP v2 train-and-step sequence."""
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
        # The reference TransformerBlock cannot use the DistributedOptimizer path, which
        # requires DDP/FSDP buffer metadata, but the optimizer factory still expects
        # every model chunk to expose ddp_config.
        reference_model.ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        model = FullyShardedDataParallel(
            config=config,
            ddp_config=DistributedDataParallelConfig(
                use_megatron_fsdp=True,
                megatron_fsdp_version=2,
                use_distributed_optimizer=False,
                data_parallel_sharding_strategy="optim_grads_params",
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
            # TODO(#7074): Exercise nonzero clipping after MFSDP v2 precision-aware
            # clipping is fixed.
            clip_grad=0.0,
        )
        reference_optimizer = get_megatron_optimizer(reference_optimizer_config, [reference_model])
        optimizer = get_megatron_optimizer(
            replace(
                reference_optimizer_config,
                use_precision_aware_optimizer=True,
                # Exercise MFSDP v2's precision-aware optimizer with BF16 moment state.
                exp_avg_dtype=torch.bfloat16,
                exp_avg_sq_dtype=torch.bfloat16,
            ),
            [model],
        )
        assert isinstance(optimizer, FullyShardedOptimizer)
        optimizer.reload_model_params()

        steps = [
            [
                torch.randn(8, 2, config.hidden_size, device="cuda", dtype=torch.bfloat16)
                for _ in range(2)
            ]
            for _ in range(10)
        ]

        def run(model, optimizer) -> torch.Tensor:
            losses = []
            for microbatches in steps:
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
            return torch.stack(losses)

        reference_losses = run(reference_model, reference_optimizer)
        losses = run(model, optimizer)

        for state in optimizer.optimizer.state.values():
            assert state["exp_avg"].dtype == torch.bfloat16
            assert state["exp_avg_sq"].dtype == torch.bfloat16
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
        optimizer = get_megatron_optimizer(optimizer_config, [model])

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

    @pytest.mark.parametrize("use_precision_aware_optimizer", [False, True])
    def test_gradient_clipping_reaches_global_norm(self, use_precision_aware_optimizer):
        """MFSDP v2 reports the true global gradient norm and clips the gradients to it.

        Main gradients are kept in the main-weight dtype so that clipping is measurable on
        parameter.grad: _copy_model_grads_to_main_grads otherwise installs a dtype-cast copy
        for non-precision-aware optimizers, clip_grad_norm scales that copy, and
        step_with_ready_grads restores the original afterwards.
        """
        clip_grad = 1.0
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
        model = FullyShardedDataParallel(
            config=config,
            ddp_config=DistributedDataParallelConfig(
                use_megatron_fsdp=True,
                megatron_fsdp_version=2,
                use_distributed_optimizer=False,
                data_parallel_sharding_strategy="optim_grads_params",
                megatron_fsdp_main_grads_dtype=torch.float32,
            ),
            module=_build_block(config),
            pg_collection=self.pg_collection,
        )
        optimizer = get_megatron_optimizer(
            OptimizerConfig(
                optimizer="adam",
                lr=1.0e-3,
                weight_decay=0.0,
                bf16=True,
                params_dtype=torch.bfloat16,
                use_distributed_optimizer=False,
                clip_grad=clip_grad,
                use_precision_aware_optimizer=use_precision_aware_optimizer,
            ),
            [model],
        )

        def global_norm(local_tensors) -> float:
            squared_norm = torch.zeros(1, dtype=torch.float32, device="cuda")
            for local_tensor in local_tensors:
                squared_norm += local_tensor.float().square().sum()
            torch.distributed.all_reduce(squared_norm)
            return squared_norm.sqrt().item()

        optimizer.zero_grad(set_to_none=True)
        output = model(
            hidden_states=(
                torch.arange(1, config.hidden_size + 1, device="cuda", dtype=torch.bfloat16)
                .view(1, 1, -1)
                .expand(8, 2, -1)
                * (torch.distributed.get_rank() + 1)
            ),
            attention_mask=None,
        )
        output.float().square().sum().backward()

        parameters = [
            parameter for parameter in optimizer.get_parameters() if parameter.grad is not None
        ]
        assert all(isinstance(parameter.grad, DTensor) for parameter in parameters)
        expected_pre_clip_norm = global_norm([p.grad.to_local() for p in parameters])
        assert (
            expected_pre_clip_norm > clip_grad
        ), "Test gradients must exceed the clipping threshold to exercise clipping."

        success, pre_clip_norm, _ = optimizer.step()

        assert success
        torch.testing.assert_close(pre_clip_norm.item(), expected_pre_clip_norm)
        torch.testing.assert_close(
            global_norm([p.grad.to_local() for p in parameters]), clip_grad, rtol=1e-3, atol=0
        )


class TestMcoreAdapterCudaGraph:
    """Exercise MFSDP v2 full-iteration and optimizer CUDA graphs together."""

    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        model_parallel_cuda_manual_seed(1234, te_rng_tracker=True, force_reset_rng=True)

    def teardown_method(self):
        # The wrappers store capture state globally. Reset it so the next test captures its
        # own work instead of replaying this test's graph.
        OptimizerCudaGraphWrapper.curr_iteration = 0
        OptimizerCudaGraphWrapper.cuda_graph = None
        OptimizerCudaGraphWrapper.result = None
        FullCudaGraphWrapper.curr_iteration = {'training': 0, 'validation': 0}
        FullCudaGraphWrapper.cuda_graph = {'training': None, 'validation': None}
        FullCudaGraphWrapper.result = {'training': None, 'validation': None}
        StaticBufferLoader.static_buffers = {'training': [], 'validation': []}
        _destroy_model_parallel()

    def test_full_iteration_and_optimizer_cuda_graph_match_eager(self):
        """Compare graph replay with an otherwise identical eager MFSDP v2 run."""
        eager_config = TransformerConfig(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=4,
            bf16=True,
            params_dtype=torch.bfloat16,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )
        graph_config = replace(eager_config, cuda_graph_impl="full_iteration")

        def build_model_and_optimizer(
            config: TransformerConfig, enable_cuda_graph: bool
        ) -> tuple[torch.nn.Module, FullyShardedOptimizer]:
            model = _build_block(config)
            model = FullyShardedDataParallel(
                config=config,
                ddp_config=DistributedDataParallelConfig(
                    use_megatron_fsdp=True,
                    megatron_fsdp_version=2,
                    use_distributed_optimizer=False,
                    data_parallel_sharding_strategy="optim_grads_params",
                    # With the default None, MFSDP v2 uses BF16 main grads with FP32 main
                    # params. Capturable FusedAdam in the TE revision under test requires
                    # matching dtypes (https://github.com/NVIDIA/TransformerEngine/issues/3358).
                    megatron_fsdp_main_grads_dtype=torch.float32,
                    megatron_fsdp_cuda_graph_mode=enable_cuda_graph,
                ),
                module=model,
                pg_collection=self.pg_collection,
            )
            optimizer = get_megatron_optimizer(
                OptimizerConfig(
                    optimizer="adam",
                    lr=1.0e-3,
                    weight_decay=0.0,
                    bf16=True,
                    params_dtype=torch.bfloat16,
                    use_distributed_optimizer=False,
                    # TODO(#7074): Exercise nonzero clipping after MFSDP v2 precision-aware
                    # clipping is fixed.
                    clip_grad=0.0,
                    optimizer_cuda_graph=enable_cuda_graph,
                    use_precision_aware_optimizer=True,
                    # Exercise MFSDP v2's precision-aware optimizer with BF16 moment state.
                    exp_avg_dtype=torch.bfloat16,
                    exp_avg_sq_dtype=torch.bfloat16,
                ),
                [model],
            )
            assert isinstance(optimizer, FullyShardedOptimizer)
            optimizer.reload_model_params()
            if enable_cuda_graph:
                # The factory makes FusedAdam capturable, while training installs this wrapper.
                optimizer.step = OptimizerCudaGraphWrapper(
                    optimizer.step, cuda_graph_warmup_steps=1
                )
            return model, optimizer

        # FullCudaGraphWrapper requires this keyword-only schedule callback signature.
        def forward_backward(*, model, data_iterator, num_microbatches, seq_length, forward_only):
            assert seq_length is None
            assert not forward_only
            microbatch_losses = []
            for _ in range(num_microbatches):
                batch = next(data_iterator[0])
                # Pipeline schedules receive model chunks as a list, including with PP=1.
                output = model[0](hidden_states=batch["hidden_states"], attention_mask=None)
                loss = output.float().square().mean()
                (loss / num_microbatches).backward()
                microbatch_losses.append({"loss": loss.detach()})
            return microbatch_losses

        steps = [
            [
                torch.randn(8, 2, eager_config.hidden_size, device="cuda", dtype=torch.bfloat16)
                for _ in range(2)
            ]
            for _ in range(10)
        ]

        def run(model, optimizer, forward_backward) -> torch.Tensor:
            losses = []
            for microbatches in steps:
                optimizer.zero_grad(set_to_none=True)
                microbatch_losses = forward_backward(
                    model=[model],
                    data_iterator=[iter([{"hidden_states": batch} for batch in microbatches])],
                    num_microbatches=len(microbatches),
                    # FullCudaGraphWrapper requires the production schedule arguments.
                    seq_length=None,
                    forward_only=False,
                )
                success, _, _ = optimizer.step()
                assert success
                losses.append(torch.stack([entry["loss"] for entry in microbatch_losses]).mean())
            return torch.stack(losses)

        eager_model, eager_optimizer = build_model_and_optimizer(
            eager_config, enable_cuda_graph=False
        )
        graph_model, graph_optimizer = build_model_and_optimizer(
            graph_config, enable_cuda_graph=True
        )
        graph_model.load_state_dict(eager_model.state_dict())

        eager_losses = run(eager_model, eager_optimizer, forward_backward)
        cuda_graph_forward_backward = FullCudaGraphWrapper(
            forward_backward, cuda_graph_warmup_steps=1
        )
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            graph_losses = run(graph_model, graph_optimizer, cuda_graph_forward_backward)

        graph_launches = sum(event.name == "cudaGraphLaunch" for event in prof.events())
        assert graph_launches == 2 * (len(steps) - 1)
        assert FullCudaGraphWrapper.cuda_graph["training"] is not None
        # Verify that both runs use the requested BF16 moment state rather than silently
        # allocating FP32 state, so the comparison isolates CUDA-graph execution.
        for optimizer in (eager_optimizer, graph_optimizer):
            for state in optimizer.optimizer.state.values():
                assert state["exp_avg"].dtype == torch.bfloat16
                assert state["exp_avg_sq"].dtype == torch.bfloat16
        torch.testing.assert_close(graph_losses, eager_losses, rtol=1e-3, atol=0)


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
        _destroy_model_parallel()

    def test_build_train_step_and_clip(self):
        """Shard experts over expert-DP and clip their combined gradients."""
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
            reference_success, reference_pre_clip_norm, _ = reference_optimizer.step()
            assert reference_success
            assert (
                reference_pre_clip_norm > optimizer_config.clip_grad
            ), "Reference gradients must exceed the clipping threshold to exercise clipping."
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


class TestMcoreAdapterHybrid:
    """Exercise MFSDP v2 over a hybrid data-parallel domain (an outer DP axis)."""

    def teardown_method(self):
        _destroy_model_parallel()

    @staticmethod
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

    @staticmethod
    def _train(config, instances, outer_strategy, steps=3):
        """Train over the already-initialized DP topology and return per-step losses.

        ``instances`` must match what initialize_model_parallel was given: it selects the
        adapter's mesh, while the process groups it maps onto come from the caller.
        """
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        model_parallel_cuda_manual_seed(1234)
        model = FullyShardedDataParallel(
            config=config,
            ddp_config=DistributedDataParallelConfig(
                use_megatron_fsdp=True,
                megatron_fsdp_version=2,
                use_distributed_optimizer=False,
                data_parallel_sharding_strategy="optim_grads_params",
                num_distributed_optimizer_instances=instances,
                outer_dp_sharding_strategy=outer_strategy,
            ),
            module=_build_block(config),
            pg_collection=pg_collection,
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
        losses = []
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            # Rank-dependent but step-deterministic input, so every configuration
            # sees the same global batch however the domain is split.
            hidden = torch.arange(
                1, config.hidden_size + 1, device="cuda", dtype=torch.bfloat16
            ).view(1, 1, -1).expand(8, 2, -1) * (torch.distributed.get_rank() + 1 + step)
            loss = model(hidden_states=hidden, attention_mask=None).float().square().mean()
            loss.backward()
            success, _, _ = optimizer.step()
            assert success
            losses.append(loss.detach())
        return torch.stack(losses)

    @pytest.mark.parametrize("outer_strategy", ["no_shard", "optim"], ids=["hsdp", "hfsdp"])
    def test_hybrid_placements(self, outer_strategy):
        """The outer axis takes its strategy's placement; the inner axis stays ZeRO-3."""
        Utils.initialize_model_parallel(1, 1, num_distributed_optimizer_instances=2)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        model_parallel_cuda_manual_seed(1234)
        config = self._config()
        model = FullyShardedDataParallel(
            config=config,
            ddp_config=DistributedDataParallelConfig(
                use_megatron_fsdp=True,
                megatron_fsdp_version=2,
                use_distributed_optimizer=False,
                data_parallel_sharding_strategy="optim_grads_params",
                num_distributed_optimizer_instances=2,
                outer_dp_sharding_strategy=outer_strategy,
            ),
            module=_build_block(config),
            pg_collection=pg_collection,
        )
        output = model(
            hidden_states=torch.randn(
                8, 2, config.hidden_size, device="cuda", dtype=torch.bfloat16
            ),
            attention_mask=None,
        )
        output.float().square().sum().backward()

        expected_outer = Replicate() if outer_strategy == "no_shard" else Shard(0)
        graded = [p for p in model.parameters() if p.grad is not None]
        assert graded, "no gradients to inspect"
        for parameter in graded:
            assert parameter.grad.device_mesh.mesh_dim_names == ("dp_outer", "dp_shard")
            assert parameter.grad.placements == (expected_outer, Shard(0))

    @pytest.mark.parametrize("outer_strategy", ["no_shard", "optim"], ids=["hsdp", "hfsdp"])
    def test_hybrid_matches_single_instance(self, outer_strategy):
        """Splitting the DP domain must not change the math: same losses as one instance."""
        config = self._config()
        # The instance count is fixed by initialize_model_parallel, so comparing two
        # topologies means initializing twice. teardown_method destroys the second.
        Utils.initialize_model_parallel(1, 1, num_distributed_optimizer_instances=1)
        reference = self._train(config, instances=1, outer_strategy="no_shard")
        _destroy_model_parallel()

        Utils.initialize_model_parallel(1, 1, num_distributed_optimizer_instances=2)
        hybrid = self._train(config, instances=2, outer_strategy=outer_strategy)
        assert torch.isfinite(reference).all()
        torch.testing.assert_close(hybrid, reference, rtol=1e-2, atol=0)

    @pytest.mark.parametrize("dense_outer_strategy", ["optim", "no_shard"])
    @pytest.mark.parametrize("expert_outer_strategy", ["optim", "no_shard"])
    def test_moe_with_independent_hybrid_placements(
        self, dense_outer_strategy, expert_outer_strategy
    ):
        """Dense and expert parameters use different placements on the same hybrid mesh."""
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size < 4 or world_size % 4:
            pytest.skip("MoE + hybrid needs a world size divisible by four (EP=2, instances=2).")

        Utils.initialize_model_parallel(
            1, 1, expert_model_parallel_size=2, num_distributed_optimizer_instances=2
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(1234)
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
        model = FullyShardedDataParallel(
            config=config,
            ddp_config=DistributedDataParallelConfig(
                use_megatron_fsdp=True,
                megatron_fsdp_version=2,
                use_distributed_optimizer=False,
                data_parallel_sharding_strategy="optim_grads_params",
                num_distributed_optimizer_instances=2,
                outer_dp_sharding_strategy=dense_outer_strategy,
                expert_outer_dp_sharding_strategy=expert_outer_strategy,
            ),
            module=HybridModel(
                config=config,
                hybrid_stack_spec=hybrid_stack_spec,
                vocab_size=128,
                max_sequence_length=8,
                hybrid_layer_pattern="*E",
                pg_collection=pg_collection,
            ).cuda(),
            pg_collection=pg_collection,
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

        optimizer.zero_grad(set_to_none=True)
        input_ids = torch.randint(0, 128, (2, 8), device="cuda")
        position_ids = torch.arange(8, device="cuda").repeat(2, 1)
        output = model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
        output.float().square().mean().backward()
        success, _, _ = optimizer.step()
        assert success

        mesh_dim_names = {
            parameter.grad.device_mesh.mesh_dim_names
            for parameter in model.parameters()
            if parameter.grad is not None
        }
        assert mesh_dim_names == {("dp_outer", "dp_shard")}

        dense_parameters = []
        expert_parameters = []
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            # In this model, expert weights live under mlp.experts; router weights are dense.
            if "experts" in name:
                expert_parameters.append((name, parameter))
            else:
                dense_parameters.append((name, parameter))
        dense_outer = Replicate() if dense_outer_strategy == "no_shard" else Shard(0)
        for name, parameter in dense_parameters:
            assert parameter.grad.placements == (dense_outer, Shard(0)), name

        expert_outer = Replicate() if expert_outer_strategy == "no_shard" else Shard(0)
        for name, parameter in expert_parameters:
            assert parameter.grad.placements == (expert_outer, Shard(0)), name
