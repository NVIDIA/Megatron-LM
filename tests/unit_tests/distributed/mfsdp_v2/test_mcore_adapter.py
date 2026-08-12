# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MCore adapter and optimizer integration tests for experimental MFSDP v2."""

import contextlib
from dataclasses import replace

import pytest
import torch

import megatron.core.distributed.fsdp.mcore_fsdp_adapter as mcore_fsdp_adapter
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.full_cuda_graph import (
    FullCudaGraphWrapper,
    StaticBufferLoader,
    get_shared_capture_stream,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.fully_sharded_optimizer import FullyShardedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import is_torch_min_version
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


def _train_config() -> TransformerConfig:
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


def _build_train_setup(config: TransformerConfig, pg_collection: ProcessGroupCollection):
    """Build a reference block and an MFSDP v2 block that start from the same weights."""
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
            fsdp_all_gather_in_start_param_sync=False,
        ),
        module=model,
        pg_collection=pg_collection,
    )

    optimizer_config = OptimizerConfig(
        optimizer="adam",
        lr=1.0e-3,
        weight_decay=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=False,
        clip_grad=0.0,
    )
    reference_optimizer = get_megatron_optimizer(
        optimizer_config, [reference_model], use_gloo_process_groups=False
    )
    return reference_model, reference_optimizer, model, optimizer_config


def _any_rank_has_nonzero_grads(model: torch.nn.Module) -> bool:
    """Whether any rank sees a non-zero gradient on its MFSDP parameter shards."""
    local_present = any(
        parameter.grad is not None
        and getattr(parameter.grad, "_local_tensor", parameter.grad).count_nonzero().item() > 0
        for parameter in model.parameters()
    )
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, local_present)
    return any(gathered)


def _any_rank_has_updated_parameters(
    model: torch.nn.Module, initial_parameters: list[torch.Tensor]
) -> bool:
    """Whether any rank's MFSDP parameter shards moved away from their initial values."""
    local_updated = any(
        not torch.equal(getattr(parameter, "_local_tensor", parameter), initial)
        for parameter, initial in zip(model.parameters(), initial_parameters)
    )
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, local_updated)
    return any(gathered)


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
                use_distributed_optimizer=False,
                data_parallel_sharding_strategy="optim_grads_params",
                megatron_fsdp_main_params_dtype=torch.float32,
                megatron_fsdp_main_grads_dtype=torch.float32,
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
        config = _train_config()
        reference_model, reference_optimizer, model, reference_optimizer_config = (
            _build_train_setup(config, self.pg_collection)
        )

        with pytest.raises(
            ValueError, match="MFSDP v2 currently requires use_distributed_optimizer=False"
        ):
            get_megatron_optimizer(
                replace(reference_optimizer_config, use_distributed_optimizer=True),
                [model],
                use_gloo_process_groups=False,
            )
        optimizer = get_megatron_optimizer(
            replace(reference_optimizer_config), [model], use_gloo_process_groups=False
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
        torch.testing.assert_close(losses, reference_losses, rtol=1e-2, atol=0)

    @pytest.mark.skipif(
        not is_torch_min_version("2.4.0"),
        reason="Full-iteration CUDA graph capture requires PyTorch 2.4.0 or later.",
    )
    def test_build_train_and_step_with_full_cuda_graph(self):
        """Run test_build_train_and_step's iteration through a captured CUDA graph.

        ``FullCudaGraphWrapper`` warms up the wrapped forward/backward eagerly, captures
        it once, and replays the graph on every later call, which is how production
        training drives ``cuda_graph_impl="full_iteration"``. The optimizer steps
        eagerly between replays because MFSDP v2 rejects ``optimizer_cuda_graph``, so
        its Adam is built without ``capturable=True``.
        """
        config = _train_config()
        reference_model, reference_optimizer, model, reference_optimizer_config = (
            _build_train_setup(config, self.pg_collection)
        )
        optimizer = get_megatron_optimizer(
            replace(reference_optimizer_config), [model], use_gloo_process_groups=False
        )
        optimizer.reload_model_params()

        # FullCudaGraphWrapper registers every tracked RNG state on the graph, which
        # requires generator states rather than the default tracker's ByteTensors.
        model_parallel_cuda_manual_seed(1234, use_cudagraphable_rng=True, force_reset_rng=True)

        num_warmup_steps = 1
        num_microbatches = 2
        # Two replays follow the warmup iterations and the capture iteration.
        steps = [
            [
                torch.randn(8, 2, config.hidden_size, device="cuda", dtype=torch.bfloat16)
                for _ in range(num_microbatches)
            ]
            for _ in range(num_warmup_steps + 3)
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

        def forward_backward_func(*, model, data_iterator, num_microbatches, **unused):
            """Run one MFSDP v2 iteration's microbatch forward/backward passes."""
            module = model[0]
            iterator = data_iterator[0]
            microbatch_losses = []
            for _ in range(num_microbatches):
                batch = next(iterator)
                output = module(hidden_states=batch["hidden_states"], attention_mask=None)
                loss = output.float().square().mean()
                (loss / num_microbatches).backward()
                microbatch_losses.append(loss.detach())
            return torch.stack(microbatch_losses).mean()

        wrapper = FullCudaGraphWrapper(
            forward_backward_func, cuda_graph_warmup_steps=num_warmup_steps
        )
        # Wrapper capture state and static input buffers are class-level attributes
        # shared by every wrapper instance in the process. Start from a clean slate and
        # restore it below so other tests are unaffected.
        wrapper.reset_cuda_graph()
        StaticBufferLoader.static_buffers = {'training': [], 'validation': []}

        capture_stream = get_shared_capture_stream()
        initial_parameters = [
            getattr(parameter, "_local_tensor", parameter).detach().clone()
            for parameter in model.parameters()
        ]

        losses = []
        try:
            for iteration, microbatches in enumerate(steps):
                # set_to_none=False keeps every sharded parameter's .grad bound to the
                # persistent main_grad buffer that the captured backward writes into.
                # set_to_none=True would drop that binding on the host, leaving the
                # optimizer with no gradient to read after a replay.
                optimizer.zero_grad(set_to_none=False)
                # Warm up on the capture stream so autograd's gradient accumulators do
                # not retain a reference to a different, non-capturing stream. The
                # wrapper captures on this same stream.
                is_warmup = iteration < num_warmup_steps
                if is_warmup:
                    capture_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(capture_stream) if is_warmup else contextlib.nullcontext():
                    loss = wrapper(
                        model=[model],
                        data_iterator=iter([{"hidden_states": batch} for batch in microbatches]),
                        num_microbatches=num_microbatches,
                        seq_length=None,
                        forward_only=False,
                    )
                if is_warmup:
                    torch.cuda.current_stream().wait_stream(capture_stream)
                assert _any_rank_has_nonzero_grads(model), (
                    f"No rank has a non-zero gradient shard at iteration {iteration}; "
                    "the captured graph did not deliver gradients to the optimizer."
                )
                # Each replay rewrites the captured graph's fixed loss storage.
                losses.append(loss.clone())
                success, _, _ = optimizer.step()
                assert success

            assert (
                FullCudaGraphWrapper.cuda_graph["training"] is not None
            ), "FullCudaGraphWrapper did not capture a training CUDA graph."
            assert FullCudaGraphWrapper.curr_iteration["training"] == len(steps)

            losses = torch.stack(losses)
            reference_losses = torch.stack(reference_losses)
            assert torch.isfinite(losses).all()
            assert torch.isfinite(reference_losses).all()
            torch.testing.assert_close(losses, reference_losses, rtol=1e-2, atol=0)
            assert _any_rank_has_updated_parameters(model, initial_parameters), (
                "MFSDP v2 parameter shards never moved, so the optimizer did not apply "
                "the gradients produced inside the CUDA graph."
            )
        finally:
            wrapper.reset_cuda_graph()
            StaticBufferLoader.static_buffers = {'training': [], 'validation': []}
            # The RNG tracker is a module global, so restore the default one.
            model_parallel_cuda_manual_seed(1234, force_reset_rng=True)
