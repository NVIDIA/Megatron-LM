# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Dense MCore adapter and optimizer integration tests for MFSDP v2."""

from dataclasses import replace

import pytest
import torch
import transformer_engine.pytorch as te
from torch.distributed.tensor import DTensor
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

import megatron.core.distributed.fsdp.mcore_adapter_v2 as mcore_adapter_v2
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    FsdpContext,
    fully_shard_context,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import ShardLayout
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.fully_sharded_optimizer import FullyShardedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.distributed.mfsdp_v2.mcore_training_utils import (
    run_forward_backward_on_microbatches,
)
from tests.unit_tests.distributed.mfsdp_v2.mcore_utils import destroy_model_parallel
from tests.unit_tests.test_utilities import Utils


def _build_layer(config: TransformerConfig) -> TransformerLayer:
    return TransformerLayer(
        config=config,
        submodules=get_gpt_layer_local_spec().submodules,
        layer_number=1,
        add_layer_offset=False,
    )


@pytest.fixture
def process_group_collection(distributed_setup):
    """Set up dense model parallelism and clean up its process groups afterward."""
    Utils.initialize_model_parallel(1, 1)
    process_group_collection = ProcessGroupCollection.use_mpu_process_groups()
    model_parallel_cuda_manual_seed(1234)
    yield process_group_collection
    destroy_model_parallel()


@pytest.mark.launch_on_gb200
@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] < 10,
    reason="MXFP8 requires Blackwell-or-newer CUDA hardware.",
)
def test_mxfp8_parameters(process_group_collection):
    """The MCore adapter preserves MXFP8 parameters when FP8 gather is enabled."""
    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        ffn_hidden_size=256,
        bf16=True,
        params_dtype=torch.bfloat16,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        fp8="hybrid",
        fp8_recipe="mxfp8",
        fp8_param=True,
    )

    block = TransformerBlock(config=config, spec=get_gpt_layer_with_transformer_engine_spec()).to(
        device="cuda", dtype=config.params_dtype
    )
    model = FullyShardedDataParallel(
        config=config,
        ddp_config=DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            megatron_fsdp_version=2,
            use_distributed_optimizer=False,
            data_parallel_sharding_strategy="optim_grads_params",
            fp8_param_gather=True,
        ),
        module=block,
        pg_collection=process_group_collection,
    )
    # FSDP installs DTensor shards; check the parameters used for compute.
    parameters = []
    for module in model.module.modules():
        if not isinstance(module, FsdpModule):
            continue
        for group in module.parameter_groups:
            for parameter in group.fsdp_parameters:
                parameters.append(parameter.unsharded)
    assert any(isinstance(p, MXFP8Tensor) for p in parameters)


def test_init_model_with_meta_device_initializes_fsdp_v2_parameters(process_group_collection):
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
        pg_collection=process_group_collection,
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


def test_wraps_fsdp_unit_modules_before_root(process_group_collection):
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
        pg_collection=process_group_collection,
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


def test_nccl_ub_enables_symmetric_memory(process_group_collection, monkeypatch):
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
    original_fully_shard_context = mcore_adapter_v2.fully_shard_context

    def record_fully_shard_context(context):
        fully_shard_context_calls.append(context.use_symmetric_memory)
        return original_fully_shard_context(context)

    monkeypatch.setattr(mcore_adapter_v2, "fully_shard_context", record_fully_shard_context)
    FullyShardedDataParallel(
        config=config,
        ddp_config=DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            megatron_fsdp_version=2,
            data_parallel_sharding_strategy="optim_grads_params",
            nccl_ub=True,
        ),
        module=model,
        pg_collection=process_group_collection,
    )

    assert fully_shard_context_calls == [True]


def test_unpadded_shards_pack_parameter_groups_without_padding(process_group_collection):
    """megatron_fsdp_padded_shards=False reaches every parameter group and rejects nccl_ub."""
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        ffn_hidden_size=32,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    ddp_config = DistributedDataParallelConfig(
        use_megatron_fsdp=True,
        megatron_fsdp_version=2,
        data_parallel_sharding_strategy="optim_grads_params",
        megatron_fsdp_padded_shards=False,
    )
    model = FullyShardedDataParallel(
        config=config,
        ddp_config=ddp_config,
        module=torch.nn.Linear(config.hidden_size, config.hidden_size).to(
            device="cuda", dtype=config.params_dtype
        ),
        pg_collection=process_group_collection,
    )
    [group] = model.module.parameter_groups
    assert group.shard_layout is ShardLayout.UNPADDED
    # Weight and bias share one bf16 group and pack back to back without padding.
    assert group.main_weight.layout.size == sum(
        parameter.sharded.numel() for parameter in group.fsdp_parameters
    )

    with pytest.raises(ValueError, match="incompatible with nccl_ub"):
        FullyShardedDataParallel(
            config=config,
            ddp_config=replace(ddp_config, nccl_ub=True),
            module=torch.nn.Linear(config.hidden_size, config.hidden_size).to(
                device="cuda", dtype=config.params_dtype
            ),
            pg_collection=process_group_collection,
        )


def test_rejects_shared_context_without_caller_managed_grad_sync(
    process_group_collection, distributed_setup
):
    """The adapter rejects an incompatible shared context before sharding the model."""
    config = TransformerConfig(num_layers=1, hidden_size=16, num_attention_heads=4)
    layer = torch.nn.Linear(16, 16, device=distributed_setup.device)
    with fully_shard_context(FsdpContext(device=distributed_setup.device)) as context:
        with pytest.raises(ValueError, match="caller_managed_grad_sync=True"):
            FullyShardedDataParallel(
                config=config,
                ddp_config=DistributedDataParallelConfig(
                    use_megatron_fsdp=True,
                    megatron_fsdp_version=2,
                    use_distributed_optimizer=False,
                    data_parallel_sharding_strategy="optim_grads_params",
                ),
                module=layer,
                pg_collection=process_group_collection,
            )
        assert not context.caller_managed_grad_sync
        assert not isinstance(layer, FsdpModule)


def test_finish_grad_sync_waits_for_delayed_wgrad(process_group_collection, distributed_setup):
    """Delayed-wgrad reductions must rejoin the compute stream before capture ends."""
    config = TransformerConfig(
        num_layers=1, hidden_size=16, num_attention_heads=4, bf16=True, params_dtype=torch.bfloat16
    )
    layer = te.Linear(
        16,
        16,
        bias=False,
        params_dtype=config.params_dtype,
        device=distributed_setup.device,
        delay_wgrad_compute=True,
        fuse_wgrad_accumulation=False,
    )
    model = FullyShardedDataParallel(
        config=config,
        ddp_config=DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            megatron_fsdp_version=2,
            use_distributed_optimizer=False,
            data_parallel_sharding_strategy="optim_grads_params",
        ),
        module=layer,
        pg_collection=process_group_collection,
    )

    x = torch.randn(
        4, 16, device=distributed_setup.device, dtype=config.params_dtype, requires_grad=True
    )

    def forward():
        output = model(x)
        return torch.nn.functional.mse_loss(
            output.float(), torch.zeros_like(output, dtype=torch.float32)
        )

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    # Warm up on the capture stream so autograd's gradient accumulators do not
    # create cross-stream memory growth. See test_cuda_graph.py.
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            # Retain the grad views so reduction includes the accumulation kernel.
            model.zero_grad(set_to_none=False)
            run_forward_backward_on_microbatches(
                model, forward, delayed_wgrad_compute=model.module.backward_dw
            )

    # Ending capture rejects an unjoined reduction stream if finish_grad_sync
    # fails to wait for the reductions launched by backward_dw.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        model.zero_grad(set_to_none=False)
        run_forward_backward_on_microbatches(
            model, forward, delayed_wgrad_compute=model.module.backward_dw
        )
    torch.cuda.current_stream().wait_stream(capture_stream)

    for _ in range(5):
        graph.replay()


def test_delayed_wgrad_matches_immediate(process_group_collection, distributed_setup):
    """Delaying weight gradients preserves training losses with microbatch accumulation."""
    num_microbatches = 3
    config = TransformerConfig(
        num_layers=1, hidden_size=16, num_attention_heads=4, bf16=True, params_dtype=torch.bfloat16
    )

    def build_layer(delay_wgrad_compute):
        return te.Linear(
            16,
            16,
            bias=False,
            params_dtype=config.params_dtype,
            device=distributed_setup.device,
            delay_wgrad_compute=delay_wgrad_compute,
            fuse_wgrad_accumulation=False,
        )

    immediate_layer = build_layer(delay_wgrad_compute=False)
    delayed_layer = build_layer(delay_wgrad_compute=True)
    delayed_layer.load_state_dict(immediate_layer.state_dict())
    # Both runs see the same inputs, with distinct data on each rank and microbatch.
    batches = [
        [
            torch.randn(4, 16, device=distributed_setup.device, dtype=config.params_dtype)
            + distributed_setup.rank
            for _ in range(num_microbatches)
        ]
        for _ in range(5)
    ]

    def train(layer, delayed_wgrad_compute=None):
        model = FullyShardedDataParallel(
            config=config,
            ddp_config=DistributedDataParallelConfig(
                use_megatron_fsdp=True,
                megatron_fsdp_version=2,
                use_distributed_optimizer=False,
                data_parallel_sharding_strategy="optim_grads_params",
            ),
            module=layer,
            pg_collection=process_group_collection,
        )
        optimizer = get_megatron_optimizer(
            OptimizerConfig(
                optimizer="adam",
                lr=1.0e-3,
                weight_decay=0.0,
                bf16=True,
                params_dtype=config.params_dtype,
                use_distributed_optimizer=False,
                use_precision_aware_optimizer=True,
                clip_grad=0.0,
            ),
            [model],
        )
        optimizer.reload_model_params()
        losses = []
        for microbatches in batches:
            microbatch_iterator = iter(microbatches)

            def forward():
                output = model(next(microbatch_iterator).detach().requires_grad_())
                return torch.nn.functional.mse_loss(
                    output.float(), torch.zeros_like(output, dtype=torch.float32)
                )

            optimizer.zero_grad()
            microbatch_losses = run_forward_backward_on_microbatches(
                model,
                forward,
                num_microbatches=num_microbatches,
                loss_scale=1 / num_microbatches,
                delayed_wgrad_compute=delayed_wgrad_compute,
            )
            success, _, _ = optimizer.step()
            assert success
            losses.append(torch.stack(microbatch_losses).mean())
        return torch.stack(losses)

    immediate_losses = train(immediate_layer)
    delayed_losses = train(delayed_layer, delayed_layer.backward_dw)
    torch.testing.assert_close(delayed_losses, immediate_losses, rtol=0, atol=0)


def test_build_train_and_step(process_group_collection):
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
    reference_model = TransformerBlock(config=config, spec=get_gpt_layer_local_spec()).to(
        device="cuda", dtype=config.params_dtype
    )
    model = TransformerBlock(config=config, spec=get_gpt_layer_local_spec()).to(
        device="cuda", dtype=config.params_dtype
    )
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
        pg_collection=process_group_collection,
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
            batches = iter(microbatches)

            def forward():
                output = model(hidden_states=next(batches), attention_mask=None)
                return torch.nn.functional.mse_loss(
                    output.float(), torch.zeros_like(output, dtype=torch.float32)
                )

            optimizer.zero_grad(set_to_none=True)
            microbatch_losses = run_forward_backward_on_microbatches(
                model, forward, num_microbatches=len(microbatches), loss_scale=1 / len(microbatches)
            )
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


def test_fused_sgd_casts_mismatched_grads(process_group_collection):
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
        module=TransformerBlock(config=config, spec=get_gpt_layer_local_spec()).to(
            device="cuda", dtype=config.params_dtype
        ),
        pg_collection=process_group_collection,
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

    hidden = torch.randn(8, 2, config.hidden_size, device="cuda", dtype=torch.bfloat16)

    def forward():
        output = model(hidden_states=hidden, attention_mask=None)
        return torch.nn.functional.mse_loss(
            output.float(), torch.zeros_like(output, dtype=torch.float32)
        )

    optimizer.zero_grad(set_to_none=True)
    run_forward_backward_on_microbatches(model, forward)

    success, _, _ = optimizer.step()
    assert success


@pytest.mark.parametrize("use_precision_aware_optimizer", [False, True])
def test_gradient_clipping_reaches_global_norm(
    process_group_collection, use_precision_aware_optimizer
):
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
        module=TransformerBlock(config=config, spec=get_gpt_layer_local_spec()).to(
            device="cuda", dtype=config.params_dtype
        ),
        pg_collection=process_group_collection,
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
    hidden = torch.arange(1, config.hidden_size + 1, device="cuda", dtype=torch.bfloat16).view(
        1, 1, -1
    ).expand(8, 2, -1) * (torch.distributed.get_rank() + 1)

    def forward():
        output = model(hidden_states=hidden, attention_mask=None)
        return torch.nn.functional.mse_loss(
            output.float(), torch.zeros_like(output, dtype=torch.float32), reduction="sum"
        )

    run_forward_backward_on_microbatches(model, forward)

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
