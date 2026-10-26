# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Full-iteration and optimizer CUDA graph integration tests for the MFSDP v2 adapter."""

from dataclasses import replace

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.full_cuda_graph import FullCudaGraphWrapper, StaticBufferLoader
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.fully_sharded_optimizer import FullyShardedOptimizer
from megatron.core.optimizer.optimizer_cuda_graph import OptimizerCudaGraphWrapper
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.distributed.mfsdp_v2.mcore_training_utils import (
    run_forward_backward_on_microbatches,
)
from tests.unit_tests.distributed.mfsdp_v2.mcore_utils import destroy_model_parallel
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def process_group_collection(distributed_setup, outer_dp_size):
    """Yield process groups with TE RNG tracking, then reset graph state and groups."""
    Utils.initialize_model_parallel(1, 1, num_distributed_optimizer_instances=outer_dp_size)
    process_group_collection = ProcessGroupCollection.use_mpu_process_groups()
    model_parallel_cuda_manual_seed(1234, te_rng_tracker=True, force_reset_rng=True)
    yield process_group_collection
    # Release captured graphs before distributed_setup destroys process groups.
    # The wrappers store capture state globally. Reset it so the next test captures its
    # own work instead of replaying this test's graph.
    OptimizerCudaGraphWrapper.curr_iteration = 0
    OptimizerCudaGraphWrapper.cuda_graph = None
    OptimizerCudaGraphWrapper.result = None
    FullCudaGraphWrapper.curr_iteration = {'training': 0, 'validation': 0}
    FullCudaGraphWrapper.cuda_graph = {'training': None, 'validation': None}
    FullCudaGraphWrapper.result = {'training': None, 'validation': None}
    StaticBufferLoader.static_buffers = {'training': [], 'validation': []}
    destroy_model_parallel()


@pytest.mark.parametrize(
    "outer_dp_size,inner_strategy",
    [(1, "optim_grads_params"), (2, "optim_grads")],
    ids=["zero3", "hybrid_zero2"],
)
def test_full_iteration_and_optimizer_cuda_graph_match_eager(
    process_group_collection, outer_dp_size, inner_strategy
):
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
        model = TransformerBlock(config=config, spec=get_gpt_layer_local_spec()).to(
            device="cuda", dtype=config.params_dtype
        )
        model = FullyShardedDataParallel(
            config=config,
            ddp_config=DistributedDataParallelConfig(
                use_megatron_fsdp=True,
                megatron_fsdp_version=2,
                use_distributed_optimizer=False,
                data_parallel_sharding_strategy=inner_strategy,
                num_distributed_optimizer_instances=outer_dp_size,
                outer_dp_sharding_strategy="optim" if outer_dp_size > 1 else "no_shard",
                # With the default None, MFSDP v2 uses BF16 main grads with FP32 main
                # params. Capturable FusedAdam in the TE revision under test requires
                # matching dtypes (https://github.com/NVIDIA/TransformerEngine/issues/3358).
                megatron_fsdp_main_grads_dtype=torch.float32,
                megatron_fsdp_cuda_graph_mode=enable_cuda_graph,
            ),
            module=model,
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
            optimizer.step = OptimizerCudaGraphWrapper(optimizer.step, cuda_graph_warmup_steps=1)
        return model, optimizer

    # FullCudaGraphWrapper requires this keyword-only schedule callback signature.
    def schedule(*, model, data_iterator, num_microbatches, seq_length, forward_only):
        assert seq_length is None
        assert not forward_only

        def forward():
            hidden = next(data_iterator[0])["hidden_states"]
            output = model[0](hidden, attention_mask=None)
            return torch.nn.functional.mse_loss(
                output.float(), torch.zeros_like(output, dtype=torch.float32)
            )

        # Pipeline schedules receive model chunks as a list, including with PP=1.
        losses = run_forward_backward_on_microbatches(
            model[0], forward, num_microbatches=num_microbatches, loss_scale=1 / num_microbatches
        )
        return [{"loss": loss} for loss in losses]

    steps = [
        [
            torch.randn(8, 2, eager_config.hidden_size, device="cuda", dtype=torch.bfloat16)
            for _ in range(2)
        ]
        for _ in range(10)
    ]

    def run(model, optimizer, schedule) -> torch.Tensor:
        losses = []
        for microbatches in steps:
            optimizer.zero_grad(set_to_none=True)
            microbatch_losses = schedule(
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

    eager_model, eager_optimizer = build_model_and_optimizer(eager_config, enable_cuda_graph=False)
    graph_model, graph_optimizer = build_model_and_optimizer(graph_config, enable_cuda_graph=True)
    graph_model.load_state_dict(eager_model.state_dict())

    eager_losses = run(eager_model, eager_optimizer, schedule)
    cuda_graph_forward_backward = FullCudaGraphWrapper(schedule, cuda_graph_warmup_steps=1)
    with torch.profiler.profile() as prof:
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

    # The optimizer holds the wrapper as its step attribute. The wrapper saves
    # the original step method, which keeps a reference to the optimizer it
    # operates on. Remove the wrapper to break this cycle so the optimizer's
    # GPU memory can be released when the test returns.
    del graph_optimizer.step
