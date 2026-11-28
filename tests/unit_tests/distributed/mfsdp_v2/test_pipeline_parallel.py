# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pipeline-parallel integration tests for MFSDP v2.

``virtual_pipeline_model_parallel_size > 1`` manifests as multiple model chunks
on a rank, and ``wrap_model_chunks_with_ddp`` wraps each chunk in its own
``FullyShardedDataParallelV2``. For MFSDP v2 the helper opens one ambient
``fully_shard_context`` around the whole per-chunk loop; each chunk's adapter
joins it through ``current_fully_shard_context()`` instead of opening a second
context, and the helper finalizes the shared context exactly once after the
loop. This test drives the helper directly and asserts that all wrapped chunks
share the same ``FsdpContext``.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.core import Timers
from megatron.core.distributed import DistributedDataParallelConfig, FullyShardedDataParallel
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallelV2
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.fully_shard import (
    current_fully_shard_context,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.energy_monitor import EnergyMonitor
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.schedules import (
    forward_backward_pipelining_without_interleaving,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.training import global_vars, training
from megatron.training.state import TrainState
from megatron.training.training import wrap_model_chunks_with_ddp
from tests.unit_tests.distributed.mfsdp_v2.mcore_utils import destroy_model_parallel
from tests.unit_tests.test_utilities import Utils


class _VirtualPipelineChunk(nn.Module):
    """One virtual-pipeline stage: a tiny dense block with its own parameters."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _mfsdp_v2_ddp_config() -> DistributedDataParallelConfig:
    """A DistributedDataParallelConfig accepted by MFSDP v2 validation."""
    return DistributedDataParallelConfig(
        use_megatron_fsdp=True,
        megatron_fsdp_version=2,
        data_parallel_sharding_strategy="optim_grads_params",
    )


def _compute_config() -> TransformerConfig:
    """A minimal TransformerConfig accepted by MFSDP v2 validation."""
    return TransformerConfig(num_layers=1, hidden_size=16, num_attention_heads=1)


@pytest.fixture
def process_group_collection(distributed_setup, request):
    """Reset model-parallel state before distributed_setup tears down the groups."""
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=getattr(request, "param", 1)
    )
    yield ProcessGroupCollection.use_mpu_process_groups()
    destroy_model_parallel()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 CUDA devices.")
def test_shared_context_across_virtual_pipeline_chunks(distributed_setup, process_group_collection):
    """Multiple virtual-pipeline chunks wrapped in one call share one FsdpContext."""
    device = distributed_setup.device

    # virtual_pipeline_model_parallel_size > 1 yields multiple chunks on this
    # rank; model two virtual-pipeline sub-stages here. The ambient
    # fully_shard_context is opened for every MFSDP v2 wrap, regardless of the
    # chunk count.
    chunks = [_VirtualPipelineChunk().to(device=device, dtype=torch.bfloat16) for _ in range(2)]

    wrapped_chunks = wrap_model_chunks_with_ddp(
        chunks,
        _compute_config(),
        _mfsdp_v2_ddp_config(),
        DP=FullyShardedDataParallel,
        pg_collection=process_group_collection,
        disable_bucketing_per_chunk=[False] * len(chunks),
    )

    # Every chunk is an MFSDP v2 wrapper whose root is an FsdpModule.
    assert len(wrapped_chunks) == len(chunks)
    for wrapped_chunk in wrapped_chunks:
        assert isinstance(wrapped_chunk, FullyShardedDataParallelV2)
        assert isinstance(wrapped_chunk.module, FsdpModule)

    # The whole point of the feature: all chunks share ONE context, so each chunk
    # must report the same context object as the first one.
    first_context = wrapped_chunks[0].module.context
    for wrapped_chunk in wrapped_chunks:
        assert wrapped_chunk.module.context is first_context

    # The shared context was finalized exactly once after the loop, so it is
    # ready to drive forward/backward without an extra finalize.
    first_context.ensure_finalized()

    # The helper left no ambient context behind: the chunks joined its scope and
    # that scope, which owned the only finalize call, has since exited.
    assert current_fully_shard_context() is None


@pytest.mark.parametrize("process_group_collection", [2], indirect=True)
@pytest.mark.skipif(Utils.world_size < 2, reason="Requires two pipeline stages.")
def test_training_reduces_pipeline_gradients(
    distributed_setup, process_group_collection, run_config, monkeypatch
):
    """Training's MFSDP v2 hook registration must allow the final backward to reduce."""
    pg = process_group_collection
    config = TransformerConfig(
        num_layers=2,
        hidden_size=1,
        num_attention_heads=1,
        pipeline_model_parallel_size=2,
        pipeline_dtype=torch.float32,
    )

    class Stage(_VirtualPipelineChunk):
        def __init__(self):
            super().__init__(dim=1)
            self.config = config
            self.pg_collection = pg
            self.pre_process = False
            self.share_embeddings_and_output_weights = False
            nn.init.ones_(self.fc.weight)

        def set_input_tensor(self, tensors):
            self.input_tensor = tensors[0]

        def forward(self, inputs):
            return super().forward(self.input_tensor if inputs is None else inputs)

    model = FullyShardedDataParallelV2(
        config,
        DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            megatron_fsdp_version=2,
            data_parallel_sharding_strategy="no_shard",
        ),
        Stage().to(distributed_setup.device),
        pg_collection=pg,
    )
    run_config.dist.align_grad_reduce = True
    for name, value in {
        "_GLOBAL_ARGS": SimpleNamespace(),
        "_GLOBAL_TRAIN_STATE": TrainState(),
        "_GLOBAL_TIMERS": Timers(0, "minmax"),
        "_GLOBAL_ENERGY_MONITOR": EnergyMonitor(),
    }.items():
        monkeypatch.setattr(global_vars, name, value)

    def forward_step(_iterator, wrapped):
        inputs = torch.full((1, 1, 1), float(pg.dp.rank() + 1), device=distributed_setup.device)
        return wrapped(inputs if pg.pp.rank() == 0 else None), lambda output: (output.sum(), {})

    class BackwardComplete(Exception):
        pass

    def run_schedule(_message):
        # Run after train() installs its hooks, before unrelated training-loop setup.
        forward_backward_pipelining_without_interleaving(
            forward_step_func=forward_step,
            data_iterator=None,
            model=[model],
            num_microbatches=1,
            seq_length=1,
            micro_batch_size=1,
            p2p_communicator=P2PCommunicator(pp_group=pg.pp, config=config),
            pg_collection=pg,
        )
        raise BackwardComplete()

    monkeypatch.setattr(training, "print_datetime", run_schedule)
    with pytest.raises(BackwardComplete):
        training.train(
            forward_step_func=forward_step,
            model=[model],
            optimizer=None,
            opt_param_scheduler=None,
            train_data_iterator=None,
            config=config,
            checkpointing_context={},
            next_train_iterator=lambda _iteration, iterator: iterator,
            run_eval=None,
        )
    gradient = model.module.fc.weight.grad
    assert not any(placement.is_partial() for placement in gradient.placements)
    # Each unit-weight stage has dloss/dweight = input; DP must average the inputs.
    torch.testing.assert_close(
        gradient.to_local(), torch.full_like(gradient.to_local(), (pg.dp.size() + 1) / 2)
    )
