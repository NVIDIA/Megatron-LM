# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MFSDP v2 + MLA activation offload integration tests.

Run under torchrun; use multiple GPUs to exercise distributed sharding.
This isolates attention offload; it does not exercise MoE/HybridEP or paged stash.
"""

import gc
from collections.abc import Iterator

import pytest
import torch
from torch.autograd import DeviceType
from torch.distributed.distributed_c10d import _world

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.optimizer import MegatronOptimizer, OptimizerConfig, get_megatron_optimizer
from megatron.core.pipeline_parallel.fine_grained_activation_offload import PipelineOffloadManager
from megatron.core.pipeline_parallel.schedules import forward_backward_no_pipelining
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.test_utilities import Utils


def _reset_offload():
    torch.cuda.synchronize()
    PipelineOffloadManager.reset_instance()
    gc.collect()


@pytest.fixture(scope="function")
def offload_pg_collection() -> Iterator[ProcessGroupCollection]:
    """Set up model parallelism and clean up offload state after each case."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    yield ProcessGroupCollection.use_mpu_process_groups()
    _reset_offload()
    Utils.destroy_model_parallel()
    # Utils clears MCore references but leaves some c10d groups alive (#6897).
    # Release their NCCL communicators while preserving the default group.
    for group in list(_world.pg_map):
        if group is not torch.distributed.group.WORLD:
            torch.distributed.destroy_process_group(group)


def _build_model(
    pg_collection: ProcessGroupCollection, *, offload: bool
) -> tuple[FullyShardedDataParallel, MegatronOptimizer]:
    """Build an identically initialized model and optimizer for each offload setting."""
    _reset_offload()
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234, te_rng_tracker=True, force_reset_rng=True)
    config = MLATransformerConfig(
        num_layers=6,
        hidden_size=128,
        num_attention_heads=4,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_head_dim=32,
        qk_pos_emb_head_dim=32,
        v_head_dim=32,
        bf16=True,
        params_dtype=torch.bfloat16,
        attention_backend=AttnBackend.unfused,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        gradient_accumulation_fusion=False,
        fine_grained_activation_offloading=offload,
        offload_modules=["core_attn", "attn_proj"] if offload else [],
        # Small model: lower the threshold so both groups actually offload.
        min_offloaded_tensor_size=1024,
        fine_grained_offloading_max_inflight_offloads=1,
    )
    model = HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        # Three MLA/MLP pairs; HybridModel counts each as a separate layer.
        hybrid_layer_pattern="+-+-+-",
        vocab_size=256,
        max_sequence_length=128,
        pg_collection=pg_collection,
    ).cuda()
    model = FullyShardedDataParallel(
        config=config,
        ddp_config=DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            megatron_fsdp_version=2,
            data_parallel_sharding_strategy="optim_grads_params",
            megatron_fsdp_main_grads_dtype=torch.bfloat16,
        ),
        module=model,
        pg_collection=pg_collection,
    )
    config.no_sync_func = model.no_sync
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            lr=1e-3,
            weight_decay=0.0,
            bf16=True,
            params_dtype=torch.bfloat16,
            clip_grad=1.0,
            use_precision_aware_optimizer=True,
            main_grads_dtype=torch.bfloat16,
        ),
        [model],
    )
    optimizer.reload_model_params()

    return model, optimizer


def _train(
    model: torch.nn.Module,
    optimizer: MegatronOptimizer,
    pg_collection: ProcessGroupCollection,
    *,
    num_steps: int = 6,
) -> torch.Tensor:
    """Run optimizer steps and return per-microbatch losses on the GPU."""
    seq_length, microbatches = 128, 2
    generator = torch.Generator(device="cuda").manual_seed(2026 + torch.distributed.get_rank())
    batches = [
        [
            {"tokens": torch.randint(0, 256, (1, seq_length), device="cuda", generator=generator)}
            for _ in range(microbatches)
        ]
        for _ in range(num_steps)
    ]
    positions = torch.arange(seq_length, device="cuda").unsqueeze(0)

    def forward_step(data_iterator, model):
        batch = next(data_iterator)
        output = model(batch["tokens"], positions, None)

        def loss_func(output):
            loss = output.float().square().mean()
            return loss, {"loss": loss.detach()}

        return output, loss_func

    losses = []
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        result = forward_backward_no_pipelining(
            forward_step_func=forward_step,
            data_iterator=[iter(batch)],
            model=[model],
            num_microbatches=microbatches,
            seq_length=seq_length,
            micro_batch_size=1,
            forward_only=False,
            pg_collection=pg_collection,
        )
        success, _, _ = optimizer.step()
        assert success
        losses.append(torch.stack([item["loss"] for item in result]).detach())
    return torch.stack(losses)


def test_mla_activation_offload_matches_baseline(
    offload_pg_collection: ProcessGroupCollection,
) -> None:
    """Compare training losses without profiling either run."""
    model, optimizer = _build_model(offload_pg_collection, offload=False)
    baseline_losses = _train(model, optimizer, offload_pg_collection)
    del model, optimizer
    model, optimizer = _build_model(offload_pg_collection, offload=True)
    offloaded_losses = _train(model, optimizer, offload_pg_collection)
    torch.testing.assert_close(offloaded_losses, baseline_losses, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("offload", [False, True], ids=["disabled", "enabled"])
def test_mla_activation_offload_transfers(
    offload_pg_collection: ProcessGroupCollection, offload: bool
) -> None:
    """Check that activation transfers occur only when offloading is enabled."""
    model, optimizer = _build_model(offload_pg_collection, offload=offload)
    # One warmup discovers offload groups; profile the next iteration.
    _train(model, optimizer, offload_pg_collection, num_steps=1)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        _train(model, optimizer, offload_pg_collection, num_steps=1)
    # Filter device copies first; their names identify the transfer direction.
    memcpy_events = [
        event
        for event in profiler.events()
        if event.device_type == DeviceType.CUDA and event.activity_type == "gpu_memcpy"
    ]
    transfer_counts = {
        direction: sum(direction in event.name for event in memcpy_events)
        for direction in ("DtoH", "HtoD")
    }
    if offload:
        manager = PipelineOffloadManager.get_instance()
        for name in ("core_attn", "attn_proj"):
            assert manager.offload_summary_bytes.get(name, 0) > 0, name
        assert all(count > 0 for count in transfer_counts.values()), transfer_counts
    else:
        assert all(count == 0 for count in transfer_counts.values()), transfer_counts
