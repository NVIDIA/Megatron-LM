# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""``fsdp_dtensor`` checkpoint round-trip tests for the MFSDP v2 MCore optimizer."""

from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.tensor import DTensor

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    preprocess_state_dict_for_uneven_dtensor,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.fully_sharded_optimizer import FullyShardedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.fsdp_dtensor_checkpoint import get_global_unique_param_name
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils

SOURCE_STEPS = 3
DESTINATION_STEPS = 1


def _build_model_and_optimizer(
    config: TransformerConfig, pg_collection: ProcessGroupCollection, *, zero_init: bool
) -> tuple[FullyShardedDataParallel, FullyShardedOptimizer]:
    """Build an MFSDP v2 sharded transformer block and its :class:`FullyShardedOptimizer`."""
    block = TransformerBlock(config=config, spec=get_gpt_layer_local_spec()).to(
        device="cuda", dtype=config.params_dtype
    )
    if zero_init:
        # Zero the destination weights so they are obviously different from the saved
        # (trained) source; a correct load must overwrite them.
        with torch.no_grad():
            for parameter in block.parameters():
                parameter.zero_()
    model = FullyShardedDataParallel(
        config=config,
        ddp_config=DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            megatron_fsdp_version=2,
            use_distributed_optimizer=True,
            data_parallel_sharding_strategy="optim_grads_params",
            megatron_fsdp_main_params_dtype=torch.float32,
            megatron_fsdp_main_grads_dtype=torch.bfloat16,
            fsdp_all_gather_in_start_param_sync=False,
        ),
        module=block,
        pg_collection=pg_collection,
    )
    optimizer_config = OptimizerConfig(
        optimizer="adam",
        lr=1.0e-2,
        # A non-zero weight decay splits the parameters into decay and no-decay groups, so
        # the param_to_group_meta round-trip is exercised with more than one group.
        weight_decay=0.1,
        bf16=True,
        params_dtype=torch.bfloat16,
        # Megatron-FSDP shards the optimizer state itself, so MFSDP v2 rejects MCore's
        # distributed optimizer on the OptimizerConfig.
        use_distributed_optimizer=False,
        clip_grad=0.0,
    )
    optimizer = get_megatron_optimizer(optimizer_config, [model], use_gloo_process_groups=False)
    assert isinstance(optimizer, FullyShardedOptimizer)
    optimizer.reload_model_params()
    return model, optimizer


def _train_step(
    config: TransformerConfig, model: FullyShardedDataParallel, optimizer: FullyShardedOptimizer
) -> None:
    """Run one optimizer step so the weights and the optimizer state become non-trivial."""
    model.zero_grad_buffer()
    optimizer.zero_grad(set_to_none=True)
    batch = torch.randn(8, 2, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    model(hidden_states=batch, attention_mask=None).square().mean().backward()
    success, _, _ = optimizer.step()
    assert success


def _local_numel(value: torch.Tensor) -> int:
    """Return the number of elements this rank physically holds."""
    return value.to_local().numel() if isinstance(value, DTensor) else value.numel()


def _assert_tensors_identical(expected: torch.Tensor, actual: torch.Tensor, what: str) -> None:
    """Assert two tensors are bit-identical, checking DTensor global metadata when applicable.

    A checkpoint round trip must reproduce the values exactly, so tolerances are zero. For
    DTensors it must also reproduce the *global* view: an entry whose global shape or placement
    changed would be silently wrong even if this rank's local shard happens to match.
    """
    assert type(expected) is type(actual), f"{what}: {type(expected)} became {type(actual)}"
    if isinstance(expected, DTensor):
        assert (
            expected.shape == actual.shape
        ), f"{what}: global shape {expected.shape} != {actual.shape}"
        assert expected.placements == actual.placements, f"{what}: placements changed"
        assert expected.device_mesh == actual.device_mesh, f"{what}: device mesh changed"
        expected, actual = expected.to_local(), actual.to_local()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=f"{what}: value mismatch")


def _snapshot_model_state(model: FullyShardedDataParallel) -> dict[str, torch.Tensor]:
    """Clone the model weights, keyed as the checkpoint state dict is.

    DTensor entries are cloned as DTensors so the comparison can check the global shape and
    placements, not just this rank's local shard.
    """
    return {
        key: value.clone()
        for key, value in model.state_dict_for_save_checkpoint().items()
        if torch.is_tensor(value)
    }


def _snapshot_optimizer_state(optimizer: FullyShardedOptimizer) -> dict[str, dict[str, Any]]:
    """Clone this rank's optimizer state, keyed by globally unique parameter name."""
    snapshot: dict[str, dict[str, Any]] = {}
    for param, state in optimizer.optimizer.state.items():
        fqn = get_global_unique_param_name(optimizer.model_chunks, param)
        snapshot[fqn] = {
            key: (value.clone() if torch.is_tensor(value) else value)
            for key, value in state.items()
        }
    return snapshot


def _snapshot_param_group_hyperparameters(optimizer: FullyShardedOptimizer) -> list[dict[str, Any]]:
    """Capture every param group's hyperparameters, in group order.

    The base optimizer (TE FusedAdam) tracks ``step`` per group rather than per parameter,
    so restoring it exercises the ``param_to_group_meta`` round-trip specifically.
    """
    return [
        {key: value for key, value in group.items() if key != "params"}
        for group in optimizer.optimizer.param_groups
    ]


def _assert_model_matches_snapshot(
    model: FullyShardedDataParallel, model_snapshot: dict[str, torch.Tensor]
) -> bool:
    """Assert the model's local weights equal the snapshot.

    Returns:
        bool: whether this rank held at least one non-empty local shard. The caller
        all-gathers this flag and asserts some rank made a real (non-empty) comparison, so
        the test cannot pass vacuously when a rank happens to own only empty shards.
    """
    current = _snapshot_model_state(model)
    assert model_snapshot.keys() == current.keys()
    local_nonempty = False
    for key, expected in model_snapshot.items():
        _assert_tensors_identical(expected, current[key], f"model[{key}]")
        local_nonempty = local_nonempty or _local_numel(expected) > 0
    return local_nonempty


def _assert_optimizer_matches_snapshot(
    optimizer: FullyShardedOptimizer,
    state_snapshot: dict[str, dict[str, Any]],
    param_group_snapshot: list[dict[str, Any]],
) -> None:
    """Assert the optimizer's local state and param-group hyperparameters equal the snapshot."""
    current_state = _snapshot_optimizer_state(optimizer)
    assert state_snapshot.keys() == current_state.keys()
    for fqn, expected_state in state_snapshot.items():
        assert expected_state.keys() == current_state[fqn].keys(), fqn
        for key, expected in expected_state.items():
            actual = current_state[fqn][key]
            if torch.is_tensor(expected):
                _assert_tensors_identical(expected, actual, f"optim[{fqn}][{key}]")
            else:
                assert expected == actual, f"optim[{fqn}][{key}] scalar mismatch"

    assert param_group_snapshot == _snapshot_param_group_hyperparameters(optimizer)


class TestOptimizerCheckpoint:
    """Round-trip an MFSDP v2 transformer block and its optimizer through a DCP checkpoint."""

    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        model_parallel_cuda_manual_seed(1234)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def test_checkpoint_roundtrip(self, tmp_path_dist_ckpt: Path) -> None:
        """Model weights and optimizer state survive an ``fsdp_dtensor`` round trip.

        The source trains a few steps and is checkpointed the way the training loop does it
        (``state_dict_for_save_checkpoint`` for the model, :meth:`FullyShardedOptimizer.
        sharded_state_dict` for the optimizer, then the uneven-DTensor preprocessing). The
        destination starts from zeroed weights and trains a *different* number of steps, so
        a dropped or mis-serialized ``step`` (and the stale momentum that comes with it)
        fails the comparison instead of matching by construction.

        With >=2 ranks this also covers the empty-local placeholder path: MFSDP v2 filters
        empty-local shards out of the base optimizer per rank, so the placeholders are what
        keep the DTensor keyspace identical across ranks.
        """
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
        world_size = torch.distributed.get_world_size()

        source_model, source_optimizer = _build_model_and_optimizer(
            config, self.pg_collection, zero_init=False
        )
        for _ in range(SOURCE_STEPS):
            _train_step(config, source_model, source_optimizer)

        model_snapshot = _snapshot_model_state(source_model)
        state_snapshot = _snapshot_optimizer_state(source_optimizer)
        param_group_snapshot = _snapshot_param_group_hyperparameters(source_optimizer)
        assert {group["step"] for group in param_group_snapshot} == {SOURCE_STEPS}

        if world_size > 1:
            # Confirm the placeholder path is not vacuous: at least one rank must have had an
            # empty-local shard filtered out of its optimizer. Count empty local shards
            # directly, which is the exact condition the filter tests.
            empty_local = sum(
                1
                for param in source_model.parameters()
                if param.requires_grad
                and isinstance(param, DTensor)
                and param.to_local().numel() == 0
            )
            empty_local_counts = [None] * world_size
            torch.distributed.all_gather_object(empty_local_counts, empty_local)
            assert any(empty_local_counts), (
                "No rank had an empty local shard, so the placeholder path is untested for "
                f"this config. empty_local_counts={empty_local_counts}"
            )

        with TempNamedDir(tmp_path_dist_ckpt / "fsdp_dtensor_optimizer", sync=True) as ckpt_dir:
            save_state_dict = {
                "model": source_model.state_dict_for_save_checkpoint(),
                "optimizer": source_optimizer.sharded_state_dict({}),
            }
            preprocess_state_dict_for_uneven_dtensor(save_state_dict)
            dcp.save(save_state_dict, checkpoint_id=ckpt_dir)

            destination_model, destination_optimizer = _build_model_and_optimizer(
                config, self.pg_collection, zero_init=True
            )
            for _ in range(DESTINATION_STEPS):
                _train_step(config, destination_model, destination_optimizer)

            load_state_dict = {
                "model": destination_model.state_dict_for_save_checkpoint(),
                "optimizer": destination_optimizer.sharded_state_dict({}, is_loading=True),
            }
            preprocess_state_dict_for_uneven_dtensor(load_state_dict)
            # DCP writes into the exposed DTensors in place, so the model weights and the
            # optimizer's momentum buffers are restored by this call alone; load_state_dict
            # then reinstalls the param-group hyperparameters.
            dcp.load(load_state_dict, checkpoint_id=ckpt_dir)
            destination_optimizer.load_state_dict(load_state_dict["optimizer"])

        local_nonempty = _assert_model_matches_snapshot(destination_model, model_snapshot)
        _assert_optimizer_matches_snapshot(
            destination_optimizer, state_snapshot, param_group_snapshot
        )

        # At least one rank must have held non-empty local shards for the check to be meaningful.
        nonempty_flags = [None] * world_size
        torch.distributed.all_gather_object(nonempty_flags, local_nonempty)
        assert any(nonempty_flags), "All ranks had empty local shards."
