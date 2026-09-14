# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Checkpoint restart phases driven by ``run_checkpoint_reshard.sh``."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import load, save
from megatron.core.models.engram.distributed_embedding import EPShardedMultiTableEmbedding
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.optimizer import FP32Optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils

TABLE_SIZES = (11, 13)
EMBEDDING_DIM = 3


def _initialize(ep_size: int):
    if not torch.distributed.is_initialized():
        Utils.initialize_distributed()
    assert torch.distributed.get_world_size() % ep_size == 0
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=ep_size,
    )
    return ProcessGroupCollection.use_mpu_process_groups(["ep", "tp", "expt_dp"])


def _build_embedding(pg_collection):
    config = SimpleNamespace(
        use_cpu_initialization=False,
        params_dtype=torch.float32,
        perform_initialization=False,
        sequence_parallel=False,
    )
    embedding = EPShardedMultiTableEmbedding(
        config=config,
        table_sizes=TABLE_SIZES,
        embedding_dim=EMBEDDING_DIM,
        init_method=lambda _: None,
        ep_group=pg_collection.ep,
        tp_group=pg_collection.tp,
        expt_dp_group=pg_collection.expt_dp,
    )
    with torch.no_grad():
        for table_id, table in enumerate(embedding.tables):
            rows = torch.arange(
                table.row_start, table.row_end, dtype=torch.float32, device="cuda"
            ).unsqueeze(1)
            columns = torch.arange(EMBEDDING_DIM, dtype=torch.float32, device="cuda").unsqueeze(0)
            table.weight.copy_(table_id * 100.0 + rows * 0.1 + columns * 0.01)
    return embedding


def _init_adam_state(optimizer, config=None):
    del config
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state[parameter]
            if not state:
                state["step"] = torch.tensor(0.0, device=parameter.device)
                state["exp_avg"] = torch.zeros_like(parameter)
                state["exp_avg_sq"] = torch.zeros_like(parameter)


def _build_optimizer(embedding):
    config = OptimizerConfig(optimizer="adam", lr=1.0e-3, min_lr=0.0)
    inner = torch.optim.Adam(
        [
            {
                "params": [table.weight for table in embedding.tables],
                "wd_mult": 1.0,
                "lr_mult": 1.0,
                "is_expert_parallel": True,
                "is_decoupled_lr": False,
            }
        ],
        lr=1.0e-3,
        betas=(0.9, 0.999),
    )
    return FP32Optimizer(inner, config, _init_adam_state)


def _set_global_row_grads(embedding, scale: float):
    for table_id, table in enumerate(embedding.tables):
        rows = torch.arange(
            table.row_start, table.row_end, dtype=torch.float32, device="cuda"
        ).unsqueeze(1)
        columns = torch.arange(EMBEDDING_DIM, dtype=torch.float32, device="cuda").unsqueeze(0)
        table.weight.grad = scale * (1.0 + table_id + rows + columns * 0.25)


def _model_sharded_state_dict(embedding):
    metadata = {"dp_cp_group": parallel_state.get_data_parallel_group(with_context_parallel=True)}
    return embedding.sharded_state_dict(prefix="", metadata=metadata)


def _build_full_reference():
    parameters = []
    for table_id, table_size in enumerate(TABLE_SIZES):
        rows = torch.arange(table_size, dtype=torch.float32, device="cuda").unsqueeze(1)
        columns = torch.arange(EMBEDDING_DIM, dtype=torch.float32, device="cuda").unsqueeze(0)
        parameters.append(torch.nn.Parameter(table_id * 100.0 + rows * 0.1 + columns * 0.01))
    optimizer = torch.optim.Adam(parameters, lr=1.0e-3, betas=(0.9, 0.999))
    return parameters, optimizer


def _step_full_reference(parameters, optimizer, scale):
    for table_id, parameter in enumerate(parameters):
        rows = torch.arange(parameter.shape[0], dtype=torch.float32, device="cuda").unsqueeze(1)
        columns = torch.arange(EMBEDDING_DIM, dtype=torch.float32, device="cuda").unsqueeze(0)
        parameter.grad = scale * (1.0 + table_id + rows + columns * 0.25)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _checkpoint_dir(source_ep):
    root = os.environ.get("ENGRAM_CHECKPOINT_DIR")
    if root is None:
        pytest.skip("run with tests/unit_tests/models/engram/run_checkpoint_reshard.sh")
    return Path(root) / f"ep{source_ep}"


def _require_phase(expected):
    if os.environ.get("ENGRAM_CHECKPOINT_PHASE") != expected:
        pytest.skip(f"checkpoint phase {expected} was not requested")


def _load_under_ep4_and_continue(checkpoint_dir):
    pg_collection = _initialize(4)
    full_parameters, full_optimizer = _build_full_reference()
    _step_full_reference(full_parameters, full_optimizer, scale=1.0)

    embedding = _build_embedding(pg_collection)
    optimizer = _build_optimizer(embedding)
    model_state = _model_sharded_state_dict(embedding)
    loaded = load(
        {
            "model": model_state,
            "optimizer": optimizer.sharded_state_dict(model_state, is_loading=True),
        },
        checkpoint_dir,
    )
    embedding.load_state_dict(loaded["model"])
    optimizer.load_state_dict(loaded["optimizer"])

    for table_id, table in enumerate(embedding.tables):
        torch.testing.assert_close(
            table.weight, full_parameters[table_id][table.row_start : table.row_end]
        )

    rank = torch.distributed.get_rank(pg_collection.ep)
    hash_ids = torch.tensor(
        [[[rank % TABLE_SIZES[0], (rank * 3 + 1) % TABLE_SIZES[1]]]],
        dtype=torch.int64,
        device="cuda",
    )
    output = embedding(hash_ids)
    expected = torch.stack(
        (full_parameters[0][hash_ids[..., 0]], full_parameters[1][hash_ids[..., 1]]), dim=-2
    )
    torch.testing.assert_close(output, expected)

    _set_global_row_grads(embedding, scale=0.5)
    optimizer.optimizer.step()
    _step_full_reference(full_parameters, full_optimizer, scale=0.5)
    for table_id, table in enumerate(embedding.tables):
        torch.testing.assert_close(
            table.weight,
            full_parameters[table_id][table.row_start : table.row_end],
            rtol=1e-6,
            atol=1e-7,
        )

    Utils.destroy_model_parallel()


@pytest.mark.parametrize("source_ep", [1, 2])
def test_save_model_and_adam_state(source_ep):
    _require_phase(f"save_ep{source_ep}")
    pg_collection = _initialize(source_ep)
    embedding = _build_embedding(pg_collection)
    optimizer = _build_optimizer(embedding)
    _set_global_row_grads(embedding, scale=1.0)
    optimizer.optimizer.step()
    optimizer.optimizer.zero_grad(set_to_none=True)

    checkpoint_dir = _checkpoint_dir(source_ep)
    if torch.distributed.get_rank() == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
    torch.distributed.barrier()
    model_state = _model_sharded_state_dict(embedding)
    save(
        {"model": model_state, "optimizer": optimizer.sharded_state_dict(model_state)},
        checkpoint_dir,
    )
    torch.distributed.barrier()
    assert (checkpoint_dir / ".metadata").is_file()
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("source_ep", [1, 2])
def test_load_under_ep4_and_continue_adam(source_ep):
    _require_phase(f"load_ep{source_ep}")
    _load_under_ep4_and_continue(_checkpoint_dir(source_ep))


@pytest.mark.parametrize("source_ep", [1, 2])
def test_in_process_reshard_to_ep4(source_ep):
    if os.environ.get("ENGRAM_CHECKPOINT_PHASE") is not None:
        pytest.skip("the cross-process checkpoint runner requested a specific phase")
    if int(os.environ.get("WORLD_SIZE", "1")) not in (4, 8):
        pytest.skip("requires torchrun with four or eight ranks")

    source_pg = _initialize(source_ep)
    source = _build_embedding(source_pg)
    source_optimizer = _build_optimizer(source)
    _set_global_row_grads(source, scale=1.0)
    source_optimizer.optimizer.step()
    source_optimizer.optimizer.zero_grad(set_to_none=True)

    checkpoint_root = Path.cwd() / ".pytest_cache" / "engram_dist_checkpoint"
    checkpoint_name = f"engram_ep{source_ep}_to_ep4"
    with TempNamedDir(checkpoint_root / checkpoint_name, sync=True) as checkpoint_dir:
        source_model_state = _model_sharded_state_dict(source)
        save(
            {
                "model": source_model_state,
                "optimizer": source_optimizer.sharded_state_dict(source_model_state),
            },
            checkpoint_dir,
        )
        torch.distributed.barrier()
        assert (checkpoint_dir / ".metadata").is_file()
        del source_optimizer, source
        Utils.destroy_model_parallel()
        _load_under_ep4_and_continue(checkpoint_dir)
