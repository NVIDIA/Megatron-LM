# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Sparse gradient and native checkpoint invariants across parallel layouts."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import load, save
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.optimizer import ChainedOptimizer
from megatron.core.optimizer.sparse_adam import RowSparseAdamOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.engram.memory import RowShardedMultiHeadEmbedding
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(Utils.world_size not in (2, 4, 8), reason="needs 2, 4, or 8 GPUs")


def _table(layer_id: int) -> RowShardedMultiHeadEmbedding:
    return RowShardedMultiHeadEmbedding([11, 13], 4, layer_id=layer_id).cuda()


def _optimizer(tables: list[RowShardedMultiHeadEmbedding]) -> RowSparseAdamOptimizer:
    groups = [{"params": [table.embedding.weight for table in tables]}]
    return RowSparseAdamOptimizer(
        groups,
        OptimizerConfig(lr=0.02, min_lr=0.002, clip_grad=0.0),
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(["tp", "mp", "tp_dp_cp"]),
    )


@pytest.mark.parametrize("per_token", [False, True])
def test_context_parallel_sparse_gradient_matches_full_sequence(per_token):
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=Utils.world_size // 2, context_parallel_size=2
    )
    try:
        table = RowShardedMultiHeadEmbedding(
            [11, 13], 4, layer_id=1, calculate_per_token_loss=per_token
        ).cuda()
        ids = torch.arange(8, device="cuda").view(1, 8, 1).expand(-1, -1, 2)
        output = table(ids)
        cp_rank = parallel_state.get_context_parallel_rank()
        local_loss = output[:, cp_rank * 4 : (cp_rank + 1) * 4].sum()
        # Megatron's ordinary loss path compensates CP before DDP averaging;
        # per-token loss instead scales the final gradients by global token count.
        (local_loss if per_token else local_loss * 2 / 8).backward()
        gradient = table.embedding.weight.grad.coalesce()
        if per_token:
            gradient.values().div_(8)
            table.embedding.weight.grad = gradient
        reference = torch.zeros(24, 4, device="cuda")
        reference[:8] = 1 / 8
        reference[11:19] = 1 / 8
        expected = torch.cat(
            [
                reference[offset + start : offset + end]
                for offset, (start, end) in zip(
                    table.layout.global_offsets, table.layout.local_bounds
                )
            ]
        )
        torch.testing.assert_close(gradient.to_dense(), expected)
        optimizer = _optimizer([table])
        assert optimizer.get_grad_norm() == pytest.approx(1.0)
        optimizer.clip_grad_by_total_norm(0.5, 1.0)
        torch.testing.assert_close(
            table.embedding.weight.grad.to_dense(), expected * (0.5 / (1 + 1e-6))
        )
    finally:
        Utils.destroy_model_parallel()


def _state(tables, optimizer, *, is_loading=False):
    result = {"optimizer": optimizer.sharded_state_dict({}, is_loading=is_loading)}
    result["model"] = {}
    for table in tables:
        result["model"].update(table.sharded_state_dict(prefix=f"memory_{table.layer_id}."))
    return result


def test_sparse_checkpoint_moves_layer_between_pipeline_stages(tmp_path_dist_ckpt):
    """Preserve moments, row steps and hyperparameters when PP owners change."""
    first_dir = tmp_path_dist_ckpt / "pipeline_source"
    second_dir = tmp_path_dist_ckpt / "tensor_source"
    if torch.distributed.get_rank() == 0:
        first_dir.mkdir()
        second_dir.mkdir()
    torch.distributed.barrier()
    try:
        Utils.initialize_model_parallel(pipeline_model_parallel_size=Utils.world_size)
        layer_id = torch.distributed.get_rank() + 10
        source = _table(layer_id)
        with torch.no_grad():
            source.embedding.weight.fill_(layer_id)
        optimizer = _optimizer([source])
        rows = torch.arange(24, device="cuda")
        source.embedding.weight.grad = torch.sparse_coo_tensor(
            rows.unsqueeze(0), torch.full((24, 4), 0.25, device="cuda"), (24, 4)
        )
        optimizer.step()
        optimizer.param_groups[0]["lr"] = 0.007
        save(_state([source], optimizer), first_dir)
        Utils.destroy_model_parallel()

        Utils.initialize_model_parallel(tensor_model_parallel_size=Utils.world_size)
        tables = [_table(layer) for layer in range(10, 10 + Utils.world_size)]
        restored = _optimizer(tables)
        loaded = load(_state(tables, restored, is_loading=True), first_dir)
        for table in tables:
            prefix = f"memory_{table.layer_id}."
            table.load_state_dict(
                {
                    key[len(prefix) :]: value
                    for key, value in loaded["model"].items()
                    if key.startswith(prefix)
                }
            )
        restored.load_state_dict(loaded["optimizer"])
        assert restored.param_groups[0]["lr"] == 0.007
        for table in tables:
            state = restored.optimizer.compact_state(table.embedding.weight)
            torch.testing.assert_close(
                table.embedding.weight,
                torch.full_like(table.embedding.weight, table.layer_id - 0.02),
            )
            torch.testing.assert_close(state["exp_avg"], torch.full_like(state["exp_avg"], 0.025))
            assert state["step"].eq(1).all()
        save(_state(tables, restored), second_dir)
        Utils.destroy_model_parallel()

        Utils.initialize_model_parallel(pipeline_model_parallel_size=Utils.world_size)
        moved = _table(10 + Utils.world_size - 1 - torch.distributed.get_rank())
        resumed = _optimizer([moved])
        loaded = load(_state([moved], resumed, is_loading=True), second_dir)
        prefix = f"memory_{moved.layer_id}."
        moved.load_state_dict({key[len(prefix) :]: value for key, value in loaded["model"].items()})
        resumed.load_state_dict(loaded["optimizer"])
        state = resumed.optimizer.compact_state(moved.embedding.weight)
        assert state["rows"].numel() == 24
        assert state["step"].eq(1).all()
        assert resumed.param_groups[0]["lr"] == 0.007
    finally:
        Utils.destroy_model_parallel()


def test_table_free_pipeline_stages_participate_in_sparse_norm():
    Utils.initialize_model_parallel(pipeline_model_parallel_size=Utils.world_size)
    try:
        tables = [_table(1)] if parallel_state.get_pipeline_model_parallel_rank() == 1 else []
        optimizer = _optimizer(tables)
        if tables:
            parameter = tables[0].embedding.weight
            parameter.grad = torch.sparse_coo_tensor(
                torch.tensor([[0]], device="cuda"), torch.ones(1, 4, device="cuda"), parameter.shape
            )
        assert optimizer.get_grad_norm() == pytest.approx(2.0)
    finally:
        Utils.destroy_model_parallel()


def test_empty_sparse_pipeline_stage_preserves_native_optimizer_chain(tmp_path_dist_ckpt):
    """Native extraction must retain empty components without saving rank-local placeholders."""
    Utils.initialize_model_parallel(pipeline_model_parallel_size=Utils.world_size)
    directory = tmp_path_dist_ckpt / "empty_sparse_stage"
    if torch.distributed.get_rank() == 0:
        directory.mkdir()
    torch.distributed.barrier()
    try:
        owns_table = parallel_state.get_pipeline_model_parallel_rank() == 0
        tables = [_table(1)] if owns_table else []
        first = _optimizer(tables)
        chain = ChainedOptimizer([first, _optimizer([])])
        if tables:
            param = tables[0].embedding.weight
            param.grad = torch.sparse_coo_tensor(
                torch.tensor([[0]], device="cuda"), torch.ones(1, 4, device="cuda"), param.shape
            )
        first.step()
        save(_state(tables, chain), directory)

        restored_tables = [_table(1)] if owns_table else []
        restored = ChainedOptimizer([_optimizer(restored_tables), _optimizer([])])
        loaded = load(_state(restored_tables, restored, is_loading=True), directory)
        assert set(loaded["optimizer"]) == {0, 1}
        assert loaded["optimizer"][1] == {"buckets": {}, "layer_param_groups": {}}
        restored.load_state_dict(loaded["optimizer"])
        if owns_table:
            state = restored.chained_optimizers[0].optimizer.compact_state(
                restored_tables[0].embedding.weight
            )
            assert state["rows"].tolist() == [0]
            assert state["step"].tolist() == [1]
        else:
            assert loaded["optimizer"][0] == {"buckets": {}, "layer_param_groups": {}}
        assert not restored.chained_optimizers[1].optimizer.state
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("invalid", [float("inf"), float("nan")])
def test_sparse_nonfinite_skips_all_pipeline_stages_without_updating(invalid):
    """A single bad row cannot corrupt the table or dense chained optimizer state."""
    Utils.initialize_model_parallel(pipeline_model_parallel_size=2)
    try:
        tables = [_table(1)] if parallel_state.get_pipeline_model_parallel_rank() == 0 else []
        optimizer = _optimizer(tables)
        before = []
        for table in tables:
            parameter = table.embedding.weight
            before.append(parameter.detach().clone())
            value = invalid if torch.distributed.get_rank() == 0 else 1.0
            parameter.grad = torch.sparse_coo_tensor(
                torch.tensor([[0]], device="cuda"),
                torch.full((1, 4), value, device="cuda"),
                parameter.shape,
            )
        assert optimizer.step() == (False, None, None)
        for table, original in zip(tables, before):
            torch.testing.assert_close(table.embedding.weight, original, rtol=0, atol=0)
            assert not optimizer.optimizer.state.get(table.embedding.weight)
        optimizer.zero_grad()
        assert not optimizer.prepare_grads()
    finally:
        Utils.destroy_model_parallel()
