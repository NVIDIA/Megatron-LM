# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import load, save
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import ChainedOptimizer, OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.sparse_adam import RowSparseAdam, RowSparseAdamOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.transformer.engram import Engram, EngramConfig, RowShardedMultiHeadEmbedding
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    Utils.world_size not in (2, 4, 8), reason="requires two, four, or eight ranks"
)

ROW_WIDTH = 80


def _set_global_row_values(memory):
    with torch.no_grad():
        for (start, end), local_offset, global_offset in zip(
            memory.layout.local_bounds, memory.layout.local_offsets, memory.layout.global_offsets
        ):
            rows = torch.arange(start, end, device="cuda") + global_offset
            values = rows[:, None] * 10 + torch.arange(memory.embedding_dim, device="cuda")
            memory.embedding.weight[local_offset : local_offset + end - start].copy_(values)


def _config(backend="row_a2a", **kwargs):
    values = dict(
        enabled=True,
        hash_table_min_sizes=(17, 19),
        max_ngram_size=3,
        embedding_dim_per_ngram=8,
        num_hash_heads_per_ngram=2,
        layer_ids=(0,),
        pad_id=0,
        seed=7,
        kernel_size=2,
        hidden_size=8,
        table_backend=backend,
        params_dtype=torch.float32,
        use_cpu_initialization=True,
    )
    values.update(kwargs)
    return EngramConfig(**values)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_row_a2a_forward_backward_duplicates_and_uneven_splits(dtype):
    Utils.initialize_model_parallel(tensor_model_parallel_size=Utils.world_size)
    try:
        memory = RowShardedMultiHeadEmbedding(
            [5, 7], D=4, layer_id=3, params_dtype=dtype, use_cpu_initialization=True
        ).cuda()
        _set_global_row_values(memory)
        ids = torch.tensor([[[0, 0], [4, 6], [0, 2]]], device="cuda")
        output = memory(ids)
        full_weight = torch.arange(12, device="cuda")[:, None] * 10 + torch.arange(4, device="cuda")
        expected = F.embedding(ids + torch.tensor([0, 5], device="cuda"), full_weight.to(dtype))
        torch.testing.assert_close(output, expected)
        output.sum().backward()
        grad = memory.embedding.weight.grad.coalesce()
        assert grad.is_sparse
        assert grad.indices().shape[0] == 1
        assert torch.isfinite(grad.values()).all()
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("calculate_per_token_loss", [False, True])
def test_row_a2a_optimizer_sees_equivalent_final_token_scaling(calculate_per_token_loss):
    tp_size = 2 if Utils.world_size == 4 else 1
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp_size)
    try:
        memory = RowShardedMultiHeadEmbedding(
            [5, 7],
            D=ROW_WIDTH,
            layer_id=3,
            use_cpu_initialization=True,
            calculate_per_token_loss=calculate_per_token_loss,
        ).cuda()
        ids = torch.tensor([[[1, 2], [1, 2]]], device="cuda")
        loss = memory(ids).sum()
        local_num_tokens = ids.shape[1]
        if not calculate_per_token_loss:
            loss = loss / local_num_tokens
        loss.backward()
        grad = memory.embedding.weight.grad.coalesce()
        dp_size = parallel_state.get_data_parallel_world_size(with_context_parallel=False)
        if calculate_per_token_loss:
            grad.values().mul_(1.0 / (local_num_tokens * dp_size))
            memory.embedding.weight.grad = grad
        torch.testing.assert_close(grad.values(), torch.ones_like(grad.values()))

        before = memory.embedding.weight.detach().clone()
        optimizer = RowSparseAdam(
            [memory.embedding.weight], lr=1.0e-2, betas=(0.9, 0.95), eps=1.0e-8
        )
        optimizer.step()
        if grad.values().numel():
            assert not torch.equal(memory.embedding.weight, before)
    finally:
        Utils.destroy_model_parallel()


def test_row_a2a_empty_request_is_collective_safe():
    Utils.initialize_model_parallel(tensor_model_parallel_size=Utils.world_size)
    try:
        memory = RowShardedMultiHeadEmbedding(
            [2, 3], D=4, layer_id=3, use_cpu_initialization=True
        ).cuda()
        ids = torch.empty((0, 0, 2), dtype=torch.long, device="cuda")
        output = memory(ids)
        assert output.shape == (0, 0, 2, 4)
        output.sum().backward()
        assert memory.embedding.weight.grad.coalesce().values().numel() == 0

    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("use_cpu_initialization", [True, False])
def test_row_a2a_initialization_preserves_cpu_and_cuda_rng(use_cpu_initialization):
    Utils.initialize_model_parallel(tensor_model_parallel_size=Utils.world_size)
    try:
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone()
        first = RowShardedMultiHeadEmbedding(
            [5, 7], D=4, layer_id=3, seed=123, use_cpu_initialization=use_cpu_initialization
        )
        assert torch.equal(torch.get_rng_state(), cpu_rng)
        assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
        second = RowShardedMultiHeadEmbedding(
            [5, 7], D=4, layer_id=3, seed=123, use_cpu_initialization=use_cpu_initialization
        )
        assert torch.equal(first.embedding.weight, second.embedding.weight)
    finally:
        Utils.destroy_model_parallel()


def test_row_a2a_full_engram_recompute_parity():
    Utils.initialize_model_parallel(tensor_model_parallel_size=Utils.world_size)
    try:
        module = Engram(engram_config=_config(), tokenizer_lookup=torch.arange(64)).cuda()
        ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], device="cuda")
        compressed_input_ids = module.compress_input_ids(ids)
        hidden = torch.randn(4, 2, 8, device="cuda")

        direct_hidden = hidden.detach().clone().requires_grad_(True)
        direct = module(direct_hidden, 0, compressed_input_ids)
        direct.square().sum().backward()
        direct_hidden_grad = direct_hidden.grad.clone()
        direct_table_grad = (
            module.layers["0"].multi_head_embedding.embedding.weight.grad.coalesce().clone()
        )
        direct_parameter_grads = {
            name: param.grad.detach().clone()
            for name, param in module.named_parameters()
            if param.grad is not None and not param.grad.is_sparse
        }

        module.zero_grad(set_to_none=True)
        recompute_hidden = hidden.detach().clone().requires_grad_(True)
        recomputed = checkpoint(
            lambda value, compressed_ids: module(value, 0, compressed_ids),
            recompute_hidden,
            compressed_input_ids,
            use_reentrant=False,
        )
        recomputed.square().sum().backward()
        torch.testing.assert_close(recomputed, direct)
        torch.testing.assert_close(recompute_hidden.grad, direct_hidden_grad)
        torch.testing.assert_close(
            module.layers["0"].multi_head_embedding.embedding.weight.grad.coalesce(),
            direct_table_grad,
        )
        for name, param in module.named_parameters():
            if name in direct_parameter_grads:
                torch.testing.assert_close(param.grad, direct_parameter_grads[name])
    finally:
        Utils.destroy_model_parallel()


def test_row_a2a_dcp_roundtrip(tmp_path_dist_ckpt):
    Utils.initialize_model_parallel(tensor_model_parallel_size=Utils.world_size)
    error = None
    try:
        try:
            source = RowShardedMultiHeadEmbedding(
                [5, 7], D=4, layer_id=3, use_cpu_initialization=True
            ).cuda()
            _set_global_row_values(source)
            expected = source.embedding.weight.detach().clone()
            checkpoint_dir = tmp_path_dist_ckpt / "row_a2a_roundtrip"
            if torch.distributed.get_rank() == 0:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.distributed.barrier()
            prefix = "engram.layers.3.multi_head_embedding."
            save(source.sharded_state_dict(prefix=prefix), checkpoint_dir)
            restored = RowShardedMultiHeadEmbedding(
                [5, 7], D=4, layer_id=3, use_cpu_initialization=True
            ).cuda()
            state = load(restored.sharded_state_dict(prefix=prefix), checkpoint_dir)
            restored.load_state_dict(
                {
                    "offsets": state[f"{prefix}offsets"],
                    "embedding.weight": state[f"{prefix}embedding.weight"],
                }
            )
            torch.testing.assert_close(restored.embedding.weight, expected)
        except Exception as exc:
            error = f"rank {torch.distributed.get_rank()}: {type(exc).__name__}: {exc}"
        errors = [None] * Utils.world_size
        torch.distributed.all_gather_object(errors, error)
        assert not any(errors), errors
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(
    Utils.world_size == 8,
    reason="the one-to-many checkpoint topology fixture is covered on two and four ranks",
)
@pytest.mark.parametrize("table_sizes", [(5, 7), (1, 11)])
def test_row_a2a_dcp_reshards_one_to_many_and_back(tmp_path_dist_ckpt, table_sizes):
    prefix = "engram.layers.3.multi_head_embedding."
    first_dir = tmp_path_dist_ckpt / "row_a2a_one_to_many"
    second_dir = tmp_path_dist_ckpt / "row_a2a_many_to_one"
    Utils.initialize_model_parallel(pipeline_model_parallel_size=Utils.world_size)
    try:
        source = None
        if torch.distributed.get_rank() == 0:
            source = RowShardedMultiHeadEmbedding(
                table_sizes, D=4, layer_id=3, use_cpu_initialization=True
            ).cuda()
            _set_global_row_values(source)
            first_dir.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()
        save(source.sharded_state_dict(prefix=prefix) if source else {}, first_dir)

        Utils.initialize_model_parallel(tensor_model_parallel_size=Utils.world_size)
        sharded = RowShardedMultiHeadEmbedding(
            table_sizes, D=4, layer_id=3, use_cpu_initialization=True
        ).cuda()
        state = load(sharded.sharded_state_dict(prefix=prefix), first_dir)
        sharded.load_state_dict(
            {
                "offsets": state[f"{prefix}offsets"],
                "embedding.weight": state[f"{prefix}embedding.weight"],
            }
        )
        expected = sharded.embedding.weight.detach().clone()
        _set_global_row_values(sharded)
        torch.testing.assert_close(sharded.embedding.weight, expected)

        if torch.distributed.get_rank() == 0:
            second_dir.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()
        save(sharded.sharded_state_dict(prefix=prefix), second_dir)

        Utils.initialize_model_parallel(pipeline_model_parallel_size=Utils.world_size)
        # Relocate the consumer from the first PP rank to the last, retaining
        # its logical table identity even when intermediate TP shards are empty.
        if torch.distributed.get_rank() == Utils.world_size - 1:
            restored = RowShardedMultiHeadEmbedding(
                table_sizes, D=4, layer_id=3, use_cpu_initialization=True
            ).cuda()
            state = load(restored.sharded_state_dict(prefix=prefix), second_dir)
            restored.load_state_dict(
                {
                    "offsets": state[f"{prefix}offsets"],
                    "embedding.weight": state[f"{prefix}embedding.weight"],
                }
            )
            _set_global_row_values(restored)
            torch.testing.assert_close(
                state[f"{prefix}embedding.weight"], restored.embedding.weight
            )
        else:
            load({}, second_dir)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sparse_adam_tracks_only_touched_rows(dtype):
    param = torch.nn.Parameter(
        torch.arange(6 * ROW_WIDTH, device="cuda").view(6, ROW_WIDTH).to(dtype)
    )
    optimizer = RowSparseAdam([param], lr=1.0e-2, betas=(0.9, 0.95), eps=1.0e-8)
    before = param.detach().clone()
    param.grad = torch.sparse_coo_tensor(
        torch.tensor([[1, 1, 4]], device="cuda"),
        torch.tensor(
            [[1] * ROW_WIDTH, [2] * ROW_WIDTH, [3] * ROW_WIDTH], device="cuda", dtype=dtype
        ),
        param.shape,
    ).coalesce()
    optimizer.step()
    state = optimizer.state[param]
    assert state["rows"].tolist() == [1, 4]
    assert state["step"].tolist() == [1, 1]
    torch.testing.assert_close(param[[0, 2, 3, 5]], before[[0, 2, 3, 5]])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sparse_adam_amortizes_growth_and_matches_exact_adam(dtype):
    param = torch.nn.Parameter(
        (torch.arange(32 * ROW_WIDTH, device="cuda").view(32, ROW_WIDTH).float() / 10).to(dtype)
    )
    optimizer = RowSparseAdam([param], lr=1.0e-2, betas=(0.9, 0.95), eps=1.0e-8)
    reference = param.detach().float().clone()
    exp_avg = {}
    exp_avg_sq = {}
    steps = {}
    storage_pointers = []

    for row in range(24):
        grad = torch.full((ROW_WIDTH,), (row + 1) / 7, device="cuda", dtype=dtype)
        param.grad = torch.sparse_coo_tensor(
            torch.tensor([[row]], device="cuda"), grad.unsqueeze(0), param.shape
        )
        optimizer.step()
        grad = grad.float()
        exp_avg[row] = 0.9 * exp_avg.get(row, torch.zeros_like(grad)) + 0.1 * grad
        exp_avg_sq[row] = 0.95 * exp_avg_sq.get(row, torch.zeros_like(grad)) + 0.05 * grad.square()
        steps[row] = steps.get(row, 0) + 1
        update = exp_avg[row] / (1.0 - 0.9 ** steps[row])
        denom = (exp_avg_sq[row] / (1.0 - 0.95 ** steps[row])).sqrt() + 1.0e-8
        reference[row].addcdiv_(update, denom, value=-1.0e-2)
        storage_pointers.append(optimizer.state[param]["_exp_avg_storage"].data_ptr())

    assert len(set(storage_pointers[:16])) == 1
    assert len(set(storage_pointers)) == 2
    torch.testing.assert_close(
        param.float(), reference, atol=3.0e-2 if dtype == torch.bfloat16 else 1.0e-6, rtol=0
    )
    compact_state = next(iter(optimizer.state_dict()["state"].values()))
    assert compact_state["rows"].numel() == 24
    assert not any(key.startswith("_") for key in compact_state)

    restored_param = torch.nn.Parameter(param.detach().clone())
    restored = RowSparseAdam([restored_param], lr=1.0e-2, betas=(0.9, 0.95), eps=1.0e-8)
    restored.load_state_dict(optimizer.state_dict())
    next_grad = torch.ones((2, ROW_WIDTH), device="cuda", dtype=dtype)
    for target_param, target_optimizer in ((param, optimizer), (restored_param, restored)):
        target_param.grad = torch.sparse_coo_tensor(
            torch.tensor([[3, 31]], device="cuda"), next_grad, target_param.shape
        )
        target_optimizer.step()
    torch.testing.assert_close(restored_param, param)


def test_sparse_adam_dcp_preserves_next_update(tmp_path_dist_ckpt):
    Utils.initialize_model_parallel(tensor_model_parallel_size=Utils.world_size)
    try:
        memory = RowShardedMultiHeadEmbedding(
            [5, 7],
            D=ROW_WIDTH,
            layer_id=3,
            params_dtype=torch.bfloat16,
            use_cpu_initialization=True,
        ).cuda()
        config = OptimizerConfig(
            lr=1.0e-2, min_lr=1.0e-4, clip_grad=0.0, log_num_zeros_in_grad=True
        )
        group = {
            "params": [memory.embedding.weight],
            "lr_mult": 5.0,
            "wd_mult": 0.0,
            "is_engram_row_parallel": True,
        }
        optimizer = RowSparseAdamOptimizer([group], config)
        table_rank = torch.distributed.get_rank(
            group=parallel_state.get_tensor_and_data_parallel_group()
        )
        local_rows = (
            torch.arange(memory.embedding.weight.shape[0], device="cuda")
            if table_rank == 0
            else torch.empty(0, dtype=torch.long, device="cuda")
        )
        initial_values = torch.zeros(
            (local_rows.numel(), ROW_WIDTH), device="cuda", dtype=torch.bfloat16
        )
        initial_values[:, 0] = (torch.arange(1, local_rows.numel() + 1, device="cuda") / 10).to(
            torch.bfloat16
        )
        memory.embedding.weight.grad = torch.sparse_coo_tensor(
            local_rows.unsqueeze(0), initial_values, memory.embedding.weight.shape
        )
        success, grad_norm, num_zeros = optimizer.step()
        assert success and grad_norm >= 0
        total_touched = torch.tensor(local_rows.numel(), device="cuda", dtype=torch.int64)
        torch.distributed.all_reduce(total_touched)
        assert num_zeros == total_touched.item() * (ROW_WIDTH - 1)
        assert optimizer.optimizer.state[memory.embedding.weight]["rows"].numel() == (
            local_rows.numel()
        )

        checkpoint_dir = tmp_path_dist_ckpt / "row_sparse_adam_roundtrip"
        if torch.distributed.get_rank() == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()
        sharded_state = optimizer.sharded_state_dict({})
        table_world = torch.distributed.get_world_size(
            group=parallel_state.get_tensor_and_data_parallel_group()
        )
        assert all(
            bucket % table_world == table_rank
            for layer_buckets in sharded_state["buckets"].values()
            for bucket in layer_buckets
        )
        assert sharded_state["layer_param_groups"][3].replica_id == table_rank
        routed_rows = optimizer._checkpoint_routing_stats["save_received_rows"]
        routed_rows_by_rank = [None] * Utils.world_size
        torch.distributed.all_gather_object(routed_rows_by_rank, routed_rows)
        local_touched_by_rank = [None] * Utils.world_size
        torch.distributed.all_gather_object(local_touched_by_rank, local_rows.numel())
        assert sum(routed_rows_by_rank) == sum(local_touched_by_rank)
        assert any(count == 0 for count in local_touched_by_rank)
        save(sharded_state, checkpoint_dir)

        restored_memory = RowShardedMultiHeadEmbedding(
            [5, 7],
            D=ROW_WIDTH,
            layer_id=3,
            params_dtype=torch.bfloat16,
            use_cpu_initialization=True,
        ).cuda()
        with torch.no_grad():
            restored_memory.embedding.weight.copy_(memory.embedding.weight)
        restored_group = dict(group, params=[restored_memory.embedding.weight])
        restored = RowSparseAdamOptimizer([restored_group], config)
        state = load(restored.sharded_state_dict({}, is_loading=True), checkpoint_dir)
        restored.load_state_dict(state)
        original_state = optimizer.optimizer.state[memory.embedding.weight]
        restored_state = restored.optimizer.state[restored_memory.embedding.weight]
        for key in ("rows", "exp_avg", "exp_avg_sq", "step", "master_param"):
            if key not in original_state:
                continue
            torch.testing.assert_close(restored_state[key], original_state[key])
        assert restored._checkpoint_routing_stats["load_received_rows"] == local_rows.numel()

        if memory.embedding.weight.shape[0]:
            next_rows = torch.tensor([[0, memory.embedding.weight.shape[0] - 1]], device="cuda")
            next_values = torch.tensor(
                [[0.3] * ROW_WIDTH, [0.7] * ROW_WIDTH], device="cuda", dtype=torch.bfloat16
            )
        else:
            next_rows = torch.empty((1, 0), dtype=torch.long, device="cuda")
            next_values = torch.empty((0, ROW_WIDTH), device="cuda", dtype=torch.bfloat16)
        for target_optimizer, target_memory in ((optimizer, memory), (restored, restored_memory)):
            target_memory.embedding.weight.grad = torch.sparse_coo_tensor(
                next_rows, next_values, target_memory.embedding.weight.shape
            )
            target_optimizer.prepare_grads()
            target_optimizer.step_with_ready_grads()
        assert torch.equal(restored_memory.embedding.weight, memory.embedding.weight)
        restored_state = restored.optimizer.state[restored_memory.embedding.weight]
        original_state = optimizer.optimizer.state[memory.embedding.weight]
        for key in ("rows", "exp_avg", "exp_avg_sq", "step", "master_param"):
            if key in original_state:
                assert torch.equal(restored_state[key], original_state[key])
    finally:
        Utils.destroy_model_parallel()


class _IntegrationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.memory = RowShardedMultiHeadEmbedding(
            [5, 7], D=ROW_WIDTH, layer_id=3, use_cpu_initialization=True
        ).cuda()
        # Keep at least one dense optimizer shard on every supported test rank.
        self.dense = torch.nn.Parameter(torch.ones(8, device="cuda"))
        self.ddp_config = SimpleNamespace(use_custom_fsdp=False, use_megatron_fsdp=False)

    def forward(self, ids):
        mask = torch.zeros(ROW_WIDTH, device="cuda")
        mask[0] = 1.0
        return (self.memory(ids) * mask).sum() + self.dense.sum()


class _ThreeOptimizerIntegrationModel(_IntegrationModel):
    def __init__(self):
        super().__init__()
        self.expert = torch.nn.Parameter(torch.ones(8, device="cuda"))
        self.expert.allreduce = False


def _transformer_config():
    return TransformerConfig(
        num_layers=1, hidden_size=ROW_WIDTH, num_attention_heads=1, use_cpu_initialization=True
    )


def test_ddp_excludes_row_table_from_dense_grad_buffer():
    Utils.initialize_model_parallel(tensor_model_parallel_size=Utils.world_size)
    try:
        model = _IntegrationModel()
        ddp = DistributedDataParallel(
            _transformer_config(), DistributedDataParallelConfig(overlap_grad_reduce=False), model
        )
        row_weight = model.memory.embedding.weight
        assert row_weight in ddp.params_without_grad_buffer
        assert row_weight not in ddp.param_to_bucket_group
        assert not hasattr(row_weight, "main_grad")
        assert hasattr(model.dense, "main_grad")
        ddp.broadcast_params()
    finally:
        Utils.destroy_model_parallel()


def test_row_a2a_coexists_with_distributed_optimizer():
    if Utils.world_size == 8:
        pytest.skip("the focused DistOpt chain fixture is covered on two and four ranks")
    Utils.initialize_model_parallel(tensor_model_parallel_size=Utils.world_size)
    try:
        raw = _IntegrationModel()
        ddp = DistributedDataParallel(
            _transformer_config(),
            DistributedDataParallelConfig(
                use_distributed_optimizer=True, overlap_grad_reduce=False
            ),
            raw,
        )
        optimizer = get_megatron_optimizer(
            OptimizerConfig(
                optimizer="adam",
                lr=2.0e-3,
                min_lr=1.0e-4,
                weight_decay=0.1,
                clip_grad=1.0,
                use_distributed_optimizer=True,
                log_num_zeros_in_grad=True,
            ),
            [ddp],
            use_gloo_process_groups=False,
        )
        OptimizerParamScheduler(
            optimizer,
            init_lr=0.0,
            max_lr=2.0e-3,
            min_lr=1.0e-4,
            lr_warmup_steps=0,
            lr_decay_steps=10,
            lr_decay_style="constant",
            start_wd=0.1,
            end_wd=0.1,
            wd_incr_steps=10,
            wd_incr_style="constant",
        )
        assert isinstance(optimizer, ChainedOptimizer)
        assert any(
            isinstance(item, RowSparseAdamOptimizer) for item in optimizer.chained_optimizers
        )
        sparse = next(
            item
            for item in optimizer.chained_optimizers
            if isinstance(item, RowSparseAdamOptimizer)
        )
        assert id(raw.memory.embedding.weight) in {
            id(param) for group in sparse.param_groups for param in group["params"]
        }
        sparse_group = next(group for group in sparse.param_groups if group["params"])
        assert sparse_group["max_lr"] == pytest.approx(1.0e-2)
        assert sparse_group["lr"] == pytest.approx(1.0e-2)
        assert sparse_group["wd_mult"] == 0.0
        assert sparse_group["weight_decay"] == 0.0
        dense_before = raw.dense.detach().clone()
        table_before = raw.memory.embedding.weight.detach().clone()
        ddp(torch.tensor([[[1, 2], [1, 2]]], device="cuda")).backward()
        # Dense values have gradient one and two sparse rows have one
        # nonzero value of two after row-A2A duplicate coalescing.
        expected_grad_norm = math.sqrt(raw.dense.numel() * 1.0**2 + 2 * 2.0**2)
        sparse_grad_before = raw.memory.embedding.weight.grad.coalesce().values().clone()
        success, grad_norm, num_zeros = optimizer.step()
        clip_coefficient = 1.0 / (expected_grad_norm + 1.0e-6)
        sparse_grad_after = raw.memory.embedding.weight.grad.coalesce().values()
        local_clip_error = torch.tensor(
            (
                (sparse_grad_after - sparse_grad_before * clip_coefficient).abs().max().item()
                if sparse_grad_after.numel()
                else 0.0
            ),
            device="cuda",
        )
        torch.distributed.all_reduce(local_clip_error, op=torch.distributed.ReduceOp.MAX)
        dense_changed = torch.tensor(int(not torch.equal(raw.dense, dense_before)), device="cuda")
        torch.distributed.all_reduce(dense_changed, op=torch.distributed.ReduceOp.MAX)
        table_changed = torch.tensor(
            int(not torch.equal(raw.memory.embedding.weight, table_before)), device="cuda"
        )
        torch.distributed.all_reduce(table_changed, op=torch.distributed.ReduceOp.MAX)
        local_sparse_delta = (raw.memory.embedding.weight - table_before).abs().max()
        torch.distributed.all_reduce(local_sparse_delta, op=torch.distributed.ReduceOp.MAX)
        dense_delta = (raw.dense - dense_before).abs().max()
        torch.distributed.all_reduce(dense_delta, op=torch.distributed.ReduceOp.MAX)

        assert success
        assert grad_norm == pytest.approx(expected_grad_norm)
        assert local_clip_error.item() < 1.0e-7
        assert num_zeros == 2 * (ROW_WIDTH - 1)
        assert dense_changed.item()
        assert table_changed.item()
        assert local_sparse_delta.item() == pytest.approx(
            5.0 * dense_delta.item(), rel=2.0e-4, abs=1.0e-6
        )
    finally:
        Utils.destroy_model_parallel()


def test_chained_distributed_optimizer_checkpoint_roundtrip(tmp_path_dist_ckpt):
    if Utils.world_size == 8:
        pytest.skip("the focused DistOpt checkpoint fixture is covered on two and four ranks")
    Utils.initialize_model_parallel(tensor_model_parallel_size=Utils.world_size)
    try:
        transformer_config = _transformer_config()
        ddp_config = DistributedDataParallelConfig(
            use_distributed_optimizer=True, overlap_grad_reduce=False
        )
        optimizer_config = OptimizerConfig(
            optimizer="adam",
            lr=1.0e-2,
            min_lr=1.0e-4,
            use_distributed_optimizer=True,
            clip_grad=0.0,
        )

        def build():
            model = _ThreeOptimizerIntegrationModel()
            ddp = DistributedDataParallel(transformer_config, ddp_config, model)
            optimizer = get_megatron_optimizer(optimizer_config, [ddp])
            return model, optimizer

        source, source_optimizer = build()
        assert len(source_optimizer.chained_optimizers) == 3
        source.dense.main_grad.fill_(0.25)
        source.expert.main_grad.fill_(0.5)
        source.memory.embedding.weight.grad = torch.sparse_coo_tensor(
            torch.tensor([[0]], device="cuda"),
            torch.full((1, ROW_WIDTH), 0.75, device="cuda"),
            source.memory.embedding.weight.shape,
        )
        assert source_optimizer.step()[0]

        checkpoint_dir = tmp_path_dist_ckpt / "chained_optimizer_roundtrip"
        if torch.distributed.get_rank() == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()
        sharding_type = "dp_zero_gather_scatter"
        save(source_optimizer.sharded_state_dict({}, sharding_type=sharding_type), checkpoint_dir)

        restored, restored_optimizer = build()
        state = load(
            restored_optimizer.sharded_state_dict({}, is_loading=True, sharding_type=sharding_type),
            checkpoint_dir,
        )
        with torch.no_grad():
            restored_optimizer.load_state_dict(state)
        restored.dense.main_grad.fill_(0.25)
        restored.expert.main_grad.fill_(0.5)
        restored.memory.embedding.weight.grad = torch.sparse_coo_tensor(
            torch.tensor([[0]], device="cuda"),
            torch.full((1, ROW_WIDTH), 0.75, device="cuda"),
            restored.memory.embedding.weight.shape,
        )
        assert restored_optimizer.step()[0]
    finally:
        Utils.destroy_model_parallel()
