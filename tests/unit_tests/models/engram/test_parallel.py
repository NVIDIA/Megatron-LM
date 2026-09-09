# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import load, save
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.sparse_adam import RowSparseAdamOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.engram import (
    Engram,
    EngramConfig,
    MultiHeadEmbedding,
    ParallelSequenceLayout,
)
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(Utils.world_size != 2, reason="requires exactly two ranks")


def _normal(weight):
    torch.nn.init.normal_(weight, mean=0.0, std=0.02)


def _config(**kwargs):
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
        init_method=_normal,
        params_dtype=torch.float32,
        use_cpu_initialization=True,
    )
    values.update(kwargs)
    return EngramConfig(**values)


def _direct(module, hidden, compressed_input_ids):
    layer = module.layers["0"]
    hashes = module.hash_mapping.forward_compressed(compressed_input_ids, 0)
    embeddings = layer.multi_head_embedding(hashes).flatten(start_dim=-2)
    return layer(hidden, embeddings)


def test_tp_memory_width_shard_matches_full_lookup_and_gradient():
    Utils.initialize_model_parallel(tensor_model_parallel_size=2)
    try:
        memory = MultiHeadEmbedding(
            [5, 7], D=8, tensor_parallel_size=2, init_method=_normal, use_cpu_initialization=True
        ).cuda()
        local_weight = memory.embedding.weight.detach()
        shards = [torch.empty_like(local_weight) for _ in range(2)]
        torch.distributed.all_gather(
            shards, local_weight, group=parallel_state.get_tensor_model_parallel_group()
        )
        full_weight = torch.cat(shards, dim=1)
        ids = torch.tensor([[[1, 2], [3, 4]]], device="cuda")
        output = memory(ids)
        expected = F.embedding(ids + memory.offsets, full_weight)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

        grad = torch.arange(output.numel(), device="cuda", dtype=output.dtype).view_as(output)
        output.backward(grad)
        reference = full_weight.detach().clone().requires_grad_(True)
        F.embedding(ids + memory.offsets, reference).backward(grad)
        rank = parallel_state.get_tensor_model_parallel_rank()
        torch.testing.assert_close(
            memory.embedding.weight.grad, reference.grad.chunk(2, dim=1)[rank], rtol=0, atol=0
        )
    finally:
        Utils.destroy_model_parallel()


def test_sequence_parallel_matches_full_sequence():
    Utils.initialize_model_parallel(tensor_model_parallel_size=2)
    try:
        # SP ranks represent partitions of one logical model/input, so both
        # module initialization and the full-sequence oracle must be identical.
        torch.manual_seed(2026)
        module = Engram(
            engram_config=_config(sequence_parallel=True), tokenizer_lookup=torch.arange(64)
        ).cuda()
        full_hidden = torch.randn(8, 2, 8, device="cuda")
        local_hidden = (
            full_hidden.chunk(2, dim=0)[parallel_state.get_tensor_model_parallel_rank()]
            .detach()
            .requires_grad_(True)
        )
        compressed_input_ids = module.compress_input_ids(
            torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]], device="cuda")
        )
        with torch.no_grad():
            expected = _direct(module, full_hidden, compressed_input_ids)
        actual = module(local_hidden, 0, compressed_input_ids)
        gathered = [torch.empty_like(actual) for _ in range(2)]
        torch.distributed.all_gather(
            gathered, actual, group=parallel_state.get_tensor_model_parallel_group()
        )
        torch.testing.assert_close(torch.cat(gathered), expected)
        actual.square().sum().backward()
        assert torch.isfinite(local_hidden.grad).all()
    finally:
        Utils.destroy_model_parallel()


def test_context_parallel_mirrored_layout_matches_full_sequence():
    Utils.initialize_model_parallel(context_parallel_size=2)
    try:
        # CP ranks select mirrored shards from the same logical full sequence.
        torch.manual_seed(2027)
        module = Engram(
            engram_config=_config(context_parallel_size=2), tokenizer_lookup=torch.arange(64)
        ).cuda()
        full_hidden = torch.randn(8, 1, 8, device="cuda")
        full_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device="cuda")
        local_hidden = (
            ParallelSequenceLayout._select_cp(full_hidden, 0).detach().requires_grad_(True)
        )
        local_compressed_input_ids = module.compress_input_ids(
            ParallelSequenceLayout._select_cp(full_ids, 1)
        )
        with torch.no_grad():
            expected = _direct(module, full_hidden, module.compress_input_ids(full_ids))
        actual = module(local_hidden, 0, local_compressed_input_ids)
        torch.testing.assert_close(actual, ParallelSequenceLayout._select_cp(expected, 0))
        actual.square().sum().backward()
        assert torch.isfinite(local_hidden.grad).all()
    finally:
        Utils.destroy_model_parallel()


def test_tp_distributed_checkpoint_roundtrip(tmp_path_dist_ckpt):
    Utils.initialize_model_parallel(tensor_model_parallel_size=2)
    error = None
    try:
        try:
            torch.manual_seed(3030)
            source = Engram(engram_config=_config(), tokenizer_lookup=torch.arange(64)).cuda()
            expected = {
                name: value.detach().clone()
                for name, value in source.state_dict().items()
                if torch.is_tensor(value)
            }
            ckpt_dir = tmp_path_dist_ckpt / "engram_tp_roundtrip"
            if torch.distributed.get_rank() == 0:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.distributed.barrier()
            save(source.sharded_state_dict(), ckpt_dir)
            restored = Engram(engram_config=_config(), tokenizer_lookup=torch.arange(64)).cuda()
            state = load(restored.sharded_state_dict(), ckpt_dir)
            result = restored.load_state_dict(state, strict=True)
            assert not result.missing_keys and not result.unexpected_keys
            for name, value in restored.state_dict().items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        except Exception as exc:
            error = f"rank {torch.distributed.get_rank()}: {type(exc).__name__}: {exc}"
        errors = [None, None]
        torch.distributed.all_gather_object(errors, error)
        assert not any(errors), errors
    finally:
        Utils.destroy_model_parallel()


def test_engram_uses_explicit_process_group_collection(monkeypatch):
    Utils.initialize_model_parallel(tensor_model_parallel_size=2)
    try:
        groups = ProcessGroupCollection(
            tp=parallel_state.get_tensor_model_parallel_group(),
            pp=parallel_state.get_pipeline_model_parallel_group(),
            cp=parallel_state.get_context_parallel_group(),
            dp=parallel_state.get_data_parallel_group(with_context_parallel=False),
            dp_cp=parallel_state.get_data_parallel_group(with_context_parallel=True),
            tp_dp_cp=parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True),
            mp=parallel_state.get_model_parallel_group(),
        )
        with pytest.raises(ValueError, match="missing required groups"):
            Engram(
                engram_config=_config(table_backend="row_a2a"),
                tokenizer_lookup=torch.arange(64),
                pg_collection=ProcessGroupCollection(tp=groups.tp),
            )

        def reject_global_group(*args, **kwargs):
            raise AssertionError("global process groups must not be consulted")

        for getter in (
            "get_tensor_and_data_parallel_group",
            "get_tensor_model_parallel_group",
            "get_data_parallel_group",
            "get_model_parallel_group",
            "get_context_parallel_group",
        ):
            monkeypatch.setattr(parallel_state, getter, reject_global_group)
        module = Engram(
            engram_config=_config(table_backend="row_a2a"),
            tokenizer_lookup=torch.arange(64),
            pg_collection=groups,
        ).cuda()
        table = module.layers["0"].multi_head_embedding
        assert table.table_group is groups.tp_dp_cp
        assert table.embedding.weight.engram_table_metadata.table_group is groups.tp_dp_cp

        hidden = torch.randn(4, 1, 8, device="cuda", requires_grad=True)
        compressed_input_ids = module.compress_input_ids(
            torch.tensor([[1, 2, 3, 4]], device="cuda")
        )
        module(hidden, 0, compressed_input_ids).sum().backward()
        module.sharded_state_dict()

        optimizer = RowSparseAdamOptimizer(
            [
                {
                    "params": [table.embedding.weight],
                    "lr_mult": 5.0,
                    "wd_mult": 0.0,
                    "is_engram_row_parallel": True,
                }
            ],
            OptimizerConfig(lr=1.0e-2, min_lr=1.0e-4, clip_grad=0.0),
        )
        assert optimizer.table_group is groups.tp_dp_cp
        assert optimizer.tp_group is groups.tp
        assert optimizer.get_grad_stats_parallel_group() is groups.mp
    finally:
        Utils.destroy_model_parallel()


def test_contiguous_boundary_tokens_and_attention_hidden_layout_match():
    from types import SimpleNamespace

    from megatron.core.context_parallel import convert_cp_layout
    from megatron.core.transformer.engram.hybrid_adapter import EngramHybridProvider

    Utils.initialize_model_parallel(context_parallel_size=2)
    try:
        groups = ProcessGroupCollection.use_mpu_process_groups()
        provider = EngramHybridProvider(
            config=SimpleNamespace(
                num_layers=2,
                engram_layer_ids=[0],
                engram_target_layer_indices=None,
                cuda_graph_impl='none',
                linear_cp_layout='contiguous',
                attention_cp_layout='zigzag',
            ),
            tokenizer_lookup=torch.arange(64),
            pad_id=0,
            hybrid_layer_pattern='*-',
            pg_collection=groups,
        )
        torch.manual_seed(2027)
        module = Engram(
            engram_config=_config(context_parallel_size=2),
            tokenizer_lookup=torch.arange(64),
            pg_collection=groups,
        ).cuda()
        full_hidden = torch.randn(8, 1, 8, device='cuda')
        full_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device='cuda')
        rank = groups.cp.rank()
        boundary_hidden = full_hidden.chunk(2, dim=0)[rank]
        boundary_tokens = full_ids.chunk(2, dim=1)[rank]
        hidden = convert_cp_layout(boundary_hidden, 'contiguous', 'zigzag', cp_group=groups.cp)
        tokens = provider.prepare(
            boundary_tokens, inference_context=None, packed_seq_params=None, cp_batch=None
        )
        torch.testing.assert_close(
            tokens, ParallelSequenceLayout._select_cp(full_ids, 1, groups.cp), rtol=0, atol=0
        )
        with torch.no_grad():
            expected = _direct(module, full_hidden, module.compress_input_ids(full_ids))
            actual = module(hidden, 0, module.compress_input_ids(tokens))
        torch.testing.assert_close(
            actual, ParallelSequenceLayout._select_cp(expected, 0, groups.cp)
        )
    finally:
        Utils.destroy_model_parallel()
