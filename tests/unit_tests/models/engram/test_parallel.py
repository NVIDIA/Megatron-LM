# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pretrain_hybrid
from megatron.core import parallel_state
from megatron.core.dist_checkpointing import load, save
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.engram import (
    Engram,
    EngramConfig,
    MultiHeadEmbedding,
    ParallelSequenceLayout,
)
from megatron.core.transformer.engram.optimizer import get_engram_optimizer
from megatron.training.datasets.data_samplers import MegatronPretrainingSampler
from tests.unit_tests.models.engram.test_integration import model_parallel, prepare_row_gradients
from tests.unit_tests.test_utilities import Utils


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


@pytest.mark.skipif(Utils.world_size != 2, reason='requires exactly two ranks')
def test_tp_memory_width_shard_matches_full_lookup_and_gradient():
    with model_parallel(tensor_model_parallel_size=2):
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


@pytest.mark.skipif(Utils.world_size != 2, reason='requires exactly two ranks')
def test_sequence_parallel_matches_full_sequence():
    with model_parallel(tensor_model_parallel_size=2):
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


@pytest.mark.skipif(Utils.world_size != 2, reason='requires exactly two ranks')
def test_context_parallel_mirrored_layout_matches_full_sequence():
    with model_parallel(context_parallel_size=2):
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


@pytest.mark.skipif(Utils.world_size != 2, reason='requires exactly two ranks')
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


@pytest.mark.skipif(Utils.world_size != 2, reason='requires exactly two ranks')
def test_engram_uses_explicit_process_group_collection(monkeypatch):
    with model_parallel(tensor_model_parallel_size=2):
        groups = ProcessGroupCollection.use_mpu_process_groups()
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

        from megatron.core.transformer.transformer_config import TransformerConfig

        ddp = DistributedDataParallel(
            TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=2),
            DistributedDataParallelConfig(),
            module,
            pg_collection=groups,
        )
        optimizer = get_engram_optimizer(
            OptimizerConfig(lr=1e-2, min_lr=1e-4, clip_grad=0.0),
            [ddp],
            pg_collection=groups,
            parallel_groups=module.parallel_groups,
            use_gloo_process_groups=False,
        )
        hidden = torch.randn(4, 1, 8, device="cuda", requires_grad=True)
        compressed_input_ids = module.compress_input_ids(
            torch.tensor([[1, 2, 3, 4]], device="cuda")
        )
        module(hidden, 0, compressed_input_ids).sum().backward()
        module.sharded_state_dict()

        assert table.embedding.weight.main_grad.dtype == torch.float32
        owner = optimizer.chained_optimizers[-1]
        assert owner.tp_group is groups.tp
        assert owner.get_grad_stats_parallel_group() is module.parallel_groups.stats_group


@pytest.mark.skipif(Utils.world_size != 2, reason='requires exactly two ranks')
def test_contiguous_boundary_tokens_and_attention_hidden_layout_match():
    from types import SimpleNamespace

    from megatron.core.context_parallel import convert_cp_layout
    from megatron.core.transformer.engram.hybrid_adapter import EngramHybridProvider

    with model_parallel(context_parallel_size=2):
        groups = ProcessGroupCollection.use_mpu_process_groups()
        provider = EngramHybridProvider(
            config=SimpleNamespace(
                num_layers=2,
                engram_layer_ids=[0],
                engram_table_backend="local",
                engram_row_parallel_size=None,
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


class _IdentifiedSamples(Dataset):
    """Encode sample identity and source position in each token."""

    def __len__(self):
        return 128

    def __getitem__(self, index):
        tokens = index * 100 + torch.arange(16)
        return {
            'tokens': tokens,
            'labels': tokens + 1,
            'position_ids': torch.arange(16),
            'loss_mask': torch.ones(16),
        }


def _iterator():
    return iter(
        DataLoader(
            _IdentifiedSamples(),
            batch_sampler=MegatronPretrainingSampler(
                total_samples=128,
                consumed_samples=8,
                micro_batch_size=2,
                data_parallel_rank=parallel_state.get_data_parallel_rank(),
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            ),
            num_workers=0,
        )
    )


def _configure(monkeypatch, tp, pp, cp, vp, mtp):
    args = SimpleNamespace(
        sequence_packing_scheduler=None,
        context_parallel_size=cp,
        sft=False,
        dataloader_inter_document_masking=False,
        create_attention_mask_in_dataloader=False,
        hybrid_context_parallel=False,
        micro_batch_size=2,
        seq_length=16,
        pipeline_model_parallel_size=pp,
    )
    config = SimpleNamespace(
        pipeline_model_parallel_layout=None,
        mtp_num_layers=1 if mtp else None,
        virtual_pipeline_model_parallel_size=vp,
        linear_cp_layout='contiguous',
        attention_cp_layout='zigzag',
        sequence_parallel=tp > 1,
        tensor_model_parallel_size=tp,
    )
    monkeypatch.setattr(pretrain_hybrid, 'get_args', lambda: args)
    monkeypatch.setattr(pretrain_hybrid, 'core_transformer_config_from_args', lambda _: config)


def _check_tokens(cp_batch, microbatch):
    dp_rank = parallel_state.get_data_parallel_rank()
    dp_size = parallel_state.get_data_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    cp_size = parallel_state.get_context_parallel_world_size()
    sample_ids = 8 + microbatch * 2 * dp_size + dp_rank * 2 + torch.arange(2, device='cuda')
    positions = torch.arange(16, device='cuda')
    expected = sample_ids[:, None] * 100 + positions[None, :]
    contiguous = expected.chunk(cp_size, dim=1)[cp_rank]
    torch.testing.assert_close(cp_batch.get_batch()['tokens'], contiguous, rtol=0, atol=0)
    if cp_size > 1:
        segments = expected.chunk(cp_size * 2, dim=1)
        zigzag = torch.cat((segments[cp_rank], segments[2 * cp_size - cp_rank - 1]), dim=1)
        torch.testing.assert_close(cp_batch.get_batch('zigzag')['tokens'], zigzag, rtol=0, atol=0)


@pytest.mark.parametrize(
    ('tp', 'pp', 'cp', 'vp', 'ep'),
    [(1, 2, 1, None, 1), (2, 2, 1, 2, 1), (1, 2, 2, 2, 1), (2, 2, 2, 2, 1), (1, 2, 1, 2, 2)],
)
@pytest.mark.parametrize('mtp', [False, True])
def test_chunk_data_streams_match_across_pipeline_and_context_ranks(
    monkeypatch, tp, pp, cp, vp, mtp, ep
):
    if Utils.world_size != tp * pp * cp * ep:
        pytest.skip('requires the topology-specific world size')
    with model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
        virtual_pipeline_model_parallel_size=vp,
        expert_model_parallel_size=ep,
    ):
        _configure(monkeypatch, tp, pp, cp, vp, mtp)
        # Every chunk owns an independent native sampler. Rank/chunk scheduling
        # must not affect the source sample consumed by its nth forward.
        chunks = list(range(vp)) if vp is not None else [None]
        iterators = {chunk: _iterator() for chunk in chunks}
        calls = []
        original_broadcast = torch.distributed.broadcast

        def record_broadcast(tensor, src, group, **kwargs):
            calls.append(group)
            return original_broadcast(tensor, src, group=group, **kwargs)

        monkeypatch.setattr(torch.distributed, 'broadcast', record_broadcast)
        for microbatch in range(3):
            # Opposite PP ranks deliberately visit their local VPP chunks in
            # different orders; no PP forward collective can be inserted here.
            local_chunks = (
                chunks if parallel_state.get_pipeline_model_parallel_rank() == 0 else chunks[::-1]
            )
            for chunk in local_chunks:
                assert pretrain_hybrid.is_dataset_built_on_rank(
                    vp_stage=chunk, requires_token_ids=True
                ) == (parallel_state.get_tensor_model_parallel_rank() == 0)
                batch = pretrain_hybrid.get_batch(
                    iterators[chunk], vp_stage=chunk, requires_token_ids=True
                )
                _check_tokens(batch, microbatch)
                last = parallel_state.is_pipeline_last_stage(
                    ignore_virtual=chunk is None, vp_stage=chunk
                )
                assert (batch.get_batch()['labels'] is not None) == last
        assert calls and all(
            group is parallel_state.get_tensor_model_parallel_group() for group in calls
        )


@pytest.mark.parametrize('mtp', [False, True])
def test_feature_off_middle_chunk_does_not_advance_data(monkeypatch, mtp):
    if Utils.world_size != 2:
        pytest.skip('requires two pipeline ranks')
    with model_parallel(pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=2):
        _configure(monkeypatch, 1, 2, 1, 2, mtp)
        rank = parallel_state.get_pipeline_model_parallel_rank()
        # Rank zero's second chunk and rank one's first chunk are both interior.
        vp_stage = 1 - rank
        assert not pretrain_hybrid.is_dataset_built_on_rank(vp_stage=vp_stage)
        iterator = _iterator()
        batch = pretrain_hybrid.get_batch(iterator, vp_stage=vp_stage)
        assert all(value is None for value in batch.get_batch().values())
        expected = next(_iterator())
        actual = next(iterator)
        torch.testing.assert_close(actual['tokens'], expected['tokens'], rtol=0, atol=0)


@pytest.mark.parametrize('mtp', [False, True])
def test_feature_off_endpoints_keep_original_batch_schema(monkeypatch, mtp):
    if Utils.world_size != 2:
        pytest.skip('requires two pipeline ranks')
    with model_parallel(pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=2):
        _configure(monkeypatch, 1, 2, 1, 2, mtp)
        rank = parallel_state.get_pipeline_model_parallel_rank()
        # These are the global embedding and output chunks respectively.
        assert pretrain_hybrid.is_dataset_built_on_rank(vp_stage=rank)
        batch = pretrain_hybrid.get_batch(_iterator(), vp_stage=rank).get_batch()
        assert (batch['tokens'] is not None) == (rank == 0 or mtp)
        assert (batch['position_ids'] is not None) == (rank == 0 or mtp)
        assert (batch['labels'] is not None) == (rank == 1)
        assert (batch['loss_mask'] is not None) == (rank == 1)
        expected = next(_iterator())
        for key in ('tokens', 'position_ids', 'labels', 'loss_mask'):
            if batch[key] is not None:
                torch.testing.assert_close(batch[key].cpu(), expected[key], rtol=0, atol=0)
