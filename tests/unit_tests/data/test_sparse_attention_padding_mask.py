# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Packed metadata and padding-mask contracts for sparse attention."""

from types import SimpleNamespace

import numpy
import pytest
import torch

from megatron.core import parallel_state
from megatron.core import utils as core_utils
from megatron.core.context_parallel import utils as cp_utils
from megatron.core.datasets import data_schedule
from megatron.core.datasets.gpt_dataset import GPTDataset
from megatron.core.packed_seq_params import build_thd_padding_mask
from megatron.core.transformer.experimental_attention_variant.dsa_indexer_loss import (
    indexer_loss_from_target,
)
from megatron.core.transformer.experimental_attention_variant.dsa_masking import (
    extract_query_valid_rows_from_packed_seq_params,
)
from tests.unit_tests.test_utilities import Utils


def _packed_batch():
    return {
        "tokens": torch.arange(8, dtype=torch.int64).view(1, 8),
        "labels": torch.arange(8, dtype=torch.int64).view(1, 8),
        "loss_mask": torch.ones(1, 8),
        "position_ids": torch.arange(8, dtype=torch.int64).view(1, 8),
        "attention_mask": None,
        "cu_seqlens": torch.tensor([[0, 3, 5]], dtype=torch.int32),
        "cu_seqlens_padded": torch.tensor([[0, 4, 8]], dtype=torch.int32),
        "max_seqlen": torch.tensor(4, dtype=torch.int32),
    }


def test_padding_mask_uses_physical_offsets():
    logical = torch.tensor([0, 3, 5, 9], dtype=torch.int32)
    physical = torch.tensor([0, 4, 8, 12], dtype=torch.int32)

    padding = build_thd_padding_mask(logical, physical, 12)

    assert padding.dtype == torch.bool
    assert padding.shape == (12,)
    assert padding.tolist() == [
        False,
        False,
        False,
        True,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
    ]


def test_public_cp_batch_builder_propagates_real_token_mask(monkeypatch):
    cp_group = object()
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)

    result = cp_utils.get_batches_on_this_cp_rank(
        _packed_batch(),
        boundary_layout="zigzag",
        is_hybrid_cp=False,
        cp_group=cp_group,
        tokens_per_sample=8,
        physical_token_count=8,
    )

    packed = result.get_packed_seq_params()
    assert result.boundary_layout == "zigzag"
    assert packed.qkv_format == "thd"
    assert packed.cu_seqlens_q.tolist() == [0, 4, 8]
    assert packed.cu_seqlens_kv.data_ptr() == packed.cu_seqlens_q.data_ptr()
    assert packed.total_tokens == 8
    assert packed.real_token_mask_q.tolist() == [True, True, True, False, True, True, False, False]


@pytest.mark.parametrize(
    ("cp_rank", "expected_real_tokens"),
    [(0, [True, False, True, False]), (1, [True, True, True, False])],
)
def test_metadata_only_cp2_batch_builds_local_real_token_mask(
    monkeypatch, cp_rank, expected_real_tokens
):
    # Intermediate pipeline stages have document metadata but no token tensors.
    batch = _packed_batch()
    for key in ("tokens", "labels", "loss_mask", "position_ids"):
        batch[key] = None
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: cp_rank)

    result = cp_utils.get_batches_on_this_cp_rank(
        batch,
        boundary_layout="zigzag",
        is_hybrid_cp=False,
        cp_group=object(),
        use_per_sequence_balancing=True,
        tokens_per_sample=8,
        physical_token_count=8,
    )

    packed = result.get_packed_seq_params()
    local_batch = result.get_batch()
    assert local_batch["tokens"] is None
    assert packed.total_tokens == 8
    assert packed.cu_seqlens_q.tolist() == [0, 3, 5]
    assert packed.cu_seqlens_q_padded.tolist() == [0, 4, 8]
    assert packed.real_token_mask_q.tolist() == expected_real_tokens
    assert local_batch["padding_mask"].tolist() == [[not real for real in expected_real_tokens]]
    assert "padding_mask" not in batch


def test_contiguous_cp_metadata_keeps_shared_logical_boundaries():
    batch = _packed_batch()
    batch["padding_mask"] = build_thd_padding_mask(
        batch["cu_seqlens"].squeeze(0), batch["cu_seqlens_padded"].squeeze(0), 8
    ).unsqueeze(0)

    packed = cp_utils._build_packed_seq_params(
        batch, layout="contiguous", cp_size=2, tokens_per_sample=8
    )

    assert packed.cu_seqlens_q.tolist() == [0, 3, 5]
    assert packed.cu_seqlens_kv.data_ptr() == packed.cu_seqlens_q.data_ptr()
    assert packed.cu_seqlens_q_padded.tolist() == [0, 4, 8]
    assert packed.max_seqlen_q == packed.max_seqlen_kv == 4
    assert packed.real_token_mask_q.tolist() == [True, True, True, False, True, True, False, False]


def test_packed_builder_rejects_physical_count_mismatch(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)

    with pytest.raises(ValueError, match="physical_token_count disagrees"):
        cp_utils.get_batches_on_this_cp_rank(
            _packed_batch(),
            boundary_layout="zigzag",
            is_hybrid_cp=False,
            cp_group=object(),
            tokens_per_sample=8,
            physical_token_count=9,
        )


def test_packed_builder_rejects_invalid_alignment(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)

    with pytest.raises(ValueError, match="local_token_alignment must be a positive integer"):
        cp_utils.get_batches_on_this_cp_rank(
            _packed_batch(),
            boundary_layout="zigzag",
            is_hybrid_cp=False,
            cp_group=object(),
            local_token_alignment=0,
            physical_token_count=8,
        )


def test_aligned_cp_shards_exclude_new_storage_padding_from_losses(monkeypatch):
    """Alignment may add physical rows, but must neither duplicate tokens nor count padding."""
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    real_tokens = []
    for rank in range(2):
        monkeypatch.setattr(torch.distributed, "get_rank", lambda group: rank)
        result = cp_utils.get_batches_on_this_cp_rank(
            _packed_batch(),
            boundary_layout="zigzag",
            is_hybrid_cp=False,
            cp_group=object(),
            use_per_sequence_balancing=True,
            tokens_per_sample=8,
            physical_token_count=8,
            local_token_alignment=32,
        )
        batch = result.get_batch()
        valid = result.get_packed_seq_params().real_token_mask_q
        assert valid.numel() == 32
        assert torch.equal(valid, ~batch["padding_mask"].squeeze(0))
        assert torch.all(batch["loss_mask"].squeeze(0)[~valid] == 0)
        real_tokens.extend(batch["tokens"].squeeze(0)[valid].tolist())
    assert sorted(real_tokens) == [0, 1, 2, 4, 5]


@pytest.mark.parametrize("physical_token_count", [None, 0])
def test_padding_mask_preserves_empty_scheduler_input(physical_token_count):
    cu_seqlens = torch.tensor([0], dtype=torch.int32)
    padding = build_thd_padding_mask(cu_seqlens, cu_seqlens, physical_token_count)
    assert padding.shape == (0,)
    assert padding.dtype == torch.bool


def test_padding_mask_preserves_scheduler_no_documents():
    logical = torch.tensor([0], dtype=torch.int32)
    physical = torch.tensor([3], dtype=torch.int32)
    assert build_thd_padding_mask(logical, physical).tolist() == [True, True, True]


@pytest.mark.parametrize("producer", ["cp_batch", "scheduler"])
@pytest.mark.parametrize("calculate_per_token_loss", [False, True])
def test_packed_producer_masks_and_normalizes_shared_indexer_loss(
    monkeypatch, producer, calculate_per_token_loss
):
    """Physical boundaries must not replace five real queries with eight loss rows."""
    if producer == "cp_batch":
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)
        monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
        packed = cp_utils.get_batches_on_this_cp_rank(
            _packed_batch(),
            boundary_layout="zigzag",
            is_hybrid_cp=False,
            cp_group=object(),
            physical_token_count=8,
        ).get_packed_seq_params()
    else:
        group = SimpleNamespace(rank=lambda: 0, size=lambda: 1)
        monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: [0])
        monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
        monkeypatch.setattr(data_schedule, "broadcast_scalars", lambda values, *a, **kw: values)
        monkeypatch.setattr(data_schedule, "broadcast_tensor", lambda *a, **kw: None)
        batch = {
            key: value.squeeze(0) if value is not None and value.ndim > 0 else value
            for key, value in _packed_batch().items()
        }
        packed = data_schedule.get_batch_on_this_rank_for_sequence_packing(
            iter([batch]), pg_collection=SimpleNamespace(tp=group, pp=group, cp=group)
        )[5]

    # Both producers use physical attention boundaries after the TE workaround.
    assert packed.cu_seqlens_q.tolist() == [0, 4, 8]
    valid = extract_query_valid_rows_from_packed_seq_params(
        packed, b=1, sq=8, device=torch.device("cpu")
    )
    assert valid.tolist() == [[True, True, True, False, True, True, False, False]]
    target = torch.tensor([[[1.0, 0.0]]]).expand(1, 8, 2)
    predicted = torch.tensor(
        [
            [
                [0.8, 0.2],
                [0.8, 0.2],
                [0.8, 0.2],
                [0.01, 0.99],
                [0.8, 0.2],
                [0.8, 0.2],
                [0.01, 0.99],
                [0.01, 0.99],
            ]
        ]
    ).log()
    loss = indexer_loss_from_target(
        target,
        predicted,
        loss_coeff=0.1,
        query_valid_rows=valid,
        calculate_per_token_loss=calculate_per_token_loss,
    )
    expected = -torch.log(torch.tensor(0.8)) * 0.1
    if calculate_per_token_loss:
        expected *= 5
    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize("tokens_present", [False, True])
@pytest.mark.parametrize("rank,indices", [(0, [0, 3, 4, 7]), (1, [1, 2, 5, 6])])
def test_per_document_te_partition_keeps_mask_aligned(monkeypatch, tokens_present, rank, indices):
    batch = _packed_batch()
    original_mask = torch.tensor([[False, False, False, True, False, False, True, True]])
    batch["padding_mask"] = original_mask
    for key in ("tokens", "labels", "loss_mask", "position_ids"):
        if key != "tokens" or not tokens_present:
            batch[key] = None
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: rank)

    def te_partition(cu_seqlens, physical_count, cp_size, cp_rank):
        assert cu_seqlens.tolist() == [0, 4, 8]
        assert (physical_count, cp_size, cp_rank) == (8, 2, rank)
        return torch.tensor(indices, dtype=torch.int64)

    monkeypatch.setattr(
        core_utils, "tex", SimpleNamespace(thd_get_partitioned_indices=te_partition)
    )
    local = core_utils._get_batch_on_this_cp_rank_per_document_balancing(batch, object())
    assert local["padding_mask"].tolist() == [[original_mask[0, i].item() for i in indices]]
    if tokens_present:
        assert local["tokens"].tolist() == [indices]
    else:
        assert local["tokens"] is None


def test_padded_zigzag_preserves_explicit_validity_when_logical_lengths_were_lost(monkeypatch):
    batch = _packed_batch()
    batch["cu_seqlens"] = batch["cu_seqlens_padded"]
    batch["padding_mask"] = torch.tensor([[False, False, False, True, False, False, True, True]])
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    real_tokens = []
    for rank in range(2):
        monkeypatch.setattr(torch.distributed, "get_rank", lambda group: rank)
        result = cp_utils.get_batches_on_this_cp_rank(
            batch,
            boundary_layout="zigzag",
            is_hybrid_cp=False,
            cp_group=object(),
            use_per_sequence_balancing=True,
            physical_token_count=8,
            local_token_alignment=32,
        )
        local = result.get_batch()
        valid = result.get_packed_seq_params().real_token_mask_q
        real_tokens.extend(local["tokens"].squeeze(0)[valid].tolist())
        assert torch.all(local["loss_mask"].squeeze(0)[~valid] == 0)
    assert sorted(real_tokens) == [0, 1, 2, 4, 5]


@pytest.mark.parametrize("backend", ["reference", "cute"])
@pytest.mark.parametrize("metadata_only", [False, True])
@pytest.mark.parametrize("explicit_mask", [False, True])
def test_hybrid_entrypoint_applies_packed_validity_and_fp8_alignment_without_backend_gate(
    monkeypatch, backend, metadata_only, explicit_mask
):
    import pretrain_hybrid

    args = SimpleNamespace(
        sequence_packing_scheduler=None,
        context_parallel_size=2,
        sft=False,
        dataloader_inter_document_masking=True,
        create_attention_mask_in_dataloader=False,
        hybrid_context_parallel=False,
        micro_batch_size=1,
        seq_length=8,
        pipeline_model_parallel_size=3 if metadata_only else 1,
    )
    config = SimpleNamespace(
        dsa_gqa_backend=backend,
        pipeline_model_parallel_layout=None,
        mtp_num_layers=None,
        linear_cp_layout="zigzag",
        attention_cp_layout="zigzag",
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        fp8=True,
        fp8_recipe="mxfp8",
    )
    monkeypatch.setattr(pretrain_hybrid, "get_args", lambda: args)
    monkeypatch.setattr(pretrain_hybrid, "core_transformer_config_from_args", lambda args: config)
    monkeypatch.setattr(pretrain_hybrid, "mtp_on_this_rank_func", lambda **kw: False)
    monkeypatch.setattr(
        pretrain_hybrid, "is_first_or_last_pipeline_stage", lambda stage: not metadata_only
    )
    monkeypatch.setattr(pretrain_hybrid.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(pretrain_hybrid.mpu, "get_tensor_model_parallel_src_rank", lambda: 0)
    monkeypatch.setattr(pretrain_hybrid.mpu, "get_tensor_model_parallel_group", lambda: object())
    monkeypatch.setattr(pretrain_hybrid.mpu, "is_pipeline_first_stage", lambda: not metadata_only)
    monkeypatch.setattr(pretrain_hybrid.mpu, "is_pipeline_last_stage", lambda: not metadata_only)
    monkeypatch.setattr(pretrain_hybrid, "get_context_parallel_group", lambda: object())
    monkeypatch.setattr(pretrain_hybrid, "get_batch_on_this_tp_rank", lambda batch, **kw: batch)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kw: self)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    batch = _packed_batch()
    if explicit_mask:
        batch["cu_seqlens"] = batch["cu_seqlens_padded"]
        batch["padding_mask"] = torch.tensor(
            [[False, False, False, True, False, False, True, True]]
        )
    if metadata_only:
        for key in ("tokens", "labels", "loss_mask", "position_ids"):
            batch[key] = None

    result = pretrain_hybrid.get_batch(iter([batch]))

    valid = result.get_packed_seq_params().real_token_mask_q
    assert valid.numel() == 32
    # Rank 0 owns real tokens 0, 4 and 5; the additional GEMM rows are padding.
    assert valid.sum().item() == 3
    assert torch.equal(valid, ~result.get_batch()["padding_mask"].squeeze(0))
    assert (result.get_batch()["tokens"] is None) == metadata_only


def test_flatten_packed_batch_keeps_provided_mask_aligned():
    tokens = torch.arange(8).view(2, 4)
    batch = {
        "tokens": tokens,
        "padding_mask": torch.tensor([[False, False, False, True], [False, False, True, True]]),
        "cu_seqlens": torch.tensor([[0, 4], [0, 4]], dtype=torch.int32),
        "cu_seqlens_padded": torch.tensor([[0, 4], [0, 4]], dtype=torch.int32),
        "max_seqlen": torch.tensor([4, 4], dtype=torch.int32),
    }
    flat = core_utils.flatten_batch_for_packed_sequences(batch)
    assert flat["tokens"].tolist() == [list(range(8))]
    assert flat["padding_mask"].tolist() == [[False, False, False, True, False, False, True, True]]
    assert flat["cu_seqlens"].tolist() == [[0, 4, 8]]


@pytest.mark.parametrize("add_extra_token", [False, True])
@pytest.mark.parametrize("sample_index", [0, None])
def test_short_gpt_input_padding_reaches_shared_indexer_loss(
    monkeypatch, add_extra_token, sample_index
):
    dataset = object.__new__(GPTDataset)
    dataset.config = SimpleNamespace(
        add_extra_token_to_sequence=add_extra_token,
        tokenizer=SimpleNamespace(eod=2),
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=True,
        create_attention_mask=False,
        inter_document_masking=True,
        sequence_length=8,
    )
    dataset._pad_token_id = -1
    dataset.masks_and_position_ids_are_cacheable = False
    dataset.masks_and_position_ids_are_cached = False
    text = numpy.array([10, 11, 2, 13, 14] + [-1] * (3 + int(add_extra_token)))
    monkeypatch.setattr(
        dataset, "_query_document_sample_shuffle_indices", lambda idx: (text.copy(), None, [3, 2])
    )
    sample = dataset[sample_index]
    assert sample["padding_mask"].tolist() == [False] * 5 + [True] * 3
    # EOD has zero LM loss but remains a real query. A dummy index reuses the
    # same input validity even though its entire LM loss is disabled.
    assert sample["loss_mask"][2].item() == 0
    if sample_index is None:
        assert torch.all(sample["loss_mask"] == 0)
    batch = core_utils.flatten_batch_for_packed_sequences(
        {key: tensor.unsqueeze(0) for key, tensor in sample.items()}
    )
    batch["cu_seqlens_padded"] = None
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    packed = cp_utils.get_batches_on_this_cp_rank(
        batch,
        boundary_layout="zigzag",
        is_hybrid_cp=False,
        cp_group=object(),
        physical_token_count=8,
    ).get_packed_seq_params()
    valid = extract_query_valid_rows_from_packed_seq_params(
        packed, b=1, sq=8, device=torch.device("cpu")
    )
    assert valid.tolist() == [[True] * 5 + [False] * 3]
    target = torch.tensor([[[1.0, 0.0]]]).expand(1, 8, 2)
    probs = torch.tensor([[[0.8, 0.2]] * 5 + [[0.01, 0.99]] * 3])
    loss = indexer_loss_from_target(target, probs.log(), 0.1, query_valid_rows=valid)
    torch.testing.assert_close(loss, -torch.log(torch.tensor(0.8)) * 0.1)


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("present", [False, True])
@pytest.mark.parametrize("metadata_only", [False, True])
def test_optional_padding_mask_broadcasts_across_real_tp(tp_size, present, metadata_only):
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp_size)
    try:
        group = parallel_state.get_tensor_model_parallel_group()
        tp_rank = group.rank()
        expected_mask = torch.tensor(
            [[False, False, False, True, False, False, True, True]], device="cuda"
        )
        batch = {}
        if tp_rank == 0:
            batch = {
                key: tensor.cuda() if isinstance(tensor, torch.Tensor) else tensor
                for key, tensor in _packed_batch().items()
            }
            if present:
                # Entrypoints may leave optional dataset metadata on the CPU.
                batch["padding_mask"] = expected_mask.cpu()
        result = core_utils.get_batch_on_this_tp_rank(
            batch,
            has_cu_seqlens=True,
            is_hybrid_cp=False,
            create_attention_mask_in_dataloader=False,
            broadcast_src_rank=torch.distributed.get_process_group_ranks(group)[0],
            broadcast_group=group,
            cp_size=1,
            tp_rank=tp_rank,
            micro_batch_size=1,
            seq_length=8,
            mtp_on_this_rank=False,
            pipeline_model_parallel_size=3 if metadata_only else 1,
            is_pipeline_first_stage=not metadata_only,
            is_pipeline_last_stage=not metadata_only,
        )
        if present:
            assert result["padding_mask"].is_cuda
            torch.testing.assert_close(result["padding_mask"], expected_mask)
        else:
            assert result.get("padding_mask") is None
        if metadata_only:
            assert result["tokens"] is None
            assert result["labels"] is None
    finally:
        Utils.destroy_model_parallel()
