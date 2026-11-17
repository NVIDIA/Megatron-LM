# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Logical and physical packed-metadata contracts for sparse attention."""

from types import SimpleNamespace

import numpy
import pytest
import torch

from megatron.core import parallel_state
from megatron.core import utils as core_utils
from megatron.core.context_parallel import utils as cp_utils
from megatron.core.datasets import data_schedule
from megatron.core.datasets.gpt_dataset import GPTDataset
from megatron.core.transformer.experimental_attention_variant.dsa_indexer_loss import (
    indexer_loss_from_target,
)
from megatron.core.transformer.experimental_attention_variant.simplified_sparse_attention_cudnn import (
    build_attention_layout,
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


def _query_valid(monkeypatch, packed, local_rows, cp_size=1, cp_rank=0):
    """Exercise SSA's metadata consumer without a distributed process group."""
    group = SimpleNamespace(size=lambda: cp_size, rank=lambda: cp_rank)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, **kwargs: None)
    return build_attention_layout(torch.empty(local_rows, 16, 256), packed, group).query_valid


def test_padding_mask_uses_physical_offsets():
    logical = torch.tensor([0, 3, 5, 9], dtype=torch.int32)
    physical = torch.tensor([0, 4, 8, 12], dtype=torch.int32)

    padding = data_schedule._build_thd_padding_mask(logical, physical)

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


def test_public_cp_batch_builder_preserves_logical_lengths_and_source_padding(monkeypatch):
    cp_group = object()
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)

    result = cp_utils.get_batches_on_this_cp_rank(
        _packed_batch(),
        boundary_layout="zigzag",
        dynamic_cp=False,
        cp_group=cp_group,
        tokens_per_sample=8,
        preserve_logical_seqlens=True,
    )

    packed = result.get_packed_seq_params()
    assert result.boundary_layout == "zigzag"
    assert packed.qkv_format == "thd"
    assert packed.cu_seqlens_q.tolist() == [0, 3, 5]
    assert packed.cu_seqlens_q_padded.tolist() == [0, 4, 8]
    assert packed.cu_seqlens_kv.data_ptr() == packed.cu_seqlens_q.data_ptr()
    assert packed.total_tokens == 8
    valid = _query_valid(monkeypatch, packed, 8)
    assert valid.tolist() == [True, True, True, False, True, True, False, False]
    assert "padding_mask" not in result.get_batch()


@pytest.mark.parametrize(
    ("cp_rank", "expected_real_tokens"),
    [(0, [True, False, True, False]), (1, [True, True, True, False])],
)
def test_metadata_only_cp2_batch_preserves_real_document_lengths(
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
        dynamic_cp=False,
        cp_group=object(),
        use_per_sequence_balancing=True,
        tokens_per_sample=8,
    )

    packed = result.get_packed_seq_params()
    local_batch = result.get_batch()
    assert local_batch["tokens"] is None
    assert packed.total_tokens == 8
    assert packed.cu_seqlens_q.tolist() == [0, 3, 5]
    assert packed.cu_seqlens_q_padded.tolist() == [0, 4, 8]
    assert _query_valid(monkeypatch, packed, 4, 2, cp_rank).tolist() == expected_real_tokens
    assert "padding_mask" not in local_batch
    assert "padding_mask" not in batch


def test_contiguous_cp_metadata_keeps_shared_logical_boundaries():
    batch = _packed_batch()
    packed = cp_utils._build_packed_seq_params(
        batch, layout="contiguous", cp_size=2, tokens_per_sample=8
    )
    assert packed.cu_seqlens_q.tolist() == [0, 3, 5]
    assert packed.cu_seqlens_kv.data_ptr() == packed.cu_seqlens_q.data_ptr()
    assert packed.cu_seqlens_q_padded.tolist() == [0, 4, 8]
    assert packed.max_seqlen_q == packed.max_seqlen_kv == 4
    assert "padding_mask" not in batch


def test_packed_builder_rejects_invalid_alignment(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)

    with pytest.raises(ValueError, match="local_token_alignment must be a positive integer"):
        cp_utils.get_batches_on_this_cp_rank(
            _packed_batch(),
            boundary_layout="zigzag",
            dynamic_cp=False,
            cp_group=object(),
            local_token_alignment=0,
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
            dynamic_cp=False,
            cp_group=object(),
            use_per_sequence_balancing=True,
            tokens_per_sample=8,
            local_token_alignment=32,
        )
        batch = result.get_batch()
        valid = _query_valid(monkeypatch, result.get_packed_seq_params(), 32, 2, rank)
        assert valid.numel() == 32
        assert "padding_mask" not in batch
        assert torch.all(batch["loss_mask"].squeeze(0)[~valid] == 0)
        real_tokens.extend(batch["tokens"].squeeze(0)[valid].tolist())
    assert sorted(real_tokens) == [0, 1, 2, 4, 5]


@pytest.mark.parametrize("source_padded", [False, True])
def test_hybrid_attention_alignment_does_not_add_model_padding_mask(monkeypatch, source_padded):
    """Temporary attention rows are invalid in SSA and absent from the model boundary."""
    batch = _packed_batch()
    if not source_padded:
        batch["cu_seqlens"] = torch.tensor([[0, 3, 8]], dtype=torch.int32)
        batch["cu_seqlens_padded"] = None
        batch["max_seqlen"] = torch.tensor(5, dtype=torch.int32)
    batch["loss_mask"][0, 1] = 0  # A real prompt/image row must remain an attention input.
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    # Collective routing is independently covered by test_layout's real all-to-all round trips.
    monkeypatch.setattr(cp_utils, "build_thd_cp_layout_plan", lambda *args, **kwargs: object())
    real_tokens = []
    for rank in range(2):
        monkeypatch.setattr(torch.distributed, "get_rank", lambda group: rank)
        result = cp_utils.get_batches_on_this_cp_rank(
            batch,
            boundary_layout="contiguous",
            additional_layouts=("zigzag",),
            dynamic_cp=False,
            cp_group=object(),
            use_per_sequence_balancing=True,
            local_token_alignment=32,
            preserve_logical_seqlens=True,
        )
        boundary = result.get_batch()
        attention = result.get_batch("zigzag")
        assert "padding_mask" not in boundary
        assert "padding_mask" not in attention
        assert boundary["tokens"].numel() == 4
        torch.testing.assert_close(
            boundary["tokens"], batch["tokens"][:, rank * 4 : (rank + 1) * 4]
        )
        valid = _query_valid(monkeypatch, result.get_packed_seq_params("zigzag"), 32, 2, rank)
        assert valid.numel() == attention["tokens"].numel() == 32
        assert not valid.all()
        real_tokens.extend(attention["tokens"][0, valid].tolist())
    assert sorted(real_tokens) == ([0, 1, 2, 4, 5] if source_padded else list(range(8)))


def test_padding_mask_preserves_empty_scheduler_input():
    cu_seqlens = torch.tensor([0], dtype=torch.int32)
    padding = data_schedule._build_thd_padding_mask(cu_seqlens, cu_seqlens)
    assert padding.shape == (0,)
    assert padding.dtype == torch.bool


def test_padding_mask_preserves_scheduler_no_documents():
    logical = torch.tensor([0], dtype=torch.int32)
    physical = torch.tensor([3], dtype=torch.int32)
    assert data_schedule._build_thd_padding_mask(logical, physical).tolist() == [True, True, True]


@pytest.mark.parametrize("preserve_logical_seqlens", [False, True])
def test_unpadded_scheduler_preserves_router_mask_contract(monkeypatch, preserve_logical_seqlens):
    """The existing scheduler mask API is unchanged for SSA and dense callers."""
    group = SimpleNamespace(rank=lambda: 0, size=lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: [0])
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(data_schedule, "broadcast_scalars", lambda values, *a, **kw: values)
    monkeypatch.setattr(data_schedule, "broadcast_tensor", lambda *a, **kw: None)
    batch = {
        key: value.squeeze(0) if value is not None and value.ndim > 0 else value
        for key, value in _packed_batch().items()
    }
    batch["cu_seqlens"] = batch["cu_seqlens_padded"].clone()

    result = data_schedule.get_batch_on_this_rank_for_sequence_packing(
        iter([batch]),
        pg_collection=SimpleNamespace(tp=group, pp=group, cp=group, tp_cp=group),
        preserve_logical_seqlens=preserve_logical_seqlens,
    )

    torch.testing.assert_close(result[6], torch.zeros(1, 8, dtype=torch.bool))
    assert result[5].cu_seqlens_q.tolist() == [0, 4, 8]
    torch.testing.assert_close(result[0], torch.arange(8, dtype=torch.int64).view(1, 8))


@pytest.mark.parametrize("producer", ["cp_batch", "scheduler"])
@pytest.mark.parametrize("calculate_per_token_loss", [False, True])
@pytest.mark.parametrize("preserve_logical_seqlens", [False, True])
def test_packed_producer_masks_and_normalizes_shared_indexer_loss(
    monkeypatch, producer, calculate_per_token_loss, preserve_logical_seqlens
):
    """Physical boundaries must not replace five real queries with eight loss rows."""
    if producer == "cp_batch":
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)
        monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
        packed = cp_utils.get_batches_on_this_cp_rank(
            _packed_batch(),
            boundary_layout="zigzag",
            dynamic_cp=False,
            cp_group=object(),
            preserve_logical_seqlens=preserve_logical_seqlens,
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
            iter([batch]),
            pg_collection=SimpleNamespace(tp=group, pp=group, cp=group, tp_cp=group),
            preserve_logical_seqlens=preserve_logical_seqlens,
        )[5]

    # The scheduler keeps logical ends for every caller. The CP batch builder keeps them for
    # SSA (and MTP), while dense callers keep the TE physical-length workaround.
    retains_logical = preserve_logical_seqlens or producer == "scheduler"
    assert packed.cu_seqlens_q.tolist() == ([0, 3, 5] if retains_logical else [0, 4, 8])
    assert packed.cu_seqlens_q_padded.tolist() == [0, 4, 8]
    assert packed.cu_seqlens_kv is packed.cu_seqlens_q
    if not retains_logical:
        return
    valid = _query_valid(monkeypatch, packed, 8).unsqueeze(0)
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
def test_per_document_te_partition_keeps_metadata_aligned(
    monkeypatch, tokens_present, rank, indices
):
    batch = _packed_batch()
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
    assert local["cu_seqlens"].tolist() == [[0, 3, 5]]
    assert local["cu_seqlens_padded"].tolist() == [[0, 4, 8]]
    assert "padding_mask" not in local
    if tokens_present:
        assert local["tokens"].tolist() == [indices]
    else:
        assert local["tokens"] is None


@pytest.mark.parametrize("attention_variant", [None, "ssa"])
@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("metadata_only", [False, True])
def test_hybrid_entrypoint_preserves_packed_metadata_and_attention_alignment(
    monkeypatch, attention_variant, cp_size, metadata_only
):
    import pretrain_hybrid

    args = SimpleNamespace(
        sft=False,
        dataloader_inter_document_masking=True,
        create_attention_mask_in_dataloader=False,
        micro_batch_size=1,
        seq_length=8,
    )
    config = SimpleNamespace(
        sequence_packing_scheduler=None,
        context_parallel_size=cp_size,
        dynamic_context_parallel=False,
        experimental_attention_variant=attention_variant,
        pipeline_model_parallel_layout=None,
        pipeline_model_parallel_size=3 if metadata_only else 1,
        mtp_num_layers=None,
        linear_cp_layout="contiguous",
        attention_cp_layout="zigzag",
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        fp8=True,
        fp8_recipe="mxfp8",
    )
    monkeypatch.setattr(pretrain_hybrid, "get_args", lambda: args)
    monkeypatch.setattr(
        pretrain_hybrid,
        "get_run_config",
        lambda: SimpleNamespace(model=SimpleNamespace(transformer=config)),
    )
    monkeypatch.setattr(pretrain_hybrid, "mtp_on_this_rank_func", lambda **kw: False)
    monkeypatch.setattr(
        pretrain_hybrid, "is_first_or_last_pipeline_stage", lambda stage: not metadata_only
    )
    monkeypatch.setattr(pretrain_hybrid.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(pretrain_hybrid.mpu, "get_tensor_model_parallel_src_rank", lambda: 0)
    monkeypatch.setattr(pretrain_hybrid.mpu, "get_tensor_model_parallel_group", lambda: object())
    monkeypatch.setattr(pretrain_hybrid.mpu, "is_pipeline_first_stage", lambda: not metadata_only)
    monkeypatch.setattr(pretrain_hybrid.mpu, "is_pipeline_last_stage", lambda: not metadata_only)
    monkeypatch.setattr(pretrain_hybrid.mpu, "get_pipeline_model_parallel_group", lambda: object())
    monkeypatch.setattr(pretrain_hybrid, "get_context_parallel_group", lambda: object())
    monkeypatch.setattr(pretrain_hybrid, "get_batch_on_this_tp_rank", lambda batch, **kw: batch)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kw: self)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: cp_size)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    monkeypatch.setattr(cp_utils, "build_thd_cp_layout_plan", lambda *a, **kw: object())
    batch = _packed_batch()
    if metadata_only:
        for key in ("tokens", "labels", "loss_mask", "position_ids"):
            batch[key] = None

    result = pretrain_hybrid.get_batch(iter([batch]))
    packed = result.get_packed_seq_params("zigzag" if cp_size > 1 else None)
    assert "padding_mask" not in result.get_batch()
    if cp_size > 1:
        assert "padding_mask" not in result.get_batch("zigzag")
    logical_ends_retained = cp_size > 1 or attention_variant == "ssa"
    assert packed.cu_seqlens_q.tolist() == ([0, 3, 5] if logical_ends_retained else [0, 4, 8])
    if attention_variant == "ssa":
        valid = _query_valid(monkeypatch, packed, 8 if cp_size == 1 else 32, cp_size)
        assert valid.sum().item() == (5 if cp_size == 1 else 3)
    assert (result.get_batch()["tokens"] is None) == metadata_only


def test_flatten_packed_batch_keeps_logical_and_physical_offsets_separate():
    batch = {
        "tokens": torch.arange(8).view(2, 4),
        "cu_seqlens": torch.tensor([[0, 3], [0, 2]], dtype=torch.int32),
        "cu_seqlens_padded": torch.tensor([[0, 4], [0, 4]], dtype=torch.int32),
        "max_seqlen": torch.tensor([4, 4], dtype=torch.int32),
    }
    flat = core_utils.flatten_batch_for_packed_sequences(batch)
    assert flat["tokens"].tolist() == [list(range(8))]
    assert flat["cu_seqlens"].tolist() == [[0, 3, 5]]
    assert flat["cu_seqlens_padded"].tolist() == [[0, 4, 8]]
    assert "padding_mask" not in flat


def test_packed_cp1_none_group_preserves_real_token_metadata(monkeypatch):
    batch = _packed_batch()
    result = cp_utils.get_batches_on_this_cp_rank(
        batch,
        boundary_layout="zigzag",
        dynamic_cp=False,
        cp_group=None,
        preserve_logical_seqlens=True,
    )
    assert result.get_batch()["tokens"] is batch["tokens"]
    assert _query_valid(monkeypatch, result.get_packed_seq_params(), 8).tolist() == [
        True,
        True,
        True,
        False,
        True,
        True,
        False,
        False,
    ]


def test_aligned_packed_cp_preserves_decoder_features_and_mtp_mask(monkeypatch):
    batch = _packed_batch()
    for key in ("tokens", "labels", "loss_mask", "position_ids"):
        batch[key] = None
    features = torch.arange(24, dtype=torch.float32).view(1, 8, 3)
    batch["decoder_input"] = features
    batch["mtp_input_mask"] = torch.ones(1, 8, dtype=torch.bool)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    real_features = []
    for rank in range(2):
        monkeypatch.setattr(torch.distributed, "get_rank", lambda group: rank)
        result = cp_utils.get_batches_on_this_cp_rank(
            batch,
            boundary_layout="zigzag",
            dynamic_cp=False,
            cp_group=object(),
            use_per_sequence_balancing=True,
            local_token_alignment=32,
        )
        local = result.get_batch()
        valid = _query_valid(monkeypatch, result.get_packed_seq_params(), 32, 2, rank)
        assert local["decoder_input"].shape == (1, 32, 3)
        assert torch.all(local["decoder_input"][0, ~valid] == 0)
        assert torch.equal(local["mtp_input_mask"].squeeze(0), valid)
        real_features.extend(local["decoder_input"][0, valid].tolist())
    assert sorted(real_features) == features[0, [0, 1, 2, 4, 5]].tolist()


@pytest.mark.parametrize("add_extra_token", [False, True])
@pytest.mark.parametrize("sample_index", [0, None])
@pytest.mark.parametrize(
    "document_lengths,preserve_logical",
    [([3, 2], False), ([3, 2], True), ([4, 1], True), ([], True)],
)
def test_short_gpt_input_lengths_reach_shared_indexer_loss(
    monkeypatch, add_extra_token, sample_index, document_lengths, preserve_logical
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
        preserve_logical_seqlens=preserve_logical,
        sequence_length=8,
    )
    dataset._pad_token_id = -1
    dataset.masks_and_position_ids_are_cacheable = False
    dataset.masks_and_position_ids_are_cached = False
    real_count = sum(document_lengths)
    text = numpy.array(
        [10, 11, 2, 13, 14][:real_count] + [-1] * (8 - real_count + int(add_extra_token))
    )
    monkeypatch.setattr(
        dataset,
        "_query_document_sample_shuffle_indices",
        lambda idx: (text.copy(), None, list(document_lengths)),
    )
    sample = dataset[sample_index]
    assert "padding_mask" not in sample
    if not preserve_logical:
        assert "cu_seqlens_padded" not in sample
        assert sample["cu_seqlens"][:3].tolist() == [0, 3, 8]
        return
    expected_logical = [0, document_lengths[0], 5] if document_lengths else [0, 0]
    expected_physical = [0, document_lengths[0], 8] if document_lengths else [0, 8]
    assert sample["cu_seqlens"][: len(expected_logical)].tolist() == expected_logical
    assert sample["cu_seqlens_padded"][: len(expected_physical)].tolist() == expected_physical
    # EOD has zero LM loss but remains a real query. A dummy index reuses the
    # same input validity even though its entire LM loss is disabled.
    assert sample["loss_mask"][2].item() == 0
    if sample_index is None:
        assert torch.all(sample["loss_mask"] == 0)
    batch = core_utils.flatten_batch_for_packed_sequences(
        {key: tensor.unsqueeze(0) for key, tensor in sample.items()}
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    packed = cp_utils.get_batches_on_this_cp_rank(
        batch,
        boundary_layout="zigzag",
        dynamic_cp=False,
        cp_group=object(),
        preserve_logical_seqlens=True,
    ).get_packed_seq_params()
    valid = _query_valid(monkeypatch, packed, 8).unsqueeze(0)
    assert valid.tolist() == [[True] * real_count + [False] * (8 - real_count)]
    target = torch.tensor([[[1.0, 0.0]]]).expand(1, 8, 2)
    probs = torch.tensor([[[0.8, 0.2]] * 5 + [[0.01, 0.99]] * 3])
    loss = indexer_loss_from_target(target, probs.log(), 0.1, query_valid_rows=valid)
    expected_loss = -torch.log(torch.tensor(0.8)) * 0.1 if real_count else torch.tensor(0.0)
    torch.testing.assert_close(loss, expected_loss)


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("metadata_only", [False, True])
def test_packed_lengths_broadcast_across_real_tp(tp_size, padded, metadata_only):
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp_size)
    try:
        group = parallel_state.get_tensor_model_parallel_group()
        tp_rank = group.rank()
        batch = {}
        if tp_rank == 0:
            batch = {
                key: tensor.cuda() if isinstance(tensor, torch.Tensor) else tensor
                for key, tensor in _packed_batch().items()
            }
            if not padded:
                batch["cu_seqlens"] = batch["cu_seqlens_padded"].clone()
                batch["cu_seqlens_padded"] = None
        result = core_utils.get_batch_on_this_tp_rank(
            batch,
            has_cu_seqlens=True,
            dynamic_cp=False,
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
        assert "padding_mask" not in result
        assert result["cu_seqlens"].is_cuda
        assert result["cu_seqlens"].tolist() == [[0, 3, 5] if padded else [0, 4, 8]]
        if padded:
            assert result["cu_seqlens_padded"].is_cuda
            assert result["cu_seqlens_padded"].tolist() == [[0, 4, 8]]
        else:
            assert result["cu_seqlens_padded"] is None
        if metadata_only:
            assert result["tokens"] is None
            assert result["labels"] is None
    finally:
        Utils.destroy_model_parallel()
