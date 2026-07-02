# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.datasets.data_schedule import (
    DpBalancedScheduler,
    _get_vpp_stages_needing_data,
    wrap_data_iterator,
)
from megatron.core.datasets.data_schedule_utils import (
    _deserialize_packed_metadata,
    _get_group_local_ranks,
    _serialize_packed_metadata,
    _unpack_batch,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.core.transformer.enums import LayerType
from megatron.core.utils import (
    _build_thd_padding_mask,
    _sanitize_thd_padding_values,
    get_batch_on_this_tp_rank,
)
from megatron.training.global_vars import unset_global_variables
from tests.unit_tests.test_utilities import Utils


def test_scheduler_thd_padding_mask_from_cu_seqlens():
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
    cu_seqlens_padded = torch.tensor([0, 4, 8], dtype=torch.int32)

    padding_mask = _build_thd_padding_mask(cu_seqlens, cu_seqlens_padded)

    assert torch.equal(
        padding_mask, torch.tensor([False, False, False, True, False, False, True, True])
    )


def test_scheduler_sanitizes_thd_padding_values():
    padding_mask = torch.tensor([[False, False, True, False, True]])
    batch = {
        "tokens": torch.tensor([[11, 12, -1, 21, -1]], dtype=torch.int64),
        "labels": torch.tensor([[12, 13, -1, 22, -1]], dtype=torch.int64),
        "loss_mask": torch.ones((1, 5), dtype=torch.float32),
        "position_ids": torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.int64),
    }

    _sanitize_thd_padding_values(batch, padding_mask)

    assert torch.equal(batch["tokens"], torch.tensor([[11, 12, 0, 21, 0]]))
    assert torch.equal(batch["labels"], torch.tensor([[12, 13, 0, 22, 0]]))
    assert torch.equal(batch["loss_mask"], torch.tensor([[1.0, 1.0, 0.0, 1.0, 0.0]]))
    assert torch.equal(batch["position_ids"], torch.tensor([[0, 1, 0, 0, 0]]))


def test_canonical_tp_batch_broadcasts_dynamic_shape_padding_and_metadata(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    payloads = []

    def _record_broadcast(tensor, *_args, **_kwargs):
        payloads.append(tensor.clone())

    monkeypatch.setattr(torch.distributed, "broadcast", _record_broadcast)
    source_batch = {
        "tokens": torch.tensor([[11, 12, -1, 21, -1, -1]], dtype=torch.int64),
        "labels": torch.tensor([[12, 13, -1, 22, -1, -1]], dtype=torch.int64),
        "loss_mask": torch.ones((1, 6), dtype=torch.float32),
        "position_ids": torch.tensor([[0, 1, 2, 0, 1, 2]], dtype=torch.int64),
        "attention_mask": None,
        "cu_seqlens": torch.tensor([[0, 2, 3]], dtype=torch.int32),
        "cu_seqlens_padded": torch.tensor([[0, 3, 6]], dtype=torch.int32),
        "max_seqlen": torch.tensor([3], dtype=torch.int32),
    }
    common_kwargs = {
        "has_cu_seqlens": True,
        "is_hybrid_cp": False,
        "create_attention_mask_in_dataloader": False,
        "broadcast_src_rank": 0,
        "broadcast_group": object(),
        "cp_size": 1,
        "micro_batch_size": 4,
        "seq_length": 128,
        "mtp_on_this_rank": False,
        "pipeline_model_parallel_size": 1,
        "variable_seq_lengths": True,
        "has_padding_mask": True,
    }

    source = get_batch_on_this_tp_rank(source_batch, tp_rank=0, **common_kwargs)

    recorded_payloads = iter(payloads)

    def _replay_broadcast(tensor, *_args, **_kwargs):
        tensor.copy_(next(recorded_payloads))

    monkeypatch.setattr(torch.distributed, "broadcast", _replay_broadcast)
    received = get_batch_on_this_tp_rank({}, tp_rank=1, **common_kwargs)

    assert received["tokens"].shape == (1, 6)
    assert torch.equal(received["tokens"], source["tokens"])
    assert torch.equal(received["padding_mask"], source["padding_mask"])
    assert torch.equal(received["cu_seqlens"], source["cu_seqlens"])
    assert torch.equal(received["cu_seqlens_padded"], source["cu_seqlens_padded"])
    assert source["padding_mask"].tolist() == [[False, False, True, False, True, True]]


def test_dp_balanced_scheduler_can_split_group_zero_for_vpp_alignment():
    scheduler = DpBalancedScheduler(
        max_seqlen_per_dp_cp_rank=2, cp_size=1, dp_size=1, microbatch_group_size_per_vp_stage=2
    )

    groups = scheduler.get_groups_and_subsamples([(7, 1), (11, 1)])

    assert groups == [[[7]], [[11]]]


def test_dp_balanced_scheduler_rejects_impossible_alignment():
    scheduler = DpBalancedScheduler(
        max_seqlen_per_dp_cp_rank=2, cp_size=1, dp_size=1, microbatch_group_size_per_vp_stage=2
    )

    with pytest.raises(ValueError, match="Not enough samples"):
        scheduler.get_groups_and_subsamples([(7, 1)])


def test_dp_balanced_scheduler_rejects_oversized_sample():
    scheduler = DpBalancedScheduler(
        max_seqlen_per_dp_cp_rank=4, cp_size=2, dp_size=1, microbatch_group_size_per_vp_stage=None
    )

    with pytest.raises(ValueError, match="exceeding"):
        scheduler.get_groups_and_subsamples([(3, 9)])


def test_packed_metadata_round_trip_is_length_prefixed_and_integer_safe():
    large_offset = 2**24 + 17
    samples = [
        {
            "max_seqlen": torch.tensor(0, dtype=torch.int32),
            "cu_seqlens": torch.tensor([0], dtype=torch.int32),
            "cu_seqlens_padded": torch.tensor([0], dtype=torch.int32),
        },
        {
            "max_seqlen": torch.tensor(large_offset, dtype=torch.int32),
            "cu_seqlens": torch.tensor([0, large_offset], dtype=torch.int32),
            "cu_seqlens_padded": torch.tensor([0, large_offset], dtype=torch.int32),
        },
    ]

    payload = _serialize_packed_metadata(samples, torch.device("cpu"))
    restored = _deserialize_packed_metadata(payload)

    assert payload.dtype == torch.int64
    assert len(restored) == len(samples)
    for actual, expected in zip(restored, samples):
        assert torch.equal(actual["max_seqlen"], expected["max_seqlen"])
        assert torch.equal(actual["cu_seqlens"], expected["cu_seqlens"])
        assert torch.equal(actual["cu_seqlens_padded"], expected["cu_seqlens_padded"])


def test_packed_metadata_decoder_rejects_trailing_values():
    payload = _serialize_packed_metadata([], torch.device("cpu"))
    malformed = torch.cat((payload, torch.tensor([0], dtype=torch.int64)))

    with pytest.raises(AssertionError, match="decoder consumed"):
        _deserialize_packed_metadata(malformed)


def test_unpack_batch_accepts_variable_length_samples_with_collate_dim():
    batch = [
        {
            "tokens": torch.tensor([[1, 2, 0]]),
            "labels": torch.tensor([[2, 3, 0]]),
            "loss_mask": torch.tensor([[1.0, 1.0, 0.0]]),
            "position_ids": torch.tensor([[0, 1, 2]]),
            "padded_seq_len": torch.tensor([[3]], dtype=torch.int32),
            "original_seq_len": torch.tensor([[2]], dtype=torch.int32),
        }
    ]

    unpacked = _unpack_batch(batch)

    assert unpacked[0]["tokens"].shape == (3,)
    assert unpacked[0]["padded_seq_len"].shape == (1,)
    assert unpacked[0]["original_seq_len"].item() == 2


def test_unpack_batch_preserves_real_and_physical_packed_lengths():
    batch = [
        {
            "tokens": torch.tensor([[1, 2, 0, 3, 0, 0]]),
            "labels": torch.tensor([[2, 0, 0, 0, 0, 0]]),
            "loss_mask": torch.tensor([[1.0, 1.0, 0.0, 1.0, 0.0, 0.0]]),
            "position_ids": torch.tensor([[0, 1, 2, 0, 1, 2]]),
            "cu_seqlens": torch.tensor([[0, 3, 6, 6]], dtype=torch.int32),
            "cu_seqlens_original": torch.tensor([[0, 2, 3, 3]], dtype=torch.int32),
        }
    ]

    unpacked = _unpack_batch(batch)

    assert [sample["original_seq_len"].item() for sample in unpacked] == [2, 1]
    assert [sample["padded_seq_len"].item() for sample in unpacked] == [3, 3]
    assert [sample["tokens"].numel() for sample in unpacked] == [3, 3]


def test_group_local_rank_mapping_does_not_assume_global_rank_order(monkeypatch):
    subgroup = object()
    parent_group = object()
    ranks = {subgroup: [0, 4], parent_group: [0, 2, 4, 6]}
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", ranks.__getitem__)

    assert _get_group_local_ranks(subgroup, parent_group) == [0, 2]


def test_vpp_data_ownership_includes_custom_layout_mtp_stage():
    config = SimpleNamespace(
        virtual_pipeline_model_parallel_size=2,
        pipeline_model_parallel_layout=SimpleNamespace(
            layout=[
                [[LayerType.embedding], [LayerType.decoder]],
                [[LayerType.decoder], [LayerType.mtp]],
                [[LayerType.decoder], [LayerType.loss]],
            ]
        ),
        mtp_num_layers=1,
    )

    assert _get_vpp_stages_needing_data(config, pp_rank=0, pp_size=3) == [True, False]
    assert _get_vpp_stages_needing_data(config, pp_rank=1, pp_size=3) == [False, True]
    assert _get_vpp_stages_needing_data(config, pp_rank=2, pp_size=3) == [False, True]


def _process_groups() -> ProcessGroupCollection:
    return ProcessGroupCollection(
        tp=parallel_state.get_tensor_model_parallel_group(),
        pp=parallel_state.get_pipeline_model_parallel_group(),
        dp=parallel_state.get_data_parallel_group(
            with_context_parallel=False, partial_data_parallel=False
        ),
        dp_cp=parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        ),
    )


def _sample(sequence_length: int) -> dict[str, torch.Tensor]:
    device = torch.device("cuda", torch.cuda.current_device())
    tokens = torch.arange(sequence_length, dtype=torch.int64, device=device)
    return {
        "tokens": tokens,
        "labels": tokens + 1,
        "loss_mask": torch.ones(sequence_length, dtype=torch.float32, device=device),
        "position_ids": torch.arange(sequence_length, dtype=torch.int64, device=device),
        "original_seq_len": torch.tensor([sequence_length], dtype=torch.int32, device=device),
        "padded_seq_len": torch.tensor([sequence_length], dtype=torch.int32, device=device),
    }


@pytest.mark.parametrize(
    ("tp", "pp", "cp", "vpp", "middle_has_legacy_iterator", "mtp_pp_rank", "mtp_vp_stage"),
    [
        (1, 1, 8, None, False, None, None),
        (2, 1, 4, None, False, None, None),
        (2, 4, 1, None, True, None, None),
        (2, 2, 1, None, False, None, None),
        (1, 4, 1, 4, True, 1, 2),
    ],
)
def test_wrap_data_iterator(tp, pp, cp, vpp, middle_has_legacy_iterator, mtp_pp_rank, mtp_vp_stage):
    Utils.initialize_model_parallel(tp, pp, vpp, context_parallel_size=cp)
    try:
        pg_collection = _process_groups()
        dp_size = pg_collection.dp.size()
        global_sequence_lengths = [128 + 64 * (index % 4) for index in range(16)]
        input_microbatches = len(global_sequence_lengths) // dp_size
        dp_rank = pg_collection.dp.rank()
        local_lengths = global_sequence_lengths[
            dp_rank * input_microbatches : (dp_rank + 1) * input_microbatches
        ]
        samples = [_sample(sequence_length) for sequence_length in local_lengths]

        pp_rank = pg_collection.pp.rank()
        tp_rank = pg_collection.tp.rank()
        is_endpoint = pp_rank in (0, pp - 1)
        if tp_rank == 0 and (is_endpoint or middle_has_legacy_iterator):
            data_iterator = RerunDataIterator(iter(samples))
        else:
            data_iterator = None

        if vpp is not None and tp_rank == 0:
            # Packed datasets are currently built on every TP-zero virtual stage.
            data_iterator = [RerunDataIterator(iter(samples)) for _ in range(vpp)]

        pipeline_layout = None
        if mtp_pp_rank is not None:
            layout = [[[LayerType.decoder] for _ in range(vpp)] for _ in range(pp)]
            layout[mtp_pp_rank][mtp_vp_stage] = [LayerType.mtp]
            pipeline_layout = SimpleNamespace(layout=layout)

        config = SimpleNamespace(
            max_seqlen_per_dp_cp_rank=512,
            microbatch_group_size_per_vp_stage=2,
            virtual_pipeline_model_parallel_size=vpp,
            sequence_packing_scheduler="dp_balanced",
            pipeline_model_parallel_layout=pipeline_layout,
            mtp_num_layers=1 if mtp_pp_rank is not None else None,
        )
        (packed_iterator, output_microbatches, total_tokens, sequence_square_sum) = (
            wrap_data_iterator(
                data_iterator, config, input_microbatches, pg_collection=pg_collection
            )
        )

        assert isinstance(output_microbatches, int)
        assert total_tokens == float(sum(global_sequence_lengths))
        assert sequence_square_sum == float(sum(length**2 for length in global_sequence_lengths))

        if tp_rank != 0:
            if vpp is None:
                assert packed_iterator is None
            else:
                assert packed_iterator == [None] * vpp
            return

        expected_metadata = {"cu_seqlens", "cu_seqlens_padded", "max_seqlen"}

        def _consume(iterator):
            batches = [next(iterator) for _ in range(output_microbatches)]
            for batch in batches:
                assert expected_metadata <= batch.keys()
                assert batch["cu_seqlens"].ndim == 2
                assert batch["cu_seqlens_padded"].ndim == 2
            return batches

        if vpp is None:
            batches = _consume(packed_iterator)
            if is_endpoint:
                assert {"tokens", "labels", "loss_mask", "position_ids"} <= batches[0].keys()
            else:
                assert set(batches[0]) == expected_metadata
        else:
            assert len(packed_iterator) == vpp
            stage_batches = [_consume(iterator) for iterator in packed_iterator]
            full_stages = set()
            if pp_rank == 0:
                full_stages.add(0)
            if pp_rank == pp - 1:
                full_stages.add(vpp - 1)
            if pp_rank == mtp_pp_rank:
                full_stages.add(mtp_vp_stage)
            for stage, batches in enumerate(stage_batches):
                if stage in full_stages:
                    assert {"tokens", "labels", "loss_mask", "position_ids"} <= batches[0].keys()
                else:
                    assert set(batches[0]) == expected_metadata
    finally:
        Utils.destroy_model_parallel()
        unset_global_variables()
