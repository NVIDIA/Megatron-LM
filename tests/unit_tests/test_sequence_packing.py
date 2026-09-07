# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from megatron.core import parallel_state
from megatron.core.datasets.data_schedule import (
    DefaultDynamicCPScheduler,
    _build_thd_padding_mask,
    _sanitize_thd_padding_values,
    get_batch_on_this_rank_for_sequence_packing,
    wrap_data_iterator,
)
from megatron.core.datasets.data_schedule_utils import (
    create_data_iterator,
    next_hdp_group_packing_aware,
    reroute_samples_to_dcp_ranks,
)
from megatron.core.rerun_state_machine import RerunDataIterator
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
    padding_mask = torch.tensor([False, False, True, False, True])
    batch = {
        'tokens': torch.tensor([11, 12, -1, 21, -1], dtype=torch.int64),
        'labels': torch.tensor([12, 13, -1, 22, -1], dtype=torch.int64),
        'loss_mask': torch.ones(5, dtype=torch.float32),
        'position_ids': torch.tensor([0, 1, 2, 0, 1], dtype=torch.int64),
    }

    _sanitize_thd_padding_values(batch, padding_mask)

    assert torch.equal(batch['tokens'], torch.tensor([11, 12, 0, 21, 0]))
    assert torch.equal(batch['labels'], torch.tensor([12, 13, 0, 22, 0]))
    assert torch.equal(batch['loss_mask'], torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0]))
    assert torch.equal(batch['position_ids'], torch.tensor([0, 1, 0, 0, 0]))


@pytest.mark.parametrize(
    ("local_cp_size", "expected_layouts"),
    [(1, {"contiguous", "zigzag"}), (2, {"contiguous", "zigzag"})],
)
def test_scheduler_builds_runtime_cp_layout_views(local_cp_size, expected_layouts):
    class _Group:
        def __init__(self, size, rank=0):
            self._size = size
            self._rank = rank

        def size(self):
            return self._size

        def rank(self):
            return self._rank

    tp_group = _Group(1)
    pp_group = _Group(1)
    static_cp_group = _Group(4)
    runtime_cp_group = _Group(local_cp_size)
    pg_collection = SimpleNamespace(tp=tp_group, pp=pp_group, cp=static_cp_group)
    config = SimpleNamespace(
        linear_cp_layout="contiguous", attention_cp_layout="zigzag", sequence_parallel=False
    )
    tokens = torch.arange(8, dtype=torch.int64)
    batch = {
        "tokens": tokens.clone(),
        "labels": tokens.clone() + 100,
        "loss_mask": torch.ones(8, dtype=torch.float32),
        "position_ids": tokens.clone(),
        "cu_seqlens": torch.tensor([0, 8], dtype=torch.int32),
        "cu_seqlens_padded": torch.tensor([0, 8], dtype=torch.int32),
        "max_seqlen": torch.tensor([8], dtype=torch.int32),
        "local_cp_size": torch.tensor([local_cp_size], dtype=torch.int32),
    }

    with (
        patch("torch.cuda.current_device", return_value=torch.device("cpu")),
        patch("torch.distributed.get_process_group_ranks", return_value=[0]),
        patch("torch.distributed.get_world_size", side_effect=lambda group: group.size()),
        patch("torch.distributed.get_rank", side_effect=lambda group: group.rank()),
        patch("megatron.core.datasets.data_schedule.broadcast_tensor"),
        patch(
            "megatron.core.parallel_state.get_dynamic_data_context_parallel_groups",
            return_value=runtime_cp_group,
        ),
    ):
        cp_batch = get_batch_on_this_rank_for_sequence_packing(
            iter([batch]),
            dynamic_cp=True,
            pg_collection=pg_collection,
            config=config,
            return_context_parallel_batch=True,
        )

    assert cp_batch.boundary_layout == "contiguous"
    assert set(cp_batch.batches_by_layout) == expected_layouts
    assert (cp_batch.thd_plan is not None) == (local_cp_size > 1)
    for layout in expected_layouts:
        packed_seq_params = cp_batch.get_packed_seq_params(layout)
        assert packed_seq_params.cp_group is runtime_cp_group
        assert packed_seq_params.local_cp_size == local_cp_size

    expected_contiguous = tokens[: 8 // local_cp_size].view(1, -1)
    torch.testing.assert_close(cp_batch.get_batch("contiguous")["tokens"], expected_contiguous)
    if local_cp_size == 2:
        torch.testing.assert_close(
            cp_batch.get_batch("zigzag")["tokens"], tokens[[0, 1, 6, 7]].view(1, -1)
        )


def test_next_hdp_group_packing_aware_can_expand_short_sequence_group():
    micro_batches, leftovers, exec_times, sample_ids = next_hdp_group_packing_aware(
        [(0, 6144), (1, 2048)], total_gpus=2, max_seq_len_per_rank=4096
    )

    assert leftovers == []
    assert micro_batches == [[6144, 2048], [6144, 2048]]
    assert sample_ids == [[0, 1], [0, 1]]
    assert exec_times[0] == exec_times[1]


def test_next_hdp_group_packing_aware_fills_non_power_of_two_group():
    micro_batches, leftovers, exec_times, sample_ids = next_hdp_group_packing_aware(
        [(0, 50), (1, 50)], total_gpus=14, max_seq_len_per_rank=100
    )

    assert leftovers == []
    assert micro_batches == [[50, 50] for _ in range(14)]
    assert sample_ids == [[0, 1] for _ in range(14)]
    assert exec_times == [exec_times[0] for _ in range(14)]


def test_default_dynamic_cp_scheduler_uses_packing_aware_grouping():
    scheduler = DefaultDynamicCPScheduler(
        max_seqlen_per_dp_cp_rank=4096,
        cp_size=2,
        dp_size=1,
        microbatch_group_size_per_vp_stage=None,
    )

    groups = scheduler.get_groups_and_subsamples([(0, 6144), (1, 2048)])

    assert groups == [[[0, 1], [0, 1]]]


def test_dynamic_cp_group_sizes_partition_dpxcp_ranks():
    assert parallel_state.get_valid_dynamic_context_parallel_group_sizes(8) == [1, 2, 4, 8]
    assert parallel_state.get_valid_dynamic_context_parallel_group_sizes(6) == [1, 2, 6]


def test_default_dynamic_cp_scheduler_rejects_uncreated_min_group_size():
    with pytest.raises(ValueError, match="min_cp_size=3.*expected one of"):
        DefaultDynamicCPScheduler(
            max_seqlen_per_dp_cp_rank=4096,
            cp_size=8,
            dp_size=1,
            microbatch_group_size_per_vp_stage=None,
            min_cp_size=3,
        )


def test_vpp_packed_iterators_are_independent_and_mtp_gets_data():
    sample = {
        "tokens": torch.tensor([1, 2]),
        "labels": torch.tensor([2, 3]),
        "loss_mask": torch.ones(2),
        "position_ids": torch.arange(2),
        "cu_seqlens": torch.tensor([0, 2], dtype=torch.int32),
        "cu_seqlens_padded": torch.tensor([0, 2], dtype=torch.int32),
        "max_seqlen": torch.tensor(2, dtype=torch.int32),
        "local_cp_size": torch.tensor(1, dtype=torch.int32),
    }
    config = SimpleNamespace(virtual_pipeline_model_parallel_size=3)
    tp_group = SimpleNamespace(rank=lambda: 0)

    iterators = create_data_iterator(
        [sample], tp_group, config, vpp_needs_data=[False, True, False], is_dynamic_cp=True
    )

    metadata_batch = next(iterators[0])
    mtp_batch = next(iterators[1])
    other_metadata_batch = next(iterators[2])
    assert "tokens" not in metadata_batch
    assert "tokens" in mtp_batch and "labels" in mtp_batch
    assert "tokens" not in other_metadata_batch
    assert metadata_batch is not other_metadata_batch
    metadata_batch["max_seqlen"] = torch.tensor(99)
    assert other_metadata_batch["max_seqlen"].item() == 2


def test_scheduler_reroute_uses_dp_all_gather(monkeypatch):
    class _Group:
        def __init__(self, size, rank):
            self._size = size
            self._rank = rank

        def size(self):
            return self._size

        def rank(self):
            return self._rank

    dp_group = _Group(size=2, rank=0)
    dp_cp_group = _Group(size=2, rank=0)
    batch = [
        {
            'tokens': torch.tensor([10, 11]),
            'labels': torch.tensor([110, 111]),
            'loss_mask': torch.tensor([1.0, 0.0]),
            'position_ids': torch.tensor([0, 1]),
            'original_seq_len': torch.tensor([2], dtype=torch.int32),
            'padded_seq_len': torch.tensor([2], dtype=torch.int32),
        },
        {
            'tokens': torch.tensor([20]),
            'labels': torch.tensor([120]),
            'loss_mask': torch.tensor([1.0]),
            'position_ids': torch.tensor([0]),
            'original_seq_len': torch.tensor([1], dtype=torch.int32),
            'padded_seq_len': torch.tensor([1], dtype=torch.int32),
        },
    ]
    remote_inputs = iter(
        [
            torch.tensor([30, 31, 32, 33]),
            torch.tensor([130, 131, 132, 133]),
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([4, 0], dtype=torch.int32),
            torch.tensor([4, 0], dtype=torch.int32),
        ]
    )
    gather_groups = []

    def _all_gather_into_tensor(output, input_, group):
        gather_groups.append(group)
        remote = next(remote_inputs)
        assert input_.numel() == remote.numel()
        output.copy_(torch.cat([input_, remote]))

    monkeypatch.setattr(torch.cuda, 'current_device', lambda: torch.device('cpu'))
    monkeypatch.setattr(torch.distributed, 'all_gather_into_tensor', _all_gather_into_tensor)
    monkeypatch.setattr(
        torch.distributed,
        'all_to_all_single',
        lambda *args, **kwargs: pytest.fail('reroute must not use all_to_all_single'),
    )

    received = reroute_samples_to_dcp_ranks(
        batch=batch,
        global_ids_this_rank=torch.tensor([0, 1]),
        global_id_seqlens=[(0, 2), (1, 1), (2, 4)],
        sample_id_groups=[[[2], [0, 1]]],
        offsets=torch.tensor([0, 2, 3]),
        dp_group=dp_group,
        dp_cp_group=dp_cp_group,
    )

    assert list(received) == [2]
    assert torch.equal(received[2]['tokens'], torch.tensor([30, 31, 32, 33]))
    assert torch.equal(received[2]['labels'], torch.tensor([130, 131, 132, 133]))
    assert torch.equal(received[2]['loss_mask'], torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert torch.equal(received[2]['position_ids'], torch.tensor([0, 1, 2, 3]))
    assert received[2]['original_seq_len'].item() == 4
    assert received[2]['padded_seq_len'].item() == 4
    assert gather_groups == [dp_group] * 6


def test_scheduler_reroute_rejects_multimodal_metadata_in_text_contract():
    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    batch = [
        {
            'tokens': torch.tensor([10]),
            'original_seq_len': torch.tensor([1], dtype=torch.int32),
            'padded_seq_len': torch.tensor([1], dtype=torch.int32),
            'image_grid_thw': torch.tensor([[1, 2, 2]]),
        }
    ]

    with pytest.raises(
        AssertionError,
        match=r"supports only the text sample schema.*unsupported sample keys \['image_grid_thw'\]",
    ):
        reroute_samples_to_dcp_ranks(
            batch=batch,
            global_ids_this_rank=torch.tensor([0]),
            global_id_seqlens=[(0, 1)],
            sample_id_groups=[[[0]]],
            offsets=torch.tensor([0, 1]),
            dp_group=group,
            dp_cp_group=group,
        )


def test_scheduler_reroute_rejects_inconsistent_sample_keys():
    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    batch = [
        {
            'tokens': torch.tensor([10]),
            'original_seq_len': torch.tensor([1], dtype=torch.int32),
            'padded_seq_len': torch.tensor([1], dtype=torch.int32),
        },
        {'tokens': torch.tensor([20]), 'original_seq_len': torch.tensor([1], dtype=torch.int32)},
    ]

    with pytest.raises(AssertionError, match='Sample 1 keys'):
        reroute_samples_to_dcp_ranks(
            batch=batch,
            global_ids_this_rank=torch.tensor([0, 1]),
            global_id_seqlens=[(0, 1), (1, 1)],
            sample_id_groups=[[[0, 1]]],
            offsets=torch.tensor([0, 2]),
            dp_group=group,
            dp_cp_group=group,
        )


class MockVariableLengthSequencePackingDataIterator:
    """
    Mock data iterator for testing get_batch_on_this_rank_for_sequence_packing.

    Generates variable-length (THD format) packed sequences with deterministic
    data for verification across parallel ranks.
    """

    def __init__(
        self,
        total_seq_length: int,
        sequence_lengths: list,
        local_cp_size: int = None,
        device: str = "cuda",
        seed: int = 42,
    ):
        """
        Args:
            total_seq_length: Total length of packed sequences
            sequence_lengths: List of individual sequence lengths (variable-length).
                              If None, generates random variable lengths.
            device: Device to create tensors on
            seed: Random seed for reproducibility
        """
        self.total_seq_length = total_seq_length
        self.sequence_lengths = sequence_lengths
        self.local_cp_size = local_cp_size
        self.device = device
        self.seed = seed
        assert (
            sum(self.sequence_lengths) == total_seq_length
        ), f"Sequence lengths sum {sum(self.sequence_lengths)} != total {total_seq_length}"

    def __iter__(self):
        """Interface for the data iterator."""
        return self

    def __next__(self):
        """Generate a mock batch with variable-length THD format."""
        dev = self.device
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        tokens = torch.randint(0, 16384, (self.total_seq_length,), dtype=torch.int64, device=dev)

        # Create position_ids that reset for each sequence (THD format)
        position_ids = []
        for seq_len in self.sequence_lengths:
            position_ids.extend(range(seq_len))
        position_ids = torch.tensor(position_ids, dtype=torch.int64, device=dev)

        # Labels are tokens shifted by 1 for easy verification
        labels = tokens + 1

        # Loss mask: 1.0 for all positions except padding (none here)
        loss_mask = torch.ones(self.total_seq_length, dtype=torch.float32, device=dev)

        # Create cu_seqlens for variable-length packed sequences
        cu_seqlens = [0]
        for seq_len in self.sequence_lengths:
            cu_seqlens.append(cu_seqlens[-1] + seq_len)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=dev)
        cu_seqlens_padded = cu_seqlens.clone()

        max_seqlen = torch.tensor([max(self.sequence_lengths)], dtype=torch.int32, device=dev)

        batch = {
            "tokens": tokens,
            "position_ids": position_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "cu_seqlens": cu_seqlens,
            "cu_seqlens_padded": cu_seqlens_padded,
            "max_seqlen": max_seqlen,
        }

        if not (
            parallel_state.is_pipeline_first_stage(ignore_virtual=True)
            or parallel_state.is_pipeline_last_stage(ignore_virtual=True)
        ):
            batch["tokens"] = None
            batch["position_ids"] = None
            batch["labels"] = None
            batch["loss_mask"] = None

        if self.local_cp_size is not None:
            batch["local_cp_size"] = torch.tensor(
                [self.local_cp_size], dtype=torch.int32, device=dev
            )

        return batch


def _gather_tensor_from_tp_group(tensor):
    """Gather tensors from all TP ranks for comparison."""
    assert tensor is not None, "Tensor should not be None"
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(tp_size)]
    torch.distributed.all_gather(
        gathered, tensor, group=parallel_state.get_tensor_model_parallel_group()
    )
    return gathered


def _gather_tensor_from_all_ranks(tensor):
    """Gather tensors from all PP ranks for comparison."""
    assert tensor is not None, "Tensor should not be None"
    if type(tensor) is int:
        tensor = torch.tensor(tensor, dtype=torch.int32, device=torch.cuda.current_device())
    gathered = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered, tensor)
    return gathered


@pytest.mark.parametrize(
    ("tp", "pp", "cp", "dynamic_cp", "local_cp_size"),
    [
        (1, 1, 1, False, None),  # Basic case: no parallelism
        (2, 1, 1, False, None),  # Tensor parallel only
        (1, 2, 1, False, None),  # Pipeline parallel only
        (2, 2, 1, False, None),  # TP + PP
        (1, 1, 2, False, None),  # CP only
        (2, 1, 2, False, None),  # TP + CP
        (1, 2, 2, False, None),  # PP + CP
        (1, 4, 1, False, None),  # Has middle pp stage
        (1, 1, 4, True, 4),
        (1, 1, 4, True, 2),
        (1, 1, 4, True, 1),
        (2, 1, 2, True, 2),
    ],
)
def test_get_batch_on_this_rank_for_sequence_packing(tp, pp, cp, dynamic_cp, local_cp_size):
    """
    Test get_batch_on_this_rank_for_sequence_packing function with variable-length THD format.

    This test verifies:
    1. TP ranks: All ranks within a TP group receive identical data after broadcast
    2. PP ranks: Middle PP ranks have the same packed_seq_params as first/last stages
    3. CP ranks: Data is correctly partitioned with proper shape and values
    4. Variable-length (THD) format: Different sequence lengths are handled correctly
    """
    args = SimpleNamespace()
    args.tensor_model_parallel_size = tp
    args.pipeline_model_parallel_size = pp
    args.context_parallel_size = cp
    args.virtual_pipeline_model_parallel_size = None
    args.data_parallel_size = 8 // (tp * pp * cp)
    args.seq_length = 8192

    # Skip invalid configurations
    if args.data_parallel_size < 1:
        raise ValueError(f"Invalid config: tp={tp}, pp={pp}, cp={cp} exceeds world size 8")

    # Initialize model parallel
    Utils.initialize_model_parallel(
        tp,
        pp,
        None,
        context_parallel_size=cp,
        dynamic_context_parallel=dynamic_cp,
        min_dynamic_context_parallel_size=1,
    )

    try:
        # Create mock data iterator with variable-length sequences
        # Only TP rank 0 needs the iterator; other TP ranks pass None
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        if tp_rank == 0:
            # Use deterministic seed based on DP rank so same data within TP/PP/CP group
            dp_rank = parallel_state.get_data_parallel_rank()
            sequence_lengths = [1024, 2048, 512, 1536, 3072]
            assert (
                sum(sequence_lengths) == args.seq_length
            ), f"Sequence lengths sum {sum(sequence_lengths)} != total {args.seq_length}"
            data_iterator = iter(
                MockVariableLengthSequencePackingDataIterator(
                    total_seq_length=args.seq_length,
                    sequence_lengths=sequence_lengths,  # Variable lengths, sum=8192
                    local_cp_size=local_cp_size,
                    seed=42 + dp_rank,  # Same seed within PP/CP group
                )
            )
        else:
            # Non-TP-rank-0 ranks don't need the iterator
            data_iterator = None

        # Call the function under test
        result = get_batch_on_this_rank_for_sequence_packing(
            data_iterator=data_iterator,
            mtp_on_this_rank=False,
            vp_stage=None,
            dynamic_cp=dynamic_cp,
        )

        # Unpack the result. Scheduler THD always returns padding_mask.
        tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params, padding_mask = (
            result
        )

        # Get parallel state info
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        is_first_stage = parallel_state.is_pipeline_first_stage(ignore_virtual=True)
        is_last_stage = parallel_state.is_pipeline_last_stage(ignore_virtual=True)
        is_first_or_last = is_first_stage or is_last_stage

        assert padding_mask is not None
        assert padding_mask.dtype == torch.bool
        assert padding_mask.dim() == 2
        assert padding_mask.size(0) == 1
        assert not padding_mask.any(), "Mock data has no per-sequence padding."

        # =====================================================================
        # TEST 1: Verify data based on pipeline stage
        # =====================================================================
        if is_first_stage:
            assert tokens is not None, "First stage should have tokens"
            assert position_ids is not None, "First stage should have position_ids"
            assert tokens.dim() == 2, "Tokens should be 2D (batch, seq)"
            assert position_ids.dim() == 2, "Position IDs should be 2D (batch, seq)"
            assert tokens.size(0) == 1, "batch should be 1 in THD format"
            assert position_ids.size(0) == 1, "batch should be 1 in THD format"
        else:
            assert tokens is None, "Non-first stage should not have tokens"
            assert position_ids is None, "Non-first stage should not have position_ids"

        if is_last_stage:
            assert labels is not None, "Last stage should have labels"
            assert loss_mask is not None, "Last stage should have loss_mask"
            assert labels.dim() == 2, "Labels should be 2D (batch, seq)"
            assert loss_mask.dim() == 2, "Loss mask should be 2D (batch, seq)"
            assert labels.size(0) == 1, "batch should be 1 in THD format"
            assert loss_mask.size(0) == 1, "batch should be 1 in THD format"
        else:
            assert labels is None, "Non-last stage should not have labels"
            assert loss_mask is None, "Non-last stage should not have loss_mask"

        # =====================================================================
        # TEST 2: Verify all ranks have consistent packed_seq_params
        # =====================================================================
        assert packed_seq_params is not None
        assert packed_seq_params.qkv_format == "thd"
        if dynamic_cp:
            assert packed_seq_params.local_cp_size == local_cp_size
            assert packed_seq_params.cp_group is not None
            assert packed_seq_params.cp_group.size() == local_cp_size

        test_keys = [
            "cu_seqlens_q",
            "cu_seqlens_q_padded",
            "max_seqlen_q",
            "cu_seqlens_kv",
            "cu_seqlens_kv_padded",
            "max_seqlen_kv",
        ]
        for key in test_keys:
            tensor = getattr(packed_seq_params, key)
            assert tensor is not None
            gathered_tensor = _gather_tensor_from_all_ranks(tensor)
            for i in range(1, len(gathered_tensor)):
                assert torch.equal(
                    gathered_tensor[0], gathered_tensor[i]
                ), f"Rank 0 and rank {i} have different {key}"

        # =====================================================================
        # TEST 3: Verify TP ranks receive identical data after broadcast
        # =====================================================================
        if tp > 1:
            test_tensors = []
            if is_first_stage:
                test_tensors.extend([tokens, position_ids])
            if is_last_stage:
                test_tensors.extend([labels, loss_mask])

            for tensor in test_tensors:
                gathered_tensors = _gather_tensor_from_tp_group(tensor)
                for i in range(1, tp):
                    assert torch.equal(
                        gathered_tensors[0], gathered_tensors[i]
                    ), f"TP rank 0 and rank {i} have different data"

        # =====================================================================
        # TEST 4: Verify CP partitioning
        # =====================================================================
        effective_cp_size = local_cp_size if dynamic_cp else cp
        if effective_cp_size > 1:
            # With CP, the sequence should be partitioned
            expected_seq_len = args.seq_length // effective_cp_size

            if is_first_stage:
                actual_seq_len = tokens.shape[1]
                assert (
                    actual_seq_len == expected_seq_len
                ), f"CP partitioned tokens have wrong shape: {actual_seq_len} != {expected_seq_len}"

            # Verify labels only if all CP ranks are at last stage
            if is_last_stage:
                actual_seq_len = labels.shape[1]
                assert (
                    actual_seq_len == expected_seq_len
                ), f"CP partitioned labels have wrong shape: {actual_seq_len} != {expected_seq_len}"

    finally:
        Utils.destroy_model_parallel()
        unset_global_variables()


@pytest.mark.parametrize(
    ("tp", "pp", "cp", "vpp", "scheduler_type"),
    [
        (1, 1, 8, None, "dp_balanced"),
        (2, 1, 4, None, "dp_balanced"),
        (2, 4, 1, None, "dp_balanced"),
        (2, 2, 1, None, "dp_balanced"),
        (1, 4, 1, 4, "dp_balanced"),
        (1, 1, 8, None, "default_dynamic_cp"),
        (2, 1, 4, None, "default_dynamic_cp"),
        (1, 2, 4, None, "default_dynamic_cp"),
        (1, 4, 2, 4, "default_dynamic_cp"),
    ],
)
def test_wrap_dataloader(tp, pp, cp, vpp, scheduler_type):
    '''
    Test wrap_dataloader function with different scheduler types.
    '''
    args = SimpleNamespace()
    args.tensor_model_parallel_size = tp
    args.pipeline_model_parallel_size = pp
    args.context_parallel_size = cp
    args.virtual_pipeline_model_parallel_size = None
    args.data_parallel_size = 8 // (tp * pp * cp)
    args.seq_length = 8192
    args.max_seqlen_per_dp_cp_rank = 8192

    # Skip invalid configurations
    if args.data_parallel_size < 1:
        raise ValueError(f"Invalid config: tp={tp}, pp={pp}, cp={cp} exceeds world size 8")

    def _create_single_sample(seq_len):
        # hard code the padding size to 16
        pad_size = 16
        seq_len_padded = ((seq_len + pad_size - 1) // pad_size) * pad_size
        device = torch.device("cuda", torch.cuda.current_device())
        tokens = torch.randint(0, 128, (seq_len_padded,), dtype=torch.int64, device=device)
        labels = tokens + 1
        position_ids = torch.arange(seq_len_padded, dtype=torch.int64, device=device)
        loss_mask = torch.ones(seq_len_padded, dtype=torch.float32, device=device)
        loss_mask[0:seq_len] = 1
        loss_mask[seq_len:] = 0
        cu_seqlens = torch.tensor([0, seq_len_padded], dtype=torch.int32, device=device)

        return {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'cu_seqlens': cu_seqlens,
        }

    # Initialize model parallel
    dynamic_cp = scheduler_type == "default_dynamic_cp"
    Utils.initialize_model_parallel(
        tp,
        pp,
        vpp,
        context_parallel_size=cp,
        dynamic_context_parallel=dynamic_cp,
        min_dynamic_context_parallel_size=1,
    )

    global_batch_size = 64
    micro_batch_size = 1
    rng = random.Random(42)
    nums = [rng.randint(2048, args.seq_length) for _ in range(global_batch_size)]

    config = SimpleNamespace()
    config.max_seqlen_per_dp_cp_rank = args.max_seqlen_per_dp_cp_rank
    config.microbatch_group_size_per_vp_stage = pp
    config.virtual_pipeline_model_parallel_size = vpp
    config.sequence_packing_scheduler = scheduler_type
    config.min_dynamic_context_parallel_size = 1
    config.pipeline_model_parallel_layout = None
    config.mtp_num_layers = None

    dp_rank = parallel_state.get_data_parallel_rank()
    dp_size = parallel_state.get_data_parallel_world_size()

    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()

    is_pp_first = pp_rank == 0
    is_pp_last = pp_rank == pp - 1
    is_pp_first_or_last = is_pp_first or is_pp_last
    is_tp_first = tp_rank == 0

    num_micro_batches_old = global_batch_size // micro_batch_size // dp_size

    # Packed datasets are built independently on TP rank zero of every PP
    # stage. CP siblings must generate byte-identical samples.
    if is_tp_first:
        torch.manual_seed(42 + dp_rank)
        torch.cuda.manual_seed(42 + dp_rank)
        samples = [
            _create_single_sample(num)
            for num in nums[dp_rank * num_micro_batches_old : (dp_rank + 1) * num_micro_batches_old]
        ]
        data_iterator = RerunDataIterator(iter(samples))
    else:
        data_iterator = None

    if is_tp_first:
        if vpp is not None and vpp > 1:
            if is_pp_first:
                data_iterator = [data_iterator] + [None for _ in range(vpp - 1)]
            elif is_pp_last:
                data_iterator = [None for _ in range(vpp - 1)] + [data_iterator]
            else:
                data_iterator = [data_iterator] + [None for _ in range(vpp - 1)]
    try:
        # Call the function under test
        (
            new_data_iterator,
            num_micro_batches,
            num_total_tokens_this_global_batch,
            sequence_square_sum_this_global_batch,
        ) = wrap_data_iterator(data_iterator, config, num_micro_batches_old)

        # check the result
        assert type(num_micro_batches) is int
        assert (
            type(num_total_tokens_this_global_batch) is float
            or type(num_total_tokens_this_global_batch) is np.float32
        )
        assert (
            type(sequence_square_sum_this_global_batch) is float
            or type(sequence_square_sum_this_global_batch) is np.float32
        )

        def _check_batch(batch_all, batch_keys):
            for batch in batch_all:
                assert set(batch_keys) <= set(
                    batch.keys()
                ), f"batch keys: {set(batch.keys())} missing {set(batch_keys) - set(batch.keys())}"
                for key in batch_keys:
                    assert batch[key] is not None

        if is_tp_first:
            # CHECK KEYS
            batch_keys = ["cu_seqlens", "max_seqlen", "cu_seqlens_padded"]
            if dynamic_cp:
                batch_keys.append("local_cp_size")
            stage_data_keys = []
            if is_pp_first:
                stage_data_keys += ["tokens", "position_ids"]
            if is_pp_last:
                stage_data_keys += ["labels", "loss_mask"]
            if vpp is not None and vpp > 1:
                # check metadata for all stages (save batches to avoid re-consuming iterators)
                all_stage_batches = []
                for temp_data_iterator in new_data_iterator:
                    stage_batch = [next(temp_data_iterator) for _ in range(num_micro_batches)]
                    all_stage_batches.append(stage_batch)
                    _check_batch(stage_batch, batch_keys)

                # check for first or last stage on first or last pp rank
                if is_pp_first_or_last:
                    batch_all = all_stage_batches[0] if is_pp_first else all_stage_batches[-1]
                    _check_batch(batch_all, batch_keys + stage_data_keys)
            else:
                # non-VPP: single iterator
                batch_all = [next(new_data_iterator) for _ in range(num_micro_batches)]
                _check_batch(batch_all, batch_keys + stage_data_keys)

            # CHECK TOKEN SUM ON FIRST OR LAST PP RANK
            # Note: data_iterator is consumed by wrap_data_iterator, new_data_iterator is consumed above.
            # Use `samples` for before-wrap, reuse `batch_all` from the check above for after-wrap.
            if is_pp_first and (vpp is None or vpp <= 1):
                max_cp = parallel_state.get_context_parallel_world_size()
                dp_group = parallel_state.get_data_parallel_group()
                dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)

                # Before scheduling, CP siblings hold identical samples. Count
                # one lane over DP, then use an integer max-CP scale.
                token_sum_before = torch.tensor(0, dtype=torch.int64, device='cuda')
                for sample in samples:
                    token_sum_before += sample['tokens'].long().sum()
                torch.distributed.all_reduce(
                    token_sum_before, op=torch.distributed.ReduceOp.SUM, group=dp_group
                )
                token_sum_before *= max_cp

                # Dynamic samples are replicated on their local runtime CP
                # group, whereas fixed-CP samples remain replicated by CP lane.
                token_sum_after = torch.tensor(0, dtype=torch.int64, device='cuda')
                if dynamic_cp:
                    for batch in batch_all:
                        local_cp_size = int(batch['local_cp_size'].item())
                        token_sum_after += batch['tokens'].long().sum() * (max_cp // local_cp_size)
                    torch.distributed.all_reduce(
                        token_sum_after, op=torch.distributed.ReduceOp.SUM, group=dp_cp_group
                    )
                else:
                    for batch in batch_all:
                        token_sum_after += batch['tokens'].long().sum()
                    torch.distributed.all_reduce(
                        token_sum_after, op=torch.distributed.ReduceOp.SUM, group=dp_group
                    )
                    token_sum_after *= max_cp

                assert (
                    token_sum_before == token_sum_after
                ), f"Token sum mismatch: before={token_sum_before.item()}, after={token_sum_after.item()}"

        else:
            if vpp is not None and vpp > 1:
                assert type(new_data_iterator) is list and len(new_data_iterator) == vpp
                for data_iterator in new_data_iterator:
                    assert data_iterator is None
            else:
                assert new_data_iterator is None

    finally:
        Utils.destroy_model_parallel()
        unset_global_variables()
