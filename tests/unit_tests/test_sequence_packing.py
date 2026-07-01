# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from megatron.core import parallel_state
from megatron.core.datasets import data_schedule
from megatron.core.datasets.data_schedule import (
    DefaultDynamicCPScheduler,
    _build_thd_padding_mask,
    _get_scheduler_max_real_num_seqs,
    _sanitize_thd_padding_values,
    get_batch_on_this_rank_for_sequence_packing,
    wrap_data_iterator,
)
from megatron.core.datasets.data_schedule_utils import next_hdp_group_packing_aware
from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.training.global_vars import unset_global_variables
from tests.unit_tests.test_utilities import Utils


def test_scheduler_max_real_num_seqs_reserves_dummy_sequence():
    config = SimpleNamespace(
        thd_max_packed_sequences=32,
        pad_packed_seq_alignment="max",
        pad_packed_seq_by_appending_dummy_seq=True,
    )

    assert _get_scheduler_max_real_num_seqs(config) == 31

    config.pad_packed_seq_by_appending_dummy_seq = False
    assert _get_scheduler_max_real_num_seqs(config) == 32

    config.pad_packed_seq_alignment = None
    config.pad_packed_seq_by_appending_dummy_seq = True
    assert _get_scheduler_max_real_num_seqs(config) == 32


def test_scheduler_max_real_num_seqs_rejects_dummy_without_capacity():
    config = SimpleNamespace(
        thd_max_packed_sequences=1,
        pad_packed_seq_alignment="max",
        pad_packed_seq_by_appending_dummy_seq=True,
    )

    with pytest.raises(ValueError, match="includes that dummy sequence"):
        _get_scheduler_max_real_num_seqs(config)


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
        padded_sequence_lengths: list = None,
        local_cp_size: int = None,
        device: str = "cuda",
        seed: int = 42,
    ):
        """
        Args:
            total_seq_length: Total length of packed sequences
            sequence_lengths: List of individual sequence lengths (variable-length).
                              If None, generates random variable lengths.
            padded_sequence_lengths: Physical storage length for each sequence.
            device: Device to create tensors on
            seed: Random seed for reproducibility
        """
        self.total_seq_length = total_seq_length
        self.sequence_lengths = sequence_lengths
        self.padded_sequence_lengths = padded_sequence_lengths or sequence_lengths
        self.local_cp_size = local_cp_size
        self.device = device
        self.seed = seed
        assert len(self.sequence_lengths) == len(self.padded_sequence_lengths)
        assert all(
            real <= padded
            for real, padded in zip(self.sequence_lengths, self.padded_sequence_lengths)
        )
        assert sum(self.padded_sequence_lengths) == total_seq_length, (
            f"Padded sequence lengths sum {sum(self.padded_sequence_lengths)} "
            f"!= total {total_seq_length}"
        )

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
        for seq_len, padded_seq_len in zip(self.sequence_lengths, self.padded_sequence_lengths):
            position_ids.extend(range(seq_len))
            position_ids.extend([0] * (padded_seq_len - seq_len))
        position_ids = torch.tensor(position_ids, dtype=torch.int64, device=dev)

        # Labels are tokens shifted by 1 for easy verification
        labels = tokens + 1

        # Loss mask: 1.0 for valid tokens and 0.0 for physical padding.
        loss_mask = []
        for seq_len, padded_seq_len in zip(self.sequence_lengths, self.padded_sequence_lengths):
            loss_mask.extend([1.0] * seq_len)
            loss_mask.extend([0.0] * (padded_seq_len - seq_len))
        loss_mask = torch.tensor(loss_mask, dtype=torch.float32, device=dev)

        # Create cu_seqlens for variable-length packed sequences
        cu_seqlens = [0]
        for seq_len in self.sequence_lengths:
            cu_seqlens.append(cu_seqlens[-1] + seq_len)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=dev)
        cu_seqlens_padded = [0]
        for seq_len in self.padded_sequence_lengths:
            cu_seqlens_padded.append(cu_seqlens_padded[-1] + seq_len)
        cu_seqlens_padded = torch.tensor(cu_seqlens_padded, dtype=torch.int32, device=dev)

        max_seqlen = torch.tensor(
            [max(self.padded_sequence_lengths)], dtype=torch.int32, device=dev
        )

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


class _MockCPGroup:
    def __init__(self, size, rank):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def test_dsv4_thd_cp_slice_uses_static_partition_total():
    from megatron.core.datasets.data_schedule_utils import get_cp_slice_for_thd

    batch = {
        "tokens": torch.arange(12, dtype=torch.int64),
        "position_ids": torch.arange(12, dtype=torch.int64),
        "labels": torch.arange(100, 112, dtype=torch.int64),
        "loss_mask": torch.ones(12, dtype=torch.float32),
        "cu_seqlens": torch.tensor([0, 12], dtype=torch.int32),
        "cu_seqlens_padded": torch.tensor([0, 12], dtype=torch.int32),
        "max_seqlen": torch.tensor([12], dtype=torch.int32),
    }

    get_cp_slice_for_thd(
        batch,
        _MockCPGroup(size=4, rank=2),
        cp_partition_mode="contiguous",
        partition_total_tokens=16,
    )

    assert torch.equal(batch["tokens"], torch.arange(8, 12, dtype=torch.int64))
    assert torch.equal(batch["position_ids"], torch.arange(8, 12, dtype=torch.int64))
    assert torch.equal(batch["labels"], torch.arange(108, 112, dtype=torch.int64))
    assert torch.equal(batch["loss_mask"], torch.ones(4, dtype=torch.float32))
    assert torch.equal(batch["cu_seqlens"], torch.tensor([0, 12], dtype=torch.int32))
    assert torch.equal(batch["cu_seqlens_padded"], torch.tensor([0, 12], dtype=torch.int32))


@pytest.mark.parametrize(
    ("alignment", "cuda_graph_impl", "local_cp_size", "total_tokens", "local_target"),
    [(4, "none", 2, 10, 8), (8, "transformer_engine", 2, 10, 8)],
)
def test_dsv4_thd_dynamic_cp_pads_before_slicing(
    monkeypatch, alignment, cuda_graph_impl, local_cp_size, total_tokens, local_target
):
    """DSv4 padding must not change rank-local row origins after CP slicing."""
    cp_rank = local_cp_size - 1
    dynamic_cp_group = _MockCPGroup(size=local_cp_size, rank=cp_rank)
    pg_collection = SimpleNamespace(
        tp=_MockCPGroup(size=1, rank=0),
        pp=_MockCPGroup(size=1, rank=0),
        cp=_MockCPGroup(size=4, rank=cp_rank),
    )
    config = SimpleNamespace(
        cp_partition_mode="contiguous",
        pad_packed_seq_alignment=alignment,
        max_seqlen_per_dp_cp_rank=8,
        thd_max_packed_sequences=None,
        cuda_graph_impl=cuda_graph_impl,
        pad_packed_seq_by_appending_dummy_seq=True,
    )
    tokens = torch.arange(1, total_tokens + 1, dtype=torch.int64)
    batch = {
        "tokens": tokens.clone(),
        "position_ids": torch.arange(total_tokens, dtype=torch.int64),
        "labels": tokens + 100,
        "loss_mask": torch.ones(total_tokens, dtype=torch.float32),
        "cu_seqlens": torch.tensor([0, total_tokens], dtype=torch.int32),
        "cu_seqlens_padded": torch.tensor([0, total_tokens], dtype=torch.int32),
        "max_seqlen": torch.tensor([total_tokens], dtype=torch.int32),
        "local_cp_size": torch.tensor([local_cp_size], dtype=torch.int32),
    }
    pad_input_lengths = []
    real_pad_sequence_for_thd = data_schedule.pad_sequence_for_thd

    def record_padding(
        tokens, labels, loss_mask, position_ids, packed_seq_params, *, padding_mask, **_
    ):
        pad_input_lengths.append(tokens.shape[-1])
        assert tokens.shape[-1] == local_target
        assert padding_mask.shape[-1] == local_target
        result = real_pad_sequence_for_thd(
            tokens,
            labels,
            loss_mask,
            position_ids,
            packed_seq_params,
            padding_mask=padding_mask,
            **_,
        )
        assert result[0].shape[-1] == local_target
        assert result[-1].shape[-1] == local_target
        return result

    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda _: [0])
    monkeypatch.setattr(
        torch.distributed,
        "get_world_size",
        lambda group=None: group.size() if group is not None else 1,
    )
    monkeypatch.setattr(
        torch.distributed, "get_rank", lambda group=None: group.rank() if group is not None else 0
    )
    monkeypatch.setattr(data_schedule, "broadcast_tensor", lambda *_: None)
    monkeypatch.setattr(data_schedule, "pad_sequence_for_thd", record_padding)
    monkeypatch.setattr(
        parallel_state,
        "get_dynamic_data_context_parallel_groups",
        lambda group_size: dynamic_cp_group,
    )

    result = get_batch_on_this_rank_for_sequence_packing(
        data_iterator=iter([batch]), dynamic_cp=True, pg_collection=pg_collection, config=config
    )

    local_tokens, _, _, _, _, packed_seq_params, padding_mask = result
    global_start = cp_rank * local_target
    expected = torch.zeros(local_target, dtype=torch.int64)
    copied = max(0, min(total_tokens - global_start, local_target))
    if copied:
        expected[:copied] = torch.arange(
            global_start + 1, global_start + copied + 1, dtype=torch.int64
        )
    assert torch.equal(local_tokens.squeeze(0), expected)
    assert padding_mask.shape == (1, local_target)
    assert torch.equal(
        padding_mask.squeeze(0),
        torch.arange(global_start, global_start + local_target) >= total_tokens,
    )
    assert packed_seq_params.local_cp_size == local_cp_size
    assert packed_seq_params.cp_partition_mode == "contiguous"
    assert pad_input_lengths == [local_target]


def test_next_hdp_group_packing_aware_can_use_larger_cp_group_for_short_sequences():
    micro_batches, leftovers, exec_times, sample_ids = next_hdp_group_packing_aware(
        [(0, 6144), (1, 2048)], total_gpus=2, max_seq_len_per_rank=4096
    )

    assert leftovers == []
    assert micro_batches == [[6144, 2048], [6144, 2048]]
    assert sample_ids == [[0, 1], [0, 1]]
    assert exec_times[0] == exec_times[1]


def test_next_hdp_group_packing_aware_fills_non_power_of_two_dpxcp_group():
    micro_batches, leftovers, exec_times, sample_ids = next_hdp_group_packing_aware(
        [(0, 50), (1, 50)], total_gpus=14, max_seq_len_per_rank=100
    )

    assert leftovers == []
    assert micro_batches == [[50, 50] for _ in range(14)]
    assert sample_ids == [[0, 1] for _ in range(14)]
    assert exec_times == [exec_times[0] for _ in range(14)]


def test_default_dynamic_cp_scheduler_uses_packing_aware_grouping_by_default():
    scheduler = DefaultDynamicCPScheduler(
        max_seqlen_per_dp_cp_rank=4096,
        cp_size=2,
        dp_size=1,
        microbatch_group_size_per_vp_stage=None,
    )

    sample_id_groups = scheduler.get_groups_and_subsamples([(0, 6144), (1, 2048)])

    assert sample_id_groups == [[[0, 1], [0, 1]]]


def _gather_tensor_from_tp_group(tensor):
    """Gather tensors from all TP ranks for comparison."""
    assert tensor is not None, "Tensor should not be None"
    if type(tensor) is int:
        tensor = torch.tensor(tensor, dtype=torch.int32, device=torch.cuda.current_device())
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
        (1, 1, 4, True, 4),  # DCP: all CP ranks participate
        (1, 1, 4, True, 2),  # DCP: partial CP (2 out of 4)
        (1, 1, 4, True, 1),  # DCP: no CP splitting
        (2, 1, 4, True, 4),  # DCP + TP
        (1, 2, 4, True, 4),  # DCP + PP
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
    init_kwargs = dict(context_parallel_size=cp)
    if dynamic_cp:
        init_kwargs['dynamic_context_parallel'] = True
        init_kwargs['min_dynamic_context_parallel_size'] = 1
    Utils.initialize_model_parallel(tp, pp, None, **init_kwargs)

    try:
        # Create mock data iterator with variable-length sequences
        # Only TP rank 0 needs the iterator; other TP ranks pass None
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        if tp_rank == 0:
            # Use deterministic seed based on DP rank so same data within TP/PP/CP group
            dp_rank = parallel_state.get_data_parallel_rank()
            sequence_lengths = [1000, 2040, 500, 1500, 3000]
            padded_sequence_lengths = [1024, 2048, 512, 1536, 3072]
            assert sum(padded_sequence_lengths) == args.seq_length
            data_iterator = iter(
                MockVariableLengthSequencePackingDataIterator(
                    total_seq_length=args.seq_length,
                    sequence_lengths=sequence_lengths,
                    padded_sequence_lengths=padded_sequence_lengths,
                    local_cp_size=local_cp_size,
                    seed=42 + dp_rank,
                )
            )
        else:
            data_iterator = None

        # Call the function under test
        result = get_batch_on_this_rank_for_sequence_packing(
            data_iterator=data_iterator,
            mtp_on_this_rank=False,
            vp_stage=None,
            dynamic_cp=dynamic_cp,
        )

        # The helper returns a 7-tuple; scheduler THD always provides padding_mask.
        tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params, padding_mask = (
            result
        )
        assert padding_mask is not None
        assert padding_mask.dtype == torch.bool
        assert padding_mask.dim() == 2
        assert packed_seq_params is not None
        has_padding = padding_mask.any().to(torch.int32)
        effective_cp_group = (
            packed_seq_params.cp_group
            if packed_seq_params.cp_group is not None
            else parallel_state.get_context_parallel_group()
        )
        torch.distributed.all_reduce(
            has_padding, op=torch.distributed.ReduceOp.MAX, group=effective_cp_group
        )
        assert has_padding.item(), "Mock data intentionally has padding in each CP group."

        # Get parallel state info
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        is_first_stage = parallel_state.is_pipeline_first_stage(ignore_virtual=True)
        is_last_stage = parallel_state.is_pipeline_last_stage(ignore_virtual=True)
        is_first_or_last = is_first_stage or is_last_stage

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
        # TEST 2: Verify packed_seq_params consistency
        # =====================================================================
        assert packed_seq_params.qkv_format == "thd"
        assert packed_seq_params.pad_between_seqs is True
        assert not torch.equal(
            packed_seq_params.cu_seqlens_q, packed_seq_params.cu_seqlens_q_padded
        )
        assert packed_seq_params.cu_seqlens_q[-1] < packed_seq_params.cu_seqlens_q_padded[-1]

        test_keys = [
            "cu_seqlens_q",
            "cu_seqlens_q_padded",
            "max_seqlen_q",
            "cu_seqlens_kv",
            "cu_seqlens_kv_padded",
            "max_seqlen_kv",
        ]

        if dynamic_cp:
            assert packed_seq_params.local_cp_size == local_cp_size
            # For DCP, only TP ranks within the same CP group should match.
            # Different CP groups can have different packed_seq_params.
            if tp > 1:
                for key in test_keys:
                    tensor = getattr(packed_seq_params, key)
                    assert tensor is not None
                    gathered = _gather_tensor_from_tp_group(tensor)
                    for i in range(1, tp):
                        assert torch.equal(
                            gathered[0], gathered[i]
                        ), f"TP rank 0 and rank {i} have different {key}"
        else:
            # For THD, all ranks share the same packing metadata.
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
            test_tensors = [padding_mask]
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
        effective_cp = local_cp_size if dynamic_cp else cp
        if effective_cp is not None and effective_cp > 1:
            expected_seq_len = args.seq_length // effective_cp

            if is_first_stage:
                actual_seq_len = tokens.shape[1]
                assert (
                    actual_seq_len == expected_seq_len
                ), f"CP partitioned tokens have wrong shape: {actual_seq_len} != {expected_seq_len}"

            if is_last_stage:
                actual_seq_len = labels.shape[1]
                assert (
                    actual_seq_len == expected_seq_len
                ), f"CP partitioned labels have wrong shape: {actual_seq_len} != {expected_seq_len}"

            actual_seq_len = padding_mask.shape[1]
            assert (
                actual_seq_len == expected_seq_len
            ), f"CP partitioned padding_mask has wrong shape: {actual_seq_len} != {expected_seq_len}"

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
    is_dynamic_cp = scheduler_type == "default_dynamic_cp"

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
    init_kwargs = dict(context_parallel_size=cp)
    if is_dynamic_cp:
        init_kwargs['dynamic_context_parallel'] = True
        init_kwargs['min_dynamic_context_parallel_size'] = 1
    Utils.initialize_model_parallel(tp, pp, vpp, **init_kwargs)

    global_batch_size = 64
    micro_batch_size = 1
    rng = random.Random(42)
    nums = [rng.randint(2048, args.seq_length) for _ in range(global_batch_size)]  # 64 sequences

    config = SimpleNamespace()
    config.max_seqlen_per_dp_cp_rank = args.max_seqlen_per_dp_cp_rank
    config.microbatch_group_size_per_vp_stage = pp
    config.virtual_pipeline_model_parallel_size = vpp
    config.sequence_packing_scheduler = scheduler_type
    # wrap_data_iterator -> mtp_on_this_rank reads these two config fields.
    # A real TransformerConfig defaults both to None when MTP is unused, so
    # mirror that here (this test does not exercise MTP).
    config.pipeline_model_parallel_layout = None
    config.mtp_num_layers = None
    if is_dynamic_cp:
        config.min_dynamic_context_parallel_size = 1

    dp_rank = parallel_state.get_data_parallel_rank()
    dp_size = parallel_state.get_data_parallel_world_size()

    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()

    is_pp_first = pp_rank == 0
    is_pp_last = pp_rank == pp - 1
    is_pp_first_or_last = is_pp_first or is_pp_last
    is_tp_first = tp_rank == 0

    num_micro_batches_old = global_batch_size // micro_batch_size // dp_size

    # In packed-sequence mode is_dataset_built_on_rank returns True for every
    # PP stage on TP rank 0 (not just first/last), so all PP stages — including
    # middle ones — own a data_iterator and run the scheduler. Middle stages
    # later strip their samples down to metadata only.
    if is_tp_first:
        # Seed torch RNG so CP siblings produce identical token values
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
                # Middle PP stage: no VPP sub-stage needs full data, but the
                # scheduler still needs an iterator (slot 0) to derive the
                # microbatch count and per-stage metadata.
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
            if is_dynamic_cp:
                batch_keys.append("local_cp_size")
            # Per-stage data field stripping (see data_schedule.py): the first PP
            # stage keeps tokens/position_ids, the last PP stage keeps labels/
            # loss_mask. When pp==1 a single stage is both first and last, so it
            # keeps all four. Middle stages carry metadata only.
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

            # CHECK TOKEN SUM ON FIRST PP RANK
            # Note: data_iterator is consumed by wrap_data_iterator, new_data_iterator is consumed above.
            # Use `samples` for before-wrap, reuse `batch_all` from the check above for after-wrap.
            # Skip for VPP: microbatch alignment may pad/duplicate samples,
            # changing the total token count.
            # Only the first PP stage is checked: with per-stage data stripping the
            # last PP stage (when pp>1) keeps labels/loss_mask and no longer carries
            # 'tokens'. The dp/dp_cp all-reduce groups are disjoint per PP rank, so
            # running this on the first PP stage alone is collective-safe.
            if is_pp_first and (vpp is None or vpp <= 1):
                dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
                cp_size = parallel_state.get_context_parallel_world_size()

                # Count each sequence exactly once using int64 for bitwise comparison.
                # THD (dp_balanced): CP siblings hold identical packed data,
                #   so reduce across DP only (not CP) on both sides.
                # DCP: wrap_data_iterator returns packed samples before
                #   get_batch_on_this_rank_for_sequence_packing applies CP
                #   slicing, so local CP siblings still hold identical packed
                #   tokens here. Scale each rank by max_cp / local_cp, then
                #   reduce across DPxCP. A local-CP all-reduce here would
                #   overcount the pre-slice tokens.
                # Both sides multiply by max_cp so DCP (with varying local_cp)
                # can be normalized to the same integer scale without division.
                max_cp = cp_size
                dp_group = parallel_state.get_data_parallel_group()

                # Before wrap: CP siblings have identical samples.
                # Reduce across DP only to count each sequence once.
                token_sum_before = torch.tensor(0, dtype=torch.int64, device='cuda')
                for sample in samples:
                    token_sum_before += sample['tokens'].long().sum()
                torch.distributed.all_reduce(
                    token_sum_before, op=torch.distributed.ReduceOp.SUM, group=dp_group
                )
                token_sum_before *= max_cp

                # After wrap.
                token_sum_after = torch.tensor(0, dtype=torch.int64, device='cuda')
                if is_dynamic_cp:
                    for batch in batch_all:
                        mb_sum = batch['tokens'].long().sum().clone()
                        local_cp = batch['local_cp_size']
                        if isinstance(local_cp, torch.Tensor):
                            local_cp = local_cp.item()
                        mb_sum *= max_cp // local_cp
                        token_sum_after += mb_sum
                    torch.distributed.all_reduce(
                        token_sum_after, op=torch.distributed.ReduceOp.SUM, group=dp_cp_group
                    )
                else:
                    # THD: CP siblings hold identical packed data.
                    # Reduce across DP only (same as before).
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
