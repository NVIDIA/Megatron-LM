# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from megatron.core import parallel_state
from megatron.core.datasets.data_schedule import (
    _build_thd_padding_mask,
    _sanitize_thd_padding_values,
    get_batch_on_this_rank_for_sequence_packing,
    wrap_data_iterator,
)
from megatron.core.process_groups_config import ProcessGroupCollection
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


def test_packed_graph_tail_preserves_document_offsets():
    from megatron.core.datasets.data_schedule_utils import pad_packed_batch_before_cp_slice

    for owns_tokens in (False, True):
        physical = torch.tensor([0, 4, 12], dtype=torch.int32)
        logical = torch.tensor([0, 3, 8], dtype=torch.int32)
        mask = _build_thd_padding_mask(logical, physical)
        batch = dict(
            cu_seqlens=logical,
            cu_seqlens_padded=physical,
            max_seqlen=torch.tensor(8),
            padding_mask=mask,
            loss_mask=(~mask).float(),
        )
        if owns_tokens:
            batch["tokens"] = torch.arange(12)
        config = SimpleNamespace(
            pad_packed_seq_alignment="max",
            max_seqlen_per_dp_cp_rank=8,
            cuda_graph_thd_max_packed_sequences=4,
        )
        pad_packed_batch_before_cp_slice(batch, config, cp_size=2, tp_size=1)
        assert ("tokens" in batch) == owns_tokens
        assert batch["cu_seqlens"].tolist() == [0, 3, 8, 8, 8]
        assert batch["cu_seqlens_padded"].tolist() == [0, 4, 12, 16, 16]
        assert batch["padding_mask"][12:].all()
        assert batch["loss_mask"].sum() == 8
        assert batch["max_seqlen"] == 16
        previous = {key: value.clone() for key, value in batch.items()}
        pad_packed_batch_before_cp_slice(batch, config, cp_size=2, tp_size=1)
        for key in previous:
            torch.testing.assert_close(batch[key], previous[key])
        config.cuda_graph_thd_max_packed_sequences = 2
        with pytest.raises(ValueError, match="documents exceed"):
            pad_packed_batch_before_cp_slice(batch, config, cp_size=2, tp_size=1)


def test_packed_graph_scheduler_reserves_dummy_document():
    from megatron.core.datasets.data_schedule import DpBalancedScheduler

    scheduler = DpBalancedScheduler(16, 2, 1, None, max_num_seqs=2)
    groups = scheduler.get_groups_and_subsamples([(i, 4) for i in range(5)])
    assert [len(group[0]) for group in groups] == [2, 2, 1]


@pytest.mark.parametrize(
    "alignment,fp8,expected",
    [(None, None, [3, 2]), ("max", None, [2, 2, 1]), (None, "hybrid", [2, 2, 1])],
)
def test_graph_scheduler_accounts_for_main_padding(monkeypatch, alignment, fp8, expected):
    from megatron.core.datasets.data_schedule import DpBalancedScheduler

    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    groups = ProcessGroupCollection(tp=group, pp=group, cp=group, dp=group, dp_cp=group)
    config = SimpleNamespace(
        sequence_packing_scheduler="dp_balanced",
        pad_packed_seq_alignment=alignment,
        fp8=fp8,
        fp8_recipe="mxfp8",
        max_seqlen_per_dp_cp_rank=128,
        cuda_graph_thd_max_packed_sequences=3,
        virtual_pipeline_model_parallel_size=None,
    )
    observed = []

    def run(scheduler, *args):
        groups = scheduler.get_groups_and_subsamples([(i, 3) for i in range(5)])
        observed.extend(len(group[0]) for group in groups)
        return None, len(groups), 15, 45

    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(DpBalancedScheduler, "run", run)
    wrap_data_iterator(None, config, 5, groups)
    assert observed == expected


@pytest.mark.parametrize("cp_size,cp_rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("layout", ["tuple", "zigzag", "contiguous"])
@pytest.mark.parametrize("pp_rank", [0, 1, 2])
@pytest.mark.parametrize("alignment,metadata_capacity", [("max", 4), (None, 4), (None, None)])
def test_scheduler_loader_preserves_graph_logical_offsets(
    monkeypatch, alignment, metadata_capacity, cp_size, cp_rank, layout, pp_rank
):
    from megatron.core.datasets import data_schedule

    group = SimpleNamespace(rank=lambda: 0, size=lambda: 1)
    groups = ProcessGroupCollection(
        tp=group,
        pp=SimpleNamespace(rank=lambda: pp_rank, size=lambda: 3),
        cp=SimpleNamespace(rank=lambda: cp_rank, size=lambda: cp_size),
    )
    if cp_size == 1:
        monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda _: [0])
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: group.size())
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: group.rank())
    monkeypatch.setattr(data_schedule, "broadcast_scalars", lambda values, *args, **kwargs: values)
    monkeypatch.setattr(data_schedule, "broadcast_tensor", lambda *args, **kwargs: None)
    batch = {
        "tokens": torch.arange(12, dtype=torch.int64),
        "labels": torch.arange(12, dtype=torch.int64) + 100,
        "position_ids": torch.arange(12, dtype=torch.int64) + 200,
        "loss_mask": torch.ones(12),
        "cu_seqlens": torch.tensor([0, 3, 8], dtype=torch.int32),
        "cu_seqlens_padded": torch.tensor([0, 4, 12], dtype=torch.int32),
        "max_seqlen": 8,
    }
    if cp_size > 1:
        batch = {
            key: value.cuda() if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
    config = SimpleNamespace(
        pad_packed_seq_alignment=alignment,
        cuda_graph_thd_max_packed_sequences=metadata_capacity,
        max_seqlen_per_dp_cp_rank=16,
        linear_cp_layout="zigzag" if layout == "tuple" else layout,
        attention_cp_layout="zigzag" if layout == "tuple" else layout,
        sequence_parallel=False,
    )
    result = get_batch_on_this_rank_for_sequence_packing(
        iter([batch]),
        pg_collection=groups,
        config=config,
        return_context_parallel_batch=layout != "tuple",
    )
    if layout == "tuple":
        tokens, labels, loss_mask, _, positions, params, padding_mask = result
    else:
        assert result.boundary_layout == layout
        values = result.get_batch()
        tokens, labels, loss_mask, positions, padding_mask = (
            values[key] for key in ("tokens", "labels", "loss_mask", "position_ids", "padding_mask")
        )
        params = result.get_packed_seq_params()

    # Hand-build the expected source indices; do not call the sharder under test.
    end = 16 * cp_size if alignment == "max" else 12
    physical_offsets = [0, 4, 12, end, end] if metadata_capacity is not None else [0, 4, 12]
    if cp_size == 1 or layout == "contiguous":
        indices = list(range(cp_rank * (end // cp_size), (cp_rank + 1) * (end // cp_size)))
    else:
        indices = []
        for begin, stop in zip(physical_offsets[:-1], physical_offsets[1:]):
            width = (stop - begin) // (2 * cp_size)
            for chunk in (cp_rank, 2 * cp_size - cp_rank - 1):
                indices.extend(range(begin + chunk * width, begin + (chunk + 1) * width))
    expected_ids = torch.tensor(indices, device=padding_mask.device)
    valid = (expected_ids < 3) | ((expected_ids >= 4) & (expected_ids < 9))
    torch.testing.assert_close(padding_mask, (~valid).view(1, -1), rtol=0, atol=0)
    if pp_rank == 0:
        torch.testing.assert_close(
            tokens, expected_ids.masked_fill(~valid, 0).view(1, -1), rtol=0, atol=0
        )
        torch.testing.assert_close(
            positions, (expected_ids + 200).masked_fill(~valid, 0).view(1, -1), rtol=0, atol=0
        )
    else:
        assert tokens is None and positions is None
    if pp_rank == 2:
        torch.testing.assert_close(
            labels, (expected_ids + 100).masked_fill(~valid, 0).view(1, -1), rtol=0, atol=0
        )
        torch.testing.assert_close(loss_mask, valid.float().view(1, -1), rtol=0, atol=0)
    else:
        assert labels is None and loss_mask is None
    assert padding_mask.shape == (1, end // cp_size)
    if metadata_capacity is not None:
        assert params.cu_seqlens_q.tolist() == params.cu_seqlens_kv.tolist() == [0, 3, 8, 8, 8]
        assert params.cu_seqlens_q_padded.tolist() == physical_offsets
        assert torch.equal(params.cu_seqlens_q_padded, params.cu_seqlens_kv_padded)
        assert params.max_seqlen_q == params.max_seqlen_kv == end
        assert params.pad_between_seqs
    else:
        expected = [0, 3, 8] if layout == "contiguous" and cp_size > 1 else [0, 4, 12]
        assert params.cu_seqlens_q.tolist() == params.cu_seqlens_kv.tolist() == expected


def test_hybrid_scheduler_requests_layout_aware_batch(monkeypatch):
    import pretrain_hybrid

    config = SimpleNamespace(
        virtual_pipeline_model_parallel_size=None,
        pipeline_model_parallel_layout=None,
        mtp_num_layers=None,
    )
    monkeypatch.setattr(
        pretrain_hybrid,
        "get_args",
        lambda: SimpleNamespace(sequence_packing_scheduler="dp_balanced"),
    )
    monkeypatch.setattr(pretrain_hybrid, "core_transformer_config_from_args", lambda _: config)
    monkeypatch.setattr(pretrain_hybrid, "mtp_on_this_rank_func", lambda **_: False)
    result = object()
    calls = []

    def get_batch(iterator, **kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setattr(pretrain_hybrid, "get_batch_on_this_rank_for_sequence_packing", get_batch)
    assert pretrain_hybrid.get_batch(iter([])) is result
    assert calls[0]["return_context_parallel_batch"] is True
    assert calls[0]["config"] is config


@pytest.mark.parametrize("scheduler", [None, "dp_balanced"])
def test_hybrid_packed_stats_respect_scheduler_totals(monkeypatch, scheduler):
    from unittest.mock import MagicMock

    import pretrain_hybrid
    from megatron.core.context_parallel import ContextParallelBatch

    offsets = torch.tensor([[0, 3, 8]], dtype=torch.int32)
    cp_batch = ContextParallelBatch.from_single_layout("contiguous", {"cu_seqlens": offsets}, None)
    monkeypatch.setattr(
        pretrain_hybrid, "get_args", lambda: SimpleNamespace(sequence_packing_scheduler=scheduler)
    )
    monkeypatch.setattr(pretrain_hybrid, "get_timers", MagicMock())
    monkeypatch.setattr(pretrain_hybrid, "stimer", MagicMock())
    monkeypatch.setattr(pretrain_hybrid, "get_attr_wrapped_model", lambda *args: None)
    monkeypatch.setattr(pretrain_hybrid, "get_batch", lambda *args: cp_batch)
    update = MagicMock()
    monkeypatch.setattr(pretrain_hybrid, "update_seqlen_stats_from_cu_seqlens", update)
    model = MagicMock()

    output, _ = pretrain_hybrid.forward_step(None, model)

    assert output is model.return_value
    assert model.call_args.kwargs["cp_batch"] is cp_batch
    if scheduler is None:
        update.assert_called_once()
        torch.testing.assert_close(update.call_args.args[0], offsets.squeeze(0))
    else:
        update.assert_not_called()


@pytest.mark.parametrize("layout", ["zigzag", "contiguous"])
@pytest.mark.parametrize("tp,cp", [(1, 2), (2, 1)])
def test_scheduler_layout_batch_with_real_process_groups(layout, tp, cp):
    if Utils.world_size < tp * cp:
        pytest.skip("requires at least two distributed ranks")
    Utils.initialize_model_parallel(tp, 1, context_parallel_size=cp)
    try:
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        cp_rank = parallel_state.get_context_parallel_rank()
        config = SimpleNamespace(
            pad_packed_seq_alignment="max",
            cuda_graph_thd_max_packed_sequences=4,
            max_seqlen_per_dp_cp_rank=8,
            linear_cp_layout=layout,
            attention_cp_layout=layout,
            sequence_parallel=False,
        )
        batch = {
            "tokens": torch.arange(8, dtype=torch.int64, device="cuda"),
            "labels": torch.arange(8, dtype=torch.int64, device="cuda") + 100,
            "position_ids": torch.arange(8, dtype=torch.int64, device="cuda") + 200,
            "loss_mask": torch.ones(8, device="cuda"),
            "cu_seqlens": torch.tensor([0, 6], dtype=torch.int32, device="cuda"),
            "cu_seqlens_padded": torch.tensor([0, 8], dtype=torch.int32, device="cuda"),
            "max_seqlen": 8,
        }
        result = get_batch_on_this_rank_for_sequence_packing(
            iter([batch]) if tp_rank == 0 else None,
            config=config,
            return_context_parallel_batch=True,
        )
        values = result.get_batch()
        if cp == 1 or layout == "contiguous":
            indices = list(range(cp_rank * 8, (cp_rank + 1) * 8))
        else:
            indices = [0, 1, 6, 7, 8, 9, 14, 15] if cp_rank == 0 else [2, 3, 4, 5, 10, 11, 12, 13]
        ids = torch.tensor(indices, device="cuda").view(1, -1)
        valid = ids < 6
        for key, offset in (("tokens", 0), ("labels", 100), ("position_ids", 200)):
            torch.testing.assert_close(
                values[key], (ids + offset).masked_fill(~valid, 0), rtol=0, atol=0
            )
        torch.testing.assert_close(values["loss_mask"], valid.float(), rtol=0, atol=0)
        torch.testing.assert_close(values["padding_mask"], ~valid, rtol=0, atol=0)
        params = result.get_packed_seq_params()
        assert params.cu_seqlens_q.tolist() == [0, 6, 6, 6, 6]
        assert params.cu_seqlens_q_padded.tolist() == [0, 8, 8 * cp, 8 * cp, 8 * cp]
        assert params.max_seqlen_q == params.max_seqlen_kv == 8 * cp
        assert params.pad_between_seqs
    finally:
        Utils.destroy_model_parallel()


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
    ("tp", "pp", "cp"),
    [
        (1, 1, 1),  # Basic case: no parallelism
        (2, 1, 1),  # Tensor parallel only
        (1, 2, 1),  # Pipeline parallel only
        (2, 2, 1),  # TP + PP
        (1, 1, 2),  # CP only
        (2, 1, 2),  # TP + CP
        (1, 2, 2),  # PP + CP
        (1, 4, 1),  # Has middle pp stage
    ],
)
def test_get_batch_on_this_rank_for_sequence_packing(tp, pp, cp):
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
    Utils.initialize_model_parallel(tp, pp, None, context_parallel_size=cp)

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
                    seed=42 + dp_rank,  # Same seed within PP/CP group
                )
            )
        else:
            # Non-TP-rank-0 ranks don't need the iterator
            data_iterator = None

        # Call the function under test
        result = get_batch_on_this_rank_for_sequence_packing(
            data_iterator=data_iterator, mtp_on_this_rank=False, vp_stage=None
        )

        # Unpack the result. Scheduler THD always returns padding_mask.
        tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params, padding_mask = (
            result
        )

        # Get parallel state info
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        cp_rank = parallel_state.get_context_parallel_rank()
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
        if cp > 1:
            # With CP, the sequence should be partitioned
            expected_seq_len = args.seq_length // cp

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
    Utils.initialize_model_parallel(tp, pp, vpp, context_parallel_size=cp)

    global_batch_size = 64
    micro_batch_size = 1
    nums = [random.randint(2048, args.seq_length) for _ in range(global_batch_size)]  # 64 sequences

    config = SimpleNamespace()
    config.max_seqlen_per_dp_cp_rank = args.max_seqlen_per_dp_cp_rank
    config.microbatch_group_size_per_vp_stage = pp
    config.virtual_pipeline_model_parallel_size = vpp
    config.sequence_packing_scheduler = scheduler_type

    dp_rank = parallel_state.get_data_parallel_rank()
    dp_size = parallel_state.get_data_parallel_world_size()

    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()

    is_pp_first = pp_rank == 0
    is_pp_last = pp_rank == pp - 1
    is_pp_first_or_last = is_pp_first or is_pp_last
    is_tp_first = tp_rank == 0

    num_micro_batches_old = global_batch_size // micro_batch_size // dp_size

    if is_tp_first and (is_pp_first or is_pp_last):
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
                data_iterator = [None for _ in range(vpp)]
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
                    batch_keys += ["tokens", "position_ids", "labels", "loss_mask"]
                    _check_batch(batch_all, batch_keys)
            else:
                # non-VPP: single iterator
                batch_all = [next(new_data_iterator) for _ in range(num_micro_batches)]
                if is_pp_first_or_last:
                    batch_keys += ["tokens", "position_ids", "labels", "loss_mask"]
                _check_batch(batch_all, batch_keys)

            # CHECK TOKEN SUM ON FIRST OR LAST PP RANK
            # Note: data_iterator is consumed by wrap_data_iterator, new_data_iterator is consumed above.
            # Use `samples` for before-wrap, reuse `batch_all` from the check above for after-wrap.
            if is_pp_first_or_last:
                # Compute token sum before wrap
                token_sum_before = torch.tensor(0, dtype=torch.int64, device='cuda')
                for sample in samples:
                    token_sum_before += sample['tokens'].long().sum()

                # Compute token sum after wrap (batch_all already collected above with tokens)
                token_sum_after = torch.tensor(0, dtype=torch.int64, device='cuda')
                for batch in batch_all:
                    token_sum_after += batch['tokens'].long().sum()

                # Reduce sum across dp_cp group and verify equality
                dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=False)
                torch.distributed.all_reduce(
                    token_sum_before, op=torch.distributed.ReduceOp.SUM, group=dp_cp_group
                )
                torch.distributed.all_reduce(
                    token_sum_after, op=torch.distributed.ReduceOp.SUM, group=dp_cp_group
                )

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
