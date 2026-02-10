# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from megatron.core import parallel_state
from megatron.core.datasets.data_schedule import (
    PackingScheduler,
    get_batch_on_this_rank_for_sequence_packing,
    scheduler_map,
    wrap_data_iterator,
)
from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.training.global_vars import unset_global_variables
from tests.unit_tests.test_utilities import Utils


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

        # Unpack the result
        tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = result

        # Get parallel state info
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        cp_rank = parallel_state.get_context_parallel_rank()
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
