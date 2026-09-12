# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import pretrain_hybrid
from megatron.core import mpu
from megatron.core.context_parallel import get_batches_on_this_cp_rank
from megatron.core.context_parallel.utils import (
    _build_packed_seq_params,
    _get_batch_on_this_cp_rank_contiguous,
)
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import (
    _get_batch_on_this_cp_rank_per_sequence_balancing,
    flatten_batch_for_packed_sequences,
)
from megatron.training.arguments import parse_args, validate_args
from megatron.training.global_vars import destroy_global_vars, set_global_variables
from pretrain_hybrid import get_batch
from tests.unit_tests.test_utilities import Utils


def initialize_test_environment(
    tp_size: int,
    pp_size: int,
    cp_size: int,
    seq_length: int,
    micro_batch_size: int,
    global_batch_size: int = 1,
    sft: bool = False,
    hybrid_context_parallel: bool = False,
    max_seqlen_per_cp_rank: int = 1024,
    create_attention_mask: bool = False,
):
    destroy_global_vars()
    destroy_num_microbatches_calculator()

    sys.argv = ['test_get_batch.py']
    args = parse_args()
    args.seq_length = seq_length
    args.tensor_model_parallel_size = tp_size
    args.sequence_parallel = True if tp_size > 1 else False
    args.pipeline_model_parallel_size = pp_size
    args.context_parallel_size = cp_size
    args.hybrid_context_parallel = hybrid_context_parallel
    args.max_seqlen_per_cp_rank = max_seqlen_per_cp_rank
    args.sft = sft
    args.micro_batch_size = micro_batch_size
    args.create_attention_mask_in_dataloader = create_attention_mask
    args.global_batch_size = global_batch_size
    args.calculate_per_token_loss = True
    args.vocab_size = 1024
    args.tokenizer_type = "NullTokenizer"
    args.num_layers = 4
    args.hidden_size = 512
    args.num_attention_heads = 8
    args.max_position_embeddings = seq_length

    os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
    os.environ['NCCL_NVLS_ENABLE'] = '0'  # NOTE(asolergi-nv): Without this, NCCL crashes

    validate_args(args)
    set_global_variables(args, True)

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        context_parallel_size=cp_size,
        hybrid_context_parallel=hybrid_context_parallel,
    )
    return args


def _add_provenance_keys(batch):
    """Inject extra fields that real dataloader stacks add to per-sample dicts.

    BlendedDataset.__getitem__ prepends a ``dataset_id`` provenance field; other
    wrappers may add arbitrary metadata. get_batch must ignore these and return
    exactly len(BATCH_KEYS) values — without this, the call-site unpacking fails
    on tp_rank 0 (PR #4952). Applied inside every iterator builder so every
    parametrized test case in this file exercises the cardinality contract.
    """
    batch["dataset_id"] = torch.tensor([0], dtype=torch.int64)
    return batch


def create_sft_data_iterator(max_seq_length: int = 1024):
    """Create a mock SFT data iterator matching the old SFTDataset output after DataLoader collation.

    The old SFTDataset (megatron/training/datasets/sft_dataset.py) returns per-sample dicts with
    keys: tokens, labels, loss_mask, position_ids, cu_seqlens, max_seqlen — all padded to
    seq_length.  After PyTorch DataLoader default_collate, tensors get a leading batch dim of 1.
    """
    min_len = max(1, int(0.1 * max_seq_length))
    max_len = max(2, int(0.4 * max_seq_length))
    candidate_lengths = [torch.randint(min_len, max_len + 1, (1,)).item() for _ in range(10)]

    lengths = []
    total = 0
    for l in candidate_lengths:
        if total + l >= max_seq_length:
            break
        lengths.append(l)
        total += l

    num_real_tokens = sum(lengths)
    assert (
        num_real_tokens < max_seq_length
    ), f"Sum of lengths {num_real_tokens} is greater than max_seq_length {max_seq_length}"

    # Generate packed token sequence (num_real_tokens + 1 for labels shift)
    text = torch.randint(0, 10000, (num_real_tokens + 1,), dtype=torch.int64)

    # Pad to max_seq_length (mimics old SFTDataset padding)
    pad_len = max_seq_length - num_real_tokens
    pad_token = 0

    tokens = torch.cat([text[:-1], torch.full((pad_len,), pad_token, dtype=torch.int64)])
    labels = torch.cat([text[1:], torch.full((pad_len,), pad_token, dtype=torch.int64)])

    # Position IDs: per-segment positions, then padding positions
    position_ids = torch.cat([torch.arange(l, dtype=torch.int64) for l in lengths])
    position_ids = torch.cat(
        [
            position_ids,
            torch.arange(
                position_ids[-1].item() + 1,
                position_ids[-1].item() + 1 + pad_len,
                dtype=torch.int64,
            ),
        ]
    )

    # Loss mask: 1 for real tokens, 0 for padding
    loss_mask = torch.cat(
        [
            torch.ones(num_real_tokens, dtype=torch.float32),
            torch.zeros(pad_len, dtype=torch.float32),
        ]
    )

    # cu_seqlens: cumulative lengths ending at max_seq_length (last entry = seq_length after padding)
    cu_seqlens = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32),
            torch.cumsum(torch.tensor(lengths, dtype=torch.int64), dim=0).to(torch.int32),
        )
    )
    cu_seqlens[-1] = max_seq_length  # last entry is padded to seq_length

    # max_seqlen: max segment length
    seg_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seqlen = torch.tensor([seg_lengths.max().item()], dtype=torch.int32)

    # Add batch dimension to all per-sample tensors to mimic DataLoader default_collate.
    # The dataset emits cu_seqlens as 1-D (S+1,) and max_seqlen as 0-D; default_collate
    # stacks them with a leading batch dim of 1. get_batch_on_this_tp_rank's sender is
    # responsible for squeezing the batch dim of cu_seqlens before broadcast.
    batch = {
        "tokens": tokens.unsqueeze(0),
        "labels": labels.unsqueeze(0),
        "loss_mask": loss_mask.unsqueeze(0),
        "position_ids": position_ids.unsqueeze(0),
        "cu_seqlens": cu_seqlens.unsqueeze(0),
        "max_seqlen": max_seqlen,
    }
    return iter([_add_provenance_keys(batch)]), num_real_tokens


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("pp_size", [1, 2, 4])
@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("seq_length", [1024, 4096])
def test_sft_batch(tp_size, pp_size, cp_size, seq_length):
    if tp_size * pp_size * cp_size > torch.cuda.device_count():
        pytest.skip(
            f"Skipping test because tp_size * pp_size * cp_size > torch.cuda.device_count() ({tp_size * pp_size * cp_size} > {torch.cuda.device_count()})"
        )

    global_batch_size = int(os.environ.get("WORLD_SIZE", 1)) // (tp_size * pp_size * cp_size)
    args = initialize_test_environment(
        tp_size,
        pp_size,
        cp_size,
        seq_length,
        micro_batch_size=1,
        global_batch_size=global_batch_size,
        sft=True,
    )

    data_iterator = None
    num_real_tokens = 0
    if mpu.get_tensor_model_parallel_rank() == 0:
        data_iterator, num_real_tokens = create_sft_data_iterator(seq_length)

    cp_batch = get_batch(data_iterator)
    assert set(cp_batch.batches_by_layout) == {args.linear_cp_layout}
    batch = cp_batch.get_batch()
    (
        attention_mask,
        cu_seqlens,
        cu_seqlens_padded,
        hybrid_cp_group,
        labels,
        local_cp_size,
        loss_mask,
        max_seqlen,
        position_ids,
        tokens,
    ) = [batch[key] for key in pretrain_hybrid.BATCH_KEYS]

    is_first = mpu.is_pipeline_first_stage()
    is_last = mpu.is_pipeline_last_stage()
    seq_len_per_rank = seq_length // cp_size

    if pp_size == 1:
        # Single pipeline stage: all tensors present
        assert tokens is not None
        assert labels is not None
        assert loss_mask is not None
        assert position_ids is not None
        assert cu_seqlens is not None
        assert max_seqlen is not None
        assert attention_mask is None
        assert hybrid_cp_group is None
        assert local_cp_size is None
        assert cu_seqlens_padded is None

        assert tokens.shape == (
            1,
            seq_len_per_rank,
        ), f"Expected tokens shape (1, {seq_len_per_rank}), got {tokens.shape}"
        assert labels.shape == (
            1,
            seq_len_per_rank,
        ), f"Expected labels shape (1, {seq_len_per_rank}), got {labels.shape}"
        assert loss_mask.shape == (
            1,
            seq_len_per_rank,
        ), f"Expected loss_mask shape (1, {seq_len_per_rank}), got {loss_mask.shape}"
        assert position_ids.shape == (
            1,
            seq_len_per_rank,
        ), f"Expected position_ids shape (1, {seq_len_per_rank}), got {position_ids.shape}"

        assert tokens.dtype == torch.int64
        assert labels.dtype == torch.int64
        assert loss_mask.dtype == torch.float32
        assert position_ids.dtype == torch.int64

        assert cu_seqlens.dim() == 2
        assert cu_seqlens.shape[0] == 1
        assert cu_seqlens.dtype == torch.int32
        assert cu_seqlens[0, 0].item() == 0
        assert cu_seqlens[0, -1].item() == seq_length
        assert cu_seqlens.shape[1] >= 2

        assert max_seqlen.shape == (1,)
        assert max_seqlen.dtype == torch.int32
        assert 0 < max_seqlen.item() <= seq_length

        assert ((loss_mask == 0.0) | (loss_mask == 1.0)).all(), "loss_mask must be binary"

    elif is_first:
        # First pipeline stage: tokens, position_ids, and SFT metadata
        assert tokens is not None
        assert position_ids is not None
        assert labels is None
        assert loss_mask is None
        assert cu_seqlens is not None
        assert max_seqlen is not None
        assert attention_mask is None
        assert hybrid_cp_group is None
        assert local_cp_size is None
        assert cu_seqlens_padded is None

        assert tokens.shape == (
            1,
            seq_len_per_rank,
        ), f"Expected tokens shape (1, {seq_len_per_rank}), got {tokens.shape}"
        assert position_ids.shape == (
            1,
            seq_len_per_rank,
        ), f"Expected position_ids shape (1, {seq_len_per_rank}), got {position_ids.shape}"

        assert tokens.dtype == torch.int64
        assert position_ids.dtype == torch.int64

        assert cu_seqlens.dim() == 2
        assert cu_seqlens.shape[0] == 1
        assert cu_seqlens.dtype == torch.int32
        assert cu_seqlens[0, 0].item() == 0
        assert cu_seqlens[0, -1].item() == seq_length
        assert cu_seqlens.shape[1] >= 2

        assert max_seqlen.shape == (1,)
        assert max_seqlen.dtype == torch.int32
        assert 0 < max_seqlen.item() <= seq_length

    elif is_last:
        # Last pipeline stage: labels, loss_mask, and SFT metadata
        assert labels is not None
        assert loss_mask is not None
        assert tokens is None
        assert position_ids is None
        assert cu_seqlens is not None
        assert max_seqlen is not None
        assert attention_mask is None
        assert hybrid_cp_group is None
        assert local_cp_size is None
        assert cu_seqlens_padded is None

        assert labels.shape == (
            1,
            seq_len_per_rank,
        ), f"Expected labels shape (1, {seq_len_per_rank}), got {labels.shape}"
        assert loss_mask.shape == (
            1,
            seq_len_per_rank,
        ), f"Expected loss_mask shape (1, {seq_len_per_rank}), got {loss_mask.shape}"

        assert labels.dtype == torch.int64
        assert loss_mask.dtype == torch.float32

        assert cu_seqlens.dim() == 2
        assert cu_seqlens.shape[0] == 1
        assert cu_seqlens.dtype == torch.int32
        assert cu_seqlens[0, 0].item() == 0
        assert cu_seqlens[0, -1].item() == seq_length
        assert cu_seqlens.shape[1] >= 2

        assert max_seqlen.shape == (1,)
        assert max_seqlen.dtype == torch.int32
        assert 0 < max_seqlen.item() <= seq_length

        assert ((loss_mask == 0.0) | (loss_mask == 1.0)).all(), "loss_mask must be binary"

    else:
        # Intermediate SFT pipeline stages: only THD metadata for PackedSeqParams
        assert tokens is None
        assert labels is None
        assert loss_mask is None
        assert position_ids is None
        assert attention_mask is None
        assert hybrid_cp_group is None
        assert local_cp_size is None

        assert cu_seqlens is not None
        assert max_seqlen is not None
        assert cu_seqlens_padded is None

        assert cu_seqlens.dim() == 2
        assert cu_seqlens.shape[0] == 1
        assert cu_seqlens.dtype == torch.int32
        assert cu_seqlens[0, 0].item() == 0
        assert cu_seqlens[0, -1].item() == seq_length
        assert cu_seqlens.shape[1] >= 2

        assert max_seqlen.shape == (1,)
        assert max_seqlen.dtype == torch.int32
        assert 0 < max_seqlen.item() <= seq_length

    Utils.destroy_model_parallel()


@pytest.mark.parametrize("micro_batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_length", [16, 1024])
def test_flatten_batch_for_packed_sequences(micro_batch_size, seq_length):
    """Verify that flatten_batch_for_packed_sequences correctly merges
    cu_seqlens across samples and flattens sequence-dimension tensors.
    """
    # Each sample: tokens = range(seq_length), two documents per sample.
    tokens = (
        torch.arange(seq_length, dtype=torch.int64)
        .unsqueeze(0)
        .expand(micro_batch_size, -1)
        .clone()
    )
    labels = tokens.clone()
    loss_mask = torch.ones(micro_batch_size, seq_length, dtype=torch.float32)
    position_ids = (
        torch.arange(seq_length, dtype=torch.int64)
        .unsqueeze(0)
        .expand(micro_batch_size, -1)
        .clone()
    )
    half = seq_length // 2
    cu_seqlens = torch.tensor([[0, half, seq_length]] * micro_batch_size, dtype=torch.int32)
    max_seqlen = torch.tensor([half] * micro_batch_size, dtype=torch.int32)

    batch = {
        'tokens': tokens,
        'labels': labels,
        'loss_mask': loss_mask,
        'position_ids': position_ids,
        'cu_seqlens': cu_seqlens,
        'max_seqlen': max_seqlen,
    }
    result = flatten_batch_for_packed_sequences(batch)

    total_tokens = micro_batch_size * seq_length

    # Sequence-dimension tensors are flattened to (1, mbs * seq_length).
    assert result['tokens'].shape == (1, total_tokens)
    assert result['labels'].shape == (1, total_tokens)
    assert result['loss_mask'].shape == (1, total_tokens)
    assert result['position_ids'].shape == (1, total_tokens)

    # cu_seqlens is 2-D (1, N), starts at 0, ends at total_tokens.
    assert result['cu_seqlens'].dim() == 2
    assert result['cu_seqlens'].shape[0] == 1
    assert result['cu_seqlens'][0, 0].item() == 0
    assert result['cu_seqlens'][0, -1].item() == total_tokens

    # Each sample contributes 3 cu_seqlens entries; the first sample's
    # leading zero is kept while subsequent samples' leading zeros are
    # dropped, so total entries = 3 + (mbs - 1) * 2.
    expected_entries = 3 + (micro_batch_size - 1) * 2
    assert result['cu_seqlens'].shape[1] == expected_entries

    # Verify offsets: sample i's boundaries are offset by i * seq_length.
    for i in range(micro_batch_size):
        offset = i * seq_length
        if i == 0:
            assert result['cu_seqlens'][0, 0].item() == 0
            assert result['cu_seqlens'][0, 1].item() == half
            assert result['cu_seqlens'][0, 2].item() == seq_length
        else:
            base = 3 + (i - 1) * 2
            assert result['cu_seqlens'][0, base].item() == offset + half
            assert result['cu_seqlens'][0, base + 1].item() == offset + seq_length

    # max_seqlen is reduced to a single value.
    assert result['max_seqlen'].numel() == 1
    assert result['max_seqlen'].item() == half


@pytest.mark.parametrize("micro_batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_length", [16, 1024])
def test_flatten_batch_for_packed_sequences_intermediate_pp_stage(micro_batch_size, seq_length):
    """On intermediate PP stages, tokens/labels/loss_mask/position_ids are None.
    seq_length should be inferred from cu_seqlens[0, -1].
    """
    half = seq_length // 2
    cu_seqlens = torch.tensor([[0, half, seq_length]] * micro_batch_size, dtype=torch.int32)
    max_seqlen = torch.tensor([half] * micro_batch_size, dtype=torch.int32)

    batch = {
        'tokens': None,
        'labels': None,
        'loss_mask': None,
        'position_ids': None,
        'cu_seqlens': cu_seqlens,
        'max_seqlen': max_seqlen,
    }
    result = flatten_batch_for_packed_sequences(batch)

    total_tokens = micro_batch_size * seq_length

    # cu_seqlens is 2-D (1, N), starts at 0, ends at total_tokens.
    assert result['cu_seqlens'].dim() == 2
    assert result['cu_seqlens'].shape[0] == 1
    assert result['cu_seqlens'][0, 0].item() == 0
    assert result['cu_seqlens'][0, -1].item() == total_tokens

    expected_entries = 3 + (micro_batch_size - 1) * 2
    assert result['cu_seqlens'].shape[1] == expected_entries

    # max_seqlen is reduced to a single value.
    assert result['max_seqlen'].numel() == 1
    assert result['max_seqlen'].item() == half

    # Sequence-dimension tensors remain None.
    assert result['tokens'] is None
    assert result['labels'] is None
    assert result['loss_mask'] is None
    assert result['position_ids'] is None


@pytest.mark.parametrize("micro_batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_length", [16, 1024])
def test_flatten_batch_for_packed_sequences_padded_cu_seqlens(micro_batch_size, seq_length):
    """Verify that _strip_padding correctly removes trailing padding from
    cu_seqlens before merging. This matches the collation padding added by
    GPTDataset and SFTDataset.
    """
    half = seq_length // 2
    # Padded cu_seqlens: valid entries [0, half, seq_length] followed by
    # trailing copies of seq_length (matching dataset collation).
    padded_len = seq_length + 1
    cu_seqlens = torch.full((micro_batch_size, padded_len), seq_length, dtype=torch.int32)
    for i in range(micro_batch_size):
        cu_seqlens[i, 0] = 0
        cu_seqlens[i, 1] = half
        cu_seqlens[i, 2] = seq_length

    tokens = (
        torch.arange(seq_length, dtype=torch.int64)
        .unsqueeze(0)
        .expand(micro_batch_size, -1)
        .clone()
    )
    labels = tokens.clone()
    loss_mask = torch.ones(micro_batch_size, seq_length, dtype=torch.float32)
    position_ids = (
        torch.arange(seq_length, dtype=torch.int64)
        .unsqueeze(0)
        .expand(micro_batch_size, -1)
        .clone()
    )
    max_seqlen = torch.tensor([half] * micro_batch_size, dtype=torch.int32)

    batch = {
        'tokens': tokens,
        'labels': labels,
        'loss_mask': loss_mask,
        'position_ids': position_ids,
        'cu_seqlens': cu_seqlens,
        'max_seqlen': max_seqlen,
    }
    result = flatten_batch_for_packed_sequences(batch)

    total_tokens = micro_batch_size * seq_length

    # After stripping padding and merging, result should be identical to the
    # unpadded case: 2-D (1, N) with correct offsets.
    assert result['cu_seqlens'].dim() == 2
    assert result['cu_seqlens'].shape[0] == 1
    assert result['cu_seqlens'][0, 0].item() == 0
    assert result['cu_seqlens'][0, -1].item() == total_tokens

    expected_entries = 3 + (micro_batch_size - 1) * 2
    assert result['cu_seqlens'].shape[1] == expected_entries


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("pp_size", [1, 2, 4])
@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("seq_length", [1024])
def test_inter_document_masking_batch(tp_size, pp_size, cp_size, seq_length):
    if tp_size * pp_size * cp_size > torch.cuda.device_count():
        pytest.skip(
            f"Skipping test because tp_size * pp_size * cp_size > torch.cuda.device_count() "
            f"({tp_size * pp_size * cp_size} > {torch.cuda.device_count()})"
        )

    global_batch_size = int(os.environ.get("WORLD_SIZE", 1)) // (tp_size * pp_size * cp_size)
    if global_batch_size < 1:
        pytest.skip("Not enough ranks for the requested parallelism configuration")
    args = initialize_test_environment(
        tp_size,
        pp_size,
        cp_size,
        seq_length,
        micro_batch_size=1,
        global_batch_size=global_batch_size,
        sft=False,
    )
    args.dataloader_inter_document_masking = True

    data_iterator = None
    if mpu.get_tensor_model_parallel_rank() == 0:
        data_iterator, _ = create_sft_data_iterator(seq_length)

    cp_batch = get_batch(data_iterator)
    batch = cp_batch.get_batch()
    (
        attention_mask,
        cu_seqlens,
        cu_seqlens_padded,
        hybrid_cp_group,
        labels,
        local_cp_size,
        loss_mask,
        max_seqlen,
        position_ids,
        tokens,
    ) = [batch[key] for key in pretrain_hybrid.BATCH_KEYS]

    is_first = mpu.is_pipeline_first_stage()
    is_last = mpu.is_pipeline_last_stage()

    # Per-sequence zigzag balancing may pad each sequence to 2 * CP alignment, so the physical
    # shard length comes from the padded metadata rather than the original sample length.
    if cp_size > 1:
        assert cu_seqlens_padded is not None
        partitioned_seq_length = cu_seqlens_padded[0, -1].item() // cp_size
    else:
        assert cu_seqlens_padded is None
        partitioned_seq_length = seq_length

    if pp_size == 1:
        assert tokens is not None
        assert labels is not None
        assert loss_mask is not None
        assert position_ids is not None
        assert cu_seqlens is not None
        assert max_seqlen is not None
        assert attention_mask is None

        assert tokens.shape[1] == partitioned_seq_length
        assert labels.shape[1] == partitioned_seq_length
        assert loss_mask.shape[1] == partitioned_seq_length
        assert position_ids.shape[1] == partitioned_seq_length

        assert cu_seqlens.dim() == 2
        assert cu_seqlens.shape[0] == 1
        assert cu_seqlens.dtype == torch.int32
        assert cu_seqlens[0, 0].item() == 0
        assert cu_seqlens[0, -1].item() == seq_length
        assert cu_seqlens.shape[1] >= 2

        assert max_seqlen.shape == (1,)
        assert max_seqlen.dtype == torch.int32
        assert 0 < max_seqlen.item() <= seq_length

    elif is_first:
        assert tokens is not None
        assert position_ids is not None
        assert labels is None
        assert loss_mask is None
        assert cu_seqlens is not None
        assert max_seqlen is not None

        assert tokens.shape[1] == partitioned_seq_length
        assert position_ids.shape[1] == partitioned_seq_length

        assert cu_seqlens.dim() == 2
        assert cu_seqlens.dtype == torch.int32
        assert cu_seqlens[0, 0].item() == 0
        assert cu_seqlens[0, -1].item() == seq_length

    elif is_last:
        assert labels is not None
        assert loss_mask is not None
        assert tokens is None
        assert position_ids is None
        assert cu_seqlens is not None
        assert max_seqlen is not None

        assert labels.shape[1] == partitioned_seq_length
        assert loss_mask.shape[1] == partitioned_seq_length

        assert cu_seqlens.dim() == 2
        assert cu_seqlens.dtype == torch.int32
        assert cu_seqlens[0, 0].item() == 0
        assert cu_seqlens[0, -1].item() == seq_length

    else:
        assert tokens is None
        assert labels is None
        assert loss_mask is None
        assert position_ids is None
        assert cu_seqlens is not None
        assert max_seqlen is not None

    Utils.destroy_model_parallel()


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("seq_length", [16, 1024])
def test_get_batch_on_this_cp_rank_per_sequence_balancing(cp_size, seq_length):
    """Verify that per-sequence zigzag balancing selects the correct chunks.

    Constructs a batch with tokens = range(seq_length) and checks that each
    simulated CP rank receives the expected zigzag-interleaved chunks. The
    batch also carries the per-sample dataset_id that BlendedDataset adds,
    which has no sequence dimension and must survive untouched.
    """
    tokens = torch.arange(seq_length, dtype=torch.int64).unsqueeze(0)
    cu_seqlens = torch.tensor([[0, seq_length // 2, seq_length]], dtype=torch.int32)
    max_seqlen = torch.tensor([seq_length // 2], dtype=torch.int32)
    dataset_id = torch.tensor([7], dtype=torch.int64)

    for cp_rank in range(cp_size):
        batch = {
            'tokens': tokens.clone(),
            'cu_seqlens': cu_seqlens.clone(),
            'max_seqlen': max_seqlen.clone(),
            'dataset_id': dataset_id.clone(),
        }

        mock_group = MagicMock()
        with (
            patch('torch.distributed.get_world_size', return_value=cp_size),
            patch('torch.distributed.get_rank', return_value=cp_rank),
        ):
            result = _get_batch_on_this_cp_rank_per_sequence_balancing(batch, cp_group=mock_group)

        if cp_size == 1:
            assert torch.equal(result['tokens'], tokens)
        else:
            # The sequence is split into 2*cp_size equal chunks. This rank
            # gets chunk cp_rank and chunk 2*cp_size - cp_rank - 1.
            chunk_size = seq_length // (2 * cp_size)
            chunk_0_start = cp_rank * chunk_size
            chunk_1_start = (2 * cp_size - cp_rank - 1) * chunk_size
            expected = torch.cat(
                [
                    tokens[0, chunk_0_start : chunk_0_start + chunk_size],
                    tokens[0, chunk_1_start : chunk_1_start + chunk_size],
                ]
            ).unsqueeze(0)
            assert torch.equal(
                result['tokens'], expected
            ), f"cp_rank={cp_rank}: expected {expected}, got {result['tokens']}"

        # cu_seqlens, max_seqlen and dataset_id must be unchanged.
        assert torch.equal(result['cu_seqlens'], cu_seqlens)
        assert torch.equal(result['max_seqlen'], max_seqlen)
        assert torch.equal(result['dataset_id'], dataset_id)


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("seq_length", [16, 1024])
def test_get_batch_on_this_cp_rank_contiguous_keeps_attention_mask_zigzag(cp_size, seq_length):
    micro_batch_size = 2
    tokens = torch.arange(micro_batch_size * seq_length, dtype=torch.int64).view(
        micro_batch_size, seq_length
    )
    sequence_tensors = {
        'tokens': tokens,
        'labels': tokens + 1,
        'loss_mask': torch.ones(micro_batch_size, seq_length),
        'position_ids': torch.arange(seq_length, dtype=torch.int64).repeat(micro_batch_size, 1),
    }
    attention_mask = torch.arange(micro_batch_size * seq_length, dtype=torch.int64).view(
        micro_batch_size, 1, seq_length, 1
    )
    cu_seqlens = torch.tensor([[0, seq_length // 2, seq_length]], dtype=torch.int32)
    dataset_id = torch.tensor(7, dtype=torch.int64)

    for cp_rank in range(cp_size):
        batch = {
            **{key: value.clone() for key, value in sequence_tensors.items()},
            'attention_mask': attention_mask.clone(),
            'cu_seqlens': cu_seqlens,
            'dataset_id': dataset_id,
        }
        mock_group = MagicMock()
        with (
            patch('torch.distributed.get_world_size', return_value=cp_size),
            patch('torch.distributed.get_rank', return_value=cp_rank),
        ):
            result = _get_batch_on_this_cp_rank_contiguous(batch, cp_group=mock_group)

        local_sequence_length = seq_length // cp_size
        local_slice = slice(cp_rank * local_sequence_length, (cp_rank + 1) * local_sequence_length)
        for key, value in sequence_tensors.items():
            assert torch.equal(result[key], value[:, local_slice])
            assert result[key].is_contiguous()

        if cp_size == 1:
            expected_mask = attention_mask
        else:
            segment_length = seq_length // (2 * cp_size)
            expected_mask = torch.cat(
                [
                    attention_mask[:, :, cp_rank * segment_length : (cp_rank + 1) * segment_length],
                    attention_mask[
                        :,
                        :,
                        (2 * cp_size - cp_rank - 1)
                        * segment_length : (2 * cp_size - cp_rank)
                        * segment_length,
                    ],
                ],
                dim=2,
            )
        assert torch.equal(result['attention_mask'], expected_mask)
        assert result['cu_seqlens'] is cu_seqlens
        assert result['dataset_id'] is dataset_id


@pytest.mark.parametrize(
    ("cp_rank", "expected_indices"),
    [
        (0, [0, -1, 4, 5, 6, -1, -1, -1]),
        (1, [1, -1, 7, 8, 9, -1, -1, -1]),
        (2, [2, -1, 10, 11, 12, -1, -1, -1]),
        (3, [3, -1, 13, 14, 15, -1, -1, -1]),
    ],
)
@pytest.mark.parametrize("with_contiguous_layout", [False, True])
def test_get_batch_on_this_cp_rank_zigzag_packed(cp_rank, expected_indices, with_contiguous_layout):
    tokens = torch.arange(1, 17).view(1, 16)
    loss_mask = torch.tensor([[1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    batch = {
        "tokens": tokens,
        "labels": tokens + 100,
        "loss_mask": loss_mask,
        "position_ids": tokens - 1,
        "cu_seqlens": torch.tensor([[0, 3, 13]], dtype=torch.int32),
        "cu_seqlens_padded": torch.tensor([[0, 4, 16]], dtype=torch.int32),
        "max_seqlen": torch.tensor([12], dtype=torch.int32),
    }
    parallel_context = SimpleNamespace(
        cp_size=4,
        cp_rank=cp_rank,
        tp_size=4,
        tp_rank=0,
        group_size=16,
        communication_group=MagicMock(),
        group_rank_by_logical_rank=tuple(range(16)),
    )
    cp_group = MagicMock()
    cp_group.size.return_value = 4
    cp_group.rank.return_value = cp_rank
    tp_group = MagicMock()
    tp_group.size.return_value = 4
    with (
        patch("torch.distributed.get_world_size", return_value=4),
        patch("torch.distributed.get_rank", return_value=cp_rank),
        patch(
            "megatron.core.context_parallel.layout._get_layout_parallel_context",
            return_value=parallel_context,
        ),
    ):
        cp_batch = get_batches_on_this_cp_rank(
            batch,
            boundary_layout="contiguous" if with_contiguous_layout else "zigzag",
            is_hybrid_cp=False,
            cp_group=cp_group,
            additional_layouts=("zigzag",) if with_contiguous_layout else (),
            use_per_sequence_balancing=True,
            sequence_parallel=True,
            tp_group=tp_group,
            tp_cp_group=MagicMock(),
            tokens_per_sample=16,
        )
        result = cp_batch.get_batch("zigzag")

    indices = torch.tensor(expected_indices)
    padding = indices < 0
    gathered_indices = indices.clamp_min(0)
    padding = padding | (gathered_indices == 3) | (gathered_indices >= 14)
    expected_tokens = tokens.index_select(1, gathered_indices).masked_fill(padding.view(1, -1), 0)
    expected_loss_mask = loss_mask.index_select(1, gathered_indices).masked_fill(
        padding.view(1, -1), 0
    )
    torch.testing.assert_close(result["tokens"], expected_tokens)
    torch.testing.assert_close(result["loss_mask"], expected_loss_mask)
    torch.testing.assert_close(
        result["cu_seqlens_padded"], torch.tensor([[0, 8, 32]], dtype=torch.int32)
    )
    assert result["max_seqlen"].item() == 24
    assert (cp_batch.thd_plan is not None) == with_contiguous_layout
    assert set(cp_batch.batches_by_layout) == (
        {"contiguous", "zigzag"} if with_contiguous_layout else {"zigzag"}
    )
    zigzag_packed_seq_params = cp_batch.get_packed_seq_params("zigzag")
    assert zigzag_packed_seq_params is not None
    torch.testing.assert_close(
        zigzag_packed_seq_params.cu_seqlens_q_padded, result["cu_seqlens_padded"].squeeze(0)
    )
    torch.testing.assert_close(
        zigzag_packed_seq_params.cu_seqlens_q, batch["cu_seqlens"].squeeze(0)
    )
    assert zigzag_packed_seq_params.max_seqlen_q == 24
    assert zigzag_packed_seq_params.pad_between_seqs
    assert zigzag_packed_seq_params.total_tokens == 32
    torch.testing.assert_close(
        zigzag_packed_seq_params.seq_idx, torch.tensor([[0] * 8 + [1] * 24], dtype=torch.int32)
    )


def test_metadata_only_cp_batch_skips_sharding():
    batch = dict.fromkeys(pretrain_hybrid.BATCH_KEYS)
    batch["cu_seqlens"] = torch.tensor([[0, 3, 13]], dtype=torch.int32)
    batch["cu_seqlens_padded"] = torch.tensor([[0, 4, 16]], dtype=torch.int32)
    batch["max_seqlen"] = torch.tensor([12], dtype=torch.int32)

    with (
        patch("torch.distributed.get_world_size", return_value=4),
        patch("megatron.core.utils.get_batch_on_this_cp_rank") as shard_batch,
    ):
        cp_batch = get_batches_on_this_cp_rank(
            batch,
            boundary_layout="zigzag",
            is_hybrid_cp=False,
            cp_group=MagicMock(),
            tokens_per_sample=16,
        )

    shard_batch.assert_not_called()
    assert cp_batch.get_batch()["tokens"] is None
    assert cp_batch.get_packed_seq_params("zigzag") is not None


def test_get_batch_builds_required_cp_layouts():
    cp_size = 4
    seq_length = 16
    if int(os.environ.get("WORLD_SIZE", "1")) < cp_size:
        pytest.skip(f"CP={cp_size} requires at least {cp_size} ranks")
    global_batch_size = int(os.environ.get("WORLD_SIZE", "1")) // cp_size
    args = initialize_test_environment(
        tp_size=1,
        pp_size=1,
        cp_size=cp_size,
        seq_length=seq_length,
        micro_batch_size=1,
        global_batch_size=global_batch_size,
    )
    args.linear_cp_layout = "contiguous"
    args.attention_cp_layout = "zigzag"

    tokens = torch.arange(seq_length, dtype=torch.int64).view(1, -1)
    batch = {
        "tokens": tokens,
        "labels": tokens + 100,
        "loss_mask": torch.ones_like(tokens, dtype=torch.float32),
        "position_ids": tokens.clone(),
    }
    data_iterator = iter([batch]) if mpu.get_tensor_model_parallel_rank() == 0 else None

    cp_batch = get_batch(data_iterator)
    local_tokens = cp_batch.get_batch()["tokens"]

    cp_rank = mpu.get_context_parallel_rank()
    contiguous_indices = torch.arange(
        cp_rank * (seq_length // cp_size), (cp_rank + 1) * (seq_length // cp_size), device="cuda"
    )
    segment_length = seq_length // (2 * cp_size)
    zigzag_indices = torch.tensor(
        [
            *range(cp_rank * segment_length, (cp_rank + 1) * segment_length),
            *range(
                (2 * cp_size - cp_rank - 1) * segment_length,
                (2 * cp_size - cp_rank) * segment_length,
            ),
        ],
        device="cuda",
    )
    global_tokens = tokens.cuda()
    torch.testing.assert_close(local_tokens, global_tokens.index_select(1, contiguous_indices))
    assert set(cp_batch.batches_by_layout) == {"contiguous", "zigzag"}
    torch.testing.assert_close(
        cp_batch.get_batch("zigzag")["tokens"], global_tokens.index_select(1, zigzag_indices)
    )


@pytest.mark.parametrize(
    ("linear_cp_layout", "cp_size", "use_actual_lengths"),
    [("zigzag", 2, False), ("contiguous", 1, False), ("contiguous", 2, True)],
)
def test_packed_seq_params_expose_actual_thd_lengths_for_contiguous_cp(
    linear_cp_layout, cp_size, use_actual_lengths
):
    actual_cu_seqlens = torch.tensor([[0, 3, 5]], dtype=torch.int32)
    padded_cu_seqlens = torch.tensor([[0, 4, 8]], dtype=torch.int32)
    max_seqlen = torch.tensor([4], dtype=torch.int32)
    batch = {
        "cu_seqlens": actual_cu_seqlens,
        "cu_seqlens_padded": padded_cu_seqlens,
        "hybrid_cp_group": None,
        "local_cp_size": None,
        "max_seqlen": max_seqlen,
    }
    packed_seq_params = _build_packed_seq_params(
        batch, linear_cp_layout, cp_size, tokens_per_sample=8
    )
    assert packed_seq_params is not None
    expected_cu_seqlens = actual_cu_seqlens[0] if use_actual_lengths else padded_cu_seqlens[0]
    assert torch.equal(packed_seq_params.cu_seqlens_q, expected_cu_seqlens)
    assert torch.equal(packed_seq_params.cu_seqlens_kv, expected_cu_seqlens)
    assert torch.equal(packed_seq_params.cu_seqlens_q_padded, padded_cu_seqlens[0])
    assert packed_seq_params.total_tokens == 8


def create_pretrain_data_iterator(
    seq_length: int = 1024, micro_batch_size: int = 1, create_attention_mask: bool = False
):
    text = torch.randint(0, 10000, (micro_batch_size, seq_length + 1), dtype=torch.int64)
    tokens = text[:, :-1].contiguous()
    labels = text[:, 1:].contiguous()
    position_ids = (
        torch.arange(seq_length, dtype=torch.long)
        .unsqueeze(0)
        .expand(micro_batch_size, -1)
        .contiguous()
    )
    loss_mask = torch.ones((micro_batch_size, seq_length), dtype=torch.float)

    batch = {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
    }

    if create_attention_mask:
        batch["attention_mask"] = torch.tril(
            torch.ones((micro_batch_size, 1, seq_length, seq_length))
        ).bool()

    return iter([_add_provenance_keys(batch)])


def test_sequence_packing_batch_uses_context_parallel_batch_interface():
    tokens = torch.tensor([[1, 2]])
    labels = torch.tensor([[2, 3]])
    loss_mask = torch.ones(1, 2)
    position_ids = torch.tensor([[0, 1]])
    padding_mask = torch.zeros(1, 2, dtype=torch.bool)
    packed_seq_params = PackedSeqParams(qkv_format="thd")
    scheduler_batch = (
        tokens,
        labels,
        loss_mask,
        None,
        position_ids,
        packed_seq_params,
        padding_mask,
    )
    args = SimpleNamespace(sequence_packing_scheduler="dp_balanced")
    config = SimpleNamespace(
        virtual_pipeline_model_parallel_size=None,
        pipeline_model_parallel_layout=None,
        mtp_num_layers=1,
        linear_cp_layout="zigzag",
    )

    with (
        patch.object(pretrain_hybrid, "get_args", return_value=args),
        patch.object(pretrain_hybrid, "core_transformer_config_from_args", return_value=config),
        patch.object(pretrain_hybrid, "mtp_on_this_rank_func", return_value=True),
        patch.object(
            pretrain_hybrid,
            "get_batch_on_this_rank_for_sequence_packing",
            return_value=scheduler_batch,
        ),
    ):
        cp_batch = get_batch(None)

    assert set(cp_batch.batches_by_layout) == {"zigzag"}
    assert cp_batch.get_packed_seq_params() is packed_seq_params
    batch = cp_batch.get_batch()
    assert batch["tokens"] is tokens
    assert batch["labels"] is labels
    assert batch["loss_mask"] is loss_mask
    assert batch["position_ids"] is position_ids
    assert batch["padding_mask"] is padding_mask


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("pp_size", [1, 2, 4])
@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("seq_length", [1024, 4096])
@pytest.mark.parametrize("create_attention_mask", [True, False])
@pytest.mark.parametrize("micro_batch_size", [1, 4])
def test_pretrain_batch(
    tp_size, pp_size, cp_size, seq_length, create_attention_mask, micro_batch_size
):
    if tp_size * pp_size * cp_size > torch.cuda.device_count():
        pytest.skip(
            f"Skipping test because tp_size * pp_size * cp_size > torch.cuda.device_count() ({tp_size * pp_size * cp_size} > {torch.cuda.device_count()})"
        )
    dp_size = int(os.environ.get("WORLD_SIZE", 1)) // (tp_size * pp_size * cp_size)
    global_batch_size = micro_batch_size * dp_size
    args = initialize_test_environment(
        tp_size,
        pp_size,
        cp_size,
        seq_length,
        micro_batch_size,
        global_batch_size=global_batch_size,
        sft=False,
        create_attention_mask=create_attention_mask,
    )

    data_iterator = None
    if mpu.get_tensor_model_parallel_rank() == 0:
        data_iterator = create_pretrain_data_iterator(
            seq_length,
            micro_batch_size=micro_batch_size,
            create_attention_mask=create_attention_mask,
        )

    cp_batch = get_batch(data_iterator)
    batch = cp_batch.get_batch()
    (
        attention_mask,
        cu_seqlens,
        cu_seqlens_padded,
        hybrid_cp_group,
        labels,
        local_cp_size,
        loss_mask,
        max_seqlen,
        position_ids,
        tokens,
    ) = [batch[key] for key in pretrain_hybrid.BATCH_KEYS]

    is_first = mpu.is_pipeline_first_stage()
    is_last = mpu.is_pipeline_last_stage()
    seq_len_per_rank = seq_length // cp_size

    if pp_size == 1:
        # Single pipeline stage: all tensors present
        assert tokens is not None
        assert labels is not None
        assert loss_mask is not None
        assert position_ids is not None
        assert cu_seqlens is None
        assert cu_seqlens_padded is None
        assert max_seqlen is None
        assert hybrid_cp_group is None
        assert local_cp_size is None

        assert tokens.shape == (
            micro_batch_size,
            seq_len_per_rank,
        ), f"Expected tokens shape ({micro_batch_size}, {seq_len_per_rank}), got {tokens.shape}"
        assert labels.shape == (
            micro_batch_size,
            seq_len_per_rank,
        ), f"Expected labels shape ({micro_batch_size}, {seq_len_per_rank}), got {labels.shape}"
        assert loss_mask.shape == (
            micro_batch_size,
            seq_len_per_rank,
        ), f"Expected loss_mask shape ({micro_batch_size}, {seq_len_per_rank}), got {loss_mask.shape}"
        assert position_ids.shape == (
            micro_batch_size,
            seq_len_per_rank,
        ), f"Expected position_ids shape ({micro_batch_size}, {seq_len_per_rank}), got {position_ids.shape}"

        assert tokens.dtype == torch.int64
        assert labels.dtype == torch.int64
        assert loss_mask.dtype == torch.float32
        assert position_ids.dtype == torch.int64

        assert loss_mask.sum().item() == micro_batch_size * seq_len_per_rank

        if create_attention_mask:
            assert attention_mask is not None
            assert attention_mask.shape == (
                micro_batch_size,
                1,
                seq_len_per_rank,
                seq_length,
            ), f"Expected attention_mask shape ({micro_batch_size}, 1, {seq_len_per_rank}, {seq_length}), got {attention_mask.shape}"
            assert attention_mask.dtype == torch.bool
        else:
            assert attention_mask is None

    elif is_first:
        # First pipeline stage: tokens, position_ids, and optionally attention_mask
        assert tokens is not None
        assert position_ids is not None
        assert labels is None
        assert loss_mask is None
        assert cu_seqlens is None
        assert cu_seqlens_padded is None
        assert max_seqlen is None
        assert hybrid_cp_group is None
        assert local_cp_size is None

        assert tokens.shape == (
            micro_batch_size,
            seq_len_per_rank,
        ), f"Expected tokens shape ({micro_batch_size}, {seq_len_per_rank}), got {tokens.shape}"
        assert position_ids.shape == (
            micro_batch_size,
            seq_len_per_rank,
        ), f"Expected position_ids shape ({micro_batch_size}, {seq_len_per_rank}), got {position_ids.shape}"

        assert tokens.dtype == torch.int64
        assert position_ids.dtype == torch.int64

        if create_attention_mask:
            assert attention_mask is not None
            assert attention_mask.shape == (
                micro_batch_size,
                1,
                seq_len_per_rank,
                seq_length,
            ), f"Expected attention_mask shape ({micro_batch_size}, 1, {seq_len_per_rank}, {seq_length}), got {attention_mask.shape}"
            assert attention_mask.dtype == torch.bool
        else:
            assert attention_mask is None

    elif is_last:
        # Last pipeline stage: labels, loss_mask, and optionally attention_mask
        assert labels is not None
        assert loss_mask is not None
        assert tokens is None
        assert position_ids is None
        assert cu_seqlens is None
        assert cu_seqlens_padded is None
        assert max_seqlen is None
        assert hybrid_cp_group is None
        assert local_cp_size is None

        assert labels.shape == (
            micro_batch_size,
            seq_len_per_rank,
        ), f"Expected labels shape ({micro_batch_size}, {seq_len_per_rank}), got {labels.shape}"
        assert loss_mask.shape == (
            micro_batch_size,
            seq_len_per_rank,
        ), f"Expected loss_mask shape ({micro_batch_size}, {seq_len_per_rank}), got {loss_mask.shape}"

        assert labels.dtype == torch.int64
        assert loss_mask.dtype == torch.float32

        assert loss_mask.sum().item() == micro_batch_size * seq_len_per_rank

        if create_attention_mask:
            assert attention_mask is not None
            assert attention_mask.shape == (
                micro_batch_size,
                1,
                seq_len_per_rank,
                seq_length,
            ), f"Expected attention_mask shape ({micro_batch_size}, 1, {seq_len_per_rank}, {seq_length}), got {attention_mask.shape}"
            assert attention_mask.dtype == torch.bool
        else:
            assert attention_mask is None

    else:
        # Intermediate pipeline stages: all None
        assert tokens is None
        assert labels is None
        assert loss_mask is None
        assert position_ids is None
        assert attention_mask is None
        assert cu_seqlens is None
        assert cu_seqlens_padded is None
        assert max_seqlen is None
        assert hybrid_cp_group is None
        assert local_cp_size is None

    Utils.destroy_model_parallel()


def create_hybrid_cp_data_iterator(seq_length: int = 1024, cp_size: int = 1):
    # Pack n_seqs equal-length sequences; total length must be divisible by 2 * cp_size for CP splitting
    n_seqs = max(2, 2 * cp_size)
    align = max(1, 2 * cp_size)
    seq_len_each = (seq_length // n_seqs // align) * align
    if seq_len_each == 0:
        seq_len_each = align
    total_seq_len = n_seqs * seq_len_each

    text = torch.randint(0, 10000, (1, total_seq_len + 1), dtype=torch.int64)
    tokens = text[:, :-1].contiguous()  # (1, total_seq_len)
    labels = text[:, 1:].contiguous()  # (1, total_seq_len)
    loss_mask = torch.ones((1, total_seq_len), dtype=torch.float32)
    position_ids = torch.cat(
        [torch.arange(seq_len_each, dtype=torch.int64) for _ in range(n_seqs)]
    ).unsqueeze(
        0
    )  # (1, total_seq_len)

    cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32),
            torch.cumsum(torch.tensor([seq_len_each] * n_seqs, dtype=torch.int64), dim=0).to(
                torch.int32
            ),
        ]
    ).unsqueeze(
        0
    )  # (1, n_seqs + 1) — dataloader always carries a batch dim
    max_seqlen = torch.tensor([seq_len_each], dtype=torch.int32)
    local_cp_size_tensor = torch.tensor([cp_size], dtype=torch.int32)

    batch = {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max_seqlen,
        "local_cp_size": local_cp_size_tensor,
    }

    if cp_size > 1:
        batch["cu_seqlens_padded"] = cu_seqlens.clone()

    return iter([batch])


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("cp_size", [2, 4, 8])
@pytest.mark.parametrize("seq_length", [1024])
@pytest.mark.parametrize("create_attention_mask", [False])
def test_hybrid_cp_batch(tp_size, cp_size, seq_length, create_attention_mask):
    if tp_size * cp_size > torch.cuda.device_count():
        pytest.skip(
            f"Skipping test because tp_size * cp_size > torch.cuda.device_count() ({tp_size * cp_size} > {torch.cuda.device_count()})"
        )

    args = initialize_test_environment(
        tp_size,
        1,
        cp_size,
        seq_length,
        1,
        16,
        sft=False,
        hybrid_context_parallel=True,
        create_attention_mask=create_attention_mask,
    )

    data_iterator = None
    if mpu.get_tensor_model_parallel_rank() == 0:
        data_iterator = create_hybrid_cp_data_iterator(seq_length, cp_size=cp_size)

    cp_batch = get_batch(data_iterator)
    batch = cp_batch.get_batch()
    (
        attention_mask,
        cu_seqlens,
        cu_seqlens_padded,
        hybrid_cp_group,
        labels,
        local_cp_size,
        loss_mask,
        max_seqlen,
        position_ids,
        tokens,
    ) = [batch[key] for key in pretrain_hybrid.BATCH_KEYS]

    # Presence checks
    assert tokens is not None
    assert labels is not None
    assert loss_mask is not None
    assert position_ids is not None
    assert attention_mask is None
    assert cu_seqlens is not None
    assert max_seqlen is not None
    assert local_cp_size is not None

    # Data iterator parameters (must match create_hybrid_cp_data_iterator)
    n_seqs = max(2, 2 * cp_size)
    align = max(1, 2 * cp_size)
    seq_len_each = (seq_length // n_seqs // align) * align
    if seq_len_each == 0:
        seq_len_each = align
    total_seq_len = n_seqs * seq_len_each

    # Shape: HybridCP CP splitting gives total_seq_len // cp_size tokens per rank
    seq_len_per_rank = total_seq_len // cp_size
    assert tokens.shape == (
        1,
        seq_len_per_rank,
    ), f"Expected tokens shape (1, {seq_len_per_rank}), got {tokens.shape}"
    assert labels.shape == (
        1,
        seq_len_per_rank,
    ), f"Expected labels shape (1, {seq_len_per_rank}), got {labels.shape}"
    assert loss_mask.shape == (
        1,
        seq_len_per_rank,
    ), f"Expected loss_mask shape (1, {seq_len_per_rank}), got {loss_mask.shape}"
    assert position_ids.shape == (
        1,
        seq_len_per_rank,
    ), f"Expected position_ids shape (1, {seq_len_per_rank}), got {position_ids.shape}"

    # Dtype checks
    assert tokens.dtype == torch.int64
    assert labels.dtype == torch.int64
    assert loss_mask.dtype == torch.float32
    assert position_ids.dtype == torch.int64

    # Loss mask is all-ones (no masking in the HybridCP pretrain dataloader)
    assert loss_mask.sum().item() == seq_len_per_rank

    # cu_seqlens: 2-D int32 (1, n_seqs + 1) after flatten_batch_for_packed_sequences.
    assert cu_seqlens.shape == (
        1,
        n_seqs + 1,
    ), f"Expected cu_seqlens shape (1, {n_seqs + 1}), got {cu_seqlens.shape}"
    assert cu_seqlens.dtype == torch.int32
    assert cu_seqlens[0, 0].item() == 0
    assert cu_seqlens[0, -1].item() == total_seq_len

    # max_seqlen: scalar int32 equal to the per-sequence length in the iterator
    assert max_seqlen.shape == (1,)
    assert max_seqlen.dtype == torch.int32
    assert max_seqlen.item() == seq_len_each

    # local_cp_size: scalar int32 equal to cp_size
    assert local_cp_size.shape == (1,)
    assert local_cp_size.dtype == torch.int32
    assert local_cp_size.item() == cp_size

    if cp_size > 1:
        assert cu_seqlens_padded is not None
        assert cu_seqlens_padded.shape == (1, n_seqs + 1)
        assert cu_seqlens_padded.dtype == torch.int32
        assert cu_seqlens_padded[0, 0].item() == 0
        assert cu_seqlens_padded[0, -1].item() == total_seq_len
        assert hybrid_cp_group is not None
    else:
        assert cu_seqlens_padded is None
        assert hybrid_cp_group is None

    Utils.destroy_model_parallel()


def create_inter_document_masking_data_iterator(seq_length: int = 1024, micro_batch_size: int = 2):
    """Create a mock data iterator for inter-document masking with mbs > 1.

    Mimics what default_collate produces from GPTDataset with
    inter_document_masking=True: each sample has its own padded cu_seqlens
    row, collated into (micro_batch_size, padded_len).
    """
    padded_len = seq_length + 1
    cu_seqlens = torch.full((micro_batch_size, padded_len), seq_length, dtype=torch.int32)
    max_seqlens = []

    for i in range(micro_batch_size):
        n_docs = torch.randint(2, 6, (1,)).item()
        boundaries = sorted(torch.randint(1, seq_length, (n_docs - 1,)).tolist())
        boundaries = [0] + boundaries + [seq_length]
        for j, val in enumerate(boundaries):
            cu_seqlens[i, j] = val
        seg_lengths = [boundaries[k + 1] - boundaries[k] for k in range(len(boundaries) - 1)]
        max_seqlens.append(max(seg_lengths))

    max_seqlen = torch.tensor(max_seqlens, dtype=torch.int32)

    tokens = torch.randint(0, 10000, (micro_batch_size, seq_length), dtype=torch.int64)
    labels = torch.randint(0, 10000, (micro_batch_size, seq_length), dtype=torch.int64)
    loss_mask = torch.ones(micro_batch_size, seq_length, dtype=torch.float32)
    position_ids = (
        torch.arange(seq_length, dtype=torch.int64)
        .unsqueeze(0)
        .expand(micro_batch_size, -1)
        .clone()
    )

    batch = {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max_seqlen,
    }
    return iter([batch])


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("micro_batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_length", [1024])
def test_inter_document_masking_multi_mbs_batch(tp_size, micro_batch_size, seq_length):
    """Verify cu_seqlens is correctly broadcast and merged when mbs > 1 with TP > 1.

    Regression test: the receiver in get_batch_on_this_tp_rank used to allocate
    cu_seqlens as (1, numel) instead of (mbs, padded_len), which caused
    flatten_batch_for_packed_sequences to silently drop all samples after the
    first on non-zero TP ranks.
    """
    if tp_size > torch.cuda.device_count():
        pytest.skip(
            f"Skipping test because tp_size > torch.cuda.device_count() "
            f"({tp_size} > {torch.cuda.device_count()})"
        )

    dp_size = int(os.environ.get("WORLD_SIZE", 1)) // tp_size
    global_batch_size = micro_batch_size * dp_size
    args = initialize_test_environment(
        tp_size,
        pp_size=1,
        cp_size=1,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        sft=False,
    )
    args.dataloader_inter_document_masking = True

    data_iterator = None
    if mpu.get_tensor_model_parallel_rank() == 0:
        data_iterator = create_inter_document_masking_data_iterator(
            seq_length, micro_batch_size=micro_batch_size
        )

    cp_batch = get_batch(data_iterator)
    batch = cp_batch.get_batch()
    (
        attention_mask,
        cu_seqlens,
        cu_seqlens_padded,
        hybrid_cp_group,
        labels,
        local_cp_size,
        loss_mask,
        max_seqlen,
        position_ids,
        tokens,
    ) = [batch[key] for key in pretrain_hybrid.BATCH_KEYS]

    total_tokens = micro_batch_size * seq_length

    assert tokens is not None
    assert tokens.shape == (1, total_tokens)
    assert labels is not None
    assert labels.shape == (1, total_tokens)
    assert loss_mask is not None
    assert loss_mask.shape == (1, total_tokens)
    assert position_ids is not None
    assert position_ids.shape == (1, total_tokens)

    assert cu_seqlens is not None
    assert cu_seqlens.dim() == 2
    assert cu_seqlens.shape[0] == 1
    assert cu_seqlens.dtype == torch.int32
    assert cu_seqlens[0, 0].item() == 0
    assert cu_seqlens[0, -1].item() == total_tokens
    assert cu_seqlens.shape[1] >= micro_batch_size + 1

    assert max_seqlen is not None
    assert max_seqlen.numel() == 1

    Utils.destroy_model_parallel()
