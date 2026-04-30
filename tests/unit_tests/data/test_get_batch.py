# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import sys

import pytest
import torch

from megatron.core import mpu
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
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
    return iter([batch]), num_real_tokens


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
    initialize_test_environment(
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
    ) = get_batch(data_iterator)

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

        assert cu_seqlens.dim() == 1
        assert cu_seqlens.dtype == torch.int32
        assert cu_seqlens[0].item() == 0
        assert cu_seqlens[-1].item() == seq_length
        assert cu_seqlens.shape[0] >= 2

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

        assert cu_seqlens.dim() == 1
        assert cu_seqlens.dtype == torch.int32
        assert cu_seqlens[0].item() == 0
        assert cu_seqlens[-1].item() == seq_length
        assert cu_seqlens.shape[0] >= 2

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

        assert cu_seqlens.dim() == 1
        assert cu_seqlens.dtype == torch.int32
        assert cu_seqlens[0].item() == 0
        assert cu_seqlens[-1].item() == seq_length
        assert cu_seqlens.shape[0] >= 2

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

        assert cu_seqlens.dim() == 1
        assert cu_seqlens.dtype == torch.int32
        assert cu_seqlens[0].item() == 0
        assert cu_seqlens[-1].item() == seq_length
        assert cu_seqlens.shape[0] >= 2

        assert max_seqlen.shape == (1,)
        assert max_seqlen.dtype == torch.int32
        assert 0 < max_seqlen.item() <= seq_length

    Utils.destroy_model_parallel()


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

    return iter([batch])


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
    initialize_test_environment(
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
    ) = get_batch(data_iterator)

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
    )
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

    initialize_test_environment(
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
    ) = get_batch(data_iterator)

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

    # cu_seqlens: 1D int32, [0, seq_len_each, 2*seq_len_each, ..., total_seq_len]
    assert cu_seqlens.shape == (
        n_seqs + 1,
    ), f"Expected cu_seqlens shape ({n_seqs + 1},), got {cu_seqlens.shape}"
    assert cu_seqlens.dtype == torch.int32
    assert cu_seqlens[0].item() == 0
    assert cu_seqlens[-1].item() == total_seq_len

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
        assert cu_seqlens_padded.shape == (n_seqs + 1,)
        assert cu_seqlens_padded.dtype == torch.int32
        assert cu_seqlens_padded[0].item() == 0
        assert cu_seqlens_padded[-1].item() == total_seq_len
        assert hybrid_cp_group is not None
    else:
        assert cu_seqlens_padded is None
        assert hybrid_cp_group is None

    Utils.destroy_model_parallel()
