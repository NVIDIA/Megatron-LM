# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import sys

import pytest
import torch

from megatron.core import mpu
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.training.arguments import parse_args, validate_args
from megatron.training.global_vars import destroy_global_vars, set_global_variables
from pretrain_mamba import get_batch
from tests.unit_tests.test_utilities import Utils


def initialize_test_environment(
    tp_size: int,
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
    args.pipeline_model_parallel_size = 1
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
        context_parallel_size=cp_size,
        hybrid_context_parallel=hybrid_context_parallel,
    )
    return args


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
@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("seq_length", [1024, 2048, 4096])
@pytest.mark.parametrize("create_attention_mask", [True, False])
@pytest.mark.parametrize("micro_batch_size", [1, 2, 4])
def test_pretrain_batch(tp_size, cp_size, seq_length, create_attention_mask, micro_batch_size):
    if cp_size * tp_size > torch.cuda.device_count():
        pytest.skip(
            f"Skipping test because cp_size * tp_size > torch.cuda.device_count() ({cp_size * tp_size} > {torch.cuda.device_count()})"
        )
    dp_size = int(os.environ.get("WORLD_SIZE", 1)) // (tp_size * cp_size)
    global_batch_size = micro_batch_size * dp_size
    initialize_test_environment(
        tp_size,
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

    # Presence checks
    assert tokens is not None
    assert labels is not None
    assert loss_mask is not None
    assert position_ids is not None
    assert cu_seqlens is None
    assert cu_seqlens_padded is None
    assert max_seqlen is None
    assert hybrid_cp_group is None
    assert local_cp_size is None

    # Shape: pretrain CP slicing takes 2 non-contiguous chunks → seq_length // cp_size tokens per rank
    seq_len_per_rank = seq_length // cp_size
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

    # Dtype checks
    assert tokens.dtype == torch.int64
    assert labels.dtype == torch.int64
    assert loss_mask.dtype == torch.float32
    assert position_ids.dtype == torch.int64

    # Pretrain loss_mask is all-ones (no masking in the dataloader)
    assert loss_mask.sum().item() == micro_batch_size * seq_len_per_rank

    if create_attention_mask:
        assert attention_mask is not None
        # attention_mask input shape (B, 1, S, S); seq_dim=2 splits the query dim → (B, 1, S // cp_size, S)
        assert attention_mask.shape == (
            micro_batch_size,
            1,
            seq_len_per_rank,
            seq_length,
        ), f"Expected attention_mask shape ({micro_batch_size}, 1, {seq_len_per_rank}, {seq_length}), got {attention_mask.shape}"
        assert attention_mask.dtype == torch.bool
    else:
        assert attention_mask is None

    Utils.destroy_model_parallel()


def create_hybrid_cp_data_iterator(seq_length: int = 1024, cp_size: int = 1):
    total_seq_len = seq_length
    text = torch.randint(0, 10000, (total_seq_len + 1,), dtype=torch.int64)
    return iter(
        [
            {
                "tokens": text[:-1].contiguous(),
                "labels": text[1:].contiguous(),
                "loss_mask": torch.ones(total_seq_len, dtype=torch.float32),
                "position_ids": torch.arange(total_seq_len, dtype=torch.int64),
                "local_cp_size": torch.tensor([cp_size], dtype=torch.int32),
            }
        ]
    )


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("cp_size", [2, 4, 8])
@pytest.mark.parametrize("seq_length", [1024])
@pytest.mark.parametrize("create_attention_mask", [False])
def test_hybrid_cp_batch(tp_size, cp_size, seq_length, create_attention_mask):
    if cp_size * tp_size > torch.cuda.device_count():
        pytest.skip(
            f"Skipping test because cp_size * tp_size > torch.cuda.device_count() ({cp_size * tp_size} > {torch.cuda.device_count()})"
        )

    initialize_test_environment(
        tp_size,
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
    assert attention_mask is None  # HybridCP does not use attention mask from dataloader
    assert cu_seqlens is not None
    assert max_seqlen is not None
    assert local_cp_size is not None

    total_seq_len = seq_length

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

    # Legacy HybridCP synthesizes a single packed segment for the sample.
    assert cu_seqlens.shape == (2,), f"Expected cu_seqlens shape (2,), got {cu_seqlens.shape}"
    assert cu_seqlens.dtype == torch.int32
    assert cu_seqlens[0].item() == 0
    assert cu_seqlens[-1].item() == total_seq_len

    # max_seqlen: scalar int32 equal to the full sample length.
    assert max_seqlen.shape == (1,)
    assert max_seqlen.dtype == torch.int32
    assert max_seqlen.item() == total_seq_len

    # local_cp_size: scalar int32 equal to cp_size
    assert local_cp_size.shape == (1,)
    assert local_cp_size.dtype == torch.int32
    assert local_cp_size.item() == cp_size

    if cp_size > 1:
        assert cu_seqlens_padded is not None
        assert cu_seqlens_padded.shape == (2,)
        assert cu_seqlens_padded.dtype == torch.int32
        assert cu_seqlens_padded[0].item() == 0
        assert cu_seqlens_padded[-1].item() == total_seq_len
        assert hybrid_cp_group is not None
    else:
        assert cu_seqlens_padded is None
        assert hybrid_cp_group is None

    Utils.destroy_model_parallel()
