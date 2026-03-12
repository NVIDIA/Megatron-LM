import sys
import os

from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.global_vars import (
    destroy_global_vars,
    get_args,
    set_args,
    set_global_variables,
)

from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)

from tests.unit_tests.test_utilities import Utils
import pytest

from megatron.core import mpu

from sft_mamba import get_batch

import torch


def initialize_test_environment(
    tp_size: int, cp_size: int, seq_length: int, micro_batch_size: int, global_batch_size: int = 1, sft: bool = False, hybrid_context_parallel: bool = False, max_seqlen_per_cp_rank: int = 1024, create_attention_mask: bool = False
):
    destroy_global_vars()
    destroy_num_microbatches_calculator()

    sys.argv = ['test_tp_cp_.py']
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

    validate_args(args)
    set_global_variables(args, True)

    Utils.initialize_model_parallel(tensor_model_parallel_size=tp_size, context_parallel_size=cp_size)
    return args


def create_sft_data_iterator(max_seq_length: int = 1024):
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
    assert num_real_tokens < max_seq_length, f"Sum of lengths {num_real_tokens} is greater than max_seq_length {max_seq_length}"
    text = torch.randint(0, 10000, (1, num_real_tokens + 1), dtype=torch.int64)
    tokens = text[:, :-1].contiguous()
    labels = text[:, 1:].contiguous()

    cu_seqlens = torch.cat((
        torch.zeros(1, dtype=torch.int32),
        torch.cumsum(torch.tensor(lengths, dtype=torch.int64), dim=0).to(torch.int32),
    ))

    loss_mask = torch.ones((1, num_real_tokens), dtype=torch.float)
    batch = {"tokens": tokens, "labels": labels, "loss_mask": loss_mask, "cu_seqlens": cu_seqlens}
    return iter([batch]), num_real_tokens


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("seq_length", [1024, 2048, 4096])
def test_sft_batch(tp_size, cp_size, seq_length):
    if cp_size * tp_size > torch.cuda.device_count():
        pytest.skip(f"Skipping test because cp_size * tp_size > torch.cuda.device_count() ({cp_size * tp_size} > {torch.cuda.device_count()})")

    global_batch_size = int(os.environ.get("WORLD_SIZE", 1)) // (tp_size * cp_size)
    initialize_test_environment(tp_size, cp_size, seq_length, micro_batch_size=1, global_batch_size=global_batch_size, sft=True)

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

    # Presence checks
    assert tokens is not None
    assert labels is not None
    assert loss_mask is not None
    assert position_ids is not None
    assert cu_seqlens is not None
    assert max_seqlen is not None
    assert attention_mask is None
    assert hybrid_cp_group is None
    assert local_cp_size is None

    # Shape: preprocess_sft_batch pads to seq_length; THD CP slicing gives seq_length // cp_size per rank
    seq_len_per_rank = seq_length // cp_size
    assert tokens.shape == (1, seq_len_per_rank), f"Expected tokens shape (1, {seq_len_per_rank}), got {tokens.shape}"
    assert labels.shape == (1, seq_len_per_rank), f"Expected labels shape (1, {seq_len_per_rank}), got {labels.shape}"
    assert loss_mask.shape == (1, seq_len_per_rank), f"Expected loss_mask shape (1, {seq_len_per_rank}), got {loss_mask.shape}"
    assert position_ids.shape == (1, seq_len_per_rank), f"Expected position_ids shape (1, {seq_len_per_rank}), got {position_ids.shape}"

    # Dtype checks
    assert tokens.dtype == torch.int64
    assert labels.dtype == torch.int64
    assert loss_mask.dtype == torch.float32
    assert position_ids.dtype == torch.int64

    # cu_seqlens: 1D int32, starts at 0, ends at seq_length (padded by preprocess_sft_batch)
    assert cu_seqlens.dim() == 1
    assert cu_seqlens.dtype == torch.int32
    assert cu_seqlens[0].item() == 0
    assert cu_seqlens[-1].item() == seq_length
    assert cu_seqlens.shape[0] >= 2  # at least one sequence

    # max_seqlen: scalar, positive, within seq_length
    assert max_seqlen.shape == (1,)
    assert max_seqlen.dtype == torch.int32
    assert 0 < max_seqlen.item() <= seq_length

    if cp_size > 1:
        assert cu_seqlens_padded is not None
        assert cu_seqlens_padded.dim() == 1
        assert cu_seqlens_padded.dtype == torch.int32
        assert cu_seqlens_padded[0].item() == 0
        assert cu_seqlens_padded[-1].item() == seq_length
        assert cu_seqlens_padded.shape == cu_seqlens.shape

        # Compute the divisibility factor (mirrors preprocess_sft_batch logic)
        sp = tp_size > 1
        divisibility_factor = cp_size * 2
        if tp_size > 1 and sp:
            divisibility_factor *= tp_size

        # Compute the segment lengths from cu_seqlens and cu_seqlens_padded
        orig_seg_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        padded_seg_lengths = cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
        num_segments = len(orig_seg_lengths)

        # cu_seqlens_padded: every segment must be divisible by divisibility_factor
        for i, seg_len in enumerate(padded_seg_lengths):
            assert seg_len.item() % divisibility_factor == 0, (
                f"Padded segment {i} length {seg_len.item()} not divisible by {divisibility_factor}"
            )

        # cu_seqlens_padded segments >= cu_seqlens segments (padding only adds tokens).
        # The last segment is excluded because pad_or_truncate_thd_tensors replaces
        # the final entry of both cu_seqlens and cu_seqlens_padded with seq_length.
        # Since cu_seqlens_padded[-2] >= cu_seqlens[-2] (from CP padding), the last
        # padded segment (seq_length - cu_seqlens_padded[-2]) can be smaller than the
        # last original segment (seq_length - cu_seqlens[-2]).
        for i in range(num_segments - 1):
            assert padded_seg_lengths[i].item() >= orig_seg_lengths[i].item(), (
                f"Segment {i}: padded length {padded_seg_lengths[i].item()} < original length {orig_seg_lengths[i].item()}"
            )

        # loss_mask: must be binary (0.0 or 1.0)
        assert ((loss_mask == 0.0) | (loss_mask == 1.0)).all(), "loss_mask must be binary"

        # Intra-sample CP padding validation: for each padded segment on this
        # CP rank, verify that position_ids and loss_mask are consistent with
        # the padding introduced by pad_thd_sequences_for_cp.
        cp_rank = mpu.get_context_parallel_rank()
        per_rank_seg_lens = (padded_seg_lengths // cp_size).tolist()

        offset = 0
        for i in range(num_segments):
            seg_len = per_rank_seg_lens[i]
            if seg_len == 0:
                continue
            seg_pos = position_ids[0, offset:offset + seg_len]
            seg_loss = loss_mask[0, offset:offset + seg_len]

            # position_ids within each segment's CP partition must be contiguous
            # (stride 1) starting at cp_rank * seg_len (THD partitioning
            # assigns the r-th contiguous chunk of each padded segment to rank r)
            expected_start = cp_rank * seg_len
            expected_pos = torch.arange(
                expected_start, expected_start + seg_len,
                dtype=torch.int64, device=seg_pos.device,
            )
            assert torch.equal(seg_pos, expected_pos), (
                f"Segment {i}: expected position_ids [{expected_start}, "
                f"{expected_start + seg_len}) but got "
                f"[{seg_pos[0].item()}, ..., {seg_pos[-1].item()}] on CP rank {cp_rank}"
            )

            # For non-last segments, cu_seqlens entries are unchanged by
            # pad_or_truncate_thd_tensors, so orig_seg_lengths[i] is the true
            # sub-sequence length.  This lets us make precise assertions:
            #   position_id >= orig_len  =>  intra-sample CP padding  =>  loss_mask == 0
            #   position_id <  orig_len  =>  real token               =>  loss_mask == 1
            # (The last segment absorbs end-of-sequence padding so its
            # orig_seg_lengths entry is inflated -- skip the precise check.)
            if i < num_segments - 1:
                orig_len = orig_seg_lengths[i].item()
                padding_mask = seg_pos >= orig_len
                if padding_mask.any():
                    assert (seg_loss[padding_mask] == 0.0).all(), (
                        f"Segment {i}: intra-sample padding tokens (pos >= {orig_len}) "
                        f"must have loss_mask=0, CP rank {cp_rank}"
                    )
                real_mask = seg_pos < orig_len
                if real_mask.any():
                    assert (seg_loss[real_mask] == 1.0).all(), (
                        f"Segment {i}: real tokens (pos < {orig_len}) "
                        f"must have loss_mask=1, CP rank {cp_rank}"
                    )

            offset += seg_len

        assert offset == seq_len_per_rank, (
            f"Total per-rank offset {offset} != expected {seq_len_per_rank}"
        )
    else:
        assert cu_seqlens_padded is None

    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def create_pretrain_data_iterator(seq_length: int = 1024, micro_batch_size: int = 1, create_attention_mask: bool = False):
    text = torch.randint(0, 10000, (micro_batch_size, seq_length + 1), dtype=torch.int64)
    tokens = text[:, :-1].contiguous()
    labels = text[:, 1:].contiguous()
    position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0).expand(micro_batch_size, -1).contiguous()
    loss_mask = torch.ones((micro_batch_size, seq_length), dtype=torch.float)

    batch = {"tokens": tokens, "labels": labels, "loss_mask": loss_mask, "position_ids": position_ids}

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
        pytest.skip(f"Skipping test because cp_size * tp_size > torch.cuda.device_count() ({cp_size * tp_size} > {torch.cuda.device_count()})")
    dp_size = int(os.environ.get("WORLD_SIZE", 1)) // (tp_size * cp_size)
    global_batch_size = micro_batch_size * dp_size
    initialize_test_environment(tp_size, cp_size, seq_length, micro_batch_size, global_batch_size=global_batch_size, sft=False, create_attention_mask=create_attention_mask)

    data_iterator = None
    if mpu.get_tensor_model_parallel_rank() == 0:
        data_iterator = create_pretrain_data_iterator(seq_length, micro_batch_size=micro_batch_size, create_attention_mask=create_attention_mask)

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
    assert tokens.shape == (micro_batch_size, seq_len_per_rank), f"Expected tokens shape ({micro_batch_size}, {seq_len_per_rank}), got {tokens.shape}"
    assert labels.shape == (micro_batch_size, seq_len_per_rank), f"Expected labels shape ({micro_batch_size}, {seq_len_per_rank}), got {labels.shape}"
    assert loss_mask.shape == (micro_batch_size, seq_len_per_rank), f"Expected loss_mask shape ({micro_batch_size}, {seq_len_per_rank}), got {loss_mask.shape}"
    assert position_ids.shape == (micro_batch_size, seq_len_per_rank), f"Expected position_ids shape ({micro_batch_size}, {seq_len_per_rank}), got {position_ids.shape}"

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
        assert attention_mask.shape == (micro_batch_size, 1, seq_len_per_rank, seq_length), \
            f"Expected attention_mask shape ({micro_batch_size}, 1, {seq_len_per_rank}, {seq_length}), got {attention_mask.shape}"
        assert attention_mask.dtype == torch.bool
    else:
        assert attention_mask is None

    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def create_hybrid_cp_data_iterator(seq_length: int = 1024, cp_size: int = 1):
    # Pack n_seqs equal-length sequences; total length must be divisible by 2 * cp_size for CP splitting
    n_seqs = max(2, 2 * cp_size)
    align = max(1, 2 * cp_size)
    seq_len_each = (seq_length // n_seqs // align) * align
    if seq_len_each == 0:
        seq_len_each = align
    total_seq_len = n_seqs * seq_len_each

    text = torch.randint(0, 10000, (1, total_seq_len + 1), dtype=torch.int64)
    tokens = text[:, :-1].contiguous()       # (1, total_seq_len)
    labels = text[:, 1:].contiguous()        # (1, total_seq_len)
    loss_mask = torch.ones((1, total_seq_len), dtype=torch.float32)
    position_ids = torch.cat([
        torch.arange(seq_len_each, dtype=torch.int64) for _ in range(n_seqs)
    ]).unsqueeze(0)  # (1, total_seq_len)

    cu_seqlens = torch.cat([
        torch.zeros(1, dtype=torch.int32),
        torch.cumsum(torch.tensor([seq_len_each] * n_seqs, dtype=torch.int64), dim=0).to(torch.int32),
    ])
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
@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("seq_length", [1024])
@pytest.mark.parametrize("create_attention_mask", [False])
def test_hybrid_cp_batch(tp_size, cp_size, seq_length, create_attention_mask):
    if cp_size * tp_size > torch.cuda.device_count():
        pytest.skip(f"Skipping test because cp_size * tp_size > torch.cuda.device_count() ({cp_size * tp_size} > {torch.cuda.device_count()})")

    initialize_test_environment(tp_size, cp_size, seq_length, 1, 16, sft=False, hybrid_context_parallel=True, create_attention_mask=create_attention_mask)

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
    assert attention_mask is None          # HybridCP does not use attention mask from dataloader
    assert cu_seqlens is not None          # HybridCP always has cu_seqlens
    assert max_seqlen is not None          # HybridCP always has max_seqlen
    assert local_cp_size is not None       # HybridCP always has local_cp_size

    # Data iterator parameters (must match create_hybrid_cp_data_iterator)
    n_seqs = max(2, 2 * cp_size)
    align = max(1, 2 * cp_size)
    seq_len_each = (seq_length // n_seqs // align) * align
    if seq_len_each == 0:
        seq_len_each = align
    total_seq_len = n_seqs * seq_len_each  # equals seq_length for the test parameters

    # Shape: HybridCP CP splitting gives total_seq_len // cp_size tokens per rank
    seq_len_per_rank = total_seq_len // cp_size
    assert tokens.shape == (1, seq_len_per_rank), f"Expected tokens shape (1, {seq_len_per_rank}), got {tokens.shape}"
    assert labels.shape == (1, seq_len_per_rank), f"Expected labels shape (1, {seq_len_per_rank}), got {labels.shape}"
    assert loss_mask.shape == (1, seq_len_per_rank), f"Expected loss_mask shape (1, {seq_len_per_rank}), got {loss_mask.shape}"
    assert position_ids.shape == (1, seq_len_per_rank), f"Expected position_ids shape (1, {seq_len_per_rank}), got {position_ids.shape}"

    # Dtype checks
    assert tokens.dtype == torch.int64
    assert labels.dtype == torch.int64
    assert loss_mask.dtype == torch.float32
    assert position_ids.dtype == torch.int64

    # Loss mask is all-ones (no masking in the HybridCP pretrain dataloader)
    assert loss_mask.sum().item() == seq_len_per_rank

    # cu_seqlens: 1D int32, [0, seq_len_each, 2*seq_len_each, ..., total_seq_len]
    assert cu_seqlens.shape == (n_seqs + 1,), f"Expected cu_seqlens shape ({n_seqs + 1},), got {cu_seqlens.shape}"
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

    if torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    test_pretrain_batch(1, 2, 1024, True, 2)
