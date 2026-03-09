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

from megatron.training.datasets.data_samplers import HybridCPMegatronPretrainingSampler

from sft_mamba import get_batch

import torch

from torch.utils.data import DataLoader, Dataset

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
    #
    args.vocab_size = 1024
    args.tokenizer_type = "NullTokenizer"

    #
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

    assert sum(lengths) < max_seq_length, f"Sum of lengths {sum(lengths)} is greater than max_seq_length {max_seq_length}"
    text = torch.randint(0, 10000, (1, sum(lengths) + 1), dtype=torch.int64)
    tokens = text[:, :-1].contiguous()
    labels = text[:, 1:].contiguous()

    cu_seqlens = torch.cat((
        torch.zeros(1, dtype=torch.int32),
        torch.cumsum(torch.tensor(lengths, dtype=torch.int64), dim=0).to(torch.int32),
    ))

    batch = {"tokens": tokens, "labels": labels, "cu_seqlens": cu_seqlens}
    return iter([batch])

@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("seq_length", [1024, 2048, 4096])
def test_sft_batch(tp_size, cp_size, seq_length):
    if cp_size * tp_size > torch.cuda.device_count():
        pytest.skip(f"Skipping test because cp_size * tp_size > torch.cuda.device_count() ({cp_size * tp_size} > {torch.cuda.device_count()})")
    
    global_batch_size = int(os.environ.get("WORLD_SIZE", 1)) // (tp_size * cp_size)
    initialize_test_environment(tp_size, cp_size, seq_length, micro_batch_size=1, global_batch_size=global_batch_size, sft=True)

    data_iterator = None
    if mpu.get_tensor_model_parallel_rank() == 0: # NOTE(asolergi-nv): Only create data iterator on TP rank 0
        data_iterator = create_sft_data_iterator(seq_length)
    
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

    assert tokens is not None
    assert labels is not None
    assert loss_mask is not None
    assert position_ids is not None
    assert cu_seqlens is not None
    assert max_seqlen is not None
    assert attention_mask is None
    assert hybrid_cp_group is None
    assert local_cp_size is None

    if cp_size > 1:
        assert cu_seqlens_padded is not None
        if os.environ.get("RANK") == "0":
            print(cu_seqlens_padded)
            print(cu_seqlens)
    else:
        assert cu_seqlens_padded is None
    
    if torch.distributed.is_initialized():
            torch.distributed.barrier()

def create_pretrain_data_iterator(seq_length: int = 1024, micro_batch_size: int = 1, create_attention_mask: bool = False):
    text = torch.randint(0, 10000, (micro_batch_size, seq_length + 1), dtype=torch.int64)
    tokens = text[:, :-1].contiguous()
    labels = text[:, 1:].contiguous()
    position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
    loss_mask = torch.ones((1, seq_length), dtype=torch.float)

    batch = {"tokens": tokens, "labels": labels, "loss_mask": loss_mask, "position_ids": position_ids}

    if create_attention_mask:
        batch["attention_mask"] = torch.tril(
            torch.ones((seq_length, seq_length))
        ).unsqueeze(0).bool()
    
    return iter([batch])

@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("seq_length", [1024, 2048, 4096])
@pytest.mark.parametrize("create_attention_mask", [True, False])
def test_pretrain_batch(tp_size, cp_size, seq_length, create_attention_mask):
    if cp_size * tp_size > torch.cuda.device_count():
        pytest.skip(f"Skipping test because cp_size * tp_size > torch.cuda.device_count() ({cp_size * tp_size} > {torch.cuda.device_count()})")
    global_batch_size = int(os.environ.get("WORLD_SIZE", 1)) // (tp_size * cp_size)
    initialize_test_environment(tp_size, cp_size, seq_length, 1, global_batch_size=global_batch_size, sft=False, create_attention_mask=create_attention_mask)

    data_iterator = None
    if mpu.get_tensor_model_parallel_rank() == 0: # NOTE(asolergi-nv): Only create data iterator on TP rank 0
        data_iterator = create_pretrain_data_iterator(seq_length, create_attention_mask=create_attention_mask)
    
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

    assert tokens is not None
    assert labels is not None   
    assert loss_mask is not None
    assert position_ids is not None
    if create_attention_mask:
        assert attention_mask is not None
    else:
        assert attention_mask is None

    assert cu_seqlens is None
    assert cu_seqlens_padded is None
    assert max_seqlen is None
    assert hybrid_cp_group is None
    assert local_cp_size is None
    
    if torch.distributed.is_initialized():
            torch.distributed.barrier()

def create_hybrid_cp_data_iterator(seq_length: int = 1024, number_of_samples: int = 100, global_batch_size: int = 16, micro_batch_size: int = 1):
    # TODO(asolergi-nv): Implement HybridCP data iterator when it's ready
    return None


@pytest.mark.skip(reason="deactivated")
@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("seq_length", [1024])
@pytest.mark.parametrize("create_attention_mask", [False])
def test_hybrid_cp_batch(tp_size, cp_size, seq_length, create_attention_mask):
    if cp_size * tp_size > torch.cuda.device_count():
        pytest.skip(f"Skipping test because cp_size * tp_size > torch.cuda.device_count() ({cp_size * tp_size} > {torch.cuda.device_count()})")

    initialize_test_environment(tp_size, cp_size, seq_length, 1, 16, sft=False, hybrid_context_parallel=True, create_attention_mask=create_attention_mask)

    data_iterator = None
    if mpu.get_tensor_model_parallel_rank() == 0: # NOTE(asolergi-nv): Only create data iterator on TP rank 0
        data_iterator = create_hybrid_cp_data_iterator(seq_length)
    
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

    assert tokens is not None
    assert labels is not None   
    assert loss_mask is not None
    assert position_ids is not None
    if create_attention_mask:
        assert attention_mask is not None
    else:
        assert attention_mask is None

    assert cu_seqlens is None
    assert cu_seqlens_padded is None
    assert max_seqlen is None
    assert hybrid_cp_group is None
    assert local_cp_size is None
    
    if torch.distributed.is_initialized():
            torch.distributed.barrier()
    
if __name__ == "__main__":
    # test_pretrain_batch(2, 1, 1024, True)
    test_hybrid_cp_batch(1, 2, 1024, False)