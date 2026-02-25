import sys
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
    tp_size, cp_size, seq_length, sft, micro_batch_size
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
    args.sft = sft
    args.micro_batch_size = micro_batch_size

    #
    args.vocab_size = 1024
    args.tokenizer_type = "NullTokenizer"

    #
    args.num_layers = 4
    args.hidden_size = 512
    args.num_attention_heads = 8
    args.max_position_embeddings = seq_length
    #args.vocab_size = 1024
    
    #args.create_attention_mask_in_dataloader = True
    #args.train_iters = 10
    #args.lr = 3e-5
    #args.bf16 = True
    #args.add_bias_linear = False

    validate_args(args)
    set_global_variables(args, True)

    Utils.initialize_model_parallel(tensor_model_parallel_size=tp_size, context_parallel_size=cp_size)
    return args

def create_data_iterator(seq_length, sft, create_attention_mask: bool = False):
    text = torch.randint(0, 10000, (1, seq_length + 1), dtype=torch.int64)
    tokens = text[:, :-1].contiguous()
    labels = text[:, 1:].contiguous()
    position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)

    batch = {"tokens": tokens, "labels": labels, "position_ids": position_ids}

    if sft:
        # Create a list of multiple lengths that sum to > seq_length
        # Let's pick segment lengths of around seq_length // 3, repeat until sum is > seq_length
        lengths = []
        total = 0
        base = max(1, seq_length // 3)
        while total <= seq_length:
            l = base
            # For the last one, just make sure we go over seq_length but not by a ton
            if total + l > seq_length:
                l = seq_length - total + 1
            lengths.append(l)
            total += l
        batch["cu_seqlens"] = torch.cat((torch.zeros(1, dtype=torch.int32), torch.cumsum(torch.tensor(lengths, dtype=torch.int64), dim=0).to(torch.int32)))  # NOTE(asolergi-nv): torch.cumsum promotes int32 to int64

    else:
        batch["loss_mask"] = torch.ones(seq_length, dtype=torch.float)
        if create_attention_mask:
            batch["attention_mask"] = torch.tril(
                torch.ones((seq_length, seq_length))
            ).unsqueeze(0).bool()
    
    return iter([batch])


"""
@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("seq_length", [1024, 2048, 4096])
"""

@pytest.mark.parametrize("tp_size", [1])
@pytest.mark.parametrize("cp_size", [1])
@pytest.mark.parametrize("seq_length", [1024])
@pytest.mark.parametrize(
    "sft,create_attention_mask",
    [
        (True, False),   # Combi 1: sft True, create_attention_mask False
        (False, True),   # Combi 2: sft False, create_attention_mask True
        (False, False),  # Combi 3: sft False, create_attention_mask False
    ],
    ids=["sft", "pretraining_create_attention_mask_in_dataloader_True", "pretraining_create_attention_mask_in_dataloader_False"]
)
def test_thd_layout(tp_size, cp_size, seq_length, sft, create_attention_mask):
    initialize_test_environment(tp_size, cp_size, seq_length, sft, 1)

    data_iterator = None
    if mpu.get_tensor_model_parallel_rank() == 0: # NOTE(asolergi-nv): Only create data iterator on TP rank 0
        data_iterator = create_data_iterator(seq_length, sft, create_attention_mask=create_attention_mask)
    
    (
        tokens,
        labels,
        loss_mask,
        position_ids,
        attention_mask,
        cu_seqlens,
        cu_seqlens_padded,
        max_seqlen,
    ) = get_batch(data_iterator)

    #print(f"Shapes! tokens: {tokens.shape}, labels: {labels.shape}, loss_mask: {loss_mask.shape}, cu_seqlens: {cu_seqlens.shape}, max_seqlen: {max_seqlen.shape}") # cu_seqlens_padded: {cu_seqlens_padded.shape}

    
