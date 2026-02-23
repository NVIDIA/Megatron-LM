# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
"""SFT Mamba."""

# Capture the true program start time BEFORE any heavy imports.
import time
_PROGRAM_START_TIME = time.time()

import json

# Suppress warnings on all ranks but rank 0.
import os
import warnings
rank = int(os.environ.get('RANK', 0))
if rank != 0:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

from functools import partial
from typing import List, Optional, Tuple

import torch

from mamba_builders import mamba_builder
from megatron.core import mpu
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import get_context_parallel_group
from megatron.core.models.mamba import MambaModel
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.utils import get_attr_wrapped_model, StragglerDetector
from megatron.training import (
    get_args,
    get_timers,
    get_tokenizer,
    inprocess_restart,
    pretrain,
    print_rank_0,
    set_startup_timestamps,
)
from megatron.training.datasets.sft_dataset import SFTDataset, IGNORE_INDEX
from megatron.training.utils import (
    get_sft_batch_on_this_cp_rank,
    get_sft_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
    is_first_or_last_pipeline_stage,
    pad_or_truncate_thd_tensors, 
)
from model_provider import model_provider

try:
    from megatron.post_training.arguments import add_modelopt_args
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt
    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

stimer = StragglerDetector()

from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import pad_thd_sequences_for_cp

# TODO(asolergi-nv): Develop tests!
# TODO(asolergi-nv): Develop pretokenized format
# TODO(asolergi-nv): Para esto si hacemos truncar --> Facil, tal y como lo tenemos ahora pero forzando siempre empezar las samples por el principio. Si deseamos no truncar tendremos que saltar a la siguiente sample una vez la siguiente sample + actual > seqlen
# TODO(asolergi-nv): Remove chat templates & refactor SFTTokenizer
# TODO(asolergi-nv): Craft docs with expected inputs & formats
def get_batch(data_iterator, vp_stage=None):
    """Generate a batch."""

    if not is_first_or_last_pipeline_stage(vp_stage): # TODO(asolergi-nv): Add HybridCP condition & MTP?
        return None, None, None, None, None, None

    args = get_args()

    cp_size = args.context_parallel_size
    tp_size = args.tensor_model_parallel_size
    sp = args.sequence_parallel
    max_seq_len = args.seq_length

    batch = {
        'tokens': None,
        'labels': None,
        'loss_mask': None,
        'cu_seqlens': None,
        'cu_seqlens_padded': None,
        'max_seqlen': None,
    }

    if mpu.get_tensor_model_parallel_rank() == 0: 
        # NOTE(asolergi-nv): Preprocessing step only on TP rank 0 before sharing with other ranks
        ### 1. Create cu_seqlens_padded if CP is enabled
        ### 2. Pad or truncate tensors if necessary
        ### 3. Create max_seqlen
        # From this point, all tensors have shape [1, sequence_length] (Except for cu_seqlens, cu_seqlens_padded and max_seqlen)
        
        # TODO(asolergi-nv): Create position_ids

        assert data_iterator is not None
        data = next(data_iterator)

        tokens, labels, cu_seqlens = data["tokens"].squeeze(0), data["labels"].squeeze(0), data["cu_seqlens"].squeeze(0) # NOTE(asolergi-nv): PyTorch DataLoader `default_collate` adds batch dimension, so we need to remove it since TE expects cu_seqlens to be 1D

        # NOTE(asolergi-nv): This is performed here https://github.com/NVIDIA-NeMo/RL/blob/2841fefb699a460cc4375fb2983b40c018ca76fe/nemo_rl/models/megatron/data.py#L393C5-L408C77
        # individual sequence needs to be splitted to CP domain, and to TP domain when SP is enabled.
        if cp_size > 1:
            divisibility_factor = 1
            if cp_size > 1:
                divisibility_factor *= cp_size * 2
            if tp_size > 1 and sp:
                divisibility_factor *= tp_size

            tokens, labels, cu_seqlens_padded = pad_thd_sequences_for_cp(
                    tokens,
                    labels,
                    cu_seqlens,
                    divisibility_factor,
                    padding_token_id=get_tokenizer().pad,
                    padding_label_id=IGNORE_INDEX,
                )
            cu_seqlens_padded = cu_seqlens_padded.to(torch.int32) # NOTE(asolergi-nv): pad_thd_sequences_for_cp uses torch.cumsum which promotes int32 cu_seqlens_padded to int64
        else:
            cu_seqlens_padded = None

        tokens, labels, cu_seqlens, cu_seqlens_padded = pad_or_truncate_thd_tensors(tokens, labels, cu_seqlens, cu_seqlens_padded, max_seq_len, get_tokenizer().pad, IGNORE_INDEX)

        # Loss mask.
        loss_mask = torch.ones(max_seq_len, dtype=torch.float32)
        loss_mask[labels == get_tokenizer().pad] = 0.0  # NOTE(asolergi-nv): Mask paddings
        loss_mask[labels == IGNORE_INDEX] = 0.0  # NOTE(asolergi-nv): Mask prompts 
        print(f"Trained tokens: {sum(loss_mask)} ({sum(loss_mask) / max_seq_len * 100:.2f}%), padding tokens: {sum(tokens == get_tokenizer().pad)} ({sum(tokens == get_tokenizer().pad) / max_seq_len * 100:.2f}%)")
        
        # NOTE(asolergi-nv): max_seqlen is computed here https://github.com/NVIDIA-NeMo/RL/blob/2841fefb699a460cc4375fb2983b40c018ca76fe/nemo_rl/models/megatron/data.py#L423-L429
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1] if cu_seqlens_padded is None else cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
        max_seqlen = torch.tensor(
            [seq_lens.max().item()],
            dtype=torch.int32,
            )
        
        batch = {
            'tokens': tokens.unsqueeze(0).cuda(non_blocking=True), # NOTE(asolergi-nv): Add back batch dimension
            'labels': labels.unsqueeze(0).cuda(non_blocking=True), # NOTE(asolergi-nv): Add back batch dimension
            'loss_mask': loss_mask.unsqueeze(0).cuda(non_blocking=True), # NOTE(asolergi-nv): Add batch dimension
            'cu_seqlens': cu_seqlens.cuda(non_blocking=True),
            'cu_seqlens_padded': cu_seqlens_padded.cuda(non_blocking=True),
            'max_seqlen': max_seqlen.cuda(non_blocking=True),
        }

    batch = get_sft_batch_on_this_tp_rank(batch) # TODO(asolergi-nv): Add mtp_on_this_rank condition?
    batch = get_sft_batch_on_this_cp_rank(batch, get_context_parallel_group())

    return batch.values()


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10

def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[MambaModel] = None):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()
    if has_nvidia_modelopt and getattr(args, 'modelopt_enabled', False):  # [ModelOpt]
        loss, num_tokens, report = loss_func_modelopt(loss_mask, output_tensor, model=model)
    else:
        losses = output_tensor.view(-1).float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses * loss_mask)

        num_tokens = loss_mask.sum().clone().detach().to(torch.int)
        report = {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are deterministic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are deterministic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,        # forward pass calculations are deterministic
            fatal=False,
        )

    return loss, num_tokens, report


def forward_step(data_iterator, model: MambaModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (MambaModel): The Mamba Model
    """
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()

    global stimer

    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        (
            tokens,
            labels,
            loss_mask,
            cu_seqlens,
            cu_seqlens_padded,
            max_seqlen,
        ) = get_batch(data_iterator, vp_stage)

    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )

    timers('batch-generator').stop()

    print(f"Shapes! tokens: {tokens.shape}, labels: {labels.shape}, loss_mask: {loss_mask.shape}, cu_seqlens: {cu_seqlens.squeeze(0).shape}, cu_seqlens_padded: {cu_seqlens_padded.squeeze(0).shape}, max_seqlen: {max_seqlen.shape}")
    # Shapes! tokens: torch.Size([1, 16384]), labels: torch.Size([1, 16384]), loss_mask: torch.Size([1, 16384]), cu_seqlens: torch.Size([2]), cu_seqlens_padded: torch.Size([2]), max_seqlen: torch.Size([1])
    # Shapes! tokens: [batch, seq_len // CP], labels: [batch, seq_len // CP], loss_mask: [batch, seq_len // CP], cu_seqlens: [Number of sequences], cu_seqlens_padded: [Number of sequences], max_seqlen: [1]  
    
    with stimer:
        output_tensor = model(
            tokens,
            None, # position_ids
            None, # attention_mask
            labels=labels,
            packed_seq_params=packed_seq_params,
            loss_mask=loss_mask
        )

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model)


def is_dataset_built_on_rank(vp_stage=None, is_packed_sequence=False):
    if mpu.get_tensor_model_parallel_rank() != 0:
        return False
    elif is_packed_sequence:
        return True
    else:
        return is_first_or_last_pipeline_stage(vp_stage)


def core_gpt_dataset_config_from_args(args):
    tokenizer = build_tokenizer(args)

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    sequences_per_dataset = None
    if args.per_dataset_sequences_path is not None:
        with open(args.per_dataset_sequences_path, "r") as f:
            sequences_per_dataset = json.load(f)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        object_storage_cache_path=args.object_storage_cache_path,
        mid_level_dataset_surplus=args.mid_level_dataset_surplus,
        allow_ambiguous_pad_tokens=args.allow_ambiguous_pad_tokens,
        fast_cache_load=args.dataloader_fast_cache_load,
        sequences_per_dataset=sequences_per_dataset,
        defer_npy_index_mmap=args.dataloader_defer_npy_index_mmap,
        context_parallel_size=args.context_parallel_size,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()
    config = core_gpt_dataset_config_from_args(args)

    is_packed_sequence = False
    if args.sft:
        dataset_type = SFTDataset
        is_packed_sequence = True  # SFT always uses packed sequence
    else:
        if args.mock_data:
            dataset_type = MockGPTDataset
        else:
            dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        partial(is_dataset_built_on_rank, vp_stage=vp_stage, is_packed_sequence=is_packed_sequence),
        config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    # Timestamp right after entering __main__ block (after all imports/library setup)
    _MAIN_ENTRY_TIME = time.time()

    # Register startup timestamps for timing report in pretrain()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(train_valid_test_datasets_provider,
             partial(model_provider, mamba_builder),
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
             store=store,
             extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
             )
