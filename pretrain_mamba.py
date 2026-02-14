# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain and SFT Mamba."""

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
from megatron.core.parallel_state import (
    get_context_parallel_rank,
    get_context_parallel_world_size,
)
from megatron.core.models.mamba import MambaModel
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.utils import get_attr_wrapped_model, is_te_min_version, StragglerDetector
from megatron.training import (
    get_args,
    get_timers,
    inprocess_restart,
    pretrain,
    print_rank_0,
    set_startup_timestamps,
)
from megatron.training.datasets.sft_dataset import SFTDataset
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
    is_first_or_last_pipeline_stage,
)
from model_provider import model_provider

try:
    from megatron.post_training.arguments import add_modelopt_args
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt
    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

try:
    # Register the TE CUDA kernels
    import transformer_engine  # pylint: disable=unused-import

    # Alias the PyTorch wrapper so we can call tex.* APIs
    import transformer_engine_torch as tex
except ImportError:
    # TE isn’t installed or the torch wrapper is missing
    tex = None

stimer = StragglerDetector()


def get_batch(data_iterator, vp_stage=None):
    """Generate a batch."""

    empty_batch = {
        'tokens': None,
        'labels': None,
        'loss_mask': None,
        'attention_mask': None,
        'position_ids': None,
        'cu_seqlens': None,
        'max_seqlen': None,
    }

    # TODO(duncan): Is there a more efficient way to access is_packed_sequence here?
    is_packed_sequence = get_args().sft  # SFT always uses packed sequence
    if not is_first_or_last_pipeline_stage(vp_stage) and not is_packed_sequence:
        return empty_batch.values()

    batch = get_batch_on_this_tp_rank(data_iterator)
    
    cu_seqlens = batch['cu_seqlens']
    # Unused at the moment
    cu_seqlens_padded = batch.pop('cu_seqlens_padded', None)
    # Support for Hybrid Context Parallel (Unused in this script)
    local_cp_size = batch.pop('local_cp_size', None)

    if cu_seqlens is not None:
        assert (
            cu_seqlens.dim() == 2 and cu_seqlens.shape[0] == 1
        ), "micro-batch-size must be 1 for packing"
        cu_seqlens = cu_seqlens[0]
        batch['cu_seqlens'] = cu_seqlens

        max_seqlen = batch['max_seqlen']
        assert max_seqlen.dim() == 1
        # TODO(duncan): can this be kept as a 0-D tensor?
        batch['max_seqlen'] = int(max_seqlen[0].item())

    if mpu.is_pipeline_first_stage(ignore_virtual=(vp_stage is None), vp_stage=vp_stage):
        total_tokens = batch['tokens'].size(1)
    elif mpu.is_pipeline_last_stage(ignore_virtual=(vp_stage is None), vp_stage=vp_stage):
        total_tokens = batch['labels'].size(1)
    else:  # packed sequence
        empty_batch['cu_seqlens'] = cu_seqlens
        empty_batch['max_seqlen'] = max_seqlen
        return empty_batch.values()

    if cu_seqlens is None:
        # slice batch along sequence dimension for context parallelism
        batch = get_batch_on_this_cp_rank(batch)  # The implementation of this function is in MCore
    else:  # Packed THD format
        cp_size = get_context_parallel_world_size()
        if cp_size > 1:  # slice batch along sequence dimension for context parallelism
            assert tex is not None and is_te_min_version("1.10.0"), (
                "Please update Transformer Engine to >= 1.10 to use "
                "Context Parallel with THD format data"
            )
            cp_rank = get_context_parallel_rank()
            index = tex.thd_get_partitioned_indices(
                cu_seqlens,
                total_tokens,
                cp_size,
                cp_rank,
            )
            for key, data in batch.items():
                if key in {'attention_mask', 'cu_seqlens', 'max_seqlen'}:
                    continue
                if data is not None:
                    # On first PP rank, labels and loss_mask can be None.
                    # On last PP rank, tokens and position_ids can be None.
                    batch[key] = data.index_select(1, index)

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
        model (MambaModel): The GPT Model
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
            attention_mask,
            position_ids,
            cu_seqlens,
            max_seqlen,
        ) = get_batch(data_iterator, vp_stage)

    if cu_seqlens is None:
        packed_seq_params = None
    else:
        # Pre-compute seq_idx for Mamba mixer CUDA graph compatibility.
        # total_tokens must be the actual tensor sequence dimension (not max_seqlen,
        # which is the max *individual* sequence length per TE convention).
        total_tokens = tokens.size(1) if tokens is not None else labels.size(1)
        total_tokens_tensor = torch.tensor(
            [total_tokens], dtype=cu_seqlens.dtype, device=cu_seqlens.device
        )
        cu_seqlens_with_max = torch.cat([cu_seqlens, total_tokens_tensor])
        seq_lengths = cu_seqlens_with_max[1:] - cu_seqlens_with_max[:-1]
        seq_idx = torch.repeat_interleave(
            torch.arange(seq_lengths.numel(), device=cu_seqlens.device), seq_lengths
        ).to(torch.int32).unsqueeze(0)

        # TODO(duncan): This class seems overly complex for what needs to be conveyed
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=None,
            cu_seqlens_kv_padded=None,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
        )
        # Override __post_init__'s seq_idx (which used max_seqlen_q) with the
        # correct value computed from the actual tensor sequence dimension.
        packed_seq_params.seq_idx = seq_idx

    timers('batch-generator').stop()

    with stimer:
        output_tensor = model(
            tokens,
            position_ids,
            attention_mask,
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
