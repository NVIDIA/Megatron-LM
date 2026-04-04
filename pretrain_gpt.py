# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pretrain and SFT GPT."""

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

from gpt_builders import gpt_builder
from megatron.core import parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.datasets.data_schedule import get_batch_on_this_rank_for_sequence_packing
from megatron.core.enums import ModelType
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.models.gpt import GPTModel
from megatron.core.models.engram.plugin import add_engram_args
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.utils import get_attr_wrapped_model, get_thd_batch_on_this_cp_rank, get_batch_on_this_dynamic_cp_rank, StragglerDetector
from megatron.training import (
    get_args,
    get_timers,
    inprocess_restart,
    pretrain,
    print_rank_0,
    set_startup_timestamps,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.datasets.sft_dataset import SFTDataset
from megatron.core.transformer.multi_token_prediction import mtp_on_this_rank, get_mtp_ranks
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.datasets.fim_dataset import GPTFIMDataset, GPTFIMDatasetConfig
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
    is_first_or_last_pipeline_stage,
)
from megatron.training.datasets.sft_dataset import SFTDataset, MockSFTDataset
from model_provider import model_provider

try:
    from megatron.post_training.arguments import add_modelopt_args
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

stimer = StragglerDetector()


def add_gpt_extra_args(parser):
    parser = add_engram_args(parser)
    if has_nvidia_modelopt:
        parser = add_modelopt_args(parser)
    return parser


def get_batch(data_iterator, vp_stage: Optional[int] = None):
    """Generate a batch.

    Packed sequence support (SFT / ``--sft`` flag):
        When ``args.sft`` is True, the dataset emits THD-format batches where
        multiple sequences are concatenated into a single flat token tensor.
        The batch includes ``cu_seqlens`` (cumulative sequence lengths, shape
        ``[1, S+1]``) and ``max_seqlen`` (shape ``[1]``) that describe the
        individual sequence boundaries.

        This function validates and squeezes those fields:
          - ``cu_seqlens``:  asserted to have shape ``[1, S+1]`` (micro-batch
            size must be 1 for packing), then squeezed to ``[S+1]``.
          - ``max_seqlen``:  asserted to be 1-D; kept as a tensor and passed
            to ``get_thd_batch_on_this_cp_rank`` which performs the final
            scalar conversion internally.

        Pipeline stage handling:
          - First/last PP stages: fetch the full batch (tokens + labels) and
            route through ``get_thd_batch_on_this_cp_rank`` to produce a
            ``PackedSeqParams`` object that carries ``cu_seqlens`` and
            ``max_seqlen`` to the attention kernel.
          - Middle PP stages: only ``cu_seqlens`` and ``max_seqlen`` are
            needed for attention masking; all other fields are returned as
            ``None`` with a ``PackedSeqParams`` built directly here.
          - MTP ranks (``mtp_on_this_rank``) also receive the full batch,
            regardless of pipeline stage.

        Difference from ``pretrain_mamba.py``:
          - Return format: GPT returns a 6-tuple
            ``(tokens, labels, loss_mask, attention_mask, position_ids,
            packed_seq_params)`` where ``packed_seq_params`` is a
            ``PackedSeqParams`` dataclass.  Mamba returns 7 values via
            ``batch.values()`` with ``cu_seqlens`` and ``max_seqlen`` as
            separate dict entries (no ``PackedSeqParams`` wrapper).
          - Middle-stage return: GPT returns ``(None×5, PackedSeqParams)``;
            Mamba returns an ``empty_batch`` dict with ``cu_seqlens`` and
            ``max_seqlen`` set.
          - CP with packed sequences: GPT delegates to
            ``get_thd_batch_on_this_cp_rank`` (MCore utility); Mamba
            implements the ``tex.thd_get_partitioned_indices`` CP slicing
            inline and does not call that helper.
          - MTP: GPT passes ``mtp_on_this_rank`` to ``get_batch_on_this_tp_rank``
            and uses it to gate the early-return; Mamba has no MTP support.
          - ``max_seqlen`` conversion: Mamba converts to a Python int scalar
            before returning (``int(max_seqlen[0].item())``); GPT keeps it as
            a tensor and lets ``get_thd_batch_on_this_cp_rank`` convert it,
            except for the middle-stage ``PackedSeqParams`` where conversion
            is done inline.
    """
    args = get_args()
    config = core_transformer_config_from_args(args)

    if args.sequence_packing_scheduler is not None:
        return get_batch_on_this_rank_for_sequence_packing(
            data_iterator,
            vpp_size=config.virtual_pipeline_model_parallel_size,
            mtp_on_this_rank=mtp_on_this_rank(config, ignore_virtual=False, vp_stage=vp_stage),
            vp_stage=vp_stage,
        )

    # TODO: this is pretty hacky, find a better way
    is_packed_sequence = get_args().sft  # SFT always uses packed sequence
    if not is_first_or_last_pipeline_stage(vp_stage) and not is_packed_sequence and (
    (not mtp_on_this_rank(config, ignore_virtual=False, vp_stage=vp_stage))):
        return None, None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(
        data_iterator,
        mtp_on_this_rank=mtp_on_this_rank(config, ignore_virtual=False, vp_stage=vp_stage)
        )

    cu_seqlens = batch.pop('cu_seqlens', None)
    cu_seqlens_padded = batch.pop('cu_seqlens_padded', None)
    max_seqlen = batch.pop('max_seqlen', None)
    local_cp_size = batch.pop('local_cp_size', None)
    if local_cp_size is not None:
        local_cp_size = int(local_cp_size.item())

    if cu_seqlens is not None:
        assert (
            cu_seqlens.dim() == 2 and cu_seqlens.shape[0] == 1
        ), "micro-batch-size must be 1 for packing"
        cu_seqlens = cu_seqlens[0]
        assert max_seqlen.dim() == 1

    # For middle pipeline stages with packed sequences, only cu_seqlens and
    # max_seqlen are needed (for attention masking); skip the full batch.
    if not is_first_or_last_pipeline_stage(vp_stage) and is_packed_sequence:
        return None, None, None, None, None, PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=int(max_seqlen[0].item()),
            max_seqlen_kv=int(max_seqlen[0].item()),
            qkv_format='thd',
        )

    if cu_seqlens is None and local_cp_size is None:
        # slice batch along sequence dimension for context parallelism
        batch = get_batch_on_this_cp_rank(batch)  # The implementation of this function is in MCore
        packed_seq_params = None
    elif local_cp_size is None:  # Packed THD format
        batch, packed_seq_params = get_thd_batch_on_this_cp_rank(batch, cu_seqlens, cu_seqlens_padded, max_seqlen)
    else: # Dynamic CP format
        batch, packed_seq_params = get_batch_on_this_dynamic_cp_rank(batch, local_cp_size)
    
    return (*batch.values(), packed_seq_params)


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(
    loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[GPTModel] = None
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        model (GPTModel, optional): The model (can be wrapped)

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
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
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
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    return loss, num_tokens, report


def forward_step(data_iterator, model: GPTModel, return_schedule_plan: bool = False):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = get_batch(data_iterator, vp_stage)
    timers('batch-generator').stop()

    with stimer:
        if args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
        else:
            if return_schedule_plan:
                assert args.overlap_moe_expert_parallel_comm, \
                    "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
                schedule_plan = model.build_schedule_plan(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )
                return schedule_plan, partial(loss_func, loss_mask, model=model)
            else:
                output_tensor = model(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask, packed_seq_params=packed_seq_params
                )

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model)


def is_dataset_built_on_rank(vp_stage=None, is_packed_sequence=False):
    args = get_args()
    config = core_transformer_config_from_args(args)
    if parallel_state.get_tensor_model_parallel_rank() != 0:
        return False
    elif is_packed_sequence:
        return True
    return (
        is_first_or_last_pipeline_stage(vp_stage)
        or mtp_on_this_rank(config, ignore_virtual=False, vp_stage=vp_stage)
    )


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

    data_args = {
        "random_seed": args.seed,
        "sequence_length": args.seq_length,
        "blend": blend,
        "blend_per_split": blend_per_split,
        "split": args.split,
        "multiple_validation_sets": args.multiple_validation_sets,
        "full_validation": args.full_validation,
        "num_dataset_builder_threads": args.num_dataset_builder_threads,
        "path_to_cache": args.data_cache_path,
        "mmap_bin_files": args.mmap_bin_files,
        "tokenizer": tokenizer,
        "reset_position_ids": args.reset_position_ids,
        "reset_attention_mask": args.reset_attention_mask,
        "eod_mask_loss": args.eod_mask_loss,
        "create_attention_mask": args.create_attention_mask_in_dataloader,
        "object_storage_cache_path": args.object_storage_cache_path,
        "mid_level_dataset_surplus": args.mid_level_dataset_surplus,
        "allow_ambiguous_pad_tokens": args.allow_ambiguous_pad_tokens,
        "fast_cache_load": args.dataloader_fast_cache_load,
        "sequences_per_dataset": sequences_per_dataset,
        "defer_npy_index_mmap": args.dataloader_defer_npy_index_mmap,
        "context_parallel_size": args.context_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "sequence_parallel_size": args.tensor_model_parallel_size*args.sequence_parallel,
        "dynamic_context_parallel": args.dynamic_context_parallel,
        "sft_mock_dataset_config_json":args.sft_mock_dataset_config_json,
    }

    # add FIM args to the config
    if args.fim_data:
        extra_tokens = {
            "prefix": args.fim_prefix_token,
            "middle": args.fim_middle_token,
            "suffix": args.fim_suffix_token,
            "pad": args.fim_pad_token,
            "eod": args.fim_eod_token,
        }
        data_args.update(
            {
                "fim_rate": args.fim_rate,
                "fim_spm_rate": args.fim_spm_rate,
                "fim_extra_tokens": extra_tokens,
                "fim_split_sample": args.fim_split_sample,
                "fim_fragment_rate": args.fim_fragment_rate,
                "fim_no_prefix": args.fim_no_prefix,
            }
        )
        return GPTFIMDatasetConfig(**data_args)

    return GPTDatasetConfig(**data_args)


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)


    is_packed_sequence = False
    if args.sft:
        if args.mock_data:
            dataset_type = MockSFTDataset
        else:
            dataset_type = SFTDataset
        is_packed_sequence = True  # SFT always uses packed sequence
    else:
        if args.mock_data:
            dataset_type = MockGPTDataset
        elif args.fim_data:
            dataset_type = GPTFIMDataset
        else:
            dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    is_dataset_built = partial(is_dataset_built_on_rank, vp_stage=vp_stage, is_packed_sequence=is_packed_sequence)
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, is_dataset_built, config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def get_embedding_ranks(pp_ranks: List[int]):
    """Get the embedding ranks."""
    embedding_ranks = [pp_ranks[0]]
    if len(pp_ranks) > 1:
        args = get_args()
        if not args.untie_embeddings_and_output_weights:
            embedding_ranks.append(pp_ranks[-1])
        config = core_transformer_config_from_args(args)
        mtp_ranks = get_mtp_ranks(pp_ranks, config)
        embedding_ranks.extend(mtp_ranks)
    embedding_ranks = list(set(embedding_ranks))
    embedding_ranks = sorted(embedding_ranks)
    return embedding_ranks


if __name__ == "__main__":
    # Timestamp right after entering __main__ block (after all imports/library setup)
    _MAIN_ENTRY_TIME = time.time()

    # Register startup timestamps for timing report in pretrain()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(
        train_valid_test_datasets_provider,
        partial(model_provider, gpt_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_gpt_extra_args,
        store=store,
        get_embedding_ranks=get_embedding_ranks,
    )
