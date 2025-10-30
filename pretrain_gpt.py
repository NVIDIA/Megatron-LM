# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain and SFT GPT."""

import torch

from functools import partial
from typing import List, Optional, Tuple
from megatron.core import parallel_state
from megatron.training import inprocess_restart
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_rank,
    get_context_parallel_world_size,
)
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import get_attr_wrapped_model, get_thd_batch_on_this_cp_rank, StragglerDetector
from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer
from megatron.training import get_args, get_timers, get_tokenizer, pretrain, print_rank_0
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
    is_first_or_last_pipeline_stage,
)
from megatron.training.datasets.sft_dataset import SFTDataset, MockSFTDataset
from model_provider import model_provider
from gpt_builders import gpt_builder

try:
    from megatron.post_training.arguments import add_modelopt_args, modelopt_args_enabled
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
    args = get_args()
    
    # TODO: this is pretty hacky, find a better way
    if not is_first_or_last_pipeline_stage(vp_stage):
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)
    
    
    if args.sft_sequence_packing:
        cu_seqlens = batch.pop('cu_seqlens')
        cu_seqlens_padded = batch.pop('cu_seqlens_padded')
        #debugmtl for debug nan
        cu_seqlens = torch.cat([cu_seqlens, torch.tensor([1024], device=cu_seqlens.device, dtype=torch.int32)], dim=0)
        cu_seqlens_padded = torch.cat([cu_seqlens_padded, torch.tensor([1024], device=cu_seqlens_padded.device, dtype=torch.int32)], dim=0)
        
        max_seqlen = int(batch.pop('max_seqlen').item())
        # local_cp_size is None if we disable hybrid-cp
        local_cp_size = int(batch.pop('local_cp_size').item()) if 'local_cp_size' in batch else None
        batch, packed_seq_params = get_thd_batch_on_this_cp_rank(batch, cu_seqlens, 
                cu_seqlens_padded, max_seqlen, local_cp_size=local_cp_size)
        
    else:
        # slice batch along sequence dimension for context parallelism
        batch = get_batch_on_this_cp_rank(batch)  # The implementation of this function is in MCore
        packed_seq_params = None
           
           
    if not hasattr(get_batch, 'microbatch_counter'):
        get_batch.microbatch_counter = 0
    
    # Debugmtl: 保存batch数据到文本文件 - 保存所有元素
    print_data = True
    if print_data:
        import os
        import numpy as np
        
        debug_dir = '/lustre/fsw/portfolios/coreai/users/tailaim/debug_data'
        os.makedirs(debug_dir, exist_ok=True)
        
        rank = parallel_state.get_data_parallel_rank(with_context_parallel=True)
        microbatch_id = get_batch.microbatch_counter
        debug_file = os.path.join(debug_dir, f'batch_microbatch{microbatch_id:06d}_rank{rank}.txt')
        
        with open(debug_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"Batch Debug Info - Microbatch {microbatch_id} - Rank {rank}\n")
            f.write("="*80 + "\n\n")
            
            # Batch shapes
            f.write("BATCH SHAPES:\n")
            f.write("-"*80 + "\n")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    f.write(f"{key}: {value.shape}, dtype: {value.dtype}, device: {value.device}\n")
            f.write("\n")
            
            # Batch values (ALL elements)
            f.write("BATCH VALUES (ALL ELEMENTS):\n")
            f.write("-"*80 + "\n")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    f.write(f"\n{key}:\n")
                    f.write(f"Shape: {value.shape}\n")
                    # 转换为numpy并设置打印选项
                    data = value.cpu().numpy()
                    # 设置numpy打印选项：打印所有元素，不省略
                    np.set_printoptions(threshold=np.inf, linewidth=200, suppress=False)
                    f.write(f"{data}\n")
            f.write("\n")
            
            # Packed seq params
            if packed_seq_params is not None:
                f.write("PACKED_SEQ_PARAMS:\n")
                f.write("-"*80 + "\n")
                if hasattr(packed_seq_params, 'cu_seqlens_q') and packed_seq_params.cu_seqlens_q is not None:
                    f.write(f"cu_seqlens_q: {packed_seq_params.cu_seqlens_q.cpu().numpy()}\n")
                if hasattr(packed_seq_params, 'cu_seqlens_kv') and packed_seq_params.cu_seqlens_kv is not None:
                    f.write(f"cu_seqlens_kv: {packed_seq_params.cu_seqlens_kv.cpu().numpy()}\n")
                if hasattr(packed_seq_params, 'max_seqlen_q'):
                    f.write(f"max_seqlen_q: {packed_seq_params.max_seqlen_q}\n")
                if hasattr(packed_seq_params, 'max_seqlen_kv'):
                    f.write(f"max_seqlen_kv: {packed_seq_params.max_seqlen_kv}\n")
                f.write("\n")
            
            # Additional info
            f.write("ADDITIONAL INFO:\n")
            f.write("-"*80 + "\n")
            f.write(f"local_cp_size: {local_cp_size}\n")
            f.write(f"max_seqlen: {max_seqlen}\n")
            f.write(f"cu_seqlens (original): {cu_seqlens.cpu().numpy()}\n")
            f.write(f"cu_seqlens_padded (original): {cu_seqlens_padded.cpu().numpy()}\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"[DEBUG] Batch data saved to: {debug_file}")
        
        # 递增计数器
        get_batch.microbatch_counter += 1
        
           
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

    if has_nvidia_modelopt and modelopt_args_enabled(args):  # [ModelOpt]
        return loss_func_modelopt(loss_mask, output_tensor, model=model)

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)

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

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    return (loss, num_tokens, {'lm loss': reporting_loss})


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


def is_dataset_built_on_rank(vp_stage=None):
    return is_first_or_last_pipeline_stage(vp_stage) and parallel_state.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    if args.legacy_tokenizer:
        tokenizer = get_tokenizer()
    else:
        tokenizer = build_tokenizer(args)

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        multiple_validation_sets=args.multiple_validation_sets,
        full_validation=args.full_validation,
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
        context_parallel_size=args.context_parallel_size,
        data_parallel_size=args.data_parallel_size,
        sequence_parallel_size=args.tensor_model_parallel_size*args.sequence_parallel,
        hybrid_context_parallel=args.hybrid_context_parallel,
        allow_ambiguous_pad_tokens=args.allow_ambiguous_pad_tokens,
        sft_mock_dataset_config_json=args.sft_mock_dataset_config_json,
        sft_sequence_packing=args.sft_sequence_packing,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.sft:
        if args.mock_data:
            dataset_type = MockSFTDataset
        else:
            dataset_type = SFTDataset
    else:
        if args.mock_data:
            dataset_type = MockGPTDataset
        else:
            dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, partial(is_dataset_built_on_rank, vp_stage=vp_stage), config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

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
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        store=store,
    )
