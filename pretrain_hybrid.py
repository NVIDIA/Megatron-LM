# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain and SFT Hybrid."""

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
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Some libraries (e.g., CUTLASS DSL) use warnings.catch_warnings() with
    # simplefilter("always"), which overrides the filters above. Override
    # showwarning as a fallback to suppress warnings that slip through.
    _original_showwarning = warnings.showwarning

    def _rank0_only_showwarning(message, category, filename, lineno, file=None, line=None):
        if issubclass(category, (UserWarning, FutureWarning, DeprecationWarning)):
            return
        _original_showwarning(message, category, filename, lineno, file, line)

    warnings.showwarning = _rank0_only_showwarning

from functools import lru_cache, partial
from typing import Any, List, Optional, Tuple

import torch

from hybrid_builders import hybrid_builder
from megatron.core import mpu
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.package_info import __version__ as mcore_version
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_hybrid_data_context_parallel_groups,
)
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.transformer.multi_token_prediction import (
    mtp_on_this_rank as mtp_on_this_rank_func,
)
from megatron.core.utils import (
    StragglerDetector,
    flatten_batch_for_packed_sequences,
    get_attr_wrapped_model,
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_te_version,
    get_torch_version,
)
from megatron.training import (
    get_args,
    get_timers,
    inprocess_restart,
    pretrain,
    print_rank_0,
    set_startup_timestamps,
)
from megatron.training.argument_utils import (
    hybrid_config_from_args,
    pretrain_cfg_container_from_args,
)
from megatron.training.arguments import core_transformer_config_from_args, parse_and_validate_args
from megatron.training.datasets.sft_dataset import SFTDataset
from megatron.training.training import update_seqlen_stats_from_cu_seqlens
from megatron.training.utils import get_blend_and_blend_per_split, is_first_or_last_pipeline_stage
from model_provider import model_provider

try:
    from megatron.post_training.arguments import add_modelopt_args
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt
    from megatron.post_training.model_builder import ModelOptHybridModelConfig
    from megatron.post_training.utils import maybe_enable_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

stimer = StragglerDetector()

# Canonical, ordered schema of the fields ``get_batch`` returns. Kept alphabetical
# to match the historical ``sorted(batch.keys())`` order that callers unpack into.
BATCH_KEYS = [
    "attention_mask",
    "cu_seqlens",
    "cu_seqlens_padded",
    "hybrid_cp_group",
    "labels",
    "local_cp_size",
    "loss_mask",
    "max_seqlen",
    "position_ids",
    "tokens",
]


def _apply_legacy_identity_sft_tokenizer_patch(args: Any) -> None:
    """Restore legacy target masking for identity SFT without editing tokenizer source."""
    if not getattr(args, "sft", False):
        return
    if getattr(args, "sft_tokenizer_prompt_format", None) != "identity":
        return

    import numpy as np

    from megatron.core.tokenizers.text.libraries.sft_tokenizer import IGNORE_INDEX, SFTTokenizer

    if getattr(SFTTokenizer, "_legacy_identity_runtime_patch_applied", False):
        return

    original_tokenize_conversation = SFTTokenizer.tokenize_conversation

    def tokenize_conversation_with_legacy_identity_masking(
        self, conversation, return_target, add_generation_prompt
    ):
        if self._prompt_format != "identity":
            return original_tokenize_conversation(
                self, conversation, return_target, add_generation_prompt
            )

        # Skip system message if the tokenizer doesn't have a system role.
        if not self._prompt_config.has_system_role and conversation[0]["role"] == "system":
            conversation = conversation[1:]

        tokens = self._extract_token_ids(
            self._tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                return_assistant_token_mask=False,
                return_tensors="np",
                chat_template=self._prompt_config.custom_chat_template,
            )
        )

        if not return_target:
            return tokens

        target = tokens.copy()

        idx = 0
        for turn_idx, turn in enumerate(conversation):

            if turn["role"].lower() == "assistant" and len(turn["content"]) == 0:
                raise ValueError(f"empty assistant turn in conversation: {conversation}.")
            if turn["role"].lower() == "assistant":
                assert conversation[turn_idx - 1]["role"].lower() in ("user", "tool")

            turn_tokens = self._extract_token_ids(
                self._tokenizer.apply_chat_template(
                    [turn], tokenize=True, chat_template=self._prompt_config.custom_chat_template
                )
            )

            # There should be only one BOS at the very beginning.
            # After the first turn, skip BOS token.
            if self._prompt_config.has_bos and turn_idx > 0:
                turn_tokens = turn_tokens[1:]
            turn_len = len(turn_tokens)

            role = turn["role"].lower()
            if role in ("system", "user", "tool"):
                target[idx : idx + turn_len] = IGNORE_INDEX
            elif role == "assistant":
                if self._prompt_config.assistant_prefix_len > 0:
                    target[idx : idx + self._prompt_config.assistant_prefix_len] = IGNORE_INDEX
            else:
                raise ValueError("Wrong role value.")

            assert np.allclose(
                tokens[idx : idx + turn_len], turn_tokens
            ), f"expected turn tokens to match tokens in conversation {conversation}"

            idx += turn_len

        assert idx == len(tokens), f"mismatch in target masking the conversation {conversation}"

        return tokens, target

    SFTTokenizer._legacy_identity_tokenize_conversation = original_tokenize_conversation
    SFTTokenizer.tokenize_conversation = tokenize_conversation_with_legacy_identity_masking
    SFTTokenizer._legacy_identity_runtime_patch_applied = True
    print_rank_0("Applied legacy identity SFTTokenizer runtime patch.")


def get_batch(data_iterator, vp_stage=None):
    """Generate a batch."""

    args = get_args()
    config = core_transformer_config_from_args(args)

    cp_size = args.context_parallel_size
    tp_rank = mpu.get_tensor_model_parallel_rank()
    is_sft = args.sft
    has_cu_seqlens = is_sft or args.dataloader_inter_document_masking
    create_attention_mask_in_dataloader = args.create_attention_mask_in_dataloader
    mtp_on_this_rank = mtp_on_this_rank_func(
        layout=config.pipeline_model_parallel_layout,
        mtp_num_layers=config.mtp_num_layers,
        ignore_virtual=False,
        vp_stage=vp_stage,
    )
    is_hybrid_cp = args.hybrid_context_parallel

    if (
        not is_first_or_last_pipeline_stage(vp_stage)
        and not mtp_on_this_rank
        and not has_cu_seqlens
    ):
        return [None for _ in BATCH_KEYS]

    batch = {}
    if tp_rank == 0:
        batch = next(data_iterator)
        for key in BATCH_KEYS:
            batch[key] = (
                batch[key].cuda(non_blocking=True)
                if key in batch and batch[key] is not None
                else None
            )

    batch = get_batch_on_this_tp_rank(
        batch,
        broadcast_src_rank=mpu.get_tensor_model_parallel_src_rank(),
        broadcast_group=mpu.get_tensor_model_parallel_group(),
        has_cu_seqlens=has_cu_seqlens,
        is_hybrid_cp=is_hybrid_cp,
        create_attention_mask_in_dataloader=create_attention_mask_in_dataloader,
        cp_size=cp_size,
        tp_rank=tp_rank,
        micro_batch_size=args.micro_batch_size,
        seq_length=args.seq_length,
        mtp_on_this_rank=mtp_on_this_rank,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        is_pipeline_first_stage=mpu.is_pipeline_first_stage(),
        is_pipeline_last_stage=mpu.is_pipeline_last_stage(),
    )

    batch = flatten_batch_for_packed_sequences(batch)

    if not is_first_or_last_pipeline_stage(vp_stage) and not mtp_on_this_rank:
        assert has_cu_seqlens
        return (
            None,
            batch['cu_seqlens'],
            batch['cu_seqlens_padded'],
            None,
            None,
            None,
            None,
            batch['max_seqlen'],
            None,
            None,
        )

    batch = get_batch_on_this_cp_rank(
        batch,
        is_hybrid_cp=is_hybrid_cp,
        cp_group=get_context_parallel_group(),
        hybrid_cp_group_func=get_hybrid_data_context_parallel_groups,
        use_per_sequence_balancing=args.dataloader_inter_document_masking and not is_sft,
    )

    # Return values in BATCH_KEYS order so callers can unpack into the fixed
    # names regardless of any provenance fields wrappers like BlendedDataset
    # add (e.g. "dataset_id"). The for-loop above already populates every
    # BATCH_KEYS entry on tp_rank 0; other tp_ranks receive a fresh dict from
    # get_batch_on_this_tp_rank. BATCH_KEYS is already alphabetical, matching
    # the historical sorted(batch.keys()) order.
    return [batch[key] for key in BATCH_KEYS]


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


@lru_cache(maxsize=1)
def _build_cached_logits_loss_func(
    logprobs_dir, decode_threads, prefetch_factor, msc_prefetch_depth, kd_loss_alpha, ignore_errors
):
    """Build (once) the offline knowledge-distillation loss callable for cached logits.

    Memoized so the teacher log-probability reader is constructed a single time per
    process, replacing the previous module-level mutable global.
    """
    from megatron.training.distillation import LossFuncCallable

    return LossFuncCallable(
        logprobs_dir=logprobs_dir,
        decode_threads=decode_threads,
        prefetch_factor=prefetch_factor,
        msc_prefetch_depth=msc_prefetch_depth,
        kd_loss_alpha=kd_loss_alpha,
        ignore_errors=ignore_errors,
    )


def loss_func(
    loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[HybridModel] = None
):
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
    if args.logits_load_dir is not None:
        # Offline knowledge distillation loss using cached teacher log-probabilities.
        loss_func_cached_logits = _build_cached_logits_loss_func(
            logprobs_dir=args.logits_load_dir,
            decode_threads=args.logits_load_decode_threads,
            prefetch_factor=args.logits_load_prefetch_factor,
            msc_prefetch_depth=args.logits_load_msc_prefetch_depth,
            kd_loss_alpha=args.logits_load_kd_loss_alpha,
            ignore_errors=args.logits_load_ignore_errors,
        )
        loss, num_tokens, report = loss_func_cached_logits(loss_mask, output_tensor, model=model)
    elif has_nvidia_modelopt and getattr(args, 'modelopt_enabled', False):  # [ModelOpt]
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
            tolerance=0.0,  # forward pass calculations are deterministic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are deterministic
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
            tolerance=0.0,  # forward pass calculations are deterministic
            fatal=False,
        )

    return loss, num_tokens, report


def forward_step(data_iterator, model: HybridModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (HybridModel): The Hybrid Model
    """
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()

    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
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
        ) = get_batch(data_iterator, vp_stage)

    packed_seq_params = None
    if cu_seqlens is not None:
        # Squeeze the batch dim: the batch dict keeps cu_seqlens as (1, N)
        # for consistency, but PackedSeqParams and TE expect 1-D.
        cu_seqlens = cu_seqlens.squeeze(0)
        if cu_seqlens_padded is not None:
            cu_seqlens_padded = cu_seqlens_padded.squeeze(0)
        # Use real (unpadded) cu_seqlens to feed the FLOPs accounting: varlen
        # attention only computes work for real tokens within each chunk.
        update_seqlen_stats_from_cu_seqlens(cu_seqlens)
        cu_seqlens_for_params = cu_seqlens_padded if cu_seqlens_padded is not None else cu_seqlens
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens_for_params,
            cu_seqlens_kv=cu_seqlens_for_params,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
            max_seqlen_q=int(max_seqlen.item()),
            max_seqlen_kv=int(max_seqlen.item()),
            local_cp_size=int(local_cp_size.item()) if local_cp_size is not None else None,
            cp_group=hybrid_cp_group,
            total_tokens=int(cu_seqlens_for_params[-1].item()),
            tokens_per_sample=args.seq_length,
        )

    timers('batch-generator').stop()

    with stimer:
        output_tensor = model(
            tokens,
            position_ids,
            attention_mask,
            labels=labels,
            packed_seq_params=packed_seq_params,
            loss_mask=loss_mask,
        )

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model)


def is_dataset_built_on_rank(vp_stage=None, is_packed_sequence=False):
    """Whether the dataset should be built on the current rank."""
    args = get_args()
    config = core_transformer_config_from_args(args)
    if mpu.get_tensor_model_parallel_rank() != 0:
        return False
    elif is_packed_sequence:
        return True
    return is_first_or_last_pipeline_stage(vp_stage) or mtp_on_this_rank_func(
        layout=config.pipeline_model_parallel_layout,
        mtp_num_layers=config.mtp_num_layers,
        ignore_virtual=False,
        vp_stage=vp_stage,
    )


def core_gpt_dataset_config_from_args(args: Any) -> GPTDatasetConfig:
    """Build the GPT dataset config from parsed CLI args."""
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
        allow_ambiguous_pad_tokens=args.allow_ambiguous_pad_tokens,
        fast_cache_load=args.dataloader_fast_cache_load,
        sequences_per_dataset=sequences_per_dataset,
        defer_npy_index_mmap=args.dataloader_defer_npy_index_mmap,
        context_parallel_size=args.context_parallel_size,
        data_parallel_size=args.data_parallel_size,
        sequence_parallel_size=args.tensor_model_parallel_size * args.sequence_parallel,
        hybrid_context_parallel=args.hybrid_context_parallel,
        inter_document_masking=args.dataloader_inter_document_masking,
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
        config,
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    # Timestamp right after entering __main__ block (after all imports/library setup)
    _MAIN_ENTRY_TIME = time.time()

    print_rank_0(f'> PyTorch version ................ {get_torch_version()}')
    print_rank_0(f'> Megatron-Core version .......... {mcore_version}')
    print_rank_0(f'> Transformer Engine version ... {get_te_version()}')

    # Register startup timestamps for timing report in pretrain()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)

    # Temporary for transition to core datasets
    setattr(train_valid_test_datasets_provider, "is_distributed", True)

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    args = parse_and_validate_args(
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
    _apply_legacy_identity_sft_tokenizer_patch(args)
    if has_nvidia_modelopt:
        maybe_enable_modelopt(args)
    if has_nvidia_modelopt and getattr(args, "modelopt_enabled", False):
        model_cfg = hybrid_config_from_args(args, model_config_cls=ModelOptHybridModelConfig)
    else:
        model_cfg = hybrid_config_from_args(args)
    full_config = pretrain_cfg_container_from_args(args, model_cfg)
    pretrain(
        full_config,
        train_valid_test_datasets_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        store=store,
    )
