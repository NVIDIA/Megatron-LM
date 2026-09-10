# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pretrain and SFT GPT."""

# Capture the true program start time BEFORE any heavy imports.
import time

_PROGRAM_START_TIME = time.time()


from megatron.rank_log_setup import suppress_duplicate_logs_off_rank0

# Quiet the duplicate warnings before the heavy imports below: torch raises its
# own deprecations while it is being imported, so a filter installed any later
# cannot reach them.
suppress_duplicate_logs_off_rank0()

from functools import lru_cache, partial
from typing import List, Optional

import torch

from gpt_builders import gpt_builder
from megatron.core import mpu
from megatron.core.context_parallel import ContextParallelBatch, get_batches_on_this_cp_rank
from megatron.core.datasets.data_schedule import get_batch_on_this_rank_for_sequence_packing
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.package_info import __version__ as mcore_version
from megatron.core.parallel_state import get_context_parallel_group
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.transformer.multi_token_prediction import (
    get_mtp_ranks,
)
from megatron.core.transformer.multi_token_prediction import (
    mtp_on_this_rank as mtp_on_this_rank_func,
)
from megatron.core.utils import (
    StragglerDetector,
    flatten_batch_for_packed_sequences,
    get_attr_wrapped_model,
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
from megatron.training.argument_utils import (  # noqa: F401  (core_gpt_dataset_config_from_args: legacy shape, importable from here)
    core_gpt_dataset_config_from_args,
    gpt_config_from_args,
    gpt_data_config_from_args,
    pretrain_cfg_container_from_args,
)
from megatron.training.arguments import core_transformer_config_from_args, parse_and_validate_args
from megatron.training.datasets.providers import build_gpt_datasets, is_dataset_built_on_rank
from megatron.training.global_vars import set_run_config
from megatron.training.training import update_seqlen_stats_from_cu_seqlens
from megatron.training.utils import is_first_or_last_pipeline_stage
from model_provider import model_provider

try:
    from megatron.post_training.arguments import add_modelopt_args
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt
    from megatron.post_training.model_builder import ModelOptModelConfig
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


# The args object and the transformer config built from it, reused until the args object
# is replaced.
_transformer_config_cache = None


def _get_transformer_config(args):
    """Return the transformer config for args, building it only when args is a new object.

    get_batch runs every microbatch, and rebuilding the config there re-runs
    TransformerConfig.__post_init__, which repeats its validation warnings in the log.
    """
    global _transformer_config_cache
    if _transformer_config_cache is None or _transformer_config_cache[0] is not args:
        _transformer_config_cache = (args, core_transformer_config_from_args(args))
    return _transformer_config_cache[1]


def get_batch(
    data_iterator,
    vp_stage: Optional[int] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
):
    """Generate a batch."""

    args = get_args()
    config = _get_transformer_config(args)

    if args.sequence_packing_scheduler is not None:
        if pg_collection is None:
            # Compatibility for direct entrypoint callers; Core receives explicit groups.
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=["tp", "pp", "cp", "tp_cp", "dynamic_cp", "dynamic_tp_cp"]
            )
        return get_batch_on_this_rank_for_sequence_packing(
            data_iterator,
            vpp_size=config.virtual_pipeline_model_parallel_size,
            mtp_on_this_rank=mtp_on_this_rank_func(
                layout=config.pipeline_model_parallel_layout,
                mtp_num_layers=config.mtp_num_layers,
                ignore_virtual=False,
                vp_stage=vp_stage,
                pp_group=pg_collection.pp,
                vp_size=config.virtual_pipeline_model_parallel_size or 1,
            ),
            vp_stage=vp_stage,
            dynamic_cp=args.dynamic_context_parallel,
            pg_collection=pg_collection,
            config=config,
            return_context_parallel_batch=True,
        )

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
    # Dynamic CP always uses the sequence-packing path above.
    assert not args.dynamic_context_parallel

    if (
        not is_first_or_last_pipeline_stage(vp_stage)
        and not mtp_on_this_rank
        and not has_cu_seqlens
    ):
        return ContextParallelBatch(
            boundary_layout=config.attention_cp_layout,
            batches_by_layout={config.attention_cp_layout: dict.fromkeys(BATCH_KEYS)},
            packed_seq_params_by_layout={config.attention_cp_layout: None},
        )

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
        dynamic_cp=False,
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
        batch = {
            **dict.fromkeys(BATCH_KEYS),
            'cu_seqlens': batch['cu_seqlens'],
            'cu_seqlens_padded': batch['cu_seqlens_padded'],
            'max_seqlen': batch['max_seqlen'],
        }

    return get_batches_on_this_cp_rank(
        batch,
        boundary_layout=config.attention_cp_layout,
        dynamic_cp=False,
        cp_group=get_context_parallel_group(),
        use_per_sequence_balancing=(batch.get('cu_seqlens') is not None),
        sequence_parallel=config.sequence_parallel,
        tp_group=mpu.get_tensor_model_parallel_group(),
        tp_cp_group=(
            mpu.get_tensor_and_context_parallel_group()
            if config.sequence_parallel and config.tensor_model_parallel_size > 1
            else None
        ),
        tokens_per_sample=args.seq_length,
    )


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


@lru_cache(maxsize=1)
def _build_cached_logits_loss_func(
    logprobs_dir,
    decode_threads,
    msc_prefetch_depth,
    kd_loss_alpha,
    ignore_errors,
    ignore_hash,
):
    """Build (once) the offline knowledge-distillation loss callable for cached logits.

    Memoized so the teacher log-probability reader is constructed a single time per
    process, replacing the previous module-level mutable global.
    """
    from megatron.training.distillation import LossFuncCallable

    return LossFuncCallable(
        logprobs_dir=logprobs_dir,
        decode_threads=decode_threads,
        msc_prefetch_depth=msc_prefetch_depth,
        kd_loss_alpha=kd_loss_alpha,
        ignore_errors=ignore_errors,
        ignore_hash=ignore_hash,
    )


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

    if args.logits_load_dir is not None:
        # Offline knowledge distillation loss using cached teacher log-probabilities.
        loss_func_cached_logits = _build_cached_logits_loss_func(
            logprobs_dir=args.logits_load_dir,
            decode_threads=args.logits_load_decode_threads,
            msc_prefetch_depth=args.logits_load_msc_prefetch_depth,
            kd_loss_alpha=args.logits_load_kd_loss_alpha,
            ignore_errors=args.logits_load_ignore_errors,
            ignore_hash=args.logits_load_ignore_hash,
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


def forward_step(data_iterator, model: GPTModel, return_schedule_plan: bool = False):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor
    """
    args = get_args()
    config = core_transformer_config_from_args(args)
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        cp_batch = get_batch(
            data_iterator, vp_stage, pg_collection=get_attr_wrapped_model(model, "pg_collection")
        )
        attention_layout = (
            config.attention_cp_layout
            if config.attention_cp_layout in cp_batch.batches_by_layout
            else None
        )
        batch = cp_batch.get_batch(attention_layout)
        attention_mask = batch.get("attention_mask")
        cu_seqlens = batch.get("cu_seqlens")
        cu_seqlens_padded = batch.get("cu_seqlens_padded")
        labels = batch.get("labels")
        loss_mask = batch.get("loss_mask")
        position_ids = batch.get("position_ids")
        tokens = batch.get("tokens")
        packed_seq_params = cp_batch.get_packed_seq_params(attention_layout)
        padding_mask = batch.get("padding_mask")
        if cu_seqlens is not None:
            update_seqlen_stats_from_cu_seqlens(cu_seqlens.squeeze(0))

        # Offline logits KD: record this microbatch's global (un-CP-sharded)
        # document boundaries so the saver can persist them alongside the
        # logits _forward_hook is about to capture -- required for
        # document-aware CP reassembly of packed (--sft) sequences.
        from megatron.training.distillation.logits_saver import get_logits_saver

        saver = get_logits_saver()
        if saver is not None:
            saver.set_current_cu_seqlens(
                cu_seqlens_padded if cu_seqlens_padded is not None else cu_seqlens
            )

    timers('batch-generator').stop()

    with stimer:
        if return_schedule_plan:
            assert (
                args.overlap_moe_expert_parallel_comm
            ), "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
            schedule_plan = model.build_schedule_plan(
                tokens,
                position_ids,
                attention_mask,
                labels=labels,
                loss_mask=loss_mask,
                packed_seq_params=packed_seq_params,
                padding_mask=padding_mask,
            )
            return schedule_plan, partial(loss_func, loss_mask, model=model)
        else:
            output_tensor = model(
                tokens,
                position_ids,
                attention_mask,
                labels=labels,
                loss_mask=loss_mask,
                packed_seq_params=packed_seq_params,
                padding_mask=padding_mask,
            )

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model)


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build the train, validation and test datasets from the run config's ``cfg.data``.

    The provider body is ``megatron.training.datasets.providers.build_gpt_datasets``, shared with
    ``pretrain_hybrid.py`` and ``pretrain_hybrid_flex.py``, called with the shared rank
    predicate.
    """
    return build_gpt_datasets(
        train_val_test_num_samples, vp_stage, built_on_rank=is_dataset_built_on_rank
    )


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
    if has_nvidia_modelopt:
        maybe_enable_modelopt(args)
    if has_nvidia_modelopt and getattr(args, "modelopt_enabled", False):
        model_cfg = gpt_config_from_args(
            args, model_config_cls=ModelOptModelConfig, vocab_size_from_tokenizer=True
        )
    else:
        model_cfg = gpt_config_from_args(args, vocab_size_from_tokenizer=True)
    full_config = pretrain_cfg_container_from_args(
        args, model_cfg, data_cfg=gpt_data_config_from_args(args)
    )
    set_run_config(full_config)
    pretrain(
        full_config,
        train_valid_test_datasets_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        store=store,
        get_embedding_ranks=get_embedding_ranks,
    )
