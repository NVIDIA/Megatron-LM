# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Forward step and masked loss adapted from Megatron-Bridge (Apache-2.0):
# https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/ec0eb0a65c42e7d2b24bf2e6bd015217e99fb4ed/src/megatron/bridge/training/gpt_step.py
# https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/ec0eb0a65c42e7d2b24bf2e6bd015217e99fb4ed/src/megatron/bridge/training/losses.py

"""Shared text pretraining step using MLM configs, batching, and runtime services.

Unlike Bridge's per-TP-rank loaders, MLM builds datasets on TP rank zero. The
batch adapter therefore uses Core's existing TP broadcast and CP layout helpers.
No root-level training script, Bridge GlobalState, or ModelOpt step is imported.
"""

from collections.abc import Callable, Iterator
from functools import partial

import torch

from megatron.core import tensor_parallel
from megatron.core.context_parallel import ContextParallelBatch, get_batches_on_this_cp_rank
from megatron.core.datasets.data_schedule import get_batch_on_this_rank_for_sequence_packing
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.transformer.multi_token_prediction import mtp_on_this_rank
from megatron.core.utils import (
    StragglerDetector,
    flatten_batch_for_packed_sequences,
    get_attr_wrapped_model,
    get_batch_on_this_tp_rank,
)
from megatron.training import get_timers
from megatron.training.datasets.utils import get_dataset_input_length
from megatron.training.global_vars import get_run_config
from megatron.training.models.hybrid import HybridModelConfig
from megatron.training.training import update_seqlen_stats_from_cu_seqlens

stimer = StragglerDetector()

_BATCH_KEYS = (
    "attention_mask",
    "cu_seqlens",
    "cu_seqlens_padded",
    "dynamic_cp_group",
    "labels",
    "local_cp_size",
    "loss_mask",
    "max_seqlen",
    "padding_mask",
    "position_ids",
    "tokens",
)


def get_batch(
    data_iterator: Iterator | None, *, pg_collection: ProcessGroupCollection, vp_stage: int | None
) -> ContextParallelBatch:
    """Read on TP rank zero, broadcast required fields, and prepare Core's CP views."""
    cfg = get_run_config()
    config = cfg.model.transformer
    use_mtp = mtp_on_this_rank(
        layout=config.pipeline_model_parallel_layout,
        mtp_num_layers=config.mtp_num_layers,
        ignore_virtual=False,
        vp_stage=vp_stage,
        pp_group=pg_collection.pp,
        vp_size=config.virtual_pipeline_model_parallel_size or 1,
    )
    if config.sequence_packing_scheduler is not None:
        return get_batch_on_this_rank_for_sequence_packing(
            data_iterator,
            vpp_size=config.virtual_pipeline_model_parallel_size,
            mtp_on_this_rank=use_mtp,
            vp_stage=vp_stage,
            dynamic_cp=config.dynamic_context_parallel,
            pg_collection=pg_collection,
            config=config,
            return_context_parallel_batch=True,
        )

    assert not config.dynamic_context_parallel
    vp_size = config.virtual_pipeline_model_parallel_size
    is_first = is_pp_first_stage(pg_collection.pp) and is_vp_first_stage(vp_stage, vp_size)
    is_last = is_pp_last_stage(pg_collection.pp) and is_vp_last_stage(vp_stage, vp_size)
    has_cu_seqlens = cfg.data.has_cu_seqlens()
    needs_masks = cfg.data.creates_attention_mask() or config.num_moe_experts is not None
    if not is_first and not is_last and not use_mtp and not has_cu_seqlens and not needs_masks:
        return ContextParallelBatch.from_single_layout(
            config.linear_cp_layout, dict.fromkeys(_BATCH_KEYS), None
        )

    cp_size = pg_collection.cp.size()
    keep_full_positions = cfg.model.position_embedding_type == "mrope" and cp_size > 1
    if keep_full_positions and has_cu_seqlens:
        raise ValueError("Packed mrope with CP>1 requires a model-specific forward step")
    tp_rank = pg_collection.tp.rank()
    batch = {}
    if tp_rank == 0:
        sample = next(data_iterator)
        batch = {
            key: sample[key].cuda(non_blocking=True) if sample.get(key) is not None else None
            for key in _BATCH_KEYS
        }
    seq_length = get_dataset_input_length(cfg.data.dataset)
    batch = get_batch_on_this_tp_rank(
        batch,
        broadcast_src_rank=torch.distributed.get_global_rank(pg_collection.tp, 0),
        broadcast_group=pg_collection.tp,
        has_cu_seqlens=has_cu_seqlens,
        dynamic_cp=False,
        create_attention_mask_in_dataloader=cfg.data.creates_attention_mask(),
        cp_size=cp_size,
        tp_rank=tp_rank,
        micro_batch_size=cfg.train.micro_batch_size,
        seq_length=seq_length,
        mtp_on_this_rank=use_mtp,
        pipeline_model_parallel_size=pg_collection.pp.size(),
        is_pipeline_first_stage=is_first,
        is_pipeline_last_stage=is_last,
        broadcast_padding_mask=True,
    )
    batch = flatten_batch_for_packed_sequences(batch)
    # mRoPE positions are [3, batch, sequence], not [batch, sequence]. The model
    # needs the full positions; omit them from the generic sequence-dimension sharder.
    full_positions = batch.pop("position_ids", None) if keep_full_positions else None
    additional_layouts = set()
    if cp_size > 1 and config.linear_cp_layout != config.attention_cp_layout:
        additional_layouts.add(config.attention_cp_layout)
    cp_batch = get_batches_on_this_cp_rank(
        batch,
        boundary_layout=config.linear_cp_layout,
        dynamic_cp=False,
        cp_group=pg_collection.cp,
        additional_layouts=additional_layouts,
        use_per_sequence_balancing=cfg.data.use_per_sequence_balancing(),
        sequence_parallel=config.sequence_parallel,
        tp_group=pg_collection.tp,
        tp_cp_group=pg_collection.tp_cp if config.sequence_parallel else None,
        tokens_per_sample=seq_length,
    )
    if full_positions is not None:
        for view in cp_batch.batches_by_layout.values():
            view["position_ids"] = full_positions
    return cp_batch


def masked_next_token_loss(
    loss_mask: torch.Tensor, output_tensor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return the masked loss sum, valid-token count, and MLM reporting metrics."""
    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)
    cfg = get_run_config()
    rerun = get_rerun_state_machine()
    if cfg.rerun_state_machine.check_for_nan_in_loss:
        for rejection, name in ((torch.isnan, "NaN"), (torch.isinf, "Inf")):
            rerun.validate_result(
                result=loss,
                rejection_func=rejection,
                message=f"found {name} in local forward loss calculation",
                tolerance=0.0,
                fatal=True,
            )
    if cfg.rerun_state_machine.check_for_spiky_loss:
        rerun.validate_result(
            result=loss,
            rejection_func=partial(
                rerun.is_unexpectedly_large,
                threshold=cfg.rerun_state_machine.spiky_loss_factor,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,
            fatal=False,
        )
    num_tokens = loss_mask.sum().detach().clone().to(torch.int)
    report = torch.cat([loss.detach().clone().view(1), num_tokens.view(1)])
    return loss, num_tokens, {"lm loss": report}


def forward_step(
    data_iterator: Iterator | None, model: torch.nn.Module, return_schedule_plan: bool = False
) -> tuple[torch.Tensor, Callable]:
    """Run a GPT/Hybrid microbatch and return the masked next-token loss callback."""
    cfg = get_run_config()
    timers = get_timers()
    pg_collection = get_attr_wrapped_model(model, "pg_collection", allow_none=False)
    vp_stage = get_attr_wrapped_model(model, "vp_stage")
    timers("batch-generator", log_level=2).start()
    with stimer(bdata=True):
        cp_batch = get_batch(data_iterator, pg_collection=pg_collection, vp_stage=vp_stage)
        batch = cp_batch.get_batch()
        # The scheduler already supplies global totals. Otherwise each physical rank
        # counts a logical microbatch only once, not once per virtual model chunk.
        if (
            cfg.model.transformer.sequence_packing_scheduler is None
            and vp_stage in (None, 0)
            and batch.get("cu_seqlens") is not None
        ):
            update_seqlen_stats_from_cu_seqlens(batch["cu_seqlens"].squeeze(0))
    timers("batch-generator").stop()

    padding_mask = batch.get("padding_mask")
    # GPT scatters alongside embeddings on its first stage only. Later stages
    # receive SP-local activations; Hybrid handles its masks inside the model.
    if (
        padding_mask is not None
        and cfg.model.transformer.sequence_parallel
        and not isinstance(cfg.model, HybridModelConfig)
        and not get_attr_wrapped_model(model, "pre_process")
    ):
        padding_mask = (
            tensor_parallel.scatter_to_sequence_parallel_region(
                padding_mask.transpose(0, 1).contiguous(), group=pg_collection.tp
            )
            .transpose(0, 1)
            .contiguous()
        )
    forward_args = {
        "input_ids": batch.get("tokens"),
        "position_ids": batch.get("position_ids"),
        "attention_mask": batch.get("attention_mask"),
        "labels": batch.get("labels"),
        "loss_mask": batch.get("loss_mask"),
        "packed_seq_params": cp_batch.get_packed_seq_params(),
        "padding_mask": padding_mask,
    }
    with stimer:
        if return_schedule_plan:
            assert (
                cfg.model.transformer.overlap_moe_expert_parallel_comm
            ), "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
            output = model.build_schedule_plan(**forward_args)
        else:
            if isinstance(cfg.model, HybridModelConfig):
                forward_args["cp_batch"] = cp_batch
            output = model(**forward_args)
    return output, partial(masked_next_token_loss, batch.get("loss_mask"))
