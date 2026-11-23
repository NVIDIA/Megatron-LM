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
# Adapted from Megatron-Bridge (Apache-2.0):
# https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/ec0eb0a65c42e7d2b24bf2e6bd015217e99fb4ed/src/megatron/bridge/training/vlm_step.py
# https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/ec0eb0a65c42e7d2b24bf2e6bd015217e99fb4ed/src/megatron/bridge/training/utils/packed_seq_utils.py

"""Vision-language step independent of the vision encoder implementation.

The model owns visual encoding, embedding insertion, multimodal position IDs,
and CP partitioning. Both HF and Megatron vision encoders use this contract.
The collator supplies full sequences and normalized visual model kwargs (or a
Bridge-compatible ``visual_inputs.normalized_for_model()`` object). Packing,
when enabled, happens in the collator, not in this step.

As in Bridge, every TP/PP/CP rank executing this step needs a matching iterator;
visual metadata is needed even on middle pipeline stages. MLM's TP-rank-zero
GPT dataset provider does not satisfy this VLM loader contract.
"""

from collections.abc import Callable, Iterator, Mapping
from functools import partial
from inspect import Parameter, signature
from typing import Any

import torch

from megatron.bridge.training.llm_step import masked_next_token_loss
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import StragglerDetector, get_attr_wrapped_model
from megatron.training import get_timers
from megatron.training.global_vars import get_run_config
from megatron.training.models.hybrid import HybridModelConfig

stimer = StragglerDetector()
_VISUAL_PAYLOAD_FIELDS = frozenset(("pixel_values", "pixel_values_videos", "images"))
_PACKED_OFFSETS = ("cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded", "cu_seqlens_kv_padded")


def _filter_visual_kwargs(
    model: torch.nn.Module, visual: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Keep processor fields accepted by the unwrapped model, as in Bridge."""
    if not visual:
        return {}
    module = model
    seen = set()
    while hasattr(module, "module") and id(module) not in seen:
        seen.add(id(module))
        wrapped = module.module
        if wrapped is None or wrapped is module:
            break
        module = wrapped
    parameters = signature(module.forward).parameters
    if any(p.kind == Parameter.VAR_KEYWORD for p in parameters.values()):
        return dict(visual)
    supported = {
        name
        for name, p in parameters.items()
        if p.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
    }
    return {key: value for key, value in visual.items() if key in supported}


def get_batch(
    data_iterator: Iterator, *, pre_process: bool, post_process: bool, use_mtp: bool
) -> dict[str, Any]:
    """Move a collated full-sequence batch to CUDA without slicing or mutating it.

    Visual payloads are needed only on the model chunk that embeds images;
    grid/position metadata and text IDs remain available on every chunk.
    Labels and supervision masks are needed on output and MTP chunks.
    """
    if data_iterator is None:
        raise ValueError("vlm_step requires a matching data iterator on every model-parallel rank")
    sample = next(data_iterator)
    tokens = sample.get("tokens")
    if tokens is None:
        tokens = sample.get("input_ids")
    batch = {
        "input_ids": tokens,
        "position_ids": sample.get("position_ids"),
        "attention_mask": sample.get("attention_mask"),
        "labels": sample.get("labels") if post_process or use_mtp else None,
        "loss_mask": sample.get("loss_mask") if post_process or use_mtp else None,
    }
    visual = sample.get("visual_inputs")
    if visual is not None:
        visual = visual if isinstance(visual, Mapping) else visual.normalized_for_model()
        batch["visual_inputs"] = {
            key: value.cuda(non_blocking=True)
            for key, value in visual.items()
            if value is not None and (pre_process or key not in _VISUAL_PAYLOAD_FIELDS)
        }
    for key in _PACKED_OFFSETS:
        batch[key] = sample.get(key)
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda(non_blocking=True)
    # Maxima stay host integers; do not introduce a GPU round trip per microbatch.
    for key in ("max_seqlen_q", "max_seqlen_kv"):
        value = sample.get(key)
        batch[key] = int(value) if value is not None else None
    return batch


def forward_step(
    data_iterator: Iterator, model: torch.nn.Module, return_schedule_plan: bool = False
) -> tuple[Any, Callable]:
    """Run a VLM microbatch and return MLM's masked next-token loss callback.

    The model may return ``(output, local_loss_mask)`` after partitioning the
    sequence for CP; that returned mask is authoritative for loss reduction.
    No encoder is called or selected here, and no text-only CP slicing is used.
    """
    cfg = get_run_config()
    dataset = cfg.data.dataset
    if getattr(dataset, "enable_in_batch_packing", False) and getattr(
        dataset, "defer_in_batch_packing_to_step", False
    ):
        raise ValueError("vlm_step requires collate-time packing, not step-owned packing")
    config = cfg.model.transformer
    timers = get_timers()
    timers("batch-generator", log_level=2).start()
    with stimer(bdata=True):
        batch = get_batch(
            data_iterator,
            pre_process=get_attr_wrapped_model(model, "pre_process"),
            post_process=get_attr_wrapped_model(model, "post_process"),
            use_mtp=(config.mtp_num_layers or 0) > 0,
        )
    timers("batch-generator").stop()

    visual = batch.pop("visual_inputs", {})
    offsets = {key: batch.pop(key) for key in _PACKED_OFFSETS}
    maxima = {key: batch.pop(key) for key in ("max_seqlen_q", "max_seqlen_kv")}
    if offsets["cu_seqlens_q"] is not None:
        offsets = {
            key: value.squeeze() if value is not None else None for key, value in offsets.items()
        }
        if offsets["cu_seqlens_kv"] is None:
            offsets["cu_seqlens_kv"] = offsets["cu_seqlens_q"]
        if maxima["max_seqlen_kv"] is None:
            maxima["max_seqlen_kv"] = maxima["max_seqlen_q"]
        batch["packed_seq_params"] = PackedSeqParams(
            **offsets,
            **maxima,
            qkv_format="thd",
            total_tokens=(
                batch["input_ids"].size(1) if isinstance(cfg.model, HybridModelConfig) else None
            ),
        )
    batch.update(_filter_visual_kwargs(model, visual))
    loss_mask = batch["loss_mask"]
    with stimer:
        if return_schedule_plan:
            assert (
                config.overlap_moe_expert_parallel_comm
            ), "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
            output = model.build_schedule_plan(**batch)
        else:
            output = model(**batch)
            if isinstance(output, tuple):
                output, loss_mask = output
    if loss_mask is not None:
        loss_mask = loss_mask.contiguous()
    return output, partial(masked_next_token_loss, loss_mask)
