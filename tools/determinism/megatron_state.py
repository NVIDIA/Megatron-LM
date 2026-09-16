# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Explicit local-state adapters for the Megatron GPT training replay recipe.

Do not use a DistributedOptimizer's outer state_dict alone: it omits moments.
These adapters read the inner optimizer and its rank-local master parameters.
They reject unimplemented backend/loader paths instead of omitting state.
"""

from __future__ import annotations

import torch

from tools.determinism.training_state import UnverifiedState


def _type_name(value) -> str:
    return f"{type(value).__module__}.{type(value).__name__}"


def capture_optimizer(optimizer) -> tuple[dict, dict]:
    """Capture every chained optimizer, local moment and master-parameter shard."""
    from megatron.core.optimizer import Adam, ChainedOptimizer
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params, FP32Optimizer

    children = optimizer.chained_optimizers if type(optimizer) is ChainedOptimizer else [optimizer]
    state, precision = {}, {}
    for index, child in enumerate(children):
        if type(child) not in (
            DistributedOptimizer,
            Float16OptimizerWithFloat16Params,
            FP32Optimizer,
        ):
            raise UnverifiedState(f"Unsupported optimizer wrapper: {_type_name(child)}")
        if child.is_stub_optimizer:
            raise UnverifiedState("A stub optimizer needs an explicit no-parameter-rank adapter")
        inner = child.optimizer
        if type(inner) not in (Adam, torch.optim.Adam, torch.optim.AdamW):
            raise UnverifiedState(f"Unsupported inner optimizer: {_type_name(inner)}")
        state[str(index)], precision[str(index)] = capture_adam_state(child)
    if not children:
        raise UnverifiedState("Empty optimizer chain")
    return state, precision


def capture_adam_state(wrapper) -> tuple[dict, dict]:
    """Read a supported Adam wrapper without a gather or a checkpoint-only filter.

    Callers must validate the wrapper/backend type before using this helper.
    The parameter order matches the IDs in the inner optimizer's state dict.
    """
    inner = wrapper.optimizer
    state = inner.state_dict()
    masters = []
    for group in inner.param_groups:
        masters.append(
            [
                {
                    "value": parameter,
                    "grad": parameter.grad,
                    "decoupled_grad": getattr(parameter, "decoupled_grad", None),
                }
                for parameter in group["params"]
            ]
        )
    if not any(masters):
        raise UnverifiedState("No local optimizer parameters")
    parameter_ids = {
        identifier for group in state["param_groups"] for identifier in group["params"]
    }
    if set(state["state"]) != parameter_ids:
        raise UnverifiedState("Optimizer state is missing local parameters")
    for value in state["state"].values():
        if not all(isinstance(value.get(key), torch.Tensor) for key in ("exp_avg", "exp_avg_sq")):
            raise UnverifiedState("Adam state is missing first or second moments")
    scaler = getattr(wrapper, "grad_scaler", None)
    precision = {
        "master_parameters": masters,
        "grad_scaler": scaler.state_dict() if scaler is not None else "disabled",
        "loss_scale": wrapper.get_loss_scale(),
        "found_inf": getattr(wrapper, "found_inf", None),
    }
    return {"wrapper": _type_name(wrapper), "inner": _type_name(inner), "state": state}, precision


def capture_model(chunks: list) -> tuple[dict, dict]:
    """Capture all model chunks and ordinary/main gradients at the same boundary."""
    model, gradients = {}, {}
    for index, chunk in enumerate(chunks):
        key = str(index)
        model[key] = chunk.state_dict()
        gradients[key] = {
            name: {"grad": parameter.grad, "main_grad": getattr(parameter, "main_grad", None)}
            for name, parameter in chunk.named_parameters()
        }
    if not chunks:
        raise UnverifiedState("No model chunks")
    return model, gradients


def capture_single_pass_loader(loader, iterator, *, consumed_samples: int) -> dict:
    """Capture the worker-free single-pass MockGPT loader's canonical next position.

    A resumed iterator has fewer local yields than an uninterrupted iterator.
    Normalize those yields into the actual absolute sample cursor and verify
    it against training's counter. Do not use that counter alone as evidence.
    """
    from megatron.core.datasets.gpt_dataset import MockGPTDataset, MockGPTLowLevelDataset
    from megatron.core.rerun_state_machine import RerunDataIterator
    from megatron.training.datasets.data_samplers import MegatronPretrainingSampler

    if loader is None and iterator is None:
        return {"mode": "no_local_loader_broadcast_from_tp_source"}
    if type(iterator) is not RerunDataIterator or type(loader) is not torch.utils.data.DataLoader:
        raise UnverifiedState("Unsupported data-loader or replay iterator")
    if loader.num_workers != 0 or type(loader.batch_sampler) is not MegatronPretrainingSampler:
        raise UnverifiedState("Require single-pass sampler with no worker/prefetch state")
    if (
        type(loader.dataset) is not MockGPTDataset
        or type(loader.dataset.dataset) is not MockGPTLowLevelDataset
    ):
        raise UnverifiedState("Real/other datasets require an immutable data identity adapter")
    return capture_mock_loader_state(loader, iterator, consumed_samples=consumed_samples)


def capture_mock_loader_state(loader, iterator, *, consumed_samples: int) -> dict:
    """Capture a validated MockGPT loader; fail if iteration state is inconsistent."""
    sampler, dataset = loader.batch_sampler, loader.dataset
    local = iterator.iterable
    if local._index_sampler is not sampler or local._dataset is not dataset:
        raise UnverifiedState("Captured iterator does not belong to the loader")
    if iterator.replaying or iterator.saved_microbatches or iterator.replay_pos:
        raise UnverifiedState("Buffered reruns require a separate adapter")
    cursor = (
        sampler.consumed_samples + local._num_yielded * sampler.micro_batch_times_data_parallel_size
    )
    if cursor != consumed_samples or cursor > sampler.total_samples:
        raise UnverifiedState("Data iterator and training sample counters disagree")
    return {
        "mode": "single_pass_mock_gpt_workers_0",
        "next_global_sample": cursor,
        "sampler": {
            key: getattr(sampler, key)
            for key in (
                "total_samples",
                "micro_batch_size",
                "data_parallel_rank",
                "micro_batch_times_data_parallel_size",
                "drop_last",
            )
        },
        "loader_generator": loader.generator.get_state(),
        "iterator_base_seed": local._base_seed,
        "dataset": {
            "description": dataset.unique_description,
            "indices": dataset.indices,
            "document_index": dataset.document_index,
            "sample_index": dataset.sample_index,
            "shuffle_index": dataset.shuffle_index,
            "sequence_lengths": dataset.dataset.sequence_lengths,
            "vocab_size": dataset.dataset.vocab_size,
            "eod_token": dataset.dataset.eod_token,
            "cacheable_masks": dataset.masks_and_position_ids_are_cacheable,
            "cached_masks": dataset.masks_and_position_ids_are_cached,
            "attention_mask": dataset.cached_attention_mask,
            "loss_mask": dataset.cached_loss_mask,
            "position_ids": dataset.cached_position_ids,
        },
    }
