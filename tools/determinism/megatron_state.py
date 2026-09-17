# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Explicit local-state adapters for the Megatron GPT training replay recipe.

Do not use a DistributedOptimizer's outer state_dict alone: it omits moments.
These adapters read the inner optimizer and its rank-local master parameters.
They reject unimplemented backend/loader paths instead of omitting state.
"""

from __future__ import annotations

import torch

from tools.determinism.hybrid_state import capture_hybrid_adam_state
from tools.determinism.training_state import UnverifiedState


def _type_name(value) -> str:
    return f"{type(value).__module__}.{type(value).__name__}"


def capture_optimizer(optimizer, *, model_chunks: list | None = None) -> tuple[dict, dict]:
    """Capture every chained optimizer, local moment and master-parameter shard."""
    from megatron.core.optimizer import Adam, ChainedOptimizer, HybridDeviceOptimizer
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
        if type(inner) is HybridDeviceOptimizer:
            if (
                type(child) is not DistributedOptimizer
                or inner.cpu_optimizer_cls is not torch.optim.AdamW
                or inner.gpu_optimizer_cls is not Adam
                or _type_name(inner.gpu_optimizer)
                != "transformer_engine.pytorch.optimizers.fused_adam.FusedAdam"
            ):
                raise UnverifiedState("Hybrid state requires distributed Torch AdamW/TE FusedAdam")
            state[str(index)], precision[str(index)] = capture_hybrid_adam_state(
                child, model_chunks
            )
            continue
        if type(inner) not in (Adam, torch.optim.Adam, torch.optim.AdamW):
            raise UnverifiedState(f"Unsupported inner optimizer: {_type_name(inner)}")
        if getattr(child.config, "use_precision_aware_optimizer", False):
            if (
                type(child) is not DistributedOptimizer
                or type(inner) is not Adam
                or _type_name(inner) != "transformer_engine.pytorch.optimizers.fused_adam.FusedAdam"
            ):
                raise UnverifiedState("Precision-aware state requires distributed TE FusedAdam")
            state[str(index)], precision[str(index)] = capture_precision_aware_adam_state(
                child, model_chunks
            )
        else:
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


def capture_precision_aware_adam_state(wrapper, model_chunks: list | None) -> tuple[dict, dict]:
    """Read raw FP16 moments, their scales and BF16 master remainders without casting.

    This explicit schema covers the named precision-aware recipe only. The caller
    must validate the distributed wrapper and exact TE optimizer type. TE's own
    state_dict converts low-precision moments and omits its separate scale map.
    """
    inner, config = wrapper.optimizer, wrapper.config
    expected = {
        "use_precision_aware_optimizer": True,
        "use_distributed_optimizer": True,
        "bf16": True,
        "store_param_remainders": True,
        "main_params_dtype": torch.float32,
        "main_grads_dtype": torch.float32,
        "exp_avg_dtype": torch.float16,
        "exp_avg_sq_dtype": torch.float16,
        "optimizer_cpu_offload": False,
    }
    if any(getattr(config, key, None) != value for key, value in expected.items()):
        raise UnverifiedState("Unsupported precision-aware optimizer configuration")
    dtype_map = {
        "exp_avg": torch.float16,
        "exp_avg_sq": torch.float16,
        "master_param": torch.float32,
    }
    policy = {
        "capturable": False,
        "master_weights": True,
        "use_decoupled_grad": True,
        "store_param_remainders": True,
        "fuse_unscale": False,
        "master_weight_dtype": torch.float32,
        "exp_avg_dtype": torch.float16,
        "exp_avg_sq_dtype": torch.float16,
        "name_to_dtype_map": dtype_map,
    }
    if any(getattr(inner, key, None) != value for key, value in policy.items()):
        raise UnverifiedState("Unsupported precision-aware Adam storage policy")
    if any(
        getattr(inner, name)
        for name in (
            "_optimizer_state_dict_pre_hooks",
            "_optimizer_state_dict_post_hooks",
            "_optimizer_step_pre_hooks",
            "_optimizer_step_post_hooks",
            "_optimizer_load_state_dict_pre_hooks",
            "_optimizer_load_state_dict_post_hooks",
        )
    ):
        raise UnverifiedState("Optimizer hooks require a separate adapter")
    parameters = [p for group in inner.param_groups for p in group["params"]]
    if (
        not parameters
        or len(set(parameters)) != len(parameters)
        or set(inner.state) != set(parameters)
    ):
        raise UnverifiedState("Missing or duplicated precision-aware optimizer parameters")
    # The base implementation maps live parameters to stable integer IDs without
    # invoking TE's conversion to unscaled checkpoint representations.
    raw = torch.optim.Optimizer.state_dict(inner)
    identifiers = [p for group in raw["param_groups"] for p in group["params"]]
    ids = dict(zip(parameters, identifiers))
    if len(ids) != len(parameters) or set(raw["state"]) != set(identifiers):
        raise UnverifiedState("Missing or duplicated precision-aware optimizer parameters")
    if set(inner._scales) != set(parameters):
        raise UnverifiedState("Missing or extra precision-aware parameter scales")
    shards, scales = {}, {}
    for parameter, identifier in ids.items():
        values = raw["state"][identifier]
        storage_dtypes = {**dtype_map, "master_param": torch.int16}
        if set(values) != set(storage_dtypes) or any(
            type(values[key]) is not torch.Tensor
            or values[key].dtype != dtype
            or values[key].shape != parameter.shape
            or values[key].device != parameter.device
            or not values[key].is_contiguous()
            for key, dtype in storage_dtypes.items()
        ):
            raise UnverifiedState("Missing or unsupported raw Adam moment/remainder storage")
        parameter_scales = inner._scales[parameter]
        if set(parameter_scales) != {"exp_avg", "exp_avg_sq"} or any(
            type(value) is not torch.Tensor
            or value.dtype != torch.float32
            or value.shape != (1,)
            or value.device != parameter.device
            for value in parameter_scales.values()
        ):
            raise UnverifiedState("Missing or unsupported raw Adam scaling metadata")
        gradient = getattr(parameter, "decoupled_grad", None)
        if (
            type(parameter) not in (torch.Tensor, torch.nn.Parameter)
            or parameter.dtype != torch.bfloat16
            or not parameter.numel()
            or not parameter.is_contiguous()
            or not isinstance(gradient, torch.Tensor)
            or type(gradient) is not torch.Tensor
            or gradient.dtype != torch.float32
            or gradient.shape != parameter.shape
            or gradient.device != parameter.device
            or not gradient.is_contiguous()
        ):
            raise UnverifiedState("Unsupported precision-aware parameter or decoupled gradient")
        shards[identifier] = {
            "value": parameter,
            "grad": parameter.grad,
            "decoupled_grad": gradient,
        }
        scales[identifier] = parameter_scales
    if set(inner.dtype_to_range_map) != {torch.float16, torch.uint8} or any(
        type(value) is not torch.Tensor or value.dtype != torch.float32 or value.shape != (1,)
        for value in inner.dtype_to_range_map.values()
    ):
        raise UnverifiedState("Unsupported Adam dtype-range storage")
    overflow = inner._dummy_overflow_buf
    if (
        type(overflow) is not torch.Tensor
        or overflow.dtype != torch.int32
        or overflow.shape != (1,)
        or overflow.device != parameters[0].device
    ):
        raise UnverifiedState("Unsupported Adam overflow buffer")
    if getattr(wrapper, "grad_scaler", None) is not None:
        raise UnverifiedState("Precision-aware BF16 recipe does not use a gradient scaler")
    state = {
        "wrapper": _type_name(wrapper),
        "inner": _type_name(inner),
        "state": raw,
        "scales": scales,
        "model_shards": capture_optimizer_shard_mapping(wrapper, model_chunks, ids),
        "options": {
            "adam_w_mode": inner.adam_w_mode,
            "policy": {
                key: str(value) if isinstance(value, torch.dtype) else value
                for key, value in policy.items()
                if key != "name_to_dtype_map"
            },
            "state_dtypes": {key: str(value) for key, value in dtype_map.items()},
            "dtype_ranges": {str(key): value for key, value in inner.dtype_to_range_map.items()},
            "dtype_range_devices": {
                str(key): value.device.type for key, value in inner.dtype_to_range_map.items()
            },
            "overflow_buffer": inner._dummy_overflow_buf,
            "defaults": inner.defaults,
            "set_grad_none": inner.set_grad_none,
        },
    }
    precision = {
        "storage": "bf16_parameters_int16_master_remainders_scaled_fp16_moments",
        "parameter_shards": shards,
        "grad_scaler": "disabled",
        "loss_scale": wrapper.get_loss_scale(),
        "found_inf": getattr(wrapper, "found_inf", None),
    }
    return state, precision


def capture_optimizer_shard_mapping(wrapper, model_chunks: list | None, ids: dict) -> dict:
    """Bind every live optimizer shard to its actual named model parameter range."""
    if not model_chunks:
        raise UnverifiedState("Precision-aware state requires actual model chunks")
    named = {
        parameter: (str(index), name)
        for index, chunk in enumerate(model_chunks)
        for name, parameter in chunk.named_parameters()
    }
    mapping = {}
    for parameter, (group, position) in wrapper.model_param_group_index_map.items():
        groups = wrapper.optimizer.param_groups
        if (
            parameter not in named
            or not 0 <= group < len(groups)
            or not 0 <= position < len(groups[group]["params"])
        ):
            raise UnverifiedState("Optimizer shard has no matching model parameter/group")
        shard = groups[group]["params"][position]
        extent = wrapper._get_model_param_range_map(parameter)["param"]
        if (
            not 0 <= extent.start < extent.end <= parameter.numel()
            or extent.end - extent.start != shard.numel()
            or parameter.dtype != shard.dtype
            or parameter.device != shard.device
            or parameter.view(-1)[extent.start : extent.end].data_ptr() != shard.data_ptr()
            or ids[shard] in mapping
        ):
            raise UnverifiedState("Optimizer shard range/storage differs from the model mapping")
        mapping[ids[shard]] = {
            "chunk": named[parameter][0],
            "parameter": named[parameter][1],
            "model_shape": list(parameter.shape),
            "start": extent.start,
            "end": extent.end,
            "group": group,
            "position": position,
        }
    if set(mapping) != set(ids.values()):
        raise UnverifiedState("Missing optimizer shard-to-model mapping")
    return mapping


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
