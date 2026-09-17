# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Capture native hybrid Adam ownership, copies and completed transfer state."""

from __future__ import annotations

import inspect
from types import CodeType

import torch

from tools.determinism.training_state import UnverifiedState

HOOKS = (
    "_optimizer_state_dict_pre_hooks",
    "_optimizer_state_dict_post_hooks",
    "_optimizer_step_pre_hooks",
    "_optimizer_step_post_hooks",
    "_optimizer_load_state_dict_pre_hooks",
    "_optimizer_load_state_dict_post_hooks",
)
COMMON_ATTRIBUTES = {
    "defaults",
    "state",
    "param_groups",
    "_zero_grad_profile_name",
    "_warned_capturable_if_run_uncaptured",
    *HOOKS,
}
HYBRID_ATTRIBUTES = {
    "offload_fraction",
    "cpu_optimizer_cls",
    "gpu_optimizer_cls",
    "pin_cpu_grads",
    "pin_cpu_params",
    "overlap_cpu_optimizer_d2h_h2d",
    "param_update_in_fp32",
    "sub_optimizer_kwargs",
    "cpu_param_groups",
    "gpu_param_groups",
    "gpu_params_map_cpu_copy",
    "cpu_copys_map_gpu_param",
    "param_to_fp32_param",
    "param_to_inner_param",
    "inner_param_to_orig_param",
    "fp32_param_to_orig_param",
    "cpu_optimizers",
    "gpu_optimizer",
    "cpu_copy_map_grad",
    "_d2h_stream",
    "_h2d_stream",
    "_cpu_optimizer_map_data_event",
}
GPU_POLICY = {
    "capturable": False,
    "master_weights": False,
    "use_decoupled_grad": False,
    "store_param_remainders": False,
    "fuse_unscale": False,
    "master_weight_dtype": torch.float32,
    "exp_avg_dtype": torch.float32,
    "exp_avg_sq_dtype": torch.float32,
}
GPU_DISPATCH = (
    "multi_tensor_adam",
    "multi_tensor_adam_capturable",
    "multi_tensor_adam_capturable_master",
    "multi_tensor_adam_fp8",
    "multi_tensor_adam_param_remainder",
)
GPU_ATTRIBUTES = {
    *GPU_POLICY,
    *GPU_DISPATCH,
    "adam_w_mode",
    "set_grad_none",
    "name_to_dtype_map",
    "dtype_to_range_map",
    "_dummy_overflow_buf",
    "_scales",
}


def _require(condition, message: str) -> None:
    if not condition:
        raise UnverifiedState(message)


def _type_name(value) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _device_type(value: torch.Tensor) -> str:
    return value.device.type


def _is_pinned(value: torch.Tensor) -> bool:
    return value.is_pinned()


def _view_key(value: torch.Tensor) -> tuple:
    return (value.device, value.dtype, value.data_ptr(), tuple(value.shape), tuple(value.stride()))


def _same_identity_map(left: dict, right: dict) -> bool:
    return left.keys() == right.keys() and all(left[key] is value for key, value in right.items())


def _same_bytes(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and torch.equal(
            left.detach().contiguous().reshape(-1).view(torch.uint8).cpu(),
            right.detach().contiguous().reshape(-1).view(torch.uint8).cpu(),
        )
    )


def _tensor(value, shape, dtype, device: str) -> None:
    _require(
        type(value) in (torch.Tensor, torch.nn.Parameter)
        and tuple(value.shape) == tuple(shape)
        and value.dtype == dtype
        and _device_type(value) == device
        and value.is_contiguous(),
        "Unsupported hybrid tensor shape, dtype, placement or storage",
    )


def _values(value, classes: tuple):
    """Encode only supported option values, retaining all tensors unconverted."""
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, type):
        _require(value in classes, "Unknown hybrid optimizer class option")
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, dict):
        return {
            str(k) if isinstance(k, torch.dtype) else k: _values(v, classes)
            for k, v in value.items()
        }
    if type(value) in (tuple, list):
        return type(value)(_values(v, classes) for v in value)
    _require(
        type(value) in (torch.Tensor, torch.nn.Parameter, str, int, bool, float, type(None)),
        "Unsupported hybrid option or hidden state",
    )
    return value


def _nested_codes(code: CodeType) -> dict:
    result = {code.co_name: code}
    for value in code.co_consts:
        if isinstance(value, CodeType):
            result.update(_nested_codes(value))
    return result


def _hooks(optimizer, hybrid, role: str) -> dict:
    """Require the native hook code and its actual owner, not just its name."""
    codes = {}
    for method in (
        type(hybrid)._register_load_state_dict_hooks,
        type(hybrid)._register_param_copy_back_gpu_hook,
    ):
        codes.update(_nested_codes(method.__code__))
    expected: dict[str, list[str]] = {key: [] for key in HOOKS}
    if role == "outer":
        expected["_optimizer_load_state_dict_pre_hooks"] = ["pre_load_state_dict_hook"]
        expected["_optimizer_load_state_dict_post_hooks"] = ["post_load_state_dict_hook"]
    else:
        expected["_optimizer_step_post_hooks"] = [
            "fp32_param_copy_back_gpu_hook" if role == "gpu" else "param_copy_back_gpu_hook"
        ]
    result = {}
    for key, names in expected.items():
        hooks = list(getattr(optimizer, key).values())
        _require(len(hooks) == len(names), "Missing or extra native hybrid hook")
        for hook, name in zip(hooks, names):
            _require(
                inspect.isfunction(hook) and hook.__code__ is codes[name],
                "Unknown hybrid optimizer hook code",
            )
            expected_closure = {} if role == "outer" else {"self": hybrid}
            _require(
                inspect.getclosurevars(hook).nonlocals == expected_closure,
                "Hybrid hook belongs to another optimizer",
            )
        result[key] = [f"{hook.__module__}.{hook.__qualname__}" for hook in hooks]
    return result


def require_completed_hybrid_transfers(hybrid) -> dict:
    """Check native completion without draining an event or synchronizing a stream."""
    current = torch.cuda.current_stream()
    state = {
        "pending_d2h_events": len(hybrid._cpu_optimizer_map_data_event),
        "d2h_complete": hybrid._d2h_stream.query(),
        "h2d_complete": hybrid._h2d_stream.query(),
        "d2h_separate_from_current": hybrid._d2h_stream != current,
        "h2d_separate_from_current": hybrid._h2d_stream != current,
        "d2h_separate_from_h2d": hybrid._d2h_stream != hybrid._h2d_stream,
    }
    _require(
        state["pending_d2h_events"] == 0
        and all(v for k, v in state.items() if k != "pending_d2h_events"),
        "Incomplete or unsupported hybrid transfers",
    )
    return state


def capture_hybrid_shard_mapping(wrapper, model_chunks: list | None, ids: dict) -> dict:
    """Resolve actual storage, including reordered groups and separate FP32 masters."""
    if not model_chunks:
        raise UnverifiedState("Hybrid capture requires named model chunks")
    named = {
        p: (str(i), name)
        for i, chunk in enumerate(model_chunks)
        for name, p in chunk.named_parameters()
    }
    by_storage = {_view_key(p): p for p in ids}
    _require(len(by_storage) == len(ids), "Aliased outer hybrid parameter shards")
    positions = {
        p: (g, i)
        for g, group in enumerate(wrapper.optimizer.param_groups)
        for i, p in enumerate(group["params"])
    }
    mapping = {}
    for model_parameter, (group, _) in wrapper.model_param_group_index_map.items():
        _require(model_parameter in named, "Unknown hybrid model parameter")
        extent = wrapper._get_model_param_range_map(model_parameter)["param"]
        _require(
            0 <= extent.start < extent.end <= model_parameter.numel(),
            "Invalid hybrid model shard extent",
        )
        model_shard = model_parameter.detach().view(-1)[extent.start : extent.end]
        separate_master = (
            model_parameter.dtype != torch.float32
            and not wrapper.config.use_precision_aware_optimizer
        )
        shard = getattr(model_parameter, "main_param", None) if separate_master else model_shard
        if not isinstance(shard, torch.Tensor) or shard.numel() != model_shard.numel():
            raise UnverifiedState("Missing hybrid main parameter shard")
        if separate_master:
            _require(
                getattr(model_parameter, "main_param_sharded", False)
                and shard.dtype == torch.float32,
                "Unsupported hybrid main parameter",
            )
        outer = by_storage.get(_view_key(shard))
        if outer is None or ids[outer] in mapping or positions[outer][0] != group:
            raise UnverifiedState("Hybrid model and optimizer storage ownership disagree")
        main_grad = getattr(model_parameter, "main_grad", None)
        if not isinstance(main_grad, torch.Tensor):
            raise UnverifiedState("Missing model main gradient")
        gradient = getattr(outer, "decoupled_grad", outer.grad)
        expected_grad = main_grad.view(-1)[extent.start : extent.end]
        _require(
            isinstance(gradient, torch.Tensor) and _view_key(gradient) == _view_key(expected_grad),
            "Hybrid gradient is not the actual model gradient shard",
        )
        mapping[ids[outer]] = {
            "chunk": named[model_parameter][0],
            "parameter": named[model_parameter][1],
            "model_shape": list(model_parameter.shape),
            "model_dtype": str(model_parameter.dtype),
            "start": extent.start,
            "end": extent.end,
            "group": group,
            "position": positions[outer][1],
            "separate_main_parameter": separate_master,
        }
    _require(set(mapping) == set(ids.values()), "Missing hybrid shard ownership")
    return mapping


def _tensor_layout(value) -> dict:
    """Describe every captured tensor and its storage aliases without pointer values."""
    groups: dict[tuple, int] = {}
    layout = []

    def visit(item, path):
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            key = (item.device, storage.data_ptr())
            group = groups.setdefault(key, len(groups))
            layout.append(
                {
                    "path": path,
                    "device_type": _device_type(item),
                    "pinned": _is_pinned(item),
                    "requires_grad": item.requires_grad,
                    "stride": list(item.stride()),
                    "storage_group": group,
                    "storage_nbytes": storage.nbytes(),
                    "byte_offset": item.storage_offset() * item.element_size(),
                    "nbytes": item.numel() * item.element_size(),
                }
            )
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda k: (type(k).__name__, k)):
                visit(item[key], [*path, key])
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                visit(child, [*path, index])

    visit(value, [])
    return {"storage_groups": len(groups), "tensors": layout}


def capture_hybrid_adam_state(wrapper, model_chunks: list | None) -> tuple[dict, dict]:
    """Capture the partial-offload FP32 Adam contract; caller validates native classes."""
    hybrid, config = wrapper.optimizer, wrapper.config
    expected = {
        "optimizer_cpu_offload": True,
        "optimizer_offload_fraction": 0.5,
        "overlap_cpu_optimizer_d2h_h2d": True,
        "pin_cpu_grads": True,
        "pin_cpu_params": True,
        "bf16": True,
        "use_distributed_optimizer": True,
        "main_params_dtype": torch.float32,
        "main_grads_dtype": torch.float32,
        "exp_avg_dtype": torch.float32,
        "exp_avg_sq_dtype": torch.float32,
    }
    _require(
        all(getattr(config, k, None) == v for k, v in expected.items()),
        "Unsupported hybrid optimizer configuration",
    )
    options = {
        "offload_fraction": 0.5,
        "overlap_cpu_optimizer_d2h_h2d": True,
        "pin_cpu_grads": True,
        "pin_cpu_params": True,
        "param_update_in_fp32": True,
    }
    _require(
        all(getattr(hybrid, k, None) == v for k, v in options.items()),
        "Unsupported native hybrid policy",
    )
    _require(
        not (set(vars(hybrid)) - COMMON_ATTRIBUTES - HYBRID_ATTRIBUTES),
        "Unknown hybrid optimizer attributes require an adapter",
    )
    _require(
        hybrid.cpu_optimizers and hybrid.gpu_optimizer is not None,
        "Hybrid capture requires actual CPU and GPU owners",
    )
    transfer = require_completed_hybrid_transfers(hybrid)
    classes = (hybrid.cpu_optimizer_cls, hybrid.gpu_optimizer_cls)
    _require(
        type(hybrid.gpu_optimizer) is classes[1]
        and all(type(c) is classes[0] for c in hybrid.cpu_optimizers),
        "Hybrid child optimizer class changed",
    )
    parameters = [p for g in hybrid.param_groups for p in g["params"]]
    _require(
        parameters
        and len(set(parameters)) == len(parameters)
        and set(hybrid.state) == set(parameters),
        "Missing or duplicate hybrid parameters",
    )
    ids = {p: i for i, p in enumerate(parameters)}
    _require(
        set(hybrid.param_to_inner_param) == set(ids)
        and len(set(hybrid.param_to_inner_param.values())) == len(ids),
        "Incomplete or aliased hybrid inner ownership",
    )
    reverse = {v: k for k, v in hybrid.param_to_inner_param.items()}
    _require(
        _same_identity_map(hybrid.inner_param_to_orig_param, reverse),
        "Hybrid reverse ownership differs",
    )
    low_precision = {p for p in ids if p.dtype != torch.float32}
    _require(
        set(hybrid.param_to_fp32_param) == low_precision
        and all(
            hybrid.param_to_fp32_param[p] is hybrid.param_to_inner_param[p] for p in low_precision
        )
        and _same_identity_map(
            hybrid.fp32_param_to_orig_param, {v: k for k, v in hybrid.param_to_fp32_param.items()}
        ),
        "Hybrid master-copy maps disagree",
    )
    offloaded = set(hybrid.gpu_params_map_cpu_copy)
    _require(
        offloaded
        and offloaded < set(ids)
        and _same_identity_map(
            hybrid.cpu_copys_map_gpu_param,
            {v: k for k, v in hybrid.gpu_params_map_cpu_copy.items()},
        )
        and all(
            hybrid.gpu_params_map_cpu_copy[p] is hybrid.param_to_inner_param[p] for p in offloaded
        )
        and set(hybrid.cpu_copy_map_grad) == set(hybrid.gpu_params_map_cpu_copy.values()),
        "Missing or inconsistent hybrid CPU copy maps",
    )
    defaults = dict(hybrid.defaults)
    _require(
        defaults.get("differentiable", False) is False,
        "Unsupported outer differentiable optimizer policy",
    )
    defaults["differentiable"] = False
    _hooks(hybrid, hybrid, "outer")
    outer = torch.optim.Optimizer.state_dict(hybrid)
    owners, children, shards = {}, {}, {}
    for name, child in [(f"cpu:{i}", c) for i, c in enumerate(hybrid.cpu_optimizers)] + [
        ("gpu", hybrid.gpu_optimizer)
    ]:
        device = "cuda" if name == "gpu" else "cpu"
        _require(
            not (
                set(vars(child)) - COMMON_ATTRIBUTES - (GPU_ATTRIBUTES if name == "gpu" else set())
            ),
            "Unknown hybrid child attributes require an adapter",
        )
        child_hooks = _hooks(child, hybrid, "gpu" if name == "gpu" else "cpu")
        child_params = [p for g in child.param_groups for p in g["params"]]
        _require(
            child_params
            and len(set(child_params)) == len(child_params)
            and set(child.state) == set(child_params),
            "Missing child optimizer state",
        )
        child_ids = []
        for inner in child_params:
            _require(inner in reverse, "Unowned hybrid child parameter")
            parameter = reverse[inner]
            identifier = ids[parameter]
            _require(
                identifier not in owners and (parameter in offloaded) == (device == "cpu"),
                "Multiple or misplaced hybrid owners",
            )
            owners[identifier] = name
            child_ids.append(identifier)
            _tensor(
                parameter,
                inner.shape,
                torch.bfloat16 if parameter in low_precision else torch.float32,
                "cuda",
            )
            _tensor(inner, parameter.shape, torch.float32, device)
            gradient = getattr(parameter, "decoupled_grad", parameter.grad)
            _tensor(gradient, parameter.shape, torch.float32, "cuda")
            _tensor(inner.grad, inner.shape, torch.float32, device)
            _require(_same_bytes(gradient, inner.grad), "Stale hybrid child gradient")
            if device == "cpu":
                _require(
                    _is_pinned(inner)
                    and _is_pinned(inner.grad)
                    and hybrid.cpu_copy_map_grad[inner] is inner.grad,
                    "Missing pinned CPU parameter/gradient buffer",
                )
            _require(
                _same_bytes(
                    parameter, inner.detach().to(device=parameter.device, dtype=parameter.dtype)
                ),
                "Stale hybrid parameter copy",
            )
            values = child.state[inner]
            _require(
                values is hybrid.state[parameter]
                and set(values)
                == {"exp_avg", "exp_avg_sq", "master_param", *(["step"] if device == "cpu" else [])}
                and values["master_param"] is inner,
                "Hybrid state/master aliases differ",
            )
            for key in ("exp_avg", "exp_avg_sq"):
                _tensor(values[key], inner.shape, torch.float32, device)
            if device == "cpu":
                _tensor(values["step"], (), torch.float32, "cpu")
            shards[identifier] = {
                "value": parameter,
                "grad": parameter.grad,
                "decoupled_grad": getattr(parameter, "decoupled_grad", None),
                "inner": inner,
                "inner_grad": inner.grad,
                "master_copy": hybrid.param_to_fp32_param.get(parameter),
                "cpu_copy": hybrid.gpu_params_map_cpu_copy.get(parameter),
            }
        children[name] = {
            "class": _type_name(child),
            "outer_ids": child_ids,
            "state": torch.optim.Optimizer.state_dict(child),
            "defaults": child.defaults,
            "hooks": child_hooks,
        }
    _require(set(owners) == set(ids.values()), "Missing hybrid parameter owner")
    gpu = hybrid.gpu_optimizer
    _require(
        all(getattr(gpu, k, None) == v for k, v in GPU_POLICY.items())
        and not gpu._scales
        and gpu.name_to_dtype_map
        == {k: torch.float32 for k in ("exp_avg", "exp_avg_sq", "master_param")},
        "Unsupported hybrid GPU Adam storage policy",
    )
    _require(
        set(gpu.dtype_to_range_map) == {torch.float16, torch.uint8}, "Unknown Adam dtype range map"
    )
    for tensor in gpu.dtype_to_range_map.values():
        _tensor(tensor, (1,), torch.float32, "cpu")
    _tensor(gpu._dummy_overflow_buf, (1,), torch.int32, "cuda")
    for group in gpu.param_groups:
        _require(
            type(group.get("step")) is int and group["step"] > 0, "Missing hybrid GPU group counter"
        )
        outer_group = next(
            g for g in hybrid.param_groups if reverse[group["params"][0]] in set(g["params"])
        )
        _require(outer_group.get("step") == group["step"], "Hybrid outer/GPU counters disagree")
    children["gpu"]["options"] = {k: getattr(gpu, k) for k in GPU_POLICY}
    children["gpu"]["options"].update(
        adam_w_mode=gpu.adam_w_mode,
        set_grad_none=gpu.set_grad_none,
        name_to_dtype_map=gpu.name_to_dtype_map,
        dtype_to_range_map=gpu.dtype_to_range_map,
        overflow_buffer=gpu._dummy_overflow_buf,
        scales=gpu._scales,
        dispatch={
            k: f"{getattr(gpu, k).__module__}.{getattr(gpu, k).__name__}" for k in GPU_DISPATCH
        },
    )
    # These constructor group containers are not read after initialization;
    # checkpoint restoration recreates their stale option dictionaries. Validate
    # and retain their parameter memberships; live child groups above are complete.
    initial_groups = {}
    for name in ("cpu_param_groups", "gpu_param_groups"):
        groups = getattr(hybrid, name)
        members = [p for group in groups for p in group["params"]]
        expected_members = {
            hybrid.param_to_inner_param[p]
            for p in ids
            if (p in offloaded) == (name == "cpu_param_groups")
        }
        _require(
            len(members) == len(expected_members) and set(members) == expected_members,
            "Hybrid constructor group membership differs",
        )
        initial_groups[name] = [[ids[reverse[p]] for p in group["params"]] for group in groups]
    _require(getattr(wrapper, "grad_scaler", None) is None, "Unsupported hybrid gradient scaler")
    state = _values(
        {
            "wrapper": _type_name(wrapper),
            "inner": _type_name(hybrid),
            "state": outer,
            "defaults": defaults,
            "owners": owners,
            "children": children,
            "hooks": _hooks(hybrid, hybrid, "outer"),
            "model_shards": capture_hybrid_shard_mapping(wrapper, model_chunks, ids),
            "copy_maps": {
                "offloaded_outer_ids": sorted(ids[p] for p in offloaded),
                "fp32_copy_outer_ids": sorted(ids[p] for p in low_precision),
                "initial_group_membership": initial_groups,
            },
            "cpu_gradient_buffers": {
                ids[reverse[p]]: v for p, v in hybrid.cpu_copy_map_grad.items()
            },
            "transfer": transfer,
            "options": {**options, "sub_optimizer_kwargs": hybrid.sub_optimizer_kwargs},
            "schema": {
                "version": 1,
                "outer_differentiable_default": "implicit_or_explicit_false",
                "constructor_groups": "membership_only_live_child_groups_are_complete",
            },
        },
        classes,
    )
    precision = {
        "storage": "hybrid_fp32_moments_and_master_parameters",
        "parameter_shards": shards,
        "grad_scaler": "disabled",
        "loss_scale": wrapper.get_loss_scale(),
        "found_inf": getattr(wrapper, "found_inf", None),
    }
    state["tensor_layout"] = _tensor_layout({"optimizer": state, "precision": precision})
    return state, precision
