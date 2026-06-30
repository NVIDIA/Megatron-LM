# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Standalone TE-compatible CUDA graph callable runtime."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
import contextlib
import gc
from math import ceil, prod
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
import warnings

UPSTREAM_TE_VERSION = "v2.16"
UPSTREAM_TE_COMMIT = "4220403e831d29e93868f7793693ea83f6b8b05b"
UPSTREAM_TE_GRAPH_PATH = "transformer_engine/pytorch/graph.py"

__all__ = [
    "UPSTREAM_TE_COMMIT",
    "UPSTREAM_TE_GRAPH_PATH",
    "UPSTREAM_TE_VERSION",
    "make_graphed_callables",
]

_torch = None
torch = None
_tree_flatten = None
_tree_unflatten = None
_graph_pool_handle = None
_TE_AVAILABLE = None
_TE_IMPORT_ERROR = None


class _UnavailableTEType:
    """Placeholder used when TransformerEngine is not installed."""


DelayedScaling = _UnavailableTEType
Recipe = Any
dist_group_type = Any
TransformerEngineBaseModule = _UnavailableTEType
BasicOperation = _UnavailableTEType
Sequential = _UnavailableTEType
OperationFuser = _UnavailableTEType


class _FP8StateStub:
    skip_fp8_weight_update_tensor = None


class _FP8GlobalStateManagerStub:
    quantization_state = _FP8StateStub()

    @staticmethod
    def is_first_fp8_module() -> bool:
        return False

    @staticmethod
    def reduce_and_update_fp8_tensors(*args, **kwargs) -> None:
        return None

    @staticmethod
    def is_fp8_enabled() -> bool:
        return False

    @staticmethod
    def get_fp8_recipe() -> None:
        return None

    @staticmethod
    def get_fp8_group() -> None:
        return None

    @staticmethod
    def add_fp8_tensors_to_global_buffer(*args, **kwargs) -> None:
        return None


FP8GlobalStateManager = _FP8GlobalStateManagerStub


@contextlib.contextmanager
def _null_autocast(*args, **kwargs):
    yield


autocast = _null_autocast


def get_default_fp8_recipe():
    raise RuntimeError(
        "FP8 graph capture requires transformer_engine. Install te-graph-runtime[te] "
        "or disable FP8/TE-specific options."
    )


def _require_torch():
    """Import torch lazily so package import does not require torch initialization."""
    global _torch, torch, _tree_flatten, _tree_unflatten, _graph_pool_handle
    if _torch is None:
        import torch as imported_torch
        from torch.utils._pytree import tree_flatten, tree_unflatten
        from torch._C import _graph_pool_handle as imported_graph_pool_handle

        _torch = imported_torch
        torch = imported_torch
        _tree_flatten = tree_flatten
        _tree_unflatten = tree_unflatten
        _graph_pool_handle = imported_graph_pool_handle
    return _torch


def _load_optional_te() -> bool:
    """Load TransformerEngine internals when available, without delegating graphing."""
    global _TE_AVAILABLE, _TE_IMPORT_ERROR
    global DelayedScaling, Recipe, dist_group_type
    global autocast, FP8GlobalStateManager, get_default_fp8_recipe
    global get_all_rng_states, graph_safe_rng_available
    global TransformerEngineBaseModule, BasicOperation, Sequential, OperationFuser

    if _TE_AVAILABLE is not None:
        return _TE_AVAILABLE
    try:
        from transformer_engine.common.recipe import DelayedScaling as te_DelayedScaling
        from transformer_engine.common.recipe import Recipe as te_Recipe
        from transformer_engine.pytorch.constants import dist_group_type as te_dist_group_type
        from transformer_engine.pytorch.quantization import (
            autocast as te_autocast,
            FP8GlobalStateManager as te_FP8GlobalStateManager,
            get_default_fp8_recipe as te_get_default_fp8_recipe,
        )
        from transformer_engine.pytorch.distributed import (
            get_all_rng_states as te_get_all_rng_states,
            graph_safe_rng_available as te_graph_safe_rng_available,
        )
        from transformer_engine.pytorch.module.base import (
            TransformerEngineBaseModule as te_TransformerEngineBaseModule,
        )
        from transformer_engine.pytorch.ops.op import BasicOperation as te_BasicOperation
        from transformer_engine.pytorch.ops import Sequential as te_Sequential
        from transformer_engine.pytorch.ops.fuser import OperationFuser as te_OperationFuser
    except Exception as exc:  # pragma: no cover - exact import failure is environment-specific
        _TE_AVAILABLE = False
        _TE_IMPORT_ERROR = exc
        return False

    DelayedScaling = te_DelayedScaling
    Recipe = te_Recipe
    dist_group_type = te_dist_group_type
    autocast = te_autocast
    FP8GlobalStateManager = te_FP8GlobalStateManager
    get_default_fp8_recipe = te_get_default_fp8_recipe
    get_all_rng_states = te_get_all_rng_states
    graph_safe_rng_available = te_graph_safe_rng_available
    TransformerEngineBaseModule = te_TransformerEngineBaseModule
    BasicOperation = te_BasicOperation
    Sequential = te_Sequential
    OperationFuser = te_OperationFuser
    _TE_AVAILABLE = True
    _TE_IMPORT_ERROR = None
    return True


def _prepare_runtime() -> bool:
    _require_torch()
    return _load_optional_te()


def get_all_rng_states() -> Dict[Any, Any]:
    return {}


def graph_safe_rng_available() -> bool:
    _torch_mod = _require_torch()
    return (
        hasattr(_torch_mod.cuda.CUDAGraph, "register_generator_state")
        and hasattr(_torch_mod.Generator, "graphsafe_set_state")
        and hasattr(_torch_mod.Generator, "graphsafe_get_state")
        and hasattr(_torch_mod.Generator, "clone_state")
    )


def _te_required_error(feature: str) -> RuntimeError:
    detail = f" Original import error: {_TE_IMPORT_ERROR}" if _TE_IMPORT_ERROR else ""
    return RuntimeError(
        f"{feature} requires transformer_engine internals compatible with {UPSTREAM_TE_VERSION}."
        f" Install te-graph-runtime[te] or disable TE-specific graph options.{detail}"
    )


def _torch_dtype_to_np_typestr(dtype):
    _torch_mod = _require_torch()
    mapping = {
        _torch_mod.float16: "<f2",
        _torch_mod.float32: "<f4",
        _torch_mod.int64: "<i8",
        _torch_mod.int32: "<i4",
        _torch_mod.int8: "|i1",
        _torch_mod.qint8: "|u1",
        _torch_mod.bool: "|b1",
        _torch_mod.bfloat16: "<f2",
    }
    float8_dtype = getattr(_torch_mod, "float8_e4m3fn", None)
    if float8_dtype is not None:
        mapping[float8_dtype] = "|i1"
    ret = mapping.get(dtype)
    if ret is None:
        supported = ", ".join(str(d) for d in mapping)
        raise TypeError(f"Unsupported dtype: {dtype}. Supported dtypes: {supported}")
    return ret


class _WeakRefTensor:
    """Tensor-like wrapper around a CUDA data pointer for graph-pool reuse."""

    def __init__(self, data_ptr: int, dtype: Any, shape: Sequence[int]):
        self._data_ptr = data_ptr
        self.dtype = dtype
        self.shape = tuple(int(i) for i in shape)

    def data_ptr(self):
        return self._data_ptr

    def numel(self):
        return prod(self.shape)

    @property
    def __cuda_array_interface__(self):
        return {
            "shape": self.shape,
            "typestr": _torch_dtype_to_np_typestr(self.dtype),
            "data": (self.data_ptr() if self.numel() > 0 else 0, False),
            "version": 3,
        }


def make_weak_ref(x):
    """Return a tensor-like weak reference so CUDA graph pool memory can be reused."""
    _torch_mod = _require_torch()

    def convert_to_torch_tensor(tensor):
        if isinstance(tensor, _torch_mod.Tensor):
            return tensor
        old_ptr = tensor.data_ptr()
        new_tensor = _torch_mod.as_tensor(tensor).view(tensor.dtype)
        if old_ptr != new_tensor.data_ptr():
            raise RuntimeError("Data pointer mismatch after converting to torch.Tensor")
        return new_tensor

    if isinstance(x, _torch_mod.Tensor):
        return convert_to_torch_tensor(_WeakRefTensor(x.data_ptr(), x.dtype, x.shape)) if x.is_cuda else x
    if isinstance(x, tuple):
        return tuple(make_weak_ref(i) for i in x)
    if isinstance(x, list):
        return [make_weak_ref(i) for i in x]
    if isinstance(x, dict):
        return {k: make_weak_ref(v) for k, v in x.items()}
    if isinstance(x, (int, float, bool)) or x is None:
        return x
    raise TypeError(
        f"Invalid type {type(x).__name__} to make weak ref. Valid types are: "
        "torch.Tensor, tuple, list, dict, int, float, bool, and None."
    )




_IS_GRAPH_CAPTURING = False

_T = TypeVar("_T")
SingleOrTuple = Union[_T, Tuple[_T, ...]]

_CAPTURE_TIME_HOOK_KEYS = (
    "forward_pre_hooks",
    "forward_pre_hooks_with_kwargs",
    "forward_hooks",
    "forward_hooks_with_kwargs",
    "backward_pre_hooks",
    "backward_hooks",
)


def _empty_capture_time_hooks() -> Dict[str, Dict[Any, Any]]:
    return {key: {} for key in _CAPTURE_TIME_HOOK_KEYS}


def _canonicalize_capture_time_hooks(
    num_callables: int,
    capture_time_hooks: Optional[List[Optional[Dict[str, Dict]]]],
) -> List[Dict[str, Dict[Any, Any]]]:
    if capture_time_hooks is None:
        return [_empty_capture_time_hooks() for _ in range(num_callables)]
    if len(capture_time_hooks) != num_callables:
        raise ValueError(
            f"capture_time_hooks has {len(capture_time_hooks)} entries but there are "
            f"{num_callables} callables"
        )

    canonicalized = []
    for callable_idx, hooks in enumerate(capture_time_hooks):
        if hooks is None:
            canonicalized.append(_empty_capture_time_hooks())
            continue
        if not isinstance(hooks, dict):
            raise TypeError(
                "capture_time_hooks entries must be dicts or None, "
                f"but entry {callable_idx} has type {type(hooks).__name__}"
            )
        unknown_keys = sorted(set(hooks) - set(_CAPTURE_TIME_HOOK_KEYS))
        if unknown_keys:
            raise ValueError(
                f"Unknown capture_time_hooks keys for callable {callable_idx}: {unknown_keys}. "
                f"Supported keys are {list(_CAPTURE_TIME_HOOK_KEYS)}"
            )

        callable_hooks = _empty_capture_time_hooks()
        for key in _CAPTURE_TIME_HOOK_KEYS:
            value = hooks.get(key, {})
            if value is None:
                value = {}
            if not isinstance(value, dict):
                raise TypeError(
                    f"capture_time_hooks[{callable_idx!r}][{key!r}] must be a dict, "
                    f"but got {type(value).__name__}"
                )
            callable_hooks[key] = dict(value)
        canonicalized.append(callable_hooks)
    return canonicalized


def _check_capture_time_hook_return(value: Any, hook_name: str, detail: str) -> None:
    if value is not None:
        raise RuntimeError(
            f"capture_time_hooks {hook_name} must not return a value ({detail} must not be "
            "modified via hook return)"
        )


def set_capture_start() -> None:
    """Record beginning of `make_graphed_callables`."""
    global _IS_GRAPH_CAPTURING
    _IS_GRAPH_CAPTURING = True


def set_capture_end() -> None:
    """Record end of `make_graphed_callables`."""
    global _IS_GRAPH_CAPTURING
    _IS_GRAPH_CAPTURING = False


def is_graph_capturing() -> bool:
    """Return whether within `make_graphed_callables`."""
    return _IS_GRAPH_CAPTURING


def graph_pool_handle():
    """
    Returns an opaque token representing the id of a graph memory pool.
    """
    _require_torch()
    return _graph_pool_handle()


@contextlib.contextmanager
def _none_grad_context_wrapper(inputs):
    """
    Wrapper to set the gradients of the inputs to None,
    in case the backward pass makes grad accumulations.
    """
    original_input_grads = []
    for input_tensor in inputs:
        original_input_grads.append(input_tensor.grad)
        input_tensor.grad = None
    yield
    for input_tensor, original_grad in zip(inputs, original_input_grads):
        input_tensor.grad = original_grad


@contextlib.contextmanager
def _graph_context_wrapper(*args, **kwargs):
    """Wrapper around `torch.cuda.graph`.

    This wrapper is a temporary workaround for a PyTorch bug:
    automatic garbage collection can destroy a graph while another
    graph is being captured, resulting in a CUDA error. See
    https://github.com/pytorch/pytorch/pull/161037.

    """
    gc_is_enabled = gc.isenabled()
    if gc_is_enabled:
        gc.disable()
    with torch.cuda.graph(*args, **kwargs):
        yield
    if gc_is_enabled:
        gc.enable()


def _make_graphed_callables(
    callables: SingleOrTuple[Callable],
    sample_args: SingleOrTuple[Tuple[torch.Tensor, ...]],
    num_warmup_iters: int = 3,
    allow_unused_input: bool = False,
    cache_quantized_params: bool = False,
    sample_kwargs: Optional[SingleOrTuple[Dict[str, Any]]] = None,
    _order: Optional[List[int]] = None,
    _num_layers_per_chunk: Optional[List[int]] = None,
    pool: Optional[Tuple[int, ...]] = None,
    retain_graph_in_backward: bool = False,
    _reuse_graph_input_output_buffers: bool = False,
    _clone_param_grads_on_return: bool = True,
    pre_warmup_hook: Optional[Callable] = None,
    post_warmup_hook: Optional[Callable] = None,
    capture_time_hooks: Optional[List[Optional[Dict[str, Dict]]]] = None,
    capture_stream: Optional[torch.cuda.Stream] = None,
) -> SingleOrTuple[Callable]:
    """
    Helper method for `make_graphed_callables`
    """

    if torch.is_autocast_enabled() and torch.is_autocast_cache_enabled():
        raise RuntimeError(
            "make_graphed_callables does not support the autocast "
            "caching. Please set `cache_enabled=False`."
        )

    # Default is to pass no kwargs to callables
    if sample_kwargs is None:
        if isinstance(callables, tuple):
            sample_kwargs = tuple({} for _ in range(len(sample_args)))
        else:
            sample_kwargs = {}

    # Canonicalize args as tuples
    just_one_callable = False
    if not isinstance(callables, tuple):
        just_one_callable = True
        callables = (callables,)
        sample_args = (sample_args,)
        sample_kwargs = (sample_kwargs,)

    capture_time_hooks = _canonicalize_capture_time_hooks(len(callables), capture_time_hooks)

    # Check training/inference
    is_training = all(c.training for c in callables)
    if not is_training and any(c.training for c in callables):
        raise RuntimeError(
            "make_graphed_callables only supports when modules are all in training or all in"
            " inference mode."
        )

    # Check sizes of args
    _order_without_wgrad = None
    delay_wgrad_compute = False
    if _order is None:
        if len(sample_args) != len(callables):
            raise ValueError(
                "Expected sample_args to have the same length as callables, "
                f"but got {len(sample_args)} sample_args for {len(callables)} callables"
            )
        if len(sample_kwargs) != len(callables):
            raise ValueError(
                "Expected sample_kwargs to have the same length as callables, "
                f"but got {len(sample_kwargs)} sample_kwargs for {len(callables)} callables"
            )
    else:
        # Custom logic for interleaved pipeline parallelism
        # Note: This is tightly coupled with the Megatron-core
        # implementation of interleaved pipeline parallelism at
        # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py.
        # Note: The model is assumed to consist of layers
        # (corresponding to callables) that are grouped into
        # model chunks. _num_layers_per_chunk is a list of integers
        # that indicates the number of layers in each model chunk.
        # _order is a list of chunk indices (1-indexed) that
        # indicates the order in which the layers are evaluated.
        # Positive values indicate forward passes and negative
        # values indicate backward passes. Each
        # entry in sample_args corresponds to one of the forward
        # passes.
        _order_without_wgrad = []
        for c_id in _order:
            if ceil(c_id) != c_id:
                delay_wgrad_compute = True
                continue
            _order_without_wgrad.append(c_id)
        num_model_chunks = max(_order_without_wgrad)
        num_microbatches = len(_order_without_wgrad) // num_model_chunks // 2
        if num_model_chunks * num_microbatches * 2 != len(_order_without_wgrad):
            raise ValueError(
                f"Pipeline-parallel order dimension mismatch: num_model_chunks ({num_model_chunks})"
                f" * num_microbatches ({num_microbatches}) * 2 ="
                f" {num_model_chunks * num_microbatches * 2}, but len(_order_without_wgrad) ="
                f" {len(_order_without_wgrad)}"
            )

        # When delay_wgrad_compute is enabled, each layer is treated as a model chunk, which
        # allows for fine-grained graph capture order.
        if delay_wgrad_compute:
            if _num_layers_per_chunk is None:
                raise ValueError(
                    "'_num_layers_per_chunk' must be provided when delay_wgrad_compute is True."
                )
            for num_layers in _num_layers_per_chunk:
                if num_layers != 1:
                    raise ValueError(
                        "Each model chunk must have only one layer when delay_wgrad_compute is"
                        f" True, but got {num_layers} layers."
                    )

        # Determine number of layers in each model chunk.
        if _num_layers_per_chunk is None:
            if not (
                len(sample_args) * 2 >= len(_order_without_wgrad)
                and (len(sample_args) * 2 % len(_order_without_wgrad) == 0)
            ):
                raise ValueError(
                    f"{len(sample_args)} * 2 >= {len(_order_without_wgrad)} and"
                    f" {len(sample_args)} * 2 % {len(_order_without_wgrad)} == 0"
                )
            num_layers = len(sample_args) // num_model_chunks // num_microbatches
            _num_layers_per_chunk = [num_layers] * num_model_chunks
        else:
            if not (
                isinstance(_num_layers_per_chunk, int)
                or len(_num_layers_per_chunk) == num_model_chunks
            ):
                raise ValueError(
                    "If _num_layers_per_chunk is provided, it must be an integer or a list of"
                    f" {num_model_chunks} integers, but got {_num_layers_per_chunk}."
                )
            if isinstance(_num_layers_per_chunk, int):
                _num_layers_per_chunk = [_num_layers_per_chunk] * num_model_chunks
        total_num_layers = sum(_num_layers_per_chunk)
        if len(callables) != total_num_layers:
            raise ValueError(
                f"Callables should have ({total_num_layers}) "
                + f"entries when order input is provided but got {len(callables)}."
            )
        if len(sample_args) != total_num_layers * num_microbatches:
            raise ValueError(
                f"Expected {total_num_layers * num_microbatches} "
                + f"args tuple, but got {len(sample_args)}."
            )

        # Calculate the starting index of each chunk in callables for future use.
        _prefix_num_layers = [0]
        for m_chunk in range(num_model_chunks):
            num_layers = _num_layers_per_chunk[m_chunk]
            _prefix_num_layers.append(_prefix_num_layers[-1] + num_layers)

        if len(sample_kwargs) != len(sample_args):
            raise ValueError(
                "Pipeline-parallel schedule requires sample_kwargs and sample_args to have "
                f"the same length, but got {len(sample_kwargs)} sample_kwargs "
                f"for {len(sample_args)} sample_args"
            )

    # Check reuse graph conditions and reorganize sample_args and sample_kwargs.
    # Note: When capturing a graph, we hold onto the args and kwargs so we have static buffers
    # when the graph is replayed. If two model chunk microbatches have no overlap between their
    # forward and backward, then we can reduce memory usage by reusing the same static buffers.
    if _reuse_graph_input_output_buffers:
        if _order is None:
            raise ValueError(
                "`_order` must be provided when `_reuse_graph_input_output_buffers` is True."
            )
        if not is_training:
            raise RuntimeError(
                "`_reuse_graph_input_output_buffers` is only available in training mode."
            )
        if isinstance(sample_args, tuple):
            sample_args = list(sample_args)
        if isinstance(sample_kwargs, tuple):
            sample_kwargs = list(sample_kwargs)

        # Reorganize args and kwargs for input tensor reuse.
        # fwd_sample_qs is keyed by model chunk index. The value is a queue of tuples.
        # Each tuple contains the sample key signature and its fwd_idx. When we finish a backward
        # chunk, we pop the corresponding fwd_idx and push to the consumed_sample_q.
        # consumed_sample_q is keyed by the sample key signature. The value is a queue of the
        # fwd_idx whose backward has been called so that we can reuse the same static buffers.
        # In this way, we can reuse the same static input buffers for the non-overlapping samples
        # with the same input signature.
        fwd_sample_qs = {}
        consumed_sample_q = {}
        fwd_idx = [0] * num_model_chunks
        for c_id in _order:
            m_chunk = abs(ceil(c_id)) - 1

            if c_id > 0:
                sample_start_idx = (_prefix_num_layers[m_chunk] * num_microbatches) + (
                    fwd_idx[m_chunk] * _num_layers_per_chunk[m_chunk]
                )
                fwd_sample_idx = [
                    sample_start_idx + i for i in range(_num_layers_per_chunk[m_chunk])
                ]
                if m_chunk not in fwd_sample_qs:
                    fwd_sample_qs[m_chunk] = []
                for per_callable_fwd_idx in fwd_sample_idx:
                    sample_args_keys = tuple(
                        (t.shape, t.dtype, t.layout) for t in sample_args[per_callable_fwd_idx]
                    )
                    sample_kwargs_keys = tuple(
                        (k, v.shape, v.dtype, v.layout)
                        for k, v in sorted(sample_kwargs[per_callable_fwd_idx].items())
                    )
                    sample_keys = sample_args_keys + sample_kwargs_keys

                    fwd_sample_qs[m_chunk].append((sample_keys, per_callable_fwd_idx))
                    if consumed_sample_q.get(sample_keys, []):
                        reuse_fwd_idx = consumed_sample_q[sample_keys].pop(0)
                        sample_args[per_callable_fwd_idx] = sample_args[reuse_fwd_idx]
                        sample_kwargs[per_callable_fwd_idx] = sample_kwargs[reuse_fwd_idx]
                fwd_idx[m_chunk] += 1
            elif ceil(c_id) != c_id:
                continue
            else:
                num_consumed_samples = min(
                    len(fwd_sample_qs[m_chunk]), _num_layers_per_chunk[m_chunk]
                )
                for sample_keys, per_callable_fwd_idx in fwd_sample_qs[m_chunk][
                    :num_consumed_samples
                ]:
                    if sample_keys not in consumed_sample_q:
                        consumed_sample_q[sample_keys] = []
                    consumed_sample_q[sample_keys].append(per_callable_fwd_idx)
                fwd_sample_qs[m_chunk] = fwd_sample_qs[m_chunk][num_consumed_samples:]

    if cache_quantized_params:
        # Initialize flag that controls FP8 weight updates
        qstate = FP8GlobalStateManager.quantization_state
        if qstate.skip_fp8_weight_update_tensor is None:
            qstate.skip_fp8_weight_update_tensor = torch.empty(
                1, dtype=torch.float32, device="cuda"
            )
        qstate.skip_fp8_weight_update_tensor.fill_(False)

    # Check callables
    for c in callables:
        if isinstance(c, torch.nn.Module):
            if not (
                len(c._backward_hooks) == 0
                and len(c._backward_pre_hooks) == 0
                and len(c._forward_hooks) == 0
                and len(c._forward_pre_hooks) == 0
            ):
                raise RuntimeError(
                    "Modules must not have hooks registered at the time they are passed. "
                    + "However, registering hooks on modules after passing them "
                    + "through make_graphed_callables is allowed. If you need hooks during "
                    + "capture, pass them with capture_time_hooks so they run outside CUDA "
                    + "graph capture and are not replayed."
                )
            if not all(b.requires_grad is False for b in c.buffers()):
                raise RuntimeError(
                    "In any :class:`~torch.nn.Module` passed to "
                    + ":func:`~make_graphed_callables`, only parameters may be trainable. "
                    + "All buffers must have ``requires_grad=False``."
                )

    # Flatten callable arguments
    per_callable_kwargs_keys = [list(kwargs.keys()) for kwargs in sample_kwargs]
    flatten_sample_args = []
    for args, kwargs, kwargs_keys in zip(sample_args, sample_kwargs, per_callable_kwargs_keys):
        flatten_arg, _ = _tree_flatten(args)
        flatten_kwarg, _ = _tree_flatten([kwargs[key] for key in kwargs_keys])
        flatten_sample_args.append(tuple(flatten_arg + flatten_kwarg))
        if not all(isinstance(arg, torch.Tensor) for arg in flatten_arg):
            raise TypeError(
                "In the beta API, sample_args "
                + "for each callable must contain only Tensors. Other types are not allowed."
            )

    # If a callable is an nn.Module, its graph's full input surface is the args the user explicitly
    # passes to forward (ie, its sample_args) AND the module's parameter attributes.
    # Note: These per_callable_* variables are not actually
    # per-callable, but per-forward-pass (see description of _order).
    # The names are kept for consistency with
    # PyTorch make_graphed_callables.
    per_callable_len_user_args = [len(args) for args in flatten_sample_args]
    if _order is None:
        per_callable_module_params = [
            tuple(c.parameters()) if isinstance(c, torch.nn.Module) else () for c in callables
        ]
        per_callable_static_input_surfaces = [
            flatten_sample_args[i] + per_callable_module_params[i] for i in range(len(callables))
        ]
    else:
        per_callable_module_params = []
        for m_chunk in range(num_model_chunks):
            for _ in range(num_microbatches):
                for l_no in range(_num_layers_per_chunk[m_chunk]):
                    per_callable_module_params.append(
                        tuple(callables[_prefix_num_layers[m_chunk] + l_no].parameters())
                        if isinstance(
                            callables[_prefix_num_layers[m_chunk] + l_no],
                            torch.nn.Module,
                        )
                        else ()
                    )
        if len(per_callable_module_params) != len(flatten_sample_args):
            raise ValueError(
                "Pipeline-parallel dimension mismatch: "
                f"per_callable_module_params has {len(per_callable_module_params)} entries, "
                f"but flatten_sample_args has {len(flatten_sample_args)} entries"
            )
        per_callable_static_input_surfaces = [
            flatten_sample_args[i] + per_callable_module_params[i]
            for i in range(len(flatten_sample_args))
        ]

    fwd_graphs = [torch.cuda.CUDAGraph() for _ in range(len(flatten_sample_args))]
    bwd_graphs = [torch.cuda.CUDAGraph() for _ in range(len(flatten_sample_args))]
    bwd_dw_graphs = [torch.cuda.CUDAGraph() for _ in range(len(flatten_sample_args))]
    graph_callables = [None for _ in range(len(flatten_sample_args))]

    def _returned_param_grad_clone_slots(static_grad_inputs, module_params):
        """Snapshot static grad slots that need clones before Graphed.backward returns."""
        if not _clone_param_grads_on_return:
            return (False,) * len(static_grad_inputs)
        module_param_start = len(static_grad_inputs) - len(module_params)
        return tuple(
            idx >= module_param_start
            and not getattr(module_params[idx - module_param_start], "skip_backward_post_hook", False)
            for idx in range(len(static_grad_inputs))
        )

    # For cases with multiple active RNG states, e.g. TP.
    if graph_safe_rng_available():
        for _, state in get_all_rng_states().items():
            for fwd_graph, bwd_graph, bwd_dw_graph in zip(fwd_graphs, bwd_graphs, bwd_dw_graphs):
                fwd_graph.register_generator_state(state)
                bwd_graph.register_generator_state(state)
                bwd_dw_graph.register_generator_state(state)

    mempool = graph_pool_handle() if pool is None else pool

    # Warmup
    # Hopefully prevents cudnn benchmarking and other lazy-initialization cuda work
    # from ending up in any captures.
    torch.cuda.synchronize()

    # Get warmup func and func_idx.
    warmup_func_idx = []
    warmup_func = []
    if _order is None:
        for func_idx, func in enumerate(callables):
            warmup_func_idx.append(func_idx)
            warmup_func.append(func)
    else:
        fwd_idx = [0] * num_model_chunks
        for c_id in _order:
            if c_id > 0:
                m_chunk = c_id - 1
                for l_no in range(_num_layers_per_chunk[m_chunk]):
                    func = callables[_prefix_num_layers[m_chunk] + l_no]
                    func_idx = (_prefix_num_layers[m_chunk] * num_microbatches) + (
                        fwd_idx[m_chunk] * _num_layers_per_chunk[m_chunk] + l_no
                    )
                    warmup_func_idx.append(func_idx)
                    warmup_func.append(func)
                fwd_idx[m_chunk] += 1
    if len(warmup_func) != len(sample_args):
        raise ValueError(f"Warmup runs {len(warmup_func)} don't match args {len(sample_args)}.")
    if len(warmup_func_idx) != len(set(warmup_func_idx)):
        raise RuntimeError(
            f"Warmup runs {len(warmup_func)} but only {len(set(warmup_func_idx))} are unique."
        )

    # Filter the TE modules that cudagraph can access.
    visited_te_modules = {}
    need_bwd_dw_graph = {}

    def _call_capture_time_forward_pre_hooks(callable_idx, func, args, kwargs) -> None:
        hooks = capture_time_hooks[callable_idx]
        with_kwargs = hooks["forward_pre_hooks_with_kwargs"]
        for hook_id, hook in hooks["forward_pre_hooks"].items():
            if hook_id in with_kwargs:
                _check_capture_time_hook_return(
                    hook(func, args, kwargs),
                    "forward_pre_hooks",
                    "args/kwargs",
                )
            else:
                _check_capture_time_hook_return(
                    hook(func, args),
                    "forward_pre_hooks",
                    "args",
                )

    def _call_capture_time_forward_hooks(callable_idx, func, args, kwargs, outputs) -> None:
        hooks = capture_time_hooks[callable_idx]
        with_kwargs = hooks["forward_hooks_with_kwargs"]
        for hook_id, hook in hooks["forward_hooks"].items():
            if hook_id in with_kwargs:
                _check_capture_time_hook_return(
                    hook(func, args, kwargs, outputs),
                    "forward_hooks",
                    "output",
                )
            else:
                _check_capture_time_hook_return(
                    hook(func, args, outputs),
                    "forward_hooks",
                    "output",
                )

    def _call_capture_time_backward_pre_hooks(callable_idx, func, grad_outputs) -> None:
        for hook in capture_time_hooks[callable_idx]["backward_pre_hooks"].values():
            _check_capture_time_hook_return(
                hook(func, grad_outputs),
                "backward_pre_hooks",
                "grad_output",
            )

    def _call_capture_time_backward_hooks(callable_idx, func, grad_inputs, grad_outputs) -> None:
        for hook in capture_time_hooks[callable_idx]["backward_hooks"].values():
            _check_capture_time_hook_return(
                hook(func, grad_inputs, grad_outputs),
                "backward_hooks",
                "grad_input",
            )

    def _make_grad_outputs(outputs):
        return tuple(
            torch.empty_like(o) if o is not None and o.requires_grad else None for o in outputs
        )

    def _run_warmup_forward(func_idx, func, callable_idx, register_discovery_hooks=True):
        args = sample_args[func_idx]
        kwargs = sample_kwargs[func_idx]

        def hook_fn(module, inputs, outputs, func_idx=func_idx):  # pylint: disable=unused-argument
            modules = set()
            if isinstance(module, TransformerEngineBaseModule):
                modules.add(module)
            # If forward is called on a BasicOperation directly the hook will run.
            elif isinstance(module, BasicOperation):
                modules.add(module)
            elif hasattr(module, "need_backward_dw") and hasattr(module, "backward_dw"):
                modules.add(module)
            # If forward is called on a te.ops.Sequential it is not called on its constituent ops.
            elif isinstance(module, Sequential):
                if module._module_groups is None:
                    raise RuntimeError("module._module_groups should have been initialized by warmup")
                for module_group in module._module_groups:
                    if isinstance(module_group, OperationFuser):
                        for basic_op in module_group._basic_ops:
                            modules.add(basic_op)
            if modules:
                if func_idx not in visited_te_modules:
                    visited_te_modules[func_idx] = modules
                else:
                    visited_te_modules[func_idx].update(modules)

        _call_capture_time_forward_pre_hooks(callable_idx, func, args, kwargs)
        hooks = []
        if register_discovery_hooks and isinstance(func, torch.nn.Module):
            for module in func.modules():
                hooks.append(module.register_forward_hook(hook_fn))
        outputs = func(*args, **kwargs)
        for hook in hooks:
            hook.remove()
        _call_capture_time_forward_hooks(callable_idx, func, args, kwargs, outputs)
        flatten_outputs, _ = _tree_flatten(outputs)
        return flatten_outputs

    def _run_warmup_backward(func_idx, func, outputs, warmup_iter, callable_idx) -> None:
        static_input_surface = per_callable_static_input_surfaces[func_idx]
        inputs = tuple(i for i in static_input_surface if i is not None and i.requires_grad)
        outputs_requiring_grad = tuple(o for o in outputs if o is not None and o.requires_grad)
        grad_outputs = _make_grad_outputs(outputs)

        _call_capture_time_backward_pre_hooks(callable_idx, func, grad_outputs)
        with _none_grad_context_wrapper(inputs):
            torch.autograd.backward(
                outputs_requiring_grad,
                grad_tensors=tuple(o for o in grad_outputs if o is not None),
            )
            grad_inputs = tuple(input.grad for input in inputs)
        _call_capture_time_backward_hooks(callable_idx, func, grad_inputs, grad_outputs)

        # Filter module params that get None grad from grad_inputs and remove them
        # from static_input_surface. This is to ensure that the backward hooks
        # registered to these params are not wrongly triggered.
        num_required_grad_sample_args = sum(
            isinstance(arg, torch.Tensor) and arg.requires_grad
            for arg in flatten_sample_args[func_idx]
        )
        required_grad_input_idx = []
        for i, arg in enumerate(static_input_surface):
            if isinstance(arg, torch.Tensor) and arg.requires_grad:
                required_grad_input_idx.append(i)
        module_params_with_grad = []
        for grad_inputs_idx, inputs_idx in enumerate(required_grad_input_idx):
            if grad_inputs[grad_inputs_idx] is None and grad_inputs_idx < num_required_grad_sample_args:
                if not allow_unused_input:
                    raise RuntimeError(
                        "The input tensor requires grad, but the grad is None after backward pass."
                    )
            elif grad_inputs[grad_inputs_idx] is not None and grad_inputs_idx >= num_required_grad_sample_args:
                module_params_with_grad.append(static_input_surface[inputs_idx])
        if len(module_params_with_grad) != len(per_callable_module_params[func_idx]):
            if warmup_iter != 0:
                raise RuntimeError(
                    "no-grad params should only be used as inputs in the first warmup"
                    f" iteration, but found in iteration {warmup_iter}"
                )
            per_callable_module_params[func_idx] = tuple(module_params_with_grad)
            static_input_surface = flatten_sample_args[func_idx] + tuple(module_params_with_grad)
            per_callable_static_input_surfaces[func_idx] = static_input_surface

        # Run wgrad. This is essential for some TE modules when they have
        # delay_wgrad_compute enabled.
        need_backward_dw = False
        for module in visited_te_modules.get(func_idx, set()):
            if hasattr(module, "need_backward_dw") and module.need_backward_dw():
                need_backward_dw = True
                module.backward_dw()
        need_bwd_dw_graph[func_idx] = need_backward_dw

    def _run_warmup_iteration(warmup_iter, register_discovery_hooks):
        if _order is None:
            warmup_outputs = []
            for func_idx, func in zip(warmup_func_idx, warmup_func):
                outputs = _run_warmup_forward(
                    func_idx,
                    func,
                    func_idx,
                    register_discovery_hooks=register_discovery_hooks,
                )
                warmup_outputs.append((func_idx, func, outputs))
            if is_training:
                for func_idx, func, outputs in reversed(warmup_outputs):
                    _run_warmup_backward(func_idx, func, outputs, warmup_iter, func_idx)
            return

        per_fwd_outputs = {}
        fwd_idx = [0] * num_model_chunks
        bwd_idx = [0] * num_model_chunks
        for c_id in _order:
            if c_id > 0:
                m_chunk = c_id - 1
                for l_no in range(_num_layers_per_chunk[m_chunk]):
                    callable_idx = _prefix_num_layers[m_chunk] + l_no
                    per_callable_fwd_idx = (_prefix_num_layers[m_chunk] * num_microbatches) + (
                        fwd_idx[m_chunk] * _num_layers_per_chunk[m_chunk] + l_no
                    )
                    func = callables[callable_idx]
                    outputs = _run_warmup_forward(
                        per_callable_fwd_idx,
                        func,
                        callable_idx,
                        register_discovery_hooks=register_discovery_hooks,
                    )
                    per_fwd_outputs[per_callable_fwd_idx] = outputs
                fwd_idx[m_chunk] += 1
            elif ceil(c_id) == c_id:
                if is_training:
                    m_chunk = -c_id - 1
                    for l_no in reversed(range(_num_layers_per_chunk[m_chunk])):
                        callable_idx = _prefix_num_layers[m_chunk] + l_no
                        per_callable_bwd_idx = (
                            _prefix_num_layers[m_chunk] * num_microbatches
                        ) + (bwd_idx[m_chunk] * _num_layers_per_chunk[m_chunk] + l_no)
                        func = callables[callable_idx]
                        outputs = per_fwd_outputs[per_callable_bwd_idx]
                        _run_warmup_backward(
                            per_callable_bwd_idx,
                            func,
                            outputs,
                            warmup_iter,
                            callable_idx,
                        )
                    bwd_idx[m_chunk] += 1

    # Run warmup on the same stream as capture so workspace buffers
    # stay in the same CUDA context and don't need re-allocation.
    capture_stream = capture_stream or torch.cuda.Stream()
    with torch.cuda.stream(capture_stream):
        if pre_warmup_hook is not None:
            pre_warmup_hook()

        for warmup_iter in range(num_warmup_iters):
            _run_warmup_iteration(warmup_iter, register_discovery_hooks=True)

        # TE discovery temporarily registers forward hooks, and Dynamo guards
        # compiled modules on hook state. Capture runs after those hooks are
        # removed, so warm the capture-equivalent specialization as well.
        compiled_callables = any(
            getattr(func, "_compiled_call_impl", None) is not None
            or hasattr(getattr(func, "forward", None), "_torchdynamo_orig_callable")
            for func in callables
        )
        if num_warmup_iters > 0 and compiled_callables:
            _run_warmup_iteration(
                num_warmup_iters,
                register_discovery_hooks=False,
            )

        if post_warmup_hook is not None:
            post_warmup_hook()
    torch.cuda.synchronize()

    import gc
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

    # All captures here share a mempool. To avoid replays corrupting each other's memory,
    # the safest approach is to capture all passes in the same order they'll run:
    # fwd 1, fwd 2, ... fwd N, then bwd N, bwd N-1, ... bwd 1.

    if _order is not None:  # pylint: disable=too-many-nested-blocks
        per_callable_static_outputs = [None] * len(flatten_sample_args)
        per_callable_output_unflatten_spec = [None] * len(flatten_sample_args)
        per_callable_static_grad_outputs = [None] * len(flatten_sample_args)
        per_callable_static_grad_inputs = [None] * len(flatten_sample_args)
        per_callable_returned_param_grad_clone_slots = [None] * len(flatten_sample_args)
        fwd_idx = [0] * num_model_chunks
        bwd_idx = [0] * num_model_chunks
        static_grad_outputs_dict = {}
        wgrad_validation_list = [None] * len(_order)
        previous_chunk_last_callable_bwd_idx = None
        for i, c_id in enumerate(_order):
            if c_id > 0:
                if not isinstance(c_id, int):
                    raise TypeError(
                        f"Forward order value must be an integer, but got {type(c_id).__name__}."
                    )
                # Capture forward graph for model chunk c_id, microbatch fwd_idx[c_id-1]
                m_chunk = c_id - 1
                for l_no in range(_num_layers_per_chunk[m_chunk]):
                    callable_idx = _prefix_num_layers[m_chunk] + l_no
                    func = callables[callable_idx]
                    per_callable_fwd_idx = (_prefix_num_layers[m_chunk] * num_microbatches) + (
                        fwd_idx[m_chunk] * _num_layers_per_chunk[m_chunk] + l_no
                    )
                    args = sample_args[per_callable_fwd_idx]
                    kwargs = sample_kwargs[per_callable_fwd_idx]
                    fwd_graph = fwd_graphs[per_callable_fwd_idx]
                    _call_capture_time_forward_pre_hooks(callable_idx, func, args, kwargs)
                    with _graph_context_wrapper(fwd_graph, pool=mempool, stream=capture_stream):
                        outputs = func(*args, **kwargs)
                    _call_capture_time_forward_hooks(callable_idx, func, args, kwargs, outputs)
                    flatten_outputs, spec = _tree_flatten(outputs)
                    per_callable_static_outputs[per_callable_fwd_idx] = tuple(flatten_outputs)
                    per_callable_output_unflatten_spec[per_callable_fwd_idx] = spec
                    graph_callables[per_callable_fwd_idx] = func
                fwd_idx[m_chunk] += 1
            else:
                # Capture backward graph for model chunk c_id, microbatch bwd_idx[-c_id-1]
                m_chunk = -ceil(c_id) - 1
                previous_per_callable_bwd_idx = None
                for l_no in list(reversed(range(_num_layers_per_chunk[m_chunk]))):
                    callable_idx = _prefix_num_layers[m_chunk] + l_no
                    per_callable_bwd_idx = (_prefix_num_layers[m_chunk] * num_microbatches) + (
                        bwd_idx[m_chunk] * _num_layers_per_chunk[m_chunk] + l_no
                    )
                    if ceil(c_id) == c_id and need_bwd_dw_graph.get(per_callable_bwd_idx, False):
                        # Check if bwd graph has corresponding wgrad graph:
                        # Number of dgrad backward graphs should be equal to number of
                        # wgrad backward graphs.
                        # Note: For MCore, the validation rule is more strict (the next backward
                        # of dgrad graph must be corresponding wgrad graph).
                        if wgrad_validation_list[i] is None:
                            same_bwd_c_id_list = [i]
                            num_wgrad_c_id = 0
                            for idx in range(i + 1, len(_order)):
                                if _order[idx] > 0:
                                    continue
                                if _order[idx] == c_id:
                                    same_bwd_c_id_list.append(idx)
                                if _order[idx] + 0.5 == c_id:
                                    num_wgrad_c_id += 1
                                if len(same_bwd_c_id_list) == num_wgrad_c_id:
                                    for same_c_id_idx in same_bwd_c_id_list:
                                        wgrad_validation_list[same_c_id_idx] = True
                                    break
                                if len(same_bwd_c_id_list) < num_wgrad_c_id:
                                    # It's impossible to have more wgrad than dgrad.
                                    wgrad_validation_list[i] = False
                                    break
                            if wgrad_validation_list[i] is None:
                                wgrad_validation_list[i] = False
                            if not wgrad_validation_list[i]:
                                raise RuntimeError(
                                    f"Number of wgrad graph({num_wgrad_c_id}) doesn't match number "
                                    f"of dgrad graphs ({len(same_bwd_c_id_list)}) for chunk {c_id}."
                                )
                    elif ceil(c_id) != c_id:
                        per_callable_bwd_idx -= _num_layers_per_chunk[m_chunk]
                        if not is_training:
                            raise RuntimeError("Only training mode supports backward_dw.")
                        # If no one module needs the backward_dw, the bwd_dw_graph will be empty.
                        # So skip capturing it. For backward_dw, the order value is c_id - 0.5 to indicate
                        # the specific order of backward_dw.
                        if ceil(c_id) - c_id != 0.5:
                            raise ValueError(
                                "The order diff of wgrad and dgrad must be 0.5, "
                                f"get {ceil(c_id) - c_id}."
                            )
                        if not need_bwd_dw_graph.get(per_callable_bwd_idx, False):
                            raise RuntimeError(
                                "No module needs wgrad computation but get float in order"
                            )
                        bwd_dw_graph = bwd_dw_graphs[per_callable_bwd_idx]
                        with _graph_context_wrapper(bwd_dw_graph, pool=mempool, stream=capture_stream):
                            for module in visited_te_modules[per_callable_bwd_idx]:
                                if (
                                    hasattr(module, "need_backward_dw")
                                    and module.need_backward_dw()
                                ):
                                    module.backward_dw()
                        continue

                    static_input_surface = per_callable_static_input_surfaces[per_callable_bwd_idx]
                    static_outputs = per_callable_static_outputs[per_callable_bwd_idx]
                    bwd_graph = bwd_graphs[per_callable_bwd_idx]
                    # For now, assumes all static_outputs require grad
                    if _reuse_graph_input_output_buffers:
                        # Note for _reuse_graph_input_output_buffers: grad output is only used
                        # within backward, so we can reuse the same static buffers every time.
                        static_grad_outputs_keys = tuple(
                            (o.shape, o.dtype, o.layout)
                            for o in static_outputs
                            if o is not None and o.requires_grad
                        )
                        if static_grad_outputs_keys in static_grad_outputs_dict:
                            static_grad_outputs = static_grad_outputs_dict[static_grad_outputs_keys]
                        else:
                            static_grad_outputs = tuple(
                                torch.empty_like(o) if o is not None and o.requires_grad else None
                                for o in static_outputs
                            )
                            static_grad_outputs_dict[static_grad_outputs_keys] = static_grad_outputs
                    else:
                        static_grad_outputs = tuple(
                            torch.empty_like(o) if o is not None and o.requires_grad else None
                            for o in static_outputs
                        )
                    if is_training:
                        func = graph_callables[per_callable_bwd_idx]
                        _call_capture_time_backward_pre_hooks(
                            callable_idx,
                            func,
                            static_grad_outputs,
                        )
                        inputs = tuple(i for i in static_input_surface if i is not None and i.requires_grad)
                        with _none_grad_context_wrapper(inputs), _graph_context_wrapper(
                            bwd_graph, pool=mempool, stream=capture_stream
                        ):
                            torch.autograd.backward(
                                tuple(
                                    o for o in static_outputs if o is not None and o.requires_grad
                                ),
                                grad_tensors=tuple(o for o in static_grad_outputs if o is not None),
                                retain_graph=retain_graph_in_backward,
                            )
                            grad_inputs = tuple(input.grad for input in inputs)
                        _call_capture_time_backward_hooks(
                            callable_idx,
                            func,
                            grad_inputs,
                            static_grad_outputs,
                        )

                    # Constructs a tuple suitable for returning from Graphed.backward:
                    # Pads out the actually-needed grads with Nones in gradient slots for inputs
                    # that don't require grad. I couldn't think of a one-liner for this pattern.
                    static_grad_inputs = []
                    grad_idx = 0
                    for arg in static_input_surface:
                        if is_training and isinstance(arg, torch.Tensor) and arg.requires_grad:
                            static_grad_inputs.append(grad_inputs[grad_idx])
                            grad_idx += 1
                        else:
                            static_grad_inputs.append(None)  # type: ignore[arg-type]
                    static_grad_inputs = tuple(static_grad_inputs)  # type: ignore[assignment]

                    per_callable_static_grad_outputs[per_callable_bwd_idx] = static_grad_outputs
                    per_callable_static_grad_inputs[per_callable_bwd_idx] = static_grad_inputs
                    returned_param_grad_clone_slots = _returned_param_grad_clone_slots(
                        static_grad_inputs,
                        per_callable_module_params[per_callable_bwd_idx],
                    )
                    per_callable_returned_param_grad_clone_slots[per_callable_bwd_idx] = (
                        returned_param_grad_clone_slots
                    )

                    # Weak ref the static outputs and static grad inputs that are no longer needed
                    # in the following steps. These two type of tensors are both in cudagraph
                    # mempool, so we just deallocate them and let PyTorch's memory allocator
                    # reuse them elsewhere.
                    if _reuse_graph_input_output_buffers:
                        # Weak ref the static outputs of the forward pass of this backward. It's
                        # no longer needed after the corresponding backward graph is built up.
                        per_callable_static_outputs[per_callable_bwd_idx] = make_weak_ref(
                            static_outputs
                        )

                        # Parameter grads can be weak-refed immediately only when Graphed.backward
                        # will clone them before returning them to autograd users.
                        static_grad_inputs = per_callable_static_grad_inputs[per_callable_bwd_idx]
                        per_callable_static_grad_inputs[per_callable_bwd_idx] = tuple(
                            (
                                make_weak_ref(grad_input)
                                if returned_param_grad_clone_slots[idx] and grad_input is not None
                                else grad_input
                            )
                            for idx, grad_input in enumerate(static_grad_inputs)
                        )

                        # Weak ref the static grad inputs of the previous backward pass within the
                        # same chunk.
                        if previous_per_callable_bwd_idx is not None:
                            idx = previous_per_callable_bwd_idx
                            per_callable_static_grad_inputs[idx] = make_weak_ref(
                                per_callable_static_grad_inputs[idx]
                            )
                        previous_per_callable_bwd_idx = per_callable_bwd_idx

                        # Weak ref the static grad inputs of the previous chunk's last backward
                        # pass.
                        # Note: After a chunk's backward pass, we assume Mcore will send the grad
                        # input to another pipeline parallel rank and that the communication is
                        # finished before the end of the next chunk's backward pass.
                        if l_no == 0:
                            if previous_chunk_last_callable_bwd_idx is not None:
                                idx = previous_chunk_last_callable_bwd_idx
                                per_callable_static_grad_inputs[idx] = make_weak_ref(
                                    per_callable_static_grad_inputs[idx]
                                )
                            previous_chunk_last_callable_bwd_idx = per_callable_bwd_idx
                if ceil(c_id) == c_id:
                    bwd_idx[m_chunk] += 1
    else:
        # Capture forward graphs
        per_callable_static_outputs = []
        per_callable_output_unflatten_spec = []
        for func_idx, (func, args, kwargs, fwd_graph) in enumerate(
            zip(callables, sample_args, sample_kwargs, fwd_graphs)
        ):
            _call_capture_time_forward_pre_hooks(func_idx, func, args, kwargs)
            with _graph_context_wrapper(fwd_graph, pool=mempool, stream=capture_stream):
                outputs = func(*args, **kwargs)
            _call_capture_time_forward_hooks(func_idx, func, args, kwargs, outputs)
            graph_callables[func_idx] = func

            flatten_outputs, spec = _tree_flatten(outputs)
            per_callable_static_outputs.append(tuple(flatten_outputs))
            per_callable_output_unflatten_spec.append(spec)

        # Capture backward graphs in reverse order
        per_callable_static_grad_outputs = []
        per_callable_static_grad_inputs = []
        per_callable_returned_param_grad_clone_slots = []
        for static_input_surface, static_outputs, bwd_graph, bwd_dw_graph, bwd_idx in zip(
            reversed(per_callable_static_input_surfaces),
            reversed(per_callable_static_outputs),
            reversed(bwd_graphs),
            reversed(bwd_dw_graphs),
            reversed(range(len(per_callable_static_input_surfaces))),
        ):
            # For now, assumes all static_outputs require grad
            static_grad_outputs = tuple(
                torch.empty_like(o) if o is not None and o.requires_grad else None
                for o in static_outputs
            )
            if is_training:
                func = graph_callables[bwd_idx]
                _call_capture_time_backward_pre_hooks(bwd_idx, func, static_grad_outputs)
                inputs = tuple(i for i in static_input_surface if i is not None and i.requires_grad)
                with _none_grad_context_wrapper(inputs), _graph_context_wrapper(
                    bwd_graph, pool=mempool
                ):
                    torch.autograd.backward(
                        tuple(o for o in static_outputs if o is not None and o.requires_grad),
                        grad_tensors=tuple(o for o in static_grad_outputs if o is not None),
                        retain_graph=retain_graph_in_backward,
                    )
                    grad_inputs = tuple(input.grad for input in inputs)
                _call_capture_time_backward_hooks(bwd_idx, func, grad_inputs, static_grad_outputs)

                if need_bwd_dw_graph.get(bwd_idx, False):
                    with _graph_context_wrapper(bwd_dw_graph, pool=mempool, stream=capture_stream):
                        for module in visited_te_modules[bwd_idx]:
                            if hasattr(module, "need_backward_dw") and module.need_backward_dw():
                                module.backward_dw()
            # Constructs a tuple suitable for returning from Graphed.backward:
            # Pads out the actually-needed grads with Nones in gradient slots for inputs that
            # don't require grad. I couldn't think of a slick one-liner for this pattern.
            static_grad_inputs = []
            grad_idx = 0
            for arg in static_input_surface:
                if is_training and isinstance(arg, torch.Tensor) and arg.requires_grad:
                    static_grad_inputs.append(grad_inputs[grad_idx])
                    grad_idx += 1
                else:
                    static_grad_inputs.append(None)  # type: ignore[arg-type]
            static_grad_inputs = tuple(static_grad_inputs)  # type: ignore[assignment]

            per_callable_static_grad_outputs.append(static_grad_outputs)
            per_callable_static_grad_inputs.append(static_grad_inputs)
            per_callable_returned_param_grad_clone_slots.append(
                _returned_param_grad_clone_slots(
                    static_grad_inputs,
                    per_callable_module_params[bwd_idx],
                )
            )

        # Reverse the most recent per-callable lists.
        per_callable_static_grad_outputs = list(reversed(per_callable_static_grad_outputs))
        per_callable_static_grad_inputs = list(reversed(per_callable_static_grad_inputs))
        per_callable_returned_param_grad_clone_slots = list(
            reversed(per_callable_returned_param_grad_clone_slots)
        )
    # Now for every per_callable list, per_callable_*[i] holds the stuff for the ith callable.

    def make_graphed_autograd_function(
        fwd_graph,
        bwd_graph,
        module_params,
        kwargs_keys,
        len_user_args,
        output_unflatten_spec,
        static_input_surface,
        static_outputs,
        static_grad_outputs,
        static_grad_inputs,
        returned_param_grad_clone_slots,
    ):
        class Graphed(torch.autograd.Function):
            """Autograd function for graph replay."""

            @staticmethod
            def forward(ctx, skip_fp8_weight_update, cuda_graph_stream, cuda_graph_event, *inputs):
                # pylint: disable=missing-function-docstring

                # Set flag for whether to update FP8 weight updates
                ctx.is_first_module = FP8GlobalStateManager.is_first_fp8_module()
                if ctx.is_first_module and skip_fp8_weight_update is not None:
                    FP8GlobalStateManager.quantization_state.skip_fp8_weight_update_tensor.fill_(
                        skip_fp8_weight_update
                    )
                ctx.cuda_graph_stream = cuda_graph_stream
                ctx.cuda_graph_event = cuda_graph_event
                # Copy values from new tensors into static tensors
                for i in range(len_user_args):
                    if (
                        isinstance(static_input_surface[i], torch.Tensor)
                        and inputs[i] is not None
                        and static_input_surface[i].data_ptr() != inputs[i].data_ptr()
                    ):
                        static_input_surface[i].copy_(inputs[i])

                # Replay forward graph
                if cuda_graph_stream != torch.cuda.current_stream():
                    cuda_graph_stream.wait_stream(torch.cuda.current_stream())
                    with cuda_graph_stream:
                        fwd_graph.replay()
                    if cuda_graph_event is not None:
                        torch.cuda.current_stream().wait_event(cuda_graph_event)
                    else:
                        torch.cuda.current_stream().wait_stream(cuda_graph_stream)
                else:
                    fwd_graph.replay()
                if not isinstance(static_outputs, tuple):
                    raise TypeError(
                        "Expected static_outputs to be a tuple, but got"
                        f" {type(static_outputs).__name__}"
                    )
                return tuple(o.detach() if o is not None else o for o in static_outputs)

            @staticmethod
            @torch.autograd.function.once_differentiable
            def backward(ctx, *grads):
                # pylint: disable=missing-function-docstring

                # Replay backward graph
                if len(grads) != len(static_grad_outputs):
                    raise ValueError(
                        "Backward graph grad dimension mismatch: "
                        f"received {len(grads)} grads, "
                        f"but expected {len(static_grad_outputs)} static_grad_outputs"
                    )
                for g, grad in zip(static_grad_outputs, grads):
                    if g is not None:
                        # don't copy if autograd gods have been kind and the
                        # incoming grad is already in the right place
                        if g.data_ptr() != grad.data_ptr():
                            g.copy_(grad)
                if ctx.cuda_graph_stream != torch.cuda.current_stream():
                    ctx.cuda_graph_stream.wait_stream(torch.cuda.current_stream())
                    with ctx.cuda_graph_stream:
                        bwd_graph.replay()
                    if ctx.cuda_graph_event is not None:
                        torch.cuda.current_stream().wait_event(ctx.cuda_graph_event)
                    else:
                        torch.cuda.current_stream().wait_stream(ctx.cuda_graph_stream)
                else:
                    bwd_graph.replay()

                # Update FP8 scale factors if needed
                if ctx.is_first_module:
                    FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)

                # Input args that didn't require grad expect a None gradient.
                if not isinstance(static_grad_inputs, tuple):
                    raise TypeError(
                        "Expected static_grad_inputs to be a tuple, but got"
                        f" {type(static_grad_inputs).__name__}"
                    )
                grad_inputs = []
                for idx, grad_input in enumerate(static_grad_inputs):
                    if grad_input is None:
                        grad_inputs.append(None)
                    elif returned_param_grad_clone_slots[idx]:
                        # Returned parameter grads may be installed directly as param.grad.
                        # Clone to avoid exposing CUDA graph static buffers to autograd users.
                        grad_inputs.append(grad_input.detach().clone())
                    else:
                        grad_inputs.append(grad_input.detach())
                return (None, None, None) + tuple(grad_inputs)

        def functionalized(*user_args, **user_kwargs):

            # Decide whether to update FP8 weights
            skip_fp8_weight_update = None
            if cache_quantized_params:
                if "is_first_microbatch" not in user_kwargs or not isinstance(
                    user_kwargs["is_first_microbatch"], bool
                ):
                    raise ValueError(
                        "`is_first_microbatch` boolean kwarg must be provided for FP8 weight"
                        " caching."
                    )

                skip_fp8_weight_update = not user_kwargs["is_first_microbatch"]

            # The cuda_graph_stream and cuda_graph_event are used in the TE CUDA graph replay.
            # When replaying the graph in the cuda graph stream, the graph replay could overlap
            # with the work on main stream.
            # When cuda_graph_event is given, it should be an external event recorded
            # in the cuda graph and is used to sync-back to the main stream.
            # If cuda_graph_event is not given, it will be None and the graph replay will block
            # the main stream until it is finished.
            if "cuda_graph_stream" in user_kwargs:
                cuda_graph_stream = user_kwargs["cuda_graph_stream"]
                user_kwargs.pop("cuda_graph_stream")
            else:
                cuda_graph_stream = torch.cuda.current_stream()
            if "cuda_graph_event" in user_kwargs:
                cuda_graph_event = user_kwargs["cuda_graph_event"]
                user_kwargs.pop("cuda_graph_event")
            else:
                cuda_graph_event = None
            # Runs the autograd function with inputs == all inputs to
            # the graph that might require grad (explicit user args +
            # module parameters)
            # Assumes module params didn't change since capture.
            # Reconstruct the same flattened arg order as capture time.
            # User may pass some recorded kwargs as positional args, so
            # check user_args first (by position), then user_kwargs.
            user_pos_args = list(user_args)
            kwarg_values = []
            for key in kwargs_keys:
                if key in user_kwargs:
                    kwarg_values.append(user_kwargs[key])
                elif user_pos_args:
                    kwarg_values.append(user_pos_args.pop(0))
                # else: key was a default not passed — skip (not a tensor)
            flatten_user_kwargs, _ = _tree_flatten(kwarg_values)
            func_args = tuple(flatten_user_kwargs) + module_params
            out = Graphed.apply(
                skip_fp8_weight_update, cuda_graph_stream, cuda_graph_event, *func_args
            )
            return _tree_unflatten(out, output_unflatten_spec)

        return functionalized

    def make_graphed_attribute_functions(graph_idx):
        # Get te modules for current graph
        te_modules = visited_te_modules.get(graph_idx, set())

        # Attach backward_dw as an attribute to the graphed callable.
        def backward_dw():
            if need_bwd_dw_graph.get(graph_idx, False):
                bwd_dw_graphs[graph_idx].replay()

                # Trigger the grad accumulation hook for wgrad graphs.
                for module in te_modules:
                    if (
                        hasattr(module, "_trigger_wgrad_accumulation_and_reduce_hooks")
                        and module.need_backward_dw()
                    ):
                        module._trigger_wgrad_accumulation_and_reduce_hooks()

        # Attach reset as an attribute to the graphed callable.
        def reset():
            fwd_graphs[graph_idx].reset()
            bwd_graphs[graph_idx].reset()
            bwd_dw_graphs[graph_idx].reset()

        return backward_dw, reset

    # Put together the final graphed callables
    ret = []
    for i in range(len(sample_args)):
        graphed = make_graphed_autograd_function(
            fwd_graphs[i],
            bwd_graphs[i],
            per_callable_module_params[i],
            per_callable_kwargs_keys[i],
            per_callable_len_user_args[i],
            per_callable_output_unflatten_spec[i],
            per_callable_static_input_surfaces[i],
            per_callable_static_outputs[i],
            per_callable_static_grad_outputs[i],
            per_callable_static_grad_inputs[i],
            per_callable_returned_param_grad_clone_slots[i],
        )

        func = graph_callables[i]
        te_modules = visited_te_modules.get(i, set())
        if isinstance(func, torch.nn.Module):

            def make_graphed_forward(func, graph_training_state, graphed, orig_fwd, te_modules):
                def new_fwd(*user_args, **user_kwargs):
                    # If the module's training-or-eval state matches what we graphed,
                    # run the graph, otherwise run the original forward method
                    if func.training == graph_training_state:
                        # Set the FP8 group from global amax reduction.
                        if FP8GlobalStateManager.is_fp8_enabled():
                            fp8_recipe = FP8GlobalStateManager.get_fp8_recipe()
                            for m in func.modules():
                                if m not in te_modules:
                                    # Only Set the FP8 meta for the modules included by forward
                                    continue
                                if isinstance(m, TransformerEngineBaseModule):
                                    from transformer_engine.pytorch.attention.dot_product_attention import (
                                        DotProductAttention,
                                    )

                                    if (
                                        isinstance(m, DotProductAttention)
                                        and not fp8_recipe.fp8_mha
                                        and not fp8_recipe.fp8_dpa
                                    ):
                                        # Don't need to update FP8 meta for non-FP8 DPA
                                        continue
                                    m.fp8_meta["fp8_group"] = FP8GlobalStateManager.get_fp8_group()
                                    m.fp8_meta["recipe"] = FP8GlobalStateManager.get_fp8_recipe()
                                    FP8GlobalStateManager.add_fp8_tensors_to_global_buffer(
                                        m.fp8_meta,
                                    )
                                elif isinstance(m, BasicOperation):
                                    for mode in ("forward", "backward"):
                                        if m.num_quantizers(mode):
                                            m._fp8_metas[mode][
                                                "fp8_group"
                                            ] = FP8GlobalStateManager.get_fp8_group()
                                            m._fp8_metas[mode][
                                                "recipe"
                                            ] = FP8GlobalStateManager.get_fp8_recipe()
                                            FP8GlobalStateManager.add_fp8_tensors_to_global_buffer(
                                                m._fp8_metas[mode],
                                            )
                        return graphed(*user_args, **user_kwargs)
                    return orig_fwd(*user_args, **user_kwargs)

                return new_fwd

            forward = make_graphed_forward(func, func.training, graphed, func.forward, te_modules)
            if _order is None:
                func.forward = forward
                ret.append(func)
            else:
                ret.append(forward)
        else:
            ret.append(graphed)

        backward_dw_func, reset_func = make_graphed_attribute_functions(i)
        setattr(ret[-1], "backward_dw", backward_dw_func)
        setattr(ret[-1], "reset", reset_func)

    if just_one_callable:
        return ret[0]

    return tuple(ret)


def save_fp8_tensors(
    modules: Iterable[torch.nn.Module],
    recipe: Optional[Recipe],
) -> Optional[List[Any]]:
    """
    Returns the FP8 tensors for all modules
    with adjusted amax history sizes.
    """

    if not isinstance(recipe, DelayedScaling):
        return None

    fp8_tensors = []
    for module in modules:
        for m in module.modules():
            module_tensors = None
            if isinstance(m, TransformerEngineBaseModule):
                if m.primary_weights_in_fp8:
                    m.adjust_amax_history_length(recipe.amax_history_len)
                module_tensors = m.get_fp8_meta_tensors()
            elif isinstance(m, BasicOperation):
                m.reset_recipe_state(recipe=recipe)
                module_tensors = m._save_fp8_metas()
            fp8_tensors.append(module_tensors)
    return fp8_tensors


def restore_fp8_tensors(
    modules: Iterable[torch.nn.Module],
    fp8_tensors: Optional[List[Any]],
) -> None:
    """Restore FP8 tensors."""

    if fp8_tensors is None:
        return

    for module in modules:
        for m in module.modules():
            module_tensors = fp8_tensors.pop(0)
            if isinstance(m, TransformerEngineBaseModule):
                m.reset_fp8_meta_tensors(module_tensors)
            elif isinstance(m, BasicOperation):
                m._load_fp8_metas(module_tensors)
    if len(fp8_tensors) != 0:
        raise RuntimeError(
            f"Got FP8 state for {len(fp8_tensors)} more modules than expected. "
            "There is probably a discrepancy with `save_fp8_tensors`."
        )


def make_graphed_callables(
    modules: SingleOrTuple[Callable],
    sample_args: SingleOrTuple[Tuple[torch.Tensor, ...]],
    num_warmup_iters: int = 3,
    allow_unused_input: bool = False,
    sample_kwargs: Optional[SingleOrTuple[Dict[str, Any]]] = None,
    fp8_enabled: Optional[SingleOrTuple[bool]] = None,
    fp8_calibrating: Optional[bool] = None,
    fp8_recipe: Optional[Recipe] = None,
    fp8_group: Optional[dist_group_type] = None,
    fp8_weight_caching: Optional[bool] = None,
    enabled: Optional[SingleOrTuple[bool]] = None,
    calibrating: Optional[bool] = None,
    recipe: Optional[Recipe] = None,
    amax_reduction_group: Optional[dist_group_type] = None,
    cache_quantized_params: Optional[bool] = None,
    _order: Optional[List[int]] = None,
    _num_layers_per_chunk: Optional[List[int]] = None,
    pool: Optional[Tuple[int, ...]] = None,
    retain_graph_in_backward: bool = False,
    _reuse_graph_input_output_buffers: bool = False,
    _clone_param_grads_on_return: bool = True,
    pre_warmup_hook: Optional[Callable] = None,
    post_warmup_hook: Optional[Callable] = None,
    capture_time_hooks: Optional[List[Optional[Dict[str, Dict]]]] = None,
    capture_stream: Optional[torch.cuda.Stream] = None,
) -> Union[Callable, Tuple[Callable, ...]]:
    """
    Make CUDA graph version of Transformer Engine modules

    A variation of PyTorch's `make_graphed_callables` utility function
    with support for Transformer Engine modules and FP8. Please see
    the
    original PyTorch implementation for more documentation.

    .. warning::

       Arguments 'fp8_enabled', 'fp8_calibrating', 'fp8_recipe', 'fp8_group', and 'fp8_weight_caching' are deprecated.
       Use arguments 'enabled', 'calibrating', 'recipe', 'amax_reduction_group', and 'cache_quantized_params' instead.

    Graphing parameters
    -------------------
    modules: (tuple of) callable
             Callable or callables to graph.
    sample_args: (tuple of) tuple of torch.Tensor
                 Positional arguments to callable(s).
    num_warmup_iters: int, default = 3
                      Number of warmup iterations.
    allow_unused_input: bool, default = False
                        Whether to handle case where callable inputs
                        and outputs are disconnected in compute graph.
    sample_kwargs: (tuple of) dict, optional
                   Keyword arguments to callable(s)
    pool: (tuple of) int, default = None, optional
          An instance returned from function `torch.cuda.graph_pool_handle` that hints
          this graph may share memory with the indicated pool.
    retain_graph_in_backward: bool, default = False
                              Whether to set retain_graph=True in backward graph capture.
    _reuse_graph_input_output_buffers: bool, default = False
        Reduce memory usage by reusing input/output data buffers between
        graphs. Only supported with Mcore interleaved pipeline parallelism, i.e.
        when `_order` is provided. All callables in `modules` are assumed to have
        inputs and outputs with the same dtype and shape.
    _clone_param_grads_on_return: bool, default = True
        Clone parameter gradients before returning them from CUDA graph replay.
        Disabling this avoids the extra clone/copy and may improve performance,
        but returned parameter gradients will alias CUDA graph static gradient
        buffers. These tensors no longer have standard PyTorch returned-gradient
        lifetime semantics: a later replay of the same graph, or reused-buffer
        replay of another callable, may overwrite retained hook or `.grad`
        tensors. Only disable this when the caller consumes returned parameter
        gradients before any such overwrite can occur.
    pre_warmup_hook: callable, default = None
                      A hook function that will be called once before all warmup iterations
                      (not once per callable).
    post_warmup_hook: callable, default = None
                      A hook function that will be called once after all warmup iterations
                      (not once per callable).
    capture_time_hooks: list of dict, optional
                        Per-callable hooks invoked during warmup and graph capture, but
                        intentionally executed outside CUDA graph capture so they are not
                        recorded into the graph and are not replayed. Each hook must return
                        ``None``. Each list entry corresponds to one callable and may include
                        these keys:
                        ``"forward_pre_hooks"`` maps hook IDs to hooks with signature
                        ``hook(module, args)`` or ``hook(module, args, kwargs)`` when the ID
                        is present in ``"forward_pre_hooks_with_kwargs"``;
                        ``"forward_hooks"`` maps hook IDs to hooks with signature
                        ``hook(module, args, output)`` or ``hook(module, args, kwargs, output)``
                        when the ID is present in ``"forward_hooks_with_kwargs"``;
                        ``"backward_pre_hooks"`` maps hook IDs to
                        ``hook(module, grad_output)``;
                        ``"backward_hooks"`` maps hook IDs to
                        ``hook(module, grad_input, grad_output)``.

    Quantization parameters
    -----------------------
    enabled: (tuple of) bool, default = False
             whether or not to enable low precision quantization (FP8/FP4).
             If tuple, the length must match the number of modules.
    calibrating: bool, default = False
                 calibration mode allows collecting statistics such as amax and scale
                 data of quantized tensors even when executing without quantization enabled.
                 This is useful for saving an inference ready checkpoint while training
                 using a higher precision.
    recipe: recipe.Recipe, default = None
            recipe used for low precision quantization.
    amax_reduction_group: torch._C._distributed_c10d.ProcessGroup, default = None
                          distributed group over which amaxes for the quantized tensors
                          are reduced at the end of each training step.
    cache_quantized_params: bool, default = False
                            Whether or not to cache quantized weights across microbatches. if set to `True`,
                            the `is_first_microbatch` boolean argument must be passed into the forward
                            method for TransformerEngine modules. When storing primary weights in low precision
                            using TE's `quantized_model_init` API and using an quantization aware optimizer,
                            this arg must be set to `False` if calculating weight transposes' outside TE, e.g.,
                            in the optimizer step.

    """

    te_available = _prepare_runtime()

    # Handle deprecated args. If old kwargs are set, they are prioritized with warning.
    if fp8_enabled is not None:
        if enabled is not None:
            raise ValueError(
                "make_graphed_callables has deprecated `fp8_enabled` kwarg "
                "in favor of `enabled`, but both kwargs are set."
            )
        warnings.warn(
            "make_graphed_callables has deprecated `fp8_enabled` kwarg in favor of `enabled`. "
            "`fp8_enabled` will be removed in a future release.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        enabled = fp8_enabled
    if enabled is None:
        enabled = False

    if fp8_calibrating is not None:
        if calibrating is not None:
            raise ValueError(
                "make_graphed_callables has deprecated `fp8_calibrating` kwarg "
                "in favor of `calibrating`, but both kwargs are set."
            )
        warnings.warn(
            "make_graphed_callables has deprecated `fp8_calibrating` kwarg in favor of "
            "`calibrating`. `fp8_calibrating` will be removed in a future release.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        calibrating = fp8_calibrating
    if calibrating is None:
        calibrating = False

    if fp8_recipe is not None:
        if recipe is None:
            warnings.warn(
                "make_graphed_callables has deprecated `fp8_recipe` kwarg in favor of "
                "`recipe`. `fp8_recipe` will be removed in a future release.",
                category=DeprecationWarning,
                stacklevel=2,
            )
        else:
            raise ValueError(
                "make_graphed_callables has deprecated `fp8_recipe` kwarg "
                "in favor of `recipe`, but both kwargs are set."
            )
        recipe = fp8_recipe

    if fp8_group is not None:
        if amax_reduction_group is None:
            warnings.warn(
                "make_graphed_callables has deprecated `fp8_group` kwarg in favor of "
                "`amax_reduction_group`. `fp8_group` will be removed in a future release.",
                category=DeprecationWarning,
                stacklevel=2,
            )
        else:
            raise ValueError(
                "make_graphed_callables has deprecated `fp8_group` kwarg "
                "in favor of `amax_reduction_group`, but both kwargs are set."
            )
        amax_reduction_group = fp8_group

    if fp8_weight_caching is not None:
        if cache_quantized_params is not None:
            raise ValueError(
                "make_graphed_callables has deprecated `fp8_weight_caching` kwarg "
                "in favor of `cache_quantized_params`, but both kwargs are set."
            )
        warnings.warn(
            "make_graphed_callables has deprecated `fp8_weight_caching` kwarg in favor of "
            "`cache_quantized_params`. `fp8_weight_caching` will be removed in a future release.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        cache_quantized_params = fp8_weight_caching
    if cache_quantized_params is None:
        cache_quantized_params = False

    set_capture_start()

    # Handle single module.
    just_one_callable = False
    if not isinstance(modules, tuple):
        just_one_callable = True
        modules = (modules,)

    if not isinstance(enabled, tuple):
        if not isinstance(enabled, bool):
            raise TypeError(
                f"enabled must be a bool or a tuple of bools, but got {type(enabled).__name__}"
            )
        enabled = (enabled,) * len(modules)
    else:
        if len(enabled) != len(modules):
            raise ValueError(
                f"enabled length ({len(enabled)}) must match modules length ({len(modules)})"
            )
    if not te_available and (
        any(enabled)
        or calibrating
        or recipe is not None
        or amax_reduction_group is not None
        or cache_quantized_params
    ):
        raise _te_required_error("FP8/TE-specific graph capture")
    if any(enabled) and recipe is None:
        recipe = get_default_fp8_recipe()
    elif not any(enabled):
        recipe = None
    module_uses_fp8 = dict(zip((id(m) for m in modules), enabled))

    # Store FP8 tensors to reset later.
    saved_fp8_tensors = save_fp8_tensors(modules, recipe=recipe)

    # FP8 wrapper.
    old_call_funcs = {}

    def wrap_autocast(block):
        block_cls = type(block)
        if block_cls in old_call_funcs:
            return

        old_call_funcs[block_cls] = block_cls.__call__

        # Wrap the original call function of the module class.
        def call_func(self, *args, **kwargs):
            with autocast(
                enabled=module_uses_fp8.get(id(self), False),
                calibrating=calibrating,
                recipe=recipe,
                amax_reduction_group=amax_reduction_group,
                _graph=True,
            ):
                outputs = old_call_funcs[block_cls](self, *args, **kwargs)
            return outputs

        block_cls.__call__ = call_func

    forward_funcs = []
    for module in modules:
        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"Graphing for {type(module)} is not supported.")
        wrap_autocast(module)
        forward_funcs.append(module)

    if just_one_callable:
        forward_funcs = forward_funcs[0]
    else:
        forward_funcs = tuple(forward_funcs)

    # Save RNG state.
    if graph_safe_rng_available():
        generators = [
            torch.cuda.default_generators[torch.cuda.current_device()],
            *get_all_rng_states().values(),
        ]
        original_rng_states = [state.get_state() for state in generators]
    else:
        original_rng_states = torch.cuda.get_rng_state()

    graphed_callables = _make_graphed_callables(
        forward_funcs,
        sample_args,
        num_warmup_iters=num_warmup_iters,
        allow_unused_input=allow_unused_input,
        cache_quantized_params=cache_quantized_params,
        sample_kwargs=sample_kwargs,
        _order=_order,
        _num_layers_per_chunk=_num_layers_per_chunk,
        pool=pool,
        retain_graph_in_backward=retain_graph_in_backward,
        _reuse_graph_input_output_buffers=_reuse_graph_input_output_buffers,
        _clone_param_grads_on_return=_clone_param_grads_on_return,
        pre_warmup_hook=pre_warmup_hook,
        post_warmup_hook=post_warmup_hook,
        capture_time_hooks=capture_time_hooks,
        capture_stream=capture_stream,
    )

    # Ensures warmup does not affect numerics for ops such as dropout.
    if graph_safe_rng_available():
        for gen, state in zip(generators, original_rng_states):
            gen.set_state(state)
    else:
        torch.cuda.set_rng_state(original_rng_states)

    # Remove FP8 wrapper.
    for module_cls, old_call in old_call_funcs.items():
        module_cls.__call__ = old_call

    # Restore FP8 state.
    restore_fp8_tensors(modules, saved_fp8_tensors)

    set_capture_end()
    return graphed_callables
