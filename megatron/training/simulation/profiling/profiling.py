import inspect
import os
from functools import lru_cache, wraps
from typing import Callable, Dict, List, Optional, Tuple

import torch

from megatron.core.utils import get_model_config


class ProfilingHook:
    def __init__(self, module_list: List[torch.nn.Module] = None):
        self.module_list = module_list
        self.module_to_name: Dict[torch.nn.Module, str] = {}
        self.profiler_hook_handlers: List[torch.utils.hooks.RemovableHandle] = []

        self.record_funcs: List[Tuple[str, torch.profiler.record_function]] = []
        self.nvtx_range_messages: List[str] = []  # Messages associated with active NVTX ranges

        if _record_func_enabled and _nvtx_enabled:
            raise ValueError(
                "Can't set record_function and NVTX at the same time."
            )

        self.model_config = None
        self.deallocate_pipeline_outputs_original = None

    def profiling_range_push(self, msg=None):
        if _record_func_enabled:
            rec_func = torch.profiler.record_function(msg)
            rec_func.__enter__()

            # Push record function
            self.record_funcs.append((msg, rec_func))
        elif _nvtx_enabled:
            # Track messages to ensure consistency when popping
            self.nvtx_range_messages.append(msg)

            # Push NVTX range
            torch.cuda.nvtx.range_push(msg)

    def profiling_range_pop(self, msg=None):
        if _record_func_enabled:
            if len(self.record_funcs) == 0:
                raise RuntimeError("Attempted to pop torch profiler record function from empty stack")

            last_msg, last_rec_func = self.record_funcs.pop()
            if msg is not None and msg != last_msg:
                raise ValueError(
                    f"Attempted to pop torch profiler record function from stack with msg={msg}, "
                    f"but last record function has msg={last_msg}"
                )

            # Pop record function
            last_rec_func.__exit__(None, None, None)
            last_rec_func = None
        elif _nvtx_enabled:
            # Update list of NVTX range messages and check for consistency
            if len(self.nvtx_range_messages) == 0:
                raise RuntimeError("Attempted to pop NVTX range from empty stack")
            last_msg = self.nvtx_range_messages.pop()
            if msg is not None and msg != last_msg:
                raise ValueError(
                    f"Attempted to pop NVTX range from stack with msg={msg}, "
                    f"but last range has msg={last_msg}"
                )

            # Pop NVTX range
            torch.cuda.nvtx.range_pop()

    def _get_module_hooks(self, is_forward):
        stage_tag = "Forward" if is_forward else "Backward"

        def _pre_hook(module: torch.nn.Module, input):
            name = f"{stage_tag}::{self.module_to_name[module]}"
            self.profiling_range_push(msg=name)

        def _post_hook(module: torch.nn.Module, input, output):
            name = f"{stage_tag}::{self.module_to_name[module]}"
            self.profiling_range_pop(msg=name)

        return _pre_hook, _post_hook

    def register_profiler_hooks(self, module_list: List[torch.nn.Module]):
        if module_list is None:
            module_list = self.module_list
        assert isinstance(module_list, List) and len(module_list) > 0

        self.model_config = get_model_config(module_list[0])

        # register_full_backward_hook will change the pytorch graph, in conflict with 'deallocate_pipeline_outputs',
        # which will cause an error "AssertionError: counter-productive to free a view of another tensor."
        self.deallocate_pipeline_outputs_original = self.model_config.deallocate_pipeline_outputs
        self.model_config.deallocate_pipeline_outputs = False

        fwd_pre_hook_func, fwd_post_hook_func = self._get_module_hooks(is_forward=True)
        bwd_pre_hook_func, bwd_post_hook_func = self._get_module_hooks(is_forward=False)

        vpp_size = len(module_list)
        for vpp_stage, model_chunk in enumerate(module_list):
            if isinstance(model_chunk, torch.nn.Module):
                for name, sub_module in model_chunk.named_modules():
                    while name.startswith("module."):
                        name = name.lstrip("module").lstrip(".")
                    # Add class name
                    if name == "":
                        name = sub_module._get_name()
                    else:
                        name = f"{name}::{sub_module._get_name()}"
                    if vpp_size > 1:
                        name = f"vpp{vpp_stage}.{name}"
                    self.module_to_name[sub_module] = name

                for module in model_chunk.modules():
                    self.profiler_hook_handlers.append(module.register_forward_pre_hook(fwd_pre_hook_func))
                    self.profiler_hook_handlers.append(module.register_forward_hook(fwd_post_hook_func))
                    self.profiler_hook_handlers.append(module.register_full_backward_pre_hook(bwd_pre_hook_func))
                    self.profiler_hook_handlers.append(module.register_full_backward_hook(bwd_post_hook_func))

    def clear_profiler_hooks(self):
        for h in self.profiler_hook_handlers:
            h.remove()
        self.profiler_hook_handlers.clear()

        self.record_funcs.clear()
        self.nvtx_range_messages.clear()

        if self.deallocate_pipeline_outputs_original is not None:
            self.model_config.deallocate_pipeline_outputs = self.deallocate_pipeline_outputs_original
            self.deallocate_pipeline_outputs_original = None
            self.model_config = None


# Whether record_function profiling is enabled
_record_func_enabled: bool = False
# Whether NVTX range profiling is enabled
_nvtx_enabled: bool = False

_profiling_hook: ProfilingHook = None


def configure_profiling(
    enable_record_function: bool = False,
    enable_nvtx: bool = False,
    module_list: List[torch.nn.Module] = None,
) -> None:
    """Configure torch profiler record_function or NVTX range to be enabled or disabled.
    """
    if enable_record_function and enable_nvtx:
        raise ValueError("Can't configure record_function and NVTX at the same time.")

    global _record_func_enabled, _nvtx_enabled, _profiling_hook
    _record_func_enabled = enable_record_function
    _nvtx_enabled = enable_nvtx

    if (_record_func_enabled or _nvtx_enabled) and _profiling_hook is None:
        _profiling_hook = ProfilingHook(module_list)
        if module_list is not None:
            _profiling_hook.register_profiler_hooks(module_list)


def clear_profiling_hooks():
    global _record_func_enabled, _nvtx_enabled, _profiling_hook
    _record_func_enabled = False
    _nvtx_enabled = False
    if _profiling_hook is not None:
        _profiling_hook.clear_profiler_hooks()
        _profiling_hook = None


def _get_func_path():
    """Get the path of a function. Assumes being called from profiling_range_push/pop.

    Returns:
        str: Module path and function name joined by a dot
    """
    # Get the caller's caller frame (go back 2 frames)
    frame = inspect.currentframe().f_back.f_back
    caller_func = inspect.getframeinfo(frame).function
    module = inspect.getmodule(frame)

    return f"{module.__name__}.{caller_func}"


def profiling_range_push(msg=None, suffix=None) -> None:
    """Push profiling range onto stack. If msg is not provided, use the calling function's path.

    Args:
        msg (str, optional): Message to associate with range
        suffix (str, optional): Suffix to append to the message
    """
    if not (_record_func_enabled or _nvtx_enabled):
        return

    if msg is None:
        msg = _get_func_path()
    if suffix is not None:
        msg = f"{msg}.{suffix}"

    global _profiling_hook

    if _profiling_hook is None:
        _profiling_hook = ProfilingHook()

    _profiling_hook.profiling_range_push(msg)


def profiling_range_pop(msg=None, suffix=None) -> None:
    """Pop profiling range from stack. If msg is not provided, use the calling function's path.

    Args:
        msg (str, optional): Message to associate with range
        suffix (str, optional): Suffix to append to the message
    """
    if not (_record_func_enabled or _nvtx_enabled):
        return

    if msg is None:
        msg = _get_func_path()
    if suffix is not None:
        msg = f"{msg}.{suffix}"

    global _profiling_hook

    assert _profiling_hook is not None, ValueError(
        "You must firstly call profiling_range_push before profiling_range_pop."
    )

    _profiling_hook.profiling_range_pop(msg)


@lru_cache(maxsize=None)
def _decorator_get_func_path(func):
    """Get the path of a function.

    Args:
        func (Callable): Function to get path for.

    Returns:
        str: Module path and function name joined by a dot
    """
    caller_func = func.__name__
    module = inspect.getmodule(func)

    return f"{module.__name__}.{caller_func}"


def profiling_decorator(msg: Optional[str] = None, insert_marker_op: bool = False):
    """Decorator to add torch profiler record function or NVTX range to a function.

    Args:
        msg (str, optional): Custom msg for record function or the NVTX range. If None, uses function path
        insert_marker_op (bool): Whether to insert marker op like torch.empty((1,)).cuda(), used to ensure that
            record_function can catch the range on special cuda stream like Backward pass in pytorch autograd, or NCCL communication.
            If record_function don't work on your function, try to set `insert_marker_op` to True.
            Attention, `insert_marker_op` will mv a tensor from CPU to GPU with a H2D op, causing CUDA synchronization.
            TODO: `insert_marker_op` will increase the number of synchronization, may decrease the speed.

    Returns:
        Callable: Decorated function with torch profiler or NVTX profiling if enabled

    Example:
        @profiling_decorator()
        def my_function():
            pass

        @profiling_decorator(msg="Custom Range", color="blue")
        def another_function():
            pass
    """

    def func_decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not (_record_func_enabled or _nvtx_enabled):
                return func(*args, **kwargs)

            func_msg = msg or _decorator_get_func_path(func)
            if _record_func_enabled:
                with torch.profiler.record_function(func_msg):
                    # TODO: use more reasonable method to catch range on 'forward_step' and 'backward_step'
                    # if insert_marker_op:
                    #     # Just for record_function
                    #     torch.empty((1,)).cuda()
                    res = func(*args, **kwargs)
                    # if insert_marker_op:
                    #     # Just for record_function
                    #     torch.empty((1,)).cuda()
                    return res
            elif _nvtx_enabled:
                with torch.cuda.nvtx.range(func_msg):
                    return func(*args, **kwargs)
        return wrapper

    return func_decorator
