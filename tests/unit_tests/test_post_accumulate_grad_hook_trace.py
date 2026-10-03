# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Regression tests for CUDA post-accumulate hooks missed by coverage.py.

The CPU autograd engine invokes the hook on the calling thread, so a tracer
installed there is visible. These tests clear that tracer before backward, which
is the state the CUDA engine actually enters the hook in. The CUDA test covers
the real engine path on a GPU.
"""

import sys

import pytest
import torch

from tests.unit_tests.post_accumulate_trace import (
    install_post_accumulate_trace,
    preserve_caller_trace,
)

install_post_accumulate_trace()


def _line_tracer(function_name: str, seen: list[int]):
    """Return a tracer that records line events from ``function_name``."""

    def tracer(frame, event, arg):
        del arg
        if event == "line" and frame.f_code.co_name == function_name:
            seen.append(frame.f_lineno)
        return tracer

    return tracer


def test_preserve_is_identity_without_a_tracer():
    previous = sys.gettrace()
    sys.settrace(None)
    try:

        def hook(parameter):
            del parameter
            return None

        assert preserve_caller_trace(hook) is hook
    finally:
        sys.settrace(previous)


def test_wrapper_restores_a_cleared_trace_and_propagates_errors():
    def tracer(frame, event, arg):
        del frame, event, arg
        return tracer

    def boom(parameter):
        del parameter
        raise RuntimeError("boom")

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        wrapped = preserve_caller_trace(boom)
    finally:
        sys.settrace(previous)

    sys.settrace(None)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            wrapped(torch.zeros(1))
        assert sys.gettrace() is None
    finally:
        sys.settrace(previous)


def test_wrapper_leaves_an_existing_tracer_installed():
    def outer(frame, event, arg):
        del frame, event, arg
        return outer

    def inner(frame, event, arg):
        del frame, event, arg
        return inner

    def hook(parameter):
        del parameter
        return None

    previous = sys.gettrace()
    sys.settrace(outer)
    try:
        wrapped = preserve_caller_trace(hook)
    finally:
        sys.settrace(previous)

    sys.settrace(inner)
    try:
        assert wrapped(torch.zeros(1)) is None
        assert sys.gettrace() is inner
    finally:
        sys.settrace(previous)


def test_cpu_backward_hook_is_traced_after_the_caller_trace_is_cleared():
    seen = []
    weight = torch.nn.Parameter(torch.ones(2, 2))

    def user_hook(parameter):
        if parameter.grad is None:
            raise RuntimeError("grad missing")
        return None

    previous = sys.gettrace()
    sys.settrace(_line_tracer("user_hook", seen))
    try:
        weight.register_post_accumulate_grad_hook(user_hook)
        sys.settrace(None)
        (weight @ torch.ones(2, 1)).sum().backward()
        assert seen
        assert sys.gettrace() is None
    finally:
        sys.settrace(previous)


def test_tensor_subclass_hook_registration_does_not_recurse():
    class TracedParameter(torch.nn.Parameter):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            del types
            if kwargs is None:
                kwargs = {}
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)

    seen = []
    weight = TracedParameter(torch.ones(2))

    def user_hook(parameter):
        if parameter.grad is None:
            raise RuntimeError("grad missing")
        return None

    previous = sys.gettrace()
    sys.settrace(_line_tracer("user_hook", seen))
    try:
        weight.register_post_accumulate_grad_hook(user_hook)
        sys.settrace(None)
        weight.sum().backward()
        assert seen
    finally:
        sys.settrace(previous)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA autograd hook coverage requires a GPU"
)
@pytest.mark.launch_on_gb200
def test_cuda_backward_hook_is_traced_while_the_caller_trace_stays_installed():
    seen = []
    weight = torch.nn.Parameter(torch.ones(2, 2, device="cuda"))

    def user_hook(parameter):
        if parameter.grad is None:
            raise RuntimeError("grad missing")
        return None

    previous = sys.gettrace()
    sys.settrace(_line_tracer("user_hook", seen))
    try:
        weight.register_post_accumulate_grad_hook(user_hook)
        (weight @ torch.ones(2, 1, device="cuda")).sum().backward()
        torch.cuda.synchronize()
        assert seen
    finally:
        sys.settrace(previous)
