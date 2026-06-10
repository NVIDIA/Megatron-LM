# Copyright (c) 2025 DeepSeek. Licensed under the MIT License.
# Ported verbatim from DeepSeek DualPipe: https://github.com/deepseek-ai/DualPipe

import queue
from typing import Callable, List

import torch
from torch.autograd import Variable


class WeightGradStore:
    """Stores delayed weight-gradient computations for zero-bubble backward passes."""

    enabled: bool = False
    cache: List[Callable] = []
    funcs_queue = queue.Queue()

    @classmethod
    def put(cls, func: Callable) -> None:
        """Add a weight-gradient function to the cache."""
        cls.cache.append(func)

    @classmethod
    def flush(cls) -> None:
        """Move the cached functions into the FIFO queue as one group."""
        cls.funcs_queue.put(cls.cache)
        cls.cache = []

    @classmethod
    def pop(cls) -> None:
        """Run the oldest group of weight-gradient functions."""
        assert not cls.funcs_queue.empty(), "Pop empty queue."
        funcs = cls.funcs_queue.get()
        for func in funcs:
            func()

    @classmethod
    def clear(cls) -> None:
        """Drop all cached and queued weight-gradient functions."""
        cls.cache = []
        cls.funcs_queue = queue.Queue()


def run_backward(tensors: List[torch.Tensor], grad_tensors: List[torch.Tensor]) -> None:
    """Run the autograd engine backward for ``tensors`` without accumulating into leaves twice."""
    kwargs = dict(
        keep_graph=False, create_graph=False, allow_unreachable=True, accumulate_grad=True
    )
    Variable._execution_engine.run_backward(tensors, grad_tensors, **kwargs)


def chunk_tensor(x, chunks, dim):
    """Split a tensor into ``chunks`` parts along ``dim``; ``None`` passes through."""
    if x is None:
        return [None for _ in range(chunks)]
    return x.tensor_split(chunks, dim=dim)


def cat_tensor(x, dim):
    """Concatenate a list/tuple of tensors along ``dim``; ``None`` passes through."""
    if isinstance(x, tuple) or isinstance(x, list):
        if len(x) == 1:
            return x[0]
        elif x[0] is None:
            assert all(y is None for y in x)
            return None
    return torch.cat(x, dim=dim)


def scatter(inputs, chunks, dim):
    """Split each input tensor into micro-batches along ``dim``."""
    assert isinstance(inputs, (torch.Tensor, tuple, list))
    if isinstance(inputs, torch.Tensor):
        inputs = (inputs,)
    assert all(x is None or isinstance(x, torch.Tensor) for x in inputs)
    inputs = [chunk_tensor(x, chunks, dim) for x in inputs]
    microbatches = [microbatch for microbatch in zip(*inputs)]
    if len(microbatches) == 0:
        microbatches = [() for _ in range(chunks)]
    return microbatches


def gather(micro_outputs, dim):
    """Concatenate per-micro-batch outputs back into full-batch tensors along ``dim``."""
    assert isinstance(micro_outputs[0], (torch.Tensor, tuple, list))
    if isinstance(micro_outputs[0], torch.Tensor):
        micro_outputs = [(x,) for x in micro_outputs]
    outputs = [x for x in zip(*micro_outputs)]
    outputs = tuple(cat_tensor(x, dim=dim) for x in outputs)
    return outputs
