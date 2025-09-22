# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
import functools
import importlib
import sys
import time
import traceback
from typing import Callable, Coroutine, Type

from pydantic import BaseModel, ConfigDict


def import_class(class_path: str) -> Type:
    """Import a class from a string path.

    Args:
        class_path: String path to the class (e.g. 'examples.rl.environments.countdown.countdown_agent.CountdownAgent')

    Returns:
        The class object
    """
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class GenericGenerationArgs(BaseModel):
    """Generic generation arguments."""

    model_config = ConfigDict(frozen=True)
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    max_tokens: int | None = None

    def add(self, generation_args: 'GenericGenerationArgs') -> 'GenericGenerationArgs':
        return GenericGenerationArgs.model_validate(
            {**self.model_dump(), **generation_args.model_dump(exclude_unset=True)}
        )


class Request(BaseModel):
    """Generation Request."""

    generation_args: GenericGenerationArgs = GenericGenerationArgs()


from collections import defaultdict

_STATS = defaultdict(lambda: [0, 0.0])  # cnt, total_time


def trace_async_exceptions(fn: Callable[..., Coroutine]) -> Callable[..., Coroutine]:
    """Decorator to be applied to every coroutine that runs in a separate task.

    This is needed because asyncio tasks do not propagate exceptions.
    Coroutines running inside separate tasks will fail silently if not decorated.
    """
    if not asyncio.iscoroutinefunction(fn):
        raise TypeError("trace_async_exceptions can only be used with async functions")

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            print(f"Exception in async function {fn.__name__}: {e}")
            traceback.print_exc()
            sys.exit(1)
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            name = fn.__qualname__
            cnt, tot = _STATS[name]
            _STATS[name] = [cnt + 1, tot + elapsed]
            avg = _STATS[name][1] / _STATS[name][0]
            import numpy as np

            log10 = np.log10(max(cnt, 1))
            if np.isclose(log10, round(log10)):
                print(
                    f"{name} completed in {elapsed:.3f} ms, lifetime avg: {avg:.3f} ms, lifetime cnt: {cnt + 1}"
                )

    return wrapper
