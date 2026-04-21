# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright 2025 The vLLM authors.
#
# This code was adopted from https://github.com/vllm-project/vllm/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from typing import Any, AsyncGenerator, Callable, Optional, Type, Union

from megatron.core.inference.inference_request import InferenceRequest

STOP_ITERATION = Exception()


class AsyncStream:
    """
    Class for encapsulating an asynchronous stream of InferenceRequest outputs.

    Adopted from https://github.com/vllm-project/vllm/blob/eb881ed006ca458b052905e33f0d16dbb428063a/vllm/v1/engine/async_stream.py # pylint: disable=line-too-long
    """

    def __init__(self, request_id: str, cancel: Callable[[str], None]) -> None:
        self._request_id = request_id
        self._cancel = cancel
        self._queue: asyncio.Queue = asyncio.Queue()
        self._finished = False
        self._loop = asyncio.get_running_loop()

    def put(self, item: Union[InferenceRequest, Exception]) -> None:
        """Adds a new value to the stream"""
        if not self._finished:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    def finish(self, exception: Optional[Union[BaseException, Type[BaseException]]] = None) -> None:
        """Completes the stream by adding a sentinel value"""
        if not self._finished:
            self._finished = True
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait,
                exception if self._is_raisable(exception) else STOP_ITERATION,
            )

    @property
    def finished(self) -> bool:
        """Whether the stream has finished"""
        return self._finished

    async def generator(self) -> AsyncGenerator[InferenceRequest, None]:
        """Creates an AsyncGenerator over the stream queue"""
        try:
            while True:
                result = await self._queue.get()
                if self._is_raisable(result):
                    if result == STOP_ITERATION:
                        return
                    raise result
                yield result
        except GeneratorExit:
            self._cancel()
            raise asyncio.CancelledError from None

    @staticmethod
    def _is_raisable(value: Any):
        return isinstance(value, BaseException) or (
            isinstance(value, type) and issubclass(value, BaseException)
        )
