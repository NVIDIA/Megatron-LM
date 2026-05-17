# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Async high-level inference API for Megatron (``MegatronAsyncLLM``)."""

from typing import List, Optional, Union

from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.inference._llm_base import _MegatronLLMBase
from megatron.inference.serve_config import ServeConfig


class MegatronAsyncLLM(_MegatronLLMBase):
    """Async high-level inference API for Megatron.

    Asyncio-native wrapper over the shared engine + runtime managed by
    :class:`_MegatronLLMBase` -- see that class for execution modes
    (direct vs coordinator), caller responsibilities, and the
    ``model.eval()`` contract.

    On top of the base, this class provides:

    - ``async generate`` accepting single or batched prompts. In direct mode
      it is single-caller -- concurrent calls (e.g. via ``asyncio.gather``)
      raise :class:`RuntimeError`; pass a list of prompts to batch.
    - ``async`` lifecycle controls: ``pause`` / ``unpause`` / ``suspend`` /
      ``resume`` / ``shutdown`` / ``wait_for_shutdown``.
    - :meth:`serve` for OpenAI-compatible HTTP serving on the primary rank.
    - ``async with`` context-manager protocol; exit calls :meth:`shutdown`.
    """

    def __init__(
        self,
        *,
        model,
        tokenizer,
        inference_config: Optional[InferenceConfig] = None,
        use_coordinator: bool = False,
        coordinator_host: Optional[str] = None,
        coordinator_port: Optional[int] = None,
    ) -> None:
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            inference_config=inference_config,
            use_coordinator=use_coordinator,
            coordinator_host=coordinator_host,
            coordinator_port=coordinator_port,
        )
        # Concurrency guard for direct-mode generate (asyncio is single-threaded;
        # a plain bool is sufficient).
        self._direct_generate_in_flight: bool = False
        # Set in serve() when this rank starts the HTTP frontend; consulted by shutdown().
        self._serve_started: bool = False

    async def generate(
        self,
        prompts: Union[str, List[int], List[str], List[List[int]]],
        sampling_params: Optional[SamplingParams] = None,
    ) -> Union["DynamicInferenceRequest", List["DynamicInferenceRequest"]]:
        """Run inference for one prompt or a batch of prompts.

        Single input (``str`` or ``list[int]``) returns a single
        ``DynamicInferenceRequest``; batched input (``list[str]`` or
        ``list[list[int]]``) returns ``list[DynamicInferenceRequest]`` in
        input order.

        In direct mode, ``generate`` is single-caller -- concurrent calls raise
        ``RuntimeError``. Pass batched input instead of using
        ``asyncio.gather``.

        Raises:
            RuntimeError: if called on a non-primary rank in coordinator mode,
                or if a second concurrent call enters in direct mode.
        """
        self._assert_primary()
        if sampling_params is None:
            sampling_params = SamplingParams()

        normalized, is_batch = self._normalize_prompts(prompts)

        if not normalized:
            # Empty batch: nothing to schedule. ``is_batch`` is always True
            # here since single input is wrapped to a one-element list.
            return []

        if self._use_coordinator:
            assert self._loop_manager is not None
            results = await self._loop_manager.run_async(
                self._generate_impl(normalized, sampling_params)
            )
        else:
            if self._direct_generate_in_flight:
                raise RuntimeError(
                    "MegatronAsyncLLM.generate in direct mode is single-caller; "
                    "pass a list of prompts instead of using asyncio.gather."
                )
            self._direct_generate_in_flight = True
            try:
                results = await self._generate_impl(normalized, sampling_params)
            finally:
                self._direct_generate_in_flight = False

        return results if is_batch else results[0]

    async def pause(self) -> None:
        """Transition the engine to ``PAUSED``.

        Raises:
            RuntimeError: in direct mode (``use_coordinator=False``).
        """
        self._assert_coordinator()
        assert self._loop_manager is not None
        await self._loop_manager.run_async(self._pause_impl())

    async def unpause(self) -> None:
        """Transition the engine from ``PAUSED`` back to ``RUNNING``.

        Raises:
            RuntimeError: in direct mode (``use_coordinator=False``).
        """
        self._assert_coordinator()
        assert self._loop_manager is not None
        await self._loop_manager.run_async(self._unpause_impl())

    async def suspend(self) -> None:
        """Transition the engine to ``SUSPENDED`` (offloads GPU buffers).

        The caller must ``pause()`` first; this method does not enforce that.

        Raises:
            RuntimeError: in direct mode (``use_coordinator=False``).
        """
        self._assert_coordinator()
        assert self._loop_manager is not None
        await self._loop_manager.run_async(self._suspend_impl())

    async def resume(self) -> None:
        """Transition the engine from ``SUSPENDED`` to ``RESUMED``.

        Raises:
            RuntimeError: in direct mode (``use_coordinator=False``).
        """
        self._assert_coordinator()
        assert self._loop_manager is not None
        await self._loop_manager.run_async(self._resume_impl())

    async def shutdown(self) -> None:
        """Stop the engine, tear down the coordinator, and join the runtime thread.

        Idempotent. No-op in direct mode.
        """
        if self._shutdown_called:
            return
        self._shutdown_called = True

        # If we started an HTTP frontend, stop it first so no new requests
        # arrive while we tear down the coordinator. Invariant:
        # ``_serve_started`` can only be True when ``use_coordinator=True``
        # because ``serve()`` raises otherwise.
        if self._serve_started:
            from megatron.core.inference.text_generation_server.dynamic_text_gen_server.text_generation_server import (
                stop_text_gen_server,
            )

            stop_text_gen_server()
            self._serve_started = False

        if not self._use_coordinator:
            return
        assert self._loop_manager is not None
        await self._loop_manager.run_async(self._shutdown_impl())
        self._loop_manager.stop()

    async def serve(
        self,
        serve_config: ServeConfig,
        *,
        blocking: bool = True,
    ) -> None:
        """Start the OpenAI-compatible HTTP frontend.

        Coordinator mode only. The HTTP frontend runs only on the primary
        rank (global rank 0); other ranks no-op the HTTP setup but still
        respect ``blocking`` (so all ranks return together).

        With ``blocking=True`` (default), this awaits the engine loop until
        :meth:`shutdown` is called -- suitable for standalone serving scripts.
        With ``blocking=False``, this returns once the HTTP frontend is up
        (primary) or immediately (workers); the engine loop continues in the
        background runtime, and the user can call :meth:`generate` /
        :meth:`shutdown` afterward.

        Raises:
            ValueError: if ``use_coordinator=False`` (HTTP serving requires
                the coordinator path).
        """
        if not self._use_coordinator:
            raise ValueError(
                "MegatronAsyncLLM.serve() requires use_coordinator=True"
            )

        if self._is_primary_rank:
            # Lazy import: keep the module importable in environments where
            # the HTTP server backend (Quart/Hypercorn) isn't installed.
            import torch.distributed as dist

            from megatron.core.inference.text_generation_server.dynamic_text_gen_server.text_generation_server import (
                start_text_gen_server,
            )

            assert self._coord_runtime is not None
            start_text_gen_server(
                coordinator_addr=self._coord_runtime.coord_addr,
                tokenizer=self._controller.tokenizer,
                rank=dist.get_rank(),
                server_port=serve_config.port,
                parsers=serve_config.parsers,
                verbose=serve_config.verbose,
                num_replicas=serve_config.frontend_replicas,
                hostname=serve_config.host,
            )
            self._serve_started = True

        if blocking:
            # Block until the engine loop terminates (shutdown was invoked
            # somewhere in this process; for serve(blocking=True) typically by
            # SIGINT or out-of-band orchestration).
            await self.wait_for_shutdown()

    async def wait_for_shutdown(self) -> None:
        """Block until the engine's background loop task terminates.

        No-op in direct mode.
        """
        if not self._use_coordinator:
            return
        assert self._loop_manager is not None
        await self._loop_manager.run_async(self._wait_for_shutdown_impl())

    async def __aenter__(self) -> "MegatronAsyncLLM":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.shutdown()
