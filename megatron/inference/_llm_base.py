# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Internal building blocks for the Megatron inference high-level API.

This module hosts private helpers shared by ``MegatronLLM`` and
``MegatronAsyncLLM``: ``_EventLoopManager``, ``_CoordinatorRuntime``, and
``_MegatronLLMBase``. The public sync/async wrappers live on the subclasses;
this base only exposes shared engine state, runtime spawn, validation
helpers, and the private ``_<method>_impl`` coroutines.
"""

import asyncio
import concurrent.futures
import threading
import time
from typing import Coroutine, List, Optional, Tuple, Union

from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine, EngineState
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)


class _EventLoopManager:
    """Per-instance background daemon thread + persistent asyncio event loop.

    Bridges sync and async user-thread callers to coroutines that run on the
    background loop via ``asyncio.run_coroutine_threadsafe``.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started: bool = False
        self._stopped: bool = False

    def start(self) -> None:
        """Spawn the daemon thread and start the event loop. Idempotent."""
        if self._started:
            return

        def _run_loop() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            loop.run_forever()

        self._thread = threading.Thread(target=_run_loop, daemon=True)
        self._thread.start()

        # Wait for the loop to be created and running before returning so
        # callers can use ``submit`` immediately. Mirrors NeMo RL's polling
        # approach.
        while self._loop is None or not self._loop.is_running():
            time.sleep(0.001)

        self._started = True

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """The background asyncio loop. Raises if ``start()`` has not been called."""
        if not self._started or self._loop is None:
            raise RuntimeError("_EventLoopManager.start() must be called before accessing loop.")
        return self._loop

    def submit(self, coro: Coroutine) -> "concurrent.futures.Future":
        """Schedule ``coro`` on the background loop and return its future.

        The caller decides how to wait on the returned future (e.g.
        ``.result()`` for blocking sync, ``asyncio.wrap_future(...)`` for
        awaiting from another loop).
        """
        if not self._started or self._loop is None:
            raise RuntimeError("_EventLoopManager.start() must be called before submit().")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def run_sync(self, coro: Coroutine):
        """Schedule ``coro`` on the background loop and block on its result."""
        return self.submit(coro).result()

    async def run_async(self, coro: Coroutine):
        """Schedule ``coro`` on the background loop and await it from any loop."""
        return await asyncio.wrap_future(self.submit(coro))

    def stop(self) -> None:
        """Stop the event loop and join the background thread. Idempotent."""
        if not self._started or self._stopped:
            return
        assert self._loop is not None
        assert self._thread is not None
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._stopped = True
        self._started = False


class _CoordinatorRuntime:
    """Owns the dynamic-inference coordinator and ``InferenceClient`` lifecycle.

    Async-native: :meth:`setup` and :meth:`teardown` are coroutines meant to
    run on a background loop owned by :class:`_EventLoopManager`. The primary
    rank additionally holds an :class:`InferenceClient` used by the high-level
    API to submit requests and send control signals.
    """

    def __init__(
        self,
        engine: "DynamicInferenceEngine",
        *,
        is_primary: bool,
        coordinator_host: Optional[str],
        coordinator_port: Optional[int],
    ) -> None:
        self._engine = engine
        self._is_primary = is_primary
        self._coordinator_host = coordinator_host
        self._coordinator_port = coordinator_port
        self._client: "InferenceClient | None" = None
        self._coord_addr: Optional[str] = None

    async def setup(self, *, loop: asyncio.AbstractEventLoop) -> None:
        """Bring the coordinator and (on primary) the ``InferenceClient`` up.

        Calls ``engine.start_listening_to_data_parallel_coordinator(loop=loop)``
        on every rank. Only host/port kwargs that the caller actually supplied
        are forwarded so the engine can auto-bind when both are ``None``.
        """
        kwargs = {"loop": loop}
        if self._coordinator_host is not None:
            kwargs["hostname"] = self._coordinator_host
        if self._coordinator_port is not None:
            kwargs["inference_coordinator_port"] = self._coordinator_port

        coord_addr = await self._engine.start_listening_to_data_parallel_coordinator(**kwargs)
        self._coord_addr = coord_addr

        if self._is_primary:
            # Lazy import: keep this module importable without pyzmq/msgpack
            # installed when the user only needs direct mode.
            from megatron.core.inference.inference_client import InferenceClient

            client = InferenceClient(coord_addr)
            client.start(loop=loop)
            self._client = client

    async def teardown(self) -> None:
        """Primary-only client shutdown.

        Worker ranks are no-ops here; their ``engine_loop_task`` is awaited by
        :meth:`_MegatronLLMBase._shutdown_impl` after the primary has issued
        the STOP signal.
        """
        if not self._is_primary:
            return
        assert self._client is not None
        self._client.shutdown_coordinator()
        self._client.stop()

    @property
    def client(self) -> "InferenceClient | None":
        """The :class:`InferenceClient` on the primary rank; ``None`` on workers."""
        return self._client

    @property
    def coord_addr(self) -> Optional[str]:
        """Address returned by ``start_listening_to_data_parallel_coordinator``."""
        return self._coord_addr


class _MegatronLLMBase:
    """Private base shared by ``MegatronLLM`` and ``MegatronAsyncLLM``.

    This base intentionally exposes no public ``generate`` / lifecycle
    methods -- those live on the subclasses, which call into the private
    ``_<method>_impl`` coroutines defined here. The base owns:

    - the engine pipeline (engine, context, controller),
    - the per-instance background runtime (``_loop_manager``,
      ``_coord_runtime``) when ``use_coordinator=True``,
    - validation helpers (``_assert_primary``, ``_assert_coordinator``) and
      the input shape helper (``_normalize_prompts``).

    Two execution modes are supported:

    - **Direct mode** (``use_coordinator=False``): every rank is treated as
      primary and ``generate`` runs the engine synchronously (offloaded to a
      thread when called from an event loop). Lifecycle methods are invalid
      and raise :class:`RuntimeError` via ``_assert_coordinator``.
    - **Coordinator mode** (``use_coordinator=True``): a background event loop
      hosts the engine pipeline and an :class:`InferenceClient` (on global
      rank 0). Only the primary rank may submit requests via ``generate``.

    ``model`` must be in eval mode before construction; this class does not
    modify the model state.
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
        if (coordinator_host is not None or coordinator_port is not None) and not use_coordinator:
            raise ValueError("coordinator_host/port require use_coordinator=True")

        if inference_config is None:
            inference_config = InferenceConfig()

        # Build the engine pipeline. Mirrors examples/inference/gpt/gpt_dynamic_inference.py.
        context = DynamicInferenceContext(model.config, inference_config)
        # TODO: extend for non-GPT models in a future iteration.
        wrapper = GPTInferenceWrapper(model, context)
        controller = TextGenerationController(inference_wrapped_model=wrapper, tokenizer=tokenizer)
        engine = DynamicInferenceEngine(controller=controller, context=context)

        if use_coordinator:
            # Lazy import so the module imports cleanly without torch installed.
            import torch.distributed as dist

            is_primary_rank = dist.get_rank() == 0
        else:
            is_primary_rank = True

        self._engine = engine
        self._context = context
        self._controller = controller
        self._use_coordinator = use_coordinator
        self._is_primary_rank = is_primary_rank
        self._loop_manager: "Optional[_EventLoopManager]" = None
        self._coord_runtime: "Optional[_CoordinatorRuntime]" = None
        self._shutdown_called: bool = False

        if use_coordinator:
            loop_manager = _EventLoopManager()
            loop_manager.start()
            try:
                coord_runtime = _CoordinatorRuntime(
                    engine,
                    is_primary=is_primary_rank,
                    coordinator_host=coordinator_host,
                    coordinator_port=coordinator_port,
                )
                loop_manager.run_sync(coord_runtime.setup(loop=loop_manager.loop))
            except BaseException:
                loop_manager.stop()
                raise
            self._loop_manager = loop_manager
            self._coord_runtime = coord_runtime

    # ---- properties ----

    @property
    def is_primary_rank(self) -> bool:
        """Whether ``generate`` may be called on this rank."""
        return self._is_primary_rank

    @property
    def engine(self) -> "DynamicInferenceEngine":
        """The underlying :class:`DynamicInferenceEngine`."""
        return self._engine

    @property
    def context(self) -> "DynamicInferenceContext":
        """The underlying :class:`DynamicInferenceContext`."""
        return self._context

    @property
    def controller(self) -> "TextGenerationController":
        """The underlying :class:`TextGenerationController`."""
        return self._controller

    # ---- internal helpers ----

    def _assert_primary(self) -> None:
        if not self._is_primary_rank:
            raise RuntimeError(
                "generate(...) is only valid on the primary rank in coordinator mode"
            )

    def _assert_coordinator(self) -> None:
        if not self._use_coordinator:
            raise RuntimeError("This method requires use_coordinator=True")

    def _normalize_prompts(
        self, prompts: Union[str, List[int], List[str], List[List[int]]]
    ) -> Tuple[Union[List[str], List[List[int]]], bool]:
        """Return ``(normalized_list, is_batch_input)``.

        - ``"abc"`` -> ``(["abc"], False)``
        - ``[1, 2, 3]`` -> ``([[1, 2, 3]], False)``  (single token-id prompt)
        - ``["abc", "def"]`` -> ``(["abc", "def"], True)``
        - ``[[1, 2], [3, 4]]`` -> ``([[1, 2], [3, 4]], True)``
        - ``[]`` -> ``([], True)``

        Only the first element is inspected to distinguish single vs batch;
        per-element type validation is left to the engine.
        """
        if isinstance(prompts, str):
            return [prompts], False
        if isinstance(prompts, list):
            if not prompts:
                return [], True
            first = prompts[0]
            if isinstance(first, int):
                return [prompts], False
            if isinstance(first, (str, list)):
                return prompts, True
            raise TypeError(
                f"Unsupported prompt element type: {type(first)}; "
                "expected str, list[int], list[str], or list[list[int]]."
            )
        raise TypeError(
            f"prompts must be str, list[int], list[str], or list[list[int]]; "
            f"got {type(prompts)}"
        )

    # ---- private impl coroutines ----
    # Subclasses' public methods bridge to these via ``_EventLoopManager``
    # (coordinator mode, on the runtime loop) or await them directly
    # (direct mode, on the caller's event loop).
    # We need this bridge in coordinator mode because the coordinator requires
    # a long running event loop, so we need to route the user's event
    # loop to our runtime loop

    async def _generate_impl(
        self,
        prompts: Union[List[str], List[List[int]]],
        sp: SamplingParams,
    ) -> List["DynamicInferenceRequest"]:
        """Run inference for a non-empty list of prompts; returns input-ordered list.

        - Coordinator mode: must run on the runtime loop (via
          ``_loop_manager.run_async``); enqueues requests through
          ``client.add_request`` and gathers all futures.
        - Direct mode: runs on the caller's event loop; offloads the synchronous
          ``engine.generate`` to a thread.
        """
        if self._use_coordinator:
            # ``add_request`` calls ``asyncio.get_running_loop().create_future()``
            # so it must be invoked from a coroutine on the runtime loop. This
            # coroutine runs on that same loop, so ``asyncio.gather`` over the
            # returned futures is safe.
            assert self._coord_runtime is not None and self._coord_runtime.client is not None
            futures = [self._coord_runtime.client.add_request(p, sp) for p in prompts]
            return list(await asyncio.gather(*futures))
        # Direct mode: ``engine.generate`` accepts ``list[str]`` or
        # ``list[list[int]]``; both flow through ``engine.add_request`` which
        # accepts ``Union[str, List[int], Tensor]`` despite the narrower declared
        # type on ``engine.generate`` itself. TODO: widen that signature upstream.
        records = await asyncio.to_thread(self._engine.generate, prompts, sp)
        return [r.merge() for r in records]

    async def _pause_impl(self) -> None:
        if self._is_primary_rank:
            assert self._coord_runtime is not None and self._coord_runtime.client is not None
            self._coord_runtime.client.pause_engines()
        await self._engine.wait_until(EngineState.PAUSED)

    async def _unpause_impl(self) -> None:
        if self._is_primary_rank:
            assert self._coord_runtime is not None and self._coord_runtime.client is not None
            self._coord_runtime.client.unpause_engines()
        await self._engine.wait_until(EngineState.RUNNING)

    async def _suspend_impl(self) -> None:
        if self._is_primary_rank:
            assert self._coord_runtime is not None and self._coord_runtime.client is not None
            self._coord_runtime.client.suspend_engines()
        await self._engine.wait_until(EngineState.SUSPENDED)

    async def _resume_impl(self) -> None:
        if self._is_primary_rank:
            assert self._coord_runtime is not None and self._coord_runtime.client is not None
            self._coord_runtime.client.resume_engines()
        await self._engine.wait_until(EngineState.RESUMED)

    async def _shutdown_impl(self) -> None:
        if self._is_primary_rank:
            assert self._coord_runtime is not None and self._coord_runtime.client is not None
            self._coord_runtime.client.stop_engines()
            await self._engine.wait_until(EngineState.STOPPED)
            await self._coord_runtime.teardown()
        else:
            await self._engine.engine_loop_task

    async def _wait_for_shutdown_impl(self) -> None:
        await self._engine.engine_loop_task

