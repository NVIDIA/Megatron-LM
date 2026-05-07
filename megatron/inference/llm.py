# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Sync high-level inference API for Megatron (``MegatronLLM``)."""

from typing import List, Optional, Union

from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.inference._llm_base import _MegatronLLMBase


class MegatronLLM(_MegatronLLMBase):
    """Sync high-level inference API for Megatron.

    See :class:`_MegatronLLMBase` for execution modes (direct vs
    coordinator), caller responsibilities, and the ``model.eval()`` contract.

    On top of the base, this class provides:

    - :meth:`generate` accepting one prompt or a batch; **always returns a
      ``list[DynamicInferenceRequest]``** (single-prompt input returns a
      one-element list -- deliberate asymmetry vs the async API).
    - Sync lifecycle controls: :meth:`pause` / :meth:`unpause` /
      :meth:`suspend` / :meth:`resume` / :meth:`shutdown` /
      :meth:`wait_for_shutdown`.
    - Context-manager protocol: ``with MegatronLLM(...) as llm:``; exit
      calls :meth:`shutdown`.

    Note:
        ``serve()`` (online HTTP serving) is async-only by design; use
        :class:`MegatronAsyncLLM` for serving.
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

    def generate(
        self,
        prompts: Union[str, List[int], List[str], List[List[int]]],
        sampling_params: Optional[SamplingParams] = None,
    ) -> List["DynamicInferenceRequest"]:
        """Run inference for one prompt or a batch.

        Returns ``list[DynamicInferenceRequest]`` in input order. Single-prompt
        input returns a one-element list -- the always-list shape is the
        deliberate sync-vs-async asymmetry.

        No concurrency guard: sync is single-caller by Python's GIL. If you
        need to call ``generate`` concurrently from multiple threads, callers
        must serialize externally.

        Raises:
            RuntimeError: if called on a non-primary rank in coordinator mode.
        """
        self._assert_primary()
        if sampling_params is None:
            sampling_params = SamplingParams()

        normalized, _is_batch = self._normalize_prompts(prompts)
        if not normalized:
            return []

        if self._use_coordinator:
            assert self._loop_manager is not None
            return self._loop_manager.run_sync(self._generate_impl(normalized, sampling_params))
        # Direct mode: bypass _generate_impl (which would use to_thread,
        # pointless for sync). Call the engine directly and merge.
        records = self._engine.generate(normalized, sampling_params)
        return [r.merge() for r in records]

    def pause(self) -> None:
        """Transition the engine to ``PAUSED``. Coordinator mode only.

        Raises:
            RuntimeError: in direct mode (``use_coordinator=False``).
        """
        self._assert_coordinator()
        assert self._loop_manager is not None
        self._loop_manager.run_sync(self._pause_impl())

    def unpause(self) -> None:
        """Transition the engine from ``PAUSED`` back to ``RUNNING``.

        Raises:
            RuntimeError: in direct mode (``use_coordinator=False``).
        """
        self._assert_coordinator()
        assert self._loop_manager is not None
        self._loop_manager.run_sync(self._unpause_impl())

    def suspend(self) -> None:
        """Transition the engine to ``SUSPENDED`` (offloads GPU buffers).

        The caller must ``pause()`` first; this method does not enforce that.

        Raises:
            RuntimeError: in direct mode (``use_coordinator=False``).
        """
        self._assert_coordinator()
        assert self._loop_manager is not None
        self._loop_manager.run_sync(self._suspend_impl())

    def resume(self) -> None:
        """Transition the engine from ``SUSPENDED`` to ``RESUMED``.

        Raises:
            RuntimeError: in direct mode (``use_coordinator=False``).
        """
        self._assert_coordinator()
        assert self._loop_manager is not None
        self._loop_manager.run_sync(self._resume_impl())

    def shutdown(self) -> None:
        """Tear down the engine and runtime. Idempotent. Direct mode is a no-op."""
        if self._shutdown_called:
            return
        self._shutdown_called = True
        if not self._use_coordinator:
            return  # direct mode: nothing to tear down
        assert self._loop_manager is not None
        self._loop_manager.run_sync(self._shutdown_impl())
        # Sync caller already on its own thread; no need for to_thread.
        self._loop_manager.stop()

    def wait_for_shutdown(self) -> None:
        """Block until the engine loop terminates. Direct mode no-op."""
        if not self._use_coordinator:
            return
        assert self._loop_manager is not None
        self._loop_manager.run_sync(self._wait_for_shutdown_impl())

    def __enter__(self) -> "MegatronLLM":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()
