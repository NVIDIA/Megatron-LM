# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Sync high-level inference API for Megatron (``MegatronLLM``)."""

from typing import List, Optional, Type, Union

from megatron.core.inference.apis._llm_base import _MegatronLLMBase
from megatron.core.inference.apis.serve_config import ServeConfig
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.model_inference_wrappers.abstract_model_inference_wrapper import (
    AbstractModelInferenceWrapper,
)
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams


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
    - :meth:`serve` for OpenAI-compatible HTTP serving on the primary rank.
    - Context-manager protocol: ``with MegatronLLM(...) as llm:``; exit
      calls :meth:`shutdown`.
    """

    def __init__(
        self,
        *,
        model,
        tokenizer,
        inference_config: Optional[InferenceConfig] = None,
        use_coordinator: bool = True,
        coordinator_host: Optional[str] = None,
        coordinator_port: Optional[int] = None,
        inference_wrapper_cls: Optional[Type[AbstractModelInferenceWrapper]] = None,
    ) -> None:
        # Resolve the default at call time so tests can monkey-patch
        # ``GPTInferenceWrapper`` on this module. Binding it as the argument
        # default would freeze the reference at import time and bypass the
        # patch, which is why the previous ``= None`` version tripped
        # ``None(model, context)``.
        if inference_wrapper_cls is None:
            inference_wrapper_cls = GPTInferenceWrapper
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            inference_config=inference_config,
            use_coordinator=use_coordinator,
            coordinator_host=coordinator_host,
            coordinator_port=coordinator_port,
            inference_wrapper_cls=inference_wrapper_cls,
        )

    def generate(
        self,
        prompts: Union[str, List[int], List[str], List[List[int]]],
        sampling_params: Optional[SamplingParams] = None,
        multi_modal_data=None,
    ) -> List["DynamicInferenceRequest"]:
        """Run inference for one prompt or a batch.

        Returns ``list[DynamicInferenceRequest]`` in input order. Single-prompt
        input returns a one-element list -- the always-list shape is the
        deliberate sync-vs-async asymmetry.

        ``multi_modal_data`` follows vLLM's modality-dictionary shape. Batched
        prompts take one modality dictionary per prompt.

        Images:
            ``"image"`` accepts raw image bytes, a list of raw image bytes, or
            a preprocessed image tensor dictionary.
        Video:
            ``"video"`` accepts raw video bytes, a list of raw video bytes, or
            a preprocessed video tensor dictionary.
        Audio:
            Audio does not yet have any supported data preprocessing or
            modeling formats.

        No concurrency guard: sync is single-caller by Python's GIL. If you
        need to call ``generate`` concurrently from multiple threads, callers
        must serialize externally.

        Raises:
            RuntimeError: if called on a non-primary rank in coordinator mode.
        """
        self._assert_primary()
        if sampling_params is None:
            sampling_params = SamplingParams()

        normalized, is_batch = self._normalize_prompts(prompts)
        if not normalized:
            return []

        per_prompt_multi_modal_data = self._normalize_multi_modal_data_list(
            multi_modal_data, num_prompts=len(normalized), is_batch=is_batch
        )

        if self._use_coordinator:
            assert self._loop_manager is not None
            return self._loop_manager.run_sync(
                self._generate_impl(normalized, sampling_params, per_prompt_multi_modal_data)
            )
        if any(per_prompt_multi_modal_data):
            raise ValueError("multi_modal_data is only supported with use_coordinator=True.")
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
        self._stop_frontend_if_started()
        if not self._use_coordinator:
            return  # direct mode: nothing to tear down
        assert self._loop_manager is not None
        self._loop_manager.run_sync(self._shutdown_impl())
        # Sync caller already on its own thread; no need for to_thread.
        self._loop_manager.stop()

    def serve(self, serve_config: ServeConfig, *, blocking: bool = True) -> None:
        """Start the OpenAI-compatible HTTP frontend.

        Coordinator mode only. The HTTP frontend runs only on the primary
        rank (global rank 0); other ranks no-op the HTTP setup but still
        respect ``blocking`` (so all ranks return together).

        With ``blocking=True`` (default), this blocks the calling thread until
        the engine loop terminates via :meth:`shutdown` -- suitable for
        standalone serving scripts. With ``blocking=False``, this returns once
        the HTTP frontend is up (primary) or immediately (workers); the engine
        loop continues in the background runtime, and the user can call
        :meth:`generate` / :meth:`shutdown` afterward.

        Raises:
            ValueError: if ``use_coordinator=False`` (HTTP serving requires
                the coordinator path).
        """
        if not self._use_coordinator:
            raise ValueError("MegatronLLM.serve() requires use_coordinator=True")

        if self._is_primary_rank:
            # Lazy import: keep the module importable in environments where
            # the HTTP server backend (Quart/Hypercorn) isn't installed.
            import torch.distributed as dist

            from megatron.core.inference.text_generation_server.dynamic_text_gen_server.text_generation_server import (  # pylint: disable=line-too-long
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
                sock=serve_config.sock,
                multimodal_prompt_config=(
                    self._controller.inference_wrapped_model.multimodal_prompt_config
                ),
                default_temperature=serve_config.default_temperature,
                default_top_p=serve_config.default_top_p,
                default_top_k=serve_config.default_top_k,
                eval_mode=serve_config.eval_mode,
            )
            self._serve_started = True

        if blocking:
            # Block until the engine loop terminates (shutdown was invoked
            # somewhere in this process; for serve(blocking=True) typically by
            # SIGINT or out-of-band orchestration).
            self.wait_for_shutdown()

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
