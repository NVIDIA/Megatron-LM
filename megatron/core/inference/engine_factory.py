# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Construction helpers for dynamic inference engines."""

from typing import Any

from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.disaggregation.coordinator_setup import (
    configure_prebuilt_disagg_engine,
)
from megatron.core.inference.disaggregation.engine import DisaggDynamicInferenceEngine
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.model_inference_wrappers.abstract_model_inference_wrapper import (
    AbstractModelInferenceWrapper,
)
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)


def build_dynamic_inference_engine(
    *,
    model: Any,
    tokenizer: Any,
    inference_config: InferenceConfig | None = None,
    inference_wrapper_cls: type[AbstractModelInferenceWrapper] = GPTInferenceWrapper,
    engine_cls: type[DynamicInferenceEngine] | None = None,
) -> DynamicInferenceEngine:
    """Build and configure a dynamic inference engine.

    This is the shared construction path for the high-level inference API and
    framework integrations. When ``disaggregation_shards`` is configured, it
    selects the disaggregated engine, reserves any recurrent-state dummy slot,
    and applies the prefill/decode coordinator configuration.

    Args:
        model: Distributed Megatron model to serve.
        tokenizer: Tokenizer consumed by the text generation controller.
        inference_config: Runtime inference configuration. Defaults to
            :class:`InferenceConfig`.
        inference_wrapper_cls: Model inference wrapper implementation.
        engine_cls: Optional engine implementation override. A disaggregated
            config requires a subclass of :class:`DisaggDynamicInferenceEngine`.

    Returns:
        A fully configured dynamic inference engine.

    Raises:
        ValueError: If a disaggregated config disables prefix caching or is
            paired with an incompatible explicit engine class.
    """
    if inference_config is None:
        inference_config = InferenceConfig()

    disaggregated = inference_config.disaggregation_shards is not None
    if disaggregated and not inference_config.enable_prefix_caching:
        raise ValueError("disaggregated inference requires prefix caching")
    if engine_cls is None:
        engine_cls = DisaggDynamicInferenceEngine if disaggregated else DynamicInferenceEngine
    elif disaggregated and not issubclass(engine_cls, DisaggDynamicInferenceEngine):
        raise ValueError("disaggregation_shards requires a DisaggDynamicInferenceEngine subclass")

    inference_config.reserve_recurrent_state_dummy_slot = (
        engine_cls.requires_recurrent_state_dummy_slot
    )
    context = DynamicInferenceContext(model.config, inference_config)
    inference_wrapped_model = inference_wrapper_cls(model, context)
    controller = TextGenerationController(
        inference_wrapped_model=inference_wrapped_model, tokenizer=tokenizer
    )
    engine = engine_cls(controller=controller, context=context)
    if disaggregated:
        configure_prebuilt_disagg_engine(engine)
    return engine
