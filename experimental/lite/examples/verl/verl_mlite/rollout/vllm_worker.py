# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""vLLM worker integration for MLite DeepSeek-V4 weight refits."""

from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger(__name__)

_NATIVE_DSV4_RELOAD_KEY = "verl_mlite_native_dsv4_layerwise"
_PATCHED_ATTR = "_verl_mlite_native_dsv4_reload_patched"
_ACTIVE_ATTR = "_verl_mlite_native_dsv4_reload_active"
_CONFIG_ATTR = "_verl_mlite_native_dsv4_reload_config"


def _needs_native_dsv4_reload() -> bool:
    """Use the compatibility lifecycle only on Hopper (SM90)."""

    import torch

    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major == 9


def install_native_dsv4_reload(vllm_config: Any) -> None:
    """Route DeepSeek-V4 refits through vLLM's native layerwise lifecycle.

    VERL's receiver invokes prepare once before the bucket stream, load once
    per bucket, and process once after the stream.  The wrappers preserve that
    orchestration while replacing only the DeepSeek-V4 implementation; every
    other quantization scheme continues through VERL's generic staging path.
    """

    from verl.utils.vllm import vllm_quant_utils as quant_utils
    from verl.workers.rollout.vllm_rollout import utils as worker_utils

    if getattr(quant_utils, _PATCHED_ATTR, False):
        return

    original_prepare = quant_utils.prepare_quanted_weights_for_loading
    original_process = quant_utils.process_quanted_weights_after_loading
    original_load = quant_utils.load_quanted_weights

    def prepare_quanted_weights_for_loading(model):
        if not _needs_native_dsv4_reload() or not quant_utils.is_deepseek_v4_model(model):
            return original_prepare(model)

        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.reload import initialize_layerwise_reload
        from vllm.model_executor.model_loader.reload.meta import SKIP_TENSORS

        # Runtime/derived tensors are synchronized separately and must not be
        # treated as checkpoint parameters by the layerwise loader.
        SKIP_TENSORS.update({"tid2eid", "expert_bias", "e_score_correction_bias", "attn_sink"})
        with set_current_vllm_config(vllm_config):
            initialize_layerwise_reload(model)
        setattr(model, _ACTIVE_ATTR, True)
        setattr(model, _CONFIG_ATTR, vllm_config)
        return {_NATIVE_DSV4_RELOAD_KEY: True}

    def process_quanted_weights_after_loading(model, reload_state):
        if not (reload_state or {}).get(_NATIVE_DSV4_RELOAD_KEY):
            return original_process(model, reload_state)

        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.reload import finalize_layerwise_processing

        config = getattr(model, _CONFIG_ATTR)
        try:
            with set_current_vllm_config(config):
                finalize_layerwise_processing(model, config.model_config)
        finally:
            setattr(model, _ACTIVE_ATTR, False)
        quant_utils.refresh_rocm_attention_weight_caches(model)

    def load_quanted_weights(weights, model_runner, is_drafter=False):
        if is_drafter:
            drafter = getattr(model_runner, "drafter", None)
            model = getattr(drafter, "model", None)
        else:
            model = model_runner.model
        if model is not None and getattr(model, _ACTIVE_ATTR, False):
            # vLLM buffers loader arguments until the complete logical layer
            # arrives, while VERL reuses or releases each IPC bucket backing
            # store after the callback returns.
            weights = [(name, tensor.clone()) for name, tensor in weights]
        return original_load(weights, model_runner, is_drafter=is_drafter)

    quant_utils.prepare_quanted_weights_for_loading = prepare_quanted_weights_for_loading
    quant_utils.process_quanted_weights_after_loading = process_quanted_weights_after_loading
    quant_utils.load_quanted_weights = load_quanted_weights
    # ``vllm_rollout.utils`` imports this symbol at module scope, so update its
    # binding as well as the defining module.
    worker_utils.load_quanted_weights = load_quanted_weights
    setattr(quant_utils, _PATCHED_ATTR, True)
    logger.info("Installed MLite native DeepSeek-V4 layerwise refit")


from verl.workers.rollout.vllm_rollout.utils import (  # noqa: E402
    vLLMColocateWorkerExtension,
)


class MLiteVLLMColocateWorkerExtension(vLLMColocateWorkerExtension):
    """Install the MLite DS4 refit protocol inside each spawned vLLM worker."""

    def __new__(cls, **kwargs):
        vllm_config = kwargs.get("vllm_config")
        if vllm_config is None:
            raise ValueError("MLite vLLM worker extension requires vllm_config")
        install_native_dsv4_reload(vllm_config)
        return super().__new__(cls, **kwargs)


__all__ = ["MLiteVLLMColocateWorkerExtension", "install_native_dsv4_reload"]
