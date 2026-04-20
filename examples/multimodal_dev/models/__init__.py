# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Model registry for multimodal_dev training.

Maps ``--model-arch`` to a set of factory functions that fully encapsulate
model-specific logic.  The training entry point (``pretrain_multimodal.py``)
remains model-agnostic — adding a new architecture only requires a new
registry entry (and its backing module) without touching the entry point.

Registry entry fields
---------------------
``model_factory_fn``  *(required)*
    ``(args, language_config, vision_config, **kwargs) -> MegatronModule``
    Builds and returns the complete model instance.

``vision_config_fn``  *(required)*
    ``(num_layers_override=None) -> TransformerConfig``
    Returns the vision encoder TransformerConfig.

``post_language_config_fn``  *(optional)*
    ``(language_config, args) -> None``
    Mutates the language TransformerConfig in-place with model-specific
    fields (e.g. ``mrope_section``).

``vision_flops_fn``  *(optional)*
    ``(args, language_config, vision_config) -> None``
    Sets vision FLOPs metadata on ``args`` for training throughput logging.

``dataset_providers``  *(optional)*
    ``Dict[str, str | callable]``
    Maps ``--dataset-provider`` names to callables (or dotted import paths
    resolved lazily) with signature
    ``(train_val_test_num_samples) -> (train_ds, val_ds, test_ds)``.
"""

from examples.multimodal_dev.models.qwen35_vl.configuration import (
    get_qwen35_vl_vision_config,
)
from examples.multimodal_dev.models.qwen35_vl.factory import (
    build_model as _build_qwen35_vl_model,
    post_language_config as _qwen35_vl_post_language_config,
    set_vision_flops_metadata as _qwen35_vl_vision_flops,
)
from examples.multimodal_dev.models.kimi_k25.factory import (
    build_model as _build_kimi_k25_model,
    get_kimi_k25_vision_config_stub as _kimi_k25_vision_config_stub,
    post_language_config as _kimi_k25_post_language_config,
)

MODEL_REGISTRY = {
    "qwen35_vl": {
        "model_factory_fn": _build_qwen35_vl_model,
        "vision_config_fn": get_qwen35_vl_vision_config,
        "post_language_config_fn": _qwen35_vl_post_language_config,
        "vision_flops_fn": _qwen35_vl_vision_flops,
        "dataset_providers": {
            "mock": (
                "examples.multimodal_dev.data.mock"
                ".train_valid_test_datasets_provider"
            ),
            "cord_v2": (
                "examples.multimodal_dev.data.vlm_dataset"
                ".train_valid_test_datasets_provider"
            ),
        },
    },
    "kimi_k25": {
        "model_factory_fn": _build_kimi_k25_model,
        "vision_config_fn": _kimi_k25_vision_config_stub,
        "post_language_config_fn": _kimi_k25_post_language_config,
        "dataset_providers": {
            "mock": (
                "examples.multimodal_dev.data.kimi_k25_vlm_mock"
                ".train_valid_test_datasets_provider"
            ),
        },
    },
}
