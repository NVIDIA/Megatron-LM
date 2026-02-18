# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import warnings
from dataclasses import dataclass, field
from typing import Dict

from megatron.core.transformer.spec_utils import ModuleSpec


@dataclass
class MimoModelConfig:
    """Configuration for a multi-modal model.

    Args:
        language_model_spec (ModuleSpec):
            Specification for the language model
        modality_submodules_spec (Dict[str, ModuleSpec]):
            Dictionary mapping modality names to their submodule specifications
        special_token_ids (Dict[str, int]):
            Dictionary mapping modality names to their special token IDs.
            For example, {"vision": -200, "audio":32000}, these represent placeholders
            in the input_ids to insert the modality embeddings at the correct positions.
    """

    warnings.warn(
        "MimoModelConfig is experimental and still under active development. "
        "The API may change without notice in future releases.",
        category=UserWarning,
        stacklevel=2,
    )

    language_model_spec: ModuleSpec = field(default_factory=ModuleSpec)
    modality_submodules_spec: Dict[str, ModuleSpec] = field(default_factory=dict)
    special_token_ids: Dict[str, int] = field(default_factory=dict)
