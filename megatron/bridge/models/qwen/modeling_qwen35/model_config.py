# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/0f06205924d4e19b0723aa8c33e34a1959ca17fe/src/megatron/bridge/models/qwen/modeling_qwen35/model_config.py
# Source license: Apache-2.0.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3.5 text Hybrid construction with explicit pipeline stage boundaries."""

from copy import copy
from dataclasses import dataclass
from typing import ClassVar

from megatron.bridge.models.hybrid.hybrid_builder import BridgeHybridModelBuilder
from megatron.bridge.models.hybrid.model_config import BridgeHybridModelConfig
from megatron.bridge.models.qwen.modeling_qwen35.hybrid import qwen35_pipeline_pattern
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.process_groups_config import ProcessGroupCollection


@dataclass(kw_only=True)
class Qwen35HybridModelConfig(BridgeHybridModelConfig):
    """Language config whose layer counts and pipeline layouts use Hybrid positions."""

    builder: ClassVar[str] = (
        "megatron.bridge.models.qwen.modeling_qwen35.model_config.Qwen35HybridModelBuilder"
    )


def prepare_qwen35_pipeline_config(
    model_config: Qwen35HybridModelConfig,
) -> Qwen35HybridModelConfig:
    """Finalize and partition a copy for construction, preserving the caller's config."""
    config = copy(model_config)
    config.transformer = copy(config.transformer)
    config.finalize()
    main_pattern, separator, mtp_pattern = config.hybrid_layer_pattern.partition("/")
    if "|" not in main_pattern:
        main_pattern = qwen35_pipeline_pattern(config.transformer, main_pattern)
        config.hybrid_layer_pattern = main_pattern + separator + mtp_pattern
    if "|" in main_pattern:
        config.num_layers_in_first_pipeline_stage = None
        config.num_layers_in_last_pipeline_stage = None
        config.pipeline_model_parallel_layout = None
    return config


class Qwen35HybridModelBuilder(BridgeHybridModelBuilder):
    """Resolve PP/VPP boundaries after the caller has applied parallelism overrides."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> HybridModel:
        """Build a stage without modifying the caller's pattern or pipeline settings."""
        config = prepare_qwen35_pipeline_config(self._model_config)

        vp_size = config.virtual_pipeline_model_parallel_size
        if pre_process is None:
            pre_process = is_vp_first_stage(
                vp_stage=vp_stage, vp_size=vp_size
            ) and is_pp_first_stage(pg_collection.pp)
        if post_process is None:
            post_process = is_vp_last_stage(
                vp_stage=vp_stage, vp_size=vp_size
            ) and is_pp_last_stage(pg_collection.pp)

        return BridgeHybridModelBuilder(config).build_model(
            pg_collection, pre_process, post_process, vp_stage
        )
