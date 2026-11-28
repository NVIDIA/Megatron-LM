# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/0f06205924d4e19b0723aa8c33e34a1959ca17fe/src/megatron/bridge/models/qwen_vl/qwen35_vl_builder.py
# Source license: Apache-2.0.
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Model builder for Qwen3.5 and Qwen3.6 vision-language models."""

from megatron.bridge.models.hybrid.hybrid_builder import BridgeHybridModelBuilder
from megatron.bridge.models.qwen.modeling_qwen35.model_config import prepare_qwen35_pipeline_config
from megatron.bridge.models.qwen_vl.modeling_qwen35.hybrid_model import Qwen35VLHybridModel
from megatron.bridge.models.qwen_vl.qwen35_vl_config import Qwen35VLModelConfig
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.process_groups_config import ProcessGroupCollection


class Qwen35VLModelBuilder(BridgeHybridModelBuilder):
    """Combine the Qwen3.5 HybridModel language decoder with the Qwen3-VL vision encoder.

    Text and VL configs both count Hybrid mixer and feed-forward positions.
    """

    model_class: type[Qwen35VLHybridModel] | None = None

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> Qwen35VLHybridModel:
        """Build one Qwen3.5-VL pipeline stage.

        Args:
            pg_collection: Process groups used for distributed construction.
            pre_process: Whether this stage owns the embeddings and the vision encoder.
            post_process: Whether this stage owns the language-model output layer.
            vp_stage: Optional virtual pipeline stage index.

        Returns:
            A Qwen3-VL model stage built around the Qwen3.5 HybridModel decoder.

        Raises:
            TypeError: If the builder was given another model config.
            ValueError: If ``vision_config`` is unset or ``transformer_layer_spec`` is set.
        """
        model_config = self._model_config
        if not isinstance(model_config, Qwen35VLModelConfig):
            raise TypeError(f"Expected Qwen35VLModelConfig, got {type(model_config).__name__}.")
        if model_config.vision_config is None:
            raise ValueError(
                "Qwen3.5-VL vision_config must be configured before model construction."
            )
        if model_config.transformer_layer_spec is not None:
            raise ValueError(
                "Qwen3.5-VL builds its HybridModel decoder from the Qwen3.5 hybrid stack spec; "
                "leave transformer_layer_spec unset."
            )

        model_config = prepare_qwen35_pipeline_config(model_config)

        vp_size = model_config.virtual_pipeline_model_parallel_size
        if pre_process is None:
            pre_process = is_vp_first_stage(
                vp_stage=vp_stage, vp_size=vp_size
            ) and is_pp_first_stage(pg_collection.pp)
        if post_process is None:
            post_process = is_vp_last_stage(
                vp_stage=vp_stage, vp_size=vp_size
            ) and is_pp_last_stage(pg_collection.pp)

        vision_config = model_config.vision_config
        vision_config.torch_dtype = model_config.params_dtype
        model_class = self.model_class or Qwen35VLHybridModel
        model = model_class(
            language_transformer_config=model_config.transformer,
            language_transformer_layer_spec=self._resolve_hybrid_stack_spec(),
            vision_transformer_config=vision_config,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            model_config=model_config,
            language_model_kwargs={"hybrid_layer_pattern": model_config.hybrid_layer_pattern},
        )
        if vp_stage is not None:
            model.vp_stage = vp_stage
        if model.vision_model is not None:
            # Conversion builds without Float16Module, so cast the initially FP32 vision encoder here.
            model.vision_model.to(dtype=model_config.params_dtype)

        if (
            model_config.freeze_language_model
            or model_config.freeze_vision_model
            or model_config.freeze_vision_projection
        ):
            model.freeze(
                freeze_language_model=model_config.freeze_language_model,
                freeze_vision_model=model_config.freeze_vision_model,
                freeze_vision_projection=model_config.freeze_vision_projection,
            )
        return model


__all__ = ["Qwen35VLModelBuilder"]
