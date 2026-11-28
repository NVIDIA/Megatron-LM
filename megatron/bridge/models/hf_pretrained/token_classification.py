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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/hf_pretrained/token_classification.py
# Source license: Apache-2.0.

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

from transformers import (
    AutoImageProcessor,
    AutoModelForTokenClassification,
    AutoProcessor,
    PreTrainedModel,
    ProcessorMixin,
)

from megatron.bridge.models.hf_pretrained.masked_lm import PreTrainedMaskedLM

if TYPE_CHECKING:
    from transformers.image_processing_utils import BaseImageProcessor


TokenClassificationType = TypeVar("TokenClassificationType", bound=PreTrainedModel)


class PreTrainedTokenClassification(
    PreTrainedMaskedLM[TokenClassificationType], Generic[TokenClassificationType]
):
    """Lazy Hugging Face token-classification model wrapper with VLM artifacts."""

    OPTIONAL_ARTIFACTS = ["processor", "image_processor"]

    def _load_model(self) -> TokenClassificationType:
        if self.model_name_or_path is None:
            raise ValueError("model_name_or_path must be provided to load model")

        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "config": self.config,
            **self.init_kwargs,
        }
        if self.torch_dtype is not None:
            model_kwargs["torch_dtype"] = self.torch_dtype

        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name_or_path, **model_kwargs
        )
        return model.to(self.device)

    def _load_processor(self) -> ProcessorMixin | None:
        """Load the optional multimodal processor."""
        if self.model_name_or_path is None:
            return None
        try:
            return AutoProcessor.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=self.trust_remote_code,
                **self.init_kwargs,
            )
        except Exception:
            return None

    def _load_image_processor(self) -> BaseImageProcessor | None:
        """Load the optional image processor, preferring the main processor."""
        processor = getattr(self, "_processor", None)
        if processor is not None and hasattr(processor, "image_processor"):
            return processor.image_processor
        if self.model_name_or_path is None:
            return None
        try:
            return AutoImageProcessor.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=self.trust_remote_code,
                **self.init_kwargs,
            )
        except Exception:
            return None

    @property
    def processor(self) -> ProcessorMixin | None:
        """Lazy-load and return the optional multimodal processor."""
        if not hasattr(self, "_processor"):
            self._processor = self._load_artifact("processor")
        return self._processor

    @processor.setter
    def processor(self, value: ProcessorMixin | None) -> None:
        """Set the multimodal processor manually."""
        self._processor = value

    @property
    def image_processor(self) -> BaseImageProcessor | None:
        """Lazy-load and return the optional image processor."""
        if not hasattr(self, "_image_processor"):
            self._image_processor = self._load_artifact("image_processor")
        return self._image_processor

    @image_processor.setter
    def image_processor(self, value: BaseImageProcessor | None) -> None:
        """Set the image processor manually."""
        self._image_processor = value
