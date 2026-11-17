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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/nemotron_omni/hf_pretrained.py
# Source license: Apache-2.0.

"""Native HF processor compatibility for the audio-free Super-VL checkpoint."""

import transformers
from packaging.version import Version
from transformers import AutoModelForImageTextToText, AutoProcessor, ProcessorMixin

from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM

# Bound the processor workaround to the releases whose loading behavior we tested.
# Future releases use their own loader and surface errors rather than silently
# dropping a required VLM artifact during conversion/export.
_PROCESSOR_ADAPTER_RELEASES = ("5.18.0", "5.19.0", "5.20.0", "5.20.1", "5.20.2")


class Nemotron35VLPretrained(PreTrainedCausalLM):
    """Keep native processor compatibility separate from legacy remote-code loading."""

    def _uses_remote_model(self) -> bool:
        """Check the checkpoint registration, not only permission to execute it."""
        return self.trust_remote_code and bool(
            (getattr(self.config, "auto_map", None) or {}).get("AutoModelForCausalLM")
        )

    def _get_auto_model_class(self) -> type:
        """Use native VL registration while retaining the remote CausalLM mapping."""
        if self._text_only:
            return super()._get_auto_model_class()
        # Permission to execute remote code does not imply that the checkpoint
        # uses it: native exports have no remote CausalLM registration.
        if self._uses_remote_model():
            return super()._get_auto_model_class()
        if Version(transformers.__version__) < Version("5.18.0"):
            raise ValueError(
                "Native Super-VL requires Transformers >=5.18.0; use trust_remote_code=True otherwise."
            )
        return AutoModelForImageTextToText

    def _load_processor(self) -> ProcessorMixin | None:
        if self.model_name_or_path is None or self._uses_remote_model():
            return super()._load_processor()

        version = Version(transformers.__version__)
        if version < Version("5.18.0"):
            raise ValueError(
                "Native Super-VL requires Transformers >=5.18.0; use trust_remote_code=True otherwise."
            )

        processor_cls = AutoProcessor
        if (
            version.base_version in _PROCESSOR_ADAPTER_RELEASES
            and getattr(self.config, "audio_config", None) is None
        ):
            from megatron.bridge.models.nemotron_super_vl.processing_super_vl import (
                Nemotron35VLProcessor,
            )

            processor_cls = Nemotron35VLProcessor
        return processor_cls.from_pretrained(
            self.model_name_or_path, trust_remote_code=False, **self.init_kwargs
        )
