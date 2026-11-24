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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/nemotron_omni/processing_super_vl.py
# Source license: Apache-2.0.

"""Bridge-local loading adapter for the native VL processor in Transformers 5.18.0 through 5.20.2.

Imported lazily only by the version-guarded Super-VL pretrained wrapper. Image,
video and text processing use the upstream implementation without monkey patches
or changes to the source checkpoint. Loading these exports on unpatched HF 5.18.0
through 5.20.2 still requires this Bridge adapter because HF insists on an audio extractor.
"""

import copy

from transformers import AutoTokenizer
from transformers.models.nemotron_h_omni.image_processing_nemotron_h_omni import (
    NemotronH_Omni_Reasoning_V3ImageProcessor,
)
from transformers.models.nemotron_h_omni.processing_nemotron_h_omni import (
    NemotronH_Omni_Reasoning_V3Processor,
)
from transformers.models.nemotron_h_omni.video_processing_nemotron_h_omni import (
    NemotronH_Omni_Reasoning_V3VideoProcessor,
)


class Nemotron35VLProcessor(NemotronH_Omni_Reasoning_V3Processor):
    """Reuse the native processor without requiring an unused audio component."""

    @classmethod
    def get_attributes(cls) -> list[str]:
        # HF 5.18-5.20 ProcessorMixin binds constructor arguments to these modality
        # attributes. The native constructor's trailing feature_extractor=None
        # is deliberately not bound or serialized for this audio-free model.
        return [name for name in super().get_attributes() if name != "feature_extractor"]

    @classmethod
    def _get_arguments_from_pretrained(
        cls, pretrained_model_name_or_path, processor_dict=None, **kwargs
    ):
        kwargs = {**kwargs, "trust_remote_code": False}
        image_processor = NemotronH_Omni_Reasoning_V3ImageProcessor.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        video_dict, video_kwargs = (
            NemotronH_Omni_Reasoning_V3VideoProcessor.get_video_processor_dict(
                pretrained_model_name_or_path, **kwargs
            )
        )
        video_dict = copy.deepcopy(video_dict)
        # Legacy files serialize this derived value, while the native video
        # processor exposes it as a read-only property. Preserve the source.
        video_dict.pop("_downsample_factor", None)
        video_processor = NemotronH_Omni_Reasoning_V3VideoProcessor.from_dict(
            video_dict, **video_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return [image_processor, video_processor, tokenizer]

    def _process_audio(self, audio, audio_kwargs):
        raise ValueError("Nemotron 3.5 Super VL has no audio encoder; audio input is unsupported.")

    def to_dict(self) -> dict:
        """Save native metadata, without introducing a remote-code dependency."""
        result = super().to_dict()
        result["processor_class"] = "NemotronH_Omni_Reasoning_V3Processor"
        result.pop("auto_map", None)
        return result
