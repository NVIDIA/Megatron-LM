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

from typing import Any, Dict, Optional

import torch

from megatron.core import tensor_parallel
from megatron.core.inference.config import MediaPromptSpec, MultimodalPromptConfig
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.model_inference_wrappers.multimodal.utils import (
    dynamic_media_embedding_counts,
    dynamic_media_replacement_counts,
)
from megatron.core.utils import get_attr_wrapped_model


class NemotronOmniInferenceWrapper(GPTInferenceWrapper):
    """Dynamic-inference adapter for canonical, expanded-sequence Nemotron Omni."""

    supports_text = True
    supports_image = True
    supports_video = True
    supports_audio = False

    def get_multimodal_prompt_config(self) -> MultimodalPromptConfig:
        """Return Nemotron Omni's compact visual-span contract."""
        return MultimodalPromptConfig(
            image_spec=MediaPromptSpec(model_token="<image>", prefix="<img>", suffix="</img>"),
            video_spec=MediaPromptSpec(model_token="<image>", prefix="<img>", suffix="</img>"),
        )

    def get_preexpanded_media_token_id(self, modality: str) -> int:
        """Return the model's internal sentinel for already-expanded visual prompts."""
        del modality
        model = get_attr_wrapped_model(self.model, "image_token_index", return_model_obj=True)
        return int(model.image_token_index)

    def run_one_forward_step(
        self, inference_input: Dict[str, Any], recv_buffer_seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """Run one TP-only forward step."""
        if getattr(self.config, "pipeline_model_parallel_size", 1) > 1:
            raise NotImplementedError(
                "NemotronOmniInferenceWrapper supports pipeline_model_parallel_size=1 only."
            )
        return super().run_one_forward_step(inference_input, recv_buffer_seq_len)

    def expand_image_tokens(
        self, tokens, num_tiles=None, imgs_sizes=None, num_frames=None, *, image_token_id=None
    ):
        """Expand compact image/video placeholders and build embedding masks."""
        if imgs_sizes is None:
            raise NotImplementedError(
                "Canonical Nemotron Omni inference supports dynamic-resolution images only."
            )
        if num_tiles is not None:
            raise ValueError("num_tiles must be omitted for dynamic-resolution Omni inference.")
        model = get_attr_wrapped_model(self.model, "image_token_index", return_model_obj=True)
        image_token_index = (
            model.image_token_index if image_token_id is None else int(image_token_id)
        )
        if not getattr(model, "dynamic_resolution", False):
            raise ValueError("NemotronOmniModel must have dynamic_resolution enabled.")

        frame_embedding_counts = dynamic_media_embedding_counts(
            imgs_sizes, patch_dim=model.patch_dim, pixel_shuffle=True
        )
        placeholder_count = sum(
            token == image_token_index for sample_tokens in tokens for token in sample_tokens
        )

        replacement_counts = dynamic_media_replacement_counts(
            frame_embedding_counts,
            num_frames=num_frames,
            temporal_patch_size=int(getattr(model.vision_model, "temporal_patch_dim", 1)),
        )
        media_kind = "video" if num_frames is not None else "image"
        if placeholder_count != len(replacement_counts):
            raise ValueError(
                f"Expected one compact placeholder per {media_kind}: "
                f"expected {len(replacement_counts)}, got {placeholder_count}."
            )

        expanded_tokens = []
        image_masks = []
        image_index = 0
        embedding_offset = 0
        for sample_tokens in tokens:
            expanded_sample = []
            mask_sample = []
            for token in sample_tokens:
                if token != image_token_index:
                    expanded_sample.append(token)
                    mask_sample.append(None)
                    continue

                replacement_count = int(replacement_counts[image_index])
                expanded_sample.extend([-1] * replacement_count)
                mask_sample.extend(range(embedding_offset, embedding_offset + replacement_count))
                image_index += 1
                embedding_offset += replacement_count

            expanded_tokens.append(expanded_sample)
            image_masks.append(mask_sample)

        return expanded_tokens, image_masks

    def _forward_vision_encoder(
        self,
        images: torch.Tensor,
        num_image_tiles: Optional[torch.Tensor] = None,
        imgs_sizes: Optional[torch.Tensor] = None,
        num_frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode and project dynamic-resolution images once per request."""
        if imgs_sizes is None:
            raise NotImplementedError("Canonical Nemotron Omni inference requires imgs_sizes.")
        if num_image_tiles is not None:
            raise ValueError("num_image_tiles is not used by canonical Nemotron Omni.")

        media_kind = "video" if num_frames is not None else "image"
        if num_frames is None:
            num_frames = torch.ones(
                imgs_sizes.shape[0], dtype=torch.int32, device=imgs_sizes.device
            )

        model = get_attr_wrapped_model(self.model, "image_token_index", return_model_obj=True)
        with torch.cuda.nvtx.range(f"megatron.multimodal.{media_kind}_encoder"):
            embeddings = model._encode_images(
                images, imgs_sizes, vision_packed_seq_params=None, num_frames=num_frames
            )
        return embeddings.unsqueeze(1)

    def _forward(self, inference_input: Dict[str, Any]) -> torch.Tensor:
        """Dispatch text-only/decode and image-prefill forwards."""
        if "image_token_mask" in inference_input:
            return self._forward_dynamic(inference_input)

        output = self.model(
            input_ids=inference_input["tokens"],
            position_ids=inference_input["position_ids"],
            attention_mask=inference_input["attention_mask"],
            inference_context=self.inference_context,
            runtime_gather_output=True,
        )
        return output[0] if isinstance(output, tuple) else output

    def _forward_dynamic(self, inference_input: Dict[str, Any]) -> torch.Tensor:
        """Splice precomputed image embeddings and run the nested HybridModel."""
        tokens = inference_input["tokens"]
        position_ids = inference_input["position_ids"]
        attention_mask = inference_input["attention_mask"]
        image_token_mask = inference_input["image_token_mask"]
        image_embeddings = inference_input.get("image_embeddings")
        model = get_attr_wrapped_model(self.model, "image_token_index", return_model_obj=True)

        # The mask covers compact-path padding and pre-expanded model sentinels.
        input_ids_text = tokens.masked_fill(image_token_mask >= 0, 0)
        decoder_input = model.language_model.embedding(
            input_ids=input_ids_text, position_ids=position_ids
        )
        combined_embeddings = decoder_input.transpose(0, 1).contiguous()

        # Inject vision embeddings into the decoder input.
        image_positions = image_token_mask >= 0
        if image_positions.any():
            if image_embeddings is None:
                raise ValueError("Image positions were provided without image embeddings.")
            flat_image_embeddings = image_embeddings.reshape(-1, image_embeddings.shape[-1]).to(
                dtype=combined_embeddings.dtype
            )
            image_indices = image_token_mask[image_positions].to(dtype=torch.long)
            max_index = int(image_indices.max().item())
            if max_index >= flat_image_embeddings.shape[0]:
                raise ValueError(
                    f"Image embedding index {max_index} exceeds "
                    f"{flat_image_embeddings.shape[0]} available embeddings."
                )
            combined_embeddings[image_positions] = flat_image_embeddings[image_indices]

        decoder_input = combined_embeddings.transpose(0, 1).contiguous()
        if model.sequence_parallel_lm:
            decoder_input = tensor_parallel.scatter_to_sequence_parallel_region(
                decoder_input, group=model.pg_collection.tp
            ).contiguous()

        return model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=None,
            inference_context=self.inference_context,
            runtime_gather_output=True,
            packed_seq_params=None,
        )
