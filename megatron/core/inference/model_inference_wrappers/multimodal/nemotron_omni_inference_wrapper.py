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

import math
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


def _render_nemotron_vl_video_prompt(
    prompt_spec: MediaPromptSpec,
    frame_indices: list[int],
    fps: float,
    temporal_patch_size: int,
) -> str:
    """Render the canonical Nemotron-VL text and compact media slot per tubelet."""
    if any(type(index) is not int or index < 0 for index in frame_indices):
        raise ValueError("Video frame indices must be non-negative integers.")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError("Video FPS must be a positive number.")

    frame_duration_ms = int(1000.0 / float(fps))
    tubelet_text = []
    for first_frame in range(0, len(frame_indices), temporal_patch_size):
        descriptions = []
        for offset in range(temporal_patch_size):
            frame_position = first_frame + offset
            if frame_position >= len(frame_indices):
                break
            frame_label = "Frame" if offset == 0 else "frame"
            timestamp = frame_indices[frame_position] * frame_duration_ms / 1000.0
            descriptions.append(
                f"{frame_label} {frame_position + 1} sampled at "
                f"{timestamp:.2f} seconds"
            )
        media_text = prompt_spec.prefix + prompt_spec.model_token + prompt_spec.suffix
        if prompt_spec.include_frame_timestamps_for_nemotron_vl:
            media_text = " and ".join(descriptions) + ": " + media_text
        tubelet_text.append(media_text)
    return "\n".join(tubelet_text)


def _replace_compact_video_slots(
    tokens: list[list[int]],
    *,
    image_token_id: int,
    compact_prefix_tokens: list[int],
    compact_suffix_tokens: list[int],
    per_video_tokens: list[list[int]],
) -> list[list[int]]:
    """Replace each compact video slot with its per-tubelet compact token sequence."""
    rewritten_tokens = []
    video_index = 0
    for sample_tokens in tokens:
        rewritten_sample = []
        token_position = 0
        while token_position < len(sample_tokens):
            token = sample_tokens[token_position]
            if token != image_token_id:
                rewritten_sample.append(token)
                token_position += 1
                continue

            prefix_length = len(compact_prefix_tokens)
            prefix_matches = not compact_prefix_tokens or (
                rewritten_sample[-prefix_length:] == compact_prefix_tokens
            )
            suffix_start = token_position + 1
            configured_suffix_end = suffix_start + len(compact_suffix_tokens)
            suffix_matches = (
                sample_tokens[suffix_start:configured_suffix_end]
                == compact_suffix_tokens
            )
            has_wrapper = (
                bool(compact_prefix_tokens) and prefix_matches
            ) or (bool(compact_suffix_tokens) and suffix_matches)
            if has_wrapper and not (prefix_matches and suffix_matches):
                raise ValueError(
                    "Compact video marker must use either both configured "
                    "wrapper boundaries or neither."
                )
            if has_wrapper:
                if compact_prefix_tokens:
                    del rewritten_sample[-prefix_length:]
                token_position = configured_suffix_end
            else:
                token_position += 1

            if video_index >= len(per_video_tokens):
                raise ValueError("Video prompt contains more compact slots than videos.")
            rewritten_sample.extend(per_video_tokens[video_index])
            video_index += 1
        rewritten_tokens.append(rewritten_sample)
    if video_index != len(per_video_tokens):
        raise ValueError("Video prompt contains fewer compact slots than videos.")
    return rewritten_tokens


class NemotronOmniInferenceWrapper(GPTInferenceWrapper):
    """Dynamic-inference adapter for canonical, expanded-sequence Nemotron Omni."""

    supports_text = True
    supports_image = True
    supports_video = True
    supports_audio = False

    multimodal_prompt_config = MultimodalPromptConfig(
        image_spec=MediaPromptSpec(
            model_token="<image>", prefix="<img>", suffix="</img>"
        ),
        video_spec=MediaPromptSpec(
            model_token="<image>",
            prefix="<img>",
            suffix="</img>",
        ),
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
        self,
        tokens,
        num_tiles=None,
        imgs_sizes=None,
        num_frames=None,
        *,
        image_token_id=None,
        tokenizer=None,
        video_frame_indices=None,
        video_fps=None,
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

        prompt_spec = self.multimodal_prompt_config.get_spec(
            "video" if num_frames is not None else "image"
        )
        temporal_expansion = (
            num_frames is not None and prompt_spec.expansion_mode == "temporal_patch"
        )
        replacement_counts = dynamic_media_replacement_counts(
            frame_embedding_counts,
            num_frames=num_frames,
            temporal_patch_size=int(getattr(model.vision_model, "temporal_patch_dim", 1)),
            aggregate_videos=not temporal_expansion,
        )
        media_kind = "video" if num_frames is not None else "image"
        if temporal_expansion:
            if hasattr(num_frames, "tolist"):
                frame_groups = num_frames.tolist()
                if not isinstance(frame_groups, list):
                    frame_groups = [frame_groups]
            else:
                frame_groups = [num_frames] if isinstance(num_frames, int) else list(num_frames)
            frame_groups = [int(value) for value in frame_groups]
            expected_placeholders = len(frame_groups)
        else:
            frame_groups = []
            expected_placeholders = len(replacement_counts)
        if placeholder_count != expected_placeholders:
            raise ValueError(
                f"Expected one compact placeholder per {media_kind}: "
                f"expected {expected_placeholders}, got {placeholder_count}."
            )

        if temporal_expansion:
            if tokenizer is None:
                raise ValueError("Temporal video prompt expansion requires a tokenizer.")
            if (
                video_frame_indices is None
                or video_fps is None
                or len(video_frame_indices) != len(frame_groups)
                or len(video_fps) != len(frame_groups)
            ):
                raise ValueError(
                    "Temporal video prompt expansion requires frame indices and FPS "
                    "for every compact video placeholder."
                )
            temporal_patch_size = int(
                getattr(model.vision_model, "temporal_patch_dim", 1)
            )
            compact_prefix_tokens = tokenizer.tokenize(prompt_spec.prefix)
            compact_suffix_tokens = tokenizer.tokenize(prompt_spec.suffix)

            per_video_tokens = []
            replacement_offset = 0
            for video_index, frame_count in enumerate(frame_groups):
                indices = video_frame_indices[video_index]
                fps = video_fps[video_index]
                if len(indices) != frame_count:
                    raise ValueError(
                        "Video frame-index metadata must match num_frames: "
                        f"{len(indices)} != {frame_count}."
                    )
                tubelet_count = (frame_count + temporal_patch_size - 1) // temporal_patch_size
                video_counts = replacement_counts[
                    replacement_offset : replacement_offset + tubelet_count
                ]
                replacement_offset += tubelet_count
                rendered_video = _render_nemotron_vl_video_prompt(
                    prompt_spec, indices, fps, temporal_patch_size
                )
                video_tokens = tokenizer.tokenize(rendered_video)
                if sum(token == image_token_index for token in video_tokens) != len(
                    video_counts
                ):
                    raise ValueError(
                        "Tokenizer did not preserve exactly one media token per "
                        "temporal video prompt group."
                    )
                per_video_tokens.append(list(video_tokens))
            if replacement_offset != len(replacement_counts):
                raise ValueError("Temporal replacement counts did not partition videos.")

            tokens = _replace_compact_video_slots(
                tokens,
                image_token_id=image_token_index,
                compact_prefix_tokens=compact_prefix_tokens,
                compact_suffix_tokens=compact_suffix_tokens,
                per_video_tokens=per_video_tokens,
            )
            expanded_placeholder_count = sum(
                token == image_token_index
                for sample_tokens in tokens
                for token in sample_tokens
            )
            if expanded_placeholder_count != len(replacement_counts):
                raise ValueError(
                    "Temporal video prompt expansion produced "
                    f"{expanded_placeholder_count} media markers for "
                    f"{len(replacement_counts)} tubelet replacement counts."
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
        # Dynamic-engine prefill supplies precomputed image embeddings, and
        # decode supplies no raw images. Both use the LM-only path.
        if "image_token_mask" in inference_input or "images" not in inference_input:
            return self._forward_dynamic(inference_input)

        # Legacy/manual path: let NemotronOmniModel encode raw images and merge
        # them with the compact placeholder tokens.
        output = self.model(
            images=inference_input["images"],
            input_ids=inference_input["tokens"],
            position_ids=inference_input["position_ids"],
            attention_mask=inference_input["attention_mask"],
            imgs_sizes=inference_input.get("imgs_sizes"),
            vision_packed_seq_params=inference_input.get("vision_packed_seq_params"),
            num_frames=inference_input.get("num_frames"),
            inference_context=self.inference_context,
            runtime_gather_output=True,
        )
        return output[0] if isinstance(output, tuple) else output

    def _forward_dynamic(self, inference_input: Dict[str, Any]) -> torch.Tensor:
        """Splice precomputed image embeddings and run the nested HybridModel."""
        tokens = inference_input["tokens"]
        position_ids = inference_input["position_ids"]
        attention_mask = inference_input["attention_mask"]
        image_token_mask = inference_input.get("image_token_mask")
        image_embeddings = inference_input.get("image_embeddings")
        model = get_attr_wrapped_model(self.model, "image_token_index", return_model_obj=True)

        # The mask covers compact-path padding and pre-expanded model sentinels.
        input_ids_text = (
            tokens if image_token_mask is None else tokens.masked_fill(image_token_mask >= 0, 0)
        )
        decoder_input = model.language_model.embedding(
            input_ids=input_ids_text, position_ids=position_ids
        )
        combined_embeddings = decoder_input.transpose(0, 1).contiguous()

        # Inject vision embeddings into the decoder input.
        image_positions = image_token_mask >= 0 if image_token_mask is not None else None
        if image_positions is not None and image_positions.any():
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
