# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Nemotron Omni implementation of `MultimodalEmbeddingProvider`.

Ties the three stages together for one request:

1. Host preprocessing -- dynamic tiling, frame resizing, mel extraction. CPU-bound, cacheable.
2. Prompt expansion -- placeholder spans materialized at exactly the token counts stage 3 will
   produce.
3. Device encoding -- one batched RADIO call for images and videos, one Parakeet call for audio.

Stage 2 sits between the other two because video token counts depend on the encoder: EVS
pruning is data-dependent, so the retained count per tubelet is only known after the tower and
projector have run. Video therefore encodes *before* expansion, while images and audio can have
their counts predicted from geometry alone.
"""

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from megatron.core.inference.multimodal.nemotron_omni.audio_processor import ParakeetAudioProcessor
from megatron.core.inference.multimodal.nemotron_omni.config import NemotronOmniConfig
from megatron.core.inference.multimodal.nemotron_omni.encoder_stack import (
    EncodedMedia,
    NemotronOmniEncoderStack,
)
from megatron.core.inference.multimodal.nemotron_omni.image_processor import (
    DynamicResolutionImageTiler,
    TiledImage,
    patchify,
)
from megatron.core.inference.multimodal.nemotron_omni.prompt import (
    NemotronOmniPromptExpander,
    video_frame_separators,
)
from megatron.core.inference.multimodal.nemotron_omni.video_processor import (
    NemotronOmniVideoProcessor,
    ProcessedVideo,
)
from megatron.core.inference.multimodal.types import (
    MultimodalData,
    MultimodalEmbeddingProvider,
    ProcessedMultimodalPrompt,
)


class NemotronOmniEmbeddingProvider(MultimodalEmbeddingProvider):
    """Expands and encodes Nemotron Omni prompts at request admission.

    Args:
        config (NemotronOmniConfig): Resolved model configuration.
        encoder_stack (NemotronOmniEncoderStack): Vision tower, projectors, audio tower.
        tokenizer: Tokenizer used for placeholder expansion.
        device (Optional[torch.device]): Device for the returned embeddings. Defaults to the
            vision tower's device.
    """

    def __init__(
        self,
        config: NemotronOmniConfig,
        encoder_stack: NemotronOmniEncoderStack,
        tokenizer: Any,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.encoder_stack = encoder_stack
        self.tokenizer = tokenizer
        self.expander = NemotronOmniPromptExpander(tokenizer, config)
        self._device = device

        vision = config.vision
        self.image_tiler = DynamicResolutionImageTiler(
            patch_size=vision.patch_size,
            min_num_patches=vision.min_num_patches,
            max_num_patches=vision.max_num_patches,
            norm_mean=vision.norm_mean,
            norm_std=vision.norm_std,
        )
        self.video_processor = NemotronOmniVideoProcessor(
            patch_size=vision.patch_size,
            target_num_patches=vision.video_target_num_patches,
            maintain_aspect_ratio=vision.video_maintain_aspect_ratio,
            norm_mean=vision.norm_mean,
            norm_std=vision.norm_std,
        )
        self.audio_processor = ParakeetAudioProcessor(config.sound)

    @property
    def hidden_size(self) -> int:
        """Language model hidden size the encoder rows are projected to."""
        return self.config.hidden_size

    @property
    def device(self) -> torch.device:
        """Device the encoder rows are returned on."""
        if self._device is not None:
            return self._device
        return next(self.encoder_stack.vision_model.parameters()).device

    def _image_token_budget(self, prompt: str) -> int:
        """Post-shuffle tokens available to share across this request's images.

        The reference implementation computes this from the engine's `max_model_len`, which
        makes image *resolution* depend on a serving flag. Pinning it to the processor's own
        sequence length keeps output reproducible across deployments; set
        `image_budget_sequence_length` to None to recover the flag-dependent behaviour.
        """
        budget_length = self.config.image_budget_sequence_length or self.config.max_sequence_length
        text_length = len(self.expander._encode(prompt))
        # The -4 reserves the span's own <img> / </img> markers.
        return max(budget_length - text_length - 4, self.config.vision.min_num_patches // 4)

    def _content_hash(self, multimodal_data: MultimodalData) -> int:
        """Stable digest over the decoded media.

        Not used for cache lookups yet -- multimodal requests bypass the prefix cache because
        placeholder ids are identical across different images. Recorded so that folding it into
        the block hash chain later is a scheduler change rather than a data-model change.
        """
        digest = hashlib.blake2b(digest_size=8)
        for group, items in (
            ("image", multimodal_data.images),
            ("video", multimodal_data.videos),
            ("audio", multimodal_data.audios),
        ):
            digest.update(group.encode())
            for item in items:
                array = item
                if isinstance(array, tuple):
                    array = array[0]
                if torch.is_tensor(array):
                    array = array.detach().cpu().numpy()
                try:
                    digest.update(memoryview(array).cast("B"))
                except (TypeError, ValueError):
                    digest.update(repr(array).encode())
        return int.from_bytes(digest.digest(), "big")

    def _prepare_visual_inputs(
        self, tiled: Sequence[TiledImage], videos: Sequence[ProcessedVideo]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Patchify images and video frames into one packed RADIO input.

        Images come first, then videos, matching the order `encode_visual` unpacks. Note this
        is *not* prompt order when the two are interleaved; `encode` re-associates by item
        index rather than by position.

        Return:
            (Tuple[torch.Tensor, torch.Tensor, List[int]]) Packed patches
            `[1, total_patches, 3 * patch**2]`, per-frame pixel sizes `[total_frames, 2]`, and
            frames per item.
        """
        patch_size = self.config.vision.patch_size
        frames: List[torch.Tensor] = []
        sizes: List[Tuple[int, int]] = []
        num_frames: List[int] = []

        for image in tiled:
            frames.append(image.pixel_values)
            sizes.append(image.size_hw)
            num_frames.append(1)

        for video in videos:
            for frame_idx in range(video.num_frames):
                frames.append(video.pixel_values[frame_idx])
                sizes.append(video.size_hw)
            num_frames.append(video.num_frames)

        pixel_values = patchify(frames, patch_size)
        imgs_sizes = torch.tensor(sizes, dtype=torch.long)
        return pixel_values, imgs_sizes, num_frames

    def encode(self, prompt: str, multimodal_data: MultimodalData) -> ProcessedMultimodalPrompt:
        """Expand placeholders and run the encoders for one request.

        Args:
            prompt (str): Prompt text containing `<image>` / `<video>` / `<so_embedding>`
                markers.
            multimodal_data (MultimodalData): Media for this request.

        Return:
            (ProcessedMultimodalPrompt) Expanded token ids plus the encoder rows they need.
        """
        images = list(multimodal_data.images)
        videos = list(multimodal_data.videos)
        audios = list(multimodal_data.audios)

        # --- Stage 1: host preprocessing -------------------------------------------------
        tiled: List[TiledImage] = []
        if images:
            tiled = self.image_tiler.process(images, self._image_token_budget(prompt))

        processed_videos: List[ProcessedVideo] = []
        for video in videos:
            frames, fps, frame_indices = video, None, None
            if isinstance(video, dict):
                frames = video["frames"]
                fps = video.get("fps")
                frame_indices = video.get("frame_indices")
            processed_videos.append(
                self.video_processor.process(frames, fps=fps, frame_indices=frame_indices)
            )

        processed_audio = self.audio_processor.process(audios) if audios else None

        # --- Stage 2/3 interleaved: encode, then expand ----------------------------------
        # Videos have to be encoded before expansion: EVS retention is data-dependent, so the
        # per-tubelet token counts are only known once the tower and projector have run.
        visual_encoded: List[EncodedMedia] = []
        if tiled or processed_videos:
            pixel_values, imgs_sizes, num_frames = self._prepare_visual_inputs(
                tiled, processed_videos
            )
            visual_encoded = self.encoder_stack.encode_visual(pixel_values, imgs_sizes, num_frames)

        image_encoded = visual_encoded[: len(tiled)]
        video_encoded = visual_encoded[len(tiled) :]

        audio_encoded: List[EncodedMedia] = []
        if processed_audio is not None:
            audio_encoded = self.encoder_stack.encode_audio(
                processed_audio.mel_features,
                processed_audio.attention_mask,
                processed_audio.clips_per_item,
            )

        image_token_counts = [encoded.embeddings.shape[0] for encoded in image_encoded]
        for predicted, encoded in zip(tiled, image_encoded):
            assert predicted.num_tokens == encoded.embeddings.shape[0], (
                f"host predicted {predicted.num_tokens} tokens for a "
                f"{predicted.patch_grid} patch grid but the encoder produced "
                f"{encoded.embeddings.shape[0]} rows"
            )

        audio_token_counts = [encoded.embeddings.shape[0] for encoded in audio_encoded]
        if processed_audio is not None:
            assert audio_token_counts == processed_audio.token_counts, (
                f"host predicted {processed_audio.token_counts} audio tokens but the encoder "
                f"produced {audio_token_counts}"
            )

        video_plans: List[Dict[str, Any]] = []
        for processed, encoded in zip(processed_videos, video_encoded):
            separators = video_frame_separators(
                processed.frame_indices,
                processed.frame_duration_ms,
                self.config.vision.video_temporal_patch_size,
            )
            assert len(separators) == len(encoded.tokens_per_tubelet), (
                f"{len(separators)} tubelet separators for "
                f"{len(encoded.tokens_per_tubelet)} encoded tubelets"
            )
            video_plans.append(
                {"tokens_per_tubelet": encoded.tokens_per_tubelet, "separators": separators}
            )

        expanded = self.expander.expand(
            prompt,
            image_token_counts=image_token_counts,
            audio_token_counts=audio_token_counts,
            video_plans=video_plans,
        )

        # --- Assemble: concatenate rows in prompt order ---------------------------------
        by_modality = {"image": image_encoded, "video": video_encoded, "audio": audio_encoded}
        rows = [by_modality[span.modality][span.item_index].embeddings for span in expanded.spans]
        mm_embeddings = (
            torch.cat(rows, dim=0) if rows else torch.empty(0, self.hidden_size, device=self.device)
        )

        positions = expanded.embed_positions
        assert mm_embeddings.shape[0] == len(
            positions
        ), f"{mm_embeddings.shape[0]} encoder rows for {len(positions)} placeholder slots"

        return ProcessedMultimodalPrompt(
            prompt_tokens=torch.tensor(expanded.token_ids, dtype=torch.int64, device=self.device),
            mm_embeddings=mm_embeddings.contiguous(),
            mm_embed_positions=torch.tensor(positions, dtype=torch.int64, device="cpu"),
            content_hash=self._content_hash(multimodal_data),
        )
