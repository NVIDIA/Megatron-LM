# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import Tensor

from megatron.core.inference.inference_request import (
    PREFIX_EOS_TOKEN_ID_FIELD,
    PREFIX_EXPANDED_TOKEN_COUNT_FIELD,
    PREFIX_MEDIA_COUNT_FIELD,
    PREFIX_TEMPLATE_TOKEN_IDS_FIELD,
    DynamicVLMInferenceRequest,
    compute_media_cache_key,
)
from megatron.core.inference.sampling_params import SamplingParams

from .dynamic_engine_requests import _weight_scoped_salt

_MULTIMODAL_STITCHING_FIELDS = {
    PREFIX_EXPANDED_TOKEN_COUNT_FIELD: (
        "a nonnegative integer counting the exact, already-expanded previous-turn tokens "
        "at the start of the prompt"
    ),
    PREFIX_MEDIA_COUNT_FIELD: (
        "a nonnegative integer counting logical image/video items in the previous prefix"
    ),
}


def _take_expanded_prefix_stitching_metadata(offload_params):
    """Remove and return engine-private expanded-prefix stitching metadata."""
    if not isinstance(offload_params, dict):
        return None, offload_params

    required_fields = set(_MULTIMODAL_STITCHING_FIELDS)
    if not required_fields.intersection(offload_params):
        return None, offload_params

    missing_fields = required_fields.difference(offload_params)
    if missing_fields:
        missing = "; ".join(
            f"{field!r}: {_MULTIMODAL_STITCHING_FIELDS[field]}" for field in sorted(missing_fields)
        )
        raise ValueError(
            "Incomplete multimodal expanded-prefix stitching metadata. "
            f"Missing {missing}. A RequestPromptPreparer must be configured and return the "
            "stitched prompt with its expanded prefix token count."
        )

    for field in required_fields:
        value = offload_params[field]
        if type(value) is not int or value < 0:
            raise ValueError(
                f"Invalid multimodal stitching field {field!r}: "
                f"expected {_MULTIMODAL_STITCHING_FIELDS[field]}."
            )

    metadata = {field: offload_params[field] for field in required_fields}
    consumed_fields = required_fields | {PREFIX_TEMPLATE_TOKEN_IDS_FIELD, PREFIX_EOS_TOKEN_ID_FIELD}
    remaining = {key: value for key, value in offload_params.items() if key not in consumed_fields}
    return metadata, remaining or None


def _prepend_expanded_multimodal_prefix(
    prefix_tokens, suffix_tokens, suffix_mask, inference_wrapper, modality
):
    """Prepend exact prior model tokens to an expanded current-turn suffix."""
    prefix_mask = inference_wrapper.build_preexpanded_media_token_mask(prefix_tokens, modality)

    # Offset suffix mask positions so we inject the suffix embeddings correctly.
    # For example: [ -1 0 1 2 -1 ] is the prefix mask, then the suffix mask
    # needs to at least start at 3, or else the first prefix embedding will be
    # used for the first suffix embedding.
    prefix_embedding_count = int((prefix_mask >= 0).sum().item())
    suffix_mask = suffix_mask.clone()
    suffix_media_positions = suffix_mask >= 0
    suffix_mask[suffix_media_positions] += prefix_embedding_count

    return torch.cat((prefix_tokens, suffix_tokens)), torch.cat((prefix_mask, suffix_mask))


def _slice_suffix_media_metadata(
    prefix_media_count, *, num_tiles, imgs_sizes, num_frames, video_frame_indices, video_fps
):
    """Select metadata for media placeholders occurring in the compact suffix."""
    if not isinstance(prefix_media_count, int) or prefix_media_count < 0:
        raise ValueError("Expanded-prefix stitching requires a nonnegative prefix media count.")

    # Retrieve the total media count and offset into the media data.
    if num_frames is not None:
        # Number of frame groups (such as videos or tubelets).
        total_media_count = len(num_frames)
        prefix_media_offset = int(num_frames[:prefix_media_count].sum().item())
    elif imgs_sizes is not None:
        # Number of frames.
        total_media_count = len(imgs_sizes)
        prefix_media_offset = prefix_media_count
    elif num_tiles is not None:
        # Number of tiles.
        total_media_count = len(num_tiles)
        prefix_media_offset = 0
    else:
        total_media_count = 0
        prefix_media_offset = 0

    if prefix_media_count > total_media_count:
        raise ValueError(
            "Expanded-prefix stitching prefix media count exceeds the request media count: "
            f"prefix={prefix_media_count}, total={total_media_count}."
        )

    return {
        "suffix_media_count": total_media_count - prefix_media_count,
        "num_tiles": (num_tiles[prefix_media_count:] if num_tiles is not None else None),
        "imgs_sizes": (imgs_sizes[prefix_media_offset:] if imgs_sizes is not None else None),
        "num_frames": (num_frames[prefix_media_count:] if num_frames is not None else None),
        "video_frame_indices": (
            video_frame_indices[prefix_media_count:] if video_frame_indices is not None else None
        ),
        "video_fps": (video_fps[prefix_media_count:] if video_fps is not None else None),
    }


@dataclass(frozen=True)
class _VisionCacheEntry:
    """Projected embedding and the preprocessed media needed to reuse it."""

    embedding: Tensor
    modality: str
    imgs: Tensor
    num_tiles: Optional[Tensor]
    num_img_embeddings_per_tile: int
    imgs_sizes: Optional[Tensor]
    num_frames: Optional[Tensor]
    video_frame_indices: Optional[List[List[int]]] = None
    video_fps: Optional[List[float]] = None


class MultimodalRequestMixin:
    """VLM request preparation and vision-embedding caching for `DynamicInferenceEngine`."""

    @staticmethod
    def _tensor_nbytes(tensor: Tensor) -> int:
        return tensor.untyped_storage().nbytes()

    def _vision_cache_entry_nbytes(self, entry: _VisionCacheEntry) -> int:
        total_bytes = 0
        seen_storages = set()
        for tensor in (
            entry.embedding,
            entry.imgs,
            entry.num_tiles,
            entry.imgs_sizes,
            entry.num_frames,
        ):
            if tensor is None or not tensor.is_cuda:
                continue
            storage = tensor.untyped_storage()
            storage_key = (tensor.device, storage.data_ptr())
            if storage_key in seen_storages:
                continue
            seen_storages.add(storage_key)
            total_bytes += self._tensor_nbytes(tensor)
        return total_bytes

    def clear_vision_embedding_cache(self) -> None:
        """Release all projected embeddings and reusable media retained by this engine."""
        self._vision_embedding_cache.clear()
        self._vision_embedding_cache_bytes = 0

    def _invalidate_vision_state(self) -> None:
        """Mark cached and request-local projected media as weight-stale."""
        if getattr(self, "allow_stale_multimodal_embeddings", False):
            return
        self.clear_vision_embedding_cache()
        for entry in self.requests.values():
            request = entry.record[-1]
            if isinstance(request, DynamicVLMInferenceRequest):
                request.image_embeddings = None

    def _refresh_vlm_request_data(self, request: DynamicVLMInferenceRequest) -> None:
        """Rebuild missing media state for a known-multimodal request."""
        if request.image_embeddings is not None and request.image_token_mask is not None:
            return
        if request.imgs is None:
            raise RuntimeError(
                f"Cannot refresh vision state for request {request.request_id}: raw media "
                "tensors were not retained."
            )

        # Move request tensors to local device for multimodal embedding recomputation.
        # Prompt token tensors are persistently stored on the inference device.
        device = request.prompt_tokens.device
        imgs = request.imgs.to(device=device)
        num_tiles = request.num_tiles.to(device=device) if request.num_tiles is not None else None
        imgs_sizes = (
            request.imgs_sizes.to(device=device) if request.imgs_sizes is not None else None
        )
        num_frames = (
            request.num_frames.to(device=device) if request.num_frames is not None else None
        )

        # Re-construct the input mask that provides multimodal embedding injection guidelines
        # for the inference wrapper decoder inputs.
        wrapper = self.controller.inference_wrapped_model
        modality = "video" if num_frames is not None else "image"
        if request.image_token_mask is not None:
            # If a request is checkpointed, retrieve the previous mask,
            # which can be composed of multiple turns.
            mask = request.image_token_mask.to(device=request.prompt_tokens.device)
            generated_suffix_length = len(request.prompt_tokens) - len(mask)
            if generated_suffix_length < 0:
                raise RuntimeError(
                    f"Preserved media mask for request {request.request_id} is longer than "
                    "its checkpointed prompt."
                )
            if generated_suffix_length > 0:
                # Mask out the previously generated tokens as invalid for media embeddings.
                # This will extend our checkpointed mask for future turns as well.
                mask = torch.cat(
                    (
                        mask,
                        torch.full(
                            (generated_suffix_length,), -1, dtype=mask.dtype, device=mask.device
                        ),
                    )
                )
        elif request.media_tokens_preexpanded:
            # Non-checkpointed pre-expanded input prompt. Build a new mask.
            mask = wrapper.build_preexpanded_media_token_mask(request.prompt_tokens, modality)
        else:
            raise RuntimeError(
                f"Cannot refresh vision state for request {request.request_id}: the media "
                "token mask was not retained."
            )

        encoder_kwargs = {"num_image_tiles": num_tiles, "imgs_sizes": imgs_sizes}
        if num_frames is not None:
            encoder_kwargs["num_frames"] = num_frames
        with torch.inference_mode():
            embeddings = wrapper._forward_vision_encoder(imgs, **encoder_kwargs)

        expected_count = int((mask >= 0).sum().item())
        actual_count = embeddings.numel() // embeddings.shape[-1] if embeddings.ndim > 0 else 0
        if actual_count != expected_count:
            raise ValueError(
                f"Refreshed request {request.request_id} has {expected_count} media-token "
                f"position(s), but the vision encoder produced {actual_count} embedding(s)."
            )

        request.image_embeddings = embeddings
        request.image_token_mask = mask
        self._cache_vision_embedding(
            request.media_cache_key,
            embeddings,
            modality=modality,
            imgs=imgs,
            num_tiles=num_tiles,
            num_img_embeddings_per_tile=request.num_img_embeddings_per_tile,
            imgs_sizes=imgs_sizes,
            num_frames=num_frames,
            video_frame_indices=request.video_frame_indices,
            video_fps=request.video_fps,
        )
        self.context.add_vlm_request_data(
            request.request_id, image_embeddings=embeddings, image_token_mask=mask
        )

    def _get_cached_vision_entry(
        self, cache_key: Optional[str], modality: Optional[str] = None
    ) -> Optional[_VisionCacheEntry]:
        """Return and promote a complete reusable vision-cache entry."""
        if (
            not cache_key
            or self.vision_embedding_cache_max_bytes == 0
            or cache_key not in self._vision_embedding_cache
        ):
            return None
        entry = self._vision_embedding_cache[cache_key]
        if modality is not None and entry.modality != modality:
            return None
        self._vision_embedding_cache.move_to_end(cache_key)
        return entry

    def _get_cached_vision_embedding(self, cache_key: Optional[str]) -> Optional[Tensor]:
        entry = self._get_cached_vision_entry(cache_key)
        return entry.embedding if entry is not None else None

    def _cache_vision_embedding(
        self,
        cache_key: Optional[str],
        embedding: Tensor,
        *,
        modality: str,
        imgs: Tensor,
        num_tiles: Optional[Tensor] = None,
        num_img_embeddings_per_tile: int = 0,
        imgs_sizes: Optional[Tensor] = None,
        num_frames: Optional[Tensor] = None,
        video_frame_indices: Optional[List[List[int]]] = None,
        video_fps: Optional[List[float]] = None,
    ) -> None:
        if not cache_key or self.vision_embedding_cache_max_bytes == 0:
            return
        entry = _VisionCacheEntry(
            embedding=embedding,
            modality=modality,
            imgs=imgs,
            num_tiles=num_tiles,
            num_img_embeddings_per_tile=num_img_embeddings_per_tile,
            imgs_sizes=imgs_sizes,
            num_frames=num_frames,
            video_frame_indices=video_frame_indices,
            video_fps=video_fps,
        )
        cache_entry_bytes = self._vision_cache_entry_nbytes(entry)
        if cache_entry_bytes > self.vision_embedding_cache_max_bytes:
            return
        previous = self._vision_embedding_cache.pop(cache_key, None)
        if previous is not None:
            self._vision_embedding_cache_bytes -= self._vision_cache_entry_nbytes(previous)
        while (
            self._vision_embedding_cache
            and self._vision_embedding_cache_bytes + cache_entry_bytes
            > self.vision_embedding_cache_max_bytes
        ):
            _, evicted = self._vision_embedding_cache.popitem(last=False)
            self._vision_embedding_cache_bytes -= self._vision_cache_entry_nbytes(evicted)
        self._vision_embedding_cache[cache_key] = entry
        self._vision_embedding_cache_bytes += cache_entry_bytes

    def _build_vlm_request(
        self,
        *,
        request_id: int,
        prompt_str: Optional[str],
        tokens: Tensor,
        sampling_params: Optional[SamplingParams],
        imgs: Optional[Tensor],
        num_tiles: Optional[Tensor],
        num_img_embeddings_per_tile: int,
        imgs_sizes: Optional[Tensor],
        precomputed_block_hashes: Optional[List[int]] = None,
        num_frames: Optional[Tensor] = None,
        video_frame_indices: Optional[List[List[int]]] = None,
        video_fps: Optional[List[float]] = None,
        media_tokens_preexpanded: bool = False,
        offload_params: Optional[Dict] = None,
        media_cache_key: Optional[str] = None,
    ) -> DynamicVLMInferenceRequest:
        """Prepare media tokens, run the vision encoder, register per-request
        media data on the context, and return a DynamicVLMInferenceRequest.
        """
        prefix_stitching_metadata, offload_params = _take_expanded_prefix_stitching_metadata(
            offload_params
        )
        cached_vision_entry = self._get_cached_vision_entry(media_cache_key)
        if cached_vision_entry is not None:
            modality = cached_vision_entry.modality
            imgs = cached_vision_entry.imgs
            num_tiles = cached_vision_entry.num_tiles
            num_img_embeddings_per_tile = cached_vision_entry.num_img_embeddings_per_tile
            imgs_sizes = cached_vision_entry.imgs_sizes
            num_frames = cached_vision_entry.num_frames
            video_frame_indices = cached_vision_entry.video_frame_indices
            video_fps = cached_vision_entry.video_fps
        elif num_frames is not None:
            modality = "video"
            missing = [
                name
                for name, value in (("imgs", imgs), ("imgs_sizes", imgs_sizes))
                if value is None
            ]
            if missing:
                raise ValueError(
                    "Video input requires imgs, imgs_sizes, and num_frames; " f"missing {missing}."
                )
        elif imgs_sizes is not None:
            modality = "image"
            if imgs is None:
                raise ValueError("Dynamic-resolution image input requires imgs and imgs_sizes.")
        else:
            modality = "image"
            if imgs is None or num_tiles is None or num_img_embeddings_per_tile <= 0:
                raise ValueError(
                    "Static-tiling image input requires imgs, num_tiles, and "
                    "num_img_embeddings_per_tile > 0."
                )

        # PP>1 needs a non-first-stage embedding recv path (the wrapper's
        # _recv_only_vision_embeds TODO). Until that lands, only PP=1 is
        # correct: non-first stages would see None embeddings but a non-None
        # mask and silently skip image splicing.
        pp_group = self.controller.pp_group
        if (
            pp_group is not None
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size(pp_group) > 1
        ):
            raise NotImplementedError(
                "Dynamic VLM inference does not support pipeline parallel. "
                "PP>1 requires the non-first-stage embedding recv path "
                "which is not yet available upstream."
            )

        # Multimodal request preparation.
        needs_media_identity = self.context.enable_prefix_caching or (
            self.vision_embedding_cache_max_bytes > 0
        )
        if media_cache_key is None and imgs is not None and needs_media_identity:
            # Compute multimodal media cache key, which is used by generators to
            # skip re-computing multimodal embeddings if the cache is hit.
            # Strongly recommend generating this hash upstream, such as via
            # the InferenceClient or providing this argument in add_request().
            media_inputs = {"imgs": imgs}
            for name, value in (
                ("num_tiles", num_tiles),
                ("imgs_sizes", imgs_sizes),
                ("num_frames", num_frames),
                ("video_frame_indices", video_frame_indices),
                ("video_fps", video_fps),
            ):
                if value is not None:
                    media_inputs[name] = value
            if num_img_embeddings_per_tile:
                media_inputs["num_img_embeddings_per_tile"] = num_img_embeddings_per_tile
            media_cache_key = compute_media_cache_key(modality, media_inputs)

        device = torch.cuda.current_device()
        num_tiles = num_tiles.to(device=device) if num_tiles is not None else None
        imgs_sizes = imgs_sizes.to(device=device) if imgs_sizes is not None else None
        num_frames = num_frames.to(device=device) if num_frames is not None else None

        # Dynamic-resolution requests derive their embedding count from
        # imgs_sizes downstream and don't need num_tiles.sum() at admission.
        # Static-tiling requests do; only pay the D2H sync on that path so
        # dynamic-res admissions stay sync-free here.
        has_images = imgs_sizes is not None and (
            imgs is not None or cached_vision_entry is not None
        )
        if not has_images:
            total_num_tiles = int(num_tiles.sum().item()) if num_tiles is not None else 0
            num_img_embeddings = num_img_embeddings_per_tile * total_num_tiles
            has_images = num_img_embeddings > 0
        if not has_images:
            raise ValueError("Multimodal input did not contain any processable media.")

        mask_tensor: Optional[Tensor] = None
        image_embeddings: Optional[Tensor] = None
        expected_embedding_count = 0

        if has_images:
            inference_wrapper = self.controller.inference_wrapped_model
            suffix_media_metadata = None
            if prefix_stitching_metadata is not None:
                # Retrieve suffix multimodal data and metadata.
                suffix_media_metadata = _slice_suffix_media_metadata(
                    prefix_stitching_metadata[PREFIX_MEDIA_COUNT_FIELD],
                    num_tiles=num_tiles,
                    imgs_sizes=imgs_sizes,
                    num_frames=num_frames,
                    video_frame_indices=video_frame_indices,
                    video_fps=video_fps,
                )

            # Compute input mask that provides multimodal embedding injection
            # guidelines for the inference wrapper decoder inputs.
            if media_tokens_preexpanded:
                mask_tensor = inference_wrapper.build_preexpanded_media_token_mask(tokens, modality)
                expected_embedding_count = int((mask_tensor >= 0).sum().item())
            else:
                media_token_id = inference_wrapper.resolve_media_token_id(
                    self.controller.tokenizer, modality
                )
                prefix_tokens = None
                if prefix_stitching_metadata is not None:
                    # Split the exact, already-expanded prefix from the compact suffix.
                    prefix_length = prefix_stitching_metadata[PREFIX_EXPANDED_TOKEN_COUNT_FIELD]
                    if prefix_length > len(tokens):
                        raise ValueError(
                            f"Expanded prefix token count {prefix_length} exceeds the prompt "
                            f"length {len(tokens)}."
                        )
                    prefix_tokens, tokens = tokens[:prefix_length], tokens[prefix_length:]
                expansion_num_tiles = num_tiles
                expansion_imgs_sizes = imgs_sizes
                expansion_num_frames = num_frames
                expansion_video_frame_indices = video_frame_indices
                expansion_video_fps = video_fps
                suffix_media_count = None
                if suffix_media_metadata is not None:
                    # Suffix / compact tokens require multi-modal expansion.
                    expansion_num_tiles = suffix_media_metadata["num_tiles"]
                    expansion_imgs_sizes = suffix_media_metadata["imgs_sizes"]
                    expansion_num_frames = suffix_media_metadata["num_frames"]
                    expansion_video_frame_indices = suffix_media_metadata["video_frame_indices"]
                    expansion_video_fps = suffix_media_metadata["video_fps"]
                    suffix_media_count = suffix_media_metadata["suffix_media_count"]

                if prefix_tokens is not None:
                    slice_placeholders = int((tokens == media_token_id).sum().item())
                    if slice_placeholders != suffix_media_count:
                        raise ValueError(
                            f"Expected {suffix_media_count} compact media placeholder(s) after "
                            f"the expanded prefix, found {slice_placeholders}."
                        )

                if suffix_media_count == 0:
                    # No multimodal data.
                    mask_tensor = torch.full_like(tokens, -1, dtype=torch.int64)
                else:
                    token_list: List[List[int]] = [tokens.tolist()]
                    expansion_kwargs = {
                        "num_tiles": expansion_num_tiles,
                        "imgs_sizes": expansion_imgs_sizes,
                    }
                    if expansion_num_frames is not None:
                        # Video and tubelet expansion kwargs.
                        expansion_kwargs["num_frames"] = expansion_num_frames
                        prompt_config = inference_wrapper.multimodal_prompt_config
                        if (
                            prompt_config is not None
                            and prompt_config.video_spec.expansion_mode == "temporal_patch"
                        ):
                            expansion_kwargs["tokenizer"] = self.controller.tokenizer
                            expansion_kwargs["video_frame_indices"] = expansion_video_frame_indices
                            expansion_kwargs["video_fps"] = expansion_video_fps
                    # Expand and prompt-format multimodal placeholder tokens.
                    expanded_tokens_list, mask_list = inference_wrapper.expand_image_tokens(
                        token_list, image_token_id=media_token_id, **expansion_kwargs
                    )
                    # Construct expanded token sequence with model-specific media tokens.
                    expanded_tokens = [
                        # Map -1 to <media>.
                        media_token_id if token < 0 else token
                        for token in expanded_tokens_list[0]
                    ]
                    tokens = torch.tensor(expanded_tokens, dtype=torch.int64, device=device)
                    # Multimodal mask: [ -1 = Text / 0, 1, 2, ... = Media Embed Indices ]
                    mask_tensor = torch.tensor(
                        [(-1 if v is None else int(v)) for v in mask_list[0]], device=device
                    )
                    expected_embedding_count = sum(value is not None for value in mask_list[0])

                if prefix_tokens is not None:
                    # Multi-Modal Prefix Stitching
                    tokens, mask_tensor = _prepend_expanded_multimodal_prefix(
                        prefix_tokens, tokens, mask_tensor, inference_wrapper, modality
                    )
                    expected_embedding_count = int((mask_tensor >= 0).sum().item())
                    media_tokens_preexpanded = True

            # Retrieve or compute the vision embedding.
            image_embeddings = self._get_cached_vision_embedding(media_cache_key)
            if image_embeddings is None and imgs is not None:
                imgs = imgs.to(device=device)
                # PP>1 is rejected above, so this is the only stage that owns
                # the vision encoder.
                with torch.inference_mode():
                    encoder_kwargs = {"num_image_tiles": num_tiles, "imgs_sizes": imgs_sizes}
                    if num_frames is not None:
                        encoder_kwargs["num_frames"] = num_frames
                    image_embeddings = (
                        self.controller.inference_wrapped_model._forward_vision_encoder(
                            imgs, **encoder_kwargs
                        )
                    )

            # Cache the image embedding.
            if image_embeddings is not None:
                actual_embedding_count = (
                    image_embeddings.numel() // image_embeddings.shape[-1]
                    if image_embeddings.ndim > 0
                    else 0
                )
                if actual_embedding_count != expected_embedding_count:
                    prompt_kind = "Pre-expanded" if media_tokens_preexpanded else "Expanded"
                    raise ValueError(
                        f"{prompt_kind} prompt has {expected_embedding_count} media-token "
                        f"position(s), but the vision encoder produced "
                        f"{actual_embedding_count} embedding(s)."
                    )
                self._cache_vision_embedding(
                    media_cache_key,
                    image_embeddings,
                    modality=modality,
                    imgs=imgs,
                    num_tiles=num_tiles,
                    num_img_embeddings_per_tile=num_img_embeddings_per_tile,
                    imgs_sizes=imgs_sizes,
                    num_frames=num_frames,
                    video_frame_indices=video_frame_indices,
                    video_fps=video_fps,
                )

        self.context.add_vlm_request_data(
            request_id, image_embeddings=image_embeddings, image_token_mask=mask_tensor
        )

        # Image-bearing requests can share KV only when their block-hash chain
        # is salted by the media identity computed from resolved media tensors.
        # Requests without enough media data to derive an identity remain
        # uncached rather than risk cross-media KV reuse for identical
        # placeholder token sequences.
        request_has_images = has_images
        enable_prefix_caching = self.context.enable_prefix_caching and (
            not request_has_images or bool(media_cache_key)
        )
        media_tensors = {
            name: tensor
            for name, tensor in (
                ("imgs", imgs),
                ("imgs_sizes", imgs_sizes),
                ("num_frames", num_frames),
                ("num_tiles", num_tiles),
            )
            if tensor is not None
        }
        return DynamicVLMInferenceRequest(
            request_id=request_id,
            prompt=prompt_str,
            prompt_tokens=tokens,
            media_tensors=media_tensors,
            sampling_params=sampling_params,
            offload_params=offload_params,
            block_size_tokens=self.context.block_size_tokens,
            enable_prefix_caching=enable_prefix_caching,
            # Recompute the block hashes for multimodal embeddings,
            # which are injected dynamically into the sequence.
            precomputed_block_hashes=[] if request_has_images else (precomputed_block_hashes or []),
            block_hash_salt=_weight_scoped_salt(
                self._weight_epoch, media_cache_key if request_has_images else None
            ),
            num_img_embeddings_per_tile=num_img_embeddings_per_tile,
            imgs=imgs,
            num_tiles=num_tiles,
            imgs_sizes=imgs_sizes,
            num_frames=num_frames,
            video_frame_indices=video_frame_indices,
            video_fps=video_fps,
            media_tokens_preexpanded=media_tokens_preexpanded,
            media_cache_key=media_cache_key,
            decoder_seq_length=0,
            image_embeddings=image_embeddings,
            image_token_mask=mask_tensor,
        )
