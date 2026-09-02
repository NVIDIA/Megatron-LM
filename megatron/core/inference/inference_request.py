# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import hashlib
import time
import uuid
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np
import torch

from megatron.core.inference.config import ImageProcessingConfig, VideoProcessingConfig
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.tokenizers import MegatronTokenizer
from megatron.core.utils import experimental_api, nvtx_range_pop, nvtx_range_push


def serialize_tensor(tensor: torch.Tensor) -> List:
    """Serialize tensor to bytes.

    Args:
        tensor (Tensor): Tensor.

    Returns:
        (List) Tensor as a list
    """
    nvtx_range_push("serialize_tensor")

    # simply convert tensor into a list
    tensor = tensor.cpu().tolist()

    nvtx_range_pop("serialize_tensor")
    return tensor


def deserialize_tensor(tensor_as_list: List) -> torch.Tensor:
    """Deserialize tensor from bytes.

    Args:
        tensor_as_list (List): List representation of tensor.

    Returns:
        (Tensor) Tensor.
    """
    tensor = torch.tensor(tensor_as_list)
    return tensor


def _normalize_raw_media_items(modality_data: Any) -> Optional[List[bytes]]:
    """Normalize supported raw-media inputs, or return None for preprocessed data."""
    if isinstance(modality_data, (bytes, bytearray)):
        return [bytes(modality_data)]
    if isinstance(modality_data, list):
        if any(not isinstance(item, (bytes, bytearray)) for item in modality_data):
            raise TypeError("Raw media lists must contain only bytes or bytearray values.")
        return [bytes(item) for item in modality_data]
    return None


def _media_tensor_keys(modality: str) -> Tuple[str, ...]:
    """Return the tensor fields that define a preprocessed media input."""
    if modality == "video":
        return ("imgs", "imgs_sizes", "num_frames")
    if modality == "image":
        return ("imgs", "imgs_sizes", "num_tiles")
    raise ValueError(f"Unsupported media modality: {modality!r}.")


def compute_media_cache_key(modality: str, modality_data: Any) -> str:
    """Return a stable content key for raw or preprocessed media.

    The key is generated inside the inference stack so callers do not need to
    understand vision-embedding cache identity. Tensor metadata is included to
    prevent equal byte streams with different shapes or dtypes from colliding.
    """
    digest = hashlib.sha256()
    digest.update(b"megatron-media-v2\0")
    digest.update(modality.encode())
    digest.update(b"\0")

    raw_items = _normalize_raw_media_items(modality_data)
    if raw_items is not None:
        digest.update(b"raw\0")
        for item in raw_items:
            digest.update(len(item).to_bytes(8, "big"))
            digest.update(item)
        return digest.hexdigest()

    if isinstance(modality_data, dict):
        digest.update(b"preprocessed\0")
        cache_fields = set(_media_tensor_keys(modality)) | {"num_img_embeddings_per_tile"}
        for name in sorted(set(modality_data) & cache_fields):
            value = modality_data[name]
            digest.update(name.encode())
            digest.update(b"\0")
            if isinstance(value, torch.Tensor):
                tensor = value.detach().contiguous().cpu()
                digest.update(str(tensor.dtype).encode())
                digest.update(b"\0")
                digest.update(repr(tuple(tensor.shape)).encode())
                digest.update(b"\0")
                # Viewing a flattened tensor as uint8 works for dtypes such as
                # bfloat16 that NumPy cannot represent directly.
                digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
            elif name == "num_img_embeddings_per_tile":
                digest.update(str(int(value)).encode())
            else:
                raise TypeError(
                    f"Cannot compute a media cache key from field {name!r} "
                    f"of type {type(value).__name__}."
                )
            digest.update(b"\0")
        return digest.hexdigest()

    raise TypeError(f"Cannot compute a media cache key for {type(modality_data).__name__}.")


def serialize_multimodal_data(multi_modal_data: Any) -> Optional[Dict[str, Any]]:
    """Serialize one request's vLLM-style multimodal dictionary.

    Supported modalities:

    Images:
        ``"image"`` accepts raw image bytes, a list of raw image bytes, or a
        preprocessed tensor dictionary containing ``imgs`` / ``imgs_sizes``
        or ``imgs`` / ``num_tiles``.
    Video:
        ``"video"`` accepts raw video bytes, a list of raw video bytes, or a
        preprocessed tensor dictionary containing ``imgs``, ``imgs_sizes``,
        and ``num_frames``.
    Audio:
        Audio does not yet have any supported data preprocessing or modeling
        formats.
    """
    if multi_modal_data is None:
        return None
    if not isinstance(multi_modal_data, dict):
        raise TypeError(f"multi_modal_data must be a dict or None, got {type(multi_modal_data)}.")

    unsupported = set(multi_modal_data) - {"image", "video", "media_tokens_preexpanded"}
    if "media_cache_key" in unsupported:
        raise ValueError(
            "multi_modal_data['media_cache_key'] is internal and must not be "
            "provided; media identity is computed automatically."
        )
    if unsupported:
        raise NotImplementedError(
            f"Unsupported multimodal modalities: {sorted(unsupported)}; "
            "supported modalities are 'image' and 'video'."
        )
    if multi_modal_data.get("image") is not None and multi_modal_data.get("video") is not None:
        raise NotImplementedError(
            "Mixing image and video inputs in one inference request is not supported."
        )
    modality = "video" if multi_modal_data.get("video") is not None else "image"
    modality_data = multi_modal_data.get(modality)
    if modality_data is None:
        return None
    media_tokens_preexpanded = multi_modal_data.get("media_tokens_preexpanded", False)
    if not isinstance(media_tokens_preexpanded, bool):
        raise TypeError(
            "multi_modal_data['media_tokens_preexpanded'] must be a bool, "
            f"got {type(media_tokens_preexpanded)}."
        )
    metadata = {"media_tokens_preexpanded": True} if media_tokens_preexpanded else {}
    raw_items = _normalize_raw_media_items(modality_data)
    if raw_items is not None:
        media_cache_key = compute_media_cache_key(modality, raw_items)
        return {modality: raw_items, "media_cache_key": media_cache_key, **metadata}
    elif isinstance(modality_data, dict):
        media_cache_key = compute_media_cache_key(modality, modality_data)
        wire: Dict[str, Any] = {}
        for key in _media_tensor_keys(modality):
            value = modality_data.get(key)
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"multi_modal_data[{modality!r}][{key!r}] must be a Tensor, "
                    f"got {type(value)}."
                )
            wire[key] = serialize_tensor(value)
        if "num_img_embeddings_per_tile" in modality_data:
            wire["num_img_embeddings_per_tile"] = int(modality_data["num_img_embeddings_per_tile"])
        return {modality: wire, "media_cache_key": media_cache_key, **metadata} if wire else None
    else:
        raise TypeError(
            f"multi_modal_data[{modality!r}] must be bytes, list[bytes], or a "
            f"preprocessed tensor dict; got {type(modality_data)}."
        )


def split_multimodal_data(
    serialized: Optional[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Any]:
    """Split serialized media into a bounded descriptor and its payload.

    The descriptor rides in the metadata frame, which the coordinator decodes and
    repacks for every request; the payload rides in a body frame it forwards
    untouched. Keeping them apart is what bounds that per-request cost: the
    descriptor is a 64-character key plus two small flags, while the payload is
    raw image or video bytes, or serialized preprocessed tensors.

    Args:
        serialized: The output of :func:`serialize_multimodal_data`, or None.

    Returns:
        ``(media_meta, payload)``, both None for a text-only request.
    """
    if serialized is None:
        return None, None
    modality = "video" if "video" in serialized else "image"
    if modality not in serialized:
        raise ValueError(f"Serialized multimodal data has no media payload: {sorted(serialized)}.")
    media_meta = {key: value for key, value in serialized.items() if key != modality}
    media_meta["modality"] = modality
    return media_meta, serialized[modality]


def merge_multimodal_data(
    media_meta: Optional[Dict[str, Any]], payload: Any
) -> Optional[Dict[str, Any]]:
    """Rebuild serialized media from its descriptor and payload frames.

    Inverse of :func:`split_multimodal_data`, returning the shape
    :func:`resolve_multimodal_data_for_engine` consumes.

    Args:
        media_meta: The bounded descriptor from the metadata frame, or None.
        payload: The media payload from its own body frame.

    Returns:
        The serialized multimodal dictionary, or None for a text-only request.
    """
    if media_meta is None:
        return None
    media_meta = dict(media_meta)
    modality = media_meta.pop("modality", None)
    if modality not in ("image", "video"):
        raise ValueError(f"Media metadata carries an unsupported modality: {modality!r}.")
    return {modality: payload, **media_meta}


def resolve_multimodal_data_for_engine(
    multi_modal_data: Any,
    *,
    image_preprocessing_config: Optional[ImageProcessingConfig] = None,
    video_preprocessing_config: Optional[VideoProcessingConfig] = None,
) -> Dict[str, Any]:
    """Resolve wire-format multimodal data into dynamic-engine arguments.

    Supported modalities:

    Images:
        Raw image bytes are preprocessed into model inputs. Serialized or
        in-process preprocessed image tensor dictionaries are passed through
        as dynamic-engine image arguments.
    Video:
        Raw video bytes are decoded and sampled into model inputs. Serialized
        or in-process preprocessed tensor dictionaries are passed through.
    Audio:
        Audio does not yet have any supported data preprocessing or modeling
        formats.
    """
    if multi_modal_data is None:
        return {}
    if not isinstance(multi_modal_data, dict):
        raise TypeError(f"multi_modal_data must be a dict or None, got {type(multi_modal_data)}.")

    unsupported = set(multi_modal_data) - {
        "image",
        "video",
        "media_cache_key",
        "media_tokens_preexpanded",
    }
    if unsupported:
        raise NotImplementedError(
            f"Unsupported multimodal modalities: {sorted(unsupported)}; "
            "supported modalities are 'image' and 'video'."
        )
    if multi_modal_data.get("image") is not None and multi_modal_data.get("video") is not None:
        raise NotImplementedError(
            "Mixing image and video inputs in one inference request is not supported."
        )
    modality = "video" if multi_modal_data.get("video") is not None else "image"
    modality_data = multi_modal_data.get(modality)
    if modality_data is None:
        return {}
    media_tokens_preexpanded = multi_modal_data.get("media_tokens_preexpanded", False)
    if not isinstance(media_tokens_preexpanded, bool):
        raise TypeError(
            "multi_modal_data['media_tokens_preexpanded'] must be a bool, "
            f"got {type(media_tokens_preexpanded)}."
        )
    metadata = {"media_tokens_preexpanded": True} if media_tokens_preexpanded else {}
    if isinstance(modality_data, list):
        from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
            image_preprocessing,
        )

        if modality == "video":
            if video_preprocessing_config is None:
                raise RuntimeError(
                    "Raw video data require InferenceConfig.video_preprocessing_config."
                )
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available()
                else None
            )
            return {
                **image_preprocessing.preprocess_video_bytes_list(
                    modality_data, video_preprocessing_config, device=device
                ),
                **metadata,
            }

        if image_preprocessing_config is None:
            raise RuntimeError("Raw image data require InferenceConfig.image_preprocessing_config.")
        device = (
            torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else None
        )
        return {
            **image_preprocessing.preprocess_image_bytes_list(
                modality_data, image_preprocessing_config, device=device
            ),
            **metadata,
        }
    if not isinstance(modality_data, dict):
        raise TypeError(
            f"Wire multi_modal_data[{modality!r}] must be list[bytes] or a "
            f"serialized tensor dict; got {type(modality_data)}."
        )

    kwargs: Dict[str, Any] = {}
    tensor_keys = (
        ("imgs", "imgs_sizes", "num_frames")
        if modality == "video"
        else ("imgs", "imgs_sizes", "num_tiles")
    )
    for key in tensor_keys:
        if key in modality_data:
            value = modality_data[key]
            kwargs[key] = value if isinstance(value, torch.Tensor) else deserialize_tensor(value)
    if "num_img_embeddings_per_tile" in modality_data:
        kwargs["num_img_embeddings_per_tile"] = int(modality_data["num_img_embeddings_per_tile"])

    if modality == "image":
        # Reject incomplete static-tiling payloads. Static tiling (imgs +
        # num_tiles, no imgs_sizes) needs num_img_embeddings_per_tile to size
        # image-token expansion. Without it, the request would silently run as
        # text-only.
        has_num_tiles = "num_tiles" in kwargs
        has_imgs_sizes = "imgs_sizes" in kwargs
        has_per_tile = kwargs.get("num_img_embeddings_per_tile", 0) > 0
        if has_num_tiles and not has_imgs_sizes and not has_per_tile:
            raise ValueError(
                "Static-tiling image payload requires num_img_embeddings_per_tile > 0 "
                "when num_tiles is provided without imgs_sizes."
            )
    else:
        missing = {"imgs", "imgs_sizes", "num_frames"} - set(kwargs)
        if missing:
            raise ValueError(
                "Preprocessed video payload requires imgs, imgs_sizes, and num_frames; "
                f"missing {sorted(missing)}."
            )
    return {**kwargs, **metadata}


def serialize_ndarray(arr: np.ndarray) -> dict:
    """Serialize numpy array to a JSON-compatible dict."""
    return {"data": arr.tolist(), "dtype": str(arr.dtype)}


def deserialize_ndarray(obj: dict) -> np.ndarray:
    """Deserialize numpy array from dict."""
    return np.array(obj["data"], dtype=np.dtype(obj["dtype"]))


def unwrap_serialized_tensors(serialized_request: dict) -> dict:
    """Unwrap ("tensor", [...]) tuples produced by serialize() into plain lists.

    Args:
        serialized_request (dict): A dict produced by `serialize()`.

    Returns:
        dict: A shallow copy with tensor wrapper tuples replaced by their inner lists.
    """
    return {
        k: v[1] if isinstance(v, (list, tuple)) and len(v) == 2 and v[0] == "tensor" else v
        for k, v in serialized_request.items()
    }


# class syntax
class Status(Enum):
    """Enum for status"""

    WAITING_IN_QUEUE = 1
    ACTIVE_AND_GENERATING_TOKENS = 2
    ACTIVE_BUT_NOT_GENERATING_TOKENS = 3
    COMPLETED = 4
    FAILED = 5


# =========================================================================
# Hash computation for prefix caching
# =========================================================================


def compute_block_hashes_batched(
    prompt_tokens: torch.Tensor, block_size: int, cache_salt: Optional[str] = None
) -> List[int]:
    """Compute SHA-256 based hashes for all complete blocks in a prompt.

    Each block hash is computed as SHA-256(parent_digest || block_bytes), where
    parent_digest chains from the previous block (starting from a zero digest).
    This provides cryptographic collision resistance with no exploitable algebraic
    structure.

    Args:
        prompt_tokens: All prompt token IDs, shape [seq_len].
        block_size: Number of tokens per block.
        cache_salt: Optional request-input identity mixed into every chained
            block hash. Multimodal requests use their generated media key so
            equal token placeholders backed by different media cannot share KV.

    Returns:
        List of positive integer hash values in [1, 2^63-1], one per complete block.
    """
    num_complete_blocks = len(prompt_tokens) // block_size
    if num_complete_blocks == 0:
        return []

    # Single GPU->CPU transfer, get contiguous bytes
    tokens_cpu = prompt_tokens[: num_complete_blocks * block_size].to(torch.int64).cpu()
    tokens_bytes = tokens_cpu.numpy().tobytes()
    block_byte_size = block_size * tokens_cpu.element_size()  # 8 bytes per int64

    hashes = []
    if cache_salt is None:
        parent_digest = b'\x00' * 32  # Preserve text-only hash compatibility.
    else:
        parent_digest = hashlib.sha256(
            b"megatron-prefix-cache-salt-v1\0" + cache_salt.encode()
        ).digest()

    for i in range(num_complete_blocks):
        block_bytes = tokens_bytes[i * block_byte_size : (i + 1) * block_byte_size]
        digest = hashlib.sha256(parent_digest + block_bytes).digest()

        # Map to positive int64 range [1, 2^63-1], avoiding sentinels -1 and 0
        raw = int.from_bytes(digest[:8], byteorder='little', signed=False)
        hash_val = (raw % (2**63 - 1)) + 1

        hashes.append(hash_val)
        parent_digest = digest  # Full 32-byte digest chains into next block

    return hashes


@dataclass(kw_only=True)
class InferenceRequest:
    """Class for one inference request

    Containing relevant data for an inference request

    """

    request_id: int
    prompt: str
    sampling_params: Optional[SamplingParams] = None
    inference_parameters: Optional[SamplingParams] = None
    prompt_tokens: Optional[List[int]] = None
    # Prompt token count. Always populated when serializing a finished request so the
    # API can report usage.prompt_tokens even when the prompt_tokens tensor itself is
    # dropped from the payload (see SamplingParams.return_prompt_tokens).
    prompt_length: Optional[int] = None
    arrival_time: Optional[float] = None
    status: Optional[Status] = None
    encoder_prompt: Optional[str] = None
    generated_text: Optional[str] = None
    segments: Optional[List[str]] = None
    generated_segments: Optional[List[str]] = None
    generated_sequence_lengths: Optional[List[int]] = None
    generated_tokens: Optional[torch.Tensor] = None
    prompt_log_probs: Optional[torch.Tensor] = None
    generated_log_probs: Optional[torch.Tensor] = None
    prompt_top_n_logprobs: Optional[List[Dict[str, float]]] = None
    generated_top_n_logprobs: Optional[List[Dict[str, float]]] = None
    generated_length: Optional[int] = None
    tpot: List[float] = field(default_factory=list)

    def __post_init__(self):
        if self.sampling_params is None and self.inference_parameters is not None:
            warnings.warn(
                "`inference_parameters` renamed to `sampling_params`, and the "
                "previous name will be removed in Mcore 0.14."
            )
            self.sampling_params = self.inference_parameters

    def serialize(self) -> dict:
        """Converts the instance into a serializable dictionary.

        Returns:
            (dict) A dictionary representation of the instance suitable for
                serialization.
        """
        # Dataclass to dict.
        # do not use asdict(self) - it has very high CPU overheads
        # and if there are tensors, it will try to deepcopy them
        obj = self.__dict__.copy()  # shallow dict copy
        obj["status"] = self.status.name if self.status else None
        obj["sampling_params"] = self.sampling_params.serialize() if self.sampling_params else None
        obj["inference_parameters"] = (
            self.inference_parameters.serialize() if self.inference_parameters else None
        )

        # Serialize tensors and numpy arrays.
        obj = {
            k: (
                ("tensor", serialize_tensor(v))
                if isinstance(v, torch.Tensor)
                else ("ndarray", serialize_ndarray(v)) if isinstance(v, np.ndarray) else v
            )
            for k, v in obj.items()
        }
        return obj

    @classmethod
    def deserialize(cls, obj: dict) -> "InferenceRequest":
        """Deserialize request.

        Args:
            obj (dict): Serialized request data.

        Returns:
            (InferenceRequest) Deserialized request.
        """

        # Initialize request.
        request = cls(**obj)
        request._post_deserialize(obj)
        return request

    def _post_deserialize(self, obj: dict):
        """
        This is called after the dataclass is initialized to handle any special
        deserialization logic.
        """
        # Deserialize status.
        self.status = None if obj["status"] is None else Status[obj["status"]]
        self.sampling_params = (
            None
            if obj["sampling_params"] is None
            else SamplingParams.deserialize(obj["sampling_params"])
        )
        self.inference_parameters = (
            None
            if obj["inference_parameters"] is None
            else SamplingParams.deserialize(obj["inference_parameters"])
        )

        # Deserialize tensors, numpy arrays, and sampling params.
        for k, v in obj.items():
            if isinstance(v, list) and len(v) == 2 and v[0] == "tensor":
                setattr(self, k, deserialize_tensor(v[1]))
            elif isinstance(v, list) and len(v) == 2 and v[0] == "ndarray":
                setattr(self, k, deserialize_ndarray(v[1]))


class DynamicInferenceEventType(Enum):
    """Dynamic inference event type."""

    ADD_ENGINE = auto()  # When request is added to engine via _add_request()
    ADD_CONTEXT = auto()  # When request is added to context (scheduled for prefill)
    GENERATED_TOKEN = auto()  # When an output token is generated (payload = {"token_id": int})
    PAUSE = auto()
    EVICT = auto()
    FINISH = auto()
    FAIL = auto()
    ERROR_TRANSIENT = auto()
    ERROR_NONTRANSIENT = auto()


@dataclass(kw_only=True)
class DynamicInferenceEvent:
    """A lifecycle event for a dynamic inference requests.

    An event is currently one of the following:

    - request added
    - request paused
    - request evicted
    - request finished
    - request failed
    - request error (transient)
    - request error (non-transient, i.e. fatal)
    """

    timestamp: Optional[float] = None
    type: DynamicInferenceEventType
    payload: Optional[Any] = None

    def __post_init__(self):

        # Timestamp.
        if self.timestamp is None:
            self.timestamp = time.time()

        # Validate type.
        assert isinstance(self.type, DynamicInferenceEventType)

        # Validate payload.
        if self.type in (
            DynamicInferenceEventType.ERROR_TRANSIENT,
            DynamicInferenceEventType.ERROR_NONTRANSIENT,
        ):
            assert self.payload is not None
        elif self.type == DynamicInferenceEventType.GENERATED_TOKEN:
            assert (
                self.payload is not None
                and isinstance(self.payload, dict)
                and "token_id" in self.payload
            )
        else:
            assert self.payload is None

    def __str__(self):
        if self.type == DynamicInferenceEventType.GENERATED_TOKEN:
            payload_str = f", token={self.payload['token_id']}"
        elif self.payload is None:
            payload_str = ""
        else:
            payload_str = f", {type(self.payload).__name__}"
        return f"[{self.timestamp:.3f}] {self.type.name}{payload_str}"

    def serialize(self) -> dict:
        """Converts the instance into a serializable dictionary.

        Returns:
            dict: Full event dict.
        """
        nvtx_range_push("DynamicInferenceEvent.serialize")
        # do not use asdict(self) - it has very high CPU overheads
        # and if there are tensors, it will try to deepcopy them
        obj = self.__dict__.copy()
        obj["type"] = self.type.name

        # Serialize payload.
        if self.payload is not None:
            if self.type in (
                DynamicInferenceEventType.ERROR_TRANSIENT,
                DynamicInferenceEventType.ERROR_NONTRANSIENT,
            ):
                from .contexts.dynamic_context import ContextErrorFactory  # avoid circular import.

                obj["payload"] = ContextErrorFactory.serialize(self.payload)

        nvtx_range_pop("DynamicInferenceEvent.serialize")
        return obj

    @classmethod
    def deserialize(cls, obj: dict) -> "DynamicInferenceEvent":
        """Deserialize event.

        Args:
            obj: Serialized event data dict.

        Returns:
            (DynamicInferenceEvent) Deserialized event.
        """
        event_type = DynamicInferenceEventType[obj["type"]]

        # Pre-process payload before construction (since __post_init__ validates types).
        init_obj = {**obj, "type": event_type}
        if obj["payload"] is not None:
            if event_type in (
                DynamicInferenceEventType.ERROR_TRANSIENT,
                DynamicInferenceEventType.ERROR_NONTRANSIENT,
            ):
                from .contexts.dynamic_context import ContextErrorFactory  # avoid circular import.

                init_obj["payload"] = ContextErrorFactory.deserialize(obj["payload"])

        return cls(**init_obj)


@experimental_api
@dataclass(kw_only=True)
class DynamicInferenceRequest(InferenceRequest):
    """Class for one inference request

    Containing relevant data for an dynamic inference request

    """

    request_id: int
    # `request_id` is per-engine, do not cross DP, and do not persist.
    # A uuid is globally unique and can be used to track down individual requests.
    uid: str = field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    prompt: Optional[str] = None
    prompt_tokens: Optional[torch.Tensor] = None
    compact_prompt_tokens: Optional[torch.Tensor] = None
    # remaining prompt tokens are used for chunked prefill
    remaining_prompt_tokens: Optional[torch.Tensor] = None
    policy_epoch: Optional[list[tuple[int, int]]] = None
    kv_cache_epoch: Optional[list[tuple[int, int]]] = None
    latency: Optional[float] = None
    # routing_indices is reconstructed from per-block storage when a request finishes.
    routing_indices: Optional[np.ndarray] = None
    finished_chunk_token_count: int = 0
    stop_word_ids: Optional[List[List[int]]] = None  # Tokenized stop words (populated internally)
    # Consecutive steps this request has been deferred by CG-aware admission gating.
    # Reset to 0 on successful admission. Used only for starvation logging.
    cg_wait_iters: int = 0

    # Prefix caching fields
    block_size_tokens: Optional[int] = None  # Block size for hash computation
    enable_prefix_caching: bool = False  # Whether prefix caching is enabled
    # Prompt tokens whose prefill was skipped via prefix caching; accumulated across chunks.
    num_cached_tokens: int = 0
    # Length of the leading run of this request's blocks that was obtained by hash
    # match rather than computed. Accumulated across prefill chunks by the context,
    # which uses it to avoid rewriting KV into blocks that already hold it.
    num_matched_prefix_blocks: int = 0
    block_hash_salt: Optional[str] = None  # Media identity for multimodal KV safety.

    # Computed field - not passed by caller
    precomputed_block_hashes: List[int] = field(default_factory=list)

    # KV handoff metadata describing this request's pinned prefill state.
    # Used by decode-side pulls and prefill-side pushes.
    # Shape: {"request_id", "block_ids", "kv_meta"}.
    # Hybrid models may add kv_meta["ssm"] for recurrent state.
    disaggregated_params: Optional[dict] = None

    # Wire marker: serialize(payload_offloaded=True) writes this key after dropping the
    # per-token payload (log probs, MoE routing indices) from the wire; the engine sets it
    # only for requests whose payload it handed to its RequestPayloadStager.
    payload_offloaded: bool = False

    def __post_init__(self):
        self.sampling_params = copy.deepcopy(self.sampling_params)
        if self.prompt_tokens is not None:
            self.remaining_prompt_tokens = self.prompt_tokens

        # Compute block hashes for prefix matching (skip if already provided, e.g. from `merge`).
        if (
            self.enable_prefix_caching
            and self.block_size_tokens is not None
            and self.prompt_tokens is not None
            and not self.precomputed_block_hashes
        ):
            self._compute_block_hashes()

    def _compute_block_hashes(self) -> None:
        """Compute hashes for all complete blocks in the prompt.

        After this call:
        - precomputed_block_hashes is [] if prompt < block_size (no complete blocks)
        - precomputed_block_hashes is [hash1, ...] for N complete blocks
        """
        self.precomputed_block_hashes = compute_block_hashes_batched(
            self.prompt_tokens, self.block_size_tokens, cache_salt=self.block_hash_salt
        )

    @property
    def remaining_prompt_length(self):
        """
        Get the length of the remaining prompt tokens.
        """
        return len(self.remaining_prompt_tokens)

    ttft: Optional[float] = None
    events: List[DynamicInferenceEvent] = field(default_factory=list)
    event_add_engine: Optional[DynamicInferenceEvent] = field(default=None, repr=False)
    generated_tokens: List[int] = field(default_factory=list)

    def __str__(self):
        return ", ".join(
            (
                f"id {self.request_id}",
                f"{self.status.name}" if self.status is not None else "[NOT ADDED]",
                f"prompt len {len(self.prompt_tokens)}",
                f"gen len {len(self.generated_tokens)}",
                f"num events {len(self.events)}",
            )
        )

    def serialize(self, payload_offloaded: bool = False):
        """Converts the instance into a serializable dictionary.

        Args:
            payload_offloaded (bool): Drop the per-token payload (log probs, MoE routing
                indices) from the wire; the engine's RequestPayloadStager has custody of it.

        Returns:
            (dict) A dictionary representation of the instance suitable for
                serialization.
        """
        nvtx_range_push("DynamicInferenceRequest.serialize")

        # The prompt length is always reported (needed for usage.prompt_tokens),
        # but the prompt_tokens tensor is dropped from the wire payload unless the
        # client asked for it back (return_prompt_tokens). This keeps the large
        # prompt tensor off the engine->coordinator->API path.
        prompt_len = len(self.prompt_tokens) if self.prompt_tokens is not None else None

        # Sanity check routing_indices: ndarray [total_tokens - 1, num_layers, topk]
        if self.routing_indices is not None:
            total_tokens = prompt_len + len(self.generated_tokens)
            # the last generated token does not undergo a forward pass
            # hence we expect routing indices for total_tokens - 1
            assert self.routing_indices.shape[0] == total_tokens - 1, (
                f"routing_indices first dimension {self.routing_indices.shape[0]} does not match "
                f"total tokens {total_tokens-1}."
            )

        sampling_params = self.sampling_params
        dropped_fields = {}
        if (
            self.prompt_tokens is not None
            and sampling_params is not None
            and not getattr(sampling_params, "return_prompt_tokens", False)
        ):
            dropped_fields["prompt_tokens"] = self.prompt_tokens
        if payload_offloaded:
            dropped_fields["generated_log_probs"] = self.generated_log_probs
            dropped_fields["prompt_log_probs"] = self.prompt_log_probs
            dropped_fields["routing_indices"] = self.routing_indices

        # Dropped fields are nulled only for the duration of super().serialize();
        # this mutate-and-restore makes serialize() non-reentrant on one request object.
        try:
            for field_name in dropped_fields:
                setattr(self, field_name, None)
            obj = super().serialize()
        finally:
            for field_name, value in dropped_fields.items():
                setattr(self, field_name, value)

        obj["events"] = [e.serialize() for e in self.events]
        obj.pop("event_add_engine", None)
        obj["prompt_length"] = prompt_len
        obj["payload_offloaded"] = payload_offloaded

        nvtx_range_pop("DynamicInferenceRequest.serialize")
        return obj

    def _post_deserialize(self, obj):
        super()._post_deserialize(obj)
        self.events = [DynamicInferenceEvent.deserialize(e) for e in obj.get("events", [])]

    @property
    def tracked_metadata(self) -> List[Any]:
        """Obtain an ordered list of all request metadata to be tracked by the context.

        This consists of metadata that is used to inform text generation.
        The values of such fields are tensorized and kept aligned with the current active batch.

        Note that while the general request object is mutable, this metadata is
        inherently assumed to remain immutable once the request becomes active.
        """
        sp = self.sampling_params
        if sp.termination_id is None:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                warnings.warn(
                    f"DynamicInferenceRequest {self.request_id} has no termination_id set "
                    "in its sampling_params. Defaulting to -1."
                )
            sp.termination_id = -1
        return [getattr(sp, field) for field, _ in self.get_metadata_types()]

    @staticmethod
    def get_metadata_types() -> List[Tuple[str, torch.dtype]]:
        """Keeps track of all request metadata names and dtypes.

        Returns:
            List[Tuple[str, torch.dtype]]: Mapping from metadata name to:
                name (str) - The name of the metadata field.
                dtype (torch.dtype) - The datatype of the metadata.
        """
        return [
            ("temperature", torch.float32),
            ("top_k", torch.int32),
            ("top_p", torch.float32),
            ("termination_id", torch.int64),
            ("return_log_probs", torch.bool),
            ("skip_prompt_log_probs", torch.bool),
            ("top_n_logprobs", torch.int32),
        ]

    def add_event(
        self, type: DynamicInferenceEventType, payload: Optional[Any] = None
    ) -> DynamicInferenceEvent:
        """Add event."""
        event = DynamicInferenceEvent(type=type, payload=payload)
        self.events.append(event)
        return event

    def add_event_add_engine(self):
        """Add 'add_engine' event - called when request enters the engine queue."""
        self.event_add_engine = self.add_event(DynamicInferenceEventType.ADD_ENGINE)
        return self.event_add_engine

    def add_event_add_context(self):
        """Add 'add_context' event - called when request is added to context for prefill."""
        return self.add_event(DynamicInferenceEventType.ADD_CONTEXT)

    def add_event_generated_token(
        self,
        token: int,
        blocks_total: Optional[int] = None,
        blocks_hashed_total: Optional[int] = None,
        blocks_hashed_active: Optional[int] = None,
        blocks_ref_count: Optional[int] = None,
        pre_fwd_active_token_count: Optional[int] = None,
        pre_fwd_step_count: Optional[int] = None,
    ):
        """Add 'generated_token' event - records each generated token.

        Args:
            token (int): The token ID that was generated.
            blocks_total (int): Total block capacity from allocator.
            blocks_hashed_total (int): All allocated (hashed) blocks.
            blocks_hashed_active (int): Blocks with ref_count > 0.
            blocks_ref_count (int): Sum of block ref counts from allocator.
            pre_fwd_active_token_count (int): Active token count before forward pass.
            pre_fwd_step_count (int): Step count before forward pass.
        """
        payload = {"token_id": token}
        if blocks_total is not None:
            payload["blocks_total"] = blocks_total
        if blocks_hashed_total is not None:
            payload["blocks_hashed_total"] = blocks_hashed_total
        if blocks_hashed_active is not None:
            payload["blocks_hashed_active"] = blocks_hashed_active
        if blocks_ref_count is not None:
            payload["blocks_ref_count"] = blocks_ref_count
        if pre_fwd_active_token_count is not None:
            payload["pre_fwd_active_token_count"] = pre_fwd_active_token_count
        if pre_fwd_step_count is not None:
            payload["pre_fwd_step_count"] = pre_fwd_step_count
        return self.add_event(DynamicInferenceEventType.GENERATED_TOKEN, payload)

    def add_event_pause(self):
        """Add 'pause' event."""
        return self.add_event(DynamicInferenceEventType.PAUSE)

    def add_event_evict(self):
        """Add 'evict' event."""
        return self.add_event(DynamicInferenceEventType.EVICT)

    def add_event_finish(self):
        """Add 'finish' event."""
        return self.add_event(DynamicInferenceEventType.FINISH)

    def add_event_fail(self):
        """Add 'fail' event."""
        return self.add_event(DynamicInferenceEventType.FAIL)

    def add_event_error_transient(self, error: Exception):
        """Add transient error event."""
        return self.add_event(DynamicInferenceEventType.ERROR_TRANSIENT, error)

    def add_event_error_nontransient(self, error: Exception):
        """Add non-transient error event."""
        return self.add_event(DynamicInferenceEventType.ERROR_NONTRANSIENT, error)

    def succeeded(self) -> bool:
        """Request experienced no non-transient errors."""
        return self.status == Status.COMPLETED

    def failed(self) -> bool:
        """Request experienced non-transient error."""
        return self.status == Status.FAILED


@dataclass(kw_only=True)
class DynamicInferenceRequestRecord:
    """History of DynamicInferenceRequest objects over multiple request
    checkpoints."""

    requests: list[DynamicInferenceRequest] = field(default_factory=list)
    latency: Optional[float] = None

    @classmethod
    def from_request(cls, request: DynamicInferenceRequest) -> "DynamicInferenceRequestRecord":
        """Initialize record from a single request.

        Args:
            request (DynamicInferenceRequest): Initial request.

        Returns:
            (DynamicInferenceRequestRecord) A record.
        """
        record = cls()
        record.requests.append(request)
        return record

    def __getitem__(self, idx: int) -> DynamicInferenceRequest:
        """Get request by index.

        Args:
            idx (int): Request index.

        Returns:
            (DynamicInferenceRequest) Request object.
        """
        return self.requests[idx]

    @property
    def request_id(self) -> int:
        """Get request id.

        Returns:
            (int) Request id.
        """
        return self.requests[0].request_id

    def checkpoint(self, tokenizer: MegatronTokenizer | None = None):
        """Maintain reference to previous request, and then append a new request
        that concatenates the previous prompt and generations.

        Args:
            tokenizer (MegatronTokenizer | None): (Deprecated) Tokenizer.
        """

        old_request = self[-1]

        # Carry forward policy_epoch as-is.
        policy_epoch = old_request.policy_epoch

        # Reset kv_cache_epoch to None: the KV cache is recomputed fresh after checkpoint;
        # the engine's stamping logic will initialize a new stamp record with the recompute epoch.
        kv_cache_epoch = None

        # New prompt (concatenate prompt + generated tokens).
        new_prompt_tokens = torch.cat(
            (
                old_request.prompt_tokens,
                torch.tensor(
                    old_request.generated_tokens,
                    dtype=old_request.prompt_tokens.dtype,
                    device=old_request.prompt_tokens.device,
                ),
            ),
            dim=0,
        )

        # New sampling params.
        new_sampling_params = SamplingParams(
            **{
                **asdict(old_request.sampling_params),
                "num_tokens_to_generate": (
                    old_request.sampling_params.num_tokens_to_generate
                    - len(old_request.generated_tokens)
                ),
            }
        )

        # Preserve prefix-cache configuration and let __post_init__ recompute hashes for the
        # expanded prompt. The previous hash list may not include newly completed blocks.
        common_kwargs = dict(
            request_id=old_request.request_id,
            prompt_tokens=new_prompt_tokens,
            compact_prompt_tokens=old_request.compact_prompt_tokens,
            sampling_params=new_sampling_params,
            policy_epoch=policy_epoch,
            kv_cache_epoch=kv_cache_epoch,
            block_size_tokens=old_request.block_size_tokens,
            enable_prefix_caching=old_request.enable_prefix_caching,
            block_hash_salt=old_request.block_hash_salt,
        )
        # Preserve the VLM subtype and multimodal fields so a suspend/resume
        # cycle doesn't downcast the request to text-only and lose its imgs /
        # embeddings / token mask.
        if isinstance(old_request, DynamicVLMInferenceRequest):
            new_request = DynamicVLMInferenceRequest(
                **common_kwargs,
                num_img_embeddings_per_tile=old_request.num_img_embeddings_per_tile,
                imgs=old_request.imgs,
                num_tiles=old_request.num_tiles,
                imgs_sizes=old_request.imgs_sizes,
                num_frames=old_request.num_frames,
                media_tokens_preexpanded=old_request.media_tokens_preexpanded,
                decoder_seq_length=old_request.decoder_seq_length,
                image_embeddings=old_request.image_embeddings,
                image_token_mask=old_request.image_token_mask,
            )
        else:
            new_request = DynamicInferenceRequest(**common_kwargs)
        # Preserve event_add_engine from old request if it exists, otherwise set it.
        # This ensures TTFT calculation works correctly for evicted/resumed requests.
        if old_request.event_add_engine is not None:
            new_request.event_add_engine = old_request.event_add_engine
        else:
            new_request.add_event_add_engine()
        self.requests.append(new_request)

    def merge(self, tokenizer: MegatronTokenizer | None = None) -> DynamicInferenceRequest:
        """Merge requests into a single checkpoint-agnostic request object.

        Args:
            tokenizer (MegatronTokenizer | None): (Deprecated) Tokenizer.

        Returns:
            (DynamicInferenceRequest) Merged request.
        """

        def merge_lists(key):
            values = [getattr(request, key) for request in self.requests]
            if all(value is None for value in values):
                return None
            return [item for value in values if value is not None for item in value]

        first_request = self.requests[0]
        prompt_tokens = first_request.prompt_tokens
        prompt_text = self.requests[0].prompt
        routing_indices = None
        routing_parts = [r.routing_indices for r in self.requests if r.routing_indices is not None]
        if routing_parts:
            routing_indices = np.concatenate(routing_parts)
        generated_tokens = merge_lists("generated_tokens")
        try:
            generated_text = "".join(r.generated_text for r in self.requests)
        except TypeError as e:  # generally means r.generated_text is None
            generated_text = None

        policy_epoch = self.requests[-1].policy_epoch
        kv_cache_epoch = self.requests[-1].kv_cache_epoch
        # Preserve KV handoff metadata when merging request segments.
        disaggregated_params = self.requests[-1].disaggregated_params

        # Merged request.
        request = DynamicInferenceRequest(
            request_id=self.requests[0].request_id,
            uid=self.requests[0].uid,
            prompt=prompt_text,
            prompt_tokens=prompt_tokens,
            compact_prompt_tokens=first_request.compact_prompt_tokens,
            prompt_log_probs=self.requests[0].prompt_log_probs,
            prompt_top_n_logprobs=self.requests[0].prompt_top_n_logprobs,
            generated_text=generated_text,
            generated_tokens=generated_tokens,
            generated_length=len(generated_tokens),
            generated_log_probs=merge_lists("generated_log_probs"),
            generated_top_n_logprobs=merge_lists("generated_top_n_logprobs"),
            sampling_params=self.requests[0].sampling_params,
            policy_epoch=policy_epoch,
            kv_cache_epoch=kv_cache_epoch,
            ttft=self.requests[0].ttft,
            tpot=merge_lists("tpot"),
            status=self.requests[-1].status,
            latency=self.latency,
            events=merge_lists("events"),
            routing_indices=routing_indices,
            block_size_tokens=self.requests[0].block_size_tokens,
            enable_prefix_caching=self.requests[0].enable_prefix_caching,
            block_hash_salt=self.requests[0].block_hash_salt,
            precomputed_block_hashes=self.requests[0].precomputed_block_hashes,
            num_cached_tokens=self.requests[0].num_cached_tokens,
            disaggregated_params=disaggregated_params,
        )

        return request

    def serialize(self) -> dict:
        """Converts the instance into a serializable dictionary.

        Returns:
            (dict) A dictionary representation of the instance suitable for
                serialization.
        """
        nvtx_range_push("DynamicInferenceRequestRecord.serialize")
        obj = self.__dict__.copy()  # shallow dict copy
        obj["requests"] = [r.serialize() for r in obj["requests"]]
        nvtx_range_pop("DynamicInferenceRequestRecord.serialize")
        return obj

    @classmethod
    def deserialize(cls, obj: dict) -> "DynamicInferenceRequestRecord":
        """Deserialize record.

        Args:
            obj (dict): Serialized record data.

        Returns:
            (DynamicInferenceRequestRecord) Deserialized record.
        """
        request = cls(**obj)
        request.requests = [DynamicInferenceRequest.deserialize(r) for r in obj["requests"]]
        return request


@dataclass
class FinishedRequestRecord:
    """Stores per-request metadata that is not meant to be passed through the RESTful server."""

    policy_epoch: Optional[list[tuple[int, int]]]
    kv_cache_epoch: Optional[list[tuple[int, int]]]
    num_evictions: int

    @classmethod
    def from_request(cls, request: "DynamicInferenceRequest") -> "FinishedRequestRecord":
        """Build the request's non-RESTful metadata from a finished request."""
        # Epoch stamps exist only while the engine has a generation epoch set.
        record = cls(
            policy_epoch=(
                None if request.policy_epoch is None else [tuple(b) for b in request.policy_epoch]
            ),
            kv_cache_epoch=(
                None
                if request.kv_cache_epoch is None
                else [tuple(b) for b in request.kv_cache_epoch]
            ),
            num_evictions=sum(
                1 for event in request.events if event.type is DynamicInferenceEventType.EVICT
            ),
        )
        return record


@dataclass
class OffloadedRequestPayload:
    """A finished request's per-token payload, kept off the RESTful reply by payload offload.

    Self-contained: a consumer can rebuild the served sequence from it alone,
    keyed by the request uid (the OpenAI response id).
    """

    prompt_token_ids: Optional[list[int]]
    generated_token_ids: list[int]
    generated_log_probs: Optional[list[float]]
    prompt_log_probs: Optional[list[float]]
    routing_indices: Optional[np.ndarray]

    @classmethod
    def from_request(cls, request: "DynamicInferenceRequest") -> "OffloadedRequestPayload":
        """Copy the payload off a finished (merged) request as plain host-side values."""

        def to_plain_list(value):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return value.cpu().tolist()
            return list(value)

        if isinstance(request.prompt_tokens, torch.Tensor):
            # One device-to-host copy serves both the payload and the wire reply.
            request.prompt_tokens = request.prompt_tokens.cpu()
        return cls(
            prompt_token_ids=to_plain_list(request.prompt_tokens),
            generated_token_ids=list(request.generated_tokens),
            generated_log_probs=to_plain_list(request.generated_log_probs),
            prompt_log_probs=to_plain_list(request.prompt_log_probs),
            routing_indices=request.routing_indices,
        )


class RequestPayloadStager(Protocol):
    """Protocol to handle offloading of request payloads to a storage backend."""

    def stage(self, uid: str, payload: OffloadedRequestPayload) -> None:
        """Take custody of one finished request's payload, keyed by its OpenAI response id."""
        ...


@dataclass(kw_only=True)
class VLMInferenceRequest(InferenceRequest):
    """Class for a VLM inference request"""

    num_img_embeddings_per_tile: int
    imgs: torch.Tensor
    num_tiles: torch.Tensor
    decoder_seq_length: int


@dataclass(kw_only=True)
class DynamicVLMInferenceRequest(DynamicInferenceRequest, VLMInferenceRequest):
    """Dynamic inference request for VLM models.

    Combines DynamicInferenceRequest (for dynamic batching) with VLMInferenceRequest
    (for multimodal fields). Also stores pre-computed image embeddings and the image
    token mask produced by expand_image_tokens.
    """

    image_embeddings: Optional[torch.Tensor] = None  # [seq_img, 1, hidden]
    image_token_mask: Optional[torch.Tensor] = None  # 1D, -1=text, >=0=image index
    imgs_sizes: Optional[torch.Tensor] = None
    num_frames: Optional[torch.Tensor] = None
    media_tokens_preexpanded: bool = False
