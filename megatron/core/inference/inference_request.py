# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import io
import time
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import torch

from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.tokenizers import MegatronTokenizer
from megatron.core.utils import experimental_api


def serialize_tensor(tensor: torch.Tensor) -> bytes:
    """Serialize tensor to bytes.

    Args:
        tensor (Tensor): Tensor.

    Returns:
        (bytes) Byte representation of tensor.
    """
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    tensor_bytes = buffer.read()
    return tensor_bytes


def deserialize_tensor(tensor_bytes: bytes) -> torch.Tensor:
    """Deserialize tensor from bytes.

    Args:
        tensor_bytes (bytes): Byte representation of tensor.

    Returns:
        (Tensor) Tensor.
    """
    buffer = io.BytesIO(tensor_bytes)
    tensor = torch.load(buffer)
    return tensor


# class syntax
class Status(Enum):
    """Enum for status"""

    WAITING_IN_QUEUE = 1
    ACTIVE_AND_GENERATING_TOKENS = 2
    ACTIVE_BUT_NOT_GENERATING_TOKENS = 3
    COMPLETED = 4
    FAILED = 5


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
    tpot: Optional[List[int]] = None

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
        obj = asdict(self)
        obj["status"] = self.status.name if self.status else None

        # Serialize tensors.
        obj = {
            k: (("tensor", serialize_tensor(v)) if isinstance(v, torch.Tensor) else v)
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
        request.status = None if obj["status"] is None else Status[obj["status"]]

        # Deserialize tensors.
        for k, v in obj.items():
            if isinstance(v, list) and len(v) == 2 and v[0] == "tensor":
                setattr(request, k, deserialize_tensor(v[1]))

        return request


class DynamicInferenceEventType(Enum):
    """Dynamic inference event type."""

    ADD = auto()
    PAUSE = auto()
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
        else:
            assert self.payload is None

    def __str__(self):
        payload_str = "" if self.payload is None else f", {type(self.payload).__name__}"
        return f"[{self.timestamp:.3f}] {self.type.name}{payload_str}"

    def serialize(self) -> dict:
        """Converts the instance into a serializable dictionary.

        Returns:
            (dict) A dictionary representation of the instance suitable for
                serialization.
        """

        # Dataclass to dict.
        obj = asdict(self)
        obj["type"] = self.type.name

        # Serialize payload.
        if self.payload:
            from .contexts.dynamic_context import ContextErrorFactory  # avoid circular import.

            obj["payload"] = ContextErrorFactory.serialize(self.payload)

        return obj

    @classmethod
    def deserialize(cls, obj: dict) -> "DynamicInferenceEvent":
        """Deserialize event.

        Args:
            obj (dict): Serialized event data.

        Returns:
            (DynamicInferenceEvent) Deserialized event.
        """

        # Initialize event.
        event = cls(**{**obj, "type": DynamicInferenceEventType[obj["type"]]})

        # Deserialize payload.
        if obj["payload"]:
            from .contexts.dynamic_context import ContextErrorFactory  # avoid circular import.

            event.payload = ContextErrorFactory.deserialize(obj["payload"])

        return event


@experimental_api
@dataclass(kw_only=True)
class DynamicInferenceRequest(InferenceRequest):
    """Class for one inference request

    Containing relevant data for an dynamic inference request

    """

    request_id: int
    generated_tokens: List[int] = field(default_factory=list)
    prompt: Optional[str] = None
    prompt_tokens: Optional[torch.Tensor] = None
    # remaining prompt tokens are used for chunked prefill
    remaining_prompt_tokens: Optional[torch.Tensor] = None
    latency: Optional[float] = None
    finished_chunk_token_count = 0

    def __post_init__(self):
        self.sampling_params = copy.deepcopy(self.sampling_params)
        if self.prompt_tokens is not None:
            self.remaining_prompt_tokens = copy.deepcopy(self.prompt_tokens)

    @property
    def remaining_prompt_length(self):
        """
        Get the length of the remaining prompt tokens.
        """
        return len(self.remaining_prompt_tokens)

    events: List[DynamicInferenceEvent] = field(default_factory=list)

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

    def serialize(self):
        """Converts the instance into a serializable dictionary.

        Returns:
            (dict) A dictionary representation of the instance suitable for
                serialization.
        """
        obj = super().serialize()
        obj["events"] = [e.serialize() for e in self.events]
        return obj

    @classmethod
    def deserialize(cls, obj: dict) -> "DynamicInferenceRequest":
        """Deserialize request.

        Args:
            obj (dict): Serialized request data.

        Returns:
            (DynamicInferenceRequest) Deserialized request.
        """
        request = super().deserialize(obj)
        request.events = [DynamicInferenceEvent.deserialize(e) for e in obj["events"]]
        return request

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
        return [getattr(sp, field) for field, _, _ in self.get_metadata_types()]

    @staticmethod
    def get_metadata_types() -> List[Tuple[str, torch.dtype, bool]]:
        """Keeps track of all request metadata names, dtypes, and target device.

        Returns:
            List[Tuple[str, torch.dtype, bool]]: Mapping from metadata name to:
                name (str) - The name of the metadata field.
                dtype (torch.dtype) - The datatype of the metadata.
                on_device (bool) - Whether the metadata lives on GPU (True) or CPU (False).
        """
        return [
            ("temperature", torch.float32, False),  # CPU for torch sampling
            ("top_k", torch.int32, False),  # CPU for torch sampling
            ("top_p", torch.float32, False),  # CPU for torch sampling
            ("termination_id", torch.int64, True),
            ("return_log_probs", torch.bool, False),  # CPU for non-selective logprobs
            ("skip_prompt_log_probs", torch.bool, False),  # CPU for non-selective logprobs
            ("top_n_logprobs", torch.int32, False),  # CPU for torch sampling
        ]

    def add_event(self, type: DynamicInferenceEventType, payload: Optional[Any] = None) -> None:
        """Add event."""
        self.events.append(DynamicInferenceEvent(type=type, payload=payload))

    def add_event_add(self):
        """Add 'add' event."""
        return self.add_event(DynamicInferenceEventType.ADD)

    def add_event_pause(self):
        """Add 'pause' event."""
        return self.add_event(DynamicInferenceEventType.PAUSE)

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
    """History of DynamicInferenceRequest objects over multiple suspend and
    resumes."""

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

    def suspend(self, tokenizer: MegatronTokenizer | None = None):
        """Suspend request by storing references to previous prompt, generations,
        and sampling params.

        Args:
            tokenizer (MegatronTokenizer | None): (Deprecated) Tokenizer.
        """

        old_request = self[-1]

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

        # New request.
        new_request = DynamicInferenceRequest(
            request_id=old_request.request_id,
            prompt_tokens=new_prompt_tokens,
            sampling_params=new_sampling_params,
        )
        self.requests.append(new_request)

    def merge(self, tokenizer: MegatronTokenizer | None = None) -> DynamicInferenceRequest:
        """Merge requests into a single suspend-agnostic request object.

        Args:
            tokenizer (MegatronTokenizer | None): (Deprecated) Tokenizer.

        Returns:
            (DynamicInferenceRequest) Merged request.
        """

        def merge_lists(key):
            if getattr(self.requests[0], key) is None:
                return None
            else:
                return [v for r in self.requests for v in getattr(r, key)]

        prompt_tokens = self.requests[0].prompt_tokens
        prompt_text = self.requests[0].prompt
        generated_tokens = merge_lists("generated_tokens")
        generated_text = "".join(r.generated_text for r in self.requests)

        # Merged request.
        request = DynamicInferenceRequest(
            request_id=self.requests[0].request_id,
            prompt=prompt_text,
            prompt_tokens=prompt_tokens,
            prompt_log_probs=self.requests[0].prompt_log_probs,
            prompt_top_n_logprobs=self.requests[0].prompt_top_n_logprobs,
            generated_text=generated_text,
            generated_tokens=generated_tokens,
            generated_length=len(generated_tokens),
            generated_log_probs=merge_lists("generated_log_probs"),
            generated_top_n_logprobs=merge_lists("generated_top_n_logprobs"),
            sampling_params=self.requests[0].sampling_params,
            tpot=merge_lists("tpot"),
            status=self.requests[-1].status,
            latency=self.latency,
            events=merge_lists("events"),
        )

        return request

    def serialize(self) -> dict:
        """Converts the instance into a serializable dictionary.

        Returns:
            (dict) A dictionary representation of the instance suitable for
                serialization.
        """
        obj = asdict(self)
        obj["requests"] = [r.serialize() for r in self.requests]
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


@dataclass(kw_only=True)
class VLMInferenceRequest(InferenceRequest):
    """Class for a VLM inference request"""

    num_img_embeddings_per_tile: int
    imgs: torch.Tensor
    num_tiles: torch.Tensor
    decoder_seq_length: int
