# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import io
import time
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import torch

from megatron.core.inference.sampling_params import SamplingParams


def serialize_tensor(tensor):
    """Serialize tensor to bytes."""
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    tensor_bytes = buffer.read()
    return tensor_bytes


def deserialize_tensor(tensor_bytes):
    """Deserialize tensor from bytes."""
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

    def serializable(self):
        """
        Converts the instance into a serializable dictionary.
        Returns:
            dict: A dictionary representation of the instance suitable for serialization.
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

    def serialize(self):
        """
        Converts the instance into a serializable dictionary.
        Returns:
            dict: A dictionary representation of the instance suitable for serialization.
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

    def serializable(self):
        """
        Converts the instance into a serializable dictionary.
        Returns:
            dict: A dictionary representation of the instance suitable for serialization.
        """
        obj = super().serializable()
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
        return [getattr(sp, field) for field in self.get_metadata_labels().keys()]

    @staticmethod
    def get_metadata_labels() -> Dict[str, int]:
        """Provides human-readable labels for the tracked metadata fields."""
        ret = [
            "temperature",
            "top_k",
            "top_p",
            "termination_id",
            "return_log_probs",
            "skip_prompt_log_probs",
        ]
        return {k: v for v, k in enumerate(ret)}

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
class VLMInferenceRequest(InferenceRequest):
    """Class for a VLM inference request"""

    num_img_embeddings_per_tile: int
    imgs: torch.Tensor
    num_tiles: torch.Tensor
    decoder_seq_length: int
