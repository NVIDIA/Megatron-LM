# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import time
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from itertools import accumulate
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.tokenizers import MegatronTokenizer
from megatron.core.utils import experimental_api


def serialize_tensor(tensor: torch.Tensor) -> List:
    """Serialize tensor to bytes.

    Args:
        tensor (Tensor): Tensor.

    Returns:
        (List) Tensor as a list
    """
    torch.cuda.nvtx.range_push("serialize_tensor")

    # simply convert tensor into a list
    tensor = tensor.cpu().tolist()

    torch.cuda.nvtx.range_pop()
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

# Constants for hash computation
# Using 2^61 - 1 (Mersenne prime) for ~10^18 hash space, reducing collision probability
# from ~10^-9 to ~10^-18 compared to the previous prime (1000000007).
HASH_PRIME = 2305843009213693951
HASH_BASE = 31

_hash_powers: Optional[torch.Tensor] = None


def compute_block_hashes_batched(prompt_tokens: torch.Tensor, block_size: int) -> List[int]:
    """Compute hashes for all complete blocks in a prompt in one batched operation.

    Reshapes prompt tokens into [num_blocks, block_size], computes all per-block
    token hashes via a single GPU matmul, transfers results with one .tolist() call,
    and chains parent hashes on CPU.

    Args:
        prompt_tokens: All prompt token IDs, shape [seq_len].
        block_size: Number of tokens per block.

    Returns:
        List of positive integer hash values (1 to HASH_PRIME), one per complete block.
    """
    num_complete_blocks = len(prompt_tokens) // block_size
    if num_complete_blocks == 0:
        return []

    global _hash_powers
    if _hash_powers is None or _hash_powers.shape[0] != block_size:
        positions = torch.arange(block_size, device=prompt_tokens.device, dtype=torch.int64)
        _hash_powers = torch.pow(HASH_BASE, positions).to(torch.int64) % HASH_PRIME

    # Reshape to [num_blocks, block_size] (zero-copy view) and compute all token hashes
    blocks = prompt_tokens[: num_complete_blocks * block_size].view(num_complete_blocks, block_size)
    token_hashes = (blocks.to(torch.int64) * _hash_powers).sum(dim=1) % HASH_PRIME

    # Single GPU→CPU transfer
    token_hashes_list = token_hashes.tolist()

    # Chain parent hashes on CPU (C-level accumulate, no Python loop)
    hashes = list(
        accumulate(
            token_hashes_list,
            lambda parent, th: (parent * HASH_BASE + th) % HASH_PRIME + 1,
            initial=0,
        )
    )[1:]

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
                ("tensor", serialize_tensor(v)) if isinstance(v, torch.Tensor)
                else ("ndarray", serialize_ndarray(v)) if isinstance(v, np.ndarray)
                else v
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
        torch.cuda.nvtx.range_push("DynamicInferenceEvent.serialize")
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

        torch.cuda.nvtx.range_pop()
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
    prompt: Optional[str] = None
    prompt_tokens: Optional[torch.Tensor] = None
    # remaining prompt tokens are used for chunked prefill
    remaining_prompt_tokens: Optional[torch.Tensor] = None
    policy_epoch: Optional[list[tuple[int, int]]] = None
    kv_cache_epoch: Optional[list[tuple[int, int]]] = None
    latency: Optional[float] = None
    # routing_indices stores MoE routing decisions for all tokens generated so far.
    # Shape: [total_tokens, num_layers, topk] - accumulated across all generation steps.
    # Stored as numpy array (int16/int32) on CPU.
    routing_indices: Optional[np.ndarray] = None
    # block_cache_key is set when routing_indices are written to block store.
    # Contains {"instance_rank", "instance_id", "req_id", "key"} for retrieval.
    block_cache_key: Optional[dict] = None
    finished_chunk_token_count: int = 0
    stop_word_ids: Optional[List[List[int]]] = None  # Tokenized stop words (populated internally)

    # Prefix caching fields
    block_size_tokens: Optional[int] = None  # Block size for hash computation
    enable_prefix_caching: bool = False  # Whether prefix caching is enabled

    # Computed field - not passed by caller
    precomputed_block_hashes: List[int] = field(default_factory=list)

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
            self.prompt_tokens, self.block_size_tokens
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

    def serialize(self):
        """Converts the instance into a serializable dictionary.

        Returns:
            (dict) A dictionary representation of the instance suitable for
                serialization.
        """
        torch.cuda.nvtx.range_push("DynamicInferenceRequest.serialize")
        obj = super().serialize()
        obj["events"] = [e.serialize() for e in self.events]
        obj.pop("event_add_engine", None)

        # Sanity check routing_indices: Tensor [total_tokens - 1, num_layers, topk]
        if self.routing_indices is not None:
            total_tokens = len(self.prompt_tokens) + len(self.generated_tokens)
            # the last generated token does not undergo a forward pass
            # hence we expect routing indices for total_tokens - 1
            assert self.routing_indices.shape[0] == total_tokens - 1, (
                f"routing_indices first dimension {self.routing_indices.shape[0]} does not match "
                f"total tokens {total_tokens-1}."
            )

        torch.cuda.nvtx.range_pop()
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

        # New request.
        new_request = DynamicInferenceRequest(
            request_id=old_request.request_id,
            prompt_tokens=new_prompt_tokens,
            sampling_params=new_sampling_params,
            policy_epoch=policy_epoch,
            kv_cache_epoch=kv_cache_epoch,
        )
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
            if getattr(self.requests[0], key) is None:
                return None
            else:
                return [v for r in self.requests for v in getattr(r, key)]

        prompt_tokens = self.requests[0].prompt_tokens
        prompt_text = self.requests[0].prompt
        routing_indices = None
        routing_parts = [r.routing_indices for r in self.requests if r.routing_indices is not None]
        if routing_parts:
            routing_indices = np.concatenate(routing_parts)
        block_cache_key = self.requests[-1].block_cache_key
        generated_tokens = merge_lists("generated_tokens")
        try:
            generated_text = "".join(r.generated_text for r in self.requests)
        except TypeError as e:  # generally means r.generated_text is None
            generated_text = None

        policy_epoch = self.requests[-1].policy_epoch
        kv_cache_epoch = self.requests[-1].kv_cache_epoch

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
            policy_epoch=policy_epoch,
            kv_cache_epoch=kv_cache_epoch,
            ttft=self.requests[0].ttft,
            tpot=merge_lists("tpot"),
            status=self.requests[-1].status,
            latency=self.latency,
            events=merge_lists("events"),
            routing_indices=routing_indices,
            block_cache_key=block_cache_key,
            block_size_tokens=self.requests[0].block_size_tokens,
            enable_prefix_caching=self.requests[0].enable_prefix_caching,
            precomputed_block_hashes=self.requests[0].precomputed_block_hashes,
        )

        return request

    def dump_routing_indices_to_block_store(self, block_store_instance, block_store_instance_rank, block_store_instance_id):
        """Write routing indices to block store and replace with cache key.

        Mirrors vLLM serving.py _maybe_store_moe_topk_indices.
        Merges routing_indices across all sub-requests, writes to block store,
        sets block_cache_key, and clears routing_indices to free memory.
        """
        import logging
        import ray

        parts = [r.routing_indices for r in self.requests if r.routing_indices is not None]
        if not parts:
            return

        routing_indices = np.concatenate(parts) if len(parts) > 1 else parts[0]
        req_id = str(self.requests[0].request_id)

        ray.get(
            block_store_instance.put_numpy.remote(
                req_id,
                "moe_topk_indices",
                routing_indices,
            )
        )

        block_cache_key = {
            "instance_rank": block_store_instance_rank,
            "instance_id": block_store_instance_id,
            "req_id": req_id,
            "key": "moe_topk_indices",
        }

        logging.info(
            f"Block store put: req_id={req_id} "
            f"shape={list(routing_indices.shape)} "
            f"dtype={routing_indices.dtype}"
        )

        # Set cache key and clear indices on all sub-requests.
        for r in self.requests:
            r.block_cache_key = block_cache_key
            r.routing_indices = None

    def serialize(self) -> dict:
        """Converts the instance into a serializable dictionary.

        Returns:
            (dict) A dictionary representation of the instance suitable for
                serialization.
        """
        torch.cuda.nvtx.range_push("DynamicInferenceRequestRecord.serialize")
        obj = self.__dict__.copy()  # shallow dict copy
        obj["requests"] = [r.serialize() for r in obj["requests"]]
        torch.cuda.nvtx.range_pop()
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
