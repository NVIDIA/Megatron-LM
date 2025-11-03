# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Context metadata management for dynamic inference.

This module provides abstraction for managing request/token metadata in dynamic batching inference.
Context metadata includes per-request state, per-token state, and KV block mappings.
This is separate from attention metadata (MHA-specific structures).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import torch

from megatron.core.transformer.enums import AttnBackend


class BaseContextMetadata(ABC):
    """Base class for context metadata management with shared utilities.

    Context metadata = per-request and per-token bookkeeping tensors.
    This is separate from attention metadata (MHA metadata structures).
    """

    def __init__(self):
        pass

    # ===== SHARED HELPER METHODS (used by both Sync and Async) =====

    def _create_single_tensor_set(
        self,
        device: torch.device,
        max_requests: int,
        max_tokens: int,
        max_kv_block_count: int
    ) -> Dict[str, torch.Tensor]:
        """Create ONE set of context metadata tensors.

        This method is called:
        - Once by SyncContextMetadata
        - Twice by AsyncContextMetadata (for slots 0 and 1)

        Args:
            device: CUDA device
            max_requests: Maximum number of requests
            max_tokens: Maximum number of tokens
            max_kv_block_count: Maximum KV blocks per sequence

        Returns:
            Dict of all context metadata tensors
        """
        return {
            # Per-request state (7 tensors)
            'request_ids': torch.full(
                (max_requests,), -1, dtype=torch.int32, device=device
            ),
            'request_query_lengths': torch.zeros(
                max_requests, dtype=torch.int32, device=device
            ),
            'request_output_lengths': torch.zeros(
                max_requests, dtype=torch.int32, device=device
            ),
            'request_kv_length_offsets': torch.zeros(
                max_requests, dtype=torch.int32, device=device
            ),
            'request_kv_block_counts': torch.zeros(
                max_requests, dtype=torch.int32, device=device
            ),
            'request_last_kv_block_id': torch.zeros(
                max_requests, dtype=torch.int32, device=device
            ),
            'request_last_kv_block_offset': torch.zeros(
                max_requests, dtype=torch.int32, device=device
            ),
            'request_last_token_ids': torch.zeros(
                max_requests, dtype=torch.long, device=device
            ),

            # Per-token state (6 tensors)
            'token_to_input_ids': torch.full(
                (max_tokens,), 0, dtype=torch.long, device=device
            ),
            'token_to_pos_ids': torch.zeros(
                max_tokens, dtype=torch.long, device=device
            ),
            'token_to_request_idx': torch.zeros(
                max_tokens, dtype=torch.long, device=device
            ),
            'token_to_block_idx': torch.zeros(
                max_tokens, dtype=torch.long, device=device
            ),
            'token_to_position_in_request': torch.zeros(
                max_tokens, dtype=torch.long, device=device
            ),
            'token_to_local_position_within_kv_block': torch.zeros(
                max_tokens, dtype=torch.long, device=device
            ),

            # Block mapping (1 tensor)
            'request_to_kv_block_ids': torch.full(
                (max_requests, max_kv_block_count),
                -1,
                dtype=torch.int,
                device=device,
            ),
        }

    def _create_single_attention_metadata_pair(
        self,
        context,
        copy_id: int = 0
    ) -> Dict[str, Any]:
        """Create ONE pair of graph/non-graph attention metadata.

        This method is called:
        - Once by SyncContextMetadata (copy_id=0)
        - Twice by AsyncContextMetadata (copy_id=0 and 1)

        Encapsulates the existing logic from lines 398-470 of dynamic_context.py.

        Args:
            context: DynamicInferenceContext instance
            copy_id: Slot ID (0 or 1), used for async mode

        Returns:
            Dict with 'graph' and 'non_graph' attention metadata objects
        """
        from .attention_context.mha_metadata import GraphMHAMetadata, NonGraphMHAMetadata
        from .attention_context.mha_splitpd_metadata import MHASplitPDMetadata
        from .attention_context.mha_flashinfer_metadata import (
            GraphMHAFlashInferMetadata,
            NonGraphMHAFlashInferMetadata
        )

        attention_backend = context.attention_backend
        block_count_total = context.block_allocator.block_count_total

        result = {'graph': {}, 'non_graph': {}}

        # Flash or Auto backend
        if attention_backend in [AttnBackend.flash, AttnBackend.auto]:
            result['graph']['mha_metadata'] = GraphMHAMetadata(
                block_count_total=block_count_total,
                max_kv_block_count=context.max_kv_block_count,
                max_requests=context.max_requests,
                block_size_tokens=context.block_size_tokens,
                max_seqlen=context.max_sequence_length,
            )

            result['non_graph']['mha_metadata'] = NonGraphMHAMetadata(
                block_count_total=block_count_total,
                max_kv_block_count=context.max_kv_block_count,
                max_requests=context.max_requests,
                block_size_tokens=context.block_size_tokens,
                max_seqlen=context.max_sequence_length,
            )

        # Flash Split backend
        elif attention_backend == AttnBackend.flash_split:
            result['graph']['mha_metadata'] = MHASplitPDMetadata(
                block_count_total=block_count_total,
                max_kv_block_count=context.max_kv_block_count,
                max_requests=context.max_requests,
                block_size_tokens=context.block_size_tokens,
                max_seqlen=context.max_sequence_length,
            )
            result['non_graph']['mha_metadata'] = MHASplitPDMetadata(
                block_count_total=block_count_total,
                max_kv_block_count=context.max_kv_block_count,
                max_requests=context.max_requests,
                block_size_tokens=context.block_size_tokens,
                max_seqlen=context.max_sequence_length,
            )

        # FlashInfer backends
        elif attention_backend in [
            AttnBackend.flashinfer_fa2,
            AttnBackend.flashinfer_fa3,
            AttnBackend.flashinfer_trt
        ]:
            result['graph']['mha_metadata'] = GraphMHAFlashInferMetadata(
                block_count_total=block_count_total,
                max_kv_block_count=context.max_kv_block_count,
                max_requests=context.max_requests,
                block_size_tokens=context.block_size_tokens,
                max_seqlen=context.max_sequence_length,
                max_num_tokens=context.max_tokens,
                backend=attention_backend,
                prefill_workspace_size=1 * 1024 * 1024 * 1024,  # 1GB
                decode_workspace_size=1 * 1024 * 1024 * 1024,   # 1GB
            )

            result['non_graph']['mha_metadata'] = NonGraphMHAFlashInferMetadata(
                block_count_total=block_count_total,
                max_kv_block_count=context.max_kv_block_count,
                max_requests=context.max_requests,
                block_size_tokens=context.block_size_tokens,
                max_seqlen=context.max_sequence_length,
                max_num_tokens=context.max_tokens,
                backend=attention_backend,
                prefill_workspace_size=1 * 1024 * 1024 * 1024,
                decode_workspace_size=1 * 1024 * 1024 * 1024,
            )

            # Set model parameters for FlashInfer planning
            for mode in ['graph', 'non_graph']:
                result[mode]['mha_metadata'].set_model_params(
                    num_qo_heads=context.num_attention_qo_heads_per_partition,
                    num_kv_heads=context.num_attention_kv_heads_per_partition,
                    head_dim=context.hidden_size_per_attention_head,
                    params_dtype=context.params_dtype,
                )

        return result

    # ===== ABSTRACT METHODS (must be implemented by subclasses) =====

    @abstractmethod
    def initialize(self, context):
        """Initialize context metadata tensors.

        Args:
            context: DynamicInferenceContext instance
        """
        pass

    @abstractmethod
    def get_active_tensors(self) -> Dict[str, torch.Tensor]:
        """Return currently active context metadata tensors.

        Returns:
            Dict mapping tensor name to tensor
        """
        pass

    @abstractmethod
    def get_active_attention_metadata(self) -> Dict[str, Any]:
        """Return currently active attention metadata.

        Returns:
            Dict with 'graph' and 'non_graph' keys
        """
        pass


class SyncContextMetadata(BaseContextMetadata):
    """Synchronous single-copy context metadata management.

    This maintains backward compatibility with existing sync behavior.
    All metadata stored as single copies (not double-buffered).
    """

    def __init__(self):
        super().__init__()
        # Single copy of tensors
        self.tensors = None
        self.attn_metadata = None

        # Scalar tracking counts
        self.total_request_count = 0
        self.active_token_count = 0
        self.paused_request_count = 0
        self.padded_active_token_count = 0
        self.padded_active_request_count = 0
        self.num_prefill_requests = 0
        self.chunked_prefill_request_id = -1
        self.paused_tokens = None

    def initialize(self, context):
        """Initialize using shared helpers - called ONCE.

        Args:
            context: DynamicInferenceContext instance
        """
        device = torch.cuda.current_device()

        # Create single set of tensors using SHARED method
        self.tensors = self._create_single_tensor_set(
            device,
            context.max_requests,
            context.max_tokens,
            context.max_kv_block_count
        )

        # Create attention metadata using SHARED method (copy_id=0)
        self.attn_metadata = self._create_single_attention_metadata_pair(context, copy_id=0)

    def get_active_tensors(self) -> Dict[str, torch.Tensor]:
        """Return the single copy of tensors.

        Returns:
            Dict of all context metadata tensors
        """
        return self.tensors

    def get_active_attention_metadata(self) -> Dict[str, Any]:
        """Return the single attention metadata pair.

        Returns:
            Dict with 'graph' and 'non_graph' keys
        """
        return self.attn_metadata

    @property
    def active_id(self) -> int:
        """Return active ID (always 0 for sync mode).

        Returns:
            0 (sync mode has only one copy)
        """
        return 0



class AsyncContextMetadata(BaseContextMetadata):
    """Asynchronous double-buffered context metadata management.

    Maintains two copies of request/token metadata so CPU preparation can overlap
    with GPU execution. Slot 0 is active by default; slot switching is managed by
    DynamicInferenceContext when async scheduling is enabled.
    """

    _PER_REQUEST_KEYS: List[str] = [
        'request_ids',
        'request_query_lengths',
        'request_output_lengths',
        'request_kv_length_offsets',
        'request_kv_block_counts',
        'request_last_kv_block_id',
        'request_last_kv_block_offset',
        'request_last_token_ids',
    ]

    _PER_REQUEST_FILL: Dict[str, int] = {
        'request_ids': -1,
        'request_query_lengths': 0,
        'request_output_lengths': 0,
        'request_kv_length_offsets': 0,
        'request_kv_block_counts': 0,
        'request_last_kv_block_id': -1,
        'request_last_kv_block_offset': 0,
        'request_last_token_ids': 0,
    }

    _PER_TOKEN_KEYS: List[str] = [
        'token_to_input_ids',
        'token_to_pos_ids',
        'token_to_request_idx',
        'token_to_block_idx',
        'token_to_position_in_request',
        'token_to_local_position_within_kv_block',
    ]

    _PER_TOKEN_FILL: Dict[str, int] = {
        'token_to_input_ids': 0,
        'token_to_pos_ids': 0,
        'token_to_request_idx': -1,
        'token_to_block_idx': -1,
        'token_to_position_in_request': 0,
        'token_to_local_position_within_kv_block': 0,
    }

    def __init__(self):
        super().__init__()

        # Double-buffered tensors and attention metadata
        self.tensors: List[Optional[Dict[str, torch.Tensor]]] = [None, None]
        self.attn_metadata: List[Optional[Dict[str, Any]]] = [None, None]

        # Double-buffered scalar tracking values
        self.total_request_count_array = [0, 0]
        self.active_token_count_array = [0, 0]
        self.paused_request_count_array = [0, 0]
        self.padded_active_token_count_array = [0, 0]
        self.padded_active_request_count_array = [0, 0]
        self.num_prefill_requests_array = [0, 0]
        self.chunked_prefill_request_id_array = [-1, -1]
        self.paused_tokens_array = [None, None]

        # Active slot tracking
        self.ACTIVE_ID = 0
        self.debug = False

        # CUDA synchronization primitives (created during initialize)
        self.execution_stream: Optional[torch.cuda.Stream] = None
        self.meta_data_ready: Optional[torch.cuda.Event] = None
        self.gpu_finished: Optional[torch.cuda.Event] = None

        # Request state tracking helpers (populated during initialize)
        self.select_map: Optional[torch.Tensor] = None
        self.finished_map: Optional[torch.Tensor] = None
        self.paused_map: Optional[torch.Tensor] = None
        self.active_map: Optional[torch.Tensor] = None

        # Buffers used by async pipeline (optional)
        self.logits: Optional[torch.Tensor] = None
        self.old_logits: Optional[torch.Tensor] = None
        self.chunked_prefill_request_id_post_process: int = -1

    # ----- Helpers -----

    def _old_idx(self) -> int:
        return 1 - self.active_id

    def _active_tensor_dict(self) -> Dict[str, torch.Tensor]:
        tensors = self.tensors[self.active_id]
        if tensors is None:
            raise RuntimeError("AsyncContextMetadata not initialized")
        return tensors

    def _old_tensor_dict(self) -> Dict[str, torch.Tensor]:
        tensors = self.tensors[self._old_idx()]
        if tensors is None:
            raise RuntimeError("AsyncContextMetadata not initialized")
        return tensors

    # ----- Initialization -----

    def initialize(self, context):
        """Create streams, events, and double-buffered tensor sets."""
        device_index = torch.cuda.current_device()
        device = torch.device('cuda', device_index)

        # Create async coordination primitives
        self.execution_stream = torch.cuda.Stream(device=device)
        self.meta_data_ready = torch.cuda.Event(enable_timing=False)
        self.gpu_finished = torch.cuda.Event(enable_timing=False)

        # Allocate tensor sets and attention metadata for both slots
        for copy_id in (0, 1):
            self.tensors[copy_id] = self._create_single_tensor_set(
                device,
                context.max_requests,
                context.max_tokens,
                context.max_kv_block_count,
            )
            self.attn_metadata[copy_id] = self._create_single_attention_metadata_pair(
                context,
                copy_id=copy_id,
            )

        # Initialize helper tensors
        self.select_map = torch.empty(0, dtype=torch.int32, device=device)
        self.finished_map = torch.empty(0, dtype=torch.int32, device=device)
        self.paused_map = torch.empty(0, dtype=torch.int32, device=device)
        self.active_map = torch.empty(0, dtype=torch.int32, device=device)

        # Allocate optional logits buffers
        self.logits = torch.empty((1, 0, 1), dtype=torch.float32, device=device)
        self.old_logits = torch.empty((1, 0, 1), dtype=torch.float32, device=device)

        self.ACTIVE_ID = 0

    # ----- Scalar properties (active) -----

    @property
    def total_request_count(self) -> int:
        return self.total_request_count_array[self.active_id]

    @total_request_count.setter
    def total_request_count(self, value: int):
        self.total_request_count_array[self.active_id] = value

    @property
    def active_token_count(self) -> int:
        return self.active_token_count_array[self.active_id]

    @active_token_count.setter
    def active_token_count(self, value: int):
        self.active_token_count_array[self.active_id] = value

    @property
    def paused_request_count(self) -> int:
        return self.paused_request_count_array[self.active_id]

    @paused_request_count.setter
    def paused_request_count(self, value: int):
        self.paused_request_count_array[self.active_id] = value

    @property
    def padded_active_token_count(self) -> int:
        return self.padded_active_token_count_array[self.active_id]

    @padded_active_token_count.setter
    def padded_active_token_count(self, value: int):
        self.padded_active_token_count_array[self.active_id] = value

    @property
    def padded_active_request_count(self) -> int:
        return self.padded_active_request_count_array[self.active_id]

    @padded_active_request_count.setter
    def padded_active_request_count(self, value: int):
        self.padded_active_request_count_array[self.active_id] = value

    @property
    def num_prefill_requests(self) -> int:
        return self.num_prefill_requests_array[self.active_id]

    @num_prefill_requests.setter
    def num_prefill_requests(self, value: int):
        self.num_prefill_requests_array[self.active_id] = value

    @property
    def chunked_prefill_request_id(self) -> int:
        return self.chunked_prefill_request_id_array[self.active_id]

    @chunked_prefill_request_id.setter
    def chunked_prefill_request_id(self, value: int):
        self.chunked_prefill_request_id_array[self.active_id] = value

    @property
    def paused_tokens(self):
        return self.paused_tokens_array[self.active_id]

    @paused_tokens.setter
    def paused_tokens(self, value):
        self.paused_tokens_array[self.active_id] = value

    # ----- Scalar properties (previous slot) -----

    @property
    def total_request_count_old(self) -> int:
        return self.total_request_count_array[self._old_idx()]

    @property
    def active_token_count_old(self) -> int:
        return self.active_token_count_array[self._old_idx()]

    @property
    def paused_request_count_old(self) -> int:
        return self.paused_request_count_array[self._old_idx()]

    @property
    def padded_active_token_count_old(self) -> int:
        return self.padded_active_token_count_array[self._old_idx()]

    @property
    def padded_active_request_count_old(self) -> int:
        return self.padded_active_request_count_array[self._old_idx()]

    @property
    def num_prefill_requests_old(self) -> int:
        return self.num_prefill_requests_array[self._old_idx()]

    @property
    def chunked_prefill_request_id_old(self) -> int:
        return self.chunked_prefill_request_id_array[self._old_idx()]

    @property
    def paused_tokens_old(self):
        return self.paused_tokens_array[self._old_idx()]

    # ----- Tensor accessors -----

    def get_active_tensors(self) -> Dict[str, torch.Tensor]:
        return self._active_tensor_dict()

    def get_old_tensors(self) -> Dict[str, torch.Tensor]:
        return self._old_tensor_dict()

    def get_active_attention_metadata(self) -> Dict[str, Any]:
        metadata = self.attn_metadata[self.active_id]
        if metadata is None:
            raise RuntimeError("AsyncContextMetadata not initialized")
        return metadata

    def get_old_attention_metadata(self) -> Dict[str, Any]:
        metadata = self.attn_metadata[self._old_idx()]
        if metadata is None:
            raise RuntimeError("AsyncContextMetadata not initialized")
        return metadata

    # ----- Slot management -----

    @property
    def active_id(self) -> int:
        """Return current active ID (0 or 1).

        Returns:
            Current active metadata slot ID
        """
        return self.ACTIVE_ID

    def activate_metadata(self, copy_id: int):
        """Switch active metadata slot."""
        if copy_id not in (0, 1):
            raise ValueError(f"copy_id must be 0 or 1, got {copy_id}")
        self.ACTIVE_ID = copy_id

    def copy_old_counts(self):
        """Copy scalar counters from previous slot to active slot."""
        new_id = self.active_id
        old_id = self._old_idx()
        self.total_request_count_array[new_id] = self.total_request_count_array[old_id]
        self.active_token_count_array[new_id] = self.active_token_count_array[old_id]
        self.paused_request_count_array[new_id] = self.paused_request_count_array[old_id]
        self.padded_active_token_count_array[new_id] = self.padded_active_token_count_array[old_id]
        self.padded_active_request_count_array[new_id] = (
            self.padded_active_request_count_array[old_id]
        )
        self.num_prefill_requests_array[new_id] = self.num_prefill_requests_array[old_id]
        self.chunked_prefill_request_id_array[new_id] = (
            self.chunked_prefill_request_id_array[old_id]
        )
        self.paused_tokens_array[new_id] = self.paused_tokens_array[old_id]

    def _reset_per_request_tail(self, tensors: Dict[str, torch.Tensor], start_idx: int):
        for key in self._PER_REQUEST_KEYS:
            fill_value = self._PER_REQUEST_FILL[key]
            tensors[key][start_idx:].fill_(fill_value)
        tensors['request_to_kv_block_ids'][start_idx:].fill_(-1)

    def select_segments(self, ltensor, rtensor, select_map):
        """Copy selected segments from rtensor to ltensor using select_map."""
        ltensor[:len(select_map)] = rtensor[select_map]

    def select_old_tensors_to_active(self, select_map: torch.Tensor):
        """Copy selected rows from old tensors into the active slot."""
        new_tensors = self._active_tensor_dict()
        old_tensors = self._old_tensor_dict()

        if select_map is None or select_map.numel() == 0:
            self._reset_per_request_tail(new_tensors, 0)
            for key in self._PER_TOKEN_KEYS:
                fill_value = self._PER_TOKEN_FILL[key]
                new_tensors[key].fill_(fill_value)
            tensors = new_tensors['request_to_kv_block_ids']
            tensors.fill_(-1)
            return

        # Per-request tensors - copy selected segments
        for key in self._PER_REQUEST_KEYS:
            self.select_segments(new_tensors[key], old_tensors[key], select_map)

        # Per-token tensors - copy selected segments
        for key in self._PER_TOKEN_KEYS:
            self.select_segments(new_tensors[key], old_tensors[key], select_map)

        # Block mapping - copy selected segments
        self.select_segments(new_tensors['request_to_kv_block_ids'], old_tensors['request_to_kv_block_ids'], select_map)

        # Reset remainder
        self._reset_per_request_tail(new_tensors, len(select_map))



    # ----- CUDA graph/event helpers -----

    def record_metadata_ready(self, stream: Optional[torch.cuda.Stream] = None):
        if self.meta_data_ready is None:
            return
        if stream is None:
            stream = torch.cuda.current_stream()
        self.meta_data_ready.record(stream)
