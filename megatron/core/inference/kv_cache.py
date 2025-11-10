# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""KV Cache implementations for different memory layouts."""

import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Tuple, Union

import torch
from torch import Tensor


class KVCacheEfficiencyWarning(UserWarning):
    """Custom warning for inefficient KV cache operations."""

    pass


class KVCacheLayout(Enum):
    """
    Enum representing the different KV cache memory layouts.
    Note: Layer dimension is NOT included - it's handled outside the cache.

    The names correspond to the data layout:
    M = Merged, S = Separate
    2 = K/V dimension
    N = Chunks, C = Chunk Size, H = Heads, D = Head Dimension
    """

    M_2NCHD = "KVCacheM2NCHD"
    """Merged cache layout: [2, Chunks, ChunkSize, Heads, Dim]"""

    M_N2CHD = "KVCacheMN2CHD"
    """Merged cache layout: [Chunks, 2, ChunkSize, Heads, Dim]"""

    M_N2HCD = "KVCacheMN2HCD"
    """Merged cache layout: [Chunks, 2, Heads, ChunkSize, Dim]"""

    S_NCHD = "KVCacheSNCHD"
    """Separate K and V caches. Layout: [Chunks, ChunkSize, Heads, Dim]"""

    S_NHCD = "KVCacheSNHCD"
    """Separate K and V caches. Layout: [Chunks, Heads, ChunkSize, Dim]"""


class KVCacheBase(ABC):
    """
    Base class for KV cache implementations.
    Each cache instance represents a single layer's cache.
    """

    def __init__(
        self,
        num_chunks: int,
        chunk_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.num_chunks: int = num_chunks
        self.chunk_size: int = chunk_size
        self.num_kv_heads: int = num_kv_heads
        self.head_dim: int = head_dim
        self.device: Optional[torch.device] = device
        self.dtype: Optional[torch.dtype] = dtype

    @abstractmethod
    def get_content(self) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Returns the cache content in its most efficient representation.

        Returns:
            - For merged layouts: single tensor
            - For separate layouts: tuple of (k_cache, v_cache)
        """
        raise NotImplementedError

    @abstractmethod
    def append(
        self,
        key: Tensor,
        value: Tensor,
        padded_active_token_count: int,
        token_to_block_idx: Tensor,
        token_to_local_position_within_kv_block: Tensor,
    ) -> None:
        """
        Appends key-value pairs to the cache.

        Args:
            key: Key tensor to append
            value: Value tensor to append
            padded_active_token_count: Number of active tokens
            token_to_block_idx: Mapping from token to block index
            token_to_local_position_within_kv_block: Mapping from token to position within block
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Resets the cache."""
        raise NotImplementedError

    def supports_triton(self) -> bool:
        """
        Returns True if this cache layout is compatible with Triton kernels.
        All layouts (M_2NCHD, M_N2CHD, M_N2HCD, S_NCHD, S_NHCD, MLA) are now Triton-compatible.
        """
        return False


class MLACache(KVCacheBase):
    """
    Cache for Multi-Latent Attention (MLA).
    Stores compressed latent representation instead of full K/V.
    Layout: [Chunks, ChunkSize, LatentDim]
    """

    def __init__(
        self,
        num_chunks: int,
        chunk_size: int,
        kv_reduced_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        # MLA doesn't use num_kv_heads or head_dim in the same way
        super().__init__(
            num_chunks=num_chunks,
            chunk_size=chunk_size,
            num_kv_heads=0,  # Not used for MLA
            head_dim=kv_reduced_dim,  # Reuse head_dim for latent dim
            device=device,
            dtype=dtype,
        )
        self.kv_reduced_dim = kv_reduced_dim
        self.cache: Tensor = torch.full(
            (num_chunks, chunk_size, kv_reduced_dim), -1, dtype=dtype, device=device
        )

    def get_content(self) -> Tensor:
        """Returns the MLA latent cache tensor."""
        return self.cache

    def append(
        self,
        key: Tensor,
        value: Tensor,
        padded_active_token_count: int,
        token_to_block_idx: Tensor,
        token_to_local_position_within_kv_block: Tensor,
    ) -> None:
        """
        Append latent representation to MLA cache.
        For MLA, 'key' contains the concatenated latent representation.
        """
        block_idx = token_to_block_idx[:padded_active_token_count]
        local_kv_seq_idx = token_to_local_position_within_kv_block[:padded_active_token_count]

        # For MLA, key contains the kv_concat latent representation
        kv_concat = key.squeeze(1)
        self.cache[block_idx, local_kv_seq_idx] = kv_concat[:padded_active_token_count]

    def reset(self):
        """Resets the MLA cache."""
        self.cache.fill_(-1)

    def supports_triton(self) -> bool:
        """MLA is Triton-compatible."""
        return True


class KVCacheM2NCHD(KVCacheBase):
    """
    Merged KV cache with shape [2, Chunks, ChunkSize, Heads, Dim].
    Layout: 2, N, C, H, D
    Triton-compatible for Flash backend.
    """

    def __init__(
        self,
        num_chunks: int,
        chunk_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(num_chunks, chunk_size, num_kv_heads, head_dim, device, dtype)
        self.cache: Tensor = torch.full(
            (2, num_chunks, chunk_size, num_kv_heads, head_dim), -1, dtype=dtype, device=device
        )

    def get_content(self) -> Tensor:
        """Returns the merged cache tensor."""
        return self.cache

    def append(
        self,
        key: Tensor,
        value: Tensor,
        padded_active_token_count: int,
        token_to_block_idx: Tensor,
        token_to_local_position_within_kv_block: Tensor,
    ) -> None:
        """Append K/V to merged cache."""
        block_idx = token_to_block_idx[:padded_active_token_count]
        local_kv_seq_idx = token_to_local_position_within_kv_block[:padded_active_token_count]

        assert key.size(1) == 1 and value.size(1) == 1, "Expected sequence length of 1"
        key = key.squeeze(1)
        value = value.squeeze(1)

        self.cache[0, block_idx, local_kv_seq_idx] = key[:padded_active_token_count]
        self.cache[1, block_idx, local_kv_seq_idx] = value[:padded_active_token_count]

    def reset(self):
        """Resets the merged cache."""
        self.cache.fill_(-1)

    def supports_triton(self) -> bool:
        """M_2NCHD is Triton-compatible."""
        return True


class KVCacheMN2CHD(KVCacheBase):
    """
    Merged KV cache with shape [Chunks, 2, ChunkSize, Heads, Dim].
    Layout: N, 2, C, H, D
    """

    def __init__(
        self,
        num_chunks: int,
        chunk_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(num_chunks, chunk_size, num_kv_heads, head_dim, device, dtype)
        self.cache: Tensor = torch.full(
            (num_chunks, 2, chunk_size, num_kv_heads, head_dim), -1, dtype=dtype, device=device
        )

    def get_content(self) -> Tensor:
        """Returns the merged cache tensor."""
        return self.cache

    def append(
        self,
        key: Tensor,
        value: Tensor,
        padded_active_token_count: int,
        token_to_block_idx: Tensor,
        token_to_local_position_within_kv_block: Tensor,
    ) -> None:
        """Append K/V to merged cache."""
        block_idx = token_to_block_idx[:padded_active_token_count]
        local_kv_seq_idx = token_to_local_position_within_kv_block[:padded_active_token_count]

        assert key.size(1) == 1 and value.size(1) == 1, "Expected sequence length of 1"
        key = key.squeeze(1)
        value = value.squeeze(1)

        self.cache[block_idx, 0, local_kv_seq_idx] = key[:padded_active_token_count]
        self.cache[block_idx, 1, local_kv_seq_idx] = value[:padded_active_token_count]

    def reset(self):
        """Resets the merged cache."""
        self.cache.fill_(-1)

    def supports_triton(self) -> bool:
        """M_N2CHD is Triton-compatible."""
        return True


class KVCacheMN2HCD(KVCacheBase):
    """
    Merged KV cache with shape [Chunks, 2, Heads, ChunkSize, Dim].
    Layout: N, 2, H, C, D
    Used by FlashInfer backends (fa2, fa3, trt).
    """

    def __init__(
        self,
        num_chunks: int,
        chunk_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(num_chunks, chunk_size, num_kv_heads, head_dim, device, dtype)
        self.cache: Tensor = torch.full(
            (num_chunks, 2, num_kv_heads, chunk_size, head_dim), -1, dtype=dtype, device=device
        )

    def get_content(self) -> Tensor:
        """Returns the merged cache tensor."""
        return self.cache

    def append(
        self,
        key: Tensor,
        value: Tensor,
        padded_active_token_count: int,
        token_to_block_idx: Tensor,
        token_to_local_position_within_kv_block: Tensor,
    ) -> None:
        """Append K/V to merged cache."""
        block_idx = token_to_block_idx[:padded_active_token_count]
        local_kv_seq_idx = token_to_local_position_within_kv_block[:padded_active_token_count]

        assert key.size(1) == 1 and value.size(1) == 1, "Expected sequence length of 1"
        key = key.squeeze(1)
        value = value.squeeze(1)

        self.cache[block_idx, 0, :, local_kv_seq_idx, :] = key[:padded_active_token_count]
        self.cache[block_idx, 1, :, local_kv_seq_idx, :] = value[:padded_active_token_count]

    def reset(self):
        """Resets the merged cache."""
        self.cache.fill_(-1)

    def supports_triton(self) -> bool:
        """M_N2HCD is Triton-compatible."""
        return True


class KVCacheSNCHD(KVCacheBase):
    """
    Separate K and V caches, both with shape [Chunks, ChunkSize, Heads, Dim].
    Layout: N, C, H, D
    Triton-compatible.
    """

    def __init__(
        self,
        num_chunks: int,
        chunk_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(num_chunks, chunk_size, num_kv_heads, head_dim, device, dtype)
        self.k_cache: Tensor = torch.full(
            (num_chunks, chunk_size, num_kv_heads, head_dim), -1, dtype=dtype, device=device
        )
        self.v_cache: Tensor = torch.full(
            (num_chunks, chunk_size, num_kv_heads, head_dim), -1, dtype=dtype, device=device
        )

    def get_content(self) -> Tuple[Tensor, Tensor]:
        """Returns (k_cache, v_cache) tuple."""
        return (self.k_cache, self.v_cache)

    def append(
        self,
        key: Tensor,
        value: Tensor,
        padded_active_token_count: int,
        token_to_block_idx: Tensor,
        token_to_local_position_within_kv_block: Tensor,
    ) -> None:
        """Append K/V to separate caches."""
        block_idx = token_to_block_idx[:padded_active_token_count]
        local_kv_seq_idx = token_to_local_position_within_kv_block[:padded_active_token_count]

        assert key.size(1) == 1 and value.size(1) == 1, "Expected sequence length of 1"
        key = key.squeeze(1)
        value = value.squeeze(1)

        self.k_cache[block_idx, local_kv_seq_idx] = key[:padded_active_token_count]
        self.v_cache[block_idx, local_kv_seq_idx] = value[:padded_active_token_count]

    def reset(self):
        """Resets all caches."""
        self.k_cache.fill_(-1)
        self.v_cache.fill_(-1)

    def supports_triton(self) -> bool:
        """S_NCHD is Triton-compatible."""
        return True


class KVCacheSNHCD(KVCacheBase):
    """
    Separate K and V caches, both with shape [Chunks, Heads, ChunkSize, Dim].
    Layout: N, H, C, D
    """

    def __init__(
        self,
        num_chunks: int,
        chunk_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(num_chunks, chunk_size, num_kv_heads, head_dim, device, dtype)
        self.k_cache: Tensor = torch.full(
            (num_chunks, num_kv_heads, chunk_size, head_dim), -1, dtype=dtype, device=device
        )
        self.v_cache: Tensor = torch.full(
            (num_chunks, num_kv_heads, chunk_size, head_dim), -1, dtype=dtype, device=device
        )

    def get_content(self) -> Tuple[Tensor, Tensor]:
        """Returns (k_cache, v_cache) tuple."""
        return (self.k_cache, self.v_cache)

    def reset(self):
        """Resets all caches."""
        self.k_cache.fill_(-1)
        self.v_cache.fill_(-1)

    def append(
        self,
        key: Tensor,
        value: Tensor,
        padded_active_token_count: int,
        token_to_block_idx: Tensor,
        token_to_local_position_within_kv_block: Tensor,
    ) -> None:
        """Append K/V to separate caches."""
        block_idx = token_to_block_idx[:padded_active_token_count]
        local_kv_seq_idx = token_to_local_position_within_kv_block[:padded_active_token_count]

        assert key.size(1) == 1 and value.size(1) == 1, "Expected sequence length of 1"
        key = key.squeeze(1)
        value = value.squeeze(1)

        self.k_cache[block_idx, :, local_kv_seq_idx, :] = key[:padded_active_token_count]
        self.v_cache[block_idx, :, local_kv_seq_idx, :] = value[:padded_active_token_count]

    def supports_triton(self) -> bool:
        """S_NHCD is Triton-compatible."""
        return True


# --- Factory Function ---
def create_mhagqa_cache(
    layout: KVCacheLayout,
    num_chunks: int,
    chunk_size: int,
    num_kv_heads: int,
    head_dim: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> KVCacheBase:
    """
    Factory function to create a KV cache instance with a specified layout.

    Args:
        layout: The desired memory layout from the KVCacheLayout enum.
        num_chunks: Number of chunks (blocks) in the cache.
        chunk_size: The number of tokens per chunk (block).
        num_kv_heads: The number of key/value heads.
        head_dim: The dimension of each head.
        device: The torch device to create the cache on.
        dtype: The torch dtype for the cache tensor.

    Returns:
        An instance of a KVCacheBase subclass.
    """
    layout_to_class = {
        KVCacheLayout.M_2NCHD: KVCacheM2NCHD,
        KVCacheLayout.M_N2CHD: KVCacheMN2CHD,
        KVCacheLayout.M_N2HCD: KVCacheMN2HCD,
        KVCacheLayout.S_NCHD: KVCacheSNCHD,
        KVCacheLayout.S_NHCD: KVCacheSNHCD,
    }

    cache_class = layout_to_class.get(layout)
    if cache_class is None:
        raise ValueError(f"Unknown KV cache layout: {layout}")

    return cache_class(
        num_chunks=num_chunks,
        chunk_size=chunk_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )
