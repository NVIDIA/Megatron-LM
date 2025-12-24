from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor, nn

from megatron.core import parallel_state

if TYPE_CHECKING:
    from megatron.core.transformer import TransformerConfig


@dataclass
class ContextParallelHandler(ABC):
    """
    Abstract base class for Context Parallel (CP) handlers.
    Manages distribution, combination, and manipulation of tensors across context parallel ranks.
    """

    # Legacy parameters for PackedSeqParams
    qkv_format: Optional[Literal["sbhd", "bshd", "thd"]] = None
    cp_group: Optional[dist.ProcessGroup] = None

    cu_seqlens_q: Optional[Tensor] = None
    cu_seqlens_kv: Optional[Tensor] = None
    cu_seqlens_q_padded: Optional[Tensor] = None
    cu_seqlens_kv_padded: Optional[Tensor] = None
    max_seqlen_q: Optional[int] = None
    max_seqlen_kv: Optional[int] = None

    # 在dcp中使用
    local_cp_size: Optional[int] = None

    # 在DefaultContextParallelHandler中使用
    seqlens_q_list: Optional[List[int]] = None
    seqlens_kv_list: Optional[List[int]] = None
    seqlens_q_padded: Optional[torch.Tensor] = None
    seqlens_kv_padded: Optional[torch.Tensor] = None
    # Lists containing flattened [actual_len, padded_len] pairs
    seqlens_q_with_padded_list: Optional[List[int]] = None
    seqlens_kv_with_padded_list: Optional[List[int]] = None
    total_seqlen_padded_q: Optional[int] = None
    total_seqlen_padded_kv: Optional[int] = None

    _post_initialized: bool = False
    _cp_size: int = 1

    def __post_init__(self):
        if self.qkv_format is None:
            self.qkv_format = "sbhd"

        if self.cp_group is None:
            self.cp_group = parallel_state.get_context_parallel_group(check_initialized=False)

        if self.cp_group is not None:
            self._cp_size = dist.get_world_size(self.cp_group)

    @abstractmethod
    def dispatch(
        self, seq_dim: int, tensor: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """Splits and dispatches the tensor to the appropriate CP rank during forward pass."""
        pass

    @abstractmethod
    def combine(
        self, seq_dim: int, tensor: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """Combines tensors from different CP ranks (gather) during forward pass."""
        pass

    @abstractmethod
    def apply_rotary_pos_emb(
        self,
        tensor: torch.Tensor,
        freq: torch.Tensor,
        config: "TransformerConfig",
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Applies Rotary Positional Embeddings considering context parallelism."""
        pass

    @abstractmethod
    def get_emb_on_this_cp_rank(self, emb: torch.Tensor) -> torch.Tensor:
        """Retrieves the slice of embeddings belonging to the current CP rank."""
        pass

    @abstractmethod
    def roll_tensor(
        self, tensor: torch.Tensor, shifts: int = -1, dims: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rolls the tensor elements along a dimension, handling communication across CP ranks."""
        pass

    @abstractmethod
    def core_attn(self, attn_mod: nn.Module, *args: Any, **kwargs: Any) -> Any:
        """Executes the core attention logic using this handler."""
        pass
