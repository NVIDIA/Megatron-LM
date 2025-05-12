from __future__ import annotations

"""Token- and weight-partitioning helper (CP, TP, SP).

The adapter slices sequences across *context-parallel* ranks and can further
scatter them across *sequence-parallel* ranks when sequence-parallelism is
enabled.
"""
import dataclasses
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Protocol, runtime_checkable

import torch
from torch.nn import functional as F
from torch.distributed import ProcessGroup
from megatron.core.utils import is_te_min_version
from megatron.core import tensor_parallel
from megatron.core.parallel_state import get_context_parallel_group
from megatron.core.models.multimodal import context_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.utils import get_batch_on_this_cp_rank

try:
    import transformer_engine_torch as tex  # type: ignore

    _HAVE_TEX = True
except ModuleNotFoundError:  # pragma: no cover
    tex = None  # type: ignore
    _HAVE_TEX = False


@dataclass(frozen=True)
class PartitionConfig:
    """Minimal runtime information needed to shard inputs.

    NOTE: Always construct PartitionConfig using the provided classmethods
    (from_model or from_mp_config) to ensure all fields, including cp_group,
    are set correctly.
    """

    cp_size: int
    tp_size: int
    seq_parallel: bool
    use_cp: bool
    tp_comm_overlap: bool
    max_seq_len: int
    kv_format: str = "sbhd"  # "sbhd" | "thd"
    cp_group: Optional[ProcessGroup] = None

    @property
    def is_partitioning_enabled(self) -> bool:
        """Returns True if context parallelism or sequence parallelism is active."""
        return self.use_cp or self.seq_parallel

    @property
    def group_size(self) -> int:
        """Returns the size of the parallelism group."""
        if self.use_cp:
            return self.cp_size
        if self.seq_parallel:
            return self.tp_size
        return 1

    @property
    def group_rank(self) -> int:
        """Returns the rank within the parallelism group."""
        if self.use_cp:
            return self.cp_group.rank() if self.cp_group else 0
        if self.seq_parallel:
            from megatron.core.parallel_state import get_tensor_model_parallel_rank

            return get_tensor_model_parallel_rank()
        return 0

    @runtime_checkable
    class _ParallelAttrs(Protocol):
        context_parallel_lm: int
        tensor_model_parallel_size_lm: int
        sequence_parallel_lm: bool
        tp_comm_overlap_lm: bool
        _language_max_sequence_length: int
        _kv_format: str
        cp_group: Optional[ProcessGroup]

    @classmethod
    def from_model(cls, model: _ParallelAttrs) -> "PartitionConfig":
        """
        Creates a PartitionConfig from a model.
        """
        cp_size = model.context_parallel_lm
        cp_group = model.cp_group if cp_size > 1 else None
        if cp_size > 1 and cp_group is None:
            cp_group = get_context_parallel_group()

        return cls(
            cp_size=cp_size,
            tp_size=model.tensor_model_parallel_size_lm,
            seq_parallel=model.sequence_parallel_lm,
            use_cp=cp_size > 1,
            tp_comm_overlap=model.tp_comm_overlap_lm,
            max_seq_len=model._language_max_sequence_length,
            kv_format=model._kv_format,
            cp_group=cp_group,
        )

    @classmethod
    def from_mp_config(
        cls,
        mp: ModelParallelConfig,
        *,
        max_seq_len: int,
        kv_format: str = "sbhd",
        cp_group: Optional[ProcessGroup] = None,
    ) -> "PartitionConfig":
        """
        Creates a PartitionConfig from a ModelParallelConfig.
        """
        if not isinstance(mp, ModelParallelConfig):
            raise TypeError("mp must be a ModelParallelConfig instance")

        cp_size = mp.context_parallel_size
        if cp_size > 1 and cp_group is None:
            cp_group = get_context_parallel_group()

        return cls(
            cp_size=cp_size,
            tp_size=mp.tensor_model_parallel_size,
            seq_parallel=mp.sequence_parallel,
            use_cp=cp_size > 1,
            tp_comm_overlap=mp.tp_comm_overlap,
            max_seq_len=max_seq_len,
            kv_format=kv_format,
            cp_group=cp_group,
        )


class PartitionAdapter:
    """Shard batch-first embeddings & label tensors for Context and Sequence Parallelism."""

    def __init__(self, cfg: PartitionConfig):
        """Initialize the partition adapter.
        Args:
            cfg: PartitionConfig, the configuration for the partition adapter.
        """
        self.cfg = cfg

    # pylint: disable=too-many-arguments
    def shard(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        packed_seq_params: Optional[PackedSeqParams] = None,
        *,
        pre: bool,
        post: bool,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[PackedSeqParams],
    ]:
        """
        Apply context parallel (CP) and sequence parallel (SP) sharding to input tensors.

        Args:
            embeddings (torch.Tensor):
                Input embeddings tensor. Shape: (B, S, H)
            labels (torch.Tensor):
                Labels tensor. Shape: (B, S)
            loss_mask (torch.Tensor):
                Loss mask tensor. Shape: (B, S)
            attention_mask (torch.Tensor):
                Attention mask tensor. Shape: (B, S)
            packed_seq_params (PackedSeqParams, optional):
                Packed sequence parameters. Defaults to None.
            pre (bool):
                Whether this is the pre-forward stage (affects which tensors are sharded).
            post (bool):
                Whether this is the post-forward stage (affects which tensors are sharded).

        Returns:
            Tuple containing:
                - embeddings (torch.Tensor): Possibly sharded embeddings. Shape: (B, S, H) or (S, B, H) after sharding.
                - labels (torch.Tensor): Possibly sharded labels. Shape: (B, S)
                - loss_mask (torch.Tensor): Possibly sharded loss mask. Shape: (B, S)
                - attention_mask (torch.Tensor): Possibly sharded attention mask. Shape: (B, S)
                - packed_seq_params (PackedSeqParams, optional): Updated packed sequence parameters.
        """
        if not (self.cfg.use_cp or self.cfg.seq_parallel):
            return embeddings, labels, loss_mask, attention_mask, packed_seq_params

        # When we are in the `pre` stage we can already sanity-check the
        # sequence length before any sharding happens.
        if pre and embeddings is not None:
            shard_factor = None
            seq_dim = None  # which dimension holds the token sequence

            if self.cfg.use_cp and self.cfg.seq_parallel:
                shard_factor = self.cfg.tp_size * self.cfg.cp_size * 2
                seq_dim = 1  # embeddings shape: [B, S, H]
            elif self.cfg.use_cp:
                shard_factor = self.cfg.cp_size * 2
                seq_dim = 1
            elif self.cfg.seq_parallel:
                shard_factor = self.cfg.tp_size
                seq_dim = 0  # embeddings shape: [S, B, H]

            if shard_factor is not None and (
                packed_seq_params is None
                or getattr(packed_seq_params, 'qkv_format', 'sbhd') == 'sbhd'
            ):
                assert (
                    embeddings.shape[seq_dim] % shard_factor == 0
                ), (
                    f"Sequence length should be divisible by {shard_factor} "
                    "for Sequence/Context parallelism"
                )

                if self.cfg.seq_parallel and self.cfg.tp_comm_overlap:
                    assert (
                        embeddings.shape[seq_dim] == self.cfg.max_seq_len
                    ), (
                        "TP Comm overlap requires Vision+Text token length "
                        "== language_max_sequence_length"
                    )

        if self.cfg.use_cp:
            embeddings, labels, loss_mask, attention_mask, packed_seq_params = self._apply_context_parallel(
                embeddings, labels, loss_mask, attention_mask, packed_seq_params, pre, post
            )

        if self.cfg.seq_parallel and pre and embeddings is not None:
            embeddings = tensor_parallel.scatter_to_sequence_parallel_region(embeddings)

        return embeddings, labels, loss_mask, attention_mask, packed_seq_params

    # pylint: disable=too-many-arguments
    def _apply_context_parallel(
        self,
        embeddings: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        loss_mask: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        packed_seq_params: Optional[PackedSeqParams],
        pre: bool,
        post: bool,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[PackedSeqParams]]:
        """
        Apply context parallel (CP) sharding to input tensors.

        Args:
            embeddings (Optional[torch.Tensor]):
                Input embeddings tensor. Shape: (B, S, H)
            labels (Optional[torch.Tensor]):
                Labels tensor. Shape: (B, S)
            loss_mask (Optional[torch.Tensor]):
                Loss mask tensor. Shape: (B, S)
            attention_mask (Optional[torch.Tensor]):
                Attention mask tensor. Shape: (B, S)
            packed_seq_params (PackedSeqParams, optional):
                Packed sequence parameters. Defaults to None.
            pre (bool):
                Whether this is the pre-forward stage (affects which tensors are sharded).
            post (bool):
                Whether this is the post-forward stage (affects which tensors are sharded).

        Returns:
            Tuple containing:
                - embeddings (Optional[torch.Tensor]): Possibly sharded embeddings. Shape: (S, B, H) after sharding.
                - labels (Optional[torch.Tensor]): Possibly sharded labels. Shape: (B, S)
                - loss_mask (Optional[torch.Tensor]): Possibly sharded loss mask. Shape: (B, S)
                - attention_mask (Optional[torch.Tensor]): Possibly sharded attention mask. Shape: (B, S)
                - packed_seq_params (PackedSeqParams, optional): Updated packed sequence parameters.
        """
        if not self.cfg.use_cp:
            return embeddings, labels, loss_mask, attention_mask, packed_seq_params

        # Distribute sequence across CP ranks
        batch = dict()
        if pre and embeddings is not None:
            batch["embeddings"] = embeddings
        if post:
            if labels is not None:
                batch["labels"] = labels
            if loss_mask is not None:
                batch["loss_mask"] = loss_mask
            if attention_mask is not None:
                batch["attention_mask"] = attention_mask

        if (
            packed_seq_params is None
            or getattr(packed_seq_params, 'qkv_format', 'sbhd') == 'sbhd'
        ):
            batch = get_batch_on_this_cp_rank(batch)
        else:
            assert _HAVE_TEX and is_te_min_version(
                "1.10.0"
            ), "Please update Transformer Engine to >= 1.10 to use Context Parallel with THD format data"
            cp_size = self.cfg.cp_group.size()
            cp_rank = self.cfg.cp_group.rank()
            for key, data in batch.items():
                index = tex.thd_get_partitioned_indices(
                    packed_seq_params.cu_seqlens_q_padded, data.size(1), cp_size, cp_rank
                )
                batch[key] = data.index_select(1, index)

        # Extract sharded tensors
        embeddings = batch.get("embeddings", None)
        if embeddings is not None:
            # Convert from [b, s/cp, h] to [s/cp, b, h]
            embeddings = embeddings.transpose(0, 1).contiguous()
        labels = batch.get("labels", None)
        loss_mask = batch.get("loss_mask", None)
        attention_mask = batch.get("attention_mask", None)

        return embeddings, labels, loss_mask, attention_mask, packed_seq_params