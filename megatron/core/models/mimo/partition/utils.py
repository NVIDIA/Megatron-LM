# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Token and weight partitioning helper (CP, TP, SP).

The adapter slices sequences across *context-parallel* ranks and can further
scatter them across *sequence-parallel* ranks when sequence-parallelism is
enabled.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch  # type: ignore[import-not-found]
from torch.distributed import ProcessGroup  # type: ignore[import-not-found]

from megatron.core import tensor_parallel
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import get_context_parallel_group, get_tensor_model_parallel_group
from megatron.core.utils import (
    get_batch_on_this_cp_rank,
    get_pg_rank,
    get_pg_size,
    is_te_min_version,
)

try:
    import transformer_engine_torch as tex  # type: ignore

    _HAVE_TEX = True
except ModuleNotFoundError:  # pragma: no cover
    tex = None  # type: ignore
    _HAVE_TEX = False


@dataclass(frozen=True)
class PartitionConfig:
    """Minimal runtime information needed to shard inputs.

    NOTE: Always construct PartitionConfig using the provided classmethod
    (from_mp_config) to ensure all fields, including cp_group and tp_group,
    are set correctly.
    """

    seq_parallel: bool
    use_cp: bool
    tp_comm_overlap: bool
    max_seq_len: int
    kv_format: str = "sbhd"  # "sbhd" | "thd"
    cp_group: Optional[ProcessGroup] = None
    tp_group: Optional[ProcessGroup] = None

    @classmethod
    def from_mp_config(
        cls,
        mp: ModelParallelConfig,
        *,
        max_seq_len: int,
        kv_format: str = "sbhd",
        cp_group: Optional[ProcessGroup] = None,
        tp_group: Optional[ProcessGroup] = None,
    ) -> "PartitionConfig":
        """
        Creates a PartitionConfig from a ModelParallelConfig.
        """
        if not isinstance(mp, ModelParallelConfig):
            raise TypeError("mp must be a ModelParallelConfig instance")

        if mp.context_parallel_size > 1 and cp_group is None:
            cp_group = get_context_parallel_group()

        if mp.sequence_parallel and tp_group is None:
            tp_group = get_tensor_model_parallel_group()

        return cls(
            seq_parallel=mp.sequence_parallel,
            use_cp=get_pg_size(cp_group) > 1,
            tp_comm_overlap=mp.tp_comm_overlap,
            max_seq_len=max_seq_len,
            kv_format=kv_format,
            cp_group=cp_group,
            tp_group=tp_group,
        )


class PartitionAdapter:
    """Shard MIMO sequence inputs (CP/SP) and return language-model-ready embeddings."""

    def __init__(self, cfg: PartitionConfig):
        """Initialize the partition adapter.
        Args:
            cfg: PartitionConfig, the configuration for the partition adapter.
        """
        self.cfg = cfg

    def shard(
        self,
        embeddings: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        loss_mask: Optional[torch.Tensor],
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[PackedSeqParams],
    ]:
        """Apply context- (CP) and sequence-parallel (SP) sharding along the sequence.

        Inputs:
            - embeddings: sequence-first ``(S, B, H)``, or ``None`` on non-first PP stages.
            - labels / loss_mask: batch-first ``(B, S)``.

        Returns ``(embeddings, labels, loss_mask, packed_seq_params)`` with embeddings in
        ``(S/(cp*tp), B, H)`` language-model layout. CP shards every tensor along the
        sequence; SP additionally scatters only the embeddings -- labels/loss_mask are not
        SP-scattered because the loss runs after the TP all-gather on the full CP-local
        sequence. A dense ``[B, S]`` attention mask is intentionally not handled: under CP
        it cannot line up with the sharded sequence, so MIMO masks via a causal
        ``attn_mask_type`` or ``packed_seq_params`` (THD) instead.
        """
        # Sanity-check the sequence length before sharding. Embeddings are sequence-first,
        # so the token sequence is dim 0.
        if embeddings is not None:
            shard_factor = None

            if self.cfg.use_cp and self.cfg.seq_parallel:
                shard_factor = get_pg_size(self.cfg.tp_group) * get_pg_size(self.cfg.cp_group) * 2
            elif self.cfg.use_cp:
                shard_factor = get_pg_size(self.cfg.cp_group) * 2
            elif self.cfg.seq_parallel:
                shard_factor = get_pg_size(self.cfg.tp_group)

            if shard_factor is not None and (
                packed_seq_params is None
                or getattr(packed_seq_params, 'qkv_format', 'sbhd') == 'sbhd'
            ):
                assert embeddings.shape[0] % shard_factor == 0, (
                    f"Sequence length should be divisible by {shard_factor} "
                    "for Sequence/Context parallelism"
                )

                if self.cfg.seq_parallel and self.cfg.tp_comm_overlap:
                    assert embeddings.shape[0] == self.cfg.max_seq_len, (
                        "TP Comm overlap requires Vision+Text token length "
                        "== language_max_sequence_length"
                    )

        if self.cfg.use_cp:
            # CP shards batch-first (get_batch_on_this_cp_rank requirement): transpose
            # (S, B, H) -> (B, S, H) in, then the CP-local result back to (S/cp, B, H).
            if embeddings is not None:
                embeddings = embeddings.transpose(0, 1).contiguous()
            embeddings, labels, loss_mask, packed_seq_params = self._apply_context_parallel(
                embeddings, labels, loss_mask, packed_seq_params
            )
            if embeddings is not None:
                embeddings = embeddings.transpose(0, 1).contiguous()

        # SP scatters the sequence (dim 0); the SP-only path needs no transpose.
        if self.cfg.seq_parallel and embeddings is not None:
            embeddings = tensor_parallel.scatter_to_sequence_parallel_region(
                embeddings, group=self.cfg.tp_group
            )

        return embeddings, labels, loss_mask, packed_seq_params

    def _apply_context_parallel(
        self,
        embeddings: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        loss_mask: Optional[torch.Tensor],
        packed_seq_params: Optional[PackedSeqParams],
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[PackedSeqParams],
    ]:
        """
        Apply context parallel (CP) sharding to input tensors.

        Args:
            embeddings (Optional[torch.Tensor]):
                Input embeddings tensor. Shape: (B, S, H)
            labels (Optional[torch.Tensor]):
                Labels tensor. Shape: (B, S)
            loss_mask (Optional[torch.Tensor]):
                Loss mask tensor. Shape: (B, S)
            packed_seq_params (PackedSeqParams, optional):
                Packed sequence parameters. Defaults to None.

        Returns:
            Tuple containing:
                - embeddings (Optional[torch.Tensor]): Sharded embeddings. Shape: (B, S/cp, H)
                - labels (Optional[torch.Tensor]): Possibly sharded labels. Shape: (B, S/cp)
                - loss_mask (Optional[torch.Tensor]): Possibly sharded loss mask. Shape: (B, S/cp)
                - packed_seq_params (PackedSeqParams, optional): Updated packed sequence parameters.
        """
        if not self.cfg.use_cp:
            return embeddings, labels, loss_mask, packed_seq_params

        # Distribute sequence across CP ranks
        batch = dict()
        if embeddings is not None:
            batch["embeddings"] = embeddings
        if labels is not None:
            batch["labels"] = labels
        if loss_mask is not None:
            batch["loss_mask"] = loss_mask

        if packed_seq_params is None or getattr(packed_seq_params, 'qkv_format', 'sbhd') == 'sbhd':
            batch = get_batch_on_this_cp_rank(batch, is_hybrid_cp=False, cp_group=self.cfg.cp_group)
        else:
            assert _HAVE_TEX and is_te_min_version("1.10.0"), (
                "Please update Transformer Engine to >= 1.10 "
                "to use Context Parallel with THD format data"
            )
            assert self.cfg.cp_group is not None
            cp_size = get_pg_size(self.cfg.cp_group)
            cp_rank = get_pg_rank(self.cfg.cp_group)
            for key, data in batch.items():
                index = tex.thd_get_partitioned_indices(
                    packed_seq_params.cu_seqlens_q_padded, data.size(1), cp_size, cp_rank
                )
                batch[key] = data.index_select(1, index)

        # Extract sharded tensors; embeddings stay in [B, S/cp, H]. shard() transposes
        # them to language-model layout [S/cp, B, H] after CP and before optional SP.
        embeddings = batch.get("embeddings", None)
        labels = batch.get("labels", None)
        loss_mask = batch.get("loss_mask", None)

        return embeddings, labels, loss_mask, packed_seq_params
