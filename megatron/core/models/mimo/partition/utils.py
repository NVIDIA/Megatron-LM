# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Token and activation partitioning helper for MIMO language models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch  # type: ignore[import-not-found]
from torch.distributed import ProcessGroup  # type: ignore[import-not-found]

from megatron.core import tensor_parallel
from megatron.core.context_parallel import (
    ContextParallelBatch,
    CPLayout,
    get_batches_on_this_cp_rank,
)
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.parallel_state import get_context_parallel_group, get_tensor_model_parallel_group
from megatron.core.utils import get_pg_size


@dataclass(frozen=True)
class PartitionConfig:
    """Minimal runtime information needed to partition language-model inputs."""

    seq_parallel: bool
    use_cp: bool
    tp_comm_overlap: bool
    max_seq_len: int
    kv_format: str = "sbhd"  # "sbhd" | "thd"
    linear_cp_layout: CPLayout = "zigzag"
    attention_cp_layout: CPLayout = "zigzag"
    cp_group: Optional[ProcessGroup] = None
    tp_group: Optional[ProcessGroup] = None
    tp_cp_group: Optional[ProcessGroup] = None

    @classmethod
    def from_mp_config(
        cls,
        mp: ModelParallelConfig,
        *,
        max_seq_len: int,
        kv_format: str = "sbhd",
        cp_group: Optional[ProcessGroup] = None,
        tp_group: Optional[ProcessGroup] = None,
        tp_cp_group: Optional[ProcessGroup] = None,
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
            linear_cp_layout=getattr(mp, "linear_cp_layout", "zigzag"),
            attention_cp_layout=getattr(mp, "attention_cp_layout", "zigzag"),
            cp_group=cp_group,
            tp_group=tp_group,
            tp_cp_group=tp_cp_group,
        )


class PartitionAdapter:
    """Partition all token-aligned MIMO language inputs with one shared CP layout."""

    def __init__(self, cfg: PartitionConfig):
        self.cfg = cfg

    def partition(
        self,
        embeddings: Optional[torch.Tensor],
        input_ids: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        loss_mask: Optional[torch.Tensor],
        mtp_input_mask: Optional[torch.Tensor],
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_padded: Optional[torch.Tensor] = None,
        max_seqlen: Optional[torch.Tensor] = None,
    ) -> ContextParallelBatch:
        """Apply CP to all inputs and SP to decoder embeddings.

        Decoder embeddings arrive sequence-first as ``[S, B, H]``. Token-aligned metadata is
        batch-first as ``[B, S, ...]`` except multi-axis position IDs, which arrive as
        ``[rope_dim, B, S]``. Under dual-layout CP, the boundary view follows the linear layer
        layout and the second view follows the attention/MTP layout.
        """
        self._validate_sequence_length(embeddings, is_packed=cu_seqlens is not None)

        is_multiaxis_position_ids = (
            self.cfg.use_cp and position_ids is not None and position_ids.dim() == 3
        )
        position_metadata = (
            position_ids.movedim(0, -1).contiguous() if is_multiaxis_position_ids else position_ids
        )
        batch = {
            "tokens": input_ids,
            "position_ids": position_metadata,
            "labels": labels,
            "loss_mask": loss_mask,
            "mtp_input_mask": mtp_input_mask,
            "decoder_input": (
                embeddings.transpose(0, 1).contiguous()
                if self.cfg.use_cp and embeddings is not None
                else embeddings
            ),
            "cu_seqlens": cu_seqlens,
            "cu_seqlens_padded": cu_seqlens_padded,
            "max_seqlen": max_seqlen,
        }

        additional_layouts = set()
        if self.cfg.use_cp and self.cfg.linear_cp_layout != self.cfg.attention_cp_layout:
            additional_layouts.add(self.cfg.attention_cp_layout)
        cp_batch = get_batches_on_this_cp_rank(
            batch,
            boundary_layout=self.cfg.linear_cp_layout,
            is_hybrid_cp=False,
            cp_group=self.cfg.cp_group,
            additional_layouts=additional_layouts,
            sequence_parallel=self.cfg.seq_parallel,
            tp_group=self.cfg.tp_group,
            tp_cp_group=self.cfg.tp_cp_group,
            tokens_per_sample=self.cfg.max_seq_len,
        )

        if is_multiaxis_position_ids:
            for layout_batch in cp_batch.batches_by_layout.values():
                local_position_ids = layout_batch.get("position_ids")
                if local_position_ids is not None:
                    layout_batch["position_ids"] = local_position_ids.movedim(-1, 0).contiguous()

        boundary_batch = cp_batch.get_batch()
        local_embeddings = boundary_batch.get("decoder_input")
        # The alternate decoder-input view is never consumed: decoder activations are converted
        # between layouts after each layer instead. Drop these references before model forward.
        for layout_batch in cp_batch.batches_by_layout.values():
            layout_batch.pop("decoder_input", None)
        if self.cfg.use_cp and local_embeddings is not None:
            local_embeddings = local_embeddings.transpose(0, 1).contiguous()
        if self.cfg.seq_parallel and local_embeddings is not None:
            local_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(
                local_embeddings, group=self.cfg.tp_group
            )
        boundary_batch["decoder_input"] = local_embeddings
        return cp_batch

    def _validate_sequence_length(
        self, embeddings: Optional[torch.Tensor], is_packed: bool
    ) -> None:
        """Validate dense sequence divisibility and TP-overlap constraints."""
        if embeddings is None:
            return

        shard_factor = None
        if self.cfg.use_cp and self.cfg.seq_parallel:
            shard_factor = get_pg_size(self.cfg.tp_group) * get_pg_size(self.cfg.cp_group) * 2
        elif self.cfg.use_cp:
            shard_factor = get_pg_size(self.cfg.cp_group) * 2
        elif self.cfg.seq_parallel:
            shard_factor = get_pg_size(self.cfg.tp_group)

        if shard_factor is not None and not is_packed:
            assert embeddings.shape[0] % shard_factor == 0, (
                f"Sequence length should be divisible by {shard_factor} "
                "for Sequence/Context parallelism"
            )

            if self.cfg.seq_parallel and self.cfg.tp_comm_overlap:
                assert embeddings.shape[0] == self.cfg.max_seq_len, (
                    "TP Comm overlap requires Vision+Text token length "
                    "== language_max_sequence_length"
                )
