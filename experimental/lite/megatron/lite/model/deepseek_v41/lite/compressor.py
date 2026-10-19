# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DS4.1 KV source selection over the native Core compressor."""
from typing import Any

import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant.csa import Compressor
from megatron.lite.model.deepseek_v41.lite.state import KVSourceState


class DS41Compressor(Compressor):
    """Select owner outputs without changing Core's layout or CP collectives.

    Core contract: dev 7b150acfacee9d1126025f6cbe0f49c588975c51,
    Compressor.forward and Compressor._forward_thd_cp. Signature tests protect
    both overrides. The parent remains responsible for every compression op.
    """

    def bind_source(self, state: KVSourceState, layer: int, owner: int) -> None:
        if not 0 <= owner <= layer:
            raise ValueError("KV source must be this layer or an earlier owner")
        self.source_state, self.source_layer, self.source_owner = state, layer, owner

    def _resolve_source(self, x, prefix, mode, max_length, global_start, produce):
        if not hasattr(self, "source_state"):
            raise RuntimeError("DS4.1 compressor has no bound KV source")
        geometry = (
            mode,
            tuple(x.shape),
            x.dtype,
            x.device,
            self.compress_ratio,
            self.head_dim,
            prefix.data_ptr() if prefix is not None else None,
            max_length,
            global_start,
            torch.is_grad_enabled(),
            id(self.pg_collection.cp),
        )
        return self.source_state.resolve(self.source_layer, self.source_owner, geometry, produce)

    def forward(self, x: torch.Tensor, packed_seq_params: PackedSeqParams | None = None) -> Any:
        prefix = (
            None
            if packed_seq_params is None
            else (
                packed_seq_params.cu_seqlens_q_padded
                if packed_seq_params.cu_seqlens_q_padded is not None
                else packed_seq_params.cu_seqlens_q
            )
        )
        parent = super().forward
        return self._resolve_source(
            x,
            prefix,
            "forward",
            None if packed_seq_params is None else packed_seq_params.max_seqlen_q,
            None,
            lambda: parent(x, packed_seq_params),
        )

    def _forward_thd_cp(
        self, x, boundary_hidden, cu_seqlens, layout, max_seqlen_q, pre_compacted_hidden=None
    ):
        parent = super()._forward_thd_cp
        return self._resolve_source(
            x,
            cu_seqlens,
            "thd_cp",
            max_seqlen_q,
            layout.global_start,
            lambda: parent(
                x,
                boundary_hidden,
                cu_seqlens,
                layout,
                max_seqlen_q,
                pre_compacted_hidden=pre_compacted_hidden,
            ),
        )
