# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from typing import Sequence

import torch

from .metadata_base import MetadataBase


class FlashInferDSAMetadata(MetadataBase):
    """Per-step metadata of DSA-over-GQA inference through FlashInfer's DSA kernels
    (``dsa_backend="flashinfer"``; ``DSAMetadata`` serves ``"triton"``).

    Holds the step plan of ``dsa_gqa_flashinfer`` (decode-path rows, 128-row prefill tiles, their
    union-core groups, threshold-sample counts) and FlashInfer's 64-key page table, in static
    buffers at their largest sizes. ``update`` builds the plan on the CPU from the step's request
    lengths; ``transfer`` copies it to the GPU and derives the page table from the step's block
    table (once per step, outside CUDA graphs); ``plan`` returns the views a DSA layer reads. A
    step's view sizes depend only on its padded batch, so captured graphs replay on any batch that
    fits them.
    """

    def __init__(
        self,
        *,
        max_requests: int,
        max_tokens: int,
        max_kv_blocks: int,
        block_size_tokens: int,
        max_sequence_length: int,
        query_heads: int,
        index_topk: int,
    ):
        super().__init__()
        from megatron.core.transformer.experimental_attention_variant import dsa_gqa_flashinfer

        dsa_gqa_flashinfer.require_flashinfer_dsa()
        self._fi = dsa_gqa_flashinfer
        self.block_size_tokens = block_size_tokens
        self.max_sequence_length = max_sequence_length
        self.query_heads = query_heads
        device = torch.cuda.current_device()
        tiles = dsa_gqa_flashinfer.plan_sizes(max_tokens, max_requests, False)["tiles"]
        tiled = tiles * dsa_gqa_flashinfer.TILE_ROWS
        self._capacity = {
            "decode_requests": max_requests,
            "decode_lengths": max_requests,
            "tile_requests": tiles,
            "tile_ordinals": tiles,
            "row_ends": tiled,
            "query_rows": tiled,
            "request_rows": max_requests,
            "sample_ends": max_requests,
            "batch_starts": tiled // dsa_gqa_flashinfer.union_group(query_heads) + 1,
        }
        total = sum(self._capacity.values())
        self._cpu_buf = torch.zeros(total, dtype=torch.int32, pin_memory=True)
        self._gpu_buf = torch.zeros(total, dtype=torch.int32, device=device)
        self._cpu_views, self._gpu_views, offset = {}, {}, 0
        for name, size in self._capacity.items():
            self._cpu_views[name] = self._cpu_buf[offset : offset + size]
            self._gpu_views[name] = self._gpu_buf[offset : offset + size]
            offset += size
        self.page_table = torch.zeros(
            max_requests,
            dsa_gqa_flashinfer.page_table_columns(
                max_kv_blocks, block_size_tokens, max_sequence_length, index_topk
            ),
            dtype=torch.int32,
            device=device,
        )
        self.reset()

    def update(
        self,
        query_lengths: Sequence[int],
        kv_lengths: Sequence[int],
        *,
        padded_tokens: int,
        padded_requests: int,
        decode_only: bool,
    ) -> None:
        """Build the step's plan on the CPU for requests that add ``query_lengths[i]`` tokens,
        the last at key ``kv_lengths[i] - 1``."""
        arrays = self._fi.build_step_plan(
            query_lengths,
            kv_lengths,
            padded_rows=padded_tokens,
            padded_requests=padded_requests,
            decode_only=decode_only,
            heads=self.query_heads,
        )
        for name, values in arrays.items():
            assert values.size <= self._capacity[name], name
            self._cpu_views[name][: values.size] = torch.from_numpy(values)
            self._lengths[name] = values.size
        self._sizes = self._fi.plan_sizes(padded_tokens, padded_requests, decode_only)
        self._page_rows = padded_requests

    def transfer(self, block_table: torch.Tensor, non_blocking: bool) -> None:
        """Copy the plan to the GPU and fill the page table from the step's GPU block table."""
        self._gpu_buf.copy_(self._cpu_buf, non_blocking=non_blocking)
        self._fi.fill_page_table(block_table, self.block_size_tokens, self.page_table)

    def plan(self):
        """The step's ``dsa_gqa_flashinfer.StepPlan``; kernel shapes use the configured bound."""
        return self._fi.StepPlan(
            decode_rows=self._sizes["decode_rows"],
            num_tiles=self._sizes["tiles"],
            max_context=self.max_sequence_length,
            page_table=self.page_table[: self._page_rows],
            **{name: view[: self._lengths[name]] for name, view in self._gpu_views.items()},
        )

    def reset(self) -> None:
        """Empty plan: no rows, tiles or requests."""
        self._lengths = {name: 0 for name in self._capacity}
        self._sizes = {"decode_rows": 0, "tiles": 0}
        self._page_rows = 0
