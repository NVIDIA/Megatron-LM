# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from typing import Any, Optional, Sequence, Tuple

import torch

from . import dsa_flashinfer_plan as plan_lib
from .metadata_base import MetadataBase


class FlashInferDSAMetadata(MetadataBase):
    """Per-step metadata of DSA-over-GQA inference through FlashInfer's DSA kernels
    (``dsa_backend="flashinfer"``; ``DSAMetadata`` serves ``"triton"``).

    Holds the step plan of ``dsa_flashinfer_plan`` (decode-path rows, 128-row prefill tiles, their
    union-core groups, threshold-sample counts) and FlashInfer's 64-key page table, in static
    buffers at their largest sizes. ``update`` builds the plan on the CPU from the step's request
    lengths; ``transfer`` copies it to the GPU and derives the page table from the step's block
    table (once per step, outside CUDA graphs); ``plan`` returns the views a DSA layer reads. A
    step's view sizes depend only on its padded batch, so captured graphs replay on any batch that
    fits them.

    With speculative decoding a decode request adds up to ``decode_tokens = 1 + k`` rows. The MTP
    draft head's forwards, whose lengths exist only on the GPU, stage an all-decode-path plan
    directly on the GPU instead (``stage_decode_rows``).
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
        decode_tokens: int = 1,
    ):
        super().__init__()
        plan_lib.check_plan_rows(max_tokens)
        self.block_size_tokens = block_size_tokens
        self.max_sequence_length = max_sequence_length
        self.query_heads = query_heads
        # Most rows a decode-path request adds: 1 + the speculative tokens it verifies.
        self.decode_tokens = decode_tokens
        device = torch.cuda.current_device()
        tiles = plan_lib.plan_sizes(max_tokens, max_requests, False)["tiles"]
        tiled = tiles * plan_lib.TILE_ROWS
        # A decode-only step puts every row on the decode path: up to `decode_tokens` rows per
        # request (the MTP commit forward stages `decode_tokens - 1`).
        decode_capacity = max_requests * decode_tokens
        self._capacity = {
            "decode_requests": decode_capacity,
            "decode_lengths": decode_capacity,
            "tile_requests": tiles,
            "tile_ordinals": tiles,
            "row_ends": tiled,
            "query_rows": tiled,
            "request_rows": max_requests,
            "sample_ends": max_requests,
            "batch_starts": tiled // plan_lib.union_group(query_heads) + 1,
            "compact_rows": max_tokens,
            "compact_ends": max_tokens,
            "compact_ordinals": max_tokens,
            "tiled_compact": tiled,
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
            plan_lib.page_table_columns(
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
        padded_prefill_requests: Optional[int] = None,
    ) -> None:
        """Build the step's plan on the CPU for requests that add ``query_lengths[i]`` tokens,
        the last at key ``kv_lengths[i] - 1``. ``padded_prefill_requests`` bounds the requests
        that take the prefill tiles (all ``padded_requests`` when not given)."""
        arrays = plan_lib.build_step_plan(
            query_lengths,
            kv_lengths,
            padded_rows=padded_tokens,
            padded_requests=padded_requests,
            decode_only=decode_only,
            heads=self.query_heads,
            padded_prefill_requests=padded_prefill_requests,
            decode_tokens=self.decode_tokens,
        )
        for name, values in arrays.items():
            assert values.size <= self._capacity[name], name
            self._cpu_views[name][: values.size] = torch.from_numpy(values)
            self._lengths[name] = values.size
        self._sizes = plan_lib.plan_sizes(
            padded_tokens, padded_requests, decode_only, padded_prefill_requests, self.decode_tokens
        )
        self._page_rows = padded_requests

    def transfer(self, block_table: torch.Tensor, non_blocking: bool) -> None:
        """Copy the plan to the GPU and fill the page table from the step's GPU block table."""
        self._gpu_buf.copy_(self._cpu_buf, non_blocking=non_blocking)
        plan_lib.fill_page_table(block_table, self.block_size_tokens, self.page_table)

    def stage_decode_rows(
        self,
        row_requests: torch.Tensor,
        row_lengths: torch.Tensor,
        padded_rows: int,
        padded_requests: int,
        block_table: torch.Tensor,
    ) -> None:
        """Stage an all-decode-path plan directly on the GPU, for the MTP draft and CUDA-graphed
        commit forwards, whose lengths exist only on the GPU.

        Row ``r`` reads the first ``row_lengths[r]`` keys of block-table row
        ``row_requests[r]``; rows past ``row_requests.numel()`` (padding) read none. The plan's
        sizes depend only on ``padded_rows`` and ``padded_requests``, so captured graphs replay.
        No host synchronization.

        Args:
            row_requests (Tensor): [n] block-table row of each real row.
            row_lengths (Tensor): [n] causal key count of each real row.
            padded_rows (int): Rows the forward runs, including padding.
            padded_requests (int): Block-table rows the forward reads.
            block_table (Tensor): [>=padded_requests, max_kv_blocks] block ids; the MTP forwards
                mark padding requests' blocks with -1.
        """
        n = row_requests.numel()
        assert n <= padded_rows <= self._capacity["decode_requests"]
        requests = self._gpu_views["decode_requests"]
        lengths = self._gpu_views["decode_lengths"]
        requests[:n].copy_(row_requests, non_blocking=True)
        requests[n:padded_rows].zero_()
        lengths[:n].copy_(row_lengths, non_blocking=True)
        lengths[n:padded_rows].zero_()
        pages = self.page_table[:padded_requests]
        plan_lib.fill_page_table(block_table[:padded_requests], self.block_size_tokens, pages)
        # -1 blocks give negative pages. No row reads them, but keep every page in range.
        pages.clamp_(min=0)
        self._lengths = {name: 0 for name in self._capacity}
        self._lengths["decode_requests"] = padded_rows
        self._lengths["decode_lengths"] = padded_rows
        self._sizes = plan_lib.plan_sizes(padded_rows, padded_requests, decode_only=True)
        self._page_rows = padded_requests

    def snapshot(self) -> Tuple[Any, ...]:
        """Opaque copy of the staged plan, for MTP forwards that run before the step's main
        forward and must leave its plan in place (``restore`` puts it back)."""
        return (
            self._gpu_buf.clone(),
            self.page_table[: self._page_rows].clone(),
            dict(self._lengths),
            dict(self._sizes),
            self._page_rows,
        )

    def restore(self, state: Tuple[Any, ...]) -> None:
        """Put back a plan saved by ``snapshot``."""
        gpu_buf, page_table, lengths, sizes, page_rows = state
        self._gpu_buf.copy_(gpu_buf)
        self.page_table[:page_rows].copy_(page_table)
        self._lengths, self._sizes, self._page_rows = lengths, sizes, page_rows

    def plan(self):
        """The step's ``dsa_flashinfer_plan.StepPlan``; kernel shapes use the configured bound."""
        return plan_lib.StepPlan(
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
