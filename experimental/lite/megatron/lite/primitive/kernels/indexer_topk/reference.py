# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The matched-precision reference selector of the indexer top-k.

The reference selector scores every visible key of a query row with DeepGEMM
``fp8_fp4_mqa_logits`` on the same quantized operands the LiteTopK plugin reads (FP8 queries
and keys from :mod:`.quant`, float32 head weights folded by
:func:`~.quant.fold_indexer_weights`), then keeps the ``topk`` best keys of the row with the
exact-tie top-k (score descending, lower key id first on equal float32 scores) or, without it,
the cuDNN frontend radix top-k (equal scores in unspecified order). It selects the rows
LiteTopK does not cover and recomputes failed LiteTopK rows, and it is the whole selector of
the reference backend. :func:`reference_topk` runs it on one layout.

One selection call batches the rows of all its segments. DeepGEMM scores each row against its
own key window ``[ks, ke)`` (the keys of its sequence it sees), and one top-k call selects each
chunk of rows. A chunk of rows of one sequence is scored against a view of that sequence's keys
(windows from column 0, plain score rows, the kernel's faster mode); a chunk that spans sequences
gets row-relative scores, so rows of different sequences still share a score call.
A chunk holds as many rows as the float32 scores of the call's widest row fit in a byte budget:
whole waves of the score kernel (:func:`plan_score_rows`) or an explicit row count, and at most
32768 rows. Every row's scores are independent of the chunking, and so is its selection.

DeepGEMM builds its score kernels for some head counts only. Other head counts are padded with
zero query heads of zero weight, which add exact zeros to every score.
"""

from __future__ import annotations

import bisect
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from megatron.lite.primitive.kernels.indexer_topk.config import (
    REFERENCE_BUDGET_BYTES,
    TOPK_ROWS_PER_CALL_LIMIT,
    ExactTopKConfig,
    IndexerTopKConfigError,
    IndexerTopKFormat,
    IndexerTopKRuntimeError,
)
from megatron.lite.primitive.kernels.indexer_topk.layout import QueryLayout, QuerySegment
from megatron.lite.primitive.kernels.indexer_topk.order import compact_valid_topk_, sort_topk_rows_
from megatron.lite.primitive.kernels.indexer_topk.plugins.loader import load_exact_topk
from megatron.lite.primitive.kernels.indexer_topk.quant import (
    fold_indexer_weights,
    quantize_indexer_fp8_rows,
)

__all__ = [
    "QuantizedKeys",
    "ReferenceSelector",
    "TopKKernel",
    "plan_score_rows",
    "quantize_keys",
    "quantize_queries",
    "reference_topk",
    "score_kernel_heads",
    "topk_kernel",
]

# A top-k kernel maps (scores, lengths, top_k) to the top_k best of the first lengths[r] columns
# of every float32 score row r, as int32 [rows, top_k] column ids with -1 for missing ones.
TopKKernel = Callable[[Tensor, Tensor, int], Tensor]

_FORMATS = ("fp8",)
# A score kernel block covers 128 (query row, head) pairs: 128 // heads rows per SM.
_SCORE_BLOCK_PAIRS = 128
_MAX_SCORE_HEADS = 128
# Keys are quantized this many rows at a time, to bound the float32 temporaries.
_KEY_QUANTIZE_ROWS = 65536
# The score kernel maps its operands with descriptors that need 16-byte aligned addresses. A view
# of the keys of a later sequence can start at any key: its key rows (64 or 128 bytes each) stay
# aligned, but its scales (4 bytes per key) may not, and are then copied to an aligned buffer.
_KERNEL_OPERAND_ALIGNMENT = 16
_HEAD_SUPPORT: dict[tuple[str, int, int, tuple[int, int]], bool] = {}


def plan_score_rows(keys: int, *, kernel_heads: int, num_sms: int, budget_bytes: int) -> int:
    """Return the query rows of one reference scoring call.

    The float32 scores of a call, ``rows * keys * 4`` bytes, must fit ``budget_bytes``. The rows
    are a whole number of score kernel waves, ``(128 // kernel_heads) * num_sms`` rows, when at
    least one wave fits, and otherwise the largest multiple of four that fits (at least four).
    A host computation.

    Args:
        keys: Keys scored per row: the widest row of the call.
        kernel_heads: Heads the score kernel runs with.
        num_sms: Streaming multiprocessors of the device.
        budget_bytes: Byte budget of the float32 scores of one call.

    Returns:
        The query rows per scoring call.

    Raises:
        ValueError: If an argument is not a positive integer.
    """
    for name, value in (
        ("keys", keys),
        ("kernel_heads", kernel_heads),
        ("num_sms", num_sms),
        ("budget_bytes", budget_bytes),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    maximum = _budget_rows(keys, budget_bytes)
    wave = max(1, _SCORE_BLOCK_PAIRS // kernel_heads) * num_sms
    return maximum // wave * wave if maximum >= wave else maximum


def _budget_rows(keys: int, budget_bytes: int) -> int:
    return max(4, budget_bytes // (4 * keys) // 4 * 4)


@dataclass(frozen=True)
class QuantizedKeys:
    """Indexer keys in a score kernel operand format.

    Attributes:
        fmt: ``fp8``.
        data: ``float8_e4m3fn`` ``[N, D]``.
        scale: float32 ``[N]`` row scales.
    """

    fmt: IndexerTopKFormat
    data: Tensor
    scale: Tensor


def quantize_keys(k: Tensor, fmt: IndexerTopKFormat, *, rows: int | None = None) -> QuantizedKeys:
    """Quantize the first ``rows`` keys to a score kernel operand format.

    Keys are quantized row by row, so the result equals a quantization of all keys at once.

    Args:
        k: Indexer keys ``[N, D]``.
        fmt: ``fp8`` (:func:`~.quant.quantize_indexer_fp8_rows`).
        rows: Keys to quantize, from the first; defaults to all.

    Returns:
        The quantized keys.

    Raises:
        ValueError: If ``fmt``, ``k`` or ``rows`` is invalid.
    """
    if fmt not in _FORMATS:
        raise ValueError(f"fmt must be one of {_FORMATS}, got {fmt!r}")
    if k.ndim != 2:
        raise ValueError(f"expected keys [N, D], got {tuple(k.shape)}")
    count = k.shape[0] if rows is None else rows
    if type(count) is not int or not 0 <= count <= k.shape[0]:
        raise ValueError(f"rows must be an integer in [0, {k.shape[0]}], got {rows!r}")
    data = torch.empty((count, k.shape[1]), dtype=torch.float8_e4m3fn, device=k.device)
    scale = torch.empty((count,), dtype=torch.float32, device=k.device)
    for start in range(0, count, _KEY_QUANTIZE_ROWS):
        end = min(start + _KEY_QUANTIZE_ROWS, count)
        chunk_data, chunk_scale = quantize_indexer_fp8_rows(k[start:end])
        data[start:end].copy_(chunk_data)
        scale[start:end].copy_(chunk_scale)
    return QuantizedKeys(fmt=fmt, data=data, scale=scale)


def _pad_heads(tensor: Tensor, heads: int, value: int | float) -> Tensor:
    """Append ``heads - tensor.shape[1]`` heads filled with ``value`` along dimension 1."""
    missing = heads - tensor.shape[1]
    if missing == 0:
        return tensor
    storage = tensor.view(torch.uint8) if tensor.dtype == torch.float8_e4m3fn else tensor
    filler = storage.new_full((storage.shape[0], missing, *storage.shape[2:]), value)
    padded = torch.cat((storage, filler), dim=1)
    return padded.view(tensor.dtype) if storage is not tensor else padded


def quantize_queries(
    q: Tensor, weights: Tensor, fmt: IndexerTopKFormat, *, softmax_scale: float, kernel_heads: int
) -> tuple[Tensor, Tensor | None, Tensor]:
    """Quantize query rows and fold their head weights into score kernel operands.

    Every selector of a call builds its query operands with this function, so the reference
    selector and the LiteTopK plugin score identical bytes. Rows are quantized independently:
    the operands of a row do not depend on which rows are quantized with it.

    Args:
        q: Indexer queries ``[n, H, D]``.
        weights: Per-head weights ``[n, H]``, not yet multiplied by ``softmax_scale``.
        fmt: ``fp8``.
        softmax_scale: Positive score scale folded into the weights.
        kernel_heads: Heads of the score kernel, at least H; the missing heads are appended as
            zero queries with zero weights, which add exact zeros to a score.

    Returns:
        ``(data, scales, weights)``: ``float8_e4m3fn`` ``[n, kernel_heads, D]``, no scales (None)
        and float32 weights with the query row scales folded in.

    Raises:
        ValueError: If ``fmt`` is not a supported operand format.
    """
    if fmt not in _FORMATS:
        raise ValueError(f"fmt must be one of {_FORMATS}, got {fmt!r}")
    data, q_scale = quantize_indexer_fp8_rows(q)
    folded = fold_indexer_weights(weights, softmax_scale=softmax_scale, q_scale=q_scale)
    return _pad_heads(data, kernel_heads, 0), None, _pad_heads(folded, kernel_heads, 0.0)


def _mqa_logits(
    q: tuple[Tensor, Tensor | None],
    kv: tuple[Tensor, Tensor],
    weights: Tensor,
    ks: Tensor,
    ke: Tensor,
    *,
    max_seqlen_k: int,
) -> Tensor:
    """float32 scores of DeepGEMM ``fp8_fp4_mqa_logits``; columns outside a window are not written.

    With ``max_seqlen_k == 0`` the scores are ``[rows, keys]`` and column ``j`` of row ``r`` holds
    the score of key ``j`` for ``ks[r] <= j < ke[r]``. With a positive ``max_seqlen_k`` they are
    row-relative, ``[rows, max_seqlen_k]``: column ``j`` holds the score of key ``ks[r] + j`` for
    ``j < ke[r] - ks[r]`` (a slower mode of the kernel).
    """
    try:
        import deep_gemm
    except ImportError as exc:
        raise IndexerTopKRuntimeError(
            "the matched-precision indexer top-k reference selector needs DeepGEMM "
            "(deep_gemm.fp8_fp4_mqa_logits); install it or keep indexer_topk.backend='default'"
        ) from exc
    return deep_gemm.fp8_fp4_mqa_logits(
        q, kv, weights, ks, ke, clean_logits=False, max_seqlen_k=max_seqlen_k
    )


def _score_kernel_accepts(fmt: str, heads: int, head_dim: int, device: torch.device) -> bool:
    key = (fmt, heads, head_dim, torch.cuda.get_device_capability(device))
    supported = _HEAD_SUPPORT.get(key)
    if supported is not None:
        return supported
    rows = max(1, _SCORE_BLOCK_PAIRS // heads)
    queries = torch.zeros((rows, heads, head_dim), dtype=torch.bfloat16, device=device)
    keys = quantize_keys(torch.zeros((256, head_dim), dtype=torch.bfloat16, device=device), fmt)
    weights = torch.zeros((rows, heads), dtype=torch.float32, device=device)
    q = (quantize_indexer_fp8_rows(queries)[0], None)
    starts = torch.zeros((rows,), dtype=torch.int32, device=device)
    ends = torch.full((rows,), 256, dtype=torch.int32, device=device)
    try:
        _mqa_logits(q, (keys.data, keys.scale), weights, starts, ends, max_seqlen_k=256)
        supported = True
    except IndexerTopKRuntimeError:
        raise
    except RuntimeError as exc:
        # DeepGEMM rejects an unsupported head count with a host assertion on num_heads.
        if "num_heads" not in str(exc):
            raise IndexerTopKRuntimeError(
                f"probing the head counts of DeepGEMM fp8_fp4_mqa_logits ({fmt}, {heads} heads, "
                f"head_dim {head_dim}) failed: {exc}"
            ) from exc
        supported = False
    _HEAD_SUPPORT[key] = supported
    return supported


def score_kernel_heads(
    num_heads: int, *, fmt: IndexerTopKFormat, head_dim: int, device: torch.device
) -> int:
    """Return the head count the reference score kernel runs ``num_heads`` indexer heads with.

    Which head counts DeepGEMM builds score kernels for depends on its version, so the answer
    is probed once per format, head count and device architecture (a tiny score call). It is
    ``num_heads`` when supported, else the smallest supported multiple of four above it (the
    queries are then padded with zero heads).

    Args:
        num_heads: Indexer heads.
        fmt: Operand format.
        head_dim: Indexer head dimension.
        device: CUDA device.

    Returns:
        The score kernel head count, at least ``num_heads``.

    Raises:
        IndexerTopKConfigError: If no head count from ``num_heads`` to 128 is supported.
        IndexerTopKRuntimeError: If DeepGEMM is missing or the probe fails for another reason.
    """
    if _score_kernel_accepts(fmt, num_heads, head_dim, device):
        return num_heads
    for heads in range(num_heads - num_heads % 4 + 4, _MAX_SCORE_HEADS + 1, 4):
        if _score_kernel_accepts(fmt, heads, head_dim, device):
            return heads
    raise IndexerTopKConfigError(
        f"DeepGEMM fp8_fp4_mqa_logits supports no head count from {num_heads} to "
        f"{_MAX_SCORE_HEADS} for {fmt} operands with head_dim {head_dim}; the indexer top-k "
        f"reference selector cannot score {num_heads} heads"
    )


def _cudnn_dsa_namespace():
    try:
        from cudnn import DSA
    except ImportError:
        try:
            from cudnn.deepseek_sparse_attention import DSA
        except ImportError as exc:
            raise IndexerTopKRuntimeError(
                "the indexer top-k reference selector without an exact_topk package needs the "
                "cuDNN frontend DeepSeek sparse attention namespace (cudnn.DSA) for its radix "
                "top-k"
            ) from exc
    return DSA


def topk_kernel(exact_topk: ExactTopKConfig | None) -> TopKKernel:
    """Return the top-k kernel of the reference selector.

    Args:
        exact_topk: The exact-tie top-k package (score descending, lower column id first on
            equal scores); None selects the cuDNN frontend radix top-k, which leaves the order
            of equal scores unspecified.

    Returns:
        The kernel.

    Raises:
        IndexerTopKPluginError: If the exact-tie package fails to load.
        IndexerTopKRuntimeError: If the cuDNN frontend radix top-k is unavailable.
    """
    if exact_topk is not None:
        selector = load_exact_topk(exact_topk)

        def exact(scores: Tensor, lengths: Tensor, top_k: int) -> Tensor:
            return selector(scores, lengths, top_k=top_k)[0]

        return exact
    namespace = _cudnn_dsa_namespace()

    def radix(scores: Tensor, lengths: Tensor, top_k: int) -> Tensor:
        result = namespace.indexer_top_k_wrapper(
            scores.contiguous(), lengths, top_k=top_k, next_n=1, return_val=False
        )
        return result["indices"].to(torch.int32)

    return radix


_Piece = tuple[QuerySegment, int, int]


def _chunks(pieces: list[_Piece], rows: int) -> Iterator[list[_Piece]]:
    """Split consecutive row pieces into chunks of ``rows`` rows (the last one shorter)."""
    chunk: list[_Piece] = []
    filled = 0
    for segment, start, end in pieces:
        while start < end:
            take = min(end - start, rows - filled)
            chunk.append((segment, start, start + take))
            filled += take
            start += take
            if filled == rows:
                yield chunk
                chunk, filled = [], 0
    if chunk:
        yield chunk


def _gather(tensor: Tensor, chunk: list[_Piece]) -> Tensor:
    if len(chunk) == 1:
        return tensor[chunk[0][1] : chunk[0][2]]
    return torch.cat([tensor[start:end] for _segment, start, end in chunk])


def _aligned(tensor: Tensor) -> bool:
    return tensor.data_ptr() % _KERNEL_OPERAND_ALIGNMENT == 0


def _plain_key_view(
    kv: tuple[Tensor, Tensor], first: int, width: int
) -> tuple[Tensor, Tensor] | None:
    """The keys ``[first, first + width)`` as aligned score kernel operands, or None.

    The key rows of the view are used in place. Scales that do not start at an aligned address
    (a view that starts at a key that is not a multiple of four) are copied to a new buffer,
    ``width * 4`` bytes; the kernel reads the same values, so the scores are unchanged. None
    when the view is empty or its key rows are not aligned: the chunk is then scored in the
    row-relative mode.
    """
    data, scale = (tensor[first : first + width] for tensor in kv)
    if data.shape[0] == 0 or not _aligned(data):
        return None
    if not _aligned(scale):
        scale = scale.clone()
        if not _aligned(scale):
            return None
    return data, scale


@dataclass(frozen=True)
class ReferenceSelector:
    """The matched-precision reference selector with fixed kernels and chunking.

    Attributes:
        fmt: Operand format: ``fp8``.
        topk_kernel: The top-k kernel (:func:`topk_kernel`).
        kernel_heads: Heads the score kernel runs with; queries with fewer heads are padded with
            zero heads (see :func:`score_kernel_heads`).
        budget_bytes: Byte budget of the float32 scores of one scoring call.
        rows_per_call: Rows per scoring call, bounded by the budget; None plans whole score
            kernel waves within it (:func:`plan_score_rows`).
        num_sms: Streaming multiprocessors of the device.
    """

    fmt: IndexerTopKFormat
    topk_kernel: TopKKernel
    kernel_heads: int
    budget_bytes: int
    rows_per_call: int | None
    num_sms: int

    def __post_init__(self) -> None:
        if self.fmt not in _FORMATS:
            raise ValueError(f"fmt must be one of {_FORMATS}, got {self.fmt!r}")
        for name in ("kernel_heads", "budget_bytes", "num_sms"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.rows_per_call is not None and (
            type(self.rows_per_call) is not int or self.rows_per_call < 1
        ):
            raise ValueError(
                f"rows_per_call must be a positive integer or None, got {self.rows_per_call!r}"
            )

    def rows_per_scoring_call(self, keys: int) -> int:
        """Return the rows of one scoring call whose widest row scores ``keys`` keys."""
        if self.rows_per_call is None:
            rows = plan_score_rows(
                keys,
                kernel_heads=self.kernel_heads,
                num_sms=self.num_sms,
                budget_bytes=self.budget_bytes,
            )
        else:
            rows = min(self.rows_per_call, _budget_rows(keys, self.budget_bytes))
        return min(rows, TOPK_ROWS_PER_CALL_LIMIT)

    def select(
        self,
        q: Tensor,
        weights: Tensor,
        keys: QuantizedKeys,
        *,
        layout: QueryLayout,
        row_ranges: Sequence[tuple[int, int]],
        topk: int,
        softmax_scale: float,
        out: Tensor,
    ) -> int:
        """Select the top-k keys of the given local rows into ``out``.

        Writes the rows of ``row_ranges`` only: each row's ids (plus its segment's
        ``index_base``) at the front in the top-k kernel's order and -1 after them. Rows outside
        every segment get only -1. Nothing is synchronized with the host.

        Args:
            q: Indexer queries ``[layout.rows, H, D]``.
            weights: Per-head weights ``[layout.rows, H]``, before ``softmax_scale``.
            keys: The quantized keys, covering every key the rows see.
            layout: The layout of the rows.
            row_ranges: Ordered, disjoint local row ranges ``(start, end)`` to select.
            topk: Keys selected per row.
            softmax_scale: Positive score scale folded into the weights.
            out: int32 ``[layout.rows, topk]`` destination.

        Returns:
            The number of scoring calls (one score kernel and one top-k kernel call each).

        Raises:
            ValueError: If shapes, formats or ranges are inconsistent.
        """
        rows = layout.rows
        if (
            q.ndim != 3
            or q.shape[0] != rows
            or weights.shape != q.shape[:2]
            or keys.fmt != self.fmt
            or keys.data.ndim != 2
            or q.shape[1] > self.kernel_heads
            or out.shape != (rows, topk)
            or out.dtype != torch.int32
        ):
            raise ValueError(
                f"expected q [{rows}, H<={self.kernel_heads}, D], weights [{rows}, H], "
                f"{self.fmt} keys and int32 out [{rows}, {topk}]; got "
                f"q {tuple(q.shape)}, weights {tuple(weights.shape)}, {keys.fmt} keys, "
                f"out {out.dtype} {tuple(out.shape)}"
            )
        pieces = self._pieces(layout, row_ranges, out)
        if not pieces:
            return 0
        width = max(layout.visible_keys(segment, end - 1) for segment, _start, end in pieces)
        if width == 0:
            for _segment, start, end in pieces:
                out[start:end].fill_(-1)
            return 0
        key_end = max(
            segment.key_start + layout.visible_keys(segment, end - 1)
            for segment, _start, end in pieces
        )
        if key_end > keys.data.shape[0]:
            raise ValueError(f"the rows see {key_end} keys; only {keys.data.shape[0]} are given")
        kv = (keys.data[:key_end], keys.scale[:key_end])
        # The keys every row sees, for all pieces at once: a chunk reads a slice of them.
        visible = []
        for segment, start, end in pieces:
            first = segment.position + start - segment.row_start + 1
            seen = torch.arange(first, first + end - start, dtype=torch.int64, device=q.device)
            visible.append(seen.clamp_(max=segment.key_count).to(torch.int32))
        windows = visible[0] if len(visible) == 1 else torch.cat(visible)
        rows_per_call = self.rows_per_scoring_call(width)
        zeros = torch.zeros(
            min(rows_per_call, windows.shape[0]), dtype=torch.int32, device=q.device
        )
        calls = offset = 0
        for chunk in _chunks(pieces, rows_per_call):
            rows = sum(end - start for _segment, start, end in chunk)
            self._select_chunk(
                q,
                weights,
                kv,
                chunk,
                windows[offset : offset + rows],
                zeros[:rows],
                width,
                topk,
                softmax_scale,
                out,
            )
            offset += rows
            calls += 1
        return calls

    @staticmethod
    def _pieces(
        layout: QueryLayout, row_ranges: Sequence[tuple[int, int]], out: Tensor
    ) -> list[_Piece]:
        """Split row ranges by segment; rows outside every segment are written as empty."""
        segments = layout.segments
        segment_starts = [segment.row_start for segment in segments]
        pieces: list[_Piece] = []
        previous_end = 0
        for start, end in row_ranges:
            if not previous_end <= start <= end <= layout.rows:
                raise ValueError(
                    f"row ranges must be ordered, disjoint and within [0, {layout.rows}); "
                    f"got {list(row_ranges)}"
                )
            previous_end = end
            cursor = start
            index = max(0, bisect.bisect_right(segment_starts, start) - 1)
            while index < len(segments) and segments[index].row_start < end:
                segment = segments[index]
                first, last = max(start, segment.row_start), min(end, segment.row_end)
                if first < last:
                    if cursor < first:
                        out[cursor:first].fill_(-1)
                    pieces.append((segment, first, last))
                    cursor = last
                index += 1
            if cursor < end:
                out[cursor:end].fill_(-1)
        return pieces

    def _select_chunk(
        self,
        q: Tensor,
        weights: Tensor,
        kv: tuple[Tensor, Tensor],
        chunk: list[_Piece],
        windows: Tensor,
        zeros: Tensor,
        width: int,
        topk: int,
        softmax_scale: float,
        out: Tensor,
    ) -> None:
        """Score and select one chunk; ``windows`` holds the keys each of its rows sees."""
        operands = quantize_queries(
            _gather(q, chunk),
            _gather(weights, chunk),
            self.fmt,
            softmax_scale=softmax_scale,
            kernel_heads=self.kernel_heads,
        )
        view = _plain_key_view(kv, chunk[0][0].key_start, width) if len(chunk) == 1 else None
        if view is not None:
            # Rows of one sequence: score them against a view of its keys. Their windows then
            # start at column 0, so the kernel writes plain score rows, its faster mode.
            scores = _mqa_logits(operands[:2], view, operands[2], zeros, windows, max_seqlen_k=0)
        else:
            starts = torch.cat(
                [zeros.new_full((end - start,), segment.key_start) for segment, start, end in chunk]
            )
            scores = _mqa_logits(
                operands[:2], kv, operands[2], starts, starts + windows, max_seqlen_k=width
            )
        selected = self.topk_kernel(scores, windows, min(topk, scores.shape[1]))
        del scores

        device = out.device
        if len(chunk) == 1:
            segment, start, end = chunk[0]
            target = out[start:end]
            compact_valid_topk_(selected, windows, target)
            if segment.index_base:
                target.copy_(torch.where(target >= 0, target + segment.index_base, target))
            return
        staged = torch.empty((selected.shape[0], topk), dtype=torch.int32, device=device)
        compact_valid_topk_(selected, windows, staged)
        offset = 0
        for segment, start, end in chunk:
            rows = staged[offset : offset + end - start]
            if segment.index_base:
                rows = torch.where(rows >= 0, rows + segment.index_base, rows)
            out[start:end].copy_(rows)
            offset += end - start


def reference_topk(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    *,
    layout: QueryLayout,
    topk: int,
    softmax_scale: float,
    fmt: IndexerTopKFormat,
    exact_topk: ExactTopKConfig | None = None,
    budget_bytes: int | None = None,
    rows_per_call: int | None = None,
    out: Tensor | None = None,
) -> Tensor:
    """Select the top-k keys of every query row with the matched-precision reference selector.

    Quantizes the queries and keys to ``fmt``, folds ``softmax_scale`` and the query scales into
    float32 weights, scores every visible key with DeepGEMM ``fp8_fp4_mqa_logits``
    and keeps the ``topk`` best keys of every row with the exact-tie top-k (score descending,
    lower key id first on equal scores) or, without ``exact_topk``, the cuDNN frontend radix
    top-k (equal scores in unspecified order). The indexer top-k bindings run the same selector
    on the rows LiteTopK does not cover; with ``exact_topk`` every row's set is determined by
    the operands.

    Args:
        q: CUDA indexer queries ``[layout.rows, H, D]`` (after RoPE).
        k: Indexer keys ``[N, D]``; a segment's keys are
            ``k[key_start : key_start + key_count]``.
        weights: Per-head weights ``[layout.rows, H]``, not yet multiplied by
            ``softmax_scale``.
        layout: The query rows and the keys each one sees.
        topk: Keys selected per row (at most 2048 with the exact-tie top-k).
        softmax_scale: Positive score scale, folded into the weights.
        fmt: ``fp8`` (E4M3 rows with float32 scales).
        exact_topk: The exact-tie top-k package; None uses the cuDNN frontend radix top-k.
        budget_bytes: Byte budget of the float32 scores of one scoring call; defaults to 2 GiB.
        rows_per_call: Rows per scoring call, bounded by the budget; defaults to whole score
            kernel waves within the budget (:func:`plan_score_rows`).
        out: Optional int32 ``[layout.rows, topk]`` destination.

    Returns:
        int32 ``[layout.rows, topk]``: every row's key ids (sequence-local ids plus the
        segment's ``index_base``) ascending and -1 after them; rows outside every segment hold
        only -1.

    Raises:
        ValueError: If the tensors or the layout are inconsistent.
        IndexerTopKConfigError: If DeepGEMM cannot score the head count.
        IndexerTopKRuntimeError: If DeepGEMM or the radix top-k is missing.
        IndexerTopKPluginError: If the exact-tie package fails to load.
    """
    if fmt not in _FORMATS:
        raise ValueError(f"fmt must be one of {_FORMATS}, got {fmt!r}")
    rows = layout.rows
    if (
        q.ndim != 3
        or k.ndim != 2
        or q.shape[0] != rows
        or q.shape[2] != k.shape[1]
        or weights.shape != q.shape[:2]
        or not q.is_cuda
        or k.device != q.device
        or weights.device != q.device
    ):
        raise ValueError(
            f"expected CUDA q [{rows}, H, D], k [N, D] and weights [{rows}, H] on one device; "
            f"got q {tuple(q.shape)} on {q.device}, k {tuple(k.shape)} on {k.device}, "
            f"weights {tuple(weights.shape)} on {weights.device}"
        )
    if type(topk) is not int or topk < 1:
        raise ValueError(f"topk must be a positive integer, got {topk!r}")
    key_rows = max(
        (segment.key_start + segment.key_count for segment in layout.segments), default=0
    )
    if key_rows > k.shape[0]:
        raise ValueError(f"the layout's segments need {key_rows} keys; k has {k.shape[0]}")
    if out is None:
        out = torch.empty((rows, topk), dtype=torch.int32, device=q.device)
    elif out.shape != (rows, topk) or out.dtype != torch.int32 or out.device != q.device:
        raise ValueError(f"out must be int32 [{rows}, {topk}] on {q.device}")
    if not layout.segments:
        return out.fill_(-1)
    selector = ReferenceSelector(
        fmt=fmt,
        topk_kernel=topk_kernel(exact_topk),
        kernel_heads=score_kernel_heads(q.shape[1], fmt=fmt, head_dim=q.shape[2], device=q.device),
        budget_bytes=REFERENCE_BUDGET_BYTES[fmt] if budget_bytes is None else budget_bytes,
        rows_per_call=rows_per_call,
        num_sms=torch.cuda.get_device_properties(q.device).multi_processor_count,
    )
    seen = max(
        segment.key_start + layout.visible_keys(segment, segment.row_end - 1)
        for segment in layout.segments
    )
    keys = quantize_keys(k, fmt, rows=seen)
    selector.select(
        q,
        weights,
        keys,
        layout=layout,
        row_ranges=((0, rows),),
        topk=topk,
        softmax_scale=softmax_scale,
        out=out,
    )
    return sort_topk_rows_(out)
