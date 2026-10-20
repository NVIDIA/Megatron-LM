# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optional DeepSelect backend for the non-deterministic DSA indexer top-k.

DeepSelect (https://github.com/deepseek-ai/DeepSelect) selects the top-k of every row in one
launch with no scratch buffer. Two builds are accepted, the first one that imports wins:

* ``ds_tune``: DeepSelect bfa4507 with the GLM-5.2 patches (a ``max_topk`` 2048 tier, config
  ``f_t512_b2_2k_d3`` for FP32).
* ``deep_select``: upstream DeepSelect.

Through :func:`dsa_cudnn_kernels._indexer_top_k_wrapper_chunked` on GB300 (CUDA graph, K=2048,
FP32, GLM-5.2 row lengths; cuDNN frontend 1.31 with its 8 GiB scratch budget), ``ds_tune``
took 5.90 ms against 14.34 ms for cuDNN at the 1M CP64 rank-0 call (16384 x 1M), 6.82 against
13.65 ms at the narrowest 1M CP64 rank (16384 x 532480) and 0.43 against 0.57 ms at 128K
(4096 rows). Narrow rows favour cuDNN, and more so the more rows a call has. With CP-balanced
row lengths, ``ds_tune`` against cuDNN is 0.94-0.96x at 65536 columns and 2048 rows, 0.98-1.01x
at 65536 and 0.91-0.94x at 81920 columns with 4096 rows (the GLM-5.2 128K call at its realized
CP32), but 1.08-1.09x at 65536 and 0.94-0.96x at 81920 columns with 16384 rows, and it is
slower everywhere below 65536 (1.01-1.7x, e.g. 1.26-1.50x at 16K). Only the edges of the
65536..81920 band were measured; interpolating both times linearly puts the 4096-row ratio at
or below 1.0 from about 67K columns (the narrowest 128K CP32 rank is 67584 wide). Upstream
``deep_select`` took 7.94 ms at 1M and 9.49 against 13.67 ms at 16384 x 532480, but 0.598
against 0.564 ms at 128K. So each build is used only from :func:`min_score_width` columns
(:data:`MIN_SCORE_WIDTH`, :data:`MANY_ROWS_MIN_SCORE_WIDTH`).

:func:`deepselect_topk` keeps the cuDNN ``indexer_top_k_wrapper`` output contract of
:func:`dsa_cudnn_kernels._indexer_top_k_wrapper_chunked` in non-deterministic mode: int32
``[rows, K]`` indices in unspecified slot order, the selected FP32 scores (or ``None``), rows
no longer than K gathered at the front of the row and padded with index ``-1`` and value
``-inf``. Only the choice among equal scores at the cutoff may differ from cuDNN's, which is
itself unspecified there. DeepSelect is deterministic run to run and does not depend on the
other rows of the batch, but it does not keep the lowest-index ties, so deterministic mode
never uses it.

NaN: DeepSelect checks rows longer than K for NaN. Instead of trapping
(``abort_when_nan_found=False``) it writes ``NAN_ROW_SENTINEL`` into slot 0 of such a row and
leaves the rest of the row unspecified. :func:`repair_nan_rows` rewrites those rows to the
first ``K`` in-row columns with NaN values, so they stay in range and unique. This reaches the
sparse attention, not only the indexer loss: with ``return_topk_scores=True``,
``_indexer_topk_bshd`` keeps only in-bounds, finite selections, so a flagged row ends with
``topk_length == 0`` and attends to no key; with ``return_topk_scores=False`` it attends to the
in-bounds part of its first K columns (the first K keys of its packed THD segment). One NaN key
column flags every later query row longer than K that sees it. The
cuDNN radix top-k instead spends one slot on the NaN (a positive NaN ranks above ``+inf``) and
keeps K - 1 real keys. Both are garbage-in rows (NaN scores already mean a broken step), so
this difference is documented rather than matched.

Requirements, checked by :func:`fallback_reason` without host synchronization: a build imports
and exposes the arguments used here, a CUDA device in :data:`SUPPORTED_SM` (SM 10.3; the builds
also ship sm_100a kernels, :data:`COMPILED_SM`, but those have not run on a GB200 yet), FP32
scores with unit column stride, a 16-byte-aligned start (the TMA descriptor's global address)
and a row stride that is a multiple of the build's input alignment (1024 bytes),
``min_score_width(build, rows) <= width < 2**23``, a compiled config for K (K <= 4096) and K a
multiple of the output alignment (8 int32), so the preallocated outputs are contiguous.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import logging
from typing import Optional

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - depends on the environment
    HAVE_TRITON = False

logger = logging.getLogger(__name__)

TOPK_BACKENDS = ("cudnn", "deepselect")
# Import order: the GLM-5.2 tuned build first, then upstream DeepSelect.
PROVIDER_MODULES = ("ds_tune", "deep_select")
# Index DeepSelect writes into slot 0 of a row containing NaN with abort_when_nan_found=False.
NAN_ROW_SENTINEL = 0x3F3F3F3F
# Narrowest score row per build at which it beat cuDNN (see the module docstring), and per build
# (row limit, width): a call of more than row-limit rows needs that wider row. Every rank of the
# GLM-5.2 128K recipe (realized CP32: 4096 rows of widths 67584..131072) and of the 1M CP64
# recipe (16384 rows of widths 532480..1048576) uses ds_tune; only the 1M ranks reach
# deep_select's.
MIN_SCORE_WIDTH = {"ds_tune": 65536, "deep_select": 524288}
MANY_ROWS_MIN_SCORE_WIDTH = {"ds_tune": (4096, 81920)}
MAX_TOPK = 4096
MAX_SCORE_WIDTH_EXCLUSIVE = 2**23
# Devices the builds ship kernels for (sm_100a, sm_103a), and the ones the backend runs on: only
# GB300 (SM 10.3) has run them. SM 10.0 (GB200) stays on cuDNN, with a warning, until
# test_dsa_deepselect_topk_gpu.py (which runs on every COMPILED_SM device) passes on a GB200.
COMPILED_SM = frozenset({(10, 0), (10, 3)})
SUPPORTED_SM = frozenset({(10, 3)})
# The kernels load the scores through a TMA descriptor, whose global address must be 16-byte
# aligned (the build checks the row stride, not the start).
INPUT_ADDRESS_ALIGNMENT_BYTES = 16
_REQUIRED_PARAMETERS = {
    "ds_tune": (
        "end",
        "cfg_id",
        "sorted_index",
        "out_val",
        "out_idx",
        "idx_oob_fill_value",
        "value_oob_fill_value",
        "abort_when_nan_found",
    ),
    "deep_select": (
        "end",
        "indices_type",
        "sorted_index",
        "output_idx",
        "idx_oob_fill_value",
        "value_oob_fill_value",
        "return_value",
        "abort_when_nan_found",
    ),
}
_REQUIRED_ATTRIBUTES = {
    "ds_tune": (
        "topk",
        "config_for",
        "INPUT_STRIDE_ALIGNMENT_BYTES",
        "OUTPUT_STRIDE_ALIGNMENT_BYTES",
    ),
    "deep_select": ("topk", "get_stride_requirement"),
}
_INDEX_PAD = -1
_VALUE_PAD = float("-inf")
_NAN_REPAIR_BLOCK = 1024

_IMPORT_STATE: dict = {}
_LOGGED_REASONS: set = set()


class _Provider:
    """The imported DeepSelect build behind one calling convention."""

    def __init__(self, kind: str, module):
        self.kind = kind
        self.module = module
        if kind == "ds_tune":
            alignments = (module.INPUT_STRIDE_ALIGNMENT_BYTES, module.OUTPUT_STRIDE_ALIGNMENT_BYTES)
        else:
            alignments = module.get_stride_requirement()
        self.input_align, self.output_align = (int(value) for value in alignments)
        self._configs: dict = {}

    @property
    def description(self) -> str:
        """Package, version and location, for the backend log line."""
        return (
            f"{self.kind} {getattr(self.module, '__version__', 'unknown')} from "
            f"{getattr(self.module, '__file__', '?')}"
        )

    def config_id(self, dtype: torch.dtype, topk_k: int) -> Optional[int]:
        """The compiled config id for ``(dtype, topk_k)``, or ``None`` if there is none."""
        if self.kind != "ds_tune":
            return 0
        key = (dtype, topk_k)
        if key not in self._configs:
            try:
                self._configs[key] = int(self.module.config_for(dtype, topk_k)["id"])
            except (ValueError, KeyError, TypeError):
                self._configs[key] = None
        return self._configs[key]

    def topk(
        self, scores: Tensor, topk_k: int, lengths: Tensor, indices: Tensor, return_values: bool
    ) -> Optional[Tensor]:
        """Write the selection into ``indices``; return the selected scores if asked."""
        common = dict(
            sorted_index=False,
            idx_oob_fill_value=_INDEX_PAD,
            value_oob_fill_value=_VALUE_PAD,
            abort_when_nan_found=False,
        )
        if self.kind == "ds_tune":
            # Every ds_tune config writes values; they are dropped when not requested.
            values = torch.empty((scores.size(0), topk_k), dtype=scores.dtype, device=scores.device)
            self.module.topk(
                scores,
                topk_k,
                lengths,
                self.config_id(scores.dtype, topk_k),
                out_val=values,
                out_idx=indices,
                **common,
            )
            return values if return_values else None
        values, _ = self.module.topk(
            scores,
            topk_k,
            end=lengths,
            indices_type=torch.int32,
            output_idx=indices,
            return_value=return_values,
            **common,
        )
        return values if return_values else None


def _missing_api(kind: str, module) -> list:
    missing = [name for name in _REQUIRED_ATTRIBUTES[kind] if not hasattr(module, name)]
    if "topk" in missing:
        return missing
    try:
        names = inspect.signature(module.topk).parameters
    except (TypeError, ValueError):
        return list(_REQUIRED_PARAMETERS[kind])
    return missing + [name for name in _REQUIRED_PARAMETERS[kind] if name not in names]


def _load_provider() -> Optional[_Provider]:
    """Import the first available DeepSelect build once; ``None`` with the reason recorded."""
    if "provider" not in _IMPORT_STATE:
        provider, errors = None, []
        for kind in PROVIDER_MODULES:
            try:
                module = importlib.import_module(kind)
            except (ImportError, OSError, RuntimeError) as exc:
                errors.append(f"cannot import {kind}: {type(exc).__name__}: {exc}")
                continue
            missing = _missing_api(kind, module)
            if missing:
                errors.append(f"{kind} lacks {', '.join(missing)}")
                continue
            provider = _Provider(kind, module)
            break
        _IMPORT_STATE["provider"] = provider
        _IMPORT_STATE["error"] = None if provider else "no DeepSelect build: " + "; ".join(errors)
    return _IMPORT_STATE["provider"]


@functools.lru_cache(maxsize=None)
def _device_capability(device_index: int) -> tuple:
    return tuple(torch.cuda.get_device_capability(device_index))


def _device_sm(device: torch.device) -> Optional[tuple]:
    if device.type != "cuda":
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _device_capability(index)


def _device_supported(device: torch.device) -> bool:
    return _device_sm(device) in SUPPORTED_SM


def _unsupported_device_reason(device: torch.device) -> str:
    sm = _device_sm(device)
    if sm in COMPILED_SM:
        return (
            f"DeepSelect is not validated on SM {sm[0]}.{sm[1]}: its kernels for it have not run "
            f"on a GPU (validated: SM {sorted(SUPPORTED_SM)})"
        )
    return f"DeepSelect needs a CUDA device of SM {sorted(SUPPORTED_SM)}"


def min_score_width(kind: str, n_rows: int) -> int:
    """Narrowest score row from which build ``kind`` is used for a call of ``n_rows`` rows."""
    width = MIN_SCORE_WIDTH[kind]
    many_rows = MANY_ROWS_MIN_SCORE_WIDTH.get(kind)
    if many_rows is not None and n_rows > many_rows[0]:
        width = max(width, many_rows[1])
    return width


def static_fallback_reason(
    *,
    n_rows: int,
    width: int,
    topk_k: int,
    dtype: torch.dtype,
    device: torch.device,
    deterministic: bool,
) -> Optional[str]:
    """Why a top-k of this geometry cannot use DeepSelect, or ``None`` if it can.

    Covers everything but the score buffer's strides, so a caller can size the buffer before
    it exists (:func:`score_row_stride`). The reason strings are stable per cause, so they
    double as the log-once key.
    """
    if deterministic:
        return "deterministic mode keeps the cuDNN tie_break=1 top-k"
    if not 0 < topk_k <= MAX_TOPK:
        return f"top-k must be in [1, {MAX_TOPK}]"
    if width < min(MIN_SCORE_WIDTH.values()):
        return f"score width below {min(MIN_SCORE_WIDTH.values())} (cuDNN is faster there)"
    if width >= MAX_SCORE_WIDTH_EXCLUSIVE:
        return "score width must be below 2**23"
    if topk_k >= width:
        return "top-k covers the whole row"
    if n_rows <= 0:
        return "no rows"
    if dtype != torch.float32:
        return "scores must be FP32"
    if not _device_supported(device):
        return _unsupported_device_reason(device)
    provider = _load_provider()
    if provider is None:
        return _IMPORT_STATE["error"]
    if width < MIN_SCORE_WIDTH[provider.kind]:
        return (
            f"score width below {MIN_SCORE_WIDTH[provider.kind]} for {provider.kind} "
            "(cuDNN is faster there)"
        )
    if width < min_score_width(provider.kind, n_rows):
        row_limit, row_width = MANY_ROWS_MIN_SCORE_WIDTH[provider.kind]
        return (
            f"score width below {row_width} for {provider.kind} with more than {row_limit} rows "
            "(cuDNN is faster there)"
        )
    if provider.config_id(dtype, topk_k) is None:
        return f"{provider.kind} has no compiled config for top-k {topk_k}"
    if (topk_k * torch.iinfo(torch.int32).bits // 8) % provider.output_align:
        return f"top-k must be a multiple of {provider.output_align // 4} for contiguous outputs"
    return None


def fallback_reason(
    scores: Tensor, seq_lens: Tensor, topk_k: int, deterministic: bool
) -> Optional[str]:
    """:func:`static_fallback_reason` plus the stride, address and length checks of a call."""
    if scores.ndim != 2:
        return "scores must be 2-D"
    n_rows, width = scores.shape
    reason = static_fallback_reason(
        n_rows=n_rows,
        width=width,
        topk_k=topk_k,
        dtype=scores.dtype,
        device=scores.device,
        deterministic=deterministic,
    )
    if reason is not None:
        return reason
    input_align = _load_provider().input_align
    if scores.stride(1) != 1 or (scores.stride(0) * scores.element_size()) % input_align:
        return f"score row stride must be a multiple of {input_align} bytes with unit columns"
    if scores.data_ptr() % INPUT_ADDRESS_ALIGNMENT_BYTES:
        return f"score buffer must start at a {INPUT_ADDRESS_ALIGNMENT_BYTES}-byte boundary"
    if seq_lens.ndim != 1 or seq_lens.numel() != n_rows or seq_lens.device != scores.device:
        return "sequence lengths must be one per row on the score device"
    return None


def score_row_stride(width: int) -> int:
    """Smallest FP32 row stride (elements) of at least ``width`` that DeepSelect accepts.

    Call only after :func:`static_fallback_reason` returned ``None`` (a build is loaded).
    """
    align = max(1, _load_provider().input_align // (torch.finfo(torch.float32).bits // 8))
    return -(-width // align) * align


def _is_unavailable(reason: str) -> bool:
    return reason.startswith(("no DeepSelect build", "DeepSelect needs", "DeepSelect is not")) or (
        "has no compiled config" in reason
    )


def log_topk_backend_once(requested: str, reason: Optional[str]) -> None:
    """Log once per outcome which backend a ``dsa_indexer_topk_backend='deepselect'`` run uses."""
    key = (requested, reason)
    if key in _LOGGED_REASONS:
        return
    _LOGGED_REASONS.add(key)
    if reason is None:
        logger.info("DSA indexer top-k uses DeepSelect (%s).", _load_provider().description)
    elif _is_unavailable(reason):
        logger.warning(
            "dsa_indexer_topk_backend='deepselect' is unavailable, the cuDNN top-k runs: %s", reason
        )
    else:
        logger.info("DSA indexer top-k uses cuDNN for some calls: %s.", reason)


if HAVE_TRITON:

    @triton.jit
    def _repair_nan_rows_kernel(
        indices_ptr,
        indices_row_stride,
        values_ptr,
        values_row_stride,
        lengths_ptr,
        K: tl.constexpr,
        HAS_VALUES: tl.constexpr,
        SENTINEL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """One program per row; rows without the sentinel only read their first index."""
        row = tl.program_id(0).to(tl.int64)
        row_indices = indices_ptr + row * indices_row_stride
        if tl.load(row_indices) == SENTINEL:
            k_eff = tl.minimum(tl.load(lengths_ptr + row), K)
            for start in tl.static_range(0, K, BLOCK):
                columns = start + tl.arange(0, BLOCK)
                in_k = columns < K
                tl.store(row_indices + columns, tl.where(columns < k_eff, columns, -1), mask=in_k)
                if HAS_VALUES:
                    nan = tl.full((BLOCK,), float("nan"), tl.float32)
                    tl.store(values_ptr + row * values_row_stride + columns, nan, mask=in_k)


def repair_nan_rows(indices: Tensor, values: Optional[Tensor], lengths: Tensor) -> None:
    """Rewrite rows flagged with :data:`NAN_ROW_SENTINEL` in place (no host synchronization).

    A flagged row becomes columns ``0 .. min(len, K) - 1`` (index ``-1`` after that) with NaN
    values. Rows without the flag are unchanged. On CUDA a Triton kernel reads only slot 0 of
    unflagged rows; elsewhere (and without Triton) a fixed-shape PyTorch version runs. The
    module docstring says what such a row does to the sparse attention.
    """
    if indices.size(0) == 0:
        return
    if HAVE_TRITON and indices.is_cuda:
        _repair_nan_rows_triton(indices, values, lengths)
    else:
        _repair_nan_rows_torch(indices, values, lengths)


def _repair_nan_rows_triton(indices: Tensor, values: Optional[Tensor], lengths: Tensor) -> None:
    n_rows, topk_k = indices.shape
    _repair_nan_rows_kernel[(n_rows,)](
        indices,
        indices.stride(0),
        values if values is not None else indices,
        values.stride(0) if values is not None else 0,
        lengths,
        K=topk_k,
        HAS_VALUES=values is not None,
        SENTINEL=NAN_ROW_SENTINEL,
        BLOCK=min(_NAN_REPAIR_BLOCK, triton.next_power_of_2(topk_k)),
    )


def _repair_nan_rows_torch(indices: Tensor, values: Optional[Tensor], lengths: Tensor) -> None:
    n_rows, topk_k = indices.shape
    flagged = indices[:, :1] == NAN_ROW_SENTINEL
    columns = torch.arange(topk_k, dtype=indices.dtype, device=indices.device).view(1, topk_k)
    in_row = columns < lengths.to(indices.dtype).clamp(max=topk_k).view(n_rows, 1)
    replacement = torch.where(in_row, columns, torch.full_like(columns, _INDEX_PAD))
    indices.copy_(torch.where(flagged, replacement, indices))
    if values is not None:
        values.copy_(torch.where(flagged, torch.full_like(values, float("nan")), values))


def deepselect_topk(
    scores: Tensor, seq_lens: Tensor, topk_k: int, return_topk_scores: bool
) -> dict:
    """Top-k of every row of ``scores[:, :seq_lens]`` in one DeepSelect launch.

    The caller must have checked :func:`fallback_reason`. Returns ``{"indices", "values"}``
    with the cuDNN wrapper's non-deterministic contract (see the module docstring). Outputs
    have fixed shapes, nothing synchronizes with the host, and no scratch is allocated, so the
    call can be captured in a CUDA graph.
    """
    n_rows, width = scores.shape
    # DeepSelect requires end <= width; the clamp is a no-op for every in-tree caller.
    lengths = seq_lens.to(dtype=torch.int32).clamp(min=0, max=width).contiguous()
    indices = torch.empty((n_rows, topk_k), dtype=torch.int32, device=scores.device)
    values = _load_provider().topk(scores, topk_k, lengths, indices, return_topk_scores)
    repair_nan_rows(indices, values, lengths)
    return {"indices": indices, "values": values}
