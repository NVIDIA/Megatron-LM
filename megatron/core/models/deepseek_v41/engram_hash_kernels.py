# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fused Engram n-gram hashing: ``EngramHasher``'s address arithmetic in one Triton kernel.

After the context-parallel halo exchange, ``EngramHasher`` maps every token of a ``[batch, length]``
block to ``layers x (max_ngram_size - 1) x n_heads`` table addresses. The PyTorch arithmetic runs
64 small kernels per call for 4-grams with a token mask and THD sequence ids (token-map gather,
gathers, boundary masks and a select per shift, the int64 products, a XOR and a modulo per n-gram
order, the concatenation and the offsets). This kernel computes the same values for a tile of
tokens and output columns at once:

* ``compressed = token_map[token]`` (negative ids wrap like PyTorch indexing), ``-1`` where the
  token mask is false;
* for shift ``k = 0 .. max_ngram_size - 1`` the history token at ``max(position - k, 0)`` is
  replaced by ``pad_id`` once it is blocked, and blocking accumulates over ``k``: ``position < k``,
  a ``-1`` (masked) history token, or a history token from another sequence id;
* column ``j = (layer, order i, head h)`` takes the XOR of ``token_k * multipliers[layer, k]``
  over ``k <= i``, its floored modulo by ``primes[layer, i - 1, h]`` and ``offsets[layer, j]``.

Every step is the int64 operation PyTorch performs (wrapping multiply, XOR, floored ``%``, add),
so the addresses are bit-identical to the reference for all inputs the reference accepts; the
XOR of the products is order-independent. A token id outside ``[-len(token_map), len(token_map))``
is an indexing error in the reference (IndexError on the host, a device assert on CUDA). By
default the kernel reads nothing for such an id and hashes it as compressed id 0;
``MCORE_ENGRAM_HASH_CHECKS=1`` asserts on the device instead (see ``DEVICE_CHECKS``).

CUDA graphs: all scalar arguments are typed and never specialized, and no pointer is specialized
on alignment, so a hasher configuration and input dtype combination has exactly one Triton
specialization and new lengths, strides or token-map sizes never recompile. Inside CUDA-graph
capture a specialization runs only if it already ran outside capture (training warm-up iterations
do that); otherwise ``fused_engram_hash`` warns once and returns ``None``, and the caller records
the PyTorch arithmetic, which is exact as well, so Triton never compiles or loads a module during
capture. A specialization that fails to compile or launch outside capture also warns once and
keeps the PyTorch arithmetic; a launch failure inside capture raises.
"""

import logging
from typing import Optional

import torch

from megatron.core.env_flags import env_flag

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # Triton ships with the PyTorch container; keep CPU-only imports working.
    HAVE_TRITON = False

logger = logging.getLogger(__name__)

# MCORE_ENGRAM_HASH_CHECKS=1 (read once at import; off by default): before every kernel launch,
# assert on the device that each token id indexes the token map, which the reference's
# ``token_map[token_ids]`` enforces. Two compares, an AND, a reduction and a
# ``torch._assert_async`` (five small kernels) per call, without a host sync and also inside CUDA
# graphs, where a failing check aborts the CUDA context without a Python stack; a debugging aid
# for new tokenizers and data pipelines.
DEVICE_CHECKS = env_flag("MCORE_ENGRAM_HASH_CHECKS", False)
_OUT_OF_RANGE_MESSAGE = (
    "Engram hash: a token id lies outside the Engram token map (tokenizer and token map differ)"
)

_INDEX_DTYPES = (torch.int32, torch.int64)
# Small one-warp programs keep many int64 divisions in flight: at the V4.1 proxy shape (4096
# tokens x 48 columns) on GB300, 2 x 64 tiles take 4.4 us of GPU time, 32 x 64 tiles on four
# warps 14.6 us.
_NUM_WARPS = 1

# Specializations (see ``_specialization_key``) launched outside CUDA-graph capture, those that
# failed to compile or launch there, and the (reason, key) pairs already warned about.
_WARM_KEYS: set = set()
_FAILED_KEYS: set = set()
_WARNED: set = set()
# Calls inside capture that returned None because their specialization was not warm yet.
capture_fallbacks = 0


if HAVE_TRITON:

    @triton.jit(
        do_not_specialize=[
            "token_stride_b",
            "token_stride_t",
            "mask_stride_b",
            "mask_stride_t",
            "sequence_stride_b",
            "sequence_stride_t",
            "map_length",
            "pad_id",
            "rows",
            "start",
        ],
        do_not_specialize_on_alignment=[
            "token_ptr",
            "mask_ptr",
            "sequence_ptr",
            "token_map_ptr",
            "multiplier_ptr",
            "prime_ptr",
            "offset_ptr",
            "out_ptr",
        ],
    )
    def _engram_hash_kernel(
        token_ptr,
        mask_ptr,
        sequence_ptr,
        token_map_ptr,
        multiplier_ptr,
        prime_ptr,
        offset_ptr,
        out_ptr,
        token_stride_b: tl.int64,
        token_stride_t: tl.int64,
        mask_stride_b: tl.int64,
        mask_stride_t: tl.int64,
        sequence_stride_b: tl.int64,
        sequence_stride_t: tl.int64,
        map_length: tl.int64,
        pad_id: tl.int64,
        rows: tl.int64,
        start: tl.int64,
        NGRAM: tl.constexpr,
        HEADS: tl.constexpr,
        WIDTH: tl.constexpr,
        HAS_MASK: tl.constexpr,
        HAS_SEQUENCE: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_J: tl.constexpr,
    ):
        """Write out[b, t, j] for BLOCK_T output rows and BLOCK_J of the WIDTH address columns.

        Output row t reads input position ``start + t`` and its history; ``j`` enumerates
        (layer, order, head) row-major, the layout of ``primes`` and ``offsets``.
        """
        COLUMNS: tl.constexpr = (NGRAM - 1) * HEADS
        row = tl.program_id(0).to(tl.int64) * BLOCK_T + tl.arange(0, BLOCK_T)
        column = tl.program_id(1) * BLOCK_J + tl.arange(0, BLOCK_J)
        batch = tl.program_id(2).to(tl.int64)
        live = row < rows
        live_column = column < WIDTH
        layer = column // COLUMNS
        # Column j of a layer hashes the n-gram of order i = (j % COLUMNS) // HEADS + 1, the XOR
        # of the products of shifts 0..i.
        order = (column % COLUMNS) // HEADS + 1
        prime = tl.load(prime_ptr + column, mask=live_column, other=1)[None, :]
        offset = tl.load(offset_ptr + column, mask=live_column, other=0)[None, :]

        position = start + row
        tokens = token_ptr + batch * token_stride_b
        masks = mask_ptr + batch * mask_stride_b
        sequences = sequence_ptr + batch * sequence_stride_b
        if HAS_SEQUENCE:
            current = tl.load(sequences + position * sequence_stride_t, mask=live, other=0)
        blocked = tl.zeros([BLOCK_T], dtype=tl.int1)
        rolling = tl.zeros([BLOCK_T, BLOCK_J], dtype=tl.int64)
        for shift in tl.static_range(NGRAM):
            source = tl.maximum(position - shift, 0)
            token = tl.load(tokens + source * token_stride_t, mask=live, other=0).to(tl.int64)
            token = tl.where(token < 0, token + map_length, token)
            in_map = live & (token >= 0) & (token < map_length)
            compressed = tl.load(token_map_ptr + token, mask=in_map, other=0)
            if HAS_MASK:
                valid = tl.load(masks + source * mask_stride_t, mask=live, other=0) != 0
                compressed = tl.where(valid, compressed, -1)
            blocked = blocked | (position < shift) | (compressed == -1)
            if HAS_SEQUENCE:
                history = tl.load(sequences + source * sequence_stride_t, mask=live, other=0)
                blocked = blocked | (history != current)
            token = tl.where(blocked, pad_id, compressed)
            # Shift k enters the orders i >= k; a zero multiplier leaves the XOR unchanged.
            multiplier = tl.load(
                multiplier_ptr + layer * NGRAM + shift, mask=live_column & (order >= shift), other=0
            )
            rolling = rolling ^ (token[:, None] * multiplier[None, :])
        # PyTorch's integer % is floored (the remainder takes the divisor's sign); Triton's is
        # truncated like C, so a nonzero remainder of the other sign moves by one divisor.
        remainder = rolling % prime
        wrong_sign = (remainder != 0) & ((remainder < 0) != (prime < 0))
        remainder = tl.where(wrong_sign, remainder + prime, remainder)
        out = out_ptr + (batch * rows + row)[:, None] * WIDTH + column[None, :]
        tl.store(out, remainder + offset, mask=live[:, None] & live_column[None, :])

else:
    _engram_hash_kernel = None


def _tile(width: int) -> tuple:
    """(BLOCK_T, BLOCK_J): up to 64 columns and 128 addresses per one-warp program."""
    block_j = min(triton.next_power_of_2(width), 64)
    return max(1, 128 // block_j), block_j


def _warn_once(reason: str, key: tuple, message: str) -> None:
    """Log ``message`` once per (reason, specialization key)."""
    if (reason, key) not in _WARNED:
        _WARNED.add((reason, key))
        logger.warning("Fused Engram hash %s: %s", key, message)


def _check_token_ids(token_ids: torch.Tensor, map_length: int) -> None:
    """Assert on the device that every id indexes a ``map_length`` map like PyTorch indexing.

    No host sync, so it is also recorded in CUDA graphs and checked at every replay.
    """
    if map_length > torch.iinfo(token_ids.dtype).max:
        return  # every id of this dtype lies in [-map_length, map_length)
    in_range = ((token_ids >= -map_length) & (token_ids < map_length)).all()
    torch._assert_async(in_range, _OUT_OF_RANGE_MESSAGE)


def _specialization_key(token_ids, token_mask, sequence_ids, ngram, heads, width) -> tuple:
    """Everything that selects a Triton specialization of ``_engram_hash_kernel``.

    Scalars are typed and unspecialized and pointers are not specialized on alignment, so the
    pointer dtypes, the constexprs and the launch options are the whole key.
    """
    return (
        token_ids.device.index,
        token_ids.dtype,
        None if token_mask is None else token_mask.dtype,
        None if sequence_ids is None else sequence_ids.dtype,
        ngram,
        heads,
        width,
        *_tile(width),
        _NUM_WARPS,
    )


def _eligible(token_ids, token_mask, sequence_ids, token_map, multipliers, primes, offsets) -> bool:
    """Whether the kernel takes this call (anything else uses the PyTorch reference)."""
    if not (
        HAVE_TRITON
        and isinstance(token_ids, torch.Tensor)
        and token_ids.is_cuda
        and token_ids.dim() == 2
        and token_ids.shape[0] <= 65535  # the batch is the third grid dimension
        and token_ids.dtype in _INDEX_DTYPES
        and token_ids.device.index == torch.cuda.current_device()
        and not torch.compiler.is_compiling()
    ):
        return False
    device, shape = token_ids.device, token_ids.shape
    if token_mask is not None and not (
        token_mask.dtype == torch.bool and token_mask.shape == shape and token_mask.device == device
    ):
        return False
    if sequence_ids is not None and not (
        sequence_ids.dtype in _INDEX_DTYPES
        and sequence_ids.shape == shape
        and sequence_ids.device == device
    ):
        return False
    if multipliers.dim() != 2 or multipliers.shape[1] < 2 or primes.dim() != 3:
        return False
    layers, ngram = multipliers.shape
    heads = primes.shape[2]
    buffers = (token_map, multipliers, primes, offsets)
    return (
        token_map.dim() == 1
        and primes.shape == (layers, ngram - 1, heads)
        and offsets.shape == (layers, (ngram - 1) * heads)
        and heads > 0
        and all(
            b.dtype == torch.int64 and b.device == device and b.is_contiguous() for b in buffers
        )
    )


def fused_engram_hash(
    token_ids: torch.Tensor,
    token_mask: Optional[torch.Tensor],
    sequence_ids: Optional[torch.Tensor],
    token_map: torch.Tensor,
    multipliers: torch.Tensor,
    primes: torch.Tensor,
    offsets: torch.Tensor,
    pad_id: int,
    local_length: int,
) -> Optional[torch.Tensor]:
    """Engram addresses of the last ``local_length`` positions, or ``None`` to use the reference.

    Args:
        token_ids: ``[batch, length]`` int32/int64 raw token ids (any strides), already extended
            by the CP halo when context parallelism is on.
        token_mask: ``[batch, length]`` bool (False marks a missing token) or None.
        sequence_ids: ``[batch, length]`` int32/int64 document ids, or None.
        token_map, multipliers, primes, offsets, pad_id: ``EngramHasher``'s buffers and pad id.
        local_length: rows to return, the trailing positions exactly as ``[:, -local_length:]``
            selects them (0 returns every row, a value above the length every row).

    Returns:
        int64 ``[batch, rows, layers, (max_ngram_size - 1) * n_heads]`` addresses, contiguous,
        bit-identical to ``EngramHasher``'s PyTorch arithmetic; ``None`` for an input the kernel
        does not take (CPU tensors, other dtypes or devices, inconsistent buffers, torch.compile
        tracing), for a specialization that failed to compile or launch, and inside CUDA-graph
        capture when the specialization never ran eagerly.
    """
    global capture_fallbacks
    if not _eligible(token_ids, token_mask, sequence_ids, token_map, multipliers, primes, offsets):
        return None
    layers, ngram = multipliers.shape
    heads = primes.shape[2]
    columns = (ngram - 1) * heads
    batch, length = token_ids.shape
    # The positions the reference's [:, -local_length:] keeps (Python slicing: -0 keeps all).
    start, stop, _ = slice(-local_length, None).indices(length)
    rows = stop - start
    if batch * rows == 0:
        return torch.empty(
            (batch, rows, layers, columns), dtype=torch.int64, device=token_ids.device
        )
    key = _specialization_key(token_ids, token_mask, sequence_ids, ngram, heads, layers * columns)
    if key in _FAILED_KEYS:
        return None
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and key not in _WARM_KEYS:
        capture_fallbacks += 1
        _warn_once(
            "cold",
            key,
            "first used inside CUDA-graph capture; this graph records the PyTorch arithmetic "
            "(run the hasher eagerly before capturing to use the kernel)",
        )
        return None
    if DEVICE_CHECKS:
        _check_token_ids(token_ids, token_map.shape[0])
    out = torch.empty((batch, rows, layers, columns), dtype=torch.int64, device=token_ids.device)
    # Absent operands alias the token ids; HAS_MASK / HAS_SEQUENCE keep them unread.
    mask = token_ids if token_mask is None else token_mask.view(torch.uint8)
    sequences = token_ids if sequence_ids is None else sequence_ids
    block_t, block_j = _tile(layers * columns)
    grid = (triton.cdiv(rows, block_t), triton.cdiv(layers * columns, block_j), batch)
    try:
        _engram_hash_kernel[grid](
            token_ids,
            mask,
            sequences,
            token_map,
            multipliers,
            primes,
            offsets,
            out,
            token_ids.stride(0),
            token_ids.stride(1),
            mask.stride(0),
            mask.stride(1),
            sequences.stride(0),
            sequences.stride(1),
            token_map.shape[0],
            int(pad_id),
            rows,
            start,
            NGRAM=ngram,
            HEADS=heads,
            WIDTH=layers * columns,
            HAS_MASK=token_mask is not None,
            HAS_SEQUENCE=sequence_ids is not None,
            BLOCK_T=block_t,
            BLOCK_J=block_j,
            num_warps=_NUM_WARPS,
        )
    except Exception as error:  # pylint: disable=broad-exception-caught
        # Compilation or launch failure: the PyTorch arithmetic gives the same addresses. Inside
        # capture the failure may have invalidated the graph, so it propagates.
        if capturing:
            raise
        _FAILED_KEYS.add(key)
        _warn_once("failed", key, f"kernel unavailable ({error!r}); using the PyTorch arithmetic")
        return None
    if not capturing:
        _WARM_KEYS.add(key)
    return out
