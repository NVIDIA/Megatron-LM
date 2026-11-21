# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared pieces of the SSM oracles for the Mamba-2 chunked-scan building blocks (test_chunk_cumsum.py,
test_chunk_state.py, test_state_passing.py, test_bmm_chunk.py, test_batch_invariant_decode_rows.py).

* Chunk windows. A launch covers `nchunks` windows of the packed token axis. Varlen mode (prefill) takes
  them from `cu_chunk_seqlens` (window c = [cu[c], cu[c+1])); fixed-window mode (batch-invariant decode:
  target rows, chunk flags, dst states) takes a start per window and a length of `chunk_size`. The
  references below take the windows as (start, length) pairs and never see the kernels' blocking.
* The tensor cores' operand precision. The training scan and the fork both feed `tl.dot` with operands in
  the input dtype: bf16 and fp16 are read as stored; fp32 is read as TF32, the tensor core ignoring the
  low 13 mantissa bits. A trusted implementation that keeps every operand exact would be far more
  accurate than any tensor-core kernel (the bound is a factor over the trusted error), so the trusted
  implementations here round the operands as the tensor cores read them (`tensor_core_operand`) and
  accumulate their products in fp64 before rounding the result to the output dtype. If Triton rounds fp32
  operands to TF32 instead of truncating them, the kernel is only more accurate than this.
* Allocations inside a launcher. `poisoned_allocations(module)` makes the module's `torch.empty` return
  NaN-poisoned tensors inside guard bands, so a test can show which outputs a mode leaves unwritten and
  that nothing is written out of bounds, through the production launcher.
* Autotuner control: `pinned(kernel, config)` leaves the autotuner one candidate; `default_config`
  is the candidate batch-invariant mode pins (`_PINNED_MAMBA_CONFIGS`, the production tile there).
"""

import contextlib

import torch

from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions
from megatron.core.inference.contexts.attention_context.mamba_metadata import MambaMetadata
from megatron.core.transformer.custom_layers.batch_invariant_kernels import _PINNED_MAMBA_CONFIGS
from tests.unit_tests.inference.oracles import oracle_utils as ou

PRODUCTION_CHUNK = 128  # MambaMixer chunk_size default (mamba_mixer.py), MambaInferenceStateConfig


# ---------------------------------------------------------------------------------------------
# Autotuners
# ---------------------------------------------------------------------------------------------


def autotuner(kernel):
    """The Triton autotuner around `kernel` (heuristics wrappers unwrapped)."""
    from triton.runtime.autotuner import Autotuner

    while not isinstance(kernel, Autotuner):
        kernel = kernel.fn
    return kernel


def kernel_configs(kernel):
    """The autotune candidates of `kernel`, as the decorator lists them."""
    return list(autotuner(kernel).configs)


@contextlib.contextmanager
def pinned(kernel, config):
    """Leave `kernel`'s autotuner one candidate, `config`; restore its list and cache afterwards."""
    tuner = autotuner(kernel)
    saved = tuner.configs, tuner.cache
    try:
        tuner.configs = [config]
        tuner.cache = {}
        yield config
    finally:
        tuner.configs, tuner.cache = saved


@contextlib.contextmanager
def with_configs(kernel, configs):
    """Give `kernel`'s autotuner the candidate list `configs`; restore its list and cache afterwards."""
    tuner = autotuner(kernel)
    saved = tuner.configs, tuner.cache
    try:
        tuner.configs = list(configs)
        tuner.cache = {}
        yield
    finally:
        tuner.configs, tuner.cache = saved


def default_config(kernel):
    """The candidate batch-invariant mode pins this kernel to (the production tile in that mode)."""
    tuner = autotuner(kernel)
    name = tuner.base_fn.__name__
    expected = _PINNED_MAMBA_CONFIGS[name]
    return next(c for c in tuner.configs if all(c.kwargs.get(k) == v for k, v in expected.items()))


def scan_forward_kernels(include_pip=True):
    """The five forward kernels of Megatron's chunked scan, then pip mamba_ssm's (when installed)."""
    from megatron.core.ssm.ops.mamba2 import (
        ssd_bmm,
        ssd_chunk_scan,
        ssd_chunk_state,
        ssd_state_passing,
    )

    kernels = [
        ssd_chunk_state._chunk_cumsum_fwd_kernel,
        ssd_chunk_state._chunk_state_fwd_kernel,
        ssd_state_passing._state_passing_fwd_kernel,
        ssd_bmm._bmm_chunk_fwd_kernel,
        ssd_chunk_scan._chunk_scan_fwd_kernel,
    ]
    if include_pip:
        try:
            from mamba_ssm.ops.triton import ssd_bmm as p_bmm
            from mamba_ssm.ops.triton import ssd_chunk_scan as p_scan
            from mamba_ssm.ops.triton import ssd_chunk_state as p_state
            from mamba_ssm.ops.triton import ssd_state_passing as p_pass
        except ImportError:
            return kernels
        kernels += [
            p_state._chunk_cumsum_fwd_kernel,
            p_state._chunk_state_fwd_kernel,
            p_pass._state_passing_fwd_kernel,
            p_bmm._bmm_chunk_fwd_kernel,
            p_scan._chunk_scan_fwd_kernel,
        ]
    return kernels


@contextlib.contextmanager
def pinned_scan_kernels(include_pip=True):
    """Pin every chunked-scan forward kernel (Megatron's and pip's) to the tile batch-invariant mode
    pins, as `_pin_mamba_autotuners` does, and restore them afterwards: one compile per shape instead of
    timing every candidate."""
    with contextlib.ExitStack() as stack:
        for kernel in scan_forward_kernels(include_pip):
            tuner = autotuner(kernel)
            if len(tuner.configs) > 1:
                stack.enter_context(pinned(kernel, default_config(kernel)))
        yield


def tile_id(config):
    """A short name for a config: its block sizes, stages and warps."""
    blocks = "".join(f"{k.replace('BLOCK_SIZE_', '')}{v}" for k, v in sorted(config.kwargs.items()))
    return f"{blocks}s{config.num_stages}w{config.num_warps}"


# ---------------------------------------------------------------------------------------------
# Launcher allocations
# ---------------------------------------------------------------------------------------------


class _PoisonedTorch:
    """Stands in for `torch` in a launcher module: `empty` and `empty_like` return poisoned, guarded
    (contiguous) tensors; everything else is torch's."""

    def __init__(self, real, buffers, guard_bytes):
        self._real = real
        self._buffers = buffers
        self._guard_bytes = guard_bytes

    def _allocate(self, shape, dtype, device):
        buf = ou.guarded_empty(
            tuple(int(s) for s in shape), dtype, device, guard_bytes=self._guard_bytes
        )
        self._buffers.append(buf)
        return buf.tensor

    def empty(self, *size, dtype=None, device=None, **kwargs):
        if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)):
            size = tuple(size[0])
        return self._allocate(size, dtype or torch.get_default_dtype(), device or "cuda")

    def empty_like(self, t, *, dtype=None, device=None, **kwargs):
        return self._allocate(t.shape, dtype or t.dtype, device or t.device)

    def __getattr__(self, name):
        return getattr(self._real, name)


@contextlib.contextmanager
def poisoned_allocations(module, guard_bytes=None):
    """Within the block, `module`'s `torch.empty` / `torch.empty_like` allocate NaN-poisoned tensors in
    guard bands (`guard_bytes` each side; default one leading-dim row, at least 4 KiB).

    Yields the list of `GuardedBuffer`s allocated so far; `check_all(buffers)` checks their guards.
    """
    buffers = []
    real = module.torch
    module.torch = _PoisonedTorch(real, buffers, guard_bytes)
    try:
        yield buffers
    finally:
        module.torch = real


def check_all(buffers, what):
    """Every guard band of `buffers` intact."""
    ou.check_guards(**{f"{what}[{i}]": b for i, b in enumerate(buffers)})


# ---------------------------------------------------------------------------------------------
# Numerics helpers
# ---------------------------------------------------------------------------------------------


def tensor_core_operand(t):
    """`t` as a tensor core reads it: 16-bit floats unchanged, fp32 as TF32 (low 13 bits dropped)."""
    if t.dtype == torch.float32:
        return (t.contiguous().view(torch.int32) & -8192).view(torch.float32)
    return t


def assert_equal(actual, expected, *, what):
    """Exact equality of computed values: bit for bit after mapping -0.0 to +0.0.

    A sum of signed zeros may come out as either zero depending on the order (a scan's identity element,
    a zero accumulator); the contracts here are numeric, so the sign of a zero carries no meaning.
    """
    ou.assert_exact(actual + 0.0, expected.to(actual.dtype) + 0.0, what=what)


def as_operand(t, dtype):
    """`t` rounded to `dtype` (as a kernel's `.to(dtype)` before a dot), then as the tensor core reads it,
    returned in fp64."""
    return tensor_core_operand(t.to(dtype)).double()


def softplus_training_formula(v):
    """softplus as the training scan and the prefill cumsum evaluate it in fp32: log(exp(v) + 1) below
    20, v above (mamba_ssm's formula, kept by #8960 so batch-invariant decode matches training)."""
    return torch.where(v <= 20.0, torch.log(torch.exp(v) + 1), v)


def group_of_head(heads, groups, device):
    return torch.arange(heads, device=device) // (heads // groups)


# ---------------------------------------------------------------------------------------------
# Chunk windows
# ---------------------------------------------------------------------------------------------


def varlen_windows(cu_chunk_seqlens):
    """(start, length) of every varlen chunk: window c = [cu[c], cu[c+1])."""
    cu = cu_chunk_seqlens.tolist()
    return [(cu[c], cu[c + 1] - cu[c]) for c in range(len(cu) - 1)]


def fixed_windows(chunk_starts, chunk_size):
    """(start, chunk_size) of every fixed window (batch-invariant decode)."""
    return [(s, chunk_size) for s in chunk_starts.tolist()]


def gather_windows(t, windows, chunk_size):
    """Token rows of every window, (nchunks, chunk_size, ...), zero past each window's length."""
    out = t.new_zeros((len(windows), chunk_size) + tuple(t.shape[1:]))
    for c, (start, length) in enumerate(windows):
        out[c, :length] = t[start : start + length]
    return out


def window_mask(windows, chunk_size, device):
    """(nchunks, chunk_size) bool: position k lies inside window c."""
    lengths = torch.tensor([n for _, n in windows], device=device)
    return torch.arange(chunk_size, device=device)[None, :] < lengths[:, None]


def production_metadata(lengths, padded_requests, padded_tail_tokens, chunk_size, device="cuda"):
    """Prefill chunk and conv metadata of a pack from `MambaMetadata.compute_cpu_metadata` (production).

    `lengths`: the real requests' token counts; `padded_requests` zero-length requests and
    `padded_tail_tokens` tokens are added as a CUDA-graph bucket does. Returns device tensors and sizes,
    including the conv kernel's per-token `conv_seq_idx` / `conv_seq_start`.
    """
    real = len(lengths)
    padded_count = real + padded_requests
    real_tokens = sum(lengths)
    padded_tokens = real_tokens + padded_tail_tokens
    metadata = MambaMetadata(
        max_requests=padded_count, max_tokens=padded_tokens, mamba_chunk_size=chunk_size
    )
    max_chunks = padded_tokens // chunk_size + padded_count
    i32 = dict(dtype=torch.int32)
    bufs = {
        "batch_indices_decode": torch.full((padded_count,), -1, dtype=torch.int64),
        "batch_indices_prefill": torch.full((padded_count,), -1, **i32),
        "seq_idx": torch.full((1, max(padded_tokens, 1)), -1, **i32),
        "cu_seqlens": torch.zeros(padded_count + 1, **i32),
        "cu_chunk_seqlens": torch.zeros(max_chunks + 1, **i32),
        "last_chunk_indices": torch.zeros(padded_count, **i32),
        "seq_idx_for_varlen": torch.zeros(max_chunks, **i32),
        "conv_seq_idx": torch.full((max(padded_tokens, 1),), -7, **i32),
        "conv_seq_start": torch.full((max(padded_tokens, 1),), -7, **i32),
    }
    metadata.bind_cpu_buffers(bufs)
    cu_query = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), **i32)
    token_to_request = torch.repeat_interleave(
        torch.arange(real, **i32), torch.tensor(lengths, dtype=torch.int64)
    )
    result = metadata.compute_cpu_metadata(
        active_mamba_indices=torch.arange(real, **i32),
        token_to_request_idx=token_to_request,
        cpu_cu_query=cu_query,
        batch_dimensions=InferenceBatchDimensions(
            token_count=real_tokens, prefill_req_count=real, decode_req_count=0
        ),
        padded_batch_dimensions=InferenceBatchDimensions(
            token_count=padded_tokens, prefill_req_count=padded_count, decode_req_count=0
        ),
        enable_chunked_prefill=True,
    )
    nchunks = result["padded_max_chunks"]
    return dict(
        cu_seqlens=bufs["cu_seqlens"][: padded_count + 1].to(device),
        cu_chunk_seqlens=bufs["cu_chunk_seqlens"][: nchunks + 1].to(device),
        last_chunk_indices=bufs["last_chunk_indices"][:padded_count].to(device),
        seq_idx=bufs["seq_idx_for_varlen"][:nchunks].to(device),
        conv_seq_idx=bufs["conv_seq_idx"][:padded_tokens].to(device),
        conv_seq_start=bufs["conv_seq_start"][:padded_tokens].to(device),
        real_requests=real,
        real_tokens=real_tokens,
        padded_tokens=padded_tokens,
    )


def decode_windows(
    entries, chunk_size, *, padding=(), rows=None, device="cuda", seed=ou.DEFAULT_SEED
):
    """Batch-invariant decode metadata for `entries` decode entries over a replay buffer of `entries`
    slots of `chunk_size` rows (`BatchInvariantDecodeBuffers`): shuffled slots, `chunk_starts = slot *
    chunk_size`, a target row per entry (`rows`, or every row in turn with every fifth entry on the last
    row), `-1` padding entries at the positions in `padding`, and chunk flags on the last row. As
    `_decode_step_prep_kernel` writes them, a padding entry keeps `chunk_starts = -chunk_size` (its slot
    times the chunk size), target row -1 and flag 0: no kernel may read through it.
    """
    generator = torch.Generator().manual_seed(seed)
    slots = torch.randperm(entries, generator=generator).to(device=device, dtype=torch.int32)
    if rows is None:
        pos = torch.arange(entries) % chunk_size
        pos = torch.where(torch.arange(entries) % 5 == 0, chunk_size - 1, pos)
    else:
        pos = torch.tensor([rows[i % len(rows)] for i in range(entries)])
    pos = pos.to(device=device, dtype=torch.int32)
    for p in padding:
        slots[p] = -1
    target_rows = torch.where(slots >= 0, pos, torch.full_like(pos, -1))
    return dict(
        slots=slots,
        chunk_starts=(slots * chunk_size).to(torch.int32),
        target_rows=target_rows,
        chunk_flags=(target_rows == chunk_size - 1).to(torch.int32),
    )
