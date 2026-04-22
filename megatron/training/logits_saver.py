# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
Utilities for saving top-K log-probabilities during model training with minimal overhead.

This module provides hook-based utilities to capture and persist top-K log-probabilities
during training while respecting tensor parallelism and minimizing communication,
computation, I/O, and disk space overhead.

The forward hook accumulates raw logits across microbatches and, once all microbatches
have been collected, converts them to log-probs and writes them to disk in a single I/O call.

For efficient global top-K log-prob computation across TP ranks:
- Local top-K is computed on fp32 logits first to avoid materializing
  a full vocab-sized log-softmax tensor
- The log-softmax denominator (log-sum-exp) is computed TP-aware and applied
  only to the top-K values
- Local candidates are gathered to TP rank 0 for global top-K selection
- TP rank 0 saves the full top-K log-probabilities (unsharded)

Index storage optimization:
- Global vocab requires 17 bits, so we store lower 16 bits as uint16
- The 17th bit is stored separately as torch.bool (same shape as indices)
- To reconstruct: global_index = (bit_17 << 16) | low_16_bits
"""

import concurrent.futures
import glob
import io
import logging
import os
import re
import tarfile
import warnings
from collections import OrderedDict
from contextlib import nullcontext
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.distributed as dist

try:
  import zstandard
except ImportError:
  zstandard = None

from megatron.core import parallel_state
from megatron.core.msc_utils import open_file
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.training import get_args
from megatron.training.utils import print_rank_0

logger = logging.getLogger(__name__)

# Module-level reference so checkpoint/shutdown code can call flush().
_ACTIVE_LOGITS_SAVER: Optional["LogitsSaverHooks"] = None


def get_logits_saver() -> Optional["LogitsSaverHooks"]:
    """Return the active :class:`LogitsSaverHooks` instance, or *None*."""
    return _ACTIVE_LOGITS_SAVER


def get_current_iteration() -> int:
    """Return the current training iteration from ``get_args()``.

    Prefers ``args.curr_iteration`` (set during forward/backward) and
    falls back to ``args.iteration``.
    """
    args = get_args()
    iteration = getattr(args, 'curr_iteration', None)
    if iteration is None:
        iteration = getattr(args, 'iteration')
    return iteration


_MAX_VOCAB_SIZE = 2 ** 17  # 131072 - maximum supported vocab size

_FOLDER_NAMES_PREFIX = "logprobs_iter"

# Matches both unsharded (``cp{C}_dp{D}__{B}.tar``) and legacy TP-sharded
# (``tp{T}_cp{C}_dp{D}__{B}.tar``) batched tar filenames.  Named groups:
#   tp   – TP rank (None when unsharded)
#   cp   – CP rank
#   dp   – DP rank
#   iter – trailing iteration number *B*
_BATCHED_TAR_RE = re.compile(
    r"^(?:tp(?P<tp>\d+)_)?cp(?P<cp>\d+)_dp(?P<dp>\d+)__(?P<iter>\d+)\.tar$"
)


def _batched_tar_filename(
    cp_rank: int, dp_rank: int, last_iter: int,
    tp_rank: Optional[int] = None,
) -> str:
    """Return the filename for a batched tar shard.

    When *tp_rank* is ``None`` (default), returns the unsharded format
    ``cp{C}_dp{D}__{B}.tar``.  When *tp_rank* is given, returns the
    legacy TP-sharded format ``tp{T}_cp{C}_dp{D}__{B}.tar``.
    """
    if tp_rank is not None:
        return f"tp{tp_rank}_cp{cp_rank}_dp{dp_rank}__{last_iter}.tar"
    return f"cp{cp_rank}_dp{dp_rank}__{last_iter}.tar"


def _batched_tar_prefix(
    cp_rank: int, dp_rank: int,
    tp_rank: Optional[int] = None,
) -> str:
    """Return the glob prefix for batched tar shards.

    When *tp_rank* is ``None`` (default), returns the unsharded prefix
    ``cp{C}_dp{D}__``.  When *tp_rank* is given, returns the legacy
    TP-sharded prefix ``tp{T}_cp{C}_dp{D}__``.
    """
    if tp_rank is not None:
        return f"tp{tp_rank}_cp{cp_rank}_dp{dp_rank}__"
    return f"cp{cp_rank}_dp{dp_rank}__"


def _sorted_batched_tars(paths: List[str]) -> List[str]:
    """Sort batched tar paths by their iteration number (numeric, ascending).

    Filenames are not zero-padded, so a plain lexicographic sort would be
    wrong (e.g. ``...10.tar`` before ``...9.tar``).  Works with both
    TP-sharded and unsharded naming conventions.
    """
    keyed = []
    for p in paths:
        if m := _BATCHED_TAR_RE.match(os.path.basename(p)):
            keyed.append((int(m.group("iter")), p))
    keyed.sort()
    return [p for _, p in keyed]


def _pack_indices(indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split 17-bit global indices into uint16 lower bits + bool 17th bit.

    Args:
        indices: Global indices tensor, values must fit in 17 bits

    Returns:
        Tuple of (low_16_bits as uint16, bit_17 as bool)
    """
    low_bits = (indices & 0xFFFF).to(torch.uint16)
    bit_17 = (indices >> 16).to(torch.bool)
    return low_bits, bit_17


def _unpack_indices(low_bits: torch.Tensor, bit_17: torch.Tensor) -> torch.Tensor:
    """Reconstruct indices from uint16 lower bits + bool 17th bit.

    Args:
        low_bits: Lower 16 bits as uint16
        bit_17: 17th bit as bool

    Returns:
        Reconstructed global indices as int64
    """
    return (bit_17.long() << 16) | low_bits.long()


def _format_folder_and_filename(
    save_dir: str,
    iteration: int,
    tp_rank: int,
    cp_rank: int,
    dp_rank: int,
    suffix: str = "",
) -> Tuple[str, str]:
    """Generate a unique folder and filename for log-prob data."""
    folder = os.path.join(save_dir, f"{_FOLDER_NAMES_PREFIX}{iteration}")
    filename = f"tp{tp_rank}_cp{cp_rank}_dp{dp_rank}{suffix}"
    return folder, filename


class LogitsSaverHooks:
    """
    Hook-based utility to save top-K log-probabilities during training with minimal overhead.

    This class provides a forward hook to attach to the module that outputs logits.
    The hook accumulates logits across microbatches and saves all log-probs to disk
    in a single I/O call once all microbatches have been collected.

    Top-K selection is performed on raw logits first, then log-probabilities
    are computed only for the selected positions using the log-softmax
    denominator (log-sum-exp).  Because log-softmax is monotonically
    increasing, the top-K positions are identical to those of the raw logits.
    This avoids materializing a full vocab-sized log-softmax tensor.

    The implementation efficiently handles tensor parallelism by:
    - Computing local top-K on fp32 logits to reduce memory and communication
    - Concatenating values and indices before all_gather
    - Computing global top-K from gathered results
    - Having each rank save only its K/N slice (evenly divisible)

    Indices are stored efficiently: uint16 for lower 16 bits + separate high bit tensor.
    tp_rank and dp_rank are stored in the filename for flexibility with multi-digit values.

    When ``flush_interval`` > 1, iteration data is accumulated in memory and
    flushed as a single tar archive every *flush_interval* iterations via a
    background thread, reducing inode usage and avoiding blocking the training
    loop.  Call :meth:`flush` at checkpoint / shutdown to write any remaining
    buffered data.

    Args:
        k: Number of top log-probabilities to save globally
        save_dir: Directory to save log-prob files
        compress_zstd: Whether to use zstd compression (requires zstandard package)
        flush_interval: Number of iterations to batch before writing a single tar
            archive.  1 (default) preserves the legacy one-file-per-iteration behaviour.
        save_dtype: String name of the dtype for top-K log-probabilities on
            disk.  One of ``'fp16'``, ``'bf16'``, or ``'fp32'``.
    """

    _DTYPE_MAP = {
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
        'fp32': torch.float32,
    }

    def __init__(
        self,
        k: int,
        save_dir: str,
        *,
        compress_zstd: bool = False,
        flush_interval: int = 1,
        save_dtype: str = 'fp16',
    ):
        assert k > 0, "Number of top log-probabilities to save must be positive"
        assert save_dir is not None, "Save directory must be provided"
        assert flush_interval >= 1, "flush_interval must be >= 1"
        if save_dtype not in self._DTYPE_MAP:
            raise ValueError(
                f"save_dtype must be one of {list(self._DTYPE_MAP)}, got '{save_dtype}'"
            )
        self._save_dtype = self._DTYPE_MAP[save_dtype]

        self.k = k
        self.save_dir = save_dir
        self.flush_interval = flush_interval

        # Parallel state info
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self.tp_group = parallel_state.get_tensor_model_parallel_group()
        self.cp_rank = parallel_state.get_context_parallel_rank()
        self.dp_rank = parallel_state.get_data_parallel_rank()
        self._tp_dst_rank_global = parallel_state.get_tensor_model_parallel_src_rank()

        # Track number of MTP outputs to ignore
        args = get_args()
        self._mtp_num_layers = args.mtp_num_layers or 0
        self._curr_mtp_passes = 0

        # Hook states – store already-processed top-K results (not full logits)
        self._accumulated_results: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._hook_handles: List[Any] = []

        # Zstd compression setup
        if zstandard is not None:
            self._zstd_compressor = zstandard.ZstdCompressor(level=3)
        else:
            warnings.warn(
                "zstandard package not found; disabling zstd compression for log-probs."
            )
            self._zstd_compressor = None

        # Batched tar state (active when flush_interval > 1)
        self._pending_writes: OrderedDict[int, Tuple[bytes, str]] = OrderedDict()
        self._flush_executor: Optional[concurrent.futures.ThreadPoolExecutor] = (
            concurrent.futures.ThreadPoolExecutor(max_workers=1)
            if self.flush_interval > 1
            else None
        )
        self._flush_futures: List[concurrent.futures.Future] = []

        # Create save directory if needed
        os.makedirs(self.save_dir, exist_ok=True)

        # Register as the active saver so checkpoint code can flush
        global _ACTIVE_LOGITS_SAVER
        _ACTIVE_LOGITS_SAVER = self

    def get_forward_hook(self) -> Callable:
        """Returns the forward hook to capture top-K log-probs per microbatch.

        This hook should be registered on the module that outputs logits.
        Each microbatch is processed immediately to extract only the small
        top-K values and indices, avoiding storage of full vocab-sized logits
        across microbatches.
        """
        def forward_hook(
            module: torch.nn.Module,
            input: Any,
            output: Tuple[torch.Tensor, ...],
        ) -> None:
            # Skip if not in training mode
            if not module.training:
                return

            # Store logits and maybe save to disk
            if self._curr_mtp_passes == self._mtp_num_layers:
                # Main output logits come after MTP logits
                logits = output[0] if isinstance(output, tuple) else output

                with torch.no_grad():
                    result = self._process_single_microbatch(logits)
                if result is not None:
                    self._accumulated_results.append(result)

                if len(self._accumulated_results) == get_num_microbatches():
                    self._save_accumulated_log_probs()
                    self._accumulated_results.clear()

                self._curr_mtp_passes = 0
            else:
                self._curr_mtp_passes += 1

        return forward_hook

    def attach_hooks(self, module: torch.nn.Module) -> None:
        """Convenience method to attach hooks to a module.

        Args:
            module: The module that outputs logits (e.g., the output layer)
        """
        fwd_handle = module.register_forward_hook(self.get_forward_hook())
        self._hook_handles.extend([fwd_handle])

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def _save_accumulated_log_probs(self) -> None:
        """Move accumulated top-K results to CPU and save to disk.

        By this point each microbatch has already been processed in the
        forward hook, so this method only transfers the small top-K tensors
        to CPU and writes them out in a single I/O call.
        """
        if self.tp_rank != 0:
            return
        if not self._accumulated_results:
            logger.warning("No accumulated log-probs to save")
            return

        all_values = []
        all_indices_low = []
        all_high_bits = []

        for values, indices_low, high_bit in self._accumulated_results:
            all_values.append(values.cpu())
            all_indices_low.append(indices_low.cpu())
            all_high_bits.append(high_bit.cpu())

        self._write_to_disk(all_values, all_indices_low, all_high_bits)

    def _process_single_microbatch(
        self, logits: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Process a single logits tensor and return top-K log-probability data.

        Selects local top-K positions on raw logits first, then computes
        the global log-sum-exp denominator using ``torch.logsumexp`` for
        the local shard (leveraging PyTorch's fused CUDA kernel) combined
        across TP ranks via a numerically stable log-space reduction.
        This avoids materializing a full vocab-sized log-softmax tensor.

        All TP ranks participate in the collective operations (all-reduce for
        the log-sum-exp combination, gather for top-K candidates), but only
        TP rank 0 performs the final global top-K and returns the result.
        Other ranks return ``None``.

        Args:
            logits: Tensor of shape (seq_len, batch, local_vocab_size)

        Returns:
            Tuple of (log_prob_values, indices_low, high_bit) on TP rank 0,
            ``None`` on other TP ranks.
        """
        local_vocab_size = logits.shape[-1]
        global_vocab_size = local_vocab_size * self.tp_size

        assert global_vocab_size <= _MAX_VOCAB_SIZE, (
            f"Global vocab size {global_vocab_size} exceeds maximum supported {_MAX_VOCAB_SIZE} (17 bits)"
        )

        effective_k = min(self.k, global_vocab_size)
        local_k = min(effective_k, local_vocab_size)

        # Convert to fp32 to lessen precision issues
        logits = logits.float()

        local_logit_vals, local_indices = torch.topk(logits, local_k, dim=-1)

        # Local log-sum-exp via PyTorch's fused CUDA kernel
        local_lse = torch.logsumexp(logits, dim=-1, keepdim=True)

        if self.tp_size > 1:
            # Combine local log-sum-exp values across TP ranks using
            # the standard numerically stable log-space reduction:
            #   global_lse = max_lse + log(sum_i(exp(lse_i - max_lse)))
            max_lse = local_lse.clone()
            dist.all_reduce(max_lse, op=dist.ReduceOp.MAX, group=self.tp_group)
            sum_exp_lse = torch.exp(local_lse - max_lse)
            dist.all_reduce(sum_exp_lse, op=dist.ReduceOp.SUM, group=self.tp_group)
            global_lse = max_lse + torch.log(sum_exp_lse)
        else:
            global_lse = local_lse

        local_logprob_vals = local_logit_vals - global_lse

        if self.tp_size > 1:
            result = self._compute_global_topk(
                local_logit_vals, local_logprob_vals, local_indices, effective_k, local_vocab_size
            )
            if result is None:
                return None
            global_values, global_indices = result
        else:
            global_values, global_indices = local_logprob_vals, local_indices

        global_values = global_values.to(self._save_dtype)
        indices_low, high_bit = _pack_indices(global_indices)

        return global_values, indices_low, high_bit

    def _compute_global_topk(
        self,
        local_logit_vals: torch.Tensor,
        local_logprob_vals: torch.Tensor,
        local_indices: torch.Tensor,
        k: int,
        local_vocab_size: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Gather local top-K candidates to rank 0 and compute global top-K.

        The ranking is performed on fp32 *logit* values for numerical
        precision; the returned log-prob values are still fp32 here and
        are only cast to the on-disk ``save_dtype`` by the caller.

        Only TP rank 0 receives the gathered data and performs the final
        top-K selection.  Other ranks return ``None`` after participating
        in the ``gather`` collective.

        Args:
            local_logit_vals: Local top-K fp32 logit values (seq, batch, local_k)
            local_logprob_vals: Local top-K fp32 log-prob values at the
                same positions (seq, batch, local_k)
            local_indices: Local top-K indices in [0, local_vocab_size)
            k: Target number of top elements
            local_vocab_size: Size of the local vocab shard (vocab_size / tp_size)

        Returns:
            Tuple of (global_logprob_values, global_indices) on TP rank 0,
            ``None`` on other TP ranks.
        """
        vocab_offset = self.tp_rank * local_vocab_size
        global_indices = local_indices + vocab_offset

        # Pack logits, log-probs, and indices into a single tensor.
        # fp32 can exactly represent integers up to 2^24, sufficient for
        # 17-bit vocab indices.  Shape: (seq, batch, local_k, 3)
        combined = torch.stack(
            [local_logit_vals,
             local_logprob_vals.float(),
             global_indices.float()],
            dim=-1,
        )

        if self.tp_rank == 0:
            gather_list = [torch.empty_like(combined) for _ in range(self.tp_size)]
        else:
            gather_list = None
        dist.gather(combined, gather_list, dst=self._tp_dst_rank_global, group=self.tp_group)

        if self.tp_rank != 0:
            return None

        # (seq, batch, tp_size * local_k, 3)
        gathered = torch.cat(gather_list, dim=-2)

        gathered_logits = gathered[..., 0]
        gathered_logprobs = gathered[..., 1].to(local_logprob_vals.dtype)
        gathered_indices = gathered[..., 2].to(local_indices.dtype)

        _, topk_positions = torch.topk(gathered_logits, k, dim=-1)

        topk_logprobs = torch.gather(gathered_logprobs, -1, topk_positions)
        topk_global_indices = torch.gather(gathered_indices, -1, topk_positions)

        return topk_logprobs, topk_global_indices

    def _write_to_disk(
        self,
        values_list: List[torch.Tensor],
        indices_low_list: List[torch.Tensor],
        bit_17_list: List[torch.Tensor],
    ) -> None:
        """Write all microbatch data to disk in a single I/O call.

        File format: serialized dict with:
        - values: list of tensors of log-probabilities (one per microbatch)
          in ``self._save_dtype`` (default ``torch.float16``)
        - indices_low: list of uint16 tensors (lower 16 bits of vocab indices)
        - bit_17: list of bool tensors (17th bit, same shape as indices_low)

        When ``flush_interval > 1``, the serialised bytes are buffered in
        memory and flushed as a tar archive every *flush_interval* iterations.
        """
        # Serialize all tensors together
        buffer = io.BytesIO()
        torch.save({
            'values': values_list,
            'indices_low': indices_low_list,
            'bit_17': bit_17_list,
        }, buffer)
        data = buffer.getvalue()

        # Optional per-entry compression
        entry_suffix = ".pt"
        if self._zstd_compressor is not None:
            data = self._zstd_compressor.compress(data)
            entry_suffix += ".zst"

        iteration = get_current_iteration()

        if self.flush_interval <= 1:
            # Legacy: one file per iteration in a per-iteration folder
            folder, filename = _format_folder_and_filename(
                self.save_dir, iteration, self.tp_rank, self.cp_rank, self.dp_rank, entry_suffix
            )
            os.makedirs(folder, exist_ok=True)
            with open_file(os.path.join(folder, filename), "wb") as f:
                f.write(data)
        else:
            # Batched: accumulate in memory, flush when the iteration
            # is a multiple of the flush interval.
            self._pending_writes[iteration] = (data, entry_suffix)
            if (iteration + 1) % self.flush_interval == 0:
                self._flush_pending()

    # ------------------------------------------------------------------
    #  Batched-tar flush helpers
    # ------------------------------------------------------------------

    def _flush_pending(self) -> None:
        """Flush buffered iteration data as a single tar archive (async)."""
        if not self._pending_writes:
            return

        # Wait for any prior background flush to finish before handing off
        # new data, bounding peak memory to one batch of pending writes.
        self._wait_for_pending_flushes()

        # Take ownership of the current pending buffer
        writes = self._pending_writes
        self._pending_writes = OrderedDict()

        last_iter = max(writes.keys())
        tar_filename = _batched_tar_filename(
            self.cp_rank, self.dp_rank, last_iter,
        )
        tar_path = os.path.join(self.save_dir, tar_filename)

        future = self._flush_executor.submit(
            LogitsSaverHooks._write_batched_tar, tar_path, writes,
        )
        self._flush_futures.append(future)

        print_rank_0(f"Flushing {len(writes)} logit iterations to disk")

    @staticmethod
    def _write_batched_tar(
        tar_path: str,
        writes: "OrderedDict[int, Tuple[bytes, str]]",
    ) -> None:
        """Write a tar archive containing multiple iterations (runs in background thread).

        Each member is named ``{iteration}{suffix}`` so that WebDataset
        groups them as individual samples keyed by iteration number.
        """
        with tarfile.open(tar_path, "w") as tar:
            for iteration, (data, entry_suffix) in writes.items():
                member_name = f"{iteration}{entry_suffix}"
                info = tarfile.TarInfo(name=member_name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

    def _wait_for_pending_flushes(self) -> None:
        """Block until all background flush futures have completed."""
        for future in self._flush_futures:
            future.result()
        self._flush_futures.clear()

    def flush(self) -> None:
        """Flush any remaining buffered data to disk.

        Must be called at checkpoint / shutdown when ``flush_interval > 1``
        to ensure no iteration data is lost.  Safe to call when there is
        nothing pending (no-op).
        """
        if self.flush_interval > 1 and self._pending_writes:
            self._flush_pending()
        self._wait_for_pending_flushes()

    def shutdown(self) -> None:
        """Flush remaining data and shut down the background executor."""
        self.flush()
        if self._flush_executor is not None:
            self._flush_executor.shutdown(wait=True)
            self._flush_executor = None


#################################################
# Loading utilities
#################################################


def _decompress_zstd(data: bytes) -> bytes:
    """Decompress zstd-compressed bytes."""
    if zstandard is None:
        raise RuntimeError("zstandard package required to read compressed log-probs files")
    return zstandard.ZstdDecompressor().decompress(data)


def _read_logprobs_data(folder: str, base_filename: str) -> Optional[bytes]:
    """Read a log-probs file from an iteration folder or its tar archive."""
    tar_path = folder + ".tar"
    folder_name = os.path.basename(folder)

    if os.path.isdir(folder):
        archive_ctx = nullcontext()

        def resolve(name: str) -> Optional[bytes]:
            path = os.path.join(folder, name)
            if not os.path.exists(path):
                return None
            with open_file(path, 'rb') as f:
                return f.read()

    elif os.path.isfile(tar_path):
        archive = tarfile.open(tar_path, "r")
        archive_ctx = archive

        def resolve(name: str) -> Optional[bytes]:
            try:
                member = archive.extractfile(f"{folder_name}/{name}")
            except KeyError:
                return None
            return member.read() if member is not None else None

    else:
        return None

    with archive_ctx:
        for suffix in [".pt.zst", ".pt"]:
            raw = resolve(base_filename + suffix)
            if raw is None:
                continue
            return _decompress_zstd(raw) if suffix == ".pt.zst" else raw


def _read_from_batched_tar(
    save_dir: str,
    iteration: int,
    tp_rank: int,
    cp_rank: int,
    dp_rank: int,
) -> Optional[bytes]:
    """Read a single iteration's data from a batched tar archive.

    Supports both legacy TP-sharded tars (``tp{T}_cp{C}_dp{D}__{B}.tar``)
    and unsharded tars (``cp{C}_dp{D}__{B}.tar``).  Tries unsharded first,
    then falls back to sharded.
    """
    # Try unsharded format first (current), then legacy TP-sharded.
    for prefix in (
        _batched_tar_prefix(cp_rank, dp_rank),
        _batched_tar_prefix(cp_rank, dp_rank, tp_rank=tp_rank),
    ):
        pattern = os.path.join(save_dir, f"{prefix}*.tar")
        for tar_path in _sorted_batched_tars(glob.glob(pattern)):
            basename = os.path.basename(tar_path)
            m = _BATCHED_TAR_RE.match(basename)
            if not m:
                continue
            if int(m.group("iter")) < iteration:
                continue
            try:
                with tarfile.open(tar_path, "r") as tar:
                    for suffix in (".pt.zst", ".pt"):
                        member_name = f"{iteration}{suffix}"
                        try:
                            member = tar.extractfile(member_name)
                            if member is not None:
                                raw = member.read()
                                return _decompress_zstd(raw) if suffix == ".pt.zst" else raw
                        except KeyError:
                            continue
            except (tarfile.TarError, OSError):
                continue
    return None


def load_log_probs_by_rank(
    save_dir: str,
    iteration: int,
    tp_rank: int,
    cp_rank: int,
    dp_rank: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Load a saved log-probs file by specifying the root folder, iteration, and parallel ranks.

    Supports four on-disk layouts (tried in order):

    1. **Legacy folder** – ``logprobs_iter{iter}/tp{tp}_cp{cp}_dp{dp}.pt[.zst]``
    2. **Legacy folder tar** – ``logprobs_iter{iter}.tar`` containing the folder
    3. **Unsharded batched tar** – ``cp{cp}_dp{dp}__{B}.tar``
       containing ``{iter}.pt[.zst]`` (full top-K, saved by TP rank 0)
    4. **TP-sharded batched tar** (legacy) – ``tp{tp}_cp{cp}_dp{dp}__{B}.tar``
       containing ``{iter}.pt[.zst]`` (K/tp_size slice per TP rank)

    Args:
        save_dir: Root path containing all log-probs data
        iteration: Training iteration number
        tp_rank: Tensor parallel rank
        cp_rank: Context parallel rank
        dp_rank: Data parallel rank

    Returns:
        Tuple of (log_prob_values_list, indices_list)
        where each list contains one tensor per microbatch (in order).
        Values are log-probabilities; indices are reconstructed full int64 global indices.

    Raises:
        FileNotFoundError: If no matching data is found in any layout
    """
    # 1 & 2: Legacy folder / folder.tar
    folder, base_filename = _format_folder_and_filename(
        save_dir, iteration, tp_rank, cp_rank, dp_rank,
    )
    data = _read_logprobs_data(folder, base_filename)

    # 3: Batched tar
    if data is None:
        data = _read_from_batched_tar(save_dir, iteration, tp_rank, cp_rank, dp_rank)

    if data is None:
        raise FileNotFoundError(
            f"No log-probs file found for tp_rank={tp_rank}, cp_rank={cp_rank}, "
            f"dp_rank={dp_rank} at iteration {iteration} "
            f"(checked folder '{folder}', tar '{folder}.tar', "
            f"and batched tars in '{save_dir}')"
        )

    # Load tensors
    tensors = torch.load(io.BytesIO(data), weights_only=True)

    # Reconstruct full indices for each microbatch
    indices_list = [
        _unpack_indices(low, bit17)
        for low, bit17 in zip(tensors['indices_low'], tensors['bit_17'])
    ]

    return tensors['values'], indices_list


#################################################
# WebDataset-based streaming loader
#################################################


def decode_logprobs_sample(
    sample: dict,
) -> Tuple[int, List[torch.Tensor], List[torch.Tensor]]:
    """WebDataset map function: decode a single tar member into tensors.

    Handles both ``.pt`` and ``.pt.zst`` extensions produced by
    :class:`LogitsSaverHooks` when ``flush_interval > 1``.

    Args:
        sample: WebDataset sample dict with ``__key__`` and one data field.

    Returns:
        ``(iteration, values_list, indices_list)`` – the same tuple shape
        returned by :func:`load_log_probs_by_rank`.
    """
    key: str = sample["__key__"]
    iteration = int(key)

    if "pt.zst" in sample:
        data = _decompress_zstd(sample["pt.zst"])
    elif "pt" in sample:
        data = sample["pt"]
    else:
        raise ValueError(
            f"Sample '{key}' contains neither .pt nor .pt.zst data"
        )

    tensors = torch.load(io.BytesIO(data), weights_only=True)
    indices_list = [
        _unpack_indices(low, bit17)
        for low, bit17 in zip(tensors["indices_low"], tensors["bit_17"])
    ]
    return iteration, tensors["values"], indices_list
