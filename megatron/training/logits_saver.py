# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
Utilities for saving top-K log-probabilities during model training with minimal overhead.

This module provides hook-based utilities to capture and persist top-K log-probabilities
during training while respecting tensor parallelism and minimizing communication,
computation, I/O, and disk space overhead.

The forward hook accumulates raw logits across microbatches and, once all microbatches
have been collected, converts them to log-probs and writes them to disk in a single I/O call.

For efficient global top-K log-prob computation across TP ranks:
- Global log-softmax is computed in a numerically stable, TP-aware manner
- Local top_k is computed on log-probs first
- Values and indices are concatenated before all_gather to minimize communication
- A final top_k is applied to the gathered results
- Each of N ranks saves only its K/N slice to distribute I/O and storage

Index storage optimization:
- Global vocab requires 17 bits, so we store lower 16 bits as uint16
- The 17th bit is stored separately as torch.bool (same shape as indices)
- To reconstruct: global_index = (bit_17 << 16) | low_16_bits
"""

import io
import logging
import os
import warnings
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.msc_utils import open_file
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.training import get_args

logger = logging.getLogger(__name__)


_MAX_VOCAB_SIZE = 2 ** 17  # 131072 - maximum supported vocab size

FOLDER_NAMES_PREFIX = "logprobs_iter"


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
    folder = os.path.join(save_dir, f"{FOLDER_NAMES_PREFIX}{iteration}")
    filename = f"tp{tp_rank}_cp{cp_rank}_dp{dp_rank}{suffix}"
    return folder, filename


class LogitsSaverHooks:
    """
    Hook-based utility to save top-K log-probabilities during training with minimal overhead.

    This class provides a forward hook to attach to the module that outputs logits.
    The hook accumulates logits across microbatches and saves all log-probs to disk
    in a single I/O call once all microbatches have been collected.

    Raw logits are converted to global log-probabilities via a numerically stable,
    TP-aware log-softmax before top-K selection.  Because log-softmax is monotonically
    increasing, the top-K positions are identical to those of the raw logits.

    The implementation efficiently handles tensor parallelism by:
    - Computing global log-probs with all_reduce (MAX then SUM) across TP ranks
    - Computing local top-K on log-probs first to reduce memory and communication
    - Concatenating values and indices before all_gather
    - Computing global top-K from gathered results
    - Having each rank save only its K/N slice (evenly divisible)

    Indices are stored efficiently: uint16 for lower 16 bits + separate high bit tensor.
    tp_rank and dp_rank are stored in the filename for flexibility with multi-digit values.

    Args:
        k: Number of top log-probabilities to save globally
        save_dir: Directory to save log-prob files
        compress_zstd: Whether to use zstd compression (requires zstandard package)
    """

    def __init__(
        self,
        k: int,
        save_dir: str,
        *,
        compress_zstd: bool = False,
    ):
        assert k > 0, "Number of top log-probabilities to save must be positive"
        assert save_dir is not None, "Save directory must be provided"

        self.k = k
        self.save_dir = save_dir

        # Parallel state info
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self.tp_group = parallel_state.get_tensor_model_parallel_group()
        self.cp_rank = parallel_state.get_context_parallel_rank()
        self.dp_rank = parallel_state.get_data_parallel_rank()

        # Track number of MTP outputs to ignore
        args = get_args()
        self._mtp_num_layers = args.mtp_num_layers or 0
        self._curr_mtp_passes = 0

        # Hook states
        self._accumulated_logits: List[torch.Tensor] = []
        self._hook_handles: List[Any] = []

        # Zstd compression setup
        self._zstd_compressor = None
        if compress_zstd:
            try:
                import zstandard

                self._zstd_compressor = zstandard.ZstdCompressor(level=3)
            except ModuleNotFoundError:
                warnings.warn(
                    "zstandard package not found; disabling zstd compression for log-probs."
                )

        # Create save directory if needed
        if "://" not in self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

    def get_forward_hook(self) -> Callable:
        """Returns the forward hook to accumulate logits.

        This hook should be registered on the module that outputs logits.
        It appends a clone of the logits tensor to an internal list.
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
                self._accumulated_logits.append(logits.detach().clone())

                # Save all if we have reached the end of the microbatches
                if len(self._accumulated_logits) == get_num_microbatches():
                    self._save_accumulated_log_probs()
                    # Clear for next iteration
                    self._accumulated_logits.clear()

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
        """Convert all accumulated logits to log-probs and save to disk in a single I/O call."""
        if not self._accumulated_logits:
            logger.warning("No accumulated log-probs to save")
            return

        all_values = []
        all_indices_low = []
        all_high_bits = []

        with torch.no_grad():
            for logits in self._accumulated_logits:
                values, indices_low, high_bit = self._process_single_microbatch(logits)
                all_values.append(values.cpu())
                all_indices_low.append(indices_low.cpu())
                all_high_bits.append(high_bit.cpu())

        if not all_values:
            return

        # Single I/O call for all microbatches
        self._write_to_disk(all_values, all_indices_low, all_high_bits)

    def _process_single_microbatch(
        self, logits: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Process a single logits tensor and return top-K log-probability data.

        Converts raw logits to log-probabilities using a numerically stable,
        TP-aware log-softmax before selecting the top-K entries.  Because
        log-softmax is monotonically increasing with respect to the logits,
        the top-K positions are identical to those of the raw logits.

        Args:
            logits: Tensor of shape (seq_len, batch, local_vocab_size)

        Returns:
            Tuple of (log_prob_values, indices_low, high_bit)
        """
        local_vocab_size = logits.shape[-1]
        global_vocab_size = local_vocab_size * self.tp_size

        assert global_vocab_size <= _MAX_VOCAB_SIZE, (
            f"Global vocab size {global_vocab_size} exceeds maximum supported {_MAX_VOCAB_SIZE} (17 bits)"
        )

        # Effective k considering global vocab size
        effective_k = min(self.k, global_vocab_size)
        # Ensure k is evenly divisible by tp_size for fair sharding
        effective_k = (effective_k // self.tp_size) * self.tp_size

        # Convert to fp32 to lessen precision issues
        dtype = logits.dtype
        logits = logits.float()

        # Compute log-probabilities (global log-softmax, TP-aware)
        if self.tp_size > 1:
            # Subtract global max across all TP ranks for numerical stability
            logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
            dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=self.tp_group)
            logits -= logits_max

            # Compute global softmax denominator via all_reduce SUM
            local_exp_sum = torch.sum(torch.exp(logits), dim=-1, keepdim=True)
            dist.all_reduce(local_exp_sum, op=dist.ReduceOp.SUM, group=self.tp_group)

            # Each rank holds log-probs for its own vocab shard
            log_probs = logits - torch.log(local_exp_sum)
        else:
            log_probs = F.log_softmax(logits, dim=-1)

        # Back to original dtype
        log_probs = log_probs.to(dtype)

        # Step 1: Compute local top-K on log-probs
        # (log-softmax is monotonically increasing, so top-K positions are the same as for logits)
        local_k = min(effective_k, local_vocab_size)
        local_values, local_indices = torch.topk(log_probs, local_k, dim=-1)

        if self.tp_size > 1:
            # Step 2: Compute global top-K across TP ranks
            global_values, global_indices = self._compute_global_topk(
                local_values, local_indices, effective_k, local_vocab_size
            )
        else:
            global_values, global_indices = local_values, local_indices

        # Step 3: Each rank saves its K/N slice
        # Since all ranks have identical copies after global topk,
        # each rank saves only its portion to distribute I/O and storage
        shard_size = effective_k // self.tp_size
        start_idx = self.tp_rank * shard_size
        end_idx = start_idx + shard_size

        shard_values = global_values[..., start_idx:end_idx]
        shard_indices = global_indices[..., start_idx:end_idx]

        # Split indices: uint16 for lower 16 bits + uint8 for 17th bit
        indices_low, high_bit = _pack_indices(shard_indices)

        return shard_values, indices_low, high_bit

    def _compute_global_topk(
        self,
        local_values: torch.Tensor,
        local_indices: torch.Tensor,
        k: int,
        local_vocab_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute global top-K across all TP ranks efficiently.

        Strategy:
        1. Convert local indices to global indices by adding rank offset
        2. Concatenate values and indices to minimize all_gather calls (single communication)
        3. All_gather the concatenated tensor
        4. Compute top-K on gathered values
        5. Re-index: gather the corresponding global indices using top-K positions

        Args:
            local_values: Local top-K values (seq, batch, local_k)
            local_indices: Local top-K indices in [0, local_vocab_size), fits in uint16
            k: Target number of top elements
            local_vocab_size: Size of the local vocab shard (vocab_size / tp_size)

        Returns:
            Tuple of (global_values, global_indices)
        """
        # Compute vocab offset for this TP rank
        # Local indices are in [0, local_vocab_size)
        # Global indices are in [tp_rank * local_vocab_size, (tp_rank + 1) * local_vocab_size)
        vocab_offset = self.tp_rank * local_vocab_size
        global_indices = local_indices + vocab_offset

        # Concatenate for a single all_gather to minimize communication
        # float32 can exactly represent integers up to 2^24, sufficient for 17-bit vocab indices
        # Stack along new dimension: shape becomes (seq, batch, local_k, 2)
        combined = torch.stack([local_values.float(), global_indices.float()], dim=-1)

        # All_gather across TP ranks
        gathered_list = [torch.empty_like(combined) for _ in range(self.tp_size)]
        dist.all_gather(gathered_list, combined, group=self.tp_group)

        # Concatenate gathered results: (seq, batch, tp_size * local_k, 2)
        gathered = torch.cat(gathered_list, dim=-2)

        # Split back into values and indices: (seq, batch, tp_size * local_k)
        gathered_values = gathered[..., 0].to(local_values.dtype)
        gathered_indices = gathered[..., 1].to(local_indices.dtype)

        # Compute final global top-K
        topk_values, topk_positions = torch.topk(gathered_values, k, dim=-1)

        # Re-index: gather the corresponding global indices using top-K positions
        topk_global_indices = torch.gather(gathered_indices, -1, topk_positions)

        return topk_values, topk_global_indices

    def _write_to_disk(
        self,
        values_list: List[torch.Tensor],
        indices_low_list: List[torch.Tensor],
        bit_17_list: List[torch.Tensor],
    ) -> None:
        """Write all microbatch data to disk in a single I/O call.

        File format: serialized dict with:
        - values: list of bfloat16 tensors of log-probabilities (one per microbatch)
        - indices_low: list of uint16 tensors (lower 16 bits of vocab indices)
        - bit_17: list of bool tensors (17th bit, same shape as indices_low)
        """
        # Serialize all tensors together
        buffer = io.BytesIO()
        torch.save({
            'values': values_list,
            'indices_low': indices_low_list,
            'bit_17': bit_17_list,
        }, buffer)
        data = buffer.getvalue()

        # Optional compression
        suffix = ".pt"
        if self._zstd_compressor is not None:
            data = self._zstd_compressor.compress(data)
            suffix += ".zst"

        # Generate filename
        args = get_args()
        iteration = getattr(args, 'curr_iteration', None)
        if iteration is None:
            iteration = getattr(args, 'iteration')
        folder, filename = _format_folder_and_filename(
            self.save_dir, iteration, self.tp_rank, self.cp_rank, self.dp_rank, suffix
        )

        # Write to disk
        os.makedirs(folder, exist_ok=True)
        with open_file(os.path.join(folder, filename), "wb") as f:
            f.write(data)


def load_log_probs_by_rank(
    save_dir: str,
    iteration: int,
    tp_rank: int,
    cp_rank: int,
    dp_rank: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Load a saved log-probs file by specifying the root folder, iteration, and parallel ranks.

    Finds the file matching the naming convention used by LogitsSaverHooks:
    - Iteration folder pattern: logprobs_iter{iteration}
    - Filename pattern: tp{tp_rank}_cp{cp_rank}_dp{dp_rank}.pt or .pt.zst

    Args:
        save_dir: Root path containing all log-probs iteration folders
        iteration: Training iteration number
        tp_rank: Tensor parallel rank
        cp_rank: Context parallel rank
        dp_rank: Data parallel rank

    Returns:
        Tuple of (log_prob_values_list, indices_list)
        where each list contains one tensor per microbatch (in order).
        Values are log-probabilities; indices are reconstructed full int64 global indices.

    Raises:
        FileNotFoundError: If no matching folder or file is found
    """
    # Use shared formatting function to get folder and base filename
    folder, base_filename = _format_folder_and_filename(
        save_dir, iteration, tp_rank, cp_rank, dp_rank,
    )
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Iteration folder not found: {folder}")

    # Check for both compressed and uncompressed files
    filepath = None
    for suffix in [".pt.zst", ".pt"]:
        candidate = os.path.join(folder, base_filename + suffix)
        if os.path.exists(candidate):
            filepath = candidate
            break
    if filepath is None:
        raise FileNotFoundError(
            f"No log-probs file found for tp_rank={tp_rank}, cp_rank={cp_rank}, "
            f"dp_rank={dp_rank} in folder: {folder}"
        )

    # Check for compression
    if filepath.endswith('.zst'):
        try:
            import zstandard
        except ModuleNotFoundError:
            raise RuntimeError("zstandard package required to read compressed log-probs files")

        with open_file(filepath, 'rb') as f:
            decompressor = zstandard.ZstdDecompressor()
            data = decompressor.decompress(f.read())
    else:
        with open_file(filepath, 'rb') as f:
            data = f.read()

    # Load tensors
    tensors = torch.load(io.BytesIO(data), weights_only=True)

    # Reconstruct full indices for each microbatch
    indices_list = [
        _unpack_indices(low, bit17)
        for low, bit17 in zip(tensors['indices_low'], tensors['bit_17'])
    ]

    return tensors['values'], indices_list
