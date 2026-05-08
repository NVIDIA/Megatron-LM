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
import hashlib
import io
import json
import logging
import os
import re
import tarfile
import warnings
from collections import OrderedDict
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

try:
  import zstandard
except ImportError:
  zstandard = None

from megatron.core import parallel_state
from megatron.core.msc_utils import open_file
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.training import get_args, get_tensorboard_writer
from megatron.training.utils import get_blend_and_blend_per_split, print_rank_0

logger = logging.getLogger(__name__)

# Module-level reference so checkpoint/shutdown code can call flush().
_ACTIVE_LOGITS_SAVER: Optional["LogitsSaverHooks"] = None


def get_logits_saver() -> Optional["LogitsSaverHooks"]:
    """Return the active :class:`LogitsSaverHooks` instance, or *None*."""
    return _ACTIVE_LOGITS_SAVER


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

# Name of the metadata member written as the first entry of every batched
# tar.  Contains the dataset-identity hash and the fields that produced it,
# so the student-side loader can verify alignment per-tar via WebDataset.
META_TAR_MEMBER = "_meta.json"


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


def _blend_identifiers(args: Any) -> Dict[str, Any]:
    """Build a path-agnostic representation of the training data blend.

    Reuses :func:`megatron.training.utils.get_blend_and_blend_per_split`
    to resolve all of Megatron's blend-source variants (``data_path``,
    ``data_args_path``, ``train_data_path`` / ``valid_data_path`` /
    ``test_data_path``, ``per_split_data_args_path``) into the canonical
    ``(prefixes, weights)`` shape, then normalises each prefix to its
    basename (without extension) so the same teacher cache can be reused
    across machines that mount the data at different locations.

    Returns ``{"mock": True}`` when no blend is configured (e.g. mock
    data runs).
    """
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    def _normalise(blend_tuple) -> Optional[List[List[Any]]]:
        if blend_tuple is None:
            return None
        prefixes, weights = blend_tuple
        if weights is None:
            weights = [1.0] * len(prefixes)
        return [
            [float(w), os.path.splitext(os.path.basename(str(p)))[0]]
            for w, p in zip(weights, prefixes)
        ]

    if blend is not None:
        return {"kind": "blend", "blend": _normalise(blend)}
    if blend_per_split is not None:
        # Index 0 is the train split (see ``megatron.core.datasets.utils.Split``).
        # Only the train blend influences which teacher logits are produced.
        return {"kind": "blend_per_split", "train": _normalise(blend_per_split[0])}
    return {"kind": "mock", "mock": True}


def compute_dataset_hash() -> Tuple[str, Dict[str, Any]]:
    """Compute the dataset-identity hash for the current training run.

    Modeled after Megatron's ``unique_description_hash`` mechanism in
    :mod:`megatron.core.datasets.megatron_dataset`: build a deterministic
    ``OrderedDict`` of identifying fields, JSON-serialise it, and MD5-hash
    the result.  Reads the current run configuration via :func:`get_args`.

    The fields included are exactly those that determine the global
    sample stream itself: ``seed``, ``sequence_length``, ``train_samples``
    (with a fall-back to ``train_iters * global_batch_size``), and the
    data ``blend``.  Tokenizer name, split string, and ``global_batch_size``
    are intentionally excluded — none of them change the underlying
    sample stream; leaving GBS out in particular keeps the cache valid
    for future GBS-rescaling on the loader side.

    Returns:
        ``(md5_hex, identifiers_dict)`` – the hash and the dict it was
        computed from (useful for diagnostics on mismatch).
    """
    args = get_args()
    train_samples = getattr(args, 'train_samples', None)
    if train_samples is None:
        train_iters = getattr(args, 'train_iters', None)
        global_batch_size = getattr(args, 'global_batch_size', None)
        if train_iters is not None and global_batch_size is not None:
            train_samples = int(train_iters) * int(global_batch_size)

    identifiers = OrderedDict()
    identifiers["seed"] = getattr(args, 'seed', None)
    identifiers["sequence_length"] = getattr(args, 'seq_length', None)
    identifiers["train_samples"] = train_samples
    identifiers["blend"] = _blend_identifiers(args)

    description = json.dumps(identifiers, sort_keys=False, separators=(',', ':'))
    md5_hex = hashlib.md5(
        description.encode("utf-8"), usedforsecurity=False
    ).hexdigest()
    return md5_hex, dict(identifiers)


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

    When ``p`` is set, a top-P (nucleus) mask is applied after global top-K
    selection.  The K dimension is truncated to the maximum per-token nucleus
    size (reducing the amount of data written to disk), and tokens whose
    nucleus is smaller than that maximum have their trailing entries masked
    with ``-1e4`` value / ``-1`` index sentinels.  These sentinels are
    compatible with :func:`topk_kl_div` in ``cached_logits_loss.py``: the
    ``-1`` index falls outside every TP rank's vocab range so the entry is
    excluded from the KL sum, and ``exp(-1e4) = 0`` so masked teacher
    probability is effectively zero.  At least ``min_k`` entries are always
    kept per token.

    Indices are stored efficiently: uint16 for lower 16 bits + separate high bit tensor.
    tp_rank and dp_rank are stored in the filename for flexibility with multi-digit values.

    When ``flush_interval`` > 1, iteration data is accumulated in memory and
    flushed as a single tar archive every *flush_interval* iterations via a
    background thread, reducing inode usage and avoiding blocking the training
    loop.  Call :meth:`flush` at checkpoint / shutdown to write any remaining
    buffered data.

    Args:
        save_dir: Directory to save log-prob files
        k: Number of top log-probabilities to save globally
        p: Optional top-P (nucleus) threshold applied after global top-K
            selection.  When set, only the smallest set of leading entries
            per token whose cumulative probability mass reaches ``p`` is
            kept; remaining entries are masked with a ``-1e4`` value
            sentinel and ``-1`` index sentinel.
        min_k: Minimum number of entries kept per token when top-P masking
            is active, regardless of cumulative mass.  Defaults to 1.
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
        save_dir: str,
        k: int,
        *,
        p: Optional[float] = None,
        min_k: int = 1,
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

        if p is not None and not (0.0 < p <= 1.0):
            raise ValueError(
                f"p must be in (0, 1] or None, got {p}"
            )
        if min_k < 1:
            raise ValueError(
                f"min_k must be >= 1, got {min_k}"
            )

        self.save_dir = save_dir
        self.k = k
        self.p = p
        self.min_k = min_k
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

        # Dataset-identity hash + serialised metadata, written as the
        # first member of every batched tar so the student loader can
        # verify alignment per-tar without an extra tarfile.open.
        self.dataset_hash, self._dataset_identifiers = compute_dataset_hash()
        self.metadata_dict: Dict[str, Any] = {
            "hash": self.dataset_hash,
            "identifiers": self._dataset_identifiers,
            "saver": {
                "k": self.k,
                "p": self.p,
                "min_k": self.min_k,
                "save_dtype": save_dtype,
                "compress_zstd": bool(compress_zstd),
                "flush_interval": self.flush_interval,
            },
        }
        self._meta_bytes: bytes = json.dumps(
            self.metadata_dict, sort_keys=False, separators=(',', ':'),
        ).encode("utf-8")

        # Hook states – store already-processed top-K results (not full logits)
        self._accumulated_results: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._hook_handles: List[Any] = []

        # Zstd compression setup
        if compress_zstd and zstandard is not None:
            self._zstd_compressor = zstandard.ZstdCompressor(level=3)
        else:
            self._zstd_compressor = None
            if compress_zstd:
                warnings.warn(
                    "zstandard package not found; disabling zstd compression for log-probs."
                )

        # Batched tar state (active when flush_interval > 1)
        self._pending_writes: OrderedDict[int, Tuple[bytes, str]] = OrderedDict()
        self._flush_executor: Optional[concurrent.futures.ThreadPoolExecutor] = (
            concurrent.futures.ThreadPoolExecutor(max_workers=1)
            if self.flush_interval > 1
            else None
        )
        self._flush_futures: List[concurrent.futures.Future] = []

        # Top-P logging: per-microbatch kept counts (populated by _apply_topp_truncation)
        self._topp_kept_counts: List[float] = []

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

        if self._topp_kept_counts:
            # Log the average number of top-P kept tokens per microbatch to tensorboard
            avg_kept = sum(self._topp_kept_counts) / len(self._topp_kept_counts)
            self._topp_kept_counts.clear()
            if (writer := get_tensorboard_writer()) is not None:
                writer.add_scalar('avg-logprobs-kept', avg_kept, get_current_iteration())

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

        if self.p is not None:
            global_values, global_indices = self._apply_topp_truncation(
                global_values, global_indices,
            )

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

    def _apply_topp_truncation(
        self,
        values: torch.Tensor,
        indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply top-P (nucleus) mask to already-sorted top-K log-probs.

        Keeps the smallest set of leading entries per token whose
        cumulative probability mass is at least ``self.p``, with a floor
        of ``self.min_k`` kept entries.

        Because the number of surviving entries varies per token, the K
        dimension is truncated to the *maximum* kept count across all
        tokens in the microbatch.  Tokens whose individual nucleus is
        smaller than that maximum have their trailing entries masked with
        ``-1e4`` value / ``-1`` index sentinels.  These sentinels are
        compatible with :func:`topk_kl_div`: the ``-1`` index falls
        outside every TP rank's vocab range so the entry is excluded from
        the KL sum, and ``exp(-1e4) = 0`` so masked teacher probability
        is effectively zero.

        Args:
            values: Top-K log-prob values, sorted descending along the
                last dimension.  Shape ``(seq, batch, K)``.
            indices: Corresponding global vocab indices (int64).

        Returns:
            Tuple of ``(values, indices)`` with K dimension truncated to
            the maximum per-token nucleus size, and trailing out-of-nucleus
            entries masked with sentinels.
        """
        probs = values.float().exp()
        cumprobs = probs.cumsum(dim=-1)
        # Standard nucleus rule: keep entry i iff the cumulative mass
        # *before* it is < p.  This guarantees we include the entry
        # that crosses the threshold and always keeps top-1.
        keep_mask = (cumprobs - probs) < self.p

        k = values.size(-1)
        min_keep = min(self.min_k, k)
        arange = torch.arange(k, device=values.device)
        keep_mask = keep_mask | (arange < min_keep)

        kept_per_token = keep_mask.sum(dim=-1)
        # For logging purposes
        self._topp_kept_counts.append(kept_per_token.float().mean().item())

        # Truncate to reduce storage
        max_kept = int(kept_per_token.max().item())
        values = values[..., :max_kept]
        indices = indices[..., :max_kept]
        keep_mask = keep_mask[..., :max_kept]

        # Mask out-of-nucleus entries
        values = torch.where(keep_mask, values, -1e4)
        indices = torch.where(keep_mask, indices, -1)

        return values, indices

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
            LogitsSaverHooks._write_batched_tar, tar_path, writes, self._meta_bytes,
        )
        self._flush_futures.append(future)

        print_rank_0(f"Flushing {len(writes)} logit iterations to disk")

    @staticmethod
    def _write_batched_tar(
        tar_path: str,
        writes: "OrderedDict[int, Tuple[bytes, str]]",
        meta_bytes: bytes,
    ) -> None:
        """Write a tar archive containing multiple iterations (runs in background thread).

        The dataset-identity metadata is written as the **first** member
        (named :data:`META_TAR_MEMBER`) so the student-side WebDataset
        loader sees it as the very first sample of the shard and can
        verify alignment before any iteration data is decoded.

        Each subsequent member is named ``{iteration}{suffix}`` so that
        WebDataset groups them as individual samples keyed by iteration
        number.
        """
        with tarfile.open(tar_path, "w") as tar:
            info = tarfile.TarInfo(name=META_TAR_MEMBER)
            info.size = len(meta_bytes)
            tar.addfile(info, io.BytesIO(meta_bytes))

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
    expected_hash: Optional[str] = None,
) -> Optional[Tuple[int, List[torch.Tensor], List[torch.Tensor]]]:
    """WebDataset map function: decode a single tar member into tensors.

    Handles both ``.pt`` and ``.pt.zst`` extensions produced by
    :class:`LogitsSaverHooks` when ``flush_interval > 1``.

    The first member of every batched tar is the :data:`META_TAR_MEMBER`
    sample (see :meth:`LogitsSaverHooks._write_batched_tar`).  When that
    sample is encountered, its parsed contents are compared against
    *expected_hash* (if provided) and the function returns ``None`` to
    signal the caller to skip it.  This makes per-tar dataset-identity
    verification automatic for any consumer of the WebDataset pipeline
    without requiring an extra filter step.

    Args:
        sample: WebDataset sample dict with ``__key__`` and one data field.
        expected_hash: Dataset-identity hash the student run expects every
            tar to advertise in its ``_meta.json`` member.  When ``None``,
            verification is skipped (legacy data, debugging).

    Returns:
        ``(iteration, values_list, indices_list)`` for iteration samples,
        or ``None`` for the per-tar metadata sample (caller should skip).
    """
    key: str = sample["__key__"]

    if key == META_TAR_MEMBER.split(".")[0]:
        saved_hash = json.loads(sample["json"]).get("hash")
        if expected_hash is not None and saved_hash != expected_hash:
            raise RuntimeError(
                f"Teacher tar {sample.get('__url__')} was saved with hash "
                f"{saved_hash} but the current student run has hash "
                f"{expected_hash}. Data does not align!"
            )
        return None

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
