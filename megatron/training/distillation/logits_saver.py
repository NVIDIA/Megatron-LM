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

import io
import json
import logging
import os
import tarfile
from collections import OrderedDict
from types import MethodType
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import zstandard

from megatron.core import parallel_state
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.msc_utils import MultiStorageClientFeature
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.training import get_args, get_tensorboard_writer
from megatron.training.utils import print_rank_0
from megatron.training.distillation.utils_logits import (
    LOGPROBS_TAR_MEMBER_SUFFIX,
    META_TAR_MEMBER,
    batched_tar_filename,
    compute_dataset_hash,
    get_current_iteration,
    is_remote_storage_path,
    open_logit_file,
    pack_indices,
    storage_makedirs,
    storage_move,
)

logger = logging.getLogger(__name__)

# Module-level reference so checkpoint/shutdown code can call flush().
_ACTIVE_LOGITS_SAVER: Optional["LogitsSaverHooks"] = None


def get_logits_saver() -> Optional["LogitsSaverHooks"]:
    """Return the active :class:`LogitsSaverHooks` instance, or *None*."""
    return _ACTIVE_LOGITS_SAVER


_MAX_VOCAB_SIZE = 2 ** 17  # 131072 - maximum supported vocab size

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
    - Having TP rank 0 save the full top-K for each CP/DP rank

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
    CP and DP ranks are stored in the tar filename for flexibility with multi-digit values.

    Iteration data is accumulated in memory and flushed as a tar archive at
    checkpoint time via the async checkpoint queue.  Call :meth:`flush` at
    shutdown to write any remaining buffered data synchronously.

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
        save_dtype: str = 'fp16',
    ):
        assert k > 0, "Number of top log-probabilities to save must be positive"
        assert save_dir is not None, "Save directory must be provided"
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
            },
        }
        self._meta_bytes: bytes = json.dumps(
            self.metadata_dict, sort_keys=False, separators=(',', ':'),
        ).encode("utf-8")

        # Hook states – store already-processed top-K results (not full logits)
        self._accumulated_results: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._hook_handles: List[Any] = []
        self._loss_overrides: List[Tuple[torch.nn.Module, Any]] = []

        # Batched tar state — accumulates until checkpoint time.
        self._pending_writes: OrderedDict[int, bytes] = OrderedDict()

        # Top-P logging: per-microbatch kept counts (populated by _apply_topp_truncation)
        self._topp_kept_counts: List[float] = []

        # Create save directory if needed
        storage_makedirs(self.save_dir, exist_ok=True)

        # Register as the active saver so checkpoint code can flush
        global _ACTIVE_LOGITS_SAVER
        _ACTIVE_LOGITS_SAVER = self

    def _forward_hook(
        self,
        module: torch.nn.Module,
        input: Any,
        output: Tuple[torch.Tensor, ...],
    ) -> None:
        """Capture top-K log-probs for one output-layer forward.

        This hook should be registered on the module that outputs logits.
        Each microbatch is processed immediately to extract only the small
        top-K values and indices, avoiding storage of full vocab-sized logits
        across microbatches.
        """
        if not module.training:
            return

        if self._curr_mtp_passes == self._mtp_num_layers:
            # Main output logits come after MTP logits.
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

    def attach_hooks(self, model: LanguageModule) -> None:
        """Attach logits-saving hooks and skip expensive LM loss computation.

        Args:
            model: Model to instrument.
        """
        fwd_handle = model.output_layer.register_forward_hook(self._forward_hook)
        self._hook_handles.append(fwd_handle)
        self._override_language_model_loss(model)

    def _override_language_model_loss(self, model: LanguageModule) -> None:
        """Replace LM loss with a zero-valued tensor that preserves gradient edges."""
        def _compute_zero_language_model_loss(_self, _labels, logits):
            return (logits * 0).sum(dim=-1).transpose(0, 1).contiguous()

        original_loss = model.compute_language_model_loss
        self._loss_overrides.append((model, original_loss))
        model.compute_language_model_loss = MethodType(_compute_zero_language_model_loss, model)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        for model_chunk, original_loss in self._loss_overrides:
            model_chunk.compute_language_model_loss = original_loss
        self._loss_overrides.clear()

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

        self._buffer_iteration(all_values, all_indices_low, all_high_bits)

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
        indices_low, high_bit = pack_indices(global_indices)

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

    def _buffer_iteration(
        self,
        values_list: List[torch.Tensor],
        indices_low_list: List[torch.Tensor],
        bit_17_list: List[torch.Tensor],
    ) -> None:
        """Serialize microbatch data and buffer for async flush at checkpoint time.

        File format: serialized dict with:
        - values: list of tensors of log-probabilities (one per microbatch)
          in ``self._save_dtype`` (default ``torch.float16``)
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
        iteration = get_current_iteration()
        self._pending_writes[iteration] = data

    # ------------------------------------------------------------------
    #  Flush helpers
    # ------------------------------------------------------------------

    def take_pending_data(self) -> Tuple[str, "OrderedDict[int, bytes]", bytes, bool]:
        """Take ownership of buffered data for async flush at checkpoint time.

        Returns:
            Tuple of (tar_path, writes, meta_bytes, msc_enabled).  If there is
            no pending data (e.g. non-TP-rank-0), ``writes`` will be an empty
            OrderedDict and ``tar_path`` will be an empty string.
        """
        # NOTE: We need to re-enabled MSC in the async saving process, so we pass this flag.
        msc_enabled = MultiStorageClientFeature.is_enabled()

        if not self._pending_writes:
            return ("", OrderedDict(), self._meta_bytes, msc_enabled)

        writes = self._pending_writes
        self._pending_writes = OrderedDict()

        last_iter = max(writes.keys())
        tar_filename = batched_tar_filename(
            self.cp_rank, self.dp_rank, last_iter,
        )
        tar_path = os.path.join(self.save_dir, tar_filename)
        print_rank_0(f"Handing off {len(writes)} logit iterations for async flush")
        return (tar_path, writes, self._meta_bytes, msc_enabled)

    @staticmethod
    def _write_batched_tar(
        tar_path: str,
        writes: "OrderedDict[int, bytes]",
        meta_bytes: bytes,
        msc_enabled: bool = False,
    ) -> None:
        """Write a tar archive containing multiple iterations.

        Called in the async checkpoint background process.  Early-returns
        when there is nothing to write (non-TP-rank-0 ranks).

        The dataset-identity metadata is written as the **first** member
        (named :data:`META_TAR_MEMBER`) so the student-side tar reader
        sees it before any payload members and can
        verify alignment before any iteration data is decoded.

        Each subsequent member is named ``{iteration}.pt.zst`` so that
        the student-side tar reader can stream iterations by member name.
        """
        if not writes:
            return
        if msc_enabled:
            # NOTE: MSC is not enabled in the async saving process by default.
            MultiStorageClientFeature.enable()

        storage_makedirs(os.path.dirname(tar_path), exist_ok=True)
        write_path = tar_path if is_remote_storage_path(tar_path) else f"{tar_path}.tmp"
        compressor = zstandard.ZstdCompressor(level=3)

        with open_logit_file(write_path, "wb") as stream:
            with tarfile.open(fileobj=stream, mode="w") as tar:
                info = tarfile.TarInfo(name=META_TAR_MEMBER)
                info.size = len(meta_bytes)
                tar.addfile(info, io.BytesIO(meta_bytes))

                for iteration, data in writes.items():
                    data = compressor.compress(data)
                    member_name = f"{iteration}{LOGPROBS_TAR_MEMBER_SUFFIX}"
                    info = tarfile.TarInfo(name=member_name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
        if write_path != tar_path:
            storage_move(write_path, tar_path)

    def flush(self) -> None:
        """Synchronously flush any remaining buffered data to disk."""
        tar_path, writes, meta_bytes, _ = self.take_pending_data()
        self._write_batched_tar(tar_path, writes, meta_bytes)
