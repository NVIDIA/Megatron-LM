# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
Offline knowledge distillation loss using cached teacher top-K log-probabilities.

This module provides a loss function that loads pre-computed teacher top-K
log-probabilities from disk (as saved by ``LogitsSaverHooks`` in ``logits_saver.py``)
and computes forward KL divergence against live student logits.

Design highlights
-----------------
* **Sparse KL** – each rank uses only the teacher top-K positions that fall in
  its local vocab shard, avoiding a full-vocab-sized dense teacher tensor.
* **TP-aware student normalisation** – the student log-probabilities are
  normalised across tensor-parallel vocab shards before gathering the sparse
  teacher top-K positions.
* **Custom tar streaming + DataLoader** – a :class:`torch.utils.data.DataLoader`
  with ``pin_memory=True`` streams batched tar shards so that disk I/O overlaps
  with GPU compute, and the CPU→GPU copy can be issued with
  ``non_blocking=True``.
* **Own-rank loading only** – each rank loads only cp-dp shards matching its
  parallelism coordinates (no cross-rank file I/O beyond DP resharding).
* **Multi-threaded decode pipeline** – the tar ``IterableDataset`` runs
  inside a single DataLoader worker (to preserve shard iteration order) and
  parallelises the CPU-bound zstd decompression + ``torch.load`` + index
  unpacking across an internal ``ThreadPoolExecutor``.  This removes the
  single-thread bottleneck that dominates per-iteration loader cost when each
  saved tar contains a large multi-microbatch blob.

Assumptions
-----------
* The student run uses the **same random seed** and data pipeline as the
  teacher run that produced the cached log-probs, so the microbatch ordering
  matches.
* **Same CP layout** (``cp_rank``) as the teacher run.  The DP size may differ:
  both upscaling (saved < current) and downscaling (saved > current) are
  supported provided one evenly divides the other.
* **Same micro-batch size** as the teacher run.
* Teacher log-probs were saved by ``LogitsSaverHooks`` (see ``logits_saver.py``).

Usage example
-------------
::

    from megatron.training.distillation import LossFuncCallable

    # Instantiate once at the start of training
    loss_func = LossFuncCallable(
        logprobs_dir="/data/teacher_logprobs",
        kd_loss_alpha=0.5,
    )

    # Pass as the loss_func to Megatron's training loop:
    #   loss, num_tokens, report = loss_func(loss_mask, output_tensor, model)
"""

import concurrent.futures
import logging
import warnings
from collections import deque
from typing import Any, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
import torch.utils.data

from megatron.core import parallel_state
from megatron.core._rank_utils import safe_get_rank
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.training.distillation.utils_logits import (
  BATCHED_TAR_RE,
  CACHED_LOGITS_LOGPROB_SENTINEL,
  LogprobsTarEntry,
  TarShardPrefetcher,
  batched_tar_prefix,
  compute_dataset_hash,
  decode_logprobs_payload,
  detect_saved_dp_size,
  get_current_iteration,
  iter_logprobs_tar_entries,
  is_remote_storage_path,
  storage_basename,
  storage_glob_with_caching,
  sorted_batched_tars,
)
from megatron.training.utils import print_rank_0

logger = logging.getLogger(__name__)

# Module-level reference so the training loss can consume logits captured
# from the model's output layer without requiring model-specific attributes.
_ACTIVE_STUDENT_LOGITS_CAPTURE: Optional["StudentLogitsCapture"] = None


def get_student_logits_capture() -> Optional["StudentLogitsCapture"]:
    """Return the active :class:`StudentLogitsCapture` instance, or *None*."""
    return _ACTIVE_STUDENT_LOGITS_CAPTURE


class StudentLogitsCapture:
    """Forward-hook helper that keeps the latest differentiable student logits."""

    def __init__(self):
        self._logits: Optional[torch.Tensor] = None
        self._hook_handles: List[Any] = []

    def attach_hooks(self, model: LanguageModule) -> None:
        """Attach forward hooks to the model's output layer."""
        handle = model.output_layer.register_forward_hook(self._capture_logits)
        self._hook_handles.append(handle)

        global _ACTIVE_STUDENT_LOGITS_CAPTURE
        _ACTIVE_STUDENT_LOGITS_CAPTURE = self

    def _capture_logits(
        self,
        module: torch.nn.Module,
        input: Any,
        output: Any,
    ) -> None:
        if not module.training:
            return
        # NOTE: Assumes main head runs after MTP layers, overwriting this value prior to pop().
        self._logits = output[0] if isinstance(output, tuple) else output

    def pop(self) -> torch.Tensor:
        """Return captured logits and clear the reference for the next forward."""
        if self._logits is None:
            raise RuntimeError(
                "No student logits were captured for cached-logits KD. "
                "Attach StudentLogitsCapture to the model output layer during setup."
            )
        logits = self._logits
        self._logits = None
        return logits

    def remove_hooks(self) -> None:
        """Remove all registered hooks and clear captured state."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._logits = None


def _compute_dp_remapping(
    logprobs_dir: str,
    dp_rank: int,
    dp_size: int,
) -> Tuple[List[int], int, int, int]:
    """Compute the DP rank remapping when loading data saved with a different DP size.

    Scans *logprobs_dir* via :func:`detect_saved_dp_size` to infer the
    saved DP world size.  When no batched tars are found, returns the
    identity mapping ``([dp_rank], 0, 1, dp_size)``.

    Data is distributed across DP ranks in a round-robin (wrapping)
    fashion: with *dp_size_saved* ranks and *G* global microbatches,
    saved rank *d* holds microbatch indices ``d, d + dp_size_saved,
    d + 2·dp_size_saved, …``.

    **Upscaling** (``dp_size_saved < dp_size``): each saved rank's data
    is shared by ``dp_size // dp_size_saved`` current ranks that stride
    through its microbatches.

    **Downscaling** (``dp_size_saved > dp_size``): each current rank
    loads from ``dp_size_saved // dp_size`` saved files and interleaves
    their microbatches to reconstruct its share of the global ordering.

    Returns:
        ``(source_dp_ranks, sub_rank, dp_ratio, dp_size_saved)`` where

        - *source_dp_ranks* is the list of saved-file DP ranks to read.
          Length 1 for identity or upscaling, length > 1 for downscaling.
        - *sub_rank* selects the strided microbatch slice within each
          source file (non-zero only when upscaling).
        - *dp_ratio* is ``dp_size / dp_size_saved``: > 1 when upscaling
          (used as the stride in microbatch slicing), < 1 when downscaling
          (informational only — the multi-source interleave path is used
          instead), and 1 for the identity case.
        - *dp_size_saved* is the DP world size used when the data was
          written.
    """
    dp_size_saved = detect_saved_dp_size(logprobs_dir)

    if dp_size_saved is None:
        return [dp_rank], 0, 1, dp_size
    if dp_size_saved == dp_size:
        return [dp_rank], 0, 1, dp_size_saved

    if dp_size_saved < dp_size:
        # Upscaling: multiple current ranks share one saved file
        if dp_size % dp_size_saved != 0:
            raise ValueError(
                f"Current DP size ({dp_size}) is not an exact multiple of "
                f"saved DP size ({dp_size_saved})."
            )
        dp_ratio = dp_size // dp_size_saved
        mapped_dp_rank = dp_rank % dp_size_saved
        sub_rank = dp_rank // dp_size_saved
        return [mapped_dp_rank], sub_rank, dp_ratio, dp_size_saved

    # Downscaling: each current rank loads from multiple saved files
    if dp_size_saved % dp_size != 0:
        raise ValueError(
            f"Saved DP size ({dp_size_saved}) is not an exact multiple of "
            f"current DP size ({dp_size})."
        )
    num_sources = dp_size_saved // dp_size
    source_dp_ranks = [dp_rank + i * dp_size for i in range(num_sources)]
    dp_ratio = dp_size / dp_size_saved  # e.g. 0.5 when halving DP
    return source_dp_ranks, 0, dp_ratio, dp_size_saved


# ---------------------------------------------------------------------------
#  Dataset – streaming batched tar shards
# ---------------------------------------------------------------------------

class TeacherTarDataset(torch.utils.data.IterableDataset):
    """Streaming dataset that reads teacher log-probs from batched tar shards.

    Supports a single on-disk layout: ``cp{C}_dp{D}__{B}.tar``.  Shards are
    saved by TP rank 0 with the full top-K; every TP rank loads the same
    cp-dp tar.

    **DP resharding** is supported in both directions:

    * **Upscaling** (saved DP < current DP) — multiple current ranks share
      one saved file and stride through its microbatches.
    * **Downscaling** (saved DP > current DP) — each current rank loads
      from multiple saved files and interleaves their microbatches to
      reconstruct its share of the global microbatch ordering.

    **Shard discovery**: after known shards are exhausted, discovery refreshes
    the directory listing so a concurrent writer can publish more data. Remote
    refreshes are issued by rank 0 and broadcast to avoid object-store list
    storms.

    Yields ``(values_list, indices_list)``.
    """

    def __init__(
        self,
        logprobs_dir: str,
        cp_rank: int,
        dp_rank: int,
        dp_size: int,
        start_iteration: int = 0,
        decode_threads: int = 1,
        decode_lookahead: Optional[int] = None,
        msc_prefetch_depth: int = 2,
        ignore_hash: bool = False,
    ):
        self.logprobs_dir = logprobs_dir
        self.cp_rank = cp_rank
        self.dp_rank = dp_rank
        self.start_iteration = start_iteration

        self._remote_logprobs = is_remote_storage_path(logprobs_dir)
        self._msc_prefetch_depth = max(0, msc_prefetch_depth)

        # Per-tar dataset-identity verification: each tar stream validates its
        # leading _meta.json before yielding any payload data.
        self._expected_hash = None if ignore_hash else compute_dataset_hash()[0]

        self._decode_threads = max(1, decode_threads)
        if decode_lookahead is None:
            decode_lookahead = max(2 * self._decode_threads, 4)
        self._decode_lookahead = max(1, decode_lookahead)
        if self._decode_threads > 1:
            print_rank_0(
                f"Teacher logits decode: using {self._decode_threads} decode threads "
                f"(lookahead={self._decode_lookahead}) in the logits loader worker"
            )
        if self._remote_logprobs and self._msc_prefetch_depth > 0:
            print_rank_0(
                f"Teacher logits remote tar prefetch: {self._msc_prefetch_depth} "
                "shard(s) ahead"
            )

        # DP remapping: detect saved DP size and compute mapping
        self._source_dp_ranks, self._sub_rank, self._dp_ratio, dp_size_saved = (
            _compute_dp_remapping(logprobs_dir, dp_rank, dp_size)
        )
        if len(self._source_dp_ranks) > 1:
            print_rank_0(
                f"DP downscaling: dp_rank {dp_rank} loads from saved dp_ranks "
                f"{self._source_dp_ranks} (saved_dp_size {dp_size_saved}, "
                f"current_dp_size {dp_size})"
            )
        elif self._dp_ratio > 1:
            print_rank_0(
                f"DP upscaling: dp_rank {dp_rank} -> mapped_dp_rank "
                f"{self._source_dp_ranks[0]} (sub_rank {self._sub_rank}, "
                f"saved_dp_size {dp_size_saved}, current_dp_size {dp_size})"
            )

    def _discover_shards(self, already_processed: set, dp_rank: int) -> list:
        """Glob for new cp-dp batched tar shards.

        Args:
            already_processed: Set of URLs already processed (to skip).
            dp_rank: Saved DP rank to discover shards for.
        """
        prefix = batched_tar_prefix(self.cp_rank, dp_rank)
        all_urls = sorted_batched_tars(
            storage_glob_with_caching(self.logprobs_dir, f"{prefix}*.tar", cached=False)
        )
        new_urls = []
        for url in all_urls:
            if url in already_processed:
                continue
            m = BATCHED_TAR_RE.match(storage_basename(url))
            if m and int(m.group("iter")) >= self.start_iteration:
                new_urls.append(url)
        return new_urls

    def _slice_microbatches(
        self,
        values_list: List[torch.Tensor],
        indices_list: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Return only this rank's strided microbatch slice when DP ratio > 1."""
        if self._dp_ratio <= 1:
            return values_list, indices_list

        num_mb = len(values_list)
        if num_mb % self._dp_ratio != 0:
            raise ValueError(
                f"Saved microbatch count ({num_mb}) is not divisible by "
                f"DP ratio ({self._dp_ratio}). Cannot evenly split "
                f"microbatches across remapped DP ranks."
            )
        return (
            values_list[self._sub_rank :: self._dp_ratio],
            indices_list[self._sub_rank :: self._dp_ratio],
        )

    @staticmethod
    def _interleave_microbatches(
        all_values: List[List[torch.Tensor]],
        all_indices: List[List[torch.Tensor]],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Interleave microbatches from multiple source dp_ranks (downscaling).

        Given *N* sources each with *M* microbatches, produces a single
        list of *N·M* microbatches ordered so that the round-robin global
        microbatch assignment is reconstructed:
        ``src0_mb0, src1_mb0, …, srcN_mb0, src0_mb1, src1_mb1, …``
        """
        num_sources = len(all_values)
        num_mb = len(all_values[0])
        merged_values: List[torch.Tensor] = []
        merged_indices: List[torch.Tensor] = []
        for mb_idx in range(num_mb):
            for src_idx in range(num_sources):
                merged_values.append(all_values[src_idx][mb_idx])
                merged_indices.append(all_indices[src_idx][mb_idx])
        return merged_values, merged_indices

    # ------------------------------------------------------------------
    #  Shard discovery
    # ------------------------------------------------------------------

    def _not_found_error(self) -> FileNotFoundError:
        """Build a descriptive FileNotFoundError for missing shards."""
        dp_ranks = self._source_dp_ranks
        if len(dp_ranks) == 1:
            dp = dp_ranks[0]
            return FileNotFoundError(
                f"No batched tar shards for "
                f"cp{self.cp_rank}_dp{dp} "
                f"found at or after iteration {self.start_iteration} "
                f"in '{self.logprobs_dir}'"
            )
        return FileNotFoundError(
            f"No batched tar shards for source dp_ranks "
            f"{dp_ranks} (cp{self.cp_rank}) "
            f"found at or after iteration {self.start_iteration} "
            f"in '{self.logprobs_dir}'"
        )

    def _shard_groups(
        self,
        processed: set,
        prefetcher: TarShardPrefetcher,
    ) -> Iterator[Tuple[str, ...]]:
        """Yield source-DP shard URL groups as new shards are discovered.

        Each yielded tuple has one URL per source DP rank.  Handles dynamic
        re-discovery: when all current groups are exhausted, re-globs for new
        shards and yields the next batch.  Raises :class:`FileNotFoundError` if
        no shards exist at all; returns normally when no *new* shards appear.
        """
        while True:
            urls_per_src = [
                self._discover_shards(processed, src_dp)
                for src_dp in self._source_dp_ranks
            ]
            if not all(urls_per_src):
                if not processed:
                    raise self._not_found_error()
                return
            groups = list(zip(*urls_per_src))
            for group in prefetcher.iter_prefetched(groups):
                processed.update(group)
                yield group

    # ------------------------------------------------------------------
    #  Tar decode and DP resharding helpers
    # ------------------------------------------------------------------

    def _decode_entry(
        self,
        entry: LogprobsTarEntry,
    ) -> Tuple[int, List[torch.Tensor], List[torch.Tensor]]:
        """Decode one raw tar payload into logical iteration tensors."""
        values_list, indices_list = decode_logprobs_payload(entry.data)
        return entry.iteration, values_list, indices_list

    def _iter_entries_parallel(
        self,
        pool: concurrent.futures.ThreadPoolExecutor,
        entries: Iterator[LogprobsTarEntry],
    ) -> Iterator[Tuple[int, List[torch.Tensor], List[torch.Tensor]]]:
        """Yield decoded entries using a decode thread pool.

        The main thread streams raw tar members one at a time and submits the
        CPU-heavy zstd + ``torch.load`` + index unpacking work to *pool*.
        Results are yielded in tar order via a FIFO of futures.
        """
        pending: "deque[concurrent.futures.Future]" = deque()
        exhausted = False

        def submit_next() -> None:
            nonlocal exhausted
            if exhausted:
                return
            try:
                entry = next(entries)
            except StopIteration:
                exhausted = True
                return
            pending.append(pool.submit(self._decode_entry, entry))

        for _ in range(self._decode_lookahead):
            submit_next()
            if exhausted:
                break

        while pending:
            fut = pending.popleft()
            decoded = fut.result()
            submit_next()
            yield decoded

    def _iter_decoded_entries(
        self,
        pool: Optional[concurrent.futures.ThreadPoolExecutor],
        url: str,
    ) -> Iterator[Tuple[int, List[torch.Tensor], List[torch.Tensor]]]:
        """Yield decoded iteration payloads from one tar URL."""
        entries = iter_logprobs_tar_entries(
            url,
            start_iteration=self.start_iteration,
            expected_hash=self._expected_hash,
        )
        if pool is None:
            for entry in entries:
                yield self._decode_entry(entry)
        else:
            yield from self._iter_entries_parallel(pool, entries)

    def _interleave_decoded_group(
        self,
        decoded_group: Tuple[Tuple[int, List[torch.Tensor], List[torch.Tensor]], ...],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Interleave matching iterations from multiple source DP ranks."""
        all_values: List[List[torch.Tensor]] = []
        all_indices: List[List[torch.Tensor]] = []
        ref_iteration: Optional[int] = None
        for decoded in decoded_group:
            iteration, vals, inds = decoded
            if ref_iteration is None:
                ref_iteration = iteration
            elif iteration != ref_iteration:
                raise RuntimeError(
                    f"Iteration mismatch across source dp_ranks during "
                    f"downscaled DP loading: expected iteration "
                    f"{ref_iteration} but got {iteration}."
                )
            all_values.append(vals)
            all_indices.append(inds)
        return self._interleave_microbatches(all_values, all_indices)

    def _iter_single_source_group(
        self,
        pool: Optional[concurrent.futures.ThreadPoolExecutor],
        url: str,
    ) -> Iterator[Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """Yield DP-sliced microbatches from one source DP tar stream."""
        for _, values_list, indices_list in self._iter_decoded_entries(pool, url):
            yield self._slice_microbatches(values_list, indices_list)

    def _iter_downscaled_group(
        self,
        pool: Optional[concurrent.futures.ThreadPoolExecutor],
        urls: Tuple[str, ...],
    ) -> Iterator[Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """Yield interleaved microbatches from source DP tar streams in lockstep."""
        decoded_iters = [self._iter_decoded_entries(pool, url) for url in urls]
        for decoded_group in zip(*decoded_iters):
            yield self._interleave_decoded_group(decoded_group)

    def _iter_group(
        self,
        pool: Optional[concurrent.futures.ThreadPoolExecutor],
        group: Tuple[str, ...],
    ) -> Iterator[Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """Yield logical training iterations from one shard group."""
        if len(self._source_dp_ranks) > 1:
            yield from self._iter_downscaled_group(pool, group)
        else:
            yield from self._iter_single_source_group(pool, group[0])

    # ------------------------------------------------------------------
    #  Main iteration entry point
    # ------------------------------------------------------------------

    def __iter__(self):
        processed: set = set()
        with TarShardPrefetcher(
            enabled=self._remote_logprobs,
            depth=self._msc_prefetch_depth,
            max_workers=max(1, self._msc_prefetch_depth * len(self._source_dp_ranks)),
        ) as prefetcher:
            if self._decode_threads == 1:
                for group in self._shard_groups(processed, prefetcher):
                    yield from self._iter_group(None, group)
            else:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self._decode_threads,
                    thread_name_prefix="teacher-decode",
                ) as pool:
                    for group in self._shard_groups(processed, prefetcher):
                        yield from self._iter_group(pool, group)


# ---------------------------------------------------------------------------
#  Top-K KL divergence
# ---------------------------------------------------------------------------

def topk_kl_div(
    student_logits: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    tp_size: int,
    tp_rank: int,
    tp_group: dist.ProcessGroup,
    add_ghost_token: bool = False,
) -> torch.Tensor:
    """Compute the KL divergence between student and teacher's top-K log-probs."""
    student_logits = student_logits.float()
    teacher_topk_logprobs = teacher_topk_logprobs.float()

    # ---- TP-aware student softmax / log-softmax ----
    logits_max, _ = student_logits.max(dim=-1, keepdim=True)
    if tp_size > 1:
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)
    student_logits -= logits_max.detach()  # purely for numerical stability

    sum_exp = student_logits.exp().sum(dim=-1, keepdim=True)
    if tp_size > 1:
        sum_exp = dist_nn.functional.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=tp_group)
    student_logprobs = student_logits - sum_exp.log()

    # ---- Gather student log-probs at teacher's top-K positions (local shard only) ----
    local_vocab_size = student_logits.size(-1)
    offset = local_vocab_size * tp_rank
    mask = (
        (teacher_topk_indices >= offset)
        & (teacher_topk_indices < offset + local_vocab_size)
        & (teacher_topk_logprobs != CACHED_LOGITS_LOGPROB_SENTINEL)
    )
    # Clamp out-of-range indices to avoid index errors; their contributions are zeroed by the mask
    teacher_local_indices = (teacher_topk_indices - offset).clamp(0, local_vocab_size - 1)
    # In TP > 1, may contain duplicate values due to clamping above, so they are masked downstream
    student_topk_logprobs = torch.gather(student_logprobs, -1, teacher_local_indices)

    # ---- Add a "ghost" token containing sum of non-top-K probabilities to both student and teacher ----
    if add_ghost_token:
        eps = 1e-8
        student_topk_logprobs_exp = student_topk_logprobs.exp() * mask  # don't sum duplicate indices if any
        student_topk_exp_sum = student_topk_logprobs_exp.sum(dim=-1, keepdim=True)
        if tp_size > 1:
            student_topk_exp_sum = dist_nn.functional.all_reduce(
                student_topk_exp_sum, op=dist.ReduceOp.SUM, group=tp_group
            )
        student_residual = torch.log((1.0 - student_topk_exp_sum).clamp(min=eps))
        teacher_residual = torch.log((1.0 - teacher_topk_logprobs.exp().sum(dim=-1, keepdim=True)).clamp(min=eps))
        student_topk_logprobs = torch.cat([student_topk_logprobs, student_residual], dim=-1)
        teacher_topk_logprobs = torch.cat([teacher_topk_logprobs, teacher_residual], dim=-1)
        mask = torch.cat([mask, mask.new_full((*mask.shape[:-1], 1), float(tp_rank==0))], dim=-1)

    # ---- Sparse KL divergence (summed over top-K dimension) ----
    kl_div = teacher_topk_logprobs.exp() * (teacher_topk_logprobs - student_topk_logprobs)
    kl_loss = torch.sum(mask * kl_div, dim=-1)

    return kl_loss.transpose(0, 1).contiguous()  # [S, B] -> [B, S]

# ---------------------------------------------------------------------------
#  KD dataloading + loss class
# ---------------------------------------------------------------------------

class CachedLogitsKDLoss:
    """Offline knowledge-distillation loss backed by cached teacher top-K log-probs.

    For each microbatch the loss function:

    1. Retrieves (or prefetches via the DataLoader) this rank's cp-dp tar shard
       containing the teacher's full top-K log-probabilities and global vocab
       indices.
    2. Computes the student's globally-normalised log-probabilities via a
       TP-aware softmax, gathers them at the teacher's top-K positions, and
       returns the **forward KL divergence**
       ``KL(teacher ‖ student) = Σ_k p_T(k)·[log p_T(k) − log p_S(k)]``
       averaged over all token positions.

    A :class:`torch.utils.data.DataLoader` with ``pin_memory=True`` streams
    batched tar shards from disk so that I/O overlaps with GPU compute.
    Because the tensors arrive in pinned host memory, the subsequent
    ``tensor.to(device, non_blocking=True)`` call can overlap the DMA transfer
    with ongoing GPU kernels.

    A custom tar reader sequentially streams ``cp{C}_dp{D}__{B}.tar`` shards
    through the same storage layer used by the writer.

    The DataLoader is initialised lazily on the first ``__call__`` because the
    starting iteration is not known at construction time.

    Args:
        logprobs_dir: Root directory containing log-probs data written by
            :class:`LogitsSaverHooks`.
        decode_threads: Number of decode threads inside the single DataLoader
            worker used to parallelise zstd decompression and ``torch.load``
            across in-flight samples.
        prefetch_factor: How many decoded iterations the PyTorch DataLoader
            worker pre-loads ahead.
        msc_prefetch_depth: For remote MSC tar shards, how many whole
            shard objects to materialize into the MSC cache ahead of
            sequential tar consumption.
    """

    def __init__(
        self,
        logprobs_dir: str,
        decode_threads: int = 1,
        prefetch_factor: int = 2,
        msc_prefetch_depth: int = 2,
        ignore_hash: bool = False,
    ):
        self.logprobs_dir = logprobs_dir
        self._decode_threads = decode_threads
        self._prefetch_factor = prefetch_factor
        self._msc_prefetch_depth = msc_prefetch_depth
        self._ignore_hash = ignore_hash

        # ---- parallel-state bookkeeping ----
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self.tp_group = parallel_state.get_tensor_model_parallel_group()
        self.cp_rank = parallel_state.get_context_parallel_rank()
        self.dp_rank = parallel_state.get_data_parallel_rank()
        self.dp_size = parallel_state.get_data_parallel_world_size()

        # ---- DataLoader (lazy-initialised on first call) ----
        self._dataloader_iter: Optional[Iterator] = None

        # ---- iteration / microbatch tracking ----
        self._current_iteration: Optional[int] = None
        self._microbatch_counter: int = 0

        # ---- current iteration's teacher data (pinned CPU tensors) ----
        self._current_values: Optional[List[torch.Tensor]] = None
        self._current_indices: Optional[List[torch.Tensor]] = None

    def _init_dataloader(self, start_iteration: int) -> None:
        """Create the DataLoader starting from *start_iteration*."""
        # TeacherTarDataset yields a single ordered stream, so multiple DataLoader
        # workers would split shards and interleave results non-deterministically.
        # Cap at 1 DataLoader worker; intra-worker parallelism is obtained via
        # the ThreadPoolExecutor inside TeacherTarDataset.
        dataset = TeacherTarDataset(
            self.logprobs_dir,
            self.cp_rank,
            self.dp_rank,
            self.dp_size,
            start_iteration=start_iteration,
            decode_threads=self._decode_threads,
            msc_prefetch_depth=self._msc_prefetch_depth,
            ignore_hash=self._ignore_hash,
        )
        # Remote shard discovery uses rank-0 collectives, so it must run in the
        # main training process rather than a DataLoader worker.
        dataloader_workers = 0 if is_remote_storage_path(self.logprobs_dir) else 1
        loader_kwargs = dict(
            dataset=dataset,
            batch_size=None,
            collate_fn=lambda x: x,
            pin_memory=True,
            num_workers=dataloader_workers,
        )
        if dataloader_workers > 0:
            loader_kwargs["prefetch_factor"] = self._prefetch_factor
            loader_kwargs["persistent_workers"] = True
        loader = torch.utils.data.DataLoader(**loader_kwargs)
        self._dataloader_iter = iter(loader)

    def _advance_iteration(self) -> None:
        """Fetch the next iteration's data from the DataLoader."""
        assert self._dataloader_iter is not None
        try:
            values_list, indices_list = next(self._dataloader_iter)
        except StopIteration as e:
            raise StopIteration(
                f"No more teacher log-prob data available in "
                f"{self.logprobs_dir}.  The DataLoader has been exhausted."
            ) from e
        self._current_values = values_list
        self._current_indices = indices_list
        self._microbatch_counter = 0

    def __call__(
        self,
        student_logits: torch.Tensor,
        iteration: Optional[int] = None,
        microbatch_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute KL-divergence loss for one microbatch.

        Args:
            student_logits: ``(seq_len, batch, local_vocab_size)`` – raw
                (pre-softmax) student logits from this TP rank's vocab shard.
            iteration: Training iteration.  Defaults to ``args.curr_iteration``.
            microbatch_idx: Microbatch index within the current iteration.
                Defaults to an auto-incremented counter that resets on each
                new iteration.

        Returns:
            Tensor of shape ``(batch, seq_len)`` – per-token forward KL
            divergence (unreduced).
        """
        # ---- resolve iteration ----
        if iteration is None:
            iteration = get_current_iteration()

        # ---- lazy DataLoader init ----
        if self._dataloader_iter is None:
            self._init_dataloader(iteration)

        # ---- detect iteration change → pull next prefetched item ----
        if self._current_iteration != iteration:
            self._advance_iteration()
            self._current_iteration = iteration

        # ---- resolve microbatch index ----
        if microbatch_idx is None:
            microbatch_idx = self._microbatch_counter
            self._microbatch_counter += 1

        if microbatch_idx >= len(self._current_values):
            raise IndexError(
                f"Microbatch index {microbatch_idx} out of range: only "
                f"{len(self._current_values)} microbatches were saved for "
                f"iteration {iteration}."
            )

        # ---- move teacher data to GPU (non-blocking: tensors are pinned) ----
        teacher_values = self._current_values[microbatch_idx].to(
            student_logits.device, non_blocking=True
        )
        teacher_indices = self._current_indices[microbatch_idx].to(
            student_logits.device, non_blocking=True
        )

        # ---- trim teacher logits to match student sequence length ----
        if teacher_values.size(0) > student_logits.size(0):
            if safe_get_rank() == 0:
                warnings.warn(
                    "CachedLogitsKDLoss: teacher logits sequence length "
                    f"({teacher_values.size(0)}) is longer than student sequence length "
                    f"({student_logits.size(0)}); trimming teacher logits to match.",
                )
            teacher_values = teacher_values[:student_logits.size(0)]
            teacher_indices = teacher_indices[:student_logits.size(0)]

        # ---- compute loss ----
        return topk_kl_div(
            student_logits,
            teacher_values,
            teacher_indices,
            self.tp_size,
            self.tp_rank,
            self.tp_group,
            add_ghost_token=True,
        )

# ---------------------------------------------------------------------------
#  Main callable wrapper to pass to Megatron LM training loop
# ---------------------------------------------------------------------------

class LossFuncCallable:
    def __init__(
        self,
        logprobs_dir: str,
        decode_threads: int = 1,
        prefetch_factor: int = 2,
        kd_loss_alpha: float = 0.5,
        ignore_errors: bool = False,
        msc_prefetch_depth: int = 2,
        ignore_hash: bool = False,
    ):
        self.logprobs_dir = logprobs_dir
        self.decode_threads = decode_threads
        self.prefetch_factor = prefetch_factor
        self.msc_prefetch_depth = msc_prefetch_depth
        self.kd_func = None
        self.alpha = kd_loss_alpha
        self.ignore_errors = ignore_errors
        self.ignore_hash = ignore_hash

    @staticmethod
    def _mask_loss(output_tensor, loss_mask):
        """Apply mask to the unreduced loss tensor."""
        losses = output_tensor.view(-1).float()
        loss_mask = loss_mask.reshape(-1).float()
        return torch.sum(losses * loss_mask)

    def __call__(self, loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: LanguageModule):
        """Loss function wrapper for compatibility with Megatron LM training loop.

        Args:
            loss_mask (Tensor): Used to mask out some portions of the loss
            output_tensor (Tensor): The tensor with the losses
            model (LanguageModule): The model (can be wrapped)
        """
        if self.kd_func is None:
            # Construct here so parallel_state is initialized by now
            self.kd_func = CachedLogitsKDLoss(
                logprobs_dir=self.logprobs_dir,
                decode_threads=self.decode_threads,
                prefetch_factor=self.prefetch_factor,
                msc_prefetch_depth=self.msc_prefetch_depth,
                ignore_hash=self.ignore_hash,
            )

        # LM loss
        loss_lm = self._mask_loss(output_tensor, loss_mask)
        num_tokens = loss_mask.sum().clone().detach().to(torch.int)
        report = {'lm loss': torch.cat([loss_lm.clone().detach().view(1), num_tokens.view(1)])}

        # Eval case
        if not model.training:
            # During evaluation, return only the LM loss, as eval logits are not stored to disk
            return loss_lm, num_tokens, report

        # KD loss
        try:
            logits_capture = get_student_logits_capture()
            if logits_capture is None:
                raise RuntimeError(
                    "Cached-logits KD requires StudentLogitsCapture to be attached "
                    "to the model output layer during setup."
                )
            logits = logits_capture.pop()

            loss_kd = self.kd_func(logits)
            loss_kd = self._mask_loss(loss_kd, loss_mask)
            # Requires extra TP reduction
            dist.all_reduce(loss_kd, group=parallel_state.get_tensor_model_parallel_group())
        except Exception as e:
            if not self.ignore_errors:
                raise
            # Don't fail the entire training process if KD loss fails
            logger.warning(f">>>>>> KD LOSS FAILED — falling back to LM loss. {type(e).__name__}: {e} <<<<<<")
            return loss_lm, num_tokens, report

        report["logits distillation loss"] = torch.cat([loss_kd.clone().detach().view(1), num_tokens.view(1)])

        loss_total = (1 - self.alpha) * loss_lm + self.alpha * loss_kd
        report["total loss"] = torch.cat([loss_total.clone().detach().view(1), num_tokens.view(1)])

        return loss_total, num_tokens, report
