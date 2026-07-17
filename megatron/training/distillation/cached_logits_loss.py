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
* **CP-agnostic loading** – each rank loads a CP-group shard for its DP mapping
  and extracts its current CP-rank sequence slice locally.
* **Multi-threaded decode pipeline** – the tar ``IterableDataset`` runs in the
  main training process (to preserve shard iteration order) and parallelises the
  CPU-bound zstd decompression + ``torch.load`` + index unpacking across an
  internal ``ThreadPoolExecutor``.  This removes the
  single-thread bottleneck that dominates per-iteration loader cost when each
  saved tar contains a large multi-microbatch blob.

Assumptions
-----------
* The student run uses the **same random seed** and data pipeline as the
  teacher run that produced the cached log-probs, so the global sample
  ordering matches.
* The current CP size may differ from the teacher run for CP-agnostic shards,
  provided the full sequence length is divisible by ``2 * context_parallel_size``.
* The current ``micro_batch_size``, ``data_parallel_size`` and
  ``global_batch_size`` may each differ from the teacher run, restricted to
  integer multiples (in either direction) of the saved value.  The loader
  builds a :class:`~megatron.training.distillation.utils.LogprobsReshardPlan`
  that maps each load microbatch to one or a few contiguous slices of saved
  monoliths.
* Teacher log-probs were saved by ``LogitsSaverHooks`` with payload
  ``format_version >= 2``.  Older ``format_version == 1`` tars are read by
  :class:`LegacyTeacherTarDataset` (selected automatically by
  :func:`make_teacher_tar_dataset`) and remain restricted to DP-only
  resharding.  That class is slated for removal once existing teacher caches
  are regenerated.
* v2 tars and their inner payload members are named by the global sample
  range they cover (``dp{d}__{start}-{end}.tar``, ``{start}-{end}.pt.zst``)
  rather than by training-iteration number.  Dedup of stale/superseded
  shards from a crash-and-resume happens lazily on the writer side, per
  flush (``megatron.training.distillation.utils.quarantine_contained_tars``),
  and only covers the narrow "new write fully contains an old shard" case
  -- not an exhaustive guarantee.  The loader complements this with a
  sequential/no-overlap check across discovered shards for a given saved
  DP rank, raising a clear error rather than silently reading
  overlapping/duplicate data if one slips through.

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
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
import torch.utils.data

from megatron.core import parallel_state
from megatron.core._rank_utils import safe_get_rank
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.training import get_args
from megatron.training.distillation.utils import (
    BATCHED_TAR_RE,
    CACHED_LOGITS_LOGPROB_SENTINEL,
    V2_BATCHED_TAR_RE,
    LogprobsReshardPlan,
    LogprobsTarEntry,
    TarShardPrefetcher,
    V2LogprobsTarEntry,
    batched_tar_prefix,
    compute_dataset_hash,
    decode_logprobs_payload,
    detect_saved_dp_size,
    get_current_iteration,
    is_remote_storage_path,
    iter_logprobs_tar_entries,
    peek_first_logprobs_metadata,
    slice_tensor_for_cp_rank,
    sorted_batched_tars,
    storage_basename,
    storage_glob_with_caching,
    v2_iter_logprobs_tar_entries,
    v2_sorted_batched_tars,
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
        """Attach forward hooks to the model's output layer.

        ``model`` must own ``output_layer`` (last PP stage).
        """
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


# ---------------------------------------------------------------------------
#  Dataset – streaming batched tar shards (v2: monolith + LogprobsReshardPlan)
# ---------------------------------------------------------------------------


# Per-(d_save) cache entry: (i_save, values_monolith, indices_monolith).
_StreamCacheEntry = Tuple[int, torch.Tensor, torch.Tensor]


class TeacherTarDataset(torch.utils.data.IterableDataset):
    """Streaming dataset that reads teacher log-probs from v2 batched tar shards.

    Each saved iteration is stored as a single ``(seq, gbs_save / dp_save, K)``
    monolithic tensor per saved DP rank, with all microbatches concatenated
    along the sample dim and right-padded to the iteration-wide ``K_max``.
    The loader maintains one decode stream per saved DP rank it reads from
    and assembles each load microbatch as a (typically single) contiguous
    slice of those monoliths, using a
    :class:`~megatron.training.distillation.utils.LogprobsReshardPlan`
    to map (load iteration, microbatch step, current DP rank) → saved
    (iteration, DP rank, row range) tuples.

    Supports flexible MBS / DP / GBS at load time, restricted to integer
    multiples in either direction.  See :class:`LogprobsReshardPlan`
    for the exact restriction.

    Tars and their inner members are named by the **global sample range**
    they cover (``dp{d}__{start}-{end}.tar``, ``{start}-{end}.pt.zst``)
    rather than by training-iteration number.  Discovery filters purely on
    sample ranges (``end_sample > save_sample_start``).  Dedup of
    stale/superseded shards from a crash-and-resume mostly happens on the
    writer side, per flush
    (``megatron.training.distillation.utils.quarantine_contained_tars``),
    but that only covers the narrow "new write fully contains an old
    shard" case; :meth:`_stream_decoded` enforces a sequential/no-overlap
    check across discovered shards as a safety net for the rest.

    Shard discovery refreshes the directory listing after known shards are
    exhausted so a concurrent writer can publish more data; remote refreshes
    are rank-0 broadcast to avoid object-store list storms.

    Yields ``(iteration, values_list, indices_list)``, one entry per load
    microbatch step, with CP sequence slicing already applied.
    """

    def __init__(
        self,
        logprobs_dir: str,
        cp_rank: int,
        cp_size: int,
        dp_rank: int,
        dp_size: int,
        *,
        start_iteration: int = 0,
        decode_threads: int = 1,
        decode_lookahead: Optional[int] = None,
        msc_prefetch_depth: int = 2,
        ignore_hash: bool = False,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.logprobs_dir = logprobs_dir
        self.cp_rank = cp_rank
        self.cp_size = cp_size
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.start_iteration = start_iteration

        self._remote_logprobs = is_remote_storage_path(logprobs_dir)
        self._msc_prefetch_depth = max(0, msc_prefetch_depth)

        # Per-tar dataset-identity verification: each tar stream validates its
        # leading _meta.json before yielding any payload data.
        self._expected_hash = None if ignore_hash else compute_dataset_hash()[0]

        self._decode_threads = max(1, decode_threads)

        if meta is None:
            meta = peek_first_logprobs_metadata(logprobs_dir)
        if meta is None:
            raise FileNotFoundError(
                f"No batched tar shards (or unreadable metadata) found in '{logprobs_dir}'."
            )
        self._meta = meta
        self._plan = self._build_plan(meta, dp_size=dp_size)

        if decode_lookahead is None:
            # When each load step consumes >1 saved iters, scale the lookahead
            # so the decode pipeline stays full.  Inverse direction (load step
            # smaller than saved step) reuses a single decoded monolith across
            # multiple yields and doesn't need extra in-flight decodes.
            lookahead_factor = max(1, self._plan.gbs_load // self._plan.gbs_save)
            decode_lookahead = max(2 * self._decode_threads, 4) * lookahead_factor
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

        p = self._plan
        if (p.mbs_save, p.dp_save, p.gbs_save) != (p.mbs_load, p.dp_load, p.gbs_load):
            print_rank_0(
                "Cached-logits resharding: "
                f"save=(mbs={p.mbs_save}, dp={p.dp_save}, gbs={p.gbs_save}) -> "
                f"load=(mbs={p.mbs_load}, dp={p.dp_load}, gbs={p.gbs_load})"
            )

        self._needed_d_saves = self._plan.needed_d_saves(self.dp_rank)
        # Global sample index the first requested load iteration starts at.
        # Used to skip earlier tar members at the tar-streaming level.  No
        # division by gbs_save is needed here since v2 tar/member names are
        # already keyed by sample range, not by an iteration number.
        self._save_sample_start = self.start_iteration * self._plan.gbs_load

    # ------------------------------------------------------------------
    #  Plan construction
    # ------------------------------------------------------------------

    @classmethod
    def _build_plan(cls, meta: Dict[str, Any], *, dp_size: int) -> LogprobsReshardPlan:
        saver = meta.get("saver", {})
        missing = [
            field for field in ("mbs_save", "dp_size_save", "gbs_save", "format_version")
            if field not in saver
        ]
        if missing:
            raise ValueError(
                f"Cached-logits _meta.json is missing required v2 fields {missing}. "
                "These tars were likely written with format_version=1; use "
                "LegacyTeacherTarDataset (or call make_teacher_tar_dataset to "
                "dispatch automatically) to read them."
            )
        if saver["format_version"] < 2:
            raise ValueError(
                f"format_version={saver['format_version']} tars are not readable "
                "by TeacherTarDataset; use LegacyTeacherTarDataset instead."
            )
        args = get_args()
        return LogprobsReshardPlan(
            mbs_save=int(saver["mbs_save"]),
            dp_save=int(saver["dp_size_save"]),
            gbs_save=int(saver["gbs_save"]),
            mbs_load=int(args.micro_batch_size),
            dp_load=int(dp_size),
            gbs_load=int(args.global_batch_size),
        )

    # ------------------------------------------------------------------
    #  Tar discovery
    # ------------------------------------------------------------------

    def _discover_shards(self, already_processed: set, dp_save: int) -> List[str]:
        """Glob for new v2 batched tar shards for a given saved DP rank.

        Returns shards in ascending ``start`` order (via
        ``v2_sorted_batched_tars``); :meth:`_stream_decoded` relies on
        that ordering to check for overlaps as a safety net, since the
        writer-side supersede cleanup only resolves the narrow case where
        a new write fully contains an old shard.
        """
        prefix = batched_tar_prefix(dp_save)
        all_urls = v2_sorted_batched_tars(
            storage_glob_with_caching(self.logprobs_dir, f"*{prefix}*.tar", cached=False)
        )
        new_urls = []
        for url in all_urls:
            if url in already_processed:
                continue
            m = V2_BATCHED_TAR_RE.match(storage_basename(url))
            if m and int(m.group("end")) > self._save_sample_start:
                new_urls.append(url)
        return new_urls

    # ------------------------------------------------------------------
    #  Decode hooks (overridable by LegacyTeacherTarDataset)
    # ------------------------------------------------------------------

    def _decode_entry(self, entry: V2LogprobsTarEntry) -> _StreamCacheEntry:
        """Decode one tar payload into a monolithic ``(values, indices)`` pair."""
        values, indices = decode_logprobs_payload(entry.data)
        span = entry.end_sample - entry.start_sample
        if span != self._plan.gbs_save:
            raise RuntimeError(
                f"Cached-logits member {entry.start_sample}-{entry.end_sample} "
                f"spans {span} samples, expected gbs_save={self._plan.gbs_save}. "
                "This indicates a corrupt shard or one written by a "
                "different run/config."
            )
        i_save = entry.start_sample // self._plan.gbs_save
        return i_save, values, indices

    def _iter_entries_parallel(
        self,
        pool: concurrent.futures.ThreadPoolExecutor,
        entries: Iterator[Any],
    ) -> Iterator[Any]:
        """Yield decoded entries using a decode thread pool.

        The main thread streams raw tar members one at a time and submits the
        CPU-heavy zstd + ``torch.load`` + index unpacking work to *pool*.
        Results are yielded in tar order via a FIFO of futures.  Shared by
        both the v2 (:class:`V2LogprobsTarEntry`) and legacy v1
        (:class:`LogprobsTarEntry`) decode paths.
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
    ) -> Iterator[Any]:
        """Yield decoded iteration payloads from one tar URL."""
        entries = v2_iter_logprobs_tar_entries(
            url,
            start_sample=self._save_sample_start,
            expected_hash=self._expected_hash,
        )
        if pool is None:
            for entry in entries:
                yield self._decode_entry(entry)
        else:
            yield from self._iter_entries_parallel(pool, entries)

    # ------------------------------------------------------------------
    #  Per-d_save stream
    # ------------------------------------------------------------------

    def _stream_decoded(
        self,
        pool: Optional[concurrent.futures.ThreadPoolExecutor],
        prefetcher: TarShardPrefetcher,
        d_save: int,
    ) -> Iterator[_StreamCacheEntry]:
        """Yield decoded ``(i_save, values, indices)`` entries for one ``d_save``.

        Uses *prefetcher* to overlap the download of the next tar object
        (on MSC paths) with the decode of the current tar, keeping
        remote-storage tar-boundary stalls out of the training critical
        path.  Local paths make the prefetcher a no-op.

        Re-globs the directory after each batch of tars is exhausted so a
        concurrent writer can publish more shards.  Raises
        :class:`FileNotFoundError` if no shards ever appear; returns normally
        once a writer has stopped publishing new shards.

        Also enforces that shards for this ``d_save`` are sequential (no
        two overlap): ``LogitsSaverHooks``'s writer-side supersede cleanup
        only handles the narrow "new write fully contains an old shard"
        case, so a reverse-containment or partial-overlap case could in
        principle leave two overlapping shards on disk.  Since
        :meth:`_discover_shards` always returns shards in ascending
        ``start`` order, checking each newly-discovered shard's ``start``
        against the running ``end`` of the last one streamed is sufficient
        to catch *any* pairwise overlap among all shards seen so far, not
        just adjacent ones (if shard A overlapped some later shard Y, A's
        end would necessarily also exceed A's immediate successor's
        start, so the very next comparison would already have raised).
        """
        processed: set = set()
        last_end: Optional[int] = None
        while True:
            urls = self._discover_shards(processed, d_save)
            if not urls:
                if not processed:
                    raise FileNotFoundError(
                        f"No batched tar shards for saved dp_rank={d_save} found "
                        f"covering global sample {self._save_sample_start} or later "
                        f"in '{self.logprobs_dir}'"
                    )
                return
            # One URL per prefetch "group": iter_prefetched schedules
            # msc_prefetch_depth URLs ahead per stream and yields each
            # after its download completes.  Multiple d_save streams
            # share the same prefetcher; their URL sets are disjoint
            # (different ``dp{d}__*.tar`` prefixes) so no scheduling
            # collision occurs.
            for (url,) in prefetcher.iter_prefetched([(u,) for u in urls]):
                m = V2_BATCHED_TAR_RE.match(storage_basename(url))
                start, end = int(m.group("start")), int(m.group("end"))
                if last_end is not None and start < last_end:
                    raise RuntimeError(
                        f"Cached-logits shards for saved dp_rank={d_save} in "
                        f"'{self.logprobs_dir}' are not sequential: shard "
                        f"'{storage_basename(url)}' (start={start}) overlaps "
                        f"the previous shard's end ({last_end}).  The "
                        "writer-side supersede cleanup only resolves the "
                        "case where a new write fully contains an old "
                        "shard; this indicates a leftover overlapping tar "
                        "that needs manual cleanup."
                    )
                last_end = end
                processed.add(url)
                yield from self._iter_decoded_entries(pool, url)

    # ------------------------------------------------------------------
    #  Load-iteration assembly
    # ------------------------------------------------------------------

    def _slice_cp_sequences(
        self,
        values_list: List[torch.Tensor],
        indices_list: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Extract this CP rank's zigzag sequence slice from full-CP tensors."""
        if self.cp_size <= 1:
            return values_list, indices_list
        return (
            [
                slice_tensor_for_cp_rank(values, self.cp_rank, self.cp_size)
                for values in values_list
            ],
            [
                slice_tensor_for_cp_rank(indices, self.cp_rank, self.cp_size)
                for indices in indices_list
            ],
        )

    def _build_load_iteration(
        self,
        i_load: int,
        cache: Dict[int, Optional[_StreamCacheEntry]],
        streams: Dict[int, Iterator[_StreamCacheEntry]],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Assemble one load iteration's ``(values_list, indices_list)``.

        For each load microbatch step, walks the
        :class:`LogprobsReshardPlan`-supplied sources, advances the
        appropriate per-d_save stream until the right saved iteration is
        cached, slices the monolith, and (in the rare cross-boundary case)
        concatenates a handful of slices.
        """
        values_list: List[torch.Tensor] = []
        indices_list: List[torch.Tensor] = []
        for m_load in range(self._plan.num_mb_load):
            slices_v: List[torch.Tensor] = []
            slices_i: List[torch.Tensor] = []
            for src in self._plan.sources_for_microbatch(i_load, m_load, self.dp_rank):
                entry = cache.get(src.d_save)
                while entry is None or entry[0] < src.iter_save:
                    try:
                        entry = next(streams[src.d_save])
                    except StopIteration:
                        # Writer stopped before the loader's planned end.
                        raise StopIteration(
                            f"Cached-logits stream for saved dp_rank={src.d_save} "
                            f"exhausted before save iteration {src.iter_save} "
                            f"(needed by load iter {i_load}, microbatch {m_load})."
                        )
                    cache[src.d_save] = entry
                if entry[0] != src.iter_save:
                    raise RuntimeError(
                        f"Expected save iteration {src.iter_save} from saved "
                        f"dp_rank={src.d_save}, got {entry[0]}."
                    )
                _, v_mon, i_mon = entry
                slices_v.append(v_mon[:, src.row_start:src.row_end])
                slices_i.append(i_mon[:, src.row_start:src.row_end])

            # The saver guarantees a single uniform K across all saved
            # microbatches / iterations (= effective_k), so the slices
            # coming from possibly-different saved iterations always
            # share a K and the cat is a single allocation.
            if len(slices_v) == 1:
                v = slices_v[0]
                i = slices_i[0]
            else:
                v = torch.cat(slices_v, dim=1)
                i = torch.cat(slices_i, dim=1)

            values_list.append(v)
            indices_list.append(i)
        return values_list, indices_list

    # ------------------------------------------------------------------
    #  Main iteration entry point
    # ------------------------------------------------------------------

    def __iter__(self):
        def run(pool, prefetcher):
            streams = {
                d_save: self._stream_decoded(pool, prefetcher, d_save)
                for d_save in self._needed_d_saves
            }
            cache: Dict[int, Optional[_StreamCacheEntry]] = {
                d_save: None for d_save in self._needed_d_saves
            }
            i_load = self.start_iteration
            while True:
                try:
                    values_list, indices_list = self._build_load_iteration(
                        i_load, cache, streams
                    )
                except StopIteration:
                    return
                values_list, indices_list = self._slice_cp_sequences(
                    values_list, indices_list
                )
                yield i_load, values_list, indices_list
                i_load += 1

        # One prefetcher shared across all needed d_save streams: each
        # stream requests msc_prefetch_depth URLs ahead, so total
        # in-flight download workers = msc_prefetch_depth * |d_saves|.
        # Local paths make TarShardPrefetcher a no-op (enabled=False).
        with TarShardPrefetcher(
            enabled=self._remote_logprobs,
            depth=self._msc_prefetch_depth,
            max_workers=max(1, self._msc_prefetch_depth * len(self._needed_d_saves)),
        ) as prefetcher:
            if self._decode_threads == 1:
                yield from run(None, prefetcher)
            else:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self._decode_threads,
                    thread_name_prefix="teacher-decode",
                ) as pool:
                    yield from run(pool, prefetcher)


# ---------------------------------------------------------------------------
#  Legacy v1 dataset (deprecated; self-contained for easy removal)
# ---------------------------------------------------------------------------


class LegacyTeacherTarDataset(TeacherTarDataset):
    """Read v1 (list-of-microbatches) cached-logits tars.

    Preserves the pre-v2 behavior verbatim: per-tar-member iteration,
    DP-only resharding via microbatch stride / interleave, and
    list-shaped decoded payloads.  Selected automatically by
    :func:`make_teacher_tar_dataset` when the on-disk ``_meta.json`` has
    ``format_version`` missing or ``< 2``.

    .. deprecated::
       Slated for removal once existing teacher caches have been
       regenerated with ``format_version=2``.  v1 tars do **not** support
       MBS or GBS resharding; this class raises a clear :class:`ValueError`
       if the current ``micro_batch_size`` or ``global_batch_size`` differs
       from the saved values when they are present in the metadata.
    """

    _DEPRECATION_WARNED: bool = False

    def __init__(
        self,
        logprobs_dir: str,
        cp_rank: int,
        cp_size: int,
        dp_rank: int,
        dp_size: int,
        *,
        start_iteration: int = 0,
        decode_threads: int = 1,
        decode_lookahead: Optional[int] = None,
        msc_prefetch_depth: int = 2,
        ignore_hash: bool = False,
        meta: Optional[Dict[str, Any]] = None,
    ):
        # NOTE: intentionally bypass the v2 base ``__init__`` because it
        # demands v2 metadata fields and constructs a LogprobsReshardPlan
        # that the v1 path does not use.
        self.logprobs_dir = logprobs_dir
        self.cp_rank = cp_rank
        self.cp_size = cp_size
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.start_iteration = start_iteration

        self._remote_logprobs = is_remote_storage_path(logprobs_dir)
        self._msc_prefetch_depth = max(0, msc_prefetch_depth)

        self._expected_hash = None if ignore_hash else compute_dataset_hash()[0]

        self._decode_threads = max(1, decode_threads)
        if decode_lookahead is None:
            decode_lookahead = max(2 * self._decode_threads, 4)
        self._decode_lookahead = max(1, decode_lookahead)

        # No reshard plan in v1; expose attribute for completeness.
        self._plan = None
        self._save_iter_start = start_iteration

        if not LegacyTeacherTarDataset._DEPRECATION_WARNED and safe_get_rank() == 0:
            warnings.warn(
                "Loading cached-logits tars in the v1 (list-of-microbatches) "
                "format. Only DP resharding is supported; MBS and GBS must "
                "match the teacher run. Regenerate the teacher cache to enable "
                "format_version=2 MBS/GBS resharding.",
                DeprecationWarning,
                stacklevel=2,
            )
            LegacyTeacherTarDataset._DEPRECATION_WARNED = True

        # If the metadata happens to record mbs_save / gbs_save (some late
        # v1-era saves did), reject mismatches up-front rather than producing
        # silently-wrong KD losses.
        args = get_args()
        saver_meta = (meta or {}).get("saver", {})
        saved_mbs = saver_meta.get("mbs_save")
        saved_gbs = saver_meta.get("gbs_save")
        if saved_mbs is not None and int(saved_mbs) != int(args.micro_batch_size):
            raise ValueError(
                f"v1 cached-logits tars require mbs_load == mbs_save. "
                f"saved={saved_mbs}, current={args.micro_batch_size}. "
                "Regenerate the teacher cache with format_version=2 to "
                "enable MBS resharding."
            )
        if saved_gbs is not None and int(saved_gbs) != int(args.global_batch_size):
            raise ValueError(
                f"v1 cached-logits tars require gbs_load == gbs_save. "
                f"saved={saved_gbs}, current={args.global_batch_size}. "
                "Regenerate the teacher cache with format_version=2 to "
                "enable GBS resharding."
            )

        # v1 DP remapping (microbatch-stride based; unchanged from pre-v2).
        self._source_dp_ranks, self._sub_rank, self._dp_ratio, dp_size_saved = (
            self._compute_dp_remapping(logprobs_dir, dp_rank, dp_size)
        )
        if len(self._source_dp_ranks) > 1:
            print_rank_0(
                f"DP downscaling (v1): dp_rank {dp_rank} loads from saved dp_ranks "
                f"{self._source_dp_ranks} (saved_dp_size {dp_size_saved}, "
                f"current_dp_size {dp_size})"
            )
        elif self._dp_ratio > 1:
            print_rank_0(
                f"DP upscaling (v1): dp_rank {dp_rank} -> mapped_dp_rank "
                f"{self._source_dp_ranks[0]} (sub_rank {self._sub_rank}, "
                f"saved_dp_size {dp_size_saved}, current_dp_size {dp_size})"
            )

    # ------------------------------------------------------------------
    #  v1 DP remapping (microbatch-stride based)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_dp_remapping(
        logprobs_dir: str,
        dp_rank: int,
        dp_size: int,
    ) -> Tuple[List[int], int, int, int]:
        """Compute the DP rank remapping based on the saved DP world size.

        Returns ``(source_dp_ranks, sub_rank, dp_ratio, dp_size_saved)``.
        Microbatches are distributed round-robin across DP ranks, so saved
        rank *d* holds microbatch indices ``d, d + dp_save, d + 2·dp_save,
        ...``.  Upscaling (``dp_save < dp_load``) strides through one
        saved file; downscaling (``dp_save > dp_load``) interleaves
        microbatches from multiple saved files.
        """
        dp_size_saved = detect_saved_dp_size(logprobs_dir)

        if dp_size_saved is None:
            return [dp_rank], 0, 1, dp_size
        if dp_size_saved == dp_size:
            return [dp_rank], 0, 1, dp_size_saved

        if dp_size_saved < dp_size:
            if dp_size % dp_size_saved != 0:
                raise ValueError(
                    f"Current DP size ({dp_size}) is not an exact multiple of "
                    f"saved DP size ({dp_size_saved})."
                )
            dp_ratio = dp_size // dp_size_saved
            mapped_dp_rank = dp_rank % dp_size_saved
            sub_rank = dp_rank // dp_size_saved
            return [mapped_dp_rank], sub_rank, dp_ratio, dp_size_saved

        if dp_size_saved % dp_size != 0:
            raise ValueError(
                f"Saved DP size ({dp_size_saved}) is not an exact multiple of "
                f"current DP size ({dp_size})."
            )
        num_sources = dp_size_saved // dp_size
        source_dp_ranks = [dp_rank + i * dp_size for i in range(num_sources)]
        dp_ratio = dp_size / dp_size_saved
        return source_dp_ranks, 0, dp_ratio, dp_size_saved

    # ------------------------------------------------------------------
    #  v1 decode + tar walking
    # ------------------------------------------------------------------

    def _decode_entry(self, entry: LogprobsTarEntry):
        # decode_logprobs_payload dispatches on format_version; for the v1
        # tars this dataset handles, it returns lists.
        values_list, indices_list = decode_logprobs_payload(entry.data)
        return entry.iteration, values_list, indices_list

    def _discover_shards(self, already_processed: set, dp_save: int) -> List[str]:
        prefix = batched_tar_prefix(dp_save)
        all_urls = sorted_batched_tars(
            storage_glob_with_caching(self.logprobs_dir, f"*{prefix}*.tar", cached=False)
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
        """Return this rank's DP microbatch slice and CP sequence slice."""
        if self._dp_ratio > 1:
            num_mb = len(values_list)
            if num_mb % self._dp_ratio != 0:
                raise ValueError(
                    f"Saved microbatch count ({num_mb}) is not divisible by "
                    f"DP ratio ({self._dp_ratio}). Cannot evenly split "
                    f"microbatches across remapped DP ranks."
                )
            values_list = values_list[self._sub_rank :: self._dp_ratio]
            indices_list = indices_list[self._sub_rank :: self._dp_ratio]

        return self._slice_cp_sequences(values_list, indices_list)

    @staticmethod
    def _interleave_microbatches(
        all_values: List[List[torch.Tensor]],
        all_indices: List[List[torch.Tensor]],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Interleave microbatches from multiple source dp_ranks (downscaling)."""
        num_sources = len(all_values)
        num_mb = len(all_values[0])
        merged_values: List[torch.Tensor] = []
        merged_indices: List[torch.Tensor] = []
        for mb_idx in range(num_mb):
            for src_idx in range(num_sources):
                merged_values.append(all_values[src_idx][mb_idx])
                merged_indices.append(all_indices[src_idx][mb_idx])
        return merged_values, merged_indices

    def _not_found_error(self) -> FileNotFoundError:
        dp_ranks = self._source_dp_ranks
        if len(dp_ranks) == 1:
            return FileNotFoundError(
                f"No batched tar shards for dp{dp_ranks[0]} "
                f"found at or after iteration {self.start_iteration} "
                f"in '{self.logprobs_dir}'"
            )
        return FileNotFoundError(
            f"No batched tar shards for source dp_ranks {dp_ranks} "
            f"found at or after iteration {self.start_iteration} "
            f"in '{self.logprobs_dir}'"
        )

    def _shard_groups(
        self,
        processed: set,
        prefetcher: TarShardPrefetcher,
    ) -> Iterator[Tuple[str, ...]]:
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

    def _iter_decoded_entries(
        self,
        pool: Optional[concurrent.futures.ThreadPoolExecutor],
        url: str,
    ):
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

    def _interleave_decoded_group(self, decoded_group):
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
        merged_values, merged_indices = self._interleave_microbatches(all_values, all_indices)
        return ref_iteration, merged_values, merged_indices

    def _iter_single_source_group(self, pool, url):
        for iteration, values_list, indices_list in self._iter_decoded_entries(pool, url):
            values_list, indices_list = self._slice_microbatches(values_list, indices_list)
            yield iteration, values_list, indices_list

    def _iter_downscaled_group(self, pool, urls):
        decoded_iters = [self._iter_decoded_entries(pool, url) for url in urls]
        for decoded_group in zip(*decoded_iters):
            iteration, values_list, indices_list = self._interleave_decoded_group(decoded_group)
            values_list, indices_list = self._slice_cp_sequences(values_list, indices_list)
            yield iteration, values_list, indices_list

    def _iter_group(self, pool, group):
        if len(self._source_dp_ranks) > 1:
            yield from self._iter_downscaled_group(pool, group)
        else:
            yield from self._iter_single_source_group(pool, group[0])

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
#  Factory + helpers
# ---------------------------------------------------------------------------


def make_teacher_tar_dataset(
    logprobs_dir: str,
    cp_rank: int,
    cp_size: int,
    dp_rank: int,
    dp_size: int,
    *,
    start_iteration: int = 0,
    decode_threads: int = 1,
    decode_lookahead: Optional[int] = None,
    msc_prefetch_depth: int = 2,
    ignore_hash: bool = False,
) -> TeacherTarDataset:
    """Build the right :class:`TeacherTarDataset` for the on-disk format.

    Peeks any existing tar's ``_meta.json`` (on rank 0, broadcast to the
    rest of the world) to read ``saver.format_version``; instantiates
    :class:`TeacherTarDataset` for v2 tars and :class:`LegacyTeacherTarDataset`
    for v1.  Callers should always go through this factory rather than
    constructing the classes directly.
    """
    meta = peek_first_logprobs_metadata(logprobs_dir)
    fmt = (meta or {}).get("saver", {}).get("format_version", 1)
    cls: type = TeacherTarDataset if fmt >= 2 else LegacyTeacherTarDataset
    return cls(
        logprobs_dir,
        cp_rank, cp_size,
        dp_rank, dp_size,
        start_iteration=start_iteration,
        decode_threads=decode_threads,
        decode_lookahead=decode_lookahead,
        msc_prefetch_depth=msc_prefetch_depth,
        ignore_hash=ignore_hash,
        meta=meta,
    )


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

    1. Retrieves (or prefetches via the DataLoader) this rank's mapped DP tar
       shard containing the teacher's full CP-group top-K log-probabilities and
       global vocab indices, then extracts the current CP-rank sequence slice.
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

    A custom tar reader sequentially streams ``dp{D}__{start}-{end}.tar``
    v2 shards (keyed by global sample range) -- or, for legacy caches,
    iteration-numbered ``dp{D}__{B}.tar`` / ``cp0_dp{D}__{B}.tar`` v1
    shards -- through the same storage layer used by the writer.

    The DataLoader is initialised lazily on the first ``__call__`` because the
    starting iteration is not known at construction time.

    Args:
        logprobs_dir: Root directory containing log-probs data written by
            :class:`LogitsSaverHooks`.
        decode_threads: Number of decode threads inside the single DataLoader
            worker used to parallelise zstd decompression and ``torch.load``
            across in-flight samples.
        msc_prefetch_depth: For remote MSC tar shards, how many whole
            shard objects to materialize into the MSC cache ahead of
            sequential tar consumption.
    """

    def __init__(
        self,
        logprobs_dir: str,
        decode_threads: int = 1,
        msc_prefetch_depth: int = 2,
        ignore_hash: bool = False,
    ):
        self.logprobs_dir = logprobs_dir
        self._decode_threads = decode_threads
        self._msc_prefetch_depth = msc_prefetch_depth
        self._ignore_hash = ignore_hash

        # ---- parallel-state bookkeeping ----
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self.tp_group = parallel_state.get_tensor_model_parallel_group()
        self.cp_rank = parallel_state.get_context_parallel_rank()
        self.cp_size = parallel_state.get_context_parallel_world_size()
        self.dp_rank = parallel_state.get_data_parallel_rank()
        self.dp_size = parallel_state.get_data_parallel_world_size()

        # ---- DataLoader (lazy-initialised on first call) ----
        self._dataloader_iter: Optional[Iterator] = None

        # ---- iteration / microbatch tracking ----
        self._current_iteration: Optional[int] = None
        # Iteration label of the shard most recently pulled from the DataLoader,
        # used to verify teacher/student alignment.
        self._loaded_iteration: Optional[int] = None
        self._microbatch_counter: int = 0

        # ---- current iteration's teacher data (pinned CPU tensors) ----
        self._current_values: Optional[List[torch.Tensor]] = None
        self._current_indices: Optional[List[torch.Tensor]] = None

    def _init_dataloader(self, start_iteration: int) -> None:
        """Create the DataLoader starting from *start_iteration*."""
        # Keep the iterable dataset in the training process so ordered shard
        # streaming, dynamic discovery, and rank-0 collectives stay simple.
        # ``make_teacher_tar_dataset`` peeks ``_meta.json`` to pick the
        # v2 (monolith + resharding) or v1 (legacy list-of-microbatches) reader.
        dataset = make_teacher_tar_dataset(
            self.logprobs_dir,
            self.cp_rank,
            self.cp_size,
            self.dp_rank,
            self.dp_size,
            start_iteration=start_iteration,
            decode_threads=self._decode_threads,
            msc_prefetch_depth=self._msc_prefetch_depth,
            ignore_hash=self._ignore_hash,
        )
        # Remote shard discovery uses rank-0 collectives, so the iterable
        # dataset stays in the main training process.
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=None,
            collate_fn=lambda x: x,
            pin_memory=True,
            num_workers=0,
        )
        self._dataloader_iter = iter(loader)

    def _advance_iteration(self) -> None:
        """Fetch the next shard from the DataLoader and record its iteration label."""
        assert self._dataloader_iter is not None
        try:
            loaded_iteration, values_list, indices_list = next(self._dataloader_iter)
        except StopIteration as e:
            raise StopIteration(
                f"No more teacher log-prob data available in "
                f"{self.logprobs_dir}.  The DataLoader has been exhausted."
            ) from e
        self._loaded_iteration = int(loaded_iteration)
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
            # ---- verify teacher/student alignment and skip shards if needed ----
            while self._loaded_iteration is not None and self._loaded_iteration < iteration:
                logger.warning(
                    "Teacher logit shard for iteration %s is behind training "
                    "iteration %s (overlapping/duplicate shard in %s); skipping "
                    "to resync alignment.",
                    self._loaded_iteration, iteration, self.logprobs_dir,
                )
                self._advance_iteration()
            if self._loaded_iteration != iteration:
                raise RuntimeError(
                    f"Teacher logit data misaligned: training iteration "
                    f"{iteration} but the next available cached shard is "
                    f"iteration {self._loaded_iteration} (gap / missing teacher "
                    f"data in '{self.logprobs_dir}'). Refusing to train on "
                    f"misaligned distillation targets."
                )
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
        kd_loss_alpha: float = 0.5,
        ignore_errors: bool = False,
        msc_prefetch_depth: int = 2,
        ignore_hash: bool = False,
    ):
        self.logprobs_dir = logprobs_dir
        self.decode_threads = decode_threads
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
        except StopIteration as e:
            logger.warning(f"Cached-logits dataloader exhausted on rank {safe_get_rank()} — stopping. {e}")
            raise SystemExit(0) from None
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
