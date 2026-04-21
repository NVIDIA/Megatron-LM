# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
Offline knowledge distillation loss using cached teacher top-K log-probabilities.

This module provides a loss function that loads pre-computed teacher top-K
log-probabilities from disk (as saved by ``LogitsSaverHooks`` in ``logits_saver.py``)
and computes forward KL divergence against live student logits.

Design highlights
-----------------
* **Sparse KL** – only the *K/tp_size* positions in this rank's saved shard are
  used, avoiding a full-vocab-sized dense teacher tensor.
* **TP-aware custom autograd** – a ``torch.autograd.Function`` computes the
  TP-aware student softmax in the forward pass (two all-reduces) and
  provides an **analytical backward** that requires **no TP communication**:
  ``∂L/∂z_j = (1/N) · (p_S(j)·C − p_T(j))`` where *C* is the total teacher
  probability mass in the top-K.
* **Map-style dataset + DataLoader** – :class:`_TeacherDataset` is a
  standard ``torch.utils.data.Dataset`` whose ``__getitem__`` accepts an
  iteration number directly.  A :class:`torch.utils.data.DataLoader` with
  ``pin_memory=True`` and background workers prefetches upcoming iterations
  so that disk I/O overlaps with GPU compute, and the CPU→GPU copy can be
  issued with ``non_blocking=True``.
* **Own-rank loading only** – since we assume the same parallelism layout as the
  teacher run, each rank loads only its own shard file (no cross-rank file I/O).

Assumptions
-----------
* The student run uses the **same random seed** and data pipeline as the
  teacher run that produced the cached log-probs, so the microbatch ordering
  matches.
* **Same parallelism layout** (``tp_size``, ``cp_rank``, ``dp_rank``) and
  **same micro-batch size** as the teacher run.
* Teacher log-probs were saved by ``LogitsSaverHooks`` (see ``logits_saver.py``).

Usage example
-------------
::

    from megatron.training.cached_logits_loss import LossFuncCallable

    # Instantiate once at the start of training
    loss_func = LossFuncCallable(
        logprobs_dir="/data/teacher_logprobs",
        kd_loss_alpha=0.5,
    )

    # Pass as the loss_func to Megatron's training loop:
    #   loss, num_tokens, report = loss_func(loss_mask, output_tensor, model)
"""

import glob
import logging
import os
import re
from typing import Iterator, List, Optional

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
import torch.utils.data

try:
  import zstandard
except ImportError:
  zstandard = None

try:
  import webdataset as wds
except ImportError:
  wds = None

from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.utils import unwrap_model
from megatron.training.logits_saver import (
  _BATCHED_TAR_RE,
  FOLDER_NAMES_PREFIX,
  _batched_tar_prefix,
  _sorted_batched_tars,
  decode_logprobs_sample,
  get_current_iteration,
  load_log_probs_by_rank,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Dataset – one item per iteration, indexed by iteration number
# ---------------------------------------------------------------------------

class _TeacherDataset(torch.utils.data.Dataset):
    """Map-style dataset mapping iteration number → ``(values_list, indices_list)``.

    ``__getitem__(iteration)`` loads this rank's saved shard file for the
    requested iteration via :func:`load_log_probs_by_rank`.

    ``__len__`` returns the iteration number one greater than the largest ``logprobs_iter*`` sub-folder detected
    in *logprobs_dir*. This ensures correct indexing even if not all iterations are present, as it reflects
    the true range of iteration numbers with available teacher log-prob data.
    """

    def __init__(
        self,
        logprobs_dir: str,
        tp_rank: int,
        cp_rank: int,
        dp_rank: int,
    ):
        self.logprobs_dir = logprobs_dir
        self.tp_rank = tp_rank
        self.cp_rank = cp_rank
        self.dp_rank = dp_rank

        # Set length to largest iteration number from legacy per-iteration
        # folders/tars.  (Batched tars are handled by _TeacherWebDataset.)
        iter_numbers = []
        for fname in os.listdir(self.logprobs_dir):
            if match := re.match(rf"{FOLDER_NAMES_PREFIX}(\d+)", fname):
                iter_numbers.append(int(match.group(1)))
        if not iter_numbers:
            raise ValueError(f"No iteration folders found in {self.logprobs_dir}")
        self._len = max(iter_numbers) + 1

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, iteration: int):
        return load_log_probs_by_rank(
            self.logprobs_dir,
            iteration,
            self.tp_rank,
            self.cp_rank,
            self.dp_rank,
        )


class _TeacherWebDataset(torch.utils.data.IterableDataset):
    """Streaming dataset that reads teacher log-probs from batched tar shards via WebDataset.

    Supports two on-disk layouts, auto-detected at iteration time:

    * **Unsharded** (current) — ``cp{C}_dp{D}__{B}.tar``.  Saved by TP rank 0
      with the full top-K; every TP rank loads the same file.
    * **TP-sharded** (legacy) — ``tp{T}_cp{C}_dp{D}__{B}.tar``.  Each TP rank
      loads its own shard containing K/tp_size entries.

    **Dynamic shard discovery**: instead of globbing once, the iterator
    processes known shards and, when they are exhausted, re-globs the
    directory to pick up newly-written shards (e.g. from a concurrent
    training job).  If no new shards are found the iterator ends normally.

    Yields ``(values_list, indices_list)`` — the same pair returned by
    ``_TeacherDataset.__getitem__``.

    Requires the ``webdataset`` package (``pip install webdataset``).
    """

    def __init__(
        self,
        logprobs_dir: str,
        tp_rank: int,
        cp_rank: int,
        dp_rank: int,
        start_iteration: int = 0,
    ):
        if wds is None:
            raise ImportError("The 'webdataset' package is required for _TeacherWebDataset.")

        self.logprobs_dir = logprobs_dir
        self.tp_rank = tp_rank
        self.cp_rank = cp_rank
        self.dp_rank = dp_rank
        self.start_iteration = start_iteration

    def _discover_shards(self, already_processed: set) -> list:
        """Glob for new shards (unsharded first, then legacy sharded)."""
        # Prefer unsharded tars; fall back to legacy TP-sharded.
        for prefix in (
            _batched_tar_prefix(self.cp_rank, self.dp_rank),
            _batched_tar_prefix(self.cp_rank, self.dp_rank, tp_rank=self.tp_rank),
        ):
            all_urls = _sorted_batched_tars(
                glob.glob(os.path.join(self.logprobs_dir, f"{prefix}*.tar"))
            )
            new_urls = []
            for url in all_urls:
                if url in already_processed:
                    continue
                m = _BATCHED_TAR_RE.match(os.path.basename(url))
                if m and int(m.group(1)) >= self.start_iteration:
                    new_urls.append(url)
            if new_urls:
                return new_urls
        return []

    def __iter__(self):
        processed = set()

        while True:
            urls = self._discover_shards(processed)
            if not urls:
                if not processed:
                    raise FileNotFoundError(
                        f"No batched tar shards for "
                        f"cp{self.cp_rank}_dp{self.dp_rank} (or "
                        f"tp{self.tp_rank}_cp{self.cp_rank}_dp{self.dp_rank}) "
                        f"found at or after iteration {self.start_iteration} "
                        f"in '{self.logprobs_dir}'"
                    )
                return
            processed.update(urls)

            pipeline = wds.WebDataset(
                urls, nodesplitter=lambda urls: urls, shardshuffle=False,
            )
            for sample in pipeline:
                iteration, values_list, indices_list = decode_logprobs_sample(sample)
                if iteration < self.start_iteration:
                    continue
                yield values_list, indices_list


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
        dist_nn.functional.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=tp_group)
    student_logprobs = student_logits - sum_exp.log()

    # ---- Gather student log-probs at teacher's top-K positions (local shard only) ----
    local_vocab_size = student_logits.size(-1)
    offset = local_vocab_size * tp_rank
    mask = (teacher_topk_indices >= offset) & (teacher_topk_indices < offset + local_vocab_size)
    # Clamp out-of-range indices to avoid index errors; their contributions are zeroed by the mask
    teacher_local_indices = (teacher_topk_indices - offset).clamp(0, local_vocab_size - 1)
    # In TP > 1, may contain duplicate values due to clamping above, so they are masked downstream
    student_topk_logprobs = torch.gather(student_logprobs, -1, teacher_local_indices)

    # ---- Add a "ghost" token containing sum of non-top-K probabilities to both student and teacher ----
    if add_ghost_token:
        eps = 1e-6
        student_topk_logprobs_exp = student_topk_logprobs.exp() * mask  # don't sum duplicate indices if any
        student_topk_exp_sum = student_topk_logprobs_exp.sum(dim=-1, keepdim=True)
        if tp_size > 1:
            dist_nn.functional.all_reduce(student_topk_exp_sum, op=dist.ReduceOp.SUM, group=tp_group)
        student_residual = torch.log(1.0 - student_topk_exp_sum + eps)
        teacher_residual = torch.log(1.0 - teacher_topk_logprobs.exp().sum(dim=-1, keepdim=True).clamp(max=1-eps))
        student_topk_logprobs = torch.cat([student_topk_logprobs, student_residual], dim=-1)
        teacher_topk_logprobs = torch.cat([teacher_topk_logprobs, teacher_residual], dim=-1)
        mask = torch.cat([mask, mask.new_ones(*mask.shape[:-1], 1)], dim=-1)

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

    1. Retrieves (or prefetches via the DataLoader) this rank's saved shard of
       the teacher's top-K log-probabilities and global vocab indices.
    2. Computes the student's globally-normalised log-probabilities via a
       TP-aware softmax, gathers them at the teacher's top-K positions, and
       returns the **forward KL divergence**
       ``KL(teacher ‖ student) = Σ_k p_T(k)·[log p_T(k) − log p_S(k)]``
       averaged over all token positions.

    A :class:`torch.utils.data.DataLoader` with ``pin_memory=True`` and
    background workers prefetches upcoming iterations from disk so that I/O
    overlaps with GPU compute.  Because the tensors arrive in pinned host
    memory, the subsequent ``tensor.to(device, non_blocking=True)`` call can
    overlap the DMA transfer with ongoing GPU kernels.

    Two dataset backends are supported and auto-detected:

    * **WebDataset** (:class:`_TeacherWebDataset`) – sequential streaming of
      batched tar shards.  Used when unsharded (``cp{C}_dp{D}__{B}.tar``)
      or legacy TP-sharded (``tp{T}_cp{C}_dp{D}__{B}.tar``) tars are found.
    * **Map-style** (:class:`_TeacherDataset`) – random-access by iteration
      number.  Fallback for legacy per-iteration folders/tars.

    The DataLoader is initialised lazily on the first ``__call__`` because the
    starting iteration is not known at construction time.

    Args:
        logprobs_dir: Root directory containing log-probs data written by
            :class:`LogitsSaverHooks`.
        num_workers: Number of DataLoader background workers (0 = main
            process).  ``1`` is recommended for sequential prefetching.
        prefetch_factor: How many iterations each worker pre-loads ahead
            (ignored when ``num_workers == 0``).
    """

    def __init__(
        self,
        logprobs_dir: str,
        num_workers: int = 1,
        prefetch_factor: int = 2,
    ):
        self.logprobs_dir = logprobs_dir
        self._num_workers = num_workers
        self._prefetch_factor = prefetch_factor

        # ---- parallel-state bookkeeping ----
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self.tp_group = parallel_state.get_tensor_model_parallel_group()
        self.cp_rank = parallel_state.get_context_parallel_rank()
        self.dp_rank = parallel_state.get_data_parallel_rank()

        # ---- backend selection ----
        self._use_webdataset, self._teacher_is_sharded = self._resolve_backend()

        # ---- DataLoader (lazy-initialised on first call) ----
        self._dataloader_iter: Optional[Iterator] = None

        # ---- iteration / microbatch tracking ----
        self._current_iteration: Optional[int] = None
        self._microbatch_counter: int = 0

        # ---- current iteration's teacher data (pinned CPU tensors) ----
        self._current_values: Optional[List[torch.Tensor]] = None
        self._current_indices: Optional[List[torch.Tensor]] = None

    def _resolve_backend(self) -> tuple:
        """Auto-detect dataset backend and whether teacher data is TP-sharded.

        Returns:
            ``(use_webdataset, teacher_is_sharded)``.
        """
        unsharded_prefix = _batched_tar_prefix(self.cp_rank, self.dp_rank)
        sharded_prefix = _batched_tar_prefix(self.cp_rank, self.dp_rank, tp_rank=self.tp_rank)

        if glob.glob(os.path.join(self.logprobs_dir, f"{unsharded_prefix}*.tar")):
            return True, False
        if glob.glob(os.path.join(self.logprobs_dir, f"{sharded_prefix}*.tar")):
            return True, True
        return False, True  # map-style fallback ⇒ legacy sharded

    def _init_dataloader(self, start_iteration: int) -> None:
        """Create the DataLoader starting from *start_iteration*."""
        if self._use_webdataset:
            dataset = _TeacherWebDataset(
                self.logprobs_dir,
                self.tp_rank,
                self.cp_rank,
                self.dp_rank,
                start_iteration=start_iteration,
            )
            sampler = None
        else:
            dataset = _TeacherDataset(
                self.logprobs_dir, self.tp_rank, self.cp_rank, self.dp_rank,
            )
            if len(dataset) <= start_iteration:
                raise ValueError(
                    f"Start iteration {start_iteration} greater than "
                    f"dataset length {len(dataset)}"
                )
            sampler = range(start_iteration, len(dataset))
        # WebDataset yields a single ordered stream, so multiple workers
        # would split shards and interleave results non-deterministically.
        # Cap at 1 worker to keep iteration order while still prefetching.
        loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=None,
            collate_fn=lambda x: x,
            pin_memory=True,
            num_workers=1 if self._use_webdataset else self._num_workers,
            prefetch_factor=self._prefetch_factor,
            persistent_workers=True,
        )
        self._dataloader_iter = iter(loader)

    def _advance_iteration(self) -> None:
        """Fetch the next iteration's data from the DataLoader."""
        assert self._dataloader_iter is not None
        try:
            values_list, indices_list = next(self._dataloader_iter)
        except StopIteration:
            raise FileNotFoundError(
                f"No more teacher log-prob data available in "
                f"{self.logprobs_dir}.  The DataLoader has been exhausted."
            )
        except Exception as e:
            if zstandard is not None and isinstance(e, zstandard.ZstdError):
                # We skip the loss on these rare missing shards later.
                values_list, indices_list = None, None
            else:
                raise
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

        # If any TP rank failed to load teacher data (e.g. ZstdError), skip KD for the whole TP group.
        local_missing = self._current_values is None or self._current_indices is None
        missing_any_tp = torch.tensor(
            [local_missing], device=student_logits.device, dtype=torch.bool
        )
        if self.tp_size > 1:
            dist.all_reduce(missing_any_tp, op=dist.ReduceOp.MAX, group=self.tp_group)
        if missing_any_tp.item():
            logger.warning(f"Skipping KD for TP rank {self.tp_rank} due to missing teacher data for iteration {iteration}")
            # Sum over vocab so shape matches topk_kl_div (B, S); *0 keeps a grad path from logits.
            return (student_logits * 0).sum(dim=-1).transpose(0, 1).contiguous()

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

        if self._teacher_is_sharded and self.tp_size > 1:
            # Legacy TP-sharded: each rank loaded K/tp_size entries, reassemble.
            teacher_values_list = [torch.empty_like(teacher_values) for _ in range(self.tp_size)]
            teacher_indices_list = [torch.empty_like(teacher_indices) for _ in range(self.tp_size)]
            torch.distributed.all_gather(teacher_values_list, teacher_values, group=self.tp_group)
            torch.distributed.all_gather(teacher_indices_list, teacher_indices, group=self.tp_group)
            teacher_values_full = torch.cat(teacher_values_list, dim=-1)
            teacher_indices_full = torch.cat(teacher_indices_list, dim=-1)
        else:
            # Unsharded: every rank already has the full top-K.
            teacher_values_full = teacher_values
            teacher_indices_full = teacher_indices

        logger.debug("[TP%s-CP%s-DP%s]: Iter_%s Batch_%s",
                     self.tp_rank, self.cp_rank, self.dp_rank, iteration, microbatch_idx)

        # ---- compute loss ----
        return topk_kl_div(
            student_logits,
            teacher_values_full,
            teacher_indices_full,
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
        num_workers: int = 1,
        prefetch_factor: int = 2,
        kd_loss_alpha: float = 0.5,
    ):
        self.logprobs_dir = logprobs_dir
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.kd_func = None
        self.alpha = kd_loss_alpha

    @staticmethod
    def _mask_loss(output_tensor, loss_mask):
        """Apply mask to the unreduced loss tensor."""
        losses = output_tensor.view(-1).float()
        loss_mask = loss_mask.reshape(-1).float()
        return torch.sum(losses * loss_mask)

    def __call__(self, loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: GPTModel):
        """Loss function wrapper for compatibility with Megatron LM training loop.

        Args:
            loss_mask (Tensor): Used to mask out some portions of the loss
            output_tensor (Tensor): The tensor with the losses
            model (GPTModel): The model (can be wrapped)
        """
        if self.kd_func is None:
            # Construct here so parallel_state is initialized by now
            self.kd_func = CachedLogitsKDLoss(
                logprobs_dir=self.logprobs_dir,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
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
        unwrapped_model = unwrap_model(model)
        logits = unwrapped_model.logits
        del unwrapped_model.logits  # free memory during next fwd step
        loss_kd = self.kd_func(logits)
        loss_kd = self._mask_loss(loss_kd, loss_mask)
        # Requires extra TP reduction
        dist.all_reduce(loss_kd, group=parallel_state.get_tensor_model_parallel_group())
        report["logits distillation loss"] = torch.cat([loss_kd.clone().detach().view(1), num_tokens.view(1)])

        # Blend the two
        if loss_lm > 0 and loss_kd > 0:
            loss_kd_scaled = loss_kd * (loss_lm.detach() / loss_kd.detach())
        else:
            loss_kd_scaled = 0
        loss_total = self.alpha * loss_kd_scaled + (1 - self.alpha) * loss_lm
        report["total loss"] = torch.cat([loss_total.clone().detach().view(1), num_tokens.view(1)])

        return loss_total, num_tokens, report
