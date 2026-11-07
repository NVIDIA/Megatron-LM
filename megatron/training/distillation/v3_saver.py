# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Paired-input saver extending the existing sparse target hook and async queue."""

import json
import os
from collections import OrderedDict
from typing import Any

import torch
import torch.distributed as dist

from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.training import get_args, get_tensorboard_writer

from .format.arrow import samples_batch, write_file
from .logits_saver import LogitsSaverHooks
from .v3_format import MAX_VOCAB_SIZE, capture_batch, finalize_record, join_inputs, tail_logprob
from .v3_replay import layout_options, token_map
from .v3_storage import Storage, data_name, parse_name, publish


class PairedLogitsSaver(LogitsSaverHooks):
    """Reuse teacher top-K extraction while persisting full CPU input records."""

    _max_vocab_size = MAX_VOCAB_SIZE

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Hooks are installed before load_checkpoint restores the sample cursor.
        # Initialize the cache generation only when training is about to start.
        self._initialized = False
        self._captured = []
        self._pending_input = None
        self._iteration_records = None
        self._attempt_microbatch = 0

    def initialize(self) -> None:
        """Initialize metadata after checkpoint restoration, once per run."""
        if self._initialized:
            return
        from .v3_runtime import initialize_dump_metadata

        self.metadata_dict = initialize_dump_metadata(self)
        self._meta_bytes = json.dumps(self.metadata_dict, sort_keys=True).encode()
        self._initialized = True

    def capture(self, batch: dict, *, hybrid: bool = False) -> None:
        """Capture the complete input before pipeline-stage fields are cleared."""
        if self.tp_rank != 0:
            return
        inputs = capture_batch(batch)
        mapping = token_map(inputs, self.cp_rank, self.cp_size, **self._layout_options(hybrid))
        self._pending_input = (inputs, mapping)

    def _process_single_microbatch(self, logits: torch.Tensor) -> tuple | None:
        """Also return each row's fp32 tail log probability, taken before downcasting."""
        result = self._topk_logprobs(logits)
        if result is None:
            return None
        values, indices = result
        return values.to(self._save_dtype), indices, tail_logprob(values)

    def _layout_options(self, hybrid: bool) -> dict:
        # The hybrid layouts come from a full TransformerConfig; build it once, not per microbatch.
        cache = self.__dict__.setdefault("_layout_cache", {})
        if hybrid not in cache:
            cache[hybrid] = layout_options(get_args(), hybrid=hybrid)
        return cache[hybrid]

    def begin_attempt(self) -> None:
        """Discard provisional data before an iteration execution or rerun."""
        self.initialize()
        self._captured.clear()
        self._iteration_records = None
        self._attempt_microbatch = 0
        self._pending_input = None
        self._curr_mtp_passes = 0
        self._topp_kept_counts = []

    def _forward_hook(self, module: Any, inputs: Any, output: Any) -> None:
        if not module.training:
            return
        logits = output[0] if isinstance(output, tuple) else output
        if self._attempt_microbatch >= get_num_microbatches():
            raise RuntimeError(
                "Unexpected duplicate teacher output; v3 does not support full-layer recomputation"
            )
        with torch.no_grad():
            result = self._process_single_microbatch(logits)
        self._attempt_microbatch += 1
        if result is None:
            return
        if self._pending_input is None:
            raise RuntimeError("Teacher v3 output was not paired with a captured input batch")
        captured, mapping = self._pending_input
        self._pending_input = None
        self._captured.append((captured, mapping, *result))
        if len(self._captured) == get_num_microbatches():
            self._serialize_attempt()

    def _serialize_attempt(self) -> None:
        batches, all_values, all_indices, all_tails = [], [], [], []
        for captured, mapping, values, indices, tail in self._captured:
            if tuple(mapping.shape) != tuple(values.shape[:2]):
                raise ValueError(
                    "Teacher output shape does not match its captured CP token mapping"
                )
            mapping = mapping.to(values.device)
            gathered = []
            for tensor in (mapping, values, indices, tail):
                parts = (
                    [torch.empty_like(tensor) for _ in range(self.cp_size)]
                    if self.cp_rank == 0
                    else None
                )
                if self.cp_size > 1:
                    dist.gather(tensor, parts, dst=self._cp_dst_rank_global, group=self.cp_group)
                else:
                    parts = [tensor]
                gathered.append(parts)
            if self.cp_rank != 0:
                continue
            k = values.shape[-1]
            full_values = torch.full((captured["tokens"].numel(), k), -1e3, dtype=values.dtype)
            full_indices = torch.zeros_like(full_values, dtype=torch.int64)
            full_tail = torch.zeros(full_values.shape[0], dtype=torch.float32)
            seen = torch.zeros(full_values.shape[0], dtype=torch.bool)
            for mapping_part, values_part, indices_part, tail_part in zip(*gathered):
                positions = mapping_part.cpu().reshape(-1)
                valid = positions >= 0
                positions = positions[valid]
                if seen[positions].any():
                    raise ValueError("Teacher CP mapping repeats cached token positions")
                seen[positions] = True
                full_values[positions] = values_part.cpu().reshape(-1, k)[valid]
                full_indices[positions] = indices_part.cpu().reshape(-1, k)[valid]
                full_tail[positions] = tail_part.cpu().reshape(-1)[valid]
            if ((~seen) & captured["loss_mask"].bool()).any():
                raise ValueError("Teacher CP output is missing unmasked target positions")
            batches.append(captured)
            all_values.append(full_values)
            all_indices.append(full_indices)
            all_tails.append(full_tail)
        if self.cp_rank == 0:
            args = get_args()
            contract = self.metadata_dict["contract"]
            # Store the padding-free record, with target rows only where loss is applied.
            inputs, values, indices, tail = finalize_record(
                join_inputs(batches),
                torch.cat(all_values),
                torch.cat(all_indices),
                torch.cat(all_tails),
                isolated=contract["attention_mode"] == "isolated_segments",
            )
            start = args.consumed_train_samples
            end = start + get_num_microbatches() * self.dp_size * args.micro_batch_size
            self._iteration_records = {
                "start": start,
                "end": end,
                "batch": samples_batch(
                    inputs, values, indices, tail, vocab_size=contract["logprob_vocab_size"]
                ),
            }
        self._captured.clear()

    def commit_attempt(self) -> None:
        """Commit only the final selected execution, before checkpoint flushing."""
        if self.tp_rank == 0 and self.cp_rank == 0:
            if self._iteration_records is None:
                raise RuntimeError("Teacher v3 iteration did not produce all paired records")
            if self._topp_kept_counts and (writer := get_tensorboard_writer()) is not None:
                args = get_args()
                iteration = getattr(args, "curr_iteration", None)
                if iteration is None:
                    iteration = args.iteration
                writer.add_scalar(
                    "avg-logprobs-kept",
                    sum(self._topp_kept_counts) / len(self._topp_kept_counts),
                    iteration,
                )
            record = self._iteration_records
            # Resume can replay a window already published before a weight checkpoint.
            if record["end"] > self.metadata_dict["published_through"]:
                self._pending_writes[(record["start"], record["end"])] = record
            self._iteration_records = None
        self._topp_kept_counts.clear()

    def take_pending_data(self) -> tuple:
        """Transfer ownership to the existing persistent async checkpoint worker."""
        writes = self._pending_writes
        self._pending_writes = OrderedDict()
        name = ""
        if writes:
            start, end = min(a for a, _ in writes), max(b for _, b in writes)
            name = os.path.join(self.save_dir, data_name(self.dp_rank, start, end))
        return (
            name,
            writes,
            self._meta_bytes,
            self.save_dir.startswith("msc://"),
            [],
            self._failure_event,
        )

    @staticmethod
    def _write_batched_tar(
        tar_path: str,
        writes: OrderedDict,
        meta_bytes: bytes,
        msc_enabled: bool = False,
        existing_tars: Any = None,
        failure_event: Any = None,
    ) -> None:
        if not writes:
            return
        try:
            if msc_enabled:
                # Persistent workers do not inherit the main process's feature flags.
                from megatron.core.msc_utils import MultiStorageClientFeature

                MultiStorageClientFeature.enable()
            metadata = json.loads(meta_bytes)
            # Resume-only state is not part of the immutable file metadata.
            metadata.pop("published_through", None)
            storage = Storage(os.path.dirname(tar_path))
            name = parse_name(os.path.basename(tar_path))
            data, index = write_file([record["batch"] for record in writes.values()], metadata)
            publish(storage, name, data, index)
        except Exception:
            if failure_event is not None:
                failure_event.set()
            raise
