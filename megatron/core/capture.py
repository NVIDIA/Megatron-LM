# Copyright (c) 2026, Zyphra Technologies Inc. All rights reserved.
"""Lightweight, env-var gated expert-affinity / route capture for the MCore GPT + MoE path.

This module provides a process-global :class:`CaptureManager` that records
per-iteration routing information (and optionally token ids / input
embeddings) during a forward pass and dumps it to disk.

It is intended for analysis runs (e.g. "which expert path does each token in
the sequence take?", from which the expert co-activation / affinity matrix is
built). When disabled (the default) every entry point is a cheap attribute
read, so it is safe to leave the instrumentation in the model.

Everything is controlled by environment variables so it can be toggled from a
launch script with no code / arg changes:

    MOE_ROUTE_CAPTURE            master switch (1/true to enable; default off)
    MOE_ROUTE_CAPTURE_EXPERTS    capture expert_paths + routing_probs (default on
                            when master switch is on)
    MOE_ROUTE_CAPTURE_TOKENS     capture token_ids                    (default on)
    MOE_ROUTE_CAPTURE_EMBEDDINGS capture input_embeddings             (default off,
                            this one is the memory-heavy stream)
    MOE_ROUTE_CAPTURE_DIR        output directory (default ./moe_route_capture)
    MOE_ROUTE_CAPTURE_START      first iteration to capture            (default 0)
    MOE_ROUTE_CAPTURE_CONS_ITERS consecutive iterations to capture per window, from
                            START (default 0 = unlimited).
    MOE_ROUTE_CAPTURE_INTERVAL   spacing (iters) between window starts (default 0 =
                            single window then stop; >0 repeats the window)
    MOE_ROUTE_CAPTURE_NUM_RANKS  capture only global ranks 0..N-1 (default 0 = all
                            ranks). N ranks => N/world_size of samples per iter.
    MOE_ROUTE_CAPTURE_VERBOSE    print one line per captured iteration on rank 0
                            (default off)

The on-disk layout is byte-for-byte compatible with the ds_v3_routing /
``inspect_routing.py`` captures, so the same downstream tooling reads both.

Output filename: every shard is written as
``{dir}/iter_{iteration:08d}_mb{mb}_rank{rank}.pt`` (including rank0/mb0), so all
files for an iteration are named uniformly and easy to enumerate. The
``inspect_routing`` consumers key off the in-file payload, not the filename, so
they still read these directly.

Each file is a dict matching the inspect_routing layout:

    {
      "iteration":          int,
      "token_ids":          Tensor[b, s]               int32   (if enabled)
      "input_embeddings":   Tensor[b, s, h]            bf16    (if enabled)
      "moe_layer_indices":  list[int]   length L_moe
      "expert_paths":       Tensor[b, s, L_moe, top_k] uint8   (ids 0..num_experts-1)
      "routing_probs":      Tensor[b, s, L_moe, top_k] <native> (bf16 under --bf16)
      # extra metadata (ignored by inspect_routing consumers):
      "micro_batch":        int,
      "parallel": {tp_rank, tp_size, cp_rank, cp_size, pp_rank, pp_size,
                   ep_rank, ep_size, dp_rank, global_rank},
    }

Note on sharding: under context-parallel or sequence-parallel, the sequence
dimension is split across ranks, so each rank's file holds only its local slice
of the sequence. Use the ``parallel`` metadata to reassemble offline. Under
pipeline parallelism each stage only holds a subset of MoE layers, so
``moe_layer_indices`` differs per pp rank. Under virtual pipeline parallelism
each model chunk's forward is a separate ``micro_batch`` shard.
"""

import os

import torch


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None or val.strip() == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


class CaptureManager:
    """Process-global capture state. Access via :func:`get_capturer`."""

    def __init__(self):
        self.enabled = _env_flag("MOE_ROUTE_CAPTURE", False)

        # Per-stream switches (only meaningful when master switch is on).
        self.capture_experts = self.enabled and _env_flag("MOE_ROUTE_CAPTURE_EXPERTS", True)
        self.capture_tokens = self.enabled and _env_flag("MOE_ROUTE_CAPTURE_TOKENS", True)
        self.capture_embeddings = self.enabled and _env_flag("MOE_ROUTE_CAPTURE_EMBEDDINGS", False)

        self.out_dir = os.environ.get("MOE_ROUTE_CAPTURE_DIR", "./moe_route_capture")
        self.start_iter = _env_int("MOE_ROUTE_CAPTURE_START", 0)
        # How many CONSECUTIVE iterations to capture per window, starting at
        # START. 0 == unlimited (capture every iter from START onwards).
        self.cons_iters = _env_int("MOE_ROUTE_CAPTURE_CONS_ITERS", 0)
        # Spacing (in iters) between window starts. 0 (default) == a single
        # window: capture cons_iters from START, then stop. > 0 repeats the
        # same-sized window every INTERVAL iters (e.g. START, START+INTERVAL,
        # ...). The window must fit inside the interval (asserted) so windows
        # never overlap.
        self.interval = _env_int("MOE_ROUTE_CAPTURE_INTERVAL", 0)
        # MOE_ROUTE_CAPTURE_MAX_MB: record only the first N microbatches of each captured iteration (0 = all).
        self.max_mb = _env_int("MOE_ROUTE_CAPTURE_MAX_MB", 0)
        # Limit capture to the first N global ranks (0..N-1). 0 (default) ==
        # all ranks. Each rank holds a different slice of the batch, so N ranks
        # captures N/world_size of the samples per iter (fewer files & storage).
        self.num_ranks = _env_int("MOE_ROUTE_CAPTURE_NUM_RANKS", 0)
        self.verbose = _env_flag("MOE_ROUTE_CAPTURE_VERBOSE", False)

        # Resolved window params, filled lazily once the global batch size is
        # known (see _resolve_window): iters-per-window, spacing, and the total
        # captured-iter cap (0 == unlimited).
        self._resolved = False
        self._win_iters = 0
        self._win_interval = 1
        self._win_max = 0

        # Runtime state.
        self._captured_iters = set()
        self._active = False  # is the current forward being captured?
        self._cur_iter = -1
        self._mb_index = 0  # micro-batch counter within an iteration
        self._last_seen_iter = -1
        self._router = {}  # layer_number -> (indices, probs)
        self._tokens = None
        self._embeddings = None

        if self.enabled and self._is_writer_rank():
            os.makedirs(self.out_dir, exist_ok=True)

    @property
    def is_active(self) -> bool:
        """True only during a forward pass that is being captured."""
        return self._active

    @property
    def wants_router(self) -> bool:
        """True when the current forward should record router decisions.

        Cheap single-call guard for the hot path in ``TopKRouter.routing``.
        """
        return self._active and self.capture_experts

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _global_rank() -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0

    def _is_writer_rank(self) -> bool:
        # Every rank writes its own file (the sequence may be sharded across
        # CP / TP-SP ranks), so all ranks are writers.
        return True

    def _current_iteration(self) -> int:
        try:
            from megatron.training.global_vars import get_args

            return int(getattr(get_args(), "curr_iteration", 0))
        except Exception:
            return 0

    # -- forward-pass lifecycle -----------------------------------------

    def begin_step(self, token_ids=None, input_embeddings=None):
        """Called at the top of the model forward. Decides whether to capture
        this forward pass and stashes token ids / embeddings if so."""
        if not self.enabled:
            return

        it = self._current_iteration()

        # Track micro-batch index within an iteration (forward is called once
        # per micro-batch with the same curr_iteration).
        if it != self._last_seen_iter:
            self._last_seen_iter = it
            self._mb_index = 0
        else:
            self._mb_index += 1

        self._cur_iter = it
        self._router = {}
        self._tokens = None
        self._embeddings = None

        self._active = self._should_capture(it)
        if not self._active:
            return

        if self.capture_tokens and token_ids is not None:
            # input_ids is [b, s] -> matches inspect_routing token_ids [b, s].
            self._tokens = token_ids.detach().to("cpu", torch.int32, copy=True)
        if self.capture_embeddings and input_embeddings is not None:
            # Megatron embeddings are [s, b, h]; inspect_routing wants [b, s, h].
            self._embeddings = (
                input_embeddings.detach().transpose(0, 1).to("cpu", torch.bfloat16, copy=True)
            )

    def _resolve_window(self):
        """Resolve the consecutive-iter capture window.

        cons_iters -> window of that many consecutive iters.
          - interval == 0: a SINGLE window -> capture cons_iters consecutive
            iters from START, then stop.
          - interval  > 0: REPEAT that window every `interval` iters; assert the
            window fits inside the interval so windows never overlap.
        cons_iters == 0 -> unlimited: capture every iter from START.
        """
        if self._resolved:
            return
        self._resolved = True

        self._win_iters = max(1, self.cons_iters) if self.cons_iters > 0 else 0

        if self.interval > 0:
            if self._win_iters and self._win_iters > self.interval:
                raise ValueError(
                    f"MOE_ROUTE_CAPTURE: consecutive window MOE_ROUTE_CAPTURE_CONS_ITERS="
                    f"{self.cons_iters} exceeds MOE_ROUTE_CAPTURE_INTERVAL={self.interval}; "
                    f"windows would overlap. Increase the interval or reduce "
                    f"MOE_ROUTE_CAPTURE_CONS_ITERS."
                )
            # Repeated windows of _win_iters every `interval` iters; unlimited.
            self._win_interval = self.interval
            self._win_max = 0
        else:
            # Single consecutive window from START, then stop.
            self._win_interval = max(1, self._win_iters)
            self._win_max = self._win_iters  # 0 (unlimited) if cons_iters==0

    def _should_capture(self, it: int) -> bool:
        self._resolve_window()
        # Only the first num_ranks global ranks capture (0 == all ranks).
        if self.num_ranks > 0 and self._global_rank() >= self.num_ranks:
            return False
        if it < self.start_iter:
            return False
        # Inside the consecutive window at the start of each interval: phase
        # 0.._win_iters-1 captures, the rest of the interval is skipped. When
        # _win_iters == 0 (unlimited) every iter from START is captured.
        if self._win_iters and (it - self.start_iter) % self._win_interval >= self._win_iters:
            return False
        if self._win_max > 0 and len(self._captured_iters) >= self._win_max:
            return it in self._captured_iters  # allow finishing an in-flight iter
        return True

    def record_router(self, layer_number: int, indices: torch.Tensor, probs: torch.Tensor):
        """Record per-token expert choices for one MoE layer.

        ``indices`` and ``probs`` are expected with shape [b, s, top_k], where
        slot k=0 is the top-1 (highest-scoring) expert and probs[..., k] is the
        gate weight of the expert at indices[..., k]. ``indices`` are
        router-local expert ids in [0, num_experts) (matching inspect_routing's
        convention).

        Note: ``indices`` already reflects the model's true selection
        (topk over scores + expert_bias), so the biases themselves are not
        captured here.
        """
        if not (self._active and self.capture_experts):
            return
        if self.max_mb and self._mb_index >= self.max_mb:
            return
        # Storage-lean, lossless capture:
        #   - expert ids are non-negative and < num_experts, so uint8 holds
        #     them exactly for the usual configs (<=256 experts) at half the
        #     size of int16. Fall back to int16 beyond that.
        #   - routing probs are kept in their NATIVE dtype (bf16 under --bf16,
        #     matching what the model/CUDA actually computed) instead of being
        #     cast to fp16, which would be a lossy reinterpretation for the
        #     same byte cost.
        idx_dtype = torch.uint8 if int(indices.max()) <= 255 else torch.int16
        self._router[layer_number] = (
            indices.detach().to("cpu", idx_dtype, copy=True),
            probs.detach().to("cpu", copy=True),
        )

    def end_step(self):
        """Called after the decoder forward. Writes the captured data to disk."""
        if not (self.enabled and self._active):
            return
        if self.max_mb and self._mb_index >= self.max_mb:
            return

        payload = {
            "iteration": self._cur_iter,
            "micro_batch": self._mb_index,
            "parallel": self._parallel_metadata(),
        }

        if self._tokens is not None:
            payload["token_ids"] = self._tokens
        if self._embeddings is not None:
            payload["input_embeddings"] = self._embeddings

        if self.capture_experts and self._router:
            layers = sorted(self._router.keys())
            # Stack along a new layer axis: [b, s, L_moe, top_k].
            expert_paths = torch.stack([self._router[l][0] for l in layers], dim=-2)
            routing_probs = torch.stack([self._router[l][1] for l in layers], dim=-2)
            payload["moe_layer_indices"] = layers
            payload["expert_paths"] = expert_paths
            payload["routing_probs"] = routing_probs

        rank = self._global_rank()
        # Uniform per-(micro-batch, rank) filename so every shard is named
        # consistently and easy to enumerate downstream (including rank0/mb0).
        fname = f"iter_{self._cur_iter:08d}_mb{self._mb_index}_rank{rank}.pt"
        torch.save(payload, os.path.join(self.out_dir, fname))

        if self.verbose and rank == 0:
            shape = tuple(payload["expert_paths"].shape) if "expert_paths" in payload else None
            print(
                f"[capture] iter={self._cur_iter} mb={self._mb_index} rank={rank} "
                f"layers={payload.get('moe_layer_indices')} expert_paths={shape} -> {fname}",
                flush=True,
            )

        self._captured_iters.add(self._cur_iter)

        # Clear buffers.
        self._active = False
        self._router = {}
        self._tokens = None
        self._embeddings = None

    def _parallel_metadata(self) -> dict:
        try:
            from megatron.core import parallel_state as ps

            return {
                "global_rank": self._global_rank(),
                "tp_rank": ps.get_tensor_model_parallel_rank(),
                "tp_size": ps.get_tensor_model_parallel_world_size(),
                "cp_rank": ps.get_context_parallel_rank(),
                "cp_size": ps.get_context_parallel_world_size(),
                "pp_rank": ps.get_pipeline_model_parallel_rank(),
                "pp_size": ps.get_pipeline_model_parallel_world_size(),
                "ep_rank": ps.get_expert_model_parallel_rank(),
                "ep_size": ps.get_expert_model_parallel_world_size(),
                "dp_rank": ps.get_data_parallel_rank(),
            }
        except Exception:
            return {"global_rank": self._global_rank()}


_CAPTURER = None


def get_capturer() -> CaptureManager:
    """Return the process-global capture manager (constructed on first use)."""
    global _CAPTURER
    if _CAPTURER is None:
        _CAPTURER = CaptureManager()
    return _CAPTURER
