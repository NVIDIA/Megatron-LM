# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Router decision tracing for MoE models for both training and inference.

Captures per-layer top-K routing decisions to a JSONL file for offline
analysis of routing patterns (e.g., layer-to-layer top-K overlap, expert
load balance, predictor accuracy for speculative expert prefetch).

- **Inference mode** (default): step boundaries are auto-detected by
  watching for layer repeats.  When a layer that already fired this step
  fires again, a new step has started.  Enable via ``--moe-routing-trace-path``
  in the inference launcher.

- **Training mode** (``training_mode=True``): the training loop drives step
  boundaries explicitly by calling ``advance_step()`` once per iteration.
  Enable via ``--moe-routing-trace-path`` in the training launcher.

Output format — one JSONL file per rank, one record per (step, layer)::

    {"step": 0, "layer": 3, "rank": 0, "num_tokens": 128, "topk": 22,
     "top_indices": [[12, 45, ...], ...]}

Optional sidecar binary files (same directory):
- ``hidden_states_rank{rank}.bin`` — bfloat16 hidden-state tensors; each
  JSONL record gains ``hs_offset``, ``hs_bytes``, ``hs_shape`` fields.
- ``logits_rank{rank}.bin`` — bfloat16 pre-topk routing logits; each JSONL
  record gains ``logit_offset``, ``logit_bytes``, ``logit_shape`` fields.

Use ``load_hidden_states_for_record`` / ``load_logits_for_record`` to read
sidecar tensors back from the binary files.

Note: Python forward hooks do not fire during CUDA graph replay.  For
complete traces, run with ``--cuda-graph-impl none``.
"""

import atexit
import json
import os
import threading
from typing import List, Optional

import torch

_TRACER: Optional["RouterTracer"] = None


def init_tracer(
    output_dir: str,
    max_steps: int,
    rank: int,
    training_mode: bool = False,
    capture_hidden_states: bool = False,
    capture_logits: bool = False,
    dump_router_weights: bool = False,
) -> None:
    """Initialize the global router tracer.

    Must be called after torch.distributed is initialized and before
    ``register_hooks`` is called on the model.

    Args:
        output_dir: Directory for JSONL trace files (and optional sidecars).
        max_steps: Maximum steps (iterations in training, decode steps in
            inference) to capture before the tracer self-disables.
        rank: Distributed rank; used to name output files.
        training_mode: If True, step boundaries are driven by explicit
            ``advance_step()`` calls from the training loop rather than the
            layer-repeat heuristic used during inference.
        capture_hidden_states: Capture the input hidden-state tensor for each
            router call.  Adds substantial disk cost.
        capture_logits: Capture pre-topk routing logits.
        dump_router_weights: Save router weight tensors to a ``.pt`` file.
    """
    global _TRACER
    if _TRACER is not None:
        return
    _TRACER = RouterTracer(
        output_dir,
        max_steps,
        rank,
        training_mode=training_mode,
        capture_hidden_states=capture_hidden_states,
        capture_logits=capture_logits,
        dump_router_weights=dump_router_weights,
    )
    atexit.register(_TRACER.flush)


def get_tracer() -> Optional["RouterTracer"]:
    """Return the active tracer, or None if tracing is disabled."""
    return _TRACER


def load_hidden_states_for_record(record: dict, trace_dir: str) -> torch.Tensor:
    """Load the hidden-state tensor for a single JSONL record.

    Args:
        record: A parsed JSONL line that contains hs_offset, hs_bytes, hs_shape.
        trace_dir: Directory containing hidden_states_rank{rank}.bin.

    Returns:
        Tensor of shape [num_tokens, hidden_size] in bfloat16.
    """
    if "hs_offset" not in record:
        raise ValueError("Record does not contain hidden-state metadata.")
    path = os.path.join(trace_dir, f"hidden_states_rank{record['rank']}.bin")
    with open(path, "rb") as f:
        f.seek(record["hs_offset"])
        data = f.read(record["hs_bytes"])
    arr = torch.frombuffer(bytearray(data), dtype=torch.int16)
    arr = arr.view(torch.bfloat16).reshape(record["hs_shape"])
    return arr.clone()


def load_logits_for_record(record: dict, trace_dir: str) -> torch.Tensor:
    """Load the pre-topk routing logits for a single JSONL record.

    Args:
        record: A parsed JSONL line that contains logit_offset, logit_bytes,
            logit_shape.
        trace_dir: Directory containing logits_rank{rank}.bin.

    Returns:
        Tensor of shape [num_tokens, num_experts] in bfloat16.
    """
    if "logit_offset" not in record:
        raise ValueError("Record does not contain logit metadata.")
    path = os.path.join(trace_dir, f"logits_rank{record['rank']}.bin")
    with open(path, "rb") as f:
        f.seek(record["logit_offset"])
        data = f.read(record["logit_bytes"])
    arr = torch.frombuffer(bytearray(data), dtype=torch.int16)
    arr = arr.view(torch.bfloat16).reshape(record["logit_shape"])
    return arr.clone()


class RouterTracer:
    """Captures router top-K decisions across all MoE layers per step.

    Works in two modes controlled by ``training_mode``:

    - **Inference mode** (``training_mode=False``): step boundaries are
      auto-detected.  When a layer that has already fired this step fires
      again, a new step has started.
    - **Training mode** (``training_mode=True``): the training loop calls
      ``advance_step()`` at each iteration boundary.  The layer-repeat
      heuristic is not used.

    Recording is skipped during CUDA graph capture — ``D2H`` copies inside a
    captured graph would record stale values on replay.
    """

    def __init__(
        self,
        output_dir: str,
        max_steps: int,
        rank: int,
        training_mode: bool = False,
        capture_hidden_states: bool = False,
        capture_logits: bool = False,
        dump_router_weights: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self.max_steps = max_steps
        self.rank = rank
        self.training_mode = training_mode
        self.step_id = 0
        self.layers_seen_this_step: set[int] = set()
        self.records: list[dict] = []
        self._lock = threading.Lock()
        self._stopped = False
        self.capture_hidden_states = capture_hidden_states
        self.capture_logits = capture_logits
        self.dump_router_weights = dump_router_weights
        self._router_state: dict = {}
        self._hook_handles: List[torch.utils.hooks.RemovableHook] = []

        os.makedirs(output_dir, exist_ok=True)
        self.output_path = os.path.join(output_dir, f"router_trace_rank{rank}.jsonl")
        open(self.output_path, "w").close()

        self.hs_path = os.path.join(output_dir, f"hidden_states_rank{rank}.bin")
        self._hs_file = None
        self._hs_offset = 0
        if self.capture_hidden_states:
            open(self.hs_path, "wb").close()

        self.logits_path = os.path.join(output_dir, f"logits_rank{rank}.bin")
        self._logits_file = None
        self._logits_offset = 0
        if self.capture_logits:
            open(self.logits_path, "wb").close()

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def register_hooks(self, model) -> None:
        """Walk *model* and register forward hooks on every TopKRouter module.

        Accepts a single model or a list of model chunks (the convention used
        throughout the training and inference code).
        """
        from megatron.core.transformer.moe.router import TopKRouter

        if not isinstance(model, (list, tuple)):
            model = [model]

        for chunk in model:
            unwrapped = _unwrap_model(chunk)
            for module in unwrapped.modules():
                if isinstance(module, TopKRouter):
                    handle = module.register_forward_hook(self.make_hook())
                    self._hook_handles.append(handle)

    def remove_hooks(self) -> None:
        """Remove all forward hooks registered by ``register_hooks``."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    # ------------------------------------------------------------------
    # Step control
    # ------------------------------------------------------------------

    def advance_step(self) -> None:
        """Advance to the next step (training mode only).

        Call once per training iteration, *after* the forward-backward pass.
        Flushes accumulated records to disk and increments the step counter.
        Disables the tracer once ``max_steps`` is reached.
        """
        with self._lock:
            self._flush_records_to_disk()
            self.step_id += 1
            self.layers_seen_this_step.clear()
            if self.step_id >= self.max_steps:
                self._stopped = True
                self.remove_hooks()

    # ------------------------------------------------------------------
    # Hook implementation
    # ------------------------------------------------------------------

    def make_hook(self):
        """Build a forward hook callable for a single TopKRouter module."""

        def hook(module, inputs, outputs):
            if self._stopped:
                return
            if torch.cuda.is_current_stream_capturing():
                return
            self._record(module, inputs, outputs)

        return hook

    def _extract_hidden_state(self, inputs, expected_num_tokens):
        if not inputs:
            return None
        hs = inputs[0]
        if not torch.is_tensor(hs):
            return None
        if hs.dim() == 2:
            pass
        elif hs.dim() == 3:
            hs = hs.reshape(-1, hs.shape[-1])
        else:
            return None
        if hs.shape[0] != expected_num_tokens:
            return None
        return hs

    def _record(self, module, inputs, outputs) -> None:
        if not isinstance(outputs, tuple) or len(outputs) != 2:
            return
        _, second = outputs
        if not torch.is_tensor(second):
            return

        layer_number = getattr(module, "layer_number", None)
        if layer_number is None:
            return

        with self._lock:
            if not self.training_mode:
                # Inference mode: detect step boundaries via layer repeats.
                if layer_number in self.layers_seen_this_step:
                    self._flush_records_to_disk()
                    self.step_id += 1
                    self.layers_seen_this_step.clear()
                    if self.step_id >= self.max_steps:
                        self._stopped = True
                        return
                self.layers_seen_this_step.add(layer_number)

            if self.dump_router_weights and layer_number not in self._router_state:
                weight = getattr(module, "weight", None)
                expert_bias = getattr(module, "expert_bias", None)
                score_fn = getattr(
                    getattr(module, "config", None), "moe_router_score_function", None
                )
                topk_attr = getattr(module, "topk", None)
                if torch.is_tensor(weight):
                    self._router_state[layer_number] = {
                        "weight": weight.detach().to("cpu", dtype=torch.float32).clone(),
                        "expert_bias": (
                            expert_bias.detach().to("cpu", dtype=torch.float32).clone()
                            if torch.is_tensor(expert_bias)
                            else None
                        ),
                        "score_function": score_fn,
                        "topk": topk_attr,
                    }

            if second.dtype == torch.bool:
                topk = getattr(module, "topk", None)
                if topk is None:
                    return
                num_tokens = second.shape[0]
                _, expert_idx = second.nonzero(as_tuple=True)
                if expert_idx.numel() != num_tokens * topk:
                    return
                top_indices = expert_idx.view(num_tokens, topk)
            else:
                top_indices = second

            top_indices_cpu = top_indices.detach().to("cpu", torch.int32).contiguous()
            num_tokens = int(top_indices_cpu.shape[0])

            record: dict = {
                "step": self.step_id,
                "layer": int(layer_number),
                "rank": self.rank,
                "num_tokens": num_tokens,
                "topk": int(top_indices_cpu.shape[1]),
                "top_indices": top_indices_cpu.tolist(),
            }

            if self.capture_hidden_states:
                hs = self._extract_hidden_state(inputs, num_tokens)
                if hs is not None:
                    hs_cpu = hs.detach().to("cpu", dtype=torch.bfloat16).contiguous()
                    hs_bytes = hs_cpu.view(torch.int16).numpy().tobytes()
                    if self._hs_file is None:
                        self._hs_file = open(self.hs_path, "ab")
                    self._hs_file.write(hs_bytes)
                    record["hs_offset"] = self._hs_offset
                    record["hs_bytes"] = len(hs_bytes)
                    record["hs_shape"] = list(hs_cpu.shape)
                    self._hs_offset += len(hs_bytes)

            if self.capture_logits:
                hs_for_gating = self._extract_hidden_state(inputs, num_tokens)
                gating_fn = getattr(module, "gating", None)
                if hs_for_gating is not None and callable(gating_fn):
                    try:
                        with torch.no_grad():
                            logits = gating_fn(hs_for_gating)
                        if isinstance(logits, tuple):
                            logits = logits[0]
                    except Exception:
                        logits = None
                    if torch.is_tensor(logits) and logits.shape[0] == num_tokens:
                        logits_cpu = logits.detach().to("cpu", dtype=torch.bfloat16).contiguous()
                        logits_bytes = logits_cpu.view(torch.int16).numpy().tobytes()
                        if self._logits_file is None:
                            self._logits_file = open(self.logits_path, "ab")
                        self._logits_file.write(logits_bytes)
                        record["logit_offset"] = self._logits_offset
                        record["logit_bytes"] = len(logits_bytes)
                        record["logit_shape"] = list(logits_cpu.shape)
                        self._logits_offset += len(logits_bytes)

            self.records.append(record)

    def _flush_records_to_disk(self) -> None:
        if not self.records:
            return
        with open(self.output_path, "a") as f:
            for rec in self.records:
                f.write(json.dumps(rec) + "\n")
        self.records.clear()
        if self._hs_file is not None:
            self._hs_file.flush()
        if self._logits_file is not None:
            self._logits_file.flush()

    def flush(self) -> None:
        """Final flush of any remaining records (registered at atexit)."""
        with self._lock:
            self._flush_records_to_disk()
            if self._hs_file is not None:
                self._hs_file.close()
                self._hs_file = None
            if self._logits_file is not None:
                self._logits_file.close()
                self._logits_file = None
            if self.dump_router_weights and self._router_state:
                weights_path = os.path.join(self.output_dir, f"router_state_rank{self.rank}.pt")
                torch.save(self._router_state, weights_path)
                self._router_state = {}


def _unwrap_model(model):
    """Unwrap DDP/FSDP wrappers to get the underlying module."""
    try:
        from megatron.core.utils import unwrap_model

        return unwrap_model(model)
    except Exception:
        # Fallback: strip common wrappers manually.
        while hasattr(model, "module"):
            model = model.module
        return model
