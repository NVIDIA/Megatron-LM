# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Router decision tracing for MoE models for both training and inference.

Captures per-layer top-K routing decisions to a JSONL file for offline analysis of routing patterns
(e.g., expert load balance, overlap between (layer N-2, N)).
Enable via `--moe-routing-trace-path` in both training and inference.

Output format: one JSONL file per rank, one record per (step, block, layer):
    {"step": 0, "stage": "pre_dispatch", "block": "decoder", "layer": 3,
     "rank": 0, "num_tokens": 128, "topk": 22, "top_indices": [[12, 45, ...], ...]}
MTP records carry an extra "mtp_idx" field so they never collide with decoder
layers that share a layer number.

Optional sidecar binary files written:
- hidden_states_rank{rank}.bin — bfloat16 hidden-state tensors; each
  JSONL record gains `hs_offset`, `hs_bytes`, `hs_shape` fields.
- logits_rank{rank}.bin — bfloat16 pre-topk routing logits; each JSONL
  record gains `logit_offset`, `logit_bytes`, `logit_shape` fields.

Use `load_hidden_states_for_record` / `load_logits_for_record` to read sidecar tensors.

Note: Python forward hooks do not fire during CUDA graph replay.  Run with `--cuda-graph-impl none`.
"""

import atexit
import json
import os
import re
import threading
from typing import List, Optional, Tuple

import torch

_MOE_ROUTER_TRACER: Optional["RouterTracer"] = None

# Locate the layer that a router module belongs to based on its name.
_MTP_LAYER_RE = re.compile(r'mtp\.layers\.(\d+)\.mtp_model_layer\.layers\.(\d+)\.')
_DECODER_LAYER_RE = re.compile(r'decoder\.layers\.(\d+)\.')


def _parse_router_module_name(module_name: str) -> Optional[Tuple[str, Optional[int], int]]:
    """Parse a router module name into (block, mtp_idx, layer).

    Returns None if the name matches neither the decoder nor the MTP pattern.

    Examples:
        decoder.layers.3.mlp.router                         -> ("decoder", None, 3)
        mtp.layers.0.mtp_model_layer.layers.1.mlp.router    -> ("mtp", 0, 1)
    """
    mtp_match = _MTP_LAYER_RE.search(module_name)
    if mtp_match:
        return "mtp", int(mtp_match.group(1)), int(mtp_match.group(2))
    decoder_match = _DECODER_LAYER_RE.search(module_name)
    if decoder_match:
        return "decoder", None, int(decoder_match.group(1))
    return None


def init_moe_router_tracer(
    output_dir: str,
    max_steps: int,
    rank: int,
    training_mode: bool = False,
    capture_hidden_states: bool = False,
    capture_logits: bool = False,
    dump_router_weights: bool = False,
) -> None:
    """Initialize the global router tracer.
    Call after torch.distributed is initialized and before `register_hooks` is called on the model.

    Args:
        output_dir: Directory for JSONL trace files (and optional sidecars).
        max_steps: Maximum steps (iterations in training, decode steps in inference) to capture.
        rank: Distributed rank.
        training_mode: If True, step boundaries are driven by advance_step() calls from the training
        loop rather than the layer-repeat heuristic used during inference.
        capture_hidden_states: Capture the input hidden-state tensor for each router call.
        capture_logits: Capture pre-topk routing logits.
        dump_router_weights: Save router weight tensors to a .pt file.
    """
    global _MOE_ROUTER_TRACER
    if _MOE_ROUTER_TRACER is not None:
        return
    _MOE_ROUTER_TRACER = RouterTracer(
        output_dir,
        max_steps,
        rank,
        training_mode=training_mode,
        capture_hidden_states=capture_hidden_states,
        capture_logits=capture_logits,
        dump_router_weights=dump_router_weights,
    )
    atexit.register(_MOE_ROUTER_TRACER.flush)


def get_moe_router_tracer() -> Optional["RouterTracer"]:
    """Return the active tracer, or None if tracing is disabled."""
    return _MOE_ROUTER_TRACER


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
    # torch.frombuffer keeps the bytearray alive via the tensor's storage, so the
    # view is safe to return without copying.
    arr = torch.frombuffer(bytearray(data), dtype=torch.int16)
    return arr.view(torch.bfloat16).reshape(record["hs_shape"])


def load_logits_for_record(record: dict, trace_dir: str) -> torch.Tensor:
    """Load the pre-topk routing logits for a single JSONL record.

    Args:
        record: A parsed JSONL line that contains logit_offset, logit_bytes, logit_shape.
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
    # torch.frombuffer keeps the bytearray alive via the tensor's storage, so the
    # view is safe to return without copying.
    arr = torch.frombuffer(bytearray(data), dtype=torch.int16)
    return arr.view(torch.bfloat16).reshape(record["logit_shape"])


class RouterTracer:
    """Captures router top-K decisions across all MoE layers per step.

    - Inference mode: step boundaries are auto-detected.  When a layer that has already fired this
        step fires again, a new step has started.
    - Training mode: the training loop calls advance_step() at each iteration boundary.

    Recording is skipped during CUDA graph capture since D2H copies inside a captured graph would
    record stale values on replay.
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

    def register_hooks(self, model) -> None:
        """Walk model and register forward hooks on every TopKRouter module.
        Accepts a single model or a list of model chunks.
        """
        from megatron.core.transformer.moe.router import TopKRouter
        from megatron.core.utils import unwrap_model

        if not isinstance(model, (list, tuple)):
            model = [model]

        for chunk in model:
            unwrapped = unwrap_model(chunk)
            for module_name, module in unwrapped.named_modules():
                if isinstance(module, TopKRouter):
                    handle = module.register_forward_hook(self.make_hook(module_name))
                    self._hook_handles.append(handle)

    def remove_hooks(self) -> None:
        """Remove all forward hooks registered by register_hooks()."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def advance_step(self) -> None:
        """Advance to the next step (training mode).

        Call once per training iteration, after the forward-backward pass.
        Flushes accumulated records to disk and increments the step counter.
        Disables the tracer once max_steps is reached.
        """
        with self._lock:
            self._flush_records_to_disk()
            self.step_id += 1
            self.layers_seen_this_step.clear()
            if self.step_id >= self.max_steps:
                self._stopped = True
                self.remove_hooks()

    def make_hook(self, module_name: str = ""):
        """Build a forward hook callable for a single TopKRouter module.

        The module's qualified name is parsed once to recover its (block, mtp_idx, layer) identity
        so decoder and MTP layers that share a layer_number are kept distinct.
        """
        identity = _parse_router_module_name(module_name)

        def hook(module, inputs, outputs):
            if self._stopped:
                return
            if torch.cuda.is_current_stream_capturing():
                return
            self._record(module, inputs, outputs, identity)

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

    def _make_index_record(self, top_indices_cpu, step, block, mtp_idx, layer) -> dict:
        """Assemble a JSONL record dict for one layer's top-K indices."""
        record: dict = {
            "step": int(step),
            "stage": "pre_dispatch",
            "block": block,
            "layer": int(layer),
            "rank": self.rank,
            "num_tokens": int(top_indices_cpu.shape[0]),
            "topk": int(top_indices_cpu.shape[1]),
            "_top_indices_tensor": top_indices_cpu,
        }
        if mtp_idx is not None:
            record["mtp_idx"] = int(mtp_idx)
        return record

    def _record(self, module, inputs, outputs, identity=None) -> None:
        if not isinstance(outputs, tuple) or len(outputs) != 2:
            return
        _, second = outputs
        if not torch.is_tensor(second):
            return

        # Resolve a collision-free layer identity.  Prefer the name parsed at
        # registration; fall back to the module's bare layer_number.
        if identity is not None:
            block, mtp_idx, layer = identity
        else:
            layer_number = getattr(module, "layer_number", None)
            if layer_number is None:
                return
            block, mtp_idx, layer = "decoder", None, int(layer_number)
        layer_key = (block, mtp_idx, layer)

        with self._lock:
            if not self.training_mode:
                # Inference mode: detect step boundaries via layer repeats.
                if layer_key in self.layers_seen_this_step:
                    self._flush_records_to_disk()
                    self.step_id += 1
                    self.layers_seen_this_step.clear()
                    if self.step_id >= self.max_steps:
                        self._stopped = True
                        return
                self.layers_seen_this_step.add(layer_key)

            # The router-weight dump is keyed by the integer `layer` to match the
            # JSONL records' `layer` field, which is how downstream analysis joins
            # weights to traces. (`layers_seen_this_step` above uses the full
            # `layer_key` only for collision-free step-boundary detection.)
            if self.dump_router_weights and layer not in self._router_state:
                weight = getattr(module, "weight", None)
                expert_bias = getattr(module, "expert_bias", None)
                score_fn = getattr(
                    getattr(module, "config", None), "moe_router_score_function", None
                )
                topk_attr = getattr(module, "topk", None)
                if torch.is_tensor(weight):
                    self._router_state[layer] = {
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

            top_indices_cpu = top_indices.detach().to("cpu", torch.int32, non_blocking=True)
            num_tokens = int(top_indices_cpu.shape[0])

            record: dict = {
                **self._make_index_record(top_indices_cpu, self.step_id, block, mtp_idx, layer)
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

    def record_indices(
        self,
        indices,
        step: Optional[int] = None,
        layer_ids: Optional[List[int]] = None,
        block: str = "decoder",
        mtp_idx: Optional[int] = None,
    ) -> None:
        """Serialize already-captured top-K routing indices through the JSONL sink.

        This is the entry point for the in-pipeline recorder (RouterReplay/RoutingMetadata).
        Instead of capturing indices with a forward hook, the caller hands over the indices the
        router pipeline recorded.  This works under CUDA graphs, because the recorder copies into a
        static buffer rather than relying on a Python hook firing during replay.

        Only the top-K indices are serialized here. The hidden-state / logit /
        weight sidecars remain hook-only.

        Args:
            indices: Either a single tensor of shape [num_tokens, num_layers, topk]
                (the layout RoutingMetadata.get_routing_indices() returns), or a list/tuple of
                per-layer tensors each shaped [num_tokens, topk].
            step: Step id stamped on the emitted records.  Defaults to the
                tracer's current step_id (drive boundaries with advance_step()).
            layer_ids: Layer numbers, one per layer in `indices`.  Defaults to
                range(num_layers).
            block: Block tag for the records ("decoder" or "mtp").
            mtp_idx: MTP head index.
        """
        if self._stopped:
            return

        if torch.is_tensor(indices):
            if indices.dim() != 3:
                raise ValueError(
                    f"Expected a [num_tokens, num_layers, topk] tensor, got shape "
                    f"{tuple(indices.shape)}"
                )
            per_layer = [indices[:, i, :] for i in range(indices.shape[1])]
        else:
            per_layer = list(indices)

        if not per_layer:
            return
        if layer_ids is not None and len(layer_ids) != len(per_layer):
            raise ValueError(
                f"layer_ids has {len(layer_ids)} entries but indices has "
                f"{len(per_layer)} layers"
            )

        step = self.step_id if step is None else step
        with self._lock:
            for i, layer_indices in enumerate(per_layer):
                layer = i if layer_ids is None else layer_ids[i]
                top_indices_cpu = layer_indices.detach().to("cpu", torch.int32, non_blocking=True)
                self.records.append(
                    self._make_index_record(top_indices_cpu, step, block, mtp_idx, layer)
                )

    def _flush_records_to_disk(self) -> None:
        if not self.records:
            return
        with open(self.output_path, "a") as f:
            for rec in self.records:
                # .tolist() syncs the async D2H copy started in _record().
                tensor = rec.pop("_top_indices_tensor")
                rec["top_indices"] = tensor.tolist()
                f.write(json.dumps(rec) + "\n")
        self.records.clear()
        if self._hs_file is not None:
            self._hs_file.flush()
        if self._logits_file is not None:
            self._logits_file.flush()

    def flush(self) -> None:
        """Flush remaining records."""
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
