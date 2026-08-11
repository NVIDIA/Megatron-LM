# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Zero-nsys host-phase timing for the dynamic inference decode loop.

The decode step's host chain between CUDA-graph replays is bracketed by NVTX
ranges (``initialize_attention_state``, ``forward_pass``, ``sampling``,
``transfer_samples_to_cpu``, ``active_request_mask``, ``update_requests`` in the
controller; ``bookkeeping`` / ``detokenization`` in the engine). Those are raw
``torch.cuda.nvtx`` / gated ``nvtx_range_*`` calls that are inert without a
profiler attached — and the profiler that would name them (nsys with NVTX under
CUDA graphs) deadlocks finalization on this workload.

This shim monkeypatches those range push/pop names with ``perf_counter``
self-timers (a shared nesting stack, so each label reports self-time excluding
children) and periodically prints a per-step breakdown to stderr. It needs no
nsys and no CUDA-graph interaction, so it safely names the between-step Python
phases that the clean-trace idle attribution localized. Env-gated
(``MCORE_INFER_HOST_TIMING``); default off (``install`` is a no-op).
"""

# Reports go to stderr on purpose: this runs inside a multi-rank inference server
# whose logger configuration is the caller's, and a diagnostic must not depend on it.
# pylint: disable=bad-builtin

import os
import sys
import threading
import time

USE_HOST_TIMING: bool = os.environ.get("MCORE_INFER_HOST_TIMING", "0") == "1"
# Print the running breakdown every N times the step label is popped.
_REPORT_EVERY: int = int(os.environ.get("MCORE_INFER_HOST_TIMING_EVERY", "32"))
# The once-per-step range whose pop delimits a decode step for the per-step avg.
_STEP_LABEL: str = os.environ.get("MCORE_INFER_HOST_TIMING_STEP_LABEL", "forward_pass")

_lock = threading.Lock()
_totals: dict = {}  # label -> [self_ns, count]
_stack: list = []  # [label, start_ns, children_ns]
_step_count: int = 0
_installed: bool = False


def _push(msg=None, *args, **kwargs):
    _stack.append([msg if msg is not None else "<anon>", time.perf_counter_ns(), 0])


def _pop(*args, **kwargs):
    global _step_count
    if not _stack:
        return
    label, start, children = _stack.pop()
    dur = time.perf_counter_ns() - start
    self_ns = dur - children
    with _lock:
        t = _totals.setdefault(label, [0, 0])
        t[0] += self_ns
        t[1] += 1
    if _stack:
        _stack[-1][2] += dur
    if label == _STEP_LABEL:
        _step_count += 1
        if _REPORT_EVERY > 0 and _step_count % _REPORT_EVERY == 0:
            report()


def report():
    """Print the accumulated per-phase self-time breakdown to stderr."""
    with _lock:
        items = sorted(_totals.items(), key=lambda kv: -kv[1][0])
        n = _step_count or 1
        lines = [f"[HOST_TIMING] after {_step_count} steps (self-time):"]
        for label, (ns, c) in items:
            lines.append(
                f"  {ns / 1e3 / n:9.1f} us/step   tot {ns / 1e6:8.2f} ms   n={c:7d}   {label}"
            )
    print("\n".join(lines), file=sys.stderr, flush=True)


# Methods to time individually, as (module, class, [methods]). The coarse ranges
# localize the cost to ``forward_pass``, but that range is almost entirely one call
# into the model, so splitting it needs per-method timers rather than more ranges.
# Missing attributes are skipped, so this list can name methods that only exist on
# some paths.
_METHOD_TARGETS = (
    (
        "megatron.core.inference.text_generation_controllers.text_generation_controller",
        "TextGenerationController",
        (
            "_dynamic_step_forward_logits",
            "_router_record_bookkeeping",
            "_dynamic_step_log_probs_bookkeeping",
            "_dynamic_step_context_init",
        ),
    ),
    (
        "megatron.core.inference.model_inference_wrappers.abstract_model_inference_wrapper",
        "AbstractModelInferenceWrapper",
        ("run_one_forward_step", "forward_pass_without_pipeline_parallel", "_forward"),
    ),
)


def _wrap_methods():
    """Wrap selected methods so their self-time is reported like a range."""
    import importlib

    for mod_name, cls_name, methods in _METHOD_TARGETS:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        cls = getattr(mod, cls_name, None)
        if cls is None:
            continue
        for meth in methods:
            fn = getattr(cls, meth, None)
            if fn is None or getattr(fn, "_host_timed", False):
                continue

            def make(fn=fn, label=f"{cls_name}.{meth}"):
                def timed(*args, **kwargs):
                    _push(label)
                    try:
                        return fn(*args, **kwargs)
                    finally:
                        _pop()

                timed._host_timed = True
                return timed

            setattr(cls, meth, make())


def install():
    """Patch the controller / engine range push+pop names with the self-timers."""
    global _installed
    if not USE_HOST_TIMING or _installed:
        return
    import importlib

    targets = [
        (
            "megatron.core.inference.text_generation_controllers.text_generation_controller",
            "range_push",
            "range_pop",
        ),
        ("megatron.core.inference.engines.dynamic_engine", "nvtx_range_push", "nvtx_range_pop"),
    ]
    for mod_name, push_attr, pop_attr in targets:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        if hasattr(mod, push_attr):
            setattr(mod, push_attr, _push)
        if hasattr(mod, pop_attr):
            setattr(mod, pop_attr, _pop)
    if os.environ.get("MCORE_INFER_HOST_TIMING_METHODS", "1") == "1":
        _wrap_methods()
    _installed = True
    print(
        f"[HOST_TIMING] installed (report every {_REPORT_EVERY} '{_STEP_LABEL}' pops)",
        file=sys.stderr,
        flush=True,
    )
