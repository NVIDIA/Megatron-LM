# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optional nemo-lens instrumentation for Megatron.

This is the only Megatron module that imports Lens instrumentation helpers or
registers Megatron span groups. Call sites import this shim unconditionally; it
degrades to no-op helpers when Lens is absent. Attribute setters exposed here
are best effort: an attribute failure must not interrupt Megatron execution.
"""

import logging
from contextlib import ExitStack, contextmanager, nullcontext
from typing import Any

logger = logging.getLogger(__name__)

_ACTIVE_TRAINER_HANDLE = None

NAMESPACE = "megatron"

JOB = "megatron.job"
TRAIN = "megatron.train"
CKPT = "megatron.ckpt"
EVAL = "megatron.eval"
INFERENCE = "megatron.inference"
DETAIL = "megatron.detail"

GROUPS = frozenset([JOB, TRAIN, CKPT, EVAL, INFERENCE, DETAIL])
PRESETS = {
    "default": frozenset([JOB, CKPT, EVAL, INFERENCE]),
    "per_step": frozenset([JOB, TRAIN, CKPT, EVAL, INFERENCE]),
    "profiling": GROUPS,
}

# Megatron-owned canonical span names.
SPAN_CHECKPOINT_SAVE_STATE_DICT = "nv.dl.training.checkpoint.save.state_dict"
SPAN_CHECKPOINT_SAVE_IO_WRITE = "nv.dl.training.checkpoint.save.io_write"
SPAN_CHECKPOINT_LOAD = "nv.dl.training.checkpoint.load"
SPAN_CHECKPOINT_LOAD_IO_READ = "nv.dl.training.checkpoint.load.io_read"
SPAN_CHECKPOINT_REPORT_MEMORY = "nv.mlm.checkpoint.report_memory"
SPAN_CHECKPOINT_TIMERS_LOG = "nv.mlm.checkpoint.timers_log"
SPAN_CHECKPOINT_FT_HEARTBEAT = "nv.mlm.checkpoint.ft_heartbeat"
SPAN_CHECKPOINT_EXPOSED_SAVE = "nv.dl.training.checkpoint.exposed_save"
SPAN_CHECKPOINT_SAVE = "nv.dl.training.checkpoint.save"
SPAN_CHECKPOINT_SAVE_FINALIZE = "nv.dl.training.checkpoint.save.finalize"
SPAN_TRAINING_ITERATION = "nv.dl.training.iteration"
SPAN_TRAINING_FORWARD_BACKWARD = "nv.dl.training.iteration.forward_backward"
SPAN_TRAINING_OPTIMIZER_STEP = "nv.dl.training.iteration.optimizer_step"
SPAN_GPU_SNIFF_PERIODIC = "nv.dl.resiliency.gpu_sniff.periodic"
SPAN_CUDA_GRAPH_CAPTURE = "nv.mcore.cuda_graph.capture"
SPAN_MEMORY_RECLAIM = "nv.mcore.memory.reclaim"
SPAN_TRAINING_ITERATION_REPORT = "nv.mlm.train.iteration_report"
SPAN_TRAINING_PARAMS_NORM = "nv.mlm.train.params_norm"
SPAN_TRAINING_LOG = "nv.mlm.train.log"
SPAN_TRAINING_FORWARD_PRE_HOOK = "nv.mlm.train.forward_pre_hook"
SPAN_TRAINING_EVALUATE = "nv.dl.training.evaluate"
SPAN_TRAINING_EVALUATE_STEP = "nv.dl.training.evaluate.step"

# Canonical span attribute names.
TRAINING_STEP = "nv.dl.training.step"
TRAINING_ITERATION_IS_FIRST = "nv.dl.training.iteration.is_first"
TRAINING_ITERATION_SKIPPED = "nv.dl.training.iteration.skipped"
TRAINING_OPTIMIZER_UPDATE_SUCCESSFUL = "nv.dl.training.optimizer.update_successful"
TRAINING_EVALUATE_ITERATION = "nv.dl.training.evaluate.iteration"
TRAINING_EVALUATE_ITERATION_COUNT = "nv.dl.training.evaluate.iteration_count"
GPU_SNIFF_TAG = "nv.dl.resiliency.gpu_sniff.tag"
MEMORY_RECLAIM_OPERATION = "nv.mcore.memory.reclaim.operation"

MEMORY_RECLAIM_GC_COLLECT = "gc_collect"
MEMORY_RECLAIM_FREE_OVERLAP_BUFFERS = "free_overlap_buffers"

try:
    from nemo.lens import SpanRegistry as _SpanRegistry
    from nemo.lens import is_span_group_enabled as _is_span_group_enabled
    from nemo.lens import managed_span as _managed_span
    from nemo.lens import safe_set_span_attributes as _safe_set_span_attributes
    from nemo.lens import span_cm as _span_cm
    from nemo.lens import trace_fn as _trace_fn
except ModuleNotFoundError as exc:
    if exc.name not in {"nemo", "nemo.lens"}:
        raise
    _AVAILABLE = False
else:
    _AVAILABLE = True
    _SpanRegistry.register(NAMESPACE, GROUPS, PRESETS)


if not _AVAILABLE:

    @contextmanager
    def _managed_span(group: str, name: str, tracer=None, **attributes: Any):
        yield None

    @contextmanager
    def _span_cm(name: str, tracer=None, record_exception: bool = True, **attributes: Any):
        yield None

    def _trace_fn(group: str, name: str, tracer=None):
        def decorator(func):
            return func

        return decorator


class _ManagedTrainerTelemetry:
    """Keep the trainer Resource carrier installed for one Lens handle lifetime."""

    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self._publication = ExitStack()
        self._closed = False

    def _adopt_publication(self, publication: ExitStack) -> None:
        self._publication = publication.pop_all()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)

    def shutdown(self) -> None:
        """Shut down Lens, then restore the exact incoming carrier once."""
        global _ACTIVE_TRAINER_HANDLE
        if self._closed:
            return
        self._closed = True
        try:
            self._handle.shutdown()
        finally:
            try:
                self._publication.close()
            finally:
                if _ACTIVE_TRAINER_HANDLE is self:
                    _ACTIVE_TRAINER_HANDLE = None


def setup(args: Any, *, include_training: bool = True) -> Any | None:
    """Configure Lens for the trainer, or return ``None`` when Lens is absent."""
    global _ACTIVE_TRAINER_HANDLE
    if not _AVAILABLE:
        return None

    from nemo.lens import NemoLensConfig, setup_telemetry

    config = NemoLensConfig.from_env(prefix="MEGATRON_OTEL", fallback_prefix="NEMO_LENS")
    if getattr(args, "otel_enabled", False):
        config.enabled = True
    if getattr(args, "otel_service_name", None):
        config.service_name = args.otel_service_name

    resource_attributes = {}
    publication = ExitStack()
    if config.enabled:
        if _ACTIVE_TRAINER_HANDLE is not None:
            raise RuntimeError("trainer telemetry is already active")

        from nemo.lens.resources import publish_otel_resource_attributes

        from megatron.core.telemetry.resource_attrs import compose_telemetry_resource_attrs

        resource_attributes = compose_telemetry_resource_attrs(
            args, include_training=include_training
        )
        config.service_name = resource_attributes["service.name"]

    handle = None
    managed_handle = None
    try:
        if config.enabled:
            publication.enter_context(publish_otel_resource_attributes(resource_attributes))
        handle = setup_telemetry(config, resource_attributes=resource_attributes)
        result = handle
        if config.enabled:
            managed_handle = _ManagedTrainerTelemetry(handle)
            managed_handle._adopt_publication(publication)
            _ACTIVE_TRAINER_HANDLE = managed_handle
            result = managed_handle

        if config.enabled and config.logs_enabled and handle.is_exporting:
            from nemo.lens.logging_bridge import setup_logging_bridge

            setup_logging_bridge()
        return result
    except Exception:
        if managed_handle is not None:
            try:
                managed_handle.shutdown()
            except Exception:
                pass
        else:
            if handle is not None:
                try:
                    handle.shutdown()
                except Exception:
                    pass
            try:
                publication.close()
            except Exception:
                pass
        raise


def is_enabled(group: str) -> bool:
    """Return whether a Megatron span group is currently enabled."""
    if not _AVAILABLE:
        return False
    try:
        return bool(_is_span_group_enabled(group))
    except Exception:
        logger.debug("Could not resolve span group %r", group, exc_info=True)
        return False


def managed_span(group: str, name: str, tracer=None, **attributes: Any):
    """Create a Lens managed span, or a no-op when Lens is unavailable."""
    return _managed_span(group, name, tracer=tracer, **attributes)


@contextmanager
def checkpoint_exposed_save_span(step: int) -> Any:
    """Create the complete trainer-visible checkpoint save span."""
    if not is_enabled(CKPT):
        yield None
        return
    with managed_span(CKPT, SPAN_CHECKPOINT_EXPOSED_SAVE, **{TRAINING_STEP: step}) as span:
        yield span


@contextmanager
def checkpoint_save_span(step: int) -> Any:
    """Create the inner trainer save span without off-group attribute work."""
    if not is_enabled(CKPT):
        yield None
        return
    with managed_span(CKPT, SPAN_CHECKPOINT_SAVE, **{TRAINING_STEP: step}) as span:
        yield span


class TrainingStepSpans:
    """Capture one training step's tracer without owning application execution."""

    def __init__(self, handle: Any) -> None:
        self._tracer = handle.tracer if is_enabled(TRAIN) else None

    def forward_backward(self, get_num_microbatches: Any) -> Any:
        """Create the forward/backward context, reading its attribute only if enabled."""
        if is_enabled(TRAIN) and self._tracer is not None:
            return span_cm(
                SPAN_TRAINING_FORWARD_BACKWARD,
                tracer=self._tracer,
                num_microbatches=get_num_microbatches(),
            )
        return nullcontext()

    def optimizer(self) -> Any:
        """Create the optimizer context with the captured step tracer."""
        if is_enabled(TRAIN) and self._tracer is not None:
            return span_cm(SPAN_TRAINING_OPTIMIZER_STEP, tracer=self._tracer)
        return nullcontext()


@contextmanager
def training_report_span(handle: Any) -> Any:
    """Measure reporting without changing its exception events or status."""
    span = None
    token = None
    if is_enabled(TRAIN):
        from opentelemetry import context, trace

        span = handle.tracer.start_span(SPAN_TRAINING_ITERATION_REPORT)
        token = context.attach(trace.set_span_in_context(span))
    try:
        yield span
    finally:
        if span is not None:
            from opentelemetry import context

            context.detach(token)
            span.end()


def training_iteration_span(step: int, is_first: bool):
    """Create one training-iteration span with its required identity attributes."""
    return managed_span(
        TRAIN,
        SPAN_TRAINING_ITERATION,
        **{TRAINING_STEP: step, TRAINING_ITERATION_IS_FIRST: is_first},
    )


def memory_reclaim_span(group: str, step: int, operation: str):
    """Create a memory-reclaim span with its required discriminators."""
    return managed_span(
        group, SPAN_MEMORY_RECLAIM, **{TRAINING_STEP: step, MEMORY_RECLAIM_OPERATION: operation}
    )


def gpu_sniff_span(span_name: str, tag: str, training_step: int | None = None):
    """Create a GPU-sniff span, attaching a step only for periodic execution."""
    attributes: dict[str, Any] = {GPU_SNIFF_TAG: tag}
    if training_step is not None:
        attributes[TRAINING_STEP] = training_step
    return managed_span(JOB, span_name, **attributes)


def set_current_evaluation_span_attributes(
    training_step: int, iteration_count: int | None = None
) -> None:
    """Attach the training step and evaluation length to the current span."""
    if not is_enabled(EVAL):
        return

    attributes: dict[str, Any] = {TRAINING_STEP: training_step}
    if iteration_count is not None:
        attributes[TRAINING_EVALUATE_ITERATION_COUNT] = iteration_count
    set_current_span_attributes(attributes)


def trace_fn(group: str, name: str, tracer=None):
    """Decorate a function with a group-gated span."""
    return _trace_fn(group, name, tracer=tracer)


def span_cm(name: str, tracer=None, record_exception: bool = True, **attributes: Any):
    """Create an ungated span context manager, or a no-op without Lens."""
    return _span_cm(name, tracer=tracer, record_exception=record_exception, **attributes)


def set_attributes(span, attributes: dict, redact_keys=None) -> None:
    """Best-effort assignment of attributes to an OpenTelemetry span.

    Unlike calling ``span.set_attribute()`` or ``span.set_attributes()``
    directly, this delegates to Lens to ignore non-recording spans, skip
    ``None`` and unsupported complex values, accept scalar values and scalar
    sequences, and redact configured sensitive strings. When Lens is absent or
    either Lens or OpenTelemetry raises, the assignment is skipped so telemetry
    cannot interrupt Megatron execution.

    Args:
        span: OpenTelemetry span to update.
        attributes: Attribute names and values to assign.
        redact_keys: Optional sensitive string keys to redact instead of Lens's
            defaults.
    """
    if not _AVAILABLE:
        return
    try:
        if redact_keys is None:
            _safe_set_span_attributes(span, attributes)
        else:
            _safe_set_span_attributes(span, attributes, redact_keys=redact_keys)
    except Exception:
        logger.debug("Could not set span attributes", exc_info=True)


def set_current_span_attributes(attributes: dict, redact_keys=None) -> None:
    """Best-effort assignment of attributes to the current OpenTelemetry span.

    The OpenTelemetry import and current-span lookup remain lazy so this shim is
    importable when Lens and OpenTelemetry are absent.

    Args:
        attributes: Attribute names and values to assign.
        redact_keys: Optional sensitive string keys to redact instead of Lens's
            defaults.
    """
    if not _AVAILABLE:
        return
    try:
        from opentelemetry import trace

        set_attributes(trace.get_current_span(), attributes, redact_keys=redact_keys)
    except Exception:
        logger.debug("Could not set current span attributes", exc_info=True)
