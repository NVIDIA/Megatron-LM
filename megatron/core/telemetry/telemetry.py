# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optional nemo-lens instrumentation for Megatron.

This is the only Megatron module that imports Lens instrumentation helpers or
registers Megatron span groups. Call sites import this shim unconditionally; it
degrades to no-op helpers when Lens is absent. Attribute setters exposed here
are best effort: an attribute failure must not interrupt Megatron execution.
"""

import logging
import math
import time
from contextlib import ExitStack, contextmanager, nullcontext
from typing import Any

logger = logging.getLogger(__name__)

_ACTIVE_TRAINER_HANDLE = None
_PYTHON_STARTUP_RECORDED = False

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
SPAN_CHECKPOINT_LOAD_READ_METADATA = "nv.mcore.checkpoint.load.read_metadata"
SPAN_CHECKPOINT_LOAD_MCORE_TO_PYT = "nv.mcore.checkpoint.load.mcore_to_pyt"
SPAN_CHECKPOINT_LOAD_FULLY_PARALLEL_PLAN = "nv.mcore.checkpoint.load.fully_parallel_plan"
SPAN_CHECKPOINT_LOAD_IO = "nv.mcore.checkpoint.load.io"
SPAN_CHECKPOINT_LOAD_EXCHANGE = "nv.mcore.checkpoint.load.exchange"
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
SPAN_MICROBATCH_FORWARD = "nv.mcore.microbatch.forward"
SPAN_MICROBATCH_BACKWARD = "nv.mcore.microbatch.backward"
SPAN_P2P_RECV_FORWARD = "nv.mcore.p2p.recv_forward"
SPAN_P2P_RECV_BACKWARD = "nv.mcore.p2p.recv_backward"
SPAN_P2P_SEND_FORWARD = "nv.mcore.p2p.send_forward"
SPAN_P2P_SEND_BACKWARD = "nv.mcore.p2p.send_backward"
SPAN_GRAD_SYNC_START = "nv.mcore.grad_sync.start"
SPAN_GRAD_SYNC_FINISH = "nv.mcore.grad_sync.finish"
SPAN_LAYER_FORWARD = "nv.mcore.layer.forward"
SPAN_LAYER_SELF_ATTENTION = "nv.mcore.layer.self_attention"
SPAN_LAYER_MLP = "nv.mcore.layer.mlp"
SPAN_LAYER_MAMBA = "nv.mcore.layer.mamba"

SPAN_CHECKPOINT_EXIT_FINALIZE = "nv.mlm.checkpoint.exit_finalize"
SPAN_TRAINING_ITER_BLOCK = "nv.dl.training.iter_block"
SPAN_TRAINING_STARTUP = "nv.dl.training.startup"
SPAN_TRAINING_STARTUP_PYTHON = "nv.dl.training.startup.python"
SPAN_TRAINING_STARTUP_IMPORTS = "nv.dl.training.startup.imports"
SPAN_TRAINING_STARTUP_ARG_PARSE = "nv.dl.training.startup.arg_parse"
SPAN_TRAINING_STARTUP_IN_JOB_SETUP = "nv.dl.training.startup.in_job_setup"
SPAN_TRAINING_STARTUP_INITIALIZE_MEGATRON = "nv.dl.training.startup.initialize_megatron"
SPAN_TRAINING_STARTUP_JIT_FUSION_OPTIONS = "nv.dl.training.startup.jit_fusion_options"
SPAN_TRAINING_STARTUP_MODEL_INIT = "nv.dl.training.startup.model_init"
SPAN_TRAINING_STARTUP_DATALOADER = "nv.dl.training.startup.dataloader"
SPAN_TRAINING_STARTUP_WEIGHT_HASH_CHECK = "nv.dl.training.startup.weight_hash_check"
SPAN_GPU_SNIFF_STARTUP = "nv.dl.resiliency.gpu_sniff.startup"

# Canonical span attribute names.
MODEL_CONFIG_MODEL_TYPE = "nv.mcore.model.config.model_type"

TRAINING_STEP = "nv.dl.training.step"
TRAINING_ITERATION_IS_FIRST = "nv.dl.training.iteration.is_first"
TRAINING_ITERATION_SKIPPED = "nv.dl.training.iteration.skipped"
TRAINING_OPTIMIZER_UPDATE_SUCCESSFUL = "nv.dl.training.optimizer.update_successful"
TRAINING_EVALUATE_ITERATION = "nv.dl.training.evaluate.iteration"
TRAINING_EVALUATE_ITERATION_COUNT = "nv.dl.training.evaluate.iteration_count"
GPU_SNIFF_TAG = "nv.dl.resiliency.gpu_sniff.tag"
MEMORY_RECLAIM_OPERATION = "nv.mcore.memory.reclaim.operation"
GRAD_SYNC_START_SITE = "nv.mcore.grad_sync.start.site"
LAYER_NUMBER = "nv.mcore.layer.number"

MEMORY_RECLAIM_GC_COLLECT = "gc_collect"
MEMORY_RECLAIM_FREE_OVERLAP_BUFFERS = "free_overlap_buffers"
GRAD_SYNC_SITE_INTERLEAVED_BACKWARD = "interleaved_backward"
GRAD_SYNC_SITE_INTERLEAVED_COOLDOWN = "interleaved_cooldown"
GRAD_SYNC_SITE_NON_INTERLEAVED_COOLDOWN = "non_interleaved_cooldown"

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
        self._lifecycle = _TrainerLifecycle(handle)

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
            self._lifecycle.close()
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


def run_checkpoint_exit_finalize(
    callback: Any, tracer: Any | None, *args: Any, **kwargs: Any
) -> Any:
    """Run the terminal queue drain in a fresh exit-finalize trace root."""
    root_state = None
    if tracer is not None:
        root_state = start_root_span(JOB, SPAN_CHECKPOINT_EXIT_FINALIZE, tracer)
    try:
        with managed_span(CKPT, SPAN_CHECKPOINT_SAVE_FINALIZE):
            return callback(*args, **kwargs)
    finally:
        if root_state is not None:
            end_root_span(*root_state)


class LoopPassSpanLifecycle:
    """Own one active loop-pass root and its same-rank link chain."""

    def __init__(self) -> None:
        self._span = None
        self._context_token = None
        self._previous_span_context = None

    def reset(self) -> None:
        """End any active pass and clear the prior-invocation link chain."""
        self.end()
        self._previous_span_context = None

    def start(
        self, tracer: Any, training_step: int, *, initial_link_context: Any | None = None
    ) -> None:
        """End the prior pass and open the next linked, fresh-root pass."""
        self.end()
        root_state = start_root_span(
            JOB,
            SPAN_TRAINING_ITER_BLOCK,
            tracer,
            link_context=self._previous_span_context or initial_link_context,
            training_step=training_step,
        )
        if root_state is not None:
            self._span, self._context_token = root_state

    def end(self) -> None:
        """End the active pass, preserving its context for the next link."""
        if self._span is not None:
            self._previous_span_context = end_root_span(self._span, self._context_token)
        self._span = None
        self._context_token = None


class _TrainerLifecycle:
    """Own startup and loop spans, but not provider or carrier teardown."""

    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self._startup_span = None
        self._startup_span_context = None
        self._startup_parent_context = None
        self._startup_start_time = None
        self._startup_open_time = None
        self._startup_context_token = None
        self._loop_passes = LoopPassSpanLifecycle()

    def start_startup(self, model_type: Any, program_start: Any, main_entry: Any) -> None:
        """Open the startup span and attach its context as the parent for later phases."""
        global _PYTHON_STARTUP_RECORDED
        if not is_enabled(JOB):
            return
        from opentelemetry import context

        state = start_startup_span(
            self._handle.tracer,
            model_type,
            program_start,
            main_entry,
            include_python_startup=not _PYTHON_STARTUP_RECORDED,
        )
        if state is None:
            return
        _PYTHON_STARTUP_RECORDED = True
        (
            self._startup_span,
            self._startup_parent_context,
            self._startup_start_time,
            self._startup_open_time,
        ) = state
        try:
            self._startup_span_context = self._startup_span.get_span_context()
        except Exception:
            self._startup_span_context = None
        self._startup_context_token = context.attach(self._startup_parent_context)

    def emit_startup_phase(self, name: str, start: Any, end: Any) -> None:
        """Emit one startup phase under the startup parent, or nothing once startup has closed."""
        if self._startup_span is None or self._startup_parent_context is None:
            return
        emit_startup_span(
            self._handle.tracer,
            name,
            start,
            end,
            context=self._startup_parent_context,
            root_start=self._startup_start_time,
            root_open_time=self._startup_open_time,
        )

    def end_startup(self) -> None:
        """Close the startup span and detach the context it attached."""
        if self._startup_span is not None:
            try:
                from opentelemetry import context

                if self._startup_context_token is not None:
                    context.detach(self._startup_context_token)
            except Exception:
                pass
            try:
                self._startup_span.end()
            except Exception:
                pass
            self._startup_span = None
            self._startup_parent_context = None
            self._startup_start_time = None
            self._startup_open_time = None

    def prepare_loop(self) -> None:
        """Reset loop-pass tracking before the training loop runs."""
        self._loop_passes.reset()

    def start_loop_pass(self, training_step: int) -> None:
        """Open a loop-pass span for this training step, linked to the startup span."""
        if not is_enabled(JOB):
            self._loop_passes.end()
            return
        self._loop_passes.start(
            self._handle.tracer, training_step, initial_link_context=self._startup_span_context
        )

    def end_loop_pass(self) -> None:
        """Close the open loop-pass span."""
        self._loop_passes.end()

    def close(self) -> None:
        """Close the loop pass and then the startup span."""
        self.end_loop_pass()
        self.end_startup()


def is_exporting() -> bool:
    """Return whether the current trainer handle exports telemetry."""
    return _ACTIVE_TRAINER_HANDLE is not None and bool(
        getattr(_ACTIVE_TRAINER_HANDLE, "is_exporting", False)
    )


def start_training_startup(model_type: Any, program_start: Any, main_entry: Any) -> None:
    """Open the trainer's fresh startup root."""
    if _ACTIVE_TRAINER_HANDLE is not None:
        _ACTIVE_TRAINER_HANDLE._lifecycle.start_startup(model_type, program_start, main_entry)


def emit_training_startup_phase(name: str, start: Any, end: Any) -> None:
    """Emit one elapsed phase under the live startup root."""
    if _ACTIVE_TRAINER_HANDLE is not None:
        _ACTIVE_TRAINER_HANDLE._lifecycle.emit_startup_phase(name, start, end)


def end_training_startup() -> None:
    """End startup while retaining its context for the first loop-pass link."""
    if _ACTIVE_TRAINER_HANDLE is not None:
        _ACTIVE_TRAINER_HANDLE._lifecycle.end_startup()


def prepare_training_loop() -> None:
    """Reset the loop-pass link chain for this training invocation."""
    if _ACTIVE_TRAINER_HANDLE is not None:
        _ACTIVE_TRAINER_HANDLE._lifecycle.prepare_loop()


def start_training_loop_pass(training_step: int) -> None:
    """Open the next fresh loop-pass root linked to its predecessor."""
    if _ACTIVE_TRAINER_HANDLE is not None:
        _ACTIVE_TRAINER_HANDLE._lifecycle.start_loop_pass(training_step)


def end_training_loop_pass() -> None:
    """End the active loop-pass root and retain its link context."""
    if _ACTIVE_TRAINER_HANDLE is not None:
        _ACTIVE_TRAINER_HANDLE._lifecycle.end_loop_pass()


def finalize_training_exit(callback: Any, *, terminate: bool) -> Any:
    """Close the last loop pass and run the application's terminal queue drain."""
    end_training_loop_pass()
    tracer = None
    if is_enabled(JOB) and _ACTIVE_TRAINER_HANDLE is not None:
        tracer = _ACTIVE_TRAINER_HANDLE.tracer
    return run_checkpoint_exit_finalize(callback, tracer, blocking=True, terminate=terminate)


def shutdown_training(handle: Any) -> None:
    """Close trainer telemetry, preserving disabled-handle shutdown delegation."""
    if handle is not None:
        handle.shutdown()


class _TrainerExitHooks:
    """Keep telemetry-only process hooks separate from application exit policy."""

    def __init__(self) -> None:
        self._installed = False
        self._previous_sigterm = None
        self._graceful_drain = False
        self._sigterm_fired = False

    def install(self, *, get_graceful_drain: Any) -> None:
        """Register the atexit and SIGTERM hooks once, and only while exporting."""
        if not is_exporting() or self._installed:
            return
        self._installed = True
        import atexit
        import signal

        atexit.register(self.shutdown)
        self._previous_sigterm = signal.getsignal(signal.SIGTERM)
        try:
            self._graceful_drain = bool(get_graceful_drain())
        except Exception:
            pass
        try:
            signal.signal(signal.SIGTERM, self.handle_sigterm)
        except (ValueError, OSError):
            pass

    def shutdown(self) -> None:
        """Shut down the active trainer handle."""
        shutdown_training(_ACTIVE_TRAINER_HANDLE)

    def force_flush(self) -> None:
        """Flush the tracer provider when the installed provider supports it."""
        try:
            from opentelemetry import trace

            provider = trace.get_tracer_provider()
            if hasattr(provider, "force_flush"):
                provider.force_flush()
        except Exception:
            pass

    def handle_sigterm(self, signum: int, frame: Any) -> None:
        """Flush on the first SIGTERM, then defer to the handler that was installed before."""
        import os
        import signal

        if not self._sigterm_fired:
            self._sigterm_fired = True
            try:
                if self._graceful_drain:
                    self.force_flush()
                else:
                    self.shutdown()
            except Exception:
                pass
        if callable(self._previous_sigterm):
            self._previous_sigterm(signum, frame)
        elif self._previous_sigterm == signal.SIG_DFL:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            os.kill(os.getpid(), signum)


_TRAINER_EXIT_HOOKS = _TrainerExitHooks()


def install_training_exit_hooks(*, get_graceful_drain: Any) -> None:
    """Install telemetry exit hooks once, only while the trainer exports."""
    _TRAINER_EXIT_HOOKS.install(get_graceful_drain=get_graceful_drain)


def start_root_span(
    group: str,
    name: str,
    tracer: Any,
    *,
    link_context: Any | None = None,
    training_step: int | None = None,
) -> tuple[Any, Any] | None:
    """Start and attach a group-gated root span with an optional trace link."""
    if not is_enabled(group):
        return None

    span = None
    try:
        from opentelemetry import context, trace
        from opentelemetry.context import Context
        from opentelemetry.trace import Link

        links = [Link(link_context)] if link_context is not None else None
        span = tracer.start_span(name, context=Context(), links=links)
        if training_step is not None:
            set_attributes(span, {TRAINING_STEP: training_step})
        token = context.attach(trace.set_span_in_context(span, Context()))
        return span, token
    except Exception:
        logger.debug("Could not start root span %r", name, exc_info=True)
        try:
            if span is not None:
                span.end()
        except Exception:
            pass
        return None


def end_root_span(span: Any, token: Any) -> Any | None:
    """Detach and end a root span, returning its context for a later link."""
    if span is None:
        return None

    span_context = None
    try:
        span_context = span.get_span_context()
    except Exception:
        logger.debug("Could not read root span context", exc_info=True)
    try:
        from opentelemetry import context

        if token is not None:
            context.detach(token)
    except Exception:
        logger.debug("Could not detach root span context", exc_info=True)
    try:
        span.end()
    except Exception:
        logger.debug("Could not end root span", exc_info=True)
    return span_context


def select_process_start_time(
    program_start: float | None, main_entry: float | None
) -> float | None:
    """Return a valid OS process-start estimate, never an entrypoint fallback."""
    if not _is_valid_epoch(main_entry):
        return None

    from nemo.lens.span_utilities import linux_process_create_time

    try:
        process_start = linux_process_create_time()
    except Exception:
        process_start = None

    latest_start = main_entry
    if _is_valid_epoch(program_start) and program_start <= main_entry:
        latest_start = program_start
    if _is_valid_epoch(process_start) and process_start <= latest_start:
        return float(process_start)
    return None


def start_startup_span(
    tracer: Any,
    model_type: Any,
    program_start: float | None,
    main_entry: float | None,
    *,
    include_python_startup: bool = True,
) -> tuple[Any, Any, float, float] | None:
    """Start the startup root and, once per process, its Python-start estimate.

    The Python child ends at the early entrypoint timestamp, before heavy
    imports. Process creation can precede exec, so this is not a measurement
    of CPython initialization alone. A fallback root never fabricates this child.
    """
    if not is_enabled(JOB):
        return None

    from opentelemetry import trace
    from opentelemetry.context import Context

    process_start = select_process_start_time(program_start, main_entry)
    root_open_time = time.time()
    root_start_time = process_start
    if root_start_time is None and (
        _is_valid_epoch(program_start)
        and _is_valid_epoch(main_entry)
        and program_start <= main_entry <= root_open_time
    ):
        root_start_time = float(program_start)
    start_kwargs = {}
    if root_start_time is not None and root_start_time <= root_open_time:
        start_kwargs["start_time"] = int(root_start_time * 1_000_000_000)
    else:
        root_start_time = root_open_time

    span = tracer.start_span(SPAN_TRAINING_STARTUP, context=Context(), **start_kwargs)
    set_attributes(span, {MODEL_CONFIG_MODEL_TYPE: str(model_type)})
    parent_context = trace.set_span_in_context(span, Context())
    if include_python_startup and (
        _is_valid_epoch(program_start)
        and _is_valid_epoch(main_entry)
        and program_start <= main_entry <= root_open_time
    ):
        emit_startup_span(
            tracer,
            SPAN_TRAINING_STARTUP_PYTHON,
            process_start,
            program_start,
            context=parent_context,
            root_start=root_start_time,
            root_open_time=root_open_time,
        )
    return span, parent_context, root_start_time, root_open_time


def emit_startup_span(
    tracer: Any,
    name: str,
    start: float | None,
    end: float | None,
    *,
    context: Any,
    root_start: float | None,
    root_open_time: float | None,
) -> Any | None:
    """Emit one valid, already-elapsed child of the training startup root."""
    if not is_enabled(JOB):
        return None
    if not all(_is_valid_epoch(value) for value in (start, end, root_start, root_open_time)):
        return None
    if start < root_start or end < start or end > root_open_time:
        return None

    from nemo.lens.span_utilities import emit_span

    try:
        return emit_span(tracer, name, start, end, context=context)
    except Exception:
        logger.debug("Could not emit explicit-timestamp startup span %r", name, exc_info=True)
        return None


def _is_valid_epoch(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


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
