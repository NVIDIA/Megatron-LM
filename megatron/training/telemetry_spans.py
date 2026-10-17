# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""OpenTelemetry span lifecycle management for Megatron training."""

import atexit
import os
import signal
from collections.abc import Mapping
from typing import Any

from .global_vars import get_args, get_telemetry

# OTel helpers are imported once at startup. The fallbacks keep this module
# importable when nemo-lens is unavailable.
try:
    from nemo.lens.helpers import managed_span as _otel_managed_span
    from nemo.lens.helpers import safe_set_span_attributes as _otel_safe_set_attrs
    from nemo.lens.helpers import trace_fn as _otel_trace_fn
    from nemo.lens.state import is_span_group_enabled as _otel_sg_enabled
except ImportError:
    from megatron.core.telemetry.fallbacks import is_span_group_enabled as _otel_sg_enabled
    from megatron.core.telemetry.fallbacks import managed_span as _otel_managed_span
    from megatron.core.telemetry.fallbacks import safe_set_span_attributes as _otel_safe_set_attrs
    from megatron.core.telemetry.fallbacks import trace_fn as _otel_trace_fn


# An incarnation emits segmented traces rather than one run-long umbrella:
#
#   trace A: pre_startup
#   trace B: megatron.startup
#   trace C_n: megatron.train, one per checkpoint interval
#
# The interval traces link back to the preceding interval, with the first linked
# to the startup trace. Resource attributes provide run identity across traces.
_otel_startup_span = None
_otel_startup_span_ctx = None
_otel_ctx_module = None
_otel_startup_ctx_token = None
_otel_shutdown_done = False

_otel_interval_span = None
_otel_interval_ctx_token = None
_otel_trace_interval_step = 0

# Process-global hooks must be installed at most once even if pretrain() is
# invoked repeatedly in the same process.
_otel_exit_hooks_installed = False


def _otel_telemetry_active() -> bool:
    """Return whether telemetry is initialized and exporting."""
    handle = get_telemetry()
    return handle is not None and bool(getattr(handle, 'is_exporting', False))


def _start_otel_job_spans(
    model_type: Any, program_start: float | None, startup_timestamps: Mapping[str, float | None]
) -> None:
    """Emit ``pre_startup`` and open the incarnation's startup trace.

    The startup root uses an empty context so it starts a new trace. Phases that
    elapsed before telemetry initialization are emitted with explicit timestamps.

    Args:
        model_type: Model type recorded on the startup root.
        program_start: Timestamp captured before application imports.
        startup_timestamps: Launch, scheduler, and startup phase timestamps.
    """
    global _otel_ctx_module, _otel_startup_ctx_token
    global _otel_startup_span, _otel_startup_span_ctx

    if not _otel_sg_enabled('job'):
        return

    from opentelemetry import context as _otel_ctx
    from opentelemetry import trace as _otel_trace
    from opentelemetry.context import Context as _OtelContext

    _otel_ctx_module = _otel_ctx
    _otel_tracer = get_telemetry().tracer

    launch_script_start = startup_timestamps.get('launch_script_start')
    launch_script_presrun = startup_timestamps.get('launch_script_presrun')
    slurm_job_start_time = startup_timestamps.get('slurm_job_start_time')
    program_start_ns = int(program_start * 1e9) if program_start is not None else None

    if (
        launch_script_start is not None
        and slurm_job_start_time is not None
        and slurm_job_start_time < launch_script_start
    ):
        _backdated_otel_span('pre_startup', slurm_job_start_time, launch_script_start)

    fold_launch_phases = (
        launch_script_start is not None
        and launch_script_presrun is not None
        and program_start is not None
        and launch_script_start <= launch_script_presrun <= program_start
    )
    startup_start_ns = int(launch_script_start * 1e9) if fold_launch_phases else program_start_ns

    _otel_startup_span = _otel_tracer.start_span(
        "megatron.startup", context=_OtelContext(), start_time=startup_start_ns
    )
    _otel_mark_goodput(_otel_startup_span)
    _otel_safe_set_attrs(_otel_startup_span, {'megatron.model_type': str(model_type)})
    try:
        _otel_startup_span_ctx = _otel_startup_span.get_span_context()
    except Exception:  # noqa: BLE001 -- telemetry must never break training
        _otel_startup_span_ctx = None
    _otel_startup_ctx_token = _otel_ctx.attach(_otel_trace.set_span_in_context(_otel_startup_span))

    if fold_launch_phases:
        _backdated_otel_span(
            'megatron.startup.launch_script', launch_script_start, launch_script_presrun
        )
        _backdated_otel_span(
            'megatron.startup.container_load', launch_script_presrun, program_start
        )


def _otel_mark_goodput(span: Any) -> None:
    """Mark a manually created span as a goodput boundary."""
    try:
        span.set_attribute('is_goodput_span', True)
    except Exception:  # noqa: BLE001 -- telemetry must never break training
        pass


def _backdated_otel_span(name: str, start: float | None, end: float | None) -> None:
    """Record an elapsed phase as a closed span with explicit timestamps."""
    if not _otel_sg_enabled('job') or start is None or end is None:
        return
    tracer = get_telemetry().tracer
    span = tracer.start_span(name, start_time=int(start * 1e9))
    _otel_mark_goodput(span)
    span.end(end_time=int(end * 1e9))


def _end_otel_startup_span() -> None:
    """End and detach the startup root, idempotently."""
    global _otel_startup_span

    if _otel_startup_span is not None:
        try:
            if _otel_ctx_module is not None and _otel_startup_ctx_token is not None:
                _otel_ctx_module.detach(_otel_startup_ctx_token)
        except Exception:  # noqa: BLE001 -- telemetry must never break training
            pass
        try:
            _otel_startup_span.end()
        except Exception:  # noqa: BLE001
            pass
        _otel_startup_span = None


def _start_otel_train_span() -> None:
    """Reset interval tracing before entering the steady-state loop."""
    global _otel_trace_interval_step

    if not _otel_sg_enabled('job'):
        return
    _otel_trace_interval_step = 0


def _reroot_otel_interval() -> None:
    """Close the current interval and open the next in a fresh linked trace."""
    global _otel_interval_ctx_token, _otel_interval_span

    if get_telemetry() is None or not _otel_sg_enabled('job'):
        return

    from opentelemetry import context as otel_context
    from opentelemetry import trace as otel_trace
    from opentelemetry.context import Context
    from opentelemetry.trace import Link

    previous = _otel_interval_span
    links = []
    try:
        if previous is not None:
            links.append(Link(previous.get_span_context()))
        elif _otel_startup_span_ctx is not None:
            links.append(Link(_otel_startup_span_ctx))
    except Exception:  # noqa: BLE001
        links = []

    if previous is not None:
        try:
            if _otel_interval_ctx_token is not None:
                otel_context.detach(_otel_interval_ctx_token)
        except Exception:  # noqa: BLE001
            pass
        try:
            previous.end()
        except Exception:  # noqa: BLE001
            pass
        _otel_interval_span = None
        _otel_interval_ctx_token = None

    try:
        span = get_telemetry().tracer.start_span('megatron.train', context=Context(), links=links)
        _otel_mark_goodput(span)
        _otel_interval_span = span
        _otel_interval_ctx_token = otel_context.attach(otel_trace.set_span_in_context(span))
    except Exception:  # noqa: BLE001
        _otel_interval_span = None
        _otel_interval_ctx_token = None


def _maybe_reroot_otel_interval() -> None:
    """Re-root tracing at checkpoint-frequency boundaries."""
    global _otel_trace_interval_step

    frequency = getattr(get_args(), 'save_interval', None) or 0
    if frequency > 0 and (_otel_trace_interval_step % frequency == 0):
        _reroot_otel_interval()
    _otel_trace_interval_step += 1


def _end_otel_interval_span() -> None:
    """Close the final interval root, idempotently."""
    global _otel_interval_ctx_token, _otel_interval_span

    if _otel_interval_span is not None:
        try:
            from opentelemetry import context as otel_context

            if _otel_interval_ctx_token is not None:
                otel_context.detach(_otel_interval_ctx_token)
        except Exception:  # noqa: BLE001
            pass
        try:
            _otel_interval_span.end()
        except Exception:  # noqa: BLE001
            pass
        _otel_interval_span = None
        _otel_interval_ctx_token = None


def _end_otel_train_span() -> None:
    """Close the final training interval span."""
    _end_otel_interval_span()


def _end_otel_job_spans() -> None:
    """Close open roots and shut down telemetry, idempotently."""
    global _otel_shutdown_done

    _end_otel_train_span()
    _end_otel_startup_span()

    if not _otel_shutdown_done:
        handle = get_telemetry()
        if handle is not None:
            handle.shutdown()
        _otel_shutdown_done = True


def _force_flush_otel() -> None:
    """Best-effort bounded flush used during graceful SIGTERM draining."""
    try:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        if hasattr(provider, 'force_flush'):
            provider.force_flush()
    except Exception:  # noqa: BLE001 -- telemetry must never break training
        pass


def _install_otel_exit_hooks() -> None:
    """Install the telemetry atexit and SIGTERM hooks exactly once."""
    global _otel_exit_hooks_installed

    if not _otel_telemetry_active() or _otel_exit_hooks_installed:
        return

    _otel_exit_hooks_installed = True
    atexit.register(_end_otel_job_spans)

    previous_sigterm = signal.getsignal(signal.SIGTERM)
    graceful_drain = False
    try:
        graceful_drain = bool(getattr(get_args(), 'exit_signal_handler', False))
    except Exception:  # noqa: BLE001 -- telemetry must never break training
        pass
    sigterm_fired = [False]

    def sigterm_handler(signum, frame):
        if not sigterm_fired[0]:
            sigterm_fired[0] = True
            try:
                if graceful_drain:
                    _force_flush_otel()
                else:
                    _end_otel_job_spans()
            except Exception:  # noqa: BLE001 -- telemetry must never break training
                pass

        if callable(previous_sigterm):
            previous_sigterm(signum, frame)
        elif previous_sigterm == signal.SIG_DFL:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        # SIG_IGN means the previous handler intentionally ignored SIGTERM.

    try:
        signal.signal(signal.SIGTERM, sigterm_handler)
    except (ValueError, OSError):
        # Not the main thread or platform unsupported; atexit remains the fallback.
        pass
