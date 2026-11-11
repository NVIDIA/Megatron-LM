# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Prometheus metrics endpoint.

GET /metrics asks every data parallel engine for its counters through the InferenceClient ->
coordinator -> engines path (GET_METRICS); each engine answers at its next step. Counters are
lifetime totals, so throughput over a window is the difference of two scrapes divided by the
window length. Each metric is reported per engine (label ``engine``) and summed over engines.

In disaggregated serving the engine identities are ``prefill_s<i>_dp<j>`` and
``decode_s<i>_dp<j>``. Prompt tokens are counted by prefill engines, generated tokens by the engine
that sampled them (the first by prefill, the rest by decode), and finished requests by decode;
prefill engines report their completed hand-offs as ``prefill_handoffs_total``.
"""

import logging

from megatron.core.inference.engines.step_time_metrics import STEP_TIME_BUCKETS_MS

logger = logging.getLogger(__name__)

# name -> (Prometheus type, help).
_METRICS = {
    "generation_tokens_total": ("counter", "Tokens generated."),
    "prompt_tokens_total": ("counter", "Prompt tokens prefilled (computed)."),
    "prompt_tokens_cached_total": ("counter", "Prompt tokens skipped by prefix-cache hits."),
    "requests_finished_total": ("counter", "Requests finished."),
    "prefill_handoffs_total": (
        "counter",
        "Disaggregated prefill requests finished and handed off to decode.",
    ),
    "engine_steps_total": ("counter", "Engine steps."),
    "requests_running": ("gauge", "Requests in the active batch."),
    "requests_paused": ("gauge", "Requests paused (out of KV blocks)."),
    "requests_waiting": ("gauge", "Requests queued for admission."),
    "kv_tokens": ("gauge", "Context tokens (prompt + generated) held by active requests."),
    "min_context_tokens": ("gauge", "Context tokens of the shortest active request."),
    "p50_context_tokens": ("gauge", "Median context tokens over active requests."),
    "p90_context_tokens": ("gauge", "90th-percentile context tokens over active requests."),
    "p99_context_tokens": ("gauge", "99th-percentile context tokens over active requests."),
    "max_context_tokens": ("gauge", "Context tokens of the longest active request."),
}


# name -> help, for histograms reported as {class: {"buckets", "sum", "count"}} (see
# step_time_metrics.StepTimeHistogram.snapshot), labeled by the class as ``step``.
_HISTOGRAMS = {
    "step_time_ms": "Time of the steps the engine times (logging steps), by step class: "
    "prefill, decode_clean, decode_interfered (another EP rank prefills), decode.",
}


# name -> help, for per-step-class counters reported as {class: {field: value}} (see
# step_time_metrics.StepMixCounter.snapshot), one counter per field, labeled by the class as
# ``step``.
_STEP_MIX_FIELDS = {
    "steps": "Engine steps (every step), by step class: prefill_only, mixed (prefill and decode "
    "requests in this rank's batch), decode_clean, decode_interfered (another EP rank "
    "prefills), decode (no EP consensus).",
    "tokens": "Input tokens of the engine's steps, by step class.",
    "wall_ms": "Host wall time of the engine's steps (gap between consecutive steps, ms), by "
    "step class.",
}


# name -> help, for per-request gauges reported as {request id: value} (one line per request,
# labeled engine and request; no total).
_PER_REQUEST = {
    "request_context_tokens": "Context tokens (prompt + generated) of each active request.",
}


def _render_histogram(metric, labels, hist):
    """Prometheus histogram lines for one {"buckets", "sum", "count"} at the given labels."""
    bounds = [f"{le:g}" for le in STEP_TIME_BUCKETS_MS] + ["+Inf"]
    lines = [
        f'{metric}_bucket{{{labels},le="{le}"}} {count}'
        for le, count in zip(bounds, hist["buckets"])
    ]
    lines.append(f"{metric}_sum{{{labels}}} {hist['sum']}")
    lines.append(f"{metric}_count{{{labels}}} {hist['count']}")
    return lines


def render_prometheus(per_engine):
    """Prometheus text exposition of {engine identity: {metric: value}}."""
    lines = []
    for name, (kind, help_text) in _METRICS.items():
        metric = f"mcore:{name}"
        lines.append(f"# HELP {metric} {help_text}")
        lines.append(f"# TYPE {metric} {kind}")
        for engine, values in sorted(per_engine.items()):
            if name in values:
                lines.append(f'{metric}{{engine="{engine}"}} {values[name]}')
        lines.append(f"{metric} {sum(values.get(name, 0) for values in per_engine.values())}")
    for name, help_text in _HISTOGRAMS.items():
        metric = f"mcore:{name}"
        lines.append(f"# HELP {metric} {help_text}")
        lines.append(f"# TYPE {metric} histogram")
        totals = {}  # class -> histogram summed over engines
        for engine, values in sorted(per_engine.items()):
            for step_class, hist in sorted(values.get(name, {}).items()):
                lines += _render_histogram(metric, f'engine="{engine}",step="{step_class}"', hist)
                total = totals.setdefault(
                    step_class, {"buckets": [0] * len(hist["buckets"]), "sum": 0.0, "count": 0}
                )
                total["buckets"] = [a + b for a, b in zip(total["buckets"], hist["buckets"])]
                total["sum"] += hist["sum"]
                total["count"] += hist["count"]
        for step_class, hist in sorted(totals.items()):
            lines += _render_histogram(metric, f'step="{step_class}"', hist)
    for field, help_text in _STEP_MIX_FIELDS.items():
        metric = f"mcore:step_mix_{field}_total"
        lines.append(f"# HELP {metric} {help_text}")
        lines.append(f"# TYPE {metric} counter")
        totals = {}
        for engine, values in sorted(per_engine.items()):
            for step_class, counters in sorted(values.get("step_mix", {}).items()):
                value = counters.get(field, 0)
                lines.append(f'{metric}{{engine="{engine}",step="{step_class}"}} {value}')
                totals[step_class] = totals.get(step_class, 0) + value
        for step_class, value in sorted(totals.items()):
            lines.append(f'{metric}{{step="{step_class}"}} {value}')
    for name, help_text in _PER_REQUEST.items():
        metric = f"mcore:{name}"
        lines.append(f"# HELP {metric} {help_text}")
        lines.append(f"# TYPE {metric} gauge")
        for engine, values in sorted(per_engine.items()):
            for request_id, value in values.get(name, {}).items():
                lines.append(f'{metric}{{engine="{engine}",request="{request_id}"}} {value}')
    return "\n".join(lines) + "\n"


try:
    from quart import Blueprint, current_app

    bp = Blueprint('metrics_api', __name__)

    @bp.route('/metrics', methods=['GET'])
    @bp.route('/v1/metrics', methods=['GET'])
    async def metrics():
        """Engine counters and gauges in the Prometheus text format."""
        client = current_app.config.get('client')
        if client is None:
            return "client not initialized\n", 503
        per_engine = await client.get_metrics()
        return render_prometheus(per_engine), 200, {"Content-Type": "text/plain; version=0.0.4"}

except ImportError as e:
    logger.warning(f"Could not import quart: {e}")
