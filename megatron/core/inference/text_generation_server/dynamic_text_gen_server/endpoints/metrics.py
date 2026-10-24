# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Prometheus metrics endpoint.

GET /metrics asks every data parallel engine for its counters through the InferenceClient ->
coordinator -> engines path (GET_METRICS); each engine answers at its next step. Counters are
lifetime totals, so throughput over a window is the difference of two scrapes divided by the
window length. Each metric is reported per engine (label ``engine``) and summed over engines.
"""

import logging

logger = logging.getLogger(__name__)

# name -> (Prometheus type, help).
_METRICS = {
    "generation_tokens_total": ("counter", "Tokens generated."),
    "prompt_tokens_total": ("counter", "Prompt tokens prefilled (computed)."),
    "prompt_tokens_cached_total": ("counter", "Prompt tokens skipped by prefix-cache hits."),
    "requests_finished_total": ("counter", "Requests finished."),
    "engine_steps_total": ("counter", "Engine steps."),
    "requests_running": ("gauge", "Requests in the active batch."),
    "requests_paused": ("gauge", "Requests paused (out of KV blocks)."),
    "requests_waiting": ("gauge", "Requests queued for admission."),
}


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
