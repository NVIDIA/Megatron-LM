# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""CUDA profiler control endpoints.

POST /start_profile and /stop_profile relay a control signal through the
InferenceClient -> data-parallel coordinator -> every connected EP/DP engine,
which calls cudaProfilerStart()/cudaProfilerStop(). Pair with an outer
`nsys profile --capture-range=cudaProfilerApi` to bracket a capture window.
"""

import logging

logger = logging.getLogger(__name__)

try:
    from quart import Blueprint, current_app, jsonify

    bp = Blueprint('profile_api', __name__)

    @bp.route('/start_profile', methods=['POST'])
    @bp.route('/v1/start_profile', methods=['POST'])
    async def start_profile():
        """Broadcast cudaProfilerStart to all engines."""
        client = current_app.config.get('client')
        if client is None:
            return jsonify({"status": "error", "details": "client not initialized"}), 503
        client.start_cuda_profiler()
        return jsonify({"status": "ok", "action": "start_profile"}), 200

    @bp.route('/stop_profile', methods=['POST'])
    @bp.route('/v1/stop_profile', methods=['POST'])
    async def stop_profile():
        """Broadcast cudaProfilerStop to all engines."""
        client = current_app.config.get('client')
        if client is None:
            return jsonify({"status": "error", "details": "client not initialized"}), 503
        client.stop_cuda_profiler()
        return jsonify({"status": "ok", "action": "stop_profile"}), 200

except ImportError as e:
    logger.warning(f"Could not import quart: {e}")
