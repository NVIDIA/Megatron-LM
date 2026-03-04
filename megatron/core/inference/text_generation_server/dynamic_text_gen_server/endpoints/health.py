# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging

logger = logging.getLogger(__name__)

try:
    from flask import Blueprint, current_app, jsonify

    bp = Blueprint('health_api', __name__)

    @bp.route('/health', methods=['GET'])
    @bp.route('/v1/health', methods=['GET'])
    async def health():
        """
        Handles GET requests for service health.
        Checks if the inference client is initialized and reachable.
        """
        status_response = {"status": "ok", "service": "Megatron Inference Server", "ready": False}

        try:
            client = current_app.config.get('client')

            if client is not None:
                status_response["ready"] = True
                return jsonify(status_response), 200
            else:
                logger.error("Health check failed: Client not found in app config.")
                status_response["status"] = "error"
                status_response["details"] = "Inference client not initialized"
                return jsonify(status_response), 503

        except Exception as e:
            logger.error(f"Health check failed with exception: {e}")
            return jsonify({"status": "error", "details": str(e)}), 500

except ImportError as e:
    logger.warning(f"Could not import flask: {e}")
