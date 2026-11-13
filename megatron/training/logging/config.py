# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Serialize the structured run configuration for logging backends."""

import json
from typing import Any


def _config_text(value: Any) -> str:
    try:
        return str(value)
    except Exception:
        return f"<{type(value).__name__}>"


def _omit_credentials(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _omit_credentials(item)
            for key, item in value.items()
            if str(key) != "mlflow_tracking_uri"
            and not any(
                part in str(key).lower()
                for part in ("password", "secret", "api_key", "access_token")
            )
        }
    if isinstance(value, (list, tuple)):
        return [_omit_credentials(item) for item in value]
    return value


def serialize_config_for_logging(cfg) -> dict[str, Any]:
    """Return JSON-compatible config metadata, omitting credential fields."""
    return json.loads(json.dumps(_omit_credentials(cfg.to_dict()), default=_config_text))
