# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Model E2E timing without a dependency on a particular attention predictor."""
from .api import E2EEstimate, ModelConfig, ModelEvaluator, NonAttentionTimes, PackGroup
from .models import qwen3_30b, qwen35_35b

__all__ = ["E2EEstimate", "ModelConfig", "ModelEvaluator", "NonAttentionTimes", "PackGroup",
           "qwen3_30b", "qwen35_35b"]
__version__ = "0.1.0.dev20261109"
