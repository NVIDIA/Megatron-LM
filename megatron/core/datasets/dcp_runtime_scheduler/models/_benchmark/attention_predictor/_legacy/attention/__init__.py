# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Frozen, hardware-calibrated core-attention estimators."""
from .estimator import AttentionEstimator, predict_attention, verify_calibration

__all__ = ["AttentionEstimator", "predict_attention", "verify_calibration"]
