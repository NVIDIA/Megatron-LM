# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Utilities for lightweight anomaly capture and channel-aware monitoring."""

from __future__ import annotations

from collections import deque
import json
import os
from typing import Any

from megatron.core import mpu


_LAST_BATCH_INFO: dict[str, Any] = {}


def set_last_batch_info(batch_info: dict[str, Any]) -> None:
    global _LAST_BATCH_INFO
    _LAST_BATCH_INFO = batch_info


def get_last_batch_info() -> dict[str, Any]:
    return _LAST_BATCH_INFO


class TrainingAnomalyMonitor:
    """Step-level anomaly monitor with buffered IO writes."""

    def __init__(self, args) -> None:
        self.enabled = bool(getattr(args, "capture_anomaly_data", False))
        if not self.enabled:
            return

        self.loss_window = max(1, int(getattr(args, "anomaly_window_size", 100)))
        self.loss_multiplier = float(getattr(args, "anomaly_dynamic_multiplier", 3.0))
        self.loss_abs_max = float(getattr(args, "anomaly_loss_abs_max", 1.0e10))
        self.grad_abs_max = float(getattr(args, "anomaly_grad_norm_abs_max", 1.0e10))
        self.flush_interval = max(1, int(getattr(args, "anomaly_flush_interval", 10)))
        self.max_buffer_size = max(1, int(getattr(args, "anomaly_buffer_size", 64)))
        self.output_path = str(getattr(args, "anomaly_output_file", "anomaly_events.jsonl"))
        self.anomaly_start_iter = int(getattr(args, "anomaly_start_iter", 0))

        self.loss_history: deque[float] = deque(maxlen=self.loss_window)
        self.grad_history: deque[float] = deque(maxlen=self.loss_window)
        self._buffer: list[dict[str, Any]] = []

    def _is_dp_rank0(self) -> bool:
        return mpu.get_data_parallel_rank(with_context_parallel=True) == 0

    def _to_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _mean_or_none(self, history: deque[float]) -> float | None:
        if not history:
            return None
        return sum(history) / len(history)

    def observe(self, iteration: int, loss_value: Any, grad_norm: Any, batch_info: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return

        loss = self._to_float(loss_value)
        grad = self._to_float(grad_norm)

        loss_avg = self._mean_or_none(self.loss_history)
        grad_avg = self._mean_or_none(self.grad_history)

        loss_dynamic_thres = None if loss_avg is None else self.loss_multiplier * loss_avg
        grad_dynamic_thres = None if grad_avg is None else self.loss_multiplier * grad_avg

        loss_is_anomaly = (
            loss is not None
            and (
                (loss_dynamic_thres is not None and loss > loss_dynamic_thres)
                or loss > self.loss_abs_max
            )
        )
        grad_is_anomaly = (
            grad is not None
            and (
                (grad_dynamic_thres is not None and grad > grad_dynamic_thres)
                or grad > self.grad_abs_max
            )
        )

        if iteration >= self.anomaly_start_iter and (loss_is_anomaly or grad_is_anomaly):
            info = {
                "step_id": int(iteration),
                "loss": loss,
                "grad_norm": grad,
                "loss_dynamic_threshold": loss_dynamic_thres,
                "grad_dynamic_threshold": grad_dynamic_thres,
                "data_info": batch_info or {},
            }
            self._buffer.append(info)

        if loss is not None:
            self.loss_history.append(loss)
        if grad is not None:
            self.grad_history.append(grad)

        if (iteration % self.flush_interval == 0) or (len(self._buffer) >= self.max_buffer_size):
            self.flush()

    def flush(self) -> None:
        if (not self.enabled) or (not self._buffer) or (not self._is_dp_rank0()):
            return

        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(self.output_path, "a", encoding="utf-8") as fout:
            for record in self._buffer:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._buffer.clear()
