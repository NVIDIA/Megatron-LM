# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Global MFU (Model FLOPs Utilization) tracker for RL training.

Tracks cumulative FLOPs and wall-clock time for both inference and training
phases, enabling combined MFU reporting.

Usage:
    from megatron.training.mfu_tracker import get_mfu_tracker
    tracker = get_mfu_tracker()
    tracker.add_inference_flops(flops, time_s)
    tracker.add_training_flops(flops, time_s)
    report = tracker.get_report(gpu_peak_tflops)
"""

import threading


class MFUTracker:
    """Thread-safe tracker for inference and training FLOPs/time."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self._inference_flops = 0.0
            self._inference_time = 0.0
            self._inference_tokens = 0
            self._training_flops = 0.0
            self._training_time = 0.0
            self._training_tokens = 0
            # Per-iteration accumulators (reset each RL iteration)
            self._iter_inference_flops = 0.0
            self._iter_inference_time = 0.0
            self._iter_inference_tokens = 0
            self._iter_logprob_time = 0.0
            self._iter_real_training_tokens = 0

    def add_inference_flops(self, flops: float, time_s: float, tokens: int = 0):
        """Called by the inference engine each step."""
        with self._lock:
            self._inference_flops += flops
            self._inference_time += time_s
            self._inference_tokens += tokens
            self._iter_inference_flops += flops
            self._iter_inference_time += time_s
            self._iter_inference_tokens += tokens

    def add_training_flops(self, flops: float, time_s: float, tokens: int = 0):
        """Called by the training loop each iteration."""
        with self._lock:
            self._training_flops += flops
            self._training_time += time_s
            self._training_tokens += tokens

    def get_iter_inference_flops(self) -> float:
        """Get inference FLOPs accumulated since last reset_iter()."""
        with self._lock:
            return self._iter_inference_flops

    def get_iter_inference_time(self) -> float:
        """Get inference time accumulated since last reset_iter()."""
        with self._lock:
            return self._iter_inference_time

    def get_iter_inference_tokens(self) -> int:
        """Get inference tokens accumulated since last reset_iter()."""
        with self._lock:
            return self._iter_inference_tokens

    def add_logprob_time(self, time_s: float):
        """Called after the compute-logprobs phase each RL iteration."""
        with self._lock:
            self._iter_logprob_time += time_s

    def get_iter_logprob_time(self) -> float:
        with self._lock:
            return self._iter_logprob_time

    def set_iter_real_training_tokens(self, tokens: int):
        """Set the real (non-padding) training token count for this iteration."""
        with self._lock:
            self._iter_real_training_tokens = tokens

    def get_iter_real_training_tokens(self) -> int:
        with self._lock:
            return self._iter_real_training_tokens

    def reset_iter(self):
        """Reset per-iteration accumulators."""
        with self._lock:
            self._iter_inference_flops = 0.0
            self._iter_inference_time = 0.0
            self._iter_inference_tokens = 0
            self._iter_logprob_time = 0.0
            self._iter_real_training_tokens = 0

    def save_iter(self) -> dict:
        """Snapshot per-iteration accumulators so they can be restored later.

        Used around evaluation to prevent eval inference from polluting
        training throughput metrics.
        """
        with self._lock:
            return {
                'inference_flops': self._iter_inference_flops,
                'inference_time': self._iter_inference_time,
                'inference_tokens': self._iter_inference_tokens,
                'logprob_time': self._iter_logprob_time,
                'real_training_tokens': self._iter_real_training_tokens,
            }

    def restore_iter(self, snapshot: dict):
        """Restore per-iteration accumulators from a previous snapshot."""
        with self._lock:
            self._iter_inference_flops = snapshot['inference_flops']
            self._iter_inference_time = snapshot['inference_time']
            self._iter_inference_tokens = snapshot['inference_tokens']
            self._iter_logprob_time = snapshot['logprob_time']
            self._iter_real_training_tokens = snapshot['real_training_tokens']

    def get_report(self, gpu_peak_tflops: float) -> dict:
        """Compute MFU breakdown.

        All FLOPs stored in this tracker are per-GPU.

        Args:
            gpu_peak_tflops: Peak BF16 TFLOP/s for one GPU.

        Returns:
            dict with keys: inference_tflops, inference_time, inference_mfu,
                           training_tflops, training_time, training_mfu,
                           total_tflops, total_time, total_mfu.
        """
        with self._lock:
            inf_tflops = self._inference_flops / 1e12
            inf_time = self._inference_time
            inf_tokens = self._inference_tokens
            train_tflops = self._training_flops / 1e12
            train_time = self._training_time
            train_tokens = self._training_tokens

        total_tflops = inf_tflops + train_tflops
        total_time = inf_time + train_time
        total_tokens = inf_tokens + train_tokens

        def _mfu(tflops, time_s):
            if time_s <= 0 or gpu_peak_tflops <= 0:
                return 0.0
            return tflops / time_s / gpu_peak_tflops * 100.0

        def _toks_per_sec(tokens, time_s):
            if time_s <= 0:
                return 0.0
            return tokens / time_s

        return {
            'inference_tflops': inf_tflops,
            'inference_time': inf_time,
            'inference_tokens': inf_tokens,
            'inference_throughput': inf_tflops / inf_time if inf_time > 0 else 0,
            'inference_mfu': _mfu(inf_tflops, inf_time),
            'inference_toks_per_sec_per_gpu': _toks_per_sec(inf_tokens, inf_time),
            'training_tflops': train_tflops,
            'training_time': train_time,
            'training_tokens': train_tokens,
            'training_throughput': train_tflops / train_time if train_time > 0 else 0,
            'training_mfu': _mfu(train_tflops, train_time),
            'training_toks_per_sec_per_gpu': _toks_per_sec(train_tokens, train_time),
            'total_tflops': total_tflops,
            'total_time': total_time,
            'total_tokens': total_tokens,
            'total_throughput': total_tflops / total_time if total_time > 0 else 0,
            'total_mfu': _mfu(total_tflops, total_time),
            'total_toks_per_sec_per_gpu': _toks_per_sec(total_tokens, total_time),
        }


_GLOBAL_TRACKER = None
_TRACKER_LOCK = threading.Lock()


def get_mfu_tracker() -> MFUTracker:
    """Get or create the global MFU tracker singleton."""
    global _GLOBAL_TRACKER
    if _GLOBAL_TRACKER is None:
        with _TRACKER_LOCK:
            if _GLOBAL_TRACKER is None:
                _GLOBAL_TRACKER = MFUTracker()
    return _GLOBAL_TRACKER
