# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared helpers for per-module determinism tests.

The env vars required for bit-exact reproducibility are set in each
subpackage's ``__init__.py`` (``correctness/`` always; ``perf/`` only when
``DETERMINISM_PERF_MODE != 'nondet'``) so they take effect on package
import, before any cuBLAS / TE call inside a test module.
"""

import random

import numpy as np
import torch

try:
    # Public-by-import helper used by PyTorch's own test_cuda.py to convert
    # milliseconds to device-cycle counts for torch.cuda._sleep.
    from torch.testing._internal.common_utils import get_cycles_per_ms
except ImportError:  # pragma: no cover — fallback only if PyTorch internals move

    def get_cycles_per_ms() -> float:
        # Rough lower bound: H100 boosts to ~1.8 GHz → ~1.8M cycles/ms. Picking
        # 1M is conservative — the jitter will be a bit shorter than requested,
        # not longer, which keeps test runtime bounded.
        return 1_000_000.0


def capture_rng_state() -> dict:
    """Snapshot every RNG that the framework consumes during a fwd+bwd pass.

    Mirrors ``RerunStateMachine._save_state`` in
    ``megatron/core/rerun_state_machine.py``. Also captures Megatron's own
    ``CudaRNGStatesTracker`` (used for TP-aware dropout), which advances
    independently of ``torch.cuda``'s RNG when any layer calls
    ``get_cuda_rng_tracker().fork()``.
    """
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    return {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),
        "mpu_tracker": get_cuda_rng_tracker().get_states(),
    }


def restore_rng_state(state: dict) -> None:
    """Inverse of ``capture_rng_state``."""
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"])
    if "mpu_tracker" in state:
        get_cuda_rng_tracker().set_states(state["mpu_tracker"])


def _strict_equal_with_nan(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Element-wise equality where NaN at the same position counts as equal.

    Plain ``torch.equal`` returns False for any NaN-vs-NaN comparison, which
    is the correct semantics for value equality but wrong for *determinism*
    where we only care that two runs produced bit-identical outputs — same
    NaN pattern included.
    """
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    eq = (a == b) | (a.isnan() & b.isnan())
    return bool(eq.all().item())


def assert_bit_exact(out_a, grads_a, out_b, grads_b) -> None:
    """Assert two (output, grad-dict) pairs are bit-exact equal.

    Uses explicit ``raise AssertionError`` rather than ``assert`` statements:
    this helper lives outside ``test_*.py`` so pytest does NOT rewrite its
    asserts, and bare ``assert`` would be stripped under ``python -O`` /
    ``PYTHONOPTIMIZE=1`` — turning every determinism check into a silent
    no-op.
    """
    if not _strict_equal_with_nan(out_a, out_b):
        raise AssertionError("Outputs differ between deterministic runs")
    if grads_a.keys() != grads_b.keys():
        raise AssertionError("Grad keys differ between runs")
    for name in grads_a:
        if not _strict_equal_with_nan(grads_a[name], grads_b[name]):
            raise AssertionError(f"Grad mismatch for {name}")


def collect_grads(modules) -> dict:
    """Snapshot every parameter's gradient across one or more modules.

    Handles BOTH eager autograd (``p.grad``) and Megatron-FSDP
    (``p.main_grad`` — the adapter ``del``s ``p.grad`` post-backward, so
    we have to fall through to ``main_grad`` when ``p.grad`` is None).
    """
    grads = {}
    for i, m in enumerate(modules):
        for name, p in m.named_parameters():
            g = getattr(p, "main_grad", None)
            if g is None:
                g = p.grad
            if g is not None:
                grads[f"chunk{i}.{name}"] = g.detach().clone()
    return grads


def zero_grads(model) -> None:
    """Reset both eager ``p.grad`` and Megatron-FSDP's grad buffer."""
    model.zero_grad(set_to_none=True)
    zero_buf = getattr(model, "zero_grad_buffer", None)
    if callable(zero_buf):
        zero_buf()


def reset_quantizer_state(modules) -> None:
    """Reset per-module TE FP8/FP4 quantizer state to its post-init values.

    Required for cross-step recipes (``delayed`` FP8): per-module
    ``fp8_meta`` / ``fp4_meta`` carries amax history + derived scale across
    forward passes. Without this reset, run B in the bit-exact harness
    sees run A's updated amax history and computes different scale factors
    → outputs diverge even though the model is deterministic in a real
    training loop.

    No-op for stateless recipes (``tensorwise`` / ``mxfp8`` / ``nvfp4``)
    and for bf16 cells — those modules either lack the ``*_meta``
    attribute or have an empty history.
    """
    for module in modules:
        for m in module.modules():
            for meta_attr in ("fp8_meta", "fp4_meta"):
                meta = getattr(m, meta_attr, None)
                if not isinstance(meta, dict):
                    continue
                for scaling_key in ("scaling_fwd", "scaling_bwd"):
                    sc = meta.get(scaling_key)
                    if sc is None:
                        continue
                    hist = getattr(sc, "amax_history", None)
                    if hist is not None:
                        hist.zero_()
                    scale = getattr(sc, "scale", None)
                    if scale is not None:
                        scale.fill_(1.0)
                    scale_inv = getattr(sc, "scale_inv", None)
                    if scale_inv is not None:
                        scale_inv.fill_(1.0)


class RacingStreams:
    """Run side-stream GEMMs in parallel with the model to perturb scheduling.

    Goal: force the CUDA scheduler to keep making different choices across
    runs so that any kernel whose bit-output depends on dispatch order
    surfaces as a bit-exact failure.

    Three things make scheduling vary more than the default "all streams
    do the same work" pattern:

    * ranks open DIFFERENT numbers of side streams (rank N opens N % 4
      more streams), so each rank presents different SM pressure → ranks
      finish model fwd/bwd at different wall-clock times → NCCL
      collectives race in different orders across runs.
    * each side stream runs GEMMs of MIXED SIZES (1024 / 2048 / 3072),
      picked from a per-rank-seeded CPU generator. Mixed sizes create
      more scheduling decision points than a uniform chain.
    * half the side streams have HIGHER priority than the default
      (priority=-1 vs 0). The scheduler must arbitrate priority classes
      under contention; the arbitration is hardware-state-dependent.

    Notes:

    * Effective only when ``CUDA_DEVICE_MAX_CONNECTIONS > 1``. Under ``=1``
      (Hopper determinism default) all streams serialise into a single
      hardware queue and this helper is a no-op. The
      ``test_bit_exact_under_racing_streams`` test skips on ``=1``.
    * Side-stream RNG uses a CPU ``torch.Generator`` — does NOT touch the
      per-device CUDA RNG, so the model's input RNG state stays clean.
    * Matrix contents are ``torch.ones`` because GEMM scheduling depends
      on shape/dtype/launch order, not values.
    """

    # GEMM sizes the stream rotates through. Mixed → adjacent kernels have
    # different completion latencies → more scheduling decision points.
    _SIZE_CHOICES = (1024, 2048, 3072)
    # Rank N opens this-many-extra streams, cycling. Smaller than world size
    # so within a node neighbouring ranks differ but bounded.
    _RANK_STREAM_CYCLE = 4
    # Alternating priorities — half HIGH, half DEFAULT. Scheduler must
    # arbitrate priority classes under contention.
    _STREAM_PRIORITIES = (-1, 0)
    # Per-rank generator seed = _BASE_SEED + rank. CPU generator, not CUDA,
    # so the model's per-device RNG state is untouched.
    _BASE_SEED = 0xC0FFEE

    def __init__(self, num_streams: int = 4, num_iters: int = 200):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.num_streams = num_streams + (rank % self._RANK_STREAM_CYCLE)
        self.num_iters = num_iters
        self._gen = torch.Generator()
        self._gen.manual_seed(self._BASE_SEED + rank)
        self.streams: list[torch.cuda.Stream] = []
        self._noise: list[torch.Tensor] = []

    def __enter__(self):
        self.streams = [
            torch.cuda.Stream(priority=self._STREAM_PRIORITIES[i % len(self._STREAM_PRIORITIES)])
            for i in range(self.num_streams)
        ]
        for stream in self.streams:
            with torch.cuda.stream(stream):
                # One randint call instead of num_iters calls.
                size_indices = torch.randint(
                    0, len(self._SIZE_CHOICES), (self.num_iters,), generator=self._gen
                ).tolist()
                # Keep only the last matmul result alive — earlier ones get
                # GC'd (PyTorch's caching allocator preserves storage while
                # the in-flight kernel still references it). Otherwise we'd
                # retain num_iters * num_streams tensors → multi-GB.
                result = None
                for idx in size_indices:
                    sz = self._SIZE_CHOICES[idx]
                    a = torch.ones(sz, sz, device="cuda", dtype=torch.bfloat16)
                    b = torch.ones(sz, sz, device="cuda", dtype=torch.bfloat16)
                    result = a @ b
                if result is not None:
                    self._noise.append(result)
        return self

    def __exit__(self, *args):
        for stream in self.streams:
            stream.synchronize()
        self.streams.clear()
        self._noise.clear()


class CudaSleepJitter:
    """Inject rank-asymmetric ``torch.cuda._sleep`` calls on every submodule
    forward.

    Same pattern PyTorch's own ``test/test_cuda.py`` uses to stress stream
    ordering: ``_sleep`` is a no-op kernel that spins for a fixed device-cycle
    count, so it perturbs scheduling without touching memory. Pairing this
    with ``NCCL_LAUNCH_RACE_FATAL=1`` turns any latent collective-ordering bug
    into a hard test failure.

    Determinism semantics: a fresh ``torch.Generator`` is seeded per-rank in
    ``__enter__``, so two consecutive ``with`` blocks see identical jitter
    sequences per rank (different across ranks). Re-create the context
    manager around each run rather than reusing one across runs.
    """

    def __init__(self, module: torch.nn.Module, max_us_per_hook: int = 200, seed: int = 0xCAFE):
        self._module = module
        self._max_us = max_us_per_hook
        self._seed = seed
        self._handles: list = []

    def __enter__(self):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        gen = torch.Generator()
        gen.manual_seed(self._seed + rank)

        # microseconds → ms → cycles
        max_cycles = int((self._max_us / 1000.0) * get_cycles_per_ms())

        def _hook(_mod, _args, _max=max_cycles, _gen=gen):
            if _max <= 0:
                return
            cycles = int(torch.randint(0, _max + 1, (1,), generator=_gen).item())
            if cycles > 0:
                torch.cuda._sleep(cycles)

        for sub in self._module.modules():
            self._handles.append(sub.register_forward_pre_hook(_hook))
        return self

    def __exit__(self, *args):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def maybe_fsdp_wrap(model: torch.nn.Module, parallelism: dict) -> torch.nn.Module:
    """If ``parallelism["FSDP"] > 1``, wrap ``model`` with Megatron-FSDP.

    Uses the production path from ``megatron/training/training.py``: the
    ``FullyShardedDataParallel`` adapter from ``mcore_fsdp_adapter``, with the
    ``ProcessGroupCollection`` derived from the current ``parallel_state``.
    This means TP/PP/CP/EP groups already initialised by
    ``Utils.initialize_model_parallel`` are honoured automatically — FSDP
    just shards along the DP dimension that ``parallel_state`` exposes.
    """
    if parallelism.get("FSDP", 1) <= 1:
        return model

    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
    from megatron.core.process_groups_config import ProcessGroupCollection

    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=False,
        overlap_grad_reduce=False,  # determinism — disable async overlap
        overlap_param_gather=False,
        use_distributed_optimizer=True,
        bucket_size=40_000_000,
    )
    config = getattr(model, "config", None)
    return FullyShardedDataParallel(
        config=config, ddp_config=ddp_config, module=model, pg_collection=pg_collection
    )
