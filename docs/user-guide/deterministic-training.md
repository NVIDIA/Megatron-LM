<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Deterministic Training

Deterministic training guarantees that two runs with identical inputs produce identical outputs at every step. Useful for debugging regressions and for reproducibility studies.

Pass `--deterministic-mode` to any Megatron training entry point (e.g. `pretrain_hybrid.py`):

```bash
python pretrain_hybrid.py \
  --deterministic-mode \
  <other args ...>
```

When enabled, Megatron applies the env vars and config overrides below via `megatron.training.determinism.apply_determinism_to_args` (called from `validate_args`).

## Environment variables

Each variable may be set by the launcher or left unset. If set, the value must be one that has been validated as deterministic — anything else fails hard with an assertion. If unset, `apply_determinism_env` fills the canonical default (except `MAMBA_DETERMINISTIC` and `CAUSAL_CONV1D_DETERMINISTIC`, which the external `mamba_ssm` and `causal_conv1d` packages auto-detect from `torch.are_deterministic_algorithms_enabled()`). Must be set before the first cuBLAS / Transformer Engine call — `apply_determinism_to_args` runs early in `validate_args` to guarantee this.

| Variable | Accepted values (or unset) | Default filled if unset | Reason |
|---|---|---|---|
| `NCCL_ALGO` | subset of `{Ring, CollnetDirect, CollnetChain, ^NVLS}` | `Ring` | Conservative default — `Ring`'s reduction order is fixed by topology, so it is bit-exact across runs on every supported NCCL version |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | `0` | `0` | Forces Transformer Engine to use deterministic algorithms |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` or `:16:8` | `:4096:8` | Deterministic cuBLAS workspace (both sizes are reproducible per NVIDIA docs; `:4096:8` is faster, `:16:8` uses less memory) |
| `TRITON_CACHE_AUTOTUNING` | `0` or `1` | *(none — opt-in)* | Persists Triton autotune winners for autotuners **outside** the pinned scope, so ranks reuse one choice instead of re-timing it. Kernels inside the scope are pinned without timing either way — see [Triton autotuning](#triton-autotuning) |
| `TRITON_CACHE_DIR` | any shared-filesystem path | *(none — required only with `TRITON_CACHE_AUTOTUNING=1`)* | No safe default exists: unset, Triton uses a node-local directory and each node autotunes on its own. Required rather than filled in |
| `TRITON_PRINT_AUTOTUNING` | `1` | *(none — not set)* | Logs the config Triton times, which only happens for autotuners outside the pinned scope. For pinned kernels use `--triton-autotune-enumerate` and `--triton-autotune-verify-every` — see [Verifying kernel-config agreement](#verifying-kernel-config-agreement) |
| `MAMBA_DETERMINISTIC` | any string starting with `'1'` | *(none — SSM auto-detects)* | Controls the external `mamba_ssm` package, which auto-follows `torch.are_deterministic_algorithms_enabled()` when unset; only an explicit non-deterministic override is rejected. Megatron's in-tree SSM kernels (`megatron.core.ssm.ops`) do not read it and follow `--deterministic-mode` instead |
| `CAUSAL_CONV1D_DETERMINISTIC` | any string starting with `'1'` | *(none — the kernel auto-detects)* | causal_conv1d ≥ 1.6.0 auto-follows `torch.are_deterministic_algorithms_enabled()` when unset, reducing the conv weight/bias gradients through a workspace instead of `atomicAdd`; the Mamba and GDP mixers reject a deterministic run without it |

If you override `NCCL_ALGO`, the value must be a subset of `{Ring, CollnetDirect, CollnetChain, ^NVLS}`. `Tree` is intentionally excluded: its intra-node chain reduction order is not user-controllable, and the inter-node tree topology can vary across runs without a pinned topology file, so it cannot be vouched for as bit-exact across stacks. `^NVLS` is accepted (banning NVLS is a legitimate user choice on hardware that exposes it); the user is responsible for ensuring whatever NCCL falls back to is deterministic on their environment.

## Config requirements

Checked against the parsed `args` Namespace in `apply_determinism_to_args`. Incompatible options are rejected with an explicit error rather than silently flipped off — you must disable them yourself so the run matches the config you asked for:

| Flag | Behavior under `--deterministic-mode` |
|---|---|
| `--cross-entropy-loss-fusion` | Must be off — asserted (fused CE is non-deterministic); drop the flag yourself |
| `--tp-comm-overlap` | Must be off — asserted (the overlap path is not bit-exact); drop the flag yourself |
| `moe_router_aux_loss_fusion` | Must be off — asserted (TE's fused aux-loss kernel is non-deterministic); follows `moe_router_fusion` when unset |
| `torch.use_deterministic_algorithms` | Set to `True` |
| `torch.utils.deterministic.fill_uninitialized_memory` | Set to `False` — see below |

Flash attention is permitted: Transformer Engine's flash-attention backend is deterministic when `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` (see the [Transformer Engine docs](https://docs.nvidia.com/deeplearning/transformer-engine/api/pytorch.html)).

## Uninitialized-memory fill

Determinism costs an *independent* output buffer: a reduction that would otherwise accumulate into shared memory with unordered atomics writes into its own buffer instead, fixing the summation order run to run. That is what makes training reproducible, and `--deterministic-mode` keeps it.

`torch.use_deterministic_algorithms(True)` also switches on a separate knob, `torch.utils.deterministic.fill_uninitialized_memory`, which fills every uninitialized allocation — `torch.empty`, `empty_like`, `empty_strided`, `Tensor.resize_` — with NaN or MAX_INT, so a kernel *reading* memory it never wrote reads the same bytes every run. Reproducibility does not need that, and it is not free: one extra fill kernel per empty allocation, serialized between real work, which suppresses the overlap between compute and communication and so costs far more wall time than GPU time. Clearing it is worth roughly **15% TFLOP/s** on large configs, and more the more `torch.empty` calls a step makes. `apply_determinism_to_args` clears it immediately after enabling deterministic algorithms.

Padding matters only if a computation reads it. **Benign — nothing reads it:** computed values are bit-identical and only saved bytes differ. Checkpoints are the example — some saved tensors carry trailing pad slots that no kernel writes, so two runs of the same configuration write files differing in those bytes while every trained value matches. **Harmful — a computation consumes it:** a reduction over a padded tail, a GEMM with a rounded-up K, an unmasked attention region. Results then differ run to run, and the fill does not make them correct, only repeatably wrong — every run reads the same NaN instead of different garbage. Fix the read; re-enabling the fill hides it.

Set the fill back to `True` while hunting such a read — turning garbage into a loud NaN is the one thing it is good for:

```python
import torch.utils.deterministic
torch.utils.deterministic.fill_uninitialized_memory = True
```

## Triton autotuning

Triton picks a kernel config by timing its candidates, so the winner depends on the machine at that instant and ranks can disagree. `--deterministic-mode` pins it: the `megatron.core.tuning` adapter selects one candidate per kernel and shape without timing anything, from a tuned table when one matches (tables ship for `sm100` and `sm103`, or can be recorded with `--triton-autotune-mode record`), otherwise from the cheapest candidate by a pure function of the candidate list. Pinned kernels never consult Triton's autotune cache, so every rank computes the same answer by construction. The cheapest candidate is not necessarily the fastest one; a recorded table recovers the throughput.

The default scope is `mamba_ssm`, `transformer_engine` and `megatron.core` (`--triton-autotune-modules`). Kernels whose outputs do not depend on the config — by default only pure data movement, Transformer Engine's MoE permutation kernels, listed in `AutotunePolicy.config_invariant` (`--triton-autotune-config-invariant`) — keep Triton's timed choice, since pinning them would only cost throughput. See `megatron/core/tuning/README.md` for recording tables, precedence rules and the full option list.

Autotuners outside the scope still time their candidates. `TRITON_CACHE_AUTOTUNING=1` with a shared `TRITON_CACHE_DIR` makes those ranks reuse one cached winner, but a rank that misses the cache re-times the selection on its own and can pick differently, so prefer adding their modules to `--triton-autotune-modules`. Setting `TRITON_CACHE_AUTOTUNING=1` without `TRITON_CACHE_DIR` is rejected — unset, Triton falls back to a node-local directory, which is exactly the case the cache is meant to prevent.

## Verifying kernel-config agreement

`--triton-autotune-verify-every N` compares, every N training steps, the configs that ranks chose for the same kernel and shape, and logs any disagreement (`--triton-autotune-verify-strict` raises instead). Each check exchanges only the choices made since the previous one. `--triton-autotune-enumerate` logs, per rank, every multi-config autotuner the run reaches and whether it is pinned, timed as config-invariant, or outside the scope.

For autotuners outside the scope, `TRITON_PRINT_AUTOTUNING=1` makes each rank log the config Triton selects; a rank only logs when it tunes, so a run where some ranks hit the autotune cache and others miss cannot be compared this way.

## Verifying determinism

The bit-exact correctness suite lives at `tests/unit_tests/determinism/correctness/`. It parametrizes over model presets (GPT-like, Llama-like, Hybrid/Mamba) × parallelism cells (TP, PP, VPP, EP, FSDP, and composites) and asserts that two runs of the same configuration produce bit-identical outputs and gradients. FP8 / FP4 recipes (`tensorwise`, `delayed`, `mxfp8`, `nvfp4`) are covered by `tests/unit_tests/determinism/correctness/test_fp8_determinism.py`; the Blackwell-only recipes are capability-skipped on Hopper.

The cost of `--deterministic-mode` is measured outside pytest by an nsys-driven per-NVTX-range breakdown: `tests/performance_tests/shell_test_utils/determinism/run_nsys_breakdown.sh` wraps any training entry point (e.g. `pretrain_hybrid.py --profile`) under nsys for a det-vs-nondet comparison, and `tests/performance_tests/shell_test_utils/determinism/print_nsys_leaderboard.py` joins the two CSVs into a side-by-side table. The CI invocation lives at `tests/test_utils/recipes/h100/determinism-perf.yaml`.
