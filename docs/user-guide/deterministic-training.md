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

Each variable may be set by the launcher or left unset. If set, the value must be one that has been validated as deterministic — anything else fails hard with an assertion. If unset, `apply_determinism_env` fills the canonical default (except `MAMBA_DETERMINISTIC` and `CAUSAL_CONV1D_DETERMINISTIC`, which their kernels auto-detect from `torch.are_deterministic_algorithms_enabled()`). Must be set before the first cuBLAS / Transformer Engine call — `apply_determinism_to_args` runs early in `validate_args` to guarantee this.

| Variable | Accepted values (or unset) | Default filled if unset | Reason |
|---|---|---|---|
| `NCCL_ALGO` | subset of `{Ring, CollnetDirect, CollnetChain, ^NVLS}` | `Ring` | Conservative default — `Ring`'s reduction order is fixed by topology, so it is bit-exact across runs on every supported NCCL version |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | `0` | `0` | Forces Transformer Engine to use deterministic algorithms |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` or `:16:8` | `:4096:8` | Deterministic cuBLAS workspace (both sizes are reproducible per NVIDIA docs; `:4096:8` is faster, `:16:8` uses less memory) |
| `TRITON_CACHE_AUTOTUNING` | `0` or `1` | *(none — opt-in)* | Persists each Triton autotune winner so every rank reuses one choice instead of re-timing it. Left unset, deterministic mode instead pins the cheapest config, which needs no cache — see [Triton autotuning](#triton-autotuning) |
| `TRITON_CACHE_DIR` | any shared-filesystem path | *(none — required only with `TRITON_CACHE_AUTOTUNING=1`)* | No safe default exists: unset, Triton uses a node-local directory and each node autotunes on its own. Required rather than filled in |
| `TRITON_PRINT_AUTOTUNING` | `1` | *(none — recommended, not set)* | Logs the config each rank selected. Changes no numerics, so it is recommended rather than forced; when `TRITON_CACHE_AUTOTUNING=1` and this is unset, a startup line reminds you. See [Verifying kernel-config agreement](#verifying-kernel-config-agreement) |
| `MAMBA_DETERMINISTIC` | any string starting with `'1'` | *(none — SSM auto-detects)* | Mamba SSM auto-follows `torch.are_deterministic_algorithms_enabled()` when unset; only an explicit non-deterministic override is rejected |
| `CAUSAL_CONV1D_DETERMINISTIC` | any string starting with `'1'` | *(none — the kernel auto-detects)* | causal_conv1d ≥ 1.6.0 auto-follows `torch.are_deterministic_algorithms_enabled()` when unset, reducing the conv weight/bias gradients through a workspace instead of `atomicAdd`; the Mamba and GDP mixers reject a deterministic run without it |

If you override `NCCL_ALGO`, the value must be a subset of `{Ring, CollnetDirect, CollnetChain, ^NVLS}`. `Tree` is intentionally excluded: its intra-node chain reduction order is not user-controllable, and the inter-node tree topology can vary across runs without a pinned topology file, so it cannot be vouched for as bit-exact across stacks. `^NVLS` is accepted (banning NVLS is a legitimate user choice on hardware that exposes it); the user is responsible for ensuring whatever NCCL falls back to is deterministic on their environment.

## Config requirements

Checked against the parsed `args` Namespace in `apply_determinism_to_args`. Incompatible options are rejected with an explicit error rather than silently flipped off — you must disable them yourself so the run matches the config you asked for:

| Flag | Behavior under `--deterministic-mode` |
|---|---|
| `--cross-entropy-loss-fusion` | Must be off — asserted (fused CE is non-deterministic); drop the flag yourself |
| `--tp-comm-overlap` | Must be off — asserted (the overlap path is not bit-exact); drop the flag yourself |
| `torch.use_deterministic_algorithms` | Set to `True` |

Flash attention is permitted: Transformer Engine's flash-attention backend is deterministic when `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` (see the [Transformer Engine docs](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html)).

## Triton autotuning

Triton picks a kernel config by timing its candidates, so the winner depends on the machine at that instant and ranks can disagree. Deterministic mode offers two ways to remove that variance:

| Strategy | How to select it | Determinism rests on |
|---|---|---|
| **Pinned config** (default) | leave `TRITON_CACHE_AUTOTUNING` unset | Nothing external. `autotune_configs` picks the cheapest config by a pure function of the candidate list, so every rank computes the same answer without timing anything. Slower, since the pinned config is not necessarily the fastest one. |
| **Cached autotuning** | `TRITON_CACHE_AUTOTUNING=1` **and** `TRITON_CACHE_DIR=<shared path>` | Every rank reading one warm cache. Autotuning still runs and still picks fast configs, but a rank that misses the cache re-times the selection on its own and can pick differently. |

Cached autotuning is opt-in because its determinism is conditional: the pinned default holds by construction, the cached path holds only while the shared cache does. Setting `TRITON_CACHE_AUTOTUNING=1` without `TRITON_CACHE_DIR` is rejected — unset, Triton falls back to a node-local directory, which is exactly the case the cache is meant to prevent.

## Verifying kernel-config agreement

Applies to cached autotuning; the pinned default has nothing to compare. `TRITON_PRINT_AUTOTUNING=1` makes each rank log the config it selects per kernel; group those lines by kernel and key across the per-rank logs, and every group should hold exactly one distinct config.

Note the limit: a rank only logs when it *tunes*, so a run where some ranks hit the cache and others miss cannot be compared this way — the hitting ranks log nothing.

## Verifying determinism

The bit-exact correctness suite lives at `tests/unit_tests/determinism/correctness/`. It parametrizes over model presets (GPT-like, Llama-like, Hybrid/Mamba) × parallelism cells (TP, PP, VPP, EP, FSDP, and composites) and asserts that two runs of the same configuration produce bit-identical outputs and gradients. FP8 / FP4 recipes (`tensorwise`, `delayed`, `mxfp8`, `nvfp4`) are covered by `tests/unit_tests/determinism/correctness/test_fp8_determinism.py`; the Blackwell-only recipes are capability-skipped on Hopper.

The cost of `--deterministic-mode` is measured outside pytest by an nsys-driven per-NVTX-range breakdown: `tests/performance_tests/shell_test_utils/determinism/run_nsys_breakdown.sh` wraps any training entry point (e.g. `pretrain_hybrid.py --profile`) under nsys for a det-vs-nondet comparison, and `tests/performance_tests/shell_test_utils/determinism/print_nsys_leaderboard.py` joins the two CSVs into a side-by-side table. The CI invocation lives at `tests/test_utils/recipes/h100/determinism-perf.yaml`.
