<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Deterministic Training

Deterministic training aims to reproduce outputs and training state for the same inputs, configuration, software stack and hardware topology. Startup settings select supported deterministic paths; replay tests must verify the particular recipe and execution environment.

Pass `--deterministic-mode` to a supported Megatron pretraining entry point (e.g. `pretrain_hybrid.py`):

```bash
python pretrain_hybrid.py \
  --deterministic-mode \
  <other args ...>
```

GPT, Hybrid, VLM, BERT, T5, MIMO and elastification pretraining scripts bootstrap `megatron.determinism.configure_determinism` before heavy imports. The deprecated Mamba entrypoint delegates to Hybrid. YAML uses the effective `deterministic_mode` from the root, `model_parallel`, then `language_model` settings, with later sections taking precedence, as in full YAML validation. Full CLI/YAML validation rechecks the resolved config and logs policy on rank zero. A config cannot silently disable deterministic mode after the process opted in.

For custom scripts or module entrypoints, use the explicit early launcher:

```bash
python -m megatron.determinism train.py <args ...>
python -m megatron.determinism -m my_package.train <args ...>
# With one fresh worker per GPU:
torchrun --nproc-per-node=8 --module megatron.determinism train.py <args ...>
```

The launcher establishes process policy before importing the target. It does not rewrite the model config: enable `deterministic_mode` in that config too. CLI callers retain the compatibility entrypoint `megatron.training.determinism.apply_determinism_to_args` for final validation.

## Library startup

`ModelParallelConfig.deterministic_mode=True` selects deterministic implementation paths, but does not configure process-wide settings. MCore library callers, including integrations built on Megatron Bridge, can use the public startup API:

```python
from megatron.determinism import configure_determinism

# Configure before importing megatron.core or megatron.bridge: their GPU
# dependencies can query CUDA during import. Use the final recipe options.
options = {
    "deterministic_mode": True,
    "cross_entropy_loss_fusion": False,
    "tp_comm_overlap": False,
    "moe_router_fusion": True,
    "moe_router_aux_loss_fusion": False,
}
policy = configure_determinism(options)

# Import Core/Bridge next, then initialize devices/process groups and seed
# all required RNGs. Construct the model with the same options and groups.
```

The API also accepts a `ModelParallelConfig`, `TransformerConfig`, or argparse Namespace when constructing that object does not initialize CUDA. `validate_determinism_config(config)` performs only the config checks and does not change the object or process. Passing a dictionary does not transfer its options to the model: the caller must use the same effective values when constructing it.

Call the startup API before importing Core or Bridge, CUDA initialization, process-group creation, and backend first use. The lightweight `megatron.determinism` package does not import the Core GPU stack. The legacy `megatron.core.determinism` path re-exports the same functions and process state, but importing its parent can already initialize CUDA. Use the early path for first setup. A first call after PyTorch CUDA or distributed initialization raises `RuntimeError`; restart the process and configure it earlier. Repeated calls after initialization are allowed only if this process already applied the policy and its tracked environment and Torch settings are unchanged. The check cannot detect CUDA contexts created outside PyTorch, or every library that caches environment variables during import. Put environment settings in the launcher before such imports.

Startup enables `torch.use_deterministic_algorithms(True, warn_only=False)`, sets `torch.backends.cudnn.deterministic=True`, and disables cuDNN benchmarking. This removes benchmark-driven algorithm selection and asks PyTorch to reject operations without a supported deterministic implementation. See [PyTorch's reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html). These settings can affect performance and need measurement on the target recipe.

The returned dictionary is JSON-serializable and is also logged at INFO. It records the validated options, tracked environment (including Triton block overrides), Torch version and effective Torch/cuDNN settings. It is a settings record, not replay evidence. In particular, a cache path does not establish cache-content agreement, and selecting `Ring` does not pin NCCL's physical reduction order across allocations.

The API does not seed RNGs. The caller still owns Python, NumPy, Torch CPU/CUDA and model-parallel RNG state, data order, precision state, optimizer/scheduler state and checkpoint restore order. Megatron training retains its existing seed initialization. Validate independent runs and checkpoint resume separately; this API alone does not establish full-state equality. Bridge recipes need the same early call or launcher before Bridge imports, followed by validation of their resolved model options.

## Environment variables

Each variable may be set by the launcher or left unset. Startup validates all supplied values before changing the environment, then fills missing defaults. Invalid values raise explicit exceptions even under `python -O`. The legacy `apply_determinism_env(env)` helper only validates/fills the supplied mapping; it does not enforce startup timing or enable Torch determinism. Unlike the complete startup API, that helper leaves unset SSM flags to follow Torch at runtime.

| Variable | Accepted values (or unset) | Default filled if unset | Reason |
|---|---|---|---|
| `NCCL_ALGO` | subset of `{Ring, CollnetDirect, CollnetChain, ^NVLS}` | `Ring` | Retains the existing default; actual reduction order still depends on topology and communicator construction |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | `0` | `0` | Forces Transformer Engine to use deterministic algorithms |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` or `:16:8` | `:4096:8` | Deterministic cuBLAS workspace (both sizes are reproducible per NVIDIA docs; `:4096:8` is faster, `:16:8` uses less memory) |
| `TRITON_CACHE_AUTOTUNING` | `0` or `1` | *(none — opt-in)* | Persists each Triton autotune winner so every rank reuses one choice instead of re-timing it. Left unset, deterministic mode instead pins the cheapest config, which needs no cache — see [Triton autotuning](#triton-autotuning) |
| `TRITON_CACHE_DIR` | any shared-filesystem path | *(none — required only with `TRITON_CACHE_AUTOTUNING=1`)* | No safe default exists: unset, Triton uses a node-local directory and each node autotunes on its own. Required rather than filled in |
| `TRITON_PRINT_AUTOTUNING` | `1` | *(none — recommended, not set)* | Logs the config each rank selected. Changes no numerics, so it is recommended rather than forced; when `TRITON_CACHE_AUTOTUNING=1` and this is unset, a startup line reminds you. See [Verifying kernel-config agreement](#verifying-kernel-config-agreement) |
| `MAMBA_DETERMINISTIC` | empty or any string starting with `'1'` | `1` | Complete startup pins the flag before backend use; an explicit empty value retains the existing auto-detection behavior |
| `CAUSAL_CONV1D_DETERMINISTIC` | empty or any string starting with `'1'` | `1` | Complete startup pins the flag; the supported causal_conv1d library follows Torch for an explicit empty value |

If you override `NCCL_ALGO`, the value must be a subset of `{Ring, CollnetDirect, CollnetChain, ^NVLS}`. `Tree` is intentionally excluded: its intra-node chain reduction order is not user-controllable, and the inter-node tree topology can vary across runs without a pinned topology file, so it cannot be vouched for as bit-exact across stacks. `^NVLS` is accepted (banning NVLS is a legitimate user choice on hardware that exposes it); the user is responsible for ensuring whatever NCCL falls back to is deterministic on their environment.

## Config requirements

Checked by the shared API against the supplied config, mapping or Namespace. Incompatible options are rejected with an explicit error rather than silently flipped off — you must disable them yourself so the run matches the config you asked for:

| Flag | Behavior under `--deterministic-mode` |
|---|---|
| `--cross-entropy-loss-fusion` | Must be off — asserted (fused CE is non-deterministic); drop the flag yourself |
| `--tp-comm-overlap` | Must be off — asserted (the overlap path is not bit-exact); drop the flag yourself |
| `moe_router_aux_loss_fusion` | Must be off — asserted (TE's fused aux-loss kernel is non-deterministic); follows `moe_router_fusion` when unset |
| `torch.use_deterministic_algorithms` | Set to `True`, with `warn_only=False` |
| `torch.utils.deterministic.fill_uninitialized_memory` | Set to `False` — see below |
| `torch.backends.cudnn.deterministic` | Set to `True` |
| `torch.backends.cudnn.benchmark` | Set to `False` |

Flash attention is permitted: Transformer Engine's flash-attention backend is deterministic when `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` (see the [Transformer Engine docs](https://docs.nvidia.com/deeplearning/transformer-engine/api/pytorch.html)).

## Uninitialized-memory fill

Determinism costs an *independent* output buffer: a reduction that would otherwise accumulate into shared memory with unordered atomics writes into its own buffer instead, fixing the summation order run to run. That is what makes training reproducible, and `--deterministic-mode` keeps it.

`torch.use_deterministic_algorithms(True)` also switches on a separate knob, `torch.utils.deterministic.fill_uninitialized_memory`, which fills every uninitialized allocation — `torch.empty`, `empty_like`, `empty_strided`, `Tensor.resize_` — with NaN or MAX_INT, so a kernel *reading* memory it never wrote reads the same bytes every run. Reproducibility does not need that, and it is not free: one extra fill kernel per empty allocation, serialized between real work, which suppresses the overlap between compute and communication and so costs far more wall time than GPU time. Clearing it is worth roughly **15% TFLOP/s** on large configs, and more the more `torch.empty` calls a step makes. The shared `configure_determinism` startup API clears it after validation and enabling deterministic algorithms; `apply_determinism_to_args` delegates to that API.

Padding matters only if a computation reads it. **Benign — nothing reads it:** computed values are bit-identical and only saved bytes differ. Checkpoints are the example — some saved tensors carry trailing pad slots that no kernel writes, so two runs of the same configuration write files differing in those bytes while every trained value matches. **Harmful — a computation consumes it:** a reduction over a padded tail, a GEMM with a rounded-up K, an unmasked attention region. Results then differ run to run, and the fill does not make them correct, only repeatably wrong — every run reads the same NaN instead of different garbage. Fix the read; re-enabling the fill hides it.

After the last startup/configuration call, set the fill back to `True` while hunting such a read — turning garbage into a loud NaN is the one thing it is good for:

```python
import torch.utils.deterministic
torch.utils.deterministic.fill_uninitialized_memory = True
```

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
