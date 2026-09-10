# Triton autotune policy

Triton's autotuner normally selects configurations by benchmarking. Independent
processes can choose different configurations for the same inputs. For kernels
whose tiling changes floating-point accumulation, this can change their results.
Caching the measured winner does not make independent cold runs choose alike.

This package can select one configuration without benchmarking. It controls
configuration selection; it does not make an otherwise nondeterministic kernel
or an entire training run deterministic.

## Selection and integration

For kernels in scope, `pinned` mode first runs the kernel's existing Triton
pruning rules, then selects from the remaining candidates:

1. A matching table entry for the current GPU architecture, kernel and tuning key.
2. A matching `TRITON_AUTOTUNE_BLOCK_*` fallback override.
3. The smallest estimated block/stage cost, with a warning, unless `on_miss=error`.

The selected **live** `triton.Config` object retains its launch hook. The adapter
temporarily supplies a one-entry candidate list to the original `Autotuner.run`,
then restores both the list and argument state, including when the launch fails.
This skips benchmarking and timing-cache lookup even on older Triton versions
that still benchmark a singleton returned by `early_config_prune`.

The static fallback is reproducible for the same candidates and inputs. Its cost
estimate is not a throughput model, and it cannot predict every compile-time
resource failure. An invalid selection raises; it never retries with timing.

Framework initialization calls `install_from_env()` from
`initialize_megatron` and `TransformerConfig.__post_init__`. The latter also
passes `TransformerConfig.deterministic_mode`. Installation does not query CUDA;
tables load on the first pinned kernel invocation for the current device.
Once a model requests pinning, a later component's default configuration does
not disable it. An explicit `MCORE_AUTOTUNE_MODE` still overrides that default.

For explicit control:

```python
from megatron.core.tuning import AutotunePolicy, install

install(AutotunePolicy(mode="pinned", modules=("mamba_ssm", "transformer_engine")))
```

The policy is process-wide. Configure it during initialization, before launching
kernels. An explicit `install(policy)` takes precedence over subsequent framework
initialization calls. Reinstalling a policy reuses the adapter, flushes pending
recordings, and starts fresh diagnostics. The adapter does not add thread-safety
to Triton's mutable autotuner state.

By default, the adapter covers `mamba_ssm` and `transformer_engine` and their
submodules. Other packages retain their normal selection. Override the scope
with `MCORE_AUTOTUNE_MODULES`. Megatron's in-tree SSM kernels separately use the
legacy `autotune_configs()` decoration-time helper; their candidate lists are
reduced when deterministic mode is already enabled at import time.

## Modes and precedence

| Mode | Behavior |
|---|---|
| `auto` | Triton chooses normally; optional diagnostics observe its choices. |
| `pinned` | Select one valid candidate without timing. |
| `record` | Triton chooses normally and the adapter records the winners. |

An explicit `MCORE_AUTOTUNE_MODE` wins. Otherwise, a recording path selects
`record`; otherwise, deterministic mode selects `pinned`. The default is `auto`.
A recording run can therefore benchmark even when deterministic algorithms are
otherwise enabled. Invalid modes and a recording mode without a path raise.

## Recording and using a table

```bash
# Record the actual workload. The variable is a file prefix, not a directory.
MCORE_AUTOTUNE_RECORD=/tmp/rec torchrun ... pretrain.py ...

# Merge rec.rank0.json, rec.rank1.json, etc. by majority vote.
python -m megatron.core.tuning merge /tmp/rec.rank*.json -o ~/.mcore/tuning/sm103.json

# Pin using the recorded table, also outside deterministic mode.
MCORE_AUTOTUNE_MODE=pinned MCORE_AUTOTUNE_TABLE_PATH=~/.mcore/tuning torchrun ... pretrain.py ...

# Inspect disagreements before merging.
python -m megatron.core.tuning report /tmp/rec.rank*.json
```

Record on the target architecture and with the intended package versions and
workload. Majority vote resolves ties by serialized configuration. It selects
the most frequently observed winner; it does not measure a globally optimal
configuration. Captures are written at normal process exit, so abnormal
termination can lose them.

Some external packages reduce their candidate lists at import time. With the
current `mamba_ssm` helper, `TRITON_CACHE_AUTOTUNING=1` preserves those candidates
when `MAMBA_DETERMINISTIC=1`. The adapter can record a singleton, but cannot recover
candidates that an external decorator already discarded.

Tables are JSON files named for their architecture, such as `sm100.json` or
`sm103.json`. User directories take precedence over bundled files. Each entry
contains `kwargs`, `num_warps`, `num_stages`, `num_ctas`, `maxnreg`, and
`ir_override`; legacy entries use defaults for the last three fields. Lookup
matches all these options against valid live candidates instead of constructing
a new configuration. An unmatched entry falls back according to the policy.

Version metadata describes the environment that writes the merged table. Merge
inside the recording environment if those versions are to describe the recording.
The bundled tables have empty version fields, so they cannot establish package
compatibility or performance provenance. A version mismatch warns; candidate
membership is still checked at use time.

## Comparing choices across ranks

`verify_choices(group=None)` compares each rank's most recently observed config
for a `(architecture, qualified kernel name, tuning key)`. Ranks that did not
execute a key are excluded from that key's comparison: pipeline stages and
expert ranks need not run the same kernels or shapes. Agreement does not prove
equal kernel coverage, cross-run repeatability, or numerical equality.

Call the check where every member of the group participates, such as a step
boundary. Calling it from individual kernels can deadlock. Megatron training
calls `maybe_verify_choices(iteration)` every step; `MCORE_AUTOTUNE_VERIFY=N`
enables a check every `N` steps. Other callers can pass an explicit process group.

The check warns on conflicting observed configurations, or raises with
`MCORE_AUTOTUNE_VERIFY_STRICT=1`. Enumeration reports which multi-config kernels
actually execute and whether the active policy pins them. Chaos mode chooses
rank-dependent configurations as a diagnostic positive control; it requires
`pinned` mode and can make results differ deliberately.

## Environment variables

| Variable | Meaning |
|---|---|
| `MCORE_AUTOTUNE_MODE` | `auto`, `pinned`, or `record`. |
| `MCORE_AUTOTUNE_MODULES` | Comma-separated package/module prefixes. |
| `MCORE_AUTOTUNE_TABLE_PATH` | Search directories, separated by the platform path separator; `~` expands. |
| `MCORE_AUTOTUNE_RECORD` | File prefix for per-rank recordings. |
| `MCORE_AUTOTUNE_ON_MISS` | `min_cost` (default), or `error` if no table/override matches. |
| `MCORE_AUTOTUNE_VERIFY` | Check cadence in steps; `0` disables. |
| `MCORE_AUTOTUNE_VERIFY_STRICT` | `1` raises on disagreement. |
| `MCORE_AUTOTUNE_ENUMERATE` | `1` reports multi-config kernels as they execute. |
| `MCORE_AUTOTUNE_CHAOS` | `1` enables rank-dependent choices in pinned mode. |
| `TRITON_AUTOTUNE_BLOCK_*` | Fallback kernel-kwarg selection when the table misses. |

The earlier `DET_AUTOTUNE_*` and `MCORE_DET_TUNE_RECORD` names remain accepted.

## Upstream path

A supported selection/pruning hook in the owning packages is preferable to a
process-wide adapter. Triton's `prune_configs_by` API provides a place for this
policy, but older versions still benchmark the one surviving candidate. Until
the supported dependency range consistently skips that benchmark and external
packages expose the policy, `interception.py` contains the compatibility shim.
