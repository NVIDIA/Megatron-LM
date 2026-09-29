# Triton autotune configuration policy

Independent timing-based autotuning can choose different tilings on different
ranks or cold runs. Those tilings can change floating-point reduction order.
Pinning chooses a configuration without benchmarking; it does not establish
determinism inside the kernel or across the whole training job.

## Configure through Python or training arguments

The policy is a first-class model configuration. Library callers use:

```python
from megatron.core.transformer import TransformerConfig
from megatron.core.tuning import AutotunePolicy

config = TransformerConfig(
    num_layers=2,
    hidden_size=128,
    num_attention_heads=4,
    deterministic_mode=True,
    triton_autotune=AutotunePolicy(
        table_path=("/path/to/tables",),
        on_miss="error",
        verify_every=10,
        verify_strict=True,
    ),
)
```

Standalone kernel callers can instead call `install(AutotunePolicy(...))` before
launching kernels. An explicit `install(policy)` takes precedence over framework
initialization. Framework initialization calls `install_from_config()` from
`initialize_megatron` and `TransformerConfig.__post_init__`; installation does
not query CUDA. Architecture tables load on the first pinned kernel invocation.

The policy is process-wide. A later component without a policy preserves the
configured policy, and its default `deterministic_mode=False` does not undo an
earlier request for pinning. An explicitly configured mode takes precedence.
Changing the effective policy clears selected configurations, tables, and
diagnostics. The adapter does not add thread-safety to Triton's mutable state.

Training accepts the corresponding `--triton-autotune-*` arguments. Legacy
`--yaml-cfg` users can put the policy under `language_model.triton_autotune`:

```yaml
language_model:
  deterministic_mode: true
  triton_autotune:
    table_path: ["/path/to/tables"]
    on_miss: error
    verify_every: 10
    verify_strict: true
```

Autotune CLI arguments cannot be combined with `--yaml-cfg`; put the options in
the YAML policy instead. Model configuration retains the nested policy for
normal configuration serialization.

## Selection and scope

For each pinned invocation, the adapter first applies the kernel's pruning rules
and selects from the remaining live candidates:

1. A matching architecture-table entry.
2. A matching `block_sizes` configuration override.
3. The lowest static-cost candidate, or an error when `on_miss="error"`.

The adapter temporarily supplies a singleton candidate list to the original
`Autotuner.run`, then restores the list and argument state, even after failure.
This skips timing and Triton's tuning-cache lookup, including versions that
benchmark a singleton returned by `early_config_prune`. Live config hooks are
preserved.

After a successful launch, the selected config is cached per autotuner, GPU
architecture, and tuning key (including tensor dtypes). Repeated calls skip
pruning and selection; launch hooks still run every time. Inputs that affect
pruning must be represented in the kernel's tuning key. Changing `block_sizes`
requires installing a new policy, which invalidates cached selections.

The static fallback is reproducible for identical candidates and inputs. Its
cost estimate is not a throughput model and cannot predict every compile-time
resource failure. Invalid selections raise; they never retry with timing.

By default the adapter covers `mamba_ssm`, `transformer_engine`, and
`megatron.core.ssm.ops`, including submodules. This also covers in-tree SSM
kernels imported before model configuration. Override `modules` to include
another package; this replaces the default list. In-tree SSM kernels also use
`autotune_configs()` at decoration time. That helper uses the explicit deterministic-mode setter or
PyTorch's deterministic flag and, if already installed, the policy's block sizes.
An import-time singleton cannot later recover its discarded candidates.

## Modes and precedence

| Mode | Behavior |
|---|---|
| `None` (default) | Derive from recording and determinism settings. |
| `auto` | Triton chooses normally; optional diagnostics observe its choices. |
| `pinned` | Select one valid candidate without timing. |
| `record` | Triton chooses normally and the adapter records winners. |

An explicit mode wins. Otherwise, `record_path` selects `record`; otherwise,
model or PyTorch deterministic mode selects `pinned`; ordinary execution uses
`auto`. Recording intentionally permits benchmarking and requires a file prefix.

## Recording and using a table

```bash
# Record representative workload shapes; the recording path is a file prefix.
uv run python -m torch.distributed.run ... pretrain_gpt.py ... \
    --triton-autotune-mode record --triton-autotune-record-path /tmp/rec

# Inspect disagreements and combine per-rank captures by majority vote.
uv run python -m megatron.core.tuning report /tmp/rec.rank*.json
uv run python -m megatron.core.tuning merge /tmp/rec.rank*.json \
    -o ~/.mcore/tuning/sm103.json

# Use the recorded selections without benchmarking.
uv run python -m torch.distributed.run ... pretrain_gpt.py ... \
    --triton-autotune-mode pinned --triton-autotune-table-path ~/.mcore/tuning
```

Record on the target architecture with the intended package versions and
workload, keeping the kernel's candidates available. Some external libraries
reduce candidates at import time; recording cannot restore them. Captures are
written on normal process exit, so abnormal termination can lose them. Merge in
the recording environment so recorded package versions describe that environment.

Tables are named for their architecture, such as `sm100.json` or `sm103.json`.
The first matching file in `table_path` wins; packaged files are searched last.
Files do not overlay one another, so preserve existing entries when extending
a table. The merge command combines raw captures and overwrites its output.

Entries store `kwargs`, `num_warps`, `num_stages`, `num_ctas`, `maxnreg`, and
`ir_override`. Lookup matches these against live candidates, preserving their
hooks; it cannot introduce new candidate configurations. Unmatched entries
follow `on_miss`. Majority vote resolves ties by serialized configuration and
does not establish global performance optimality. A version mismatch warns;
bundled tables have empty version metadata and provide no compatibility proof.

## Diagnostics and configuration reference

`verify_choices(group=None)` compares each rank's most recently observed config
for matching architecture, qualified kernel name, and tuning key. Ranks that did
not execute a key are excluded from that comparison. Agreement does not prove
equal coverage, cross-run repeatability, or numerical equality.

Call verification where all group members participate, such as a step boundary.
Megatron training calls `maybe_verify_choices(iteration)` every step;
`verify_every=N` enables checks every N steps. `verify_strict=True` raises on
disagreement. Enumeration reports executing multi-config autotuners and whether
they are pinned. Chaos mode deliberately makes ranks choose different configs;
use it only as a diagnostic in pinned mode.

| `AutotunePolicy` field | Training argument |
|---|---|
| `mode` | `--triton-autotune-mode {auto,pinned,record}` |
| `modules` | `--triton-autotune-modules mamba_ssm transformer_engine megatron.core.ssm.ops my_package` |
| `table_path` | `--triton-autotune-table-path /tables/first /tables/second` |
| `record_path` | `--triton-autotune-record-path /tmp/rec` |
| `on_miss` | `--triton-autotune-on-miss {min_cost,error}` |
| `block_sizes` | `--triton-autotune-block-sizes BLOCK_C=512 BLOCK_S=1` |
| `verify_every` | `--triton-autotune-verify-every 10` |
| `verify_strict` | `--triton-autotune-verify-strict` |
| `enumerate_autotuners` | `--triton-autotune-enumerate` |
| `chaos` | `--triton-autotune-chaos` |

The earlier tuning environment variables (`MCORE_AUTOTUNE_*`, `DET_AUTOTUNE_*`,
`MCORE_DET_TUNE_RECORD`, and `TRITON_AUTOTUNE_BLOCK_*`) are no longer read. The
policy also does not read `MAMBA_DETERMINISTIC`; external libraries may still
have their own environment controls. Distributed launcher rank metadata remains
used for per-rank recordings and diagnostics.

## Upstream path

Package-owned selection/pruning hooks are preferable to a process-wide adapter.
Triton's `prune_configs_by` supplies a hook, but older versions still benchmark
a sole surviving candidate. Until supported dependencies consistently skip that
benchmark and expose policy hooks, `interception.py` provides the adapter.
