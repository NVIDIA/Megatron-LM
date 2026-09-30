# Triton autotune configuration policy

Independent timing-based autotuning can choose different tilings on different
ranks or cold runs. Those tilings can change floating-point reduction order.
Pinning chooses a configuration without benchmarking; it does not establish
determinism inside the kernel or across the whole training job.

## Configure through Python or training arguments

`AutotunePolicy` is an immutable configuration object. It holds user-supplied
settings such as selection mode, module scope, table paths, and verification
cadence. Library callers pass these settings through `TransformerConfig`:

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

`TransformerConfig.triton_autotune` exposes kernel-selection settings alongside
`deterministic_mode` to library callers, including applications that do not use
Megatron's training initializer. Constructing the config applies these settings
before the model executes its kernels.

Installation reads the settings and resolves an omitted mode into a separate
policy value. It does not modify the supplied `AutotunePolicy` or store results
in `TransformerConfig`. The runtime adapter in `interception.py` separately
owns loaded tables, selected-config caches, recorded winners, and diagnostics.
For example, `table_path` specifies where to read a table; the table's entries
are loaded into runtime state when a pinned kernel first executes.

Standalone kernel callers can instead call `install(AutotunePolicy(...))` before
launching kernels. An explicit `install(policy)` takes precedence over the
policies framework configuration supplies, and `install(None)` withdraws it.
Framework initialization calls `install_from_config()` from
`initialize_megatron` and at the end of `TransformerConfig.__post_init__`, so a
configuration rejected by its own validation leaves the process-wide policy
unchanged; a mapping is converted to `AutotunePolicy` and any other type is
rejected first. A policy that fails to install, such as one whose recording path
is not writable, also leaves the previous policy in place. Installation does not
query CUDA. Architecture tables load on the first pinned kernel invocation.

The policy is process-wide. A later component without a policy preserves the
configured policy, and its default `deterministic_mode=False` does not undo an
earlier request for pinning. A policy that omits `mode`, including an explicitly
installed one, keeps deriving it: a deterministic request made before or after
installation selects `pinned`. An explicitly set mode takes precedence.
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

On the first pinned invocation for an autotuner, architecture, and tuning key,
the adapter applies the kernel's pruning rules and selects from the remaining
live candidates:

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
`megatron.core`, including submodules, so every in-tree Triton autotuner (SSM
ops and fusions such as the MLA RoPE and mHC kernels) is covered, including those
imported before model configuration. Override `modules` to include another
package; this replaces the default list.

Some covered kernels produce the same values whatever configuration they run.
`config_invariant` lists them by qualified name (`module.function`); they keep
Triton's timed choice under pinning, since pinning would only cost throughput,
and their choices are not logged or compared. By default it lists only pure data
movement, Transformer Engine's MoE permutation kernels: even elementwise
arithmetic can round differently between configurations, because the layout
decides whether values are computed packed or promoted and whether multiplies
and adds are fused. A GPU test forces every candidate of each default entry and
checks for bit-identical outputs. Covered reductions, such as the MLA RoPE and
mHC backward kernels, fall back to the cheapest candidate without a table; record
one to recover their throughput.

In-tree SSM kernels also use `autotune_configs()` at decoration time. That helper
uses the explicit deterministic-mode setter or PyTorch's deterministic flag and,
if already installed, the policy's block sizes. An import-time singleton cannot
later recover its discarded candidates.

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
`MAMBA_DETERMINISTIC` does not select `pinned`: it only controls the external
`mamba_ssm` package, and the adapter logs a notice when it is set while the
policy resolves to `auto`.

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
reduce candidates at import time; recording cannot restore them, and autotuners
left with a single candidate are not recorded, since nothing was timed. Winners
are recorded under the qualified kernel name (`module.function`), so kernels
that share a function name in different packages keep separate entries; lookup
also accepts the bare function name used by older tables.

`record_path` is a file prefix: `~` is expanded, and a path ending in a
separator is rejected. Its directory is created and checked for write access
when the policy is installed. Each rank writes `<record_path>.rank<N>.json`, with
the rank read from the process group while recording, on normal process exit;
the file is replaced atomically, and a failed write is logged. Abnormal
termination can lose captures. Merge in the recording environment so recorded
package versions describe that environment.

Tables are named for their architecture, such as `sm100.json` or `sm103.json`.
The first matching file in `table_path` wins; packaged files are searched last.
Missing directories and files whose recorded `arch` differs from their name are
skipped with a warning. Files do not overlay one another, so preserve existing
entries when extending a table. The merge command combines raw per-rank captures
and overwrites its output; it rejects table files, truncated captures, and an
`--arch` the captures do not contain, naming the offending input.

Entries store `kwargs`, `num_warps`, `num_stages`, `num_ctas`, `maxnreg`, and
`ir_override`. Lookup matches these against live candidates, preserving their
hooks; it cannot introduce new candidate configurations. Unmatched entries
follow `on_miss`. Majority vote resolves ties by serialized configuration and
does not establish global performance optimality. A version mismatch warns;
bundled tables have empty version metadata and provide no compatibility proof.

## Diagnostics and configuration reference

The adapter logs a choice once, when it is made: the first pinned selection
for an autotuner, architecture, and tuning key, or a new timed winner of a
covered kernel. Steady-state launches do no bookkeeping. Kernels outside the
scope and config-invariant kernels are not logged.

`verify_choices(group=None)` exchanges the choices each rank made since the
previous check and compares them, for matching architecture, qualified kernel
name, and tuning key, with each other and with those already agreed on. Once
every kernel and shape has been seen, a check moves an empty payload. Ranks that
did not execute a key are excluded from that comparison. Agreement does not prove
equal coverage, cross-run repeatability, or numerical equality.

Call verification where all group members participate, such as a step boundary.
Megatron training calls `maybe_verify_choices(iteration)` every step;
`verify_every=N` enables checks every N steps. A disagreement is logged once, on
the group's first rank; `verify_strict=True` raises on every rank instead.
Enumeration logs, on each rank, every multi-config autotuner reached and whether
it is pinned, timed because it is config-invariant, or outside the scope. The
adapter reports through the `logging` module rather than `warnings`, so launcher
warning filters do not hide it. Chaos mode deliberately makes ranks choose
different configs; use it only as a diagnostic in pinned mode.

| `AutotunePolicy` field | Training argument |
|---|---|
| `mode` | `--triton-autotune-mode {auto,pinned,record}` |
| `modules` | `--triton-autotune-modules mamba_ssm transformer_engine megatron.core my_package` |
| `config_invariant` | `--triton-autotune-config-invariant pkg.module.kernel ...` (no names: pin everything) |
| `table_path` | `--triton-autotune-table-path /tables/first /tables/second` |
| `record_path` | `--triton-autotune-record-path /tmp/rec` |
| `on_miss` | `--triton-autotune-on-miss {min_cost,error}` |
| `block_sizes` | `--triton-autotune-block-sizes BLOCK_C=512 BLOCK_S=1` |
| `verify_every` | `--triton-autotune-verify-every 10` |
| `verify_strict` | `--triton-autotune-verify-strict` |
| `enumerate_autotuners` | `--triton-autotune-enumerate` |
| `chaos` | `--triton-autotune-chaos` |

Configure this policy through Python, training arguments, or YAML. In YAML, a
`null` value means the default, and a mistyped value (such as a quoted `"false"`
for a boolean) or an unknown key is rejected. External libraries may have their
own environment controls. Per-rank recordings and the chaos diagnostic use the
process-group rank, falling back to the launcher's `RANK` or `SLURM_PROCID`.

## Upstream path

Package-owned selection/pruning hooks are preferable to a process-wide adapter.
Triton's `prune_configs_by` supplies a hook, but older versions still benchmark
a sole surviving candidate. Until supported dependencies consistently skip that
benchmark and expose policy hooks, `interception.py` provides the adapter.
