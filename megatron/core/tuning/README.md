# Triton Autotune Policy

Triton picks a kernel config by timing its candidates at first call. The winner
is a property of the machine at that instant, and it fixes `BLOCK_SIZE`,
`num_warps` and `num_stages` — which is to say it fixes how a reduction is
tiled, and so the order floating-point values are accumulated in. Addition is
not associative in floating point, so two identical runs that pick differently
produce different numbers.

That has two costs. Runs stop being reproducible, and every cold process pays
for a benchmark whose result is noisy enough that ranks disagree with each
other. Recording the winners across 96 GPUs of one run, on identical shapes and
identical hardware, six of sixteen kernel/shape entries disagreed; one kernel
produced seven distinct winners.

This package fixes the choice.

## Pinning

Pinning replaces the candidate list with exactly one entry, chosen by a rule
that never reads a clock, before Triton can benchmark:

```python
candidates = self.configs
try:
    self.configs = [chosen]          # Triton now sees one option
    return original_run(self, *args, **kwargs)
finally:
    self.configs = candidates        # restore: the choice is per shape
```

Triton gates benchmarking on `len(self.configs) > 1`, so a single-entry list
skips the timing loop, the disk cache and `do_bench` entirely. The list is
restored afterwards because a later call with a different shape must still see
every candidate.

**What makes it deterministic is the rule, not which config it picks.** The
selection is a pure function of the candidate list and the call signature, so
every rank computes the same answer:

1. the tuned table entry for this kernel, shape and architecture
2. an explicit `TRITON_AUTOTUNE_BLOCK_*` override
3. otherwise the cheapest config by static estimate, with a warning

Step 3 is exactly as reproducible as step 1. The table only decides whether the
fixed choice is also the fast one.

Pinning is useful outside determinism too: it removes autotuning from startup
and makes benchmarks repeatable.

## Modes

| Mode | Behaviour |
|---|---|
| `auto` | Leave Triton alone. |
| `pinned` | Replace the candidate list with one timing-free choice. |
| `record` | Let Triton benchmark as usual and capture the winners. **Not reproducible** — its output is the table. |

Deterministic mode implies `pinned`. An explicit `MCORE_AUTOTUNE_MODE` wins over
that, so a determinism run can still be put into `record`.

## Pretuning: recording a table

The cheapest-config fallback is deterministic but slower than the config the
autotuner would have chosen. A tuned table closes that gap. Three steps, no
source edit and no rebuild:

```bash
# 1. Record. A normal run of the workload whose shapes you want covered.
MCORE_AUTOTUNE_RECORD=/tmp/rec  torchrun ... pretrain.py ...

# 2. Merge the per-rank captures into one table file.
python -m megatron.core.tuning merge /tmp/rec/*.json -o ~/.mcore/tuning/sm103.json

# 3. Use it. User paths are searched before the packaged defaults.
MCORE_AUTOTUNE_TABLE_PATH=~/.mcore/tuning  torchrun ... pretrain.py ...
```

Inspect a recording before trusting it:

```bash
python -m megatron.core.tuning report /tmp/rec/*.json
```

Ranks routinely disagree about the winner — that disagreement *is* the variance
the table removes — so the merge takes a **majority vote**, breaking ties on the
serialized config so a given set of recordings always produces the same table.
The report names the entries that disagreed.

Record on the architecture you will run on. `sm100` (GB200) and `sm103` (GB300)
are different tables; a miss on an unrecorded architecture is not an error, it
just falls back to the cheapest config and warns.

### Recording gotcha

Set `TRITON_CACHE_AUTOTUNING=1` for the recording run if an installed package
ships its own import-time pin. `mamba_ssm` does, and with the flag unset it has
already reduced every config list to one entry, leaving no benchmark to record —
the run produces an empty table and no error.

## Table format

One file per architecture, carrying the provenance needed to notice staleness:

```json
{"arch": "sm103",
 "triton": "3.6.0",
 "packages": {"mamba-ssm": "2.3.1", "transformer-engine": "2.14.0"},
 "source": "nemotron_3_ultra 96-GPU recording, majority vote over ranks",
 "kernels": {"_chunk_scan_fwd_kernel": {
     "chunk_size=128|...": {"kwargs": {"BLOCK_SIZE_M": 64}, "num_warps": 4, "num_stages": 3}}}}
```

A stored entry is matched back against the kernel's **live** candidate list
rather than rebuilt. Two things follow: fields the table does not carry
(`pre_hook`, `num_ctas`, `maxnreg`) survive intact, and an entry that no longer
names a real candidate degrades to a miss instead of producing an invalid
launch. A table recorded against a different Triton warns rather than being
discarded.

## Where the policy is installed

From `TransformerConfig.__post_init__`, so every model construction is covered,
and from `initialize_megatron` for training scripts that run kernels before
building a config. Both are idempotent.

It is deliberately not installed from any one layer type. Autotuning is not a
property of the model: Mamba's SSD kernels and Transformer Engine's MoE
permutation kernels both select by timing, so activating from `MambaMixer` — as
an earlier version did — left every non-Mamba model unprotected. The patch
replaces a method on Triton's `Autotuner` class, so it only has to be installed
before the first kernel *call*, not before the decorators are evaluated.

## Scope

By default `mamba_ssm` and `transformer_engine`. Everything else keeps its tuned
performance. Widen or narrow with `MCORE_AUTOTUNE_MODULES`.

Two families are known to matter:

- **`mamba_ssm`** SSD kernels, reached by the Mamba memory-efficient path.
- **`transformer_engine.common.triton.permutation`** — `_unpermute_kernel` and
  `_unpermute_bwd_with_merging_probs_kernel` sum each token's top-k expert
  contributions, so the block size sets the accumulation order. Reachable
  whenever `moe_permute_fusion` is on, which the nemotronh recipes set even
  though the mcore default is off.

`fla` carries more autotuners than either, but they sit behind `GatedDeltaNet`;
add it to the scope for models that use it.

## Checking what a run did

```bash
MCORE_AUTOTUNE_ENUMERATE=1 ...    # every multi-config autotuner reached, pinned or not
MCORE_AUTOTUNE_VERIFY=1 ...       # assert all ranks chose alike, at each step boundary
```

`ENUMERATE` turns "which kernels autotune here" from a static audit into a
measurement — on Nemotron-H it reports 18, of which three were unpinned before
the scope was widened.

`verify_choices()` all-gathers a digest of each rank's `(kernel, shape) -> config`
map and names the kernels ranks disagree on. It catches the reduction-order bug
at its cause, rather than waiting for it to surface as diverging tensors many
iterations later. Call it where every rank arrives, such as a step boundary:
calling it at the moment of choice would deadlock, since ranks reach a given
kernel at different times.

`MCORE_AUTOTUNE_VERIFY=N` is that call at a cadence. Megatron's training loop
runs `maybe_verify_choices(iteration)` every step, which checks every `N`th one
and does nothing when the variable is unset. Setting it also installs the
interception on its own, so an `auto`-mode run still logs what Triton timed —
which is the case worth checking, since it is the one that can differ. Outside
the training loop, call `maybe_verify_choices()` from your own step boundary, or
`verify_choices()` directly to check on demand; pass `group=` to verify within a
subgroup rather than the whole world.

`MCORE_AUTOTUNE_CHAOS=1` makes each rank pick a different config on purpose,
reproducibly. It is a positive control — every other check is a negative one,
and "the runs matched" cannot distinguish a working detector from a blind one.
Never use it for a real run.

## Environment variables

| Variable | Meaning |
|---|---|
| `MCORE_AUTOTUNE_MODE` | `auto`, `pinned` or `record`. Overrides the deterministic-mode default. |
| `MCORE_AUTOTUNE_MODULES` | Comma-separated module prefixes to act on. |
| `MCORE_AUTOTUNE_TABLE_PATH` | Directories searched before the packaged tables. |
| `MCORE_AUTOTUNE_RECORD` | Path prefix for a recording run's per-rank captures. |
| `MCORE_AUTOTUNE_ON_MISS` | `min_cost` (default) or `error` to refuse the fallback. |
| `MCORE_AUTOTUNE_VERIFY` | Cross-rank agreement check cadence, in steps. `0` disables. |
| `MCORE_AUTOTUNE_VERIFY_STRICT` | Raise instead of warning on disagreement. |
| `MCORE_AUTOTUNE_ENUMERATE` | Report every multi-config autotuner reached. |
| `MCORE_AUTOTUNE_CHAOS` | Per-rank divergence on purpose, as a positive control. |
| `TRITON_AUTOTUNE_BLOCK_*` | Force a specific kernel kwarg, e.g. `TRITON_AUTOTUNE_BLOCK_SIZE_M=64`. |

The earlier `DET_AUTOTUNE_*` and `MCORE_DET_TUNE_RECORD` names are still
accepted.

## Upstream

Patching `Autotuner.run` is a stopgap, confined to `interception.py` so a
supported hook replaces one file. The real fixes belong upstream: Transformer
Engine's permutation autotuners should honour
`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`, and `mamba_ssm` should raise rather than
silently discard `MAMBA_DETERMINISTIC=1` when `TRITON_CACHE_AUTOTUNING=1` is
also set. Those two flags read as complementary and are in direct conflict; the
latency one currently wins, without a word.
