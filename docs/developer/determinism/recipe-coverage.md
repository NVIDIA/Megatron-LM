---
orphan: true
---

# Recipe coverage from operation evidence

This tool answers: **which observed operation configurations have matching
replay evidence, and which are known failures or gaps?** It does not certify a
recipe. Independent training runs and checkpoint-resume verification remain
required even when every observed operation has matching evidence.

## Capture selected entrypoints

Create a bindings JSON file containing module-level functions to observe. IDs
must use the same operation and implementation contract as the replay tests:

```json
[
  {
    "target": "megatron.core.transformer.mlp:bias_gelu_impl",
    "op_id": "fused_bias_gelu",
    "implementation": "torch.compile:bias_gelu"
  }
]
```

Bind the attribute looked up at the training call site. This example observes
the dense GELU MLP. Core imports the MLP module before capture installs bindings,
so wrapping the fusion module's original attribute does not replace the MLP's
retained alias. Select the entrypoint actually used by your recipe; a gated
activation needs its own binding and matching implementation contract.

Run the original recipe through the wrapper, using the same launch environment:

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
uv run python -m torch.distributed.run --nproc-per-node 8 \
  -m tools.determinism.capture_recipe \
  --bindings /tmp/bindings.json --output /tmp/recipe-inventory \
  --recipe-id my-dense-recipe -- pretrain_gpt.py <recipe arguments>
```

Each rank writes `rank-N.json`. Use a fresh directory for every run. The wrapper
records shapes, strides, dtypes, gradient requirements, mode, invocation counts,
and source/environment provenance. It records successful forward calls and uses
output-gradient hooks to record that backward traversed a call. It neither
copies tensor contents nor adds CUDA synchronization.
Hooks recheck runtime settings. When backward uses a different policy, the
signature retains both the forward settings and explicit `backward_runtime` and
`backward_deterministic_algorithms` fields. That mixed signature needs its own
matching evidence. Multiple output hooks are counted once per distinct backward
policy for a wrapped call; a later traversal under another policy is retained.
Check every rank's observed signatures and invocation counts against the recipe.
A training run can finish with an empty inventory when a bound entrypoint is
unused or calls bypass the binding.

The capture entrypoint uses `megatron.determinism.bootstrap_training_determinism`
before importing bound modules or initializing CUDA. It honors both
`--deterministic-mode` and the effective `--yaml-cfg` policy, including YAML
precedence over CLI flags. This requires the early MCore startup API from
[#7419](https://github.com/NVIDIA/Megatron-LM/pull/7419). The training program
still validates its full configuration. Capture with determinism disabled does
not opt the recipe into deterministic mode.

Bindings are explicit because a function name alone cannot establish the
backend or numerical variant. Include every numerical option in the arguments
or give distinct variants distinct implementation IDs. A binding must cover
the same callable contract as the corresponding test, including hidden state.
Module attributes that alias `torch.autograd.Function.apply` are supported,
including bias-GELU's explicit classmethod override. The original bound class,
autograd behavior and class descriptor are preserved. Instance-bound methods
(including native methods), arbitrary classmethods and class descriptors are
rejected; their hidden state needs an explicit adapter.

## Query evidence

The consumer accepts schema-version-1 `determinism_coverage` reports from the
measured coverage producer. It is independently runnable without importing that
producer or any GPU package:

```bash
python -m tools.determinism.recipe_coverage /tmp/recipe-inventory \
  --evidence /tmp/coverage.json --output /tmp/recipe-report.json --strict
```

Source revision and the entire recorded environment must match. Shape, stride,
dtype, mode, implementation, and phase matches are exact; no shape-range or
backend equivalence is inferred. Driver versions and call-time autocast, TF32,
cuDNN settings and the memory-fill flag are part of that match. Both inventory
and replay must record `runtime.fill_uninitialized_memory` as an explicit boolean.
Missing or non-boolean values remain unverified, including when both old records
omit the field. The consumer does not infer a setting from source revision or a
current default. If backward uses another runtime policy, its fill flag must also
be explicit. Historical reports are not rewritten to the new default.
Triton cache policy/directory and all
`TRITON_AUTOTUNE_BLOCK_*` overrides are matched at startup and at each call,
including changes made after capture starts. Older evidence without these fields
does not match a new capture. A cache directory is provenance, not proof that
cache contents or selected configurations are unchanged.
New captures retain source/environment context before and after training. A
changed revision, dirty-tree status, or recorded environment prevents completion
and leaves the inventory unverified. The consumer independently rejects recorded
context drift, even if a capture claims completion. These boundary checks do not
detect a change that is reverted before the final check.
A passing signature shared across ranks requires matching protocol evidence from
every required rank. Rank-dependent signatures can also match when **one case in
one replay run, using one protocol and phase**, contains exactly one signature per
required rank and the inventory contains every signature on that same rank.
The JSON retains the complete `rank_assignment` with each match. Group membership,
local rank, input hashes and other recorded configuration fields remain part of
the exact signature; none are dropped to make a collective match.

Missing peers, changed rank assignments, ambiguous variants on a rank, or evidence
split across cases, protocols or runs cannot supply this match. A signature used
on an additional recipe rank needs evidence for that rank as well. Repeated calls
do not prove that their order or interactions match the test; independent recipe
replay remains required.

A passing forward+backward replay can supply forward evidence. A backward mismatch
is not projected into a forward failure.
Matching negative evidence takes precedence over intermittent passing evidence.

JSON and Markdown reports list matching case/run IDs, unknown configurations,
rejected stale evidence, and incomplete or truncated captures. The denominator
is **unique observed signatures**, not invocation count. Missing ranks,
interruption, dirty source, and truncation prevent verified coverage. Empty
inventories have null percentages.

Without `--strict`, the CLI writes a diagnostic report and exits 0. Strict mode
exits 1 for matching nondeterministic operations and 2 for unknown or incomplete
coverage. Even a strict exit 0 only means the observed operation inventory has
evidence; the report retains `recipe_status: replay_required`.

## Scope and limits

Only explicitly bound Python functions are observed. Previously imported
aliases, unbound functions, native calls, compiler internals, and CUDA graph
replays can bypass these wrappers. The inventory is therefore partial by
construction. Wrappers/hooks can also affect compilation and scheduling: use
the original uninstrumented recipe for final replay and performance validation.

New model shapes, backends, or environment settings remain unverified until
matching replay evidence exists. Generic bindings do not recover an explicit
process group's semantics or input hashes from opaque argument types. The
collective adapter below records those separately; it does not match old seeded
microbenchmarks by dropping their seeds, hashes or other configuration fields.

## Explicit collective capture and replay

For bounded diagnostics, a binding can select `"adapter": "tensor_parallel_collective"`.
This mode **copies tensor contents to host memory and synchronizes device work**.
Use the original uninstrumented recipe for final state replay and performance.
The default metadata-only mode retains its behavior.

Example binding at an actual call site:

```json
[
  {
    "target": "megatron.core.tensor_parallel.layers:reduce_from_tensor_model_parallel_region",
    "op_id": "tensor_parallel_mappings",
    "implementation": "mcore:reduce_from_tensor_model_parallel_region",
    "adapter": "tensor_parallel_collective"
  }
]
```

Add `--collective-capture /shared/capture --max-collective-bytes 268435456`
to the inventory command, keeping the original training script and arguments.
Use a fresh directory on shared storage accessible to every rank. Each rank
writes a manifest and SHA-256-addressed binary blobs. Raw tensor contents remain
there; coverage JSON contains metadata and hashes. The byte limit applies per
rank, including a bound on the storage needed to restore each tensor.
`--max-signatures` also limits the number of recorded collective events. Exceeded
limits and unsupported invocations produce capture issues, preventing D coverage
while allowing the original training calls to proceed.

The adapter supports the six direct TP/SP mappings (copy, reduce, first/last
dimension all-gather and reduce-scatter), explicit multi-rank groups, FP32/BF16,
equal shards and default mapping options. It records actual group membership,
local rank, backend, NCCL version/settings, local GPU UUID, forward/backward
policy, and input/upstream-gradient bytes. Each invocation is retained in order,
including repeated backward traversals. Input snapshots precede possible
in-place reduction. Restoration preserves logical bytes, shapes, strides,
storage offsets and broadcast views; unused storage is zero-filled. No seed is
inferred for materialized recipe tensors.

Replay uses the same source, dependencies, environment, allocation and physical
rank assignment as capture. The launcher applies the standard early MCore
determinism policy, then checks it against the capture. Preparation rejects a
different captured policy rather than relabelling it. With all rank captures
visible, run under the original torchrun topology:

```bash
python -m torch.distributed.run --nproc-per-node 8 \
  -m tools.determinism.replay_collectives \
  --capture /shared/capture --evidence /shared/replay-evidence
```

Use the recipe's actual GPU count rather than assuming eight. The launcher uses
the existing evidence plugin with a dedicated fixture, excluding generic
unit-test defaults that would change the recipe's NCCL settings. Every rank's
manifest, blob hashes and peer contracts are checked before collective replay.
The initial protocol requires the same mapping/invocation/phase order on all
global ranks; disjoint TP groups are supported. Each event runs three times with
side-stream contention, compares all output/input-gradient bytes, and checks an
independent CPU FP64 reference using the captured rank inputs. Forward and
forward/backward evidence are produced separately, with their complete matching
signatures. The existing consumer joins the resulting report to the inventory.

Implicit groups, TP1, uneven splits, global buffers, alternate output-gradient
semantics, higher-order autograd, mixed forward/backward runtime, incompatible
rank schedules, quantized tensors, overlap and unbound/native collectives need
separate adapters. Missing or corrupt blobs, truncation and incomplete ranks
cannot establish replay evidence. This remains an operation diagnostic: matching
results do not certify full training state, restarts, collective interactions,
cross-allocation behavior, or performance.

Full automatic operation discovery, first-divergence tensor comparison, and
checkpoint-state certification are separate extensions; this PR provides the
inventory-to-evidence contract and a bounded runtime producer.

CPU contract tests:

```bash
PYTHONPATH=. python -m pytest --confcutdir=tests/unit_tests/determinism_reporting \
  tests/unit_tests/determinism_reporting/test_recipe_coverage.py
```
