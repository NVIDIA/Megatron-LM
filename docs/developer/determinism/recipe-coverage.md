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
    "target": "megatron.core.fusions.fused_bias_swiglu:bias_swiglu_impl",
    "op_id": "fused_bias_swiglu",
    "implementation": "torch.compile:bias_swiglu"
  }
]
```

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

When the command requests `--deterministic-mode`, canonical determinism policy
is applied before importing bound modules, including import-time SSM policy.
The training program still validates its own configuration.

Bindings are explicit because a function name alone cannot establish the
backend or numerical variant. Include every numerical option in the arguments
or give distinct variants distinct implementation IDs. A binding must cover
the same callable contract as the corresponding test, including hidden state.

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
and cuDNN settings are part of that match. A passing signature must have matching
protocol evidence from every required rank. A passing forward+backward replay can supply
forward evidence. A backward mismatch is not projected into a forward failure.
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

The current coverage producer initially annotates fused activations. New model
shapes, backends, or environment settings will often be unverified until a
corresponding test is added. This exposes the work needed without extrapolating
from a nearby passing configuration.

Full automatic operation discovery, first-divergence tensor comparison, and
checkpoint-state certification are separate extensions; this PR provides the
inventory-to-evidence contract and a bounded runtime producer.

CPU contract tests:

```bash
PYTHONPATH=. python -m pytest --confcutdir=tests/unit_tests/determinism_reporting \
  tests/unit_tests/determinism_reporting/test_recipe_coverage.py
```
