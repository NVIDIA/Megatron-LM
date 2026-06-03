---
name: add-inference-performance-test
description: Add a new inference performance test to Megatron-LM — author the test_case YAML, register the recipe, capture the baseline through cog, and scp the baseline back. Pairs with run-inference-performance-tests, which only knows how to run already-registered tests. - Adding a new perf test for a model/parallelism config not yet covered; porting an existing H100 perf test to a new platform (GB200, B200, …); 'add a perf test', 'create new inference perf test_case', 'onboard <model> to inference perf', 'register an inference perf recipe'.
---

# Add a new inference performance test

This skill walks the full author-and-onboard flow for a brand-new
inference performance test case. The runner skill
[[run-inference-performance-tests]] knows how to *run* tests that are
already registered; this skill creates the registration in the first
place.

> ## 🌱 This is an evolving skill — keep it that way.
>
> Every cluster, every checkpoint, and every base-image bump exposes a
> new edge case. **When you hit an error this skill didn't predict —
> after fixing it — update the "Pitfalls" section with the symptom,
> root cause, and the change you made.** The value of this skill is
> the accumulated failure modes; a skill that doesn't grow falls
> behind the codebase within a release.
>
> Concretely:
> - New error you debug → add a row to the "Pitfalls" table.
> - New cluster you onboard → add cluster-specific notes inline near
>   the affected step (don't bury them at the bottom).
> - Tooling/harness change in `run_perf_test.sh` /
>   `compare_to_baseline.py` → update the relevant step here so the
>   instructions stay live.
> - Skill turned out to be wrong about a step → fix the prose, don't
>   work around it.
>
> If you find yourself doing the same diagnostic twice in a month, the
> skill is missing a pitfall row. Add it.

---

## When to use

- A new model checkpoint exists in `mcore_ci/model/...` and there's no
  perf test for it yet.
- An existing H100 perf test needs a counterpart on a new platform
  (GB200, B200, …) because the existing one doesn't fit the new node
  layout (e.g. DP=8 → DP=4 for 4-GPU GB200 nodes).
- Same model, different parallelism / batch-size / dataset sweep —
  enough variance that the baseline shouldn't be merged with the
  existing test.

Don't use this skill to *re-run* an already-registered test — that's
[[run-inference-performance-tests]].

## Prerequisites

- `cog` set up and a cluster registered. See [[cog-setup-and-help]].
- The checkpoint and tokenizer for the model live under
  `${CHECKPOINT_LOAD_PATH}/model/...` on the cluster. **Never download
  or copy checkpoints yourself** — if missing, stop and ask the user
  where the artifacts live.
- You've read [[run-inference-performance-tests]] at least once. This
  skill assumes you understand the cog driver, `--gpus` allocation,
  and the `RECORD_BASELINE=1` / `SKIP_COMPARE=1` knobs.

---

## Step 1 — Decide the test shape

Pin these dimensions before writing any files:

| Dimension | Notes |
|---|---|
| **Model** (`MODEL`) | Must match a `tests/performance_tests/server/model_args/<MODEL>.args` file. If the model isn't there yet, you're in Step 1b (new model). |
| **Platform** | `dgx_h100`, `dgx_gb200`, etc. Determines the recipe directory (`tests/test_utils/recipes/{h100,gb200}/`) and the GPU count per node. |
| **Parallelism** (`TP / PP / DP / EP`) | World size is `TP * PP * max(DP, EP)`. Must be ≤ GPUs/node for single-node tests. `run_perf_test.sh` overrides TP/PP/EP from the test case's `model_config.yaml`, so the `.args` file's defaults don't bind you. |
| **Dataset** | `synthetic` (default, identical tokens) or `gsm8k` (real prompts). MoE and hybrid models **must** use `gsm8k` — synthetic input misleadingly hits the same expert / KV cache slot every step. |
| **Batch sizes** | Sweep that covers small-batch latency and large-batch throughput regimes. Typical: `[1, 8, 32, 128]`. |
| **NUM_OUTPUT_TOKENS** | Smaller for slow models (16B MoE → 256), default 128 otherwise. Drives wall time. |
| **Metrics** | Drop `p99_latency_ms` from the assertion set unless `NUM_TIMED_ITERS` ≥ ~20 (p99 of 5 samples is `max(5)`, very noisy). |

### Step 1b — Model not yet onboarded

If `tests/performance_tests/server/model_args/<MODEL>.args` doesn't
exist, you need to create it before any test case will load.
Cross-reference an existing
`tests/functional_tests/test_cases/<family>/<test_case>/model_config.yaml`
that loads the same checkpoint, and lift its `--load`, `--tokenizer-*`,
`--model-provider`, and architecture-specific flags into the new
`.args` file. **Use `${CHECKPOINT_LOAD_PATH}/model/...` as a prefix**
so the path resolves at runtime against the artifacts root the user
chooses in [[run-inference-performance-tests]] Step 2.

The `.args` file should **not** hardcode TP/PP/DP/EP flags — the test
case `model_config.yaml` provides those and `run_perf_test.sh`
overrides anything in the .args file. (One exception: MoE checkpoints
sometimes hardcode `--expert-model-parallel-size` for the
distributed-checkpoint loader to find the right shard layout —
acceptable, since the script overrides it from config.)

---

## Step 2 — Create the test_case directory

```text
tests/performance_tests/test_cases/<family>/<test_case_name>/
└── model_config.yaml
```

`<family>` is the model family (`gpt`, `hybrid`, …) and matches the
recipe's `spec.model`. `<test_case_name>` should be unique across the
family and self-describing — include parallelism mode and platform if
disambiguating against an existing case (e.g. `gpt_583m_perf_gb200_4gpu`
sits next to `gpt_583m_perf`).

Template `model_config.yaml`:

```yaml
# Inference perf test: <one-line description>.
# <optional: cite the functional test or upstream test this was derived from>.

MODEL: <model_name>          # must match server/model_args/<MODEL>.args
TP: 1                        # tensor-model-parallel
PP: 1                        # pipeline-model-parallel
DP: 1                        # data-parallel
EP: 1                        # expert-parallel (MoE only; omit or set 1 for dense)
DATASET: synthetic           # or gsm8k for MoE/hybrid
NUM_INPUT_TOKENS: 512        # only used for synthetic; gsm8k pulls real prompts
NUM_OUTPUT_TOKENS: 128
NUM_WARMUP_ITERS: 2
NUM_TIMED_ITERS: 5
BATCH_SIZES:
  - 1
  - 8
  - 32
  - 128
TOLERANCE_PCT: 10            # regression tolerance (%) for compare_to_baseline.py
# UPPER_TOLERANCE_PCT: 20    # default 20 — uncomment if you need to override

METRICS:                     # which metrics compare_to_baseline.py asserts on
  - throughput_tok_per_sec
  - avg_latency_ms
  - p50_latency_ms
  - tpot_ms_per_tok
  # - p99_latency_ms         # uncomment only if NUM_TIMED_ITERS >= ~20
```

> **Don't pre-write `baseline_values.json`.** `RECORD_BASELINE=1` will
> create it in Step 6.

---

## Step 3 — Create or update the recipe YAML

Recipes live under
`tests/test_utils/recipes/<platform_short>/<family>-perf[-<modifier>].yaml`
where `<platform_short>` is `h100`, `gb200`, etc. Pick the closest
existing perf recipe to copy from and adjust the `products` block to
reference your new test case.

Key fields:

| Field | Value |
|---|---|
| `spec.nodes` | Almost always `1` for inference perf. Multi-node perf is uncharted (see Pitfalls). |
| `spec.gpus` | Must satisfy the cluster's QOS minimum (see Step 4). For GB200 this is `4` even for a single-GPU test. |
| `spec.platforms` | `dgx_h100`, `dgx_gb200`, … must match the recipe's directory. |
| `script: GPUS_PER_NODE=<gpus>` | Documentation only — the actual world size still comes from `model_config.yaml`. Set it equal to `spec.gpus`. |
| `products.test_case` | Your new test_case directory name. |
| `products.products.{environment, scope, platforms}` | Typically `[dev]`, `[mr]`, `[dgx_<platform>]`. |

**If you're adding a test for a model that already has a recipe on
this platform** (same `gpus` / `script` block), append your test_case
to the existing recipe's `products.test_case` list rather than
creating a new YAML.

---

## Step 4 — Check the cluster QOS GPU minimum

Before running, confirm whether the cluster you'll execute on enforces
a minimum GPU count per job:

```bash
ssh "$COG_SSH_HOST" 'sacctmgr show qos format=Name,MinTRES | head -20'
```

If `MinTRES=gres/gpu=N` appears, set `COG_GPUS_PER_NODE_LIMIT=N` in
your cluster's `~/.cog/setup.env.<cluster>` file. The
[[run-inference-performance-tests]] driver reads this and rounds the
allocation up — your test still runs with the world size from
`model_config.yaml`, just inside a larger Slurm reservation.

Known floors as of 2026-06:
- `cw-dfw` (H100, 8/node): no MinTRES — any GPU count works.
- `oci-hsg` (GB200, 4/node): `MinTRES=gres/gpu=4`. All jobs must
  request ≥ 4 GPUs.

---

## Step 5 — First run with `SKIP_COMPARE=1`

This verifies the test wiring (model loads, server starts, benchmark
sweeps complete) before committing any baseline numbers.

```bash
source ~/.cog/setup.env.<cluster>          # picks the right cluster
export COG_ARTIFACTS_ROOT=<mcore_ci path on that cluster>
/tmp/launch_perf_test.sh <family>/<test_case_name> <world_size> SKIP_COMPARE=1
```

(`launch_perf_test.sh` is the wrapper [[run-inference-performance-tests]]
documents. If you don't have it, paste the
`submit_inference_perf_test` driver from that skill's Step 4 and call
it.)

A successful run prints `"returncode": 0` and writes
`$RUN_DIR/perf_results/results.json` with one entry per batch size.
Read that file — if any batch's throughput is wildly off from
your back-of-envelope expectation, fix the test config before
recording the baseline.

---

## Step 6 — Record the baseline

```bash
/tmp/launch_perf_test.sh <family>/<test_case_name> <world_size> RECORD_BASELINE=1
```

This re-runs the test and writes `baseline_values.json` in the cluster's
synced workspace under the **detected platform key** (auto-detect via
`nvidia-smi -L`; override with a `PLATFORM=<name>` arg). The merge is
non-destructive — other platforms' subtrees in the same file are
preserved.

Then `scp` the file back into your local checkout:

```bash
WS_HASH=$(grep -oE '"workspace_hash":\s*"[^"]+"' /tmp/perf-<safe-name>.log | head -1 | sed 's/.*"\([^"]*\)"$/\1/')
scp "$COG_SSH_HOST:$COG_SCRATCH_ROOT/workspaces/megatron_lm/$WS_HASH/repo/tests/performance_tests/test_cases/<family>/<test_case_name>/baseline_values.json" \
    "$COG_MEGATRON_REPO/tests/performance_tests/test_cases/<family>/<test_case_name>/baseline_values.json"
```

Verify locally that the file has the expected platform key(s):

```bash
python3 -c "import json; print(list(json.load(open('tests/performance_tests/test_cases/<family>/<test_case_name>/baseline_values.json')).keys()))"
```

---

## Step 7 — Verify enumeration picks up the new test

```bash
uv run --no-project --with pyyaml python \
  skills/run-inference-performance-tests/list_inference_perf_tests.py \
  | grep <test_case_name>
```

You should see your test in the TSV with the right `gpus` and
`platforms` fields. If not, the recipe's `scope` / `environment` /
`platforms` filter doesn't match what the helper looks for — check
the helper code.

---

## Step 8 — Sanity-check the comparison path

Run once more without `SKIP_COMPARE` or `RECORD_BASELINE` — the test
should now pass against the baseline you just recorded:

```bash
/tmp/launch_perf_test.sh <family>/<test_case_name> <world_size>
```

If you see `OK: all metrics within tolerance.` you're done. If it
fails on the same numbers it just recorded, the auto-detect for
`PLATFORM` probably differed between the record run and the compare
run — check both slurm logs' `[run_perf_test] ... PLATFORM=...` lines
and pin via `PLATFORM=<name>` if needed.

---

## Pitfalls (grow this section)

Each row: symptom → root cause → fix. **Add a new row whenever you
debug a fresh failure.** Don't delete rows when fixing the underlying
bug — future you may regress and want to find this entry again.

| Symptom | Root cause | Fix |
|---|---|---|
| `cannot execute binary file: Exec format error` from `run_perf_test.sh` line ~52 | yq binary architecture mismatch (e.g. `yq_linux_amd64` on aarch64 GB200) | Dispatch on `uname -m` in the launcher's inline yq install — `aarch64 → yq_linux_arm64`. The [[run-inference-performance-tests]] driver already does this. |
| `srun: error: QOSMinGRES` / `Unable to allocate resources: Job violates accounting/QOS policy` | Cluster QOS sets a per-job GPU minimum (e.g. GB200 enforces ≥ 4) and your `--gpus` is below it | Set `COG_GPUS_PER_NODE_LIMIT=N` in `~/.cog/setup.env.<cluster>`; the driver rounds `--gpus` up. |
| `RuntimeError: world_size (X) is not divisible by expert_tensor_model_pipeline_parallel size (Y)` | MoE checkpoint's `.args` file hardcodes `--expert-model-parallel-size Y` but `model_config.yaml` set `EP=X` and `X != Y` | `run_perf_test.sh` overrides EP from config — if you see this, the override block at the top of the script is missing or stale. Re-add `MODEL_ARGS+=(--expert-model-parallel-size "$EP")`. |
| `ImportError: MambaSSM is not installed` on a hybrid/mamba test | Base image doesn't have mamba-ssm/causal-conv1d (true for bare NGC PyTorch — only the `mcore_ci_dev` image ships them under `/opt/venv`) | `run_perf_test.sh` has two fallbacks: shim `.pth` pointing at `/opt/venv` (H100 / mcore_ci_dev), and `uv pip install --no-build-isolation mamba-ssm causal-conv1d` (GB200 / bare NGC, relies on PyPI's `linux_aarch64` wheels). |
| `server did not become ready within 15 min` then test fails | Server crashed during init. Causes vary: wrong checkpoint path, tokenizer mismatch, CUDA/TE version skew, OOM on the chosen batch size | Read `$RUN_DIR/perf_results/server_logs/server.log` for the actual traceback, fix at the source. |
| `tokenizer_path: ... is invalid` | `COG_ARTIFACTS_ROOT` doesn't point at the real mcore_ci tree on this cluster, or the lustre mount isn't there | Re-do [[run-inference-performance-tests]] Step 2 and confirm `$COG_ARTIFACTS_ROOT/model/` exists. |
| `error: no baseline_values.json at ...` from compare step | First-time bootstrap or you deleted the baseline | Re-run with `RECORD_BASELINE=1`, scp back. |
| `ERROR: no baseline for platform 'X' in baseline_values.json` | The baseline file exists but doesn't have a subtree for the platform you ran on | Record on that platform with `RECORD_BASELINE=1`. The merge-write preserves other platforms. |
| Comparison reports throughput regression on the same run that recorded the baseline | Auto-detected `PLATFORM` differed between record and compare runs (e.g. heterogeneous cluster) | Pin `PLATFORM=<name>` explicitly via the third positional arg of the perf-test launcher. |
| `Improvement too large — refresh baseline` | `UPPER_TOLERANCE_PCT` exceeded — typically because of a real perf improvement, occasionally measurement noise | Intentional speedup → refresh baseline with `RECORD_BASELINE=1`. Otherwise investigate before refreshing. |
| 8-GPU H100 test fails on GB200's 4-GPU/node nodes | DP=8 / EP=8 needs world_size 8, but a GB200 node only has 4 GPUs and single-node is the only path `run_perf_test.sh` knows | Create a 4-GPU variant (`<test_case>_gb200_4gpu`) with `DP=4` or `EP=4` — different test, different baseline. Multi-node single-test perf is currently unsupported. |
| Cog `returncode: 126` with empty stderr | Almost always something `bash -c` couldn't execute inside the container — usually a binary architecture mismatch (yq on ARM, …) or a missing `/usr/local/bin/...` | Read the SLURM stderr (`cog logs slurm --stream stderr`) — line 1 of the wrapper's bash usually shows what failed. |

---

## Cross-references

- [[run-inference-performance-tests]] — how to run an already-registered
  perf test through cog. Owns the canonical `submit_inference_perf_test`
  driver function this skill calls into.
- [[cog-setup-and-help]] — cog install, cluster registration, and the
  full CLI reference. This skill defers there for every cog command.
- [[testing]] — recipe YAML schema, scope/environment/platform
  vocabulary, golden values for functional tests (different concept
  from baselines here).
- `tests/performance_tests/shell_test_utils/run_perf_test.sh` — the
  in-container driver. If you change PLATFORM detection, the merge
  logic, or the override block here, the steps above must be updated
  in lockstep.
- `tests/performance_tests/shell_test_utils/compare_to_baseline.py` —
  baseline comparator. Takes `--platform` and indexes
  `baseline[platform]`.
