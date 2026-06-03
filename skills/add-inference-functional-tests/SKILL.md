---
name: add-inference-functional-tests
description: Add a new inference functional test to Megatron-LM — author the test_case YAML, register the recipe, bootstrap the golden_values JSON through cog, and scp it back. Pairs with run-inference-functional-tests, which only knows how to run already-registered tests.
when_to_use: Adding a new inference functional test for a model / parallelism / sampling config not yet covered; porting an existing test to a new platform; 'add a functional test', 'create new inference test_case', 'onboard a new inference recipe', 'record golden values for a new functional test'.
---

# Add a new inference functional test

This skill walks the full author-and-onboard flow for a brand-new
inference functional test case. The runner skill
[[run-inference-functional-tests]] knows how to *run* tests that are
already registered; this skill creates the registration in the first
place and records the initial `golden_values_*.json` file through cog
rather than waiting on the CI label.

> ## 🌱 This is an evolving skill — keep it that way.
>
> Every new test case, every base-image bump, every new sampling
> backend exposes a fresh edge case. **When you hit an error this
> skill didn't predict — after fixing it — update the "Pitfalls"
> section with the symptom, root cause, and the change you made.**
> A skill that doesn't grow falls behind the codebase within a release.
>
> Concretely:
> - New error you debug → add a row to the "Pitfalls" table.
> - New cluster / platform variant → add cluster-specific notes inline
>   near the affected step (don't bury them at the bottom).
> - Tooling/harness change in `run_ci_test.sh` / `_run_training.sh` /
>   `test_inference_regular_pipeline.py` → update the relevant step
>   here so the instructions stay live.
> - Skill turned out to be wrong about a step → fix the prose, don't
>   work around it.
>
> If you find yourself doing the same diagnostic twice in a month, the
> skill is missing a pitfall row. Add it.

---

## When to use

- A new sampling / engine / parallelism config exists that isn't covered
  by any test under `tests/functional_tests/test_cases/<family>/*inference*`.
- An existing H100 test needs a counterpart on a new platform (GB200,
  B200, …) because checkpoint sharding / numerics differ enough that
  the existing goldens won't reuse.
- A bug repro needs to become a permanent regression test — wrap it as
  a new functional test rather than carrying a one-off script.

Don't use this skill to *re-run* an already-registered test — that's
[[run-inference-functional-tests]]. Don't use it to refresh existing
goldens after an intentional numeric change — that's
[[update-golden-values]].

## Prerequisites

- `cog` set up and a cluster registered. See [[cog-setup-and-help]].
- The checkpoint and tokenizer for the model live under
  `${CHECKPOINT_LOAD_PATH}/model/...` on the cluster. **Never download
  or copy checkpoints yourself** — if the artifacts you need aren't
  already on the cluster filesystem, stop and ask the user where their
  copy lives. Tokenizers are multi-GB, full checkpoints are tens of
  GB, and the on-disk layout is load-bearing for goldens.
- You've read [[run-inference-functional-tests]] at least once. This
  skill reuses its `submit_inference_test` driver, the cog
  `COG_EXTRA_MOUNTS` plumbing, the venv shebang fix, and the artifacts
  root prompt from its Step 2.

---

## Step 1 — Decide the test shape

Pin these dimensions before writing any files:

| Dimension | Notes |
|---|---|
| **Model family** | `gpt`, `mamba`, `moe`, `hybrid`, … — must match the `spec.model` of a recipe under `tests/test_utils/recipes/<platform_short>/`. Determines the `tests/functional_tests/test_cases/<family>/` subdirectory. |
| **Mode** | `static` (single shot, all prompts arrive together) vs `dynamic` (rolling batches via the dynamic engine) vs `dynamic-with-coordinator` (multi-DP via the ZMQ coordinator). Drives which inference example script is invoked. |
| **TRAINING_SCRIPT_PATH** | One of `examples/inference/gpt/gpt_static_inference.py`, `gpt_dynamic_inference.py`, or `gpt_dynamic_inference_with_coordinator.py`. The model family is independent of the script — `moe-dynamic-inference.yaml` still uses `gpt_dynamic_inference.py`. The lookup is in the recipe's `script` block. |
| **Parallelism** (`TP / PP / DP`) | World size = `TP * PP * DP`. CI allocates 8 GPUs/node on H100 and runs even tp=1 pp=1 dp=1 tests across 8 ranks; the [[run-inference-functional-tests]] driver matches this by always passing `--gpus 8`. Pick TP/PP/DP that *together with the checkpoint shard layout* are loadable — see Pitfalls. |
| **Backend** | `flash` vs `flashinfer` vs cuda_graphs / chunked_prefill / prefix-caching — encoded as model_config flags. Choose what the test exercises. |
| **Sampling** | `--top_k`, `--top_p`, `--temperature`, stop words, n-logprobs. Whatever the test is meant to cover. |
| **Test case name** | Convention: `<family>_<mode>_inference_tp<n>_pp<n>[_dp<n>]_<size>_<feature>`. Examples: `gpt_static_inference_tp1_pp1_583m_logitsmatch`, `gpt_dynamic_inference_tp2_pp2_583m_prefix_caching_cuda_graphs`. Keep it self-describing — the recipe row, the test_case dir, and the golden filename all share this slug. |
| **Platform** | `dgx_h100` (default) and/or `dgx_gb200`. Each platform gets its own golden file: `golden_values_dev_dgx_h100.json`, `golden_values_dev_dgx_gb200.json`. |
| **Environment** | `dev` (current PyTorch base image) — the only env we run via cog today. `lts` is CI-only. |
| **Metrics** | What `test_inference_regular_pipeline.py` will compare against the golden — listed in `MODEL_ARGS.METRICS` of the model_config. For dynamic/static engines the typical set is `generated_tokens` + `logprobs`. Coordinator tests add per-rank or per-request structure — copy the closest existing case. |

---

## Step 2 — Create the test_case directory

```text
tests/functional_tests/test_cases/<family>/<test_case_name>/
└── model_config.yaml
```

Template (lifted from `gpt_static_inference_tp1_pp1_583m_logitsmatch` —
the simplest passing case as of 2026-05). Pull the architecture flags
(`--num-layers`, `--hidden-size`, `--num-attention-heads`,
`--max-position-embeddings`, `--seq-length`, …) from the closest
existing model_config that loads the **same checkpoint**:

```yaml
# Inference functional test: <one-line description>.
# <optional: cite the upstream test or model card this was derived from>.

ENV_VARS:
  CUDA_DEVICE_MAX_CONNECTIONS: 1
  NVTE_ALLOW_NONDETERMINISTIC_ALGO: 0
  NCCL_ALGO: Ring
  CUBLAS_WORKSPACE_CONFIG: :4096:8
  # Do NOT add SKIP_PYTEST here — the bootstrap pass in Step 4 sets it via
  # ENABLE_LIGHTWEIGHT_MODE so this file stays the same in CI.

TEST_TYPE: frozen-start
MODE: inference

MODEL_ARGS:
  --use-mcore-models: true

  # Tokenizer & checkpoint — paths must use ${CHECKPOINT_LOAD_PATH} so
  # they resolve against the artifacts root the user picks at runtime.
  --tokenizer-type: TikTokenizer
  --tokenizer-model: ${CHECKPOINT_LOAD_PATH}/model/mcore_mistral/nemo_minitron-0.5b/v1/multiMixV8.gpt4o_nc_sd.500000.128k.vocab.json
  --load: ${CHECKPOINT_LOAD_PATH}/model/mcore_mistral/nemo_minitron-0.5b/v1/
  --tiktoken-pattern: v2

  --use-checkpoint-args: true
  --no-load-optim: true
  --no-use-tokenizer-model-from-checkpoint-args: true
  --auto-detect-ckpt-format: true
  --ckpt-format: torch_dist
  --dist-ckpt-strictness: log_unexpected

  # Parallelism — these are the dims the test name advertises.
  --tensor-model-parallel-size: 1
  --pipeline-model-parallel-size: 1

  # Architecture flags — copy from a model_config that loads the same ckpt.
  --num-layers: 24
  --hidden-size: 1152
  --num-attention-heads: 16
  --max-position-embeddings: 1024
  --seq-length: 1024
  --transformer-impl: transformer_engine
  --distributed-backend: nccl
  --bf16: true
  --attention-backend: flash
  --deterministic-mode: true
  --micro-batch-size: 1
  --max-tokens-to-oom: 3600000
  --inference-max-seq-length: 4096
  --log-interval: 1
  --timing-log-level: 0
  --log-memory-to-tensorboard: true
  --log-num-zeros-in-grad: true
  --log-validation-ppl-to-tensorboard: true
  --log-timers-to-tensorboard: true

  # Sampling.
  --temperature: 1.0
  --top_k: 1
  --return-log-probs: true
  --num-tokens-to-generate: 30

  # Engine-specific (uncomment for dynamic):
  # --incoming-requests-per-step: 32
  # --inference-dynamic-batching-buffer-size-gb: 10
  # --inference-repeat-n: 8
  # --use-flashinfer-fused-rope: true
  # --inference-logging-step-interval: 1

  # Engine-specific (uncomment for static):
  # --flash-decode: true
  # --use-legacy-static-engine: true
  # --incoming-requests-per-sec: -1   # all up front

  # Output — must use ${INFERENCE_OUTPUT_PATH}, the runner writes it here
  # and pytest compares it against GOLDEN_VALUES_PATH.
  --output-path: ${INFERENCE_OUTPUT_PATH}
  --inference-ckpt-non-strict: true

  --prompts: "Time travel to 2008, and go to a bar or a club or one of the myriad disco-basements on the Lower East Side that does not quite know which of those it is. Dance awkwardly in a room full of other glittered-up nerds, and wait for something to happen, buoyed on the feeling that this is the big swollen heart of life, that this is New York like the movies."

METRICS:
  - "generated_tokens"
  - "logprobs"
```

> **Do not pre-write `golden_values_dev_dgx_h100.json`.** It gets
> generated by Step 4's bootstrap pass and copied back in Step 5.
> If you ship a hand-written file the goldens compare will silently
> be wrong.

---

## Step 3 — Add the test case to the recipe YAML

Recipes live under `tests/test_utils/recipes/<platform_short>/`. Look
up the right one by mode:

| Mode | Recipe (h100) | TRAINING_SCRIPT_PATH |
|---|---|---|
| dynamic engine (single DP) | `<family>-dynamic-inference.yaml` | `examples/inference/gpt/gpt_dynamic_inference.py` |
| static engine | `<family>-static-inference.yaml` | `examples/inference/gpt/gpt_static_inference.py` |
| dynamic + ZMQ coordinator (multi-DP) | `<family>-dynamic-inference-with-coordinator.yaml` | `examples/inference/gpt/gpt_dynamic_inference_with_coordinator.py` |

**If the right recipe already exists** (almost always — most new test
cases are sampling/feature variations of an existing model+mode),
append your test_case slug to its `products` list:

```yaml
products:
  - test_case: [<your_new_test_case>]
    products:
      - environment: [dev]
        scope: [mr]                # or [mr, mr-github] for github MR CI too
        platforms: [dgx_h100]
```

**Only create a new recipe** if the model family / mode pair is new
(e.g. first `moe-static-inference.yaml`). In that case copy the closest
existing recipe, change `spec.model`, change the
`TRAINING_SCRIPT_PATH=` inside its `script` block, and trim `products`
down to your one test case. Keep `spec.gpus: 1` and `spec.nodes: 1` —
the runner driver allocates 8 GPUs regardless (`_run_training.sh`
defaults `GPUS_PER_NODE=8`), and the recipe `gpus` field is mostly
metadata for the recipe parser.

> **Scope choice.** `mr` makes the test run in the GitLab internal CI
> matrix; `mr-github` also runs it in the GitHub MR matrix. Use both
> for any test that should gate external PRs; use just `mr` for tests
> that are too expensive or too platform-specific. `mr-broken` /
> `mr-github-broken` mark a known-broken case without deleting it —
> see the testing skill's "Disabling a test without deleting it"
> section.

---

## Step 4 — Bootstrap the golden values file (lightweight run)

For `MODE: inference`, the test driver's `ENABLE_LIGHTWEIGHT_MODE=true`
flag does exactly one thing: it sets `ENV_VARS.SKIP_PYTEST=1` inside
the model_config at runtime, which short-circuits the
`test_inference_regular_pipeline.py` comparison step. The
training/inference still runs end-to-end and still writes the result
JSON to `$INFERENCE_OUTPUT_PATH`. That JSON is exactly the structure a
golden file must have.

(Confirmed at `tests/functional_tests/shell_test_utils/run_ci_test.sh`
line ~117: the `MODE == "inference" && ENABLE_LIGHTWEIGHT_MODE == "true"`
branch only flips `SKIP_PYTEST=1` — unlike the pretraining branch, it
does NOT mutate `--exit-interval` or otherwise truncate the run.)

The runner skill's `submit_inference_test` driver accepts a
`LIGHTWEIGHT_MODE` env var to enable this. Confirm the runner skill is
loaded and call:

```bash
# Steps 1–3 from run-inference-functional-tests must already be done:
#   - cog ready (its Step 1)
#   - $COG_ARTIFACTS_ROOT exported (its Step 2)
#   - venv shebangs patched (its Step 3)
#
# Then bootstrap the golden:
LIGHTWEIGHT_MODE=true submit_inference_test \
  <test_case_name> \
  <family> \
  <TRAINING_SCRIPT_PATH> \
  2>&1 | tee /tmp/ftest-bootstrap-<test_case_name>.log
```

A successful bootstrap prints `"returncode": 0` *and* shows
`Skipping Pytest checks.` (or similar) in the slurm log — i.e. the
test completed without trying to compare. If you get a real pytest
failure here, your bootstrap is doing a real comparison against a
file you didn't mean to create — make sure no
`golden_values_<env>_<platform>.json` already exists in the test_case
directory.

---

## Step 5 — `scp` the bootstrap output back as the golden

The inference output JSON lives at
`$RUN_DIR/inference_output.json` on the cluster. Pull the run-dir path
from the cog JSON in the log, then `scp` the file into the local
test_case directory:

```bash
source ~/.cog/setup.env
RUN_ROOT=$(grep -oE '"run_root":\s*"[^"]+"' /tmp/ftest-bootstrap-<test_case>.log \
           | head -1 | sed 's/.*"\([^"]*\)"$/\1/')
echo "run_root on cluster: $RUN_ROOT"

scp "$COG_SSH_HOST:$RUN_ROOT/inference_output.json" \
    "$COG_MEGATRON_REPO/tests/functional_tests/test_cases/<family>/<test_case>/golden_values_dev_dgx_h100.json"
```

Verify locally that the file is well-formed JSON and contains every
metric listed in `model_config.yaml`'s `METRICS`:

```bash
python3 - <<PY
import json, sys
p = "tests/functional_tests/test_cases/<family>/<test_case>/golden_values_dev_dgx_h100.json"
data = json.load(open(p))
print("top-level keys:", list(data)[:10])
PY
```

If the file is missing keys you expect, the inference script didn't
emit them — go fix the test (often a missing `--return-log-probs` or
`--output-path` flag, or `METRICS` referencing a name the engine
doesn't produce) and re-bootstrap.

---

## Step 6 — Verify enumeration picks up the new test

```bash
uv run --no-project --with pyyaml python \
  skills/run-inference-functional-tests/list_inference_tests.py \
  | grep <test_case>
```

You should see one row with the right model and TRAINING_SCRIPT_PATH.
If not, the recipe row's `scope` / `environment` / `platforms` doesn't
match what the helper filters on (`scope ∈ {mr, mr-github}`,
`environment ∋ dev`, `platforms ∋ dgx_h100`). Fix the recipe.

---

## Step 7 — Sanity-check the comparison path

Now run the test **without** lightweight mode — `test_inference_regular_pipeline.py`
will load the golden we just recorded and compare it to a fresh
inference output. With deterministic seeding (the model_config sets
`--deterministic-mode: true` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` +
`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`) the second run should match
bit-for-bit:

```bash
submit_inference_test \
  <test_case_name> \
  <family> \
  <TRAINING_SCRIPT_PATH> \
  2>&1 | tee /tmp/ftest-verify-<test_case>.log
```

Expected: `"returncode": 0` and `1 passed in 0.XXs` from pytest. If
the second run *fails* the comparison, the test isn't actually
deterministic — investigate before merging (the most common cause is
a missing `--deterministic-mode` flag or a non-deterministic kernel
in the chosen attention backend).

---

## Pitfalls (grow this section)

Each row: symptom → root cause → fix. **Add a new row whenever you
debug a fresh failure.** Don't delete rows when fixing the underlying
bug — future you may regress and want to find this entry again.

| Symptom | Root cause | Fix |
|---|---|---|
| `tokenizer_path: ... is invalid` during bootstrap | `$COG_ARTIFACTS_ROOT` doesn't point at the real `mcore_ci` tree on this cluster, or the lustre mount isn't there | Re-do [[run-inference-functional-tests]] Step 2 and confirm `$COG_ARTIFACTS_ROOT/model/<your_subpath>` exists on the cluster. Never download a replacement — ask the user. |
| `RuntimeError: tensor model parallel size … does not match checkpoint` | TP/PP in your `model_config.yaml` doesn't match how the checkpoint was sharded on disk | Look at the closest passing test's TP/PP for the same checkpoint and match it; or re-shard the checkpoint (out of scope for this skill — ask the user). |
| pytest `1 failed` with `AssertionError: Mismatch in 'generated_tokens'` on the verify pass *immediately after* bootstrap | Run isn't deterministic — different seed, missing `--deterministic-mode`, or a kernel in the chosen attention backend that's nondeterministic | Add `--deterministic-mode: true` and `ENV_VARS: { CUBLAS_WORKSPACE_CONFIG: :4096:8, NVTE_ALLOW_NONDETERMINISTIC_ALGO: 0, NCCL_ALGO: Ring }` to the model_config. If still flaky, switch `--attention-backend` to `flash` and re-bootstrap. |
| pytest `KeyError: 'logprobs'` in `test_inference_regular_pipeline.py` | `METRICS` includes a key the inference script didn't emit (usually because the corresponding flag — `--return-log-probs`, `--top_n_logprobs` — wasn't passed) | Add the flag in `MODEL_ARGS` and re-bootstrap. Or trim `METRICS` to what the engine actually produces. |
| Bootstrap returns `"returncode": 0` but `$RUN_DIR/inference_output.json` is empty/missing | Inference script didn't honor `--output-path`, or wrote to a different path | Confirm the model_config has `--output-path: ${INFERENCE_OUTPUT_PATH}` (literally — the envsubst pass in `_run_training.sh` substitutes `INFERENCE_OUTPUT_PATH`). Some legacy inference examples ignore the flag — pick a different `TRAINING_SCRIPT_PATH`. |
| `Providing $ENABLE_LIGHTWEIGHT_MODE is mandatory` from the bootstrap submit | `ENABLE_LIGHTWEIGHT_MODE=` was expanded to empty in the cog `--command` (driver bug) | Confirm you're calling the driver as `LIGHTWEIGHT_MODE=true submit_inference_test …`, not `ENABLE_LIGHTWEIGHT_MODE=true …` (the env var the driver reads is `LIGHTWEIGHT_MODE`, which it then forwards as `ENABLE_LIGHTWEIGHT_MODE` inside the cog command). |
| Recipe row not picked up by `list_inference_tests.py` | scope/environment/platforms filter mismatch | The helper keeps rows where `scope ∋ {mr, mr-github} ∧ environment ∋ dev ∧ platforms ∋ dgx_h100`. Use `scope: [mr]` or `scope: [mr, mr-github]` (not `mr-broken`). |
| `examples/inference/gpt/gpt_dynamic_inference_with_coordinator.py` fails with `RuntimeError: world_size … not divisible by …` for a multi-DP test | `TP * PP * DP` in your model_config doesn't equal `GPUS_PER_NODE` (default 8) | World size must equal the number of ranks the launcher spawns. Either set `TP*PP*DP=8` or pin `ENV_VARS.GPUS_PER_NODE` in the model_config. |
| Two model_config entries both use `${CHECKPOINT_LOAD_PATH}` but resolve to different paths in the same cog run | One of them is in `ENV_VARS` (gets re-exported into the test's environment) and the other is read by `envsubst` from `MODEL_ARGS` — both should resolve to the same value, but only if `CHECKPOINT_LOAD_PATH` is in the cog `ARGUMENTS=(…)` block | Don't redeclare `CHECKPOINT_LOAD_PATH` inside `ENV_VARS` — let the runner driver's `ARGUMENTS=(…)` own it. |
| You wrote a hand-crafted `golden_values_dev_dgx_h100.json` and the verify pass fails on every numeric | Golden file structure / numerics drift from what the engine actually produces | Always bootstrap goldens via Step 4. Delete the hand-written file, re-run lightweight, scp the produced JSON back. |

---

## Cross-references

- [[run-inference-functional-tests]] — how to run an already-registered
  functional test through cog. Owns the canonical `submit_inference_test`
  driver function this skill calls into. Steps 1 (cog ready), 2 (ask
  for artifacts root), and 3 (venv shebang fix) are shared
  prerequisites.
- [[cog-setup-and-help]] — cog install, cluster registration, full CLI
  reference. This skill defers there for every cog command.
- [[testing]] — recipe YAML schema, scope/environment/platform
  vocabulary, the "disable without deleting" `-broken` suffix trick,
  CI parity notes.
- [[update-golden-values]] — refresh existing goldens after an
  intentional numeric change. Use that, not this skill, when the test
  already exists.
- `tests/functional_tests/shell_test_utils/run_ci_test.sh` — the
  in-container driver. The `ENABLE_LIGHTWEIGHT_MODE` branch this skill
  relies on for bootstrapping is at line ~117. If that branch's
  semantics ever change (e.g. someone adds `--exit-interval`
  truncation for inference too), the bootstrap path here breaks —
  update this skill in lockstep.
- `tests/functional_tests/python_test_utils/test_inference_regular_pipeline.py`
  — the comparator. Reads `--golden-values-path` and `--test-values-path`,
  asserts metric-by-metric. If you change the JSON schema, both this
  skill's template and that comparator need to move together.
