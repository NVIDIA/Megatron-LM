---
name: run-inference-functional-tests
description: Run Megatron-LM inference functional tests (every test_case with `inference` in the name from the H100 recipes) on a Slurm cluster via cog, and report pass/fail per test case. Auto-bootstraps cog via the cog-setup-and-help skill if cog is not installed or `~/.cog/setup.env` is missing.
when_to_use: Running the inference functional-test suite end-to-end on the cluster from a laptop; validating inference-side changes with golden-value comparison before pushing; 'run inference functional tests', 'inference goldens', 'verify inference goldens on cluster', 'run functional tests on the cluster'.
---

# Run Megatron-LM inference functional tests on the cluster via cog

The CI Slurm bucket that runs every `tests/functional_tests/test_cases/**/*inference*`
end-to-end against its golden value file takes 30-60 min and queues behind the
full pipeline. This skill submits the same tests from your laptop via cog so
you can verify inference-side changes (engine, sampling, attention backend,
prefix caching, …) before pushing.

The first part of this skill is a hard dependency on **`cog-setup-and-help`**.
If cog is missing or `~/.cog/setup.env` is not populated, invoke
[[cog-setup-and-help]] first; do not proceed until those preconditions hold.

> **🔁 Iterating on the same test repeatedly?** Don't `cog submit` in a
> loop — each submit re-queues for a fresh Slurm allocation. Instead,
> `cog session start` once with a 2-3 hour wall-clock budget and then
> `cog session exec` for each iteration. See the "Iterating" section
> in [[cog-setup-and-help]] for the exact recipe — that skill is the
> canonical in-tree cog reference (commands, flags, error codes).

The second part depends on the **`testing`** skill's recipe structure — every
recipe under `tests/test_utils/recipes/h100/*inference*.yaml` maps each
`test_case` to a `TRAINING_SCRIPT_PATH` in its `script` block, and the helper
`list_inference_tests.py` shipped with this skill parses that mapping so we
don't have to hardcode it.

---

## Step 1 — Verify cog is ready (or bootstrap it)

Run all three checks. **If any check fails, invoke the `cog-setup-and-help` skill and
re-run the checks before proceeding.**

```bash
if ! test -f ~/.cog/setup.env || \
   ! grep -q '^export COG_MEGATRON_REPO=' ~/.cog/setup.env || \
   ! grep -q '^export COG_INTERACTIVE_PARTITION=' ~/.cog/setup.env; then
  echo "MISSING: ~/.cog/setup.env — run cog-setup-and-help skill"; exit 1
fi
cog --help > /dev/null 2>&1 || { echo "cog import failed — run cog-setup-and-help skill"; exit 1; }
source ~/.cog/setup.env
cog doctor --repo "$COG_MEGATRON_REPO" 2>&1 | grep -q '"overall":"ok"' \
  || { echo "cog doctor failed — run cog-setup-and-help skill"; exit 1; }
echo "cog ready: $COG_CLUSTER_NAME @ $COG_SSH_HOST"
```

---

## Step 2 — Ask the user where the CI models/data live on the cluster

Every inference `model_config.yaml` references `${CHECKPOINT_LOAD_PATH}/model/...`
and `${CHECKPOINT_LOAD_PATH}/...` paths — those resolve against an
**artifacts root directory** on the cluster filesystem that holds the
checkpoint tree, tokenizer files, and any auxiliary data the tests load.

**Default (used by NVIDIA CI on cw-dfw):**

```text
/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_mcore/mcore_ci
```

**Procedure (the agent running this skill must do this — do not skip):**

1. Ask the user, via `AskUserQuestion`, whether to use the default path or a
   different one. Make the default the first option and recommend it.
2. **Never attempt to download, copy, mirror, or stage model checkpoints
   yourself.** Tokenizers are multi-GB; full checkpoints are tens of GB; and
   the file layout under `mcore_ci/model/...` is opinionated (recipe paths
   like `model/mcore_mistral/nemo_minitron-0.5b/v1/...` are checked into
   `model_config.yaml` and are matched bit-for-bit by goldens). If the
   default path is not accessible, **stop and ask the user to point you at
   their copy** rather than guessing or downloading anything.
3. Verify the chosen path actually exists on the cluster and contains the
   `model/` subtree before proceeding. Bail out with a clear error if it
   doesn't — do not silently fall back.
4. Export it as `COG_ARTIFACTS_ROOT` so the rest of this skill (Step 5's
   driver) picks it up consistently:

```bash
# Default if the user accepts it; otherwise use the path they gave.
export COG_ARTIFACTS_ROOT="${COG_ARTIFACTS_ROOT:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_mcore/mcore_ci}"

# Verify the chosen path is reachable and has the expected `model/` subdir.
source ~/.cog/setup.env
if ! ssh -o BatchMode=yes "$COG_SSH_HOST" \
     "test -d '$COG_ARTIFACTS_ROOT/model'" ; then
  cat <<EOF >&2
ERROR: artifacts root '$COG_ARTIFACTS_ROOT' is not accessible on cluster
       '$COG_SSH_HOST', or it does not contain a 'model/' subdirectory.

       This skill does NOT download checkpoints/data. Ask the user where
       their copy of the mcore_ci artifacts lives and re-run with
       \`export COG_ARTIFACTS_ROOT=<their path>\`.
EOF
  exit 1
fi
echo "artifacts root: $COG_ARTIFACTS_ROOT"
```

> **Why not auto-download:** the artifacts tree is large, has restricted
> internal sources, and the path layout is load-bearing for golden-value
> comparison. Silent download/mirror would silently diverge from CI.
> Always defer to the user's existing copy.

---

## Step 3 — One-time fix: rewrite the cog venv's stale `.partial.<hash>` shebangs

The cog shared `.venv` is built under `…/envs/<recipe>/<hash>.partial.<rand>/.venv`
and then `mv`-d into the canonical `…/envs/<recipe>/<hash>/.venv`. Every bin
script in the venv (pytest, isort, …) has the **partial** path baked into its
shebang, so `uv run --no-sync pytest` (called by `tests/functional_tests/shell_test_utils/run_ci_test.sh`)
exits 127 with `python3: not found`. This is the same root cause as the
`python -m <module>` workaround documented in [[cog-setup-and-help]] — but
`run_ci_test.sh` is checked-in code we can't easily route through
`python -m pytest`, so we patch the shebangs directly.

The patch is **idempotent** — once the canonical path is in place, re-running
this step is a no-op:

```bash
source ~/.cog/setup.env

# Resolve the venv's canonical path from cog's env layout.
VENV=$(ssh -o BatchMode=yes "$COG_SSH_HOST" \
  "ls -d $COG_SCRATCH_ROOT/envs/megatron_lm/*/.venv 2>/dev/null \
     | grep -v '.partial.' | head -1")
if [ -z "$VENV" ]; then
  echo "No cog venv found — submit any cog job once (e.g. \`cog ensure-env\`) and retry"; exit 1
fi

ssh -o BatchMode=yes "$COG_SSH_HOST" "VENV='$VENV' bash -s" <<'REMOTE'
for f in "$VENV/bin"/*; do
  test -f "$f" || continue
  if head -3 "$f" 2>/dev/null | grep -q '\.partial\.'; then
    PARTIAL=$(head -3 "$f" | grep -oE '/lustre[^"'"'"']+\.partial\.[a-f0-9]+/\.venv' | head -1)
    if [ -n "$PARTIAL" ]; then
      sed -i "s|$PARTIAL|$VENV|g" "$f"
    fi
  fi
done
head -3 "$VENV/bin/pytest"
REMOTE
```

The expected final `head -3` shows pytest's `exec` line pointing at the
**canonical** `…/<hash>/.venv/bin/python3` (no `.partial.` segment).

> **Why fix the venv instead of patching `run_ci_test.sh`:** modifying the
> checked-in test driver would drift from CI. Fixing the venv is a real bug
> fix at the layer that caused it (cog's partial→canonical rename), and it
> persists across submits because the venv lives on shared scratch.

---

## Step 4 — Enumerate inference test cases

This skill ships a helper, `list_inference_tests.py`, that walks every
H100 recipe matching `*inference*.yaml`, keeps the rows with `scope ∋ {mr,
mr-github}`, `environment ∋ dev`, `platforms ∋ dgx_h100`, and prints one TSV
line per case:

```text
<test_case>\t<model>\t<TRAINING_SCRIPT_PATH>\t<n_repeat>
```

Run it via `uv` to pick up `pyyaml` on the fly without touching the project
venv:

```bash
cd "$COG_MEGATRON_REPO"
uv run --no-project --with pyyaml python \
  .claude/skills/run-inference-functional-tests/list_inference_tests.py \
  | tee /tmp/inference-tests.tsv
wc -l /tmp/inference-tests.tsv   # ~45 rows as of 2026-05
```

The `gpt-dynamic-inference-cuda-graphs.yaml` recipe is intentionally skipped
by the helper — its only test is marked `mr-broken` and its `script` block
inlines a one-off `torch.distributed.run` invocation rather than going
through `run_ci_test.sh`, so the rest of this skill's machinery does not
apply.

---

## Step 5 — Submit one functional test through cog

Two things make functional tests harder than unit tests through cog:

1. **`run_ci_test.sh` hardcodes `/usr/local/bin/yq`, and `_run_training.sh`
   calls `envsubst`.** Neither exists in the default NGC PyTorch base image
   (`nvcr.io/nvidia/pytorch:26.04-py3`) cog imports — they only exist in the
   CI-built `mcore_ci_dev` image. The submit command below installs both at
   the start of the job (yq: static binary download; envsubst: tiny
   `gettext-base` apt package).
2. **CI mounts cluster test data at `/mnt/artifacts`; cog does not.** Every
   `model_config.yaml` for inference references
   `${CHECKPOINT_LOAD_PATH}/model/mcore_mistral/...` which (via the recipe's
   `ARGUMENTS=` block) resolves to `/mnt/artifacts/model/...`. We pass the
   user-confirmed `$COG_ARTIFACTS_ROOT` from Step 2 directly as
   `CHECKPOINT_LOAD_PATH`/`DATA_PATH` and mount it into the container via
   `COG_EXTRA_MOUNTS` so it resolves both inside the container *and* matches
   the envsubst substitution inside `model_config.yaml`.

Driver function (paste into your bash, then call it). It reads
`$COG_ARTIFACTS_ROOT` (populated in Step 2). If unset, it bails — **do not
hardcode an artifacts path here**; always come from Step 2 so the user's
choice is honored:

```bash
# Args:  test_case  model  training_script_path
# Env:   $LIGHTWEIGHT_MODE=true (default false) — for MODE=inference this
#        flips ENV_VARS.SKIP_PYTEST=1 so the goldens comparison step is
#        skipped. The training/inference itself still runs and still writes
#        $INFERENCE_OUTPUT_PATH. Use this when bootstrapping a brand-new
#        test case (see add-inference-functional-tests skill).
submit_inference_test() {
  local TEST_CASE="$1" MODEL="$2" TRAINING_SCRIPT_PATH="$3"
  local LIGHTWEIGHT="${LIGHTWEIGHT_MODE:-false}"

  if [ -z "${COG_ARTIFACTS_ROOT:-}" ]; then
    echo "COG_ARTIFACTS_ROOT is not set — go back to Step 2 and ask the user" >&2
    return 2
  fi
  local ARTIFACTS="$COG_ARTIFACTS_ROOT"

  source ~/.cog/setup.env
  export COG_EXTRA_MOUNTS="$ARTIFACTS:$ARTIFACTS"

  cog --pretty submit \
    --repo "$COG_MEGATRON_REPO" \
    --run-name "ftest-${TEST_CASE}" \
    --gpus 8 --nodes 1 --ntasks-per-node 1 \
    --time 00:30:00 \
    --partition "$COG_INTERACTIVE_PARTITION" \
    --command "set -euo pipefail
if ! test -x /usr/local/bin/yq; then
  curl -fsSL https://github.com/mikefarah/yq/releases/download/v4.44.1/yq_linux_amd64 -o /usr/local/bin/yq
  chmod +x /usr/local/bin/yq
fi
if ! command -v envsubst >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq --no-install-recommends gettext-base
fi
ARGUMENTS=(
    CHECKPOINT_LOAD_PATH=$ARTIFACTS
    CHECKPOINT_SAVE_PATH=\$RUN_DIR/checkpoints
    DATA_PATH=$ARTIFACTS
    DATA_CACHE_PATH=\$RUN_DIR/data_cache
    TRAINING_SCRIPT_PATH=$TRAINING_SCRIPT_PATH
    TRAINING_PARAMS_PATH=./tests/functional_tests/test_cases/$MODEL/$TEST_CASE/model_config.yaml
    GOLDEN_VALUES_PATH=./tests/functional_tests/test_cases/$MODEL/$TEST_CASE/golden_values_dev_dgx_h100.json
    OUTPUT_PATH=\$RUN_DIR
    TENSORBOARD_PATH=\$RUN_DIR/tensorboard
    INFERENCE_OUTPUT_PATH=\$RUN_DIR/inference_output.json
    N_REPEAT=1
    ENABLE_LIGHTWEIGHT_MODE=$LIGHTWEIGHT
    RECORD_CHECKPOINTS=false
)
mkdir -p \"\$RUN_DIR/checkpoints\" \"\$RUN_DIR/data_cache\" \"\$RUN_DIR/tensorboard\"
bash ./tests/functional_tests/shell_test_utils/run_ci_test.sh \"\${ARGUMENTS[@]}\"
"
}
```

### Why `--gpus 8` for what looks like a 1-GPU test

The recipes' top-level `gpus: 1` field is mostly metadata — CI's nemo-run
launcher uses `--num_gpus=-1` (= all 8 GPUs on the node) regardless, and
`tests/functional_tests/shell_test_utils/_run_training.sh` defaults
`GPUS_PER_NODE=8` and feeds that to `torch.distributed.run --nproc_per_node`.
If you allocate only 1 GPU through cog but `_run_training.sh` still launches
8 worker processes, ranks 1-7 die immediately with
`torch.AcceleratorError: CUDA error: invalid device ordinal`. Allocate the
full node (8 GPUs) to match CI behavior.

### Why every flag in `ARGUMENTS=(…)` is required

`run_ci_test.sh` validates every entry in its `MANDATORY_VARS` list and
exits 1 with `Providing $X is mandatory.` if any is empty. The first
unhelpful empty I tripped over was `ENABLE_LIGHTWEIGHT_MODE=` (empty
string) — pass `false` explicitly. The same holds for `RECORD_CHECKPOINTS`.

---

## Step 6 — Run the full inference suite

Sequentially (one cog submit at a time, ~5-15 min per test, simpler to debug):

```bash
source ~/.cog/setup.env
cd "$COG_MEGATRON_REPO"

uv run --no-project --with pyyaml python \
  .claude/skills/run-inference-functional-tests/list_inference_tests.py \
  > /tmp/inference-tests.tsv

declare -a RESULTS=()
while IFS=$'\t' read -r TEST_CASE MODEL TRAINING_SCRIPT N_REPEAT; do
  [ -z "$TEST_CASE" ] && continue
  echo "===== $TEST_CASE ====="
  if submit_inference_test "$TEST_CASE" "$MODEL" "$TRAINING_SCRIPT" \
        2>&1 | tee "/tmp/ftest-$TEST_CASE.log" \
        | grep -q '"returncode": 0'; then
    RESULTS+=("PASS  $TEST_CASE")
  else
    RESULTS+=("FAIL  $TEST_CASE")
  fi
done < /tmp/inference-tests.tsv

printf '%s\n' "${RESULTS[@]}"
```

For faster turnaround, submit in parallel (cog uses connection multiplexing,
so concurrent submits are cheap; the limit is your Slurm fair-share queue,
not cog):

```bash
while IFS=$'\t' read -r TEST_CASE MODEL TRAINING_SCRIPT N_REPEAT; do
  [ -z "$TEST_CASE" ] && continue
  ( submit_inference_test "$TEST_CASE" "$MODEL" "$TRAINING_SCRIPT" \
      > "/tmp/ftest-$TEST_CASE.log" 2>&1 ) &
done < /tmp/inference-tests.tsv
wait

# Tally
for f in /tmp/ftest-*.log; do
  TC=$(basename "$f" .log); TC=${TC#ftest-}
  if grep -q '"returncode": 0' "$f"; then echo "PASS  $TC"; else echo "FAIL  $TC"; fi
done
```

---

## Step 7 — Read failures

For any FAIL row, fetch the slurm log and look at the standard pytest /
torchrun signatures:

```bash
TEST_CASE=<the_failing_one>
JOB_ID=$(grep -oE '"job_id": "[0-9]+"' /tmp/ftest-$TEST_CASE.log | head -1 \
         | grep -oE '[0-9]+')
cog logs slurm --run-name "ftest-$TEST_CASE" --job-id "$JOB_ID" \
  --stream both --lines 500
```

Common patterns to grep for:

| Pattern | Meaning |
|---------|---------|
| `Providing $X is mandatory.` | An `ARGUMENTS=(…)` flag is missing or empty — re-check the driver |
| `tokenizer_path: … is invalid` | Cluster artifacts not mounted, or `$COG_ARTIFACTS_ROOT` points at the wrong dir — re-run Step 2 |
| `torch.AcceleratorError: CUDA error: invalid device ordinal` | Allocated < 8 GPUs but `_run_training.sh` is launching 8 ranks |
| `pytest: command not found` / `.partial.<hash>… not found` | Step 2 not done (or the venv was rebuilt) — re-run Step 2 |
| `AssertionError: Mismatch` (in `test_inference_regular_pipeline.py`) | Real golden-value divergence — your change moved numerics |

---

## Common pitfalls

- **First submit per cluster is slow (~5-10 min)** because cog runs
  `enroot import` + `uv sync` once. Subsequent submits start in seconds.
  Pre-warm via `cog prepare-image` + `cog ensure-env` from the
  [[cog-setup-and-help]] skill so the slow path is out of the critical path.
- **Job exits 127 with `/usr/local/bin/yq: No such file or directory`** —
  the submit command's `yq` install step didn't run. Confirm the heredoc
  in `submit_inference_test` is intact and the `curl` line is reachable
  from the cluster (it usually is — gitlab/github outbound HTTPS is open
  on `cw-dfw`).
- **`run_ci_test.sh` exits 1 right after `Running pytest checks against
  golden values`** with `1 failed`. Look at the *actual* metric diff in the
  log: `assert <metric> ≈ <golden>` — this is the test doing its job. Either
  your change is wrong, or the goldens need refreshing (see the
  `update-golden-values` skill).
- **Cog `job.returncode: 0` but the test "failed"** — should not happen for
  this skill. If you see it, the pytest comparison was skipped (e.g.
  `SKIP_PYTEST=1` set somewhere or the recipe's `script` block changed
  shape). Re-read the `model_config.yaml`'s `ENV_VARS`.
- **`stale env-ready marker` warning on the cog submit** — harmless if the
  `pyproject.toml` hasn't changed. Run `cog ensure-env --repo
  $COG_MEGATRON_REPO --run-name env-warmup --gpus 1 --time 00:20:00
  --partition $COG_INTERACTIVE_PARTITION` to refresh.
- **Container can't reach github.com to download yq.** Switch to the
  Slurm-side cached binary if you've pre-staged one under
  `$COG_SCRATCH_ROOT/bin/yq`; mount that path via `COG_EXTRA_MOUNTS` and
  `cp` it to `/usr/local/bin/yq` instead of `curl`.

## When in doubt

- Cog command details / flags / output fields / error codes: read
  [[cog-setup-and-help]] — it is the canonical in-tree cog reference.
  Only fall through to `~/cog/docs/cli-guide.md` for details that
  skill doesn't cover.
- Recipe YAML shape, scope/environment/platform vocabulary, lightweight
  mode: the **testing** skill.
- SLURM-level details (sbatch, multi-node, `CUDA_DEVICE_MAX_CONNECTIONS`):
  the **run-on-slurm** skill.
- Refreshing golden values after a real numeric change: the
  **update-golden-values** skill.
