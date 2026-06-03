---
name: run-inference-performance-tests
description: Run Megatron-LM inference performance tests (every test_case from the H100 perf recipes that drives run_perf_test.sh) on a Slurm cluster via cog, sweep batch sizes, and compare against checked-in baseline_values.json. Auto-bootstraps cog via the cog-setup-and-help skill if cog is not installed or ~/.cog/setup.env is missing. Supports running a single test case (<family>/<test_case>) or the whole inference-perf suite.
when_to_use: Running the inference performance-test suite end-to-end on the cluster from a laptop; bootstrapping or refreshing a perf baseline; investigating an inference throughput / latency regression; 'run inference perf tests', 'run inference performance tests', 'inference benchmark', 'inference throughput regression', 'baseline_values.json'.
---

# Run Megatron-LM inference performance tests on the cluster via cog

The CI Slurm bucket that runs every
`tests/test_utils/recipes/h100/*perf*.yaml` recipe drives the OpenAI-style
inference perf harness end-to-end against its checked-in
`baseline_values.json`. Each test spins up the dynamic text-generation
server with a real checkpoint, fires concurrent `/v1/completions`
requests at a sweep of batch sizes, and compares
throughput / latency / TPOT against the baseline (default 10% lower
tolerance, 20% upper tolerance on throughput).

This skill submits the same tests from your laptop via cog so you can
verify inference perf changes (engine, sampling, dynamic batching,
attention backend, …) before pushing.

The first part of this skill is a hard dependency on **`cog-setup-and-help`**.
If cog is missing or `~/.cog/setup.env` is not populated, invoke
[[cog-setup-and-help]] first; do not proceed until those preconditions hold.

> **🔁 Iterating on the same perf test repeatedly?** Don't `cog submit`
> in a loop — each submit re-queues for a fresh Slurm allocation
> (minutes of latency per attempt). Instead, `cog session start` once
> with a 2-3 hour wall-clock budget on the same partition / GPU count,
> then `cog session exec` to re-run `run_perf_test.sh` for each
> iteration. Especially useful when you're tuning `BATCH_SIZES`,
> `RECORD_BASELINE`, or chasing a server-startup error. See the
> "Iterating" section in [[cog-setup-and-help]] for the exact recipe —
> that skill is the canonical in-tree cog reference (commands, flags,
> error codes).

The second part follows the same artifacts-root + per-test-case
submission pattern as [[run-inference-functional-tests]] — every
`model_config.yaml` references `${CHECKPOINT_LOAD_PATH}/model/...` paths
which must resolve against the same on-cluster mcore_ci tree the CI
recipes use.

---

## Step 1 — Verify cog is ready (or bootstrap it)

Run all three checks. **If any check fails, invoke the `cog-setup-and-help` skill and
re-run the checks before proceeding.**

```bash
if ! test -f ~/.cog/setup.env || \
   ! grep -q '^export COG_MEGATRON_REPO=' ~/.cog/setup.env || \
   ! grep -q '^export COG_BATCH_PARTITION=' ~/.cog/setup.env; then
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

Every inference perf `model_args/<MODEL>.args` file references
`${CHECKPOINT_LOAD_PATH}/model/...` (tokenizer, checkpoint, etc.). The
`run_perf_test.sh` driver expects `CHECKPOINT_LOAD_PATH=` as a positional
arg pointing at the **artifacts root** on the cluster — the same tree
the CI `.yaml` recipes mount at `/mnt/artifacts/`.

**Default (used by NVIDIA CI on cw-dfw):**

```text
/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_mcore/mcore_ci
```

**Procedure (the agent running this skill must do this — do not skip):**

1. Ask the user, via `AskUserQuestion`, whether to use the default path
   or a different one. Make the default the first option and recommend
   it.
2. **Never attempt to download, copy, mirror, or stage checkpoints
   yourself.** The artifacts tree is large, the file layout under
   `mcore_ci/model/...` is opinionated, and baselines are tied to the
   exact checkpoints CI uses. If the default path is not accessible,
   **stop and ask the user to point you at their copy** rather than
   guessing.
3. Verify the chosen path actually exists on the cluster and contains a
   `model/` subtree before proceeding. Bail out with a clear error if it
   doesn't.
4. Export it as `COG_ARTIFACTS_ROOT` so the rest of this skill picks it
   up consistently:

```bash
export COG_ARTIFACTS_ROOT="${COG_ARTIFACTS_ROOT:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_mcore/mcore_ci}"

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

---

## Step 3 — Enumerate inference perf test cases

This skill ships a helper, `list_inference_perf_tests.py`, that walks
every recipe matching `tests/test_utils/recipes/{h100,gb200}/*perf*.yaml`,
keeps the ones that actually drive `run_perf_test.sh` (skipping
module-level perf recipes like `module_performance.yaml`), filters to
`scope ∋ {mr, mr-github}`, `environment ∋ dev`, `platforms ∋
{dgx_h100, dgx_gb200}`, and prints one TSV line per (test_case,
platform):

```text
<family>/<test_case>\t<gpus>\t<platform>
```

Run via `uv` to pick up `pyyaml` on the fly without touching the project
venv:

```bash
cd "$COG_MEGATRON_REPO"
uv run --no-project --with pyyaml python \
  skills/run-inference-performance-tests/list_inference_perf_tests.py \
  | tee /tmp/inference-perf-tests.tsv
```

As of 2026-05 this emits 8 (test_case, platform) rows:

| `<family>/<test_case>` | GPUs (world size) | Platform | Notes |
|---|---|---|---|
| `gpt/gpt_583m_perf` | 8 | `dgx_h100` | DP=8 dynamic-batching path with ZMQ DP coordinator (H100 only — 8 GPUs/node) |
| `gpt/gpt_583m_perf_gb200_4gpu` | 4 | `dgx_gb200` | DP=4 GB200 variant (single node, 4 GPUs/node) |
| `gpt/gpt_16b_perf` | 1 | `dgx_h100` | 16B MoE deepseek-style (gsm8k dataset) |
| `gpt/gpt_16b_perf` | 1 | `dgx_gb200` | Same test case, GB200 baseline subtree |
| `hybrid/hybrid_2b_perf` | 1 | `dgx_h100` | mamba hybrid 2B (gsm8k dataset) |
| `hybrid/hybrid_2b_perf` | 1 | `dgx_gb200` | Same test case, GB200 baseline subtree |
| `hybrid/hybrid_nanov3_3b_perf` | 8 | `dgx_h100` | mamba hybrid MoE EP=8 (H100 only) |
| `hybrid/hybrid_nanov3_3b_perf_gb200_4gpu` | 4 | `dgx_gb200` | EP=4 GB200 variant |

**Filter the TSV by platform** before iterating, so you only submit
the tests that match the cluster you're pointed at:

```bash
# H100 (cw-dfw):  awk -F'\t' '==:"dgx_h100"' /tmp/inference-perf-tests.tsv
# GB200 (oci-hsg): awk -F'\t' '==:"dgx_gb200"' /tmp/inference-perf-tests.tsv
```

---

## Step 4 — Submit one inference perf test through cog

Three things make perf tests harder than unit tests through cog:

1. **`run_perf_test.sh` requires `/usr/local/bin/yq`.** It doesn't exist
   in the default NGC PyTorch base image (`nvcr.io/nvidia/pytorch:…`)
   cog imports — only in the CI-built `mcore_ci_dev` image. We install
   the static binary inline at the start of the job (~3 seconds, free
   cache hit on repeat).
2. **CI mounts cluster test data at `/mnt/artifacts`; cog does not.**
   Every `model_args/<MODEL>.args` file references
   `${CHECKPOINT_LOAD_PATH}/model/...` — we pass the user-confirmed
   `$COG_ARTIFACTS_ROOT` from Step 2 directly as `CHECKPOINT_LOAD_PATH`
   and mount it identity-mapped into the container via `COG_EXTRA_MOUNTS`.
3. **`run_perf_test.sh` launches its own `torch.distributed.run`.** Pass
   `--ntasks-per-node 1` so cog allocates one srun task per node, then
   the script spawns `WORLD_SIZE = TP*PP*max(DP,EP)` worker processes
   itself. Wrong here and you get either no distributed group (cog
   launched nothing) or 64 ranks (cog spawned 8 srun tasks, each
   re-launching 8 torchrun workers).

Driver function (paste into your bash, then call it). It reads
`$COG_ARTIFACTS_ROOT` (populated in Step 2). If unset, it bails — **do
not hardcode an artifacts path here**; always come from Step 2 so the
user's choice is honored.

```bash
# Args: test_id (e.g. "gpt/gpt_583m_perf") gpus [extra_perf_env]
# extra_perf_env is an optional bash word like "RECORD_BASELINE=1" or
# "SKIP_COMPARE=1" — passed verbatim into the ARGUMENTS positional list
# that run_perf_test.sh consumes.
submit_inference_perf_test() {
  local TEST_ID="$1" GPUS="$2" EXTRA_PERF_ENV="${3:-}"
  local FAMILY="${TEST_ID%%/*}" TEST_CASE="${TEST_ID#*/}"

  if [ -z "${COG_ARTIFACTS_ROOT:-}" ]; then
    echo "COG_ARTIFACTS_ROOT is not set — go back to Step 2 and ask the user" >&2
    return 2
  fi
  local ARTIFACTS="$COG_ARTIFACTS_ROOT"
  local SAFE_NAME
  SAFE_NAME=$(echo "$TEST_ID" | tr '/' '-')

  source ~/.cog/setup.env
  export COG_EXTRA_MOUNTS="$ARTIFACTS:$ARTIFACTS"

  # Some clusters (oci-hsg / GB200) enforce a per-job minimum GPU count via QOS
  # because nodes are full reservations. Set COG_GPUS_PER_NODE_LIMIT=4 (or
  # whatever the node holds) in ~/.cog/setup.env on those clusters; the driver
  # rounds $GPUS up to that floor for Slurm while torchrun inside the container
  # still only spawns the WORLD_SIZE the model_config.yaml computes.
  local ALLOC_GPUS="$GPUS"
  if [ -n "${COG_GPUS_PER_NODE_LIMIT:-}" ] && [ "$ALLOC_GPUS" -lt "$COG_GPUS_PER_NODE_LIMIT" ]; then
    ALLOC_GPUS="$COG_GPUS_PER_NODE_LIMIT"
  fi

  # Use $COG_BATCH_PARTITION (not interactive) because perf tests need
  # stable, exclusive-ish node access to produce reproducible numbers.
  cog --pretty submit \
    --repo "$COG_MEGATRON_REPO" \
    --run-name "perf-${SAFE_NAME}" \
    --gpus "$ALLOC_GPUS" --nodes 1 --ntasks-per-node 1 \
    --time 01:00:00 \
    --partition "$COG_BATCH_PARTITION" \
    --command "set -euo pipefail
# yq isn't in the bare NGC PyTorch image (only in mcore_ci_dev). Pick the right
# binary for the host arch — GB200 / Grace nodes are aarch64, H100 are x86_64.
if ! test -x /usr/local/bin/yq; then
  ARCH=\$(uname -m)
  case \"\$ARCH\" in
    aarch64) YQ_ASSET=yq_linux_arm64 ;;
    x86_64)  YQ_ASSET=yq_linux_amd64 ;;
    *) echo \"unsupported arch: \$ARCH\" >&2; exit 2 ;;
  esac
  curl -fsSL https://github.com/mikefarah/yq/releases/download/v4.44.1/\$YQ_ASSET -o /usr/local/bin/yq
  chmod +x /usr/local/bin/yq
fi
ARGUMENTS=(
    \"CONFIG_PATH=tests/performance_tests/test_cases/$FAMILY/$TEST_CASE/model_config.yaml\"
    \"CHECKPOINT_LOAD_PATH=$ARTIFACTS\"
    \"RESULTS_ROOT=\$RUN_DIR/perf_results\"
    ${EXTRA_PERF_ENV:+\"$EXTRA_PERF_ENV\"}
)
GPUS_PER_NODE=$GPUS bash ./tests/performance_tests/shell_test_utils/run_perf_test.sh \"\${ARGUMENTS[@]}\"
"
}
```

### Why `GPUS_PER_NODE=$GPUS` and `--ntasks-per-node 1`

`run_perf_test.sh` reads `GPUS_PER_NODE` only as a documentation
breadcrumb — the actual world size is computed from
`TP*PP*max(DP,EP)` in `model_config.yaml`. But the recipes set
`GPUS_PER_NODE` for the same reason: it must match cog's `--gpus` so
the cluster scheduler reserves enough devices for the torchrun worker
group the script will launch. `--ntasks-per-node 1` keeps cog out of
the torchrun launch loop.

### Why `--partition $COG_BATCH_PARTITION`

Perf measurements need exclusive-ish node access for stable throughput
numbers. The interactive partition often shares nodes, which inflates
run-to-run noise above the harness's 10% regression floor and produces
false positives. Use batch.

### `EXTRA_PERF_ENV` knob

`run_perf_test.sh` parses `KEY=VALUE` positional args including
`RECORD_BASELINE=1` (overwrite `baseline_values.json` in the test case
dir) and `SKIP_COMPARE=1` (emit `results.json` without asserting). Pass
either through the third positional argument of the driver:

```bash
submit_inference_perf_test gpt/gpt_583m_perf 8 RECORD_BASELINE=1
submit_inference_perf_test hybrid/hybrid_2b_perf 1 SKIP_COMPARE=1
```

`RECORD_BASELINE=1` writes to the **synced workspace on cluster
scratch** (`$COG_SCRATCH_ROOT/workspaces/megatron_lm/<hash>/repo/...`).
`scp` the resulting file back into your local checkout to commit it —
see [[run-performance-tests]] for the exact path.

---

## Step 5 — Run modes

### 5a. Single test case (arg form)

When the user invokes the skill with `<family>/<test_case>`, look up
the GPU count from the TSV in Step 3 and run that one test:

```bash
TEST_ID="gpt/gpt_583m_perf"   # from user
GPUS=$(awk -F'\t' -v t="$TEST_ID" '$1==t {print $2; exit}' /tmp/inference-perf-tests.tsv)
if [ -z "$GPUS" ]; then
  echo "Unknown test case '$TEST_ID' — must be one of:" >&2
  cut -f1 /tmp/inference-perf-tests.tsv | sed 's/^/  /' >&2
  exit 2
fi
submit_inference_perf_test "$TEST_ID" "$GPUS" 2>&1 | tee "/tmp/perf-$(echo $TEST_ID | tr / -).log"
```

### 5b. Whole inference-perf suite (no arg)

Sequentially (one cog submit at a time, ~5-15 min per test, simpler to
debug):

```bash
source ~/.cog/setup.env
cd "$COG_MEGATRON_REPO"

uv run --no-project --with pyyaml python \
  skills/run-inference-performance-tests/list_inference_perf_tests.py \
  > /tmp/inference-perf-tests.tsv

declare -a RESULTS=()
while IFS=$'\t' read -r TEST_ID GPUS; do
  [ -z "$TEST_ID" ] && continue
  echo "===== $TEST_ID (gpus=$GPUS) ====="
  SAFE_NAME=$(echo "$TEST_ID" | tr '/' '-')
  if submit_inference_perf_test "$TEST_ID" "$GPUS" \
        2>&1 | tee "/tmp/perf-$SAFE_NAME.log" \
        | grep -q '"returncode": 0'; then
    RESULTS+=("PASS  $TEST_ID")
  else
    RESULTS+=("FAIL  $TEST_ID")
  fi
done < /tmp/inference-perf-tests.tsv

printf '%s\n' "${RESULTS[@]}"
```

For faster turnaround, submit in parallel:

```bash
while IFS=$'\t' read -r TEST_ID GPUS; do
  [ -z "$TEST_ID" ] && continue
  SAFE_NAME=$(echo "$TEST_ID" | tr '/' '-')
  ( submit_inference_perf_test "$TEST_ID" "$GPUS" \
      > "/tmp/perf-$SAFE_NAME.log" 2>&1 ) &
done < /tmp/inference-perf-tests.tsv
wait

for f in /tmp/perf-*.log; do
  TC=$(basename "$f" .log); TC=${TC#perf-}
  if grep -q '"returncode": 0' "$f"; then echo "PASS  $TC"; else echo "FAIL  $TC"; fi
done
```

---

## Step 6 — Read failures and report

For any FAIL row, fetch the slurm log:

```bash
TEST_ID=<the_failing_one>
SAFE_NAME=$(echo "$TEST_ID" | tr '/' '-')
JOB_ID=$(grep -oE '"job_id": "[0-9]+"' /tmp/perf-$SAFE_NAME.log | head -1 \
         | grep -oE '[0-9]+')
cog logs slurm --run-name "perf-$SAFE_NAME" --job-id "$JOB_ID" \
  --stream both --lines 500
```

Common signatures in the slurm log:

| Pattern | Meaning |
|---|---|
| `server did not become ready within 15 min` | Server crashed during init. Check `$RUN_DIR/perf_results/server_logs/server.log` for the traceback (bad `--load` path, checkpoint format mismatch, TE/CUDA mismatch) |
| `aiohttp.client_exceptions.ServerDisconnectedError` mid-benchmark | Server died under load. Look for OOM in `server.log` — `--inference-dynamic-batching-buffer-size-gb` too high, or the model + KV cache won't fit the chosen batch size |
| `yq: not found` | The inline yq install at the top of the submit command didn't run. Confirm the heredoc isn't truncated; verify cluster can reach `github.com` |
| `ImportError: MambaSSM is not installed` (hybrid/mamba tests) | `run_perf_test.sh` installs a `.pth` shim in the cog venv pointing at `/opt/venv/lib/python3.12/site-packages`. If you see this error, grep the perf log for `installing mamba-ssm shim .pth` — if missing, `$VIRTUAL_ENV` was unset (cog change) and the shim block was skipped |
| `tokenizer_path: … is invalid` | `$COG_ARTIFACTS_ROOT` is wrong or the lustre mount isn't there. Re-run Step 2 |
| `error: no baseline_values.json at …` | `compare_to_baseline.py` couldn't find a baseline. Either the test is brand-new (run with `RECORD_BASELINE=1` first — see [[run-performance-tests]]) or the file wasn't checked into the workspace |
| `Throughput regression … below baseline_X * 0.9` | Real perf regression. Use `SKIP_COMPARE=1` to gather numbers without asserting, then bisect |
| `Improvement too large — refresh baseline` | Throughput exceeded `baseline * 1.2`. Intentional → refresh baseline with `RECORD_BASELINE=1`. Unintentional → investigate why before refreshing |

Report format (when reporting back to the user):

```markdown
## Inference perf-test results

**Job(s):** <job_id_1>, <job_id_2>, …
**Results:**
- PASS  gpt/gpt_583m_perf
- PASS  gpt/gpt_16b_perf
- FAIL  hybrid/hybrid_2b_perf  (throughput regression at batch 32)
- PASS  hybrid/hybrid_nanov3_3b_perf

### Failures
<for each fail, copy the per-metric diff block from compare_to_baseline.py>
```

---

## Common pitfalls

- **First submit per cluster is slow (~5-10 min)** because cog runs
  `enroot import` + `uv sync` once. Subsequent submits start in
  seconds. Pre-warm via `cog prepare-image` + `cog ensure-env` from
  [[cog-setup-and-help]] so the slow path is out of the critical path.
- **Allocating only 1 GPU for a DP=8 test.** `run_perf_test.sh` computes
  `WORLD_SIZE = TP*PP*max(DP,EP)` from `model_config.yaml`. If
  `--gpus 1` doesn't cover that, ranks ≥ 1 die immediately with
  `invalid device ordinal`. Always use the GPU count from the helper
  (Step 3), never hardcode 1 or 8.
- **Hardcoding `cw-dfw` / `/lustre/.../mcore_ci` / `coreai_dlalgo_llm`
  in `cog submit`.** Read `$COG_*` from `~/.cog/setup.env` and
  `$COG_ARTIFACTS_ROOT` from Step 2. The [[cog-setup-and-help]] skill exists
  precisely so cluster-specific values are not embedded in this skill.
- **Container can't reach github.com to download yq.** Cluster
  outbound HTTPS to github.com / gitlab-master.nvidia.com is normally
  open on cw-dfw. If it isn't on your cluster, pre-stage the yq binary
  under `$COG_SCRATCH_ROOT/bin/yq`, add it to `COG_EXTRA_MOUNTS`, and
  replace the `curl` line with `cp $COG_SCRATCH_ROOT/bin/yq /usr/local/bin/yq`.
- **Trying to refresh a baseline locally.** `RECORD_BASELINE=1` writes
  inside the synced cluster workspace, not your laptop. You must `scp`
  the file back into your local checkout before `git commit` will see
  the change. See the matching section of [[run-performance-tests]] for
  the path layout (`$COG_SCRATCH_ROOT/workspaces/megatron_lm/<hash>/repo/...`).
- **`cog job.returncode: 0` but the test "failed".** Either
  `SKIP_COMPARE=1` was set (no comparison ran) or
  `compare_to_baseline.py` was already past — re-read the slurm log
  for the actual exit signal.
- **Run-to-run noise above the 10% floor.** H100s show ~3-5% run-to-run
  noise on throughput. If a regression is just over the line, re-run
  once. Bigger drift → real regression or stale baseline; bisect with
  `SKIP_COMPARE=1` to gather numbers without asserting.

### GB200 / oci-hsg-specific gotchas

These bit us onboarding oci-hsg (GB200, ARM64). They also apply to any
future Grace-Blackwell or ARM cluster.

- **`yq_linux_amd64` won't run on ARM nodes.** The inline yq install in
  the cog submit command must pick the right binary for the host arch.
  Use `uname -m` to dispatch — `x86_64 → yq_linux_amd64`,
  `aarch64 → yq_linux_arm64`. Symptom: `cannot execute binary file:
  Exec format error` from line 1 of `run_perf_test.sh`.
- **GB200 QOS rejects sub-node allocations.** All QOS on `oci-hsg` set
  `MinTRES=gres/gpu=4`. Even a `DP=1` perf test must request `--gpus 4`
  from cog (full node); torchrun inside still only spawns
  `WORLD_SIZE = TP*PP*max(DP,EP)` workers. Symptom on too-small alloc:
  `QOSMinGRES` / `Unable to allocate resources: Job violates
  accounting/QOS policy` at submit time. The launcher script in this
  skill hardcodes `ALLOC_GPUS=4` for that reason.
- **MoE checkpoints with `--expert-model-parallel-size N` in their
  `.args` file pin world size to `N`.** `run_perf_test.sh` overrides
  TP/PP/EP from `model_config.yaml` so the config wins, but if you ever
  hand-build a `--command` that doesn't go through `run_perf_test.sh`,
  remember the override. Symptom: `RuntimeError: world_size (4) is not
  divisible by expert_tensor_model_pipeline_parallel size (8)`.
- **Mamba-ssm is missing from the bare NGC PyTorch image on GB200.**
  The H100 `mcore_ci_dev` image happens to have prebuilt extras in
  `/opt/venv/lib/python3.12/site-packages` and the skill's shim
  `.pth` trick exposes them. On GB200's ARM image `/opt/venv` doesn't
  exist, so `run_perf_test.sh` falls back to
  `uv pip install --no-build-isolation mamba-ssm causal-conv1d`. Relies
  on PyPI's `linux_aarch64` wheels (mamba-ssm ≥ 2.2.2, causal-conv1d
  ≥ 1.4.0). Adds ~30–60 s to the first hybrid run after a clean env;
  cached on later runs.
- **8-GPU H100 tests don't fit on GB200 (4 GPUs/node).** The 8-GPU
  recipes (`gpt-perf-dp8.yaml`, `hybrid-perf-ep8.yaml`) have no
  single-node GB200 counterpart. The skill ships matching
  `gpt-perf-dp4.yaml` / `hybrid-perf-ep4.yaml` with different
  `test_case` dirs (`gpt_583m_perf_gb200_4gpu`,
  `hybrid_nanov3_3b_perf_gb200_4gpu`) and their own baselines.
- **`baseline_values.json` is platform-keyed.** Schema is
  `{ "h100": { batch_1: {...}, ... }, "gb200": { ... } }`.
  `run_perf_test.sh` auto-detects `PLATFORM` from `nvidia-smi -L`
  (override via `PLATFORM=<name>` arg) and merges only that subtree on
  `RECORD_BASELINE=1`, preserving other platforms' numbers.
  `compare_to_baseline.py --platform` picks the subtree to compare.

---

## When in doubt

- For the single-test perf harness reference (test cases table, dataset
  knobs, baseline refresh procedure, full pitfall list including
  `flash-decode` and EP-shaped world sizes): read
  [[run-performance-tests]] in this repo.
- For cog command details / flags / output fields / error codes: read
  [[cog-setup-and-help]] — it is the canonical in-tree cog reference.
  Only fall through to `~/cog/docs/cli-guide.md` for details that
  skill doesn't cover.
- For recipe YAML shape, scope/environment/platform vocabulary: read
  the **testing** skill.
- For SLURM-level details (sbatch, multi-node,
  `CUDA_DEVICE_MAX_CONNECTIONS`): read the **run-on-slurm** skill.
