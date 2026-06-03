---
name: run-inference-unit-tests
description: Run the Megatron-LM inference unit-test suite (tests/unit_tests/inference) on a Slurm cluster via cog, collecting pytest-cov coverage of megatron.core.inference, and report both pass/fail counts and the coverage summary. Auto-bootstraps cog via the cog-setup-and-help skill if cog is not installed or ~/.cog/setup.env is missing.
when_to_use: Running the inference unit-test suite end-to-end on the cluster; producing a coverage number for megatron/core/inference; validating inference changes before pushing; 'run inference tests', 'inference coverage', 'inference unit tests', 'test inference', 'coverage for inference'.
---

# Run Megatron-LM inference unit tests with coverage

This skill submits the inference unit-test suite at
`tests/unit_tests/inference/` to a Slurm cluster via cog and reports
**test status** — pass / fail / skip counts and any failing test IDs.

**Coverage status:** the `pytest-cov` path is currently broken on the
container image cog uses — `--cov=megatron.core.inference` triggers
`RuntimeError: function '_has_torch_function' already has a docstring`
at `import torch._dynamo` in `tests/unit_tests/__init__.py` on every
rank, before any test runs. Reproduced with `COVERAGE_CORE=sysmon` as
well; the collision is at C-extension init, not at the tracer level.
Coverage is therefore behind an opt-in flag (`COVERAGE=1`) and is
documented in Step 2b — leave it off until the upstream image is
updated.

The first part of this skill is a hard dependency on **`cog-setup-and-help`**.
If cog is missing or `~/.cog/setup.env` is not populated, invoke
[[cog-setup-and-help]] first; do not proceed until those preconditions hold.

> **🔁 Iterating on a single failing test?** Don't `cog submit` in a
> loop — each submit re-queues for a fresh Slurm allocation. Instead,
> `cog session start` once with a 2-3 hour wall-clock budget and then
> `cog session exec` to re-run `python -m pytest <path>` for each
> iteration. Especially useful when bisecting a single failure or
> tweaking pytest filters / fixtures. See the "Iterating" section
> in [[cog-setup-and-help]] for the exact recipe — that skill is the
> canonical in-tree cog reference (commands, flags, error codes).

## Step 1 — Verify cog is ready (or bootstrap it)

Run all three checks. **If any check fails, invoke the `cog-setup-and-help`
skill and re-run the checks before proceeding.** Do not attempt to
hand-fix cog state — `cog-setup-and-help` handles the full cleanup +
re-install flow.

```bash
# (a) setup.env exists and exports the variables we depend on
if ! test -f ~/.cog/setup.env || \
   ! grep -q '^export COG_MEGATRON_REPO=' ~/.cog/setup.env || \
   ! grep -q '^export COG_INTERACTIVE_PARTITION=' ~/.cog/setup.env; then
  echo "MISSING: ~/.cog/setup.env (or required keys) — run cog-setup-and-help skill"
  exit 1
fi

# (b) cog CLI is on PATH and importable
cog --help > /dev/null 2>&1 || { echo "BROKEN: cog import failed — run cog-setup-and-help skill"; exit 1; }

# (c) cog doctor passes against the registered cluster
source ~/.cog/setup.env
cog doctor --repo "$COG_MEGATRON_REPO" 2>&1 | grep -q '"overall":"ok"' \
  || { echo "DOCTOR FAILED — run cog-setup-and-help skill"; exit 1; }

echo "cog ready: $COG_CLUSTER_NAME @ $COG_SSH_HOST"
```

## Step 2a — Submit the test job (test status only)

This is the default, reliable path. 8 GPUs, 1 node, 1 srun task (single
torchrun worker group with 8 processes), 2-hour cap (inference suite
is the slowest unit-test subset). **No coverage** — see Step 2b for
why and how to opt in.

```bash
source ~/.cog/setup.env

cog --pretty submit \
  --repo "$COG_MEGATRON_REPO" \
  --run-name inference-utests \
  --gpus 8 \
  --nodes 1 \
  --ntasks-per-node 1 \
  --time 02:00:00 \
  --partition "$COG_INTERACTIVE_PARTITION" \
  --command 'set -e
echo "===== inference unit tests ====="
python -m torch.distributed.run --nproc-per-node 8 \
  --log-dir "$TORCHRUN_LOG_DIR" \
  -m pytest \
    -m "not flaky and not flaky_in_dev" \
    --ignore=tests/unit_tests/inference/engines/test_dynamic_engine.py \
    -v -o addopts= \
    tests/unit_tests/inference/
'
```

## Step 2b — Coverage (opt-in, currently broken on the default image)

Coverage of `megatron.core.inference` would be useful, but
**`pytest-cov` is incompatible with the torch + conftest import chain
in the current container** (see header callout for the exact error).
The command that *would* be correct once the image is fixed is below;
leave it commented unless you've verified the image no longer hits the
`_has_torch_function` collision.

```bash
# DO NOT RUN until you've verified pytest-cov works on the current
# container image. Reproduced as broken with COVERAGE_CORE=sysmon as
# well — the conflict is at torch C-extension init, not the tracer.
#
# source ~/.cog/setup.env
# cog --pretty submit \
#   --repo "$COG_MEGATRON_REPO" \
#   --run-name inference-utests-cov \
#   --gpus 8 --nodes 1 --ntasks-per-node 1 \
#   --time 02:00:00 --partition "$COG_INTERACTIVE_PARTITION" \
#   --command 'set -e
# mkdir -p "$RUN_DIR/coverage"
# cat > "$RUN_DIR/coverage/.coveragerc" <<COVRC
# [run]
# parallel = True
# branch = True
# source = megatron.core.inference
# COVRC
# export COVERAGE_FILE="$RUN_DIR/coverage/.coverage"
# python -m torch.distributed.run --nproc-per-node 8 \
#   --log-dir "$TORCHRUN_LOG_DIR" \
#   -m pytest \
#     --cov=megatron.core.inference \
#     --cov-config="$RUN_DIR/coverage/.coveragerc" \
#     --cov-report= \
#     -m "not flaky and not flaky_in_dev" \
#     -v -o addopts= \
#     tests/unit_tests/inference/
# cd "$RUN_DIR/coverage"
# python -m coverage combine
# python -m coverage report --skip-empty
# python -m coverage xml -o "$RUN_DIR/coverage.xml"
# '
```

**To probe whether the image has been fixed**, submit Phase B from the
diagnostic recipe (single file, with `--cov`) and check that all 8
ranks reach `1 passed` instead of dying at `import torch._dynamo`. If
that works, uncomment the block above and use it.

> **Why `--ntasks-per-node 1` and `--nproc-per-node 8`:** cog populates
> distributed env vars (`MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`)
> per-task. We want a single torchrun launcher per node that itself
> spawns 8 worker processes — not 8 srun tasks each launching their
> own torchrun. The latter would produce 64 worker processes and break
> the distributed group.
>
> **Why `python -m torch.distributed.run`, not bare `torchrun`:** see
> the [[cog-setup-and-help]] pitfall — the venv's bin shebangs are broken by
> cog's `.partial→canonical` rename, but `python -m <module>` works.
>
> **Why `-m "not flaky and not flaky_in_dev"`:** matches the testing
> skill's recommended local-dev filter. Override if you need a strict
> CI-parity run.

## Step 3 — Report results

Once `cog submit` returns, parse its JSON output and the captured
stdout to produce a structured summary:

1. **Top-level job status** — read `job.returncode` and `job.state`
   from the cog JSON payload. `returncode: 0` + `state: completed`
   means torch.distributed.run + pytest all succeeded. Non-zero means
   at least one rank's pytest failed.

2. **Pass/fail summary** — grep the captured stdout for pytest's final
   summary line (e.g. `123 passed, 4 failed, 2 skipped in 318.42s`).
   With 8 ranks you get the line 8 times (one per rank); they should
   agree. If the stdout in the cog JSON is truncated, fetch the full
   log via:

   ```bash
   cog logs slurm --run-name inference-utests --job-id <job_id> --stream stdout --lines 5000
   ```

3. **Failing tests** (if any) — grep stdout for `FAILED ` lines
   (pytest emits one per failing test, e.g.
   `FAILED tests/unit_tests/inference/test_inference_utils.py::… - AssertionError: …`).
   De-duplicate across ranks before reporting.

4. **Format** the user-facing report as a short Markdown block:

   ```markdown
   ## Inference unit-test results

   **Job:** <job_id>, state <state>, returncode <rc>
   **Tests:** <passed> passed · <failed> failed · <skipped> skipped (<duration>s)

   <if failures>
   ### Failures
   - <test_id_1>
   - <test_id_2>
   …
   </if>
   ```

   Mention coverage explicitly as **N/A — currently disabled due to
   pytest-cov / torch import collision** so the user knows it's not an
   omission.

## Common pitfalls

- **First run is slow.** If `cog ensure-env` hasn't been called for the
  current `pyproject.toml` recipe, the submit job pays the one-time
  `uv sync` cost on top of pytest. Pre-warm the env once with the
  cog-setup-and-help skill's Step 4a, then this skill's `cog submit` skips
  straight to pytest.
- **`No data to report.` from `coverage combine`.** Either every rank
  crashed before importing `megatron.core.inference`, or `COVERAGE_FILE`
  wasn't honored (each subprocess wrote to its own `cwd`). Re-check
  the heredoc: `parallel = True` must be in `.coveragerc`, and
  `COVERAGE_FILE` must be exported **before** `torch.distributed.run`.
- **Test passes locally but skill reports failure.** Run with
  `--time 04:00:00` if the suite is being killed by the time cap, or
  drop the `-m "not flaky"` filter if a test marked flaky has now been
  fixed and you want it included.
- **Coverage looks lower than expected.** `pytest-cov` with parallel
  mode only counts lines actually executed. Tests that import a module
  but never call its functions won't bump the percentage. Use
  `coverage report --skip-empty` (already in the command) to hide
  files with 0 statements, which makes the real gaps clearer.
- **`cog submit` exits 127 with `pytest: command not found` in the
  Slurm log.** You're using bare `pytest`, not `python -m pytest`.
  The command above already uses `python -m pytest` indirectly via
  `python -m torch.distributed.run -m pytest`; if you adapted the
  command, keep the `python -m` prefix.
- **`cog doctor` shows `overall: ok` but `submit` errors out with
  `RUN_ROOT_OUTSIDE_SCRATCH`.** `$COG_SCRATCH_ROOT` in the env file
  doesn't match the cluster's registered scratch root. Re-run
  [[cog-setup-and-help]]'s Step 3a to refresh `~/.cog/setup.env`.
- **Whole-suite SIGABRT mid-run with `delete_cuda_graphs` in the
  faulthandler dump.** Observed in
  `tests/unit_tests/inference/engines/test_dynamic_engine.py` — the
  `teardown_method` calls `cuda_graphs.delete_cuda_graphs` which
  SIGABRTs the rank, and torchrun then kills every peer with SIGTERM,
  losing the partial pass/fail report. The Step 2a command above
  already passes
  `--ignore=tests/unit_tests/inference/engines/test_dynamic_engine.py`
  to skip the offending file. Drop the `--ignore` only after the
  upstream bug is fixed; verify by running that one file alone first.
  **Reproduces on both the default NGC base image
  (`nvcr.io/nvidia/pytorch:26.04-py3`) and the CI-built
  `mcore_ci_dev:main` image — confirmed empirically.** Switching base
  images is not a fix; the SIGABRT is in the test's teardown code,
  not in the env. CI is green because CI's launcher (`nemo-run`)
  retries known-transient patterns (segfault is on the list) up to 3
  times per the testing skill — cog's one-shot `submit` has no such
  retry layer, so a single teardown crash kills the whole run.

- **Tried switching to `mcore_ci_dev` and the suite still crashed?**
  That is expected — see the bullet above. The CI image *does* work
  with cog (verified: `cog ensure-env --base-image
  gitlab-master.nvidia.com/adlr/megatron-lm/mcore_ci_dev:main` builds
  cleanly; `cog submit --base-image … --skip-uv-sync` then runs
  pytest), and may help if you hit pytest-cov / `_has_torch_function`
  issues, but it does **not** fix the inference-suite SIGABRT. If you
  want to try it anyway, drop the `:5005` port from the registry URL
  (`enroot import` rejects it with "Invalid image reference") and
  run `ensure-env` *before* the first `submit` so the venv exists
  when `--skip-uv-sync` is used (otherwise the activate step fails
  with "No such file or directory").
- **Test suite finishes but the cog `job.returncode` is non-zero.**
  Expected when any test failed — pytest exits 1 with failures, and
  cog propagates that. The skill's Step 3 still produces a useful
  pass/fail report from the captured stdout; treat the non-zero exit
  as a signal to read the `FAILED ` lines, not as a sign the run was
  unusable.

## Why this skill exists

`tests/unit_tests/inference/` is the largest single subdirectory of the
unit-test suite and the most common place inference-side changes need
quick verification before pushing. The CI bucket that runs these tests
takes 30-60 min; running them through cog from your laptop with the
same 8-GPU configuration gives near-CI-parity results in roughly the
same wall time, without queueing behind the full CI pipeline.

Coverage of `megatron.core.inference` is a useful proxy for "did my
change exercise the code I think it did" — particularly when adding
new tests for a new code path. The skill bundles test status and
coverage into a single submit so you don't have to re-run the suite
twice.

## When in doubt

- For cog command details / flags / output fields / error codes:
  read [[cog-setup-and-help]] — it is the canonical in-tree cog
  reference. Only fall through to `~/cog/docs/cli-guide.md` for
  details that skill doesn't cover.
- For Megatron-LM test layout, markers, fixtures, recipe YAML: read
  the **testing** skill.
- For SLURM-level details (sbatch, CUDA_DEVICE_MAX_CONNECTIONS,
  multi-node rules): read the **run-on-slurm** skill.
