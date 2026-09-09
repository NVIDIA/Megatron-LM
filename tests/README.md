# Megatron-LM Tests

## Selective Unit Testing

Add the **`Run selective unit tests`** label to a PR to select H100 unit-test
**files** affected by the whole PR diff using `pytest-impacted`, including
transitive import dependencies. PRs without this label run the full unit-test
suite, including documentation-only PRs. CI adds the 10 baseline files in
[`unit_tests/always_run_tests.json`](unit_tests/always_run_tests.json)
to every selection, maps the union to the existing buckets, and launches only
buckets with selected files. Each file can contain multiple parametrized test
cases. Documentation-only PRs with this label run the baseline after the usual
CI authorization. `Run full unit tests` overrides selection and requests the
full suite when both labels are present.
The existing GB200 hardware-specific marker suite remains enabled separately.

Test selection runs on a CPU runner alongside the Docker image build, using its
own Python environment. It adds no packages to the GPU image. The GPU test jobs
wait for both the selection plan and a successful image build before starting.

The baseline covers basic setup, model-parallel configuration, process groups,
rank utilities, shared utilities, tensor-parallel cross entropy, transformer
configuration, GPT construction, and checkpoint mappings. Edit the JSON list to
change the baseline; entries must be unique, existing unit-test files owned by a
CI bucket. Changes to the baseline itself run the full suite for validation.

| Build or change | H100 unit tests |
| --- | --- |
| PR without `Run selective unit tests`, including documentation-only PRs | Full suite |
| PR labeled `Run selective unit tests`, source or test changes | Affected files plus the baseline |
| Documentation-only PR labeled `Run selective unit tests` | Baseline only |
| PR labeled `Run full unit tests`, including when both labels are present | Full suite |
| Merge queue, nightly/CI workload, manual dispatch | Full suite |
| Unsupported change or failed/ambiguous analysis | Full suite |

`Run tests` and `Run functional tests` retain their functional-test behavior;
they do not enable or disable unit-test selection. The older
`Run selective unit tests on latest commit` label does not enable selection.
Use `Run selective unit tests`; analysis always covers the entire PR, including
changes in earlier commits. The full-suite override takes precedence.

For labeled PRs, CI compares the exact synthetic merge commit being built and
tested with its first parent (`git diff HEAD^1 HEAD`). That parent is the
revision of `main` used to create the merge, so the comparison includes all PR
changes without including unrelated changes already on `main`. It does not
depend on the base SHA reported separately in PR metadata. The selection
artifact records `tested_sha` and `diff_base_sha`, and the job summary shows
the comparison. `diff_base_sha` is null when selection is bypassed before a
comparison base is established, such as for an unlabeled PR.

Selection is conservative: missing/invalid base commits, selector errors,
timeouts, deleted or renamed files, shared fixtures, test runners, dependency
and CI configuration changes, package `__init__.py` changes, and unsupported
files request the full matrix. Tokenizer changes also run the full suite because
their dynamic imports hide consumers from static analysis. Empty impact results for source changes also
request the full suite; the baseline cannot hide an analysis failure. The
analyzed base must be an exact ancestor of the same commit used by test jobs.
Static imports cannot prove complete runtime coverage of dynamic imports,
plugins, or monkeypatching; the full merge-queue suite remains the final check.

The selection summary reports the mode, fallback reason, affected and baseline
file counts, total selected files, buckets, and selector overhead. The dependency
graph is rebuilt on each selective run, so there is no persistent impact cache. Invalid
per-job payloads and selections that collect no runnable tests fail the job.

### Compare runs with and without selection

CI records metrics for both labeled and unlabeled runs in its workflow summary
and the `unit-test-metrics-<run-id>-<run-attempt>` artifact, retained for 90 days.
It contains `unit-test-metrics.json`, `unit-test-metrics.csv`, and
`unit-test-metrics.md`; the CSV contains one row per run. The separate
`unit-test-selection-<run-id>-<run-attempt>` artifact preserves the selection
manifest. Compare `selective_label_present` and `mode` together:
a labeled PR can still fall back to the full suite, with the reason recorded in
`selection_reason`.

Download a run's metrics with:

```bash
gh run download <run-id> --repo NVIDIA/Megatron-LM \
  --pattern 'unit-test-metrics-*' --dir ./unit-test-metrics/<run-id>
```

The report includes selected and available H100 test-file counts, selected
bucket count, selector duration, observed job counts, and job outcomes. These
file counts describe the test plan; a file can contain many parametrized cases.
Job timing comes from the actual GitHub Actions jobs:

- `sum_job_execution_seconds` totals execution time across H100 unit-test jobs.
- `job_execution_span_seconds` measures the interval from the first H100 job
  starting to the last finishing, including overlap between parallel jobs.
- `timing_complete` indicates whether every expected job has usable timing.
  Complete timing totals are null for incomplete data; any partial duration is
  reported separately as `observed_job_execution_seconds`.

Compare equivalent commits and environments when assessing the opt-in mode.
These metrics exclude GB200 jobs and do not measure queue time or predict how
long a skipped test would have taken.

### Analyze changes locally

The selector has its own locked Python environment in `.github/test-selection`.
It does not install Megatron or require GPUs for analysis, and its dependencies
do not conflict with the main package's linting environment. Run the following
inside the development container, from the repository root:

```bash
bucket_file="$(mktemp)"
selection_file="$(mktemp)"
trap 'rm -f "$bucket_file" "$selection_file"' EXIT

python - <<'PYTHON' > "$bucket_file"
import json
from pathlib import Path

import yaml

recipe = yaml.safe_load(Path("tests/test_utils/recipes/h100/unit-tests.yaml").read_text())
print(json.dumps([bucket for product in recipe["products"] for bucket in product["test_case"]]))
PYTHON

uv run --locked --project .github/test-selection \
  python .github/scripts/select_unit_tests.py \
  --buckets-file "$bucket_file" \
  --git-mode unstaged \
  --output "$selection_file"
```

`unstaged` mode includes untracked files. Staged changes cause a full-suite
fallback. For committed changes, replace `--git-mode unstaged` with
`--git-mode branch --base-ref "$(git merge-base <base-branch> HEAD)"`, using your
current target branch. The report contains the selected files and CI matrix.

To execute the selection, use the **GPU development environment** rather than
the selector environment, which only contains analysis dependencies:

```bash
python - "$selection_file" <<'PYTHON'
import json
import subprocess
import sys

with open(sys.argv[1]) as stream:
    report = json.load(stream)
targets = report["selected_files"] if report["mode"] == "selective" else ["tests/unit_tests"]
raise SystemExit(subprocess.call([
    sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", "8",
    "-m", "pytest", "-q", "-m", "not flaky_in_dev", *targets,
]))
PYTHON
```

Run the CPU regression checks for the selector, workflow, and launch path with:

```bash
uv run --locked --project .github/test-selection \
  python -m unittest discover -s .github/scripts -p 'test_select*.py' -v
```

## Updating Functional Test Golden Values

When adding new functional tests, it may be necessary to update the golden values used to verify if the test is
passing as expected.

1. Add the new functional test case with the scope set to `mr-github`
2. Open a PR with the new test. Ensure the label `Run functional tests` is added
3. Run the PR CI tests
4. Run the script to download golden values from a Github CI run
    a. Ensure click, requests, and python-gitlab are installed in your environment
    b. Ensure a Github access token is set as an environment variable `GITHUB_TOKEN`
    c. Run the script `python tests/test_utils/python_scripts/download_golden_values.py --source github --pipeline-id <github-workflow-run-id>`
    d. Optionally pass in `--only-failing` to only download golden values for failing tests only
    e. Ensure you are only checking-in golden values for tests are you updating

### Golden-value precision

New training golden files preserve the full scalar precision available in TensorBoard and mark each metric with
`"value_precision": "full"`. Deterministic checks compare these values without rounding. Approximate checks round
both the golden and actual values to five decimal places before applying their configured tolerances.

Existing golden files do not need to be regenerated. A metric without `value_precision` is treated as a legacy
five-decimal golden, so deterministic checks retain their previous behavior until that golden is deliberately
regenerated.

The Github CI infra may not be appropriate for Perf tests. Perf tests may be more appropriate for nightly jobs on other infra.
