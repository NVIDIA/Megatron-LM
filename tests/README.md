# Megatron-LM Tests

## Selective Unit Testing

PR builds select H100 unit-test **files** affected by the whole PR diff using
`pytest-impacted`, including transitive import dependencies. CI adds the 10
baseline files in [`unit_tests/always_run_tests.json`](unit_tests/always_run_tests.json)
to every selection, maps the union to the existing buckets, and launches only
buckets with selected files. Each file can contain multiple parametrized test
cases. Documentation-only PRs run the baseline after the usual CI authorization.
The existing GB200 hardware-specific marker suite remains enabled separately.

The baseline covers basic setup, model-parallel configuration, process groups,
rank utilities, shared utilities, tensor-parallel cross entropy, transformer
configuration, GPT construction, and checkpoint mappings. Edit the JSON list to
change the baseline; entries must be unique, existing unit-test files owned by a
CI bucket. Changes to the baseline itself run the full suite for validation.

| Build or change | H100 unit tests |
| --- | --- |
| PR source or test changes | Affected files plus the baseline |
| Documentation-only PR | Baseline only |
| PR labeled `Run full unit tests` | Full suite |
| Merge queue, nightly/CI workload, manual dispatch | Full suite |
| Unsupported change or failed/ambiguous analysis | Full suite |

`Run tests` and `Run functional tests` retain their functional-test behavior;
they do not disable unit-test selection. No opt-in label is required. The old
`Run selective unit tests` and `Run selective unit tests on latest commit`
labels are unnecessary; selection always covers the entire PR, including
changes in earlier commits.

Selection is conservative: missing/invalid base commits, selector errors,
timeouts, deleted or renamed files, shared fixtures, test runners, dependency
and CI configuration changes, package `__init__.py` changes, and unsupported
files request the full matrix. Tokenizer changes also run the full suite because
their dynamic imports hide consumers from static analysis. Empty impact results for source changes also
request the full suite; the baseline cannot hide an analysis failure. The
analyzed base must be an exact ancestor of the same commit used by test jobs.
Static imports cannot prove complete runtime coverage of dynamic imports,
plugins, or monkeypatching; the full merge-queue suite remains the final check.

The workflow summary reports the mode, fallback reason, affected and baseline
file counts, total selected files, buckets, and selector overhead. The dependency
graph is rebuilt on every run, so there is no persistent impact cache. Invalid
per-job payloads and selections that collect no runnable tests fail the job.

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

The Github CI infra may not be appropriate for Perf tests. Perf tests may be more appropriate for nightly jobs on other infra.
