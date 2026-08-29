# Megatron-LM Tests

## Selective Unit Testing

Pull requests labeled `Run selective unit tests` use `pytest-impacted` to select
H100 unit-test files affected by the PR diff. The selected files are mapped to
the existing unit-test buckets, and only buckets with selected files are
launched. Without the label, CI uses the standard full H100 unit-test matrix.
The GB200 unit-test job still runs its full recipe.

Selection is fail-closed: CI runs the full H100 unit-test matrix whenever the
analysis is unavailable or ambiguous. Full-suite fallbacks include a missing or
invalid base commit, selector errors, empty or invalid selector output, deleted
or renamed files, changes to CI/test configuration or dependencies, shared
fixtures and test runners, package `__init__.py` files, selector timeouts, and
files that `pytest-impacted` cannot analyze safely. The PR base must be an exact
commit that is an ancestor of the tested head. The merge queue and
scheduled/nightly workflows also run the full matrix. On a PR, either the `Run
tests` or `Run functional tests` label requests a full matrix and takes
precedence over `Run selective unit tests`. The workflow summary reports the
selection mode, fallback reason, selector overhead, file count, and bucket
count.

After the matrix is emitted, an invalid per-job payload or a selected bucket
that collects no tests fails the job rather than allowing a green partial run.

`pytest-impacted` is stateless between invocations. CI does not restore or save
a dependency-analysis cache; the dependency graph is rebuilt from the checked
out commit on every run. Cache-hit rate is therefore not applicable; selector
overhead is measured directly instead.

### Run impacted tests locally

Generate the bucket list once per shell session:

```bash
bucket_file="$(mktemp)"
selection_file="$(mktemp)"
trap 'rm -f "$bucket_file" "$selection_file"' EXIT

yq -o=json '[.products[].test_case[] | {"bucket": .}]' \
  tests/test_utils/recipes/h100/unit-tests.yaml > "$bucket_file"
```

To analyze unstaged and untracked changes, run:

```bash
uv run --locked --no-default-groups \
  --group build --group test --group selective-testing \
  python .github/scripts/select_unit_tests.py \
  --buckets-file "$bucket_file" \
  --git-mode unstaged \
  --output "$selection_file" \
  --run
```

Staged-only changes are intentionally not analyzed in `unstaged` mode and cause
a full-suite fallback. To analyze all committed changes on a branch, first make
sure `origin/main` is current, then use its exact merge-base commit:

```bash
base_sha="$(git merge-base origin/main HEAD)"

uv run --locked --no-default-groups \
  --group build --group test --group selective-testing \
  python .github/scripts/select_unit_tests.py \
  --buckets-file "$bucket_file" \
  --git-mode branch \
  --base-ref "$base_sha" \
  --output "$selection_file" \
  --run
```

Both commands print whether selection was selective or fell back to the full
suite and record the details in `$selection_file`. They launch tests with eight
processes by default, matching CI.

To bypass selection and run the full unit-test suite directly:

```bash
uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests
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
