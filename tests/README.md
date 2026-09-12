# Megatron-LM Tests

## Selective Unit Testing

Add the **`Run selective unit tests`** label to select H100 unit-test files
affected by the entire PR using `pytest-impacted`, plus the 10 baseline files in
[`unit_tests/always_run_tests.json`](unit_tests/always_run_tests.json).
CI runs only the selected files within their existing recipe buckets.

Without this label, CI runs the full suite without impact analysis.
**`Run full unit tests`** overrides selection. Functional-test labels do not
enable selection. Merge queues, scheduled workloads, and manual runs use the
full suite. Docs-only PRs retain the existing CI skips, and the separate GB200
marker-based suite is unchanged.

Maintain `ALWAYS_RUN_ONLY_PATTERNS` in
[`select_unit_tests.py`](../.github/scripts/select_unit_tests.py) for PR changes
that need only the existing always-run files:

```python
ALWAYS_RUN_ONLY_PATTERNS = (
    "tests/functional_tests/**",
    "skills/**",
    ".github/workflows/claude_review.yml",
)
```

When the selector receives a nonempty PR diff and every changed path matches a
pattern, it selects only the baseline without running `pytest-impacted`. Both
the old and new paths of renames/copies must match. If any path does not match,
the entire diff follows the normal selection policy. Full-suite overrides and
baseline validation remain in effect. With the unchanged CI workflow, this rule
applies to H100 PRs with the selective-testing label; unlabeled PRs still run
the full suite, and GB200 behavior is unchanged.

Selection compares the tested synthetic merge with its first parent
(`git diff HEAD^1 HEAD`), covering all PR commits without unrelated main
changes. Missing merge history, analyzer failures, empty impact results, and
unsupported or high-impact changes fall back to the full suite. These include
shared fixtures, package `__init__.py` files, CI/dependency configuration, and
dynamically loaded tokenizers. An unchanged `conftest.py` reported by the
analyzer does not itself trigger a fallback.

The analyzer runs on a CPU runner in the locked `.github/test-selection`
environment, alongside the image build. It adds no dependencies to the GPU
image. GPU jobs wait for both the image build and the selection plan.
Invalid file payloads or selections with no tests surviving collection and
marker filtering fail the job.

The job summary and `unit-test-selection-<run-id>-<run-attempt>` artifact
contain the plan and fallback reason. File counts describe **planned inputs**,
not passing tests or measured runtime savings; pytest still applies markers
and skips. Static import analysis cannot cover every dynamic dependency, so
full-suite validation remains necessary.

The focused CPU regression checks in `.github/scripts/test_select_unit_tests.py`
cover selection policy, CI wiring, and runtime safeguards. Run them inside the
development container:

```bash
uv run --locked --project .github/test-selection \
  python .github/scripts/test_select_unit_tests.py -v
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
