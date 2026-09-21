# Functional tests changed by a pull request

GitHub PR CI runs the usual functional-test suite **plus active functional tests
added or updated by the PR**, with or without the `Run functional tests` label.
No additional label is required. Existing labels,
repeat counts, lightweight mode, and H100/GB200 availability gates still apply.
Unit-test selection is unchanged.

The functional matrix compares the tested commit with the merge base of its
pinned PR base SHA. This covers the entire PR, including changes made before the
most recent push. It adds tests when:

- Any file in `tests/functional_tests/test_cases/<model>/<test_case>/` changes,
  including model configuration, golden values, or test-specific inputs.
- A recipe in `tests/test_utils/recipes/` adds or changes that test's expanded
  workload. Editing one product selects that product's affected cases; changing
  a shared recipe specification selects all affected cases in that recipe.
  Comments and formatting alone do not add tests.

Additional tests must have an active GitHub tier (`L0`, `L1`, `L2`, or `L3`,
including legacy aliases), a `dev` recipe for the platform being tested, and an
existing `model_config.yaml`. They must fit the existing single-node GitHub
runner (8 GPUs on H100 or 4 GPUs on GB200); multi-node recipes still require
the existing JET/GitLab infrastructure. Disabled scopes such as `mr-github-broken`,
GitLab-only scopes, unit-test recipes, and deleted cases are not added. New
functional cases therefore still need a recipe registration.

Changed cases can run outside their normal tier or cadence, so a nightly case
edited by a PR is included in that PR's matrix. A case already in the usual
suite runs once with its existing selection settings. Each additional matrix
entry carries its own scope and bypasses cadence filtering when launched.

Merge-queue, scheduled, release, and manual runs keep their existing selection.
Changes to shared test harnesses outside case directories or recipes do not
automatically expand the suite; use the existing scope labels for broader
validation.
