# Bisect: Golden-Value Update Script

`tools/bisect.sh` checks out a specific commit, runs the `gpt3_mcore_te_tp2_pp1_resume_torch_dist_cp2_nondeterministic` functional test to generate fresh golden values, copies those values into the test-case directory, then re-runs the test to confirm it passes against the new baseline.

It is used during a manual `git bisect` workflow to determine which commit first broke checkpoint-resume determinism.

## Usage

```bash
./tools/bisect.sh <COMMIT> <MODEL> <TESTCASE>
```

| Argument | Description |
|----------|-------------|
| `<COMMIT>` | Any ref (SHA, branch, tag) that `git checkout` can resolve |
| `<MODEL>` | Model subdirectory under `test_cases/` (e.g. `gpt`) |
| `<TESTCASE>` | Test-case name (e.g. `gpt3_mcore_te_tp2_pp1_resume_torch_dist_cp2_nondeterministic`) |

Example:

```bash
./tools/bisect.sh abc1234 gpt gpt3_mcore_te_tp2_pp1_resume_torch_dist_cp2_nondeterministic
```

## What it does

| Step | Action |
|------|--------|
| 1 | *(commented out)* Stash local changes |
| 2 | *(commented out)* Fetch and checkout `<COMMIT>` |
| 3 | *(commented out)* Cherry-pick each commit in `CHERRYPICK_SHAS` onto the tested commit, in order |
| 4 | Generate local job configs (`generate_local_jobs --environment dev --scope mr`) |
| 5 | Run `test_cases/gpt/gpt3_mcore_te_tp2_pp1_resume_torch_dist_cp2_nondeterministic.sh`, tee output to `log.txt` |
| 6 | Parse `OUTPUT_PATH` from the line `This test wrote results into …` |
| 7 | Copy `golden_values*.json` from `OUTPUT_PATH` into `tests/functional_tests/test_cases/gpt/gpt3_mcore_te_tp2_pp1_resume_torch_dist_cp2_nondeterministic/` |
| 8 | Re-run the test to validate against the newly copied golden values |

Steps 1–3 are commented out because during a `git bisect run` workflow the caller is expected to handle the checkout externally.

## Manual bisect workflow

Because `git bisect run` conflicts with the pre-installed megatron-lm at `/opt/megatron-lm` inside the NeMo container, use a manual binary-search approach:

1. Collect the commit list between GOOD and BAD, oldest-first.
2. Pick the midpoint commit.
3. Run `bisect.sh <midpoint>` inside the container.
4. Mark the commit as GOOD or BAD based on the exit code / log output.
5. Narrow the range and repeat.

## Key constants (as of 2026-04-10)

| Label | SHA |
|-------|-----|
| GOOD  | `696f164de1076f46c87c904e58ca293108459572` |
| BAD   | `d30c3ae5469fe3f6a64d4fd2e63b6e7f7844ea81` |
| Cherry-picks (applied in order) | `bfa0f308aa5f2df76eb24e6b9fb86de5b39b5334` |

## Critical caveat — clone location

The NeMo 25.11 image ships megatron-lm pre-installed at `/opt/megatron-lm`. Cloning into `/opt/megatron-lm` silently reuses the existing directory (which is not a git repo), causing every bisect step to test the image's bundled code instead of the target commit. Always clone into `/tmp/megatron-lm`.

## Output

- `log.txt` — cumulative log for both test runs (appended with `tee -a`).
- `tests/functional_tests/test_cases/gpt/gpt3_mcore_te_tp2_pp1_resume_torch_dist_cp2_nondeterministic/golden_values*.json` — updated golden values copied from the first run.
