---
name: update-golden-values
description: Refresh golden values from a GitHub Actions workflow run (failing-only or all jobs), score the change with average normalized relative differences, and produce a PR-ready summary. Use when the user asks to update goldens for a CI run, refresh golden values from a workflow ID, or generate a golden-value diff summary for a PR description.
when_to_use: User provides a GitHub Actions workflow run ID and asks to refresh golden values; user asks to update goldens for "failing tests only" or "all tests"; user asks for a per-metric relative-difference summary of the golden-value diff; user wants a PR description blurb after running download_golden_values.py.
---

# Update golden values + relative-diff summary

End-to-end workflow for refreshing golden values from a GitHub Actions workflow run, scoring the update with a per-metric average normalized relative difference, and writing a PR-ready summary.

The skill orchestrates two scripts that already live in the repo:

- `tests/test_utils/python_scripts/download_golden_values.py` — pulls artifacts from a workflow run and overwrites `tests/functional_tests/test_cases/**/golden_values_*.json`.
- `tests/test_utils/python_scripts/compare_golden_values_kl.py` — diffs the working-tree goldens against `git HEAD` and reports per-metric `avg_rel_diff = mean((old − new) / old)`. (Filename keeps the legacy `_kl` suffix; the script no longer computes KL divergence.)

## Inputs to gather from the user

1. **GitHub Actions workflow run ID** (e.g. `25341543542`). It's the numeric ID in the run URL.
2. **Source**: should be `github` for this workflow. (`gitlab` is supported by the download script but uses a different env path.)
3. **Scope** — accept one of:
   - `only-failing` → run with `--only-failing` (download from failing/cancelled jobs only). Use this for "fix the broken tests" workflows.
   - `all` → run without `--only-failing` (download from every job that produced golden values). Use this when the user wants a full refresh.

   If the user doesn't specify, ask. Don't silently default.

## Workflow

```
- [ ] Step 1: Set up env (token + venv with deps)
- [ ] Step 2: Reset prior golden-value edits
- [ ] Step 3: Download goldens (scope = only-failing | all)
- [ ] Step 4: Run relative-diff comparison + capture CSV
- [ ] Step 5: Produce summary blurb
```

### Step 1 — Environment

The download script needs `GITHUB_TOKEN`. If the user has the `gh` CLI authenticated, derive it; do NOT export the token into a long-lived shell or commit it.

```bash
# token (one-shot, scoped to the command)
export GITHUB_TOKEN="$(gh auth token)"

# python deps (the script imports click, gitlab, requests)
python3 -m venv /tmp/gv_venv
/tmp/gv_venv/bin/pip install --quiet click python-gitlab requests
```

Reuse `/tmp/gv_venv` if it already exists. The comparison script only depends on `click` (also in the venv).

### Step 2 — Reset prior edits (only if user re-runs)

If the working tree already has prior golden-value modifications you want to discard before re-downloading:

```bash
git checkout -- tests/functional_tests/test_cases/
git ls-files --others --exclude-standard tests/functional_tests/test_cases/ \
  | while IFS= read -r f; do rm -f "$f"; done
```

Skip this step when the user explicitly wants to layer a new download on top of an in-progress branch.

### Step 3 — Download

Build the command from the user-provided scope:

```bash
# scope = only-failing (default for "fix broken tests")
/tmp/gv_venv/bin/python tests/test_utils/python_scripts/download_golden_values.py \
  --source github --pipeline-id <WORKFLOW_RUN_ID> --only-failing

# scope = all (full refresh; omit the flag)
/tmp/gv_venv/bin/python tests/test_utils/python_scripts/download_golden_values.py \
  --source github --pipeline-id <WORKFLOW_RUN_ID>
```

When `--only-failing` is set, the GitHub path filters at `_fetch_and_filter_artifacts` on `matched_job["conclusion"] == "success"`, so only failing/cancelled jobs contribute artifacts. Without the flag, every job's golden-value artifact is pulled.

Capture the final two log lines for the summary; they look like:

```
INFO:__main__:Total tests with golden values: <N>
INFO:__main__:Total golden values found: <M>
```

### Step 4 — Relative-diff comparison

```bash
/tmp/gv_venv/bin/python tests/test_utils/python_scripts/compare_golden_values_kl.py \
  --top 20 --csv /tmp/reldiff_summary.csv
```

The CSV holds one row per `(file, metric)` with four columns:

`file, metric, n_steps, avg_rel_diff`

- `n_steps` — count of shared steps that contributed (steps where `|old| < 1e-12` are skipped to avoid div-by-zero; NaN/inf are dropped).
- `avg_rel_diff` — `mean((old − new) / old)`. **Signed**: positive = the new run is smaller than the old run at the typical step (e.g. loss decreased), negative = larger.

Then derive aggregates from the CSV (do this in Python; do not paste raw CSV into the summary):

```python
import csv, collections
rows = list(csv.DictReader(open('/tmp/reldiff_summary.csv')))
for r in rows:
    r['n_steps']      = int(r['n_steps'])
    r['avg_rel_diff'] = float(r['avg_rel_diff'])
    r['abs']          = abs(r['avg_rel_diff'])

by_metric = collections.defaultdict(list)
for r in rows:
    by_metric[r['metric']].append(r['abs'])

# headline numbers per metric (using |avg_rel_diff|)
for m, vs in sorted(by_metric.items()):
    vs.sort()
    print(m, len(vs), 'median', vs[len(vs)//2], 'max', vs[-1])

# bucket counts across all rows, on |avg_rel_diff|
buckets = [('==0',      lambda x: x == 0),
           ('(0,1e-6)', lambda x: 0 < x < 1e-6),
           ('[1e-6,1e-4)', lambda x: 1e-6 <= x < 1e-4),
           ('[1e-4,1e-3)', lambda x: 1e-4 <= x < 1e-3),
           ('[1e-3,1e-2)', lambda x: 1e-3 <= x < 1e-2),
           ('[1e-2,1e-1)', lambda x: 1e-2 <= x < 1e-1),
           ('>=1e-1',   lambda x: x >= 1e-1)]
abs_all = [r['abs'] for r in rows]
for label, pred in buckets:
    print(label, sum(1 for v in abs_all if pred(v)))
```

### Step 5 — Summary blurb

Use this template verbatim, filling in `<…>` from steps 3–4. Drop sections that don't apply to the run.

Pick the wording for the first line based on the scope used:

- `only-failing` → "Refresh of golden values for failing functional tests from GitHub workflow run …"
- `all` → "Full refresh of golden values from GitHub workflow run …"

Match the `download_golden_values.py` command in the bullet list to the scope used (with or without `--only-failing`).

````markdown
### Summary

<scope-appropriate sentence> from GitHub workflow run `<WORKFLOW_RUN_ID>`.

**Golden value updates**

- Re-ran `tests/test_utils/python_scripts/download_golden_values.py --source github --pipeline-id <WORKFLOW_RUN_ID> <--only-failing if scope=only-failing>`.
- Updated **<N> golden-value files** under `tests/functional_tests/test_cases/`.

### Relative-difference summary

Comparison covers <FILES_WITH_BASELINE> files × <NUM_METRICS> metrics = **<TOTAL_ROWS> `(file, metric)` pairs**. Per row: `avg_rel_diff = mean((old − new) / old)` over shared steps.

**Per-metric headline numbers** (over `|avg_rel_diff|`)

| metric                    |   n | median \|avg_rel_diff\| | max \|avg_rel_diff\| |
| ------------------------- | --: | -----------------------: | -------------------: |
| `lm loss`                 | <…> |                    <…>   |                <…>   |
| `num-zeros`               | <…> |                    <…>   |                <…>   |
| `iteration-time`          | <…> |                    <…>   |                <…>   |
| `mem-allocated-bytes`     | <…> |                    <…>   |                <…>   |
| `mem-max-allocated-bytes` | <…> |                    <…>   |                <…>   |

**Distribution of `|avg_rel_diff|` across all <TOTAL_ROWS> rows**

| \|avg_rel_diff\| bucket | count |
| ----------------------- | ----: |
| `== 0`                  |  <…>  |
| `(0, 1e-6)`             |  <…>  |
| `[1e-6, 1e-4)`          |  <…>  |
| `[1e-4, 1e-3)`          |  <…>  |
| `[1e-3, 1e-2)`          |  <…>  |
| `[1e-2, 1e-1)`          |  <…>  |
| `>= 1e-1`               |  <…>  |

**Interpretation** (apply only the bullets that match the data)

- `lm loss` max `|avg_rel_diff|` <X> / median <Y> — loss trajectories match old goldens to numerical noise (sub-1e-4 is within run-to-run variance).
- `mem-*` metrics typically sit at `== 0` or `(0, 1e-6)`; flag any row that lands above `[1e-4, 1e-3)`.
- `iteration-time` movement is dominated by warmup/scheduler noise; signed avg near zero means the run was simply jitterier, not slower or faster on average.
- `num-zeros` shifts cluster on `<list of test patterns>`; within historical run-to-run variance.
````

## Reading the columns

| column         | meaning                                                                                          |
| -------------- | ------------------------------------------------------------------------------------------------ |
| `n_steps`      | shared step indices used in the average (NaN/inf and steps with `\|old\| < 1e-12` are dropped). |
| `avg_rel_diff` | `mean((old − new) / old)` over `n_steps`. Signed: positive = new < old, negative = new > old.    |

When sorting / filtering, the script ranks by `|avg_rel_diff|`. Keep the sign in the printed table so reviewers can see direction.

Triage rules of thumb:

- `lm loss` / `num-zeros` rows with `|avg_rel_diff|` ≲ 1e-4 are run-to-run noise.
- `iteration-time` divergences are usually warmup/scheduler noise; a small signed mean near zero says the run was jitterier, not systematically faster or slower.
- Focus reviewer attention on `lm loss` and `num-zeros` rows with `|avg_rel_diff|` ≥ ~1e-3.

## Notes & gotchas

- The download script's `_fetch_and_filter_artifacts` honors `--only-failing` only on the GitHub path. The Gitlab path applies it per-job inside `download_from_gitlab`.
- A brand-new golden file (no `git HEAD` baseline) is silently skipped by the comparison script with a warning. Subtract these from the file count when reporting "files with baseline".
- Steps where `|old|` is below `1e-12` are excluded from the average — division blows up there (think `num-zeros` step 0 on a dense model, or `mem-*` before allocation). If every shared step is excluded for a metric, that `(file, metric)` row is omitted entirely.
- Some artifacts have a literal string `"nan"` in step 1 of `iteration-time`; the comparison script filters those out, so other steps for that metric still contribute. Don't flag `iteration-time` as a correctness problem unless something else also moved.
- The script's filename is `compare_golden_values_kl.py` for legacy reasons; it no longer computes KL divergence. The function and CSV column names reflect what it actually does (`avg_rel_diff`).
- Never commit `GITHUB_TOKEN`, `RO_API_TOKEN`, or any value derived from `gh auth token`. If the user wants you to commit, only stage golden-value files and the optional CSV — not the env or the venv.
