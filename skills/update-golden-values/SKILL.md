---
name: update-golden-values
description: Refresh golden values from a GitHub Actions workflow run (failing-only or all jobs), calculate signed per-model percentage changes, and produce a PR-ready summary. Use when the user asks to update goldens for a CI run, refresh golden values from a workflow ID, or generate a golden-value diff summary for a PR description.
---

# Update golden values + signed per-model percentage summary

End-to-end workflow for refreshing golden values from a GitHub Actions workflow run, reporting signed percentage changes per test/model environment, and writing a PR-ready summary.

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
- [ ] Step 4: Run relative-diff comparison + generate per-model percentage table
- [ ] Step 5: Produce PR-ready summary
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

Convert the raw ratio to a percentage for the report:

`avg_rel_diff_pct = 100 × avg_rel_diff`

Always include the `%` symbol and preserve the sign. Do not take the absolute value, produce magnitude-only statistics, or combine models into magnitude buckets. Generate one row per test/model environment instead:

```python
import collections
import csv
from pathlib import Path

rows = list(csv.DictReader(open('/tmp/reldiff_summary.csv')))
for r in rows:
    r['n_steps'] = int(r['n_steps'])
    r['avg_rel_diff_pct'] = 100 * float(r['avg_rel_diff'])

by_file = collections.defaultdict(dict)
for r in rows:
    by_file[r['file']][r['metric']] = {
        'n_steps': r['n_steps'],
        'pct': r['avg_rel_diff_pct'],
    }

preferred_metrics = [
    'lm loss',
    'mtp_1 loss',
    'num-zeros',
    'iteration-time',
    'mem-allocated-bytes',
    'mem-max-allocated-bytes',
]
present_metrics = {r['metric'] for r in rows}
metrics = [m for m in preferred_metrics if m in present_metrics]
metrics.extend(sorted(present_metrics - set(metrics)))

print('| Test / environment | Steps | ' + ' | '.join(f'`{m}` (%)' for m in metrics) + ' |')
print('| --- | --: | ' + ' | '.join('--:' for _ in metrics) + ' |')

for file_name in sorted(by_file):
    path = Path(file_name)
    test_name = path.parent.name
    environment = path.stem.removeprefix('golden_values_')
    values = by_file[file_name]
    steps = max(v['n_steps'] for v in values.values())

    cells = []
    for metric in metrics:
        if metric not in values:
            cells.append('—')
            continue
        pct = values[metric]['pct']
        cells.append('`0.000000%`' if pct == 0 else f'`{pct:+.6f}%`')

    print(f'| `{test_name}` / {environment} | {steps} | ' + ' | '.join(cells) + ' |')

print()
print('`Steps` is the largest shared-step count for the row. State any metric-specific')
print('difference, such as `iteration-time` having one fewer step after a leading NaN is filtered.')
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

### Signed per-model relative differences

Comparison covers <FILES_WITH_BASELINE> files across <NUM_METRICS> distinct metrics = **<TOTAL_ROWS> `(file, metric)` pairs**. The reported percentage is `100 × mean((old − new) / old)` over shared steps.

Positive percentages mean the new run is lower; negative percentages mean it is higher.

<INSERT THE GENERATED PER-MODEL MARKDOWN TABLE HERE>

State when a metric uses fewer shared steps than the row's `Steps` value—for example, when a leading `iteration-time` NaN was filtered.

**Interpretation** (apply only statements supported by the signed percentages)

- `lm loss` changes between `-0.01%` and `+0.01%` generally match old goldens to numerical noise.
- For `lm loss` or `num-zeros`, call out values below `-0.1%` or above `+0.1%` for review.
- Negative `iteration-time` percentages mean the new run was slower; positive percentages mean it was faster. Treat timing changes as scheduler/warmup noise unless they repeat.
- Describe large `num-zeros` changes per model; do not hide them in a cross-model magnitude aggregate.
````

## Reading the columns

| column             | meaning                                                                                               |
| ------------------ | ----------------------------------------------------------------------------------------------------- |
| `n_steps`          | shared step indices used in the average (NaN/inf and steps with `\|old\| < 1e-12` are dropped).      |
| `avg_rel_diff`     | raw ratio: `mean((old − new) / old)` over `n_steps`; positive = new < old, negative = new > old.      |
| `avg_rel_diff_pct` | report value: `100 × avg_rel_diff`; retain the sign and append `%` so the unit is unambiguous.         |

Keep the sign in every table cell and sort rows by test/model and environment. Do not rank or summarize the report using unsigned magnitudes.

Triage rules of thumb:

- `lm loss` changes from `-0.01%` through `+0.01%` are generally run-to-run noise.
- A negative `iteration-time` percentage means the new run was slower; a positive percentage means it was faster. Treat timing changes as scheduler/warmup noise unless they repeat.
- Focus reviewer attention on `lm loss` and `num-zeros` values outside `-0.1%` through `+0.1%`.

## Notes & gotchas

- The download script's `_fetch_and_filter_artifacts` honors `--only-failing` only on the GitHub path. The Gitlab path applies it per-job inside `download_from_gitlab`.
- A brand-new golden file (no `git HEAD` baseline) is silently skipped by the comparison script with a warning. Subtract these from the file count when reporting "files with baseline".
- Steps where `|old|` is below `1e-12` are excluded from the average — division blows up there (think `num-zeros` step 0 on a dense model, or `mem-*` before allocation). If every shared step is excluded for a metric, that `(file, metric)` row is omitted entirely.
- Functional-test metric collection removes leading `iteration-time` NaNs before artifacts are uploaded. The comparison script filters NaN/inf values at other steps, so other values for that metric still contribute.
- The script's filename is `compare_golden_values_kl.py` for legacy reasons; it no longer computes KL divergence. The function and CSV column names reflect what it actually does (`avg_rel_diff`).
- Never commit `GITHUB_TOKEN`, `RO_API_TOKEN`, or any value derived from `gh auth token`. If the user wants you to commit, only stage golden-value files and the optional CSV — not the env or the venv.
