---
name: update-golden-values
description: Refresh golden values from a GitHub Actions workflow run (failing-only or all jobs), score the change with KL divergence, and produce a PR-ready summary. Use when the user asks to update goldens for a CI run, refresh golden values from a workflow ID, or generate a golden-value diff summary for a PR description.
when_to_use: User provides a GitHub Actions workflow run ID and asks to refresh golden values; user asks to update goldens for "failing tests only" or "all tests"; user asks for a KL-divergence summary of the golden-value diff; user wants a PR description blurb after running download_golden_values.py.
---

# Update golden values + KL summary

End-to-end workflow for refreshing golden values from a GitHub Actions workflow run, validating the update with KL divergence, and writing a PR-ready summary.

The skill orchestrates two scripts that already live in the repo:

- `tests/test_utils/python_scripts/download_golden_values.py` — pulls artifacts from a workflow run and overwrites `tests/functional_tests/test_cases/**/golden_values_*.json`.
- `tests/test_utils/python_scripts/compare_golden_values_kl.py` — diffs the working-tree goldens against `git HEAD` and reports per-metric KL.

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
- [ ] Step 4: Run KL comparison + capture CSV
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

Reuse `/tmp/gv_venv` if it already exists. The KL script only depends on `click` (also in the venv).

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

### Step 4 — KL comparison

```bash
/tmp/gv_venv/bin/python tests/test_utils/python_scripts/compare_golden_values_kl.py \
  --top 20 --csv /tmp/kl_summary.csv
```

The CSV holds one row per `(file, metric)` with columns:
`file, metric, n_steps, KL(old||new), KL(new||old), sym_KL, max|d|, mean|d|, mean rel|d|`.

Then derive aggregates from the CSV (do this in Python; do not paste raw CSV into the summary):

```python
import csv, collections
rows = list(csv.DictReader(open('/tmp/kl_summary.csv')))
for r in rows:
    for k in ('KL(old||new)','KL(new||old)','sym_KL','max|d|','mean|d|','mean rel|d|'):
        r[k] = float(r[k])

by_metric = collections.defaultdict(list)
for r in rows:
    by_metric[r['metric']].append(r['sym_KL'])

# headline numbers per metric
for m, syms in sorted(by_metric.items()):
    syms.sort()
    print(m, len(syms), 'median', syms[len(syms)//2], 'max', syms[-1])

# bucket counts across all rows
buckets = [('==0',lambda x:x==0), ('(0,1e-9)',lambda x:0<x<1e-9),
           ('[1e-9,1e-6)',lambda x:1e-9<=x<1e-6), ('[1e-6,1e-3)',lambda x:1e-6<=x<1e-3),
           ('[1e-3,1e-2)',lambda x:1e-3<=x<1e-2), ('[1e-2,1e-1)',lambda x:1e-2<=x<1e-1),
           ('>=1e-1',lambda x:x>=1e-1)]
syms_all = [r['sym_KL'] for r in rows]
for label, pred in buckets:
    print(label, sum(1 for s in syms_all if pred(s)))
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

### KL divergence summary

Comparison covers <FILES_WITH_BASELINE> files × <NUM_METRICS> metrics = **<TOTAL_ROWS> `(file, metric)` pairs**.

**Per-metric headline numbers**

| metric                    |   n | median sym_KL | max sym_KL |
| ------------------------- | --: | ------------: | ---------: |
| `lm loss`                 | <…> |         <…>   |       <…>  |
| `num-zeros`               | <…> |         <…>   |       <…>  |
| `iteration-time`          | <…> |         <…>   |       <…>  |
| `mem-allocated-bytes`     | <…> |         <…>   |       <…>  |
| `mem-max-allocated-bytes` | <…> |         <…>   |       <…>  |

**Distribution of symmetric KL across all <TOTAL_ROWS> rows**

| sym_KL bucket  | count |
| -------------- | ----: |
| `== 0`         |  <…>  |
| `(0, 1e-9)`    |  <…>  |
| `[1e-9, 1e-6)` |  <…>  |
| `[1e-6, 1e-3)` |  <…>  |
| `[1e-3, 1e-2)` |  <…>  |
| `[1e-2, 1e-1)` |  <…>  |
| `>= 1e-1`      |  <…>  |

**Interpretation** (apply only the bullets that match the data)

- `lm loss` max sym_KL <X> / median <Y> — loss trajectories match old goldens to numerical noise.
- `mem-*` metrics are flat in KL even when raw-byte `max|d|` is large (constant offset).
- `iteration-time` divergences are warmup/scheduler noise, not a correctness signal.
- `num-zeros` shifts cluster on `<list of test patterns>`; within historical run-to-run variance.
````

## Reading the KL columns

| column | meaning |
| --- | --- |
| `n_steps` | shared step indices after dropping NaN/inf |
| `KL(old\|\|new)` / `KL(new\|\|old)` | KL divergence in nats, both directions (asymmetric) |
| `sym_KL` | `KL(old\|\|new) + KL(new\|\|old)`; primary ranking column |
| `max\|d\|`, `mean\|d\|` | step-wise absolute diffs in raw metric units |
| `mean rel\|d\|` | average of `\|old − new\| / max(\|old\|, ε)`; scale-free |

Triage rules of thumb:

- `lm loss` / `num-zeros` rows with `sym_KL` ≲ 1e-6 are run-to-run noise.
- `iteration-time` divergences are usually warmup/scheduler noise, not correctness.
- Focus reviewer attention on `lm loss` and `num-zeros` rows with `sym_KL` ≥ ~1e-3, and check `mean rel|d|` for an intuitive magnitude.

## Notes & gotchas

- The download script's `_fetch_and_filter_artifacts` honors `--only-failing` only on the GitHub path. The Gitlab path applies it per-job inside `download_from_gitlab`.
- A brand-new golden file (no `git HEAD` baseline) is silently skipped by the KL script with a warning. Subtract these from the file count when reporting "files with baseline".
- Some artifacts have a literal string `"nan"` in step 1 of `iteration-time`; the KL script filters those out, so divergences for that metric still come through. Don't flag `iteration-time` as a correctness problem unless something else also moved.
- Never commit `GITHUB_TOKEN`, `RO_API_TOKEN`, or any value derived from `gh auth token`. If the user wants you to commit, only stage golden-value files and the optional CSV — not the env or the venv.
