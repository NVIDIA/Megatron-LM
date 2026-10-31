# Functional-test coverage baselines

Add both **Collect functional coverage** and **Run functional tests** PR labels before the synthetic PR push
(or rerun CI after applying it). This opt-in selects the merge-queue functional
matrix: `scope=L1`, `cadence=mergegroup`, one repeat, and lightweight mode off.
The existing full-suite queue policy, including `ALLOW_PR_RUN_ALL_TESTS` and
functional-run serialization, still applies. The coverage label alone does not
enable collection or expand the matrix. Existing maintainer/runner gates still
apply. Normal PR and merge-queue runs do
not collect functional coverage without this label. No recipes are removed.

The collector runs inside each disposable test container. A temporary `.pth`
hook calls `coverage.process_startup()` before imports in Python workers using
the CI interpreter. It also enables multiprocessing support. This covers
training, inference, custom recipe scripts, and checkpoint-resume phases, unlike
wrapping only the final pytest validator. Interpreters that disable site startup
or use a different environment need separate instrumentation.

Coverage measures Python lines and branches under `megatron/`, including core
and training. It does not measure CUDA/native/Triton kernel bodies or establish
numerical, topology, assertion, or defect-detection equivalence. Module import
execution contributes coverage too. Rank, repeat, phase, and launcher restart
are recorded as contexts. Review observed ranks against the recipe and actual
launcher topology before accepting a baseline. An allocated GPU count is not
always a workload's worker count (notably custom inference/performance recipes).

Artifacts are named
`coverage-functional-<platform>-<model>-<test_case>-<run_id>-<run_attempt>`.
Each workload retry has its own `attempt-N/` directory containing:

- `metadata.json`: source SHA, image reference, configuration hash, original
  workload exit code, collection status, elapsed time, and observed contexts/ranks.
- `model_config.yaml`, when the recipe has one, copied before the harness edits it.
- `.coverage.*`: raw process databases, retained for rank/phase analysis.
- `.coverage` and `coverage.json`: combined data for that attempt only.

The top-level manifest and workload script record the requested recipe. Separate
`coverage-functional-matrix-*` artifacts contain the expected matrix on each
platform. Failed attempts are uploaded for diagnosis but excluded from baseline
analysis. A shell-only success with no measured Python code fails collection.
Abrupt termination can prevent data flush; a missing/failing attempt must not be
treated as zero coverage or an excuse to shrink the denominator. Instrumentation
adds overhead; do not use its timings as uninstrumented performance results.
Existing numerical checks, training lengths, and validation remain unchanged.

Download one run (do not mix GitHub run attempts) and analyze it:

```bash
gh run download "$RUN_ID" --repo NVIDIA/Megatron-LM \
  --pattern 'coverage-functional-*' --dir functional-coverage
python tests/test_utils/python_scripts/analyze_functional_coverage.py \
  functional-coverage --output functional-coverage-analysis.json
```

The analyzer requires successful coverage for every entry in both expected
matrices and a consistent source/run identity. It reports each test's line and
branch counts and coverage exclusive to that test on its platform. It then
tries greedy 10% and 20% reductions. For each platform independently, both

`lost previously covered lines / full-suite covered lines`

and

`lost previously covered branches / full-suite covered branches`

must remain **strictly below 5%**. This is relative loss of observed coverage,
not five percentage points of a repository coverage percentage. Marginal loss
is recomputed after each removal. A greedy failure is not proof no feasible
subset exists. Candidate lists are advisory; review hardware, precision,
checkpoint/resume, collectives, and assertions before changing CI selection.

The initial motivating run, `40628051353` at
`a296523510e11badf977fdff9f3ba4a4db14b2e9`, had 80 H100 and 35 GB200 functional
jobs. Its coverage artifacts were unit-test-only, so functional per-test
coverage cannot be reconstructed from that run's TensorBoard metrics or logs.
