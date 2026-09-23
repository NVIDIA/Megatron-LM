# Explicit source-to-test mappings

`testmon_source_mapping.json` maps source directories to unit-test buckets that
must run in full when those directories change. These mappings supplement
Testmon for execution paths that coverage.py cannot reliably trace, such as
CUDA autograd callbacks:

```json
{
  "megatron/core/distributed/fsdp/src/megatron_fsdp/experimental": {
    "dgx_h100": ["tests/unit_tests/distributed/mfsdp_v2/**/*.py"],
    "dgx_gb200": ["tests/unit_tests/**/*.py"]
  }
}
```

Keys are repository-relative directories without wildcards or trailing slashes.
Each value maps recipe platforms (`dgx_h100` or `dgx_gb200`) to lists of exact
`test_case` bucket names from that platform's unit-test recipe. Omit a platform
when it needs no override. Add more entries or buckets to extend the mapping.
A directory can require multiple buckets, and a bucket can depend on multiple
directories.

MFSDP v2 has a dedicated H100 bucket, but its Blackwell-specific tests run in
GB200's shared `tests/unit_tests/**/*.py` bucket. The mapping therefore forces
that whole GB200 bucket to run when experimental MFSDP sources change. This
conservative choice also runs unrelated GB200 tests in the same bucket; it
preserves coverage without adding a job. H100's shared bucket remains selective.

The shared baseline records paths and content hashes for each bucket's mapped
source files, including subdirectories. An added, modified, deleted, or renamed
file makes that bucket incompatible with its restored baseline, so the existing
fallback runs the full configured bucket. Other eligible buckets continue using
Testmon. Normal hardware, experimental, and flaky-test filters still apply;
the mapping adds no jobs or duplicate test executions.

Comparison is against the actual cached baseline, rather than only the PR's
base commit. This also covers changes already merged into `main` after that
baseline was recorded. Unchanged mapped sources still allow Testmon selection.
Mapping changes invalidate old baselines through the normal compatibility checks.

Python bytecode (`.pyc` and `.pyo`) and generated `__pycache__`, `.pytest_cache`,
`.mypy_cache`, and `.ruff_cache` directories are excluded. Invalid mappings,
unreadable source files, or symlinks in mapped trees fail identity calculation
and take the existing full-test fallback. Producer errors prevent publishing
an incomplete baseline.

This safeguard does not repair missing trace events or protect unmapped paths.
Keep exhaustive merge validation and audit other untraced paths separately.
