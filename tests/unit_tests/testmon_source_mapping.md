# Explicit source-to-test mappings

`testmon_source_mapping.yml` maps source directories to unit-test buckets that
must run in full when those directories change. These mappings supplement
Testmon for execution paths that coverage.py cannot reliably trace, such as
CUDA autograd callbacks:

```yaml
mappings:
  - source_dir: megatron/core/distributed/fsdp/src/megatron_fsdp/experimental
    test_buckets:
      dgx_h100:
        - tests/unit_tests/distributed/mfsdp_v2/**/*.py
```

| Key | Meaning |
| --- | --- |
| `mappings` | List of source-directory rules. Use `[]` to configure no overrides. |
| `source_dir` | Repository-relative source directory to watch recursively, without wildcards or a trailing slash. |
| `test_buckets` | Hardware platforms and their buckets to run fully when the source directory changes. |
| `dgx_h100` / `dgx_gb200` | Optional platform keys; each value is a list of exact `test_case` bucket names from that platform's unit-test recipe. |

Add another list entry for a new source directory, or add more buckets under a
platform. A directory can require multiple buckets, and a bucket can depend on
multiple directories. Each source directory must have a single entry.

The example overrides only H100's MFSDP v2 bucket. GB200 is omitted, so its jobs
keep their normal Testmon selection. Omitted platforms and empty bucket lists
do not create an override or select a broader bucket automatically. This also
means the mapping does not protect GB200 against missing trace dependencies.

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

Identity calculation uses PyYAML's safe loader. The CI action supplies the pinned
parser with `uv run --no-project --with`, without synchronizing Megatron's project
dependencies. Cache validation and publication do not need PyYAML. A parser setup
failure takes the existing full-test fallback for selective runs.

Python bytecode (`.pyc` and `.pyo`) and generated `__pycache__`, `.pytest_cache`,
`.mypy_cache`, and `.ruff_cache` directories are excluded. Invalid mappings,
unreadable source files, or symlinks in mapped trees fail identity calculation
and take the existing full-test fallback. Producer errors prevent publishing
an incomplete baseline.

This safeguard does not repair missing trace events or protect unmapped paths.
Keep exhaustive merge validation and audit other untraced paths separately.
