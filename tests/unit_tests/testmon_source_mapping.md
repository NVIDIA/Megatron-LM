# Explicit source-to-test mappings

`testmon_source_mapping.yml` maps source directories to unit-test buckets that
must run in full when those directories change. These mappings supplement
Testmon for execution paths that coverage.py cannot reliably trace, such as
CUDA autograd callbacks:

```yaml
mappings:
  - source_dirs:
      - megatron/core/distributed/fsdp/src/megatron_fsdp/experimental
    test_buckets:
      dgx_h100:
        - tests/unit_tests/distributed/mfsdp_v2/**/*.py
```

| Key | Meaning |
| --- | --- |
| `mappings` | List of rules. Add multiple entries for independent groups; use `[]` to configure no overrides. |
| `source_dirs` | Nonempty list of repository-relative source directories to watch recursively, without wildcards or trailing slashes. |
| `test_buckets` | Hardware platforms and their buckets to run fully when **any** listed source directory changes. Multiple platform keys are supported. |
| `dgx_h100` / `dgx_gb200` | Optional platform keys; each value is a list of exact `test_case` bucket names from that platform's unit-test recipe. |

Every rule can list multiple source directories, multiple hardware platforms,
and multiple buckets per platform. A change in **any** listed directory forces
**all** that rule's configured buckets on their respective platforms. Separate
rules are additive: if a source appears in several rules, its targets are merged.
Repeated source paths or bucket names do not create duplicate test executions.
An empty list does not cancel a target added by another rule.

The example overrides only H100's MFSDP v2 bucket. GB200 is omitted, so its jobs
keep their normal Testmon selection. Omitted platforms and empty bucket lists
do not create an override or select a broader bucket automatically. This also
means the mapping does not protect GB200 against missing trace dependencies.

For illustration, a separate rule could group several source directories and
test buckets as follows. This rule is not enabled by the example configuration:

```yaml
mappings:
  - source_dirs:
      - megatron/core/transformer
      - megatron/core/tensor_parallel
    test_buckets:
      dgx_h100:
        - tests/unit_tests/transformer/**/*.py
        - tests/unit_tests/models/**/*.py
      dgx_gb200:
        - tests/unit_tests/**/*.py
        - tests/unit_tests/generalized_tensor_parallel/**/*.py
  - source_dirs:
      - megatron/core/distributed/fsdp/src/megatron_fsdp/experimental
    test_buckets:
      dgx_h100:
        - tests/unit_tests/distributed/mfsdp_v2/**/*.py
```

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
