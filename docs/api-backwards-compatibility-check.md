---
orphan: true
---

<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# API Backward Compatibility Checking

## Overview

Megatron Core uses automated API compatibility checking to ensure stable interfaces between releases. This prevents accidental breaking changes that could affect users upgrading between versions.

## How It Works

The compatibility checker:
1. Compares the current code against the latest release
2. Detects breaking changes in function signatures
3. Fails CI if breaking changes are found (unless explicitly exempted)
4. Runs automatically on every PR that modifies `megatron/core`

## What Gets Checked

### ✅ Breaking Changes Detected

- **Parameter removed** - Removing a function parameter
- **Parameter added without default** - Adding a required parameter
- **Parameter order changed** - Changing the order of parameters
- **Optional→Required** - Removing a default value from a parameter
- **Function removed** - Deleting a public function
- **Return type changed** - Changing the return type annotation (warning)

### ⏭️ What Gets Skipped

- **Test functions** - Functions starting with `test_`
- **Exempt decorators** - Functions marked with `@internal_api`, `@experimental_api`, or `@deprecated`
- **Excluded paths** - Code in `tests/`, `experimental/`, `legacy/`

### ✅ Allowed Changes

- **Adding optional parameters** - Adding parameters with default values
- **Adding new functions** - New public APIs
- **Making parameters optional** - Adding default values to required parameters

## For Developers

### Running Locally

```bash
# Install griffe
pip install griffe

# Check against latest release
python scripts/check_api_backwards_compatibility.py --baseline core_r0.8.0

# Check with verbose output
python scripts/check_api_backwards_compatibility.py --baseline core_r0.8.0 -v

# Compare two specific branches
python scripts/check_api_backwards_compatibility.py --baseline core_r0.8.0 --current main
```

### Marking Functions as Exempt

If you need to make breaking changes to internal or experimental APIs:

#### Internal API (for internal implementation details)

```python
from megatron.core.utils import internal_api

@internal_api
def experimental_feature(x, y):
    """
    This API is experimental and may change.
    NOT FOR EXTERNAL USE.
    """
    pass
```

**When to use `@internal_api`:**
- Internal APIs not documented for external use
- Experimental features explicitly marked as unstable
- Functions in development that haven't been released yet

#### Experimental API (for experimental features)

```python
from megatron.core.utils import experimental_api

@experimental_api
def new_experimental_feature(x, y):
    """
    This API is experimental and may change without notice.
    """
    pass
```

**When to use `@experimental_api`:**
- Experimental features explicitly marked as unstable
- New APIs under active development
- Features that haven't been stabilized yet

### Deprecating APIs

For planned API changes, use the deprecation workflow:

```python
from megatron.core.backwards_compatibility_decorators import deprecated

@deprecated(
    version="1.0.0",           # When deprecation starts
    removal_version="2.0.0",    # When it will be removed
    alternative="new_function", # Recommended replacement
    reason="Improved performance and cleaner API"
)
def old_function(x):
    """This function is deprecated."""
    pass
```

**Deprecation Timeline:**
1. **Version N** - Add `@deprecated` decorator, function still works
2. **Version N+1** - Keep function with deprecation warnings
3. **Version N+2** - Remove function (users have been warned)

### Deprecated `parallel_state` Accessors

The functions of `megatron.core.parallel_state` that read the global parallel grid, and
`ProcessGroupCollection.use_mpu_process_groups()`, which builds a collection from them, are
deprecated in Megatron Core 0.21 and removed in 0.23. The global grid belongs to a single model;
code that needs process groups takes them from the `ProcessGroupCollection` that its caller
passes. Called from outside Megatron Core, each function emits a
`ParallelStateDeprecationWarning`, a `FutureWarning`, once per call site. The warning names the
replacement:

| Deprecated functions | Replacement |
|---|---|
| Group accessors, such as `get_tensor_model_parallel_group()`, including the Gloo and dynamic context-parallel ones | The matching field of the model's `ProcessGroupCollection`, such as `pg_collection.tp` |
| Rank, size and global-rank accessors, such as `get_tensor_model_parallel_rank()` and `get_pipeline_model_parallel_first_rank()` | `rank()` or `size()` of that field, `torch.distributed.get_global_rank(group, 0)`, or the `get_pp_*_rank(pp_group)` helpers in `megatron.core.pipeline_parallel.utils` |
| Stage predicates: `is_pipeline_first_stage()`, `is_pipeline_last_stage()`, `is_rank_in_embedding_group()`, `is_rank_in_position_embedding_group()` | `is_pp_first_stage(pp_group)`, `is_pp_last_stage(pp_group)`, `is_vp_first_stage()` and `is_vp_last_stage()` in `megatron.core.pipeline_parallel.utils`; membership in `pg_collection.embd` or `pg_collection.pos_embd` |
| Virtual-pipeline rank and size getters and setters | The `vp_stage` of the model chunk, passed explicitly, and the `virtual_pipeline_model_parallel_size` of the model's configuration |
| Rank and size overrides, such as `set_tensor_model_parallel_rank()` | A `ProcessGroupCollection` whose groups have that rank and size, passed to the code that needs it or registered with `register_process_groups()` |
| `get_global_memory_buffer()`, `destroy_global_memory_buffer()`, `_set_global_memory_buffer()` | `get_global_memory_buffer()` and `destroy_global_memory_buffer()` in `megatron.core.utils`. `get_global_memory_buffer()` creates the buffer on first use, so `_set_global_memory_buffer()` needs no replacement |
| `ProcessGroupCollection.use_mpu_process_groups()` | The `ProcessGroupCollection` that the caller passes, or `parallel_state.get_registered_process_groups()` for the groups that `initialize_model_parallel()` or `register_process_groups()` registered for the run. It returns the registered collection itself, not a new one, so set a field such as `dp_cp_ag` on a `copy.copy()` of it: `copy.deepcopy()` returns the same collection, and `dataclasses.replace()` returns one without groups |

Bootstrap, teardown and the registry are not deprecated: `initialize_model_parallel()`,
`destroy_model_parallel()`, `is_initialized()`, `model_parallel_is_initialized()`,
`register_process_groups()`, `get_registered_process_groups()` and
`build_process_group_collection()`. Neither are `create_group()` and `update_pg_timeout()`, which
manage the lifetime of the groups that `parallel_state` creates, nor `create_all_gather_groups()`
with a `pg_collection`; without one, it falls back to the global grid with a
`ProcessGroupFallbackWarning`.

- **Megatron Core's own calls do not warn.** Its compatibility fallbacks emit a
  `ProcessGroupFallbackWarning` for the caller that omitted its groups instead.
  `tools/check_process_group_usage.py` ratchets Megatron Core's calls of every deprecated
  function outside `parallel_state.py` and `process_groups_config.py`: the group, rank and size
  accessors, the stage predicates, the rank and size overrides, the virtual-pipeline getters and
  setters, `get_all_ranks()`, the dynamic context-parallel group maps, the memory-buffer
  aliases and `use_mpu_process_groups()`. It also counts `get_registered_process_groups()`,
  which is not deprecated but returns the same global groups: Megatron Core takes its groups
  from its caller. It counts the calls of the fallbacks of `process_groups_config.py`, which
  read the grid when the caller omits its groups, as guarded: `resolve_process_groups()`,
  `resolve_gtp_remat_group()`, and `ProcessGroupCollection.setup_process_groups_for_ddp()` and
  `setup_process_groups_for_optimizer()`, and any read, write or deletion of a private attribute
  of `parallel_state`. It counts a reference whether or not the code calls it, through imports,
  assigned aliases, `getattr()`, `setattr()` or `delattr()` with a constant name,
  `importlib.import_module()` and `sys.modules`. A new call of any of them makes the check exit
  with an error. Its allowlist of the existing calls records which ones are warned: preceded, in
  the same function, by a `warn_global_process_group_fallback()` call in their block or in a
  block that contains it, or inside a `with global_process_group_fallback(...)` block. A warned
  call that loses its warning is an error too.
- **A fallback outside Megatron Core** reads the grid inside
  `with global_process_group_fallback(owner, argument, deprecated_in=..., removed_in=...):`
  from `megatron.core.process_groups_config`. It emits one `ProcessGroupFallbackWarning` for the
  fallback, and the accessors and `use_mpu_process_groups()` it calls do not warn again.
  `tests/unit_tests/test_parallel_state_accessor_uses.py` checks that the code of this
  repository outside Megatron Core and its tests calls them only in such a block or in
  bootstrap code that ignores `ParallelStateDeprecationWarning` after warning with that category
  itself, earlier in the same function, and never touches a private attribute of
  `parallel_state`.
- **In CI**, the *Process group usage check* workflow
  (`.github/workflows/process-group-usage-check.yml`) runs on pull requests and in the merge
  queue. Its job fails when the tests of the checker,
  `tests/unit_tests/test_check_process_group_usage.py`, fail; they also check that each call in
  the allowlist, and each read that the checker counts in Megatron Core's code, is warned or
  guarded, apart from those the test lists (none now), and that the checker counts every helper of
  `process_groups_config.py` that reads the grid. A new unwarned read in Megatron Core therefore
  fails the job. It also fails when `tests/unit_tests/test_removal_versions.py` fails (see
  *Removal*). The job then runs
  `tools/check_process_group_usage.py` and `tests/unit_tests/test_parallel_state_accessor_uses.py`
  as report-only steps (`continue-on-error`): a new call in Megatron Core, a call that lost its
  warning, or a deprecated call outside Megatron Core is listed in the step's log, and the job
  still passes. Whether that should block a pull request is still to be decided, see
  [#6307](https://github.com/NVIDIA/Megatron-LM/issues/6307). The unit-test jobs run these
  three test files in the `tests/unit_tests/**/*.py` bucket, where a failure of any of them
  fails the job; nothing there runs the checker on the repository's code. A pull request that
  runs selective (Testmon) unit tests may skip them, because Testmon selects tests by the code
  they execute and they read the files they check as text; the merge queue runs every test.
- **Removal** is due when the version in `megatron/core/package_info.py` reaches the
  `removed_in` version of a fallback, the default of that argument, or a `*_REMOVED_IN`
  constant such as `_ACCESSOR_REMOVED_IN` in `parallel_state.py`.
  `tests/unit_tests/test_removal_versions.py` then fails and lists that code. It fails the
  *Process group usage check* job, which runs it on every pull request that changes code,
  including the one that bumps the version, and the unit-test jobs.
- **Strict mode** (`--strict-process-group-fallbacks`, `enable_strict_process_group_fallbacks()`)
  turns `ProcessGroupFallbackWarning` into an error. It does not affect
  `ParallelStateDeprecationWarning`.
- **Unit tests** set up the global grid on purpose, so `pyproject.toml` ignores
  `ParallelStateDeprecationWarning` when a test calls a deprecated function. When the caller is
  a module of this repository outside the tests, imported under its package name, such as
  `megatron.training.*` or `examples.*`, or a top-level script such as `pretrain_gpt`, it turns
  the warning into an error, so the first test that reaches such a call site fails. Inside
  `pytest.warns()` or another filter the warning is recorded or ignored instead, and the call
  site, which warns only once per process, then stays silent for the later tests. A module that
  a test imports under another name, through `sys.path` or a file location, such as the
  converters of `tools/checkpoint`, and the `megatron.*` packages of other projects, such as
  `megatron.energon`, stay ignored; `tests/unit_tests/test_parallel_state_accessor_uses.py`
  still checks the calls of the former in their source. A test that checks the warning uses
  `pytest.warns(ParallelStateDeprecationWarning)` on a call site that has not warned yet in the
  same process.

### Handling CI Failures

If the compatibility check fails on your PR:

1. **Review the breaking changes** in the CI logs
2. **Choose an action:**
   - **Fix the code** - Revert the breaking change
   - **Add exemption** - Use `@internal_api` if intentional
   - **Use deprecation** - For planned API changes
3. **Update your PR** with the fix

## Examples

### Example 1: Compatible Change

```python
# ✅ BEFORE (v1.0)
def train_model(config, dataloader):
    pass

# ✅ AFTER (v1.1) - Added optional parameter
def train_model(config, dataloader, optimizer="adam"):
    pass
```
**Result:** ✅ Check passes

---

### Example 2: Breaking Change

```python
# BEFORE (v1.0)
def train_model(config, dataloader, optimizer="adam"):
    pass

# ❌ AFTER (v1.1) - Removed parameter
def train_model(config, dataloader):
    pass
```
**Result:** ❌ Check fails - "Parameter 'optimizer' removed"

---

### Example 3: Exempt Internal API

```python
from megatron.core.utils import internal_api

# BEFORE (v1.0)
@internal_api
def _internal_compute(x, y):
    pass

# ✅ AFTER (v1.1) - Can change freely
@internal_api
def _internal_compute(x, y, z):  # Added parameter
    pass
```
**Result:** ✅ Check passes (function is exempt)

---

### Example 4: Deprecation Workflow

```python
from megatron.core.utils import deprecated

# Version 1.0 - Add deprecation
@deprecated(
    version="1.0.0",
    removal_version="2.0.0",
    alternative="train_model_v2"
)
def train_model(config):
    """Old training function - DEPRECATED"""
    pass

def train_model_v2(config, **options):
    """New improved training function"""
    pass

# Version 1.1 - Keep both (users migrate)
# Version 2.0 - Remove train_model()
```

## Architecture

```
Developer commits code
    ↓
GitHub Actions triggers
    ↓
CI runs check_api_backwards_compatibility.py
    ↓
Script loads code via griffe:
  • Baseline: latest release (e.g., core_r0.8.0)
  • Current: PR branch
    ↓
Apply filtering:
  • Skip @internal_api, @experimental_api, and @deprecated
  • Skip private functions (_prefix)
  • Skip test/experimental paths
    ↓
Griffe compares signatures:
  • Parameters
  • Types
  • Return types
  • Defaults
    ↓
Report breaking changes
    ↓
Exit: 0=pass, 1=fail
    ↓
CI fails if breaking changes detected
```

## Configuration

### Customizing Filters

Edit `scripts/check_api_backwards_compatibility.py`:

```python
# Add more exempt decorators
EXEMPT_DECORATORS = [
    "internal_api",
    "experimental_api",
    "deprecated",
]

# Add more path exclusions
EXCLUDE_PATHS = {
    "tests",
    "experimental",
    "legacy",
    "your_custom_path",  # ← Add here
}
```

### Changing the Baseline

The workflow auto-detects the latest `core_r*` tag. To manually specify:

```yaml
# In .github/workflows/check_api_backwards_compatibility_workflow.yml
- name: Run compatibility check
  run: |
    python scripts/check_api_backwards_compatibility.py \
      --baseline your_custom_baseline
```

## FAQ

### Q: Why did my PR fail the compatibility check?

**A:** Your code introduced breaking changes compared to the last release. Review the CI logs to see what changed.

### Q: Can I disable the check for my PR?

**A:** No, but you can mark specific functions as exempt using `@internal_api` or `@experimental_api`.

### Q: What if I need to make a breaking change?

**A:** Use the `@deprecated` decorator for a gradual transition, or mark the function as exempt using `@internal_api` (for internal code) or `@experimental_api` (for experimental features).

### Q: Does this check all of Megatron-LM?

**A:** No, only `megatron/core/**` (Megatron Core). Legacy code is excluded.

### Q: What about class methods?

**A:** Yes, class methods are checked just like functions.

### Q: Can I run this locally before pushing?

**A:** Yes! Run `python scripts/check_api_backwards_compatibility.py --baseline core_r0.8.0`

### Q: What if there's no release tag yet?

**A:** The workflow will use `main` as the baseline. Update it once you have release tags.

## Troubleshooting

### Error: "griffe is not installed"

```bash
pip install griffe
```

### Error: "No core_r* tags found"

The repository doesn't have release tags yet. The workflow will fall back to `main`.

### False Positives

If the checker reports a breaking change that isn't actually breaking, file an issue and use `@internal_api` as a temporary workaround.

## References

- **Script:** `scripts/check_api_backwards_compatibility.py`
- **Workflow:** `.github/workflows/check_api_backwards_compatibility_workflow.yml`
- **Decorators:** `megatron/core/backwards_compatibility_decorators.py`
- **Griffe Documentation:** https://mkdocstrings.github.io/griffe/

## Support

For questions or issues:
1. Check this documentation
2. Review existing PRs with compatibility checks
3. Ask in the Megatron-LM Slack/Discord
4. File an issue on GitHub

