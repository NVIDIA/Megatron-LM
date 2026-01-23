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

