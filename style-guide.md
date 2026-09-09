# Megatron-LM Style Guide

Use this guide for new code and changes to existing code. Keep changes focused:
follow the surrounding conventions where this guide permits alternatives, and
avoid unrelated formatting or annotation rewrites. The repository is primarily
Python; preserve local conventions for shell scripts, C++, and CUDA.

This guide builds on [PR #3149](https://github.com/NVIDIA/Megatron-LM/pull/3149)
and its review discussion, checked against the current repository. Configuration
files and executable checks are authoritative for tool behavior. Recommendations
below also cover matters that the tools do not enforce. Follow applicable
[repository instructions](AGENTS.md) and the
[contribution policy](docs/developer/contribute.md).

General Python conventions are covered by Google's guidance on
[indentation](https://google.github.io/styleguide/pyguide.html#s3.4-indentation),
[strings](https://google.github.io/styleguide/pyguide.html#s3.10-strings),
[naming](https://google.github.io/styleguide/pyguide.html#s3.16-naming),
[type annotations](https://google.github.io/styleguide/pyguide.html#s3.19-type-annotations),
[comments and docstrings](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings),
[exceptions](https://google.github.io/styleguide/pyguide.html#s2.4-exceptions), and
[resource handling](https://google.github.io/styleguide/pyguide.html#s3.11-files-sockets-and-similar-stateful-resources).
Those conventions are referenced here rather than repeated. The repository-specific
rules below take precedence where they differ; these references do not adopt
unrelated Google policies or add lint checks.

## Python and formatting

The root package requires **Python 3.12 or newer**, as declared in
[pyproject.toml](pyproject.toml). Keep code compatible with the minimum supported
version; subpackages with their own metadata may have a different minimum.

- Keep Python lines within **100 characters**, matching Black, isort, and Pylint.
  Black may leave some lines over the limit; address remaining lint findings.
- Let Black handle wrapping and spacing. The repository disables its magic
  trailing comma behavior and string normalization.
- Use the versions and settings in `pyproject.toml` and `uv.lock`.

### Imports

Let isort organize imports using the repository's **Black profile** and configured
sections. `megatron` is first-party; `torch` and `transformer_engine` are third-party.
Relative imports have a separate final section. Explicit symbol imports and relative
imports are accepted here, differing from Google's import restrictions. Preserve
deliberate optional-dependency guards and lazy imports where they are needed.

After changing Python imports, run:

```bash
uv run isort path/to/changed_file.py
```

### Running the checks

In a configured development environment, with the lint tools on `PATH`, run
from the repository root:

```bash
# Format and lint changes relative to the PR's target branch.
BASE_REF=main CHECK_ONLY=false bash tools/autoformat.sh

# Check without applying formatting fixes.
BASE_REF=main CHECK_ONLY=true SKIP_DOCS=false bash tools/autoformat.sh
```

Set `BASE_REF` to the actual target branch. The script fetches that branch from
NVIDIA/Megatron-LM and selects changed, tracked Python files under `megatron/core`
and `tests`. Untracked files and files elsewhere need explicit file checks.
Pre-commit hooks have their own, narrower scope; see
[.pre-commit-config.yaml](.pre-commit-config.yaml).

[tools/autoformat.sh](tools/autoformat.sh) runs Black, isort, Pylint, Ruff, and
attempts mypy. Mypy failures are currently non-blocking, and mypy is not included
in the linting dependency group. Pylint enables a small set of checks, including
missing class/function docstrings, unused imports, long lines, and `print` usage;
see [.pylintrc](.pylintrc). Ruff currently selects `S506` (unsafe YAML loading),
not a general Google-style or docstring rule set. Passing these checks does not
establish complete type or style correctness.

For environment setup and formatting commands, see the
[linting skill](skills/mcore-linting-and-formatting/SKILL.md).

## Interfaces

- Avoid shadowing builtins or names from an enclosing scope when it obscures
  their meaning.
- Initialize ordinary instance state in `__init__`. Document state that is
  intentionally created later; properties, dataclass initialization, and staged
  setup do not require artificial constructor assignments.
- Dynamic interfaces remain appropriate for module factories and compatibility
  adapters.

## Type hints

Annotate constructor return types with `-> None`.
Document tensor shapes, layout, dtype requirements, and ownership in docstrings
where a `torch.Tensor` annotation cannot express the contract.

No current lint rule chooses between annotation spellings. Preserve nearby
annotation conventions when extending an existing interface.

## Docstrings and comments

Use **MyST (Markdown) markup** within Google-style docstrings, as configured by
[docs/conf.py](docs/conf.py) and the
[docstring parser](docs/autodoc2_docstrings_parser.py). Use Markdown backticks
for inline code and MyST syntax for links, lists, and math.

The contribution policy requires removing commented-out code before submitting
a change.

## Configuration

Extend the existing configuration dataclass when adding an option to a component.
For new configuration objects, prefer dataclasses with typed fields. Put a
docstring immediately after each public configuration field, following
[TransformerConfig](megatron/core/transformer/transformer_config.py).

Make required values explicit where the existing inheritance/API permits it.
Choose meaningful defaults and explain sentinel values such as `None` or `0`.
Use `field(default_factory=...)` for mutable defaults. Validate invalid values and
incompatible options in the existing validation path, often `__post_init__`.

```python
from dataclasses import dataclass


@dataclass
class LayerOptions:
    """Illustrative configuration for a layer."""

    hidden_size: int
    """Width of the hidden representation."""

    dropout: float = 0.0
    """Dropout probability; zero disables dropout."""
```

Use `NamedTuple` for a small immutable record when that fits the interface;
it is not a general replacement for configuration dataclasses.

## Logging

In Core library code, use a module logger (`logging.getLogger(__name__)`) for
diagnostics. Pylint flags direct `print` calls. Follow existing rank-aware logging
patterns when a message should appear only once across distributed workers.
Training entry points may use their existing training logging helpers.

## Process groups in Megatron Core

For new production code in `megatron/core`, accept a
[`ProcessGroupCollection`](megatron/core/process_groups_config.py) or an explicit
`torch.distributed.ProcessGroup` from the caller and pass it through. Avoid adding
direct reads of global groups through `parallel_state.get_*_group()` or directly
imported group getters.

Compatibility points include `parallel_state.py`, `process_groups_config.py`,
bootstrap code that constructs a collection from global state, tests, docs, and
migration fallbacks with an explicit comment. This is review guidance, not a CI
gate. It applies to Core library code; applying the same migration to
`megatron/training` or other training-loop code requires that change's scope to
include it. See [AGENTS.md](AGENTS.md) for the repository policy.

## Tests

Add focused regression coverage for behavior changes. Reuse nearby fixtures and
helpers, assert externally meaningful behavior, and cover relevant edge cases.
Keep tests appropriate to the change; documentation-only changes do not need GPU
training runs.

- Put unit tests in `tests/unit_tests/` and use `test_*.py` filenames. Reuse
  [shared fixtures](tests/unit_tests/conftest.py) and
  [distributed helpers](tests/unit_tests/test_utilities.py) where applicable.
- Many unit tests need GPUs and distributed initialization. Match the process
  count and hardware to the test; do not assume every test is a CPU test or
  requires eight GPUs. The [CI runner](tests/unit_tests/run_ci_test.sh) defines
  the CI launch and marker filters.
- Put end-to-end cases in `tests/functional_tests/test_cases/`, with model
  configuration and appropriate recipe entries under `tests/test_utils/recipes/`.
  Follow an existing case and the [functional runner](tests/functional_tests/shell_test_utils/run_ci_test.sh).
  Record hardware and parallelism requirements. Investigate numerical differences
  before updating golden values.

Markers are declared in `pyproject.toml`; the CI runner applies filters:

| Marker | CI behavior |
| --- | --- |
| `internal` | Excluded when running the legacy suite. |
| `flaky` | Excluded in the LTS environment. |
| `flaky_in_dev` | Excluded in the dev environment. |
| `launch_on_gb200` | Selects tests for the GB200 unit-test run. |

The first three markers do not skip tests by themselves in an ordinary pytest
invocation. Apply them for their actual CI purpose, not as general labels for
private helpers or distributed tests. For detailed workflows, see the
[testing skill](skills/mcore-testing/SKILL.md).

## Documentation and file headers

Document new user-facing features with their purpose, configuration, constraints,
and usage examples. Explain implementation details when they help users or
maintainers work with the feature.

When adding or moving a documentation page, update its appropriate toctree and
incoming links. Navigation may live in [docs/index.md](docs/index.md) or a nested
index. Use the [documentation build instructions](docs/developer/generate_docs.md)
to check rendered content.

Preserve existing copyright and license notices. For a new NVIDIA-authored Python
file, the current CI checker suggests this short header (use the year of creation):

```python
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
```

The repository contains both short copyright notices and longer license headers;
do not mechanically replace existing notices with one universal template. Follow
the applicable convention in nearby files.

The [copyright workflow](.github/workflows/copyright-check.yml) currently checks
changed Python files, not shell scripts. Its
[checker](https://github.com/NVIDIA-NeMo/FW-CI-templates/blob/v0.66.7/.github/actions/copyright-checker/check_copyright.py)
looks for a copyright comment in the initial comment block; it does not enforce
the exact holder wording or year.
