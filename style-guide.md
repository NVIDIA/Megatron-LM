# Megatron-LM Coding Guidelines

Note: This repository is Python-first. Prefer the Python guidelines in this document.

## Style Guides We Follow

If there is a conflict between this file and the Google style guides, prioritize this file's guidance.

- Python: [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- Shell: [Google Shell Style Guide](https://google.github.io/styleguide/shellguide.html)

## Python Coding Guidelines

### Python Standard

1. The code developed for Megatron-LM should conform to Python 3.10+.

### Line Length

1. Maximum line length is **100 characters**.

### Indentation

1. Indent code with 4 spaces. Do not use tabs.

### Naming

#### Identifier Format

1. Files
   - snake_case: `some_file.py`

2. Classes
   - PascalCase: `class SomeClass`

3. Functions and Methods
   - snake_case: `def my_awesome_function():`

4. Local Variables
   - snake_case: `my_variable = ...`
   - prefix `k` for variable names that start with a number: `k_99th_percentile = ...`

5. Global Variables
   - upper snake_case: `MY_GLOBAL = ...`

6. Constants
   - upper snake_case: `MY_CONSTANT = ...`

#### Identifier Guidelines

1. Avoid shadowing variables declared in an outer scope.
2. Initialize all externally visible members of a class in the constructor.

### Imports

Organize imports in the following order, separated by blank lines:

1. Future imports
2. Standard library imports
3. Third-party imports (including `torch`, `transformers`)
4. First-party imports (`megatron.*`)
5. Local folder imports

Example:

```python
from __future__ import annotations

import abc
import logging

import torch
from transformers import PreTrainedModel

from megatron.core import parallel_state as mpu
```

### String Quotes

1. Use **double quotes** for strings (matching ruff formatter configuration).

### Comments

1. For interfaces that may be used outside a file, prefer docstrings over comments.
2. Comments should be reserved for code within a function, or interfaces that are local to a file.
3. If a piece of code is commented out, there should be a comment around that piece of code describing its usage and why it's commented out. Otherwise that is a debug comment and it should be removed before merging.

### Docstring Syntax

#### Classes and Functions

Use the [Google style](https://google.github.io/styleguide/pyguide.html), which can be parsed by Sphinx.

Example:

```python
def sharded_state_dict(
    self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[Dict] = None
) -> ShardedStateDict:
    """Sharded state dict implementation for GPTModel backward-compatibility.

    Removing extra state.
    Tie word embeddings and output layer in mtp process stage.

    Args:
        prefix (str): Module name prefix.
        sharded_offsets (tuple): PP related offsets, expected to be empty at this module level.
        metadata (Optional[Dict]): metadata controlling sharded state dict creation.

    Returns:
        ShardedStateDict: sharded state dict for the GPTModel
    """
    ...
```

### Error Handling

1. When using try-except blocks, limit the except to the smallest set of errors possible.

For example, instead of:

```python
try:
    open(path, "r").read()
except:
    print("Failed to open file")
```

Do:

```python
try:
    open(path, "r").read()
except FileNotFoundError:
    print("Failed to open file")
```

2. When using try-except blocks to handle multiple possible variable types (i.e. duck-typing), keep the body of the try as small as possible, using the else block to implement the logic.

For example, instead of:

```python
try:
    f.seek(0)
    f.read()
except AttributeError:
    ... # Not a file-like object, do something else
```

Do:

```python
try:
    f.seek  # Do not call to minimize chance of unrelated failure
except AttributeError:
    ... # Not a file-like object, do something else
else:
    f.seek(0)
    f.read()
```

### Type Hints

1. Use type hints for function arguments and return types.
2. Use `T | None` for nullable types (not `Optional[T]`).
3. Use `X | Y` for union types (not `Union[X, Y]`).
4. Use `TypeVar` for generic type parameters.
5. Use built-in generics (`list`, `dict`, `tuple`) instead of `typing` equivalents.

Example:

```python
from typing import TypeVar

T = TypeVar("T", bound=torch.nn.Module)

def get_module_by_name(
    model: T,
    name: str,
    default: torch.nn.Module | None = None,
) -> torch.nn.Module | None:
    """Get a module from a model by its name."""
    ...

def convert_weights(
    weights: torch.Tensor | dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert weights, accepting either a single tensor or a dict."""
    ...
```

### Configuration and Dataclasses

1. Use `dataclasses` or `NamedTuple` for configuration objects.
2. Be explicit about required vs optional fields.
3. Do not add arbitrary defaults for configs; be as explicit as possible.

Example:

```python
from dataclasses import dataclass

@dataclass
class GPTModel(LanguageModule):
"""GPT Transformer language model.

Args:
    config (TransformerConfig):
        Transformer config
    transformer_layer_spec (ModuleSpec):
        Specifies module to use for transformer layers
    vocab_size (int):
        Vocabulary size
    max_sequence_length (int):
        maximum size of sequence. This is used for positional embedding
    pre_process (bool, optional):
        Include embedding layer (used with pipeline parallelism). Defaults to True.
    post_process (bool, optional):
        Include an output layer (used with pipeline parallelism). Defaults to True.
    fp16_lm_cross_entropy (bool, optional):
        Defaults to False.
"""

def __init__(
    self,
    config: TransformerConfig,
    transformer_layer_spec: ModuleSpec,
    vocab_size: int,
    max_sequence_length: int,
    pre_process: bool = True,
    post_process: bool = True,
    fp16_lm_cross_entropy: bool = False
) -> None:
    ...
```

### Avoid Reflection

Avoid using reflection when functionality can be easily achieved without reflection.

For example, instead of:

```python
def make_complex(*args):
    x, y = args
    return dict(**locals())
```

Do:

```python
def make_complex(x, y):
    return {"x": x, "y": y}
```

## Documentation Guidelines

### Ensure docs/index.md is up to date

When a new markdown doc is added under `docs/**/*.md` or a markdown file is renamed, ensure that `docs/index.md` is updated and the document appears in the most appropriate section.

### Documentation Requirements

**Important**: All new key features (e.g., enabling a new model, enabling a new parallelism strategy) must include documentation updates. This should:

- Explain the motivation and purpose of the feature
- Outline the technical approach and architecture
- Provide clear usage examples and instructions for users
- Document internal implementation details where appropriate

## Testing Guidelines

### Unit Tests

- Place unit tests in `tests/unit_tests/`
- Name test files with `test_` prefix: `test_gpt_model.py`
- Use pytest fixtures for common setup
- Use `pytest.mark` to categorize tests (unit, integration, system)

### Functional Tests

- Place functional tests in `tests/functional_tests/`
- Use subprocess for tests that require process isolation
- Document hardware requirements for GPU tests

### Test Markers

Use appropriate pytest markers:

```python
import pytest

@pytest.mark.unit
def test_parameter_mapping():
    """Test that parameter mapping is correct."""
    ...

@pytest.mark.integration
def test_model_loading():
    """Test end-to-end model loading."""
    ...
```

## NVIDIA Copyright

Add the following NVIDIA copyright header to all Python files and shell scripts. The header should appear at the top of the file:

```python
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
```