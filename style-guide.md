# Megatron-LM Coding Guidelines

Note: This repository is Python-first. Prefer the Python guidelines in this document.

## Style Guides We Follow

If there is a conflict between this file and the Google style guides, prioritize this file's guidance.

- Python: [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- Shell: [Google Shell Style Guide](https://google.github.io/styleguide/shellguide.html)

## Python Coding Guidelines

### Python Standard

1. The code developed for Megatron-LM should conform to Python 3.12+.

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

1. Use **double quotes** for strings.

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
2. Use `Optional[T]` for nullable types.
3. Use `Union[X, Y]` for union types.
4. Use `TypeVar` for generic type parameters.
5. Use `typing` generics (`List`, `Dict`, `Tuple`) for type annotations. Built-in generics (`list`, `dict`) are also acceptable.

Example:

```python
from typing import Dict, List, Optional, TypeVar, Union

T = TypeVar("T", bound=torch.nn.Module)

def get_module_by_name(
    model: T,
    name: str,
    default: Optional[torch.nn.Module] = None,
) -> Optional[torch.nn.Module]:
    """Get a module from a model by its name."""
    ...

def convert_weights(
    weights: Union[torch.Tensor, Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Convert weights, accepting either a single tensor or a dict."""
    ...
```

### Configuration and Dataclasses

1. Use `dataclasses` or `NamedTuple` for configuration objects.
2. Be explicit about required vs optional fields.
3. Do not add arbitrary defaults for configs; be as explicit as possible.

Example:

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TransformerConfig(ModelParallelConfig):
    """Configuration object for megatron-core transformers.

    The initialization function has an argument for each parameter,
    including those in ModelParallelConfig.
    """

    num_layers: int = field(default=0, metadata={"argparse_meta": {"default": None}})
    """Number of transformer layers in a transformer block."""

    mtp_num_layers: Optional[int] = None
    """Number of Multi-Token Prediction (MTP) Layers."""

    mtp_loss_scaling_factor: Optional[float] = 0.1
    """Weighting factor of Multi-Token Prediction (MTP) loss."""

    hidden_size: int = field(default=0, metadata={"argparse_meta": {"default": None}})
    """Transformer hidden size."""

    num_attention_heads: int = field(default=0, metadata={"argparse_meta": {"default": None}})
    """Number of transformer attention heads."""
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
- Use `pytest.mark` to categorize tests (`internal`, `flaky`, `flaky_in_dev`)

### Functional Tests

- Place functional tests in `tests/functional_tests/`
- Use subprocess for tests that require process isolation
- Document hardware requirements for GPU tests

### Test Markers

Use appropriate pytest markers:

```python
import pytest

@pytest.mark.internal
def test_private_helper():
    """Test a private/internal function."""
    ...

@pytest.mark.flaky
@pytest.mark.flaky_in_dev
def test_distributed_grad_sync():
    """Test that may be flaky in both LTS and DEV environments."""
    ...
```

## NVIDIA Copyright

Add the following NVIDIA copyright header to all Python files and shell scripts. The header should appear at the top of the file:

```python
# Copyright (c) 2026 NVIDIA CORPORATION. All rights reserved.
```