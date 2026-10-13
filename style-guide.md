# Megatron-LM Style Guide

Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
with the repository overrides and additional guidance below:

- **Formatting:** Use a **100-character** line limit and Black with the settings in
  [pyproject.toml](pyproject.toml), including disabled string normalization and
  magic trailing comma behavior.
- **Strings:** Use double quotes for strings. Black preserves quote style, so
  this convention is checked in review.
- **Imports:** Use isort's **Black profile** and repository configuration.
  Explicit symbol imports and relative imports are allowed. Preserve required
  optional-dependency guards and lazy imports.
- **Linting:** Use the repository's [Pylint configuration](.pylintrc) and
  [Ruff configuration](pyproject.toml).
- **Constructors:** Annotate constructor return types with `-> None`.
- **Configuration documentation:** Put MyST docstrings immediately after each
  public configuration field declaration, following
  [TransformerConfig](megatron/core/transformer/transformer_config.py).
- **Dynamic interfaces:** Dynamic imports and reflection are allowed in module
  factories and compatibility adapters.

## Dataclasses

Keep each dataclass specific to the component or concept it represents. Use a
descriptive name and explicit, typed fields so its purpose and valid state are clear.

- Avoid catch-all dataclasses that represent unrelated configurations through mode
  flags, arbitrary option dictionaries, or many fields that apply to only one case.
- Model distinct concerns with separate dataclasses and compose them when needed.
- Share a base dataclass only when its fields have the same meaning and validation
  rules across the components that use it. Generalize when concrete use cases
  demonstrate the need.
