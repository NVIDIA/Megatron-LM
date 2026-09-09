# Megatron-LM Style Guide

Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
with these repository overrides:

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
