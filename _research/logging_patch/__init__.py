"""Apertus 2 ablations logging and reproducibility patch.

Importing this module has no side effects. Call ``install()`` from
``pretrain_gpt.py`` (before ``pretrain(...)``) to wire the hooks in.

See ``README.md`` for the full list of environment variables and the JSON
schema written to ``_research/results/performance/<run_name>.json``.
"""

from .install import install

__all__ = ["install"]
