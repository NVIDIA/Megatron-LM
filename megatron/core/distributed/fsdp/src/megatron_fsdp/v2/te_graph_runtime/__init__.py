"""TE-compatible CUDA graph callable runtime.

Vendored from https://github.com/buptzyb/te-graph-runtime
Prefer ``pip install te-graph-runtime`` instead of the vendored copy.
"""

from .graph import (
    UPSTREAM_TE_COMMIT,
    UPSTREAM_TE_GRAPH_PATH,
    UPSTREAM_TE_VERSION,
    make_graphed_callables,
)

__all__ = [
    "UPSTREAM_TE_COMMIT",
    "UPSTREAM_TE_GRAPH_PATH",
    "UPSTREAM_TE_VERSION",
    "make_graphed_callables",
]

