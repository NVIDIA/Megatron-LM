from typing import Literal, Optional

from .backend import (
    ContextParallelHandler,
    DefaultContextParallelHandler,
    MagiAttnContextParallelHandler,
    TEDynamicContextParallelHandler,
)


def get_cp_handler_cls(
    backend: Optional[Literal["transformer_engine", "local", "magi"]] = None,
    cp_comm_type: Optional[str] = None,
) -> type[ContextParallelHandler]:
    """
    Factory function to get the appropriate Context Parallel Handler class based on the backend.

    Args:
        backend: The attention backend to use ('transformer_engine', 'local', or 'magi').
        cp_comm_type: Optional communication type identifier (unused in current logic).

    Returns:
        The class definition of the appropriate ContextParallelHandler.

    Raises:
        ValueError: If an unsupported backend is provided.
    """
    if backend == "transformer_engine" or backend == "local":
        return DefaultContextParallelHandler
    elif backend == "magi":
        return MagiAttnContextParallelHandler
    elif backend == "transformer_engine_dynamic":
        return TEDynamicContextParallelHandler
    else:
        raise ValueError(f"Unsupported attention backend for context parallel: {backend}")
