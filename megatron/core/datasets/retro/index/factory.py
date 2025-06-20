# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""The IndexFactory constructs an index from an index type string."""

from megatron.core.datasets.retro.index.index import Index

from .indexes import FaissBaseIndex, FaissParallelAddIndex


class IndexFactory:
    """Get index.

    Index type generally read from argument '--retro-index-ty'.
    """

    @classmethod
    def get_index_class(cls, index_type: str) -> type:
        """Get an index class, given a type string.

        Args:
            index_type (str): One of 'faiss-base' (naive Faiss index wrapper) or 'faiss-par-add' (Faiss index wrapper with near embarrassingly parallel index.add().

        Returns:
            An `Index` sub-type corresponding to the `index_type`.
        """
        return {"faiss-base": FaissBaseIndex, "faiss-par-add": FaissParallelAddIndex}[index_type]

    @classmethod
    def get_index(cls, index_type: str) -> Index:
        """Construct an index from an index type string.

        Args:
            index_type (str): One of 'faiss-base' (naive Faiss index wrapper) or 'faiss-par-add' (Faiss index wrapper with near embarrassingly parallel index.add().

        Returns:
            An `Index` instance corresponding to the `index_type`.
        """
        index_class = cls.get_index_class(index_type)
        index = index_class()
        return index
