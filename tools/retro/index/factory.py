# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from .indexes import FaissBaseIndex, FaissParallelAddIndex


class IndexFactory:
    '''Get index.

    Index type generally read from argument '--retro-index-ty'.
    '''

    @classmethod
    def get_index_class(cls, index_type):
        return {
            "faiss-base" : FaissBaseIndex,
            "faiss-par-add" : FaissParallelAddIndex,
        }[index_type]

    @classmethod
    def get_index(cls, index_type):
        index_class = cls.get_index_class(index_type)
        index = index_class()
        return index
