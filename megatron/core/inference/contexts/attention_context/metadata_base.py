# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.


class MetadataBase:
    """
    Base class for attention metadata.
    High-performance attention kernels often require input metadata in specific
    formats—such as cumulative query lengths, cumulative key/value lengths,
    and similar structures. Moreover, when using CUDA Graphs, these metadata
    buffers must be statically allocated. This class serves as a unified container
    that manages all such metadata in one place.
    """

    def __init__(self):
        """
        Initialize the metadata.
        """
        self.state_data = {}

    def __str__(self):
        """
        Return a string representation of the metadata.
        """
        return "\n".join([f"{key}: {value}" for key, value in self.state_data.items()])
