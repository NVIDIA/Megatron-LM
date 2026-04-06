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

    def update(self, *args, **kwargs):
        """
        Construct the metadata from request states.
        """
        pass

    def reset(self):
        """
        Reset the metadata.
        """
        pass

    def tensor_pad(
        self,
        tensor_buf,
        real_batch_size,
        padded_batch_size,
        is_cumulative_tensor=False,
        pad_value=0,
    ):
        """Pad tensor_buf.

        For non-cumulative tensors,
          the pad region starts at real_batch_size and fills with pad_value.
        For cumulative tensors,
          the pad region starts at real_batch_size + 1 and fills with the last entry.
        """
        if is_cumulative_tensor:
            if real_batch_size == 0:
                value = pad_value
            else:
                value = tensor_buf[real_batch_size]
            tensor_slice = slice(real_batch_size + 1, padded_batch_size + 1)
        else:
            value = pad_value
            tensor_slice = slice(real_batch_size, padded_batch_size)
        tensor_buf[tensor_slice].fill_(value)

    def __str__(self):
        """
        Return a string representation of the metadata.
        """
        return "\n".join([f"{key}: {value}" for key, value in self.state_data.items()])
