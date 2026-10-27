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

    def tensor_copy_and_pad(
        self,
        tensor_buf,
        unpadded_tensor,
        real_batch_size,
        padded_batch_size,
        is_cumulative_tensor=False,
        pad_value=0,
    ):
        """
        Copy the unpadded tensor to the tensor_buf,
        pad the tensor_buf with zero or the last value of the tensor,
        depending on whether the tensor is cumulative.
        Args:
            tensor_buf: The destination tensor, at least padded_batch_size long.
            unpadded_tensor: The tensor to copy, at least real_batch_size long.
            real_batch_size: The real batch size.
            padded_batch_size: Padded boundary of the tensor.
            is_cumulative_tensor: Whether the tensor is cumulative.
                If True, we pad the tensor_buf with the last value of the unpadded_tensor.
            pad_value: The value to pad the tensor_buf with when the tensor is not cumulative.
        """
        assert real_batch_size <= padded_batch_size
        assert tensor_buf.shape[0] >= padded_batch_size
        assert unpadded_tensor.shape[0] >= real_batch_size
        if is_cumulative_tensor:
            if real_batch_size == 0:
                value = pad_value
            else:
                value = unpadded_tensor[real_batch_size - 1]
        else:
            value = pad_value
        tensor_buf[0:real_batch_size] = unpadded_tensor[:real_batch_size]
        tensor_buf[real_batch_size:padded_batch_size] = value
        return tensor_buf

    def __str__(self):
        """
        Return a string representation of the metadata.
        """
        return "\n".join([f"{key}: {value}" for key, value in self.state_data.items()])
