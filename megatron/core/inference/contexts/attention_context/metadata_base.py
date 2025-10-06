# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import logging


class MetadataBase:
    """
    Base class for attention metadata.
    """

    def __init__(self, debug):
        """
        Initialize the metadata.
        """
        self.debug = debug
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
        tensor,
        real_batch_size,
        graph_batch_size,
        is_cumulative_tensor=False,
        pad_value=0,
    ):
        """
        Copy the tensor to the tensor_buf,
        pad the tensor_buf with zero or the last value of the tensor,
        depending on whether the tensor is cumulative.
        Args:
            tensor_buf: The destination tensor.
            tensor: The tensor to copy.
            real_batch_size: The real batch size.
            graph_batch_size: Padded boundary of the tensor.
            is_cumulative_tensor: Whether the tensor is cumulative.
        """
        assert real_batch_size <= graph_batch_size
        assert tensor_buf.shape[0] >= graph_batch_size
        assert tensor.shape[0] >= real_batch_size
        if is_cumulative_tensor:
            if real_batch_size == 0:
                value = pad_value
            else:
                value = tensor[real_batch_size - 1]
        else:
            value = pad_value
        tensor_buf[0:real_batch_size] = tensor[:real_batch_size]
        tensor_buf[real_batch_size:graph_batch_size] = value
        return tensor_buf

    def print_all_data(self):
        """
        Print all the data in the metadata.
        """
        for key, value in self.state_data.items():
            logging.info(f"{key}: {value}")
