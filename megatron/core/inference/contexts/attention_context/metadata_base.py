import torch

class MetadataBase:
    def __init__(self, debug):
        self.debug = debug

    def update(self, *args, **kwargs):
        pass

    def reset(self):
        pass

    def extend_save(self, tensor_buf, tensor, real_batch_size, graph_batch_size, value=None):
        assert real_batch_size <= graph_batch_size
        assert tensor_buf.shape[0] >= graph_batch_size
        assert tensor.shape[0] >= real_batch_size
        if value is None:
            if real_batch_size == 0:
                value = 0
            else:
                value = tensor[real_batch_size - 1]
        tensor_buf[0:real_batch_size] = tensor[:real_batch_size]
        tensor_buf[real_batch_size:graph_batch_size] = value
        return tensor_buf

    def print_all_data(self):
        for key, value in self.all_data.items():
            print(f"{key}: {value}")