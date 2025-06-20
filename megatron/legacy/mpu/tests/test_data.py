# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from commons import print_separator
from commons import initialize_distributed
from mpu import data as data_utils
import mpu
import torch
import functools
import operator
import sys
sys.path.append("../..")


def test_broadcast_data(tensor_model_parallel_size):

    if torch.distributed.get_rank() == 0:
        print('> testing broadcast_data with model parallel size {} ...'.
              format(tensor_model_parallel_size))

    mpu.initialize_model_parallel(tensor_model_parallel_size)
    torch.manual_seed(1234 + mpu.get_data_parallel_rank())
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    key_size_t = {'key1': [7, 11],
                  'key2': [8, 2, 1],
                  'key3': [13],
                  'key4': [5, 1, 2],
                  'key5': [5, 12]}
    keys = list(key_size_t.keys())

    data = {}
    data_t = {}
    for key in key_size_t:
        data[key] = torch.LongTensor(size=key_size_t[key]).random_(0, 1000)
        data_t[key] = data[key].clone()
    data['keyX'] = torch.FloatTensor(size=(5, )).random_(0, 1000)
    data_t['keyX'] = data['keyX'].clone()
    if mpu.get_tensor_model_parallel_rank() != 0:
        data = None

    data_utils._check_data_types(keys, data_t, torch.int64)
    key_size, key_numel, \
        total_numel = data_utils._build_key_size_numel_dictionaries(keys, data)
    for key in keys:
        assert key_size[key] == key_size_t[key]
    total_numel_t = 0
    for key in keys:
        target_size = functools.reduce(operator.mul, key_size_t[key], 1)
        assert key_numel[key] == target_size
        total_numel_t += target_size
    assert total_numel == total_numel_t

    data_b = data_utils.broadcast_data(keys, data, torch.int64)
    for key in keys:
        tensor = data_t[key].cuda()
        assert data_b[key].sub(tensor).abs().max() == 0

    # Reset groups
    mpu.destroy_tensor_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print('>> passed the test :-)')


if __name__ == '__main__':

    initialize_distributed()
    world_size = torch.distributed.get_world_size()

    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        print_separator('test test broadcast data')
        test_broadcast_data(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2
