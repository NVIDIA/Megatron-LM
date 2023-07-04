# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from commons import set_random_seed
from commons import IdentityLayer
from commons import print_separator
from commons import initialize_distributed
from mpu.cross_entropy import vocab_parallel_cross_entropy
import mpu
import torch.nn.functional as F
import torch
import random
import sys
from deepspeed.accelerator import get_accelerator
sys.path.append("../..")


def torch_cross_entropy(batch_size, seq_length, vocab_size,
                        logits_scale, seed):
    set_random_seed(seed)
    identity = IdentityLayer((batch_size, seq_length, vocab_size),
                             scale=logits_scale).to(get_accelerator().device_name())
    logits = identity()
    target = get_accelerator().LongTensor(
        size=(batch_size, seq_length)).random_(0, vocab_size)
    loss = F.cross_entropy(logits.view(-1, logits.size()[-1]),
                           target.view(-1),
                           reduction='none').view_as(target).mean()
    loss.backward()
    return loss, identity.weight.grad


def mpu_cross_entropy(batch_size, seq_length, vocab_size,
                      logits_scale, seed):
    set_random_seed(seed)
    identity = IdentityLayer((batch_size, seq_length, vocab_size),
                             scale=logits_scale).to(get_accelerator().device_name())
    logits = identity()
    logits_parallel = mpu.scatter_to_tensor_model_parallel_region(logits)
    target = get_accelerator().LongTensor(
        size=(batch_size, seq_length)).random_(0, vocab_size)
    loss = vocab_parallel_cross_entropy(logits_parallel, target).mean()
    loss.backward()
    return loss, identity.weight.grad


def test_cross_entropy(tensor_model_parallel_size):

    if torch.distributed.get_rank() == 0:
        print('> testing cross entropy with model parallel size {} ...'.
              format(tensor_model_parallel_size))

    mpu.initialize_model_parallel(tensor_model_parallel_size)
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    batch_size = 13
    seq_length = 17
    vocab_size_per_partition = 11
    logits_scale = 1000.0
    vocab_size = vocab_size_per_partition * tensor_model_parallel_size
    seed = 1234

    loss_torch, grad_torch = torch_cross_entropy(batch_size, seq_length,
                                                 vocab_size, logits_scale,
                                                 seed)
    loss_mpu, grad_mpu = mpu_cross_entropy(batch_size, seq_length,
                                           vocab_size, logits_scale,
                                           seed)

    error = loss_torch.sub_(loss_mpu).abs().max()
    print('   max error in loss on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    error = grad_torch.sub_(grad_mpu).abs().max()
    print('   max error in grad on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-6

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
        print_separator('test cross entropy')
        test_cross_entropy(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2
