
import os

import torch
import torch.distributed as dist

from megatron.core.optimizer.muon import Muon, MuonDistMeta, normalize_range

def is_rank_0():
    return torch.distributed.get_rank() == 0

def print_rank_0(*args):
    if is_rank_0():
        print(*args)

def cdiv(x: int, y: int):
    return (x + y - 1) // y

def gen_param_and_grads():

    # reset manual seed
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = 'cuda'
    dtype = torch.float32

    # gen params
    params = [ torch.randn(shape, device=device, dtype=dtype) for shape in [
            (100, 100), (124, 324), (456, 124), (676, 876), (128, 128), ] ]

    # gen grads [ [ grad-list ] * step ]
    grads = [ [ torch.randn_like(param) for param in params ] for _ in range(10) ]
    
    return params, grads
    
def distribute_params(params, grads, tp_dims, dist_group, tp_group):
    """ 将 param 进行 dist & tp shard, 仅保留自己的一部分 """

    params = params.copy()
    grads = [ step_grads.copy() for step_grads in grads ]

    # tp dist
    tp_size = dist.get_world_size(tp_group)
    tp_rank = dist.get_rank(tp_group)
    for i, param in enumerate(params):
        tp_dim = tp_dims[i]
        if tp_dim == -1:
            continue
        assert param.shape[tp_dim] % tp_size == 0
        local_range_start = param.shape[tp_dim] // tp_size * tp_rank
        local_range_end = param.shape[tp_dim] // tp_size * (tp_rank + 1)
        params[i] = param[local_range_start:local_range_end, :] if tp_dim == 0 else \
                    param[:, local_range_start:local_range_end].contiguous()
        
        for step_grads in grads:
            step_grads[i] = step_grads[i][local_range_start:local_range_end, :] if tp_dim == 0 else \
                            step_grads[i][:, local_range_start:local_range_end].contiguous()

    # distributed
    world_size = dist.get_world_size(dist_group)
    rank = dist.get_rank(dist_group)

    global_buffer_size = sum(param.numel() for param in params)
    local_buffer_size = cdiv(global_buffer_size, world_size)
    local_buffer_range = (local_buffer_size * rank, local_buffer_size * (rank + 1))
    global_buffer_size = local_buffer_size * world_size # fix global buffer size
    
    numel_acc = 0
    dist_params = []
    dist_grads = [[] for _ in grads]
    dist_metas = {}
    for i, param in enumerate(params):

        # gen meta
        numel = param.numel()
        dist_meta = MuonDistMeta(0, 0, param.shape, (numel_acc, numel_acc + numel), tp_dims[i])
        dist_meta.set_local_buffer_range(local_buffer_range)
        numel_acc += numel

        # skip if no element in this shard
        if dist_meta.local_range[0] == dist_meta.local_range[1]:
            continue

        # gen param
        local_range = normalize_range(dist_meta.local_range, dist_meta.global_range[0])
        dist_param = param.view(-1)[local_range[0]:local_range[1]]
        dist_params.append(dist_param)
        dist_metas[dist_param] = dist_meta

        # gen grad
        for step, step_grads in enumerate(grads):
            dist_grad = step_grads[i].view(-1)[local_range[0]:local_range[1]]
            dist_grads[step].append(dist_grad)

    return dist_params, dist_grads, global_buffer_size, dist_metas


def test_muon_dist(dp_size, tp_size):

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert dp_size * tp_size == world_size

    # init dist group
    for i in range(tp_size):
        ranks = range(i, world_size, tp_size)
        group = dist.new_group(ranks)
        if rank in ranks:
            dist_group = group
    # init tp group
    for i in range(dp_size):
        ranks = range(i * tp_size, (i + 1) * tp_size)
        group = dist.new_group(ranks)
        if rank in ranks:
            tp_group = group

    print_rank_0("process group initialized")

    params_ref, grads_ref = gen_param_and_grads()
    params_test, grads_test = gen_param_and_grads()
    tp_dims = [0, 1, -1, 1, 0]

    params_test, grads_test, global_buffer_size, dist_metas \
         = distribute_params(params_test, grads_test, tp_dims, dist_group, tp_group)

    muon_args = {
        "use_muon": True,
        "lr": 0.1,
        "momentum": 0.9,
        "nesterov": True,
        "ns_steps": 5,
        "weight_decay": 0.1,
    }

    # gen params
    ref_param_groups = [{
        "params": params_ref,
        **muon_args
    }]
    test_param_groups = [{
        "params": params_test,
        **muon_args
    }]

    ref_muon  = Muon(ref_param_groups)
    test_muon = Muon(test_param_groups)
    test_muon.enable_distributed_mode([[(global_buffer_size, 0)]], dist_group, tp_group, dist_metas)

    for step in range(10):

        # add grad
        for i, grad in enumerate(grads_ref[step]):
            params_ref[i].grad = grad.clone()
        for i, grad in enumerate(grads_test[step]):
            params_test[i].grad = grad.clone()
        # step
        ref_muon.step()
        test_muon.step()
        # distribute ref params
        dist_ref_params, _, _, _ = distribute_params(params_ref, [], tp_dims, dist_group, tp_group)
        # verify
        for i, params_x2 in enumerate(zip(dist_ref_params, params_test)):
            assert (params_x2[0] == params_x2[1]).all(), f"rank {rank} param {i} verify failed"
        print_rank_0(f" - step {step} verify passed")
    
    print_rank_0(f"dist dp = {dp_size} tp = {tp_size} test passed")

def run_process(rank, world_size):

    # init dist
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    test_muon_dist(dp_size=4, tp_size=2)
    test_muon_dist(dp_size=2, tp_size=4)

    dist.destroy_process_group()

if __name__ == "__main__":

    world_size = 8
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

    torch.multiprocessing.spawn(run_process, args=(world_size,), nprocs=world_size, join=True)
