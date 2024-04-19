import os
import time
import urllib.request as req

import numpy as np
import pytest
import torch

import megatron.core.utils as util
from tests.unit_tests.test_utilities import Utils


def test_divide_properly():
    assert util.divide(4,2) == 2

def test_divide_improperly():
    with pytest.raises(AssertionError):
        util.divide(4,5)

def test_global_memory_buffer():
    global_memory_buffer = util.GlobalMemoryBuffer()
    obtained_tensor = global_memory_buffer.get_tensor((3,2), torch.float32, "test_tensor")
    expected_tensor = torch.empty((3,2), dtype=torch.float32, device=torch.cuda.current_device())
    assert torch.equal(obtained_tensor, expected_tensor)

def test_make_viewless_tensor():
    inp = torch.rand((3,4))
    assert(torch.equal(inp, util.make_viewless_tensor(inp, True, True)))
    assert(torch.equal(inp, util.make_viewless_tensor(inp, True, False)))

def test_safely_set_viewless_tensor_data():
    tensor = torch.zeros((3,4))
    new_data_tensor = torch.tensor(np.random.rand(3,4))
    util.safely_set_viewless_tensor_data(tensor, new_data_tensor)
    assert(torch.equal(tensor, new_data_tensor))

def test_assert_viewless_tensor():
    tensor = torch.rand((3,4))
    assert(torch.equal(util.assert_viewless_tensor(tensor), tensor))
    input_tensor_list=[tensor,tensor,tensor]
    output_tensor_list = util.assert_viewless_tensor(input_tensor_list)
    for inp,out in zip(input_tensor_list, output_tensor_list):
        assert(torch.equal(inp,out))

def test_straggler_detector():
    # Environment from Workload manager
    world = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))
    master = os.getenv('MASTER_ADDR', 'localhost')
    master_port = int(os.getenv('MASTER_PORT', '60000'))
    port = 65535

    # Helpers
    # initialize torch.distributed
    # do not call init_process_group here, call Utils.initialize_distributed()
    def init_distributed():
        Utils.initialize_distributed()
        # Validate Environment from Workload manager
        assert torch.distributed.is_initialized() == True
        assert torch.distributed.get_rank() == rank
        assert torch.cuda.device_count() == world
        torch.distributed.barrier()

    # deinit and cleanup
    # do not call torch.distributed.destroy_process_group, may be needed by other tests
    def deinit_distributed():
        assert torch.distributed.is_initialized() == True
        torch.distributed.barrier()

    # checks if the instance is disabled
    def straggler_detector_disabled():
        assert stimer.enabled == False

    # checks if the instance is enabled
    def straggler_detector_enabled():
        assert stimer.enabled == True

    # enable, simulate one rank only on global rank-0
    def straggler_detector_enable():
        if rank == 0:
            resp = req.urlopen(f"http://{master}:{port}").read().decode().split()
            assert resp[3] == "ON"
        # call the reporting function, this will propagate the change
        stimer.report()

    # time an operation
    def straggler_detector_timeit():
        s = 2  # sleep for 2 sec
        M = 20
        K = 30
        N = 40
        mat1 = torch.randn(M, K, device='cuda')
        mat2 = torch.randn(K, N, device='cuda')
        # batch_data
        with stimer(bdata=True):
            time.sleep(s)
        # GEMM
        with stimer:
            res = torch.matmul(mat1, mat2)
        delta, batch_delta, _, _, _, _, = stimer.elapsed()
        assert delta > 0.0
        assert batch_delta >= s

    # reporting
    def straggler_detector_report():
        s = 2  # sleep for 2 sec
        N = 20
        P = 30
        M = 40
        mat1 = torch.randn(N, P, device='cuda')
        mat2 = torch.randn(P, M, device='cuda')
        tfp = (N * M) * (2 * P - 1)  # theoretical
        iter = 10  # mock
        # batch_data
        with stimer(bdata=True):
            time.sleep(s)
        # GEMM
        with stimer:
            res = torch.matmul(mat1, mat2)
        r = stimer.report(total_flops=tfp, log_interval=iter)
        rb = True if rank == 0 else False
        assert r == rb

    # Test steps start..
    # init
    init_distributed()

    # create a straggler_detector with enabled set to false
    stimer = util.StragglerDetector()
    stimer.configure(world, rank, enabled=False, port=port)
    # check if configuration was success
    assert stimer.configured == True

    # check if the instance is in disabled state
    straggler_detector_disabled()
    # enable it now, must call report
    straggler_detector_enable()
    # check if all ranks had it enabled
    straggler_detector_enabled()
    # time some operation
    straggler_detector_timeit()
    # report only from rank=0
    straggler_detector_report()

    # cleanup
    deinit_distributed()
