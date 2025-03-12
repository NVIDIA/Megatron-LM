import os
import time
import urllib.request as req
from types import SimpleNamespace

import mock
import numpy as np
import pytest
import torch

import megatron.core.utils as util
import megatron.training.utils as training_util
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def test_divide_properly():
    assert util.divide(4, 2) == 2


def test_divide_improperly():
    with pytest.raises(AssertionError):
        util.divide(4, 5)


def test_global_memory_buffer():
    global_memory_buffer = util.GlobalMemoryBuffer()
    obtained_tensor = global_memory_buffer.get_tensor((3, 2), torch.float32, "test_tensor")
    expected_tensor = torch.empty((3, 2), dtype=torch.float32, device=torch.cuda.current_device())
    assert obtained_tensor.shape == expected_tensor.shape


def test_make_viewless_tensor():
    inp = torch.rand((3, 4))
    assert torch.equal(inp, util.make_viewless_tensor(inp, True, True))
    assert torch.equal(inp, util.make_viewless_tensor(inp, True, False))


def test_safely_set_viewless_tensor_data():
    tensor = torch.zeros((3, 4))
    new_data_tensor = torch.tensor(np.random.rand(3, 4))
    util.safely_set_viewless_tensor_data(tensor, new_data_tensor)
    assert torch.equal(tensor, new_data_tensor)


def test_assert_viewless_tensor():
    tensor = torch.rand((3, 4))
    assert torch.equal(util.assert_viewless_tensor(tensor), tensor)
    input_tensor_list = [tensor, tensor, tensor]
    output_tensor_list = util.assert_viewless_tensor(input_tensor_list)
    for inp, out in zip(input_tensor_list, output_tensor_list):
        assert torch.equal(inp, out)


# Initialize torch.distributed; do not call init_process_group here, call
# Utils.initialize_distributed() instead.
def _init_distributed(world, rank):
    Utils.initialize_distributed()
    assert torch.distributed.is_initialized() == True
    assert torch.distributed.get_rank() == rank
    assert torch.cuda.device_count() == world
    torch.distributed.barrier()


# Deinitialization and cleanup.
# Do not call torch.distributed.destroy_process_group, may be needed by other tests.
def _deinit_distributed():
    assert torch.distributed.is_initialized() == True
    torch.distributed.barrier()


@pytest.mark.flaky_in_dev
def test_check_param_hashes_across_dp_replicas():
    world = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))

    # Setup.
    _init_distributed(world, rank)
    Utils.initialize_model_parallel()
    model = torch.nn.Linear(100, 100, bias=False, device='cuda')

    # First check case where all replicas agree.
    model.weight.data.fill_(1.0)
    assert util.check_param_hashes_across_dp_replicas([model])

    # Now check case where replica 0 disagrees with all other replicas.
    if rank == 0:
        model.weight.data.fill_(0.0)
    param_hashes_match = util.check_param_hashes_across_dp_replicas([model])
    expected_param_hashes_match = rank == 0
    assert param_hashes_match == expected_param_hashes_match

    # Teardown.
    _deinit_distributed()


@pytest.mark.flaky_in_dev
def test_cross_check_param_hashes_across_dp_replicas():
    world = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))

    # Setup.
    _init_distributed(world, rank)
    Utils.initialize_model_parallel()
    model = torch.nn.Linear(100, 100, bias=False, device='cuda')

    # First check case where all replicas agree.
    model.weight.data.fill_(1.0)
    assert util.check_param_hashes_across_dp_replicas([model], True)

    # Now check case where replica 0 disagrees with all other replicas.
    if rank == 0:
        model.weight.data.fill_(0.0)
    assert not util.check_param_hashes_across_dp_replicas([model], True)

    # Teardown.
    _deinit_distributed()


@pytest.mark.parametrize("use_distributed_optimizer", [False, True])
@pytest.mark.flaky_in_dev
def test_param_norm(use_distributed_optimizer: bool):
    world = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))

    # Setup: distributed, model, mock_args.
    _init_distributed(world, rank)
    Utils.initialize_model_parallel()
    model = torch.nn.Linear(100, 100, bias=False, dtype=torch.bfloat16, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)
    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=use_distributed_optimizer)
    # Use dummy TransformerConfig which doesn't trigger __post_init__ assertions.
    model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
    )
    for param in model.parameters():
        assert param.requires_grad
    mock_args = SimpleNamespace(bf16=True)

    with mock.patch('megatron.training.utils.get_args', new=lambda: mock_args):
        # Make sure norm is correct when `main_param` attribute is not available.
        assert training_util.calc_params_l2_norm(
            model, force_create_fp32_copy=False
        ) == pytest.approx(100.0)
        assert training_util.calc_params_l2_norm(
            model, force_create_fp32_copy=True
        ) == pytest.approx(100.0)

        # Make sure norm is correct when `main_param` attribute is available.
        optimizer_config = OptimizerConfig(
            bf16=True, use_distributed_optimizer=use_distributed_optimizer
        )
        _ = get_megatron_optimizer(optimizer_config, [model])
        for param in model.parameters():
            assert hasattr(param, 'main_param')
            if use_distributed_optimizer:
                assert getattr(param, 'main_param_sharded', False)
        assert training_util.calc_params_l2_norm(
            model, force_create_fp32_copy=False
        ) == pytest.approx(100.0)
        assert training_util.calc_params_l2_norm(
            model, force_create_fp32_copy=True
        ) == pytest.approx(100.0)

    # Teardown.
    _deinit_distributed()


@pytest.mark.flaky_in_dev
def test_straggler_detector():
    world = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))
    master = os.getenv('MASTER_ADDR', 'localhost')
    port = 65535

    # Checks if the instance is disabled.
    def straggler_detector_disabled():
        assert stimer.enabled == False

    # Checks if the instance is enabled.
    def straggler_detector_enabled():
        assert stimer.enabled == True

    # Enable.
    def straggler_detector_enable():
        if rank == 0:
            resp = req.urlopen(f"http://{master}:{port}").read().decode().split()
            assert resp[3] == "ON"
        # Call the report function, this will propagate the change.
        stimer.report()

    # Time an operation.
    def straggler_detector_timeit():
        s = 2  # Sleep for 2 seconds.
        M = 20
        K = 30
        N = 40
        mat1 = torch.randn(M, K, device='cuda')
        mat2 = torch.randn(K, N, device='cuda')
        # batch_data.
        with stimer(bdata=True):
            time.sleep(s)
        # GEMM.
        with stimer:
            res = torch.matmul(mat1, mat2)
        delta, batch_delta, _, _, _, _ = stimer.elapsed()
        assert delta > 0.0
        assert batch_delta >= s

    # Test function to raise ValueError
    def straggler_value_error():
        raise ValueError("Exception value raised")

    # Check that exception is not suppressed.
    def straggler_detector_exception_propagate():
        # batch_data
        with pytest.raises(ZeroDivisionError):
            with stimer(bdata=True):
                x = 1 / 0
        # non-batch-data
        with pytest.raises(ValueError, match=r".* value .*"):
            with stimer():
                straggler_value_error()

    # Reporting.
    def straggler_detector_report():
        s = 2  # Sleep for 2 seconds.
        N = 20
        P = 30
        M = 40
        mat1 = torch.randn(N, P, device='cuda')
        mat2 = torch.randn(P, M, device='cuda')
        tfp = (N * M) * (2 * P - 1)  # Theoretical.
        iter = 10  # Mock.
        # batch_data.
        with stimer(bdata=True):
            time.sleep(s)
        # GEMM.
        with stimer:
            res = torch.matmul(mat1, mat2)
        r = stimer.report(total_flops=tfp, log_interval=iter)
        rb = True if rank == 0 else False
        assert r == rb

    # Start test.
    # Setup.
    _init_distributed(world, rank)

    # Create a straggler_detector with enabled set to false.
    stimer = util.StragglerDetector()
    stimer.configure(world, rank, enabled=False, port=port)
    # Check if configuration was success.
    assert stimer.configured == True

    # Check if the instance is in disabled state.
    straggler_detector_disabled()
    # Enable it now, must call report.
    straggler_detector_enable()
    # Check if all ranks have straggler detector enabled.
    straggler_detector_enabled()
    # Time some operation.
    straggler_detector_timeit()
    # Report only from rank 0.
    straggler_detector_report()
    # Check that exception is not suppressed.
    straggler_detector_exception_propagate()
    util.StragglerDetector._configured = False
    # Teardown.
    _deinit_distributed()
