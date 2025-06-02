import os
import time
import urllib.request as req
from types import SimpleNamespace
from unittest.mock import patch

import unittest.mock as mock
import numpy as np
import pytest
import torch

from megatron.core.device_utils import get_current_device, get_local_device_count, get_xla_model
import megatron.core.utils as util
import megatron.training.utils as training_util
from megatron.core import config
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

xm = get_xla_model()

success_string = "hello,world"


@util.experimental_cls(introduced_with_version="0.1.0")
class A:

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        self.device=get_current_device()
        
    def __init__(self):
        pass

    def some_method(self):
        return success_string

    @classmethod
    def some_static_method(cls):
        return success_string


def test_divide_properly():
    assert util.divide(4, 2) == 2


def test_divide_improperly():
    with pytest.raises(AssertionError):
        util.divide(4, 5)


def test_experimental_cls_init():
    with patch.object(config, 'ENABLE_EXPERIMENTAL', True):
        # Check that initialization works
        a = A()
        assert a.__class__.__qualname__ == "A"
        assert a.some_method() == success_string
        assert a.is_experimental is True


def test_experimental_cls_static():
    with patch.object(config, 'ENABLE_EXPERIMENTAL', True):
        # Check that static methods work
        assert A.__class__.__qualname__ == "A"
        assert A.some_static_method() == success_string
        assert A.is_experimental is True


def test_experimental_cls_exception_init():
    with patch.object(config, 'ENABLE_EXPERIMENTAL', False), pytest.raises(
        util.ExperimentalNotEnabledError
    ):
        a = A()
        assert a.some_method() == success_string
        assert a.is_experimental is False


def test_experimental_cls_exception_static():
    with patch.object(config, 'ENABLE_EXPERIMENTAL', False), pytest.raises(
        util.ExperimentalNotEnabledError
    ):
        assert A.some_static_method() == success_string

    assert A.is_experimental is False


def test_global_memory_buffer():
    Utils.initialize_model_parallel(1, 1)
    global_memory_buffer = util.GlobalMemoryBuffer()
    obtained_tensor = global_memory_buffer.get_tensor((3, 2), torch.float32, "test_tensor")
    expected_tensor = torch.empty((3, 2), dtype=torch.float32, device=get_current_device())
    assert obtained_tensor.shape == expected_tensor.shape
    Utils.destroy_model_parallel()


def test_make_viewless_tensor():
    Utils.initialize_model_parallel(1, 1)
    inp = torch.rand((3, 4))
    assert torch.equal(inp, util.make_viewless_tensor(inp, True, True))
    assert torch.equal(inp, util.make_viewless_tensor(inp, True, False))
    Utils.destroy_model_parallel()


def test_safely_set_viewless_tensor_data():
    Utils.initialize_model_parallel(1, 1)
    tensor = torch.zeros((3, 4))
    new_data_tensor = torch.tensor(np.random.rand(3, 4))
    util.safely_set_viewless_tensor_data(tensor, new_data_tensor)
    assert torch.equal(tensor, new_data_tensor)
    Utils.destroy_model_parallel()


def test_assert_viewless_tensor():
    Utils.initialize_model_parallel(1, 1)
    tensor = torch.rand((3, 4))
    assert torch.equal(util.assert_viewless_tensor(tensor), tensor)
    input_tensor_list = [tensor, tensor, tensor]
    output_tensor_list = util.assert_viewless_tensor(input_tensor_list)
    for inp, out in zip(input_tensor_list, output_tensor_list):
        assert torch.equal(inp, out)
    Utils.destroy_model_parallel()


@pytest.mark.skipif(not util.HAVE_NVTX, reason="NVTX module not available")
@pytest.mark.parametrize(
    "msg,suffix",
    [(None, None), ("test_message", None), (None, "test_suffix"), ("test_message", "test_suffix")],
)
def test_nvtx_range(msg, suffix):
    # Track function execution
    execution_tracker = {'ranges': False}

    def _call_nvtx_range():
        util.nvtx_range_push(msg, suffix)
        execution_tracker['ranges'] = True
        util.nvtx_range_pop(msg, suffix)

    # Test with NVTX disabled
    with patch.dict('os.environ', {'MEGATRON_NVTX_ENABLED': '0'}):
        _call_nvtx_range()
        assert execution_tracker['ranges']

    # Reset tracker
    execution_tracker['ranges'] = False

    # Test with NVTX enabled
    with patch.dict('os.environ', {'MEGATRON_NVTX_ENABLED': '1'}):
        _call_nvtx_range()
        assert execution_tracker['ranges']

@pytest.mark.skipif(not util.HAVE_NVTX, reason="NVTX module not available")
def test_nvtx_decorator():
    # Track function execution
    execution_tracker = {'decorated': False, 'decorated_with_message': False}

    # Create decorated functions
    @util.nvtx_decorator()
    def nvtx_decorated_function():
        execution_tracker['decorated'] = True

    @util.nvtx_decorator(message="test_nvtx_decorator", color="red")
    def nvtx_decorated_function_with_message():
        execution_tracker['decorated_with_message'] = True

    # Test with NVTX disabled
    with patch.dict('os.environ', {'MEGATRON_NVTX_ENABLED': '0'}):
        nvtx_decorated_function()
        nvtx_decorated_function_with_message()
        assert all(execution_tracker.values())

    # Reset tracker
    execution_tracker = {'decorated': False, 'decorated_with_message': False}

    # Test with NVTX enabled
    with patch.dict('os.environ', {'MEGATRON_NVTX_ENABLED': '1'}):
        nvtx_decorated_function()
        nvtx_decorated_function_with_message()
        assert all(execution_tracker.values())


@pytest.mark.flaky_in_dev
def test_check_param_hashes_across_dp_replicas():
    world = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))

    # Setup.
    Utils.initialize_model_parallel()
    model = torch.nn.Linear(100, 100, bias=False, device=get_current_device())

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
    Utils.destroy_model_parallel()


@pytest.mark.flaky_in_dev
def test_cross_check_param_hashes_across_dp_replicas():
    rank = int(os.getenv('RANK', '0'))

    # Setup.
    Utils.initialize_model_parallel()
    model = torch.nn.Linear(100, 100, bias=False, device=get_current_device())

    # First check case where all replicas agree.
    model.weight.data.fill_(1.0)
    assert util.check_param_hashes_across_dp_replicas([model], True)

    # Now check case where replica 0 disagrees with all other replicas.
    if rank == 0:
        model.weight.data.fill_(0.0)
    assert not util.check_param_hashes_across_dp_replicas([model], True)

    # Teardown.
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("use_distributed_optimizer", [False, True])
@pytest.mark.flaky_in_dev
def test_param_norm(use_distributed_optimizer: bool):

    if xm and use_distributed_optimizer:
        return

    # Setup: distributed, model, mock_args.
    Utils.initialize_model_parallel()
    model = torch.nn.Linear(100, 100, bias=False, dtype=torch.bfloat16, device=get_current_device())
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
    Utils.destroy_model_parallel()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
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
        mat1 = torch.randn(M, K, device=get_current_device())
        mat2 = torch.randn(K, N, device=get_current_device())
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
        mat1 = torch.randn(N, P, device=get_current_device())
        mat2 = torch.randn(P, M, device=get_current_device())
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
    Utils.initialize_model_parallel()

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
    Utils.destroy_model_parallel()
    
