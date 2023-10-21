import pytest
import torch
import megatron.core.utils as util
import numpy as np

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
