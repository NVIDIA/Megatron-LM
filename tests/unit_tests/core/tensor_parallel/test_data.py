import torch

from megatron.core.tensor_parallel.data import broadcast_data
from tests.unit_tests.test_utilities import Utils


def test_broadcast_data():
    Utils.initialize_model_parallel(2, 4)
    input_data = {
        0: torch.ones((8, 8)).cuda() * 0.0,
        1: torch.ones((8, 8)).cuda() * 1.0,
        2: torch.ones((8, 8)).cuda() * 2.0,
        3: torch.ones((8, 8)).cuda() * 3.0,
        4: torch.ones((8, 8)).cuda() * 4.0,
        5: torch.ones((8, 8)).cuda() * 5.0,
        6: torch.ones((8, 8)).cuda() * 6.0,
        7: torch.ones((8, 8)).cuda() * 7.0,
    }
    dtype = torch.float32
    actual_output = broadcast_data([0, 1], input_data, dtype)
    assert torch.equal(actual_output[0], input_data[0])
    assert torch.equal(actual_output[1], input_data[1])
    Utils.destroy_model_parallel()
