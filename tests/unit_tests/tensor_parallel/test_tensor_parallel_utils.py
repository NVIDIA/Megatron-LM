import torch

import megatron.core.parallel_state as ps
import megatron.core.tensor_parallel.utils as util
from tests.unit_tests.test_utilities import Utils

rank = Utils.rank


def test_split_tensor_along_last_dim():
    input_tensor = torch.rand((3, 4))
    torch.equal(input_tensor[0:2, 0:2], util.split_tensor_along_last_dim(input_tensor, 2)[0])
    torch.equal(input_tensor[2:, 2:], util.split_tensor_along_last_dim(input_tensor, 2)[1])


def test_split_tensor_into_1d_equal_chunks():
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=4)
    input_tensor = torch.rand((3, 4))
    output_tensor = util.split_tensor_into_1d_equal_chunks(input_tensor)
    if rank % 2 == 0:
        start = 0
        end = int(input_tensor.numel() / 2)
    else:
        start = int(input_tensor.numel() / 2)
        end = input_tensor.numel()

    assert torch.equal(output_tensor, input_tensor.flatten()[start:end])
    Utils.destroy_model_parallel()


def test_gather_split_1d_tensor():
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=4)
    input_tensor = torch.ones((2, 4)).cuda() * rank
    actual_output_tensor = util.gather_split_1d_tensor(input_tensor)
    if rank % 2 == 0:
        expected_output_tensor = torch.concat((input_tensor.flatten(), input_tensor.flatten() + 1))
    else:
        expected_output_tensor = torch.concat((input_tensor.flatten() - 1, input_tensor.flatten()))
    assert torch.equal(actual_output_tensor, expected_output_tensor)
    Utils.destroy_model_parallel()


def test_vocab():
    global_vocab_size = 1600
    per_partition_vocab_size = 1600 / Utils.world_size
    assert (rank * per_partition_vocab_size, (rank + 1) * per_partition_vocab_size) == (
        util.VocabUtility.vocab_range_from_per_partition_vocab_size(
            global_vocab_size // Utils.world_size, rank, Utils.world_size
        )
    )
    assert (rank * per_partition_vocab_size, (rank + 1) * per_partition_vocab_size) == (
        util.VocabUtility.vocab_range_from_global_vocab_size(
            global_vocab_size, rank, Utils.world_size
        )
    )
