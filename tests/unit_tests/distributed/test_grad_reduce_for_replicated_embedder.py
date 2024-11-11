import pytest
import torch

from megatron.core import ModelParallelConfig, parallel_state
from megatron.core.distributed.finalize_model_grads import _allreduce_conditional_embedding_grads
from tests.unit_tests.test_utilities import Utils

rank = Utils.rank


def test_allreduce_conditional_embedding_grads():

    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=4)

    # For virtual pipeline parallelism.
    model = [torch.nn.Linear(10, 10, bias=True).cuda() for _ in range(2)]
    # Here we only reduce weights, not bias to compare the results.
    for chunk in model:
        setattr(chunk.weight, "pipeline_parallel", True)

    config = ModelParallelConfig(
        pipeline_model_parallel_size=4, sequence_parallel=False, pipeline_dtype=torch.float
    )
    config.has_cond_embedder = True

    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_world_size = parallel_state.get_pipeline_model_parallel_world_size()

    # Init different grads for each model chunk and rank.
    for i, chunk in enumerate(model):
        for param in chunk.parameters():
            param.main_grad = torch.ones_like(param) * (pp_rank * 10.0 + i)

    _allreduce_conditional_embedding_grads(model, config)

    expect_value = 0
    for i in range(len(model)):
        for j in range(pp_world_size):
            expect_value += j * 10.0 + i
    expect_weight_grad = torch.ones([10, 10]).cuda() * expect_value

    for i, chunk in enumerate(model):
        expect_bias_grad = torch.ones([10]).cuda() * (pp_rank * 10.0 + i)
        assert torch.equal(chunk.weight.main_grad, expect_weight_grad)
        assert torch.equal(chunk.bias.main_grad, expect_bias_grad)

    Utils.destroy_model_parallel()
