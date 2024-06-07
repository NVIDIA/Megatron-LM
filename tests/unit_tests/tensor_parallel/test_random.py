from megatron.core.tensor_parallel.random import CudaRNGStatesTracker
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed,get_cuda_rng_tracker
from megatron.core.tensor_parallel.random import checkpoint
from tests.unit_tests.test_utilities import Utils
import pytest
import torch

def test_cuda_rng_states_tracker():
    rng_tracker = CudaRNGStatesTracker()
    rng_tracker.set_states({"state1":1234})
    assert(rng_tracker.get_states()["state1"] == 1234)
    rng_tracker.reset()
    assert(rng_tracker.get_states() == {})
    seed = 1111
    rng_tracker.add("state2",seed)
    with pytest.raises(Exception):
        assert(rng_tracker.add("state3",seed))
    with pytest.raises(Exception):
        assert(rng_tracker.add("state2",111))
    assert(rng_tracker.get_states()['state2'] is not None)
    with pytest.raises(Exception):
        assert()
    
    rng_tracker.fork("state2")
    torch.cuda.manual_seed(seed)
    rng_state = torch.cuda.get_rng_state()
    assert torch.equal(rng_tracker.get_states()['state2'], rng_state)

def test_model_parallel_cuda_manual_seed():
    Utils.initialize_model_parallel(4,2)
    model_parallel_cuda_manual_seed(0)
    rng_tracker = get_cuda_rng_tracker()
    assert(rng_tracker.get_states()['model-parallel-rng'] is not None)
    Utils.destroy_model_parallel()

def test_checkpoint():
    def test_forward(*input):
        return input[0]+input[1]
    assert(torch.equal(torch.ones(16)*3,checkpoint(test_forward, None, torch.ones(16), torch.ones(16)*2)))
    Utils.initialize_model_parallel()
    input1 = torch.ones((4,4))
    checkpoint(test_forward, True, input1, torch.ones((4,4))*2)
    assert(torch.equal(torch.ones(input1.numel()).cuda(), input1))
    Utils.destroy_model_parallel()
