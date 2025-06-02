from megatron.core.device_utils import get_current_device, get_current_rng_state, get_xla_model
from megatron.core.tensor_parallel.random import DeviceRNGStatesTracker
from megatron.core.tensor_parallel.random import model_parallel_device_manual_seed,get_device_rng_tracker
from megatron.core.tensor_parallel.random import checkpoint
from megatron.core.device_utils import set_manual_seed
from tests.unit_tests.test_utilities import Utils
import pytest
import torch

from megatron.core.tensor_parallel.random import (
    CheckpointWithoutOutput,
    DeviceRNGStatesTracker,
    checkpoint,
    model_parallel_device_manual_seed,
)
from tests.unit_tests.test_utilities import Utils

try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

def test_device_rng_states_tracker():
    rng_tracker = DeviceRNGStatesTracker()
    rng_tracker.set_states({"state1": 1234})
    assert rng_tracker.get_states()["state1"] == 1234
    rng_tracker.reset()
    assert rng_tracker.get_states() == {}
    seed = 1111
    rng_tracker.add("state2", seed)
    with pytest.raises(Exception):
        assert rng_tracker.add("state3", seed)
    with pytest.raises(Exception):
        assert rng_tracker.add("state2", 111)
    assert rng_tracker.get_states()['state2'] is not None
    with pytest.raises(Exception):
        assert ()

    rng_tracker.fork("state2")
    set_manual_seed(seed)
    rng_state = get_current_rng_state()
    xm = get_xla_model()
    if xm is None:
        assert torch.equal(rng_tracker.get_states()['state2'], rng_state)
    else:
        assert int(rng_tracker.get_states()['state2']) == rng_state

def test_model_parallel_device_manual_seed():
    Utils.initialize_model_parallel(4,2)
    model_parallel_device_manual_seed(0)
    rng_tracker = get_device_rng_tracker()
    assert(rng_tracker.get_states()['model-parallel-rng'] is not None)
    Utils.destroy_model_parallel()


def test_checkpoint():
    def test_forward(*input):
        return input[0] + input[1]

    assert torch.equal(
        torch.ones(16) * 3, checkpoint(test_forward, None, torch.ones(16), torch.ones(16) * 2)
    )
    Utils.initialize_model_parallel()
    input1 = torch.ones((4,4))
    checkpoint(test_forward, True, input1, torch.ones((4,4))*2)
    assert(torch.equal(torch.ones(input1.numel()).to(device=get_current_device()), input1))
    Utils.destroy_model_parallel()


@pytest.mark.skipif(not HAVE_TE, reason="Transformer engine required" )
def test_checkpoint_without_output():
    def normal_forward(input):
        x = torch.nn.functional.gelu(input)
        y = x * input
        return y

    def checkpoint_forward(input):
        checkpoint = CheckpointWithoutOutput()
        x = checkpoint.checkpoint(torch.nn.functional.gelu, input)
        y = x * input
        checkpoint.discard_output_and_register_recompute(y)
        return y

    Utils.initialize_model_parallel()

    input1 = torch.ones((4, 4))
    input1.requires_grad_(True)
    output1 = normal_forward(input1)
    input2 = torch.ones((4, 4))
    input2.requires_grad_(True)
    output2 = checkpoint_forward(input2)
    assert torch.equal(output1, output2)

    output1.backward(torch.ones((4, 4)), retain_graph=True)
    output2.backward(torch.ones((4, 4)), retain_graph=True)
    assert torch.equal(input1.grad, input2.grad)

    Utils.destroy_model_parallel()
