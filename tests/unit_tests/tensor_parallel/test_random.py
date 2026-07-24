# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.tensor_parallel.random import (
    CheckpointWithoutOutput,
    CudaRNGStatesTracker,
    checkpoint,
    convert_cuda_rng_state,
    cuda_rng_namespace,
    fork_process_rng_state,
    get_cuda_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from tests.unit_tests.test_utilities import Utils


def test_cuda_rng_states_tracker():
    rng_tracker = CudaRNGStatesTracker()
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
    torch.cuda.manual_seed(seed)
    rng_state = torch.cuda.get_rng_state()
    assert torch.equal(rng_tracker.get_states()['state2'], rng_state)


@pytest.mark.parametrize("use_cudagraphable_rng", [True, False])
def test_double_fork_cuda_rng_states_tracker(use_cudagraphable_rng):
    rng_tracker = CudaRNGStatesTracker(use_cudagraphable_rng=use_cudagraphable_rng)
    rng_tracker.add("state1", 1234)
    rng_tracker.add("state2", 5678)
    randn_double_fork_1 = []
    randn_double_fork_2 = []
    with rng_tracker.fork("state1"):
        randn_double_fork_1.append(torch.randn(10, device="cuda"))
        with rng_tracker.fork("state2"):
            randn_double_fork_2.append(torch.randn(10, device="cuda"))
            with rng_tracker.fork("state1"):
                randn_double_fork_1.append(torch.randn(10, device="cuda"))
            randn_double_fork_2.append(torch.randn(10, device="cuda"))
        randn_double_fork_1.append(torch.randn(10, device="cuda"))
    if use_cudagraphable_rng:
        double_fork_state1 = rng_tracker.get_states()["state1"].get_state()
        double_fork_state2 = rng_tracker.get_states()["state2"].get_state()
    else:
        double_fork_state1 = rng_tracker.get_states()["state1"]
        double_fork_state2 = rng_tracker.get_states()["state2"]

    rng_tracker.reset()
    rng_tracker.add("state1", 1234)
    rng_tracker.add("state2", 5678)
    randn_single_fork_1 = []
    randn_single_fork_2 = []
    with rng_tracker.fork("state1"):
        randn_single_fork_1.append(torch.randn(10, device="cuda"))
        randn_single_fork_1.append(torch.randn(10, device="cuda"))
        randn_single_fork_1.append(torch.randn(10, device="cuda"))
    with rng_tracker.fork("state2"):
        randn_single_fork_2.append(torch.randn(10, device="cuda"))
        randn_single_fork_2.append(torch.randn(10, device="cuda"))
    if use_cudagraphable_rng:
        single_fork_state1 = rng_tracker.get_states()["state1"].get_state()
        single_fork_state2 = rng_tracker.get_states()["state2"].get_state()
    else:
        single_fork_state1 = rng_tracker.get_states()["state1"]
        single_fork_state2 = rng_tracker.get_states()["state2"]

    assert torch.equal(randn_double_fork_1[0], randn_single_fork_1[0])
    assert torch.equal(randn_double_fork_1[1], randn_single_fork_1[1])
    assert torch.equal(randn_double_fork_1[2], randn_single_fork_1[2])
    assert torch.equal(randn_double_fork_2[0], randn_single_fork_2[0])
    assert torch.equal(randn_double_fork_2[1], randn_single_fork_2[1])
    assert torch.equal(double_fork_state1, single_fork_state1)
    assert torch.equal(double_fork_state2, single_fork_state2)


def test_convert_cuda_rng_state():
    ## Get the default rng state
    torch.cuda.manual_seed(999)
    randn = torch.randn(10, device="cuda")
    rng_state = torch.cuda.get_rng_state()

    try:
        from megatron.core.extensions.transformer_engine import TECudaRNGStatesTracker
    except ImportError:
        TECudaRNGStatesTracker = None

    ## from non-graphable RNG to graphable RNG
    # get state from non-graphable RNG
    tracker = CudaRNGStatesTracker(use_cudagraphable_rng=False)
    tracker.add("state1", 123)
    for i in range(3):
        with tracker.fork("state1"):
            randn = torch.randn(10, device="cuda")
    state = convert_cuda_rng_state(tracker.states_["state1"], to_graphable=True)
    rand_tensors = []
    for i in range(3):
        with tracker.fork("state1"):
            randn = torch.randn(10, device="cuda")
            rand_tensors.append(randn)

    # set state to local graph RNG
    cudagraphable_tracker = CudaRNGStatesTracker(use_cudagraphable_rng=True)
    cudagraphable_tracker.set_states({"state1": state.clone_state()})
    for i in range(3):
        with cudagraphable_tracker.fork("state1"):
            randn = torch.randn(10, device="cuda")
            assert torch.equal(randn, rand_tensors[i])

    # set state to TE RNG
    if TECudaRNGStatesTracker is not None:
        te_tracker = TECudaRNGStatesTracker()
        te_tracker.set_states({"state1": state})
        for i in range(3):
            with te_tracker.fork("state1"):
                randn = torch.randn(10, device="cuda")
                assert torch.equal(randn, rand_tensors[i])

    ## from graphable RNG to non-graphable RNG
    # get state from graphable RNG
    cudagraphable_tracker = CudaRNGStatesTracker(use_cudagraphable_rng=True)
    cudagraphable_tracker.add("state2", 123)
    for i in range(3):
        with cudagraphable_tracker.fork("state2"):
            randn = torch.randn(10, device="cuda")
    state = convert_cuda_rng_state(cudagraphable_tracker.states_["state2"], to_graphable=False)
    rand_tensors = []
    for i in range(3):
        with cudagraphable_tracker.fork("state2"):
            randn = torch.randn(10, device="cuda")
            rand_tensors.append(randn)

    # set state to non-graphable RNG
    tracker = CudaRNGStatesTracker(use_cudagraphable_rng=False)
    tracker.set_states({"state2": state})
    for i in range(3):
        with tracker.fork("state2"):
            randn = torch.randn(10, device="cuda")
            assert torch.equal(randn, rand_tensors[i])

    ## from TE RNG to non-graphable RNG
    if TECudaRNGStatesTracker is not None:
        # get state from TE RNG
        cudagraphable_tracker = TECudaRNGStatesTracker()
        cudagraphable_tracker.add("state3", 123)
        for i in range(3):
            with cudagraphable_tracker.fork("state3"):
                randn = torch.randn(10, device="cuda")
        state = convert_cuda_rng_state(cudagraphable_tracker.states_["state3"], to_graphable=False)
        rand_tensors = []
        for i in range(3):
            with cudagraphable_tracker.fork("state3"):
                randn = torch.randn(10, device="cuda")
                rand_tensors.append(randn)

        # set state to non-graphable RNG
        tracker = CudaRNGStatesTracker(use_cudagraphable_rng=False)
        tracker.set_states({"state3": state})
        for i in range(3):
            with tracker.fork("state3"):
                randn = torch.randn(10, device="cuda")
                assert torch.equal(randn, rand_tensors[i])

    ## After all tests, check if the default rng state is still the same.
    rng_state_final = torch.cuda.get_rng_state()
    assert torch.equal(rng_state, rng_state_final)


def test_model_parallel_cuda_manual_seed():
    Utils.initialize_model_parallel(4, 2)
    model_parallel_cuda_manual_seed(0, force_reset_rng=True)
    rng_tracker = get_cuda_rng_tracker()
    assert rng_tracker.get_states()['model-parallel-rng'] is not None
    Utils.destroy_model_parallel()


def test_namespaced_seed_extends_singleton_without_changing_ambient_cuda():
    model_parallel_cuda_manual_seed(11, force_reset_rng=True, tp_rank=0, ep_rank=0, etp_rank=0)
    tracker = get_cuda_rng_tracker()
    base_states = {name: state.clone() for name, state in tracker.get_states().items()}
    ambient = torch.cuda.get_rng_state().clone()

    model_parallel_cuda_manual_seed(101, tp_rank=1, ep_rank=2, etp_rank=0, rng_namespace="vision")

    assert set(tracker.get_states()) == set(base_states) | {
        "vision::data-parallel-rng",
        "vision::model-parallel-rng",
        "vision::expert-parallel-rng",
    }
    assert all(
        torch.equal(tracker.get_states()[name], state) for name, state in base_states.items()
    )
    assert torch.equal(torch.cuda.get_rng_state(), ambient)


def test_duplicate_seed_is_scoped_to_rng_namespace():
    tracker = CudaRNGStatesTracker()
    tracker.add("encoder::state", 123)
    tracker.add("language::state", 123)
    with pytest.raises(Exception, match="seed 123 already exists"):
        tracker.add("encoder::other", 123)


@pytest.mark.parametrize(
    "tracker",
    [
        object(),
        CudaRNGStatesTracker(is_inference_rng_tracker=True),
        CudaRNGStatesTracker(use_cudagraphable_rng=True),
    ],
)
def test_cuda_rng_namespace_rejects_unsupported_tracker(monkeypatch, tracker):
    from megatron.core.tensor_parallel import random

    monkeypatch.setattr(random, "_CUDA_RNG_STATE_TRACKER", tracker)
    monkeypatch.setattr(random, "_CUDA_RNG_STATE_TRACKER_INITIALIZED", True)
    with pytest.raises(RuntimeError, match="require Megatron CudaRNGStatesTracker"):
        with cuda_rng_namespace("encoder"):
            pass


def test_namespaced_default_and_data_parallel_streams_are_independent():
    model_parallel_cuda_manual_seed(13, force_reset_rng=True, tp_rank=0, ep_rank=0, etp_rank=0)
    model_parallel_cuda_manual_seed(113, tp_rank=0, ep_rank=0, etp_rank=0, rng_namespace="encoder")
    model_parallel_cuda_manual_seed(213, tp_rank=0, ep_rank=0, etp_rank=0, rng_namespace="language")
    tracker = get_cuda_rng_tracker()

    with cuda_rng_namespace("encoder"):
        with tracker.fork():
            encoder_model = torch.rand(4, device="cuda")
    with cuda_rng_namespace("language"):
        with tracker.fork():
            language_model = torch.rand(4, device="cuda")
    with cuda_rng_namespace("encoder", fork_data_parallel_rng=True):
        encoder_data = torch.rand(4, device="cuda")

    assert not torch.equal(encoder_model, language_model)
    assert not torch.equal(encoder_model, encoder_data)


def test_fork_process_rng_state_is_repeatable_and_preserves_cuda():
    cpu_state = torch.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state().clone()
    with fork_process_rng_state(123):
        first = torch.rand(4)
    with fork_process_rng_state(123):
        second = torch.rand(4)
    assert torch.equal(first, second)
    assert torch.equal(torch.get_rng_state(), cpu_state)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_state)


def test_checkpoint():
    def test_forward(*input):
        return input[0] + input[1]

    assert torch.equal(
        torch.ones(16) * 3, checkpoint(test_forward, None, torch.ones(16), torch.ones(16) * 2)
    )

    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    input1 = torch.ones((4, 4)).cuda()
    input1.requires_grad_(True)
    input2 = torch.ones((4, 4)).cuda() * 2
    output = checkpoint(test_forward, True, input1, input2)

    assert torch.equal(output, torch.ones((4, 4)).cuda() * 3)
    assert input1.data.shape == (8,)

    output.sum().backward()
    assert input1.grad is not None
    assert torch.equal(input1.grad, torch.ones((4, 4)).cuda())

    Utils.destroy_model_parallel()


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


class _ViewSavingLinear(torch.autograd.Function):
    """Saves view tensors in forward to mimic TE GroupedLinear-style backward inputs."""

    @staticmethod
    def forward(ctx, inp, weight):
        inp_2d = inp.reshape(-1, inp.shape[-1])
        inputmats = torch.tensor_split(inp_2d, 2, dim=0)
        ctx.save_for_backward(*inputmats, weight)
        ctx.input_shape = inp.shape
        out_2d = inp_2d.matmul(weight.t())
        return out_2d.reshape(*inp.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        *inputmats, weight = ctx.saved_tensors
        for inputmat in inputmats:
            if inputmat.numel() > 0 and inputmat.untyped_storage().size() == 0:
                raise RuntimeError("Saved view tensor points to an empty storage.")

        inp_2d = torch.cat(inputmats, dim=0)
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
        grad_input_2d = grad_output_2d.matmul(weight)
        grad_weight = grad_output_2d.t().matmul(inp_2d)
        grad_input = grad_input_2d.reshape(ctx.input_shape)
        return grad_input, grad_weight


def test_checkpoint_without_output_view_sharing_regression():
    def normal_forward(input_, weight):
        x = torch.nn.functional.gelu(input_)
        return _ViewSavingLinear.apply(x, weight)

    def checkpoint_forward(input_, weight):
        checkpoint = CheckpointWithoutOutput()
        x = checkpoint.checkpoint(torch.nn.functional.gelu, input_)
        y = _ViewSavingLinear.apply(x, weight)
        checkpoint.discard_output_and_register_recompute(y)
        return y

    Utils.initialize_model_parallel()
    try:
        input1 = torch.randn((3, 2, 8), requires_grad=True)
        weight1 = torch.randn((6, 8), requires_grad=True)

        input2 = input1.detach().clone().requires_grad_(True)
        weight2 = weight1.detach().clone().requires_grad_(True)

        output1 = normal_forward(input1, weight1)
        output2 = checkpoint_forward(input2, weight2)
        assert torch.allclose(output1, output2)

        grad = torch.randn_like(output1)
        output1.backward(grad, retain_graph=True)
        output2.backward(grad, retain_graph=True)
        assert torch.allclose(input1.grad, input2.grad)
        assert torch.allclose(weight1.grad, weight2.grad)
    finally:
        Utils.destroy_model_parallel()
