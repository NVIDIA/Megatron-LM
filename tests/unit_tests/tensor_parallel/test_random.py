# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest import mock

import pytest
import torch
from contextlib import contextmanager


from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import (
    CheckpointWithoutOutput,
    CudaRNGStatesTracker,
    checkpoint,
    convert_cuda_rng_state,
    ensure_context_parallel_rng_tracker_states,
    ensure_model_and_context_parallel_rng_tracker_states,
    get_context_parallel_rng_tracker_name,
    get_cuda_rng_tracker,
    get_model_and_context_parallel_rng_tracker_name,
    get_model_parallel_rng_tracker_name,
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


@mock.patch("megatron.core.tensor_parallel.random.get_context_parallel_rank", return_value=1)
@mock.patch("megatron.core.tensor_parallel.random.get_context_parallel_world_size", return_value=2)
@mock.patch("megatron.core.tensor_parallel.random.get_tensor_and_context_parallel_rank", return_value=3)
def test_cp_only_and_tpxcp_tracker_seed_semantics(
    mock_tpcp_rank, mock_cp_world_size, mock_cp_rank
):
    Utils.initialize_model_parallel(4, 1)
    model_parallel_cuda_manual_seed(123, force_reset_rng=True)
    tracker_states = get_cuda_rng_tracker().get_states()
    assert get_context_parallel_rng_tracker_name() in tracker_states
    assert get_model_and_context_parallel_rng_tracker_name() in tracker_states

    expected_context_seed = 123 + 2718 + 2048 + 1
    expected_tpcp_seed = 123 + 2718 + 4 + 3

    expected_context_tracker = CudaRNGStatesTracker()
    expected_context_tracker.add(get_context_parallel_rng_tracker_name(), expected_context_seed)
    expected_tpcp_tracker = CudaRNGStatesTracker()
    expected_tpcp_tracker.add(get_model_and_context_parallel_rng_tracker_name(), expected_tpcp_seed)

    assert torch.equal(
        tracker_states[get_context_parallel_rng_tracker_name()],
        expected_context_tracker.get_states()[get_context_parallel_rng_tracker_name()],
    )
    assert torch.equal(
        tracker_states[get_model_and_context_parallel_rng_tracker_name()],
        expected_tpcp_tracker.get_states()[get_model_and_context_parallel_rng_tracker_name()],
    )
    Utils.destroy_model_parallel()


def test_ensure_context_parallel_rng_tracker_states_uses_data_parallel_fallback():
    source_tracker = CudaRNGStatesTracker()
    source_tracker.add("data-parallel-rng", 456)
    states = ensure_context_parallel_rng_tracker_states(source_tracker.get_states())
    assert get_context_parallel_rng_tracker_name() in states
    assert torch.equal(states[get_context_parallel_rng_tracker_name()], states["data-parallel-rng"])


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


@contextmanager
def _rng_context_from_config(sequence_parallel, context_parallel_size):
    if sequence_parallel:
        with get_cuda_rng_tracker().fork(get_model_and_context_parallel_rng_tracker_name()):
            yield
    elif context_parallel_size > 1:
        with get_cuda_rng_tracker().fork(get_context_parallel_rng_tracker_name()):
            yield
    else:
        yield


@pytest.mark.skipif(Utils.world_size < 4, reason="requires at least 4 ranks for tp=2 and cp=2")
def test_checkpoint_recompute_preserves_context_parallel_rng_behavior():
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, context_parallel_size=2)
    try:
        def dropout_forward(input_, sequence_parallel=False):
            with _rng_context_from_config(sequence_parallel, context_parallel_size=2):
                return torch.nn.functional.dropout(input_, p=0.5, training=True)

        input_plain = torch.full((8, 4), 4.0, device="cuda", requires_grad=True)
        input_ckpt = input_plain.detach().clone().requires_grad_(True)

        model_parallel_cuda_manual_seed(999, force_reset_rng=True)
        plain_output = dropout_forward(input_plain)

        model_parallel_cuda_manual_seed(999, force_reset_rng=True)
        ckpt_output = checkpoint(lambda x: dropout_forward(x), False, input_ckpt)

        torch.testing.assert_close(plain_output, ckpt_output)

        grad = torch.ones_like(plain_output)
        plain_output.backward(grad)
        ckpt_output.backward(grad)
        torch.testing.assert_close(input_plain.grad, input_ckpt.grad)

        dropout_mask = plain_output.eq(0).cpu()
        payload = {
            "cp_rank": parallel_state.get_context_parallel_rank(),
            "mask": dropout_mask,
        }
        gathered = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered, payload)

        masks_by_cp_rank = {}
        for item in gathered:
            masks_by_cp_rank.setdefault(item["cp_rank"], []).append(item["mask"])

        for masks in masks_by_cp_rank.values():
            reference = masks[0]
            for mask in masks[1:]:
                assert torch.equal(reference, mask)

        cp_ranks = sorted(masks_by_cp_rank)
        assert len(cp_ranks) == 2
        assert not torch.equal(
            masks_by_cp_rank[cp_ranks[0]][0],
            masks_by_cp_rank[cp_ranks[1]][0],
        )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(Utils.world_size < 4, reason="requires at least 4 ranks for tp=2 and cp=2")
def test_checkpoint_recompute_preserves_model_and_context_parallel_rng_behavior():
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, context_parallel_size=2)
    try:
        def dropout_forward(input_):
            with _rng_context_from_config(sequence_parallel=True, context_parallel_size=2):
                return torch.nn.functional.dropout(input_, p=0.5, training=True)

        input_plain = torch.full((8, 4), 4.0, device="cuda", requires_grad=True)
        input_ckpt = input_plain.detach().clone().requires_grad_(True)

        model_parallel_cuda_manual_seed(1001, force_reset_rng=True)
        plain_output = dropout_forward(input_plain)

        model_parallel_cuda_manual_seed(1001, force_reset_rng=True)
        ckpt_output = checkpoint(lambda x: dropout_forward(x), False, input_ckpt)

        torch.testing.assert_close(plain_output, ckpt_output)

        grad = torch.ones_like(plain_output)
        plain_output.backward(grad)
        ckpt_output.backward(grad)
        torch.testing.assert_close(input_plain.grad, input_ckpt.grad)

        dropout_mask = plain_output.eq(0).cpu()
        payload = {
            "cp_rank": parallel_state.get_context_parallel_rank(),
            "tp_rank": parallel_state.get_tensor_model_parallel_rank(),
            "mask": dropout_mask,
        }
        gathered = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered, payload)

        masks_by_rank = {(item["cp_rank"], item["tp_rank"]): item["mask"] for item in gathered}
        assert len(masks_by_rank) == 4

        for cp_rank in range(2):
            assert not torch.equal(masks_by_rank[(cp_rank, 0)], masks_by_rank[(cp_rank, 1)])

        for tp_rank in range(2):
            assert not torch.equal(masks_by_rank[(0, tp_rank)], masks_by_rank[(1, tp_rank)])
    finally:
        Utils.destroy_model_parallel()


def test_ensure_model_and_context_parallel_rng_tracker_states_clones_model_parallel_state():
    source_tracker = CudaRNGStatesTracker()
    source_tracker.add(get_model_parallel_rng_tracker_name(), 789)
    source_state = source_tracker.get_states()[get_model_parallel_rng_tracker_name()]

    states = ensure_model_and_context_parallel_rng_tracker_states(source_tracker.get_states())
    assert get_model_and_context_parallel_rng_tracker_name() in states
    assert torch.equal(states[get_model_and_context_parallel_rng_tracker_name()], source_state)
    assert states[get_model_and_context_parallel_rng_tracker_name()] is not source_state

    existing_target = CudaRNGStatesTracker()
    existing_target.add(get_model_and_context_parallel_rng_tracker_name(), 321)
    expected_target_state = existing_target.get_states()[get_model_and_context_parallel_rng_tracker_name()]
    states_with_target = {
        get_model_parallel_rng_tracker_name(): source_state,
        get_model_and_context_parallel_rng_tracker_name(): expected_target_state,
    }
    preserved_states = ensure_model_and_context_parallel_rng_tracker_states(states_with_target)
    assert (
        preserved_states[get_model_and_context_parallel_rng_tracker_name()] is expected_target_state
    )
