# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.tensor_parallel.random import (
    HAVE_TE,
    CheckpointFunction,
    CheckpointWithoutOutput,
    CudaRNGStatesTracker,
    checkpoint,
    convert_cuda_rng_state,
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


def _fp8_recipe_or_skip(name=None):
    """An FP8 recipe the current GPU can run; skips when it is unavailable.

    ``name`` selects ``"mxfp8"`` or ``"delayed"`` scaling; ``None`` prefers MXFP8 and falls back
    to delayed scaling.
    """
    from transformer_engine.common import recipe as te_recipe
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    if not FP8GlobalStateManager.is_fp8_available()[0]:
        pytest.skip("FP8 is not available on this GPU")
    mxfp8_available = FP8GlobalStateManager.is_mxfp8_available()[0]
    if name == "mxfp8" and not mxfp8_available:
        pytest.skip("MXFP8 is not available on this GPU")
    if name == "mxfp8" or (name is None and mxfp8_available):
        return te_recipe.MXFP8BlockScaling()
    return te_recipe.DelayedScaling()


@pytest.mark.skipif(not (HAVE_TE and torch.cuda.is_available()), reason="TE and CUDA required")
def test_checkpoint_recompute_reenters_forward_fp8_autocast():
    """backward() recomputes outside the caller's fp8_autocast; the recompute must see the FP8
    state the forward ran under (and none when the forward had none)."""
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    recipe = _fp8_recipe_or_skip()
    Utils.initialize_model_parallel()
    model_parallel_cuda_manual_seed(123)
    try:
        seen = []

        def record_and_scale(x):
            enabled = FP8GlobalStateManager.is_fp8_enabled()
            seen.append(
                (enabled, type(FP8GlobalStateManager.get_fp8_recipe()) if enabled else None)
            )
            return x * 2

        x = torch.ones(8, device="cuda", requires_grad=True)
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            y = checkpoint(record_and_scale, False, x)
        assert not FP8GlobalStateManager.is_fp8_enabled()
        y.sum().backward()  # the recompute runs here
        assert seen == [(True, type(recipe)), (True, type(recipe))]
        assert torch.equal(x.grad, torch.full((8,), 2.0, device="cuda"))

        seen.clear()
        x = torch.ones(8, device="cuda", requires_grad=True)
        y = checkpoint(record_and_scale, False, x)
        y.sum().backward()
        assert seen == [(False, None), (False, None)]
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not (HAVE_TE and torch.cuda.is_available()), reason="TE and CUDA required")
@pytest.mark.parametrize("fp8_enabled", [False, True])
def test_checkpoint_backward_restores_fp8_state(fp8_enabled: bool) -> None:
    """Exercise backward on the Python thread so coverage can trace FP8 state restoration."""
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    class CheckpointContext:
        """Save inputs without routing through the autograd engine."""

        def save_for_backward(self, *tensors):
            """Provide the context interface used by CheckpointFunction.forward."""
            self.saved_tensors = tensors

    recipe = _fp8_recipe_or_skip("delayed") if fp8_enabled else None
    Utils.initialize_model_parallel()
    model_parallel_cuda_manual_seed(123)
    try:
        seen = []

        def record_and_square(x):
            enabled = FP8GlobalStateManager.is_fp8_enabled()
            active_recipe = FP8GlobalStateManager.get_fp8_recipe() if enabled else None
            seen.append((enabled, active_recipe, torch.is_grad_enabled()))
            return x.square()

        x = torch.arange(1, 9, dtype=torch.float32, device="cuda", requires_grad=True)
        ctx = CheckpointContext()
        with te.fp8_autocast(enabled=fp8_enabled, fp8_recipe=recipe):
            output = CheckpointFunction.forward(ctx, record_and_square, False, x)
        assert not FP8GlobalStateManager.is_fp8_enabled()

        # CUDA autograd callbacks may run on C++ threads that coverage cannot trace.
        # Call the same entry point directly, with the grad mode used by autograd.
        grad_output = torch.full_like(output, 3.0)
        with torch.no_grad():
            grads = CheckpointFunction.backward(ctx, grad_output)

        assert len(seen) == 2
        for (enabled, active_recipe, grad_enabled), expected_grad_enabled in zip(
            seen, (False, True)
        ):
            assert enabled == fp8_enabled
            assert active_recipe is recipe
            assert grad_enabled == expected_grad_enabled
        assert not FP8GlobalStateManager.is_fp8_enabled()
        assert grads[:2] == (None, None)
        torch.testing.assert_close(output, x.square())
        torch.testing.assert_close(grads[2], 2 * x * grad_output)
        assert x.grad is None
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not (HAVE_TE and torch.cuda.is_available()), reason="TE and CUDA required")
@pytest.mark.parametrize("recipe_name", ["mxfp8", "delayed"])
def test_checkpoint_recomputes_te_linear_with_fp8_parameters(recipe_name):
    """A TE Linear with FP8 parameters and fused wgrad accumulation inside a checkpointed region:
    without the recorded FP8 state the recompute dequantizes the weight into a plain tensor and
    the backward fails on ``weight.main_grad``; with it the checkpointed pass matches the plain one.
    """
    import transformer_engine.pytorch as te

    recipe = _fp8_recipe_or_skip(recipe_name)
    Utils.initialize_model_parallel()
    model_parallel_cuda_manual_seed(123)
    try:
        inputs = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")

        def run(use_checkpoint):
            # A fresh, identically initialised module per pass: FP8 scaling state (amax history
            # under delayed scaling) evolves with every forward, so the two passes must not share
            # one module.
            torch.manual_seed(0)
            with te.fp8_model_init(enabled=True, recipe=recipe):
                linear = te.Linear(
                    256,
                    256,
                    bias=False,
                    params_dtype=torch.bfloat16,
                    fuse_wgrad_accumulation=True,
                    device="cuda",
                )
            linear.weight.main_grad = torch.zeros(256, 256, dtype=torch.float32, device="cuda")
            x = inputs.clone().requires_grad_(True)
            with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                out = checkpoint(linear, False, x) if use_checkpoint else linear(x)
            out.float().sum().backward()
            return out.detach().clone(), x.grad.clone(), linear.weight.main_grad.clone()

        ref_out, ref_x_grad, ref_main_grad = run(use_checkpoint=False)
        out, x_grad, main_grad = run(use_checkpoint=True)
        torch.testing.assert_close(out, ref_out)
        torch.testing.assert_close(x_grad, ref_x_grad)
        torch.testing.assert_close(main_grad, ref_main_grad, rtol=1e-3, atol=1e-2)
    finally:
        Utils.destroy_model_parallel()
