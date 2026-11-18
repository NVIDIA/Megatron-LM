# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Test the inter-sublayer dependency introduced by V4.1 mHC."""

import copy
from types import SimpleNamespace

import pytest
import torch

from megatron.core.tensor_parallel import random as tensor_parallel_random
from megatron.core.transformer import hyper_connection
from megatron.core.transformer.hyper_connection import (
    HyperConnectionModule,
    SinglePassMHCState,
    get_mhc_aggregate,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class _CoefficientProjection(torch.nn.Module):
    """Exercise the production mapping without requiring a distributed model."""

    def __init__(self, device):
        super().__init__()
        self.n = 4
        self.hidden_size = 32
        self.norm_eps = 1e-6
        self.compute_h_eps = 1e-6
        self.config = SimpleNamespace(use_fused_mhc=False)
        self.mapping_proj = torch.nn.Linear(128, 24, bias=False, device=device)
        self.alpha_pre = torch.nn.Parameter(torch.tensor([0.3], device=device))
        self.alpha_post = torch.nn.Parameter(torch.tensor([-0.2], device=device))
        self.alpha_res = torch.nn.Parameter(torch.tensor([0.1], device=device))
        self.bias = torch.nn.Parameter(torch.randn(24, device=device) * 0.03)

    forward = HyperConnectionModule._single_pass_coefficients
    aggregate = HyperConnectionModule._single_pass_aggregate
    combine = HyperConnectionModule._single_pass_bda


def _coefficient_reference(module, hidden):
    """Keep the original FP32 formula independent of the optimized helpers."""
    x = hidden.float()
    with torch.autocast(device_type=x.device.type, enabled=False):
        projected = torch.nn.functional.linear(x, module.mapping_proj.weight.float())
        projected = projected * torch.rsqrt(x.square().mean(-1, keepdim=True) + module.norm_eps)
        pre = (projected[..., :4] * module.alpha_pre + module.bias[:4]).sigmoid()
        post = (projected[..., 4:8] * module.alpha_post + module.bias[4:8]).sigmoid()
        residual = (projected[..., 8:] * module.alpha_res + module.bias[8:]).unflatten(-1, (4, 4))
    return pre + module.compute_h_eps, 2 * post, residual


@pytest.fixture(params=["cpu", "cuda"])
def coefficient_device(request, monkeypatch):
    """CPU checks execute the real autograd boundary; CUDA also compiles its helpers."""
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "cpu":
        for name in (
            "_single_pass_projection_rms_forward",
            "_single_pass_projection_rms_backward",
            "_single_pass_coefficient_activations",
        ):
            helper = getattr(hyper_connection, name)
            monkeypatch.setattr(hyper_connection, name, helper._torchdynamo_orig_callable)
    else:
        # The oracle promises FP32 arithmetic rather than TF32 GEMMs.
        monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    return device


def _assert_coefficient_close(actual, expected, activation_dtype):
    torch.testing.assert_close(
        actual,
        expected,
        rtol=2e-2 if activation_dtype == torch.bfloat16 else 2e-5,
        atol=2e-3 if activation_dtype == torch.bfloat16 else 2e-6,
    )
    if actual.numel() and expected.float().norm() > 0:
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0, eps=1e-30
        )
        assert cosine >= (0.99999 if activation_dtype == torch.bfloat16 else 0.999999)


def _assert_original_mapping_storage(saved, hidden, module):
    hidden_storage = hidden.untyped_storage()._cdata
    parameter_storages = {p.untyped_storage()._cdata for p in module.parameters()}
    assert any(
        tensor.shape == hidden.shape
        and tensor.dtype == hidden.dtype
        and tensor.untyped_storage()._cdata == hidden_storage
        for tensor in saved
    )
    assert not any(
        tensor.dtype == torch.float32
        and tensor.numel() >= hidden.numel()
        and tensor.untyped_storage()._cdata not in parameter_storages | {hidden_storage}
        for tensor in saved
    ), "mapping saved an independent full-width FP32 activation"


@pytest.mark.parametrize(
    "dtype,autocast,layout,trainable,used_outputs",
    [
        pytest.param(torch.float32, False, "normal", "all", (0, 1, 2), id="fp32-all"),
        pytest.param(torch.bfloat16, False, "normal", "all", (0, 1, 2), id="bf16-all"),
        pytest.param(torch.float32, True, "normal", "hidden", (0, 1, 2), id="amp-hidden-only"),
        pytest.param(torch.bfloat16, True, "normal", "weight", (0, 1, 2), id="amp-weight-only"),
        pytest.param(torch.float32, True, "normal", "coefficients", (0,), id="amp-pre-only"),
        pytest.param(torch.bfloat16, False, "normal", "all", (0,), id="bf16-pre-only"),
        pytest.param(torch.bfloat16, True, "noncontiguous", "all", (1,), id="amp-post-strided"),
        pytest.param(torch.float32, False, "tiny", "all", (2,), id="tiny-residual-only"),
        pytest.param(torch.bfloat16, True, "zero", "all", (0, 1, 2), id="amp-zero"),
        pytest.param(torch.float32, False, "empty", "all", (0, 1, 2), id="empty-fp32"),
        pytest.param(torch.bfloat16, True, "empty", "all", (0, 1, 2), id="empty-bf16"),
    ],
)
def test_single_pass_coefficients_match_original_formula(
    coefficient_device, dtype, autocast, layout, trainable, used_outputs
):
    """Compare coefficients and all requested gradients, including unused output branches."""
    device = coefficient_device
    torch.manual_seed(274)
    module = _CoefficientProjection(device)
    if layout == "tiny":
        module.norm_eps = 1e-12
    hidden = torch.randn(0 if layout == "empty" else 5, 2, 128, device=device, dtype=dtype)
    if layout == "noncontiguous":
        hidden = hidden.transpose(0, 1).contiguous().transpose(0, 1)
        assert not hidden.is_contiguous()
    elif layout == "tiny":
        hidden.mul_(1e-9)
    elif layout == "zero":
        hidden.zero_()
    hidden.requires_grad_(trainable in ("all", "hidden"))
    for name, parameter in module.named_parameters():
        parameter.requires_grad_(
            trainable == "all"
            or (trainable == "weight" and name == "mapping_proj.weight")
            or (trainable == "coefficients" and name != "mapping_proj.weight")
        )
    reference_module = copy.deepcopy(module)
    reference_hidden = hidden.detach().clone(memory_format=torch.preserve_format)
    reference_hidden.requires_grad_(hidden.requires_grad)
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=autocast):
        outputs = module(hidden)
        references = _coefficient_reference(reference_module, reference_hidden)
    for actual, expected in zip(outputs, references):
        assert actual.dtype == expected.dtype == torch.float32
        _assert_coefficient_close(actual, expected, torch.float32)
    probes = [torch.randn_like(output) for output in outputs]
    loss = sum((outputs[index] * probes[index]).sum() for index in used_outputs)
    reference_loss = sum((references[index] * probes[index]).sum() for index in used_outputs)
    loss.backward()
    reference_loss.backward()
    for actual, expected in zip(
        (hidden, *module.parameters()), (reference_hidden, *reference_module.parameters())
    ):
        if actual.requires_grad:
            if expected.grad is None:
                # AOTAutograd may materialize zero gradients for unused outputs.
                assert actual.grad is None or torch.count_nonzero(actual.grad) == 0
            else:
                assert actual.grad is not None
                assert torch.isfinite(actual.grad).all()
                _assert_coefficient_close(actual.grad, expected.grad, actual.grad.dtype)
        else:
            assert actual.grad is None and expected.grad is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("hidden_requires_grad", [False, True])
def test_single_pass_mapping_retains_original_activation_storage(
    coefficient_device, dtype, hidden_requires_grad
):
    """Saving the original alias must replace the independent wide FP32 activation."""
    torch.manual_seed(275)
    module = _CoefficientProjection(coefficient_device)
    hidden = torch.randn(32, 2, 128, device=coefficient_device, dtype=dtype)
    hidden.requires_grad_(hidden_requires_grad)
    saved = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        outputs = module(hidden)
    _assert_original_mapping_storage(saved, hidden, module)
    sum(output.square().sum() for output in outputs).backward()
    assert module.mapping_proj.weight.grad is not None


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_single_pass_compiled_mapping_matches_reference_and_storage(device, monkeypatch):
    """Default Inductor must preserve the original-storage boundary as well as gradients."""
    if device == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    torch.manual_seed(277)
    module = _CoefficientProjection(device)
    reference_module = copy.deepcopy(module)
    hidden = torch.randn(32, 2, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    saved = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            outputs = module(hidden)
    references = _coefficient_reference(reference_module, reference_hidden)
    _assert_original_mapping_storage(saved, hidden, module)
    probes = [torch.randn_like(output) for output in outputs]
    for actual, expected in zip(outputs, references):
        _assert_coefficient_close(actual, expected, torch.float32)
    sum((output * probe).sum() for output, probe in zip(outputs, probes)).backward()
    sum((output * probe).sum() for output, probe in zip(references, probes)).backward()
    for actual, expected in zip(
        (hidden, *module.parameters()), (reference_hidden, *reference_module.parameters())
    ):
        assert actual.grad is not None and expected.grad is not None
        _assert_coefficient_close(actual.grad, expected.grad, actual.grad.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_single_pass_mapping_survives_two_recompute_groups(coefficient_device, dtype, monkeypatch):
    """Cross-group pre_mix gradients must wait for discarded mapping inputs to be restored."""
    device = coefficient_device
    if device == "cpu":
        # These deterministic operations need no CUDA RNG; storage discard and the
        # production C++ StorageImpl rebind still run without substitutes.
        monkeypatch.setattr(
            tensor_parallel_random,
            "_get_all_rng_states",
            lambda: (torch.get_rng_state(), None, None),
        )
        monkeypatch.setattr(
            tensor_parallel_random,
            "_set_all_rng_states",
            lambda cpu, _cuda, _tracker: torch.set_rng_state(cpu),
        )
    torch.manual_seed(276)
    modules = torch.nn.ModuleList([_CoefficientProjection(device) for _ in range(4)])
    reference_modules = copy.deepcopy(modules)
    hidden = torch.randn(4, 2, 128, device=device, dtype=dtype, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    checkpointed_interiors = []
    retained_boundaries = []
    mapping_aliases = []

    def run_stack(layers, initial, grouped):
        output, pre_mix = initial, None
        for group_start in (0, 2):
            manager = tensor_parallel_random.CheckpointWithoutOutputManager() if grouped else None
            for index in range(group_start, group_start + 2):
                layer = layers[index]
                if grouped:
                    saved = []
                    with torch.autograd.graph.saved_tensors_hooks(
                        lambda tensor: (saved.append(tensor), tensor)[1], lambda tensor: tensor
                    ):
                        pre, post, residual = layer(output)
                    mapping_aliases.extend(
                        tensor
                        for tensor in saved
                        if tensor.shape == output.shape
                        and tensor.untyped_storage()._cdata == output.untyped_storage()._cdata
                    )
                else:
                    pre, post, residual = _coefficient_reference(layer, output)
                # A small dense mixer isolates the mapping/checkpoint dependency
                # from the unrelated Sinkhorn and fused aggregation kernels.
                residual = residual.softmax(-1)
                if manager is None:
                    branch = layer.aggregate(output, pre_mix)
                else:
                    branch = tensor_parallel_random.CheckpointWithoutOutput(
                        fp8=None, ckpt_manager=manager
                    ).checkpoint(layer.aggregate, output, pre_mix)
                branch = torch.tanh(branch * 0.3 + 0.1)
                args = (residual, output, post, branch, None, 0.0, True)
                if manager is not None and index == group_start:
                    output = tensor_parallel_random.CheckpointWithoutOutput(
                        fp8=None, ckpt_manager=manager
                    ).checkpoint(layer.combine, *args)
                    checkpointed_interiors.append(output)
                else:
                    output = layer.combine(*args)
                pre_mix = pre
            if manager is not None:
                retained_boundaries.append(output)
                manager.discard_all_outputs_and_register_unified_recompute(output)
                assert checkpointed_interiors[-1].untyped_storage().nbytes() == 0
                assert output.untyped_storage().nbytes() == output.numel() * output.element_size()
        return SinglePassMHCState(pre_mix).contract(output, 4)

    expected = run_stack(reference_modules, reference_hidden, False)
    actual = run_stack(modules, hidden, True)
    _assert_coefficient_close(actual, expected, dtype)
    assert len(mapping_aliases) == 4
    for interior in checkpointed_interiors:
        assert any(
            alias.untyped_storage()._cdata == interior.untyped_storage()._cdata
            and alias.untyped_storage().nbytes() == 0
            for alias in mapping_aliases
        )
    probe = torch.randn_like(actual)
    expected.backward(probe)
    actual.backward(probe)
    for interior in checkpointed_interiors:
        assert interior.untyped_storage().nbytes() == interior.numel() * interior.element_size()
    assert all(boundary.untyped_storage().nbytes() > 0 for boundary in retained_boundaries)
    for observed, reference in zip(
        (hidden, *modules.parameters()), (reference_hidden, *reference_modules.parameters())
    ):
        assert observed.grad is not None and reference.grad is not None
        _assert_coefficient_close(observed.grad, reference.grad, observed.grad.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("fused", [False, True])
def test_previous_mix_controls_input_and_receives_gradients(dtype, fused):
    config = TransformerConfig(
        num_layers=2,
        hidden_size=16,
        num_attention_heads=2,
        layernorm_epsilon=1e-20,
        enable_mhc_connections=True,
        mhc_single_pass=True,
        use_fused_mhc=fused,
    )
    module = HyperConnectionModule(config, 1).cuda()
    x = torch.randn(5, 2, 64, device="cuda", dtype=dtype, requires_grad=True)
    previous = torch.randn(5, 2, 4, device="cuda", requires_grad=True)
    state = SinglePassMHCState(previous)
    branch, residual, post, residual_input = module(x, mhc_state=state, return_residual=True)
    following = state.pre_mix
    oracle = torch.einsum("sbn,sbnc->sbc", previous, x.float().unflatten(-1, (4, 16)))
    torch.testing.assert_close(branch, oracle.to(dtype))
    assert following.dtype == post.dtype == residual.dtype == torch.float32
    bias = torch.randn(16, device="cuda", requires_grad=True)
    output = module.fused_h_res_h_post_bda(
        residual, residual_input, post, (branch, bias), 0.0, True, False
    )
    reference = torch.matmul(residual.transpose(-1, -2), x.float().unflatten(-1, (4, 16)))
    reference = reference + post.unsqueeze(-1) * (branch.float() + bias).unsqueeze(-2)
    torch.testing.assert_close(output, reference.flatten(-2).to(dtype))
    # Direct activation-dtype stores must preserve FP32 coefficient gradients.
    leaves = (x, previous, module.mapping_proj.weight, bias)
    actual_grads = torch.autograd.grad(output.float().square().mean(), leaves, retain_graph=True)
    reference_grads = torch.autograd.grad(
        reference.flatten(-2).to(dtype).float().square().mean(), leaves, retain_graph=True
    )
    for actual, expected in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(actual_grads[-1], reference_grads[-1], atol=1e-6, rtol=1e-5)
    state.contract(output, 4).square().mean().backward()
    assert previous.grad is not None and previous.grad.abs().sum() > 0
    assert module.mapping_proj.weight.grad is not None
    assert torch.isfinite(x.grad).all()


def test_entry_selects_first_stream_and_exit_uses_last_mix():
    x = torch.arange(24.0, device="cuda").reshape(2, 1, 12).requires_grad_()
    torch.testing.assert_close(SinglePassMHCState().contract(x, 3), x[..., :4])
    mix = torch.tensor([0.0, 0.0, 1.0], device="cuda").expand(2, 1, 3)
    torch.testing.assert_close(SinglePassMHCState(mix).contract(x, 3), x[..., -4:])
    with pytest.raises(ValueError, match="preceding"):
        SinglePassMHCState(mix[:, :, :2]).contract(x, 3)


@pytest.mark.parametrize("backend", ["none", "native", "triton", "cutile"])
def test_final_contraction_preserves_fp32_mix_and_gradients(backend):
    """The final readout must use the requested backend without rounding its mix."""
    from megatron.core.fusions import fused_mhc_kernels

    if backend == "cutile" and not fused_mhc_kernels.is_cutile_available():
        pytest.skip("cuTile unavailable")
    cfg = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        enable_mhc_connections=True,
        mhc_single_pass=True,
        use_fused_mhc=backend != "none",
        mhc_fused_backend="auto" if backend == "none" else backend,
    )
    torch.manual_seed(432)
    x = torch.randn(129, 1, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    mix = torch.randn(129, 1, 4, device="cuda", dtype=torch.float32, requires_grad=True)
    aggregate = get_mhc_aggregate(cfg)
    assert (aggregate is None) == (backend == "none")
    actual = SinglePassMHCState(mix).contract(x, 4, aggregate=aggregate)
    expected = (x.float().unflatten(-1, (4, 128)) * mix.unsqueeze(-1)).sum(-2).bfloat16()
    probe = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (x, mix), probe, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (x, mix), probe)
    assert actual.dtype == x.dtype
    assert actual_grads[1].dtype == torch.float32
    for observed, reference in zip((actual, *actual_grads), (expected, *expected_grads)):
        cosine = torch.nn.functional.cosine_similarity(
            observed.float().flatten(), reference.float().flatten(), dim=0
        )
        assert cosine >= 0.99999


@pytest.mark.parametrize("backend", ["none", "native", "triton"])
def test_fp32_mixing_is_preserved_under_outer_autocast(backend):
    """AMP must not round the residual-mixing matmul to BF16 for FP32 activations."""
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        enable_mhc_connections=True,
        mhc_single_pass=True,
        use_fused_mhc=backend != "none",
        mhc_fused_backend="auto" if backend == "none" else backend,
    )
    module = HyperConnectionModule(config, 1).cuda()
    x = torch.randn(5, 2, 64, device="cuda")
    previous = torch.randn(5, 2, 4, device="cuda")
    outputs = []
    for enabled in (False, True):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled):
            state = SinglePassMHCState(previous)
            branch, residual_mix, post, residual = module(x, mhc_state=state, return_residual=True)
            outputs.append(
                module.fused_h_res_h_post_bda(
                    residual_mix, residual, post, (branch, None), 0.0, False, False
                )
            )
    assert all(output.dtype == torch.float32 for output in outputs)
    torch.testing.assert_close(outputs[1], outputs[0], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("fused", [False, True])
def test_single_pass_mhc_replays_bit_exact(fused):
    """Replay the new inter-sublayer gradient edge as well as the parameter gradients."""
    from tests.unit_tests.determinism.kernels.harness import assert_module_replays_bit_exact

    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        enable_mhc_connections=True,
        mhc_single_pass=True,
        use_fused_mhc=fused,
    )

    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mhc = HyperConnectionModule(config, 1).cuda()

        def forward(self, hidden, previous):
            state = SinglePassMHCState(previous)
            branch, residual_mix, post, residual = self.mhc(
                hidden, mhc_state=state, return_residual=True
            )
            output = self.mhc.fused_h_res_h_post_bda(
                residual_mix, residual, post, (branch, None), 0.0, True, False
            )
            return state.contract(
                output, config.mhc_num_residual_streams, aggregate=get_mhc_aggregate(config)
            )

    assert_module_replays_bit_exact(
        Projection(),
        (
            torch.randn(4, 1, 64, device="cuda", requires_grad=True),
            torch.randn(4, 1, 4, device="cuda", requires_grad=True),
        ),
        replays=3,
        what="single-pass mHC",
    )


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_residual_grad_add_matches_autograd_accumulation(
    monkeypatch, dtype, fused, recompute
):
    """MCORE_MHC_SP_FUSED_GRAD_ADD=exact (the default) sums each single-pass residual input's
    three gradients in the order autograd accumulates them: a chain of sublayers gets bit-
    identical outputs and gradients with the switch off and on. 3-tuple callers never split."""
    from megatron.core.fusions import fused_mhc_kernels
    from tests.unit_tests.determinism.kernels.harness import bytes_equal

    config = TransformerConfig(
        num_layers=3,
        hidden_size=64,
        num_attention_heads=2,
        enable_mhc_connections=True,
        mhc_single_pass=True,
        use_fused_mhc=fused,
    )
    n = config.mhc_num_residual_streams
    calls = []
    add_3 = fused_mhc_kernels.fused_add_3_sequential

    def counting_add_3(*args):
        calls.append(args[0].shape)
        return add_3(*args)

    monkeypatch.setattr(fused_mhc_kernels, "fused_add_3_sequential", counting_add_3)
    torch.manual_seed(1203)
    hidden = torch.randn(256, 2, n * 64, device="cuda", dtype=dtype)
    previous = torch.randn(256, 2, n, device="cuda")
    probe = torch.randn(256, 2, 64, device="cuda", dtype=dtype)

    def run(mode, return_residual):
        monkeypatch.setattr(hyper_connection, "_SINGLE_PASS_FUSED_GRAD_ADD", mode)
        torch.manual_seed(7)
        mhcs = [HyperConnectionModule(config, i + 1).cuda() for i in range(3)]
        branches = [torch.nn.Linear(64, 64).cuda().to(dtype) for _ in range(3)]
        x = hidden.clone().requires_grad_()
        mix = previous.clone().requires_grad_()
        state = SinglePassMHCState(mix)
        output = x
        for mhc, branch in zip(mhcs, branches):
            manager = tensor_parallel_random.CheckpointWithoutOutputManager() if recompute else None
            result = mhc(
                output,
                mhc_recompute_manager=manager,
                mhc_state=state,
                return_residual=return_residual,
            )
            aggregated, residual_mix, post = result[:3]
            residual = result[3] if return_residual else output
            branch_output = torch.tanh(branch(aggregated))
            output = mhc.fused_h_res_h_post_bda(
                residual_mix, residual, post, (branch_output, None), 0.0, True, False
            )
            if manager is not None:
                manager.discard_all_outputs_and_register_unified_recompute(output)
        final = state.contract(output, n, aggregate=get_mhc_aggregate(config))
        leaves = [x, mix] + [p for module in (*mhcs, *branches) for p in module.parameters()]
        # backward(), not autograd.grad(): the unified recompute hook rejects .grad(). Every run
        # builds fresh leaves, so their .grad holds this run's gradients only.
        final.backward(probe)
        return final.detach(), [leaf.grad for leaf in leaves]

    for return_residual in (True, False):
        calls.clear()
        expected_output, expected_grads = run(None, return_residual)
        assert not calls
        output, grads = run("exact", return_residual)
        assert len(calls) == (3 if return_residual else 0)
        assert bytes_equal(output, expected_output)
        assert len(grads) == len(expected_grads)
        for index, (actual, expected) in enumerate(zip(grads, expected_grads)):
            assert bytes_equal(actual, expected), index


@pytest.mark.parametrize(
    "value,mode",
    [
        (None, "exact"),
        ("1", "exact"),
        (" Exact ", "exact"),
        ("on", "exact"),
        ("fp32", "fp32"),
        ("0", None),
        ("off", None),
        ("", ValueError),
        ("2", ValueError),
    ],
)
def test_fused_grad_add_mode_parsing(value, mode):
    """MCORE_MHC_SP_FUSED_GRAD_ADD: unset is exact; other values must name a mode."""
    if mode is ValueError:
        with pytest.raises(ValueError, match="MCORE_MHC_SP_FUSED_GRAD_ADD"):
            hyper_connection._parse_single_pass_fused_grad_add(value)
    else:
        assert hyper_connection._parse_single_pass_fused_grad_add(value) == mode


def test_cutile_proj_rms_is_opt_in_and_falls_back(monkeypatch):
    """MCORE_MHC_SP_CUTILE_PROJ_RMS is off by default, so single-pass modules keep the Inductor
    mapping. On, residuals the cuTile kernels do not tile (K = n * hidden < 8192 here) warn and
    keep the Inductor mapping, bit for bit."""
    import os

    from megatron.core.fusions import fused_mhc_kernels
    from tests.unit_tests.determinism.kernels.harness import bytes_equal

    if os.environ.get("MCORE_MHC_SP_CUTILE_PROJ_RMS") is None:
        assert fused_mhc_kernels.SINGLE_PASS_CUTILE_PROJ_RMS is False
    config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=2,
        enable_mhc_connections=True,
        mhc_single_pass=True,
        use_fused_mhc=True,
    )
    monkeypatch.setattr(fused_mhc_kernels, "SINGLE_PASS_CUTILE_PROJ_RMS", False)
    assert not HyperConnectionModule(config, 1)._single_pass_cutile_proj_rms
    if not fused_mhc_kernels.is_cutile_available():
        pytest.skip("cuTile unavailable")
    monkeypatch.setattr(fused_mhc_kernels, "SINGLE_PASS_CUTILE_PROJ_RMS", True)
    torch.manual_seed(1302)
    module = HyperConnectionModule(config, 1).cuda()
    assert module._single_pass_cutile_proj_rms
    hidden = torch.randn(128, 1, 4 * 64, device="cuda", dtype=torch.bfloat16)
    with pytest.warns(UserWarning, match="MCORE_MHC_SP_CUTILE_PROJ_RMS"):
        routed = module.compute_mappings(hidden)
    monkeypatch.setattr(module, "_single_pass_cutile_proj_rms", False)
    for actual, expected in zip(routed, module.compute_mappings(hidden)):
        assert bytes_equal(actual, expected)


@pytest.mark.parametrize(
    "cuda_graph_impl,supported", [("none", True), ("full_iteration", True), ("local", False)]
)
def test_single_pass_graph_validation(cuda_graph_impl, supported):
    """A full-iteration graph captures the whole step, so single-pass mHC state needs no
    tensor-state adapter there; per-layer local graphs still require one."""
    kwargs = dict(
        num_layers=2,
        hidden_size=16,
        num_attention_heads=2,
        enable_mhc_connections=True,
        mhc_single_pass=True,
        cuda_graph_impl=cuda_graph_impl,
    )
    if supported:
        assert TransformerConfig(**kwargs).cuda_graph_impl == cuda_graph_impl
    else:
        with pytest.raises(NotImplementedError, match="tensor-state graph adapter"):
            TransformerConfig(**kwargs)
