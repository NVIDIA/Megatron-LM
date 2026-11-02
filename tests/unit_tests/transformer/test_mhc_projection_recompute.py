# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import hashlib
import math
import os

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from megatron.core.fusions import fused_mhc_glm_projection as glm
from megatron.core.fusions import fused_mhc_kernels as fmk
from megatron.core.transformer.hyper_connection import HyperConnectionModule, _proj_rms_impl
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.transformer.mhc_fused_op_spy import spy_fused_mhc_ops


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("eps_inside_sqrt", [False, True])
def test_mhc_projection_matches_with_and_without_grad(device, dtype, eps_inside_sqrt):
    """Full-recompute forward and backward replay must use identical FP32 arithmetic."""
    torch.manual_seed(10)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        use_cpu_initialization=True,
        enable_mhc_connections=True,
        mhc_num_residual_streams=4,
        mhc_norm_eps_inside_sqrt=eps_inside_sqrt,
        mhc_keep_mappings_in_fp32=True,
    )
    module = HyperConnectionModule(config, layer_number=1).to(device)
    x = torch.randn(8, 2, 128, device=device, dtype=dtype, requires_grad=True)
    proj, norm = module._projection_and_get_norm(x)
    with torch.no_grad():
        replay_proj, replay_norm = module._projection_and_get_norm(x)
    assert torch.equal(replay_proj, proj)
    assert torch.equal(replay_norm, norm)

    expected = module.compute_mappings(x)
    sum(t.float().square().sum() for t in expected).backward()
    expected_grads = [x.grad.clone()] + [p.grad.clone() for p in module.parameters()]
    module.zero_grad(set_to_none=True)
    replay_x = x.detach().clone().requires_grad_(True)
    actual = checkpoint(module.compute_mappings, replay_x, use_reentrant=True)
    for a, e in zip(actual, expected):
        assert torch.equal(a, e)
    sum(t.float().square().sum() for t in actual).backward()
    for a, e in zip([replay_x.grad] + [p.grad for p in module.parameters()], expected_grads):
        assert torch.equal(a, e)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Inductor and eager differ on CUDA")
@pytest.mark.parametrize("normalize_before_projection", [False, True])
@pytest.mark.parametrize("eps_inside_sqrt", [False, True])
def test_mhc_mappings_replay_matches_when_dynamo_cache_is_full(
    eps_inside_sqrt, normalize_before_projection
):
    """A full-recompute forward must not mix compiled and eager mapping arithmetic.

    Dynamo's per-function recompile limit is shared by every layer and test in the
    process. Pin the state in which the no-grad forward of a reentrant checkpoint
    finds the cache full after the grad-mode forward has been compiled.
    """
    config_names = ("recompile_limit", "cache_size_limit")
    limit = next(name for name in config_names if hasattr(torch._dynamo.config, name))
    torch._dynamo.reset()
    try:
        torch.manual_seed(10)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_mhc_connections=True,
            mhc_num_residual_streams=4,
            mhc_norm_eps_inside_sqrt=eps_inside_sqrt,
            mhc_normalize_before_projection=normalize_before_projection,
            mhc_keep_mappings_in_fp32=True,
        )
        module = HyperConnectionModule(config, layer_number=1).cuda()
        # Another sequence length first: with dynamic shapes, the entry it leads to
        # would not accept the detached inputs of the no-grad forward below.
        module.compute_mappings(torch.randn(16, 2, 128, device="cuda", requires_grad=True))
        x = torch.randn(8, 2, 128, device="cuda", requires_grad=True)
        expected = module.compute_mappings(x)
        sum(t.float().square().sum() for t in expected).backward()
        expected_grads = [x.grad.clone()] + [p.grad.clone() for p in module.parameters()]
        module.zero_grad(set_to_none=True)
        replay_x = x.detach().clone().requires_grad_(True)
        with torch._dynamo.config.patch(**{limit: 1}):
            actual = checkpoint(module.compute_mappings, replay_x, use_reentrant=True)
            for a, e in zip(actual, expected):
                assert torch.equal(a, e)
            sum(t.float().square().sum() for t in actual).backward()
        grads = [replay_x.grad] + [p.grad for p in module.parameters()]
        for a, e in zip(grads, expected_grads):
            assert torch.equal(a, e)
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("input_requires_grad", [True, False])
@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("normalize_before_projection", [False, True])
def test_mhc_compute_h_runs_one_specialization(
    device, input_requires_grad, frozen, normalize_before_projection
):
    """Both halves of a full-recompute pair run the same compiled ``_compute_h`` entry.

    The no-grad forward and the grad-mode replay must not need a second Dynamo
    entry, whatever the inputs' requires_grad flags and the module's frozen
    parameters, also after another sequence length has been compiled. With
    mhc_normalize_before_projection, _compute_h does not read its norm input.
    """
    config_names = ("recompile_limit", "cache_size_limit")
    limit = next(name for name in config_names if hasattr(torch._dynamo.config, name))
    fail_names = ("fail_on_recompile_limit_hit", "fail_on_cache_limit_hit")
    fail = next((name for name in fail_names if hasattr(torch._dynamo.config, name)), None)
    torch._dynamo.reset()
    try:
        torch.manual_seed(10)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_mhc_connections=True,
            mhc_num_residual_streams=4,
            mhc_normalize_before_projection=normalize_before_projection,
            mhc_keep_mappings_in_fp32=True,
        )
        module = HyperConnectionModule(config, layer_number=1).to(device)
        module.requires_grad_(not frozen)

        def inputs(s):
            base = torch.randn(s * 2, 24, device=device, requires_grad=input_requires_grad)
            # Views of 2-D tensors, as _projection_and_get_norm returns them.
            return (base * 1).view(s, 2, 24), (base[:, :1].abs() + 1).view(s, 2, 1)

        module._compute_h_one_specialization(*inputs(16))
        proj, r = inputs(8)
        replay = module._compute_h_one_specialization(proj, r)
        # A second entry must fail hard instead of running eagerly; the unit-test
        # package turns Dynamo errors into eager fallbacks (suppress_errors).
        patch = {limit: 2, "suppress_errors": False}
        if fail is not None:
            patch[fail] = True
        with torch._dynamo.config.patch(**patch):
            with torch.no_grad():
                forward = module._compute_h_one_specialization(proj, r)
            again = module._compute_h_one_specialization(proj, r)
        needs_grad = input_requires_grad or not frozen
        for f, g, a in zip(forward, replay, again):
            assert not f.requires_grad
            assert g.requires_grad == needs_grad
            assert torch.equal(f, g) and torch.equal(a, g)
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mhc_mappings_under_inference_mode(device):
    """Inference mode records no autograd; its mappings match the no-grad forward's."""
    torch.manual_seed(10)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        use_cpu_initialization=True,
        enable_mhc_connections=True,
        mhc_num_residual_streams=4,
        mhc_keep_mappings_in_fp32=True,
    )
    module = HyperConnectionModule(config, layer_number=1).to(device)
    x = torch.randn(8, 2, 128, device=device)
    with torch.no_grad():
        expected = module.compute_mappings(x)
    with torch.inference_mode():
        actual = module.compute_mappings(x.clone())
    for a, e in zip(actual, expected):
        assert not a.requires_grad
        torch.testing.assert_close(a, e)


def _plain_proj_rms(x, weight, eps, eps_inside_sqrt):
    x = x.reshape(x.shape[0] * x.shape[1], -1).to(torch.float32)
    weight = weight.to(torch.float32)
    proj = torch.matmul(x, weight.t())
    if eps_inside_sqrt:
        r = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    else:
        r = (x.norm(dim=-1, keepdim=True) / (x.shape[-1] ** 0.5) + eps).reciprocal()
    return proj, r


@pytest.mark.parametrize("eps_inside_sqrt", [False, True])
@pytest.mark.parametrize("used_outputs", ["both", "projection", "norm"])
def test_mhc_projection_recompute_preserves_math_and_saves_activation_dtype(
    eps_inside_sqrt, used_outputs
):
    torch.manual_seed(123)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        use_cpu_initialization=True,
        enable_mhc_connections=True,
        mhc_norm_eps_inside_sqrt=eps_inside_sqrt,
    )
    module = HyperConnectionModule(config, layer_number=1)
    x = torch.randn(4, 2, 32).bfloat16()
    weight = torch.randn(24, 32) * 0.02
    module.mapping_proj.weight.data.copy_(weight)
    grad_proj = torch.randn(4, 2, 24)
    grad_norm = torch.randn(4, 2, 1)

    def run_plain():
        x_run = x.detach().clone().requires_grad_(True)
        weight_run = weight.detach().clone().requires_grad_(True)
        proj, norm = _plain_proj_rms(x_run, weight_run, module.norm_eps, eps_inside_sqrt)
        proj = proj.view(4, 2, 24)
        norm = norm.view(4, 2, 1)
        loss = 0
        if used_outputs in ("both", "projection"):
            loss = loss + (proj * grad_proj).sum()
        if used_outputs in ("both", "norm"):
            loss = loss + (norm * grad_norm).sum()
        loss.backward()
        return (proj.detach(), norm.detach()), (x_run.grad, weight_run.grad)

    def run_module():
        x_run = x.detach().clone().requires_grad_(True)
        module.mapping_proj.weight.grad = None
        saved = []

        def pack(tensor):
            saved.append((tensor.dtype, tuple(tensor.shape)))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            proj, norm = module._projection_and_get_norm(x_run)
            loss = 0
            if used_outputs in ("both", "projection"):
                loss = loss + (proj * grad_proj).sum()
            if used_outputs in ("both", "norm"):
                loss = loss + (norm * grad_norm).sum()
            loss.backward()
        return (proj.detach(), norm.detach()), (x_run.grad, module.mapping_proj.weight.grad), saved

    (proj_ref, norm_ref), (x_grad_ref, weight_grad_ref) = run_plain()
    (proj, norm), (x_grad, weight_grad), saved = run_module()

    assert torch.equal(proj, proj_ref)
    assert torch.equal(norm, norm_ref)
    assert torch.equal(x_grad, x_grad_ref)
    assert (weight_grad is None) == (weight_grad_ref is None)
    if weight_grad_ref is not None:
        assert torch.equal(weight_grad, weight_grad_ref)

    fp32_input = (torch.float32, (8, 32))
    bf16_input = (torch.bfloat16, (8, 32))
    assert fp32_input not in saved
    assert bf16_input in saved


@pytest.mark.parametrize("eps_inside_sqrt", [False, True])
def test_mhc_projection_stays_fp32_under_autocast(eps_inside_sqrt):
    """The FP32 projection/RMS recipe must not be downcast by an outer autocast scope."""
    torch.manual_seed(456)
    x = torch.randn(8, 32)
    weight = torch.randn(24, 32)
    eps = 1e-6

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual_proj, actual_norm = _proj_rms_impl(x, weight, eps, eps_inside_sqrt)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        with torch.autocast(device_type="cpu", enabled=False):
            expected_proj = torch.matmul(x.float(), weight.float().t())
            if eps_inside_sqrt:
                expected_norm = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
            else:
                expected_norm = 1.0 / (x.float().norm(dim=-1, keepdim=True) / math.sqrt(32) + eps)

    assert actual_proj.dtype == torch.float32
    assert actual_norm.dtype == torch.float32
    torch.testing.assert_close(actual_proj, expected_proj)
    torch.testing.assert_close(actual_norm, expected_norm)


# --- The Triton GLM projection pair through the module (CUDA) -------------------------------
# Per-op proj_rms with mhc_norm_eps_inside_sqrt and mhc_normalize_before_projection binds the
# pair, whose kernels are validated on compute capability (10, 3) only. The explicit triton
# policy raises on any input the pair refuses, so these tests never pass on the unfused
# fallback. Every name contains "cuda".

_NEEDS_CUDA_TRITON = pytest.mark.skipif(
    not (torch.cuda.is_available() and fmk.is_triton_available()), reason="needs CUDA and Triton"
)


def _local_rank_device():
    """The rank's own GPU: LOCAL_RANK under torchrun, 0 without it."""
    return torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))


def _skip_unless_glm_pair_device(device):
    """Skip unless the device's capability is one the GLM projection pair accepts."""
    capability = torch.cuda.get_device_capability(device)
    if capability not in glm.GLM_PAIR_DEVICE_CAPABILITIES:
        pytest.skip(
            f"the GLM projection pair is validated on {glm.GLM_PAIR_DEVICE_CAPABILITIES} only, "
            f"not {capability}"
        )


@pytest.fixture
def glm_pair_cuda_device():
    """Run the test on its rank's own GPU (LOCAL_RANK's, as the current device) and skip unless
    the pair accepts that device. Only the tests below request it."""
    device = _local_rank_device()
    _skip_unless_glm_pair_device(device)
    with torch.cuda.device(device):
        yield device


def _glm_pair_module(hidden_size, num_attention_heads, fused=True, **overrides):
    """A CUDA HyperConnectionModule with the GLM norm and order; fused binds the pair for
    proj_rms."""
    if fused:
        overrides.update(use_fused_mhc=True, mhc_fused_ops=["proj_rms"], mhc_fused_backend="triton")
    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        use_cpu_initialization=True,
        enable_mhc_connections=True,
        mhc_norm_eps_inside_sqrt=True,
        mhc_normalize_before_projection=True,
        **overrides,
    )
    module = HyperConnectionModule(config, layer_number=1).cuda()
    assert module._glm_proj_rms_op is (glm.glm_proj_rms_op("rn") if fused else None)
    return module


def _same_bits(a, b):
    """Bitwise equality (signs of zero included); None equals only None."""
    if a is None or b is None:
        return a is None and b is None
    as_int = {torch.float32: torch.int32, torch.bfloat16: torch.int16}
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and torch.equal(
            a.detach().contiguous().view(as_int[a.dtype]),
            b.detach().contiguous().view(as_int[b.dtype]),
        )
    )


def _count_backward_launches(monkeypatch):
    """Count the calls of the pair's module-level backward launcher: one GP-4 launch each."""
    original, calls = glm._glm_proj_bwd, []

    def counting(*args, **kwargs):
        calls.append(kwargs.get("mode"))
        return original(*args, **kwargs)

    monkeypatch.setattr(glm, "_glm_proj_bwd", counting)
    return calls


def _sha256_16(tensor):
    """An input's fingerprint, recorded with the results: the first 16 hex digits of the sha256 of
    the tensor's contiguous CPU bytes."""
    data = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


def _assert_pair_ran(spy, min_calls=1):
    spy.assert_ran("_glm_proj_rms_op", glm.glm_proj_rms_op("rn"), min_calls=min_calls)
    spy.assert_no_unsupported()


@_NEEDS_CUDA_TRITON
@pytest.mark.parametrize("hidden_size", [32, 4096], ids=["hidden32", "hidden4096"])
def test_mhc_glm_pair_cuda_matches_with_and_without_grad(
    glm_pair_cuda_device, monkeypatch, hidden_size
):
    """The pair runs before the checkpoint, so a full-recompute forward under no_grad and its
    replay with gradients run the same op: the grad and no-grad forwards are bitwise equal, and a
    reentrant checkpoint of compute_mappings equals the direct call in outputs and gradients."""
    torch.manual_seed(10)
    module = _glm_pair_module(
        hidden_size, 4, mhc_num_residual_streams=4, mhc_keep_mappings_in_fp32=True
    )
    x = torch.randn(8, 2, 4 * hidden_size).bfloat16().cuda().requires_grad_()
    backward_launches = _count_backward_launches(monkeypatch)
    with spy_fused_mhc_ops(module) as spy:
        proj, norm = module._projection_and_get_norm(x)
        with torch.no_grad():
            replay_proj, replay_norm = module._projection_and_get_norm(x)
        assert _same_bits(replay_proj, proj)
        assert _same_bits(replay_norm, norm)

        expected = module.compute_mappings(x)
        sum(t.float().square().sum() for t in expected).backward()
        expected_grads = [x.grad.clone()] + [p.grad.clone() for p in module.parameters()]
        module.zero_grad(set_to_none=True)
        replay_x = x.detach().clone().requires_grad_(True)
        actual = checkpoint(module.compute_mappings, replay_x, use_reentrant=True)
        for a, e in zip(actual, expected):
            assert _same_bits(a, e)
        sum(t.float().square().sum() for t in actual).backward()
        for a, e in zip([replay_x.grad] + [p.grad for p in module.parameters()], expected_grads):
            assert _same_bits(a, e)
    # Two forwards above, the direct compute_mappings, and the checkpoint's forward and replay.
    assert spy.calls("_glm_proj_rms_op") == 5
    _assert_pair_ran(spy, min_calls=5)
    assert len(backward_launches) == 2


@_NEEDS_CUDA_TRITON
@pytest.mark.parametrize("used_outputs", ["both", "projection", "norm"])
def test_mhc_glm_pair_cuda_recompute_contract(glm_pair_cuda_device, monkeypatch, used_outputs):
    """The drop-in and saved-tensor contract of the bound pair, at T 16 x K 16384 and T 8 x K 512.

    The module's outputs and gradients equal a direct fused_glm_proj_rms call bit for bit (views,
    eps and rounding wired through); without the projection's gradient there is no weight
    gradient, as on today's unfused module; and the forward saves exactly x (the residual's own
    storage), W, s and p through ctx.save_for_backward, never an fp32 copy of x.
    """
    contexts = []
    original_save = torch.autograd.function.FunctionCtx.save_for_backward

    def counting_save(ctx, *tensors):
        contexts.append(len(tensors))
        return original_save(ctx, *tensors)

    monkeypatch.setattr(torch.autograd.function.FunctionCtx, "save_for_backward", counting_save)
    backward_launches = _count_backward_launches(monkeypatch)

    for s, b, K in ((8, 2, 16384), (4, 2, 512)):
        T = s * b
        module = _glm_pair_module(K // 4, 2)
        unfused = _glm_pair_module(K // 4, 2, fused=False)
        torch.manual_seed(123)
        x = torch.randn(s, b, K).bfloat16().cuda()
        weight = (torch.randn(24, K) * 0.02).cuda()
        grad_proj = torch.randn(s, b, 24).cuda()
        grad_norm = torch.randn(s, b, 1).cuda()
        for m in (module, unfused):
            m.mapping_proj.weight.data.copy_(weight)
        # The generated inputs' fingerprints, taken before anything runs on them.
        fingerprints = (
            f"x_sha256_16={_sha256_16(x)} W_sha256_16={_sha256_16(weight)} "
            f"G_p_sha256_16={_sha256_16(grad_proj)} G_s_sha256_16={_sha256_16(grad_norm)}"
        )

        def used(proj, norm):
            outputs, grads = [], []
            if used_outputs in ("both", "projection"):
                outputs.append(proj)
                grads.append(grad_proj)
            if used_outputs in ("both", "norm"):
                outputs.append(norm)
                grads.append(grad_norm)
            return outputs, grads

        def run_module(m):
            x_run = x.detach().clone().requires_grad_(True)
            m.mapping_proj.weight.grad = None
            saved = []

            def pack(tensor):
                saved.append((tensor.dtype, tuple(tensor.shape), tensor.data_ptr(), tensor.nbytes))
                return tensor

            contexts.clear()
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                proj, norm = m._projection_and_get_norm(x_run)
            forward_contexts = list(contexts)
            backward_launches.clear()
            torch.autograd.backward(*used(proj, norm))
            outputs = (proj.detach(), norm.detach(), x_run.grad, m.mapping_proj.weight.grad)
            return outputs, saved, forward_contexts, list(backward_launches), x_run

        saved = unfused_saved = None
        try:
            with spy_fused_mhc_ops(module) as spy:
                fused, saved, forward_contexts, launches, x_run = run_module(module)
            _assert_pair_ran(spy)
            # One forward through the module, so one call of the bound pair.
            assert spy.calls("_glm_proj_rms_op") == 1, (T, K)
            assert forward_contexts == [4], (T, K, forward_contexts)
            assert len(launches) == 1, (T, K, launches)
            assert sorted((str(dtype), shape) for dtype, shape, _, _ in saved) == sorted(
                [
                    (str(torch.bfloat16), (T, K)),
                    (str(torch.float32), (24, K)),
                    (str(torch.float32), (T, 1)),
                    (str(torch.float32), (T, 24)),
                ]
            ), (T, K, saved)
            assert (torch.float32, (T, K)) not in [(dtype, shape) for dtype, shape, _, _ in saved]
            assert [ptr for dtype, _, ptr, _ in saved if dtype == torch.bfloat16] == [
                x_run.data_ptr()
            ]

            reference, unfused_saved, _, unfused_launches, _ = run_module(unfused)
            assert unfused_launches == []
            weight_grad, weight_grad_ref = fused[3], reference[3]
            assert (weight_grad is None) == (weight_grad_ref is None) == (used_outputs == "norm")

            # The same tensors through the pair directly, with the eps and rounding the module uses.
            x_leaf = x.reshape(T, K).detach().clone().requires_grad_(True)
            w_leaf = weight.detach().clone().requires_grad_(True)
            p, r = glm.fused_glm_proj_rms(
                x_leaf, w_leaf, module.config.layernorm_epsilon, rounding="rn"
            )
            backward_launches.clear()
            torch.autograd.backward(*used(p.view(s, b, 24), r.view(s, b, 1)))
            assert len(backward_launches) == 1
            direct = (p.view(s, b, 24), r.view(s, b, 1), x_leaf.grad.view(s, b, K), w_leaf.grad)
            for name, got, want in zip(("proj", "norm", "x grad", "weight grad"), fused, direct):
                assert _same_bits(got, want), (T, K, name)
        finally:
            # Recorded, not gated, for every case whether it passes or fails: the persistent saved
            # bytes of one layer's projection, fused and today's unfused path (the bf16 input is the
            # residual's own storage in both; a run that did not return leaves its count out), and
            # the fingerprints of the generated inputs.
            fields = [f"T={T}", f"K={K}", f"used={used_outputs}"]
            for field, entries in (("fused_bytes", saved), ("unfused_bytes", unfused_saved)):
                if entries is not None:
                    fields.append(f"{field}={sum(entry[3] for entry in entries)}")
            print("GLM_PAIR_PRC_SAVED", *fields, f"input_bytes={x.nbytes}", fingerprints)


@_NEEDS_CUDA_TRITON
def test_mhc_glm_pair_cuda_stays_fp32_under_autocast(glm_pair_cuda_device, monkeypatch):
    """An outer CUDA bf16 autocast does not reach the bound pair through the module: fp32 outputs,
    and outputs and gradients bitwise equal to the same call outside autocast."""
    backward_launches = _count_backward_launches(monkeypatch)
    for s, b, K in ((8, 2, 16384), (4, 2, 512)):
        module = _glm_pair_module(K // 4, 2)
        # The recompute-contract test's inputs: seed 123, then x, W, G_p and G_s in that order.
        torch.manual_seed(123)
        x = torch.randn(s, b, K).bfloat16().cuda()
        weight = (torch.randn(24, K) * 0.02).cuda()
        module.mapping_proj.weight.data.copy_(weight)
        grad_proj = torch.randn(s, b, 24).cuda()
        grad_norm = torch.randn(s, b, 1).cuda()

        def run(autocast):
            x_run = x.detach().clone().requires_grad_(True)
            module.zero_grad(set_to_none=True)
            backward_launches.clear()
            with spy_fused_mhc_ops(module) as spy:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                    proj, norm = module._projection_and_get_norm(x_run)
                torch.autograd.backward((proj, norm), (grad_proj, grad_norm))
            _assert_pair_ran(spy)
            # One forward through the module, so one call of the bound pair.
            assert spy.calls("_glm_proj_rms_op") == 1, (s * b, K, autocast)
            assert len(backward_launches) == 1
            return proj.detach(), norm.detach(), x_run.grad, module.mapping_proj.weight.grad

        outside, inside = run(False), run(True)
        for result in (outside, inside):
            assert result[0].dtype == result[1].dtype == torch.float32
        for name, got, want in zip(("proj", "norm", "x grad", "weight grad"), inside, outside):
            assert _same_bits(got, want), (s * b, K, name)
