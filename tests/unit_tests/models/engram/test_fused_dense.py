# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fused Triton dense path vs. the PyTorch reference path of the Engram module.

The gate is ``sigmoid(sign(s) * sqrt(|s|))`` of a bf16 score ``s``. Its derivative grows like
``1 / sqrt(|s|)`` near zero, so the gradients upstream of the score (key projection, key and
query norms, the Engram input, the tables) amplify any one-ulp rounding difference in ``s``:
two valid bf16 evaluations differ there by a few 1e-3 on random inputs, and both are 1e-2 to
1e0 away from fp64. The tests separate that conditioning from implementation error:

* ``test_fused_matches_reference_well_conditioned``: scores bounded away from zero. Every
  output and gradient must match the reference to rel-L2 <= 1e-3.
* ``test_fused_matches_reference``: random inputs through the whole module, including tiny
  and batched shapes. Output and the gradients that do not pass through the score must
  match to rel-L2 <= 1e-3. For the score-chain gradients the fused path must be no further
  from an fp64 evaluation than the reference is; a last-bit input perturbation of the
  reference is printed alongside for scale.
"""

import math

import pytest
import torch

from megatron.core.models.engram.config import EngramConfig
from megatron.core.models.engram.engram import Engram
from megatron.core.models.engram.fused_dense import (
    HAVE_TRITON,
    FusedEngramDense,
    fused_engram_dense,
)

from ._test_utils import make_module_config, make_pg_collection

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_TRITON), reason="needs CUDA and Triton"
)

_EOS = 7


def _build(hidden_size, num_streams, memory_dim, heads, order, kernel_size, vocab, dtype):
    config = make_module_config(num_streams=num_streams, dtype=dtype)
    config.hidden_size = hidden_size
    config.layernorm_epsilon = 1e-6
    config.init_method = lambda tensor: torch.nn.init.normal_(tensor, mean=0.0, std=0.02)
    engram_config = EngramConfig(
        global_vocab_sizes=(vocab,) * (order - 1),
        layer_ids=(1,),
        max_ngram_order=order,
        num_hash_heads=heads,
        memory_dim=memory_dim,
        kernel_size=kernel_size,
        hash_seed=1234,
        boundary_token_id=_EOS,
        variant="qwen",
        unigram_vocab_size=vocab,
    )
    torch.manual_seed(1234)
    module = Engram(
        config=config,
        engram_config=engram_config,
        layer_number=1,
        pg_collection=make_pg_collection(),
    )
    with torch.no_grad():
        # Non-trivial convolution and gammas; production initializes the convolution to zero.
        module.short_conv.weight.normal_(mean=0.0, std=0.3)
        for norm in (module.key_norm, module.query_norm, module.conv_norm):
            norm.weight.normal_(mean=0.0, std=0.2)
        for table in module.embedding.tables:
            table.weight.normal_(mean=0.0, std=1.0)
    return module.cuda()


def _run(module, impl, hidden, tokens, grad):
    module.dense_impl = impl
    for parameter in module.parameters():
        parameter.grad = None
    hidden = hidden.detach().clone().requires_grad_(True)
    output = module(hidden, tokens)
    output.backward(grad)
    grads = {name: p.grad.detach().clone() for name, p in module.named_parameters()}
    grads["hidden"] = hidden.grad.detach().clone()
    return output.detach().clone(), grads


def _rel_l2(actual, expected):
    actual, expected = actual.double(), expected.double()
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-30)).item()


_SHAPES = [
    # name, seq, batch, hidden, streams, memory_dim, heads, order, kernel_size
    pytest.param(4096, 1, 2560, 4, 1280, 8, 3, 4, id="proxy-shape"),
    pytest.param(37, 3, 96, 2, 32, 2, 3, 4, id="odd-batched"),
    pytest.param(5, 2, 64, 4, 32, 2, 3, 4, id="shorter-than-history"),
    pytest.param(50, 2, 128, 1, 64, 4, 2, 3, id="single-stream-k3-d2"),
]


@pytest.mark.parametrize("zero_centered", [True, False])
@pytest.mark.parametrize("seq,batch,hidden_size,streams,memory_dim,heads,order,kernel", _SHAPES)
def test_fused_matches_reference(
    seq, batch, hidden_size, streams, memory_dim, heads, order, kernel, zero_centered
):
    module = _build(hidden_size, streams, memory_dim, heads, order, kernel, 1000, torch.bfloat16)
    for norm in (module.key_norm, module.query_norm, module.conv_norm):
        norm.zero_centered = zero_centered
    generator = torch.Generator(device="cuda").manual_seed(7)
    tokens = torch.randint(0, 1000, (batch, seq), device="cuda", generator=generator)
    width = streams * hidden_size
    hidden = torch.randn(seq, batch, width, device="cuda", generator=generator)
    grad = torch.randn(seq, batch, width, device="cuda", generator=generator)
    hidden, grad = hidden.bfloat16(), grad.bfloat16()

    assert module._use_fused_dense(hidden)
    out_ref, grads_ref = _run(module, "reference", hidden, tokens, grad)
    out_fused, grads_fused = _run(module, "fused", hidden, tokens, grad)
    # Noise floor: flip the last mantissa bit of 2 % of the input elements.
    flip = torch.rand(hidden.shape, device="cuda", generator=generator) < 0.02
    hidden_flipped = (hidden.view(torch.int16) ^ flip.to(torch.int16)).view(torch.bfloat16)
    _, grads_flip = _run(module, "reference", hidden_flipped, tokens, grad)

    # fp64 evaluation of the same module, as the accuracy anchor for both bf16 paths.
    module64 = module.double()
    out_64, grads_64 = _run(module64, "reference", hidden.double(), tokens, grad.double())
    _, grads_64_flip = _run(module64, "reference", hidden_flipped.double(), tokens, grad.double())
    module.bfloat16()

    report = {
        "output": (
            _rel_l2(out_fused, out_ref),
            0.0,
            0.0,
            _rel_l2(out_ref, out_64),
            _rel_l2(out_fused, out_64),
        )
    }
    for name in grads_ref:
        report[name] = (
            _rel_l2(grads_fused[name], grads_ref[name]),
            _rel_l2(grads_flip[name], grads_ref[name]),
            _rel_l2(grads_64_flip[name], grads_64[name]),
            _rel_l2(grads_ref[name], grads_64[name]),
            _rel_l2(grads_fused[name], grads_64[name]),
        )
    print(
        "\nname: fused-vs-ref, ref-floor(1ulp input flip), fp64-floor, ref-vs-fp64, fused-vs-fp64"
    )
    for name, values in report.items():
        print(f"  {name}: " + ", ".join(f"{v:.3e}" for v in values))
    well_conditioned = (
        "output",
        "value_projection.weight",
        "value_projection.bias",
        "conv_norm.weight",
        "short_conv.weight",
    )
    for name, (vs_ref, _, _, ref_err, fused_err) in report.items():
        # Score-chain gradients: a sanity cap only; the fp64 criterion below is the bar.
        bar = 1e-3 if name in well_conditioned else 2e-2
        assert vs_ref <= bar, f"{name}: fused vs reference rel-L2 {vs_ref:.3e} > {bar:.3e}"
        assert fused_err <= 1.25 * ref_err + 1e-4, (
            f"{name}: fused is further from fp64 ({fused_err:.3e}) than the reference "
            f"({ref_err:.3e})"
        )


def _reference_intermediates(module, key, hidden, value, grad):
    """Reference forward/backward with the score-chain intermediates retained."""
    seq, batch, width = hidden.shape
    hidden_size = module.hidden_size
    shape = (seq, batch, width // hidden_size, hidden_size)
    key = key.detach().clone().requires_grad_(True)
    hidden = hidden.detach().clone().requires_grad_(True)
    key_n = module.key_norm(key).view(shape)
    query_n = module.query_norm(hidden).view(shape)
    dot = (key_n * query_n).sum(dim=-1)
    dot.retain_grad()
    score = dot / math.sqrt(hidden_size)
    gate = (score.abs().clamp_min(1e-6).sqrt() * score.sign()).sigmoid()
    gate.retain_grad()
    gated = gate.unsqueeze(-1) * value.unsqueeze(2)
    output = gated + module._short_convolution(gated, None)
    output.reshape(seq, batch, width).backward(grad)
    for parameter in module.parameters():
        parameter.grad = None
    return dict(score=score.detach(), gate=gate.detach(), grad_gate=gate.grad, grad_sum=dot.grad)


def _mismatch(name, fused, reference):
    fused, reference = fused.reshape(-1).float(), reference.reshape(-1).float()
    differ = (fused != reference).float().mean().item()
    print(f"  {name}: differing fraction {differ:.3e}, rel-L2 {_rel_l2(fused, reference):.3e}")


@pytest.mark.parametrize("chain", ["torch", "torch+gate-grad", "triton"])
@pytest.mark.parametrize("zero_centered", [True, False])
@pytest.mark.parametrize("seq,batch", [(1024, 1), (300, 2)])
def test_fused_matches_reference_well_conditioned(seq, batch, zero_centered, chain, monkeypatch):
    """Same projections fed to both paths, with every score bounded away from zero.

    ``chain`` selects where the scalar score-chain backward runs (``_SCALAR_CHAIN``) and
    where the gate gradient is summed (``_GATE_GRAD``). The default ("torch") must meet the
    bar; the other variants are reported (xfail) for comparison.
    """
    from megatron.core.models.engram import fused_dense

    monkeypatch.setattr(fused_dense, "_SCALAR_CHAIN", chain.split("+")[0])
    monkeypatch.setattr(fused_dense, "_GATE_GRAD", "torch" if "gate-grad" in chain else "triton")
    hidden_size, streams = 2560, 4
    module = _build(hidden_size, streams, 1280, 8, 3, 4, 1000, torch.bfloat16)
    with torch.no_grad():
        module.query_norm.weight.copy_(module.key_norm.weight)
    for norm in (module.key_norm, module.query_norm, module.conv_norm):
        norm.zero_centered = zero_centered
    generator = torch.Generator(device="cuda").manual_seed(11)
    width = streams * hidden_size

    def rand(*shape):
        return torch.randn(*shape, device="cuda", generator=generator)

    key = rand(seq, batch, streams, hidden_size)
    # Query = +/- a fixed fraction of the key plus noise: with equal key/query gammas the score
    # concentrates around +/- several standard deviations, far from zero.
    sign = torch.where(rand(seq, batch, streams, 1) > 0, 1.0, -1.0)
    hidden = sign * 0.25 * key + rand(seq, batch, streams, hidden_size)
    key, hidden = key.reshape(seq, batch, width), hidden.reshape(seq, batch, width)
    value = rand(seq, batch, hidden_size)
    grad = rand(seq, batch, width)
    key, hidden, value, grad = (t.bfloat16() for t in (key, hidden, value, grad))

    def run(impl):
        for parameter in module.parameters():
            parameter.grad = None
        leaves = [t.detach().clone().requires_grad_(True) for t in (key, hidden, value)]
        if impl == "reference":
            out = module._dense_reference(None, leaves[1], None, projections=(leaves[0], leaves[2]))
        else:
            out = fused_engram_dense(
                leaves[0],
                leaves[1],
                leaves[2],
                module.key_norm.weight,
                module.query_norm.weight,
                module.conv_norm.weight,
                module.short_conv.weight,
                streams,
                module.key_norm.eps,
                zero_centered,
                module.engram_config.max_ngram_order,
            )
        out.backward(grad)
        grads = {n: leaf.grad.clone() for n, leaf in zip(("key", "hidden", "value"), leaves)}
        for name in ("key_norm", "query_norm", "conv_norm"):
            grads[name] = getattr(module, name).weight.grad.clone()
        grads["short_conv"] = module.short_conv.weight.grad.clone()
        return out.detach(), grads

    with torch.no_grad():
        kn = module.key_norm(key).float().view(seq, batch, streams, hidden_size)
        qn = module.query_norm(hidden).float().view(seq, batch, streams, hidden_size)
        score = (kn * qn).sum(-1) / hidden_size**0.5
    min_score = score.abs().min().item()
    out_ref, grads_ref = run("reference")
    debug = {}
    monkeypatch.setattr(fused_dense, "_DEBUG", debug)
    out_fused, grads_fused = run("fused")
    monkeypatch.setattr(fused_dense, "_DEBUG", None)
    inter = _reference_intermediates(module, key, hidden, value, grad)
    print(f"\n[{chain}] fused vs reference score-chain intermediates:")
    shape = inter["score"].shape
    _mismatch("score", debug["stats"][1].view(shape), inter["score"])
    _mismatch("gate", debug["stats"][0].view(shape), inter["gate"])
    _mismatch("grad_gate", debug["grad_gate"].view(shape), inter["grad_gate"])
    _mismatch("grad_sum (torch chain)", debug["grad_sum"].view(shape), inter["grad_sum"])
    report = {"output": _rel_l2(out_fused, out_ref)}
    report.update({n: _rel_l2(grads_fused[n], grads_ref[n]) for n in grads_ref})
    print(f"\nmin |score| {min_score:.3e}; fused-vs-ref rel-L2:")
    for name, value_ in report.items():
        print(f"  {name}: {value_:.3e}")
    assert min_score > 0.05, f"test setup: scores not bounded away from zero ({min_score})"
    worst = max(report, key=report.get)
    if chain != "torch" and report[worst] > 1e-3:
        pytest.xfail(f"variant {chain}: {worst} rel-L2 {report[worst]:.3e}")
    for name, value_ in report.items():
        assert value_ <= 1e-3, f"{name}: fused vs reference rel-L2 {value_:.3e}"


def _fused_inputs(seq, batch, hidden_size, streams, kernel, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    width = streams * hidden_size

    def rand(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda", generator=generator) * scale).bfloat16()

    return dict(
        key=rand(seq, batch, width),
        hidden=rand(seq, batch, width),
        value=rand(seq, batch, hidden_size),
        key_gamma=rand(width, scale=0.2),
        query_gamma=rand(width, scale=0.2),
        conv_gamma=rand(width, scale=0.2),
        conv_weight=rand(width, 1, kernel, scale=0.3),
    )


def _fused_step(inputs, grad, dilation=3):
    leaves = {name: t.detach().clone().requires_grad_(True) for name, t in inputs.items()}
    output = FusedEngramDense.apply(
        leaves["key"],
        leaves["hidden"],
        leaves["value"],
        leaves["key_gamma"],
        leaves["query_gamma"],
        leaves["conv_gamma"],
        leaves["conv_weight"],
        inputs["key"].shape[-1] // inputs["value"].shape[-1],
        1e-6,
        True,
        dilation,
    )
    output.backward(grad)
    return output.detach(), {name: leaf.grad for name, leaf in leaves.items()}


def test_fused_is_bitwise_repeatable():
    inputs = _fused_inputs(512, 2, 256, 4, 4)
    grad = torch.randn_like(inputs["key"])
    out_a, grads_a = _fused_step(inputs, grad)
    out_b, grads_b = _fused_step(inputs, grad)
    assert torch.equal(out_a, out_b)
    for name in grads_a:
        assert torch.equal(grads_a[name], grads_b[name]), name


def test_fused_is_cuda_graph_capturable():
    inputs = _fused_inputs(256, 1, 256, 4, 4, seed=3)
    grad = torch.randn_like(inputs["key"])
    eager_out, eager_grads = _fused_step(inputs, grad)

    static = {name: t.detach().clone().requires_grad_(True) for name, t in inputs.items()}
    static_grad = grad.clone()
    num_streams = inputs["key"].shape[-1] // inputs["value"].shape[-1]

    def step():
        output = FusedEngramDense.apply(
            static["key"],
            static["hidden"],
            static["value"],
            static["key_gamma"],
            static["query_gamma"],
            static["conv_gamma"],
            static["conv_weight"],
            num_streams,
            1e-6,
            True,
            3,
        )
        grads = torch.autograd.grad(output, list(static.values()), static_grad)
        return output, grads

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()  # warm-up: compiles the kernels outside the capture
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out, graph_grads = step()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(graph_out, eager_out)
    for name, value in zip(static, graph_grads):
        assert torch.equal(value, eager_grads[name]), name


def test_token_count_does_not_recompile():
    from megatron.core.models.engram import fused_dense

    _fused_step(_fused_inputs(64, 1, 256, 4, 4), torch.randn(64, 1, 1024, device="cuda").bfloat16())
    kernels = [
        fused_dense._gate_fwd_kernel,
        fused_dense._conv_fwd_kernel,
        fused_dense._conv_bwd_kernel,
        fused_dense._cnorm_dot_kernel,
        fused_dense._gval_kernel,
        fused_dense._value_bwd_kernel,
        fused_dense._kq_bwd_kernel,
    ]
    device = torch.cuda.current_device()

    def count(kernel):
        if hasattr(kernel, "device_caches"):  # Triton >= 3.3: (kernel_cache, ...) per device
            return len(kernel.device_caches[device][0])
        if hasattr(kernel, "cache"):
            return len(kernel.cache.get(device, {}))
        pytest.skip("this Triton version does not expose the per-kernel compile cache")

    def counts():
        return [count(kernel) for kernel in kernels]

    before = counts()
    for seq in (77, 128, 1000, 4097):
        _fused_step(
            _fused_inputs(seq, 1, 256, 4, 4), torch.randn(seq, 1, 1024, device="cuda").bfloat16()
        )
    assert counts() == before
