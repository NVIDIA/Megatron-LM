# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for the sink-softmax post-correction of dynamic and static flash-decode attention.

Call sites: ``Attention._apply_sink_softmax_correction_varlen`` and
``Attention._apply_sink_softmax_correction_bshd`` (megatron/core/transformer/attention.py). Every
paged FlashAttention (FA2/FA3/FA4), FlashMLA and static ``flash_decode`` call of a layer with a
sink (``softmax_type`` off-by-one or learnable) asks the kernel for its log-sum-exp and corrects
the vanilla output with it, using the per-head logits from
``Attention._get_inference_softmax_offset``. The corrections through the real kernels' LSE
layouts are checked end to end by the sink cases of ``test_flash_attn_paged.py``,
``test_flash_mla_decode.py``, ``test_static_flash_decode.py`` and ``test_mla_dynamic_path.py``;
this module checks the correction itself on every input it can be given, and the offset source.

Contract (from the sink softmax's definition and the call sites' tensor layouts):

* Sink softmax gives head ``h`` an extra logit ``sink_h`` that joins the softmax denominator and
  contributes no value: ``p_j = exp(s_j) / (exp(sink_h) + sum_k exp(s_k))``. From a kernel's
  vanilla output ``o`` and log-sum-exp ``lse = log sum_k exp(s_k)``, the sink output is
  ``o * exp(lse) / (exp(lse) + exp(sink_h))``.
* Layouts: varlen ``o [T, H, D]`` with ``lse [H, T]``; bshd ``o [B, S, H, D]`` with
  ``lse [B, H, S]``; sink ``[H]`` (any float dtype, on any device the call can move it from).
* A row whose kernel attended no key (``lse = -inf``) has only the sink in its denominator: the
  output is 0. A row whose LSE is NaN (a row the kernel did not compute) is returned unchanged,
  as the methods document; ``lse = +inf`` leaves the row unchanged (the limit).
* Computed in fp32 and rounded once to the dtype of ``o``; shape and dtype kept; ``o``, ``lse``
  and the sink are read only.
* Offset source: ``vanilla`` gives no correction (None); ``off-by-one`` the logit 0 for each of
  the layer's heads (an extra 1 in the denominator); ``learnable`` the layer's learned per-head
  logits, ``[num_attention_heads_per_partition]``; with the local and the TE core attention.

Reference: fp64 ``o * exp(lse - logaddexp(lse, sink))``. Tolerance class: floating point, per
(token, head) row for bf16/fp16 outputs and over the whole tensor for fp32 outputs (see
``_check``), bounded by the same formula in fp32 with torch's ops (not the methods' sigmoid
form), rounded once to the dtype. Special rows: exact (zero, or bitwise unchanged).

Variants: both layouts; bf16, fp16 and fp32 outputs; sink logits in bf16 (the layers'
``params_dtype``) and fp32 (TE's); off-by-one (0), learnable (random per head), and saturating
logits (+-30 against the LSE); LSE values over several units (shapes with ``T``, ``B``, ``S``,
``H`` and ``D`` all different, so a transposed layout cannot pass); -inf, NaN and +inf LSE
rows; a 4096 x 32-head x 128 production-size varlen case.

Hardware: H100 (torch ops on the device the attention runs on).
"""

import pytest
import torch

from megatron.core.transformer.attention import Attention
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.test_utilities import Utils

H100 = ou.requires_sm(ou.HOPPER)


def sink_reference(o, lse, sink):
    """fp64 ``o * exp(lse - logaddexp(lse, sink))``; ``lse`` and ``sink`` broadcast to ``o``."""
    return o * torch.exp(lse - torch.logaddexp(lse, sink))


def sink_trusted(o, lse, sink):
    """The same formula with torch's fp32 ops, rounded once to o's dtype."""
    lse, sink = lse.float(), sink.float()
    return (o.float() * torch.exp(lse - torch.logaddexp(lse, sink))).to(o.dtype)


def _sink(kind: str, heads: int, dtype: torch.dtype, g) -> torch.Tensor:
    if kind == "off_by_one":
        return torch.zeros(heads, device="cuda", dtype=dtype)
    if kind == "saturating":
        values = torch.tensor([30.0, -30.0], device="cuda").repeat((heads + 1) // 2)[:heads]
        return values.to(dtype)
    return (torch.randn(heads, device="cuda", generator=g) * 2 + 1).to(dtype)


def _lse(shape, g) -> torch.Tensor:
    """Log-sum-exps as kernels return them (fp32), spread over several units."""
    return torch.randn(shape, device="cuda", generator=g) * 3 + 2


def _varlen(o, lse, sink):
    return Attention._apply_sink_softmax_correction_varlen(o, lse, sink)


def _bshd(o, lse, sink):
    return Attention._apply_sink_softmax_correction_bshd(o, lse, sink)


def _aligned(layout, lse, sink, heads):
    """lse and sink broadcast to o's layout: [T, H, 1] or [B, S, H, 1]."""
    if layout == "varlen":
        return lse.transpose(0, 1)[..., None], sink.reshape(1, heads, 1)
    return lse.permute(0, 2, 1)[..., None], sink.reshape(1, 1, heads, 1)


def _case(layout, dtype, sink_kind, sink_dtype, shape):
    g = ou.make_generator(device="cuda")
    heads = shape[-2]
    o = torch.randn(shape, device="cuda", generator=g).to(dtype)
    if layout == "varlen":
        lse = _lse((heads, shape[0]), g)
    else:
        lse = _lse((shape[0], heads, shape[1]), g)
    sink = _sink(sink_kind, heads, sink_dtype, g)
    return o, lse, sink


def _check(layout, o, lse, sink, what):
    heads = o.shape[-2]
    before = (o.clone(), lse.clone(), sink.clone())
    fn = _varlen if layout == "varlen" else _bshd
    out = fn(o, lse, sink)
    torch.cuda.synchronize()
    for t, b, name in zip((o, lse, sink), before, ("output", "lse", "sink")):
        ou.assert_exact(t, b, what=f"{what}: {name} (read-only)")
    assert out.shape == o.shape and out.dtype == o.dtype, f"{what}: {out.shape} {out.dtype}"
    lse_a, sink_a = _aligned(layout, lse, sink, heads)
    ref = ou.fp64_reference(sink_reference, o, lse_a, sink_a)
    trusted = sink_trusted(o, lse_a, sink_a)
    rows = lambda t: t.reshape(-1, t.shape[-1])  # noqa: E731  (one row per token and head)
    # 16-bit outputs: the final rounding dominates, bound each (token, head) row. fp32 outputs:
    # a row's error is the rounding of its single fp32 factor (1 to 3 ulps for either formula),
    # so per-row ratios vary by chance (up to 15x on CPU); bound the whole tensor instead.
    dim = None if o.dtype == torch.float32 else -1
    ou.assert_close_to_reference(rows(out), rows(ref), rows(trusted), dim=dim, what=what)
    return out


SHAPES = {"varlen": (37, 6, 40), "bshd": (3, 5, 6, 40)}  # T/B, S, H, D all different


def _cases():
    params = []
    for case in ou.pairwise_cases(
        {
            "layout": ["varlen", "bshd"],
            "dtype": [torch.bfloat16, torch.float16, torch.float32],
            "sink": ["off_by_one", "learnable", "saturating"],
            "sink_dtype": [torch.bfloat16, torch.float32],
        }
    ):
        name = lambda d: str(d).replace("torch.", "")  # noqa: E731
        case_id = (
            f"{case['layout']}-{name(case['dtype'])}-{case['sink']}-sink_{name(case['sink_dtype'])}"
        )
        params.append(pytest.param(case, id=case_id, marks=H100))
    return params


@pytest.mark.parametrize("case", _cases())
def test_correction_matches_reference(case):
    """Both layouts, every output dtype, off-by-one, learnable and saturating sink logits."""
    layout = case["layout"]
    o, lse, sink = _case(layout, case["dtype"], case["sink"], case["sink_dtype"], SHAPES[layout])
    _check(layout, o, lse, sink, f"sink correction {case}")


@pytest.mark.parametrize("layout", ["varlen", "bshd"])
@H100
def test_rows_without_keys_or_lse(layout):
    """A row with no attended key (lse -inf) becomes 0; a row with a NaN or +inf LSE is
    returned unchanged; the other rows are corrected as usual."""
    o, lse, sink = _case(layout, torch.bfloat16, "learnable", torch.bfloat16, SHAPES[layout])
    special = {"-inf": float("-inf"), "nan": float("nan"), "+inf": float("inf")}
    if layout == "varlen":
        cells = {"-inf": (1, 4), "nan": (0, 9), "+inf": (5, 20)}  # (head, token)
    else:
        cells = {"-inf": (0, 1, 2), "nan": (2, 4, 0), "+inf": (1, 5, 3)}  # (batch, head, s)
    for kind, cell in cells.items():
        lse[cell] = special[kind]
    fn = _varlen if layout == "varlen" else _bshd
    out = fn(o, lse, sink)
    torch.cuda.synchronize()

    def row(t, cell):  # the output row of an LSE cell
        return t[cell[1], cell[0]] if layout == "varlen" else t[cell[0], cell[2], cell[1]]

    assert bool((row(out, cells["-inf"]) == 0).all()), "lse -inf: row must be 0 (sink only)"
    ou.assert_exact(row(out, cells["nan"]), row(o, cells["nan"]), what="NaN lse row unchanged")
    ou.assert_exact(row(out, cells["+inf"]), row(o, cells["+inf"]), what="+inf lse row unchanged")
    finite = torch.isfinite(lse)
    lse_a, sink_a = _aligned(layout, torch.where(finite, lse, 0.0), sink, o.shape[-2])
    keep = _aligned(layout, finite, sink, o.shape[-2])[0][..., 0]
    ref = ou.fp64_reference(sink_reference, o, lse_a, sink_a)[keep]
    trusted = sink_trusted(o, lse_a, sink_a)[keep]
    ou.assert_close_to_reference(out[keep], ref, trusted, dim=-1, what="finite rows")


@H100
def test_production_size():
    """A 4096-token varlen output of 32 heads of 128 (bf16, learnable bf16 sink)."""
    o, lse, sink = _case("varlen", torch.bfloat16, "learnable", torch.bfloat16, (4096, 32, 128))
    _check("varlen", o, lse, sink, "production-size varlen correction")


@pytest.mark.sanitizer
@H100
def test_small_case_for_sanitizer():
    """Both layouts on small shapes (compute-sanitizer memcheck case)."""
    for layout in ("varlen", "bshd"):
        o, lse, sink = _case(layout, torch.bfloat16, "learnable", torch.float32, SHAPES[layout])
        _check(layout, o, lse, sink, f"sanitizer {layout}")


# ---------------------------------------------------------------------------------------------
# Offset source
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def model_parallel():
    Utils.initialize_model_parallel(1, 1)
    yield
    Utils.destroy_model_parallel()


def _layer(spec: str, softmax_type: str, heads: int = 8):
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_submodules,
        get_gpt_layer_with_transformer_engine_submodules,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.attention import SelfAttention
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.transformer_config import TransformerConfig

    model_parallel_cuda_manual_seed(ou.DEFAULT_SEED)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=heads * 64,
        num_attention_heads=heads,
        num_query_groups=2,
        kv_channels=64,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        softmax_type=softmax_type,
    )
    builder = (
        get_gpt_layer_with_transformer_engine_submodules
        if spec == "te"
        else get_gpt_layer_local_submodules
    )
    submodules = builder().self_attention.submodules
    layer = SelfAttention(config, submodules, layer_number=1, attn_mask_type=AttnMaskType.causal)
    return layer.cuda().eval()


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize("softmax_type", ["vanilla", "off-by-one", "learnable"])
@pytest.mark.parametrize("spec", ["te", "local"])
@H100
def test_offset_source(spec, softmax_type):
    """The sink logits the dynamic and static flash-decode paths hand the correction: none for
    vanilla, 0 per head for off-by-one, the learned logits (as they change) for learnable."""
    layer = _layer(spec, softmax_type)
    heads = layer.num_attention_heads_per_partition
    offset = layer._get_inference_softmax_offset()
    if softmax_type == "vanilla":
        assert offset is None, f"vanilla softmax gave sink logits {offset}"
        return
    assert offset is not None and offset.shape == (heads,), f"sink logits {offset}"
    if softmax_type == "off-by-one":
        assert bool((offset == 0).all()), f"off-by-one sink logits {offset}"
        return
    learned = layer.core_attention.softmax_offset
    g = ou.make_generator(seed=ou.DEFAULT_SEED + 5, device="cuda")
    with torch.no_grad():
        learned.copy_(torch.randn(learned.shape, device="cuda", generator=g).to(learned.dtype))
    offset = layer._get_inference_softmax_offset()
    ou.assert_exact(offset.detach(), learned.detach(), what="learnable sink logits")


# ---------------------------------------------------------------------------------------------
# CPU check of the reference itself (no kernel)
# ---------------------------------------------------------------------------------------------


def test_reference_is_the_softmax_with_an_extra_logit():
    g = torch.Generator().manual_seed(0)
    scores = torch.randn(4, 7, generator=g, dtype=torch.float64)
    v = torch.randn(7, 3, generator=g, dtype=torch.float64)
    sink = torch.tensor([0.0, 1.5, -2.0, 30.0], dtype=torch.float64)
    vanilla = torch.softmax(scores, -1) @ v
    lse = torch.logsumexp(scores, -1)
    with_sink = torch.softmax(torch.cat([scores, sink[:, None]], -1), -1)[:, :-1] @ v
    got = sink_reference(vanilla, lse[:, None], sink[:, None])
    torch.testing.assert_close(got, with_sink)
    # No attended key: only the sink in the denominator.
    no_keys = torch.full((4, 1), float("-inf"), dtype=torch.float64)
    empty = sink_reference(vanilla, no_keys, torch.zeros(4, 1, dtype=torch.float64))
    assert bool((empty == 0).all())
