# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for the LM head's fp32 logits in inference (main #8235, `a5f5036346`).

Kernel and call sites: `ColumnParallelLinear.forward(output_dtype=torch.float32)` ->
`_linear_forward` (`megatron/core/tensor_parallel/layers.py`), which runs Transformer Engine's
`general_gemm(weight, x, out_dtype=fp32, layout="TN", out=view)` (cuBLASLt) through Megatron's
`te_general_gemm`; selected under `InferenceMode` by `GPTModel._postprocess`, `HybridModel`'s
forward and `LanguageModule.compute_mtp_single_step` when `lm_head_can_emit(output_layer,
torch.float32)` (`megatron/core/inference/utils.py`) holds. Otherwise (batch-invariant mode, no
TE, a head that is not a `ColumnParallelLinear`) the head runs in the parameter dtype and the
model upcasts with `.float()`. Every GPT and Hybrid model under `InferenceMode` with TE takes this
path by default since 2026-11-06.

Contract (written from #8235's description, the call sites and the model math before reading
`_linear_forward`):
- Under `InferenceMode` the logits a GPT / Hybrid model returns, and the MTP logits of
  `compute_mtp_single_step`, are fp32 `[B, S, V]` (MTP: `[N, 1, V]`).
- When the head can emit fp32: logits = fp32(h @ W^T), the product of the bf16 hidden state `h`
  (the head's input) and the bf16 weight `W` (the head's own, or the tied embedding weight)
  accumulated in fp32 and never rounded to bf16. MuP's `mup_output_mult` multiplies the fp32
  logits. Training (outside `InferenceMode`) keeps bf16 logits.
- When it cannot (batch-invariant mode): logits are the parameter-dtype head output, MuP-scaled
  in that dtype, then upcast: bitwise `training_logits.float()` ("batch-invariant mode
  guarantees logits bitwise-identical to the training forward", `lm_head_can_emit`).
- Tensor parallelism: each rank's GEMM gives its vocabulary shard `h @ W_r^T`;
  `runtime_gather_output=True` gathers the full `[*, V]` (NCCL, or the NVLS multimem all-gather
  for `transformer_impl="inference_optimized"`); with sequence parallelism the head first gathers
  the hidden states along the sequence.
- Every output element is written (the GEMM writes into the returned allocation), inputs are
  read only, M = 0 tokens gives an empty result, and a CUDA graph of the head replays the eager
  bits.
- `lm_head_can_emit(head, dtype)`: True for the parameter dtype; for another dtype only for a
  `ColumnParallelLinear` with TE's `general_gemm` outside batch-invariant mode. Asking
  `_linear_forward` for a mixed-precision output without TE raises RuntimeError.

Reference: fp64 `h @ W^T` of the exact bf16 operands (computed in column chunks of W to bound
memory). Trusted implementation: aten's own bf16 GEMM with an fp32 output on the same device
(`torch.mm(h, W.t(), out_dtype=torch.float32)`, cuBLAS; independent of TE and Megatron).
Tolerance class: floating point, per row of logits (default factor 2) for random operands; exact
for integer-valued operands (`ou.dyadic_tensor` with |m| < 8: every partial sum is an integer
below 2**24, so fp32 accumulation in any order is exact, while bf16 output would round any logit
above 256); bitwise for batch-invariant mode, MuP scaling, the returned layout and CUDA-graph
replay. A spy on `te_general_gemm` records the dispatch and fills the output buffer with NaN
just before the GEMM, so an element the GEMM leaves unwritten fails the fp64 comparison.

Structure (read after the contract): `_linear_forward` takes the TE path only when
`output_dtype` differs from the input dtype; it flattens the input to `[M, H]`, allocates the
final-shaped fp32 output and has TE write through a 2-D view; trainable weights go through
`LinearWithGradAccumulationAndAsyncCommunication` (which gathers sequence-parallel inputs into
the global memory buffer), frozen ones through `LinearWithFrozenWeight`. cuBLASLt chooses tiles
and split-K by heuristics, so sizes are taken around 64/128/256 multiples: V in 1..257 edges,
odd vocabularies (50257), 32000 and the production vocabularies 128256, 151936, 152064 and
262144; H 64, 2880 (not a multiple of 128) and 4096..8192; M 0, 1, 7..9, 63..65, 127..129, 1024
and 8192; 2-D, `[S, 1, H]` and row-strided inputs; the head's own and a runtime (tied) weight;
frozen and trainable weights. Model level: GPT and Hybrid (`"*-"`), tied and untied, MuP on and
off, training vs `InferenceMode`; batch-invariant mode on the te_native, triton and deepgemm
backends; the MTP step through `LanguageModule.compute_mtp_single_step`; TP 2 with and without
`runtime_gather_output`, sequence parallelism and the inference-optimized NVLS gather (2 GPUs);
the dynamic-inference prefill of a batch-invariant hybrid MoE model against its training forward
(4 GPUs, the configuration of `test_hybrid_moe.py::test_batch_invariant_prefill_matches_full_forward`).

Main vs open PRs: every case is expected to pass on main.

Hardware: H100 and GB200 (1 GPU); the TP 2 cases need 2 NVLS-connected H100s, the hybrid MoE
prefill case 4 (H100 extra tasks).
"""

import contextlib
import functools
import types

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.inference.utils import InferenceMode, lm_head_can_emit
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.tensor_parallel import layers as tp_layers
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.test_utilities import Utils

pytestmark = [pytest.mark.launch_on_gb200, ou.requires_sm(ou.HOPPER, ou.BLACKWELL)]

_DEVICE = "cuda"
FP32 = torch.float32
BF16 = torch.bfloat16
# Rows of logits compared per chunk: bounds the fp64 reference and the trusted copy to ~1 GiB.
_REF_ELEMS = 2**27
# Columns of W converted to fp64 at a time for the reference.
_REF_W_ROWS = 16384


@pytest.fixture(autouse=True)
def _modes_off():
    """Batch-invariant mode and InferenceMode are process-wide: each test starts and ends with
    both off."""
    assert not bik.is_batch_invariant_mode_enabled(), "batch-invariant mode leaked into the test"
    assert not InferenceMode.is_active(), "InferenceMode leaked into the test"
    yield
    left_active = InferenceMode.is_active()
    InferenceMode.unset_active()
    assert not bik.is_batch_invariant_mode_enabled(), "the test left batch-invariant mode on"
    assert not left_active, "the test left InferenceMode active"


@pytest.fixture
def parallel(request):
    tp = getattr(request, "param", 1)
    if Utils.world_size < tp:
        pytest.skip(f"needs {tp} ranks (run with nproc >= {tp}); the manifest selects it so")
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp)
    model_parallel_cuda_manual_seed(123, force_reset_rng=True)
    SymmetricMemoryManager.destroy("tp")
    yield tp
    SymmetricMemoryManager.destroy("tp")
    Utils.destroy_model_parallel()


# ---------------------------------------------------------------------------------------------
# Inputs, reference, spies
# ---------------------------------------------------------------------------------------------


def bf16(shape, seed, scale=1.0):
    """The same values on every rank (CPU generator)."""
    g = ou.make_generator(seed)
    return (torch.randn(*shape, generator=g) * scale).to(BF16).to(_DEVICE)


def integers(shape, seed):
    """bf16 integers in (-8, 8): a dot product of length k < 2**18 is exact in fp32."""
    g = ou.make_generator(seed)
    return ou.dyadic_tensor(shape, mantissa_bits=3, exponents=(0, 0), generator=g).to(_DEVICE)


def operands(tokens, hidden, vocab, kind, seed):
    if kind == "integer":
        return integers((tokens, hidden), seed), integers((vocab, hidden), seed + 1)
    return bf16((tokens, hidden), seed), bf16((vocab, hidden), seed + 1, scale=hidden**-0.5)


def fp64_logits(x2d, w):
    """fp64 x @ W^T of the exact bf16 operands, built from column blocks of W."""
    blocks = [
        ou.fp64_reference(lambda a, b: a @ b.t(), x2d, w[lo : lo + _REF_W_ROWS])
        for lo in range(0, w.shape[0], _REF_W_ROWS)
    ]
    return torch.cat(blocks, dim=-1)


def check_logits(out2d, x2d, w, what, *, exact=False):
    """`out2d` [M, V] (fp32) against fp64 x @ W^T: exact for integer operands, else per row
    within twice aten's bf16 -> fp32 GEMM error."""
    assert out2d.dtype == FP32, f"{what}: logits are {out2d.dtype}, expected fp32"
    assert out2d.shape == (x2d.shape[0], w.shape[0]), (what, tuple(out2d.shape))
    rows = max(1, _REF_ELEMS // max(1, w.shape[0]))
    for lo in range(0, x2d.shape[0], rows):
        sl = slice(lo, lo + rows)
        ref = fp64_logits(x2d[sl], w)
        if exact:
            ou.assert_exact(out2d[sl], ref.to(FP32), what=f"{what} rows {lo}..")
            continue
        trusted = torch.mm(x2d[sl], w.t(), out_dtype=FP32)
        ou.assert_close_to_reference(out2d[sl], ref, trusted, dim=-1, what=f"{what} rows {lo}..")


class GemmSpy:
    """Records every `te_general_gemm` call (Megatron's wrapper of TE's `general_gemm`) and fills
    its `out` buffer with NaN just before the real GEMM writes it."""

    def __init__(self, monkeypatch):
        from megatron.core.extensions import transformer_engine as te_ext

        real = te_ext.te_general_gemm
        assert real is not None, "Transformer Engine's general_gemm is required"
        self.calls = []

        def spy(A, B, *args, **kwargs):
            out = kwargs.get("out")
            self.calls.append(
                dict(
                    out_dtype=kwargs.get("out_dtype"),
                    layout=kwargs.get("layout"),
                    a_ptr=A.data_ptr(),
                    out_ptr=None if out is None else out.data_ptr(),
                )
            )
            if out is not None:
                ou.poison_(out)
            return real(A, B, *args, **kwargs)

        monkeypatch.setattr(te_ext, "te_general_gemm", spy)


def make_config(hidden, *, tp=1, sp=False, impl="transformer_engine", **extra):
    kwargs = dict(
        num_layers=1,
        hidden_size=hidden,
        num_attention_heads=8,
        tensor_model_parallel_size=tp,
        sequence_parallel=sp,
        params_dtype=BF16,
        bf16=True,
        add_bias_linear=False,
        normalization="RMSNorm",
        perform_initialization=False,
        use_cpu_initialization=False,
        transformer_impl=impl,
    )
    if impl == "inference_optimized":
        kwargs["inference_disable_triton_nvls_kernels"] = False
    kwargs.update(extra)
    return TransformerConfig(**kwargs)


def build_head(config, w_full, *, frozen=True):
    """A ColumnParallelLinear LM head holding this rank's vocabulary rows of `w_full`."""
    tp = parallel_state.get_tensor_model_parallel_world_size()
    rank = parallel_state.get_tensor_model_parallel_rank()
    vocab, hidden = w_full.shape
    head = tp_layers.ColumnParallelLinear(
        hidden,
        vocab,
        config=config,
        init_method=torch.nn.init.zeros_,
        bias=False,
        gather_output=False,
        skip_bias_add=False,
    )
    with torch.no_grad():
        head.weight.copy_(w_full.chunk(tp, dim=0)[rank])
    head.weight.requires_grad_(not frozen)
    return head


def shape_input(x2d, layout, seed):
    """`x2d` [M, H] as the head's input: 2-D, [S, 1, H] (the dynamic batch), or a view whose
    rows are strided: every other row of a twice-as-tall buffer ("row_strided"), or the left
    half of a twice-as-wide one ("column_slice")."""
    if layout == "2d":
        return x2d.clone()
    if layout == "s1h":
        return x2d.clone().unsqueeze(1)
    tokens, hidden = x2d.shape
    if layout == "row_strided":
        tall = bf16((2 * tokens, hidden), seed + 7)
        tall[::2] = x2d
        view = tall[::2]
    else:
        wide = bf16((tokens, 2 * hidden), seed + 7)
        wide[:, :hidden] = x2d
        view = wide[:, :hidden]
    assert tokens < 2 or not view.is_contiguous()
    return view


# ---------------------------------------------------------------------------------------------
# The head itself: values at structure-derived and production sizes
# ---------------------------------------------------------------------------------------------

VOCABS = ou.boundary_sizes(
    128, 256, multiples=(), degenerate=False, production=(32000, 50257, 128256, 151936)
)
HIDDENS = [64, 2880, 4096]
TOKENS = ou.boundary_sizes(8, 64, 128, multiples=(), degenerate=True, production=(1024,))
HEAD_CASES = ou.pairwise_cases(
    {
        "vocab": VOCABS,
        "hidden": HIDDENS,
        "tokens": TOKENS,
        "layout": ["s1h", "2d"],
        "weight": ["own", "runtime"],
        "frozen": [True, False],
    }
)


def _head_id(c):
    frozen = "frozen" if c["frozen"] else "trainable"
    return f"V{c['vocab']}-H{c['hidden']}-M{c['tokens']}-{c['layout']}-{c['weight']}-{frozen}"


def run_head(head, x, weight, *, gather=None):
    before = x.clone()
    with torch.no_grad(), InferenceMode.active():
        out, out_bias = head(x, weight=weight, runtime_gather_output=gather, output_dtype=FP32)
    ou.assert_exact(x, before, what="the hidden states are read only")
    assert out_bias is None
    return out


@pytest.mark.parametrize("kind", ["random", "integer"])
@pytest.mark.parametrize("case", [pytest.param(c, id=_head_id(c)) for c in HEAD_CASES])
def test_head_emits_fp32_logits(parallel, case, kind, monkeypatch):
    """The fp32 head: dispatch to TE with an fp32 output written into the returned tensor, every
    element written, and values within twice aten's fp32-output GEMM error of fp64 (random) or
    exactly fp64 (integer operands)."""
    vocab, hidden, tokens = case["vocab"], case["hidden"], case["tokens"]
    x2d, w = operands(tokens, hidden, vocab, kind, seed=vocab + hidden + tokens)
    head = build_head(make_config(hidden), w, frozen=case["frozen"])
    runtime_w = None
    if case["weight"] == "runtime":  # the tied embedding weight, passed per call
        runtime_w = w.clone()
        with torch.no_grad():
            head.weight.zero_()
    spy = GemmSpy(monkeypatch)
    x = shape_input(x2d, case["layout"], seed=tokens)
    out = run_head(head, x, runtime_w)
    assert out.shape == (*x.shape[:-1], vocab) and out.is_contiguous(), tuple(out.shape)
    assert len(spy.calls) == 1, f"te_general_gemm ran {len(spy.calls)} times"
    call = spy.calls[0]
    assert (call["out_dtype"], call["layout"]) == (FP32, "TN"), call
    assert call["out_ptr"] == out.data_ptr(), "the GEMM did not write the returned logits"
    assert call["a_ptr"] == (head.weight if runtime_w is None else runtime_w).data_ptr()
    if tokens == 0:
        assert out.numel() == 0
        return
    check_logits(out.reshape(-1, vocab), x2d, w, _head_id(case), exact=kind == "integer")
    if kind == "integer" and hidden >= 1024:
        ref = fp64_logits(x2d[:64], w[:4096])
        assert bool((ref.abs() > 256).any()), "no logit above 256: bf16 output would pass"


STRIDED_CASES = [  # (vocab, hidden, tokens, layout, weight, frozen)
    (129, 4096, 7, "row_strided", "own", True),
    (32000, 2880, 64, "column_slice", "runtime", False),
    (151936, 4096, 129, "row_strided", "own", False),
    (257, 64, 2, "column_slice", "runtime", True),
]


@pytest.mark.parametrize(
    "vocab,hidden,tokens,layout,weight,frozen",
    [pytest.param(*c, id=f"V{c[0]}-H{c[1]}-M{c[2]}-{c[3]}-{c[4]}") for c in STRIDED_CASES],
)
def test_strided_hidden_states(
    parallel, vocab, hidden, tokens, layout, weight, frozen, monkeypatch
):
    """Hidden states whose rows are strided (a view, as `ColumnParallelLinear` accepts and
    `torch.matmul` handles) give the logits of their values, exactly (integer operands).

    Finding O8a-P1 (H100 dfw 21627808): on main the fp32 path reshapes such a view to [M, H]
    without a copy and hands it to TE's `general_gemm`, which reads it as contiguous: every row
    but the first holds another row's logits. Expected to fail on main (expect-fail task); the
    model call sites pass contiguous hidden states (reach in the packet doc)."""
    x2d, w = operands(tokens, hidden, vocab, "integer", seed=vocab + tokens)
    head = build_head(make_config(hidden), w, frozen=frozen)
    runtime_w = None
    if weight == "runtime":
        runtime_w = w.clone()
        with torch.no_grad():
            head.weight.zero_()
    GemmSpy(monkeypatch)
    out = run_head(head, shape_input(x2d, layout, seed=tokens), runtime_w)
    check_logits(out.reshape(-1, vocab), x2d, w, f"{layout} input", exact=True)


PRODUCTION = [  # (vocab, hidden, tokens)
    (151936, 4096, 1),  # Qwen3-8B
    (151936, 4096, 8192),
    (152064, 8192, 64),  # Qwen2.5-72B
    (128256, 8192, 1024),  # Llama-3.1-70B
    (262144, 5376, 1024),  # Gemma-3-27B
    (262144, 8192, 1),
]


@pytest.mark.parametrize("kind", ["random", "integer"])
@pytest.mark.parametrize(
    "vocab,hidden,tokens", [pytest.param(*p, id=f"V{p[0]}-H{p[1]}-M{p[2]}") for p in PRODUCTION]
)
def test_production_vocabularies(parallel, vocab, hidden, tokens, kind, monkeypatch):
    x2d, w = operands(tokens, hidden, vocab, kind, seed=vocab + tokens)
    head = build_head(make_config(hidden), w)
    spy = GemmSpy(monkeypatch)
    out = run_head(head, x2d.unsqueeze(1), None)
    assert [c["out_dtype"] for c in spy.calls] == [FP32]
    check_logits(
        out.view(-1, vocab), x2d, w, f"V{vocab} H{hidden} M{tokens}", exact=kind == "integer"
    )


@pytest.mark.sanitizer
def test_sanitizer_case(parallel, monkeypatch):
    """Small odd shape (V 257, H 64, M 9): partial tiles in both output dimensions."""
    x2d, w = operands(9, 64, 257, "random", seed=5)
    head = build_head(make_config(64), w)
    GemmSpy(monkeypatch)
    out = run_head(head, x2d.unsqueeze(1), None)
    check_logits(out.view(-1, 257), x2d, w, "sanitizer")


def test_cuda_graph_replays_the_eager_logits(parallel):
    """A CUDA graph of the fp32 head (captured after a warm-up) replays the eager bits for new
    hidden states, and its logits stay within the fp64 bound."""
    vocab, hidden, tokens = 32000, 1024, 33
    _, w = operands(tokens, hidden, vocab, "random", seed=11)
    head = build_head(make_config(hidden), w)
    static_x = bf16((tokens, 1, hidden), 12)
    with torch.no_grad(), InferenceMode.active():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                head(static_x, runtime_gather_output=True, output_dtype=FP32)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out, _ = head(static_x, runtime_gather_output=True, output_dtype=FP32)
        for seed in (13, 14):
            fresh = bf16((tokens, 1, hidden), seed)
            static_x.copy_(fresh)
            graph.replay()
            eager, _ = head(fresh, runtime_gather_output=True, output_dtype=FP32)
            ou.assert_exact(static_out, eager, what=f"replay {seed} vs eager")
            check_logits(static_out.view(-1, vocab), fresh.view(-1, hidden), w, f"replay {seed}")
    del graph


# ---------------------------------------------------------------------------------------------
# Who emits: the predicate and the no-TE path
# ---------------------------------------------------------------------------------------------


def test_lm_head_can_emit_decisions(parallel, monkeypatch):
    """`lm_head_can_emit` and `_linear_forward` on every branch: the parameter dtype always;
    fp32 only for a TE-backed ColumnParallelLinear outside batch-invariant mode; the TE LM head
    (MXFP8 output projection) never; and a mixed-precision request without TE raises."""
    from megatron.core.extensions import transformer_engine as te_ext

    head = build_head(make_config(64), bf16((128, 64), 1))
    assert lm_head_can_emit(head, BF16) is True
    assert lm_head_can_emit(head, FP32) is True
    with bik.set_batch_invariant_mode(True, backend="triton"):
        assert lm_head_can_emit(head, FP32) is False, "batch-invariant mode keeps the head bf16"
        assert lm_head_can_emit(head, BF16) is True
    te_head_cls = getattr(te_ext, "TELMHeadColumnParallelLinear", None)
    if te_head_cls is not None:
        te_head = te_head_cls.__new__(te_head_cls)
        torch.nn.Module.__init__(te_head)
        te_head.config = head.config
        assert lm_head_can_emit(te_head, FP32) is False, "the TE LM head cannot emit fp32"
    fp32_head = types.SimpleNamespace(config=make_config(64, params_dtype=FP32, bf16=False))
    assert lm_head_can_emit(fp32_head, FP32) is True
    x = bf16((4, 64), 2)
    monkeypatch.setattr(te_ext, "te_general_gemm", None)
    assert lm_head_can_emit(head, FP32) is False
    with pytest.raises(RuntimeError, match="general_gemm"):
        tp_layers._linear_forward(x, head.weight, None, FP32)
    same = tp_layers._linear_forward(x, head.weight, None, BF16)
    assert same.dtype == BF16, "an output in the input dtype needs no TE"


# ---------------------------------------------------------------------------------------------
# Tensor-parallel vocabulary shards (2 GPUs)
# ---------------------------------------------------------------------------------------------

TP_CASES = ou.pairwise_cases(
    {
        "vocab": [258, 32000, 151936],
        "tokens": [2, 64, 130],
        "gather": [True, False],
        "sp": [False, True],
        "impl": ["transformer_engine", "inference_optimized"],
    }
)


def _tp_id(c):
    return (
        f"V{c['vocab']}-M{c['tokens']}-{'gather' if c['gather'] else 'shard'}-"
        f"{'sp' if c['sp'] else 'nosp'}-{c['impl']}"
    )


@pytest.mark.parametrize("parallel", [2], indirect=True)
@pytest.mark.parametrize("case", [pytest.param(c, id=_tp_id(c)) for c in TP_CASES])
def test_tp_vocabulary_shards(parallel, case, monkeypatch):
    """TP 2: each rank's GEMM emits fp32 logits for its vocabulary rows; the gather (NCCL, or the
    NVLS multimem all-gather of fp32 under `inference_optimized`) assembles the full V; sequence
    parallelism gathers the hidden states first."""
    from megatron.core.tensor_parallel import inference_layers

    tp = parallel
    rank = parallel_state.get_tensor_model_parallel_rank()
    vocab, tokens, hidden = case["vocab"], case["tokens"], 512
    x2d, w = operands(tokens, hidden, vocab, "random", seed=vocab + tokens)
    config = make_config(hidden, tp=tp, sp=case["sp"], impl=case["impl"])
    head = build_head(config, w)
    nvls_calls = []
    real_ag = inference_layers.multimem_all_gather

    def ag_spy(*args, **kwargs):
        nvls_calls.append(args[1].dtype)
        return real_ag(*args, **kwargs)

    monkeypatch.setattr(inference_layers, "multimem_all_gather", ag_spy)
    if case["impl"] == "inference_optimized":
        buf = SymmetricMemoryManager.get_buffer(
            "tp", process_group=parallel_state.get_tensor_model_parallel_group()
        )
        if buf.symm_buffer is not None:
            buf.symm_buffer.view(torch.uint8).fill_(0xFF)  # NaN in fp32 and bf16
        torch.cuda.synchronize()
        torch.distributed.barrier()
    spy = GemmSpy(monkeypatch)
    x = x2d.unsqueeze(1)
    if case["sp"]:
        x = x.chunk(tp, dim=0)[rank].contiguous()
    out = run_head(head, x, None, gather=case["gather"])
    assert [c["out_dtype"] for c in spy.calls] == [FP32]
    if case["gather"]:
        assert out.shape == (tokens, 1, vocab)
        check_logits(out.view(-1, vocab), x2d, w, _tp_id(case))
    else:
        assert out.shape == (tokens, 1, vocab // tp)
        check_logits(out.view(-1, vocab // tp), x2d, w.chunk(tp, dim=0)[rank], _tp_id(case))
    eligible = (tokens * (vocab // tp) * 4) % 16 == 0
    expect_nvls = case["impl"] == "inference_optimized" and case["gather"] and eligible
    assert nvls_calls == ([FP32] if expect_nvls else []), (nvls_calls, expect_nvls)


# ---------------------------------------------------------------------------------------------
# Models: GPT, Hybrid, the MTP step, batch-invariant mode
# ---------------------------------------------------------------------------------------------

MODEL_HIDDEN = 256
MODEL_SEQ = 37


def build_model(kind, vocab, *, tied, mup_mult=None, **config_overrides):
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.models.hybrid.hybrid_model import HybridModel

    extra = {}
    if mup_mult is not None:
        extra = dict(use_mup=True, mup_base_hidden_size=MODEL_HIDDEN, mup_output_mult=mup_mult)
    config = TransformerConfig(
        num_layers=2,
        hidden_size=MODEL_HIDDEN,
        num_attention_heads=4,
        params_dtype=BF16,
        bf16=True,
        add_bias_linear=False,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        **extra,
        **config_overrides,
    )
    model_parallel_cuda_manual_seed(123, force_reset_rng=True)
    common = dict(
        config=config,
        vocab_size=vocab,
        max_sequence_length=64,
        share_embeddings_and_output_weights=tied,
    )
    if kind == "gpt":
        # RoPE, as production GPT models; an unwrapped learned position table would be fp32.
        model = GPTModel(
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            position_embedding_type="rope",
            **common,
        )
    else:
        model = HybridModel(
            hybrid_stack_spec=hybrid_stack_spec, hybrid_layer_pattern="*-", **common
        )
    return model.cuda().eval()


class HeadTrace:
    """Records the LM head's input, the `weight` and `output_dtype` it is called with, and its
    output, for one model forward."""

    def __init__(self, head):
        self.records = []

        def pre(module, args, kwargs):
            self.records.append(dict(h=args[0].detach().clone(), kwargs=dict(kwargs)))

        def post(module, args, kwargs, output):
            self.records[-1]["out"] = output[0].detach().clone()

        self.handles = [
            head.register_forward_pre_hook(pre, with_kwargs=True),
            head.register_forward_hook(post, with_kwargs=True),
        ]

    def remove(self):
        for h in self.handles:
            h.remove()


def model_forward(model, input_ids, *, inference):
    trace = HeadTrace(model.output_layer)
    positions = torch.arange(input_ids.shape[1], device=_DEVICE).expand_as(input_ids)
    mode = InferenceMode.active() if inference else contextlib.nullcontext()
    dtypes = []
    hooks = [
        module.register_forward_hook(
            lambda mod, args, out, name=name: dtypes.append(
                (name, getattr(out[0] if isinstance(out, tuple) else out, "dtype", None))
            )
        )
        for name, module in model.named_modules()
        if name
    ]
    try:
        with torch.no_grad(), mode:
            logits = model(
                input_ids=input_ids,
                position_ids=positions,
                attention_mask=None,
                runtime_gather_output=True,
            )
    except Exception as error:
        record = trace.records[-1] if trace.records else {}
        h = record.get("h")
        weight = record.get("kwargs", {}).get("weight")
        weight = model.output_layer.weight if weight is None else weight
        raise AssertionError(
            f"forward ({'inference' if inference else 'training'}) raised {error!r}; head input "
            f"{None if h is None else h.dtype}, head weight {getattr(weight, 'dtype', None)}; "
            f"module output dtypes in order: {dtypes}"
        ) from error
    finally:
        trace.remove()
        for h in hooks:
            h.remove()
    assert len(trace.records) == 1, f"the head ran {len(trace.records)} times"
    return logits, trace.records[0]


def head_weight(model, record):
    weight = record["kwargs"].get("weight")
    return model.output_layer.weight if weight is None else weight


MODEL_CASES = ou.pairwise_cases(
    {
        "kind": ["gpt", "hybrid"],
        "vocab": [50257, 151936],
        "tied": [False, True],
        "mup": [None, 0.3],
        "batch": [1, 3],
    }
)


def _model_id(c):
    tied = "tied" if c["tied"] else "untied"
    return f"{c['kind']}-V{c['vocab']}-{tied}-mup{c['mup']}-B{c['batch']}"


@pytest.mark.parametrize("case", [pytest.param(c, id=_model_id(c)) for c in MODEL_CASES])
def test_model_logits_are_the_fp32_head_output(parallel, case, monkeypatch):
    """Under InferenceMode the model asks its head for fp32, with the tied embedding weight when
    weights are shared, and returns the head's fp32 output MuP-scaled in fp32 as [B, S, V],
    within the fp64 bound of the head's own input; training keeps bf16 logits."""
    vocab, mult = case["vocab"], case["mup"]
    model = build_model(case["kind"], vocab, tied=case["tied"], mup_mult=mult)
    g = ou.make_generator(vocab)
    input_ids = torch.randint(0, vocab, (case["batch"], MODEL_SEQ), generator=g).to(_DEVICE)
    spy = GemmSpy(monkeypatch)
    logits, record = model_forward(model, input_ids, inference=True)
    assert record["kwargs"].get("output_dtype") == FP32, record["kwargs"]
    assert [c["out_dtype"] for c in spy.calls] == [FP32], spy.calls
    weight = head_weight(model, record)
    if case["tied"]:
        embedding = model.embedding.word_embeddings.weight
        assert weight.data_ptr() == embedding.data_ptr(), "the tied head must read the embedding"
    out = record["out"]
    assert out.dtype == FP32 and logits.dtype == FP32, (out.dtype, logits.dtype)
    expected = (out if mult is None else out * mult).transpose(0, 1)
    ou.assert_exact(logits, expected, what="returned logits = scaled head output, [B, S, V]")
    h = record["h"]
    check_logits(out.reshape(-1, vocab), h.reshape(-1, MODEL_HIDDEN), weight, _model_id(case))
    train_logits, train_record = model_forward(model, input_ids, inference=False)
    assert train_record["kwargs"].get("output_dtype") is None, train_record["kwargs"]
    assert train_logits.dtype == BF16, f"training logits are {train_logits.dtype}"


class _MTPHost:
    """Runs `LanguageModule.compute_mtp_single_step` and the properties it reads on a host whose
    MTP layer returns given hidden states (the head's part of the step, unchanged)."""

    compute_mtp_single_step = LanguageModule.compute_mtp_single_step
    lm_head_can_emit_fp32 = LanguageModule.lm_head_can_emit_fp32
    _scale_logits = LanguageModule._scale_logits

    def __init__(self, head, mtp_hidden, embedding_weight, config):
        def forward_single_position(**kwargs):
            return mtp_hidden

        layer = types.SimpleNamespace(forward_single_position=forward_single_position)
        self.mtp = types.SimpleNamespace(layers=[layer])
        self.output_layer = head
        self.embedding = None
        self.config = config
        self.share_embeddings_and_output_weights = embedding_weight is not None
        self._embedding_weight = embedding_weight

    def shared_embedding_or_output_weight(self):
        return self._embedding_weight


@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
@pytest.mark.parametrize("mup", [None, 0.3], ids=["nomup", "mup0.3"])
@pytest.mark.parametrize("backend", [None, "triton"], ids=["emit", "bi_triton"])
def test_mtp_step_logits(parallel, tied, mup, backend, monkeypatch):
    """`compute_mtp_single_step`: fp32 [N, 1, V] logits. Emitting head: the TE fp32 GEMM, scaled
    in fp32, within the fp64 bound. Batch-invariant mode: the bf16 head output scaled in bf16
    (as training does), then upcast, bitwise."""
    vocab, hidden, n = 32000, 512, 5
    extra = {} if mup is None else dict(use_mup=True, mup_output_mult=mup)
    config = make_config(hidden, **extra)
    _, w = operands(n, hidden, vocab, "random", seed=21)
    head = build_head(config, torch.zeros_like(w) if tied else w)
    mtp_hidden = bf16((n, 1, hidden), 22)
    host = _MTPHost(head, mtp_hidden, w if tied else None, config)
    ids = torch.zeros((1, n), dtype=torch.long, device=_DEVICE)
    spy = GemmSpy(monkeypatch)
    mode = (
        bik.set_batch_invariant_mode(True, backend=backend) if backend else contextlib.nullcontext()
    )
    trace = HeadTrace(head)
    try:
        with mode:
            hidden_out, logits = host.compute_mtp_single_step(mtp_hidden, ids, ids, depth=0)
    finally:
        trace.remove()
    (record,) = trace.records
    assert hidden_out is mtp_hidden
    assert record["kwargs"].get("runtime_gather_output") is True, record["kwargs"]
    if tied:
        assert record["kwargs"].get("weight") is w, "the tied MTP head must read the embedding"
    assert logits.dtype == FP32 and logits.shape == (n, 1, vocab), (logits.dtype, logits.shape)
    out = record["out"]
    expected = (out if mup is None else out * mup).float()
    ou.assert_exact(logits, expected, what="logits = head output, MuP-scaled in its dtype")
    if backend:
        assert spy.calls == [], "batch-invariant mode must not take the fp32 TE GEMM"
        assert record["kwargs"].get("output_dtype") is None, record["kwargs"]
        assert out.dtype == BF16, f"the batch-invariant head returned {out.dtype}"
        return
    assert record["kwargs"].get("output_dtype") == FP32, record["kwargs"]
    assert [c["out_dtype"] for c in spy.calls] == [FP32], spy.calls
    check_logits(out.view(-1, vocab), mtp_hidden.view(-1, hidden), w, "MTP step")


BI_BACKENDS = ["triton", "deepgemm", "te_native"]


@pytest.mark.parametrize("backend", BI_BACKENDS)
@pytest.mark.parametrize("kind", ["gpt", "hybrid"])
def test_batch_invariant_logits_are_the_training_logits_upcast(
    parallel, kind, backend, monkeypatch
):
    """Batch-invariant mode: the head runs in bf16 (no fp32 TE GEMM), and the inference logits
    are bitwise the training forward's bf16 logits (MuP-scaled in bf16) upcast to fp32."""
    if backend == "deepgemm" and not bik.HAVE_DEEPGEMM_BF16:
        pytest.fail("the deepgemm backend needs DeepGEMM's bf16 bindings", pytrace=False)
    vocab = 32000
    model = build_model(kind, vocab, tied=kind == "gpt", mup_mult=0.3)
    g = ou.make_generator(7)
    input_ids = torch.randint(0, vocab, (2, MODEL_SEQ), generator=g).to(_DEVICE)
    spy = GemmSpy(monkeypatch)
    with bik.set_batch_invariant_mode(True, backend=backend):
        train_logits, train_record = model_forward(model, input_ids, inference=False)
        logits, record = model_forward(model, input_ids, inference=True)
    assert spy.calls == [], "batch-invariant mode must not take the fp32 TE GEMM"
    assert record["kwargs"].get("output_dtype") is None, record["kwargs"]
    assert train_logits.dtype == BF16 and logits.dtype == FP32, (train_logits.dtype, logits.dtype)
    ou.assert_exact(record["h"], train_record["h"], what="the head's input (decoder output)")
    ou.assert_exact(logits, train_logits.float(), what="inference logits vs training logits")


@pytest.mark.parametrize("kind", ["gpt", "hybrid"])
def test_batch_invariant_fp32_residual_stream(parallel, kind, monkeypatch):
    """`fp32_residual_connection=True`: the residual stream is fp32, and TE's final norm returns
    its parameter dtype, so the head reads bf16 hidden states (checked without the mode). Under
    batch-invariant mode (triton) the forward must run alike, and the inference logits must be
    the training logits upcast, bit for bit.

    Suspected finding O8a-P2: the batch-invariant TE RMSNorm returns its input's dtype (fp32),
    so the head's `torch.matmul` gets an fp32 input and a bf16 weight and
    `matmul_persistent` raises "Incompatible dtypes" (first seen with an fp32 learned position
    table, dfw 21627808). Focused case:
    `test_rmsnorm_batch_invariant.py::test_te_rmsnorm_module_output_dtype_follows_te`."""
    vocab = 32000
    model = build_model(kind, vocab, tied=False, fp32_residual_connection=True)
    g = ou.make_generator(17)
    input_ids = torch.randint(0, vocab, (2, MODEL_SEQ), generator=g).to(_DEVICE)
    _, native = model_forward(model, input_ids, inference=False)
    assert native["h"].dtype == BF16, f"without the mode the head reads {native['h'].dtype}"
    with bik.set_batch_invariant_mode(True, backend="triton"):
        train_logits, train_record = model_forward(model, input_ids, inference=False)
        logits, record = model_forward(model, input_ids, inference=True)
    assert (
        train_record["h"].dtype == BF16
    ), f"under the mode the head reads {train_record['h'].dtype}"
    ou.assert_exact(record["h"], train_record["h"], what="the head's input (decoder output)")
    ou.assert_exact(logits, train_logits.float(), what="inference logits vs training logits")


# ---------------------------------------------------------------------------------------------
# The dynamic-inference prefill of a batch-invariant hybrid MoE model (4 GPUs)
# ---------------------------------------------------------------------------------------------

_EP = 4


def test_hybrid_moe_batch_invariant_prefill_logits(monkeypatch):
    """The configuration of `tests/unit_tests/inference/test_hybrid_moe.py::
    TestDynamicInferenceNVLS::test_batch_invariant_prefill_matches_full_forward` (hybrid "ME*",
    EP 4, NVLS dispatcher, batch-invariant mode, a 64-token prompt), checked against #8235's
    contract: the dynamic prefill's logits are fp32 and bitwise the full training forward's bf16
    logits upcast. The head's input is compared first, so a mismatch is attributed to the decoder
    (batch-invariance of Mamba / MoE / attention, issue #7958) or to the head."""
    ou.require_sm(ou.HOPPER)
    if Utils.world_size < _EP or Utils.world_size % _EP:
        pytest.skip(f"needs a multiple of {_EP} ranks; the manifest selects it so")
    from megatron.core.inference.inference_request import DynamicInferenceRequest
    from megatron.core.inference.sampling_params import SamplingParams
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_inference_stack_spec
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.ssm.mamba_mixer import _check_mamba_sequence_packing_support
    from megatron.core.transformer.attention import HAVE_FA4
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher
    from tests.unit_tests.inference.test_hybrid_moe import _TestDynamicInferenceBase
    from tests.unit_tests.inference.test_moe_dispatching_and_routing import _make_base_config
    from tests.unit_tests.test_utilities import clear_nvte_env_vars

    available, reason = _check_mamba_sequence_packing_support(for_inference_not_training=True)
    assert available, reason
    Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=_EP)
    try:
        model_parallel_cuda_manual_seed(123, inference_rng_tracker=True, force_reset_rng=True)
        clear_nvte_env_vars()
        config = _make_base_config(
            num_layers=3,
            batch_invariant_mode=True,
            attention_backend=AttnBackend.flash,
            attention_dropout=0.0,
            flash_attention_version=4 if HAVE_FA4 else 3,
            inference_grouped_gemm_backend="torch",
            inference_moe_token_dispatcher_type="nvls",
        )
        base = _TestDynamicInferenceBase()
        model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_inference_stack_spec,
            vocab_size=base.VOCAB_SIZE,
            max_sequence_length=base.MAX_SEQ_LEN,
            hybrid_layer_pattern="ME*",
        ).cuda()
        model.eval()
        input_ids = torch.arange(64, device=_DEVICE, dtype=torch.long).unsqueeze(0)
        trace = HeadTrace(model.output_layer)
        try:
            with torch.inference_mode(), bik.set_batch_invariant_mode(True):
                full_logits = model(
                    input_ids=input_ids,
                    position_ids=None,
                    attention_mask=None,
                    runtime_gather_output=True,
                )
                ctx = base._build_context(
                    model,
                    num_cuda_graphs=0,
                    use_cuda_graphs_for_non_decode_steps=False,
                    max_requests=4,
                    max_tokens=128,
                )
                request = DynamicInferenceRequest(
                    request_id=0,
                    prompt_tokens=input_ids.cpu().squeeze(0),
                    sampling_params=SamplingParams(num_tokens_to_generate=1, termination_id=-1),
                )
                ctx.add_request(request)
                ctx.initialize_attention_state()
                with InferenceMode.active():
                    inference_logits = model(
                        input_ids=input_ids,
                        position_ids=None,
                        attention_mask=None,
                        inference_context=ctx,
                        runtime_gather_output=True,
                    )
        finally:
            trace.remove()
        train_rec, inf_rec = trace.records
        assert full_logits.dtype == BF16 and inference_logits.dtype == FP32, (
            full_logits.dtype,
            inference_logits.dtype,
        )
        assert inf_rec["kwargs"].get("output_dtype") is None, inf_rec["kwargs"]
        h_train, h_inf = train_rec["h"], inf_rec["h"]
        assert h_train.shape == h_inf.shape, (h_train.shape, h_inf.shape)
        head_inputs_equal = torch.equal(h_train.view(torch.int16), h_inf.view(torch.int16))
        head_outputs_equal = torch.equal(
            train_rec["out"].view(torch.int16), inf_rec["out"].view(torch.int16)
        )
        upcast_equal = torch.equal(inference_logits, full_logits.float())
        h_diff = (h_train.float() - h_inf.float()).abs()
        l_diff = (inference_logits - full_logits.float()).abs()
        assert upcast_equal, (
            "dynamic prefill logits != training logits upcast: head input equal="
            f"{head_inputs_equal} ({int((h_diff > 0).sum())} of {h_diff.numel()} differ, max "
            f"{h_diff.max().item():.3e}, rows {sorted(set((h_diff > 0).nonzero()[:, 0].tolist()))[:16]}), "
            f"head output (bf16) equal={head_outputs_equal}, logits max diff {l_diff.max().item():.3e}; "
            + (
                "the decoder output differs, so the head is not the cause"
                if not head_inputs_equal
                else "the head turned equal inputs into different logits"
            )
        )
    finally:
        torch.cuda.synchronize()
        torch.distributed.barrier()
        NVLSAllGatherVDispatcher._delete_buffers()
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()
