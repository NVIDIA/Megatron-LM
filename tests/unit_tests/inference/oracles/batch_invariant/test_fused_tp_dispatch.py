# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for the fused-TP inference kernel selection, `configure_fused_tp_inference`.

Code: `megatron/core/transformer/transformer_layer.py` (`configure_fused_tp_inference`, the
fused-TP branches of the attention and MLP residual steps), `transformer_block.py`
(`_setup_fused_tp_communication`) and `megatron/core/tensor_parallel/inference_layers.py`
(`InferenceRowParallelLinear._matmul_reduce_scatter`, which picks the fused NVLS reduce-scatter
+ residual add + RMSNorm + all-gather kernel, a plain NVLS reduce-scatter or an NCCL one, and
`InferenceLayerNormColumnParallelLinear.forward`, which reads its input from the symmetric
buffer in fused mode). Active for inference-optimized dense GPT models with
`inference_fuse_tp_communication=True`.

Contract (written from the layer math and the option's meaning, a communication fusion):
- A layer computes h1 = h + Attn(norm_qkv(h)) and h2 = h1 + MLP(norm_fc1(h1)). With the flag on,
  every configuration validation accepts must compute the same function, whichever kernel runs:
  the residual is added exactly once per sublayer, and each column-parallel GEMM's input is the
  all-gathered RMSNorm of the current stream with that layer's own norm weight (QKV: its
  `linear_qkv` norm; FC1: its `linear_fc1` norm). The configurations include pipeline stages
  (each stage's block maps its input stream to its output stream; the last stage applies the
  final norm), the MTP inner layer and HybridStack layers (built outside TransformerBlock, never
  configured), MLA layers (whose attention applies its own input norm), TP up to 8, and CUDA-graph
  capture of single layers (local graphs) or of the whole block (full-iteration graphs).
- A consumer reads the symmetric buffer only when this forward's producer wrote it; a replayed
  graph replays the decisions made at its capture, and they must be consistent with whatever
  runs before and after it (eager steps, other graphs).
- Dispatch (exact): a row-parallel layer runs the fused NVLS kernel if and only if TP > 1, the
  Triton NVLS kernels are allowed, activations are bf16, the sizes are NVLS-eligible, the
  symmetric buffer holds the tensor and batch-invariant mode is off; every rank of a TP group
  takes the same sequence of branches.

Oracles:
- Exact metamorphic: with every `linear_proj` and `linear_fc2` weight zero, each sublayer adds 0,
  so a block's (stage's, layer's, graph's, engine step's) output stream must equal its input bit
  for bit on every path (the fused kernel computes bf16(0 + R) = R). A dropped residual gives 0,
  a doubled one 2R. The `"tp"` symmetric buffer is NaN-poisoned before every forward or replay,
  so a stale read turns the output NaN.
- fp64 reference for the GEMM inputs: each QKV / FC1 GEMM input, recorded at `_apply_linear`
  (under CUDA graphs by a copy captured in the graph, so each replay refreshes it), against
  RMSNorm(input; its own norm weight) in fp64 (bounded by an fp32-computed RMSNorm rounded to the
  activation dtype), with distinct norm gains per layer. The last pipeline stage's output against
  fp64 RMSNorm with the final norm weight.
- Dispatch: spies on the fused kernel, the NVLS reduce-scatter and NCCL's; fused calls and the
  norm weight each applies checked exactly (`_check_fused_norms`); traces gathered across ranks
  must be identical within a stage.
Tolerance class: exact for the outputs, the dispatch and the traces; floating point (default
factor, per row) for the GEMM inputs and the final norm.

Structure (read after the contract): `_matmul_reduce_scatter` takes the NVLS branch on
triton_nvls_kernels_allowed, bf16, `are_tensors_nvls_eligible` (16-byte multiple, SM >= 9), a
buffer handle and not batch-invariant mode, and fuses when the next norm weights are set; the
layer skips its own residual add whenever the flag is on. `_setup_fused_tp_communication` wires
the layers of one TransformerBlock (one pipeline stage): the first layer's QKV applies its own
norm, the last FC2 gets a placeholder weight; MTP inner layers (`MultiTokenPredictionLayer`) and
HybridStack layers are never wired. The buffer holds [T, B, H] bf16 iff T*B*H*2 bytes fit, so the
decision changes with the step's token count. Covered:
- TP = 1, 2 and 8 (flag off as controls at TP 1 and 2; NVLS at TP 2 and 8), 3 layers (first,
  middle, last);
- TP 2 x PP 2 with 3 layers per stage and the final norm on the last stage (NVLS);
- step token counts of decode (2 tp, 4 tp), mixed (64 tp, 65 tp) and prefill (1024 tp) steps in
  dynamic batching's [T, 1, H] layout, on one block and buffer;
- full-iteration graphs (the block captured at two sizes) and local graphs (each layer captured
  alone, replayed as a chain, with eager steps in between);
- the real engine (`DynamicInferenceEngine`) with staggered requests, so prefill, decode and
  mixed steps, eager;
- production widths (Llama-3-8B, H 4096, and Qwen2.5-72B, H 8192: SwiGLU, grouped-query
  attention, 3 layers) at TP 2 and TP 8 with NVLS, over steps of 8192, tp, 1032, 256 and tp
  tokens on one block and buffer.

Main vs open PRs: every case here passes on main. The paths that fail on main are tested by the
PRs that fix them:
- Wherever the fused kernel does not run (TP = 1, NVLS disabled, batch-invariant mode, fp16, a
  buffer too small for the step, the MTP inner layer), main's row-parallel layer drops the
  residual it was handed: #8561 (d1, `tensor_parallel/test_inference_layers.py`). #8712 (n16),
  #8704 (n8) and #8868 (t3-r04f3) change the same wiring and must keep these tests passing.
- One token per rank at a toy H (a shard narrower than a warp, as in CUDA-graph decode steps):
  main's fused-kernel launcher computes num_warps = 0; #8880 (t3-r09f2,
  `determinism/kernels/test_nvls_fused_collectives.py`).
- MLA layers fail at construction on main (no `linear_qkv` norm weights): #8704
  (`transformer/test_mla_fused_tp_communication.py`).
- HybridStack layers hand the residual to an IdentityOp's missing projection (AttributeError,
  finding O8a-N7): #8965 (`ssm/test_hybrid_fused_tp_communication.py`).

Hardware: H100 with Transformer Engine; one GPU for TP = 1, two NVLS-connected GPUs for TP = 2
(`nproc` 2), four for TP 2 x PP 2 (`nproc` 4), eight for TP = 8 (`nproc` 8).
"""

import inspect
import types

import pytest
import torch
import torch.distributed as dist

import megatron.core.tensor_parallel.inference_layers as inference_layers
from megatron.core import parallel_state
from megatron.core.inference.config import AsyncScheduleMode, InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_inference_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import attention as attention_module
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.core.transformer.enums import InferenceCudaGraphScope
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.test_utilities import Utils

pytestmark = [ou.requires_sm(ou.HOPPER)]

_DEVICE = "cuda"
SEQ, BATCH, HIDDEN, FFN, HEADS, LAYERS = 64, 2, 128, 256, 4, 3

# (case, tensor-parallel size, flag on): the flag off as controls, and the fused NVLS path. The
# fallback paths drop the residual on main; #8561 tests them.
CASES = [
    ("flag_off", 1, False),
    ("flag_off_tp2", 2, False),
    ("nvls", 2, True),
    ("tp8_nvls", 8, True),
]
FUSED_EXPECTED = {"nvls", "tp8_nvls"}


@pytest.fixture
def parallel(request):
    tp, pp = request.param if isinstance(request.param, tuple) else (request.param, 1)
    if Utils.world_size < tp * pp:
        pytest.skip(
            f"needs {tp * pp} ranks (run with nproc >= {tp * pp}); the manifest selects it so"
        )
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp)
    SymmetricMemoryManager.destroy("tp")
    yield tp
    SymmetricMemoryManager.destroy("tp")
    Utils.destroy_model_parallel()


def make_config(tp, case, fuse, *, config_class=TransformerConfig, **overrides):
    """The inference-optimized config of every case: bf16 (fp16 for "fp16"), RMSNorm, no bias."""
    dtype = torch.float16 if case == "fp16" else torch.bfloat16
    options = dict(
        num_layers=LAYERS,
        hidden_size=HIDDEN,
        ffn_hidden_size=FFN,
        num_attention_heads=max(HEADS, tp),
        tensor_model_parallel_size=tp,
        sequence_parallel=tp > 1,
        transformer_impl="inference_optimized",
        normalization="RMSNorm",
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        params_dtype=dtype,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        init_method_std=0.05,
        use_cpu_initialization=False,
        inference_fuse_tp_communication=fuse,
        inference_disable_triton_nvls_kernels=case.endswith("nvls_disabled"),
    )
    options.update(overrides)
    return config_class(**options), dtype


def norm_consumers(layer):
    """(name, linear) for each GEMM of the layer that applies the input norm itself: QKV (not in
    MLA or MLP-only layers) and FC1 (not in attention-only layers)."""
    found = []
    for name, owner, attr in (
        ("qkv", "self_attention", "linear_qkv"),
        ("fc1", "mlp", "linear_fc1"),
    ):
        linear = getattr(getattr(layer, owner, None), attr, None)
        if hasattr(linear, "layer_norm_weight"):
            found.append((name, linear))
    return found


def zero_projections_(layers, seed=4321):
    """Zero every attention / MLP output projection and give each norm distinct gains, in place:
    the fused wiring holds the norm weights by reference."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for layer in layers:
            for owner, attr in (("self_attention", "linear_proj"), ("mlp", "linear_fc2")):
                linear = getattr(getattr(layer, owner, None), attr, None)
                if linear is not None:
                    linear.weight.zero_()
            for name, param in layer.named_parameters():
                if name.endswith(("layer_norm_weight", "layernorm.weight")):
                    param.copy_(1.0 + 0.5 * torch.randn(param.shape, generator=g))


def build_block(tp, case, fuse, *, pp=1, num_layers=LAYERS, **block_kwargs):
    config, dtype = make_config(
        tp,
        case,
        fuse,
        num_layers=num_layers,
        pipeline_model_parallel_size=pp,
        pipeline_dtype=torch.bfloat16 if pp > 1 else None,
    )
    model_parallel_cuda_manual_seed(123, force_reset_rng=True)
    block = TransformerBlock(
        config, get_gpt_layer_with_inference_spec(), post_layer_norm=False, **block_kwargs
    )
    block = block.cuda().eval()
    zero_projections_(block.layers)
    return block, config, dtype


def full_input(dtype, shape=(SEQ, BATCH, HIDDEN), seed=1234):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(*shape, generator=g).to(dtype).to(_DEVICE)


def tp_shard(x_full):
    """This rank's sequence-parallel shard of x_full."""
    tp = parallel_state.get_tensor_model_parallel_world_size()
    rank = parallel_state.get_tensor_model_parallel_rank()
    return x_full.chunk(tp, dim=0)[rank].contiguous()


def _poison_tp_buffer(tp):
    if tp == 1:
        return
    buf = SymmetricMemoryManager.get_buffer(
        "tp", process_group=parallel_state.get_tensor_model_parallel_group()
    )
    if buf.symm_buffer is not None:
        buf.symm_buffer.view(torch.bfloat16).fill_(float("nan"))
    torch.cuda.synchronize()
    dist.barrier()


class _Trace:
    """Spies on the reduce-scatter kernels and records every column-parallel GEMM input.

    For each fused-kernel call it also records the address of the RMSNorm weight it applied, to
    check the wiring of `configure_fused_tp_inference`.
    """

    def __init__(self, monkeypatch):
        self.branches = []
        self.fused_norms = []
        self.gemm_inputs = []
        for name in (
            "fused_multimem_rs_add_norm_ag",
            "multimem_reduce_scatter",
            "reduce_scatter_along_first_dim",
        ):
            real = getattr(inference_layers, name)
            monkeypatch.setattr(inference_layers, name, self._spy(name, real))
        real_linear = inference_layers._apply_linear

        def apply_linear(x, weight, config, out=None):
            self.gemm_inputs.append((weight, x.detach().clone()))
            return real_linear(x, weight, config, out=out)

        monkeypatch.setattr(inference_layers, "_apply_linear", apply_linear)

    def _spy(self, name, real):
        signature = inspect.signature(real)

        def spy(*args, **kwargs):
            self.branches.append(name)
            if name == "fused_multimem_rs_add_norm_ag":
                weights = signature.bind(*args, **kwargs).arguments["rms_norm_weights"]
                self.fused_norms.append(weights.data_ptr())
            return real(*args, **kwargs)

        return spy

    def clear(self):
        self.branches.clear()
        self.fused_norms.clear()
        self.gemm_inputs.clear()

    def input_of(self, weight, records=None):
        """The input of the GEMM with this weight, from ``records`` (a list kept from a graph
        capture, whose captured copies each replay refreshes) or from the last forward."""
        found = [x for w, x in (self.gemm_inputs if records is None else records) if w is weight]
        assert len(found) == 1, f"expected one GEMM with this weight, found {len(found)}"
        return found[0]


def _rms_norm_reference(x, w, eps):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * w


def _rms_norm_trusted(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


def _check_exact(actual, expected, what, problems):
    try:
        ou.assert_exact(actual, expected, what=what)
    except AssertionError as e:
        problems.append(str(e))


def _check_gemm_inputs(layers, trace, x_full, eps, problems, *, records=None, what=""):
    """Each QKV / FC1 GEMM input is RMSNorm(stream; that GEMM's own norm weight), all-gathered.
    With zero projections the stream is the input at every layer."""
    for i, layer in enumerate(layers):
        for name, linear in norm_consumers(layer):
            w = linear.layer_norm_weight.detach()
            try:
                got = trace.input_of(linear.weight, records).reshape(x_full.shape)
                ou.assert_close_to_reference(
                    got,
                    ou.fp64_reference(_rms_norm_reference, x_full, w, eps),
                    _rms_norm_trusted(x_full, w, eps),
                    dim=-1,
                    what=f"{what}layer {i} {name} GEMM input",
                )
            except AssertionError as e:
                problems.append(str(e))


def _check_fused_norms(layers, fused_norms, expect_fused, problems, what=""):
    """The fused kernel's calls and norm weights.

    With ``expect_fused`` it must run on every row-parallel layer that feeds a norm-applying
    consumer, with that consumer's norm weight, in order: layer i's attention projection with its
    own FC1 norm, layer i's FC2 with layer i+1's QKV norm. An FC2 with no such consumer (the last
    one, or one before an MLA layer, which applies its own input norm) may or may not fuse, with
    any weight: not part of the contract (main fuses it with a placeholder, #8704 does not).
    Without ``expect_fused`` the kernel never runs.
    """
    if not expect_fused:
        if fused_norms:
            problems.append(f"{what}fused kernel ran {len(fused_norms)} times; preconditions fail")
        return
    want, real, free = [], set(), 0
    for i, layer in enumerate(layers):
        consumers = dict(norm_consumers(layer))
        real.update(lin.layer_norm_weight.data_ptr() for lin in consumers.values())
        want.append(consumers["fc1"].layer_norm_weight.data_ptr())
        next_qkv = dict(norm_consumers(layers[i + 1])).get("qkv") if i + 1 < len(layers) else None
        if next_qkv is not None:
            want.append(next_qkv.layer_norm_weight.data_ptr())
        else:
            free += 1
    got_real = [p for p in fused_norms if p in real]
    if got_real != want or len(fused_norms) - len(got_real) > free:
        problems.append(
            f"{what}fused kernel norm weights {fused_norms}, expected {want} "
            f"(+ up to {free} unconstrained FC2 calls)"
        )


def _verdict(case, branches, problems, *, stage=0):
    """Share every rank's verdict before asserting (no rank may stop early); the ranks of a
    stage must have taken the same branches."""
    # The verdicts are Python objects: exchange them over gloo, off the GPUs under test.
    group = dist.new_group(backend="gloo")
    try:
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, (stage, branches, problems), group=group)
    finally:
        dist.destroy_process_group(group)
    for s in sorted({st for st, _, _ in gathered}):
        traces = [b for st, b, _ in gathered if st == s]
        assert all(t == traces[0] for t in traces), f"stage {s}: ranks took different branches"
    failures = [f"rank {r}: {p}" for r, (_, _, ps) in enumerate(gathered) for p in ps]
    assert not failures, f"{case}:\n" + "\n".join(failures)


@pytest.mark.parametrize(
    "case,parallel,fuse", CASES, ids=[c[0] for c in CASES], indirect=["parallel"]
)
def test_zero_projections_keep_the_residual(case, parallel, fuse, monkeypatch):
    """Zero projections: the output is the input, bit for bit, on every path; exact dispatch."""
    tp = parallel
    block, config, dtype = build_block(tp, case, fuse)
    x_full = full_input(dtype)
    x = tp_shard(x_full)
    trace = _Trace(monkeypatch)
    outputs = []
    for _ in range(2):  # the second forward reuses the buffer the first one left
        _poison_tp_buffer(tp)
        trace.clear()
        with torch.no_grad(), InferenceMode.active():
            out = block(hidden_states=x.clone(), attention_mask=None)
        torch.cuda.synchronize()
        outputs.append(out)

    # Judge locally, then share every rank's verdict before asserting.
    problems = []
    for j, out in enumerate(outputs):
        _check_exact(out, x, f"block output, forward {j}", problems)
    _check_gemm_inputs(block.layers, trace, x_full, config.layernorm_epsilon, problems)
    _check_fused_norms(block.layers, trace.fused_norms, case in FUSED_EXPECTED, problems)
    _verdict(case, trace.branches, problems)


# ---- Pipeline stages --------------------------------------------------------------------------

PP_LAYERS = 6
PP_CASES = ["pp2_nvls"]


@pytest.mark.parametrize("parallel", [(2, 2)], indirect=True, ids=["tp2pp2"])
@pytest.mark.parametrize("case", PP_CASES)
def test_pipeline_stages_keep_the_residual(case, parallel, monkeypatch):
    """TP 2 x PP 2: each stage's block is wired on its own. Its first layer's QKV applies its own
    norm (the previous stage's last FC2 is on other GPUs), its last FC2 has no consumer, and the
    last stage applies the final norm to the stream. With zero projections each stage maps its
    input stream to itself, so feeding both stages the same input checks the boundary on both
    sides without the point-to-point transfer."""
    tp = parallel
    pp = parallel_state.get_pipeline_model_parallel_world_size()
    stage = parallel_state.get_pipeline_model_parallel_rank()
    first, last = stage == 0, stage == pp - 1
    case_kind = "nvls" if case == "pp2_nvls" else "nvls_disabled"
    config, dtype = make_config(
        tp,
        case_kind,
        True,
        num_layers=PP_LAYERS,
        pipeline_model_parallel_size=pp,
        pipeline_dtype=torch.bfloat16,
    )
    model_parallel_cuda_manual_seed(123, force_reset_rng=True)
    block = TransformerBlock(
        config,
        get_gpt_layer_with_inference_spec(),
        post_layer_norm=True,
        pre_process=first,
        post_process=last,
    )
    block = block.cuda().eval()
    assert len(block.layers) == PP_LAYERS // pp, f"stage {stage} built {len(block.layers)} layers"
    assert (block.final_layernorm is not None) == last, "the final norm belongs to the last stage"
    zero_projections_(block.layers, seed=4321 + stage)
    if last:
        with torch.no_grad():
            g = torch.Generator(device="cpu").manual_seed(99)
            w = block.final_layernorm.weight
            w.copy_(1.0 + 0.5 * torch.randn(w.shape, generator=g))
    streams = []
    if last:
        block.final_layernorm.register_forward_pre_hook(
            lambda module, args: streams.append(args[0].detach().clone())
        )
    x_full = full_input(dtype)
    x = tp_shard(x_full)
    trace = _Trace(monkeypatch)
    eps = config.layernorm_epsilon
    problems, branches = [], []
    for j in range(2):
        _poison_tp_buffer(tp)
        trace.clear()
        streams.clear()
        with torch.no_grad(), InferenceMode.active():
            if not first:
                block.set_input_tensor(x.clone())
            out = block(hidden_states=x.clone(), attention_mask=None)
        torch.cuda.synchronize()
        what = f"stage {stage} forward {j}: "
        if last:
            _check_exact(streams[0], x, f"{what}stream before the final norm", problems)
            w = block.final_layernorm.weight.detach()
            try:
                ou.assert_close_to_reference(
                    out,
                    ou.fp64_reference(_rms_norm_reference, x, w, eps),
                    _rms_norm_trusted(x, w, eps),
                    dim=-1,
                    what=f"{what}final norm",
                )
            except AssertionError as e:
                problems.append(str(e))
        else:
            _check_exact(out, x, f"{what}stage output", problems)
        _check_gemm_inputs(block.layers, trace, x_full, eps, problems, what=what)
        _check_fused_norms(block.layers, trace.fused_norms, case == "pp2_nvls", problems, what)
        branches.append(list(trace.branches))
    _verdict(case, branches, problems, stage=stage)


# ---- Step sizes (decode, mixed, prefill) ------------------------------------------------------

# Token counts per step, in units of tp, in dynamic batching's [T, 1, H] layout: decode (2, 4
# tokens per rank pair), mixed prefill/decode (64, 65) and a long prefill (1024), in an order
# where each kind follows another. One token per rank (T = tp) is not a case here: with this toy
# H (128) that shard is under a warp, and main's fused-kernel launcher computes num_warps = 0
# there (#8880 fixes and tests it).
STEP_TOKENS = {"decode_prefill_sizes": (64, 2, 1024, 4, 65, 64, 2)}


@pytest.mark.parametrize("parallel", [2], indirect=True, ids=["tp2"])
@pytest.mark.parametrize("case", list(STEP_TOKENS))
def test_step_sizes_keep_the_residual(case, parallel, monkeypatch):
    """Consecutive steps of different token counts on one model and one buffer: with the
    default buffer every step fuses. (Steps that fall back between fused ones drop the residual
    on main; #8561's `test_fusion_is_decided_every_forward` tests that.)"""
    tp = parallel
    block, config, dtype = build_block(tp, "nvls", True)
    trace = _Trace(monkeypatch)
    eps = config.layernorm_epsilon
    problems, branches = [], []
    for step, units in enumerate(STEP_TOKENS[case]):
        tokens = tp * units
        x_full = full_input(dtype, (tokens, 1, HIDDEN), seed=step)
        x = tp_shard(x_full)
        _poison_tp_buffer(tp)
        trace.clear()
        with torch.no_grad(), InferenceMode.active():
            out = block(hidden_states=x.clone(), attention_mask=None)
        torch.cuda.synchronize()
        what = f"step {step} (T = {tokens}): "
        _check_exact(out, x, f"{what}block output", problems)
        _check_gemm_inputs(block.layers, trace, x_full, eps, problems, what=what)
        _check_fused_norms(block.layers, trace.fused_norms, True, problems, what)
        branches.append(list(trace.branches))
    _verdict(case, branches, problems)


# ---- Production sizes -------------------------------------------------------------------------

# Production widths, LAYERS layers each (first, middle, last), with the SwiGLU MLP and the
# grouped-query attention of these models. No QKV bias: the selection never reads it.
PRODUCTION_MODELS = {
    # Llama-3-8B.
    "llama3_8b": dict(
        hidden_size=4096, ffn_hidden_size=14336, num_attention_heads=32, num_query_groups=8
    ),
    # Qwen2.5-72B; Llama-3-70B and Nemotron-H-56B have the same hidden width.
    "qwen25_72b": dict(
        hidden_size=8192, ffn_hidden_size=29568, num_attention_heads=64, num_query_groups=8
    ),
}
# Tokens per step over all TP ranks (sequence parallel), in dynamic batching's [T, 1, H] layout:
# an 8192-token prefill chunk, one decode token per rank ("tp"), a mixed step of 1032 tokens (129
# per rank at TP 8), a 256-request decode step, and one token per rank again. Every step fits the
# default 512 MiB buffer ([8192, 1, 8192] bf16 is 128 MiB), so every step fuses. One token per
# rank is H / 8 >= 512 threads of the fused kernel here, a whole number of warps, so main's
# launcher takes it (its narrow-rows bug, #8880, needs fewer than 32 threads: H / 8 < 32).
PRODUCTION_STEPS = (8192, "tp", 1032, 256, "tp")


@pytest.mark.parametrize("parallel", [2, 8], indirect=True, ids=["tp2", "tp8"])
@pytest.mark.parametrize("model", list(PRODUCTION_MODELS))
def test_production_sizes_keep_the_residual(model, parallel, monkeypatch):
    """Production widths at TP 2 and TP 8 with NVLS: prefill, decode and mixed steps in turn on
    one block and one buffer. Every step must take the fused kernel with each consumer's own norm
    weight, give back its input bit for bit with the "tp" buffer NaN-poisoned before it, and feed
    each QKV / FC1 GEMM the fp64 RMSNorm of the stream."""
    tp = parallel
    sizes = PRODUCTION_MODELS[model]
    hidden = sizes["hidden_size"]
    config, dtype = make_config(
        tp, "nvls", True, gated_linear_unit=True, activation_func=torch.nn.functional.silu, **sizes
    )
    model_parallel_cuda_manual_seed(123, force_reset_rng=True)
    block = TransformerBlock(config, get_gpt_layer_with_inference_spec(), post_layer_norm=False)
    block = block.cuda().eval()
    zero_projections_(block.layers)
    trace = _Trace(monkeypatch)
    eps = config.layernorm_epsilon
    problems, branches = [], []
    for step, tokens in enumerate(PRODUCTION_STEPS):
        tokens = tp if tokens == "tp" else tokens
        x_full = full_input(dtype, (tokens, 1, hidden), seed=step)
        x = tp_shard(x_full)
        _poison_tp_buffer(tp)
        trace.clear()
        with torch.no_grad(), InferenceMode.active():
            out = block(hidden_states=x.clone(), attention_mask=None)
        torch.cuda.synchronize()
        what = f"{model} step {step} (T = {tokens}): "
        _check_exact(out, x, f"{what}block output", problems)
        _check_gemm_inputs(block.layers, trace, x_full, eps, problems, what=what)
        _check_fused_norms(block.layers, trace.fused_norms, True, problems, what)
        branches.append(list(trace.branches))
        trace.clear()  # the recorded GEMM inputs are hundreds of MiB at these sizes
    _verdict(f"{model}-tp{tp}", branches, problems)


# ---- CUDA graphs: the block (full-iteration) or each layer (local) captured -------------------

GRAPH_CASES = ["full_iteration", "local"]


def _warm_up(fn):
    """Run fn twice on a side stream, as before a capture."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()


@pytest.mark.parametrize("parallel", [2], indirect=True, ids=["tp2"])
@pytest.mark.parametrize("case", GRAPH_CASES)
def test_cuda_graph_replay_keeps_the_residual(case, parallel, monkeypatch):
    """Graphs bake the decisions of their capture; every replay, and every eager step between
    replays, must still compute the identity with a NaN-poisoned buffer.

    full_iteration: the block captured at 64 tp and 32 tp tokens (both fused), replayed
    alternately. local: each layer captured alone at 64 tp tokens and replayed as a chain, with
    an eager step at 32 tp tokens between replays. (Sizes that decide differently drop the
    residual on main where the kernel falls back; #8561 fixes and tests that.)
    """
    tp = parallel
    sizes = (tp * 64, tp * 32)
    block, config, dtype = build_block(tp, "nvls", True)
    # Seed before capturing, as the engine does: inference has no dropout, so the inference RNG
    # tracker's fork is a no-op. The training tracker's fork (TransformerBlock.forward under
    # sequence parallelism) would reseed the CUDA generator inside the capture, which CUDA
    # graphs refuse.
    model_parallel_cuda_manual_seed(123, inference_rng_tracker=True, force_reset_rng=True)
    trace = _Trace(monkeypatch)
    eps = config.layernorm_epsilon
    problems, branches = [], []
    pool = torch.cuda.graph_pool_handle()

    def step_input(tokens, seed):
        x_full = full_input(dtype, (tokens, 1, HIDDEN), seed=seed)
        return x_full, tp_shard(x_full)

    captured = []  # every CUDAGraph, released before the verdict
    try:
        with torch.no_grad(), InferenceMode.active():
            if case.startswith("full_iteration"):
                graphs = []
                for tokens in sizes:
                    _, x = step_input(tokens, seed=tokens)
                    x_buf = x.clone()
                    _warm_up(lambda: block(hidden_states=x_buf.clone(), attention_mask=None))
                    trace.clear()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=pool):
                        out = block(hidden_states=x_buf, attention_mask=None)
                    what = f"capture T = {tokens}: "
                    _check_fused_norms(block.layers, trace.fused_norms, True, problems, what)
                    branches.append(list(trace.branches))
                    captured.append(graph)
                graphs.append((tokens, graph, x_buf, out, list(trace.gemm_inputs)))
                for rnd in range(2):
                    for tokens, graph, x_buf, out, records in graphs:
                        x_full, x = step_input(tokens, seed=1000 * (rnd + 1) + tokens)
                        x_buf.copy_(x)
                        _poison_tp_buffer(tp)
                        graph.replay()
                        torch.cuda.synchronize()
                        what = f"replay {rnd} T = {tokens}: "
                        _check_exact(out, x, f"{what}block output", problems)
                        _check_gemm_inputs(
                            block.layers, trace, x_full, eps, problems, records=records, what=what
                        )
            else:
                graph_tokens, eager_tokens = sizes
                _, x = step_input(graph_tokens, seed=graph_tokens)
                x_buf = x.clone()
                _warm_up(lambda: block(hidden_states=x_buf.clone(), attention_mask=None))
                trace.clear()
                graphs, outs, h = [], [], x_buf
                for layer in block.layers:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=pool):
                        # hidden_states is positional-only since #8523.
                        h, _ = layer(h, attention_mask=None)
                    graphs.append(graph)
                    captured.append(graph)
                    outs.append(h)
                what = f"capture T = {graph_tokens}: "
                _check_fused_norms(block.layers, trace.fused_norms, True, problems, what)
                branches.append(list(trace.branches))
                records = list(trace.gemm_inputs)
                for rnd in range(2):
                    x_full, x = step_input(graph_tokens, seed=1000 * (rnd + 1))
                    x_buf.copy_(x)
                    _poison_tp_buffer(tp)
                    for graph in graphs:
                        graph.replay()
                    torch.cuda.synchronize()
                    what = f"replay {rnd} T = {graph_tokens}: "
                    for i, out in enumerate(outs):
                        _check_exact(out, x, f"{what}layer {i} output", problems)
                    _check_gemm_inputs(
                        block.layers, trace, x_full, eps, problems, records=records, what=what
                    )
                    x_full, x = step_input(eager_tokens, seed=1000 * (rnd + 1) + 1)
                    _poison_tp_buffer(tp)
                    trace.clear()
                    out = block(hidden_states=x.clone(), attention_mask=None)
                    torch.cuda.synchronize()
                    what = f"eager step {rnd} T = {eager_tokens}: "
                    _check_exact(out, x, f"{what}block output", problems)
                    _check_gemm_inputs(block.layers, trace, x_full, eps, problems, what=what)
                    _check_fused_norms(block.layers, trace.fused_norms, True, problems, what)
                    branches.append(list(trace.branches))
    finally:
        # Release the graphs before anything can fail or tear down: a graph that captured an
        # NCCL fallback keeps the communicator busy, and destroying the process groups under it
        # hangs (pdx 7856500: the failing case timed out in teardown).
        torch.cuda.synchronize()
        for graph in captured:
            graph.reset()
        captured.clear()
        torch.cuda.synchronize()
        dist.barrier()

    _verdict(case, branches, problems)


# ---- The engine: CudaGraphManager graphs, prefill, decode and mixed steps ---------------------

ENGINE_CASES = ["eager"]
ENGINE_VOCAB, ENGINE_MAX_SEQ = 128, 1024
# (arrival step, prompt length): two short prompts, then a 600-token prompt (a mixed step above
# cuda_graph_max_tokens = 512, so eager) and a 130-token one (a mixed step that is graphed).
ENGINE_PROMPTS = ((0, 9), (0, 37), (1, 600), (2, 130))


@pytest.mark.parametrize("parallel", [2], indirect=True, ids=["tp2"])
@pytest.mark.parametrize("case", ENGINE_CASES)
def test_engine_steps_keep_the_residual(case, parallel, monkeypatch):
    """A GPT model with zero projections served by the dynamic engine: in every forward (eager,
    replayed from per-layer graphs, or from block graphs) the decoder's output is its input.
    The decoder's final norm is removed so its output is the stream; hooks on the embedding
    (which also NaN-poisons the buffer) and the output layer, which run eagerly around graphed
    decoders, compare the two on the step's active tokens (padding rows are unspecified).

    Only `eager` is a case here. With layer- or block-scope graphs, CUDA-graph decode steps run
    one token per rank, a shard narrower than a warp at this toy H (128), and main's fused-kernel
    launcher computes num_warps = 0 ("num_warps must be a power of 2"; dfw 21235396); #8880
    (t3-r09f2) fixes and tests that. The graph setup below stays, for those cases to return once
    #8880 is on main."""
    tp = parallel
    graphs = case != "eager"
    scope = {"eager": InferenceCudaGraphScope.none, "local": InferenceCudaGraphScope.layer}.get(
        case, InferenceCudaGraphScope.block
    )
    model_parallel_cuda_manual_seed(
        123, inference_rng_tracker=True, use_cudagraphable_rng=False, force_reset_rng=True
    )
    config, _ = make_config(
        tp,
        "nvls",
        True,
        inference_rng_tracker=True,
        cuda_graph_impl="local" if graphs else "none",
        inference_cuda_graph_scope=scope,
        # Pinned: auto-selection picks FA4 when installed, and on Hopper main then asks FA4 for
        # split-KV, which its SM90 forward rejects (#8584, n2). The attention backend is not what
        # this test is about; FA3 (FA2 where it is absent) runs the paged KV cache on Hopper.
        flash_attention_version=3 if attention_module.HAVE_FA3 else 2,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_inference_spec(),
        vocab_size=ENGINE_VOCAB,
        max_sequence_length=ENGINE_MAX_SEQ,
        parallel_output=True,
        pre_process=True,
        post_process=True,
        # No learned position embedding: it is a plain fp32 torch.nn.Embedding, which would make
        # the decoder input fp32, and TE's RMSNorm has no fp32-input / bf16-weight kernel.
        position_embedding_type="none",
    )
    model = model.cuda().eval()
    with torch.no_grad():
        for param in model.parameters():
            # As Float16Module does, but only for parameters not yet in bf16: the fused wiring
            # holds the (bf16) norm weights by reference, so they must keep their storage.
            if param.dtype != torch.bfloat16:
                param.data = param.data.to(torch.bfloat16)
    zero_projections_(model.decoder.layers)
    model.decoder.final_layernorm = None

    problems, inputs, counts = [], [], {"forwards": 0}

    def after_embedding(module, args, output):
        if torch.cuda.is_current_stream_capturing():
            return
        inputs.append(output.detach().clone())
        _poison_tp_buffer(tp)

    def before_output_layer(module, args):
        if torch.cuda.is_current_stream_capturing() or not inputs:
            return
        # Only the step's active tokens: the rows that pad the step to its token bucket (and
        # this rank's sequence-parallel shard of them) carry unspecified attention outputs, and
        # a zero projection maps a non-finite one to NaN (pdx 7986825: rank 1's padding rows).
        out, expected = args[0], inputs.pop()
        local = out.shape[0]
        rank = parallel_state.get_tensor_model_parallel_rank()
        live = min(max(context.active_token_count - rank * local, 0), local)
        _check_exact(
            out[:live], expected[:live], f"decoder output, forward {counts['forwards']}", problems
        )
        counts["forwards"] += 1

    model.embedding.register_forward_hook(after_embedding)
    model.output_layer.register_forward_pre_hook(before_output_layer)

    context = DynamicInferenceContext(
        model_config=config,
        inference_config=InferenceConfig(
            max_sequence_length=ENGINE_MAX_SEQ,
            buffer_size_gb=0.25,
            block_size_tokens=256,
            max_requests=16,
            materialize_only_last_token_logits=False,
            unified_memory_level=0,
            num_cuda_graphs=4 if graphs else None,
            use_cuda_graphs_for_non_decode_steps=True,
            async_sched_mode=AsyncScheduleMode.LEGACY,
        ),
    )
    wrapper = GPTInferenceWrapper(model, context)
    wrapper.model_is_pipeline_parallel = False
    controller = TextGenerationController(
        inference_wrapped_model=wrapper,
        tokenizer=types.SimpleNamespace(
            vocab_size=ENGINE_VOCAB, detokenize=lambda tokens: "tokenized_prompt"
        ),
    )
    delete_cuda_graphs()
    steps = []
    try:
        with torch.inference_mode():
            engine = DynamicInferenceEngine(controller, context)
            g = torch.Generator(device="cpu").manual_seed(77)
            arrivals = [
                (
                    arrival,
                    DynamicInferenceRequest(
                        request_id=i,
                        prompt_tokens=torch.randint(0, ENGINE_VOCAB - 1, (length,), generator=g)
                        .to(torch.int64)
                        .to(_DEVICE),
                        sampling_params=SamplingParams(
                            num_tokens_to_generate=4, termination_id=-1, top_k=1
                        ),
                    ),
                )
                for i, (arrival, length) in enumerate(ENGINE_PROMPTS)
            ]
            last_arrival = max(a for a, _ in arrivals)
            step = 0
            while step <= last_arrival or engine.has_unfinished_requests():
                for arrival, request in arrivals:
                    if arrival == step:
                        engine._add_request(request)
                engine.step_modern()
                dims = context.batch_dimensions
                steps.append(
                    (
                        dims.prefill_req_count,
                        dims.decode_req_count,
                        context.using_cuda_graph_this_step(),
                    )
                )
                step += 1
    finally:
        delete_cuda_graphs()
    graphed = [s for s in steps if s[2]]
    if graphs:
        assert graphed, f"no step replayed a CUDA graph: {steps}"
        assert any(p > 0 and d > 0 for p, d, _ in graphed), f"no graphed mixed step: {steps}"
        assert len(graphed) < len(steps), f"no eager step: {steps}"
    else:
        assert not graphed, f"eager engine used CUDA graphs: {steps}"
    assert counts["forwards"] >= len(steps), f"{counts['forwards']} forwards checked, {steps}"
    _verdict(case, [len(steps)], problems)
