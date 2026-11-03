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
- TP = 1, 2 and 8 (flag off as controls; NVLS, NVLS disabled, batch-invariant mode, fp16, a
  1 KiB buffer), 3 layers (first, middle, last);
- TP 2 x PP 2 with 3 layers per stage and the final norm on the last stage (NVLS and NCCL);
- step token counts of decode (tp, 2 tp), mixed (64 tp, 65 tp) and prefill (1024 tp) steps in
  dynamic batching's [T, 1, H] layout, with the default buffer and with one holding exactly
  64 tp tokens (so consecutive steps decide differently, 65 tp is one row per rank over);
- full-iteration graphs (the block captured at two sizes) and local graphs (each layer captured
  alone, replayed as a chain, with eager steps in between), at sizes that decide alike and that
  decide differently;
- the real engine (`DynamicInferenceEngine`) with staggered requests, so prefill, decode and
  mixed steps, eager or with layer-scope (local) or block-scope (full-iteration) CUDA graphs from
  `CudaGraphManager`, with the default buffer and one whose capacity splits the graph sizes;
- the MTP inner layer at depths 1 and 2 (`forward_single_position`, as speculative decoding calls
  it), a HybridStack of attention and MLP layers, and MLA blocks (MLA only, and GPT/MLA mixed).

Main vs open PRs:
- On main a row-parallel layer adds the residual only inside the fused kernel, and the consumer
  decides fused mode from its own flag and the buffer handle. So every case where the kernel
  does not run drops the residual: TP = 1, NVLS disabled, batch-invariant mode, fp16, the small
  buffer, `tp8_nvls_disabled`, `pp2_nccl`, `sizes_across_capacity`, the `*_capacity` graph and
  engine cases, and the MTP inner layer at every TP (never wired, so its projections never
  fuse). #8561 (d1) fixes them; #8712 (n16) and #8704 (n8) change the same wiring and must keep
  these tests passing; #8868 (t3-r04f3) changes the batch-invariant branch to a rank-ordered
  reduce-scatter (still no residual add without d1).
- MLA (`test_mla_layers_keep_the_residual`) fails on main at construction: the block asks every
  layer for `linear_qkv`'s norm weights, which MLA does not have (AttributeError). #8704 (n8,
  stacked on #8561) wires MLA and must make it pass; the MLA up projections all-gather back to
  back into the shared buffer without a barrier, a separate race (#8709, n12), so the test takes
  them through NCCL, as #8704's own test does.
- HybridStack (`test_hybrid_stack_keeps_the_residual`) fails on main and, by reading, with #8561:
  an attention-only layer hands its residual to `self.mlp.linear_fc2` and an MLP-only layer to
  `self.self_attention.linear_proj`, which are IdentityOps there (AttributeError, finding
  O8a-N7). No open PR changes it.

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
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_inference_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_inference_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik
from megatron.core.transformer.enums import InferenceCudaGraphScope
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_block import TransformerBlock, TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.test_utilities import Utils

pytestmark = [ou.requires_sm(ou.HOPPER)]

_DEVICE = "cuda"
SEQ, BATCH, HIDDEN, FFN, HEADS, LAYERS = 64, 2, 128, 256, 4, 3

# (case, tensor-parallel size, flag on): every path the row-parallel layer can take.
CASES = [
    ("flag_off", 1, False),
    ("tp1", 1, True),
    ("flag_off_tp2", 2, False),
    ("nvls", 2, True),
    ("nvls_disabled", 2, True),
    ("batch_invariant", 2, True),
    ("fp16", 2, True),
    ("small_buffer", 2, True),
    ("tp8_nvls", 8, True),
    ("tp8_nvls_disabled", 8, True),
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
    model_parallel_cuda_manual_seed(123)
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
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, (stage, branches, problems))
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
    if case == "small_buffer":
        # A symmetric buffer of 1 KiB holds no [S, B, H] tensor: every NVLS path is unavailable.
        monkeypatch.setattr(SymmetricMemoryManager, "_default_size_mb", 1.0 / 1024)
    block, config, dtype = build_block(tp, case, fuse)
    x_full = full_input(dtype)
    x = tp_shard(x_full)
    trace = _Trace(monkeypatch)
    outputs = []
    for _ in range(2):  # the second forward reuses the buffer the first one left
        _poison_tp_buffer(tp)
        trace.clear()
        with torch.no_grad(), InferenceMode.active():
            if case == "batch_invariant":
                with bik.set_batch_invariant_mode(True, backend="te_native"):
                    out = block(hidden_states=x.clone(), attention_mask=None)
            else:
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
PP_CASES = ["pp2_nvls", "pp2_nccl"]


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
    model_parallel_cuda_manual_seed(123)
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

# Token counts per step, in units of tp, in dynamic batching's [T, 1, H] layout: decode (1, 2),
# mixed prefill/decode (64, 65) and a long prefill (1024), in an order where each kind follows
# another.
STEP_TOKENS = (64, 1, 1024, 2, 65, 64, 1)
CAPACITY_TOKENS = 64  # in units of tp: the "capacity" buffers hold [64 tp, 1, H] bf16 exactly


def _capacity_bytes(tp):
    return tp * CAPACITY_TOKENS * HIDDEN * 2


@pytest.mark.parametrize("parallel", [2], indirect=True, ids=["tp2"])
@pytest.mark.parametrize("case", ["decode_prefill_sizes", "sizes_across_capacity"])
def test_step_sizes_keep_the_residual(case, parallel, monkeypatch):
    """Consecutive steps of different token counts on one model and one buffer. With the
    default buffer every step fuses; with a buffer of exactly 64 tp tokens the 65 tp and
    1024 tp steps fall back between fused steps (fails on main: the fallback drops the
    residual, #8561)."""
    tp = parallel
    capacity = _capacity_bytes(tp) if case == "sizes_across_capacity" else None
    if capacity is not None:
        monkeypatch.setattr(SymmetricMemoryManager, "_default_size_mb", capacity / 2**20)
    block, config, dtype = build_block(tp, "nvls", True)
    trace = _Trace(monkeypatch)
    eps = config.layernorm_epsilon
    problems, branches = [], []
    for step, units in enumerate(STEP_TOKENS):
        tokens = tp * units
        x_full = full_input(dtype, (tokens, 1, HIDDEN), seed=step)
        x = tp_shard(x_full)
        _poison_tp_buffer(tp)
        trace.clear()
        with torch.no_grad(), InferenceMode.active():
            out = block(hidden_states=x.clone(), attention_mask=None)
        torch.cuda.synchronize()
        what = f"step {step} (T = {tokens}): "
        fits = capacity is None or tokens * HIDDEN * 2 <= capacity
        _check_exact(out, x, f"{what}block output", problems)
        _check_gemm_inputs(block.layers, trace, x_full, eps, problems, what=what)
        _check_fused_norms(block.layers, trace.fused_norms, fits, problems, what)
        branches.append(list(trace.branches))
    _verdict(case, branches, problems)


# ---- CUDA graphs: the block (full-iteration) or each layer (local) captured -------------------

GRAPH_CASES = ["full_iteration", "full_iteration_capacity", "local", "local_capacity"]


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
    an eager step at 32 tp tokens between replays. The *_capacity variants hold exactly 64 tp
    tokens in the buffer and use 128 tp for the second size, so the two decide differently
    (fail on main: the fallback drops the residual, #8561).
    """
    tp = parallel
    capacity = _capacity_bytes(tp) if case.endswith("_capacity") else None
    if capacity is not None:
        monkeypatch.setattr(SymmetricMemoryManager, "_default_size_mb", capacity / 2**20)
    sizes = (tp * 64, tp * (128 if capacity is not None else 32))
    block, config, dtype = build_block(tp, "nvls", True)
    trace = _Trace(monkeypatch)
    eps = config.layernorm_epsilon
    problems, branches = [], []
    pool = torch.cuda.graph_pool_handle()

    def fits(tokens):
        return capacity is None or tokens * HIDDEN * 2 <= capacity

    def step_input(tokens, seed):
        x_full = full_input(dtype, (tokens, 1, HIDDEN), seed=seed)
        return x_full, tp_shard(x_full)

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
                _check_fused_norms(block.layers, trace.fused_norms, fits(tokens), problems, what)
                branches.append(list(trace.branches))
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
                    h, _ = layer(hidden_states=h, attention_mask=None)
                graphs.append(graph)
                outs.append(h)
            what = f"capture T = {graph_tokens}: "
            _check_fused_norms(block.layers, trace.fused_norms, fits(graph_tokens), problems, what)
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
                _check_fused_norms(
                    block.layers, trace.fused_norms, fits(eager_tokens), problems, what
                )
                branches.append(list(trace.branches))
    _verdict(case, branches, problems)


# ---- The engine: CudaGraphManager graphs, prefill, decode and mixed steps ---------------------

ENGINE_CASES = ["eager", "local", "full_iteration", "full_iteration_capacity"]
ENGINE_VOCAB, ENGINE_MAX_SEQ = 128, 1024
# (arrival step, prompt length): two short prompts, then a 600-token prompt (a mixed step above
# cuda_graph_max_tokens = 512, so eager) and a 130-token one (a mixed step that is graphed).
ENGINE_PROMPTS = ((0, 9), (0, 37), (1, 600), (2, 130))
ENGINE_CAPACITY_TOKENS = 64  # graphs of up to 64 padded tokens fuse, larger ones fall back


@pytest.mark.parametrize("parallel", [2], indirect=True, ids=["tp2"])
@pytest.mark.parametrize("case", ENGINE_CASES)
def test_engine_steps_keep_the_residual(case, parallel, monkeypatch):
    """A GPT model with zero projections served by the dynamic engine: in every forward (eager,
    replayed from per-layer graphs, or from block graphs) the decoder's output is its input.
    The decoder's final norm is removed so its output is the stream; hooks on the embedding
    (which also NaN-poisons the buffer) and the output layer, which run eagerly around graphed
    decoders, compare the two. With `full_iteration_capacity` the buffer holds 64 tokens, so the
    small (decode) graphs fuse and the larger (mixed) ones do not (fails on main, #8561)."""
    tp = parallel
    graphs = case != "eager"
    if case.endswith("_capacity"):
        size_mb = ENGINE_CAPACITY_TOKENS * HIDDEN * 2 / 2**20
        monkeypatch.setattr(SymmetricMemoryManager, "_default_size_mb", size_mb)
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
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_inference_spec(),
        vocab_size=ENGINE_VOCAB,
        max_sequence_length=ENGINE_MAX_SEQ,
        parallel_output=True,
        pre_process=True,
        post_process=True,
    )
    model = model.cuda().eval()
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
        _check_exact(
            args[0], inputs.pop(), f"decoder output, forward {counts['forwards']}", problems
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


# ---- Layers outside TransformerBlock's wiring: MTP, HybridStack -------------------------------

MTP_CASES = [("tp1", 1, 1), ("tp2_nvls", 2, 1), ("tp2_nvls_depth2", 2, 2)]


@pytest.mark.parametrize(
    "case,parallel,depth", MTP_CASES, ids=[c[0] for c in MTP_CASES], indirect=["parallel"]
)
def test_mtp_inner_layer_keeps_the_residual(case, parallel, depth, monkeypatch):
    """The MTP inner layer (the GPT inference spec's layer, built by MultiTokenPredictionLayer and
    never wired) run through `forward_single_position`, as speculative decoding runs it: its
    output stream is its input. Fails on main: the flag makes it hand its residual to row-parallel
    layers that add it only inside a fused kernel they never run (#8561)."""
    tp = parallel
    config, dtype = make_config(tp, "nvls", True, mtp_num_layers=2)
    model_parallel_cuda_manual_seed(123)
    spec = get_gpt_layer_with_inference_spec()
    mtp_spec = get_gpt_mtp_block_spec(config, spec, use_transformer_engine=True)
    mtp = build_module(
        mtp_spec.layer_specs[depth - 1],
        config=config,
        layer_number=depth,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )
    mtp = mtp.cuda().eval()
    inner = mtp.mtp_model_layer
    zero_projections_([inner])
    seen = {}
    inner.register_forward_pre_hook(
        lambda m, args, kwargs: seen.__setitem__("in", kwargs["hidden_states"].detach().clone()),
        with_kwargs=True,
    )
    inner.register_forward_hook(
        lambda m, args, kwargs, out: seen.__setitem__("out", out[0].detach().clone()),
        with_kwargs=True,
    )
    hidden = tp_shard(full_input(dtype, seed=1234))
    decoder_input = tp_shard(full_input(dtype, seed=5678))
    ids = torch.zeros(BATCH, SEQ, dtype=torch.int64, device=_DEVICE)
    positions = torch.arange(SEQ, device=_DEVICE).expand(BATCH, SEQ)

    def embedding(input_ids, position_ids):
        return decoder_input.clone()

    trace = _Trace(monkeypatch)
    fc1_norm = inner.mlp.linear_fc1.layer_norm_weight.data_ptr()
    problems, branches = [], []
    for j in range(2):
        _poison_tp_buffer(tp)
        trace.clear()
        seen.clear()
        # torch.inference_mode, as LanguageModule.compute_mtp_single_step runs it.
        with torch.inference_mode(), InferenceMode.active():
            mtp.forward_single_position(
                hidden_states=hidden.clone(),
                next_token_ids=ids,
                position_ids=positions,
                embedding=embedding,
            )
        torch.cuda.synchronize()
        stream = seen["in"]
        _check_exact(seen["out"], stream, f"inner layer output, forward {j}", problems)
        stream_full = stream
        if tp > 1:
            stream_full = torch.empty(
                (stream.shape[0] * tp, *stream.shape[1:]), dtype=stream.dtype, device=_DEVICE
            )
            dist.all_gather_into_tensor(
                stream_full, stream, group=parallel_state.get_tensor_model_parallel_group()
            )
        _check_gemm_inputs([inner], trace, stream_full, config.layernorm_epsilon, problems)
        stray = [p for p in trace.fused_norms if p != fc1_norm]
        if stray:
            problems.append(f"forward {j}: fused kernel applied a norm other than FC1's: {stray}")
        branches.append(list(trace.branches))
    _verdict(case, branches, problems)


HYBRID_PATTERN = "*-*-"  # attention-only and MLP-only layers, as a HybridStack builds them
HYBRID_CASES = [("tp1", 1), ("tp2_nvls", 2)]


@pytest.mark.parametrize(
    "case,parallel", HYBRID_CASES, ids=[c[0] for c in HYBRID_CASES], indirect=["parallel"]
)
def test_hybrid_stack_keeps_the_residual(case, parallel, monkeypatch):
    """A HybridStack of attention-only and MLP-only TransformerLayers (the hybrid inference spec)
    with the flag on, as the Core API accepts it: its output stream is its input. Fails on main
    and, by reading, with #8561: the layers hand their residual to an IdentityOp's missing
    linear_fc2 / linear_proj (AttributeError, O8a-N7)."""
    tp = parallel
    config, dtype = make_config(tp, "nvls", True, num_layers=len(HYBRID_PATTERN))
    model_parallel_cuda_manual_seed(123)
    stack = HybridStack(
        config,
        hybrid_inference_stack_spec.submodules,
        layer_config_list=validate_segment_layers(HYBRID_PATTERN, config),
        pp_layer_offset=0,
        post_layer_norm=False,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )
    stack = stack.cuda().eval()
    zero_projections_(stack.layers)
    x_full = full_input(dtype)
    x = tp_shard(x_full)
    trace = _Trace(monkeypatch)
    real = {
        lin.layer_norm_weight.data_ptr()
        for layer in stack.layers
        for _, lin in norm_consumers(layer)
    }
    problems, branches = [], []
    for j in range(2):
        _poison_tp_buffer(tp)
        trace.clear()
        with torch.no_grad(), InferenceMode.active():
            out = stack(hidden_states=x.clone(), attention_mask=None)
        torch.cuda.synchronize()
        _check_exact(out, x, f"stack output, forward {j}", problems)
        _check_gemm_inputs(stack.layers, trace, x_full, config.layernorm_epsilon, problems)
        stray = [p for p in trace.fused_norms if p not in real]
        if stray:
            problems.append(f"forward {j}: fused kernel applied no consumer's norm: {stray}")
        branches.append(list(trace.branches))
    _verdict(case, branches, problems)


# ---- MLA --------------------------------------------------------------------------------------

# (case, layer pattern: M = MLA, G = GPT, TP).
MLA_CASES = [("mla_tp1", "MM", 1), ("mla_tp2_nvls", "MM", 2), ("mixed_tp2_nvls", "GMG", 2)]


@pytest.mark.parametrize(
    "case,pattern,parallel", MLA_CASES, ids=[c[0] for c in MLA_CASES], indirect=["parallel"]
)
def test_mla_layers_keep_the_residual(case, pattern, parallel, monkeypatch):
    """MLA blocks (DeepSeek-V3 style: q_lora_rank, qk_layernorm) with the flag on, which config
    validation accepts: the output is the input, every FC1 (and GPT QKV) GEMM input is its own
    norm of the stream, each MLA attention projection fuses with its own FC1 norm and an FC2
    fuses only into a next GPT layer's QKV norm. Fails on main at construction (MLA has no
    linear_qkv for the block's wiring); #8704 fixes it."""
    tp = parallel
    config, dtype = make_config(
        tp,
        "nvls",
        True,
        config_class=MLATransformerConfig,
        num_layers=len(pattern),
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_head_dim=64,
        qk_pos_emb_head_dim=64,
        v_head_dim=128,
        qk_layernorm=True,
        rope_type="rope",
    )
    model_parallel_cuda_manual_seed(123)
    specs = {
        kind: get_gpt_layer_with_inference_spec(
            qk_layernorm=True, multi_latent_attention=kind == "M"
        )
        for kind in set(pattern)
    }
    submodules = TransformerBlockSubmodules(layer_specs=[specs[kind] for kind in pattern])
    block = TransformerBlock(config, submodules, post_layer_norm=False)
    block = block.cuda().eval()
    for kind, layer in zip(pattern, block.layers):
        if kind == "M":
            # The query and KV up projections all-gather back to back into the shared buffer
            # without a barrier (#8709, n12): a separate race, so take them through NCCL.
            for name in ("linear_q_up_proj", "linear_kv_up_proj"):
                proj = getattr(layer.self_attention, name, None)
                if proj is not None:
                    proj.triton_nvls_kernels_allowed = False
    zero_projections_(block.layers)
    x_full = full_input(dtype)
    x = tp_shard(x_full)
    trace = _Trace(monkeypatch)
    eps = config.layernorm_epsilon
    problems, branches = [], []
    for j in range(2):
        _poison_tp_buffer(tp)
        trace.clear()
        with torch.no_grad(), InferenceMode.active():
            out = block(hidden_states=x.clone(), attention_mask=None)
        torch.cuda.synchronize()
        what = f"forward {j}: "
        _check_exact(out, x, f"{what}block output", problems)
        _check_gemm_inputs(block.layers, trace, x_full, eps, problems, what=what)
        _check_fused_norms(block.layers, trace.fused_norms, tp > 1, problems, what)
        branches.append(list(trace.branches))
    _verdict(case, branches, problems)
