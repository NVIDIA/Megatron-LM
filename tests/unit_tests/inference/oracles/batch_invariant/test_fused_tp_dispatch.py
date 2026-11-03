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
  `linear_qkv` norm; FC1: its `linear_fc1` norm).
- A consumer reads the symmetric buffer only when this forward's producer wrote it.
- Dispatch (exact): a row-parallel layer runs the fused NVLS kernel if and only if TP > 1, the
  Triton NVLS kernels are allowed, activations are bf16, the sizes are NVLS-eligible, the
  symmetric buffer holds the tensor and batch-invariant mode is off; every rank takes the same
  sequence of branches.

Oracles:
- Exact metamorphic: with every `linear_proj` and `linear_fc2` weight zero, each sublayer adds 0,
  so the block's output must equal its input bit for bit on every path (the fused kernel
  computes bf16(0 + R) = R). A dropped residual gives 0, a doubled one 2R. The `"tp"` symmetric
  buffer is NaN-poisoned before every forward, so a stale read turns the output NaN.
- fp64 reference for the GEMM inputs: each QKV / FC1 GEMM input, recorded at `_apply_linear`,
  against RMSNorm(input; its own norm weight) in fp64 (bounded by an fp32-computed RMSNorm
  rounded to the activation dtype), with distinct norm gains per layer.
- Dispatch: spies on the fused kernel, the NVLS reduce-scatter and NCCL's; fused calls counted
  exactly; traces gathered across ranks must be identical.
Tolerance class: exact for the output, the dispatch and the traces; floating point (default
factor, per row) for the GEMM inputs.

Structure (read after the contract): `_matmul_reduce_scatter` takes the NVLS branch on
triton_nvls_kernels_allowed, bf16, `are_tensors_nvls_eligible` (16-byte multiple, SM >= 9), a
buffer handle and not batch-invariant mode, and fuses when the next norm weights are set; the
layer skips its own residual add whenever the flag is on. Covered: TP = 1 and 2 (both with the
flag off as controls), NVLS kernels disabled, batch-invariant mode, fp16, a symmetric buffer too
small for the tensor, 3 layers (first, middle, last). Not covered: PP stage boundaries, MTP and
HybridStack layers, MLA (#8704), CUDA graphs, TP = 8.

Main vs open PRs: on main a row-parallel layer adds the residual only inside the fused kernel,
and the consumer decides fused mode from its own flag and the buffer handle, so TP = 1, NVLS
disabled, batch-invariant mode, fp16 and the small buffer drop the residual (and the first three
with a buffer also read it stale): `test_zero_projections_keep_the_residual` fails there for
every case except "nvls" and the flag-off controls. #8561 (d1) fixes both; #8712 (n16) and #8704
(n8) change the same wiring and must keep these tests passing; #8868 (t3-r04f3) changes the
batch-invariant branch to a rank-ordered reduce-scatter (still no residual add without d1).

Hardware: H100; one GPU for TP = 1, two NVLS-connected GPUs for TP = 2 (`nproc` 2), with
Transformer Engine.
"""

import inspect

import pytest
import torch
import torch.distributed as dist

import megatron.core.tensor_parallel.inference_layers as inference_layers
from megatron.core import parallel_state
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_inference_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
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
]
FUSED_EXPECTED = {"nvls"}


@pytest.fixture
def parallel(request):
    tp = request.param
    if Utils.world_size < tp:
        pytest.skip(f"needs {tp} ranks (run with nproc >= {tp}); the manifest selects it so")
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp)
    SymmetricMemoryManager.destroy("tp")
    yield tp
    SymmetricMemoryManager.destroy("tp")
    Utils.destroy_model_parallel()


def build_block(tp, case, fuse):
    dtype = torch.float16 if case == "fp16" else torch.bfloat16
    config = TransformerConfig(
        num_layers=LAYERS,
        hidden_size=HIDDEN,
        ffn_hidden_size=FFN,
        num_attention_heads=HEADS,
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
        inference_disable_triton_nvls_kernels=case == "nvls_disabled",
    )
    model_parallel_cuda_manual_seed(123)
    block = TransformerBlock(config, get_gpt_layer_with_inference_spec(), post_layer_norm=False)
    block = block.cuda().eval()
    g = torch.Generator(device="cpu").manual_seed(4321)
    with torch.no_grad():
        for layer in block.layers:
            layer.self_attention.linear_proj.weight.zero_()
            layer.mlp.linear_fc2.weight.zero_()
            for norm in (layer.self_attention.linear_qkv, layer.mlp.linear_fc1):
                # In-place, distinct per layer and sublayer: the fused wiring holds references.
                gains = 1.0 + 0.5 * torch.randn(norm.layer_norm_weight.shape, generator=g)
                norm.layer_norm_weight.copy_(gains)
    return block, config, dtype


def full_input(dtype):
    g = torch.Generator(device="cpu").manual_seed(1234)
    return torch.randn(SEQ, BATCH, HIDDEN, generator=g).to(dtype).to(_DEVICE)


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

    def input_of(self, weight):
        found = [x for w, x in self.gemm_inputs if w is weight]
        assert len(found) == 1, f"expected one GEMM with this weight, found {len(found)}"
        return found[0]


def _rms_norm_reference(x, w, eps):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * w


def _rms_norm_trusted(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


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
    rank = parallel_state.get_tensor_model_parallel_rank()
    x = x_full.chunk(tp, dim=0)[rank].contiguous()
    trace = _Trace(monkeypatch)
    outputs = []
    for _ in range(2):  # the second forward reuses the buffer the first one left
        _poison_tp_buffer(tp)
        trace.branches.clear()
        trace.fused_norms.clear()
        trace.gemm_inputs.clear()
        with torch.no_grad(), InferenceMode.active():
            if case == "batch_invariant":
                with bik.set_batch_invariant_mode(True, backend="te_native"):
                    out = block(hidden_states=x.clone(), attention_mask=None)
            else:
                out = block(hidden_states=x.clone(), attention_mask=None)
        torch.cuda.synchronize()
        outputs.append(out)

    # Judge locally, then share every rank's verdict before asserting (no rank may stop early).
    problems = []
    for j, out in enumerate(outputs):
        try:
            ou.assert_exact(out, x, what=f"block output, forward {j}")
        except AssertionError as e:
            problems.append(str(e))
    eps = config.layernorm_epsilon
    for i, layer in enumerate(block.layers):
        for name, linear in (
            ("qkv", layer.self_attention.linear_qkv),
            ("fc1", layer.mlp.linear_fc1),
        ):
            w = linear.layer_norm_weight.detach()
            try:
                got = trace.input_of(linear.weight).reshape(SEQ, BATCH, HIDDEN)
                ou.assert_close_to_reference(
                    got,
                    ou.fp64_reference(_rms_norm_reference, x_full, w, eps),
                    _rms_norm_trusted(x_full, w, eps),
                    dim=-1,
                    what=f"layer {i} {name} GEMM input",
                )
            except AssertionError as e:
                problems.append(str(e))
    # The fused kernel must run exactly on the row-parallel layers that feed a consumer, with
    # that consumer's norm weight: layer i's attention projection with its own FC1 norm, layer
    # i's FC2 with layer i+1's QKV norm. The last FC2 has no consumer here; whether it fuses is
    # not part of the contract (main fuses it with a placeholder weight, #8704 does not).
    if case in FUSED_EXPECTED:
        want = []
        for i, layer in enumerate(block.layers):
            want.append(layer.mlp.linear_fc1.layer_norm_weight.data_ptr())
            if i + 1 < len(block.layers):
                want.append(
                    block.layers[i + 1].self_attention.linear_qkv.layer_norm_weight.data_ptr()
                )
        got_norms = trace.fused_norms
        if got_norms[: len(want)] != want or len(got_norms) > len(want) + 1:
            problems.append(f"fused kernel norm weights {got_norms}, expected {want} (+ last FC2)")
    elif trace.fused_norms:
        problems.append(f"fused kernel ran {len(trace.fused_norms)} times; preconditions fail")

    group = parallel_state.get_tensor_model_parallel_group()
    gathered = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, (trace.branches, problems), group=group)
    assert all(
        b == gathered[0][0] for b, _ in gathered
    ), f"ranks took different branches: {gathered}"
    failures = [f"rank {r}: {p}" for r, (_, ps) in enumerate(gathered) for p in ps]
    assert not failures, f"{case}:\n" + "\n".join(failures)
