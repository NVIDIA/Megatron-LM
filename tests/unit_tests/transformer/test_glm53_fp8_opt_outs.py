# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GLM-5.3 FP8 opt-outs are bitwise inert under the target blockwise recipe.

The GLM53 model configuration keeps every KDA projection (``kda_disable_fp8``),
every absorbed-MLA projection of DSA layers (``mla_proj_use_quantization=False``),
the DSA indexer Q/K and weights projections
(``dsa_indexer_{qk,weights}_proj_use_quantization=False``) and the KPool score
inputs (``dsa_indexer_kpool_use_quantization=False``) outside FP8. Each mixer is
built from ``hybrid_glm53_stack_spec`` with FP8 disabled and again with the target
recipe (E4M3, blockwise scaling, FP8 primary weights). The FP8 build and forward
use Megatron's own parameter-initialization and forward quantization contexts
(``get_fp8_context``), as ``HybridStack`` applies them per layer. From identical
parameters, inputs and output cotangent, the output, the input gradient, every
parameter gradient and the output of every executed projection must be bitwise
identical, and no parameter of the FP8 build may be a Transformer Engine quantized
tensor. Projection outputs are compared directly because the frozen DSA indexer
feeds only a discrete KPool top-k: a quantized indexer projection changes the
mixer output only when it flips a pool ranking.

Non-vacuity: a pre-hook proves the enclosing context really is blockwise FP8, and
forward hooks read the quantization each projection's Transformer Engine module
itself applied (``fp8`` and the recipe set by ``prepare_forward``), so a module
that re-enters FP8 inside its own forward is caught. A spy on the KPool selector
records its ``fp8_indexer`` flag and counts KPool FP8 input conversions. A BF16
control instance runs first as a warm-up, and its result tells a failure caused by
kernel nondeterminism apart from one caused by FP8. Negative controls (1) flip each
TE-module opt-out at construction and require exactly its projections to become
FP8 primary weights; (2) let the MLA projections, the indexer Q/K projections, or
(through a regressed ``KimiDeltaAttention._proj``) the KDA input and output
projections execute in FP8, and require the TE flags and the same bitwise
comparator to report those projections' outputs (and, for MLA and KDA, the mixer
output and gradients) as changed, after an identical BF16 instance reproduced the
reference bit for bit; (3) enable KPool quantization and require the spy to see it.

Scope: TP=PP=EP=CP=1 mixer modules in the dev proxy geometry (hidden 1024, eight
attention, indexer and KDA heads). DSA uses the PyTorch sparse-attention path,
whose masked dense GEMMs are run-to-run reproducible. The fused cuDNN backward
scatters latent K/V gradients into rows selected by many queries, so bitwise
reproducibility is not established for it; the attention kernel choice is
independent of projection precision. The indexer weights and KDA beta projections
have no runtime FP8 control: their eight output features are below the 16-element
FP8 GEMM alignment (``get_fp8_align_size``). Their opt-outs are checked at
construction, through their TE execution flags and by direct output comparison.
"""

import dataclasses
import inspect
import os

import pytest
import torch
import torch.nn.functional as F
from transformer_engine.common import recipe
from transformer_engine.pytorch.fp8 import (
    FP8GlobalStateManager,
    check_fp8_block_scaling_support,
    check_fp8_support,
)
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule

from megatron.core.fp8_utils import get_fp8_context, is_float8tensor
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_glm53_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import KimiDeltaAttention
from megatron.core.ssm.gated_delta_net import kda as kda_module
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.experimental_attention_variant import dsa as dsa_module
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
)
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.training.utils import get_device_arch_version
from tests.unit_tests.test_utilities import Utils

_SEED = 20261015

# Dev proxy geometry (configs/dev/models/glm5_3_flash_proxy.yaml).
_HIDDEN = 1024
_HEADS = 8
_Q_LORA_RANK = 512
_KV_LORA_RANK = 512
_ATTENTION_HEAD_DIM = 256
_INDEXER_HEADS = 8
_INDEXER_HEAD_DIM = 128
_KPOOL = 4
_KDA_HEADS = 8
_KDA_HEAD_DIM = 128

# 128 tokens span two FLA chunks and are a whole number of 128-element FP8 blocks.
# The proxy selects 128 of 512 tokens; 32 of 128 keeps that 1/4 ratio, so the
# frozen KPool selector prunes eight of 32 pools rather than selecting everything.
_SEQUENCE = 128
_BATCH = 1
_INDEXER_TOPK = 32

# Target precision of configs/dev/training/glm53_flash_fp8_readiness.yaml.
_TARGET_FP8 = {"fp8": "e4m3", "fp8_recipe": "blockwise", "fp8_param": True}
# Runtime controls keep BF16 primary weights so that only the forward quantization
# context under test differs; an FP8 K/V-up weight would additionally be dequantized
# by the absorption einsum's view.
_RUNTIME_FP8 = {**_TARGET_FP8, "fp8_param": False}

_KDA_PROJECTIONS = (
    "in_proj",
    "beta_proj",
    "f_a_proj",
    "f_b_proj",
    "g_a_proj",
    "g_b_proj",
    "out_proj",
)
# 3072 and 1024 output features; every other KDA projection stays opted out.
_KDA_REGRESSED_PROJECTIONS = ("in_proj", "out_proj")
_MLA_PROJECTIONS = (
    "linear_q_down_proj",
    "linear_q_up_proj",
    "linear_kv_down_proj",
    "linear_kv_up_proj",
    "linear_proj",
)
_INDEXER_PREFIX = "core_attention.indexer."
_INDEXER_QK_PROJECTIONS = (f"{_INDEXER_PREFIX}linear_wq_b", f"{_INDEXER_PREFIX}linear_wk")
_INDEXER_WEIGHTS_PROJECTIONS = (f"{_INDEXER_PREFIX}linear_weights_proj",)
_DSA_PROJECTIONS = _MLA_PROJECTIONS + _INDEXER_QK_PROJECTIONS + _INDEXER_WEIGHTS_PROJECTIONS
# Absorbed MLA reads the K/V up-projection weight directly in its einsums and never
# calls that module as a GEMM; its precision is covered by the parameter checks.
_MLA_EXECUTED_PROJECTIONS = tuple(name for name in _MLA_PROJECTIONS if name != "linear_kv_up_proj")
_DSA_EXECUTED_PROJECTIONS = tuple(name for name in _DSA_PROJECTIONS if name != "linear_kv_up_proj")

_BIT_VIEWS = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}

_HAS_CUDA = torch.cuda.is_available()
if _HAS_CUDA:
    _FP8_AVAILABLE, _NO_FP8_REASON = check_fp8_support()
    _FP8_BLOCK_AVAILABLE, _NO_FP8_BLOCK_REASON = check_fp8_block_scaling_support()
else:
    _FP8_AVAILABLE = _FP8_BLOCK_AVAILABLE = False
    _NO_FP8_REASON = _NO_FP8_BLOCK_REASON = "CUDA required"

pytestmark = [
    pytest.mark.internal,
    pytest.mark.skipif(not _HAS_CUDA, reason="CUDA is required."),
    pytest.mark.skipif(
        _HAS_CUDA and get_device_arch_version() != 9,
        reason="blockwise is only supported on Hopper architecture",
    ),
    pytest.mark.skipif(not _FP8_AVAILABLE, reason=_NO_FP8_REASON),
    pytest.mark.skipif(not _FP8_BLOCK_AVAILABLE, reason=_NO_FP8_BLOCK_REASON),
    pytest.mark.skipif(not is_te_min_version("2.4.0.dev0"), reason="TE 2.4.0.dev0 is required"),
]

_requires_fla_kda = pytest.mark.skipif(
    not kda_module.HAVE_FLA_KDA or kda_module.rms_norm_gated is None,
    reason="FLA with chunk KDA and fused RMSNorm-gate support is required.",
)


def _config(**overrides) -> MLATransformerConfig:
    """GLM53 proxy model configuration with FP8 disabled unless overridden."""
    values = dict(
        num_layers=1,
        hidden_size=_HIDDEN,
        num_attention_heads=_HEADS,
        num_query_groups=_HEADS,
        kv_channels=_ATTENTION_HEAD_DIM,
        q_lora_rank=_Q_LORA_RANK,
        kv_lora_rank=_KV_LORA_RANK,
        qk_head_dim=_ATTENTION_HEAD_DIM,
        qk_pos_emb_head_dim=0,
        v_head_dim=_ATTENTION_HEAD_DIM,
        qk_layernorm=True,
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        attention_latent_norm_epsilon=1e-5,
        rmsnorm_cast_before_weight=True,
        activation_func=F.silu,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=False,
        gradient_accumulation_fusion=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        attention_backend=AttnBackend.unfused,
        dsa_kernel_backend="none",
        experimental_attention_variant="dsa",
        dsa_indexer_n_heads=_INDEXER_HEADS,
        dsa_indexer_head_dim=_INDEXER_HEAD_DIM,
        dsa_indexer_topk=_INDEXER_TOPK,
        dsa_indexer_kpool=_KPOOL,
        dsa_indexer_kpool_use_quantization=False,
        dsa_indexer_kpool_always_select_tail=True,
        dsa_indexer_kpool_key_dtype_compression=True,
        dsa_indexer_hf_scoring=True,
        dsa_indexer_loss_coeff=0.0,
        dsa_indexer_k_norm_epsilon=1e-6,
        dsa_indexer_scoring_relu=True,
        dsa_indexer_rotate_activation=False,
        dsa_indexer_qk_proj_use_quantization=False,
        dsa_indexer_weights_proj_use_quantization=False,
        dsa_indexer_weights_proj_output_dtype="bf16",
        mla_proj_use_quantization=False,
        rope_type="rope",
        rotary_scaling_factor=1.0,
        mscale_all_dim=0.0,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=_KDA_HEAD_DIM,
        linear_value_head_dim=_KDA_HEAD_DIM,
        linear_num_key_heads=_KDA_HEADS,
        linear_num_value_heads=_KDA_HEADS,
        kda_two_stage_gates=True,
        kda_disable_fp8=True,
        kda_conv_in_fp32=True,
        kda_core_in_fp32=True,
        kda_safe_gate=True,
        kda_lower_bound=-5.0,
        deterministic_mode=False,
        enable_hyper_connections=True,
        num_residual_streams=4,
        mhc_sinkhorn_iterations=20,
        mhc_norm_eps_inside_sqrt=True,
        mhc_keep_mappings_in_fp32=False,
        mhc_keep_pre_in_fp32=True,
        mhc_preserve_activation_dtype_rounding=True,
        mhc_learned_output_contract=False,
        is_hybrid_model=True,
        transformer_impl="transformer_engine",
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
    )
    values.update(overrides)
    return MLATransformerConfig(**values)


def _process_groups() -> ProcessGroupCollection:
    return ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])


def _build_kda(config: MLATransformerConfig) -> KimiDeltaAttention:
    """Build the GLM53 ``K`` mixer exactly as the hybrid GLM53 spec declares it."""
    spec = hybrid_glm53_stack_spec(config).submodules.kda_layer.submodules.self_attention
    layer = build_module(spec, config=config, layer_number=1, pg_collection=_process_groups())
    assert isinstance(layer, KimiDeltaAttention)
    assert layer.two_stage_gates and layer.core_in_fp32 is config.kda_core_in_fp32
    return layer


def _build_dsa(config: MLATransformerConfig) -> AbsorbedMLASelfAttention:
    """Build the GLM53 ``D`` mixer: NoPE absorbed MLA with a frozen KPool DSA selector."""
    spec = hybrid_glm53_stack_spec(config).submodules.dsa_layer.submodules.self_attention
    layer = build_module(spec, config=config, layer_number=1, pg_collection=_process_groups())
    assert isinstance(layer, AbsorbedMLASelfAttention)
    assert not layer.use_rope
    assert layer.core_attention.indexer.index_kpool == _KPOOL
    return layer


# kind -> (builder, every TE base module, modules executed as GEMMs, frozen prefix)
_LAYERS = {
    "kda": (_build_kda, _KDA_PROJECTIONS, _KDA_PROJECTIONS, None),
    "dsa": (_build_dsa, _DSA_PROJECTIONS, _DSA_EXECUTED_PROJECTIONS, _INDEXER_PREFIX),
}


@pytest.fixture
def model_parallel():
    """TP=PP=EP=CP=1 on every rank; teardown is collective, so it always runs."""
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    try:
        yield
    finally:
        Utils.destroy_model_parallel()


class _KPoolSpy:
    """Record each KPool selection's ``fp8_indexer`` flag and every FP8 input conversion.

    Both names are module globals of ``dsa`` and every caller looks them up there.
    ``substitute_quantizer`` replaces the conversion with its dtype-only effect, so a
    routing control does not need ``fast_hadamard_transform``.
    """

    def __init__(self, monkeypatch, *, substitute_quantizer: bool = False):
        self.selections: list[bool] = []
        self.quantized_inputs = 0
        select = dsa_module.fused_qk_topk_kpool
        quantize = dsa_module._kpool_fp8_input
        signature = inspect.signature(select)

        def _select(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            self.selections.append(bound.arguments["fp8_indexer"])
            return select(*args, **kwargs)

        def _quantize(x):
            self.quantized_inputs += 1
            return x.float() if substitute_quantizer else quantize(x)

        monkeypatch.setattr(dsa_module, "fused_qk_topk_kpool", _select)
        monkeypatch.setattr(dsa_module, "_kpool_fp8_input", _quantize)


@dataclasses.dataclass
class _Run:
    """Every compared tensor of one forward/backward and how each part was quantized."""

    results: dict
    outer_states: list
    projection_states: dict
    kpool_selections: list
    kpool_quantized_inputs: int


@dataclasses.dataclass
class _Comparison:
    """A BF16 control (run first), a BF16 reference and a candidate from equal state."""

    reference: torch.nn.Module
    candidate: torch.nn.Module
    trainable: set
    repeated: _Run
    expected: _Run
    actual: _Run


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    """Reinterpret floating-point storage as integers, so equality is bitwise."""
    return tensor.detach().contiguous().view(_BIT_VIEWS[tensor.dtype])


def _quantization_state() -> tuple[bool, type | None]:
    if not FP8GlobalStateManager.is_fp8_enabled():
        return False, None
    return True, type(FP8GlobalStateManager.get_fp8_recipe())


def _te_execution_state(module: TransformerEngineBaseModule) -> tuple[bool, type | None]:
    """Quantization TE itself applied in the module's latest forward (``prepare_forward``)."""
    if not module.fp8:
        return False, None
    return True, type(module.fp8_meta["recipe"])


def _state_recorder(sink: list):
    """Forward pre-hook recording the global quantization state a mixer is entered under."""

    def _record(module, args):
        del module, args
        sink.append(_quantization_state())

    return _record


def _projection_recorder(name: str, states: list, outputs: dict):
    """Forward hook recording TE's own execution state and a copy of the output."""

    def _record(module, args, output):
        del args
        states.append(_te_execution_state(module))
        tensor = output[0] if isinstance(output, tuple) else output
        outputs[f"out:{name}#{len(states) - 1}"] = tensor.detach().clone()

    return _record


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(_SEED)
    shape = (_SEQUENCE, _BATCH, _HIDDEN)
    hidden = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    cotangent = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    return hidden, cotangent


def _assert_quantizable_modules(layer: torch.nn.Module, projections: tuple[str, ...]) -> None:
    """Every TE base module (each module running a quantizable GEMM) is a named projection.

    TE's fusible-op norms are not base modules; run standalone they have no consumer
    input quantizer and return high-precision outputs inside or outside FP8 autocast.
    """
    actual = {
        name
        for name, module in layer.named_modules()
        if isinstance(module, TransformerEngineBaseModule)
    }
    assert actual == set(projections), f"TE modules {sorted(actual)} != {sorted(projections)}"


def _assert_high_precision_parameters(
    candidate: torch.nn.Module, reference: torch.nn.Module
) -> None:
    """No FP8 primary weight, and every parameter keeps the BF16 build's dtype."""
    for name, module in candidate.named_modules():
        if isinstance(module, TransformerEngineBaseModule):
            assert not module.primary_weights_in_fp8, f"{name} holds FP8 primary weights"
    reference_parameters = dict(reference.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    assert candidate_parameters.keys() == reference_parameters.keys()
    for name, parameter in candidate_parameters.items():
        assert not is_float8tensor(parameter), f"{name} is a TE quantized tensor"
        assert parameter.dtype == reference_parameters[name].dtype, name
        assert parameter.device.type == "cuda", name


def _copy_parameters(source: torch.nn.Module, target: torch.nn.Module) -> None:
    source_parameters = dict(source.named_parameters())
    target_parameters = dict(target.named_parameters())
    assert target_parameters.keys() == source_parameters.keys()
    with torch.no_grad():
        for name, parameter in target_parameters.items():
            assert not is_float8tensor(parameter), name
            parameter.copy_(source_parameters[name])
    for name, parameter in target_parameters.items():
        assert torch.equal(_bits(parameter), _bits(source_parameters[name])), name


def _trainable_parameters(layer: torch.nn.Module, frozen_prefix: str | None) -> set[str]:
    """Return trainable names; the frozen KPool selector must not require gradients."""
    names = {name for name, _ in layer.named_parameters()}
    frozen = {name for name in names if frozen_prefix and name.startswith(frozen_prefix)}
    if frozen_prefix:
        assert frozen, f"no parameters under {frozen_prefix}"
    requires_grad = {
        name for name, parameter in layer.named_parameters() if parameter.requires_grad
    }
    assert requires_grad == names - frozen
    return names - frozen


def _forward_backward(layer, config, hidden, cotangent, projections, kpool_spy=None) -> _Run:
    """Run one forward under ``get_fp8_context(config)`` and one backward outside it.

    Returns every result tensor by label (mixer output, input and parameter gradients,
    and each executed projection's output per call) plus the quantization observed by
    the whole mixer, by each projection's TE module and by the KPool selector.
    """
    outer_states = []
    projection_states = {name: [] for name in projections}
    projection_outputs = {}
    modules = dict(layer.named_modules())
    handles = [layer.register_forward_pre_hook(_state_recorder(outer_states))]
    handles.extend(
        modules[name].register_forward_hook(
            _projection_recorder(name, projection_states[name], projection_outputs)
        )
        for name in projections
    )
    selections_before = len(kpool_spy.selections) if kpool_spy is not None else 0
    quantized_before = kpool_spy.quantized_inputs if kpool_spy is not None else 0
    layer_input = hidden.detach().clone().requires_grad_(True)
    try:
        with get_fp8_context(config):
            output, bias = layer(layer_input, attention_mask=None)
    finally:
        for handle in handles:
            handle.remove()
    assert bias is None
    assert output.shape == hidden.shape and output.dtype == hidden.dtype
    output.backward(cotangent)
    results = {"output": output.detach(), "input_grad": layer_input.grad}
    results.update((f"grad:{name}", parameter.grad) for name, parameter in layer.named_parameters())
    results.update(projection_outputs)
    if kpool_spy is None:
        kpool_selections, kpool_quantized_inputs = [], 0
    else:
        kpool_selections = kpool_spy.selections[selections_before:]
        kpool_quantized_inputs = kpool_spy.quantized_inputs - quantized_before
    return _Run(
        results=results,
        outer_states=outer_states,
        projection_states=projection_states,
        kpool_selections=kpool_selections,
        kpool_quantized_inputs=kpool_quantized_inputs,
    )


def _assert_quantization_states(run: _Run, *, outer_fp8, fp8_projections):
    """Compare observed TE quantization states with the expected execution contract."""
    blockwise = (True, recipe.Float8BlockScaling)
    expected_outer = blockwise if outer_fp8 else (False, None)
    assert run.outer_states == [expected_outer], f"enclosing context: {run.outer_states}"
    for name, observed in run.projection_states.items():
        assert observed, f"{name} was never executed"
        expected = blockwise if name in fp8_projections else (False, None)
        assert all(state == expected for state in observed), f"{name}: {observed}"


def _assert_informative(results: dict, trainable: set[str]) -> None:
    """Bitwise equality of zero or non-finite tensors would prove nothing."""
    for label, tensor in results.items():
        if label.startswith("grad:") and label[len("grad:") :] not in trainable:
            assert tensor is None, f"frozen {label} received a gradient"
            continue
        assert tensor is not None, f"missing {label}"
        assert torch.isfinite(tensor).all(), f"non-finite {label}"
        assert torch.count_nonzero(tensor).item() > 0, f"all-zero {label}"


def _bitwise_mismatches(expected: dict, actual: dict) -> list[str]:
    """Describe every differing tensor, with the count and size of its differences."""
    assert expected.keys() == actual.keys()
    mismatches = []
    for label, reference in expected.items():
        candidate = actual[label]
        if reference is None or candidate is None:
            if (reference is None) != (candidate is None):
                mismatches.append(f"{label}: gradient presence differs")
            continue
        if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
            mismatches.append(
                f"{label}: {reference.dtype}{tuple(reference.shape)} != "
                f"{candidate.dtype}{tuple(candidate.shape)}"
            )
            continue
        differing = torch.count_nonzero(_bits(reference) != _bits(candidate)).item()
        if differing:
            largest = (reference.float() - candidate.float()).abs().max().item()
            mismatches.append(
                f"{label}: {differing}/{reference.numel()} elements differ "
                f"(max |diff| {largest:.3e})"
            )
    return mismatches


def _compare_bf16_and_fp8(kind, overrides, fp8_overrides, kpool_spy=None) -> _Comparison:
    """Build BF16 reference and control instances and an FP8 candidate, then run all three.

    ``overrides`` apply to every instance, ``fp8_overrides`` only to the candidate,
    which is built and executed under Megatron's FP8 contexts for its configuration.
    The BF16 control executes first, absorbing one-time Triton autotuning and lazy
    library initialization, so the compared reference and candidate runs follow it.
    """
    build, projections, executed, frozen_prefix = _LAYERS[kind]
    model_parallel_cuda_manual_seed(_SEED)
    torch.manual_seed(_SEED)
    bf16_config = _config(**overrides)
    fp8_config = _config(**{**overrides, **fp8_overrides})
    reference = build(bf16_config)
    control = build(bf16_config)
    with get_fp8_context(fp8_config, is_init=True):
        assert FP8GlobalStateManager.with_fp8_parameters() == bool(
            fp8_config.fp8_param
        ), "FP8 parameter initialization does not follow fp8_param"
        candidate = build(fp8_config)
    for layer in (reference, control, candidate):
        _assert_quantizable_modules(layer, projections)
    _assert_high_precision_parameters(candidate, reference)
    _copy_parameters(reference, control)
    _copy_parameters(reference, candidate)
    trainable = _trainable_parameters(reference, frozen_prefix)

    hidden, cotangent = _inputs()
    repeated = _forward_backward(control, bf16_config, hidden, cotangent, executed, kpool_spy)
    _assert_quantization_states(repeated, outer_fp8=False, fp8_projections=())
    expected = _forward_backward(reference, bf16_config, hidden, cotangent, executed, kpool_spy)
    _assert_quantization_states(expected, outer_fp8=False, fp8_projections=())
    actual = _forward_backward(candidate, fp8_config, hidden, cotangent, executed, kpool_spy)
    return _Comparison(reference, candidate, trainable, repeated, expected, actual)


def _assert_opt_outs_bitwise_inert(kind: str, kpool_spy=None, **overrides) -> None:
    runs = _compare_bf16_and_fp8(kind, overrides, _TARGET_FP8, kpool_spy)
    _assert_quantization_states(runs.actual, outer_fp8=True, fp8_projections=())
    if kpool_spy is not None:
        for run in (runs.repeated, runs.expected, runs.actual):
            assert run.kpool_selections, "KPool selection was never executed"
            assert not any(run.kpool_selections), f"KPool FP8 scoring: {run.kpool_selections}"
            assert run.kpool_quantized_inputs == 0, "KPool inputs were converted to FP8"

    _assert_informative(runs.expected.results, runs.trainable)
    mismatches = _bitwise_mismatches(runs.expected.results, runs.actual.results)
    control_mismatches = _bitwise_mismatches(runs.expected.results, runs.repeated.results)
    control_note = (
        "an identical BF16 instance also differs, so kernels are nondeterministic: "
        + "; ".join(control_mismatches)
        if control_mismatches
        else "an identical BF16 instance is bitwise reproducible, so FP8 caused this"
    )
    assert not mismatches, f"FP8 opt-outs are not inert: {'; '.join(mismatches)} ({control_note})"
    _assert_high_precision_parameters(runs.candidate, runs.reference)


def _assert_fp8_execution_detected(kind, opt_out_overrides, fp8_projections, changed) -> None:
    """Negative control: FP8 execution of ``fp8_projections`` is visible to every check."""
    runs = _compare_bf16_and_fp8(kind, {}, {**opt_out_overrides, **_RUNTIME_FP8})
    _assert_quantization_states(runs.actual, outer_fp8=True, fp8_projections=fp8_projections)
    _assert_informative(runs.expected.results, runs.trainable)
    control_mismatches = _bitwise_mismatches(runs.expected.results, runs.repeated.results)
    assert not control_mismatches, (
        "an identical BF16 instance differs, so a difference cannot be attributed to FP8: "
        + "; ".join(control_mismatches)
    )
    mismatches = _bitwise_mismatches(runs.expected.results, runs.actual.results)
    mismatched = {mismatch.split(": ", 1)[0] for mismatch in mismatches}
    projection_outputs = {
        label
        for label in runs.actual.results
        if label.startswith("out:") and label[len("out:") :].split("#")[0] in fp8_projections
    }
    missing = (projection_outputs | set(changed)) - mismatched
    assert (
        projection_outputs and not missing
    ), f"FP8 execution went undetected in {sorted(missing)}; mismatches: {mismatches}"


@_requires_fla_kda
@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize("kda_core_in_fp32", [False, True], ids=["fla-core", "fp32-core"])
def test_glm53_kda_fp8_opt_out_is_bitwise_inert(kda_core_in_fp32):
    """``kda_disable_fp8`` keeps the two-stage GLM KDA mixer bitwise BF16 under FP8."""
    _assert_opt_outs_bitwise_inert("kda", kda_core_in_fp32=kda_core_in_fp32)


@pytest.mark.usefixtures("model_parallel")
def test_glm53_dsa_fp8_opt_outs_are_bitwise_inert(monkeypatch):
    """MLA, indexer and KPool opt-outs keep the NoPE KPool DSA mixer bitwise BF16 under FP8."""
    _assert_opt_outs_bitwise_inert("dsa", kpool_spy=_KPoolSpy(monkeypatch))


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize(
    ("kind", "overrides", "expected_fp8"),
    [
        pytest.param(
            "kda",
            {"kda_disable_fp8": False},
            _KDA_PROJECTIONS,
            id="kda_disable_fp8",
            marks=_requires_fla_kda,
        ),
        pytest.param(
            "dsa",
            {"mla_proj_use_quantization": True},
            _MLA_PROJECTIONS,
            id="mla_proj_use_quantization",
        ),
        pytest.param(
            "dsa",
            {"dsa_indexer_qk_proj_use_quantization": True},
            _INDEXER_QK_PROJECTIONS,
            id="dsa_indexer_qk_proj_use_quantization",
        ),
        pytest.param(
            "dsa",
            {"dsa_indexer_weights_proj_use_quantization": True},
            _INDEXER_WEIGHTS_PROJECTIONS,
            id="dsa_indexer_weights_proj_use_quantization",
        ),
    ],
)
def test_disabled_opt_out_quantizes_exactly_its_projections(kind, overrides, expected_fp8):
    """Negative control: each opt-out alone decides whether its weights become FP8."""
    build, projections, _, _ = _LAYERS[kind]
    model_parallel_cuda_manual_seed(_SEED)
    torch.manual_seed(_SEED)
    config = _config(**overrides, **_TARGET_FP8)
    with get_fp8_context(config, is_init=True):
        layer = build(config)
    _assert_quantizable_modules(layer, projections)
    modules = dict(layer.named_modules())
    primary_fp8 = {name for name in projections if modules[name].primary_weights_in_fp8}
    quantized = {name for name, parameter in layer.named_parameters() if is_float8tensor(parameter)}
    assert primary_fp8 == set(expected_fp8)
    assert quantized == {f"{name}.weight" for name in expected_fp8}


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize(
    ("opt_out", "fp8_projections", "changed"),
    [
        pytest.param(
            "mla_proj_use_quantization",
            _MLA_EXECUTED_PROJECTIONS,
            ("output", "input_grad")
            + tuple(f"grad:{name}.weight" for name in _MLA_EXECUTED_PROJECTIONS),
            id="mla_proj_use_quantization",
        ),
        # The frozen indexer feeds only the discrete top-k, so the mixer output is
        # not required to change; the indexer projection outputs must.
        pytest.param(
            "dsa_indexer_qk_proj_use_quantization",
            _INDEXER_QK_PROJECTIONS,
            (),
            id="dsa_indexer_qk_proj_use_quantization",
        ),
    ],
)
def test_fp8_dsa_projection_execution_is_detected(opt_out, fp8_projections, changed):
    """Negative control: TE flags and the bitwise comparator detect FP8 DSA projections."""
    _assert_fp8_execution_detected("dsa", {opt_out: True}, fp8_projections, changed)


@_requires_fla_kda
@pytest.mark.usefixtures("model_parallel")
def test_fp8_kda_projection_execution_is_detected(monkeypatch):
    """Negative control: a ``_proj`` that drops the KDA runtime opt-out is detected."""
    opted_out_proj = KimiDeltaAttention._proj

    def _regressed_proj(self, proj, x):
        if any(proj is getattr(self, name) for name in _KDA_REGRESSED_PROJECTIONS):
            return proj(x)
        return opted_out_proj(self, proj, x)

    monkeypatch.setattr(KimiDeltaAttention, "_proj", _regressed_proj)
    changed = ("output", "input_grad") + tuple(
        f"grad:{name}.weight" for name in _KDA_REGRESSED_PROJECTIONS
    )
    _assert_fp8_execution_detected("kda", {}, _KDA_REGRESSED_PROJECTIONS, changed)


@pytest.mark.usefixtures("model_parallel")
def test_kpool_quantization_is_visible_to_the_kpool_spy(monkeypatch):
    """Negative control: the KPool spy observes the quantization GLM53 disables."""
    spy = _KPoolSpy(monkeypatch, substitute_quantizer=True)
    model_parallel_cuda_manual_seed(_SEED)
    torch.manual_seed(_SEED)
    config = _config(dsa_indexer_kpool_use_quantization=True, **_TARGET_FP8)
    with get_fp8_context(config, is_init=True):
        layer = _build_dsa(config)
    hidden, cotangent = _inputs()
    run = _forward_backward(layer, config, hidden, cotangent, _DSA_EXECUTED_PROJECTIONS, spy)
    _assert_quantization_states(run, outer_fp8=True, fp8_projections=())
    assert run.kpool_selections and all(run.kpool_selections), run.kpool_selections
    # Each selection converts its queries and its pooled keys.
    assert run.kpool_quantized_inputs == 2 * len(run.kpool_selections)
