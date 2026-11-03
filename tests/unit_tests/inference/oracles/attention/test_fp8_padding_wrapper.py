# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for the FP8 inference padding wrapper (``prepare_model_for_fp8_inference``).

Call site: ``megatron/core/fp8_utils.py`` ``_wrap_te_linear_for_padding``, applied by
``prepare_model_for_fp8_inference`` (from the inference model wrapper) to every TE linear of the
model. It sits on the attention projections (``linear_qkv``, ``linear_proj``, the MLA up/down
projections) and every other TE linear of an FP8 inference model.

Contract (from the wrapper's documented purpose and the TE linear it wraps):

* Outside an FP8/FP4 autocast the wrapper is transparent.
* Inside one, for input ``x [S, B, H_in]`` the result is the module's own forward on ``x``
  zero-padded along the sequence to the recipe's alignment (16 rows; 32 for MXFP8), with the
  padded rows dropped: ``wrapped(x) == forward(pad(x))[:S]``. With current (tensorwise) scaling
  zero rows do not change any amax, so this holds bit for bit.
* With tensor and sequence parallelism, each rank gets exactly its part of ``LN?(x) @ W^T``
  where ``x``/``W`` are the full operands: a column-parallel linear takes this rank's
  ``S / TP`` rows and returns all ``S`` rows of its ``H_out / TP`` columns; a row-parallel linear
  takes all rows of its ``H_in / TP`` input columns and returns its ``S / TP`` rows of the full
  ``H_out`` (one reduction over the TP group, never two). The module's ``sequence_parallel``
  attribute is the same after the call as before.

Reference: fp64 ``x @ W^T`` (after an fp64 LayerNorm for ``TELayerNormColumnParallelLinear``) from
the full operands gathered over the TP group. Tolerance class: floating point, bounded by a
trusted implementation in the same dtype: Transformer Engine's own ``Linear``/``LayerNormLinear``
holding the full weight, unwrapped and unsharded, under the same FP8 recipe, on the manually
padded full input. At TP=1 the wrapped output must also equal the unwrapped module on the padded
input bit for bit (exact, metamorphic; tensorwise and MXFP8 current scaling).

Variants covered: TELinear (duplicated), TEColumnParallelLinear, TERowParallelLinear and
TELayerNormColumnParallelLinear; TP 1, 2, 4 (sequence parallel when TP > 1); full sequence
``TP, 16 - TP, 16, 16 + TP, 32 + TP`` around the 16-row alignment (32-row for MXFP8); batch 1
and 2; tensorwise and delayed scaling on H100, MXFP8 on GB200.

Open PRs (written against main):
* #8567 (s7): with sequence parallelism the wrapper clears ``module.sequence_parallel``, so TE's
  row-parallel linear all-reduces, and then reduce-scatters the result again: the row-parallel
  output is TP times too large. Every ``row`` case with TP > 1 fails on main and passes with s7.
"""

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.test_utilities import Utils

IN_FEATURES, OUT_FEATURES = 256, 128
MODES = ["duplicated", "column", "row", "layernorm_column"]


def _sizes(tp: int, align: int):
    """Full sequence lengths around the alignment, each divisible by TP."""
    return sorted({tp, align - tp, align, align + tp, 2 * align + tp})


@pytest.fixture
def fresh_wrapper_registry():
    from megatron.core import fp8_utils

    fp8_utils._fp8_inference_wrapped_modules.clear()
    yield
    fp8_utils._fp8_inference_wrapped_modules.clear()


def _config(tp: int, recipe):
    from megatron.core.transformer.transformer_config import TransformerConfig

    return TransformerConfig(
        num_layers=1,
        hidden_size=IN_FEATURES,
        num_attention_heads=4,
        tensor_model_parallel_size=tp,
        sequence_parallel=tp > 1,
        params_dtype=torch.bfloat16,
        bf16=True,
        add_bias_linear=False,
        fp8="e4m3",
        fp8_recipe=recipe,
    )


def _module(mode: str, config, tp_group):
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TELayerNormColumnParallelLinear,
        TELinear,
        TERowParallelLinear,
    )

    kwargs = dict(
        config=config,
        init_method=config.init_method,
        bias=False,
        skip_bias_add=True,
        is_expert=False,
    )
    if mode == "duplicated":
        return TELinear(
            IN_FEATURES,
            OUT_FEATURES,
            parallel_mode="duplicated",
            tp_comm_buffer_name=None,
            skip_weight_param_allocation=False,
            tp_group=None,
            **kwargs,
        )
    if mode == "row":
        return TERowParallelLinear(
            IN_FEATURES, OUT_FEATURES, input_is_parallel=True, tp_group=tp_group, **kwargs
        )
    cls = TELayerNormColumnParallelLinear if mode == "layernorm_column" else TEColumnParallelLinear
    return cls(IN_FEATURES, OUT_FEATURES, gather_output=False, tp_group=tp_group, **kwargs)


def _full_weight(module, mode: str, tp_group) -> torch.Tensor:
    """The full [OUT, IN] weight, gathered over the TP group."""
    shard = module.weight.detach()
    if mode == "duplicated" or tp_group.size() == 1:
        return shard
    parts = [torch.empty_like(shard) for _ in range(tp_group.size())]
    torch.distributed.all_gather(parts, shard.contiguous(), group=tp_group)
    return torch.cat(parts, dim=1 if mode == "row" else 0)


def _rank_part(full: torch.Tensor, mode: str, tp: int, rank: int) -> torch.Tensor:
    """This rank's part of a full [S, B, OUT] result."""
    if mode == "duplicated" or tp == 1:
        return full
    if mode == "row":
        return full.chunk(tp, dim=0)[rank]
    return full.chunk(tp, dim=2)[rank]


def _rank_input(x: torch.Tensor, mode: str, tp: int, rank: int) -> torch.Tensor:
    if mode == "duplicated" or tp == 1:
        return x
    if mode == "row":
        return x.chunk(tp, dim=2)[rank].contiguous()
    return x.chunk(tp, dim=0)[rank].contiguous()


def _pad(x: torch.Tensor, align: int) -> torch.Tensor:
    rows = -(-x.shape[0] // align) * align
    return torch.cat([x, x.new_zeros(rows - x.shape[0], *x.shape[1:])])


def _trusted(mode: str, weight: torch.Tensor, module, config, x_full: torch.Tensor, align: int):
    """Transformer Engine's own unsharded, unwrapped linear holding the full weight."""
    import transformer_engine.pytorch as te

    from megatron.core.fp8_utils import get_fp8_context

    if mode == "layernorm_column":
        ref = te.LayerNormLinear(
            IN_FEATURES,
            weight.shape[0],
            eps=config.layernorm_epsilon,
            bias=False,
            normalization=config.normalization,
            params_dtype=torch.bfloat16,
        ).cuda()
        with torch.no_grad():
            ref.layer_norm_weight.copy_(module.layer_norm_weight)
            if getattr(module, "layer_norm_bias", None) is not None:
                ref.layer_norm_bias.copy_(module.layer_norm_bias)
    else:
        ref = te.Linear(IN_FEATURES, weight.shape[0], bias=False, params_dtype=torch.bfloat16)
        ref = ref.cuda()
    with torch.no_grad():
        ref.weight.copy_(weight)
        with get_fp8_context(config):
            out = ref(_pad(x_full, align))
    return out[: x_full.shape[0]]


def _reference(mode, module, weight, x_full, config):
    def fn(x, w, gamma, beta):
        if mode == "layernorm_column":
            x = torch.nn.functional.layer_norm(
                x, (x.shape[-1],), gamma, beta, config.layernorm_epsilon
            )
        return torch.einsum("sbk,nk->sbn", x, w)

    gamma = getattr(module, "layer_norm_weight", None)
    beta = getattr(module, "layer_norm_bias", None)
    gamma = None if gamma is None else gamma.detach()
    beta = None if beta is None else beta.detach()
    return ou.fp64_reference(fn, x_full, weight, gamma, beta)


def _run(mode: str, tp: int, recipe_name: str, seq: int, batch: int, align: int):
    from megatron.core import parallel_state
    from megatron.core.enums import Fp8Recipe
    from megatron.core.fp8_utils import get_fp8_context, prepare_model_for_fp8_inference

    recipe = Fp8Recipe(recipe_name)
    if tp > 1 and Utils.world_size % tp:
        pytest.fail(
            f"TP={tp} needs a world size divisible by it, got {Utils.world_size}", pytrace=False
        )
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp)
    try:
        tp_group = parallel_state.get_tensor_model_parallel_group()
        rank = tp_group.rank()
        config = _config(tp, recipe)
        module = _module(mode, config, tp_group).cuda().eval()
        # Each rank draws its own weight shard; every rank draws the same full input and feeds
        # its own part of it.
        g_rank = ou.make_generator(seed=ou.DEFAULT_SEED + 1000 * (rank + 1) + seq, device="cuda")
        with torch.no_grad():
            module.weight.copy_(
                torch.randn(module.weight.shape, device="cuda", generator=g_rank) * 0.05
            )
        weight = _full_weight(module, mode, tp_group)
        g = ou.make_generator(seed=ou.DEFAULT_SEED + seq, device="cuda")
        x_full = torch.randn(seq, batch, IN_FEATURES, device="cuda", generator=g).to(torch.bfloat16)
        x = _rank_input(x_full, mode, tp, rank)
        original_forward = module.forward
        container = torch.nn.ModuleList([module])
        prepare_model_for_fp8_inference(container)
        assert module.forward is not original_forward, f"{mode}: the wrapper did not wrap"
        sequence_parallel = getattr(module, "sequence_parallel", None)
        x_before = x.clone()
        with torch.no_grad(), get_fp8_context(config):
            out, _ = module(x)
        assert getattr(module, "sequence_parallel", None) == sequence_parallel, "flag not restored"
        ou.assert_exact(x, x_before, what="input (read-only)")
        reference = _rank_part(_reference(mode, module, weight, x_full, config), mode, tp, rank)
        trusted = _rank_part(_trusted(mode, weight, module, config, x_full, align), mode, tp, rank)
        assert (
            out.shape == reference.shape
        ), f"{mode}: shape {tuple(out.shape)} vs {tuple(reference.shape)}"
        ou.assert_close_to_reference(
            out.reshape(-1, out.shape[-1]),
            reference.reshape(-1, reference.shape[-1]),
            trusted.reshape(-1, trusted.shape[-1]),
            what=f"{mode} TP={tp} S={seq} B={batch} ({recipe})",
        )
        if tp == 1 and recipe_name in ("tensorwise", "mxfp8"):
            # Current scaling: zero rows change no amax, so padding by hand is the same GEMM.
            with torch.no_grad(), get_fp8_context(config):
                unwrapped, _ = original_forward(_pad(x, align))
            ou.assert_exact(out, unwrapped[:seq], what=f"{mode}: wrapped vs unwrapped on padded x")
    finally:
        Utils.destroy_model_parallel()


def _cases(recipe_name: str, align: int, marks):
    params = []
    for tp in (1, 2, 4):
        modes = MODES if tp == 1 else ["column", "row", "layernorm_column"]
        factors = {"mode": modes, "seq": _sizes(tp, align), "batch": [1, 2]}
        for case in ou.pairwise_cases(factors):
            params.append(
                pytest.param(
                    recipe_name,
                    align,
                    tp,
                    case["mode"],
                    case["seq"],
                    case["batch"],
                    id=f"{recipe_name}-tp{tp}-{case['mode']}-S{case['seq']}-B{case['batch']}",
                    marks=marks,
                )
            )
    return params


HOPPER = [ou.requires_sm(ou.HOPPER)]
BLACKWELL = [ou.requires_sm(ou.BLACKWELL), pytest.mark.launch_on_gb200]


@pytest.mark.usefixtures("fresh_wrapper_registry")
@pytest.mark.parametrize(
    "recipe_name, align, tp, mode, seq, batch",
    _cases("tensorwise", 16, HOPPER)
    + _cases("delayed", 16, HOPPER)
    + _cases("mxfp8", 32, BLACKWELL),
)
def test_wrapped_linear_matches_reference(recipe_name, align, tp, mode, seq, batch):
    """Each rank's wrapped FP8 output is its part of the full ``x @ W^T`` (bounded by TE's own
    unsharded linear), and at TP=1 it equals the unwrapped module on the padded input."""
    _run(mode, tp, recipe_name, seq, batch, align)


@pytest.mark.usefixtures("fresh_wrapper_registry")
@pytest.mark.sanitizer
@ou.requires_sm(ou.HOPPER)
def test_small_case_for_sanitizer():
    """One padded sequence-parallel case (compute-sanitizer memcheck case)."""
    tp = 2 if Utils.world_size % 2 == 0 else 1
    _run("column", tp, "tensorwise", 2 * tp + (0 if tp > 1 else 1), 1, 16)


@pytest.mark.usefixtures("fresh_wrapper_registry")
@pytest.mark.sanitizer
@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
def test_sanitizer_mxfp8_small_case():
    """The GB200 counterpart of ``test_small_case_for_sanitizer``: one padded sequence-parallel
    MXFP8 case, padded to the 32-row alignment."""
    tp = 2 if Utils.world_size % 2 == 0 else 1
    _run("column", tp, "mxfp8", 2 * tp + (0 if tp > 1 else 1), 1, 32)


@pytest.mark.usefixtures("fresh_wrapper_registry")
@ou.requires_sm(ou.HOPPER)
def test_wrapper_is_transparent_outside_fp8():
    """Outside an FP8 autocast the wrapped module returns its unwrapped output bit for bit."""
    from megatron.core import parallel_state
    from megatron.core.enums import Fp8Recipe
    from megatron.core.fp8_utils import prepare_model_for_fp8_inference

    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    try:
        config = _config(1, Fp8Recipe.tensorwise)
        module = _module("column", config, parallel_state.get_tensor_model_parallel_group())
        module = module.cuda().eval()
        g = ou.make_generator(device="cuda")
        x = torch.randn(17, 2, IN_FEATURES, device="cuda", generator=g).to(torch.bfloat16)
        with torch.no_grad():
            before, _ = module(x)
            prepare_model_for_fp8_inference(torch.nn.ModuleList([module]))
            after, _ = module(x)
        ou.assert_exact(after, before, what="wrapped vs unwrapped outside FP8")
    finally:
        Utils.destroy_model_parallel()
