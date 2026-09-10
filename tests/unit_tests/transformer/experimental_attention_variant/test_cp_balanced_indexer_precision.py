# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Selection projections must preserve their precision across activation recompute."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.extensions.transformer_engine import (
    HAVE_TE,
    TELinear,
    TEQuantizationParams,
    TEQuantizationRecipe,
)
from megatron.core.fp8_utils import dequantize_fp8_tensor, get_fp8_context, is_float8tensor
from megatron.core.transformer.experimental_attention_variant.cp_balanced_indexer import (
    _project_selection_q,
    _selection_uses_delayed_scaling,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = [
    pytest.mark.launch_on_gb200,
    pytest.mark.skipif(not HAVE_TE or not torch.cuda.is_available(), reason="Requires CUDA and TE"),
]


@pytest.fixture(autouse=True)
def _model_parallel():
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    yield
    Utils.destroy_model_parallel()


def _projection(recipe, fp8_param=False):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        params_dtype=torch.bfloat16,
        bf16=True,
        fp8=None if recipe == "bf16" else "e4m3",
        fp8_recipe="delayed" if recipe == "bf16" else recipe,
        fp8_param=fp8_param,
        use_cpu_initialization=False,
    )
    with get_fp8_context(config, layer_no=0, is_init=True):
        linear = TELinear(
            128,
            256,
            config=config,
            init_method=torch.nn.init.normal_,
            bias=False,
            skip_bias_add=False,
            parallel_mode="duplicated",
        ).cuda()
    linear.finish_init(None)
    return config, linear


def _amax_snapshot(linear):
    result = {}
    for name, value in linear.fp8_meta.items():
        history = getattr(value, "amax_history", None)
        if history is not None:
            result[name] = history.clone()
    return result


@pytest.mark.parametrize(
    ("recipe", "fp8_param"),
    [
        ("bf16", False),
        ("mxfp8", False),
        ("mxfp8", True),
        ("tensorwise", False),
        ("blockwise", False),
        ("delayed", False),
        ("delayed", True),
    ],
)
def test_selection_q_recompute_precision(recipe, fp8_param):
    from transformer_engine.pytorch.distributed import activation_recompute_forward

    if recipe in ("mxfp8", "blockwise") and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("This recipe requires Blackwell")
    torch.manual_seed(1234)
    config, linear = _projection(recipe, fp8_param)
    assert is_float8tensor(linear.weight) == fp8_param
    qr = torch.randn(256, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    projections = []
    for recompute in (False, True):
        with get_fp8_context(config, layer_no=0):
            with activation_recompute_forward(activation_recompute=True, recompute_phase=recompute):
                grad_context = torch.enable_grad() if recompute else torch.no_grad()
                with grad_context:
                    # Match CSA: the original checkpoint forward skips the loss
                    # projection, except for delayed scaling's required history.
                    if recompute or recipe == "delayed":
                        linear(qr)
                    before = _amax_snapshot(linear)
                    head = _project_selection_q(linear, qr[:128])
                    tail = _project_selection_q(linear, qr[128:])
                    projections.append(torch.cat((head, tail)))
                    if recipe == "delayed":
                        after = _amax_snapshot(linear)
                        assert before.keys() == after.keys()
                        assert before, "The canonical projection must initialize delayed amax"
                        for key in before:
                            torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)
                    else:
                        assert linear.fp8 == (recipe != "bf16")
    torch.testing.assert_close(projections[0], projections[1], rtol=0, atol=0)
    if recipe in ("bf16", "delayed"):
        weight = dequantize_fp8_tensor(linear.weight) if fp8_param else linear.weight
        expected = torch.cat((F.linear(qr[:128], weight), F.linear(qr[128:], weight)))
        torch.testing.assert_close(projections[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize("ambient_recipe", ["delayed", "mxfp8"])
@pytest.mark.parametrize("training", [True, False], ids=["train", "eval"])
def test_selection_q_preserves_explicit_bf16_override(ambient_recipe, training):
    if ambient_recipe == "mxfp8" and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 requires Blackwell")
    config, linear = _projection(ambient_recipe)
    linear.te_quant_params = TEQuantizationParams(
        training_recipe=TEQuantizationRecipe(), evaluation_recipe=TEQuantizationRecipe()
    )
    linear.train(training)
    qr = torch.randn(128, 1, 128, device="cuda", dtype=torch.bfloat16)
    with get_fp8_context(config, layer_no=0):
        assert not _selection_uses_delayed_scaling(linear)
        actual = _project_selection_q(linear, qr)
    assert not linear.fp8
    torch.testing.assert_close(actual, F.linear(qr, linear.weight), rtol=0, atol=0)


def test_delayed_selection_waits_for_parameter_gather():
    config, linear = _projection("delayed")
    qr = torch.randn(128, 1, 128, device="cuda", dtype=torch.bfloat16)
    ready = []
    linear.weight._ensure_param_ready_callback = lambda: ready.append(True)
    with get_fp8_context(config, layer_no=0):
        _project_selection_q(linear, qr)
    assert ready == [True]
