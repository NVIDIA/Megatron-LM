# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Simplified sparse-attention precision, gradient and cuDNN-adapter contracts."""

import sys
from argparse import ArgumentParser
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import (
    simplified_sparse_attention,
    simplified_sparse_attention_cudnn,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.argument_utils import core_transformer_config_from_args
from megatron.training.arguments import add_megatron_arguments


def _config(**overrides):
    values = dict(
        num_layers=2,
        hidden_size=256,
        num_attention_heads=16,
        num_query_groups=1,
        kv_channels=256,
        add_bias_linear=False,
        experimental_attention_variant="ssa",
        dsa_indexer_topk=512,
        dsa_indexer_loss_coeff=0.1,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    values.update(overrides)
    return TransformerConfig(**values)


def _inputs(sequence_length=16, heads=16, device="cpu"):
    return tuple(
        torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True)
        for shape in (
            (sequence_length, 1, heads, 256),
            (sequence_length, 1, 1, 256),
            (sequence_length, 1, 1, 256),
            (sequence_length, 1, 1, 128),
            (sequence_length, 1, 1, 128),
        )
    )


def test_ssa_rejects_deterministic_mode():
    with pytest.raises(ValueError, match="uses atomics"):
        _config(deterministic_mode=True)


@pytest.mark.parametrize("topk", [256, 4096, None])
def test_ssa_rejects_unsupported_topk(topk):
    with pytest.raises(ValueError, match="top-k 512, 1024 or 2048"):
        _config(dsa_indexer_topk=topk)


def test_indexer_only_training_requires_ssa_and_a_positive_coefficient():
    with pytest.raises(ValueError, match="requires experimental_attention_variant='ssa'"):
        _config(experimental_attention_variant=None, ssa_train_indexer_only=True)
    with pytest.raises(ValueError, match="positive dsa_indexer_loss_coeff"):
        _config(ssa_train_indexer_only=True, dsa_indexer_loss_coeff=0.0)
    with pytest.raises(ValueError, match="select different trainable parameters"):
        _config(ssa_train_indexer_only=True, freeze_base_model_for_mtp=True)
    assert _config(ssa_train_indexer_only=True).ssa_train_indexer_only


@pytest.mark.parametrize(
    "pattern,selector,expected",
    [
        (Symbols.ATTENTION, "ssa", "ssa"),
        (Symbols.DS_ATTENTION, None, "dsa"),
        (Symbols.DS_ATTENTION, "ssa", "error"),
    ],
)
def test_hybrid_argument_conversion_preserves_attention_selection(pattern, selector, expected):
    parser = ArgumentParser()
    add_megatron_arguments(parser)
    options = [
        "--num-layers",
        "1",
        "--hidden-size",
        "256",
        "--num-attention-heads",
        "16",
        "--group-query-attention",
        "--num-query-groups",
        "1",
        "--kv-channels",
        "256",
        "--disable-bias-linear",
        "--no-bias-gelu-fusion",
        "--no-rope-fusion",
        "--bf16",
        "--dsa-indexer-topk",
        "512",
        "--dsa-indexer-loss-coeff",
        "0.1",
        "--hybrid-layer-pattern",
        pattern,
    ]
    if selector is not None:
        options.extend(["--experimental-attention-variant", selector])
    args = parser.parse_args(options)
    # validate_args normally derives params_dtype from --bf16.
    args.params_dtype = torch.bfloat16

    if expected == "error":
        with pytest.raises(ValueError, match="cannot be combined with MLA DSA 'D' layers"):
            core_transformer_config_from_args(args)
    else:
        config = core_transformer_config_from_args(args)
        assert config.experimental_attention_variant == expected


def _metadata(**fields):
    return SimpleNamespace(**fields)


def _kernels(sparse_attention):
    return simplified_sparse_attention_cudnn.CudnnSparseAttentionKernels(
        sparse_attention=sparse_attention, row_metadata=_metadata
    )


def _bind_mock_kernels(*, dense_teacher=False):
    return _kernels(None)


@pytest.mark.parametrize("coefficient", [0.0, 0.03])
@pytest.mark.parametrize("per_token", [False, True])
def test_adapter_applies_coefficient_and_normalization_once(coefficient, per_token):
    inputs = _inputs()
    seen = {}

    def sparse_attention(q, k, v, qi, ki, metadata, topk, *, loss_coeff):
        seen.update(q=q, k=k, v=v, qi=qi, ki=ki, metadata=metadata, topk=topk)
        return (
            q + k.expand_as(q) + v.expand_as(q),
            (qi.float().square().sum() + ki.float().square().sum())
            * loss_coeff
            / metadata.loss_denominator.squeeze(),
        )

    output, loss = simplified_sparse_attention_cudnn.run_sparse_attention(
        _kernels(sparse_attention),
        *inputs,
        topk=512,
        loss_coeff=coefficient,
        calculate_per_token_loss=per_token,
    )
    denominator = 1 if per_token else 16
    assert output.shape == (16, 1, 4096)
    assert seen["topk"] == 512
    for name, original in zip(("q", "k", "v", "qi", "ki"), inputs):
        assert seen[name].data_ptr() == original.data_ptr()
    metadata = seen["metadata"]
    torch.testing.assert_close(metadata.key_starts, torch.zeros(16, dtype=torch.int32))
    torch.testing.assert_close(metadata.key_ends, torch.arange(1, 17, dtype=torch.int32))
    assert bool(metadata.query_valid.all())
    assert metadata.triangular_scores and not metadata.mask_invalid_rows
    (output.float().sum() + loss).backward()
    torch.testing.assert_close(inputs[0].grad, torch.ones_like(inputs[0]))
    for tensor in inputs[1:3]:
        torch.testing.assert_close(tensor.grad, torch.full_like(tensor, 16))
    for tensor in inputs[3:]:
        expected = (2 * tensor.detach().float() * coefficient / denominator).to(tensor.dtype)
        torch.testing.assert_close(tensor.grad, expected)


def test_bind_reports_missing_cudnn_frontend(monkeypatch):
    monkeypatch.setitem(sys.modules, "cudnn", SimpleNamespace(__version__="1.32.0"))
    monkeypatch.setitem(sys.modules, "cudnn.ops", SimpleNamespace())
    with pytest.raises(ImportError, match="Simplified sparse attention requires cudnn.ops"):
        simplified_sparse_attention_cudnn.bind_cudnn_sparse_attention()


def test_bind_dense_teacher_requires_indexer_kl_and_flash_attention_4(monkeypatch):
    from megatron.core.transformer import attention

    bind = simplified_sparse_attention_cudnn.bind_cudnn_sparse_attention
    ops = SimpleNamespace(
        simplified_sparse_attention=object(), SimplifiedSparseAttentionMetadata=object
    )
    monkeypatch.setitem(sys.modules, "cudnn", SimpleNamespace(__version__="1.33.0"))
    monkeypatch.setitem(sys.modules, "cudnn.ops", ops)
    assert bind().dense_attention is None and bind().dense_indexer_kl is None
    with pytest.raises(ImportError, match="cudnn.ops.dense_indexer_kl"):
        bind(dense_teacher=True)
    ops.dense_indexer_kl = object()
    monkeypatch.setattr(attention, "HAVE_FA4", False)
    with pytest.raises(ImportError, match="flash-attn-4"):
        bind(dense_teacher=True)
    flash_attention = object()
    monkeypatch.setattr(attention, "HAVE_FA4", True)
    monkeypatch.setattr(attention, "flash_attn4_varlen_func", flash_attention)
    kernels = bind(dense_teacher=True)
    assert kernels.dense_attention is flash_attention
    assert kernels.dense_indexer_kl is ops.dense_indexer_kl


@pytest.mark.parametrize("coefficient", [0.0, 0.03])
@pytest.mark.parametrize("per_token", [False, True])
def test_dense_teacher_is_detached_and_feeds_its_lse_to_the_indexer_kl(coefficient, per_token):
    # Distinct sequence length and head count make an LSE layout mix-up visible.
    inputs = _inputs(sequence_length=24)
    seen = {}
    lse = torch.arange(24 * 16, dtype=torch.float32).reshape(24, 16)

    def dense_attention(q, k, v, **kwargs):
        assert not torch.is_grad_enabled()
        seen.update(q=q, k=k, v=v, **kwargs)
        return q + k + v, lse.t().contiguous()

    def dense_indexer_kl(q, k, qi, ki, **kwargs):
        seen.update(kl_q=q, kl_k=k, **{f"kl_{name}": value for name, value in kwargs.items()})
        loss = qi.float().square().sum() + ki.float().square().sum()
        return loss * kwargs["loss_coeff"] / kwargs["loss_denominator"].squeeze()

    kernels = simplified_sparse_attention_cudnn.CudnnSparseAttentionKernels(
        sparse_attention=None,
        row_metadata=None,
        dense_attention=dense_attention,
        dense_indexer_kl=dense_indexer_kl,
    )
    output, loss = simplified_sparse_attention_cudnn.run_dense_teacher(
        kernels, *inputs, loss_coeff=coefficient, calculate_per_token_loss=per_token
    )
    assert output.shape == (24, 1, 4096) and not output.requires_grad
    assert seen["q"].shape == (24, 16, 256) and seen["k"].shape == seen["v"].shape == (24, 1, 256)
    torch.testing.assert_close(seen["cu_seqlens_q"], torch.tensor([0, 24], dtype=torch.int32))
    assert seen["cu_seqlens_k"] is seen["cu_seqlens_q"]
    assert seen["max_seqlen_q"] == seen["max_seqlen_k"] == 24
    assert seen["softmax_scale"] == 256**-0.5 and seen["causal"] and seen["return_lse"]
    if coefficient == 0:
        assert loss is None and "kl_q" not in seen
        return
    assert seen["kl_q"] is seen["q"] and seen["kl_k"] is seen["k"]
    assert seen["kl_teacher_lse"].is_contiguous()
    torch.testing.assert_close(seen["kl_teacher_lse"], lse)
    assert seen["kl_use_kf"] and seen["kl_loss_coeff"] == coefficient
    assert seen["kl_teacher_softmax_scale"] == 256**-0.5
    assert seen["kl_student_softmax_scale"] == 128**-0.5
    torch.testing.assert_close(seen["kl_key_starts"], torch.zeros(24, dtype=torch.int32))
    torch.testing.assert_close(seen["kl_key_ends"], torch.arange(1, 25, dtype=torch.int32))
    loss.backward()
    assert all(tensor.grad is None for tensor in inputs[:3])
    denominator = 1 if per_token else 24
    for tensor in inputs[3:]:
        expected = (2 * tensor.detach().float() * coefficient / denominator).to(tensor.dtype)
        torch.testing.assert_close(tensor.grad, expected)


class _Linear(torch.nn.Module):
    def __init__(self, input_size, output_size, *, config, **kwargs):
        super().__init__()
        self.config = config
        self.weight = torch.nn.Parameter(
            torch.randn(output_size, input_size, dtype=config.params_dtype) * 0.05
        )

    def forward(self, inputs):
        return torch.nn.functional.linear(inputs, self.weight), None


def _mock_core(monkeypatch, *, coefficient=0.1, per_token=False, layer_number=1, **overrides):
    monkeypatch.setattr(simplified_sparse_attention, "TELinear", _Linear)
    monkeypatch.setattr(
        simplified_sparse_attention,
        "get_fp8_disabled_context",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        simplified_sparse_attention, "bind_cudnn_sparse_attention", _bind_mock_kernels
    )
    monkeypatch.setattr(
        simplified_sparse_attention.DSAIndexerLossAutoScaler, "main_loss_backward_scale", None
    )
    config = _config(
        hidden_size=16,
        dsa_indexer_loss_coeff=coefficient,
        calculate_per_token_loss=per_token,
        **overrides,
    )
    group = SimpleNamespace(size=lambda: 1)
    return simplified_sparse_attention.SimplifiedSparseAttentionCore(
        config,
        layer_number=layer_number,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        pg_collection=SimpleNamespace(tp=group, cp=group),
        softmax_scale=config.softmax_scale,
    )


def test_indexer_shape_is_fixed_and_dsa_only_options_do_not_change_it(monkeypatch):
    core = _mock_core(
        monkeypatch,
        dsa_indexer_n_heads=8,
        dsa_indexer_head_dim=64,
        dsa_indexer_rotate_activation=True,
        dsa_indexer_scoring_relu=True,
    )
    assert core.indexer.linear_q.weight.shape == (128, 16)
    assert core.indexer.linear_k.weight.shape == (128, 16)
    assert set(dict(core.named_parameters())) == {
        "indexer.linear_q.weight",
        "indexer.linear_k.weight",
    }


def test_core_rejects_nonstandard_scale_at_construction(monkeypatch):
    with pytest.raises(ValueError, match="softmax scale"):
        _mock_core(monkeypatch, softmax_scale=0.125)


def test_attention_passes_normalized_indexer_input():
    owner = SimpleNamespace(config=_config())
    hidden = torch.randn(16, 1, 256)
    normalized = torch.randn_like(hidden)
    kwargs = simplified_sparse_attention.SimplifiedSparseAttention._get_core_attention_extra_kwargs(
        owner, hidden_states=hidden, qkv_layernorm_output=normalized, inference_context=None
    )
    assert kwargs["indexer_input"] is normalized
    assert set(kwargs) == {"indexer_input"}


def test_ssa_factory_preserves_dense_spec_and_uses_indexer_loss_scale_hook():
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        _get_self_attention_module_spec,
        get_experimental_attention_variant_module_spec,
    )
    from megatron.core.pipeline_parallel.schedules import (
        _get_experimental_attention_variant_loss_scale_func,
    )
    from megatron.core.transformer.attention import SelfAttention

    config = _config()
    dense_before = _get_self_attention_module_spec(config)
    selected = get_experimental_attention_variant_module_spec(config)
    dense_after = _get_self_attention_module_spec(config)
    assert selected.module is simplified_sparse_attention.SimplifiedSparseAttention
    assert dense_before.module is dense_after.module is SelfAttention
    assert selected.submodules == dense_before.submodules == dense_after.submodules
    assert selected.metainfo == dense_before.metainfo == dense_after.metainfo
    assert _get_experimental_attention_variant_loss_scale_func(config) == (
        simplified_sparse_attention.DSAIndexerLossAutoScaler.set_loss_scale
    )


def _fake_attention(kernels, q, k, v, qi, ki, **kwargs):
    output = q + k.expand_as(q) + v.expand_as(q)
    denominator = 1 if kwargs["calculate_per_token_loss"] else q.size(0)
    loss = qi.float().square().sum() + ki.float().square().sum()
    return output.flatten(2), loss * kwargs["loss_coeff"] / denominator


@pytest.mark.parametrize("coefficient", [0.0, 0.2])
@pytest.mark.parametrize("per_token", [False, True])
def test_core_auxiliary_gradients_are_detached_and_logging_is_mean(
    monkeypatch, coefficient, per_token
):
    core = _mock_core(monkeypatch, coefficient=coefficient, per_token=per_token)
    logged = {}
    monkeypatch.setattr(
        simplified_sparse_attention.DSAIndexerLossLoggingHelper,
        "save_loss_to_tracker",
        lambda **kw: logged.update(kw),
    )
    monkeypatch.setattr(simplified_sparse_attention, "run_sparse_attention", _fake_attention)
    hidden = torch.randn(16, 1, 16, dtype=torch.bfloat16, requires_grad=True)
    query, key, value, _, _ = _inputs()
    output = core(query, key, value, None, indexer_input=hidden)
    output.float().sum().backward()
    assert hidden.grad is None
    for parameter in core.indexer.parameters():
        assert parameter.grad is not None
        assert bool(torch.count_nonzero(parameter.grad)) == (coefficient > 0)
    qi, ki = core.indexer(hidden)
    expected_log = (qi.float().square().sum() + ki.float().square().sum()) * coefficient / 16
    torch.testing.assert_close(logged["loss"], expected_log)
    if coefficient > 0:
        torch.testing.assert_close(logged["raw_loss"], expected_log / coefficient)
    assert logged["layer_number"] == 1 and logged["num_layers"] == 2
    torch.testing.assert_close(query.grad, torch.ones_like(query))


def test_core_eval_skips_indexer_objective(monkeypatch):
    core = _mock_core(monkeypatch).eval()
    seen = {}

    def attention(kernels, q, k, v, qi, ki, **kwargs):
        seen.update(kwargs)
        return _fake_attention(kernels, q, k, v, qi, ki, **kwargs)

    monkeypatch.setattr(simplified_sparse_attention, "run_sparse_attention", attention)
    monkeypatch.setattr(
        simplified_sparse_attention.DSAIndexerLossLoggingHelper,
        "save_loss_to_tracker",
        lambda **kw: pytest.fail("Evaluation must not log the indexer objective"),
    )
    query, key, value, _, _ = _inputs()
    with torch.no_grad():
        core(query, key, value, None, indexer_input=torch.randn(16, 1, 16, dtype=torch.bfloat16))
    assert seen["loss_coeff"] == 0.0


def test_core_indexer_only_training_uses_the_dense_teacher(monkeypatch):
    bound = {}
    core = _mock_core(monkeypatch, ssa_train_indexer_only=True)
    monkeypatch.setattr(
        simplified_sparse_attention,
        "bind_cudnn_sparse_attention",
        lambda **kwargs: bound.update(kwargs) or _kernels(None),
    )
    simplified_sparse_attention.SimplifiedSparseAttentionCore(
        core.config,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        pg_collection=SimpleNamespace(
            tp=SimpleNamespace(size=lambda: 1), cp=SimpleNamespace(size=lambda: 1)
        ),
    )
    assert bound == {"dense_teacher": True}
    seen = []

    def dense_teacher(kernels, q, k, v, qi, ki, **kwargs):
        seen.append(kwargs["loss_coeff"])
        output, loss = _fake_attention(kernels, q, k, v, qi, ki, topk=None, **kwargs)
        return output.detach(), loss if kwargs["loss_coeff"] > 0 else None

    logged = []
    monkeypatch.setattr(simplified_sparse_attention, "run_dense_teacher", dense_teacher)
    monkeypatch.setattr(
        simplified_sparse_attention,
        "run_sparse_attention",
        lambda *args, **kwargs: pytest.fail("Indexer-only training must not run sparse attention"),
    )
    monkeypatch.setattr(
        simplified_sparse_attention.DSAIndexerLossLoggingHelper,
        "save_loss_to_tracker",
        lambda **kw: logged.append(kw),
    )
    hidden = torch.randn(16, 1, 16, dtype=torch.bfloat16)
    query, key, value, _, _ = _inputs()
    output = core(query, key, value, None, indexer_input=hidden)
    output.float().sum().backward()
    assert all(parameter.grad is not None for parameter in core.indexer.parameters())
    assert query.grad is None and len(logged) == 1
    with torch.no_grad():
        core.eval()(query, key, value, None, indexer_input=hidden)
    assert seen == [0.1, 0.0] and len(logged) == 1


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("precision", ["bf16", "mxfp8", "mxfp8_params"])
@pytest.mark.parametrize("recompute", ["none", "core_attn"])
def test_transformer_layer_reuses_norm_and_preserves_bf16_indexer(
    monkeypatch, precision, recompute
):
    from megatron.core.enums import Fp8Recipe
    from megatron.core.fp8_utils import get_fp8_context, is_mxfp8tensor
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.experimental_attention_variant.simplified_sparse_attention_layer_specs import (  # pylint: disable=line-too-long
        ssa_stack_spec,
    )
    from megatron.core.transformer.spec_utils import build_module
    from tests.unit_tests.test_utilities import Utils

    if not torch.cuda.is_available():
        pytest.skip("Transformer Engine requires CUDA")
    mxfp8 = precision != "bf16"
    if mxfp8 and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 requires Blackwell")
    monkeypatch.setattr(
        simplified_sparse_attention, "bind_cudnn_sparse_attention", _bind_mock_kernels
    )
    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(1224)
        config = _config(
            ffn_hidden_size=512,
            recompute_granularity="selective" if recompute != "none" else None,
            recompute_modules=[recompute] if recompute != "none" else [],
            fp8="hybrid" if mxfp8 else None,
            fp8_recipe=Fp8Recipe.mxfp8,
            fp8_param=precision == "mxfp8_params",
        )
        # Enter the real outer parameter-quantization context: the indexer's
        # initialization guard must override it while QKV remains quantized.
        with get_fp8_context(config, is_init=True):
            layer = build_module(ssa_stack_spec.submodules.attention_layer, config=config).cuda()
        attention = layer.self_attention
        assert is_mxfp8tensor(attention.linear_qkv.weight) == config.fp8_param
        captured = {"qkv_calls": 0, "core_calls": 0}

        def capture_projection(module, inputs, output):
            captured["qkv_calls"] += 1
            captured["norm"] = output[2]

        attention.linear_qkv.register_forward_hook(capture_projection)
        attention.core_attention.register_forward_hook(
            lambda module, inputs, output: captured.update(core_calls=captured["core_calls"] + 1)
        )
        attention.core_attention.indexer.linear_q.register_forward_pre_hook(
            lambda module, inputs: captured.update(indexer=inputs[0])
        )
        monkeypatch.setattr(
            simplified_sparse_attention.DSAIndexerLossLoggingHelper,
            "save_loss_to_tracker",
            lambda **kw: None,
        )
        monkeypatch.setattr(
            simplified_sparse_attention.DSAIndexerLossAutoScaler, "main_loss_backward_scale", None
        )
        monkeypatch.setattr(simplified_sparse_attention, "run_sparse_attention", _fake_attention)
        hidden = torch.randn(32, 1, 256, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        with get_fp8_context(config):
            output, _ = layer(hidden, attention_mask=None)
        output.float().square().mean().backward()
        assert captured["qkv_calls"] == 1
        assert captured["core_calls"] == (2 if recompute == "core_attn" else 1)
        assert captured["indexer"].data_ptr() == captured["norm"].data_ptr()
        assert not captured["indexer"].requires_grad
        assert hidden.grad is not None
        for name, parameter in layer.named_parameters():
            if ".indexer." in name or "linear_qkv.weight" in name:
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
            if ".indexer." in name:
                assert parameter.dtype == torch.bfloat16
                assert type(parameter).__name__ == "Parameter"
        assert attention.core_attention.indexer.linear_q.config.fp8 is None
        assert attention.core_attention.indexer.linear_q.config.quant_recipe is None
        if precision == "bf16" and recompute == "none":

            def unexpected_decode(*args, **kwargs):
                pytest.fail("SSA must reject caching before entering inherited decode paths")

            monkeypatch.setattr(attention, "flash_decode", unexpected_decode)
            monkeypatch.setattr(attention, "_adjust_key_value_for_inference", unexpected_decode)
            context = SimpleNamespace(
                is_dynamic_batching=lambda: False,
                is_static_batching=lambda: True,
                is_decode_only=lambda: True,
            )
            with pytest.raises(NotImplementedError, match="decode caching"):
                attention.forward_pre_attn_and_core_attn(
                    hidden, attention_mask=None, inference_context=context
                )
    finally:
        Utils.destroy_model_parallel()


def _attention_oracle(inputs, coefficient, denominator, topk=None):
    q, k, v, qi, ki = (tensor.float() for tensor in inputs)
    q, k, v = q[:, 0], k[:, 0, 0], v[:, 0, 0]
    scores = torch.einsum("qhd,kd->hqk", q, k) * 256**-0.5
    causal = torch.ones(scores.shape[-2:], device=scores.device, dtype=torch.bool).tril()
    logits = qi[:, 0, 0] @ ki[:, 0, 0].T * 128**-0.5
    if topk is not None:
        indices = logits.detach().masked_fill(~causal, -torch.inf).topk(topk, dim=-1).indices
        selected = torch.zeros_like(causal).scatter_(1, indices, True)
        causal &= selected
    probability = scores.masked_fill(~causal, -torch.inf).softmax(-1)
    output = torch.einsum("hqk,kd->qhd", probability, v).reshape(inputs[0].size(0), 1, -1)
    target = probability.detach().mean(0)
    log_prediction = logits.masked_fill(~causal, -torch.inf).log_softmax(-1)
    terms = target * (target.clamp_min(1e-10).log() - log_prediction)
    loss = terms.masked_fill(~causal, 0).sum() * coefficient / denominator
    return output, loss


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize(
    "coefficient,per_token,sequence_length",
    [(0.0, False, 128), (0.03, False, 128), (0.2, True, 128), (0.03, False, 1024)],
)
def test_cudnn_attention_matches_output_loss_and_gradients(coefficient, per_token, sequence_length):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("cuDNN simplified sparse attention requires SM100 or SM103")
    try:
        kernels = simplified_sparse_attention_cudnn.bind_cudnn_sparse_attention()
    except ImportError as error:
        pytest.skip(str(error))
    torch.manual_seed(654)
    inputs = _inputs(sequence_length=sequence_length, device="cuda")
    selected_support = sequence_length > 512
    if selected_support:
        # Exact BF16 inputs give strictly ordered indexer scores. The selected
        # keys are unambiguous; no tolerance hides different top-k membership.
        with torch.no_grad():
            inputs[3].zero_()
            inputs[3][..., :2] = 1
            positions = torch.arange(sequence_length, device="cuda")
            inputs[4][:, 0, 0, 0] = (positions // 32) * 0.125
            inputs[4][:, 0, 0, 1] = (positions % 32) / 256
    refs = tuple(tensor.detach().float().requires_grad_() for tensor in inputs)
    output, loss = simplified_sparse_attention_cudnn.run_sparse_attention(
        kernels, *inputs, topk=512, loss_coeff=coefficient, calculate_per_token_loss=per_token
    )
    expected_output, expected_loss = _attention_oracle(
        refs,
        coefficient,
        1 if per_token else sequence_length,
        topk=512 if selected_support else None,
    )
    if selected_support:
        dense_output, _ = _attention_oracle(refs, coefficient, sequence_length)
        assert not torch.allclose(expected_output[512:], dense_output[512:], rtol=2e-2, atol=2e-2)
    upstream = torch.randn_like(output)
    ((output.float() * upstream).sum() + loss).backward()
    ((expected_output * upstream).sum() + expected_loss).backward()
    torch.testing.assert_close(output.float(), expected_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(loss, expected_loss, rtol=2e-2, atol=2e-4)
    for actual, expected in zip(inputs, refs):
        torch.testing.assert_close(actual.grad.float(), expected.grad, rtol=4e-2, atol=4e-2)
        reference_norm = torch.linalg.vector_norm(expected.grad)
        if reference_norm.item() == 0:
            assert torch.count_nonzero(actual.grad) == 0
        else:
            relative_error = (
                torch.linalg.vector_norm(actual.grad.float() - expected.grad) / reference_norm
            )
            assert relative_error.item() < 0.08


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize(
    "coefficient,per_token,sequence_length",
    [(0.03, False, 128), (0.2, True, 128), (0.03, False, 1024)],
)
def test_dense_teacher_matches_output_loss_and_indexer_gradients(
    coefficient, per_token, sequence_length
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("cuDNN simplified sparse attention requires SM100 or SM103")
    try:
        kernels = simplified_sparse_attention_cudnn.bind_cudnn_sparse_attention(dense_teacher=True)
    except ImportError as error:
        pytest.skip(str(error))
    torch.manual_seed(655)
    inputs = _inputs(sequence_length=sequence_length, device="cuda")
    refs = tuple(tensor.detach().float().requires_grad_() for tensor in inputs)
    output, loss = simplified_sparse_attention_cudnn.run_dense_teacher(
        kernels, *inputs, loss_coeff=coefficient, calculate_per_token_loss=per_token
    )
    expected_output, expected_loss = _attention_oracle(
        refs, coefficient, 1 if per_token else sequence_length
    )
    assert not output.requires_grad
    loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(output.float(), expected_output.detach(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(loss, expected_loss, rtol=2e-2, atol=2e-4)
    assert all(tensor.grad is None for tensor in inputs[:3])
    for actual, expected in zip(inputs[3:], refs[3:]):
        relative_error = torch.linalg.vector_norm(
            actual.grad.float() - expected.grad
        ) / torch.linalg.vector_norm(expected.grad)
        assert relative_error.item() < 0.08
