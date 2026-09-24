# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Contracts for the optional CuTe simplified sparse-attention backend."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import (
    dsa_cute_kernels,
    dsa_indexer_loss,
)
from megatron.core.transformer.transformer_config import TransformerConfig


def _config(**overrides):
    values = dict(
        num_layers=1,
        hidden_size=4096,
        num_attention_heads=16,
        num_query_groups=1,
        kv_channels=256,
        add_bias_linear=False,
        experimental_attention_variant="dsa",
        dsa_gqa_backend="cute",
        dsa_indexer_mode="simplified",
        dsa_simplified_use_learned_k=True,
        dsa_indexer_n_heads=1,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=512,
        dsa_indexer_loss_coeff=0.1,
        dsa_indexer_use_sparse_loss=True,
        dsa_indexer_rotate_activation=False,
        dsa_indexer_scoring_relu=False,
        rotary_percent=0.0,
        transformer_impl="transformer_engine",
        attention_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    values.update(overrides)
    return TransformerConfig(**values)


@pytest.mark.parametrize("topk", [512, 1024, 2048])
@pytest.mark.parametrize("heads", [16, 32, 96])
@pytest.mark.parametrize("topk_only", [False, True])
def test_cute_configuration_accepts_supported_shapes(topk, heads, topk_only):
    config = _config(
        dsa_indexer_topk=topk,
        num_attention_heads=heads,
        dsa_indexer_sparse_loss_use_topk_only=topk_only,
    )
    assert config.dsa_gqa_backend == "cute"


@pytest.mark.parametrize("coefficient", [0.0, 0.03, 0.1, 0.2])
def test_cute_configuration_accepts_nonnegative_loss_coefficient(coefficient):
    assert _config(dsa_indexer_loss_coeff=coefficient).dsa_indexer_loss_coeff == coefficient


@pytest.mark.parametrize("coefficient", [-0.1, float("nan"), float("inf")])
def test_cute_configuration_rejects_invalid_loss_coefficient(coefficient):
    with pytest.raises((ValueError, AssertionError), match="coefficient|loss_coeff"):
        _config(dsa_indexer_loss_coeff=coefficient)


def test_dsa_gqa_backend_defaults_to_reference():
    config = TransformerConfig(num_layers=1, hidden_size=32, num_attention_heads=4)
    assert config.dsa_gqa_backend == "reference"


@pytest.mark.parametrize("backend", ["reference", "torch-min-memory", "triton-min-memory"])
def test_simplified_learned_indexer_does_not_require_cute(backend):
    config = _config(dsa_gqa_backend=backend)
    assert config.dsa_indexer_mode == "simplified"
    assert config.dsa_gqa_backend == backend


@pytest.mark.parametrize("backend", ["reference", "torch-min-memory", "triton-min-memory", "cute"])
def test_dsa_gqa_backend_cli_reaches_configuration(backend):
    from argparse import ArgumentParser

    from megatron.training.arguments import _add_network_size_args

    parser = _add_network_size_args(ArgumentParser(exit_on_error=False))
    defaults = parser.parse_args([])
    assert defaults.dsa_gqa_backend == "reference"
    parsed = parser.parse_args(["--dsa-gqa-backend", backend])
    config = _config(dsa_gqa_backend=parsed.dsa_gqa_backend)
    assert config.dsa_gqa_backend == backend


def test_dsa_gqa_backend_cli_rejects_unknown_choice():
    from argparse import ArgumentError, ArgumentParser

    from megatron.training.arguments import _add_network_size_args

    parser = _add_network_size_args(ArgumentParser(exit_on_error=False))
    with pytest.raises(ArgumentError, match="invalid choice"):
        parser.parse_args(["--dsa-gqa-backend", "unknown"])


@pytest.mark.parametrize(
    "saved_backend,runtime_backend",
    [
        ("reference", "cute"),
        ("triton-min-memory", "cute"),
        ("cute", "reference"),
        ("cute", "triton-min-memory"),
    ],
)
def test_cute_checkpoint_restores_architecture_without_overriding_backend(
    monkeypatch, saved_backend, runtime_backend
):
    from megatron.training import checkpointing

    saved_args = SimpleNamespace(**vars(_config(dsa_gqa_backend=saved_backend)))
    runtime_args = SimpleNamespace(**vars(_config(dsa_gqa_backend=runtime_backend)))
    runtime_args.__dict__.update(
        load="checkpoint",
        vocab_file=None,
        data_parallel_random_init=False,
        phase_transition_iterations=None,
        use_dist_ckpt=True,
        use_tokenizer_model_from_checkpoint_args=False,
        use_mp_args_from_checkpoint_args=False,
    )
    runtime_args.dsa_indexer_head_dim = 64
    runtime_args.dsa_indexer_topk = 1024
    monkeypatch.setattr(checkpointing, "print_rank_0", lambda *_args: None)
    monkeypatch.setattr(checkpointing, "get_args", lambda: runtime_args)
    monkeypatch.setattr(checkpointing, "get_checkpoint_version", lambda: 3.0)
    monkeypatch.setattr(
        checkpointing,
        "_load_base_checkpoint",
        lambda *_args, **_kwargs: (
            {"args": saved_args, "iteration": 17},
            "checkpoint.pt",
            False,
            None,
        ),
    )

    restored, _ = checkpointing.load_args_from_checkpoint(runtime_args)
    assert restored.dsa_indexer_head_dim == 128
    assert restored.dsa_indexer_topk == 512
    assert restored.dsa_gqa_backend == runtime_backend
    checkpointing.check_checkpoint_args(saved_args)
    restored.dsa_indexer_head_dim = 64
    with pytest.raises(AssertionError, match="dsa_indexer_head_dim"):
        checkpointing.check_checkpoint_args(saved_args)


@pytest.mark.parametrize(
    "override",
    [
        {"context_parallel_size": 2},
        {"tensor_model_parallel_size": 2},
        {"num_query_groups": 2},
        {"num_attention_heads": 64},
        {"kv_channels": 128},
        {"dsa_indexer_head_dim": 64},
        {"dsa_indexer_topk": 256},
        {"dsa_simplified_use_learned_k": False},
        {"dsa_indexer_use_sparse_loss": False},
        {"dsa_indexer_topk_freq": 2},
        {"dsa_fwd_use_dense_attn": True},
        {"dsa_train_indexer_only": True},
        {"attention_dropout": 0.1},
        {"deterministic_mode": True},
    ],
)
def test_cute_configuration_rejects_unsupported_modes(override):
    with pytest.raises((ValueError, AssertionError)):
        _config(**override)


@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("valid_rows", [None, 0.0, 3.0])
def test_shared_loss_denominator_preserves_sum_and_mean_semantics(per_token, valid_rows):
    count = None if valid_rows is None else torch.tensor(valid_rows)
    expected = 1.0 if per_token else max(8 if count is None else valid_rows, 1.0)
    denominator = dsa_indexer_loss.get_indexer_loss_denominator(
        num_rows=8, calculate_per_token_loss=per_token, valid_row_count=count
    )
    torch.testing.assert_close(
        torch.as_tensor(denominator), torch.tensor(expected), check_dtype=False
    )
    value = torch.tensor(24.0, requires_grad=True)
    reduced = dsa_indexer_loss.reduce_indexer_kl_sum(
        value, num_rows=8, calculate_per_token_loss=per_token, valid_row_count=count
    )
    reduced.backward()
    torch.testing.assert_close(reduced, torch.tensor(24.0 / expected))
    torch.testing.assert_close(value.grad, torch.tensor(1.0 / expected))


def _inputs(sequence_length=16, heads=16, device="cuda"):
    shapes = (
        (sequence_length, 1, heads, 256),
        (sequence_length, 1, 1, 256),
        (sequence_length, 1, 1, 256),
        (sequence_length, 1, 1, 128),
        (sequence_length, 1, 1, 128),
    )
    return tuple(
        torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True)
        for shape in shapes
    )


@pytest.mark.parametrize("coefficient", [0.0, 0.03, 0.2])
@pytest.mark.parametrize("per_token", [False, True])
def test_adapter_preserves_views_loss_scaling_and_all_gradient_paths(
    monkeypatch, coefficient, per_token
):
    inputs = _inputs(device="cpu")
    seen = {}
    denominator = 1 if per_token else inputs[0].size(0)

    def sparse_attention(q, k, v, qi, ki, layout, topk, *, loss_coeff):
        seen.update(q=q, k=k, v=v, qi=qi, ki=ki, layout=layout, topk=topk)
        # Distinct paths expose missing gradients, double scaling and unintended detaches.
        output = q + k.expand_as(q) + v.expand_as(q)
        loss = (qi.float().square().sum() + ki.float().square().sum()) * loss_coeff
        return output, loss / layout.loss_denominator.squeeze()

    def metadata(starts, ends, valid, denominator, **kwargs):
        return SimpleNamespace(
            key_starts=starts,
            key_ends=ends,
            query_valid=valid,
            loss_denominator=denominator,
            **kwargs,
        )

    monkeypatch.setitem(
        sys.modules,
        "simplified_sparse_attention",
        SimpleNamespace(RowMetadata=metadata, sparse_attention=sparse_attention),
    )
    output, loss = dsa_cute_kernels.run_cute_sparse_attention(
        *inputs,
        topk=512,
        softmax_scale=256**-0.5,
        loss_coeff=coefficient,
        loss_denominator=denominator,
    )
    assert output.shape == (16, 1, 16 * 256)
    for name, original in zip(("q", "k", "v", "qi", "ki"), inputs):
        assert seen[name].data_ptr() == original.data_ptr()
    assert seen["qi"].shape == (16, 128)
    assert seen["layout"].triangular_scores
    torch.testing.assert_close(seen["layout"].key_starts, torch.zeros(16, dtype=torch.int64))
    torch.testing.assert_close(seen["layout"].key_ends, torch.arange(1, 17, dtype=torch.int64))
    (output.float().sum() + loss).backward()
    torch.testing.assert_close(inputs[0].grad, torch.ones_like(inputs[0]))
    for tensor in inputs[1:3]:
        torch.testing.assert_close(tensor.grad, torch.full_like(tensor, 16))
    for tensor in inputs[3:]:
        expected = (2 * tensor.detach().float() * coefficient / denominator).to(tensor.dtype)
        torch.testing.assert_close(tensor.grad, expected)


def test_explicit_cute_selection_reports_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "simplified_sparse_attention", None)
    with pytest.raises(RuntimeError, match="simplified-sparse-attention package"):
        dsa_cute_kernels.run_cute_sparse_attention(
            *_inputs(device="cpu"),
            topk=512,
            softmax_scale=256**-0.5,
            loss_coeff=0.1,
            loss_denominator=16,
        )


@pytest.mark.parametrize("return_bias", [False, True])
@pytest.mark.parametrize("return_norm", [False, True])
def test_te_normalized_output_is_opt_in_without_changing_ordinary_return(
    monkeypatch, return_bias, return_norm
):
    from megatron.core.extensions import transformer_engine as te_wrappers

    wrapper_type = te_wrappers.TELayerNormColumnParallelLinear
    wrapper = wrapper_type.__new__(wrapper_type)
    torch.nn.Module.__init__(wrapper)
    wrapper.layer_norm_weight = torch.nn.Parameter(torch.ones(4))
    wrapper.te_quant_params = None
    wrapper.te_return_bias = return_bias
    wrapper.return_layernorm_output = return_norm
    output, bias, normalized = (torch.randn(2, 4) for _ in range(3))
    native_result = [output]
    if return_bias:
        native_result.append(bias)
    if return_norm:
        native_result.append(normalized)
    native_result = tuple(native_result) if len(native_result) > 1 else output
    monkeypatch.setattr(wrapper_type.__bases__[0], "forward", lambda *args, **kwargs: native_result)
    monkeypatch.setattr(te_wrappers, "_resolve_is_first_microbatch", lambda _module: True)
    monkeypatch.setattr(
        te_wrappers, "_get_fp8_autocast_for_quant_params", lambda *_args: nullcontext()
    )
    result = wrapper(torch.randn(2, 4))
    assert len(result) == (3 if return_norm else 2)
    assert result[0] is output
    assert result[1] is (bias if return_bias else None)
    if return_norm:
        assert result[2] is normalized


class _TestLinear(torch.nn.Module):
    """Ordinary linear with the projection tuple contract, without TE initialization."""

    def __init__(self, input_size, output_size, *, config, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(output_size, input_size, dtype=config.params_dtype) * 0.05
        )

    def forward(self, inputs):
        return torch.nn.functional.linear(inputs, self.weight), None


def _core_with_indexer(monkeypatch, coefficient):
    from megatron.core.transformer.experimental_attention_variant import dsa_gqa

    monkeypatch.setattr(dsa_gqa, "get_fp8_disabled_context", lambda *args, **kwargs: nullcontext())
    config = _config(hidden_size=16, dsa_indexer_loss_coeff=coefficient)
    core = dsa_gqa.DSGQACoreAttention.__new__(dsa_gqa.DSGQACoreAttention)
    torch.nn.Module.__init__(core)
    core.config = config
    core.layer_number = 1
    core.softmax_scale = 256**-0.5
    core.indexer = dsa_gqa.SimplifiedDSGQAIndexer(
        config=config,
        submodules=dsa_gqa.SimplifiedDSGQAIndexerSubmodules(
            linear_q=_TestLinear, linear_k=_TestLinear
        ),
        pg_collection=SimpleNamespace(tp=SimpleNamespace(size=lambda: 1), cp=None, dp_cp=None),
    )
    return core


@pytest.mark.parametrize("coefficient", [0.0, 0.2])
def test_cute_core_trains_existing_indexer_without_auxiliary_backbone_gradients(
    monkeypatch, coefficient
):
    from megatron.core.transformer.experimental_attention_variant import dsa_gqa

    torch.manual_seed(42)
    core = _core_with_indexer(monkeypatch, coefficient)
    monkeypatch.setattr(
        dsa_gqa.DSAIndexerLossLoggingHelper, "save_loss_to_tracker", lambda **kwargs: None
    )
    monkeypatch.setattr(dsa_gqa.DSAIndexerLossAutoScaler, "main_loss_backward_scale", None)
    hidden = torch.randn(16, 1, 16, dtype=torch.bfloat16, requires_grad=True)
    query, key, value, _, _ = _inputs(device="cpu")

    def fake_attention(query, key, value, qi, ki, **kwargs):
        assert qi.dtype == ki.dtype == torch.bfloat16
        output = query + key.expand_as(query) + value.expand_as(query)
        loss = qi.float().square().sum() + ki.float().square().sum()
        loss = loss * kwargs["loss_coeff"] / kwargs["loss_denominator"]
        return output.flatten(2), loss

    monkeypatch.setattr(dsa_cute_kernels, "run_cute_sparse_attention", fake_attention)
    output = core._forward_cute(query, key, value, hidden, indexer_input_norm=None)
    output.float().sum().backward()
    assert hidden.grad is None
    assert set(dict(core.named_parameters())) == {
        "indexer.linear_q.weight",
        "indexer.linear_k.weight",
    }
    for weight in core.indexer.parameters():
        assert weight.grad is not None
        if coefficient == 0:
            assert torch.count_nonzero(weight.grad) == 0
        else:
            assert torch.count_nonzero(weight.grad) > 0
    torch.testing.assert_close(query.grad, torch.ones_like(query))
    for tensor in (key, value):
        torch.testing.assert_close(tensor.grad, torch.full_like(tensor, 16))


def test_cute_core_rejects_runtime_rope_before_dispatch(monkeypatch):
    from megatron.core.transformer.enums import AttnMaskType

    core = _core_with_indexer(monkeypatch, 0.1)
    query, key, value, _, _ = _inputs(device="cpu")
    with pytest.raises(NotImplementedError, match="NoPE"):
        core(
            query,
            key,
            value,
            attention_mask=None,
            hidden_states=torch.zeros(16, 1, 16, dtype=torch.bfloat16),
            attn_mask_type=AttnMaskType.causal,
            use_indexer_rope=True,
        )


@pytest.mark.parametrize("indexer_clip", [None, 0.0, 2.0])
def test_cute_indexer_preserves_existing_separate_clip_groups(monkeypatch, indexer_clip):
    from megatron.core.optimizer import OptimizerConfig, _get_param_groups
    from megatron.core.optimizer.optimizer import FP32Optimizer

    if not torch.cuda.is_available():
        pytest.skip("Existing FP32 optimizer clipping uses CUDA multi-tensor kernels")
    core = _core_with_indexer(monkeypatch, 0.1).float().cuda()
    core.register_parameter("main_weight", torch.nn.Parameter(torch.zeros(1, device="cuda")))
    config = OptimizerConfig(
        lr=0.01,
        clip_grad=1.0,
        dsa_separate_indexer_grad_clip=True,
        dsa_indexer_clip_grad=indexer_clip,
    )
    assert not OptimizerConfig().dsa_separate_indexer_grad_clip

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda **kwargs: 1)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda results, value, **kwargs: results.__setitem__(0, value),
    )
    groups = _get_param_groups([core], config, config_overrides={})
    indexed_parameters = {
        id(p) for group in groups if group["is_dsa_indexer"] for p in group["params"]
    }
    assert indexed_parameters == {id(p) for p in core.indexer.parameters()}

    optimizer = FP32Optimizer(
        torch.optim.SGD(groups, lr=config.lr), config, init_state_fn=lambda _: None
    )
    for parameter in core.parameters():
        parameter.grad = torch.zeros_like(parameter)
    # Distinct bucket norms catch accidental joint clipping or use of the main threshold.
    core.main_weight.grad.fill_(3.0)
    core.indexer.linear_q.weight.grad.view(-1)[0] = 4.0
    monkeypatch.setattr(optimizer, "get_dsa_split_grad_norms", lambda: (4.0, 3.0))
    assert optimizer.clip_grad_norm_separate_dsa_indexer(config.clip_grad) == 5.0
    torch.testing.assert_close(
        core.main_weight.grad, torch.ones_like(core.main_weight), rtol=1e-5, atol=1e-5
    )
    expected_indexer = 1.0 if indexer_clip is None else (4.0 if indexer_clip == 0 else indexer_clip)
    torch.testing.assert_close(
        core.indexer.linear_q.weight.grad.view(-1)[0],
        torch.tensor(expected_indexer, device="cuda"),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("checkpoint_core", [False, True])
@pytest.mark.parametrize("mxfp8", [False, True])
def test_cute_transformer_layer_reuses_normalized_input_and_backpropagates(
    monkeypatch, checkpoint_core, mxfp8
):
    from megatron.core.enums import Fp8Recipe
    from megatron.core.fp8_utils import get_fp8_context
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.experimental_attention_variant import dsa_gqa
    from megatron.core.transformer.experimental_attention_variant.dsa_layer_specs import (
        dsa_stack_spec,
    )
    from megatron.core.transformer.spec_utils import build_module
    from tests.unit_tests.test_utilities import Utils

    if not torch.cuda.is_available():
        pytest.skip("Transformer Engine construction requires CUDA")
    if mxfp8 and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 GEMMs require Blackwell")
    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(1224)
        config = _config(
            hidden_size=256,
            ffn_hidden_size=512,
            hidden_dropout=0.0,
            recompute_granularity="selective" if checkpoint_core else None,
            recompute_modules=["core_attn"] if checkpoint_core else [],
            fp8="hybrid" if mxfp8 else None,
            fp8_recipe=Fp8Recipe.mxfp8,
        )
        layer = build_module(dsa_stack_spec.submodules.attention_layer, config=config).cuda()
        attention = layer.self_attention
        assert isinstance(attention, dsa_gqa.DSGroupedSelfAttention)
        assert attention.linear_qkv.return_layernorm_output
        captured = {}
        attention.linear_qkv.register_forward_hook(
            lambda _module, _inputs, result: captured.update(normalized=result[2])
        )
        attention.core_attention.indexer.linear_q.register_forward_pre_hook(
            lambda _module, inputs: captured.update(indexer_input=inputs[0])
        )
        monkeypatch.setattr(
            dsa_gqa.DSAIndexerLossLoggingHelper, "save_loss_to_tracker", lambda **kwargs: None
        )
        monkeypatch.setattr(dsa_gqa.DSAIndexerLossAutoScaler, "main_loss_backward_scale", None)

        def fake_attention(q, k, v, qi, ki, **kwargs):
            output = q + k.expand_as(q) + v.expand_as(q)
            loss = qi.float().square().sum() + ki.float().square().sum()
            return output.flatten(2), loss * kwargs["loss_coeff"] / kwargs["loss_denominator"]

        monkeypatch.setattr(dsa_cute_kernels, "run_cute_sparse_attention", fake_attention)
        hidden = torch.randn(32, 1, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        with get_fp8_context(config):
            output, _ = layer(hidden, attention_mask=None)
        output.float().square().mean().backward()
        assert captured["indexer_input"].data_ptr() == captured["normalized"].data_ptr()
        assert not captured["indexer_input"].requires_grad
        assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
        for name, parameter in layer.named_parameters():
            if "linear_qkv.weight" in name or ".indexer." in name:
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
                assert torch.count_nonzero(parameter.grad) > 0, name
    finally:
        Utils.destroy_model_parallel()


def _dense_causal_oracle(inputs, coefficient, denominator):
    query, key, value, qi, ki = (tensor.float() for tensor in inputs)
    query, key, value = query[:, 0], key[:, 0, 0], value[:, 0, 0]
    scores = torch.einsum("qhd,kd->hqk", query, key) * 256**-0.5
    causal = torch.ones(scores.shape[-2:], device=scores.device, dtype=torch.bool).tril()
    probabilities = scores.masked_fill(~causal, -torch.inf).softmax(-1)
    output = torch.einsum("hqk,kd->qhd", probabilities, value)
    target = probabilities.detach().mean(0)
    index_scores = qi[:, 0, 0] @ ki[:, 0, 0].T * 128**-0.5
    log_prediction = index_scores.masked_fill(~causal, -torch.inf).log_softmax(-1)
    terms = target * (target.clamp_min(1e-10).log() - log_prediction)
    loss = terms.masked_fill(~causal, 0).sum() * coefficient / denominator
    return output.reshape(inputs[0].size(0), 1, -1), loss


@pytest.mark.parametrize("heads,topk", [(16, 512), (32, 1024), (96, 2048)])
@pytest.mark.parametrize("coefficient", [0.0, 0.03, 0.1, 0.2])
@pytest.mark.parametrize("per_token", [False, True])
def test_cute_gpu_output_loss_and_gradients_match_causal_reference(
    heads, topk, coefficient, per_token
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("CuTe simplified sparse attention requires SM103")
    pytest.importorskip("simplified_sparse_attention")
    torch.manual_seed(654)
    # Top-k covers every causal key, isolating differentiable math from top-k ties.
    inputs = _inputs(sequence_length=128, heads=heads)
    reference_inputs = tuple(t.detach().float().requires_grad_() for t in inputs)
    denominator = 1 if per_token else 128
    output, loss = dsa_cute_kernels.run_cute_sparse_attention(
        *inputs,
        topk=topk,
        softmax_scale=256**-0.5,
        loss_coeff=coefficient,
        loss_denominator=denominator,
    )
    reference_output, reference_loss = _dense_causal_oracle(
        reference_inputs, coefficient, denominator
    )
    upstream = torch.randn_like(output)
    (output.float() * upstream).sum().add(loss).backward()
    (reference_output * upstream).sum().add(reference_loss).backward()
    torch.testing.assert_close(output.float(), reference_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(loss, reference_loss, rtol=2e-2, atol=2e-4)
    for name, actual, expected in zip(
        ("query", "key", "value", "indexer_q", "indexer_k"), inputs, reference_inputs
    ):
        actual_grad = actual.grad.float()
        expected_grad = expected.grad
        torch.testing.assert_close(actual_grad, expected_grad, rtol=4e-2, atol=4e-2)
        reference_norm = torch.linalg.vector_norm(expected_grad)
        if reference_norm.item() == 0.0:
            assert torch.count_nonzero(actual_grad).item() == 0, name
        else:
            relative_error = torch.linalg.vector_norm(actual_grad - expected_grad) / reference_norm
            assert (
                relative_error.item() < 0.08
            ), f"{name}: relative gradient error {relative_error.item():.4f}"
