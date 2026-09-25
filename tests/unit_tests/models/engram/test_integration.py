# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import copy
from argparse import Namespace
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from megatron.core import recompute
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.ssm.gated_delta_net import HAVE_FLA
from megatron.core.ssm.mamba_mixer import HAVE_MAMBA_SSM
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.engram import Engram, hybrid_adapter
from megatron.core.transformer.engram.addressing import NgramHashMapping
from megatron.core.transformer.engram.hybrid_adapter import (
    EngramAttentionLayer,
    EngramHybridProvider,
    resolve_engram_targets,
)
from megatron.core.transformer.engram.tokenizer import CompressedTokenizer
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.training.models.hybrid import HybridModelBuilder, engram_context_provider_spec
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.training.models.test_hybrid_builder import (
    _make_hybrid_config,
    _make_transformer,
)


@contextmanager
def model_parallel(*args, **kwargs):
    """Scope native process groups for generated-tensor regression cases."""
    Utils.initialize_model_parallel(*args, **kwargs)
    try:
        yield
    finally:
        Utils.destroy_model_parallel()


def prepare_row_gradients(model):
    """Bind standalone lookup gradients for tests without a native DDP wrapper."""
    for param in model.parameters():
        if getattr(getattr(param, 'engram_table_metadata', None), 'row_parallel', False):
            param.main_grad = torch.zeros_like(param, dtype=torch.float32)
    return model


def parameter_gradient(param):
    """Read the gradient representation owned by the parameter's training path."""
    return (
        param.main_grad
        if getattr(getattr(param, 'engram_table_metadata', None), 'row_parallel', False)
        else param.grad
    )


def make_transformer_config(**kwargs):
    values = dict(
        num_layers=6,
        hidden_size=32,
        num_attention_heads=4,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=True,
        add_bias_linear=False,
        engram_hash_table_min_sizes=[17, 19],
        engram_max_ngram_size=3,
        engram_embedding_dim_per_ngram=8,
        engram_num_hash_heads_per_ngram=2,
        engram_layer_ids=[0, 2],
        engram_seed=7,
    )
    values.update(kwargs)
    return TransformerConfig(**values)


def make_hybrid(
    config,
    pattern='*-*-*-',
    *,
    vocab_size=64,
    sequence_length=8,
    position_embedding_type='none',
    pad_id=2,
    **model_kwargs,
):
    provider = (
        ModuleSpec(
            module=EngramHybridProvider,
            params=dict(
                tokenizer_lookup=torch.arange(vocab_size),
                pad_id=pad_id,
                hybrid_layer_pattern=pattern,
            ),
        )
        if config.engram_layer_ids
        else None
    )
    return prepare_row_gradients(
        HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            hybrid_layer_pattern=pattern,
            vocab_size=vocab_size,
            max_sequence_length=sequence_length,
            position_embedding_type=position_embedding_type,
            token_context_provider_spec=provider,
            **model_kwargs,
        ).cuda()
    )


def model_inputs():
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device="cuda")
    positions = torch.arange(8, device="cuda").unsqueeze(0)
    mask = torch.triu(torch.ones(8, 8, dtype=torch.bool, device="cuda"), diagonal=1).view(
        1, 1, 8, 8
    )
    return tokens, positions, mask


def test_feature_off_builder_does_not_access_tokenizer(monkeypatch):
    def unexpected_tokenizer_access():
        raise AssertionError("feature-off must not access the tokenizer")

    monkeypatch.setattr("megatron.training.get_tokenizer", unexpected_tokenizer_access)
    assert (
        engram_context_provider_spec(make_transformer_config(engram_layer_ids=None), "*-*-*-")
        is None
    )


def test_engram_requires_lookup_and_main_tokenizer_pad_id():
    config = make_transformer_config()
    with pytest.raises(ValueError, match="precomputed tokenizer lookup"):
        Engram(config=config)
    with pytest.raises(ValueError, match="raw pad token ID"):
        Engram(config=config, tokenizer_lookup=torch.arange(64))


@pytest.mark.parametrize("full_recompute", [False, True])
def test_engram_and_hash_moe_token_inputs_coexist(full_recompute):
    """Hybrid forwards raw hash-MoE IDs and Engram context through the same call."""
    with model_parallel():
        overrides = dict(
            num_layers=4,
            engram_layer_ids=[0],
            num_moe_experts=4,
            moe_num_hash_layers=1,
            hash_moe_vocab_size=64,
            moe_router_topk=1,
            moe_router_pre_softmax=True,
            moe_router_load_balancing_type="none",
            moe_token_dispatcher_type="allgather",
        )
        if full_recompute:
            overrides.update(
                recompute_granularity="full", recompute_method="uniform", recompute_num_layers=2
            )
        model = make_hybrid(make_transformer_config(**overrides), pattern="*-*E")
        output = model(*model_inputs())
        output.float().mean().backward()
        assert torch.isfinite(output).all()
        table_grads = [
            parameter_gradient(param)
            for param in model.parameters()
            if getattr(param, "engram_table_metadata", None) is not None
        ]
        assert table_grads and all(grad is not None for grad in table_grads)
        assert all(torch.isfinite(grad).all() for grad in table_grads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_feature_off_hybrid_has_no_engram_state_and_forwards():
    with model_parallel(1, 1):
        model_parallel_cuda_manual_seed(123)
        model = make_hybrid(make_transformer_config(engram_layer_ids=None))
        assert not any(isinstance(layer, EngramAttentionLayer) for layer in model.decoder.layers)
        assert not any("engram" in name for name in model.state_dict())
        with torch.no_grad():
            output = model(*model_inputs())
        assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_selected_layers_forward_and_strict_state_roundtrip():
    with model_parallel(1, 1):
        model_parallel_cuda_manual_seed(123)
        config = make_transformer_config()
        model = make_hybrid(config)
        assert not hasattr(model, 'engram')
        assert list(model.decoder.layers[0].engram.layers) == ['0']
        assert list(model.decoder.layers[4].engram.layers) == ['2']
        first_hashes = model.decoder.layers[0].engram.hash_mapping.hash_moduli_by_layer
        last_hashes = model.decoder.layers[4].engram.hash_mapping.hash_moduli_by_layer
        assert first_hashes == last_hashes
        assert first_hashes[0] != first_hashes[2]
        selected = [
            i
            for i, layer in enumerate(model.decoder.layers)
            if isinstance(layer, EngramAttentionLayer)
        ]
        assert selected == [0, 4]
        parameter_ids = [id(p) for _, p in model.named_parameters(remove_duplicate=False)]
        assert len(parameter_ids) == len(set(parameter_ids))
        with torch.no_grad():
            output = model(*model_inputs())
        assert torch.isfinite(output).all()
        restored = make_hybrid(config)
        result = restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        assert not result.missing_keys and not result.unexpected_keys
        with torch.no_grad():
            torch.testing.assert_close(restored(*model_inputs()), output, rtol=0, atol=0)
        with pytest.raises(NotImplementedError, match="inference"):
            model(*model_inputs(), inference_context=object())
        with pytest.raises(ValueError, match="packed sequences"):
            model(*model_inputs(), packed_seq_params=object())

        original_decoder_forward = model.decoder.forward

        def fail_after_compression(*args, **kwargs):
            assert kwargs["token_context"] is not None
            raise RuntimeError("decoder failed")

        model.decoder.forward = fail_after_compression
        with pytest.raises(RuntimeError, match="decoder failed"):
            model(*model_inputs())
        model.decoder.forward = original_decoder_forward
        with torch.no_grad():
            torch.testing.assert_close(model(*model_inputs()), output, rtol=0, atol=0)


def test_cli_exposes_only_supported_engram_options(monkeypatch):
    from megatron.training.arguments import parse_args

    argv = [
        "pretrain_hybrid.py",
        "--hybrid-layer-pattern",
        "*-" * 16,
        "--hidden-size",
        "8",
        "--num-attention-heads",
        "2",
        "--seq-length",
        "4",
        "--max-position-embeddings",
        "4",
        "--micro-batch-size",
        "1",
        "--num-experts",
        "8",
        "--engram-layer-ids",
        "1",
        "9",
        "--engram-table-backend",
        "row_a2a",
        "--engram-hash-table-min-sizes",
        "17",
        "19",
    ]
    monkeypatch.setattr("sys.argv", argv)
    args = parse_args(ignore_unknown_args=True)
    assert not hasattr(args, "engram_enabled")
    assert not hasattr(args, "engram_schema_version")
    assert not hasattr(args, "engram_recompute_policy")
    assert not hasattr(args, "hc_mult")
    assert args.engram_layer_ids == [1, 9]
    assert args.engram_table_backend == "row_a2a"
    assert args.engram_hash_table_min_sizes == [17, 19]


def test_engram_cli_validation_rejects_incompatible_modes(monkeypatch):
    from megatron.training.arguments import parse_args, validate_engram_args

    def parse(extra):
        argv = [
            "pretrain_hybrid.py",
            "--hybrid-layer-pattern",
            "*-*-",
            "--hidden-size",
            "8",
            "--num-attention-heads",
            "2",
            "--max-position-embeddings",
            "4",
            "--engram-layer-ids",
            "0",
            "--engram-hash-table-min-sizes",
            "17",
            "19",
            "--tokenizer-model",
            "/local/tokenizer",
            *extra,
        ]
        monkeypatch.setattr("sys.argv", argv)
        return parse_args(ignore_unknown_args=True)

    with pytest.raises(ValueError, match="init-model-with-meta-device"):
        validate_engram_args(parse(["--init-model-with-meta-device"]))
    validate_engram_args(parse([]))
    with pytest.raises(ValueError, match="overlap-moe-expert-parallel-comm"):
        validate_engram_args(parse(["--overlap-moe-expert-parallel-comm"]))
    validate_engram_args(
        parse(["--engram-table-backend", "row_a2a", "--ckpt-format", "torch_dist", "--bf16"])
    )
    with pytest.raises(ValueError, match="unity loss scaling"):
        validate_engram_args(parse(["--engram-table-backend", "row_a2a", "--loss-scale", "128"]))
    with pytest.raises(ValueError, match="ckpt-format torch_dist"):
        validate_engram_args(parse(["--engram-table-backend", "row_a2a", "--ckpt-format", "torch"]))
    with pytest.raises(ValueError, match="native Adam"):
        validate_engram_args(parse(["--engram-table-backend", "row_a2a", "--optimizer", "sgd"]))
    with pytest.raises(ValueError, match="enable-mhc-connections"):
        validate_engram_args(parse(["--enable-mhc-connections"]))


@pytest.mark.parametrize('method', [None, 'uniform', 'block'])
def test_mtp_forward_backward_and_recompute_matches_eager(method):
    with model_parallel(1, 1):
        torch.manual_seed(2026)
        model_parallel_cuda_manual_seed(2026)
        config = make_transformer_config(
            mtp_num_layers=1, mtp_loss_scaling_factor=0.3, recompute_granularity=None
        )
        reference = make_hybrid(config, '*-*-*-/*-')
        candidate_config = make_transformer_config(
            mtp_num_layers=1,
            mtp_loss_scaling_factor=0.3,
            recompute_granularity='full' if method else None,
            recompute_method=method,
            recompute_num_layers=1 if method else None,
        )
        candidate = make_hybrid(candidate_config, '*-*-*-/*-')
        candidate.load_state_dict(reference.state_dict(), strict=True)
        assert not any(isinstance(module, Engram) for module in candidate.mtp.modules())
        inputs = model_inputs()
        labels = inputs[0].roll(-1, dims=1)
        mask = torch.ones_like(labels, dtype=torch.float)
        expected = reference(*inputs, labels=labels, loss_mask=mask)
        expected.mean().backward()
        actual = candidate(*inputs, labels=labels, loss_mask=mask)
        actual.mean().backward()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        expected_parameters = dict(reference.named_parameters())
        for name, parameter in candidate.named_parameters():
            other = expected_parameters[name]
            if other.grad is None:
                assert parameter.grad is None, name
            else:
                torch.testing.assert_close(parameter.grad, other.grad, rtol=2e-4, atol=1e-6)
        assert candidate.decoder.layers[0].engram.layers['0'].value_proj.weight.grad is not None


CASES = [
    ("local", "*-*-*-", {}),
    ("row_a2a", "*-*-*-", {"engram_table_backend": "row_a2a"}),
    ("moe", "*E*E*E", {"num_moe_experts": 4, "moe_ffn_hidden_size": 32, "moe_router_topk": 2}),
    ("mixed", "*-*E*E", {"num_moe_experts": 4, "moe_ffn_hidden_size": 32, "moe_router_topk": 2}),
    (
        "recompute",
        "*-*-*-",
        {"recompute_granularity": "full", "recompute_method": "uniform", "recompute_num_layers": 2},
    ),
]


@pytest.mark.parametrize("name,pattern,overrides", CASES)
def test_engram_hybrid_configuration_matrix(name, pattern, overrides):
    with model_parallel(1, 1):
        model_parallel_cuda_manual_seed(123)
        model = make_hybrid(make_transformer_config(**overrides), pattern)
        selected = [
            i
            for i, layer in enumerate(model.decoder.layers)
            if isinstance(layer, EngramAttentionLayer)
        ]
        assert selected == [0, 4]
        output = model(*model_inputs())
        assert torch.isfinite(output).all(), name
        output.float().square().mean().backward()
        for index in selected:
            assert any(p.grad is not None for p in model.decoder.layers[index].engram.parameters())


def test_hybrid_full_recompute_preserves_output_and_gradients():
    with model_parallel(1, 1):
        model_parallel_cuda_manual_seed(123)
        direct = make_hybrid(make_transformer_config())
        recomputed = make_hybrid(
            make_transformer_config(
                recompute_granularity="full", recompute_method="uniform", recompute_num_layers=2
            )
        )
        recomputed.load_state_dict(copy.deepcopy(direct.state_dict()), strict=True)
        expected = direct(*model_inputs())
        actual = recomputed(*model_inputs())
        expected.float().square().mean().backward()
        actual.float().square().mean().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected_grads = {name: p.grad for name, p in direct.named_parameters()}
        for name, p in recomputed.named_parameters():
            grad = expected_grads[name]
            assert (p.grad is None) == (grad is None), name
            if grad is not None:
                torch.testing.assert_close(p.grad, grad, rtol=1e-5, atol=1e-7, msg=name)


@pytest.mark.parametrize("method,num_layers", [("uniform", 2), ("block", 5)])
def test_row_a2a_hybrid_recompute_preserves_interleaved_microbatches(method, num_layers):
    with model_parallel(1, 1):
        model_parallel_cuda_manual_seed(123)
        direct = make_hybrid(make_transformer_config(engram_table_backend="row_a2a"))
        recomputed = make_hybrid(
            make_transformer_config(
                engram_table_backend="row_a2a",
                recompute_granularity="full",
                recompute_method=method,
                recompute_num_layers=num_layers,
            )
        )
        recomputed.load_state_dict(copy.deepcopy(direct.state_dict()), strict=True)
        tokens, positions, mask = model_inputs()
        microbatches = [(tokens, positions, mask), (tokens.flip(-1) + 8, positions, mask)]

        # Both forwards precede either backward. Replaying the first microbatch
        # must use its own tokens even after the provider has prepared the second.
        expected = [direct(*inputs) for inputs in microbatches]
        actual = [recomputed(*inputs) for inputs in microbatches]
        expected_parameters = dict(direct.named_parameters())
        actual_parameters = dict(recomputed.named_parameters())
        assert expected_parameters.keys() == actual_parameters.keys()

        for microbatch, (actual_output, expected_output) in enumerate(zip(actual, expected)):
            torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
            expected_output.float().square().mean().backward()
            actual_output.float().square().mean().backward()

            # Check both the first gradient and the accumulated gradient without
            # clearing either model between microbatches. Compare all owner rows,
            # including untouched zeros, to detect incorrect token context.
            row_gradients = 0
            dense_gradients = 0
            for name, parameter in actual_parameters.items():
                actual_grad = parameter_gradient(parameter)
                expected_grad = parameter_gradient(expected_parameters[name])
                message = f"{method}, microbatch {microbatch}, {name}"
                assert (actual_grad is None) == (expected_grad is None), message
                if expected_grad is None:
                    continue
                if getattr(
                    getattr(parameter, 'engram_table_metadata', None), 'row_parallel', False
                ):
                    row_gradients += 1
                else:
                    dense_gradients += 1
                torch.testing.assert_close(
                    actual_grad, expected_grad, rtol=1e-5, atol=1e-7, msg=message
                )
            assert row_gradients == len(direct.config.engram_layer_ids)
            assert dense_gradients > 0


def _heterogeneous_model(enabled, backend, family):
    pattern = family + '*-*E/*E'
    config = TransformerConfig(
        num_layers=5,
        hidden_size=128,
        num_attention_heads=2,
        kv_channels=64,
        ffn_hidden_size=256,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        add_bias_linear=False,
        normalization='RMSNorm',
        activation_func=F.silu,
        gated_linear_unit=True,
        mamba_num_groups=1,
        mamba_state_dim=16,
        mamba_head_dim=64,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_ffn_hidden_size=128,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type='alltoall',
        mtp_num_layers=1,
        mtp_loss_scaling_factor=0.3,
        engram_layer_ids=[0] if enabled else None,
        engram_target_layer_indices=[1] if enabled else None,
        engram_hash_table_min_sizes=[101, 103],
        engram_embedding_dim_per_ngram=16,
        engram_num_hash_heads_per_ngram=2,
        engram_table_backend=backend,
    )
    model = make_hybrid(
        config,
        pattern,
        vocab_size=256,
        sequence_length=128,
        position_embedding_type='rope',
        pad_id=0,
    )
    return Float16Module(config, model)


@pytest.mark.parametrize('family', ['M', 'G'])
@pytest.mark.parametrize('backend', ['local', 'row_a2a'])
def test_engram_composes_with_recurrent_attention_moe_and_mtp(monkeypatch, backend, family):
    if family == 'M' and not HAVE_MAMBA_SSM:
        pytest.skip('Mamba SSM extension is unavailable')
    if family == 'G' and not HAVE_FLA:
        pytest.skip('Flash Linear Attention extension is unavailable')
    Utils.initialize_model_parallel()
    monkeypatch.setenv('NVTE_ALLOW_NONDETERMINISTIC_ALGO', '1')
    try:
        torch.manual_seed(2026)
        model_parallel_cuda_manual_seed(2026)
        reference = _heterogeneous_model(False, backend, family)
        enabled = _heterogeneous_model(True, backend, family)
        result = enabled.module.load_state_dict(reference.module.state_dict(), strict=False)
        assert not result.unexpected_keys
        assert result.missing_keys and all('.engram.' in key for key in result.missing_keys)
        assert not any(isinstance(module, Engram) for module in reference.modules())
        assert not any(isinstance(module, Engram) for module in enabled.module.mtp.modules())
        consumers = [
            index
            for index, layer in enumerate(enabled.module.decoder.layers)
            if hasattr(layer, 'engram')
        ]
        assert consumers == [1]
        memory = enabled.module.decoder.layers[1].engram.layers['0']
        # A zero memory residual gives a direct feature-off/on control through
        # actual heterogeneous kernels; memory still receives a trainable gradient.
        with torch.no_grad():
            memory.value_proj.weight.zero_()
            memory.value_proj.bias.zero_()
        calls = []
        handles = []
        for index, layer in enumerate(enabled.module.decoder.layers):

            def before(module, args, kwargs, index=index):
                calls.append((index, 'token_context' in kwargs))

            handles.append(layer.register_forward_pre_hook(before, with_kwargs=True))
        tokens = ((torch.arange(128, device='cuda') * 37 + 7) % 255 + 1).view(1, -1)
        positions = torch.arange(128, device='cuda').view(1, -1)
        labels = (tokens + 17) % 256
        mask = torch.ones_like(tokens, dtype=torch.float32)
        expected = reference(tokens, positions, None, labels=labels, loss_mask=mask)
        expected.mean().backward()
        actual = enabled(tokens, positions, None, labels=labels, loss_mask=mask)
        actual.mean().backward()
        for handle in handles:
            handle.remove()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        assert sorted(calls) == [(0, False), (1, True), (2, False), (3, False), (4, False)]
        for model in (reference, enabled):
            grads = {
                name: parameter_gradient(parameter)
                for name, parameter in model.module.named_parameters()
                if parameter_gradient(parameter) is not None
            }
            assert grads and all(torch.isfinite(gradient).all() for gradient in grads.values())
            for prefix in ('decoder.layers.0.', 'decoder.layers.1.', 'decoder.layers.4.', 'mtp.'):
                assert any(
                    name.startswith(prefix) and grad.norm() > 0 for name, grad in grads.items()
                ), prefix
        assert memory.value_proj.weight.grad.norm() > 0
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize('table_backend', [None, 'local', 'row_a2a'])
def test_declarative_modelopt_hybrid_preserves_engram_scope(monkeypatch, table_backend):
    """The alternative builder must reject Engram rather than silently construct a plain model."""
    module = pytest.importorskip('megatron.post_training.model_builder')
    transformer = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=1,
        engram_layer_ids=[0] if table_backend else None,
        engram_hash_table_min_sizes=[17, 19],
        engram_table_backend=table_backend or 'local',
    )
    config = module.ModelOptHybridModelConfig(
        transformer=transformer, vocab_size=64, hybrid_layer_pattern='*-'
    )
    builder = module.ModelOptHybridModelBuilder(config)
    args = Namespace()
    get_args = Mock(return_value=args)
    returned = object()
    delegate = Mock(return_value=returned)
    monkeypatch.setattr(module, 'get_args', get_args)
    monkeypatch.setattr(module, 'modelopt_gpt_hybrid_builder', delegate)
    groups = Mock()

    if table_backend:
        with pytest.raises(ValueError, match='Engram does not support ModelOpt'):
            builder.build_model(groups, pre_process=True, post_process=False)
        get_args.assert_not_called()
        delegate.assert_not_called()
    else:
        assert builder.build_model(groups, pre_process=True, post_process=False) is returned
        delegate.assert_called_once_with(args, True, False, None, pg_collection=groups)


def test_legacy_modelopt_builder_rejects_engram_before_constructing_config(monkeypatch):
    """The legacy model_provider route enforces the same explicit unsupported combination."""
    module = pytest.importorskip('megatron.post_training.model_builder')
    configure = Mock()
    monkeypatch.setattr(module, 'core_transformer_config_from_args', configure)
    with pytest.raises(ValueError, match='Engram does not support ModelOpt'):
        module.modelopt_gpt_hybrid_builder(Namespace(engram_layer_ids=[0]), True, True)
    configure.assert_not_called()


class TestHybridModelBuilderEngram:
    """Exercise the real provider helper through the declarative Hybrid builder."""

    @pytest.mark.parametrize(('pad_override', 'resolved_pad'), [(None, 7), (0, 0)])
    def test_enabled_provider_forwards_tokenizer_lookup_and_pad(self, pad_override, resolved_pad):
        from megatron.core.transformer.engram.hybrid_adapter import EngramHybridProvider

        transformer = _make_transformer(
            num_layers=4,
            num_moe_experts=4,
            moe_ffn_hidden_size=128,
            engram_layer_ids=[1],
            engram_target_layer_indices=[2],
            engram_hash_table_min_sizes=[17, 19],
        )
        config = _make_hybrid_config(
            transformer=transformer, hybrid_layer_pattern='*-*E', engram_pad_id=pad_override
        )
        tokenizer = Mock()
        lookup = torch.arange(16)
        pg = Mock()
        with (
            patch('megatron.training.models.hybrid.HybridModel') as model,
            patch('megatron.training.get_tokenizer', return_value=tokenizer) as get_tokenizer,
            patch(
                'megatron.core.transformer.engram.tokenizer.build_engram_tokenizer_lookup',
                return_value=lookup,
            ) as build_lookup,
            patch(
                'megatron.core.transformer.engram.tokenizer.get_engram_tokenizer_pad_id',
                return_value=resolved_pad,
            ) as get_pad,
        ):
            HybridModelBuilder(config).build_model(pg, pre_process=True, post_process=True)

        get_tokenizer.assert_called_once_with()
        build_lookup.assert_called_once_with(tokenizer)
        get_pad.assert_called_once_with(tokenizer, pad_override)
        kwargs = model.call_args.kwargs
        provider = kwargs['token_context_provider_spec']
        assert isinstance(provider, ModuleSpec)
        assert provider.module is EngramHybridProvider
        assert provider.params['tokenizer_lookup'] is lookup
        assert provider.params['pad_id'] == resolved_pad
        assert provider.params['hybrid_layer_pattern'] == '*-*E'
        assert kwargs['config'] is transformer
        assert kwargs['pg_collection'] is pg
        assert transformer.engram_layer_ids == [1]
        assert transformer.engram_target_layer_indices == [2]

    def setup_method(self):
        self.config = _make_hybrid_config()
        self.builder = HybridModelBuilder(self.config)
        self.pg = Mock()

    @pytest.mark.parametrize('fsdp_flag', ['use_megatron_fsdp', 'use_torch_fsdp2'])
    @pytest.mark.parametrize('table_backend', [None, 'local', 'row_a2a'])
    @patch('megatron.training.models.hybrid.unimodal_build_distributed_models')
    def test_fsdp_wrapper_flags_respect_engram_scope(self, mock_unimodal, table_backend, fsdp_flag):
        """Reject unsupported wrappers from the actual builder API before allocating a model."""
        model_list = [Mock()]
        mock_unimodal.return_value = model_list
        if table_backend is not None:
            self.config.transformer.engram_layer_ids = [0]
            self.config.transformer.engram_table_backend = table_backend

        if table_backend is not None:
            with pytest.raises(ValueError, match='ordinary Megatron DDP only'):
                self.builder.build_distributed_models(self.pg, **{fsdp_flag: True})
            mock_unimodal.assert_not_called()
        else:
            result = self.builder.build_distributed_models(self.pg, **{fsdp_flag: True})
            assert result is model_list
            flag_position = 5 if fsdp_flag == 'use_megatron_fsdp' else 6
            assert mock_unimodal.call_args.args[flag_position] is True


def _provider(**changes):
    values = dict(
        num_layers=4,
        engram_table_backend='local',
        engram_row_parallel_size=None,
        engram_layer_ids=[1],
        engram_target_layer_indices=[2],
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        cuda_graph_impl='none',
        linear_cp_layout='zigzag',
        attention_cp_layout='zigzag',
    )
    values.update(changes)
    return EngramHybridProvider(
        config=SimpleNamespace(**values),
        tokenizer_lookup=torch.arange(8),
        pad_id=0,
        hybrid_layer_pattern='*-*E',
        pg_collection=SimpleNamespace(cp=SimpleNamespace(size=lambda: 1)),
    )


def test_target_mapping_preserves_memory_identity_and_hashes():
    pattern = '*-' + '*E' * 11
    assert resolve_engram_targets([1], pattern) == {2: 1}
    assert resolve_engram_targets([1], pattern, [4]) == {4: 1}
    tokenizer = CompressedTokenizer(torch.arange(8))

    def mapping(ids):
        return NgramHashMapping(
            hash_table_min_sizes=[17, 19],
            max_ngram_size=3,
            num_hash_heads_per_ngram=2,
            layer_ids=ids,
            pad_id=0,
            seed=123,
            compressed_tokenizer=tokenizer,
        )

    original = mapping([1])
    relocated = mapping(list(resolve_engram_targets([1], pattern, [4]).values()))
    tokens = torch.tensor([[1, 3, 2, 4]])
    assert torch.equal(
        original.forward_compressed(tokens, 1), relocated.forward_compressed(tokens, 1)
    )
    assert original.hash_moduli_by_layer == relocated.hash_moduli_by_layer


@pytest.mark.parametrize(
    ('memory_ids', 'pattern', 'targets'),
    [
        ([1], '*-*E', [3]),
        ([1, 2], '*-*E', [2, 2]),
        ([1, 1], '*-*E', None),
        ([2], '*-*E', None),
        ([1], '*-*E/Z', [2]),
        ([0], 'MGE', None),
        ([1], '*-*E', [-1]),
    ],
)
def test_invalid_targets_fail_before_model_construction(memory_ids, pattern, targets):
    with pytest.raises(ValueError):
        resolve_engram_targets(memory_ids, pattern, targets)


@pytest.mark.parametrize(
    'setting',
    [
        'enable_mhc_connections',
        'moe_shortcut_connection',
        'fine_grained_activation_offloading',
        'overlap_moe_expert_parallel_comm',
        'use_megatron_fsdp',
        'fp8',
    ],
)
def test_provider_rejects_unsupported_combinations(setting):
    with pytest.raises(ValueError, match='Engram does not support'):
        _provider(**{setting: True})


def test_specs_replace_only_selected_attention_without_mutating_defaults():
    provider = _provider()
    original = ModuleSpec(
        module=TransformerLayer,
        params={'hidden_dropout': 0.0},
        submodules=TransformerLayerSubmodules(),
    )
    layer_config = AttentionLayerConfig.__new__(AttentionLayerConfig)
    overrides = provider.layer_spec_overrides(
        SimpleNamespace(attention_layer=original), [None, None, layer_config, None], 0
    )
    assert set(overrides) == {2}
    assert overrides[2].module is EngramAttentionLayer
    assert overrides[2].params['hidden_dropout'] == 0.0
    assert overrides[2].params['memory_id'] == 1
    assert overrides[2].params['engram_model_config'] is provider.config
    assert overrides[2].submodules is original.submodules
    assert original.module is TransformerLayer
    assert original.params == {'hidden_dropout': 0.0}


def test_pipeline_and_mtp_patterns_preserve_main_stack_targets():
    assert resolve_engram_targets([1], 'M*G-|*E/*E') == {4: 1}
    assert resolve_engram_targets([7], 'M*G-|*E/*E', [4]) == {4: 7}
    provider = _provider(pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=2)
    layer_config = AttentionLayerConfig.__new__(AttentionLayerConfig)
    spec = ModuleSpec(module=TransformerLayer, submodules=TransformerLayerSubmodules())
    submodules = SimpleNamespace(attention_layer=spec)
    assert not provider.layer_spec_overrides(submodules, [None, None], 0)
    assert set(provider.layer_spec_overrides(submodules, [layer_config, None], 2)) == {2}


def test_provider_uses_attention_view_instead_of_contiguous_boundary_tokens():
    from megatron.core.context_parallel import ContextParallelBatch

    provider = _provider(linear_cp_layout='contiguous')
    contiguous_tokens = torch.tensor([[0, 1, 2, 3]])
    attention_tokens = torch.tensor([[0, 1, 6, 7]])
    cp_batch = ContextParallelBatch(
        boundary_layout='contiguous',
        batches_by_layout={
            'contiguous': {'tokens': contiguous_tokens},
            'zigzag': {'tokens': attention_tokens},
        },
        packed_seq_params_by_layout={'contiguous': None, 'zigzag': None},
    )
    actual = provider.prepare(
        contiguous_tokens, inference_context=None, packed_seq_params=None, cp_batch=cp_batch
    )
    assert torch.equal(actual, attention_tokens)
