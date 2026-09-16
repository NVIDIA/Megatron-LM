# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tests for streamwise wide-residual branch connections."""

from dataclasses import fields
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.residual_connection import ResidualConnection
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig, WideResidualConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.wide_residual_layer import (
    LearnedWideResidualRetention,
    StreamwiseSigmoidMap,
    StreamwiseSigmoidResidualReadout,
    StreamwiseSigmoidWideResidualConnection,
    WideResidualTransformerLayer,
    expand_wide_residual_stream,
)
from tests.unit_tests.test_utilities import Utils


def _wide_config(
    *,
    num_streams: int = 3,
    init_scale: float = 0.0,
    learned_retention: bool = False,
    **config_overrides,
) -> TransformerConfig:
    values = dict(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        hidden_dropout=0.0,
        bias_dropout_fusion=False,
        use_cpu_initialization=True,
        wide_residual=WideResidualConfig(
            num_streams=num_streams,
            streamwise_sigmoid_init_scale=init_scale,
            learned_retention=learned_retention,
            retention_init=0.999,
            retention_max_forget=0.10,
        ),
    )
    values.update(config_overrides)
    return TransformerConfig(**values)


def _process_groups() -> ProcessGroupCollection:
    return ProcessGroupCollection()


class _ActiveBranch(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        del args, kwargs

    def forward(self, hidden_states, *args, **kwargs):
        del args, kwargs
        return hidden_states


class _AdditiveAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        del args, kwargs
        self.last_hidden_size = None

    def forward(self, hidden_states, *args, **kwargs):
        del args, kwargs
        self.last_hidden_size = hidden_states.shape[-1]
        bias = torch.ones(
            hidden_states.shape[-1], device=hidden_states.device, dtype=hidden_states.dtype
        )
        return 2.0 * hidden_states, bias


class _AdditiveMLP(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        del args, kwargs
        self.last_hidden_size = None

    def forward(self, hidden_states, *args, **kwargs):
        del args, kwargs
        self.last_hidden_size = hidden_states.shape[-1]
        return 3.0 * hidden_states, None


class _ParameterizedAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        del args, kwargs
        self.scale = nn.Parameter(torch.randn(()))

    def forward(self, hidden_states, *args, **kwargs):
        del args, kwargs
        return self.scale * hidden_states, None


class _ParameterizedMLP(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        del args, kwargs
        self.scale = nn.Parameter(torch.randn(()))

    def forward(self, hidden_states, *args, **kwargs):
        del args, kwargs
        return self.scale * hidden_states, None


class _ResidualReturningNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        del args, kwargs
        self.returns_residual = True

    def forward(self, hidden_states):
        return hidden_states, hidden_states


class _ProtocolResidualConnection(ResidualConnection):
    """Minimal configurable connection for exercising the abstract runtime contract."""

    def __init__(
        self,
        *,
        residual_stream_hidden_size=12,
        branch_hidden_size=4,
        read_behavior="valid",
        write_behavior="valid",
    ):
        super().__init__(residual_stream_hidden_size, branch_hidden_size)
        self.read_behavior = read_behavior
        self.write_behavior = write_behavior

    def _read(self, hidden_states):
        if self.read_behavior == "non_tensor":
            return object(), ()
        if self.read_behavior == "wrong_leading_shape":
            return torch.empty(3, self.branch_hidden_size), ()
        if self.read_behavior == "wrong_hidden_size":
            return torch.empty(*hidden_states.shape[:-1], self.branch_hidden_size + 1), ()
        if self.read_behavior == "non_tuple_state":
            return hidden_states[..., : self.branch_hidden_size], []
        if self.read_behavior == "non_tensor_state":
            return hidden_states[..., : self.branch_hidden_size], (object(),)
        return hidden_states[..., : self.branch_hidden_size], ()

    def _write(self, branch_output, state, *, dropout_probability, training):
        del branch_output, dropout_probability, training
        if self.write_behavior == "non_tensor":
            return object()
        if self.write_behavior == "wrong_shape":
            return state[0][..., :-1]
        return state[0]


class TestResidualConnectionContract:
    @pytest.mark.parametrize(
        ("residual_stream_hidden_size", "branch_hidden_size"), [(0, 4), (12, 0)]
    )
    def test_constructor_requires_positive_hidden_sizes(
        self, residual_stream_hidden_size, branch_hidden_size
    ):
        with pytest.raises(ValueError, match="hidden size must be positive"):
            _ProtocolResidualConnection(
                residual_stream_hidden_size=residual_stream_hidden_size,
                branch_hidden_size=branch_hidden_size,
            )

    def test_forward_rejects_operation_argument_mismatches(self):
        connection = _ProtocolResidualConnection()
        residual = torch.randn(2, 12)
        branch = torch.randn(2, 4)

        with pytest.raises(TypeError, match="read expects a tensor"):
            connection((residual, None), operation="read")
        with pytest.raises(TypeError, match="write-only arguments"):
            connection(residual, operation="read", state=())
        with pytest.raises(TypeError, match="read-only arguments"):
            connection(
                branch,
                operation="write",
                state=(residual,),
                fp32_residual_connection=True,
                dropout_probability=0.0,
                training=False,
            )
        with pytest.raises(TypeError, match="requires connection state"):
            connection(branch, operation="write", dropout_probability=0.0, training=False)
        with pytest.raises(TypeError, match="requires dropout_probability and training"):
            connection(branch, operation="write", state=(residual,))
        with pytest.raises(ValueError, match="Unsupported residual connection operation"):
            connection(residual, operation="invalid")

    @pytest.mark.parametrize(
        ("read_behavior", "expected_exception", "expected_error"),
        [
            ("non_tensor", TypeError, "expected a tensor"),
            ("wrong_leading_shape", ValueError, "changed non-hidden dimensions"),
            ("wrong_hidden_size", ValueError, "expected branch hidden size"),
            ("non_tuple_state", TypeError, "possibly empty tuple of tensors"),
            ("non_tensor_state", TypeError, "contain only tensors"),
        ],
    )
    def test_read_validates_implementation_outputs(
        self, read_behavior, expected_exception, expected_error
    ):
        connection = _ProtocolResidualConnection(read_behavior=read_behavior)

        with pytest.raises(expected_exception, match=expected_error):
            connection(torch.randn(2, 12), operation="read")

    def test_read_validates_input_width(self):
        with pytest.raises(ValueError, match="expected residual-stream hidden size"):
            _ProtocolResidualConnection()(torch.randn(2, 11), operation="read")

    def test_write_validates_state_branch_output_and_implementation_output(self):
        connection = _ProtocolResidualConnection()
        residual = torch.randn(2, 12)
        branch = torch.randn(2, 4)
        write_kwargs = dict(
            operation="write", state=(residual,), dropout_probability=0.0, training=False
        )

        with pytest.raises(TypeError, match="non-empty tuple of tensors"):
            connection(branch, **{**write_kwargs, "state": ()})
        with pytest.raises(TypeError, match="contain only tensors"):
            connection(branch, **{**write_kwargs, "state": (object(),)})
        with pytest.raises(TypeError, match="tensor or an .* tensor tuple"):
            connection((branch,), **write_kwargs)
        with pytest.raises(TypeError, match="tensor and an optional tensor"):
            connection((object(), None), **write_kwargs)
        with pytest.raises(ValueError, match="incompatible non-hidden dimensions"):
            connection(torch.randn(3, 4), **write_kwargs)
        with pytest.raises(ValueError, match="expected branch output hidden size"):
            connection(torch.randn(2, 5), **write_kwargs)

        for write_behavior, expected_error in (
            ("non_tensor", "expected a tensor"),
            ("wrong_shape", "changed the residual-stream shape"),
        ):
            invalid_connection = _ProtocolResidualConnection(write_behavior=write_behavior)
            with pytest.raises(
                TypeError if write_behavior == "non_tensor" else ValueError, match=expected_error
            ):
                invalid_connection(branch, **write_kwargs)


@pytest.mark.parametrize("num_streams", [True, 1, 0, -1])
def test_wide_residual_config_rejects_invalid_num_streams(num_streams):
    error = TypeError if isinstance(num_streams, bool) else ValueError
    with pytest.raises(error):
        WideResidualConfig(num_streams=num_streams)


@pytest.mark.parametrize(
    ("retention_init", "max_forget"), [(0.999, 0.0), (0.999, 1.0), (0.90, 0.10), (1.0, 0.10)]
)
def test_wide_residual_config_validates_bounded_retention(retention_init, max_forget):
    with pytest.raises(ValueError):
        WideResidualConfig(
            num_streams=3,
            learned_retention=True,
            retention_init=retention_init,
            retention_max_forget=max_forget,
        )


def test_wide_residual_config_rejects_negative_map_init_scale():
    with pytest.raises(ValueError, match="streamwise_sigmoid_init_scale"):
        WideResidualConfig(num_streams=3, streamwise_sigmoid_init_scale=-0.01)


@pytest.mark.parametrize(
    ("override", "expected_error"),
    [
        ({"enable_mhc_connections": True}, "mutually exclusive"),
        ({"mtp_num_layers": 1}, "Multi-Token Prediction"),
        ({"inference_fuse_tp_communication": True}, "fuse_tp_communication"),
        ({"fp32_residual_connection": True}, "fp32_residual_connection"),
        ({"heterogeneous_block_specs": True}, "heterogeneous_block_specs"),
        ({"overlap_moe_expert_parallel_comm": True}, "overlap_moe_expert_parallel_comm"),
        (
            {"num_layers": 2, "pipeline_model_parallel_size": 2, "pipeline_dtype": torch.bfloat16},
            "pipeline_model_parallel_size",
        ),
    ],
)
def test_transformer_config_rejects_unsupported_wide_residual_modes(override, expected_error):
    with pytest.raises((ValueError, NotImplementedError), match=expected_error):
        _wide_config(**override)


class TestStreamwiseSigmoidWideResidualConnection:
    def test_maps_materialize_expected_initial_factors(self):
        config = _wide_config(num_streams=3, init_scale=0.2)
        read_map = StreamwiseSigmoidMap(config, map_kind="read")
        write_map = StreamwiseSigmoidMap(config, map_kind="write")

        assert read_map(return_logits=True) is read_map.logit
        assert write_map(return_logits=True) is write_map.logit
        assert torch.allclose(read_map(), torch.full((3,), 1.0 / 3.0))
        assert torch.allclose(write_map().mean(), torch.tensor(1.0))

    def test_controller_at_minimum_shard_size_is_not_padded_again(self):
        read_map = StreamwiseSigmoidMap(_wide_config(num_streams=128), map_kind="read")

        assert read_map.logit.numel() == 128

    def test_component_constructors_validate_wide_configuration(self):
        base_config = TransformerConfig(
            num_layers=1, hidden_size=8, num_attention_heads=2, use_cpu_initialization=True
        )

        with pytest.raises(ValueError, match="StreamwiseSigmoidMap requires"):
            StreamwiseSigmoidMap(base_config, map_kind="read")
        with pytest.raises(ValueError, match="LearnedWideResidualRetention requires"):
            LearnedWideResidualRetention(
                base_config, layer_number=1, branch_name="test", num_streams=3
            )
        with pytest.raises(ValueError, match="StreamwiseSigmoidWideResidualConnection requires"):
            StreamwiseSigmoidWideResidualConnection(
                config=base_config,
                layer_number=1,
                branch_name="test",
                pg_collection=_process_groups(),
            )
        with pytest.raises(ValueError, match="StreamwiseSigmoidResidualReadout requires"):
            StreamwiseSigmoidResidualReadout(base_config)

    def test_components_validate_controller_and_activation_shapes(self):
        config = _wide_config()

        with pytest.raises(ValueError, match="Unsupported streamwise sigmoid map kind"):
            StreamwiseSigmoidMap(config, map_kind="unsupported")
        with pytest.raises(ValueError, match="one controller per full-width stream"):
            LearnedWideResidualRetention(config, layer_number=1, branch_name="test", num_streams=2)
        with pytest.raises(ValueError, match="expected hidden size"):
            StreamwiseSigmoidResidualReadout(config)(torch.randn(2, 8))
        with pytest.raises(ValueError, match="greater than one"):
            expand_wide_residual_stream(torch.randn(2, 8), 1)

    def test_initial_read_write_and_readout_preserve_base_stream(self):
        config = _wide_config(init_scale=0.0)
        connection = StreamwiseSigmoidWideResidualConnection(
            config=config, layer_number=1, branch_name="test", pg_collection=_process_groups()
        )
        readout = StreamwiseSigmoidResidualReadout(config)
        base = torch.randn(2, config.hidden_size)
        residual_stream = expand_wide_residual_stream(base, 3)

        branch_input, state = connection(residual_stream, operation="read")
        branch_update = torch.randn_like(base)
        output = connection(
            branch_update, operation="write", state=state, dropout_probability=0.0, training=False
        )

        assert torch.allclose(branch_input, base)
        assert torch.allclose(output, expand_wide_residual_stream(base + branch_update, 3))
        assert torch.allclose(readout(output), base + branch_update)

    @pytest.mark.parametrize("sequence_parallel", [False, True])
    def test_controllers_have_replicated_tp_gradient_metadata(self, sequence_parallel):
        config = _wide_config(learned_retention=True)
        config.sequence_parallel = sequence_parallel
        connection = StreamwiseSigmoidWideResidualConnection(
            config=config, layer_number=1, branch_name="test", pg_collection=_process_groups()
        )

        parameters = (
            connection.read_map.logit,
            connection.write_map.logit,
            connection.retention.retention_logit,
        )
        for parameter in parameters:
            assert parameter.allreduce
            assert not parameter.tensor_model_parallel
            assert parameter.sequence_parallel == sequence_parallel
            assert parameter.average_gradients_across_tp_domain is not sequence_parallel

    def test_nested_module_hooks_run_before_controller_access(self):
        config = _wide_config(learned_retention=True)
        connection = StreamwiseSigmoidWideResidualConnection(
            config=config, layer_number=1, branch_name="test", pg_collection=_process_groups()
        )
        operations = []
        connection.read_map.register_forward_pre_hook(lambda *_: operations.append("read"))
        connection.write_map.register_forward_pre_hook(lambda *_: operations.append("write"))
        connection.retention.register_forward_pre_hook(lambda *_: operations.append("retention"))

        residual_stream = torch.randn(2, 3 * config.hidden_size)
        branch_input, state = connection(residual_stream, operation="read")
        connection(
            branch_input, operation="write", state=state, dropout_probability=0.0, training=False
        )

        assert operations == ["read", "retention", "write"]

    def test_gradients_reach_active_controllers_but_not_padding(self):
        config = _wide_config(init_scale=0.01, learned_retention=True)
        connection = StreamwiseSigmoidWideResidualConnection(
            config=config, layer_number=1, branch_name="test", pg_collection=_process_groups()
        )
        residual_stream = torch.randn(4, 3 * config.hidden_size, requires_grad=True)
        branch_input, state = connection(residual_stream, operation="read")
        output = connection(
            branch_input.square(),
            operation="write",
            state=state,
            dropout_probability=0.0,
            training=True,
        )
        output.square().mean().backward()

        parameters = (
            connection.read_map.logit,
            connection.write_map.logit,
            connection.retention.retention_logit,
        )
        for parameter in parameters:
            assert parameter.grad is not None
            assert torch.count_nonzero(parameter.grad[: connection.num_streams]) > 0
            assert torch.count_nonzero(parameter.grad[connection.num_streams :]) == 0

    def test_checkpoint_round_trip_is_strict(self):
        config = _wide_config(learned_retention=True)
        source = StreamwiseSigmoidWideResidualConnection(
            config=config, layer_number=1, branch_name="test", pg_collection=_process_groups()
        )
        destination = StreamwiseSigmoidWideResidualConnection(
            config=config, layer_number=1, branch_name="test", pg_collection=_process_groups()
        )
        with torch.no_grad():
            source.read_map.logit[0].add_(0.25)
            source.write_map.logit[1].sub_(0.125)
            source.retention.retention_logit[2].add_(0.5)

        state = source.state_dict()
        destination.load_state_dict(state, strict=True)
        assert set(state) == {"read_map.logit", "write_map.logit", "retention.retention_logit"}
        for name, value in state.items():
            assert torch.equal(destination.state_dict()[name], value)

    def test_retention_initializes_to_requested_factor(self):
        config = _wide_config(learned_retention=True)
        retention = LearnedWideResidualRetention(
            config, layer_number=1, branch_name="test", num_streams=3
        )

        assert torch.allclose(retention(), torch.full((3,), 0.999), atol=1.0e-7)


class TestWideResidualStaticConstruction:
    def test_transformer_layer_submodules_has_no_wide_residual_fields(self):
        field_names = {field.name for field in fields(TransformerLayerSubmodules)}

        assert field_names.isdisjoint(
            {
                "residual_connection_self_attn",
                "residual_connection_cross_attn",
                "residual_connection_mlp",
            }
        )

    @pytest.mark.parametrize(
        ("self_attention", "mlp", "expected_connections"),
        [
            (_ActiveBranch, IdentityOp, {"self_attention"}),
            (IdentityOp, _ActiveBranch, {"mlp"}),
            (_ActiveBranch, _ActiveBranch, {"self_attention", "mlp"}),
        ],
        ids=("attention_only", "mlp_only", "attention_and_mlp"),
    )
    def test_static_subclass_constructs_connections_for_active_branches(
        self, self_attention, mlp, expected_connections
    ):
        layer_spec = ModuleSpec(
            module=WideResidualTransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=self_attention,
                self_attn_bda=get_bias_dropout_add,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
            ),
        )

        layer = build_module(
            layer_spec,
            config=_wide_config(),
            layer_number=1,
            add_layer_offset=False,
            pg_collection=_process_groups(),
        )

        assert layer_spec.module is WideResidualTransformerLayer
        assert type(layer) is WideResidualTransformerLayer
        connections = {
            "self_attention": layer.residual_connection_self_attn,
            "mlp": layer.residual_connection_mlp,
        }
        for branch_name, connection in connections.items():
            if branch_name in expected_connections:
                assert isinstance(connection, StreamwiseSigmoidWideResidualConnection)
            else:
                assert connection is None
        if len(expected_connections) == 2:
            assert layer.residual_connection_self_attn is not layer.residual_connection_mlp

    def test_cross_attention_is_rejected_explicitly(self):
        layer_spec = ModuleSpec(
            module=WideResidualTransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=_ActiveBranch, cross_attention=_ActiveBranch, mlp=_ActiveBranch
            ),
        )

        with pytest.raises(NotImplementedError, match="cross-attention"):
            build_module(
                layer_spec,
                config=_wide_config(),
                layer_number=1,
                add_layer_offset=False,
                pg_collection=_process_groups(),
            )

    def test_ordinary_spec_with_wide_config_fails_without_mutation(self):
        submodules = TransformerLayerSubmodules(
            self_attention=_ActiveBranch,
            self_attn_bda=get_bias_dropout_add,
            mlp=_ActiveBranch,
            mlp_bda=get_bias_dropout_add,
        )
        layer_spec = ModuleSpec(module=TransformerLayer, submodules=submodules)
        original_submodules = dict(vars(submodules))

        with pytest.raises(ValueError, match="WideResidualTransformerLayer"):
            TransformerBlock(
                _wide_config(), layer_spec, post_layer_norm=False, pg_collection=_process_groups()
            )

        assert layer_spec.module is TransformerLayer
        assert layer_spec.submodules is submodules
        assert vars(submodules) == original_submodules

    def test_static_subclass_requires_wide_residual_config(self):
        config = TransformerConfig(
            num_layers=1, hidden_size=8, num_attention_heads=2, use_cpu_initialization=True
        )
        layer_spec = ModuleSpec(
            module=WideResidualTransformerLayer,
            submodules=TransformerLayerSubmodules(self_attention=_ActiveBranch),
        )

        with pytest.raises(ValueError, match="requires wide_residual"):
            build_module(
                layer_spec,
                config=config,
                layer_number=1,
                add_layer_offset=False,
                pg_collection=_process_groups(),
            )

    @pytest.mark.parametrize(
        "config_overrides",
        [
            {"cuda_graph_impl": "local"},
            {"cuda_graph_impl": "transformer_engine"},
            {"cuda_graph_impl": "full_iteration"},
            {"enable_cuda_graph": True},
            {"external_cuda_graph": True},
        ],
        ids=("local", "transformer_engine", "full_iteration", "legacy_local", "legacy_te"),
    )
    def test_wide_residual_rejects_cuda_graphs(self, config_overrides):
        with pytest.raises(NotImplementedError, match="does not yet support CUDA graphs"):
            _wide_config(**config_overrides)


class TestWideResidualLayerIntegration:
    @staticmethod
    def _layer_spec(*, module=WideResidualTransformerLayer) -> ModuleSpec:
        return ModuleSpec(
            module=module,
            submodules=TransformerLayerSubmodules(
                self_attention=_AdditiveAttention,
                self_attn_bda=get_bias_dropout_add,
                mlp=_AdditiveMLP,
                mlp_bda=get_bias_dropout_add,
            ),
        )

    def test_disabled_path_preserves_ordinary_transformer_behavior(self):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=2,
            hidden_dropout=0.0,
            bias_dropout_fusion=False,
            use_cpu_initialization=True,
        )
        layer = build_module(
            self._layer_spec(module=TransformerLayer),
            config=config,
            layer_number=1,
            add_layer_offset=False,
            pg_collection=_process_groups(),
        )
        hidden_states = torch.randn(2, 3, config.hidden_size, requires_grad=True)

        output, context = layer(hidden_states=hidden_states, attention_mask=None)

        assert context is None
        assert type(layer) is TransformerLayer
        assert not hasattr(layer, "residual_connection_self_attn")
        assert not hasattr(layer, "residual_connection_mlp")
        assert torch.allclose(output, 12.0 * hidden_states + 4.0)

    def test_disabled_block_keeps_optional_readout_defined_when_layer_build_is_overridden(
        self, monkeypatch
    ):
        config = TransformerConfig(
            num_layers=1, hidden_size=8, num_attention_heads=2, use_cpu_initialization=True
        )

        def build_layers_without_wide_residual(block):
            block.layers = torch.nn.ModuleList()
            block.final_layernorm = None

        monkeypatch.setattr(TransformerBlock, "_build_layers", build_layers_without_wide_residual)
        block = TransformerBlock(
            config,
            self._layer_spec(module=TransformerLayer),
            post_layer_norm=False,
            pg_collection=_process_groups(),
        )

        assert block.residual_stream_readout is None

    @pytest.mark.parametrize(
        ("norm_field", "expected_error"),
        [
            ("input_layernorm", "self-attention residual connection"),
            ("pre_mlp_layernorm", "MLP residual connection"),
        ],
    )
    def test_residual_returning_norm_is_rejected_at_construction(self, norm_field, expected_error):
        config = _wide_config()
        layer_spec = self._layer_spec()
        setattr(layer_spec.submodules, norm_field, _ResidualReturningNorm)

        with pytest.raises(ValueError, match=expected_error):
            build_module(
                layer_spec,
                config=config,
                layer_number=1,
                add_layer_offset=False,
                pg_collection=_process_groups(),
            )

    def test_connection_construction_preserves_base_parameter_rng_order(self):
        base_config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=2,
            hidden_dropout=0.0,
            use_cpu_initialization=True,
        )
        wide_config = _wide_config()
        submodules = TransformerLayerSubmodules(
            self_attention=_ParameterizedAttention,
            self_attn_bda=get_bias_dropout_add,
            mlp=_ParameterizedMLP,
            mlp_bda=get_bias_dropout_add,
        )
        baseline_spec = ModuleSpec(module=TransformerLayer, submodules=submodules)
        wide_spec = ModuleSpec(
            module=WideResidualTransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=_ParameterizedAttention,
                self_attn_bda=get_bias_dropout_add,
                mlp=_ParameterizedMLP,
                mlp_bda=get_bias_dropout_add,
            ),
        )

        torch.manual_seed(1234)
        baseline = build_module(
            baseline_spec,
            config=base_config,
            layer_number=1,
            add_layer_offset=False,
            pg_collection=_process_groups(),
        )
        torch.manual_seed(1234)
        wide = build_module(
            wide_spec,
            config=wide_config,
            layer_number=1,
            add_layer_offset=False,
            pg_collection=_process_groups(),
        )

        assert torch.equal(wide.self_attention.scale, baseline.self_attention.scale)
        assert torch.equal(wide.mlp.scale, baseline.mlp.scale)

    def test_layer_carries_wide_stream_while_branches_remain_ordinary_width(self):
        config = _wide_config(init_scale=0.0)
        layer = build_module(
            self._layer_spec(),
            config=config,
            layer_number=1,
            add_layer_offset=False,
            pg_collection=_process_groups(),
        )
        base = torch.randn(2, 3, config.hidden_size, requires_grad=True)
        residual_stream = expand_wide_residual_stream(base, 3)

        output, context = layer(hidden_states=residual_stream, attention_mask=None)

        expected = expand_wide_residual_stream(12.0 * base + 4.0, 3)
        assert context is None
        assert output.shape[-1] == 3 * config.hidden_size
        assert torch.allclose(output, expected)
        assert layer.self_attention.last_hidden_size == config.hidden_size
        assert layer.mlp.last_hidden_size == config.hidden_size

        output.sum().backward()
        connection_parameters = {
            name: parameter
            for name, parameter in layer.named_parameters()
            if "residual_connection" in name
        }
        assert connection_parameters
        assert all(parameter.grad is not None for parameter in connection_parameters.values())

    def test_layer_checkpoint_surface_and_strict_round_trip(self):
        config = _wide_config(learned_retention=True)
        source = build_module(
            self._layer_spec(),
            config=config,
            layer_number=1,
            add_layer_offset=False,
            pg_collection=_process_groups(),
        )
        destination = build_module(
            self._layer_spec(),
            config=config,
            layer_number=1,
            add_layer_offset=False,
            pg_collection=_process_groups(),
        )
        with torch.no_grad():
            source.residual_connection_self_attn.read_map.logit[0].add_(0.25)
            source.residual_connection_mlp.write_map.logit[1].sub_(0.125)

        state = source.state_dict()
        expected_connection_keys = {
            f"residual_connection_{branch}.{parameter}"
            for branch in ("self_attn", "mlp")
            for parameter in ("read_map.logit", "write_map.logit", "retention.retention_logit")
        }
        connection_keys = {key for key in state if key.startswith("residual_connection_")}

        assert connection_keys == expected_connection_keys
        destination.load_state_dict(state, strict=True)
        for key in state:
            assert torch.equal(destination.state_dict()[key], state[key])

    def test_layer_forward_invokes_connection_module_hooks(self):
        config = _wide_config()
        layer = build_module(
            self._layer_spec(),
            config=config,
            layer_number=1,
            add_layer_offset=False,
            pg_collection=_process_groups(),
        )
        operations = []

        def record_operation(module, args, kwargs):
            del args
            operations.append((module.branch_name, kwargs["operation"]))

        layer.residual_connection_self_attn.register_forward_pre_hook(
            record_operation, with_kwargs=True
        )
        layer.residual_connection_mlp.register_forward_pre_hook(record_operation, with_kwargs=True)
        base = torch.randn(2, 3, config.hidden_size)

        layer(hidden_states=expand_wide_residual_stream(base, 3), attention_mask=None)

        assert operations == [
            ("self_attention", "read"),
            ("self_attention", "write"),
            ("mlp", "read"),
            ("mlp", "write"),
        ]

    def test_transformer_block_expands_and_reads_out_the_wide_stream(self):
        config = _wide_config(init_scale=0.01)
        block = TransformerBlock(
            config, self._layer_spec(), post_layer_norm=False, pg_collection=_process_groups()
        )
        hidden_states = torch.randn(2, 3, config.hidden_size, requires_grad=True)

        output = block(hidden_states=hidden_states, attention_mask=None)

        assert output.shape == hidden_states.shape
        assert torch.allclose(output, 12.0 * hidden_states + 4.0)
        assert block.residual_stream_readout is not None
        assert block.layers[0].residual_stream_hidden_size == 3 * config.hidden_size

        output.sum().backward()
        assert block.residual_stream_readout.exit_map.logit.grad is not None

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_full_recompute_preserves_wide_block_forward_and_backward(self):
        Utils.initialize_model_parallel(1, 1)
        try:
            model_parallel_cuda_manual_seed(123)
            config = _wide_config(
                init_scale=0.01,
                recompute_granularity="full",
                recompute_method="uniform",
                recompute_num_layers=1,
            )
            block = TransformerBlock(
                config,
                self._layer_spec(),
                post_layer_norm=False,
                pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
            ).cuda()
            hidden_states = torch.randn(2, 3, config.hidden_size, device="cuda", requires_grad=True)

            output = block(hidden_states=hidden_states, attention_mask=None)

            assert torch.allclose(output, 12.0 * hidden_states + 4.0)
            output.sum().backward()
            assert hidden_states.grad is not None
            assert block.residual_stream_readout.exit_map.logit.grad is not None
        finally:
            Utils.destroy_model_parallel()
