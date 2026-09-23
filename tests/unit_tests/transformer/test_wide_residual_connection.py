# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tests for streamwise wide-residual branch connections."""

import pytest
import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.wide_residual_config import WideResidualConfig
from megatron.core.transformer.wide_residual_layer import (
    LearnedWideResidualRetention,
    StreamwiseSigmoidMap,
    StreamwiseSigmoidResidualReadout,
    StreamwiseSigmoidWideResidualConnection,
    expand_wide_residual_stream,
)


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
        assert connection.retention.retention_logit.is_wide_residual_retention_parameter

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
