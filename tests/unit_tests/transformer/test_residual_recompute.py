# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tests for selective wide-residual replay and offload ownership."""

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface,
    PipelineOffloadManager,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.tensor_parallel.random import initialize_rng_tracker
from megatron.core.transformer.residual_connection import (
    ResidualBranchOutput,
    ResidualConnection,
    ResidualConnectionState,
    ResidualConnectionWriteState,
)
from megatron.core.transformer.residual_recompute import (
    ResidualStreamRecomputeContext,
    build_residual_stream_recompute_plan,
    checkpoint_residual_read,
    checkpoint_residual_write,
    residual_stream_recompute_enabled,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig, WideResidualConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.wide_residual_layer import specialize_wide_residual_layer_spec
from megatron.core.typed_torch import apply_module
from tests.unit_tests.test_utilities import Utils


class _StaticTestConnection(ResidualConnection):
    def __init__(self, stream_width: int, branch_width: int):
        super().__init__(stream_width, branch_width)
        self.read_map = nn.Parameter(torch.randn(stream_width, branch_width) / stream_width**0.5)
        self.write_map = nn.Parameter(torch.randn(branch_width, stream_width) / branch_width**0.5)

    def _read(self, hidden_states: Tensor) -> tuple[Tensor, ResidualConnectionWriteState]:
        return hidden_states @ self.read_map, ()

    def _write(
        self,
        branch_output: ResidualBranchOutput,
        state: ResidualConnectionState,
        *,
        dropout_probability: float,
        training: bool,
    ) -> Tensor:
        update, bias = branch_output if isinstance(branch_output, tuple) else (branch_output, None)
        if bias is not None:
            update = update + bias
        update = F.dropout(update, p=dropout_probability, training=training)
        return state[0] + update @ self.write_map


class _HeavyTestBranch(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.projection = nn.Linear(width, width)
        self.output_bias = nn.Parameter(torch.randn(width))

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        return F.gelu(self.projection(hidden_states)), self.output_bias


class _StaticResidualChain(nn.Module):
    def __init__(self, stream_width: int = 12, branch_width: int = 6):
        super().__init__()
        self.connections = nn.ModuleList(
            [_StaticTestConnection(stream_width, branch_width) for _ in range(2)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(branch_width) for _ in range(2)])
        self.branches = nn.ModuleList([_HeavyTestBranch(branch_width) for _ in range(2)])

    def forward(
        self, hidden_states: Tensor, contexts: list[ResidualStreamRecomputeContext] | None = None
    ) -> tuple[Tensor, list[Tensor]]:
        managed_outputs = []
        for index, (connection, norm, branch) in enumerate(
            zip(self.connections, self.norms, self.branches)
        ):
            context = contexts[index] if contexts is not None else None
            if context is None:
                branch_input, state = apply_module(connection)(
                    hidden_states, operation="read", fp32_residual_connection=False
                )
                branch_input = apply_module(norm)(branch_input)
            else:
                branch_input, state = checkpoint_residual_read(
                    connection, hidden_states, context, fp32_residual_connection=False
                )
                branch_input = context.checkpoint(apply_module(norm), branch_input)

            branch_output = apply_module(branch)(branch_input)
            if context is not None and not context.is_block_end:
                hidden_states = checkpoint_residual_write(
                    connection,
                    branch_output,
                    state,
                    context,
                    dropout_probability=0.2,
                    training=True,
                )
            else:
                hidden_states = apply_module(connection)(
                    branch_output,
                    operation="write",
                    state=state,
                    dropout_probability=0.2,
                    training=True,
                )

            if context is not None and context.is_block_end:
                managed_outputs = [
                    output
                    for checkpoint in context.manager.checkpoints
                    for output in checkpoint.outputs
                ]
                context.finalize(hidden_states)

        return hidden_states, managed_outputs


class _ConfigNorm(nn.LayerNorm):
    def __init__(self, config, hidden_size, eps=None):
        del config
        super().__init__(hidden_size, eps=eps or 1.0e-5)


class _TransformerBranch(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        del args, kwargs
        self.projection = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, *args, **kwargs):
        del args, kwargs
        return F.gelu(self.projection(hidden_states)), None


class _MambaMixer(_TransformerBranch):
    pass


class _RecordingTransformerLayer(TransformerLayer):
    def forward(self, *args, **kwargs):
        self.received_residual_recompute_kwarg = "residual_stream_recompute_context" in kwargs
        return super().forward(*args, **kwargs)


class _RecordingMambaLayer(MambaLayer):
    def forward(
        self,
        hidden_states,
        *args,
        packed_seq_params=None,
        residual_stream_recompute_context=None,
        **kwargs,
    ):
        del args, kwargs
        self.event_log.append("mamba_forward")
        self.received_packed_seq_params = packed_seq_params
        self.received_residual_recompute_context = residual_stream_recompute_context
        return hidden_states


class _OffloadedQKVBranch(_TransformerBranch):
    """Small branch that exercises a nested fine-grained offload group."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.offload_qkv_linear = config.fine_grained_activation_offloading and "qkv_linear" in (
            config.offload_modules or []
        )
        self.input_offload_exclusions: list[bool] = []

    def forward(self, hidden_states, *args, **kwargs):
        del args, kwargs
        self.input_offload_exclusions.append(bool(getattr(hidden_states, "_do_not_offload", False)))
        prepared = self.projection(hidden_states)
        offload_manager = FineGrainedActivationOffloadingInterface(
            self.offload_qkv_linear, prepared, "qkv_linear"
        )
        with offload_manager as prepared:
            output = F.gelu(prepared)
        output = offload_manager.group_offload(output)
        return output, None


class _RecordingOffloadInterface:
    created: list[tuple[str, bool]] = []
    excluded: list[Tensor] = []

    def __init__(self, offload: bool, tensor: Tensor, name: str):
        self.offload = offload
        self.tensor = tensor
        self.name = name
        self.created.append((name, offload))

    def __enter__(self) -> Tensor:
        return self.tensor

    def __exit__(self, *_args) -> None:
        return None

    def group_offload(self, tensor: Tensor, **_kwargs) -> Tensor:
        return tensor

    @classmethod
    def mark_not_offload(cls, tensor: Tensor) -> None:
        tensor._do_not_offload = True
        cls.excluded.append(tensor)

    @classmethod
    def reset(cls) -> None:
        cls.created.clear()
        cls.excluded.clear()


def _process_groups() -> ProcessGroupCollection:
    return ProcessGroupCollection.use_mpu_process_groups()


def _wide_recompute_config(
    *, num_layers: int = 2, activation_offloading: bool = False
) -> TransformerConfig:
    return TransformerConfig(
        num_layers=num_layers,
        hidden_size=8,
        num_attention_heads=2,
        hidden_dropout=0.0,
        bias_dropout_fusion=False,
        use_cpu_initialization=True,
        recompute_granularity="selective",
        recompute_modules=["residual_stream"],
        residual_stream_recompute_num_layers=2,
        fine_grained_activation_offloading=activation_offloading,
        offload_modules=(["attn_norm", "mlp_norm", "qkv_linear"] if activation_offloading else []),
        min_offloaded_tensor_size=1,
        wide_residual=WideResidualConfig(
            num_streams=3,
            streamwise_sigmoid_init_scale=0.01,
            learned_retention=True,
            retention_init=0.999,
            retention_max_forget=0.10,
        ),
    )


def _layer_spec() -> ModuleSpec:
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=_ConfigNorm,
            self_attention=_TransformerBranch,
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=_ConfigNorm,
            mlp=_TransformerBranch,
            mlp_bda=get_bias_dropout_add,
        ),
    )


def _offloaded_qkv_layer_spec() -> ModuleSpec:
    spec = _layer_spec()
    spec.submodules.self_attention = _OffloadedQKVBranch
    return spec


def _recording_layer_spec() -> ModuleSpec:
    spec = _layer_spec()
    spec.module = _RecordingTransformerLayer
    return spec


def _mamba_spec() -> ModuleSpec:
    return ModuleSpec(
        module=MambaLayer,
        submodules=MambaLayerSubmodules(
            norm=_ConfigNorm, mixer=_MambaMixer, mamba_bda=get_bias_dropout_add
        ),
    )


def _recording_mamba_spec() -> ModuleSpec:
    spec = _mamba_spec()
    spec.module = _RecordingMambaLayer
    return spec


def _assert_matching_gradients(reference: nn.Module, recomputed: nn.Module) -> None:
    for (reference_name, reference_parameter), (recomputed_name, recomputed_parameter) in zip(
        reference.named_parameters(), recomputed.named_parameters()
    ):
        assert reference_name == recomputed_name
        torch.testing.assert_close(recomputed_parameter.grad, reference_parameter.grad)


class TestResidualStreamRecomputePlan:
    def test_partitions_local_layers_into_ordered_blocks(self):
        contexts = build_residual_stream_recompute_plan(num_layers=5, block_size=2)

        assert [context.is_block_end for context in contexts] == [False, True, False, True, True]
        assert contexts[0].manager is contexts[1].manager
        assert contexts[2].manager is contexts[3].manager
        assert contexts[1].manager is not contexts[2].manager

    @pytest.mark.parametrize("invalid_block_size", [False, 0, -1, 1.5])
    def test_rejects_invalid_block_size(self, invalid_block_size):
        with pytest.raises(ValueError, match="positive integer"):
            build_residual_stream_recompute_plan(2, invalid_block_size)

    def test_training_gate(self):
        config = _wide_recompute_config()
        assert residual_stream_recompute_enabled(config, training=True)
        assert not residual_stream_recompute_enabled(config, training=False)
        with torch.no_grad():
            assert not residual_stream_recompute_enabled(config, training=True)


class TestResidualStreamRecomputeConfig:
    def test_requires_wide_residual(self):
        with pytest.raises(ValueError, match="requires a configured wide residual"):
            TransformerConfig(
                num_layers=2,
                hidden_size=8,
                num_attention_heads=2,
                recompute_granularity="selective",
                recompute_modules=["residual_stream"],
            )

    def test_block_size_requires_residual_stream_recompute(self):
        with pytest.raises(ValueError, match="residual_stream_recompute_num_layers requires"):
            TransformerConfig(
                num_layers=2,
                hidden_size=8,
                num_attention_heads=2,
                residual_stream_recompute_num_layers=2,
                wide_residual=WideResidualConfig(num_streams=3),
            )

    @pytest.mark.parametrize("block_size", [False, 0, -1, 1.5])
    def test_rejects_invalid_block_size(self, block_size):
        with pytest.raises(ValueError, match="positive integer"):
            TransformerConfig(
                num_layers=2,
                hidden_size=8,
                num_attention_heads=2,
                recompute_granularity="selective",
                recompute_modules=["residual_stream"],
                residual_stream_recompute_num_layers=block_size,
                wide_residual=WideResidualConfig(num_streams=3),
            )

    def test_rejects_cuda_graph_capture(self):
        with pytest.raises(ValueError, match="requires cuda_graph_impl='none'"):
            TransformerConfig(
                num_layers=2,
                hidden_size=8,
                num_attention_heads=2,
                recompute_granularity="selective",
                recompute_modules=["residual_stream"],
                cuda_graph_impl="local",
                wide_residual=WideResidualConfig(num_streams=3),
            )

    def test_reports_connected_norm_offload_ownership(self):
        with pytest.warns(UserWarning, match="Residual-stream recomputation owns"):
            config = _wide_recompute_config(activation_offloading=True)

        assert config.offload_modules == ["attn_norm", "mlp_norm", "qkv_linear"]


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestResidualStreamRecomputeIntegration:
    def setup_method(self):
        Utils.initialize_model_parallel()
        initialize_rng_tracker(force_reset=True)

    def teardown_method(self):
        FineGrainedActivationOffloadingInterface.reset_instance()
        Utils.destroy_model_parallel()

    def test_ordered_replay_matches_reference_and_keeps_boundary_live(self):
        torch.manual_seed(1234)
        reference = _StaticResidualChain().cuda()
        recomputed = copy.deepcopy(reference).cuda()
        reference_input = torch.randn(4, 3, 12, device="cuda", requires_grad=True)
        recomputed_input = reference_input.detach().clone().requires_grad_(True)

        torch.manual_seed(5678)
        torch.cuda.manual_seed_all(5678)
        reference_output, _ = reference(reference_input)
        reference_output.square().mean().backward()

        operation_counts = {"connections": 0, "norms": 0, "branches": 0}
        for connection in recomputed.connections:
            connection.register_forward_pre_hook(
                lambda *_: operation_counts.__setitem__(
                    "connections", operation_counts["connections"] + 1
                )
            )
        for norm in recomputed.norms:
            norm.register_forward_pre_hook(
                lambda *_: operation_counts.__setitem__("norms", operation_counts["norms"] + 1)
            )
        for branch in recomputed.branches:
            branch.register_forward_pre_hook(
                lambda *_: operation_counts.__setitem__(
                    "branches", operation_counts["branches"] + 1
                )
            )

        torch.manual_seed(5678)
        torch.cuda.manual_seed_all(5678)
        recomputed_output, managed_outputs = recomputed(
            recomputed_input, build_residual_stream_recompute_plan(2, 2)
        )
        torch.testing.assert_close(recomputed_output, reference_output)
        assert recomputed_input.untyped_storage().size() > 0
        assert recomputed_output.untyped_storage().size() > 0
        assert managed_outputs
        assert all(output.untyped_storage().size() == 0 for output in managed_outputs)

        recomputed_output.square().mean().backward()

        assert all(output.untyped_storage().size() > 0 for output in managed_outputs)
        torch.testing.assert_close(recomputed_input.grad, reference_input.grad)
        _assert_matching_gradients(reference, recomputed)
        assert operation_counts == {"connections": 7, "norms": 4, "branches": 2}

    def test_transformer_block_matches_eager_forward_and_backward(self):
        recomputed_config = _wide_recompute_config()
        reference_config = copy.deepcopy(recomputed_config)
        reference_config.recompute_granularity = None
        reference_config.recompute_modules = ["core_attn"]
        reference_config.residual_stream_recompute_num_layers = None

        torch.manual_seed(1234)
        reference = TransformerBlock(
            reference_config, _layer_spec(), post_layer_norm=False, pg_collection=_process_groups()
        ).cuda()
        recomputed = TransformerBlock(
            recomputed_config, _layer_spec(), post_layer_norm=False, pg_collection=_process_groups()
        ).cuda()
        recomputed.load_state_dict(reference.state_dict())

        reference_input = torch.randn(4, 3, 8, device="cuda", requires_grad=True)
        recomputed_input = reference_input.detach().clone().requires_grad_(True)
        reference_output = reference(hidden_states=reference_input, attention_mask=None)
        reference_output.square().mean().backward()
        recomputed_output = recomputed(hidden_states=recomputed_input, attention_mask=None)
        recomputed_output.square().mean().backward()

        torch.testing.assert_close(recomputed_output, reference_output)
        torch.testing.assert_close(recomputed_input.grad, reference_input.grad)
        _assert_matching_gradients(reference, recomputed)

    def test_replay_owns_connected_norms_under_fine_grained_offload(self):
        with pytest.warns(UserWarning, match="Residual-stream recomputation owns"):
            config = _wide_recompute_config(num_layers=1, activation_offloading=True)
        layer = build_module(
            specialize_wide_residual_layer_spec(_layer_spec(), config),
            config=config,
            layer_number=1,
            add_layer_offset=False,
            pg_collection=_process_groups(),
        ).cuda()
        layer.off_interface = _RecordingOffloadInterface
        _RecordingOffloadInterface.reset()

        hidden_states = torch.randn(4, 3, 24, device="cuda", requires_grad=True)
        context = build_residual_stream_recompute_plan(1, 1)[0]
        output, _ = layer(
            hidden_states=hidden_states,
            attention_mask=None,
            residual_stream_recompute_context=context,
        )
        context.finalize(output)
        output.square().mean().backward()

        assert dict(_RecordingOffloadInterface.created) == {"attn_norm": False, "mlp_norm": False}
        assert len(_RecordingOffloadInterface.excluded) == 2
        assert all(
            getattr(output, "_do_not_offload", False)
            for output in _RecordingOffloadInterface.excluded
        )

    def test_replay_composes_with_nested_qkv_offload(self):
        with pytest.warns(UserWarning, match="Residual-stream recomputation owns"):
            offloaded_config = _wide_recompute_config(activation_offloading=True)
        reference_config = copy.deepcopy(offloaded_config)
        reference_config.fine_grained_activation_offloading = False
        reference_config.offload_modules = []

        torch.manual_seed(1234)
        reference = TransformerBlock(
            reference_config,
            _offloaded_qkv_layer_spec(),
            post_layer_norm=False,
            pg_collection=_process_groups(),
        ).cuda()
        offloaded = TransformerBlock(
            offloaded_config,
            _offloaded_qkv_layer_spec(),
            post_layer_norm=False,
            pg_collection=_process_groups(),
        ).cuda()
        offloaded.load_state_dict(reference.state_dict())

        reference_input = torch.randn(32, 3, 8, device="cuda", requires_grad=True)
        offloaded_input = reference_input.detach().clone().requires_grad_(True)
        reference_output = reference(hidden_states=reference_input, attention_mask=None)
        reference_output.float().square().mean().backward()

        FineGrainedActivationOffloadingInterface.reset_instance()
        FineGrainedActivationOffloadingInterface.init_chunk_handler(
            pp_rank=0,
            vp_size=None,
            vp_stage=None,
            min_offloaded_tensor_size=1,
            delta_offload_bytes_across_pp_ranks=0,
            activation_offload_fraction=1.0,
        )
        offloaded_output = offloaded(hidden_states=offloaded_input, attention_mask=None)
        offloaded_output.float().square().mean().backward()
        FineGrainedActivationOffloadingInterface.reset()

        offload_summary = PipelineOffloadManager.get_instance().offload_summary_bytes
        assert offload_summary.get("qkv_linear", 0) > 0
        assert "attn_norm" not in offload_summary
        assert "mlp_norm" not in offload_summary
        assert all(
            layer.self_attention.input_offload_exclusions == [True] for layer in offloaded.layers
        )
        torch.testing.assert_close(offloaded_output, reference_output)
        torch.testing.assert_close(offloaded_input.grad, reference_input.grad)
        _assert_matching_gradients(reference, offloaded)

    def test_mamba_layer_matches_eager_forward_and_backward(self):
        config = _wide_recompute_config(num_layers=1)
        layer_spec = specialize_wide_residual_layer_spec(_mamba_spec(), config)
        torch.manual_seed(1234)
        reference = build_module(
            layer_spec, config=config, layer_number=1, pg_collection=_process_groups()
        ).cuda()
        recomputed = build_module(
            layer_spec, config=config, layer_number=1, pg_collection=_process_groups()
        ).cuda()
        recomputed.load_state_dict(reference.state_dict())

        reference_input = torch.randn(4, 3, 24, device="cuda", requires_grad=True)
        recomputed_input = reference_input.detach().clone().requires_grad_(True)
        reference_output = reference(hidden_states=reference_input)
        reference_output.square().mean().backward()

        context = build_residual_stream_recompute_plan(1, 1)[0]
        recomputed_output = recomputed(
            hidden_states=recomputed_input, residual_stream_recompute_context=context
        )
        assert len(context.manager.checkpoints) == 2
        context.finalize(recomputed_output)
        recomputed_output.square().mean().backward()

        torch.testing.assert_close(recomputed_output, reference_output)
        torch.testing.assert_close(recomputed_input.grad, reference_input.grad)
        _assert_matching_gradients(reference, recomputed)

    def test_hybrid_stack_matches_eager_forward_and_backward(self):
        recomputed_config = _wide_recompute_config()
        reference_config = copy.deepcopy(recomputed_config)
        reference_config.recompute_granularity = None
        reference_config.recompute_modules = ["core_attn"]
        reference_config.residual_stream_recompute_num_layers = None
        submodules = HybridStackSubmodules(
            mamba_layer=_mamba_spec(), attention_layer=_recording_layer_spec()
        )
        layer_types = [Symbols.MAMBA, Symbols.ATTENTION]

        torch.manual_seed(1234)
        reference = HybridStack(
            reference_config,
            submodules,
            layer_type_list=layer_types,
            post_layer_norm=False,
            pg_collection=_process_groups(),
        ).cuda()
        recomputed = HybridStack(
            recomputed_config,
            submodules,
            layer_type_list=layer_types,
            post_layer_norm=False,
            pg_collection=_process_groups(),
        ).cuda()
        recomputed.load_state_dict(reference.state_dict())

        reference_input = torch.randn(4, 3, 8, device="cuda", requires_grad=True)
        recomputed_input = reference_input.detach().clone().requires_grad_(True)
        reference_output = reference(hidden_states=reference_input, attention_mask=None)
        reference_output.square().mean().backward()
        recomputed_output = recomputed(hidden_states=recomputed_input, attention_mask=None)
        recomputed_output.square().mean().backward()

        assert reference_output.shape == reference_input.shape
        assert not reference.layers[1].received_residual_recompute_kwarg
        assert recomputed.layers[1].received_residual_recompute_kwarg
        torch.testing.assert_close(recomputed_output, reference_output)
        torch.testing.assert_close(recomputed_input.grad, reference_input.grad)
        _assert_matching_gradients(reference, recomputed)

    def test_hybrid_mamba_cp_layout_finalizes_before_residual_replay(self, monkeypatch):
        config = _wide_recompute_config(num_layers=1)
        stack = HybridStack(
            config,
            HybridStackSubmodules(mamba_layer=_recording_mamba_spec()),
            layer_type_list=[Symbols.MAMBA],
            post_layer_norm=False,
            pg_collection=_process_groups(),
        ).cuda()

        events = []
        boundary_packed_seq_params = object()
        layer_packed_seq_params = object()

        class _RecordingCPLayoutState:
            finalized_hidden_states = None

            def prepare_layer(self, layer_index, hidden_states):
                assert layer_index == 0
                assert hidden_states.shape[-1] == 3 * config.hidden_size
                events.append("cp_prepare")
                return hidden_states, layer_packed_seq_params

            def finalize_layer(self, layer_index, hidden_states):
                assert layer_index == 0
                events.append("cp_finalize")
                self.finalized_hidden_states = hidden_states + 1.0
                return self.finalized_hidden_states

        cp_layout_state = _RecordingCPLayoutState()

        class _RecordingCPLayoutManager:
            def build_forward_state(self, packed_seq_params):
                assert packed_seq_params is boundary_packed_seq_params
                events.append("cp_build")
                return cp_layout_state

        stack._cp_layout_manager = _RecordingCPLayoutManager()
        stack.layers[0].event_log = events

        replay_inputs = []
        original_finalize = ResidualStreamRecomputeContext.finalize

        def record_replay_finalize(context, hidden_states):
            events.append("replay_finalize")
            replay_inputs.append(hidden_states)
            return original_finalize(context, hidden_states)

        monkeypatch.setattr(ResidualStreamRecomputeContext, "finalize", record_replay_finalize)

        hidden_states = torch.randn(4, 3, config.hidden_size, device="cuda", requires_grad=True)
        output = stack(
            hidden_states=hidden_states,
            attention_mask=None,
            packed_seq_params=boundary_packed_seq_params,
        )

        assert events == [
            "cp_build",
            "cp_prepare",
            "mamba_forward",
            "cp_finalize",
            "replay_finalize",
        ]
        assert stack.layers[0].received_packed_seq_params is layer_packed_seq_params
        assert stack.layers[0].received_residual_recompute_context is not None
        assert len(replay_inputs) == 1
        assert replay_inputs[0] is cp_layout_state.finalized_hidden_states
        assert output.shape == hidden_states.shape
