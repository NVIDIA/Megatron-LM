# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Config-only extension acceptance, using a layer that is not registered in Core."""

from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch

from megatron.core import dist_checkpointing
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_layer_allocation import (
    MTPSplit,
    PipelineSplit,
    clone_hybrid_layer_config_list,
    parse_hybrid_layer_config_list,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.hybrid.layers import utils as layer_utils
from megatron.core.models.hybrid.layers.hybrid_hyper_connection import HyperConnectionHybridLayer
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.models.hybrid.external_layer import ExternalGainConfig, ExternalGainLayer
from tests.unit_tests.test_utilities import Utils


def _root_config(**overrides):
    fields = dict(
        num_layers=3,
        hidden_size=32,
        num_attention_heads=4,
        use_cpu_initialization=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        params_dtype=torch.float32,
    )
    fields.update(overrides)
    return TransformerConfig(**fields)


def _single_rank_groups():
    """Only group size is consulted by the CPU, unsharded construction path."""
    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    return SimpleNamespace(tp=group, pp=group, cp=group, tp_cp=group, dp_cp=group)


def test_external_config_initializes_new_defaults_and_clones_mutable_fields():
    root = _root_config()
    root.normalized_extra = {"values": [7]}
    first = ExternalGainConfig.from_config(root)
    second = ExternalGainConfig.from_config(root)
    assert first.gain_init == 1.25
    assert first.notes == ["external"]
    assert first.notes is not second.notes
    assert first.normalized_extra == root.normalized_extra
    assert first.normalized_extra is not root.normalized_extra

    first.gain_init = 0.75
    clones = clone_hybrid_layer_config_list([first, first])
    assert all(type(config) is ExternalGainConfig for config in clones)
    assert clones[0] is not clones[1] and clones[0] is not first
    assert all(config.gain_init == 0.75 for config in clones)
    clones[0].notes.append("changed")
    clones[0].normalized_extra["values"].append(8)
    assert first.notes == clones[1].notes == ["external"]
    assert root.normalized_extra == first.normalized_extra == {"values": [7]}


@pytest.mark.parametrize("bad_type", [object(), torch.nn.Linear, "ExternalGainLayer"])
def test_external_config_rejects_invalid_layer_binding(bad_type):
    class InvalidGainConfig(ExternalGainConfig):
        layer_type: ClassVar[object] = bad_type

    with pytest.raises((TypeError, ValueError), match="layer_type"):
        parse_hybrid_layer_config_list([InvalidGainConfig.from_config(_root_config())])


def test_external_config_parser_preserves_pipeline_and_mtp_identity():
    root = _root_config()
    custom = ExternalGainConfig.from_config(root)
    attention = AttentionLayerConfig.from_config(root)
    entries = [custom, PipelineSplit, attention, custom, MTPSplit, custom, MTPSplit, custom]
    parsed = parse_hybrid_layer_config_list(
        entries, expected_num_layers=3, expected_mtp_num_layers=2
    )
    assert parsed.main_layer_config_list == (custom, PipelineSplit, attention, custom)
    assert parsed.main_layer_config_list[0] is parsed.main_layer_config_list[-1]
    assert parsed.mtp_layer_config_list == (custom,)
    assert parsed.mtp_layer_config_list[0] is custom
    assert parsed.mtp_num_depths == 2


@pytest.mark.parametrize("shortcut", [False, True])
def test_external_layer_cpu_stack_forward_backward_and_context(shortcut):
    root = _root_config(
        num_layers=2, moe_shortcut_connection=shortcut, num_moe_experts=2 if shortcut else None
    )
    source = ExternalGainConfig.from_config(root)
    physical_configs = clone_hybrid_layer_config_list([source, source])
    groups = _single_rank_groups()
    stack = HybridStack(
        config=root,
        submodules=HybridStackSubmodules(),
        layer_config_list=physical_configs,
        pg_collection=groups,
        pre_process=True,
        post_process=False,
        post_layer_norm=False,
        pp_layer_offset=4,
    )
    assert all(type(layer) is source.layer_type for layer in stack.layers)
    assert [layer.layer_number for layer in stack.layers] == [5, 6]
    assert stack.layers[0].config is physical_configs[0]
    assert stack.layers[1].config is physical_configs[1]
    assert stack.layers[0].gain is not stack.layers[1].gain

    hidden = torch.randn(4, 2, root.hidden_size, requires_grad=True)
    attention_mask = torch.zeros(2, 1, 4, 4, dtype=torch.bool)
    rotary = torch.randn(4, 1, 1, root.kv_channels)
    padding_mask = torch.zeros(2, 4, dtype=torch.bool)
    input_ids = torch.arange(4).expand(2, -1)
    output = stack(
        hidden,
        attention_mask=attention_mask,
        rotary_pos_emb=rotary,
        padding_mask=padding_mask,
        input_ids=input_ids,
    )
    torch.testing.assert_close(output, hidden * source.gain_init**2)
    for layer in stack.layers:
        observed = layer.last_forward_inputs
        assert observed[0] is attention_mask
        assert observed[1] is rotary
        assert observed[3] is padding_mask
        assert observed[4] is input_ids
    output.square().mean().backward()
    assert hidden.grad is not None
    before = stack.layers[0].gain.detach().clone()
    optimizer = torch.optim.SGD(stack.parameters(), lr=0.1)
    optimizer.step()
    assert not torch.equal(stack.layers[0].gain, before)
    assert source.notes == ["external"]
    assert source.gain_init == 1.25


def test_external_forward_receives_packing():
    config = ExternalGainConfig.from_config(_root_config())
    layer = config.layer_type(config=config, pg_collection=_single_rank_groups(), layer_number=1)
    hidden = torch.randn(4, 1, config.hidden_size)
    packed = PackedSeqParams(qkv_format="thd")
    layer_utils.invoke_custom_layer(layer, hidden, packed_seq_params=packed)
    assert layer.last_forward_inputs[2] is packed


@pytest.mark.parametrize("recompute", [False, True])
def test_external_stack_rejects_inference_before_execution(recompute):
    root = _root_config(num_layers=1)
    if recompute:
        root.recompute_granularity = "full"
        root.recompute_method = "uniform"
        root.recompute_num_layers = 1
    custom = ExternalGainConfig.from_config(root)
    stack = HybridStack(
        config=root,
        submodules=HybridStackSubmodules(),
        layer_config_list=[custom],
        pg_collection=_single_rank_groups(),
        post_process=False,
        post_layer_norm=False,
    )
    with pytest.raises(ValueError, match="do not support inference contexts"):
        stack(torch.randn(4, 1, root.hidden_size), None, inference_context=object())
    assert stack.layers[0].last_forward_inputs is None


@pytest.mark.parametrize("tp_size,cp_size", [(2, 1), (1, 2), (2, 2)])
def test_external_layer_receives_parallel_process_groups(tp_size, cp_size):
    root = _root_config(
        num_layers=1, tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
    )
    config = ExternalGainConfig.from_config(root)
    # Constructor-only coverage: the custom layer receives the caller's parallel grid.
    groups = _single_rank_groups()
    groups.tp = SimpleNamespace(size=lambda: tp_size, rank=lambda: 0)
    groups.cp = SimpleNamespace(size=lambda: cp_size, rank=lambda: 0)
    groups.tp_cp = SimpleNamespace(size=lambda: tp_size * cp_size, rank=lambda: 0)
    stack = HybridStack(
        config=root,
        submodules=HybridStackSubmodules(),
        layer_config_list=[config],
        pg_collection=groups,
        post_process=False,
        post_layer_norm=False,
    )
    assert stack.layers[0].pg_collection is groups
    assert stack.layers[0].config is config
    assert stack.layers[0].tp_group is groups.tp


@pytest.fixture
def gpu_model_parallel():
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA for a real HybridModel and distributed checkpoint")
    Utils.initialize_model_parallel(1, 1)
    # CPU-initialized weights must match across the checkpoint's data-parallel replicas.
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234)
    yield ProcessGroupCollection.use_mpu_process_groups()
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("recompute_mode", [None, "uniform", "block", "mhc"])
def test_external_layer_mhc_matches_eager(gpu_model_parallel, recompute_mode):
    def make_stack(mode):
        options = {}
        if mode == "mhc":
            options = dict(
                recompute_granularity="selective",
                recompute_modules=["mhc"],
                mhc_recompute_layer_num=2,
            )
        elif mode is not None:
            options = dict(
                recompute_granularity="full", recompute_method=mode, recompute_num_layers=1
            )
        root = _root_config(
            num_layers=2,
            enable_mhc_connections=True,
            mhc_num_residual_streams=2,
            mhc_sinkhorn_iterations=3,
            **options,
        )
        custom = ExternalGainConfig.from_config(root)
        return HybridStack(
            config=root,
            submodules=HybridStackSubmodules(),
            layer_config_list=clone_hybrid_layer_config_list([custom, custom]),
            pg_collection=gpu_model_parallel,
            post_layer_norm=False,
        ).cuda()

    eager = make_stack(None)
    actual = make_stack(recompute_mode)
    actual.load_state_dict(eager.state_dict())
    hidden = torch.randn(4, 2, eager.config.hidden_size, device="cuda")
    inputs = dict(
        attention_mask=torch.zeros(2, 1, 4, 4, device="cuda", dtype=torch.bool),
        rotary_pos_emb=torch.randn(4, 1, 1, eager.config.kv_channels, device="cuda"),
        packed_seq_params=PackedSeqParams(qkv_format="sbhd"),
        padding_mask=torch.zeros(2, 4, device="cuda", dtype=torch.bool),
        input_ids=torch.arange(8, device="cuda").reshape(2, 4),
    )
    seen_shapes = []
    for layer in actual.layers:
        assert isinstance(layer, HyperConnectionHybridLayer)
        layer.inner_layer.register_forward_pre_hook(
            lambda _layer, args: seen_shapes.append(args[0].shape)
        )

    eager_input = hidden.clone().requires_grad_()
    actual_input = hidden.clone().requires_grad_()
    expected = eager(eager_input, **inputs)
    output = actual(actual_input, **inputs)
    torch.testing.assert_close(output, expected)
    assert output.shape == hidden.shape
    expected.square().mean().backward()
    output.square().mean().backward()
    torch.testing.assert_close(actual_input.grad, eager_input.grad)
    assert seen_shapes and all(shape == hidden.shape for shape in seen_shapes)
    for layer in actual.layers:
        for observed, supplied in zip(
            layer.inner_layer.last_forward_inputs, inputs.values(), strict=True
        ):
            if isinstance(supplied, torch.Tensor):
                torch.testing.assert_close(observed, supplied)
            else:
                assert observed is supplied
        for parameter in (layer.inner_layer.gain, layer.hyper_connection.mapping_proj.weight):
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
    for name, parameter in actual.named_parameters():
        torch.testing.assert_close(parameter.grad, eager.get_parameter(name).grad, msg=name)


def test_external_transformer_subclass_mhc_calls_custom_forward(gpu_model_parallel):
    class CustomTransformerLayer(TransformerLayer):
        def __init__(self, config):
            MegatronModule.__init__(self, config)
            self.layer_number = 1
            self.self_attention = torch.nn.Identity()
            self.cross_attention = IdentityOp()
            self.mlp = IdentityOp()
            self.forward_calls = 0

        def forward(self, hidden_states, **_kwargs):
            self.forward_calls += 1
            return hidden_states * 1.25

        def _forward_self_attention_output_with_bias(self, **_kwargs):
            raise AssertionError("Custom TransformerLayer.forward must not be bypassed")

    class CustomTransformerConfig(ExternalGainConfig):
        layer_type = CustomTransformerLayer

    config = CustomTransformerConfig.from_config(
        _root_config(num_layers=1, enable_mhc_connections=True, mhc_num_residual_streams=2)
    )
    inner = CustomTransformerLayer(config)
    wrapper = HyperConnectionHybridLayer(config, inner).cuda()
    hidden = torch.randn(4, 2, config.hidden_size * 2, device="cuda", requires_grad=True)
    output, context = wrapper(hidden, attention_mask=None)
    assert inner.forward_calls == 1
    assert context is None
    assert output.shape == hidden.shape
    output.square().mean().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


def _gpu_model(groups, recompute_method=None, with_mtp=False):
    root = _root_config(attention_backend=AttnBackend.unfused)
    if with_mtp:
        root.mtp_num_layers = 2
    if recompute_method is not None:
        root.recompute_granularity = "full"
        root.recompute_method = recompute_method
        root.recompute_num_layers = 2
    custom = ExternalGainConfig.from_config(root)
    layers = [AttentionLayerConfig.from_config(root), custom, custom]
    if with_mtp:
        layers.extend([MTPSplit, custom, MTPSplit, custom])
    model = HybridModel(
        config=root,
        vocab_size=64,
        max_sequence_length=8,
        hybrid_layer_config_list=layers,
        position_embedding_type="rope",
        pg_collection=groups,
    ).cuda()
    return model, custom


def _gpu_loss(model):
    tokens = torch.arange(16, device="cuda", dtype=torch.long).reshape(2, 8)
    positions = torch.arange(8, device="cuda", dtype=torch.long).expand(2, -1)
    labels = (tokens + 1) % 64
    attention_mask = torch.triu(torch.ones(1, 1, 8, 8, device="cuda", dtype=torch.bool), 1)
    return model(tokens, positions, attention_mask, labels=labels).float().mean()


@pytest.mark.parametrize("recompute_method", ["uniform", "block"])
def test_external_layer_real_model_recompute_matches_eager(gpu_model_parallel, recompute_method):
    eager, source = _gpu_model(gpu_model_parallel)
    recomputed, _ = _gpu_model(gpu_model_parallel, recompute_method)
    recomputed.load_state_dict(eager.state_dict())
    eager_loss = _gpu_loss(eager)
    eager_loss.backward()
    recomputed_loss = _gpu_loss(recomputed)
    recomputed_loss.backward()
    torch.testing.assert_close(recomputed_loss, eager_loss)
    eager_params = dict(eager.named_parameters())
    recomputed_params = dict(recomputed.named_parameters())
    assert eager_params.keys() == recomputed_params.keys()
    for name, parameter in eager_params.items():
        assert parameter.grad is not None, name
        torch.testing.assert_close(recomputed_params[name].grad, parameter.grad)
    for layer in eager.decoder.layers[1:]:
        assert isinstance(layer, ExternalGainLayer)
        assert layer.config is not source
        assert layer.gain.grad.count_nonzero() > 0


def test_external_layer_real_model_mtp_clones_and_backward(gpu_model_parallel):
    model, source = _gpu_model(gpu_model_parallel, with_mtp=True)
    mtp_layers = [layer for layer in model.mtp.modules() if isinstance(layer, ExternalGainLayer)]
    assert len(mtp_layers) == 2
    all_custom_layers = [*model.decoder.layers[1:], *mtp_layers]
    assert len({id(layer.config) for layer in all_custom_layers}) == 4
    assert len({id(layer.gain) for layer in all_custom_layers}) == 4
    assert all(layer.config is not source for layer in all_custom_layers)
    assert all(layer.is_mtp_layer for layer in mtp_layers)
    _gpu_loss(model).backward()
    for layer in all_custom_layers:
        assert layer.gain.grad is not None
        assert layer.gain.grad.count_nonzero() > 0
        assert torch.isfinite(layer.gain.grad).all()
    mtp_layers[0].config.notes.append("physical MTP layer")
    assert source.notes == ["external"]
    assert mtp_layers[1].config.notes == ["external"]


def test_external_layer_real_model_checkpoint_resume(gpu_model_parallel, tmp_path_dist_ckpt):
    model, source = _gpu_model(gpu_model_parallel)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    def step(current, current_optimizer):
        current_optimizer.zero_grad()
        loss = _gpu_loss(current)
        loss.backward()
        current_optimizer.step()
        return loss.detach()

    step(model, optimizer)
    metadata = {"dp_cp_group": gpu_model_parallel.dp_cp}
    with TempNamedDir(tmp_path_dist_ckpt / "external_hybrid_layer") as checkpoint_dir:
        dist_checkpointing.save(
            {
                "model": model.sharded_state_dict(metadata=metadata),
                "optimizer": optimizer.state_dict(),
            },
            checkpoint_dir,
        )
        resumed, _ = _gpu_model(gpu_model_parallel)
        resumed_optimizer = torch.optim.SGD(resumed.parameters(), lr=0.01, momentum=0.9)
        restored = dist_checkpointing.load(
            {"model": resumed.sharded_state_dict(metadata=metadata)}, checkpoint_dir
        )
        resumed.load_state_dict(restored["model"])
        resumed_optimizer.load_state_dict(restored["optimizer"])
        torch.testing.assert_close(step(resumed, resumed_optimizer), step(model, optimizer))
        for original, recovered in zip(model.parameters(), resumed.parameters(), strict=True):
            torch.testing.assert_close(recovered, original)
    assert source.notes == ["external"]
