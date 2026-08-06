# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU-only tests for HybridStack distributed-checkpoint layer keys."""

from inspect import signature
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.models.hybrid import hybrid_block
from megatron.core.models.hybrid.hybrid_architecture import (
    HYBRID_LAYER_TYPE,
    HybridLayerSpec,
    PipelineSplit,
    resolve_hybrid_architecture,
)
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec


class _CheckpointLayer(nn.Module):
    """Minimal layer exposing a real parameter and a sharded checkpoint entry."""

    def __init__(self, layer_number: int):
        super().__init__()
        self.layer_number = layer_number
        self.weight = nn.Parameter(torch.tensor([float(layer_number)]))

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        del sharded_offsets, metadata
        key = f'{prefix}weight'
        return {key: ShardedTensor.from_rank_offsets(key, self.weight)}


MAMBA_SPEC = ModuleSpec(module=_CheckpointLayer, metainfo={HYBRID_LAYER_TYPE: "mamba"})
STACK_SPEC = ModuleSpec(
    module=HybridStack, submodules=HybridStackSubmodules(mamba_layer=MAMBA_SPEC)
)


def _config() -> TransformerConfig:
    return TransformerConfig(
        num_layers=8,
        hidden_size=16,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=4,
        ffn_hidden_size=32,
        mamba_state_dim=8,
        mamba_head_dim=4,
        mamba_num_heads=4,
        mamba_num_groups=2,
        pipeline_model_parallel_size=2,
        pipeline_dtype=torch.float32,
        use_cpu_initialization=True,
    )


def _stack(config, *, layer_specs=None, layer_type_list=None, offset=0) -> HybridStack:
    return HybridStack(
        config=config,
        submodules=STACK_SPEC.submodules,
        layer_specs=layer_specs,
        layer_type_list=layer_type_list,
        pp_layer_offset=offset,
        pre_process=False,
        post_process=False,
        post_layer_norm=False,
        pg_collection=SimpleNamespace(pp=object(), tp=object()),
    )


def _checkpoint_keys(stack: HybridStack) -> set[str]:
    return {entry.key for entry in stack.sharded_state_dict(prefix="decoder.").values()}


def test_pp2_vpp2_checkpoint_keys_use_global_offsets_and_match_legacy(monkeypatch):
    config = _config()
    layer = HybridLayerSpec(MAMBA_SPEC, config)
    architecture = resolve_hybrid_architecture(
        config=config,
        hybrid_stack_spec=STACK_SPEC,
        layer_specs=[
            [layer] * 2,
            PipelineSplit(),
            [layer],
            PipelineSplit(),
            [layer] * 3,
            PipelineSplit(),
            [layer] * 2,
        ],
    )

    monkeypatch.setattr(
        hybrid_block,
        "build_module",
        lambda _spec, **kwargs: _CheckpointLayer(kwargs["layer_number"]),
    )

    expected_chunks = {(0, 0): (2, 0), (1, 0): (1, 2), (0, 1): (3, 3), (1, 1): (2, 6)}
    all_direct_checkpoint_keys = set()

    for (pp_rank, vp_stage), (expected_length, expected_offset) in expected_chunks.items():
        layer_specs, offset = architecture.select_segment(
            pp_rank=pp_rank, pp_size=2, vp_stage=vp_stage
        )
        assert len(layer_specs) == expected_length
        assert offset == expected_offset

        direct_stack = _stack(config, layer_specs=layer_specs, offset=offset)
        legacy_stack = _stack(
            config, layer_type_list=[Symbols.MAMBA] * len(layer_specs), offset=offset
        )

        local_keys = {f'layers.{index}.weight' for index in range(expected_length)}
        local_prefixed_keys = {f'decoder.layers.{index}.weight' for index in range(expected_length)}
        global_checkpoint_keys = {
            f'decoder.layers.{offset + index}.weight' for index in range(expected_length)
        }

        assert set(direct_stack.state_dict()) == local_keys
        assert set(legacy_stack.state_dict()) == local_keys

        direct_sharded_state = direct_stack.sharded_state_dict(prefix="decoder.")
        legacy_sharded_state = legacy_stack.sharded_state_dict(prefix="decoder.")
        assert set(direct_sharded_state) == local_prefixed_keys
        assert set(legacy_sharded_state) == local_prefixed_keys
        assert _checkpoint_keys(direct_stack) == global_checkpoint_keys
        assert _checkpoint_keys(legacy_stack) == global_checkpoint_keys

        assert [layer.layer_number for layer in direct_stack.layers] == list(
            range(offset + 1, offset + expected_length + 1)
        )
        all_direct_checkpoint_keys.update(global_checkpoint_keys)

    assert all_direct_checkpoint_keys == {
        f'decoder.layers.{index}.weight' for index in range(config.num_layers)
    }


def test_raw_hybrid_model_derives_ownership_after_inferring_vpp():
    config = _config()
    config.num_layers = 4
    layer = HybridLayerSpec(MAMBA_SPEC, config)
    layer_specs = [layer, PipelineSplit(), layer, PipelineSplit(), layer, PipelineSplit(), layer]
    pg_collection = SimpleNamespace(tp=object(), cp=object(), pp=object(), embd=None)

    with (
        patch(
            "megatron.core.models.common.language_module.language_module."
            "LanguageModule._set_attention_backend"
        ),
        patch("megatron.core.models.hybrid.hybrid_model.get_pg_rank", return_value=1),
        patch(
            "megatron.core.models.hybrid.hybrid_model.build_module",
            return_value=torch.nn.Identity(),
        ),
    ):
        model = HybridModel(
            config=config,
            hybrid_stack_spec=STACK_SPEC,
            vocab_size=128,
            max_sequence_length=16,
            layer_specs=layer_specs,
            pg_collection=pg_collection,
            vp_stage=0,
        )

    assert config.virtual_pipeline_model_parallel_size == 2
    assert model.vp_size == 2
    assert model.pre_process is False
    assert model.post_process is False


def test_raw_legacy_hybrid_model_preserves_default_ownership_and_symbol_adapter():
    config = _config()
    config.num_layers = 1
    calls = []
    pg_collection = SimpleNamespace(tp=None, cp=None, pp=None, embd=None)

    def fake_build_module(_spec, *args, **kwargs):
        calls.append((args, kwargs))
        return torch.nn.Identity()

    with (
        patch(
            "megatron.core.models.common.language_module.language_module."
            "LanguageModule._set_attention_backend"
        ),
        patch(
            "megatron.core.models.hybrid.hybrid_model.LanguageModelEmbedding",
            return_value=torch.nn.Identity(),
        ),
        patch(
            "megatron.core.models.hybrid.hybrid_model.tensor_parallel.ColumnParallelLinear",
            return_value=torch.nn.Identity(),
        ),
        patch(
            "megatron.core.models.hybrid.hybrid_model.HybridModel."
            "setup_embeddings_and_output_layer"
        ),
        patch(
            "megatron.core.models.hybrid.hybrid_model.resolve_hybrid_architecture",
            side_effect=AssertionError("legacy construction must not invoke the direct resolver"),
        ),
        patch("megatron.core.models.hybrid.hybrid_model.build_module", fake_build_module),
    ):
        model = HybridModel(
            config=config,
            hybrid_stack_spec=STACK_SPEC,
            vocab_size=128,
            max_sequence_length=16,
            hybrid_layer_pattern="M",
            pg_collection=pg_collection,
        )

    assert model.pre_process is True
    assert model.post_process is True
    assert not hasattr(model, "resolved_hybrid_architecture")
    assert calls[0][1]["layer_type_list"] == [Symbols.MAMBA]
    assert "layer_specs" not in calls[0][1]


def test_raw_legacy_hybrid_model_preserves_explicit_none_ownership():
    config = _config()
    config.num_layers = 1
    pg_collection = SimpleNamespace(tp=None, cp=None, pp=None, embd=None)

    with (
        patch(
            "megatron.core.models.common.language_module.language_module."
            "LanguageModule._set_attention_backend"
        ),
        patch(
            "megatron.core.models.hybrid.hybrid_model.build_module",
            return_value=torch.nn.Identity(),
        ),
    ):
        model = HybridModel(
            config=config,
            hybrid_stack_spec=STACK_SPEC,
            vocab_size=128,
            max_sequence_length=16,
            hybrid_layer_pattern="M",
            pre_process=None,
            post_process=None,
            pg_collection=pg_collection,
        )

    assert model.pre_process is None
    assert model.post_process is None
    assert not hasattr(model, "embedding")
    assert not hasattr(model, "output_layer")


def test_direct_hybrid_model_parameters_are_appended_after_legacy_signature():
    parameters = signature(HybridModel.__init__).parameters

    parameter_names = list(parameters)
    assert parameter_names.index("layer_specs") > parameter_names.index("vp_stage")
    assert parameter_names.index("mtp_layer_specs") > parameter_names.index("vp_stage")
    assert parameter_names.index("resolved_hybrid_architecture") > parameter_names.index("vp_stage")
    assert parameters["pre_process"].default is True
    assert parameters["post_process"].default is True
    assert parameters["pre_process"].annotation is bool
    assert parameters["post_process"].annotation is bool


def test_legacy_hybrid_model_keeps_string_mtp_construction_path():
    config = _config()
    config.num_layers = 1
    config.mtp_num_layers = 2
    stack_spec = ModuleSpec(
        module=HybridStack,
        submodules=HybridStackSubmodules(
            mamba_layer=MAMBA_SPEC, mtp_block_spec=ModuleSpec(module=torch.nn.Identity)
        ),
    )
    pg_collection = SimpleNamespace(tp=None, cp=None, pp=None, embd=None)

    with (
        patch(
            "megatron.core.models.common.language_module.language_module."
            "LanguageModule._set_attention_backend"
        ),
        patch(
            "megatron.core.models.hybrid.hybrid_model.LanguageModelEmbedding",
            return_value=torch.nn.Identity(),
        ),
        patch(
            "megatron.core.models.hybrid.hybrid_model.tensor_parallel.ColumnParallelLinear",
            return_value=torch.nn.Identity(),
        ),
        patch(
            "megatron.core.models.hybrid.hybrid_model.HybridModel."
            "setup_embeddings_and_output_layer"
        ),
        patch("megatron.core.models.hybrid.hybrid_model.HybridModel._setup_mtp_cuda_graphs"),
        patch(
            "megatron.core.models.hybrid.hybrid_model.build_module",
            return_value=torch.nn.Identity(),
        ),
        patch("megatron.core.models.hybrid.hybrid_model.mtp_on_this_rank", return_value=True),
        patch(
            "megatron.core.models.hybrid.hybrid_model.MultiTokenPredictionBlock",
            return_value=torch.nn.Identity(),
        ) as mtp_block,
    ):
        model = HybridModel(
            config=config,
            hybrid_stack_spec=stack_spec,
            vocab_size=128,
            max_sequence_length=16,
            hybrid_layer_pattern="M/M/M",
            pg_collection=pg_collection,
        )

    assert model.mtp_pattern == "M"
    assert model.mtp_num_depths == 2
    assert mtp_block.call_args.kwargs["mtp_layer_pattern"] == "M"
    assert "mtp_layer_specs" not in mtp_block.call_args.kwargs
    assert "moe_metric_num_layers" not in mtp_block.call_args.kwargs
