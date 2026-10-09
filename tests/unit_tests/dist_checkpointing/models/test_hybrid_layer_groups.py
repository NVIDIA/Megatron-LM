# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Distributed checkpoints of grouped and ungrouped HybridModel patterns load into each other."""

import pytest
import torch

from megatron.core import parallel_state as ps
from megatron.core.dist_checkpointing import load, save
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def _hybrid_model(seed, pattern, pp, moe):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    config_kwargs = dict(
        num_layers=len(pattern.translate(str.maketrans('', '', '[]|'))),
        num_attention_heads=8,
        hidden_size=256,
        use_cpu_initialization=True,
        pipeline_dtype=torch.bfloat16,
        pipeline_model_parallel_size=pp,
    )
    if moe:
        config_kwargs.update(
            num_moe_experts=8, moe_grouped_gemm=True, add_bias_linear=False, moe_router_topk=2
        )
    return HybridModel(
        config=TransformerConfig(**config_kwargs),
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=128,
        max_sequence_length=4,
        hybrid_layer_pattern=pattern,
        pre_process=ps.is_pipeline_first_stage(),
        post_process=ps.is_pipeline_last_stage(),
    )


def _physical_layers(model):
    """Return the decoder's layers in pattern order, looking inside bracketed groups."""
    layers = []
    for layer in model.decoder.layers:
        layers.extend(layer.layers if isinstance(layer, HybridStack) else [layer])
    return layers


def _tensors(module):
    return {
        name: tensor
        for name, tensor in module.state_dict().items()
        if isinstance(tensor, torch.Tensor) and not name.endswith('_extra_state')
    }


class TestGroupedHybridCheckpoints:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("pp", [1, 2])
    @pytest.mark.parametrize(
        ("saved_pattern", "loaded_pattern", "moe"),
        [
            ("[*-][*-]", "*-*-", False),
            ("*-*-", "[*-][*-]", False),
            ("M[*E]M[*E]", "M*EM*E", True),
            ("M*EM*E", "[M*E][M*E]", True),
        ],
    )
    def test_grouped_and_ungrouped_checkpoints_load_into_each_other(
        self, tmp_path_dist_ckpt, saved_pattern, loaded_pattern, moe, pp
    ):
        Utils.initialize_model_parallel(1, pp)
        with TempNamedDir(tmp_path_dist_ckpt / 'hybrid_layer_groups') as ckpt_dir:
            saved_model = _hybrid_model(1, saved_pattern, pp, moe)
            with torch.no_grad():
                for param in saved_model.parameters():
                    param.random_()
            save(saved_model.sharded_state_dict(), ckpt_dir)

            loaded_model = _hybrid_model(2, loaded_pattern, pp, moe)
            loaded_model.load_state_dict(load(loaded_model.sharded_state_dict(), ckpt_dir))

        saved_layers = _physical_layers(saved_model)
        loaded_layers = _physical_layers(loaded_model)
        assert [layer.layer_number for layer in loaded_layers] == [
            layer.layer_number for layer in saved_layers
        ]
        for saved_layer, loaded_layer in zip(saved_layers, loaded_layers, strict=True):
            saved_tensors = _tensors(saved_layer)
            loaded_tensors = _tensors(loaded_layer)
            assert loaded_tensors.keys() == saved_tensors.keys()
            for name, tensor in saved_tensors.items():
                assert torch.equal(
                    loaded_tensors[name], tensor
                ), f'layer {saved_layer.layer_number} {name}'
        if loaded_model.post_process:
            for name, tensor in _tensors(saved_model.decoder.final_norm).items():
                assert torch.equal(_tensors(loaded_model.decoder.final_norm)[name], tensor)
