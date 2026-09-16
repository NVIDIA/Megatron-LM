# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Public refit API regression with real MIMO and LLaVA models on disjoint GPUs."""

import pytest
import torch
import torch.distributed as dist

from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo import MimoModel, MimoModelConfig
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.resharding.refit import (
    clear_all_caches,
    prepare_swap_model_weights,
    swap_model_weights,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _grid(tp, offset):
    grid = HyperCommGrid([tp, 1, 1, 1], ['tp', 'pp', 'dp', 'cp'], rank_offset=offset)
    for axis in ('tp', 'pp', 'dp', 'cp'):
        grid.create_pg(axis)
    if not grid.is_current_rank_in_grid():
        return grid, None
    tp_group, singleton = grid.get_pg('tp'), grid.get_pg('dp')
    return grid, ProcessGroupCollection(
        tp=tp_group,
        pp=grid.get_pg('pp'),
        dp=singleton,
        cp=singleton,
        dp_cp=singleton,
        expt_tp=tp_group,
        ep=singleton,
        expt_dp=singleton,
        embd=singleton,
        pos_embd=singleton,
        mp=tp_group,
        tp_cp=tp_group,
    )


def _config(tp):
    return TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        tensor_model_parallel_size=tp,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        params_dtype=torch.bfloat16,
        bf16=True,
    )


def _models(language_tp, vision_tp, language_grid, image_grid, language_pg, image_pg, target_pg):
    rank = dist.get_rank()
    layer = get_gpt_layer_with_transformer_engine_spec()
    projection = MLPSubmodules(linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear)
    if target_pg is not None:
        vision_config = _config(1)
        vision_config.vision_model_type = 'clip'
        target = (
            LLaVAModel(
                language_transformer_config=_config(1),
                language_transformer_layer_spec=layer,
                language_vocab_size=256,
                language_max_sequence_length=64,
                language_position_embedding_type='rope',
                share_embeddings_and_output_weights=False,
                vision_transformer_config=vision_config,
                vision_transformer_layer_spec=layer,
                drop_vision_class_token=False,
                vision_projection_config=_config(1),
                vision_projection_type='mlp',
                vision_projection_layer_spec=projection,
                img_h=32,
                img_w=32,
                patch_dim=16,
                parallel_output=False,
                pg_collection=target_pg,
            )
            .cuda()
            .bfloat16()
        )
        target.vision_model.register_buffer(
            'refit_counter', torch.zeros(1, device='cuda', dtype=torch.bfloat16)
        )
        return None, target, {}
    if rank >= language_tp + vision_tp:
        return None, None, {}
    language = ModuleSpec(
        module=GPTModel,
        params=dict(
            config=_config(language_tp),
            transformer_layer_spec=layer,
            vocab_size=256,
            max_sequence_length=64,
            parallel_output=False,
            position_embedding_type='rope',
            share_embeddings_and_output_weights=False,
            pg_collection=language_pg,
        ),
    )
    vision = ModuleSpec(
        module=CLIPViTModel,
        params=dict(
            transformer_config=_config(vision_tp),
            transformer_layer_spec=layer,
            img_h=32,
            img_w=32,
            patch_dim=16,
            pg_collection=image_pg,
        ),
    )
    projector = ModuleSpec(
        module=MultimodalProjector,
        params=dict(
            config=_config(vision_tp),
            submodules=projection,
            projector_type='mlp',
            input_size=128,
            pg_collection=image_pg,
        ),
    )
    images = ModuleSpec(
        module=VisionModalitySubmodules,
        params={'pg_collection': image_pg},
        submodules={'encoders': {'clip': vision}, 'input_projections': [projector]},
    )
    source = (
        MimoModel(
            MimoModelConfig(
                language_model_spec=language,
                modality_submodules_spec={'images': images},
                special_token_ids={'images': 255},
                module_to_grid_map={'language': language_grid, 'images': image_grid},
            )
        )
        .cuda()
        .bfloat16()
    )
    if language_pg is not None:
        components = {'language_model': source.language_model}
    else:
        tower = source.modality_submodules['images']
        components = {
            'vision_model': tower.encoders['clip'],
            'vision_projection': tower.input_projections[0],
        }
        components['vision_model'].register_buffer('refit_counter', torch.zeros(1, device='cuda'))
    return source, None, components


def _tensor_state(module):
    # Use PyTorch's state enumeration independently of refit's tensor helpers.
    # Transformer Engine's serialized extra state is not a parameter or buffer.
    return {
        k: v
        for k, v in module.state_dict(keep_vars=True).items()
        if isinstance(v, torch.Tensor) and not k.endswith('_extra_state')
    }


def _collect_source_state(components):
    local = {}
    for prefix, module in components.items():
        for name, tensor in _tensor_state(module).items():
            local[f'{prefix}.{name}'] = (
                tensor.detach().cpu().clone(),
                getattr(tensor, 'tensor_model_parallel', False),
                getattr(tensor, 'partition_dim', 0),
                getattr(tensor, 'partition_stride', 1),
            )
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    return gathered


def _assert_refit(components, target, gathered):
    original = gathered[dist.get_rank()]
    current = {
        f"{prefix}.{name}": tensor
        for prefix, module in components.items()
        for name, tensor in _tensor_state(module).items()
    }
    assert current.keys() == original.keys()
    for name, tensor in current.items():
        torch.testing.assert_close(
            tensor.cpu(),
            original[name][0],
            rtol=0,
            atol=0,
            msg=lambda m: f"Source changed during refit: {name}: {m}",
        )
    if target is None:
        return
    target_state = _tensor_state(target)
    assert set(target_state) == set().union(*(part.keys() for part in gathered))
    for name, tensor in target_state.items():
        parts = [part[name] for part in gathered if name in part]
        expected, sharded, dim, stride = parts[0]
        if sharded:
            chunks = [part[0].chunk(stride, dim=dim) for part in parts]
            expected = torch.cat([chunk[i] for i in range(stride) for chunk in chunks], dim=dim)
        else:
            for replica in parts[1:]:
                torch.testing.assert_close(replica[0], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            tensor.cpu(), expected, rtol=0, atol=0, msg=lambda m: f'{name}: {m}'
        )


@pytest.mark.parametrize('wrapped', [False, True])
@pytest.mark.parametrize('language_tp,vision_tp', [(2, 1), (1, 2)])
@pytest.mark.parametrize('backend', ['nccl', 'gloo'])
def test_mimo_to_llava_repeated_refit(language_tp, vision_tp, backend, wrapped):
    Utils.initialize_model_parallel()
    if dist.get_world_size() < 4:
        Utils.destroy_model_parallel()
        pytest.skip('MIMO and LLaVA use four disjoint GPUs')
    language_grid, language_pg = _grid(language_tp, 0)
    image_grid, image_pg = _grid(vision_tp, language_tp)
    target_grid, target_pg = _grid(1, language_tp + vision_tp)
    try:
        pg = language_pg or image_pg or target_pg
        tp_rank = dist.get_rank(pg.tp) if pg is not None else 0
        torch.manual_seed(1234)
        model_parallel_cuda_manual_seed(1234, tp_rank=tp_rank, ep_rank=0, etp_rank=0)
        source, target, components = _models(
            language_tp, vision_tp, language_grid, image_grid, language_pg, image_pg, target_pg
        )
        if wrapped and source is not None:
            if source.language_model is not None:
                source.language_model = Float16Module(_config(language_tp), source.language_model)
            for name, tower in list(source.modality_submodules.items()):
                source.modality_submodules[name] = Float16Module(_config(vision_tp), tower)
            # Mimic training's FP32 persistent state after precision wrapping.
            if 'vision_model' in components:
                components['vision_model'].refit_counter = components[
                    'vision_model'
                ].refit_counter.float()
        prepare_swap_model_weights(source, target)
        for update in range(2):
            if update:
                with torch.no_grad():
                    for module in components.values():
                        for tensor in _tensor_state(module).values():
                            tensor.add_(1)
            expected = _collect_source_state(components)
            swap_model_weights(source, target, backend)
            _assert_refit(components, target, expected)
    finally:
        clear_all_caches()
        for grid in (language_grid, image_grid, target_grid):
            grid.destroy()
        Utils.destroy_model_parallel()
