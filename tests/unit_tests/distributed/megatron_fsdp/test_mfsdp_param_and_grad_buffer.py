# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from torch.testing._internal.distributed.fake_pg import FakeStore

from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    BucketingPolicy,
    DataParallelBuffer,
    _get_parameter_groups,
)

FACTORY_META = {"device": "meta", "dtype": torch.bfloat16}


class AllInOneDummyFSDPModel(nn.Module):
    """
    One dummy module that recreates all parameter shapes and counts
    from the original paligemma_with_expert + heads, on meta device.
    Forward is not implemented; this is only for FSDP sharding tests.
    """

    def __init__(self):
        super().__init__()

        #
        # Language model parts
        #
        # embed_tokens.weight (257152, 2048)
        self.lm_embed_tokens = nn.Embedding(
            num_embeddings=257152, embedding_dim=2048, **FACTORY_META
        )

        # 18 language layers: self-attn + MLP + norms
        self.lm_layers = nn.ModuleList([LanguageLayerFlat() for _ in range(18)])

        # final norm.weight (2048,)
        self.lm_final_norm = nn.LayerNorm(
            normalized_shape=2048, device="meta", dtype=torch.bfloat16
        )

        #
        # Vision tower parts
        #
        # patch_embedding.weight (1152, 3, 14, 14)
        self.vision_patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=1152,
            kernel_size=14,
            stride=14,
            device="meta",
            dtype=torch.bfloat16,
        )
        # position_embedding.weight (256, 1152)
        self.vision_position_embedding = nn.Embedding(
            num_embeddings=256, embedding_dim=1152, **FACTORY_META
        )

        # 27 encoder layers: self-attn + MLP + norms
        self.vision_layers = nn.ModuleList([VisionLayerFlat() for _ in range(27)])

        # post_layer_norm.weight/bias (1152,)
        self.vision_post_layernorm = nn.LayerNorm(
            normalized_shape=1152, device="meta", dtype=torch.bfloat16
        )

        #
        # Multi-modal projector
        #
        # linear.weight (2048, 1152)
        self.mm_projector = nn.Linear(1152, 2048, **FACTORY_META)

        #
        # Gemma expert LM
        #
        # 18 expert layers
        self.gemma_layers = nn.ModuleList([GemmaLayerFlat() for _ in range(18)])
        # norm.dense (3072, 1024) + bias (3072,)
        self.gemma_norm_dense = nn.Linear(1024, 3072, **FACTORY_META)
        # lm_head.weight (257152, 1024)
        self.gemma_lm_head = nn.Linear(1024, 257152, bias=False, **FACTORY_META)

        #
        # Action head + time MLP
        #
        # action_in_proj.weight (1024, 32), bias (1024,)
        self.action_in_proj = nn.Linear(32, 1024, **FACTORY_META)
        # action_out_proj.weight (32, 1024), bias (32,)
        self.action_out_proj = nn.Linear(1024, 32, **FACTORY_META)

        # time_mlp_in.weight (1024, 1024), bias (1024,)
        self.time_mlp_in = nn.Linear(1024, 1024, **FACTORY_META)
        # time_mlp_out.weight (1024, 1024), bias (1024,)
        self.time_mlp_out = nn.Linear(1024, 1024, **FACTORY_META)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Dummy model for FSDP tests only.")


class LanguageLayerFlat(nn.Module):
    """
    One language layer matching:
      - self_attn.{q,k,v,o}_proj
      - mlp.{gate,up,down}_proj
      - input_layernorm, post_attention_layernorm (2048,)
    """

    def __init__(self):
        super().__init__()

        # Self-attention projections
        # q_proj.weight (2048, 2048)
        self.self_attn_q_proj = nn.Linear(2048, 2048, **FACTORY_META)
        # k_proj.weight (256, 2048)
        self.self_attn_k_proj = nn.Linear(2048, 256, **FACTORY_META)
        # v_proj.weight (256, 2048)
        self.self_attn_v_proj = nn.Linear(2048, 256, **FACTORY_META)
        # o_proj.weight (2048, 2048)
        self.self_attn_o_proj = nn.Linear(2048, 2048, **FACTORY_META)

        # MLP
        # gate_proj.weight (16384, 2048)
        self.mlp_gate_proj = nn.Linear(2048, 16384, **FACTORY_META)
        # up_proj.weight (16384, 2048)
        self.mlp_up_proj = nn.Linear(2048, 16384, **FACTORY_META)
        # down_proj.weight (2048, 16384)
        self.mlp_down_proj = nn.Linear(16384, 2048, **FACTORY_META)

        # input_layernorm.weight/bias (2048,)
        self.input_layernorm = nn.LayerNorm(
            normalized_shape=2048, device="meta", dtype=torch.bfloat16
        )
        # post_attention_layernorm.weight/bias (2048,)
        self.post_attention_layernorm = nn.LayerNorm(
            normalized_shape=2048, device="meta", dtype=torch.bfloat16
        )


class VisionLayerFlat(nn.Module):
    """
    One vision encoder layer matching:
      - self_attn.{k,v,q,out}_proj (1152, 1152)
      - mlp.fc1 (4304, 1152), mlp.fc2 (1152, 4304)
      - layer_norm1, layer_norm2 (1152,)
    """

    def __init__(self):
        super().__init__()

        # Self-attention projections (1152, 1152)
        self.self_attn_k_proj = nn.Linear(1152, 1152, **FACTORY_META)
        self.self_attn_v_proj = nn.Linear(1152, 1152, **FACTORY_META)
        self.self_attn_q_proj = nn.Linear(1152, 1152, **FACTORY_META)
        self.self_attn_out_proj = nn.Linear(1152, 1152, **FACTORY_META)

        # MLP
        # fc1.weight (4304, 1152)
        self.mlp_fc1 = nn.Linear(1152, 4304, **FACTORY_META)
        # fc2.weight (1152, 4304)
        self.mlp_fc2 = nn.Linear(4304, 1152, **FACTORY_META)

        # layer_norm1.weight/bias (1152,)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=1152, device="meta", dtype=torch.bfloat16)
        # layer_norm2.weight/bias (1152,)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=1152, device="meta", dtype=torch.bfloat16)


class GemmaLayerFlat(nn.Module):
    """
    One gemma_expert layer matching:
      - self_attn.{q,k,v,o}_proj
      - mlp.{gate,up,down}_proj
      - input_layernorm.dense, post_attention_layernorm.dense (3072, 1024)
    """

    def __init__(self):
        super().__init__()

        # Self-attention
        # q_proj.weight (2048, 1024)
        self.self_attn_q_proj = nn.Linear(1024, 2048, **FACTORY_META)
        # k_proj.weight (256, 1024)
        self.self_attn_k_proj = nn.Linear(1024, 256, **FACTORY_META)
        # v_proj.weight (256, 1024)
        self.self_attn_v_proj = nn.Linear(1024, 256, **FACTORY_META)
        # o_proj.weight (1024, 2048)
        self.self_attn_o_proj = nn.Linear(2048, 1024, **FACTORY_META)

        # MLP
        # gate_proj.weight (4096, 1024)
        self.mlp_gate_proj = nn.Linear(1024, 4096, **FACTORY_META)
        # up_proj.weight (4096, 1024)
        self.mlp_up_proj = nn.Linear(1024, 4096, **FACTORY_META)
        # down_proj.weight (1024, 4096)
        self.mlp_down_proj = nn.Linear(4096, 1024, **FACTORY_META)

        # input_layernorm.dense (3072, 1024)
        self.input_layernorm_dense = nn.Linear(1024, 3072, **FACTORY_META)
        # post_attention_layernorm.dense (3072, 1024)
        self.post_attention_layernorm_dense = nn.Linear(1024, 3072, **FACTORY_META)


@pytest.mark.parametrize("dp_world_size", [1, 3, 8, 17])
@pytest.mark.parametrize(
    ("data_parallel_sharding_strategy", "suggested_bucket_size", "fsdp_unit_modules"),
    [
        ("no_shard", 40_000_000, [LanguageLayerFlat, VisionLayerFlat]),
        ("optim", None, []),
        ("optim_grads", 40_000_000, [LanguageLayerFlat]),
        ("optim_grads_params", None, [LanguageLayerFlat, VisionLayerFlat, GemmaLayerFlat]),
    ],
)
def test_parameter_splitting(
    dp_world_size, suggested_bucket_size, fsdp_unit_modules, data_parallel_sharding_strategy
):
    model = AllInOneDummyFSDPModel()
    param_to_name = {param: name for name, param in model.named_parameters()}
    bucketing_policy = BucketingPolicy(
        suggested_bucket_size=suggested_bucket_size,
        fsdp_unit_modules=fsdp_unit_modules,
        data_parallel_sharding_strategy=data_parallel_sharding_strategy,
    )

    ddp_config = Mock()
    ddp_config.data_parallel_sharding_strategy = data_parallel_sharding_strategy

    bucket_groups, _, _ = _get_parameter_groups(model, bucketing_policy, {})
    for rank in range(dp_world_size):
        try:
            store = FakeStore()
            init_process_group_kwargs = dict(
                backend="fake", store=store, world_size=dp_world_size, rank=rank
            )
            torch.distributed.init_process_group(**init_process_group_kwargs)

            for param_group in bucket_groups:
                param_list = param_group.params
                dp_buf = DataParallelBuffer(
                    bucket_id=0,
                    ddp_config=ddp_config,
                    params=param_list,
                    chunk_size_factor=param_group.chunk_size_factor,
                    is_data_distributed=(data_parallel_sharding_strategy != "no_shard"),
                )

                for item_id, param in enumerate(param_list):
                    elem_per_slice = param.shape[1:].numel()

                    # Tensor shard
                    if data_parallel_sharding_strategy != "no_shard":
                        start, end = dp_buf._get_item_local_shard_index(item_id)
                        local_numel = end - start
                        assert local_numel % elem_per_slice == 0, (
                            f"[local_shard] param_name={param_to_name[param]}, "
                            f"world_size={dp_world_size}, "
                            f"rank={rank}, item_id={item_id}, "
                            f"param_shape={tuple(param.shape)}, "
                            f"local_numel={local_numel}, "
                            f"elem_per_slice={elem_per_slice}"
                        )

                    # Full tensor
                    start, end = dp_buf._get_item_local_index(item_id)
                    local_numel = end - start
                    assert local_numel % elem_per_slice == 0, (
                        f"[full_tensor] param_name={param_to_name[param]}, "
                        f"world_size={dp_world_size}, "
                        f"rank={rank}, item_id={item_id}, "
                        f"param_shape={tuple(param.shape)}, "
                        f"local_numel={local_numel}, "
                        f"elem_per_slice={elem_per_slice}"
                    )
        finally:
            # Clean up process group if created
            try:
                torch.distributed.destroy_process_group()
            except Exception:
                pass
