# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay MLA and absorbed MLA projection/RoPE kernels across runtime CP sizes."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from tests.unit_tests.determinism.kernels.harness import (
    assert_module_replays_bit_exact,
    bytes_equal,
    seeded,
)
from tests.unit_tests.test_utilities import Utils

pytestmark = [
    pytest.mark.launch_on_gb200,
    pytest.mark.skipif(
        not (torch.cuda.is_available() and HAVE_TE), reason="needs a GPU and Transformer Engine"
    ),
]


@pytest.mark.parametrize("absorbed", [False, True])
@pytest.mark.parametrize("fused_rope", [False, True])
def test_mla_runtime_cp_projection_rope_replays(absorbed, fused_rope):
    """Replay the actual MLA projection/RoPE dispatch with runtime CP1/2/4 metadata."""
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_submodules,
    )
    from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
        AbsorbedMLASelfAttention,
        AbsorbedMLASelfAttentionSubmodules,
    )
    from megatron.core.transformer.multi_latent_attention import MLASelfAttention
    from megatron.core.transformer.transformer_config import MLATransformerConfig
    from tests.unit_tests.context_parallel.runtime.replay_test_utils import _zigzag_shard

    if Utils.world_size < 4 or Utils.world_size % 4:
        pytest.skip("requires four distributed ranks")
    Utils.initialize_model_parallel(dynamic_context_parallel=True)
    try:
        model_parallel_cuda_manual_seed(123)
        seeded()
        config = MLATransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            q_lora_rank=32,
            kv_lora_rank=32,
            qk_head_dim=64,
            v_head_dim=64,
            qk_pos_emb_head_dim=32,
            rope_type="yarn" if fused_rope else "rope",
            add_bias_linear=False,
            apply_rope_fusion=fused_rope,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_cpu_initialization=True,
            deterministic_mode=True,
        )
        submodules = get_gpt_layer_with_transformer_engine_submodules(
            multi_latent_attention=True
        ).self_attention.submodules
        cls = MLASelfAttention
        if absorbed:
            cls = AbsorbedMLASelfAttention
            submodules = AbsorbedMLASelfAttentionSubmodules(
                **{
                    name: getattr(submodules, name)
                    for name in AbsorbedMLASelfAttentionSubmodules.__dataclass_fields__
                }
            )
        attention = cls(
            config, submodules, layer_number=1, attn_mask_type=AttnMaskType.causal
        ).cuda()

        class Projections(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, hidden, packed):
                tensors = self.module.get_query_key_value_tensors(hidden, packed_seq_params=packed)
                return torch.cat(
                    [tensor.reshape(-1) for tensor in tensors if isinstance(tensor, torch.Tensor)]
                )

        module = Projections(attention)
        hidden = torch.randn(320, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        bounds = torch.tensor([0, 128, 320], dtype=torch.int32, device="cuda")
        build_group = attention.pg_collection.cp
        reference = None
        for size in (1, 2, 4, 1):
            group = parallel_state.get_dynamic_data_context_parallel_groups(group_size=size)
            packed = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=bounds,
                cu_seqlens_kv=bounds,
                max_seqlen_q=192,
                max_seqlen_kv=192,
                local_cp_size=size,
                cp_group=group,
            )
            local = _zigzag_shard(hidden, (128, 192), size, group.rank()).detach().requires_grad_()
            result = assert_module_replays_bit_exact(
                module,
                (local, packed),
                replays=3,
                contention=True,
                what=f"MLA[absorbed={absorbed}, fused={fused_rope}, CP{size}]",
            )
            assert attention.pg_collection.cp is build_group
            assert result[1]
            for tensors in result:
                assert all(torch.isfinite(tensor).all() for tensor in tensors.values())
            if size == 1:
                if reference is None:
                    reference = result
                for expected, actual in zip(reference, result):
                    assert expected.keys() == actual.keys()
                    assert all(bytes_equal(expected[name], actual[name]) for name in expected)
    finally:
        Utils.destroy_model_parallel()
