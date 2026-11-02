# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Muon's QKV layout metadata and the opt-in per-head and MLA up-projection splits."""

import os
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer import OptimizerConfig, emerging_optimizers, get_megatron_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from megatron.core.transformer import MLATransformerConfig
from megatron.core.transformer.qkv_layout import QKVLayout
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    not emerging_optimizers.HAVE_EMERGING_OPTIMIZERS, reason="requires emerging-optimizers"
)


def test_qkv_layouts():
    config = SimpleNamespace(
        num_attention_heads=8, num_query_groups=2, kv_channels=16, attention_output_gate=True
    )
    layout = QKVLayout.from_transformer_config(config)
    assert layout.num_groups == 2 and not layout.is_mla
    assert layout.projection_split_shapes == (64, 64, 16, 16)
    assert layout.per_head_split_shapes == (16,) * 10
    # The default projection split matches the shapes Muon always used.
    assert emerging_optimizers._get_qkv_split_shapes(config) == [64, 64, 16, 16]
    assert emerging_optimizers._get_qkv_split_shapes(config, split_qkv_per_head=True) == [16] * 10

    mla = QKVLayout.for_mla_up_projection(4, (24, 8))
    assert mla.is_mla and mla.num_groups == 4
    assert mla.projection_split_shapes == mla.per_head_split_shapes == (24, 8)
    with pytest.raises(ValueError, match="same rows"):
        QKVLayout(num_groups=1, projection_split_shapes=(4, 4), per_head_split_shapes=(4,))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(muon_split_qkv=False, muon_split_qkv_per_head=True), "muon_split_qkv_per_head"),
        (dict(muon_split_qkv=False, muon_split_mla_qkv=True), "muon_split_mla_qkv"),
    ],
)
def test_split_options_require_qkv_split(kwargs, message):
    with pytest.raises(ValueError, match=message):
        OptimizerConfig(optimizer='muon', **kwargs).validate()


@pytest.mark.parametrize(
    "head_split_shapes",
    [
        pytest.param((8, 8, 8, 8, 8, 8), id="uniform-heads"),
        pytest.param((12, 4), id="mla-unequal-components"),
        pytest.param((8, 8, 8, 8, 4, 4), id="unequal-qkv-heads"),
    ],
)
def test_per_head_orthogonalization_matches_each_head(head_split_shapes):
    torch.manual_seed(0)
    optimizer = emerging_optimizers.TensorParallelMuon(
        [torch.nn.Parameter(torch.zeros(1, 1))],
        split_qkv=True,
        split_qkv_per_head=True,
        fp32_matmul_prec="highest",
    )
    num_groups, cols = 3, 20
    grad = torch.randn(num_groups * sum(head_split_shapes), cols)
    param = SimpleNamespace(is_qkv=True, qkv_split_shapes=list(head_split_shapes))
    result = optimizer._orthogonalize_qkv_per_head(param, grad, list(head_split_shapes))
    # Reference: the optimizer's own 2D Newton-Schulz on every head, one at a time.
    expected = torch.cat(
        [
            optimizer.scaled_orthogonalize_fn(head, None, None, "duplicated")
            for group in grad.split(sum(head_split_shapes), dim=0)
            for head in group.split(list(head_split_shapes), dim=0)
        ]
    )
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)


_MLA_SUFFIXES = ("linear_q_up_proj.weight", "linear_kv_up_proj.weight")


def _mla_model_and_optimizer(**optimizer_kwargs):
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.training.training import wrap_model_chunks_with_ddp

    torch.manual_seed(0)
    model_parallel_cuda_manual_seed(0)
    config = MLATransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        tensor_model_parallel_size=2,
        multi_latent_attention=True,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_head_dim=32,
        qk_pos_emb_head_dim=16,
        v_head_dim=32,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
            multi_latent_attention=True
        ),
        vocab_size=128,
        max_sequence_length=16,
    ).cuda()
    ddp = wrap_model_chunks_with_ddp(
        [model],
        model.config,
        DistributedDataParallelConfig(layer_wise_param_layout="compact"),
        use_layer_wise_distributed_optimizer=True,
    )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            bf16=True,
            use_distributed_optimizer=True,
            muon_tp_mode='duplicated',
            **optimizer_kwargs,
        ),
        ddp,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        use_gloo_process_groups=False,
    )
    return ddp[0], optimizer


@pytest.mark.skipif(int(os.getenv('WORLD_SIZE', '1')) < 2, reason="needs TP=2")
class TestMuonMLATagging:

    @pytest.fixture(autouse=True)
    def model_parallel(self):
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("split_mla, per_head", [(False, False), (True, False), (True, True)])
    def test_mla_up_projections_split_only_when_requested(self, split_mla, per_head):
        model, optimizer = _mla_model_and_optimizer(
            muon_split_mla_qkv=split_mla, muon_split_qkv_per_head=per_head
        )
        mla = {n: p for n, p in model.named_parameters() if n.endswith(_MLA_SUFFIXES)}
        assert len(mla) == 4
        for name, param in mla.items():
            if not split_mla:
                # Default: whole-matrix Newton-Schulz, as before.
                assert not getattr(param, 'is_qkv', False)
                continue
            assert param.qkv_layout.is_mla and param.is_qkv
            assert param.qkv_split_shapes == ([32, 16] if "linear_q_up_proj" in name else [32, 32])

        tokens = torch.randint(0, 128, (2, 16), device='cuda')
        position_ids = torch.arange(16, device='cuda').unsqueeze(0).expand(2, -1)
        model.zero_grad_buffer()
        model(tokens, position_ids, None, labels=tokens).float().mean().backward()
        model.finish_grad_sync()
        success, grad_norm, _ = optimizer.step()
        assert success and grad_norm > 0
