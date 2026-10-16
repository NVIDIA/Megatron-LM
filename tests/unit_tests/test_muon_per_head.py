# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Per-head Muon layout, dispatch, and independent-head update regressions."""

import os

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.emerging_optimizers import HAVE_EMERGING_OPTIMIZERS, TensorParallelMuon
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = [
    pytest.mark.skipif(
        not HAVE_EMERGING_OPTIMIZERS, reason="emerging_optimizers package is not installed"
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="per-head Muon tests require CUDA"),
]


@pytest.fixture(autouse=True)
def select_local_cuda_device():
    """Allocate standalone optimizer tensors on this torchrun worker's GPU."""
    if not torch.cuda.is_available():
        pytest.skip("per-head Muon tests require CUDA")
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))


@pytest.mark.skipif(int(os.getenv('WORLD_SIZE', '1')) < 2, reason="Requires at least two ranks")
class TestMuonPerHeadMultiRankTP:
    """Check module-owned head layouts and reconstruction across TP ranks."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test with tensor parallel."""
        world = int(os.getenv('WORLD_SIZE', '1'))
        Utils.initialize_model_parallel(tensor_model_parallel_size=min(world, 2))
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("variant", ["gated_delta_net", "gdn2"])
    def test_gdn_variant_owned_semantic_layout(self, variant):
        """Instantiate the real variant; its physical TP-local sections drive routing."""
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            get_experimental_attention_variant_module_spec,
        )
        from megatron.core.ssm.gated_delta_net import HAVE_FLA, HAVE_FLA_GDN2

        if not HAVE_FLA or (variant == "gdn2" and not HAVE_FLA_GDN2):
            pytest.skip("The GDN variant's FLA kernel is unavailable")
        pg = ProcessGroupCollection.use_mpu_process_groups()
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            hidden_size=128,
            num_layers=1,
            num_attention_heads=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
            linear_conv_kernel_dim=4,
            normalization="RMSNorm",
            tensor_model_parallel_size=2,
            experimental_attention_variant=variant,
            linear_attention_freq=[1],
            transformer_impl="transformer_engine",
        )
        spec = get_experimental_attention_variant_module_spec(config=config)
        model = spec.module(
            config,
            submodules=spec.submodules,
            layer_number=1,
            bias=False,
            conv_bias=False,
            pg_collection=pg,
        ).cuda()
        p = model.in_proj.weight
        assert sum(p.muon_layout.splits) == p.shape[0]
        controls = sum(model.in_proj_split_sections[3:])
        assert sum(n for n, a in zip(p.muon_layout.splits, p.muon_layout.adamw) if a) == controls
        opt = TensorParallelMuon([p], split_qkv=True, split_qkv_per_head=True, pg_collection=pg)
        p.grad = torch.randn_like(p)
        opt.step()
        start = sum(model.in_proj_split_sections[:3])
        assert torch.count_nonzero(opt.state[p]['gate_exp_avg'][:start]) == 0
        assert torch.count_nonzero(opt.state[p]['gate_exp_avg'][start:]) > 0

    def test_attention_gate_and_swiglu_owned_semantic_layouts(self):
        """Fused module parameters retain routing through the public optimizer factory."""
        pg = ProcessGroupCollection.use_mpu_process_groups()
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=4,
            tensor_model_parallel_size=2,
            ffn_hidden_size=32,
            attention_output_gate=True,
            gated_linear_unit=True,
            activation_func=F.silu,
            add_bias_linear=False,
        )
        model = GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_layer_local_spec(),
            vocab_size=32,
            max_sequence_length=8,
            pre_process=False,
            post_process=False,
            pg_collection=pg,
        )
        optimizer = get_megatron_optimizer(
            config=OptimizerConfig(
                optimizer='muon',
                lr=0.01,
                use_distributed_optimizer=False,
                muon_split_qkv=True,
                muon_split_qkv_per_head=True,
            ),
            model_chunks=[model],
            use_gloo_process_groups=False,
            pg_collection=pg,
        )
        layer = model.decoder.layers[0]
        assert any(layer.self_attention.linear_qkv.weight.muon_layout.adamw)
        assert layer.mlp.linear_fc1.weight.muon_layout.splits == (16, 16)
        assert layer.mlp.linear_fc1.weight.muon_layout.tp_partitioned
        assert optimizer is not None

    @pytest.mark.parametrize("split_per_head", [False, True])
    def test_kitchen_layout_requires_explicit_support(self, split_per_head):
        """Exercise the Kitchen config branch without claiming backend validation."""
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
        from megatron.core.transformer.mlp import MLP, MLPSubmodules

        pg = ProcessGroupCollection.use_mpu_process_groups()
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=4,
            tensor_model_parallel_size=2,
            ffn_hidden_size=32,
            gated_linear_unit=True,
            activation_func=F.silu,
            use_kitchen=True,
            add_bias_linear=False,
        )
        # The public Kitchen backend is a stub. Local linears let us test the
        # stride-1 metadata/validation branch without pretending to run Kitchen.
        model = MLP(
            config,
            MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear),
            tp_group=pg.tp,
            ffn_hidden_size=32,
        )
        assert model.linear_fc1.weight.partition_stride == 1
        assert getattr(model.linear_fc1.weight, "muon_layout", None) is None
        kwargs = dict(
            config=OptimizerConfig(
                optimizer="muon", lr=0.01, muon_split_qkv_per_head=split_per_head
            ),
            model_chunks=[model],
            use_gloo_process_groups=False,
            pg_collection=pg,
        )
        if split_per_head:
            with pytest.raises(ValueError, match="does not support Kitchen projection layouts"):
                get_megatron_optimizer(**kwargs)
        else:
            assert get_megatron_optimizer(**kwargs) is not None


def test_muon_optimizer_uniform_splits_accept_noncontiguous_batched_output():
    """Uniform logical splits can return transposed, noncontiguous NS outputs."""
    grad = torch.arange(48, dtype=torch.float32, device='cuda').view(12, 4)
    param = torch.nn.Parameter(torch.zeros_like(grad))
    param.is_qkv = True
    param.qkv_split_shapes = [6, 6]
    optimizer = TensorParallelMuon(
        params=[param],
        split_qkv=True,
        split_qkv_per_head=True,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=[6, 6],
        pg_collection=None,
    )

    def noncontiguous_identity(x, tp_group=None, partition_dim=None, tp_mode_this_group=None):
        del tp_group, partition_dim, tp_mode_this_group
        return x.mT.contiguous().mT

    optimizer.scaled_orthogonalize_fn = noncontiguous_identity
    actual = optimizer.orthogonalize(param, grad)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, grad)


@pytest.mark.parametrize('batched_available', [True, False])
@pytest.mark.parametrize('splits', [[2, 1, 2, 1, 2], [2, 2, 2, 2], [1, 3, 1, 3]])
def test_muon_optimizer_mixed_heads_preserve_order_and_scale(
    monkeypatch, batched_available, splits
):
    """Interleaved head sizes retain independent normalization and row scaling."""
    monkeypatch.setattr(
        'megatron.core.optimizer.emerging_optimizers._supports_batched_newton_schulz',
        lambda: batched_available,
    )
    grad = torch.arange(sum(splits) * 4, dtype=torch.float32, device='cuda').reshape(-1, 4)
    grad[2].zero_()
    optimizer = TensorParallelMuon(
        params=[nn.Parameter(torch.zeros_like(grad))], split_qkv=True, split_qkv_per_head=True
    )
    call_shapes = []

    def normalize_head(x):
        call_shapes.append(tuple(x.shape))
        result = F.normalize(x, dim=(-2, -1), eps=1e-7) * x.shape[-2]
        return result.mT.contiguous().mT

    actual = optimizer._orthogonalize_split_qkv(
        grad, splits, normalize_head, adamw=(False,) * len(splits)
    )
    expected_calls = [(splits.count(s), s, 4) for s in dict.fromkeys(splits)]
    assert call_shapes == (expected_calls if batched_available else [(s, 4) for s in splits])
    expected = torch.cat([normalize_head(head) for head in grad.split(splits)])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.is_contiguous()


def test_muon_optimizer_mixed_heads_ns_matches_individual_updates():
    """Batched mixed-size NS preserves a multi-step per-head optimizer reference."""
    torch.manual_seed(1234)
    splits = [8, 1, 8, 1, 8]
    initial = torch.randn(sum(splits), 16, device='cuda')
    fused = nn.Parameter(initial.clone())
    heads = [nn.Parameter(head.clone()) for head in initial.split(splits)]
    from megatron.core.muon_layout import MuonProjectionLayout

    fused.muon_layout = MuonProjectionLayout.matrices(tuple(splits))
    kwargs = dict(
        lr=0.001,
        momentum=0.95,
        nesterov=True,
        num_ns_steps=5,
        fp32_matmul_prec='highest',
        scale_mode='spectral',
        extra_scale_factor=0.2,
    )
    actual = TensorParallelMuon(
        [fused],
        split_qkv=True,
        split_qkv_per_head=True,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        **kwargs,
    )
    reference = TensorParallelMuon(heads, **kwargs)
    for _ in range(5):
        gradient = torch.randn_like(initial)
        fused.grad = gradient.clone()
        for head, grad in zip(heads, gradient.split(splits)):
            head.grad = grad.clone()
        actual.step()
        reference.step()
        torch.testing.assert_close(fused, torch.cat(heads), rtol=1e-5, atol=1e-6)
