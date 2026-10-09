# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Per-head Muon layout, dispatch, and independent-head update regressions."""

import json
import logging
import os

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.emerging_optimizers import (
    HAVE_EMERGING_OPTIMIZERS,
    TensorParallelMuon,
    _get_qkv_split_shapes,
    _localize_qkv_split_shapes,
    _qkv_split_groups_are_complete,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import MLATransformerConfig, TransformerConfig
from megatron.core.transformer.attention import QKVLayout
from megatron.core.transformer.heterogeneous.heterogeneous_config import (
    HeterogeneousTransformerConfig,
)
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


def test_muon_qkv_split_shapes():
    config = TransformerConfig(
        num_layers=1, hidden_size=1024, num_attention_heads=16, num_query_groups=8
    )
    gated_config = TransformerConfig(
        num_layers=1,
        hidden_size=1024,
        num_attention_heads=16,
        num_query_groups=8,
        attention_output_gate=True,
    )

    assert _get_qkv_split_shapes(config) == [128, 64, 64]
    assert _get_qkv_split_shapes(gated_config) == [128, 128, 64, 64]
    assert _get_qkv_split_shapes(config, split_qkv_per_head=True) == [64] * 32
    assert _get_qkv_split_shapes(gated_config, split_qkv_per_head=True) == [64] * 48

    mla_layout = QKVLayout.from_splits(4, (128, 64))
    assert _get_qkv_split_shapes(mla_layout) == [128, 64]
    assert _get_qkv_split_shapes(mla_layout, split_qkv_per_head=True) == [128, 64] * 4


def test_muon_local_qkv_head_split_shapes_can_differ_by_tp_rank():
    """Rank-local per-head layouts report complete and fragmented heads."""
    global_split_shapes = [64] * 20

    rank_0_shapes, rank_0_complete = _localize_qkv_split_shapes(
        global_split_shapes, local_start=0, local_rows=160
    )
    rank_1_shapes, rank_1_complete = _localize_qkv_split_shapes(
        global_split_shapes, local_start=160, local_rows=160
    )
    aligned_shapes, aligned_complete = _localize_qkv_split_shapes(
        global_split_shapes, local_start=0, local_rows=640
    )

    assert rank_0_shapes == [64, 64, 32]
    assert rank_1_shapes == [32, 64, 64]
    assert not rank_0_complete
    assert not rank_1_complete
    assert aligned_shapes == [64] * 10
    assert aligned_complete


def test_muon_qkv_query_group_layout_localization():
    """Projection splitting detects query groups fragmented by TP row ranges."""
    split_shapes = [256, 64, 64]

    assert _qkv_split_groups_are_complete(split_shapes, local_start=0, local_rows=384)
    assert _qkv_split_groups_are_complete(split_shapes, local_start=384, local_rows=768)
    assert not _qkv_split_groups_are_complete(split_shapes, local_start=0, local_rows=192)
    assert not _qkv_split_groups_are_complete(split_shapes, local_start=192, local_rows=192)


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

    def test_optimizer_factory_preserves_default_when_tp_exceeds_query_groups(self):
        """Without the per-head flag, retain upstream tagging for incomplete TP groups."""
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        tp_size = pg_collection.tp.size()
        assert tp_size == 2
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=2,
            num_query_groups=1,
            kv_channels=4,
            tensor_model_parallel_size=tp_size,
            use_cpu_initialization=False,
            add_bias_linear=False,
        )
        model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_local_spec(),
            vocab_size=32,
            max_sequence_length=8,
            pre_process=False,
            post_process=False,
            pg_collection=pg_collection,
        )
        optimizer_config = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            use_distributed_optimizer=False,
            muon_split_qkv=True,
            muon_tp_mode="blockwise",
        )

        optimizer = get_megatron_optimizer(
            config=optimizer_config,
            model_chunks=[model],
            use_gloo_process_groups=False,
            pg_collection=pg_collection,
        )

        qkv_weight = model.decoder.layers[0].self_attention.linear_qkv.weight
        assert optimizer is not None
        assert qkv_weight.shape[0] == 8
        # Upstream's default path does not tag a TP shard containing a partial group.
        assert not getattr(qkv_weight, "is_qkv", False)

    @pytest.mark.parametrize("split_per_head", [False, True])
    def test_optimizer_factory_uses_heterogeneous_layer_qkv_layout(self, split_per_head):
        """Each heterogeneous attention layer supplies its own logical QKV layout."""
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        tp_size = pg_collection.tp.size()
        assert tp_size == 2
        model_parallel_cuda_manual_seed(123)
        block_configs = {
            "block_configs": [
                {
                    "attention": {
                        "no_op": False,
                        "replace_with_linear": False,
                        "num_query_groups": 2,
                    },
                    "mlp": {"no_op": False, "replace_with_linear": False, "ffn_hidden_size": 16},
                },
                {
                    "attention": {
                        "no_op": False,
                        "replace_with_linear": False,
                        "num_query_groups": 1,
                    },
                    "mlp": {"no_op": False, "replace_with_linear": False, "ffn_hidden_size": 16},
                },
            ]
        }
        transformer_config = HeterogeneousTransformerConfig(
            num_layers=2,
            hidden_size=8,
            num_attention_heads=2,
            kv_channels=4,
            tensor_model_parallel_size=tp_size,
            use_cpu_initialization=False,
            add_bias_linear=False,
            heterogeneous_layers_config_encoded_json=json.dumps(block_configs),
        )
        model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_heterogeneous_layer_spec(transformer_config),
            vocab_size=32,
            max_sequence_length=8,
            pre_process=False,
            post_process=False,
            pg_collection=pg_collection,
        )
        optimizer_config = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            use_distributed_optimizer=False,
            muon_split_qkv=True,
            muon_split_qkv_per_head=split_per_head,
            muon_tp_mode="blockwise",
        )

        optimizer = get_megatron_optimizer(
            config=optimizer_config,
            model_chunks=[model],
            use_gloo_process_groups=False,
            pg_collection=pg_collection,
        )

        first_qkv = model.decoder.layers[0].self_attention.linear_qkv.weight
        second_qkv = model.decoder.layers[1].self_attention.linear_qkv.weight
        assert optimizer is not None
        assert transformer_config.num_query_groups == 2
        assert first_qkv.qkv_layout.num_groups == 2
        assert second_qkv.qkv_layout.num_groups == 1
        assert first_qkv.shape[0] == 12
        assert second_qkv.shape[0] == 8
        assert first_qkv.is_qkv
        if split_per_head:
            assert second_qkv.is_qkv
            assert first_qkv.qkv_split_shapes_global == [4] * 6
            assert second_qkv.qkv_split_shapes_global == [4] * 4
            assert first_qkv.qkv_split_heads_are_complete
            assert second_qkv.qkv_split_heads_are_complete
        else:
            assert first_qkv.qkv_split_shapes == [4, 4, 4]
            assert not getattr(second_qkv, "is_qkv", False)

    @pytest.mark.parametrize("split_per_head", [False, True])
    @pytest.mark.parametrize("q_lora_rank", [None, 4])
    def test_optimizer_factory_uses_mla_up_projection_layouts(self, split_per_head, q_lora_rank):
        """MLA up-projections use module-owned layouts with TP-aware Muon splitting."""
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        tp_size = pg_collection.tp.size()
        assert tp_size == 2
        model_parallel_cuda_manual_seed(123)
        transformer_config = MLATransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=2,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=4,
            qk_head_dim=4,
            qk_pos_emb_head_dim=2,
            v_head_dim=3,
            tensor_model_parallel_size=tp_size,
            use_cpu_initialization=False,
            add_bias_linear=False,
            multi_latent_attention=True,
            rope_type="rope",
            rotary_base=10000,
            original_max_position_embeddings=8,
        )
        model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
                multi_latent_attention=True
            ),
            vocab_size=32,
            max_sequence_length=8,
            pre_process=False,
            post_process=False,
            pg_collection=pg_collection,
        )
        optimizer_config = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            use_distributed_optimizer=False,
            muon_split_qkv=True,
            muon_split_qkv_per_head=split_per_head,
            muon_tp_mode="blockwise",
        )

        optimizer = get_megatron_optimizer(
            config=optimizer_config,
            model_chunks=[model],
            use_gloo_process_groups=False,
            pg_collection=pg_collection,
        )

        attention = model.decoder.layers[0].self_attention
        q_up_weight = (
            attention.linear_q_proj.weight
            if q_lora_rank is None
            else attention.linear_q_up_proj.weight
        )
        kv_up_weight = attention.linear_kv_up_proj.weight
        assert optimizer is not None
        if split_per_head:
            assert q_up_weight.is_qkv
            assert kv_up_weight.is_qkv
            assert q_up_weight.qkv_split_shapes_global == [4, 2] * 2
            assert kv_up_weight.qkv_split_shapes_global == [4, 3] * 2
            assert q_up_weight.qkv_split_heads_are_complete
            assert kv_up_weight.qkv_split_heads_are_complete
        else:
            assert not getattr(q_up_weight, "is_qkv", False)
            assert not getattr(kv_up_weight, "is_qkv", False)
        if q_lora_rank is not None:
            assert not getattr(attention.linear_q_down_proj.weight, 'is_qkv', False)
        assert not getattr(attention.linear_kv_down_proj.weight, 'is_qkv', False)
        assert attention.linear_kv_down_proj.weight.muon_layout.splits == (4, 2)

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

    def test_optimizer_factory_skips_mismatched_qkv_layout(self):
        """A QKV layout mismatch falls back to whole-matrix Muon orthogonalization."""
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        tp_size = pg_collection.tp.size()
        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=2,
            num_query_groups=1,
            kv_channels=4,
            tensor_model_parallel_size=tp_size,
        )
        model = torch.nn.Module()
        model.config = transformer_config
        model.linear_qkv = torch.nn.Linear(8, 7, bias=False, dtype=torch.float32, device='cuda')
        model.linear_qkv.weight.tensor_model_parallel = True
        model.linear_qkv.weight.partition_dim = 0
        optimizer_config = OptimizerConfig(
            optimizer='muon', lr=0.01, use_distributed_optimizer=False, muon_split_qkv=True
        )

        optimizer = get_megatron_optimizer(
            config=optimizer_config,
            model_chunks=[model],
            use_gloo_process_groups=False,
            pg_collection=pg_collection,
        )

        qkv_weight = model.linear_qkv.weight
        assert optimizer is not None
        assert not getattr(qkv_weight, "is_qkv", False)

    def test_muon_optimizer_per_head_split_gathers_fragmented_heads(self):
        """Per-head splitting reconstructs heads that cross TP rank boundaries."""
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        tp_group = pg_collection.tp
        tp_rank = tp_group.rank()
        local_grad = torch.arange(12, dtype=torch.float32, device='cuda').view(3, 4)
        local_grad = local_grad + tp_rank * local_grad.numel()
        param = torch.nn.Parameter(torch.zeros_like(local_grad))
        param.partition_dim = 0
        param.is_qkv = True
        param.qkv_split_shapes = [2, 2]
        param.qkv_split_shapes_global = [2, 2, 2]
        param.qkv_split_heads_are_complete = False

        optimizer = TensorParallelMuon(
            params=[param],
            split_qkv=True,
            split_qkv_per_head=True,
            is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
            qkv_split_shapes=[2, 2, 2],
            pg_collection=pg_collection,
            tp_mode="blockwise",
        )

        def center_rows(x, tp_group=None, partition_dim=None, tp_mode_this_group=None):
            del tp_group, partition_dim, tp_mode_this_group
            return x - x.mean(dim=-2, keepdim=True)

        optimizer.scaled_orthogonalize_fn = center_rows
        actual = optimizer.orthogonalize(param, local_grad)

        shards = [torch.empty_like(local_grad) for _ in range(tp_group.size())]
        torch.distributed.all_gather(shards, local_grad, tp_group)
        global_grad = torch.cat(shards, dim=0)
        expected_global = torch.cat(
            [center_rows(head) for head in torch.split(global_grad, [2, 2, 2], dim=0)], dim=0
        )
        expected = expected_global[tp_rank * 3 : (tp_rank + 1) * 3]
        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("split_shapes", ([4, 2, 2], [4, 4, 2, 2], [3, 5]))
    def test_muon_optimizer_per_head_split_gathers_unequal_fragments(self, split_shapes):
        """Per-head splitting reconstructs unequal logical matrices cut by TP."""
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        tp_group = pg_collection.tp
        tp_rank = tp_group.rank()
        global_rows = sum(split_shapes)
        assert global_rows % tp_group.size() == 0
        local_rows = global_rows // tp_group.size()
        local_grad = torch.arange(local_rows * 4, dtype=torch.float32, device='cuda').view(
            local_rows, 4
        )
        local_grad = local_grad + tp_rank * local_grad.numel()
        param = torch.nn.Parameter(torch.zeros_like(local_grad))
        param.partition_dim = 0
        param.is_qkv = True
        param.qkv_split_shapes = split_shapes
        param.qkv_split_shapes_global = split_shapes
        param.qkv_split_groups_are_complete = False

        optimizer = TensorParallelMuon(
            params=[param],
            split_qkv=True,
            split_qkv_per_head=True,
            is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
            qkv_split_shapes=split_shapes,
            pg_collection=pg_collection,
            tp_mode="blockwise",
        )

        def center_rows(x, tp_group=None, partition_dim=None, tp_mode_this_group=None):
            del tp_group, partition_dim, tp_mode_this_group
            return x - x.mean(dim=-2, keepdim=True)

        optimizer.scaled_orthogonalize_fn = center_rows
        actual = optimizer.orthogonalize(param, local_grad)

        shards = [torch.empty_like(local_grad) for _ in range(tp_group.size())]
        torch.distributed.all_gather(shards, local_grad, tp_group)
        global_grad = torch.cat(shards, dim=0)
        expected_global = torch.cat(
            [
                center_rows(projection)
                for projection in torch.split(global_grad, split_shapes, dim=0)
            ],
            dim=0,
        )
        expected = expected_global[tp_rank * local_rows : (tp_rank + 1) * local_rows]
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("layout", "expects_fallback"), (("local_projection", False), ("per_head", True))
)
def test_muon_qkv_distributed_mode_routing_warns_once(monkeypatch, layout, expects_fallback):
    """Only complete local projection splits without GTP retain distributed NS."""
    grad = torch.arange(16, dtype=torch.float32, device='cuda').view(4, 4)
    param = torch.nn.Parameter(torch.zeros_like(grad))
    param.partition_dim = 0
    param.is_qkv = True
    param.qkv_split_shapes = [2, 1, 1]
    param.qkv_split_shapes_global = [2, 1, 1]
    param.qkv_split_groups_are_complete = layout != "fragmented"
    param.qkv_split_heads_are_complete = True
    if layout == "gtp":
        param.is_gtp_weight_remat = True

    optimizer = TensorParallelMuon(
        params=[param],
        split_qkv=True,
        split_qkv_per_head=layout == "per_head",
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=[2, 1, 1],
        pg_collection=None,
        tp_mode="distributed",
    )
    orthogonalize_args = []
    log_records = []

    def passthrough(x, tp_group=None, partition_dim=None, tp_mode_this_group=None):
        del tp_mode_this_group
        orthogonalize_args.append((tp_group, partition_dim))
        return x

    def record_log(_logger, level, message):
        log_records.append((level, message))

    optimizer.scaled_orthogonalize_fn = passthrough
    monkeypatch.setattr("megatron.core.optimizer.emerging_optimizers.log_single_rank", record_log)

    torch.testing.assert_close(optimizer.orthogonalize(param, grad), grad)
    torch.testing.assert_close(optimizer.orthogonalize(param, grad), grad)

    warning_messages = [message for level, message in log_records if level == logging.WARNING]
    if expects_fallback:
        assert len(warning_messages) == 1
        assert "falling back to non-TP Newton-Schulz" in warning_messages[0]
        assert all(
            tp_group is None and partition_dim is None
            for tp_group, partition_dim in orthogonalize_args
        )
    else:
        assert warning_messages == []
        assert all(partition_dim == 0 for _, partition_dim in orthogonalize_args)


def test_muon_optimizer_qkv_split_per_head_is_opt_in():
    """Per-head splitting is guarded and differs from projection splitting."""
    grad = torch.arange(48, dtype=torch.float32, device='cuda').view(16, 3)
    projection_param = torch.nn.Parameter(torch.zeros_like(grad))
    projection_param.is_qkv = True
    projection_param.qkv_split_shapes = [4, 2, 2]
    head_param = torch.nn.Parameter(torch.zeros_like(grad))
    head_param.is_qkv = True
    head_param.qkv_split_shapes = [2] * 8
    orthogonalize_call_shapes = []

    def center_rows(x, tp_group=None, partition_dim=None, tp_mode_this_group=None):
        del tp_group, partition_dim, tp_mode_this_group
        orthogonalize_call_shapes.append(tuple(x.shape))
        return x - x.mean(dim=-2, keepdim=True)

    projection_optimizer = TensorParallelMuon(
        params=[projection_param],
        split_qkv=True,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=[4, 2, 2],
        pg_collection=None,
    )
    projection_optimizer.scaled_orthogonalize_fn = center_rows
    projection_out = projection_optimizer.orthogonalize(projection_param, grad)
    orthogonalize_call_shapes.clear()

    head_optimizer = TensorParallelMuon(
        params=[head_param],
        split_qkv=True,
        split_qkv_per_head=True,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=[2] * 8,
        pg_collection=None,
    )
    head_optimizer.scaled_orthogonalize_fn = center_rows
    head_out = head_optimizer.orthogonalize(head_param, grad)
    assert orthogonalize_call_shapes == [(8, 2, 3)]

    expected_head_out = torch.cat(
        [center_rows(head) for head in torch.split(grad, [2] * 8, dim=0)], dim=0
    )
    torch.testing.assert_close(head_out, expected_head_out)
    assert not torch.equal(projection_out, head_out)


def test_muon_optimizer_qkv_split_per_head_requires_split_qkv():
    """The per-head switch cannot enable QKV splitting by itself."""
    param = torch.nn.Parameter(torch.zeros(4, 4, dtype=torch.float32, device='cuda'))
    with pytest.raises(ValueError, match="split_qkv_per_head requires split_qkv=True"):
        TensorParallelMuon(
            params=[param], split_qkv=False, split_qkv_per_head=True, pg_collection=None
        )


def test_muon_optimizer_uniform_per_head_splits_use_batched_ns():
    """Uniform per-head splits use Emerging-Optimizers' batched Newton-Schulz path."""
    grad = torch.arange(16, dtype=torch.float32, device='cuda').view(4, 4)
    param = torch.nn.Parameter(torch.zeros_like(grad))
    param.is_qkv = True
    param.qkv_split_shapes = [2, 2]
    optimizer = TensorParallelMuon(
        params=[param],
        split_qkv=True,
        split_qkv_per_head=True,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=[2, 2],
        pg_collection=None,
    )
    call_shapes = []

    def center_rows(x, tp_group=None, partition_dim=None, tp_mode_this_group=None):
        del tp_group, partition_dim, tp_mode_this_group
        call_shapes.append(tuple(x.shape))
        return x - x.mean(dim=-2, keepdim=True)

    optimizer.scaled_orthogonalize_fn = center_rows
    actual = optimizer.orthogonalize(param, grad)
    assert call_shapes == [(2, 2, 4)]
    expected = torch.cat(
        [head - head.mean(dim=-2, keepdim=True) for head in torch.split(grad, [2, 2])]
    )
    torch.testing.assert_close(actual, expected)


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

    def noncontiguous_identity(x, tp_group=None, partition_dim=None):
        del tp_group, partition_dim
        return x.mT.contiguous().mT

    optimizer.scaled_orthogonalize_fn = noncontiguous_identity
    actual = optimizer.orthogonalize(param, grad)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, grad)


def test_muon_optimizer_batched_per_head_ns_matches_individual_heads():
    """The pinned Emerging-Optimizers 3D Newton-Schulz path matches 2D head calls."""
    from emerging_optimizers.utils import fp32_matmul_precision

    torch.manual_seed(42)
    grad = torch.randn(8, 16, dtype=torch.float32, device='cuda')
    param = torch.nn.Parameter(torch.zeros_like(grad))
    param.is_qkv = True
    param.qkv_split_shapes = [2] * 4
    optimizer = TensorParallelMuon(
        params=[param],
        split_qkv=True,
        split_qkv_per_head=True,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=[2] * 4,
        fp32_matmul_prec="highest",
        num_ns_steps=2,
        pg_collection=None,
    )

    # step() establishes this context; direct orthogonalize() calls must do so too.
    with fp32_matmul_precision(optimizer.fp32_matmul_prec):
        actual = optimizer.orthogonalize(param, grad)
        expected = torch.cat(
            [
                optimizer.scaled_orthogonalize_fn(head, tp_group=None, partition_dim=None)
                for head in torch.split(grad, [2] * 4)
            ]
        )
    torch.testing.assert_close(actual, expected)


def test_muon_optimizer_nonuniform_per_head_splits_use_unbatched_ns():
    """Nonuniform per-head splits keep using individual Newton-Schulz calls."""
    grad = torch.arange(12, dtype=torch.float32, device='cuda').view(3, 4)
    param = torch.nn.Parameter(torch.zeros_like(grad))
    param.is_qkv = True
    param.qkv_split_shapes = [2, 1]
    optimizer = TensorParallelMuon(
        params=[param],
        split_qkv=True,
        split_qkv_per_head=True,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=[2, 1],
        pg_collection=None,
    )
    call_shapes = []

    def center_rows(x, tp_group=None, partition_dim=None, tp_mode_this_group=None):
        del tp_group, partition_dim, tp_mode_this_group
        call_shapes.append(tuple(x.shape))
        return x - x.mean(dim=-2, keepdim=True)

    optimizer.scaled_orthogonalize_fn = center_rows
    actual = optimizer.orthogonalize(param, grad)
    assert call_shapes == [(2, 4), (1, 4)]
    expected = torch.cat([center_rows(head) for head in torch.split(grad, [2, 1])])
    torch.testing.assert_close(actual, expected)


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

    actual = optimizer._orthogonalize_split_qkv(grad, splits, normalize_head)
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
    fused.is_qkv = True
    fused.qkv_split_shapes = splits
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
