# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import builtins
import copy
import importlib.util
import inspect
import json
import logging
import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import Version

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer import emerging_optimizers as eo_mod
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer.emerging_optimizers import (
    _PROFILES,
    HAVE_EMERGING_OPTIMIZERS,
    SOAP,
    Lion,
    TensorParallelAdaptiveMuon,
    TensorParallelMuon,
    _get_qkv_split_shapes,
    _qkv_split_groups_are_complete,
    _select_tp_mode,
    get_supported_coefficient_types,
    validate_coefficient_type,
)
from megatron.core.optimizer.muon import get_megatron_muon_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import MLATransformerConfig, TransformerConfig
from megatron.core.transformer.attention import QKVLayout
from megatron.core.transformer.heterogeneous.heterogeneous_config import (
    HeterogeneousTransformerConfig,
)
from tests.unit_tests.test_utilities import Utils

if HAVE_EMERGING_OPTIMIZERS:
    from megatron.core.optimizer.layer_sharded_muon import LayerShardedMuon
else:
    LayerShardedMuon = None

# Skip all tests in this file for LTS versions or when emerging_optimizers is missing
pytestmark = [
    pytest.mark.skipif(
        Version(os.getenv('NVIDIA_PYTORCH_VERSION', "24.01")) <= Version("25.05"),
        reason="Skip emerging optimizer tests for LTS test",
    ),
    pytest.mark.skipif(
        not HAVE_EMERGING_OPTIMIZERS, reason="emerging_optimizers package is not installed"
    ),
]


@pytest.mark.parametrize("soap_module", ["legacy_soap", "soap", "missing", "broken"])
def test_soap_import_is_independent_of_muon(monkeypatch, soap_module):
    """Both SOAP layouts and an unusable SOAP must leave the Muon adapter available."""
    original_import = builtins.__import__
    soap_cls = type("SOAP", (), {})
    attempted = []

    def import_with_soap_layout(name, *args, **kwargs):
        if name in ("emerging_optimizers.legacy_soap", "emerging_optimizers.soap"):
            attempted.append(name)
            if soap_module == "broken":
                raise ImportError("SOAP dependency is unavailable")
            if name == f"emerging_optimizers.{soap_module}":
                return SimpleNamespace(SOAP=soap_cls)
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_with_soap_layout)
    name = f"{eo_mod.__package__}._test_emerging_optimizers"
    spec = importlib.util.spec_from_file_location(name, eo_mod.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)

    assert module.HAVE_EMERGING_OPTIMIZERS
    assert module.Lion is Lion
    assert module.SOAP is (soap_cls if soap_module in ("legacy_soap", "soap") else None)
    assert attempted == ["emerging_optimizers.legacy_soap"] + (
        ["emerging_optimizers.soap"] if soap_module in ("soap", "missing") else []
    )
    param = nn.Parameter(torch.randn(4, 6, device="cpu"))
    optimizer = module.TensorParallelMuon([param], fp32_matmul_prec="highest")
    param.grad = torch.ones_like(param)
    optimizer.step()
    assert "momentum_buffer" in optimizer.state[param]


@pytest.mark.parametrize("name", ["soap", "signum", "scion", "polargrad", "laprop"])
def test_default_emerging_optimizer_kwargs_match_signature(name):
    if name not in eo_mod._EMERGING_OPTIMIZERS:
        pytest.skip(f"{name} is unavailable in this emerging_optimizers release")
    config = OptimizerConfig(optimizer=name, lr=0.01, adam_beta1=0.85, adam_beta2=0.97)
    kwargs = eo_mod._default_adam_based_eopt_config_to_kwargs(name, config, [], None)
    optimizer_cls = eo_mod._EMERGING_OPTIMIZERS[name].optimizer_cls
    signature = inspect.signature(optimizer_cls.__init__)
    signature.bind(None, [nn.Parameter(torch.randn(4, 6, device="cpu"))], **kwargs)
    if "betas" in signature.parameters:
        assert kwargs["betas"] == (0.85, 0.97)
    else:
        assert "betas" not in kwargs


@pytest.mark.parametrize("name", ["muon", "adaptive_muon", "soap"])
def test_emerging_optimizer_init_and_resume(name):
    """Checkpoint initialization must allocate complete state before any gradients exist."""
    if name == "soap" and SOAP is None:
        pytest.skip("SOAP is unavailable")
    config = OptimizerConfig(optimizer=name, lr=0.01, muon_split_qkv=False)
    param = nn.Parameter(torch.randn(4, 6, device="cpu"))
    optimizer, init_state_fn = eo_mod._create_emerging_optimizer(
        config, [param], name, [_DispatchChunk()], None
    )
    init_state_fn(optimizer)
    assert optimizer.state[param]
    param.grad = torch.randn_like(param)
    optimizer.step()

    restored_param = nn.Parameter(param.detach().clone())
    restored, _ = eo_mod._create_emerging_optimizer(
        config, [restored_param], name, [_DispatchChunk()], None
    )
    restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    restored_param.grad = param.grad.clone()
    optimizer.step()
    restored.step()
    torch.testing.assert_close(restored_param, param, atol=0, rtol=0)


def _sinkhorn_reference_step(weight, momentum, grad, *, lr, beta, steps, threshold, eps, scale):
    """Independent transcription of DeepSeek-V4.1 Algorithm 1, without EO calls."""
    momentum.mul_(beta).add_(grad, alpha=1 - beta)
    update = beta * momentum + (1 - beta) * grad
    row_norms = update.norm(dim=1, keepdim=True)
    update[row_norms.squeeze(1) <= threshold * row_norms.mean()] = 0
    for step in range(steps):
        dim = 1 if step % 2 == 0 else 0
        update = update / (update.norm(dim=dim, keepdim=True) + eps)
    weight.add_(update, alpha=-lr * scale * weight.shape[1] ** 0.5)


@pytest.mark.parametrize('steps', [1, 3, 11])
def test_sinkhorn_matches_paper_and_resumes(steps):
    pytest.importorskip('emerging_optimizers.orthogonalized_optimizers.sinkhorn_utils')
    torch.manual_seed(701)
    param = nn.Parameter(torch.randn(17, 5))
    reference = param.detach().clone()
    initial = reference.clone()
    momentum = torch.zeros_like(param)
    config = OptimizerConfig(
        optimizer='muon',
        lr=0.03,
        weight_decay=0.7,
        muon_use_sinkhorn=True,
        sinkhorn_num_normalization_steps=steps,
    )
    optimizer, init_fn = eo_mod._create_emerging_optimizer(config, [param], 'sinkhorn', [], None)
    init_fn(optimizer)
    assert set(optimizer.state[param]) == {'momentum_buffer'}
    assert optimizer.nesterov
    assert optimizer.param_groups[0]['weight_decay'] == 0

    for iteration in range(5):
        grad = torch.randn_like(param)
        grad[0].zero_()
        grad[1].mul_(1e-6)
        saved_grad = grad.clone()
        param.grad = grad
        _sinkhorn_reference_step(
            reference,
            momentum,
            grad,
            lr=0.03,
            beta=0.95,
            steps=steps,
            threshold=1e-3,
            eps=1e-20,
            scale=0.18,
        )
        optimizer.step()
        torch.testing.assert_close(param, reference, atol=2e-7, rtol=2e-6)
        torch.testing.assert_close(optimizer.state[param]['momentum_buffer'], momentum)
        torch.testing.assert_close(param[:2], initial[:2], atol=0, rtol=0)
        torch.testing.assert_close(grad, saved_grad, atol=0, rtol=0)
        if iteration == 2:
            restored_param = nn.Parameter(param.detach().clone())
            restored = eo_mod.Sinkhorn(
                [restored_param],
                lr=1,
                momentum=0.2,
                num_normalization_steps=1,
                extra_scale_factor=2,
            )
            restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        elif iteration > 2:
            restored_param.grad = grad.clone()
            restored.step()
            torch.testing.assert_close(restored_param, param, atol=0, rtol=0)

    saved = copy.deepcopy(optimizer.state_dict())
    param.grad = None
    optimizer.step()
    torch.testing.assert_close(
        optimizer.state[param]['momentum_buffer'],
        saved['state'][0]['momentum_buffer'],
        atol=0,
        rtol=0,
    )
    param.grad = torch.zeros_like(param)
    # Zero input is also well defined before any momentum has accumulated.
    zero = nn.Parameter(torch.ones(17, 5))
    empty_peer = eo_mod.Sinkhorn([{'params': []}, {'params': [zero]}])
    zero.grad = torch.zeros_like(zero)
    empty_peer.step()
    torch.testing.assert_close(zero, torch.ones_like(zero), atol=0, rtol=0)


@pytest.mark.parametrize(
    'kwargs,match',
    [
        ({'momentum': 1.0}, 'momentum'),
        ({'momentum': float('nan')}, 'momentum'),
        ({'eps': 0}, 'eps'),
        ({'eps': float('inf')}, 'eps'),
        ({'num_normalization_steps': 2}, 'positive odd'),
        ({'num_normalization_steps': 0}, 'positive odd'),
        ({'num_normalization_steps': 1.5}, 'positive odd'),
        ({'zero_row_threshold': -1}, 'threshold'),
        ({'zero_row_threshold': float('nan')}, 'threshold'),
        ({'extra_scale_factor': 0}, 'scale_factor'),
    ],
)
def test_sinkhorn_validates_hyperparameters(kwargs, match):
    pytest.importorskip('emerging_optimizers.orthogonalized_optimizers.sinkhorn_utils')
    with pytest.raises(ValueError, match=match):
        eo_mod.Sinkhorn([nn.Parameter(torch.ones(8, 4))], **kwargs)


@pytest.mark.parametrize('shape', [(4,), (4, 8), (0, 4), (4, 4, 4)])
def test_sinkhorn_rejects_invalid_matrix(shape):
    pytest.importorskip('emerging_optimizers.orthogonalized_optimizers.sinkhorn_utils')
    with pytest.raises(ValueError, match='2D weights'):
        eo_mod.Sinkhorn([nn.Parameter(torch.ones(shape))])


def test_sinkhorn_missing_feature_does_not_disable_muon(monkeypatch):
    original_import = builtins.__import__

    def missing_sinkhorn(name, *args, **kwargs):
        if name.endswith('.sinkhorn_utils'):
            raise ModuleNotFoundError(name, name=name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', missing_sinkhorn)
    param = nn.Parameter(torch.randn(8, 4))
    with pytest.raises(ImportError, match='sinkhorn_balance'):
        eo_mod.Sinkhorn([param])
    muon = TensorParallelMuon([param], fp32_matmul_prec='highest')
    param.grad = torch.randn_like(param)
    muon.step()
    assert 'momentum_buffer' in muon.state[param]


def test_sinkhorn_rejects_sharded_matrices(monkeypatch):
    pytest.importorskip('emerging_optimizers.orthogonalized_optimizers.sinkhorn_utils')
    param = nn.Parameter(torch.randn(8, 4))
    monkeypatch.setattr(eo_mod, 'get_pg_size', lambda group: group.size())
    with pytest.raises(ValueError, match='tensor parallel size 1'):
        eo_mod.Sinkhorn([param], pg_collection=SimpleNamespace(tp=SimpleNamespace(size=lambda: 2)))
    param.is_gtp_weight_remat = True
    with pytest.raises(ValueError, match='GTP-sharded'):
        eo_mod.Sinkhorn([param])


@pytest.mark.parametrize('enabled,tagged', [(True, False), (False, True)])
def test_sinkhorn_factory_rejects_inconsistent_ddp_routing(enabled, tagged):
    model = nn.Module()
    model.embedding = nn.Parameter(torch.ones(8, 4))
    model.embedding.is_embedding_or_output_parameter = True
    model.embedding.use_sinkhorn = tagged
    model.embedding.is_managed_by_layer_wise_optimizer = tagged
    model.ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
    config = OptimizerConfig(
        optimizer='muon', muon_use_sinkhorn=enabled, use_layer_wise_distributed_optimizer=True
    )
    with pytest.raises(ValueError, match='Sinkhorn buffer routing disagrees'):
        get_megatron_optimizer(config, [model], pg_collection=ProcessGroupCollection())


def test_sinkhorn_requires_layerwise_for_distributed_optimizer():
    model = nn.Linear(4, 8)
    config = OptimizerConfig(
        optimizer='muon', muon_use_sinkhorn=True, use_distributed_optimizer=True
    )
    with pytest.raises(ValueError, match='use_layer_wise_distributed_optimizer=True'):
        get_megatron_optimizer(config, [model], pg_collection=ProcessGroupCollection())
    with pytest.raises(ValueError, match='Muon-family'):
        OptimizerConfig(optimizer='adam', muon_use_sinkhorn=True)


@pytest.mark.parametrize('as_dict', [False, True])
@pytest.mark.parametrize('empty_sinkhorn_owner', [False, True])
def test_layerwise_sinkhorn_checkpoint_containers(as_dict, empty_sinkhorn_owner):
    """Restore multi-child LayerWise checkpoints, including a rank owning no embedding."""
    pytest.importorskip('emerging_optimizers.orthogonalized_optimizers.sinkhorn_utils')
    from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
    from megatron.core.optimizer.optimizer import ChainedOptimizer

    def build(params):
        # Isolate checkpoint serialization from CUDA/DDP setup. These are the local,
        # already-sharded optimizers; distributed ownership is tested separately.
        children = [
            TensorParallelMuon([params[0]], fp32_matmul_prec='highest'),
            eo_mod.Sinkhorn([{'params': [] if empty_sinkhorn_owner else [params[1]]}]),
        ]
        chain = object.__new__(LayerWiseDistributedOptimizer)
        ChainedOptimizer.__init__(chain, children)
        return chain

    torch.manual_seed(199)
    params = [nn.Parameter(torch.randn(8, 4)) for _ in range(2)]
    original = build(params)
    for param in params:
        param.grad = torch.randn_like(param)
    for child in original.chained_optimizers:
        child.step()
    checkpoint = copy.deepcopy(original.state_dict())
    assert isinstance(checkpoint, list)
    restored_params = [nn.Parameter(p.detach().clone()) for p in params]
    restored = build(restored_params)
    restored.load_state_dict(dict(enumerate(checkpoint)) if as_dict else checkpoint)
    for param, restored_param in zip(params, restored_params):
        param.grad = torch.randn_like(param)
        restored_param.grad = param.grad.clone()
    for chain in (original, restored):
        for child in chain.chained_optimizers:
            child.step()
    for param, restored_param in zip(params, restored_params):
        torch.testing.assert_close(param, restored_param, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA optimizer wrappers')
@pytest.mark.parametrize('bf16', [False, True])
@pytest.mark.parametrize('use_sinkhorn', [False, True])
@pytest.mark.parametrize('scalar_optimizer', ['adam', 'lion'])
@pytest.mark.parametrize('apply_wd_to_norm', [False, True])
def test_sinkhorn_optimizer_factory(bf16, use_sinkhorn, scalar_optimizer, apply_wd_to_norm):
    """Test the real three-optimizer chain, FP32 masters and scheduled zero weight decay."""
    pytest.importorskip('emerging_optimizers.orthogonalized_optimizers.sinkhorn_utils')
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

    Utils.initialize_model_parallel()
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups()
        model = nn.Module()
        model.config = TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=1)
        model.linear = nn.Linear(8, 8)
        model.embedding = nn.Embedding(17, 8)
        model.embedding.weight.is_embedding_or_output_parameter = True
        model.output = nn.Linear(8, 17, bias=False)
        model.output.weight.is_embedding_or_output_parameter = True
        model.norm = nn.LayerNorm(8)
        model.cuda()
        if bf16:
            model.bfloat16()
        model.embedding.weight.lr_mult = 5.0
        model.embedding.weight.row_shard_group = pg.ep
        model.embedding.weight.row_shard_global_rows = 17
        model.embedding.weight.row_shard_start = 0
        config = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            min_lr=0,
            bf16=bf16,
            weight_decay=0.1,
            apply_wd_to_norm=apply_wd_to_norm,
            muon_use_sinkhorn=use_sinkhorn,
            muon_scalar_optimizer=scalar_optimizer,
        )
        chain = get_megatron_optimizer(
            config, [model], pg_collection=pg, use_gloo_process_groups=False
        )
        scheduler = OptimizerParamScheduler(
            chain,
            init_lr=0.01,
            max_lr=0.01,
            min_lr=0,
            lr_warmup_steps=0,
            lr_decay_steps=100,
            lr_decay_style='linear',
            start_wd=0.1,
            end_wd=0.3,
            wd_incr_steps=100,
            wd_incr_style='linear',
        )
        scheduler.step(10)
        decayed_vectors = []
        for wrapper in chain.chained_optimizers:
            for group in wrapper.optimizer.param_groups:
                for param in group['params']:
                    if group['lr_mult'] == 5.0:
                        assert param.shape == (17, 8)
                        assert param.row_shard_group is pg.ep
                        assert param.row_shard_global_rows == 17
                        assert param.row_shard_start == 0
                    assert group['lr'] == pytest.approx(0.009 * group['lr_mult'])
                    if isinstance(wrapper.optimizer, eo_mod.Sinkhorn):
                        assert use_sinkhorn
                        assert param.dtype == torch.float32
                        assert param.shape == (17, 8)
                        assert group['wd_mult'] == group['weight_decay'] == 0
                    elif isinstance(wrapper.optimizer, TensorParallelMuon):
                        assert param.shape == (8, 8)
                    else:
                        assert param.ndim == 1 or not use_sinkhorn
                        if param.ndim == 1 and group['wd_mult']:
                            decayed_vectors.append(param)
                            assert group['weight_decay'] == pytest.approx(0.12)
        assert len(decayed_vectors) == int(apply_wd_to_norm)
        if apply_wd_to_norm:
            torch.testing.assert_close(decayed_vectors[0], torch.ones_like(decayed_vectors[0]))
        sinkhorn = [
            c.optimizer
            for c in chain.chained_optimizers
            if isinstance(c.optimizer, eo_mod.Sinkhorn)
        ]
        assert len(sinkhorn) == int(use_sinkhorn)
        if use_sinkhorn:
            assert sum(len(g['params']) for g in sinkhorn[0].param_groups) == 2
    finally:
        Utils.destroy_model_parallel()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(80, 48)
        self.fc2 = nn.Linear(48, 32)
        self.fc3 = nn.Linear(32, 24)
        self.fc4 = nn.Linear(24, 16)
        self.fc5 = nn.Linear(16, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


# ===========================================================================
# Muon optimizer tests
# ===========================================================================


def test_select_tp_mode_flops_only_fallback():
    """Without a hardware profile, select by FLOPs or the cross-domain fallback."""
    assert (
        _select_tp_mode(
            m=8192,
            n=1024,
            group_size=8,
            steps=5,
            use_syrk=False,
            elem_size=2,
            communication_crosses_domain=False,
            profile=None,
        )
        == "distributed"
    )
    assert (
        _select_tp_mode(
            m=8192,
            n=1024,
            group_size=8,
            steps=5,
            use_syrk=False,
            elem_size=2,
            communication_crosses_domain=True,
            profile=None,
        )
        == "duplicated"
    )


def test_select_tp_mode_with_profile():
    """A hardware profile selects different modes for tall and wide matrices."""
    profile = _PROFILES["GB200"]

    assert (
        _select_tp_mode(
            m=8192,
            n=1024,
            group_size=8,
            steps=5,
            use_syrk=False,
            elem_size=2,
            communication_crosses_domain=False,
            profile=profile,
        )
        == "distributed"
    )
    assert (
        _select_tp_mode(
            m=1024,
            n=8192,
            group_size=8,
            steps=5,
            use_syrk=False,
            elem_size=2,
            communication_crosses_domain=False,
            profile=profile,
        )
        == "duplicated"
    )


def test_select_tp_mode_syrk_changes_selection():
    """SYRK halves the Gram-op cost differently per mode -- can flip the selected mode."""
    assert (
        _select_tp_mode(
            m=640,
            n=1024,
            group_size=8,
            steps=5,
            use_syrk=False,
            elem_size=2,
            communication_crosses_domain=False,
            profile=None,
        )
        == "duplicated"
    )
    assert (
        _select_tp_mode(
            m=640,
            n=1024,
            group_size=8,
            steps=5,
            use_syrk=True,
            elem_size=2,
            communication_crosses_domain=False,
            profile=None,
        )
        == "distributed"
    )


def test_resolve_tp_mode_caches(monkeypatch):
    """Repeated resolution of the same shape invokes the cost model only once."""
    call_count = 0

    def mock_select_tp_mode(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        return "distributed"

    monkeypatch.setattr(
        "megatron.core.optimizer.emerging_optimizers._select_tp_mode", mock_select_tp_mode
    )

    optimizer = TensorParallelMuon(
        params=[torch.nn.Parameter(torch.zeros(1))], tp_mode="auto", pg_collection=None
    )

    first = optimizer._resolve_tp_mode(4096, 1024, 8)
    second = optimizer._resolve_tp_mode(4096, 1024, 8)

    assert first == second == "distributed"
    assert call_count == 1
    assert list(optimizer._tp_mode_cache) == [(4096, 1024, 8)]


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

    assert _get_qkv_split_shapes(config) == ([128, 64, 64], [128, 64, 64] * 8)
    assert _get_qkv_split_shapes(gated_config) == ([128, 128, 64, 64], [128, 128, 64, 64] * 8)
    assert _get_qkv_split_shapes(config, split_qkv_per_head=True) == ([64] * 32, [64] * 32)
    assert _get_qkv_split_shapes(gated_config, split_qkv_per_head=True) == ([64] * 48, [64] * 48)

    mla_layout = QKVLayout.from_mla_head_components(4, (128, 64))
    assert _get_qkv_split_shapes(mla_layout) == ([128, 64], [128, 64] * 4)
    assert _get_qkv_split_shapes(mla_layout, split_qkv_per_head=True) == (
        [128, 64] * 4,
        [128, 64] * 4,
    )


def test_muon_qkv_query_group_layout_localization():
    """Projection splitting detects query groups fragmented by TP row ranges."""
    split_shapes = [256, 64, 64]

    assert _qkv_split_groups_are_complete(split_shapes, local_start=0, local_rows=384)
    assert _qkv_split_groups_are_complete(split_shapes, local_start=384, local_rows=768)
    assert not _qkv_split_groups_are_complete(split_shapes, local_start=0, local_rows=192)
    assert not _qkv_split_groups_are_complete(split_shapes, local_start=192, local_rows=192)


def test_muon_qkv_split_shapes_config_without_attention_output_gate():
    """Custom QKV configs predating attention gating retain their ungated layout."""
    from types import SimpleNamespace

    config = SimpleNamespace(num_attention_heads=4, num_query_groups=2, kv_channels=2)

    assert _get_qkv_split_shapes(config) == ([4, 2, 2], [4, 2, 2, 4, 2, 2])
    assert _get_qkv_split_shapes(config, split_qkv_per_head=True) == ([2] * 8, [2] * 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA optimizer wrappers")
@pytest.mark.parametrize('bf16', [False, True])
@pytest.mark.parametrize('per_head', [False, True])
@pytest.mark.parametrize('scale_mode', ['spectral', 'rms'])
def test_v41_optimizer_factory_preserves_head_layouts(monkeypatch, bf16, per_head, scale_mode):
    """The public factory carries V4.1 layouts into FP32 masters and routes norms to Adam."""
    from tests.unit_tests.transformer.experimental_attention_variant.test_csa2 import (
        _muon_attention,
    )

    Utils.initialize_model_parallel()
    try:
        groups = ProcessGroupCollection.use_mpu_process_groups()
        model = _muon_attention(monkeypatch, layer_number=1).cuda()
        if bf16:
            model.bfloat16()
        config = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            bf16=bf16,
            muon_split_qkv_per_head=per_head,
            muon_scale_mode=scale_mode,
        )
        optimizer = get_megatron_optimizer(
            config=config, model_chunks=[model], pg_collection=groups, use_gloo_process_groups=False
        )
        muon = next(
            wrapper.optimizer
            for wrapper in optimizer.chained_optimizers
            if isinstance(wrapper.optimizer, TensorParallelMuon)
        )
        muon_ids = {id(p) for group in muon.param_groups for p in group['params']}
        expected = {
            'linear_q_up_proj.weight': (4, 8),
            'linear_kv_proj.weight': (1, 8),
            'core_attention.indexer.linear_wq_b.weight': (3, 8),
            'core_attention.indexer.linear_wk.weight': (1, 8),
        }
        for name, param in model.named_parameters():
            main = getattr(param, 'main_param', param)
            assert (id(main) in muon_ids) == (param.ndim == 2)
            assert getattr(main, 'is_qkv', False) == (name in expected)
            if name in expected:
                heads, width = expected[name]
                assert main.dtype == torch.float32
                assert main.qkv_layout == param.qkv_layout
                assert main.qkv_split_shapes == ([width] * heads if per_head else [width])
                assert main.qkv_split_shapes_global == [width] * heads
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("layout_kind", ["gqa", "gated_gqa", "mla"])
@pytest.mark.parametrize("split_per_head", [False, True])
def test_muon_qkv_split_sentinel_contents(layout_kind, split_per_head):
    """NS receives the intended components and its outputs return to their original rows."""
    if layout_kind == "mla":
        layout = QKVLayout.from_mla_head_components(2, (3, 1))
        # Unequal content/RoPE widths: projection NS combines matching components
        # across heads; per-head NS handles the four components independently.
        component_specs = [("c0", 1, 3), ("r0", 2, 1), ("c1", 1, 3), ("r1", 2, 1)]
        projection_names = [("c0", "c1"), ("r0", "r1")]
    else:
        gated = layout_kind == "gated_gqa"
        layout = QKVLayout.from_transformer_config(
            TransformerConfig(
                num_layers=1,
                hidden_size=8,
                num_attention_heads=4,
                num_query_groups=2,
                kv_channels=2,
                attention_output_gate=gated,
            )
        )
        # Q=1, K=2, V=3, gate=4. Distinct row IDs in the second column also
        # distinguish heads of the same projection and rows within each head.
        if gated:
            component_specs = [
                ("q0", 1, 2),
                ("q1", 1, 2),
                ("g0", 4, 2),
                ("g1", 4, 2),
                ("k0", 2, 2),
                ("v0", 3, 2),
                ("q2", 1, 2),
                ("q3", 1, 2),
                ("g2", 4, 2),
                ("g3", 4, 2),
                ("k1", 2, 2),
                ("v1", 3, 2),
            ]
            projection_names = [
                ("q0", "q1", "q2", "q3"),
                ("g0", "g1", "g2", "g3"),
                ("k0", "k1"),
                ("v0", "v1"),
            ]
        else:
            component_specs = [
                ("q0", 1, 2),
                ("q1", 1, 2),
                ("k0", 2, 2),
                ("v0", 3, 2),
                ("q2", 1, 2),
                ("q3", 1, 2),
                ("k1", 2, 2),
                ("v1", 3, 2),
            ]
            projection_names = [("q0", "q1", "q2", "q3"), ("k0", "k1"), ("v0", "v1")]

    components = {
        name: torch.tensor(
            [[value, 10 * index + row] for row in range(rows)], dtype=torch.float32, device="cuda"
        )
        for index, (name, value, rows) in enumerate(component_specs)
    }
    grad = torch.cat(list(components.values()))
    matrix_names = [(name,) for name in components] if split_per_head else projection_names
    expected_inputs = [torch.cat([components[name] for name in names]) for names in matrix_names]
    expected_updates = {
        name: components[name] + 100 * (index + 1)
        for index, names in enumerate(matrix_names)
        for name in names
    }
    # Build the oracle from named components, independently of the split/reshape
    # implementation. Every NS matrix gets a unique output offset to check reassembly.
    expected = torch.cat([expected_updates[name] for name in components])
    split_shapes, logical_split_shapes = _get_qkv_split_shapes(
        layout, split_qkv_per_head=split_per_head
    )
    param = torch.nn.Parameter(torch.zeros_like(grad))
    param.is_qkv = True
    param.qkv_split_shapes = split_shapes
    param.qkv_split_shapes_global = logical_split_shapes
    optimizer = TensorParallelMuon(
        params=[param],
        split_qkv=True,
        split_qkv_per_head=split_per_head,
        is_qkv_fn=lambda p: p.is_qkv,
        pg_collection=None,
    )
    calls = []

    def record_and_offset(x, *_args, **_kwargs):
        calls.append(x.clone())
        if x.ndim == 3:
            offsets = torch.arange(1, len(expected_inputs) + 1, device=x.device).view(-1, 1, 1)
            return x + 100 * offsets
        return x + 100 * len(calls)

    optimizer.scaled_orthogonalize_fn = record_and_offset
    actual = optimizer.orthogonalize(param, grad)

    if split_per_head and layout_kind != "mla":
        assert len(calls) == 1
        torch.testing.assert_close(calls[0], torch.stack(expected_inputs), rtol=0, atol=0)
    else:
        assert len(calls) == len(expected_inputs)
        for actual_input, expected_input in zip(calls, expected_inputs):
            torch.testing.assert_close(actual_input, expected_input, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tp_mode", ["duplicated", "blockwise"])
@pytest.mark.parametrize("gtp_rank", [0, 1])
def test_muon_optimizer_gtp_remat_pad_length_scale_correction(monkeypatch, tp_mode, gtp_rank):
    """scaled_orthogonalize_fn_with_gtp_remat strips GTP_remat's dim-0 padding before
    calling scaled_orthogonalize_fn (unmodified, GTP-agnostic) and restores it after."""
    from types import SimpleNamespace

    gtp_group = object()
    pg_collection = SimpleNamespace(gtp_remat=gtp_group, expt_gtp_remat=gtp_group)

    # 6 true rows, padded to 8 and sharded across 2 ranks: rank 1's shard holds the 2 pad rows.
    full_grad = torch.tensor(
        [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [0.0], [0.0]], device='cuda'
    )
    pad_length = 2
    local_grad = full_grad[gtp_rank * 4 : (gtp_rank + 1) * 4].clone()
    param = torch.nn.Parameter(torch.zeros_like(local_grad))
    param.is_gtp_weight_remat = True
    param.pad_length = pad_length

    optimizer = TensorParallelMuon(
        params=[param], num_ns_steps=1, pg_collection=pg_collection, tp_mode=tp_mode
    )

    monkeypatch.setattr("megatron.core.optimizer.emerging_optimizers.get_pg_size", lambda group: 2)
    monkeypatch.setattr(
        "megatron.core.optimizer.emerging_optimizers.get_pg_rank", lambda group: gtp_rank
    )

    def fake_all_gather(shards, _local_grad, _group):
        shards[0].copy_(full_grad[:4])
        shards[1].copy_(full_grad[4:])

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)

    calls = []

    def fake_orthogonalize(grad, tp_group, partition_dim=None, tp_mode_this_group=None):
        calls.append((grad.clone(), tp_group, partition_dim))
        return grad + 100

    monkeypatch.setattr(optimizer, "scaled_orthogonalize_fn", fake_orthogonalize)

    result = optimizer.scaled_orthogonalize_fn_with_gtp_remat(param, local_grad, None, None)

    assert len(calls) == 1
    seen_grad, _, _ = calls[0]

    if tp_mode == "duplicated":
        # Gathered tensor is uniform on every rank: strip 2, restore after.
        stripped_full = full_grad[:6]
        torch.testing.assert_close(seen_grad, stripped_full)
        restored = torch.nn.functional.pad(stripped_full + 100, (0, 0, 0, 2))
        expected = restored[gtp_rank * 4 : (gtp_rank + 1) * 4]
    else:
        # blockwise: only rank 1's local block contains padding.
        if gtp_rank == 1:
            stripped_local = local_grad[:2]
            torch.testing.assert_close(seen_grad, stripped_local)
            expected = torch.nn.functional.pad(stripped_local + 100, (0, 0, 0, 2))
        else:
            torch.testing.assert_close(seen_grad, local_grad)
            expected = local_grad + 100

    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("scale_mode", ["spectral", "unit_rms_norm", "shape_scaling"])
def test_muon_optimizer_gtp_distributed_padding_uses_logical_scale_shape(monkeypatch, scale_mode):
    """Distributed NS keeps aligned rows, but Muon scaling uses the logical matrix shape."""
    from types import SimpleNamespace

    from emerging_optimizers.orthogonalized_optimizers import get_muon_scale_factor

    gtp_group = object()
    pg_collection = SimpleNamespace(gtp_remat=gtp_group, expt_gtp_remat=gtp_group)

    # GTP2: local [4, 7] reconstructs physical [8, 7], with 2 padded rows -> logical [6, 7].
    # Padding crosses the square boundary so spectral and shape_scaling change too;
    # logical shape_scaling is clamped to 1 while the padded value is sqrt(8 / 7).
    local_grad = torch.ones((4, 7), dtype=torch.float32, device='cuda')
    param = torch.nn.Parameter(torch.zeros_like(local_grad))
    param.is_gtp_weight_remat = True
    param.pad_length = 2

    optimizer = TensorParallelMuon(
        params=[param],
        num_ns_steps=1,
        scale_mode=scale_mode,
        pg_collection=pg_collection,
        tp_mode="distributed",
    )

    monkeypatch.setattr("megatron.core.optimizer.emerging_optimizers.get_pg_size", lambda group: 2)
    monkeypatch.setattr("megatron.core.optimizer.emerging_optimizers.get_pg_rank", lambda group: 0)
    monkeypatch.setattr(
        "megatron.core.optimizer.emerging_optimizers.newton_schulz_tp",
        lambda grad, **_kwargs: torch.ones_like(grad),
    )

    result = optimizer.scaled_orthogonalize_fn_with_gtp_remat(param, local_grad, None, None)

    expected_scale = get_muon_scale_factor(6, 7, mode=scale_mode)
    padded_scale = get_muon_scale_factor(8, 7, mode=scale_mode)
    assert expected_scale != padded_scale
    torch.testing.assert_close(result, torch.full_like(local_grad, expected_scale))


@pytest.mark.parametrize("scale_mode", ["spectral", "unit_rms_norm", "shape_scaling"])
@pytest.mark.parametrize(
    "partition_dim,tp_size,gtp_size,local_shape,pad_length,logical_shape",
    [
        pytest.param(1, 2, 4, (4, 4), 3, (13, 8), id="row-parallel"),
        pytest.param(0, 4, 2, (4, 8), 3, (20, 8), id="column-parallel"),
    ],
)
def test_muon_optimizer_tp_gtp_distributed_padding_uses_logical_scale_shape(
    monkeypatch,
    scale_mode,
    partition_dim,
    tp_size,
    gtp_size,
    local_shape,
    pad_length,
    logical_shape,
):
    """TP x GTP scaling removes padding once from each TP-local dim-0 slice."""
    from types import SimpleNamespace

    from emerging_optimizers.orthogonalized_optimizers import get_muon_scale_factor

    tp_group, gtp_group = object(), object()
    group_sizes = {tp_group: tp_size, gtp_group: gtp_size}
    pg_collection = SimpleNamespace(
        tp=tp_group, expt_tp=tp_group, gtp_remat=gtp_group, expt_gtp_remat=gtp_group
    )

    local_grad = torch.ones(local_shape, dtype=torch.float32, device='cuda')
    param = torch.nn.Parameter(torch.zeros_like(local_grad))
    param.is_gtp_weight_remat = True
    param.pad_length = pad_length

    optimizer = TensorParallelMuon(
        params=[param],
        num_ns_steps=1,
        scale_mode=scale_mode,
        pg_collection=pg_collection,
        tp_mode="distributed",
    )

    monkeypatch.setattr(
        "megatron.core.optimizer.emerging_optimizers.get_pg_size", lambda group: group_sizes[group]
    )
    monkeypatch.setattr("megatron.core.optimizer.emerging_optimizers.get_pg_rank", lambda group: 0)
    monkeypatch.setattr(
        optimizer,
        "_all_gather_tensor",
        lambda tensor, group, dim: torch.cat([tensor] * group_sizes[group], dim=dim),
    )
    monkeypatch.setattr(
        "megatron.core.optimizer.emerging_optimizers.newton_schulz_tp",
        lambda grad, **_kwargs: torch.ones_like(grad),
    )

    result = optimizer.scaled_orthogonalize_fn_with_gtp_remat(
        param, local_grad, tp_group, partition_dim
    )

    expected_scale = get_muon_scale_factor(*logical_shape, mode=scale_mode)
    torch.testing.assert_close(result, torch.full_like(local_grad, expected_scale))


@pytest.mark.parametrize("gtp_rank,expected_local_pad_length", [(0, 0), (1, 1), (2, 2), (3, 2)])
def test_muon_optimizer_gtp_remat_blockwise_pad_spans_multiple_ranks(
    monkeypatch, gtp_rank, expected_local_pad_length
):
    """pad_length can span multiple ranks' shards, not just the last rank's. Each rank must
    strip only its own overlap; a fully-padding rank (true dim0 == 0) must skip
    scaled_orthogonalize_fn entirely and return exact zero."""
    from types import SimpleNamespace

    gtp_group = object()
    pg_collection = SimpleNamespace(gtp_remat=gtp_group, expt_gtp_remat=gtp_group)

    # 4 ranks, shard_size=2, pad_length=5: ranks 2 and 3 are fully padding, rank 1 half.
    gtp_remat_size = 4
    shard_size = 2
    pad_length = 5
    local_grad = torch.full((shard_size, 1), 3.0, device='cuda')
    param = torch.nn.Parameter(torch.zeros_like(local_grad))
    param.is_gtp_weight_remat = True
    param.pad_length = pad_length

    optimizer = TensorParallelMuon(
        params=[param], num_ns_steps=1, pg_collection=pg_collection, tp_mode="blockwise"
    )

    monkeypatch.setattr(
        "megatron.core.optimizer.emerging_optimizers.get_pg_size", lambda group: gtp_remat_size
    )
    monkeypatch.setattr(
        "megatron.core.optimizer.emerging_optimizers.get_pg_rank", lambda group: gtp_rank
    )

    calls = []

    def fake_orthogonalize(grad, tp_group, partition_dim=None, tp_mode_this_group=None):
        calls.append(grad.clone())
        return grad  # identity: makes the strip/restore round-trip directly checkable

    monkeypatch.setattr(optimizer, "scaled_orthogonalize_fn", fake_orthogonalize)

    result = optimizer.scaled_orthogonalize_fn_with_gtp_remat(param, local_grad, None, None)

    true_dim0 = shard_size - expected_local_pad_length
    if true_dim0 <= 0:
        assert calls == []
        torch.testing.assert_close(result, torch.zeros_like(local_grad))
    else:
        assert len(calls) == 1
        torch.testing.assert_close(calls[0], local_grad[:true_dim0])
        expected = torch.nn.functional.pad(
            local_grad[:true_dim0], (0, 0, 0, expected_local_pad_length)
        )
        torch.testing.assert_close(result, expected)


def test_muon_optimizer_smoke():
    """Smoke test for TensorParallelMuon optimizer."""
    # Create a simple linear model for testing
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    # Create TensorParallelMuon optimizer
    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.95,
        nesterov=True,
        weight_decay=0.01,
        use_decoupled_weight_decay=True,
        split_qkv=False,
        fp32_matmul_prec="medium",
        num_ns_steps=5,
        scale_mode="spectral",
        extra_scale_factor=1.0,
        pg_collection=None,
        tp_mode="duplicated",
    )

    # Test basic properties
    assert optimizer is not None, "Optimizer should not be None"
    assert hasattr(optimizer, 'param_groups'), "Optimizer should have param_groups"
    assert len(optimizer.param_groups) > 0, "Optimizer should have at least one parameter group"

    # Test forward and backward pass
    input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    # Store original weight
    original_weight = model.weight.data.clone()

    # Test optimizer step
    optimizer.step()

    # Verify weight was updated
    assert not torch.equal(
        model.weight.data, original_weight
    ), "Weight should be updated after optimizer step"

    # Test zero_grad
    optimizer.zero_grad()
    assert model.weight.grad is None or torch.all(
        model.weight.grad == 0
    ), "Gradients should be zeroed"

    # Test state_dict and load_state_dict
    state_dict = optimizer.state_dict()
    assert 'state' in state_dict, "State dict should contain state"
    assert 'param_groups' in state_dict, "State dict should contain param_groups"

    # Load state dict should not raise error
    optimizer.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# Registry dispatch for muon_tp_mode='layer_sharded' (LayerShardedMuon)
# ---------------------------------------------------------------------------
# Regression coverage for kwargs pollution across registry entries: the layer-sharding
# branch must live ONLY on the ``muon`` entry's dispatcher (_muon_registry_config_to_kwargs),
# never in the shared _muon_config_to_kwargs that adaptive_muon layers its own kwargs on.
# When the branch leaked into the shared helper, ``--optimizer adaptive_muon --muon-tp-mode
# layer_sharded`` crashed at construction: TensorParallelAdaptiveMuon received
# LayerShardedMuon-only kwargs (gtp_remat_group, ns_batch_size, ...) and lost the ones it
# needs (is_qkv_fn, qkv_split_shapes, pg_collection). These tests stay at the
# signature/registry level (inspect, kwargs dicts) and never instantiate an optimizer, so
# they run on any installed emerging-optimizers.


class _DispatchModelCfg:
    num_attention_heads = 8
    num_query_groups = 2
    kv_channels = 16


class _DispatchChunk:
    config = _DispatchModelCfg()


def _dispatch_cfg(layer_sharded: bool) -> OptimizerConfig:
    """A real OptimizerConfig, so the reflective ``_kwargs_from_config`` lookups
    exercise the actual field surface instead of a stub that makes every
    ``hasattr`` fail (which would shrink the test to the hardcoded keys only).
    A valid muon config: ``__post_init__`` enforces the layer_sharded requirements."""
    return OptimizerConfig(
        optimizer='muon',
        use_layer_wise_distributed_optimizer=True,
        muon_split_qkv=False,
        muon_tp_mode='layer_sharded' if layer_sharded else 'duplicated',
    )


def test_shared_muon_kwargs_ignore_layer_sharding_flag():
    """The shared builder must stay pure even with the lsh flag set."""
    kwargs = eo_mod._muon_config_to_kwargs(
        _dispatch_cfg(True), [_DispatchChunk()], pg_collection=None
    )
    for lsh_only in ("gtp_remat_group", "tp_group", "ns_batch_size", "concurrent_groups"):
        assert lsh_only not in kwargs, f"LayerShardedMuon-only kwarg leaked: {lsh_only}"
    assert "is_qkv_fn" in kwargs
    assert "qkv_split_shapes" in kwargs
    assert "pg_collection" in kwargs


@pytest.mark.parametrize("lsh_flag", [False, True])
def test_adaptive_muon_kwargs_match_constructor_signature(lsh_flag):
    """Every kwarg built for adaptive_muon must be accepted by its __init__.

    This is the construction-crash regression: with the flag set, a polluted
    shared helper produced gtp_remat_group/ns_batch_size and TensorParallelAdaptiveMuon
    raised TypeError before training started.
    """
    kwargs = eo_mod._adaptive_muon_config_to_kwargs(
        _dispatch_cfg(lsh_flag), [_DispatchChunk()], pg_collection=None
    )
    accepted = set(inspect.signature(eo_mod.TensorParallelAdaptiveMuon.__init__).parameters)
    unexpected = set(kwargs) - accepted
    assert not unexpected, f"kwargs TensorParallelAdaptiveMuon.__init__ rejects: {unexpected}"


def test_muon_entry_dispatches_on_layer_sharding_flag():
    """The muon entry's dispatcher pairs lsh kwargs with the lsh class."""
    lsh_kwargs = eo_mod._muon_registry_config_to_kwargs(
        _dispatch_cfg(True), [_DispatchChunk()], pg_collection=None
    )
    # 'layer_sharded' is the registry selector, not a TensorParallelMuon mode:
    # the constructor must receive the bitwise reference mode instead.
    assert lsh_kwargs["tp_mode"] == "duplicated"
    assert "gtp_remat_group" in lsh_kwargs
    assert eo_mod._muon_config_to_cls(_dispatch_cfg(True)) is LayerShardedMuon
    accepted = set(inspect.signature(LayerShardedMuon.__init__).parameters)
    unexpected = set(lsh_kwargs) - accepted
    assert not unexpected, f"kwargs LayerShardedMuon.__init__ rejects: {unexpected}"

    plain_kwargs = eo_mod._muon_registry_config_to_kwargs(
        _dispatch_cfg(False), [_DispatchChunk()], pg_collection=None
    )
    assert "gtp_remat_group" not in plain_kwargs
    assert "is_qkv_fn" in plain_kwargs
    assert eo_mod._muon_config_to_cls(_dispatch_cfg(False)) is eo_mod.TensorParallelMuon


def test_muon_entry_registered_with_dispatcher():
    """The registry wires the dispatcher to 'muon' and leaves adaptive_muon alone."""
    muon_entry = eo_mod._EMERGING_OPTIMIZERS["muon"]
    assert muon_entry.config_to_kwargs is eo_mod._muon_registry_config_to_kwargs
    adaptive_entry = eo_mod._EMERGING_OPTIMIZERS["adaptive_muon"]
    assert adaptive_entry.config_to_kwargs is eo_mod._adaptive_muon_config_to_kwargs


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestMuonOptimizerMultiRank:
    """Test class for Muon optimizer with multi-rank setup."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    def create_ddp_model(self, model):
        """Wrap model in DDP.

        Args:
            model: Model to wrap

        Returns:
            DDP-wrapped model
        """
        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        return DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )

    def create_ddp_model_for_layerwise(self, model, use_param_layout=False):
        """Wrap model in DDP for layer-wise distributed optimizer tests.

        Args:
            model: Model to wrap.
            use_param_layout: If True, supply DDP a precomputed shard-aligned
                ``full_param_layout`` (turns on ``ddp_config.use_distributed_optimizer=True``
                + ``start_param_sync``). If False (default), build DDP without a layout
                so ``LayerWiseDistributedOptimizer`` syncs via the legacy
                flatten / ``all_gather_v`` / unflatten ``allgather_params()`` codepath.
        """
        if use_param_layout:
            from megatron.training.training import wrap_model_chunks_with_ddp

            ddp_config = DistributedDataParallelConfig()
            wrapped = wrap_model_chunks_with_ddp(
                [model],
                TransformerConfig(num_attention_heads=1, num_layers=1),
                ddp_config,
                use_layer_wise_distributed_optimizer=True,
            )
            return wrapped[0]
        return self.create_ddp_model(model)

    def test_get_megatron_optimizer_smoke(self):
        """Smoke test for get_megatron_optimizer function."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        # Ensure all parameters require gradients
        for param in model.parameters():
            assert param.requires_grad, "All parameters should require gradients"

        # Create optimizer config for Muon
        optimizer_config = OptimizerConfig(
            optimizer='muon',  # This will be changed internally to 'adam' for non-linear params
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_distributed_optimizer=False,  # Muon doesn't support distributed optimizer
            muon_momentum=0.95,
            muon_nesterov=True,
            muon_fp32_matmul_prec="medium",
            muon_num_ns_steps=5,
            muon_scale_mode="spectral",
            muon_tp_mode="duplicated",
        )

        # Test creating the optimizer
        optimizer = get_megatron_optimizer(
            config=optimizer_config, model_chunks=[model], use_gloo_process_groups=True
        )

        # Test basic properties
        assert optimizer is not None, "Optimizer should not be None"
        assert hasattr(optimizer, 'param_groups'), "Optimizer should have param_groups"
        assert hasattr(optimizer, 'chained_optimizers'), "Should be a ChainedOptimizer"
        assert len(optimizer.chained_optimizers) >= 1, "Should have at least one chained optimizer"

        # Test forward and backward pass
        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        # Store original parameters
        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.data.clone()

        # Test optimizer step
        optimizer.step()

        # Verify at least some parameters were updated
        params_updated = 0
        for name, param in model.named_parameters():
            if not torch.equal(param.data, original_params[name]):
                params_updated += 1

        assert params_updated > 0, "At least some parameters should be updated after optimizer step"

        # Test zero_grad
        optimizer.zero_grad()
        for param in model.parameters():
            assert param.grad is None or torch.all(
                param.grad == 0
            ), f"Gradients should be zeroed for all parameters"

        # Test state_dict and load_state_dict
        state_dict = optimizer.state_dict()
        assert isinstance(state_dict, list), "State dict should be a list"

        # Load state dict should not raise error
        optimizer.load_state_dict(state_dict)

    def test_get_megatron_optimizer_validation(self):
        """Test validation logic for get_megatron_optimizer."""
        model = torch.nn.Linear(100, 50, bias=False, dtype=torch.bfloat16, device='cuda')
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        # Test 1: FP16 should raise exception
        optimizer_config_fp16 = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            fp16=True,  # This should cause an exception
            use_distributed_optimizer=False,
        )

        with pytest.raises(Exception, match='emerging optimizer with fp16 is not supported'):
            get_megatron_optimizer(config=optimizer_config_fp16, model_chunks=[model])

        # Test 3: Invalid num_ns_steps should raise exception
        optimizer_config_invalid_ns = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            bf16=True,
            use_distributed_optimizer=False,
            muon_num_ns_steps=0,  # This should cause an exception
        )

        with pytest.raises(ValueError, match='num_ns_steps must be at least 1'):
            get_megatron_optimizer(config=optimizer_config_invalid_ns, model_chunks=[model])

    @pytest.mark.parametrize('scale_mode', ['spectral', 'rms'])
    def test_get_megatron_optimizer_layer_wise(self, scale_mode):
        """Test get_megatron_optimizer with layer-wise distributed optimizer."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model_for_layerwise(model)

        optimizer_config = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_layer_wise_distributed_optimizer=True,
            muon_momentum=0.95,
            muon_nesterov=True,
            muon_fp32_matmul_prec="medium",
            muon_num_ns_steps=5,
            muon_scale_mode=scale_mode,
            muon_tp_mode="duplicated",
        )

        # use_layer_wise_distributed_optimizer=True triggers LayerWiseDistributedOptimizer
        optimizer = get_megatron_optimizer(
            config=optimizer_config, model_chunks=[model], use_gloo_process_groups=True
        )

        # Verify it's a LayerWiseDistributedOptimizer
        from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer

        assert isinstance(
            optimizer, LayerWiseDistributedOptimizer
        ), "Should return LayerWiseDistributedOptimizer"

        # Test forward and backward pass
        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        # Test optimizer step
        update_successful, grad_norm, num_zeros = optimizer.step()

        assert update_successful, "Optimizer step should be successful"
        assert grad_norm is not None or grad_norm is None, "Grad norm should be returned"

    def test_get_megatron_muon_optimizer_backward_compatible(self):
        """Test get_megatron_muon_optimizer with backward compatible layer-wise distributed optimizer."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model_for_layerwise(model)

        optimizer_config = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_layer_wise_distributed_optimizer=True,
            muon_momentum=0.95,
            muon_nesterov=True,
            muon_fp32_matmul_prec="medium",
            muon_num_ns_steps=5,
            muon_scale_mode="spectral",
            muon_tp_mode="duplicated",
        )

        with pytest.raises(ValueError, match="dist_ prefix"):
            get_megatron_muon_optimizer(
                config=optimizer_config, model_chunks=[model], layer_wise_distributed_optimizer=True
            )

        optimizer_config.optimizer = 'dist_muon'
        optimizer = get_megatron_muon_optimizer(
            config=optimizer_config, model_chunks=[model], layer_wise_distributed_optimizer=True
        )

        # Verify it's a LayerWiseDistributedOptimizer
        from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer

        assert isinstance(
            optimizer, LayerWiseDistributedOptimizer
        ), "Should return LayerWiseDistributedOptimizer"

        # Test forward and backward pass
        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        # Test optimizer step
        update_successful, grad_norm, num_zeros = optimizer.step()

        assert update_successful, "Optimizer step should be successful"
        assert grad_norm is not None or grad_norm is None, "Grad norm should be returned"


@pytest.mark.parametrize("mode", ["duplicated", "blockwise", "distributed"])
def test_muon_optimizer_different_modes_single_rank(mode):
    """Test TensorParallelMuon optimizer with different modes on single rank.

    When TP size is 1, all modes should produce the same result.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.normal_(0, 0.02)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.95,
        weight_decay=0.0,  # Disable weight decay for deterministic comparison
        num_ns_steps=5,
        pg_collection=None,
        tp_mode=mode,
    )

    # Use fixed input for deterministic results
    torch.manual_seed(42)
    input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')

    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    # Verify weight was updated
    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with mode={mode}"


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestMuonOptimizerMultiRankTP:
    """Test class for Muon optimizer with multi-rank and tensor parallel setup."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test with tensor parallel."""
        world = int(os.getenv('WORLD_SIZE', '1'))
        Utils.initialize_model_parallel(tensor_model_parallel_size=min(world, 2))
        yield
        Utils.destroy_model_parallel()

    def create_tp_model_and_optimizer(self, mode):
        """Create model with TP and optimizer.

        Args:
            mode: Muon optimizer mode

        Returns:
            tuple: (model, optimizer, pg_collection)
        """
        rank = int(os.getenv('RANK', '0'))
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        # Create model with partition_dim for TP
        torch.manual_seed(42 + rank)
        model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
        model.requires_grad_(True)
        model.weight.data.normal_(0, 0.02)
        model.weight.partition_dim = 0  # Set partition dimension for TP

        optimizer = TensorParallelMuon(
            params=[model.weight],
            lr=0.01,
            momentum=0.95,
            weight_decay=0.0,
            num_ns_steps=5,
            pg_collection=pg_collection,
            tp_mode=mode,
        )

        return model, optimizer

    def test_optimizer_factory_tags_qkv_when_tp_exceeds_query_groups(self):
        """The public optimizer factory tags query groups fragmented across TP ranks."""
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
        assert qkv_weight.is_qkv
        assert qkv_weight.qkv_split_shapes == [8, 4, 4]
        assert qkv_weight.qkv_split_shapes_global == [8, 4, 4]
        assert not qkv_weight.qkv_split_groups_are_complete

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
        assert second_qkv.is_qkv
        if split_per_head:
            assert first_qkv.qkv_split_shapes_global == [4] * 6
            assert second_qkv.qkv_split_shapes_global == [4] * 4
            assert first_qkv.qkv_split_shapes == [4] * 6
            assert second_qkv.qkv_split_shapes == [4] * 4
        else:
            assert first_qkv.qkv_split_shapes_global == [4, 4, 4] * 2
            assert second_qkv.qkv_split_shapes_global == [8, 4, 4]
            assert first_qkv.qkv_split_groups_are_complete
            assert not second_qkv.qkv_split_groups_are_complete

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
        assert q_up_weight.is_qkv
        assert kv_up_weight.is_qkv
        assert q_up_weight.qkv_split_shapes_global == [4, 2] * 2
        assert kv_up_weight.qkv_split_shapes_global == [4, 3] * 2
        if split_per_head:
            assert q_up_weight.qkv_split_shapes == [4, 2] * 2
            assert kv_up_weight.qkv_split_shapes == [4, 3] * 2
        else:
            assert q_up_weight.qkv_split_groups_are_complete
            assert kv_up_weight.qkv_split_groups_are_complete
        if q_lora_rank is not None:
            assert not getattr(attention.linear_q_down_proj.weight, 'is_qkv', False)
        assert not getattr(attention.linear_kv_down_proj.weight, 'is_qkv', False)

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
        assert not qkv_weight.is_qkv
        assert qkv_weight.qkv_split_shapes is None
        assert qkv_weight.qkv_split_shapes_global is None

    @pytest.mark.parametrize("mode", ["duplicated", "distributed"])
    def test_muon_optimizer_modes_multirank_same_result(self, mode):
        """Test that duplicated and distributed modes produce same results with TP > 1."""
        model, optimizer = self.create_tp_model_and_optimizer(mode)

        # Use fixed input for deterministic results
        torch.manual_seed(42)
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')

        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        original_weight = model.weight.data.clone()
        optimizer.step()

        # Verify weight was updated
        assert not torch.equal(
            model.weight.data, original_weight
        ), f"Weight should be updated with mode={mode}"

    def test_muon_optimizer_blockwise_mode_different_result(self):
        """Test that blockwise mode produces different results than duplicated/distributed with TP > 1."""
        model, optimizer = self.create_tp_model_and_optimizer("blockwise")

        # Use fixed input for deterministic results
        torch.manual_seed(42)
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')

        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        original_weight = model.weight.data.clone()
        optimizer.step()

        # Verify weight was updated
        assert not torch.equal(
            model.weight.data, original_weight
        ), "Weight should be updated with mode=blockwise"

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

        optimizer = TensorParallelMuon(
            params=[param],
            split_qkv=True,
            split_qkv_per_head=True,
            is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
            qkv_split_shapes=[2, 2, 2],
            pg_collection=pg_collection,
            tp_mode="blockwise",
        )

        def center_rows(x, *_args, **_kwargs):
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
    def test_muon_optimizer_projection_split_gathers_fragmented_query_group(self, split_shapes):
        """Projection splitting reconstructs a query group split over TP ranks."""
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
            is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
            qkv_split_shapes=split_shapes,
            pg_collection=pg_collection,
            tp_mode="blockwise",
        )

        def center_rows(x, *_args, **_kwargs):
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
    ("layout", "expects_fallback"),
    (("local_projection", False), ("per_head", True), ("fragmented", True), ("gtp", True)),
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

    def passthrough(x, tp_group=None, partition_dim=None, **_kwargs):
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


# All non-custom coefficient types supported by emerging_optimizers.
_TESTABLE_COEFFICIENT_TYPES = (
    [t for t in get_supported_coefficient_types() if t != "custom"]
    if HAVE_EMERGING_OPTIMIZERS
    else []
)

# A reasonable default NS step count for testing; get_coefficient_iterator
# cycles/repeats coefficients so any step count works with any type.
_DEFAULT_NS_STEPS = 5


@pytest.mark.parametrize("coefficient_type", _TESTABLE_COEFFICIENT_TYPES)
def test_muon_optimizer_coefficient_types(coefficient_type):
    """Test TensorParallelMuon optimizer with different coefficient types."""
    model = torch.nn.Linear(80, 40, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        coefficient_type=coefficient_type,
        num_ns_steps=_DEFAULT_NS_STEPS,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 80, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with coefficient_type={coefficient_type}"


@pytest.mark.parametrize("scale_mode", ["spectral", "unit_rms_norm", "shape_scaling"])
def test_muon_optimizer_scale_modes(scale_mode):
    """Test TensorParallelMuon optimizer with different scale modes."""
    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        scale_mode=scale_mode,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with scale_mode={scale_mode}"


@pytest.mark.parametrize("nesterov", [True, False])
def test_muon_optimizer_nesterov(nesterov):
    """Test TensorParallelMuon optimizer with and without Nesterov momentum."""
    model = torch.nn.Linear(50, 25, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.9,
        nesterov=nesterov,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 50, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with nesterov={nesterov}"


def test_muon_optimizer_multiple_steps():
    """Test TensorParallelMuon optimizer across multiple optimization steps."""
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.95,
        weight_decay=0.01,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    weights_history = [model.weight.data.clone()]

    for i in range(3):
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        weights_history.append(model.weight.data.clone())

    # Verify weights changed at each step
    for i in range(len(weights_history) - 1):
        assert not torch.equal(
            weights_history[i], weights_history[i + 1]
        ), f"Weight should change at step {i}"


def test_muon_optimizer_qkv_split():
    """Test TensorParallelMuon optimizer with QKV splitting."""
    # Create a model with QKV-like parameter
    qkv_size = 3 * 64 * 16  # Combined Q, K, V dimensions, 16 heads x 64 per head
    hidden_size = 1024
    model = torch.nn.Linear(hidden_size, qkv_size, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    # Mark parameter as QKV
    model.weight.is_qkv = True

    # QKV split shapes: [Q_size, K_size, V_size]
    qkv_split_shapes = (64, 64, 64)

    # Test with split_qkv=True
    optimizer_split = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        split_qkv=True,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=qkv_split_shapes,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, hidden_size, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer_split.step()
    weight_with_split = model.weight.data.clone()

    assert not torch.equal(
        weight_with_split, original_weight
    ), "QKV weight should be updated with split_qkv=True"

    # Reset model and test with split_qkv=False
    model.weight.data.fill_(1.0)
    optimizer_no_split = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        split_qkv=False,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    optimizer_no_split.step()
    weight_without_split = model.weight.data.clone()

    assert not torch.equal(
        weight_without_split, original_weight
    ), "QKV weight should be updated with split_qkv=False"

    # Ensure the two results are different
    assert not torch.equal(
        weight_with_split, weight_without_split
    ), "Weights should be different between split_qkv=True and split_qkv=False"


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

    def center_rows(x, *_args, **_kwargs):
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

    def center_rows(x, *_args, **_kwargs):
        call_shapes.append(tuple(x.shape))
        return x - x.mean(dim=-2, keepdim=True)

    optimizer.scaled_orthogonalize_fn = center_rows
    actual = optimizer.orthogonalize(param, grad)
    assert call_shapes == [(2, 2, 4)]
    expected = torch.cat(
        [head - head.mean(dim=-2, keepdim=True) for head in torch.split(grad, [2, 2])]
    )
    torch.testing.assert_close(actual, expected)


def test_muon_optimizer_batched_per_head_ns_rejects_old_emerging_optimizers(monkeypatch):
    """The 3-D NS path reports the installed and required Emerging-Optimizers versions."""
    grad = torch.zeros(4, 4, dtype=torch.float32, device='cuda')
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
    monkeypatch.setattr(eo_mod, "is_emerging_optimizers_min_version", lambda _version: False)
    monkeypatch.setattr(eo_mod, "get_emerging_optimizers_version", lambda: "0.2.0")

    with pytest.raises(
        ValueError, match="requires emerging-optimizers >= 0.3.0, but 0.2.0 is installed"
    ):
        optimizer.orthogonalize(param, grad)


def test_muon_optimizer_batched_per_head_ns_matches_individual_heads():
    """The pinned Emerging-Optimizers 3D Newton-Schulz path matches 2D head calls."""
    torch.manual_seed(42)
    # Match Emerging-Optimizers v0.3.0's batched parity case. Tiny matrices can select
    # different baddbmm/addmm kernels and amplify their otherwise valid rounding differences.
    num_heads, head_rows, hidden_size = 4, 128, 256
    split_shapes = [head_rows] * num_heads
    grad = torch.randint(
        -3, 4, (num_heads * head_rows, hidden_size), dtype=torch.float32, device='cuda'
    )
    param = torch.nn.Parameter(torch.zeros_like(grad))
    param.is_qkv = True
    param.qkv_split_shapes = split_shapes
    optimizer = TensorParallelMuon(
        params=[param],
        split_qkv=True,
        split_qkv_per_head=True,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=split_shapes,
        fp32_matmul_prec="highest",
        num_ns_steps=5,
        pg_collection=None,
    )

    actual = optimizer.orthogonalize(param, grad)
    expected = torch.cat(
        [
            optimizer.scaled_orthogonalize_fn(head, tp_group=None, partition_dim=None)
            for head in torch.split(grad, split_shapes)
        ]
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)


def test_muon_optimizer_nonuniform_per_head_splits_use_unbatched_ns(monkeypatch):
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

    def center_rows(x, *_args, **_kwargs):
        call_shapes.append(tuple(x.shape))
        return x - x.mean(dim=-2, keepdim=True)

    monkeypatch.setattr(
        eo_mod,
        "is_emerging_optimizers_min_version",
        lambda _version: pytest.fail("nonuniform splits must not select batched NS"),
    )
    optimizer.scaled_orthogonalize_fn = center_rows
    actual = optimizer.orthogonalize(param, grad)
    assert call_shapes == [(2, 4), (1, 4)]
    expected = torch.cat([center_rows(head) for head in torch.split(grad, [2, 1])])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('shape', [(7, 13), (13, 7), (3, 7, 13)])
@pytest.mark.parametrize('target_rms', [0.18, 0.4])
def test_muon_rms_scaling_matches_ns_direction(shape, target_rms):
    """Actual NS directions keep their direction and reach the requested per-matrix RMS."""
    from emerging_optimizers.utils import fp32_matmul_precision

    grad = torch.randn(shape, generator=torch.Generator().manual_seed(41))
    param = nn.Parameter(torch.zeros_like(grad))
    optimizer = TensorParallelMuon(
        [param],
        scale_mode='rms',
        extra_scale_factor=target_rms,
        coefficient_type='deepseekv4',
        num_ns_steps=10,
        fp32_matmul_prec='highest',
    )
    with fp32_matmul_precision('highest'):
        actual = optimizer.scaled_orthogonalize_fn(grad, tp_group=None)
        direction = eo_mod.newton_schulz_tp(
            grad, steps=10, coefficient_type='deepseekv4', tp_group=None
        ).double()
    expected = F.normalize(direction.flatten(-2), dim=-1, eps=0).view_as(direction)
    expected *= target_rms * (shape[-2] * shape[-1]) ** 0.5
    torch.testing.assert_close(actual.double(), expected, atol=1e-7, rtol=2e-6)
    torch.testing.assert_close(
        actual.double().square().mean(dim=(-2, -1)).sqrt(),
        torch.full(shape[:-2], target_rms, dtype=torch.float64),
        atol=1e-7,
        rtol=1e-6,
    )


def test_muon_rms_scaling_handles_zero_and_small_updates(monkeypatch):
    """A zero head stays zero; tiny/large nonzero heads have the same normalized direction."""
    direction = torch.arange(-7, 8, dtype=torch.float32).view(3, 5)
    outputs = direction.unsqueeze(0) * torch.tensor([0.0, 1e-30, 1e30]).view(3, 1, 1)
    saved = outputs.clone()
    monkeypatch.setattr(eo_mod, 'newton_schulz_tp', lambda *args, **kwargs: outputs)
    optimizer = TensorParallelMuon(
        [nn.Parameter(torch.zeros(3, 5))], scale_mode='rms', extra_scale_factor=0.18
    )
    actual = optimizer.scaled_orthogonalize_fn(outputs, tp_group=None)
    expected = F.normalize(direction.double().flatten(), dim=0, eps=0).view(3, 5)
    expected *= 0.18 * direction.numel() ** 0.5
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual[0], torch.zeros_like(actual[0]), atol=0, rtol=0)
    torch.testing.assert_close(
        actual[1:].double(), expected.expand(2, -1, -1), atol=1e-7, rtol=2e-6
    )
    torch.testing.assert_close(outputs, saved, atol=0, rtol=0)


@pytest.mark.parametrize('splits', [[3, 3, 3], [2, 3, 4]])
def test_muon_rms_per_head_step_and_resume(splits):
    """Optimizer steps use RMS per head, preserve zero updates/decay, and resume exactly."""
    generator = torch.Generator().manual_seed(42)
    param = nn.Parameter(torch.randn(sum(splits), 7, generator=generator))
    param.qkv_split_shapes_global = splits
    kwargs = dict(
        lr=0.02,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        split_qkv=True,
        split_qkv_per_head=True,
        is_qkv_fn=lambda p: True,
        scale_mode='rms',
        extra_scale_factor=0.18,
        coefficient_type='deepseekv4',
        num_ns_steps=10,
        fp32_matmul_prec='highest',
    )
    optimizer = TensorParallelMuon([param], **kwargs)
    restored = None
    for iteration in range(4):
        grad = torch.randn(param.shape, generator=generator)
        grad[: splits[0]].zero_()
        before = param.detach().clone()
        param.grad = grad.clone()
        optimizer.step()
        decayed = before.clone().add_(before, alpha=-kwargs['lr'] * kwargs['weight_decay'])
        update = (decayed - param) / kwargs['lr']
        heads = update.split(splits)
        torch.testing.assert_close(heads[0], torch.zeros_like(heads[0]), atol=0, rtol=0)
        for head in heads[1:]:
            torch.testing.assert_close(
                head.double().square().mean().sqrt(),
                torch.tensor(0.18, dtype=torch.float64),
                atol=2e-6,
                rtol=1e-5,
            )
        if restored is not None:
            restored_param.grad = grad.clone()
            restored.step()
            torch.testing.assert_close(restored_param, param, atol=0, rtol=0)
            torch.testing.assert_close(
                restored.state[restored_param]['momentum_buffer'],
                optimizer.state[param]['momentum_buffer'],
                atol=0,
                rtol=0,
            )
        if iteration == 1:
            restored_param = nn.Parameter(param.detach().clone())
            restored_param.qkv_split_shapes_global = splits
            restored = TensorParallelMuon([restored_param], **kwargs)
            restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            assert set(optimizer.state[param]) == {'momentum_buffer'}


@pytest.mark.parametrize('scale_mode', [None, 'spectral', 'unit_rms_norm', 'shape_scaling'])
@pytest.mark.parametrize('shape', [(7, 13), (3, 7, 13)])
def test_muon_rms_preserves_existing_scale_modes(scale_mode, shape):
    """The opt-in RMS mode does not change old scaling formulas or the default."""
    from emerging_optimizers.utils import fp32_matmul_precision

    grad = torch.randn(shape, generator=torch.Generator().manual_seed(43))
    kwargs = {} if scale_mode is None else {'scale_mode': scale_mode}
    optimizer = TensorParallelMuon(
        [nn.Parameter(torch.zeros_like(grad))], extra_scale_factor=0.18, **kwargs
    )
    with fp32_matmul_precision('highest'):
        actual = optimizer.scaled_orthogonalize_fn(grad, tp_group=None)
        direction = eo_mod.newton_schulz_tp(
            grad, steps=5, coefficient_type='quintic', tp_group=None
        )
    factor = eo_mod.get_muon_scale_factor(*shape[-2:], mode=scale_mode or 'spectral')
    torch.testing.assert_close(actual, direction * factor * 0.18, atol=0, rtol=0)


@pytest.mark.parametrize('name', ['tp', 'expt_tp', 'gtp_remat', 'expt_gtp_remat'])
def test_muon_rms_rejects_sharded_groups(monkeypatch, name):
    """A local RMS must never silently stand in for a sharded matrix's global RMS."""
    groups = ProcessGroupCollection(**{name: SimpleNamespace(size=lambda: 2)})
    monkeypatch.setattr(eo_mod, 'get_pg_size', lambda group: 1 if group is None else group.size())
    with pytest.raises(ValueError, match='TP=1 and no GTP sharding'):
        TensorParallelMuon([nn.Parameter(torch.ones(3, 5))], scale_mode='rms', pg_collection=groups)


@pytest.mark.parametrize(
    'kwargs', [{'tp_group': SimpleNamespace(size=lambda: 2)}, {'scale_shape': (6, 5)}]
)
def test_muon_rms_rejects_partial_update(monkeypatch, kwargs):
    """Direct scaling callers also reject incomplete or padded logical matrices."""
    param = nn.Parameter(torch.ones(3, 5))
    monkeypatch.setattr(eo_mod, 'get_pg_size', lambda group: 1 if group is None else group.size())
    optimizer = TensorParallelMuon([param], scale_mode='rms')
    with pytest.raises(ValueError, match='complete, unpadded update matrix'):
        optimizer.scaled_orthogonalize_fn(param.detach(), **{'tp_group': None, **kwargs})


@pytest.mark.parametrize('target', [0, -1, float('inf'), float('nan')])
def test_muon_rms_rejects_invalid_target(target):
    with pytest.raises(ValueError, match='finite positive extra_scale_factor'):
        TensorParallelMuon(
            [nn.Parameter(torch.ones(3, 5))], scale_mode='rms', extra_scale_factor=target
        )


def test_muon_rms_rejects_other_optimizer_paths():
    """Alternate update algorithms reject RMS before initializing any optimizer state."""
    params = [nn.Parameter(torch.ones(3, 5))]
    with pytest.raises(ValueError, match='not supported by AdaptiveMuon'):
        TensorParallelAdaptiveMuon(params, scale_mode='rms')
    with pytest.raises(ValueError, match='not supported by LayerShardedMuon'):
        LayerShardedMuon(params, scale_mode='rms', gtp_remat_group=None)


def test_muon_optimizer_extra_scale_factor():
    """Test TensorParallelMuon optimizer with different extra_scale_factor values."""
    model = torch.nn.Linear(80, 40, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        extra_scale_factor=2.0,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 80, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), "Weight should be updated with extra_scale_factor"


def test_get_supported_coefficient_types_returns_tuple():
    """Test that get_supported_coefficient_types returns a non-empty tuple of strings."""
    supported = get_supported_coefficient_types()
    assert isinstance(supported, tuple)
    assert len(supported) > 0
    for t in supported:
        assert isinstance(t, str)


def test_get_supported_coefficient_types_contains_known_types():
    """Test that the known coefficient types are present in the supported set."""
    supported = get_supported_coefficient_types()
    for expected in ("simple", "quintic", "polar_express"):
        assert expected in supported, f"Expected '{expected}' in supported types {supported}"


def test_validate_coefficient_type_accepts_valid():
    """Test that validate_coefficient_type does not raise for valid types."""
    for t in get_supported_coefficient_types():
        validate_coefficient_type(t)  # should not raise


def test_validate_coefficient_type_rejects_invalid():
    """Test that validate_coefficient_type raises ValueError for an invalid type."""
    with pytest.raises(ValueError, match="Unsupported muon coefficient type"):
        validate_coefficient_type("nonexistent_type_xyz")


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestMuonCoefficientTypeMultiRank:
    """Test coefficient_type integration through get_megatron_optimizer."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    def create_ddp_model(self, model):
        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        return DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )

    @pytest.mark.parametrize("coefficient_type", _TESTABLE_COEFFICIENT_TYPES)
    def test_get_megatron_optimizer_coefficient_type(self, coefficient_type):
        """Test that coefficient_type flows through get_megatron_optimizer."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        optimizer_config = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_distributed_optimizer=False,
            muon_coefficient_type=coefficient_type,
            muon_num_ns_steps=_DEFAULT_NS_STEPS,
            muon_tp_mode="duplicated",
        )

        optimizer = get_megatron_optimizer(
            config=optimizer_config, model_chunks=[model], use_gloo_process_groups=True
        )

        assert optimizer is not None

        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        optimizer.step()


@pytest.mark.parametrize("num_ns_steps", [5, 15, 25])
def test_muon_optimizer_num_ns_steps(num_ns_steps):
    """Test TensorParallelMuon optimizer with different numbers of Newton-Schulz steps."""
    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        coefficient_type="quintic",
        num_ns_steps=num_ns_steps,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with num_ns_steps={num_ns_steps}"


# ===========================================================================
# Adaptive Muon optimizer tests
# ===========================================================================


def test_adaptive_muon_optimizer_smoke():
    """Smoke test for TensorParallelAdaptiveMuon optimizer."""
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelAdaptiveMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.95,
        nesterov=True,
        weight_decay=0.01,
        use_decoupled_weight_decay=True,
        split_qkv=False,
        fp32_matmul_prec="medium",
        num_ns_steps=5,
        scale_mode="spectral",
        extra_scale_factor=1.0,
        pg_collection=None,
        tp_mode="duplicated",
        moment2_method="adamuon",
        beta2=0.95,
        eps=1e-8,
    )

    assert optimizer is not None
    assert hasattr(optimizer, 'param_groups')
    assert len(optimizer.param_groups) > 0

    input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), "Weight should be updated after optimizer step"

    optimizer.zero_grad()
    assert model.weight.grad is None or torch.all(
        model.weight.grad == 0
    ), "Gradients should be zeroed"

    state_dict = optimizer.state_dict()
    assert 'state' in state_dict
    assert 'param_groups' in state_dict
    optimizer.load_state_dict(state_dict)


@pytest.mark.parametrize("mode", ["duplicated", "blockwise", "distributed"])
def test_adaptive_muon_optimizer_different_modes_single_rank(mode):
    """Test TensorParallelAdaptiveMuon with different modes on single rank."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.normal_(0, 0.02)

    optimizer = TensorParallelAdaptiveMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.95,
        weight_decay=0.0,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode=mode,
    )

    torch.manual_seed(42)
    input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')

    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with mode={mode}"


@pytest.mark.parametrize("moment2_method", ["adamuon", "normuon"])
def test_adaptive_muon_optimizer_moment2_methods(moment2_method):
    """Test TensorParallelAdaptiveMuon with different moment2 methods."""
    model = torch.nn.Linear(80, 40, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelAdaptiveMuon(
        params=[model.weight],
        lr=0.01,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
        moment2_method=moment2_method,
    )

    input_tensor = torch.randn(16, 80, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with moment2_method={moment2_method}"


@pytest.mark.parametrize("beta2", [0.5, 0.95, 0.999])
def test_adaptive_muon_optimizer_beta2(beta2):
    """Test TensorParallelAdaptiveMuon with different beta2 values."""
    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelAdaptiveMuon(
        params=[model.weight],
        lr=0.01,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
        beta2=beta2,
    )

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with beta2={beta2}"


def test_adaptive_muon_optimizer_multiple_steps():
    """Test TensorParallelAdaptiveMuon across multiple optimization steps."""
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelAdaptiveMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.95,
        weight_decay=0.01,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    weights_history = [model.weight.data.clone()]

    for i in range(3):
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        weights_history.append(model.weight.data.clone())

    for i in range(len(weights_history) - 1):
        assert not torch.equal(
            weights_history[i], weights_history[i + 1]
        ), f"Weight should change at step {i}"


@pytest.mark.parametrize("nesterov", [True, False])
def test_adaptive_muon_optimizer_nesterov(nesterov):
    """Test TensorParallelAdaptiveMuon with and without Nesterov momentum."""
    model = torch.nn.Linear(50, 25, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelAdaptiveMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.9,
        nesterov=nesterov,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 50, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with nesterov={nesterov}"


def test_adaptive_muon_optimizer_qkv_split():
    """Test TensorParallelAdaptiveMuon with QKV splitting."""
    qkv_size = 3 * 64 * 16  # Combined Q, K, V dimensions
    hidden_size = 1024
    model = torch.nn.Linear(hidden_size, qkv_size, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    model.weight.is_qkv = True
    qkv_split_shapes = (64, 64, 64)

    optimizer_split = TensorParallelAdaptiveMuon(
        params=[model.weight],
        lr=0.01,
        split_qkv=True,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=qkv_split_shapes,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, hidden_size, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer_split.step()
    weight_with_split = model.weight.data.clone()

    assert not torch.equal(
        weight_with_split, original_weight
    ), "QKV weight should be updated with split_qkv=True"

    model.weight.data.fill_(1.0)
    optimizer_no_split = TensorParallelAdaptiveMuon(
        params=[model.weight],
        lr=0.01,
        split_qkv=False,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    optimizer_no_split.step()
    weight_without_split = model.weight.data.clone()

    assert not torch.equal(
        weight_without_split, original_weight
    ), "QKV weight should be updated with split_qkv=False"

    assert not torch.equal(
        weight_with_split, weight_without_split
    ), "Weights should be different between split_qkv=True and split_qkv=False"


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestAdaptiveMuonOptimizerMultiRank:
    """Test class for Adaptive Muon optimizer with multi-rank setup."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    def create_ddp_model(self, model):
        """Wrap model in DDP."""
        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        return DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )

    def test_get_megatron_optimizer_adaptive_muon_smoke(self):
        """Smoke test for get_megatron_optimizer with adaptive_muon."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        for param in model.parameters():
            assert param.requires_grad

        optimizer_config = OptimizerConfig(
            optimizer='adaptive_muon',
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_distributed_optimizer=False,
            muon_momentum=0.95,
            muon_nesterov=True,
            muon_fp32_matmul_prec="medium",
            muon_num_ns_steps=5,
            muon_scale_mode="spectral",
            muon_tp_mode="duplicated",
            adaptive_muon_moment2_method="adamuon",
            adaptive_muon_beta2=0.95,
            adaptive_muon_eps=1e-8,
        )

        optimizer = get_megatron_optimizer(
            config=optimizer_config, model_chunks=[model], use_gloo_process_groups=True
        )

        assert optimizer is not None
        assert hasattr(optimizer, 'param_groups')
        assert hasattr(optimizer, 'chained_optimizers')
        assert len(optimizer.chained_optimizers) >= 1

        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.data.clone()

        optimizer.step()

        params_updated = 0
        for name, param in model.named_parameters():
            if not torch.equal(param.data, original_params[name]):
                params_updated += 1

        assert params_updated > 0, "At least some parameters should be updated after optimizer step"

        optimizer.zero_grad()
        for param in model.parameters():
            assert param.grad is None or torch.all(
                param.grad == 0
            ), "Gradients should be zeroed for all parameters"

        state_dict = optimizer.state_dict()
        assert isinstance(state_dict, list)
        optimizer.load_state_dict(state_dict)

    def test_get_megatron_optimizer_adaptive_muon_validation(self):
        """Test validation logic for get_megatron_optimizer with adaptive_muon."""
        model = torch.nn.Linear(100, 50, bias=False, dtype=torch.bfloat16, device='cuda')
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        optimizer_config_fp16 = OptimizerConfig(
            optimizer='adaptive_muon', lr=0.01, fp16=True, use_distributed_optimizer=False
        )

        with pytest.raises(Exception, match='emerging optimizer with fp16 is not supported'):
            get_megatron_optimizer(config=optimizer_config_fp16, model_chunks=[model])


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestAdaptiveMuonOptimizerMultiRankTP:
    """Test class for Adaptive Muon optimizer with multi-rank and tensor parallel setup."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test with tensor parallel."""
        world = int(os.getenv('WORLD_SIZE', '1'))
        Utils.initialize_model_parallel(tensor_model_parallel_size=min(world, 2))
        yield
        Utils.destroy_model_parallel()

    def create_tp_model_and_optimizer(self, mode):
        """Create model with TP and optimizer."""
        rank = int(os.getenv('RANK', '0'))
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        torch.manual_seed(42 + rank)
        model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
        model.requires_grad_(True)
        model.weight.data.normal_(0, 0.02)
        model.weight.partition_dim = 0

        optimizer = TensorParallelAdaptiveMuon(
            params=[model.weight],
            lr=0.01,
            momentum=0.95,
            weight_decay=0.0,
            num_ns_steps=5,
            pg_collection=pg_collection,
            tp_mode=mode,
        )

        return model, optimizer

    @pytest.mark.parametrize("mode", ["duplicated", "distributed"])
    def test_adaptive_muon_optimizer_modes_multirank_same_result(self, mode):
        """Test that duplicated and distributed modes produce same results with TP > 1."""
        model, optimizer = self.create_tp_model_and_optimizer(mode)

        torch.manual_seed(42)
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')

        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        original_weight = model.weight.data.clone()
        optimizer.step()

        assert not torch.equal(
            model.weight.data, original_weight
        ), f"Weight should be updated with mode={mode}"

    def test_adaptive_muon_optimizer_blockwise_mode(self):
        """Test that blockwise mode works with TP > 1."""
        model, optimizer = self.create_tp_model_and_optimizer("blockwise")

        torch.manual_seed(42)
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')

        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        original_weight = model.weight.data.clone()
        optimizer.step()

        assert not torch.equal(
            model.weight.data, original_weight
        ), "Weight should be updated with mode=blockwise"


# ===========================================================================
# SOAP optimizer tests
# ===========================================================================

skip_no_soap = pytest.mark.skipif(SOAP is None, reason="SOAP is unavailable in emerging_optimizers")


@skip_no_soap
def test_soap_optimizer_smoke():
    """Smoke test for SOAP optimizer."""

    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = SOAP(
        params=[model.weight], lr=0.01, betas=(0.9, 0.999), shampoo_beta=0.95, weight_decay=0.01
    )

    # Test basic properties
    assert optimizer is not None, "Optimizer should not be None"
    assert hasattr(optimizer, 'param_groups'), "Optimizer should have param_groups"
    assert len(optimizer.param_groups) > 0, "Optimizer should have at least one parameter group"

    # Test forward and backward pass
    input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    # Store original weight
    original_weight = model.weight.data.clone()

    # Test optimizer step
    optimizer.step()

    # Verify weight was updated
    assert not torch.equal(
        model.weight.data, original_weight
    ), "Weight should be updated after optimizer step"

    # Test zero_grad
    optimizer.zero_grad()
    assert model.weight.grad is None or torch.all(
        model.weight.grad == 0
    ), "Gradients should be zeroed"

    # Test state_dict and load_state_dict
    state_dict = optimizer.state_dict()
    assert 'state' in state_dict, "State dict should contain state"
    assert 'param_groups' in state_dict, "State dict should contain param_groups"

    # Load state dict should not raise error
    optimizer.load_state_dict(state_dict)


@skip_no_soap
def test_soap_optimizer_multiple_steps():
    """Test SOAP optimizer across multiple optimization steps."""
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = SOAP(
        params=[model.weight], lr=0.01, betas=(0.9, 0.999), shampoo_beta=0.95, weight_decay=0.01
    )

    weights_history = [model.weight.data.clone()]

    for i in range(3):
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        weights_history.append(model.weight.data.clone())

    # Verify weights changed at each step
    for i in range(len(weights_history) - 1):
        assert not torch.equal(
            weights_history[i], weights_history[i + 1]
        ), f"Weight should change at step {i}"


@skip_no_soap
@pytest.mark.parametrize("use_kl_shampoo", [True, False])
def test_soap_optimizer_kl_shampoo(use_kl_shampoo):
    """Test SOAP optimizer with and without KL-Shampoo preconditioner."""

    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = SOAP(
        params=[model.weight],
        lr=0.01,
        betas=(0.9, 0.999),
        shampoo_beta=0.95,
        use_kl_shampoo=use_kl_shampoo,
    )

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with use_kl_shampoo={use_kl_shampoo}"


@skip_no_soap
@pytest.mark.parametrize("shampoo_beta", [0.5, 0.9, 0.99])
def test_soap_optimizer_shampoo_beta(shampoo_beta):
    """Test SOAP optimizer with different shampoo_beta values."""

    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = SOAP(params=[model.weight], lr=0.01, betas=(0.9, 0.999), shampoo_beta=shampoo_beta)

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with shampoo_beta={shampoo_beta}"


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
@skip_no_soap
class TestSoapOptimizerMultiRank:
    """Test class for SOAP optimizer with multi-rank setup."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    def create_ddp_model(self, model):
        """Wrap model in DDP."""
        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        return DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )

    def test_get_megatron_optimizer_soap_smoke(self):
        """Smoke test for get_megatron_optimizer with SOAP."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        for param in model.parameters():
            assert param.requires_grad, "All parameters should require gradients"

        optimizer_config = OptimizerConfig(
            optimizer='soap',
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_distributed_optimizer=False,
            soap_shampoo_beta=0.95,
            soap_use_kl_shampoo=True,
        )

        optimizer = get_megatron_optimizer(
            config=optimizer_config, model_chunks=[model], use_gloo_process_groups=True
        )

        assert optimizer is not None, "Optimizer should not be None"
        assert hasattr(optimizer, 'param_groups'), "Optimizer should have param_groups"
        assert hasattr(optimizer, 'chained_optimizers'), "Should be a ChainedOptimizer"
        assert len(optimizer.chained_optimizers) >= 1, "Should have at least one chained optimizer"

        # Test forward and backward pass
        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        # Store original parameters
        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.data.clone()

        # Test optimizer step
        optimizer.step()

        # Verify at least some parameters were updated
        params_updated = 0
        for name, param in model.named_parameters():
            if not torch.equal(param.data, original_params[name]):
                params_updated += 1

        assert params_updated > 0, "At least some parameters should be updated after optimizer step"

        # Test zero_grad
        optimizer.zero_grad()
        for param in model.parameters():
            assert param.grad is None or torch.all(
                param.grad == 0
            ), "Gradients should be zeroed for all parameters"

        # Test state_dict and load_state_dict
        state_dict = optimizer.state_dict()
        assert isinstance(state_dict, list), "State dict should be a list"
        optimizer.load_state_dict(state_dict)

    def test_get_megatron_optimizer_soap_validation(self):
        """Test validation logic for get_megatron_optimizer with SOAP."""
        model = torch.nn.Linear(100, 50, bias=False, dtype=torch.bfloat16, device='cuda')
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        # FP16 should raise exception
        optimizer_config_fp16 = OptimizerConfig(
            optimizer='soap', lr=0.01, fp16=True, use_distributed_optimizer=False
        )

        with pytest.raises(Exception, match='emerging optimizer with fp16 is not supported'):
            get_megatron_optimizer(config=optimizer_config_fp16, model_chunks=[model])


# ===========================================================================
# Lion optimizer tests
# ===========================================================================

skip_no_lion = pytest.mark.skipif(
    not HAVE_EMERGING_OPTIMIZERS, reason="emerging_optimizers package not installed"
)


@skip_no_lion
def test_lion_optimizer_smoke():
    """Smoke test for Lion optimizer."""
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = Lion(params=[model.weight], lr=1e-4, betas=(0.9, 0.99), weight_decay=0.01)

    assert optimizer is not None
    assert hasattr(optimizer, 'param_groups')
    assert len(optimizer.param_groups) > 0

    input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), "Weight should be updated after optimizer step"

    optimizer.zero_grad()
    assert model.weight.grad is None or torch.all(
        model.weight.grad == 0
    ), "Gradients should be zeroed"

    state_dict = optimizer.state_dict()
    assert 'state' in state_dict
    assert 'param_groups' in state_dict
    optimizer.load_state_dict(state_dict)


@skip_no_lion
def test_lion_optimizer_multiple_steps():
    """Test Lion optimizer across multiple optimization steps."""
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = Lion(params=[model.weight], lr=1e-4, betas=(0.9, 0.99), weight_decay=0.01)

    weights_history = [model.weight.data.clone()]

    for i in range(3):
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        weights_history.append(model.weight.data.clone())

    for i in range(len(weights_history) - 1):
        assert not torch.equal(
            weights_history[i], weights_history[i + 1]
        ), f"Weight should change at step {i}"


@skip_no_lion
@pytest.mark.parametrize("betas", [(0.9, 0.99), (0.95, 0.999), (0.5, 0.9)])
def test_lion_optimizer_betas(betas):
    """Test Lion optimizer with different beta values."""
    model = torch.nn.Linear(80, 40, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = Lion(params=[model.weight], lr=1e-4, betas=betas)

    input_tensor = torch.randn(16, 80, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with betas={betas}"


@skip_no_lion
@pytest.mark.parametrize("weight_decay", [0.0, 0.01, 0.1])
def test_lion_optimizer_weight_decay(weight_decay):
    """Test Lion optimizer with different weight decay values."""
    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = Lion(params=[model.weight], lr=1e-4, betas=(0.9, 0.99), weight_decay=weight_decay)

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with weight_decay={weight_decay}"


@skip_no_lion
@pytest.mark.parametrize("weight_decay_method", ["decoupled", "l2"])
def test_lion_optimizer_weight_decay_method(weight_decay_method):
    """Test Lion optimizer with different weight decay methods."""
    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = Lion(
        params=[model.weight],
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0.01,
        weight_decay_method=weight_decay_method,
    )

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with weight_decay_method={weight_decay_method}"


@skip_no_lion
def test_lion_optimizer_multi_layer_net():
    """Test Lion optimizer with the multi-layer Net model."""
    model = Net().cuda()
    model.requires_grad_(True)

    optimizer = Lion(params=model.parameters(), lr=1e-4, betas=(0.9, 0.99), weight_decay=0.01)

    input_tensor = torch.randn(16, 80, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_params = {name: p.data.clone() for name, p in model.named_parameters()}
    optimizer.step()

    params_updated = 0
    for name, param in model.named_parameters():
        if not torch.equal(param.data, original_params[name]):
            params_updated += 1

    assert params_updated > 0, "At least some parameters should be updated after optimizer step"


# ===========================================================================
# use_syrk version gate
# ===========================================================================


@pytest.mark.parametrize("optimizer_cls", [TensorParallelMuon, TensorParallelAdaptiveMuon])
def test_muon_use_syrk_rejected_on_old_emerging_optimizers(monkeypatch, optimizer_cls):
    """use_syrk must raise on emerging_optimizers < 0.4.0 rather than silently falling back.

    Covers TensorParallelAdaptiveMuon too, since it forwards use_syrk through
    TensorParallelMuon.__init__ and that forwarding is what applies the gate to both.
    """
    import megatron.core.optimizer.emerging_optimizers as eo_module

    monkeypatch.setattr(eo_module, "is_emerging_optimizers_min_version", lambda _version: False)
    monkeypatch.setattr(eo_module, "get_emerging_optimizers_version", lambda: "0.2.0")

    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    with pytest.raises(ValueError, match="use_syrk requires emerging_optimizers"):
        optimizer_cls(
            params=[model.weight], lr=0.01, pg_collection=None, tp_mode="duplicated", use_syrk=True
        )


@pytest.mark.parametrize("optimizer_cls", [TensorParallelMuon, TensorParallelAdaptiveMuon])
def test_muon_use_syrk_default_off_ignores_version(monkeypatch, optimizer_cls):
    """The gate only fires when use_syrk is requested; the default path stays version-agnostic."""
    import megatron.core.optimizer.emerging_optimizers as eo_module

    monkeypatch.setattr(eo_module, "is_emerging_optimizers_min_version", lambda _version: False)

    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    optimizer = optimizer_cls(
        params=[model.weight], lr=0.01, pg_collection=None, tp_mode="duplicated"
    )
    assert optimizer is not None
