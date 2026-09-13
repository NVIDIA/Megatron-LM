# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU behavior contracts for QAT and R3 across every MLite MoE model."""

from __future__ import annotations

import copy
import importlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from megatron.lite.primitive.modules.router_replay import (
    attach_router_replay,
    detach_router_replay,
)
from megatron.lite.primitive.quantization.qat import (
    QATSpec,
    apply_qat_to_chunks,
    normalize_qat_spec,
)

pytestmark = pytest.mark.mlite

MODEL_NAMES = ("qwen3_5", "qwen3_moe", "deepseek_v4", "glm5", "kimi_k2")


class _TinyRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(8, 4, bias=False)
        self.router_replay = None


class _TinyDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.moe = nn.Module()
        self.moe.router = _TinyRouter()
        self.mlp = nn.Module()
        self.mlp.gate_up = nn.Linear(8, 16, bias=False)
        self.mlp.down = nn.Linear(16, 8, bias=False)


class _TinyModel(nn.Module):
    def __init__(self, model_name: str, *, mtp_enabled: bool):
        super().__init__()
        layers = [_TinyDecoderLayer(), _TinyDecoderLayer()]
        if model_name == "deepseek_v4":
            self.layers = nn.ModuleDict(
                {str(i): layer for i, layer in enumerate(layers)}
            )
        else:
            self.layers = nn.ModuleList(layers)
        if mtp_enabled:
            self.mtp = nn.ModuleList([_TinyDecoderLayer()])


class _TinyChunk(nn.Module):
    def __init__(self, model_name: str, *, mtp_enabled: bool):
        super().__init__()
        self.model = _TinyModel(model_name, mtp_enabled=mtp_enabled)


class _CpuTELinear(nn.Linear):
    """State-dict-compatible TE Linear stand-in for CPU-only construction."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        return_bias: bool = False,
        **_kwargs,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.return_bias = return_bias

    def forward(self, x):
        output = super().forward(x)
        return (output, self.bias) if self.return_bias else output


class _CpuTELayerNormLinear(nn.Module):
    """Minimal TE LayerNormLinear parameter layout; forward is not exercised."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        return_bias: bool = False,
        zero_centered_gamma: bool = False,
        **_kwargs,
    ):
        super().__init__()
        self.layer_norm_weight = nn.Parameter(torch.zeros(in_features))
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.return_bias = return_bias
        self.zero_centered_gamma = zero_centered_gamma


class _CpuTERMSNorm(nn.Module):
    """State-dict-compatible TE RMSNorm stand-in for CPU-only construction."""

    def __init__(
        self,
        hidden_size: int,
        *,
        zero_centered_gamma: bool = False,
        **_kwargs,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.zero_centered_gamma = zero_centered_gamma


class _CpuTEDotProductAttention(nn.Module):
    def __init__(self, *_args, **_kwargs):
        super().__init__()


class _CpuTEGroupedLinear(nn.Module):
    """TE GroupedLinear parameter names without requiring its CUDA kernels."""

    def __init__(
        self,
        num_gemms: int,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        **_kwargs,
    ):
        super().__init__()
        for index in range(num_gemms):
            self.register_parameter(
                f"weight{index}",
                nn.Parameter(torch.empty(out_features, in_features)),
            )
            if bias:
                self.register_parameter(
                    f"bias{index}",
                    nn.Parameter(torch.zeros(out_features)),
                )


@dataclass(frozen=True)
class _ModelCase:
    name: str
    protocol: object
    chunk: _TinyChunk


def _protocol(model_name: str, transformer_engine_import_stub, monkeypatch):
    transformer_engine_import_stub()
    if model_name == "deepseek_v4":
        # The package __init__ eagerly imports the fused CSA model, which is not
        # needed for this CPU protocol contract. Load the real protocol and
        # checkpoint modules without importing a GPU-only model implementation.
        package_name = "megatron.lite.model.deepseek_v4.lite"
        package = types.ModuleType(package_name)
        package.__path__ = [
            str(
                Path(__file__).parents[3]
                / "megatron"
                / "lite"
                / "model"
                / "deepseek_v4"
                / "lite"
            )
        ]
        monkeypatch.setitem(sys.modules, package_name, package)
    return importlib.import_module(f"megatron.lite.model.{model_name}.lite.protocol")


def _layers(chunk: _TinyChunk) -> list[nn.Module]:
    layers = chunk.model.layers
    return list(layers.values()) if isinstance(layers, nn.ModuleDict) else list(layers)


def _case(
    model_name: str,
    transformer_engine_import_stub,
    monkeypatch,
    *,
    mtp_enabled: bool,
) -> _ModelCase:
    return _ModelCase(
        name=model_name,
        protocol=_protocol(model_name, transformer_engine_import_stub, monkeypatch),
        chunk=_TinyChunk(model_name, mtp_enabled=mtp_enabled),
    )


def _install_cpu_te_construction_stubs(transformer_engine_import_stub, monkeypatch):
    transformer_engine_import_stub()
    te = importlib.import_module("transformer_engine.pytorch")
    te_types = {
        "Linear": _CpuTELinear,
        "LayerNormLinear": _CpuTELayerNormLinear,
        "RMSNorm": _CpuTERMSNorm,
        "DotProductAttention": _CpuTEDotProductAttention,
        "GroupedLinear": _CpuTEGroupedLinear,
    }
    for name, replacement in te_types.items():
        monkeypatch.setattr(te, name, replacement, raising=False)
    # Parameterized tests may have imported model/primitive modules under a
    # previous fixture-owned TE stub. Patch every retained module-local ``te``
    # reference as well as the current sys.modules entry.
    for module in tuple(sys.modules.values()):
        module_te = getattr(module, "te", None)
        if not isinstance(module_te, types.ModuleType):
            continue
        if module_te.__name__ != "transformer_engine.pytorch":
            continue
        for name, replacement in te_types.items():
            monkeypatch.setattr(module_te, name, replacement, raising=False)

    te_root = importlib.import_module("transformer_engine")
    monkeypatch.setattr(te_root, "__version__", "2.0.0")
    te_tensor = types.ModuleType("transformer_engine.pytorch.tensor")
    te_tensor.QuantizedTensor = torch.Tensor
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.tensor", te_tensor)
    dsa = importlib.import_module("megatron.lite.primitive.modules.attention.dsa")
    monkeypatch.setattr(dsa, "RMSNorm", _CpuTERMSNorm)


def _install_csa_import_stubs(monkeypatch):
    """Expose the Core CSA symbols needed to construct DS4 without GPU extras."""

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("CSA execution is outside this state-dict-only CPU test")

    core = types.ModuleType("megatron.core")
    tensor_parallel = types.ModuleType("megatron.core.tensor_parallel")
    mappings = types.ModuleType("megatron.core.tensor_parallel.mappings")
    mappings.gather_from_sequence_parallel_region = unavailable
    transformer = types.ModuleType("megatron.core.transformer")
    variants = types.ModuleType(
        "megatron.core.transformer.experimental_attention_variant"
    )
    layout = types.ModuleType(
        "megatron.core.transformer.experimental_attention_variant.csa_cp_layout_kernels"
    )
    cp_utils = types.ModuleType(
        "megatron.core.transformer.experimental_attention_variant.csa_cp_utils"
    )
    core_csa = types.ModuleType(
        "megatron.core.transformer.experimental_attention_variant.csa"
    )
    core_dsa = types.ModuleType(
        "megatron.core.transformer.experimental_attention_variant.dsa"
    )
    dsa_kernels = types.ModuleType(
        "megatron.core.transformer.experimental_attention_variant.dsa_kernels"
    )
    core_csa._unfused_indexer_sparse_attn_from_topk = unavailable
    core_csa.unfused_compressed_sparse_attn = unavailable
    core_dsa.DSAIndexerLossAutoScaler = torch.autograd.Function
    core_dsa.DSAIndexerLossLoggingHelper = type("DSAIndexerLossLoggingHelper", (), {})
    dsa_kernels.FusedIndexerSparseAttnFromTopkFunc = torch.autograd.Function
    dsa_kernels.dsa_sparse_attn = unavailable
    variants.csa_cp_layout_kernels = layout
    variants.csa_cp_utils = cp_utils
    modules = {
        "megatron.core": core,
        "megatron.core.tensor_parallel": tensor_parallel,
        "megatron.core.tensor_parallel.mappings": mappings,
        "megatron.core.transformer": transformer,
        "megatron.core.transformer.experimental_attention_variant": variants,
        "megatron.core.transformer.experimental_attention_variant.csa_cp_layout_kernels": layout,
        "megatron.core.transformer.experimental_attention_variant.csa_cp_utils": cp_utils,
        "megatron.core.transformer.experimental_attention_variant.csa": core_csa,
        "megatron.core.transformer.experimental_attention_variant.dsa": core_dsa,
        "megatron.core.transformer.experimental_attention_variant.dsa_kernels": dsa_kernels,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _train_config():
    return types.SimpleNamespace(
        tp=1,
        ep=1,
        etp=1,
        pp=1,
        cp=1,
        vpp=None,
        use_deepep=False,
        fp8=False,
        recompute_modules=[],
        offload_modules=[],
        deterministic=True,
    )


def _real_tiny_model(model_name: str, monkeypatch):
    from megatron.lite.primitive.parallel import ParallelState

    ps = ParallelState()
    if model_name == "qwen3_moe":
        from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
        from megatron.lite.model.qwen3_moe.lite.model import Qwen3MoEModel

        config = Qwen3MoEConfig(
            num_hidden_layers=2,
            hidden_size=16,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            vocab_size=32,
            num_experts=3,
            num_experts_per_tok=1,
            moe_intermediate_size=8,
            max_position_embeddings=16,
            layer_types=["full_attention", "full_attention"],
        )
        return Qwen3MoEModel(config, ps, use_deepep=False)
    if model_name == "qwen3_5":
        from megatron.lite.model.qwen3_5.config import Qwen35Config
        from megatron.lite.model.qwen3_5.lite.model import Qwen35Model

        config = Qwen35Config(
            num_hidden_layers=2,
            hidden_size=16,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            vocab_size=32,
            num_experts=3,
            num_experts_per_tok=1,
            moe_intermediate_size=8,
            shared_expert_intermediate_size=8,
            max_position_embeddings=16,
            partial_rotary_factor=1.0,
            layer_types=["full_attention", "full_attention"],
        )
        return Qwen35Model(config, _train_config(), ps)
    if model_name == "kimi_k2":
        from megatron.lite.model.kimi_k2.config import KimiK2Config
        from megatron.lite.model.kimi_k2.lite.model import KimiK2Model

        config = KimiK2Config(
            num_hidden_layers=2,
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=32,
            intermediate_size=48,
            moe_intermediate_size=8,
            n_routed_experts=3,
            n_shared_experts=1,
            num_experts_per_tok=1,
            n_group=1,
            topk_group=1,
            first_k_dense_replace=0,
            q_lora_rank=8,
            kv_lora_rank=8,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            max_position_embeddings=16,
        )
        return KimiK2Model(config, _train_config(), ps)
    if model_name == "glm5":
        from megatron.lite.model.glm5.config import Glm5Config
        from megatron.lite.model.glm5.lite.model import Glm5Model

        config = Glm5Config(
            num_hidden_layers=2,
            hidden_size=16,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            vocab_size=32,
            max_position_embeddings=16,
            q_lora_rank=8,
            kv_lora_rank=4,
            qk_head_dim=8,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            index_head_dim=8,
            index_n_heads=2,
            index_topk=2,
            intermediate_size=20,
            moe_intermediate_size=6,
            first_k_dense_replace=0,
            n_routed_experts=3,
            n_shared_experts=1,
            num_experts_per_tok=2,
        )
        return Glm5Model(config, _train_config(), ps)
    if model_name == "deepseek_v4":
        _install_csa_import_stubs(monkeypatch)
        from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
        from megatron.lite.model.deepseek_v4.lite.model import DeepseekV4Model

        config = DeepseekV4Config(
            vocab_size=32,
            hidden_size=32,
            moe_intermediate_size=8,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=8,
            qk_rope_head_dim=4,
            q_lora_rank=16,
            o_lora_rank=16,
            o_groups=2,
            n_routed_experts=3,
            n_shared_experts=1,
            num_experts_per_tok=1,
            max_position_embeddings=16,
            compress_ratios=[4, 4],
            sliding_window=4,
            num_hash_layers=1,
            hc_mult=2,
            index_head_dim=8,
            index_n_heads=4,
            index_topk=4,
            num_nextn_predict_layers=0,
        )
        return DeepseekV4Model(config, _train_config(), ps)
    raise AssertionError(f"unsupported model: {model_name}")


R3_SUPPORTED_MODEL_NAMES = MODEL_NAMES


def test_mapped_model_state_has_no_optional_override(
    transformer_engine_import_stub,
    monkeypatch,
):
    for model_name, spec_name in (
        ("kimi_k2", "KimiK2WeightSpec"),
        ("glm5", "Glm5WeightSpec"),
        ("deepseek_v4", "DeepseekV4WeightSpec"),
    ):
        _protocol(model_name, transformer_engine_import_stub, monkeypatch)
        checkpoint = importlib.import_module(
            f"megatron.lite.model.{model_name}.lite.checkpoint"
        )
        spec_type = getattr(checkpoint, spec_name)
        assert not hasattr(spec_type, "optional_for_load")
        assert not hasattr(spec_type, "expected_buffers")
        assert not hasattr(spec_type, "is_export_buffer")


@pytest.mark.parametrize(
    ("model_name", "spec_name"),
    [
        ("kimi_k2", "KimiK2WeightSpec"),
        ("glm5", "Glm5WeightSpec"),
    ],
)
def test_persistent_router_bias_is_required(
    model_name,
    spec_name,
    tmp_path,
    transformer_engine_import_stub,
    monkeypatch,
):
    from safetensors.torch import save_file

    from megatron.lite.primitive.ckpt.hf_weights import load_hf_weights

    _protocol(model_name, transformer_engine_import_stub, monkeypatch)
    checkpoint = importlib.import_module(
        f"megatron.lite.model.{model_name}.lite.checkpoint"
    )
    spec = getattr(checkpoint, spec_name)(types.SimpleNamespace(num_experts=0))

    class Router(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("expert_bias", torch.zeros(3))

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.moe = nn.Module()
            self.moe.router = Router()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.required = nn.Parameter(torch.zeros(1))
            self.layers = nn.ModuleList([Layer()])

    model = Model()
    monkeypatch.setattr(
        spec,
        "weight_map",
        lambda: {
            "required": ["hf.required"],
            "layers.0.moe.router.expert_bias": ["hf.router.expert_bias"],
        },
    )
    save_file(
        {"hf.required": torch.ones(1)},
        str(tmp_path / "model.safetensors"),
    )
    ps = types.SimpleNamespace(
        ep_size=1,
        ep_rank=0,
        tp_size=1,
        tp_rank=0,
        etp_size=1,
        etp_rank=0,
        pp_size=1,
    )

    with pytest.raises(
        RuntimeError,
        match=rf"{spec_name}.*layers\.0\.moe\.router\.expert_bias",
    ):
        load_hf_weights(model, str(tmp_path), spec, ps)


def test_deepseek_v4_router_buffer_remains_required(
    tmp_path,
    transformer_engine_import_stub,
    monkeypatch,
):
    from safetensors.torch import save_file

    from megatron.lite.primitive.ckpt.hf_weights import load_hf_weights

    _protocol("deepseek_v4", transformer_engine_import_stub, monkeypatch)
    checkpoint = importlib.import_module(
        "megatron.lite.model.deepseek_v4.lite.checkpoint"
    )
    spec = checkpoint.DeepseekV4WeightSpec(
        types.SimpleNamespace(num_hash_layers=0, n_routed_experts=1)
    )

    class Gate(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("expert_bias", torch.zeros(1))

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.gate = Gate()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer()])

    save_file(
        {"unrelated": torch.ones(1)},
        str(tmp_path / "model.safetensors"),
    )
    ps = types.SimpleNamespace(
        ep_size=1,
        ep_rank=0,
        tp_size=1,
        tp_rank=0,
        etp_size=1,
        etp_rank=0,
    )

    with pytest.raises(
        RuntimeError,
        match=r"DeepseekV4WeightSpec.*layers\.0\.mlp\.gate\.expert_bias"
        r".*layers\.0\.ffn\.gate\.bias",
    ):
        load_hf_weights(Model(), str(tmp_path), spec, ps)


def test_kimi_router_bias_92_key_roundtrip_uses_weight_map(
    tmp_path,
    transformer_engine_import_stub,
    monkeypatch,
):
    from safetensors.torch import save_file

    from megatron.lite.primitive.ckpt.hf_weights import (
        export_hf_weights,
        load_hf_weights,
    )

    _protocol("kimi_k2", transformer_engine_import_stub, monkeypatch)
    checkpoint = importlib.import_module("megatron.lite.model.kimi_k2.lite.checkpoint")
    spec = checkpoint.KimiK2WeightSpec(types.SimpleNamespace(num_experts=0))

    class Router(nn.Module):
        def __init__(self, layer_idx):
            super().__init__()
            self.register_buffer(
                "expert_bias",
                torch.full((2,), float(layer_idx)),
            )

    class Layer(nn.Module):
        def __init__(self, layer_idx):
            super().__init__()
            self.moe = nn.Module()
            self.moe.router = Router(layer_idx)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer(idx) for idx in range(92)])

    weight_map = {
        f"layers.{idx}.moe.router.expert_bias": [
            f"model.layers.{idx}.mlp.gate.e_score_correction_bias"
        ]
        for idx in range(92)
    }
    monkeypatch.setattr(spec, "weight_map", lambda: weight_map)
    ps = types.SimpleNamespace(
        pp_size=1,
        tp_size=1,
        tp_rank=0,
        tp_group=None,
        ep_size=1,
        ep_rank=0,
        ep_group=None,
        etp_size=1,
        etp_rank=0,
        etp_group=None,
    )

    exported = dict(export_hf_weights(Model(), spec, ps))
    assert set(exported) == {
        hf_name for hf_names in weight_map.values() for hf_name in hf_names
    }
    assert len(exported) == 92

    save_file(exported, str(tmp_path / "model.safetensors"))
    loaded = Model()
    for buffer in loaded.buffers():
        buffer.zero_()
    load_hf_weights(loaded, str(tmp_path), spec, ps)

    assert all(
        torch.equal(
            loaded.layers[idx].moe.router.expert_bias,
            torch.full((2,), float(idx)),
        )
        for idx in range(92)
    )


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_glm5_r3_route_packing_matches_contiguous_forward_layout(
    cp_rank,
    transformer_engine_import_stub,
    monkeypatch,
):
    from megatron.lite.runtime.contracts import PackedBatch

    glm5 = _protocol("glm5", transformer_engine_import_stub, monkeypatch)
    deepseek_v4 = _protocol("deepseek_v4", transformer_engine_import_stub, monkeypatch)
    model = nn.Module()
    model.ps = types.SimpleNamespace(
        tp_size=1,
        tp_rank=0,
        cp_size=2,
        cp_rank=cp_rank,
        cp_group=None,
    )
    batch = PackedBatch(
        input_ids=torch.arange(8),
        labels=torch.arange(8),
        seq_lens=torch.tensor([4, 4]),
        r3_replay_mask=torch.tensor(
            [True, False, True, False, False, True, False, True]
        ),
    )
    routes = torch.arange(8).view(2, 4, 1, 1)

    glm5_routes = glm5.pack_routed_experts(model, batch, routes)
    deepseek_v4_routes = deepseek_v4.pack_routed_experts(model, batch, routes)
    glm5_mask = glm5.pack_r3_replay_mask(model, batch)
    deepseek_v4_mask = deepseek_v4.pack_r3_replay_mask(model, batch)

    expected = torch.arange(cp_rank * 4, (cp_rank + 1) * 4).view(4, 1)
    assert len(glm5_routes) == 1
    assert torch.equal(glm5_routes[0], expected)
    assert torch.equal(glm5_routes[0], deepseek_v4_routes[0])
    route_token_ids = glm5_routes[0].squeeze(-1)
    assert torch.equal(glm5_mask, batch.r3_replay_mask[route_token_ids])
    assert torch.equal(glm5_mask, deepseek_v4_mask)


@pytest.mark.parametrize("model_name", R3_SUPPORTED_MODEL_NAMES)
@pytest.mark.parametrize("mtp_enabled", [False, True], ids=["mtp-off", "mtp-on"])
def test_supported_model_replay_roots_are_exact_decoder_layers(
    model_name: str,
    mtp_enabled: bool,
    transformer_engine_import_stub,
    monkeypatch,
):
    case = _case(
        model_name,
        transformer_engine_import_stub,
        monkeypatch,
        mtp_enabled=mtp_enabled,
    )
    expected = _layers(case.chunk)

    roots = case.protocol.router_replay_roots(case.chunk)

    assert roots == expected
    assert len(roots) == len(case.chunk.model.layers)
    assert roots != [case.chunk], (
        "falling back to the whole chunk would include MTP routers"
    )
    if mtp_enabled:
        mtp_modules = set(case.chunk.model.mtp.modules())
        assert all(root not in mtp_modules for root in roots)


@pytest.mark.parametrize("model_name", R3_SUPPORTED_MODEL_NAMES)
def test_supported_model_mtp_off_replay_attachment_count_is_unchanged(
    model_name: str,
    transformer_engine_import_stub,
    monkeypatch,
):
    case = _case(
        model_name,
        transformer_engine_import_stub,
        monkeypatch,
        mtp_enabled=False,
    )
    old_count = attach_router_replay(case.chunk, reset=False)
    detach_router_replay(case.chunk)

    roots = case.protocol.router_replay_roots(case.chunk)
    new_count = sum(attach_router_replay(root, reset=False) for root in roots)
    try:
        assert old_count == len(case.chunk.model.layers)
        assert new_count == old_count
    finally:
        for root in roots:
            detach_router_replay(root)


@pytest.mark.parametrize("model_name", R3_SUPPORTED_MODEL_NAMES)
def test_supported_model_attaches_replay_only_to_decoder_router_count(
    model_name: str,
    transformer_engine_import_stub,
    monkeypatch,
):
    case = _case(
        model_name,
        transformer_engine_import_stub,
        monkeypatch,
        mtp_enabled=True,
    )
    roots = case.protocol.router_replay_roots(case.chunk)

    count = sum(attach_router_replay(root, reset=False) for root in roots)
    try:
        assert count == len(case.chunk.model.layers)
        assert all(
            layer.moe.router.router_replay is not None for layer in _layers(case.chunk)
        )
        assert all(
            layer.moe.router.router_replay is None for layer in case.chunk.model.mtp
        )
    finally:
        for root in roots:
            detach_router_replay(root)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_real_tiny_model_replay_attachment_count_matches_decoder_layers(
    model_name: str,
    transformer_engine_import_stub,
    monkeypatch,
):
    _install_cpu_te_construction_stubs(transformer_engine_import_stub, monkeypatch)
    model = _real_tiny_model(model_name, monkeypatch)

    count = attach_router_replay(model, reset=False)
    try:
        assert count == len(model.layers)
    finally:
        detach_router_replay(model)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("layers.0.mlp.weight", "layers.0.mlp.weight"),
        (
            "layers.0.mlp.parametrizations.weight.original",
            "layers.0.mlp.weight",
        ),
        (
            "layers.0.mlp.parametrizations.weight.0.amax",
            "layers.0.mlp.parametrizations.weight.0.amax",
        ),
        (
            "model.decoder.layers.7.mlp.down_proj.parametrizations.weight.original",
            "model.decoder.layers.7.mlp.down_proj.weight",
        ),
    ],
)
def test_every_model_uses_shared_canonical_state_key(
    model_name: str,
    key: str,
    expected: str,
    transformer_engine_import_stub,
    monkeypatch,
):
    from megatron.lite.primitive.ckpt.hf_weights import (
        canonical_state_key as loader_canonical_state_key,
    )
    from megatron.lite.primitive.quantization.qat import canonical_state_key

    _protocol(model_name, transformer_engine_import_stub, monkeypatch)
    assert loader_canonical_state_key is canonical_state_key
    assert loader_canonical_state_key(key) == expected


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_every_model_qat_off_canonicalization_is_identity_for_all_real_state_keys(
    model_name: str,
    transformer_engine_import_stub,
    monkeypatch,
):
    _install_cpu_te_construction_stubs(transformer_engine_import_stub, monkeypatch)
    model = _real_tiny_model(model_name, monkeypatch)
    from megatron.lite.primitive.ckpt.hf_weights import canonical_state_key

    state_keys = tuple(model.state_dict())

    assert state_keys
    assert {canonical_state_key(key): key for key in state_keys} == {
        key: key for key in state_keys
    }


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_every_model_qat_none_is_bitwise_inert(
    model_name: str,
    transformer_engine_import_stub,
    monkeypatch,
):
    case = _case(
        model_name,
        transformer_engine_import_stub,
        monkeypatch,
        mtp_enabled=False,
    )
    implicit = copy.deepcopy(case.chunk)
    explicit = copy.deepcopy(case.chunk)

    implicit_cfg = case.protocol.ImplConfig()
    explicit_cfg = case.protocol.ImplConfig(qat=None)
    implicit_stats = apply_qat_to_chunks(
        [implicit], normalize_qat_spec(implicit_cfg.qat)
    )
    explicit_stats = apply_qat_to_chunks(
        [explicit], normalize_qat_spec(explicit_cfg.qat)
    )

    assert implicit_stats["quantized_modules"] == 0
    assert explicit_stats["quantized_modules"] == 0
    implicit_state = implicit.state_dict()
    explicit_state = explicit.state_dict()
    assert implicit_state.keys() == explicit_state.keys()
    assert all(
        torch.equal(implicit_state[key], explicit_state[key]) for key in implicit_state
    )


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_every_model_qat_quantizes_gate_up_but_not_router_gate(
    model_name: str,
    transformer_engine_import_stub,
    monkeypatch,
):
    case = _case(
        model_name,
        transformer_engine_import_stub,
        monkeypatch,
        mtp_enabled=False,
    )

    stats = apply_qat_to_chunks(
        [case.chunk],
        QATSpec(enabled=True, format="int8", group_size=-1),
    )

    assert stats["quantized_modules"] > 0
    assert stats["skipped_ignored"] > 0
    for layer in _layers(case.chunk):
        assert not parametrize.is_parametrized(layer.moe.router.gate, "weight")
        assert parametrize.is_parametrized(layer.mlp.gate_up, "weight")


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_every_model_qat_expert_load_target_resolves_master_weight(
    model_name: str,
    transformer_engine_import_stub,
    monkeypatch,
):
    _install_cpu_te_construction_stubs(transformer_engine_import_stub, monkeypatch)
    model = _real_tiny_model(model_name, monkeypatch)
    stats = apply_qat_to_chunks(
        [model],
        QATSpec(enabled=True, format="int8", group_size=-1),
    )
    assert stats["quantized_modules"] > 0

    checkpoint = importlib.import_module(
        f"megatron.lite.model.{model_name}.lite.checkpoint"
    )
    spec_type = {
        "qwen3_moe": "Qwen3MoEWeightSpec",
        "qwen3_5": "Qwen35WeightSpec",
        "kimi_k2": "KimiK2WeightSpec",
        "glm5": "Glm5WeightSpec",
        "deepseek_v4": "DeepseekV4WeightSpec",
    }[model_name]
    spec = getattr(checkpoint, spec_type)(model.config)

    from megatron.lite.primitive.ckpt.hf_weights import (
        _resolve_param_name,
        canonical_state_key,
    )

    state = model.state_dict()
    logical_state_keys = tuple(canonical_state_key(name) for name in state)
    load_weight_map = getattr(spec, "load_weight_map", None)
    weight_map = (
        load_weight_map(model, model.ps, logical_state_keys)
        if callable(load_weight_map)
        else spec.weight_map()
    )
    expert_names = [
        name
        for name in weight_map
        if spec.expert_global_id(name) is not None and not name.startswith("mtp.")
    ]
    assert expert_names

    for native_name in expert_names:
        global_id = spec.expert_global_id(native_name)
        assert global_id is not None
        local_name = spec.expert_local_name(native_name, global_id)
        assert _resolve_param_name(local_name, state) is not None, (
            model_name,
            native_name,
            local_name,
        )
