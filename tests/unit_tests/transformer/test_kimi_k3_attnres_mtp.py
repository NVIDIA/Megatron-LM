# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""K3 residual math and MCore MTP state/gradient contracts.

Reference: moonshotai/Kimi-K3, f831ab66814297da540d832a5235f8e904f29d06,
modeling_kimi_linear.py: _apply_attn_res and _forward_attn_residual.
https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06

MTP uses MCore's shift/projection semantics, not an unpublished K3 training
topology. Diagonal branch fixtures isolate the residual and MTP plumbing; the
integration test separately executes real KDA, gated MLA, and MoE kernels.
"""

from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.hybrid.hybrid_block import AttnResHybridLayer
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention_residual import AttentionResidual
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionBlock
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from tests.unit_tests.test_utilities import Utils


def _native_attn_res(values, query, norm_weight, eps=1e-6):
    values_fp32 = torch.stack([value.float() for value in values])
    keys = values_fp32 * torch.rsqrt(values_fp32.square().mean(-1, keepdim=True) + eps)
    scores = (keys * (norm_weight.float() * query.float())).sum(-1)
    weights = scores.softmax(dim=0)
    return (values_fp32 * weights.unsqueeze(-1)).sum(0).to(values[0].dtype)


def _native_norm(value, weight):
    normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + 1e-6)
    return (normalized * weight.float()).to(value.dtype)


def _assert_similarity(actual, expected, name, eps=1e-3):
    assert actual is not None and expected is not None, name
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), name
    actual, expected = actual.flatten().double(), expected.flatten().double()
    denom = (actual.square() + expected.square()).sum()
    if denom == 0:
        return
    cosine = F.cosine_similarity(actual[None], expected[None], eps=1e-30).item()
    similarity = (2 * (actual * expected).sum() / denom).item()
    assert cosine > 1 - eps, f"{name}: cosine={cosine}"
    assert similarity > 1 - eps, f"{name}: tensor_similarity={similarity}"


def _config(impl, dtype=torch.bfloat16, **kwargs):
    return TransformerConfig(
        num_layers=32,
        hidden_size=7168,
        num_attention_heads=96,
        kv_channels=128,
        is_hybrid_model=True,
        enable_attention_residuals=True,
        attn_res_block_layers=24,
        attn_res_impl=impl,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        add_bias_linear=False,
        params_dtype=dtype,
        bf16=dtype == torch.bfloat16,
        gradient_accumulation_fusion=False,
        use_cpu_initialization=True,
        **kwargs,
    )


@pytest.fixture
def pg_collection():
    """Initialize real distributed groups and reproducible per-test RNG state."""
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(1234)
    torch.manual_seed(1234)
    yield ProcessGroupCollection.use_mpu_process_groups()
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("impl", ["fla", "compile"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("n_sources", [1, 3, 9])
def test_k3_aggregation_native_parity(pg_collection, impl, dtype, n_sources):
    """Compare aggregation outputs and every input/parameter VJP to native Torch."""
    module = AttentionResidual(_config(impl, dtype)).cuda()
    with torch.no_grad():
        module.pseudo_query.normal_(std=0.005)
        module.key_norm_weight.uniform_(0.8, 1.2)
    values = [
        torch.randn(4, 1, 7168, device="cuda", dtype=dtype, requires_grad=True)
        for _ in range(n_sources)
    ]
    reference_values = [value.detach().clone().requires_grad_() for value in values]
    reference_params = {
        name: p.detach().clone().requires_grad_() for name, p in module.named_parameters()
    }
    assert set(reference_params) == {"pseudo_query", "key_norm_weight"}
    actual = module(values)
    expected = _native_attn_res(
        reference_values, reference_params["pseudo_query"], reference_params["key_norm_weight"]
    )
    _assert_similarity(actual, expected, "output")
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    for index, (value, reference) in enumerate(zip(values, reference_values)):
        _assert_similarity(value.grad, reference.grad, f"source {index}")
    for name, param in module.named_parameters():
        _assert_similarity(param.grad, reference_params[name].grad, name)


class _DiagonalBranch(torch.nn.Module):
    """Small differentiable branch fixture, not a KDA/MLA kernel reference."""

    def __init__(self, config, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((config.hidden_size,), 0.001))
        self.bias = torch.nn.Parameter(torch.full((config.hidden_size,), 0.0001))

    def forward(self, hidden_states, **kwargs):
        """Return a branch contribution and an unfused bias."""
        return hidden_states * self.weight, self.bias


class _Embedding(torch.nn.Embedding):
    add_position_embedding = False

    def forward(self, input_ids, position_ids=None):
        """Adapt a native embedding to MCore's sequence-major embedding interface."""
        return super().forward(input_ids).transpose(0, 1).contiguous()


def _fixture_submodules():
    submodules = deepcopy(hybrid_stack_spec.submodules)
    submodules.mla_layer = ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=TENorm,
            self_attention=ModuleSpec(module=_DiagonalBranch),
            self_attn_bda=get_bias_dropout_add,
        ),
    )
    submodules.moe_layer = ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            pre_mlp_layernorm=TENorm, mlp=_DiagonalBranch, mlp_bda=get_bias_dropout_add
        ),
    )
    return submodules


@pytest.mark.parametrize("impl", ["fla", "compile"])
@pytest.mark.parametrize(
    "depths,detach,repeated",
    [(1, False, False), (2, False, False), (2, True, False), (2, False, True)],
)
def test_k3_mtp_native_dataflow(pg_collection, impl, depths, detach, repeated):
    """Compare MTP state transitions, module hooks, and all gradients independently."""
    config = _config(
        impl, mtp_num_layers=depths, mtp_detach_heads=detach, mtp_use_repeated_layer=repeated
    )
    submodules = _fixture_submodules()
    block = MultiTokenPredictionBlock(
        config,
        spec=submodules.mtp_block_spec,
        pg_collection=pg_collection,
        mtp_layer_pattern="+E",
        mtp_num_depths=depths,
        hybrid_submodules=submodules,
    ).cuda()
    convert_module_to_dtype_except_fp32_marked(block, torch.bfloat16)
    embedding = _Embedding(16, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        for module in block.modules():
            if isinstance(module, AttentionResidual):
                module.pseudo_query.normal_(std=0.005)
                module.key_norm_weight.uniform_(0.8, 1.2)
    hidden = torch.randn(
        4, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    sources = tuple(torch.randn_like(hidden, requires_grad=True) for _ in range(3))
    ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    positions = torch.arange(4, device="cuda")[None]
    partials, hook_calls = [], []
    handles = []
    for depth in block.layers:
        for entry in depth.mtp_model_layer.layers:
            handles.append(entry.register_forward_hook(lambda m, a, out: partials.append(out)))
            handles.append(
                entry.inner_layer.register_forward_pre_hook(lambda m, a: hook_calls.append(m))
            )
    params = dict(block.named_parameters())
    reference_params = {name: p.detach().clone().requires_grad_() for name, p in params.items()}
    used_params = set()

    def param(name):
        used_params.add(name)
        return reference_params[name]

    reference_hidden = hidden.detach().clone().requires_grad_()
    reference_sources = [value.detach().clone().requires_grad_() for value in sources]
    reference_embedding = embedding.weight.detach().clone().requires_grad_()
    current = reference_hidden.detach() if detach else reference_hidden
    source_values = [v.detach() for v in reference_sources] if detach else reference_sources
    expected_depths, expected_partials = [], []
    for index in range(depths):
        prefix = f"layers.{0 if repeated else index}."
        shifted = F.pad(ids[:, index + 1 :], (0, index + 1))
        token_embedding = F.embedding(shifted, reference_embedding).transpose(0, 1)
        if detach:
            token_embedding = token_embedding.detach()
        partial = F.linear(
            torch.cat(
                (
                    _native_norm(token_embedding, param(prefix + "enorm.weight")),
                    _native_norm(current, param(prefix + "hnorm.weight")),
                ),
                dim=-1,
            ),
            param(prefix + "eh_proj.weight"),
        )
        for entry_index, branch, norm in [
            (0, "self_attention", "input_layernorm"),
            (1, "mlp", "pre_mlp_layernorm"),
        ]:
            entry = prefix + f"mtp_model_layer.layers.{entry_index}."
            aggregate = _native_attn_res(
                source_values + [partial],
                param(entry + "attn_res.pseudo_query"),
                param(entry + "attn_res.key_norm_weight"),
            )
            normalized = _native_norm(aggregate, param(entry + "inner_layer." + norm + ".weight"))
            delta = normalized * param(entry + "inner_layer." + branch + ".weight")
            delta = delta + param(entry + "inner_layer." + branch + ".bias")
            partial = partial + delta
            expected_partials.append(partial)
        current = _native_attn_res(
            source_values + [partial],
            param(prefix + "final_attn_res.pseudo_query"),
            param(prefix + "final_attn_res.key_norm_weight"),
        )
        current = _native_norm(current, param(prefix + "final_layernorm.weight"))
        expected_depths.append(current)
    assert used_params == set(params), set(params) - used_params
    try:
        actual = block(
            ids,
            positions,
            hidden,
            attention_mask=None,
            embedding=embedding,
            attn_res_sources=sources,
        )[hidden.shape[0] :]
    finally:
        for handle in handles:
            handle.remove()
    expected = torch.cat(expected_depths)
    assert len(partials) == len(hook_calls) == 2 * depths
    for index, (got, want) in enumerate(zip(partials, expected_partials)):
        _assert_similarity(got, want, f"partial {index}")
    _assert_similarity(actual, expected, "MTP outputs")
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    for name, value, reference in [
        ("hidden", hidden, reference_hidden),
        ("embedding", embedding.weight, reference_embedding),
        *[(f"source {i}", v, r) for i, (v, r) in enumerate(zip(sources, reference_sources))],
    ]:
        if detach:
            assert value.grad is None or torch.count_nonzero(value.grad) == 0, name
        else:
            _assert_similarity(value.grad, reference.grad, name)
    for name, value in params.items():
        _assert_similarity(value.grad, reference_params[name].grad, name)


@pytest.mark.parametrize("attention", [True, False])
def test_hybrid_delta_does_not_cancel_small_bf16_branch(pg_collection, attention):
    """Preserve a small branch update that BF16 residual subtraction would erase."""
    config = _config("fla")
    submodules = _fixture_submodules()
    inner = build_module(
        submodules.mla_layer if attention else submodules.moe_layer,
        config=config,
        layer_number=1,
        pg_collection=pg_collection,
    )
    layer = AttnResHybridLayer(config, inner).cuda()
    convert_module_to_dtype_except_fp32_marked(layer, torch.bfloat16)
    source = torch.ones(4, 1, 7168, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    actual = layer(source, attn_res_sources=(source,))
    branch = inner.self_attention if attention else inner.mlp
    expected = source * branch.weight + branch.bias
    # This tiny contribution is lost entirely by the old (1 + delta) - 1 path.
    assert torch.count_nonzero((source + expected) - source) == 0
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.sum().backward()
    assert torch.count_nonzero(branch.weight.grad) == branch.weight.numel()


@pytest.mark.parametrize("mtp_pattern", ["+E", "KE"])
def test_k3_real_hybrid_mtp_optimizer_updates(pg_collection, mtp_pattern):
    """Real KDA/NoPE gated MLA/MoE: two MTP depths, real losses and updates."""
    from megatron.core.activations import situlu
    from megatron.core.ssm.gated_delta_net import KimiDeltaAttention
    from megatron.core.transformer.multi_latent_attention import MLASelfAttention

    config = MLATransformerConfig(
        num_layers=8,
        hidden_size=1024,
        num_attention_heads=8,
        kv_channels=128,
        is_hybrid_model=True,
        enable_attention_residuals=True,
        attn_res_block_layers=6,
        attn_res_impl="fla",
        mtp_num_layers=2,
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        qk_layernorm=True,
        attention_latent_norm_epsilon=1e-6,
        q_lora_rank=512,
        kv_lora_rank=512,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=128,
        rope_type="rope",
        no_rope_freq=1,
        attention_output_gate=True,
        linear_num_key_heads=8,
        linear_num_value_heads=8,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        kda_safe_gate=True,
        kda_lower_bound=-5.0,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=0.01,
        moe_grouped_gemm=True,
        moe_ffn_hidden_size=512,
        ffn_hidden_size=2048,
        moe_latent_size=512,
        moe_latent_up_projection_rmsnorm=True,
        moe_shared_expert_intermediate_size=1024,
        gated_linear_unit=True,
        activation_func=situlu,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        add_bias_linear=False,
        params_dtype=torch.bfloat16,
        bf16=True,
        gradient_accumulation_fusion=False,
        use_cpu_initialization=True,
        attention_backend=AttnBackend.fused,
    )
    model = (
        HybridModel(
            config,
            hybrid_stack_spec,
            vocab_size=4096,
            max_sequence_length=128,
            hybrid_layer_pattern=f"K-KEKE+E/{mtp_pattern}/{mtp_pattern}",
            pg_collection=pg_collection,
        )
        .cuda()
        .train()
    )
    convert_module_to_dtype_except_fp32_marked(model, torch.bfloat16)
    assert any(isinstance(m, KimiDeltaAttention) for m in model.decoder.modules())
    assert any(isinstance(m, MLASelfAttention) for m in model.decoder.modules())
    for module in model.modules():
        if isinstance(module, KimiDeltaAttention):
            assert module.act_fn is F.silu
            assert module.activation == "silu"
            assert module.config.activation_func is situlu
    for depth in model.mtp.layers:
        attention = depth.mtp_model_layer.layers[0].inner_layer.self_attention
        assert attention.is_mtp_layer
        assert isinstance(
            attention, MLASelfAttention if mtp_pattern == "+E" else KimiDeltaAttention
        )
    watched = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if name.startswith("mtp.") and (name.endswith("pseudo_query") or "eh_proj.weight" in name)
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    ids = torch.randint(0, 4096, (1, 128), device="cuda")
    positions = torch.arange(128, device="cuda")[None]
    labels = ids.roll(-1, dims=-1)
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        loss = model(ids, positions, None, labels=labels, loss_mask=torch.ones_like(ids))
        assert torch.isfinite(loss).all()
        loss.float().mean().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name
        optimizer.step()
    for name, param in model.named_parameters():
        if name in watched:
            assert not torch.equal(param, watched[name]), name


def test_k3_nope_mla_native_parity(pg_collection):
    """Keep all 192 Q/K channels, skip RoPE, and retain the gated output VJP."""
    config = MLATransformerConfig(
        num_layers=1,
        hidden_size=7168,
        num_attention_heads=96,
        kv_channels=128,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=128,
        qk_layernorm=True,
        layernorm_epsilon=1e-6,
        attention_latent_norm_epsilon=1e-6,
        rope_type="rope",
        no_rope_freq=1,
        attention_output_gate=True,
        normalization="RMSNorm",
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=True,
        params_dtype=torch.bfloat16,
        bf16=True,
        gradient_accumulation_fusion=False,
        attention_backend=AttnBackend.unfused,
    )
    spec = deepcopy(hybrid_stack_spec.submodules.mla_layer.submodules.self_attention)
    spec.submodules.q_layernorm = TENorm
    spec.submodules.kv_layernorm = TENorm
    module = build_module(spec, config=config, layer_number=1, pg_collection=pg_collection).cuda()
    convert_module_to_dtype_except_fp32_marked(module, torch.bfloat16)
    params = dict(module.named_parameters())
    reference_params = {name: p.detach().clone().requires_grad_() for name, p in params.items()}
    expected_names = {
        f"{name}.weight"
        for name in (
            "linear_q_down_proj",
            "linear_q_up_proj",
            "linear_kv_down_proj",
            "linear_kv_up_proj",
            "q_layernorm",
            "kv_layernorm",
            "linear_proj",
            "linear_gate",
        )
    }
    assert set(params) == expected_names
    hidden = torch.randn(4, 1, 7168, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    p = reference_params
    q = F.linear(reference_hidden, p["linear_q_down_proj.weight"])
    q = _native_norm(q, p["q_layernorm.weight"])
    q = F.linear(q, p["linear_q_up_proj.weight"]).view(4, 1, 96, 192)
    kv = F.linear(reference_hidden, p["linear_kv_down_proj.weight"])
    compressed, k_shared = kv.split([512, 64], dim=-1)
    compressed = _native_norm(compressed, p["kv_layernorm.weight"])
    kv = F.linear(compressed, p["linear_kv_up_proj.weight"]).view(4, 1, 96, 256)
    k, v = kv.split([128, 128], dim=-1)
    k = torch.cat((k, k_shared.unsqueeze(-2).expand(4, 1, 96, 64)), dim=-1)
    scores = torch.einsum("sbhd,tbhd->bhst", q.float(), k.float()) * (192**-0.5)
    mask = torch.ones(4, 4, device="cuda", dtype=torch.bool).triu(1)
    weights = scores.masked_fill(mask, float("-inf")).softmax(-1).to(v.dtype)
    context = torch.einsum("bhst,tbhd->sbhd", weights, v).reshape(4, 1, -1)
    gate = F.linear(reference_hidden, p["linear_gate.weight"]).float().sigmoid().to(context.dtype)
    expected = F.linear(context * gate, p["linear_proj.weight"])
    actual, _ = module(hidden, attention_mask=mask[None, None])
    _assert_similarity(actual, expected, "NoPE gated MLA output")
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    _assert_similarity(hidden.grad, reference_hidden.grad, "NoPE gated MLA input")
    for name, param in params.items():
        _assert_similarity(param.grad, p[name].grad, name)
