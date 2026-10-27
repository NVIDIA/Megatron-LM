# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Production-geometry GLM53 logical blocks versus independent HF93c8 math.

Only depth/sequence/batch are reduced. A real HybridStack/spec owns every norm,
mHC, attention and FFN. References never call target or fused math. No official
checkpoint, distributed TP/PP qualification, FP8 or CP coverage is claimed.

Comparison criteria of the all-gradient mHC cases. Two discrete or
ill-conditioned quantities are replaced, and everything else keeps the 0.999
bars. (1) SwiGLU clamp decisions are replayed from the target into the
reference (glm53_native_reference.ClampReplay): rounding-level FFN-input
differences otherwise move a few std-0.5 coverage-channel values across the
threshold, and the backward amplifies each moved decision into every upstream
gradient link. Each replayed decision must be a near tie. (2) The shape-[1] mHC
scale gradients are compared through their per-token contribution tensors
(assert_mhc_decomposition): each scalar is a heavily cancelling sum, on which
equally valid reference executions fail the bar against each other. Routing,
DSA indices and all other links are unchanged.
"""

import gc
import os
from contextlib import contextmanager, nullcontext

import pytest
import torch
import torch.nn.functional as F

from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_glm53_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import common as gdn_common
from megatron.core.ssm.gated_delta_net import kda as kda_module
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.experimental_attention_variant import dsa as dsa_module
from megatron.core.transformer.experimental_attention_variant import dsa_kernels
from megatron.core.transformer.hyper_connection import (
    HyperConnectionModule,
    _get_precision_preserving_h_post_bda,
)
from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.models.glm53_native_reference import (
    HIDDEN,
    KDA_WIDTH,
    BufferContract,
    ClampReplay,
    ComparisonLedger,
    HostParameters,
    NativeFFN,
    NativeHybridBlock,
    ParameterLink,
    assert_all_gradients,
    assert_mhc_decomposition,
    assert_similarity,
    balanced_router_weight,
    clamp_category_counts,
    copy_parameters,
    record_details,
)
from tests.unit_tests.test_utilities import Utils

BLOCK_MHC_SITES = (
    ('layers.0.hyper_connection', 'attn_hc'),
    ('layers.1.hyper_connection', 'ffn_hc'),
)


def make_config(*, deterministic=False, backend='none', sparse=False, mtp=False):
    return MLATransformerConfig(
        num_layers=2,
        hidden_size=HIDDEN,
        ffn_hidden_size=12288,
        num_attention_heads=64,
        num_query_groups=64,
        kv_channels=256,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_head_dim=256,
        qk_pos_emb_head_dim=0,
        v_head_dim=256,
        qk_layernorm=True,
        normalization='RMSNorm',
        layernorm_epsilon=1e-5,
        attention_latent_norm_epsilon=1e-5,
        rmsnorm_cast_before_weight=True,
        activation_func=F.silu,
        gated_linear_unit=True,
        activation_func_clamp_value=10.0,
        bias_activation_fusion=False,
        use_te_activation_func=False,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=False,
        gradient_accumulation_fusion=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        attention_backend=AttnBackend.unfused if backend == 'none' else AttnBackend.auto,
        dsa_kernel_backend=backend,
        experimental_attention_variant='dsa',
        dsa_indexer_n_heads=32,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=2048,
        dsa_indexer_kpool=4,
        dsa_indexer_kpool_use_quantization=False,
        dsa_indexer_kpool_always_select_tail=True,
        dsa_indexer_kpool_key_dtype_compression=True,
        dsa_indexer_hf_scoring=True,
        dsa_indexer_loss_coeff=0.0,
        dsa_indexer_k_norm_epsilon=1e-6,
        dsa_indexer_scoring_relu=True,
        dsa_indexer_rotate_activation=False,
        dsa_indexer_qk_proj_use_quantization=False,
        dsa_indexer_weights_proj_use_quantization=False,
        dsa_indexer_weights_proj_output_dtype='bf16',
        mla_proj_use_quantization=False,
        rope_type='rope',
        rotary_scaling_factor=1.0,
        mscale_all_dim=0.0,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=64,
        linear_num_value_heads=64,
        kda_two_stage_gates=True,
        kda_disable_fp8=True,
        kda_conv_in_fp32=True,
        kda_core_in_fp32=True,
        kda_safe_gate=True,
        kda_lower_bound=-5.0,
        deterministic_mode=deterministic,
        enable_hyper_connections=True,
        num_residual_streams=4,
        mhc_sinkhorn_iterations=20,
        mhc_norm_eps_inside_sqrt=True,
        mhc_keep_mappings_in_fp32=False,
        mhc_keep_pre_in_fp32=True,
        mhc_preserve_activation_dtype_rounding=True,
        mhc_learned_output_contract=False,
        num_moe_experts=288 if sparse else None,
        moe_ffn_hidden_size=2048 if sparse else None,
        moe_shared_expert_intermediate_size=2048 if sparse else None,
        moe_grouped_gemm=sparse,
        moe_use_grouped_tensor=False,
        moe_single_grouped_weight=False,
        use_transformer_engine_op_fuser=False,
        moe_permute_fusion=False,
        moe_router_fusion=False,
        moe_router_topk=8,
        moe_router_score_function='sigmoid',
        moe_router_dtype='fp32',
        moe_apply_probs_after_fc2=sparse,
        moe_router_num_groups=1 if sparse else None,
        moe_router_group_topk=1 if sparse else None,
        moe_router_topk_scaling_factor=2.5,
        moe_router_enable_expert_bias=True,
        moe_router_bias_update_rate=0.0,
        moe_router_load_balancing_type='none',
        moe_aux_loss_coeff=0.0,
        moe_token_dispatcher_type='alltoall',
        mtp_num_layers=1 if mtp else None,
        mtp_loss_scaling_factor=0.1,
        mtp_loss_type='cross_entropy',
        mtp_detach_heads=False,
        mtp_enable_hyper_connections=False,
        calculate_per_token_loss=False,
        cross_entropy_loss_fusion=False,
        is_hybrid_model=True,
        transformer_impl='transformer_engine',
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
    )


def process_groups():
    return ProcessGroupCollection.use_mpu_process_groups()


def build_stack(config, pattern, *, expand=False):
    stack = HybridStack(
        config=config,
        submodules=hybrid_glm53_stack_spec(config).submodules,
        layer_type_list=validate_segment_layers(pattern),
        pre_process=expand,
        post_process=True,
        post_layer_norm=True,
        pg_collection=process_groups(),
    ).cuda()
    convert_module_to_dtype_except_fp32_marked(stack, torch.bfloat16)
    return stack


def _record_mhc_precision(module):
    """Observe real configuration/dispatch without changing fixture math or RNG."""
    config = next(child.config for child in module.modules() if hasattr(child, 'config'))
    mhc_modules = [
        (name, child)
        for name, child in module.named_modules()
        if isinstance(child, HyperConnectionModule)
    ]
    assert config.mhc_preserve_activation_dtype_rounding
    assert config.kda_core_in_fp32
    assert all(
        child.config.mhc_preserve_activation_dtype_rounding
        and child._h_post_bda_op is _get_precision_preserving_h_post_bda()
        for _, child in mhc_modules
    )
    record_details(
        'mhc_precision',
        {
            'mhc_preserve_activation_dtype_rounding': config.mhc_preserve_activation_dtype_rounding,
            'compile_options': {'emulate_precision_casts': True},
            'applicable': bool(mhc_modules),
            'module_count': len(mhc_modules),
            'actual_operator_selection_verified': bool(mhc_modules),
            'module_names': [name for name, _ in mhc_modules],
            'not_applicable_reason': None if mhc_modules else 'single-stream MTP has no mHC module',
            'global_matmul_precision_unchanged': torch.get_float32_matmul_precision(),
            'kda_core_in_fp32_requested': config.kda_core_in_fp32,
        },
    )


def _record_moe_probability_placement(module):
    """Record real expert calls without retaining tensors or changing their graph."""
    config = next(child.config for child in module.modules() if hasattr(child, 'config'))
    requested = config.moe_apply_probs_after_fc2
    assert type(requested) is bool
    experts = [
        (name, child) for name, child in module.named_modules() if type(child) is TEGroupedMLP
    ]
    assert bool(experts) is requested
    for _, child in experts:
        assert child.config.moe_apply_probs_after_fc2 is True
        assert child._with_fused_impl is False and child._use_grouped_tensor is False
    profile = {
        'requested': requested,
        'applicable': bool(experts),
        'module_count': len(experts),
        'module_names': [name for name, _ in experts],
        'forward_calls': [],
        'not_applicable_reason': None if experts else 'no routed experts in this case',
    }
    # record_details stores this same dictionary; actual hooks append before publish.
    record_details('moe_probability_placement', profile)

    def observe(child, inputs, kwargs, output, *, module_name):
        hidden = inputs[0] if inputs else kwargs['permuted_local_hidden_states']
        probs = inputs[2] if len(inputs) > 2 else kwargs['permuted_probs']
        assert type(child) is TEGroupedMLP
        assert child.config.moe_apply_probs_after_fc2 is True
        assert child._with_fused_impl is False and child._use_grouped_tensor is False
        assert isinstance(hidden, torch.Tensor) and isinstance(probs, torch.Tensor)
        assert hidden.dtype == torch.bfloat16 and probs.dtype == torch.float32
        assert hidden.ndim == 2 and hidden.shape[1] == HIDDEN
        assert probs.ndim == 1 and probs.shape[0] == hidden.shape[0]
        assert isinstance(output, tuple) and len(output) == 2
        result, bias = output
        assert isinstance(result, torch.Tensor) and bias is None
        assert result.dtype == torch.bfloat16 and result.shape == hidden.shape
        assert hidden.device == probs.device == result.device
        profile['forward_calls'].append(
            {
                'module_name': module_name,
                'requested': child.config.moe_apply_probs_after_fc2,
                'input_dtype': str(hidden.dtype),
                'prob_dtype': str(probs.dtype),
                'output_dtype': str(result.dtype),
                'input_shape': list(hidden.shape),
                'prob_shape': list(probs.shape),
                'output_shape': list(result.shape),
                'fused_impl': child._with_fused_impl,
                'grouped_tensor': child._use_grouped_tensor,
                'bias_is_none': bias is None,
            }
        )
        # None preserves the original output object and all original autograd edges.

    for name, child in experts:

        def hook(child, inputs, kwargs, output, module_name=name):
            return observe(child, inputs, kwargs, output, module_name=module_name)

        child.register_forward_hook(hook, with_kwargs=True)


def kda_links(native, actual):
    names = {
        'conv1d.weight': 'conv1d.weight',
        'forget_gate.f_a_proj.weight': 'f_a_proj.weight',
        'forget_gate.f_b_proj.weight': 'f_b_proj.weight',
        'forget_gate.dt_bias': 'dt_bias',
        'forget_gate.A_log': 'A_log',
        'b_proj.weight': 'beta_proj.weight',
        'g_a_proj.weight': 'g_a_proj.weight',
        'g_b_proj.weight': 'g_b_proj.weight',
        'o_norm.weight': 'out_norm.weight',
        'o_proj.weight': 'out_proj.weight',
    }
    links = [
        ParameterLink(f'{native}.{left}', f'{actual}.{right}') for left, right in names.items()
    ]
    for index, name in enumerate(('q_proj', 'k_proj', 'v_proj')):
        links.append(
            ParameterLink(
                f'{native}.{name}.weight',
                f'{actual}.in_proj.weight',
                actual_slice=slice(index * KDA_WIDTH, (index + 1) * KDA_WIDTH),
            )
        )
    return links


def mhc_links(native, actual):
    links = [
        ParameterLink(f'{native}.fn', f'{actual}.mapping_proj.weight'),
        ParameterLink(f'{native}.base', f'{actual}.bias'),
    ]
    for index, kind in enumerate(('pre', 'post', 'res')):
        links.append(
            ParameterLink(
                f'{native}.scale', f'{actual}.alpha_{kind}', native_slice=slice(index, index + 1)
            )
        )
    return links


def ffn_links(native, actual, width):
    return [
        ParameterLink(
            f'{native}.gate_proj.weight',
            f'{actual}.linear_fc1.weight',
            actual_slice=slice(0, width),
        ),
        ParameterLink(
            f'{native}.up_proj.weight',
            f'{actual}.linear_fc1.weight',
            actual_slice=slice(width, 2 * width),
        ),
        ParameterLink(f'{native}.down_proj.weight', f'{actual}.linear_fc2.weight'),
    ]


def dsa_links(native, actual):
    names = {
        'q_a_proj.weight': 'linear_q_down_proj.weight',
        'q_a_layernorm.weight': 'q_layernorm.weight',
        'q_b_proj.weight': 'linear_q_up_proj.weight',
        'kv_a_proj_with_mqa.weight': 'linear_kv_down_proj.weight',
        'kv_a_layernorm.weight': 'kv_layernorm.weight',
        'kv_b_proj.weight': 'linear_kv_up_proj.weight',
        'o_proj.weight': 'linear_proj.weight',
        'indexer.wq_b.weight': 'core_attention.indexer.linear_wq_b.weight',
        'indexer.wk.weight': 'core_attention.indexer.linear_wk.weight',
        'indexer.k_norm.weight': 'core_attention.indexer.k_norm.weight',
        'indexer.k_norm.bias': 'core_attention.indexer.k_norm.bias',
        'indexer.weights_proj.weight': 'core_attention.indexer.linear_weights_proj.weight',
        'indexer.index_kpool_compress_gate': 'core_attention.indexer.index_kpool_compress_gate',
        'indexer.index_kpool_compress_ape': 'core_attention.indexer.index_kpool_compress_ape',
    }
    return [ParameterLink(f'{native}.{left}', f'{actual}.{right}') for left, right in names.items()]


def moe_links(native, actual):
    links = [ParameterLink(f'{native}.gate.weight', f'{actual}.router.weight')]
    links.extend(ffn_links(f'{native}.shared_experts', f'{actual}.shared_experts', 2048))
    # The declared non-grouped-storage TEGroupedMLP still executes real grouped
    # GEMMs. Numbered per-expert storage avoids a huge reference scatter gradient.
    for expert in range(288):
        prefix = f'{native}.experts.{expert}'
        links.extend(
            [
                ParameterLink(
                    f'{prefix}.gate_proj.weight',
                    f'{actual}.experts.linear_fc1.weight{expert}',
                    actual_slice=slice(0, 2048),
                ),
                ParameterLink(
                    f'{prefix}.up_proj.weight',
                    f'{actual}.experts.linear_fc1.weight{expert}',
                    actual_slice=slice(2048, 4096),
                ),
                ParameterLink(
                    f'{prefix}.down_proj.weight', f'{actual}.experts.linear_fc2.weight{expert}'
                ),
            ]
        )
    return links


def block_links(*, sparse=False):
    attention = 'layers.0.inner_layer'
    ffn = 'layers.1.inner_layer'
    return [
        *mhc_links('attn_hc', 'layers.0.hyper_connection'),
        *mhc_links('ffn_hc', 'layers.1.hyper_connection'),
        ParameterLink('input_layernorm.weight', f'{attention}.input_layernorm.weight'),
        ParameterLink('post_attention_layernorm.weight', f'{ffn}.pre_mlp_layernorm.weight'),
        *(
            dsa_links('self_attn', f'{attention}.self_attention')
            if sparse
            else kda_links('self_attn', f'{attention}.self_attention')
        ),
        *(moe_links('mlp', f'{ffn}.mlp') if sparse else ffn_links('mlp', f'{ffn}.mlp', 12288)),
        ParameterLink('final_norm.weight', 'final_norm.weight'),
    ]


@torch.no_grad()
def initialize_fixture(module):
    """Shared deterministic weights; nonunit norm weights expose HF cast boundaries."""
    _record_mhc_precision(module)
    _record_moe_probability_placement(module)
    for name, parameter in module.named_parameters():
        if 'hyper_connection' in name:
            if 'alpha_' in name:
                parameter.fill_(0.2)
            elif name.endswith('.bias'):
                parameter.uniform_(-0.2, 0.2)
            else:
                parameter.normal_(std=0.01)
        elif name.endswith('dt_bias'):
            parameter.uniform_(-2.0, -0.5)
        elif name.endswith('A_log'):
            parameter.uniform_(-1.0, 0.5)
        elif any(
            key in name
            for key in (
                'layernorm.weight',
                'final_norm.weight',
                'out_norm.weight',
                'k_norm.weight',
                'enorm.weight',
                'hnorm.weight',
            )
        ):
            parameter.uniform_(0.8, 1.2)
        else:
            parameter.normal_(std=0.02)
    # A few explicit gate/up rows exercise both clamp branches without changing
    # the target activation. Most rows remain unclamped and differentiable.
    for name, parameter in module.named_parameters():
        if name.endswith('mlp.linear_fc1.weight'):
            parameter[:3].normal_(std=0.5)
            half = parameter.size(0) // 2
            parameter[half : half + 3].normal_(std=0.5)
        elif '.experts.linear_fc1.weight' in name:
            parameter[:3].normal_(std=0.5)
            parameter[2048:2051].normal_(std=0.5)


@contextmanager
def observe_dsa(monkeypatch):
    """Observe real dispatch and indices, returning every original result unchanged."""
    records = {'indices': [], 'fused': []}
    sparse = dsa_module._run_sparse_attention
    fused = dsa_kernels.run_fused_absorbed_sparse_attention

    def capture_indices(**kwargs):
        records['indices'].append(kwargs['topk_indices'].detach().cpu())
        return sparse(**kwargs)

    def capture_fused(*args, **kwargs):
        output = fused(*args, **kwargs)
        records['fused'].append(output is not None)
        return output

    with monkeypatch.context() as patch:
        patch.setattr(dsa_module, '_run_sparse_attention', capture_indices)
        patch.setattr(dsa_kernels, 'run_fused_absorbed_sparse_attention', capture_fused)
        yield records


@contextmanager
def observe_kda_convolution(monkeypatch, deterministic):
    """Observe the real convolution dispatch, without changing its calculation."""
    counts = {'fla': 0, 'torch': 0}
    fused = kda_module.causal_conv1d
    native = kda_module.KimiDeltaAttention._torch_causal_conv1d

    def capture_fused(*args, **kwargs):
        counts['fla'] += 1
        assert fused is not None
        return fused(*args, **kwargs)

    def capture_native(module, *args, **kwargs):
        counts['torch'] += 1
        return native(module, *args, **kwargs)

    with monkeypatch.context() as patch:
        if fused is not None:
            patch.setattr(kda_module, 'causal_conv1d', capture_fused)
        patch.setattr(kda_module.KimiDeltaAttention, '_torch_causal_conv1d', capture_native)
        yield
    assert counts == {'fla': int(not deterministic), 'torch': int(deterministic)}, counts
    record_details('kda_convolution_dispatch', counts)


@contextmanager
def observe_kda_core_precision(module, monkeypatch):
    """Observe the real configured FLA boundary; never alter inputs or output."""
    modules = [
        child for child in module.modules() if isinstance(child, kda_module.KimiDeltaAttention)
    ]
    assert modules and all(child.config.kda_core_in_fp32 for child in modules)
    records = []

    def wrap(method):
        def call(*args, **kwargs):
            assert not args
            assert all(kwargs[key].dtype == torch.float32 for key in ('q', 'k', 'v', 'g', 'beta'))
            assert kwargs['use_qk_l2norm_in_kernel'] is False
            assert kwargs['use_gate_in_kernel'] is False
            assert kwargs.get('use_beta_sigmoid_in_kernel', False) is False
            assert kwargs.get('scale') in (None, kwargs['q'].shape[-1] ** -0.5)
            output = method(*args, **kwargs)
            assert output[0].dtype == torch.float32
            records.append(
                {
                    'input_dtypes': {
                        key: str(kwargs[key].dtype) for key in ('q', 'k', 'v', 'g', 'beta')
                    },
                    'raw_output_dtype': str(output[0].dtype),
                    'internal_qk_norm': kwargs['use_qk_l2norm_in_kernel'],
                    'scale': kwargs.get('scale'),
                }
            )
            return output

        return call

    with monkeypatch.context() as patch:
        for child in modules:
            patch.setattr(child, 'gated_delta_rule', wrap(child.gated_delta_rule))
        yield
    assert len(records) == len(modules), (records, len(modules))
    record_details('kda_core_precision', records)


def record_router(router, records):
    def capture(_module, _inputs, output):
        probabilities, routing = output
        assert routing.dtype == torch.bool and routing.shape[-1] == 288
        assert routing.sum(-1).eq(8).all()
        counts = routing.sum(0)
        assert counts.gt(0).all(), f'all288 experts must execute: {counts.cpu().tolist()}'
        records.append((probabilities.detach().cpu(), routing.detach().cpu()))

    return router.register_forward_hook(capture)


def assert_sparse_observations(records, router_records, native_attention, native_moe, backend):
    assert len(records['indices']) == 1
    actual = records['indices'][0].sort(-1).values
    expected = native_attention.last_indices.cpu().sort(-1).values
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert (actual[:, -1] >= 0).sum(-1).eq(130).all(), 'retain the incomplete two-token pool tail'
    assert records['fused'] == ([True] if backend == 'cudnn' else [])
    assert len(router_records) == 1
    probabilities, routing = router_records[0]
    torch.testing.assert_close(routing, native_moe.last_routing.cpu(), rtol=0, atol=0)
    assert_similarity(probabilities, native_moe.last_probabilities.cpu(), 'router probabilities')
    record_details(
        'sparse_dispatch',
        {
            'backend': backend,
            'optimized_backend_meaning': (
                'FlashMLA forward + cuDNN backward'
                if backend == 'cudnn'
                else 'unfused absorbed attention'
            ),
            'fused_returned_non_none': records['fused'],
            'selected_token_set_exact': True,
            'selector_shape': list(actual.shape),
            'last_query_visible_tokens': (actual[:, -1] >= 0).sum(-1).tolist(),
            'router_exact': True,
            'expert_token_counts': routing.sum(0).tolist(),
        },
    )


@contextmanager
def observe_clamps(model, values=None):
    """Count real FC1 clamp categories; values, if given, receives each site's FC1 output."""
    records, handles = [], []

    def capture(module_name, output):
        value = output[0] if isinstance(output, tuple) else output
        if values is not None:
            assert module_name not in values, f'{module_name}: one FC1 call per forward'
            values[module_name] = value.detach().clone()
        records.append(clamp_category_counts(*value.chunk(2, -1)).cpu())

    for name, module in model.named_modules():
        if name.endswith('linear_fc1') and ('.mlp.' in name or name.startswith('mlp.')):
            handles.append(
                module.register_forward_hook(
                    lambda _module, _inputs, output, name=name: capture(name, output)
                )
            )
    assert handles, 'must observe real FFN FC1 outputs'
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def observe_expert_dispatch(model):
    """Record each routed-expert call's token counts and permuted probabilities."""
    records, handles = {}, []

    def capture(module_name, inputs, kwargs):
        tokens = inputs[1] if len(inputs) > 1 else kwargs['tokens_per_expert']
        probs = inputs[2] if len(inputs) > 2 else kwargs['permuted_probs']
        assert module_name not in records, f'{module_name}: one expert call per forward'
        tokens = tokens.tolist() if torch.is_tensor(tokens) else list(tokens)
        records[module_name] = {
            'tokens_per_expert': [int(value) for value in tokens],
            'permuted_probs': probs.detach().cpu(),
        }

    for name, module in model.named_modules():
        if type(module) is TEGroupedMLP:
            handles.append(
                module.register_forward_pre_hook(
                    lambda _module, inputs, kwargs, name=name: capture(name, inputs, kwargs),
                    with_kwargs=True,
                )
            )
    assert handles, 'must observe real routed-expert calls'
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def observe_mhc_projections(model, monkeypatch):
    """Capture each mHC projection (the _compute_h input) and its gradient.

    The uncompiled class-level _projection_and_get_norm is wrapped; values and
    the compiled _compute_h are unchanged.
    """
    names = {
        id(child): name
        for name, child in model.named_modules()
        if isinstance(child, HyperConnectionModule)
    }
    assert names, 'must observe real mHC mappings'
    for child in model.modules():
        if id(child) in names:
            assert child._proj_rms_compute_h_op is None, f'{names[id(child)]}: native path'
    captures = {}
    original = HyperConnectionModule._projection_and_get_norm

    def projection_and_get_norm(self, x):
        proj, r = original(self, x)
        name = names.get(id(self))
        if name is not None:
            assert name not in captures, f'{name}: one mapping per forward'
            entry = captures[name] = {'proj': proj.detach().clone(), 'grad': None}

            def keep(gradient):
                assert entry['grad'] is None, name
                entry['grad'] = gradient.detach().clone()

            proj.register_hook(keep)
        return proj, r

    with monkeypatch.context() as patch:
        patch.setattr(HyperConnectionModule, '_projection_and_get_norm', projection_and_get_norm)
        yield captures
    assert set(captures) == set(names.values()), sorted(captures)


def bind_moe_replay(replay, prefix, native_moe, fc1, dispatch, router_records):
    """Bind one MoE layer's shared and routed FC1 outputs to the reference."""
    assert len(router_records) == 1
    probabilities, routing = router_records[0]
    replay.bind(
        f'{prefix}.shared_experts',
        native_moe.shared_experts,
        fc1.pop(f'{prefix}.shared_experts.linear_fc1'),
    )
    calls = dispatch.pop(f'{prefix}.experts')
    replay.bind_routed(
        f'{prefix}.experts',
        native_moe,
        fc1.pop(f'{prefix}.experts.linear_fc1'),
        routing,
        calls['tokens_per_expert'],
        calls['permuted_probs'],
        probabilities,
    )


def assert_clamp_coverage(actual, native):
    observed = torch.stack(actual).sum(0)
    expected = torch.stack(
        [module.clamp_counts.cpu() for module in native.modules() if isinstance(module, NativeFFN)]
    ).sum(0)
    assert observed.gt(0).all(), f'target clamp categories unexercised: {observed.tolist()}'
    assert expected.gt(0).all(), f'native clamp categories unexercised: {expected.tolist()}'
    record_details(
        'clamp_categories',
        {
            'order': ['gate_upper', 'up_upper', 'up_lower', 'unclamped'],
            'actual': observed.tolist(),
            'native': expected.tolist(),
            'note': 'Category coverage; full tensor/gradient parity is checked separately.',
        },
    )


def release_cuda():
    gc.collect()
    torch.cuda.empty_cache()


def capture_stream_wire(stack, values):
    """Read-only real block observation before final mean hides residual mixing."""

    def capture(_module, _inputs, output):
        assert isinstance(output, tuple) and output[0].shape[-1] == 4 * HIDDEN
        values.append(output[0])

    return stack.layers[-1].register_forward_hook(capture)


@pytest.fixture(autouse=True)
def _model_parallel(request, record_property):
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(531530)
    torch.manual_seed(531530)
    with ComparisonLedger(request.node.nodeid, int(os.environ.get('RANK', '0'))) as ledger:
        try:
            yield
        finally:
            ledger.publish(record_property)
            Utils.destroy_model_parallel()


def apply_mutation(mutate, point, *args):
    """Diagnostic hook points; production tests pass mutate=None."""
    if mutate is not None and point in mutate:
        mutate[point](*args)


def _run_kda_dense(deterministic, expand, monkeypatch, *, compare_all_parameters, mutate=None):
    assert gdn_common.HAVE_FLA, 'KDA production execution requires the pinned FLA dependency'
    if not deterministic:
        assert gdn_common.causal_conv1d is not None, 'the fused-convolution case must actually run'
    config = make_config(deterministic=deterministic)
    apply_mutation(mutate, 'config', config)
    real = build_stack(config, 'K-', expand=expand)
    initialize_fixture(real)
    native = NativeHybridBlock(torch.device('cuda'), expand=expand)
    links = block_links()
    copy_parameters(real, native, links)
    apply_mutation(mutate, 'native', native)
    buffers = BufferContract(real, native)
    width = HIDDEN if expand else 4 * HIDDEN
    hidden = torch.randn(17, 1, width, device='cuda', dtype=torch.bfloat16)
    actual_input, native_input = (hidden.clone().requires_grad_() for _ in range(2))
    if not expand:
        real.set_input_tensor(actual_input)
    wires, fc1 = [], {}
    hook = capture_stream_wire(real, wires)
    try:
        with (
            observe_clamps(real, fc1) as clamp_records,
            observe_kda_convolution(monkeypatch, deterministic),
            observe_kda_core_precision(real, monkeypatch),
            (
                observe_mhc_projections(real, monkeypatch)
                if compare_all_parameters
                else nullcontext({})
            ) as projections,
        ):
            actual = real(actual_input, attention_mask=None)
    finally:
        hook.remove()
    if compare_all_parameters:
        replay = ClampReplay()
        replay.bind(
            'layers.1.inner_layer.mlp', native.mlp, fc1.pop('layers.1.inner_layer.mlp.linear_fc1')
        )
        assert not fc1, sorted(fc1)
    expected = native(native_input)
    if compare_all_parameters:
        replay.assert_consumed()
    assert actual.shape == (17, 1, HIDDEN)
    assert_similarity(actual, expected, 'K- contracted output')
    assert len(wires) == 1
    assert_similarity(wires[0], native.last_streams, 'K- full stream wire')
    gradient = torch.randn_like(actual)
    wire_gradient = torch.randn_like(wires[0])
    torch.autograd.backward((actual, wires[0]), (gradient, wire_gradient))
    torch.autograd.backward((expected, native.last_streams), (gradient, wire_gradient))
    record_details('objective', 'paired contracted-output and complete-stream-wire random VJPs')
    apply_mutation(mutate, 'actual_gradients', real)
    apply_mutation(mutate, 'native_gradients', native)
    assert_similarity(actual_input.grad, native_input.grad, 'K- hidden gradient')
    if compare_all_parameters:
        decomposed = assert_mhc_decomposition(real, native, projections, BLOCK_MHC_SITES)
        assert_all_gradients(real, native, links, decomposed=decomposed)
    assert_clamp_coverage(clamp_records, native)
    buffers.assert_actual_unchanged(real)
    buffers.assert_native_unchanged(native)


@pytest.mark.internal
@pytest.mark.parametrize('deterministic', [False, True])
def test_glm53_kda_dense_mhc_native_forward_and_all_gradients(deterministic, monkeypatch):
    """Paired output+wire VJPs expose mHC mixing hidden by the final mean head.

    Clamp decisions are replayed and the mHC scales are decomposed (module docstring).
    """
    _run_kda_dense(deterministic, False, monkeypatch, compare_all_parameters=True)


@pytest.mark.internal
def test_glm53_kda_dense_input_expansion_and_contraction(monkeypatch):
    """Normal entry boundary: output/input parity, not all-parameter qualification.

    Repeating the initial stream makes first-site residual mixing unobservable.
    All parameter gradients are instead compared with distinct legal wire inputs
    in both kernel-profile cases above; no gradient assertion is relaxed there.
    """
    _run_kda_dense(True, True, monkeypatch, compare_all_parameters=False)


def _run_dsa_moe(backend, monkeypatch, mutate=None):
    config = make_config(backend=backend, sparse=True)
    apply_mutation(mutate, 'config', config)
    real = build_stack(config, 'DE')
    initialize_fixture(real)
    native = NativeHybridBlock(torch.device('cuda'), sparse=True)
    links = block_links(sparse=True)
    copy_parameters(real, native, links)
    hidden = torch.randn(130, 4, 4 * HIDDEN, device='cuda', dtype=torch.bfloat16)
    router = real.layers[1].inner_layer.mlp.router
    with torch.no_grad():
        bias = torch.linspace(-0.02, 0.02, 288, device='cuda')
        # The production router restores this buffer to FP32 before routing;
        # initialize the fixture after that same dtype boundary, without losing
        # the correction values to a prior blanket BF16 module conversion.
        router.expert_bias.data = router.expert_bias.data.float()
        router.expert_bias.copy_(bias)
        native.mlp.e_score_correction_bias.copy_(bias)
        router.weight.copy_(balanced_router_weight(native.ffn_input(hidden)))
    buffers = BufferContract(
        real,
        native,
        [('layers.1.inner_layer.mlp.router.expert_bias', 'mlp.e_score_correction_bias')],
        ['layers.1.inner_layer.mlp.router.local_tokens_per_expert'],
    )
    del native, router
    release_cuda()
    actual_input = hidden.clone().requires_grad_()
    real.set_input_tensor(actual_input)
    router_records = []
    wires, fc1 = [], {}
    handle = record_router(real.layers[1].inner_layer.mlp.router, router_records)
    wire_hook = capture_stream_wire(real, wires)
    try:
        with (
            observe_dsa(monkeypatch) as records,
            observe_clamps(real, fc1) as clamp_records,
            observe_expert_dispatch(real) as dispatch,
            observe_mhc_projections(real, monkeypatch) as projections,
        ):
            actual = real(actual_input, attention_mask=None)
            gradient = torch.randn_like(actual)
            assert len(wires) == 1
            wire_gradient = torch.randn_like(wires[0])
            torch.autograd.backward((actual, wires[0]), (gradient, wire_gradient))
    finally:
        handle.remove()
        wire_hook.remove()
    apply_mutation(mutate, 'actual_gradients', real)
    output_gold, input_gold = actual.detach().cpu(), actual_input.grad.detach().cpu()
    wire_gold = wires[0].detach().cpu()
    projections = {
        name: {key: value.cpu() for key, value in entry.items()}
        for name, entry in projections.items()
    }
    buffers.assert_actual_unchanged(real, router_records)
    gold = HostParameters(real)
    wires.clear()
    del real, actual, actual_input
    release_cuda()
    native = NativeHybridBlock(torch.device('cuda'), sparse=True)
    copy_parameters(gold, native, links)
    buffers.copy_native(gold, native)
    apply_mutation(mutate, 'native', native)
    replay = ClampReplay()
    bind_moe_replay(replay, 'layers.1.inner_layer.mlp', native.mlp, fc1, dispatch, router_records)
    assert not fc1 and not dispatch, (sorted(fc1), sorted(dispatch))
    native_input = hidden.clone().requires_grad_()
    expected = native(native_input)
    replay.assert_consumed()
    assert_similarity(output_gold, expected, 'DE contracted output')
    assert_similarity(wire_gold, native.last_streams, 'DE full stream wire')
    torch.autograd.backward((expected, native.last_streams), (gradient, wire_gradient))
    record_details('objective', 'paired contracted-output and complete-stream-wire random VJPs')
    apply_mutation(mutate, 'native_gradients', native)
    assert_similarity(input_gold, native_input.grad, 'DE hidden gradient')
    decomposed = assert_mhc_decomposition(gold, native, projections, BLOCK_MHC_SITES)
    assert_all_gradients(gold, native, links, decomposed=decomposed)
    assert_sparse_observations(records, router_records, native.self_attn, native.mlp, backend)
    assert_clamp_coverage(clamp_records, native)
    buffers.assert_native_unchanged(native)


@pytest.mark.internal
@pytest.mark.parametrize('backend', ['none', 'cudnn'])
def test_glm53_dsa_moe_mhc_native_forward_and_all_gradients(backend, monkeypatch):
    """Full288E DE, with sequential CUDA implementations and host gold gradients.

    Clamp decisions are replayed and the mHC scales are decomposed (module docstring).
    """
    _run_dsa_moe(backend, monkeypatch)
