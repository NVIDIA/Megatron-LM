# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Selective and full Hybrid recompute with shared CSA2 state and outstanding microbatches.

Native CPU adapters exercise the production stack, state, attention and checkpoint
engine. Pipeline cases simulate chunk handoff without claiming NCCL/TE coverage.
"""

from argparse import ArgumentParser
from contextlib import contextmanager
from dataclasses import replace

import pytest
import torch

from megatron.core.enums import Fp8Recipe
from megatron.core.extensions import transformer_engine as te_runtime
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.hybrid import hybrid_block as hybrid_runtime
from megatron.core.models.hybrid import hybrid_stack_adapter as recompute_runtime
from megatron.core.tensor_parallel import random as checkpoint_runtime
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import (
    deepseek_v4_hybrid_attention as dsv4_attention,
)
from megatron.core.transformer.experimental_attention_variant.csa import (
    CompressedSparseAttentionSubmodules,
    CompressorSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.csa2 import (
    CompressedSparseAttention2,
    CSA2Compressor,
    CSA2Indexer,
    CSA2IndexerSubmodules,
    CSA2State,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_pipeline import (
    build_csa2_pipeline_plan,
)
from megatron.core.transformer.hyper_connection import HyperConnectionModule, SinglePassMHCState
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_layer import (
    HyperConnectionTransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.training.arguments import _add_network_size_args
from tests.unit_tests.tensor_parallel.test_boundary_checkpoint import (
    cpu_te_checkpoint as cpu_te_checkpoint,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2 import (
    _CPUFrequencyTable,
    _groups,
    _Linear,
    _packed,
    _record_losses,
    _RMSNorm,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2_pipeline import (
    _FFN,
    _Attention,
    _config,
    _pattern,
    _run_chunks,
    _stack,
)


@pytest.fixture
def cpu_checkpoint_rng(monkeypatch):
    """Keep real CPU RNG snapshot/restore on hosts without a CUDA generator."""
    if not torch.cuda.is_available():
        monkeypatch.setattr(checkpoint_runtime, "_get_cuda_rng_state", lambda **kwargs: None)
        monkeypatch.setattr(checkpoint_runtime, "_set_cuda_rng_state", lambda *args, **kwargs: None)


@pytest.fixture
def checkpoints(monkeypatch, cpu_checkpoint_rng):
    """Observe the real manager's storage discard and replay, without replacing either."""
    managers = []
    original_init = checkpoint_runtime.MHCCheckpointManager.__init__

    def record_init(manager):
        original_init(manager)
        managers.append(manager)

    monkeypatch.setattr(checkpoint_runtime.MHCCheckpointManager, "__init__", record_init)
    return managers


def _recompute_config(config, group_size):
    return replace(
        config,
        recompute_granularity="selective",
        recompute_modules=["mhc"],
        mhc_recompute_layer_num=group_size,
    )


def _stacks(reference, config, cuts, layout, attention=_Attention):
    plan = build_csa2_pipeline_plan(config, _pattern(cuts), qkv_format=layout)
    stacks = [_stack(config, chunk, attention=attention) for chunk in plan]
    for stack, chunk in zip(stacks, plan):
        for local, layer in enumerate(stack.layers):
            layer.load_state_dict(reference.layers[chunk.layer_offset + local].state_dict())
    return stacks, plan


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["sbhd", "thd", "short"])
@pytest.mark.parametrize("coefficient", [0.0, 0.3])
@pytest.mark.parametrize("group_size", [None, 1, 3])
def test_recompute_shared_state_and_microbatches(
    monkeypatch, checkpoints, dtype, layout, coefficient, group_size
):
    """Replay must preserve every input/parameter gradient and log the aux loss only once."""
    torch.manual_seed(391)
    records = _record_losses(monkeypatch)
    reference = _stack(_config(dtype, coefficient))
    config = _recompute_config(reference.config, group_size)
    stacks, plan = _stacks(
        reference, config, (1, 5, 7, 8, 10), "thd" if layout == "thd" else "sbhd"
    )
    params = None
    shape = (1, 1) if layout == "short" else (9, 2)
    if layout == "thd":
        params, _, valid = _packed([1, 4, 0, 2], [2, 5, 0, 4], tail=3)
        shape = (valid.numel(), 1)
    pairs = []
    for _ in range(2):
        x = torch.randn(*shape, config.hidden_size, dtype=dtype, requires_grad=True)
        ref_x = x.detach().clone().requires_grad_()
        expected = reference(ref_x, None, packed_seq_params=params)
        actual, payloads = _run_chunks(stacks, plan, x, params)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        # Real output storage has been freed, but live PP state must remain readable.
        assert checkpoints and all(m._outputs_discarded for m in checkpoints)
        assert all(
            output.untyped_storage().nbytes() == 0
            for manager in checkpoints
            for checkpoint in manager.checkpoints
            for output in checkpoint.outputs
        )
        for payload in payloads:
            for tensor in payload.tensors:
                assert tensor.numel() == 0 or tensor.untyped_storage().nbytes() > 0
        pairs.append((actual, expected, x, ref_x))
    for actual, expected, x, ref_x in reversed(pairs):
        probe = torch.randn_like(actual) * 0.1
        (actual * probe).sum().backward()
        (expected * probe).sum().backward()
        torch.testing.assert_close(x.grad, ref_x.grad, atol=2e-7, rtol=2e-5)
    for stack, chunk in zip(stacks, plan):
        for local, layer in enumerate(stack.layers):
            ref_params = dict(reference.layers[chunk.layer_offset + local].named_parameters())
            for name, param in layer.named_parameters():
                tolerance = (
                    dict(atol=2e-7, rtol=2e-5)
                    if dtype == torch.float32
                    else dict(atol=2e-3, rtol=2e-2)
                )
                torch.testing.assert_close(param.grad, ref_params[name].grad, **tolerance, msg=name)
            assert layer._mhc_recompute_manager is None
    assert all(m._recomputed for m in checkpoints)
    assert all(c.ctx is None and c.outputs is None for m in checkpoints for c in m.checkpoints)
    assert len(records) == (12 if coefficient else 0)


@pytest.mark.parametrize("field", ["pre_mix", "global_kv", "indexer_k"])
@pytest.mark.parametrize("modules", [["mhc"], ["layernorm"]])
def test_recompute_when_only_shared_output_receives_gradient(
    monkeypatch, checkpoints, field, modules
):
    """A side-output-only objective never reaches the hidden-state recompute hook."""
    _record_losses(monkeypatch)
    torch.manual_seed(182)
    config = _config()
    reference = _stack(config)
    recompute_config = replace(config, recompute_granularity="selective", recompute_modules=modules)
    stacks, _ = _stacks(reference, recompute_config, (7,), "sbhd")
    ref_stacks, _ = _stacks(reference, config, (7,), "sbhd")
    x = torch.randn(9, 2, config.hidden_size, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    actual = stacks[0](x, None)
    expected = ref_stacks[0](ref_x, None)
    index = tuple(s.name for s in actual.tensor_specs).index(field)
    actual.tensors[index].square().sum().backward()
    expected.tensors[index].square().sum().backward()
    assert all(manager._recomputed for manager in checkpoints)
    torch.testing.assert_close(x.grad, ref_x.grad)
    for param, ref_param in zip(stacks[0].parameters(), ref_stacks[0].parameters()):
        torch.testing.assert_close(param.grad, ref_param.grad)


@pytest.mark.parametrize("group_size", [None, 1, 3])
def test_recompute_preserves_dropout_rng(monkeypatch, checkpoints, group_size):
    _record_losses(monkeypatch)
    torch.manual_seed(816)
    config = replace(_config(), hidden_dropout=0.3)
    reference = _stack(config)
    actual = _stack(_recompute_config(config, group_size))
    actual.load_state_dict(reference.state_dict())
    x = torch.randn(9, 2, config.hidden_size, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    state = torch.get_rng_state()
    expected = reference(ref_x, None)
    torch.set_rng_state(state)
    output = actual(x, None)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    after_forward = torch.get_rng_state()
    output.square().sum().backward()
    assert torch.equal(torch.get_rng_state(), after_forward)
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad)
    for param, ref_param in zip(actual.parameters(), reference.parameters()):
        torch.testing.assert_close(param.grad, ref_param.grad)


def test_no_grad_forward_does_not_discard_or_retain_checkpoints(monkeypatch, checkpoints):
    _record_losses(monkeypatch)
    config = _recompute_config(_config(), None)
    stack = _stack(config)
    with torch.no_grad():
        output = stack(torch.randn(9, 2, config.hidden_size), None)
    assert torch.isfinite(output).all()
    assert not checkpoints


@pytest.mark.parametrize("module", ["core_attn", "mlp"])
def test_unsupported_recompute_is_rejected(module):
    with pytest.raises(ValueError, match="activation recomputation"):
        replace(_config(), recompute_granularity="selective", recompute_modules=[module])


@pytest.mark.parametrize(
    "modules",
    [
        ["layernorm"],
        ["mla_up_proj"],
        ["moe_act"],
        ["moe"],
        ["shared_experts"],
        ["mhc", "layernorm", "mla_up_proj", "moe_act", "moe", "shared_experts"],
    ],
)
@pytest.mark.parametrize("ep_size, etp_size", [(1, 1), (2, 1), (4, 1), (1, 2), (2, 2)])
def test_selective_modules_and_expert_parallelism_use_existing_dsv4_configuration(
    modules, ep_size, etp_size
):
    config = replace(
        _config(),
        recompute_granularity="selective",
        recompute_modules=modules,
        moe_grouped_gemm=True,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=etp_size,
    )
    assert config.recompute_modules == modules
    assert config.expert_model_parallel_size == ep_size
    assert config.expert_tensor_parallel_size == etp_size
    assert config.tensor_model_parallel_size == 1


@pytest.mark.parametrize("recipe", ["delayed", "tensorwise", "mxfp8", "blockwise"])
@pytest.mark.parametrize("modules", [[], ["mhc"], ["mla_up_proj"], ["layernorm"], ["moe_act"]])
def test_fp8_keeps_common_dsv4_recipe_validation(monkeypatch, recipe, modules):
    # Exercise configuration only; this deliberately does not claim TE/GPU execution.
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_config.is_te_min_version", lambda *args: True
    )
    options = dict(
        fp8="e4m3",
        fp8_recipe=recipe,
        moe_grouped_gemm=True,
        recompute_granularity="selective" if modules else None,
        recompute_modules=modules,
    )
    if recipe == "delayed" and set(modules) & {"layernorm", "moe_act"}:
        with pytest.raises(ValueError, match="Delayed scaling does not support"):
            replace(_config(torch.bfloat16), **options)
    else:
        config = replace(_config(torch.bfloat16), **options)
        assert config.fp8 == "e4m3"


def test_gpt_stack_rejects_single_pass_recompute_without_state_adapter():
    with pytest.raises(ValueError, match="requires HybridModel"):
        TransformerBlock(_recompute_config(_config(), None), spec=None)


def test_recompute_cli_configuration():
    parser = ArgumentParser()
    _add_network_size_args(parser)
    args = parser.parse_args(
        [
            "--recompute-granularity",
            "selective",
            "--recompute-modules",
            "mhc",
            "mla_up_proj",
            "--mhc-recompute-layer-num",
            "3",
        ]
    )
    config = replace(
        _config(),
        multi_latent_attention=True,
        recompute_granularity=args.recompute_granularity,
        recompute_modules=args.recompute_modules,
        mhc_recompute_layer_num=args.mhc_recompute_layer_num,
    )
    assert config.recompute_modules == ["mhc", "mla_up_proj"]
    assert config.mhc_recompute_layer_num == 3


class _NativeDSv4Attention(dsv4_attention.DSv4HybridSelfAttention):
    """Execute production attention/QKV/recompute with ordinary PyTorch linear modules."""

    def __init__(self, config, layer_number, pg_collection, **kwargs):
        super().__init__(
            config,
            dsv4_attention.DSv4HybridSelfAttentionSubmodules(
                q_layernorm=_RMSNorm,
                kv_layernorm=_RMSNorm,
                linear_q_down_proj=_Linear,
                linear_q_up_proj=_Linear,
                linear_kv_proj=_Linear,
                linear_proj=_Linear,
                core_attention=ModuleSpec(
                    CompressedSparseAttention2,
                    submodules=CompressedSparseAttentionSubmodules(
                        compressor=ModuleSpec(
                            CSA2Compressor,
                            submodules=CompressorSubmodules(_Linear, _Linear, _RMSNorm),
                        ),
                        indexer=ModuleSpec(
                            CSA2Indexer,
                            submodules=CSA2IndexerSubmodules(_Linear, _Linear, _RMSNorm, _Linear),
                        ),
                    ),
                ),
            ),
            layer_number,
            attn_mask_type=AttnMaskType.causal,
            pg_collection=pg_collection,
        )


@pytest.fixture
def native_attention(monkeypatch):
    """Adapt GPU frequency-table allocation and the TE constructor gate, preserving math."""

    class RotaryTable(_CPUFrequencyTable):
        def __init__(self, dim, **kwargs):
            super().__init__(dim)

        def get_rotary_seq_len(self, inference_context, transformer, hidden, config, params):
            return hidden.shape[0] if params is None else params.max_seqlen_q

    class YarnTable(RotaryTable):
        def forward(self, length, packed_seq=False):
            return super().forward(length, packed_seq=packed_seq), 1.0

    monkeypatch.setattr(dsv4_attention, "TELinear", _Linear)
    monkeypatch.setattr(dsv4_attention, "RotaryEmbedding", RotaryTable)
    monkeypatch.setattr(dsv4_attention, "YarnRotaryEmbedding", YarnTable)
    return _NativeDSv4Attention


@pytest.mark.parametrize("field", ["global_kv", "indexer_k"])
def test_gpt_norm_recompute_restores_shared_branch_inputs(
    monkeypatch, cpu_checkpoint_rng, native_attention, field
):
    """The GPT mHC layer must also restore standalone norm before shared-K-only backward."""
    _record_losses(monkeypatch)
    config = _config()
    submodules = TransformerLayerSubmodules(
        self_attention_hyper_connection=HyperConnectionModule,
        input_layernorm=_RMSNorm,
        self_attention=native_attention,
        self_attn_bda=get_bias_dropout_add,
        pre_mlp_layernorm=_RMSNorm,
        mlp=_FFN,
        mlp_bda=get_bias_dropout_add,
        mlp_hyper_connection=HyperConnectionModule,
    )
    groups = _groups()
    groups.pp = groups.tp
    reference = HyperConnectionTransformerLayer(
        config, submodules, layer_number=3, pg_collection=groups
    )
    actual = HyperConnectionTransformerLayer(
        replace(config, recompute_granularity="selective", recompute_modules=["layernorm"]),
        submodules,
        layer_number=3,
        pg_collection=groups,
    )
    actual.load_state_dict(reference.state_dict())
    x = torch.randn(9, 2, config.hidden_size * config.num_residual_streams, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    for layer, inputs in ((actual, x), (reference, ref_x)):
        state = CSA2State()
        layer(inputs, None, cross_layer_state=state, mhc_state=SinglePassMHCState())
        getattr(state, field).square().sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad)
    for parameter, ref_parameter in zip(actual.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, ref_parameter.grad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["sbhd", "thd"])
@pytest.mark.parametrize(
    "modules,group_size",
    [
        (["mhc", "mla_up_proj", "layernorm"], None),
        (["mhc", "mla_up_proj", "layernorm"], 1),
        (["mhc", "mla_up_proj", "layernorm"], 3),
        (["mla_up_proj", "layernorm"], None),
        (["mla_up_proj"], None),
        (["layernorm"], None),
    ],
)
def test_norm_and_mla_up_proj_recompute_with_optional_mhc(
    monkeypatch, checkpoints, native_attention, dtype, layout, group_size, modules
):
    records = _record_losses(monkeypatch)
    torch.manual_seed(542)
    config = replace(_config(dtype), multi_latent_attention=True)
    reference = _stack(config, attention=native_attention)
    recompute_config = replace(
        config,
        recompute_granularity="selective",
        recompute_modules=modules,
        mhc_recompute_layer_num=group_size,
    )
    stacks, plan = _stacks(reference, recompute_config, (5, 7, 8, 10), layout, native_attention)
    params = None
    shape = (9, 2)
    if layout == "thd":
        params, _, valid = _packed([1, 4, 0, 2], [2, 5, 0, 4], tail=3)
        shape = (valid.numel(), 1)
    pairs = []
    for _ in range(2):
        x = torch.randn(*shape, config.hidden_size, dtype=dtype, requires_grad=True)
        ref_x = x.detach().clone().requires_grad_()
        expected = reference(ref_x, None, packed_seq_params=params)
        actual, _ = _run_chunks(stacks, plan, x, params)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        pairs.append((actual, expected, x, ref_x))
    for actual, expected, x, ref_x in reversed(pairs):
        actual.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(x.grad, ref_x.grad)
    for stack, chunk in zip(stacks, plan):
        for local, layer in enumerate(stack.layers):
            ref_params = dict(reference.layers[chunk.layer_offset + local].named_parameters())
            for name, param in layer.named_parameters():
                tolerance = (
                    dict(atol=2e-7, rtol=2e-5)
                    if dtype == torch.float32
                    else dict(atol=2e-3, rtol=2e-2)
                )
                torch.testing.assert_close(param.grad, ref_params[name].grad, **tolerance, msg=name)
    assert len(records) == 12
    with torch.no_grad():
        actual, _ = _run_chunks(stacks, plan, x, params)
        expected = reference(x, None, packed_seq_params=params)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _full_config(config, method, count):
    return replace(
        config, recompute_granularity="full", recompute_method=method, recompute_num_layers=count
    )


def _assert_gradient(actual, expected, *, dtype=torch.float32):
    # Reentrant checkpoints may materialize zero gradients on unused side outputs.
    if actual is None and expected is not None:
        actual = torch.zeros_like(expected)
    elif expected is None and actual is not None:
        expected = torch.zeros_like(actual)
    tolerance = dict(atol=2e-7, rtol=2e-5) if dtype == torch.float32 else dict(atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(actual, expected, **tolerance)


def _assert_parameter_gradients(reference, stacks, plan, dtype=torch.float32):
    for stack, chunk in zip(stacks, plan):
        for local, layer in enumerate(stack.layers):
            expected = dict(reference.layers[chunk.layer_offset + local].named_parameters())
            for name, parameter in layer.named_parameters():
                try:
                    _assert_gradient(parameter.grad, expected[name].grad, dtype=dtype)
                except AssertionError as error:
                    raise AssertionError(f"Layer {chunk.layer_offset + local}: {name}") from error


def _observe_layers(stacks):
    calls = []
    for stack in stacks:
        for layer in stack.layers:
            layer.register_forward_pre_hook(
                lambda module, inputs: calls.append((module.layer_number, torch.is_grad_enabled()))
            )
    return calls


def _assert_losses(actual, expected):
    assert len(actual) == len(expected)
    assert {item["layer_number"] for item in actual} == {item["layer_number"] for item in expected}
    for layer_number in {item["layer_number"] for item in expected}:
        actual_losses = (
            torch.stack([item["loss"] for item in actual if item["layer_number"] == layer_number])
            .sort()
            .values
        )
        expected_losses = (
            torch.stack([item["loss"] for item in expected if item["layer_number"] == layer_number])
            .sort()
            .values
        )
        torch.testing.assert_close(actual_losses, expected_losses)


@pytest.mark.usefixtures("cpu_checkpoint_rng")
@pytest.mark.parametrize("layout", ["sbhd", "thd"])
@pytest.mark.parametrize(
    "method,count,cuts,mhc,coefficient,dtype",
    [
        ("uniform", 1, (), True, 0.3, torch.float32),
        ("uniform", 3, (), True, 0.3, torch.float32),
        ("uniform", 5, (), False, 0.3, torch.float32),
        ("block", 3, (), True, 0.3, torch.float32),
        ("block", 7, (), False, 0.3, torch.float32),
        ("uniform", 3, (1, 5, 7, 8, 10), True, 0.3, torch.float32),
        ("uniform", 5, (4, 7, 8, 10), False, 0.3, torch.float32),
        ("block", 3, (1, 5, 7, 8, 10), True, 0.3, torch.float32),
        ("uniform", 3, (1, 5, 7, 8, 10), True, 0.0, torch.float32),
        ("block", 3, (4, 7, 8, 10), False, 0.0, torch.float32),
        ("uniform", 3, (1, 5, 7, 8, 10), True, 0.3, torch.bfloat16),
    ],
)
def test_full_recompute_shared_state_and_outstanding_microbatches(
    monkeypatch, layout, method, count, cuts, mhc, coefficient, dtype
):
    """Group cuts cross Full/Reindex/Reuse and an FFN-only chunk relays shared state."""
    torch.manual_seed(391)
    records = _record_losses(monkeypatch)
    reference = _stack(_config(dtype, coefficient, mhc))
    config = _full_config(reference.config, method, count)
    stacks, plan = _stacks(reference, config, cuts, layout)
    calls = _observe_layers(stacks)
    params, shape = None, (9, 2)
    if layout == "thd":
        params, _, valid = _packed([1, 4, 0, 2], [2, 5, 0, 4], tail=3)
        shape = (valid.numel(), 1)
    expected_records, actual_records, pairs = [], [], []
    for _ in range(2):
        x = torch.randn(*shape, config.hidden_size, dtype=dtype, requires_grad=True)
        reference_x = x.detach().clone().requires_grad_()
        expected = reference(reference_x, None, packed_seq_params=params)
        expected_records.extend(records)
        records.clear()
        actual, payloads = _run_chunks(stacks, plan, x, params)
        actual_records.extend(records)
        records.clear()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for stack, payload in zip(stacks, payloads):
            assert (
                payload.tensor_specs
                == stack.forward_adapter.pipeline_payload_spec(*shape, params)[1].tensor_specs
            )
        pairs.append((actual, expected, x, reference_x))
    assert len(calls) == 2 * config.num_layers
    assert any(not grad_enabled for _, grad_enabled in calls)
    for actual, expected, x, reference_x in reversed(pairs):
        probe = torch.randn_like(actual) * 0.1
        (actual * probe).sum().backward()
        actual_records.extend(records)
        records.clear()
        (expected * probe).sum().backward()
        assert not records
        _assert_gradient(x.grad, reference_x.grad, dtype=dtype)
    replayed_layers = sum(
        len(stack.layers) if method == "uniform" else min(count, len(stack.layers))
        for stack in stacks
    )
    assert len(calls) == 2 * (config.num_layers + replayed_layers)
    _assert_parameter_gradients(reference, stacks, plan, dtype)
    assert len(expected_records) == (6 if coefficient else 0)
    _assert_losses(actual_records, expected_records)


@pytest.mark.usefixtures("cpu_checkpoint_rng")
@pytest.mark.parametrize("field", ["pre_mix", "global_kv"])
@pytest.mark.parametrize("method,count", [("uniform", 3), ("block", 5)])
def test_full_recompute_side_output_only_backward(monkeypatch, field, method, count):
    """A PP side-output objective must replay the producer without a hidden-state hook."""
    torch.manual_seed(182)
    _record_losses(monkeypatch)
    config = _config(coefficient=0)
    reference = _stack(config)
    stacks, _ = _stacks(reference, _full_config(config, method, count), (7,), "sbhd")
    reference_stacks, _ = _stacks(reference, config, (7,), "sbhd")
    x = torch.randn(9, 2, config.hidden_size, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    actual = stacks[0](x, None)
    expected = reference_stacks[0](reference_x, None)
    index = tuple(s.name for s in actual.tensor_specs).index(field)
    actual.tensors[index].square().sum().backward()
    expected.tensors[index].square().sum().backward()
    _assert_gradient(x.grad, reference_x.grad)
    for parameter, ref_parameter in zip(stacks[0].parameters(), reference_stacks[0].parameters()):
        _assert_gradient(parameter.grad, ref_parameter.grad)


@pytest.mark.usefixtures("cpu_checkpoint_rng")
@pytest.mark.parametrize("method,count", [("uniform", 3), ("block", 5)])
def test_full_recompute_indexer_k_boundary_gradient(monkeypatch, method, count):
    """Exercise indexer K gradients with the auxiliary objective explicitly enabled."""
    torch.manual_seed(192)
    _record_losses(monkeypatch)
    config = _config()
    reference = _stack(config)
    stacks, _ = _stacks(reference, _full_config(config, method, count), (7,), "sbhd")
    reference_stacks, _ = _stacks(reference, config, (7,), "sbhd")
    x = torch.randn(9, 2, config.hidden_size, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    actual = stacks[0](x, None)
    expected = reference_stacks[0](reference_x, None)
    index = tuple(s.name for s in actual.tensor_specs).index("indexer_k")
    # The zero hidden objective invokes the same auxiliary autoscaler in both
    # paths: reentrant checkpoint backward also materializes unused-output zeros.
    (actual.tensors[index].square().sum() + actual.tensors[0].sum() * 0).backward()
    (expected.tensors[index].square().sum() + expected.tensors[0].sum() * 0).backward()
    _assert_gradient(x.grad, reference_x.grad)
    for parameter, ref_parameter in zip(stacks[0].parameters(), reference_stacks[0].parameters()):
        _assert_gradient(parameter.grad, ref_parameter.grad)


@pytest.mark.usefixtures("cpu_checkpoint_rng")
@pytest.mark.parametrize("method,count", [("uniform", 3), ("block", 7)])
@pytest.mark.parametrize("mhc", [False, True])
def test_full_recompute_restores_dropout_rng(monkeypatch, method, count, mhc):
    torch.manual_seed(816)
    _record_losses(monkeypatch)
    config = replace(_config(enable_hyper_connections=mhc), hidden_dropout=0.3)
    reference = _stack(config)
    actual = _stack(_full_config(config, method, count))
    actual.load_state_dict(reference.state_dict())
    x = torch.randn(9, 2, config.hidden_size, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    rng = torch.get_rng_state()
    expected = reference(reference_x, None)
    torch.set_rng_state(rng)
    output = actual(x, None)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    after_forward = torch.get_rng_state()
    output.square().sum().backward()
    assert torch.equal(torch.get_rng_state(), after_forward)
    expected.square().sum().backward()
    _assert_gradient(x.grad, reference_x.grad)
    for parameter, ref_parameter in zip(actual.parameters(), reference.parameters()):
        _assert_gradient(parameter.grad, ref_parameter.grad)


@pytest.mark.usefixtures("cpu_checkpoint_rng")
@pytest.mark.parametrize("method", ["uniform", "block"])
def test_full_recompute_no_grad_bypasses_checkpoint(monkeypatch, method):
    _record_losses(monkeypatch)
    stack = _stack(_full_config(_config(), method, 3))

    def unexpected_checkpoint(*args, **kwargs):
        raise AssertionError("A no-grad forward must not allocate activation checkpoints")

    monkeypatch.setattr(checkpoint_runtime.CheckpointFunction, "forward", unexpected_checkpoint)
    with torch.no_grad():
        output = stack(torch.randn(9, 2, stack.config.hidden_size), None)
    assert torch.isfinite(output).all()


@pytest.mark.usefixtures("cpu_checkpoint_rng")
def test_full_recompute_short_sequence_has_empty_compressed_state(monkeypatch):
    torch.manual_seed(642)
    _record_losses(monkeypatch)
    reference = _stack(_config())
    stacks, plan = _stacks(reference, _full_config(reference.config, "uniform", 3), (5, 7), "sbhd")
    x = torch.randn(1, 1, reference.config.hidden_size, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    actual, _ = _run_chunks(stacks, plan, x)
    expected = reference(reference_x, None)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    _assert_gradient(x.grad, reference_x.grad)
    _assert_parameter_gradients(reference, stacks, plan)


@pytest.mark.usefixtures("cpu_checkpoint_rng")
def test_full_recompute_reduces_saved_activation_storage(monkeypatch):
    """Checkpoints must not retain the interior CSA2 graph through mutable shared state."""
    torch.manual_seed(219)
    _record_losses(monkeypatch)
    reference = _stack(_config())
    actual = _stack(_full_config(reference.config, "uniform", 3))
    actual.load_state_dict(reference.state_dict())
    saved_bytes = []
    for stack in (reference, actual):
        parameter_ptrs = {
            parameter.untyped_storage().data_ptr() for parameter in stack.parameters()
        }
        saved = {}

        def observe(tensor):
            storage = tensor.untyped_storage()
            if storage.data_ptr() not in parameter_ptrs:
                saved[storage.data_ptr()] = storage.nbytes()
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(observe, lambda tensor: tensor):
            output = stack(torch.randn(9, 2, stack.config.hidden_size, requires_grad=True), None)
        saved_bytes.append(sum(saved.values()))
        output.square().sum().backward()
    assert 0 < saved_bytes[1] < saved_bytes[0]


@pytest.mark.usefixtures("cpu_checkpoint_rng")
@pytest.mark.parametrize("method", ["uniform", "block"])
@pytest.mark.parametrize("mhc", [False, True])
def test_full_recompute_frozen_hidden_input_preserves_parameter_gradients(monkeypatch, method, mhc):
    """A frozen embedding must not disconnect trainable layers from reentrant backward."""
    torch.manual_seed(924)
    _record_losses(monkeypatch)
    reference = _stack(_config(enable_hyper_connections=mhc))
    actual = _stack(_full_config(reference.config, method, 3))
    actual.load_state_dict(reference.state_dict())
    x = torch.randn(9, 2, reference.config.hidden_size)
    expected = reference(x, None)
    output = actual(x, None)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    output.square().sum().backward()
    expected.square().sum().backward()
    assert any(parameter.grad is not None for parameter in actual.parameters())
    for parameter, ref_parameter in zip(actual.parameters(), reference.parameters()):
        _assert_gradient(parameter.grad, ref_parameter.grad)


@pytest.mark.usefixtures("cpu_checkpoint_rng")
@pytest.mark.parametrize("quantization", ["fp8", "fp4", "quant_recipe"])
@pytest.mark.parametrize("layout", ["sbhd", "thd"])
def test_full_recompute_quantization_dispatch_and_layer_context(
    monkeypatch, cpu_te_checkpoint, quantization, layout
):
    """Validate TE checkpoint dispatch and global layer contexts without quantized kernels."""
    torch.manual_seed(437)
    records = _record_losses(monkeypatch)
    reference = _stack(_config())
    config = _full_config(reference.config, "uniform", 3)
    stacks, plan = _stacks(reference, config, (5,), layout)
    # These adapters contain ordinary CPU linears. Enabling only dispatch after
    # construction isolates checkpoint/context wiring from TE recipe dependencies.
    if quantization == "fp8":
        config.fp8 = "e4m3"
        config.fp8_recipe = Fp8Recipe.tensorwise
    elif quantization == "fp4":
        config.fp4 = "e2m1"
    else:
        config.quant_recipe = object()
    contexts, checkpoint_calls = [], []

    @contextmanager
    def quantization_context(config, layer_number=None):
        contexts.append((layer_number, torch.is_grad_enabled()))
        yield

    def te_checkpoint(function, distribute, rng_tracker, tp_group, *inputs, boundary_policy):
        assert rng_tracker is checkpoint_runtime.get_cuda_rng_tracker
        assert any(tp_group is stack.pg_collection.tp for stack in stacks)
        checkpoint_calls.append(len(inputs))
        return te_runtime.te_checkpoint(
            function, distribute, rng_tracker, tp_group, *inputs, boundary_policy=boundary_policy
        )

    monkeypatch.setattr(recompute_runtime, "te_checkpoint", te_checkpoint)
    monkeypatch.setattr(hybrid_runtime, "get_fp8_context", quantization_context)
    monkeypatch.setattr(hybrid_runtime, "get_fp4_context", quantization_context)
    params = None if layout == "sbhd" else _packed([3, 0, 4], [4, 0, 5])[0]
    x = torch.randn(9, 2 if params is None else 1, reference.config.hidden_size, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    actual, _ = _run_chunks(stacks, plan, x, params)
    assert not records  # The original TE body still runs under no-grad.
    expected = reference(reference_x, None, packed_seq_params=params)
    expected_records = {r["layer_number"]: r["loss"] for r in records}
    records.clear()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    assert len(records) == len(expected_records)
    for record in records:
        torch.testing.assert_close(record["loss"], expected_records[record["layer_number"]])
    _assert_gradient(x.grad, reference_x.grad)
    _assert_parameter_gradients(reference, stacks, plan)
    assert len(checkpoint_calls) == 5
    if quantization == "quant_recipe":
        assert not contexts
    else:
        assert sorted(contexts) == [
            (layer, grad_enabled)
            for layer in range(config.num_layers)
            for grad_enabled in (False, True)
        ]


@pytest.mark.usefixtures("cpu_checkpoint_rng")
@pytest.mark.parametrize("method", ["uniform", "block"])
def test_full_recompute_legacy_mhc(monkeypatch, method):
    """The existing multi-pass mHC wrapper shares the same full-recompute state path."""
    if not torch.cuda.is_available():
        monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda message: None)
        monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)
    torch.manual_seed(845)
    _record_losses(monkeypatch)
    config = replace(_config(), mhc_single_pass=False)
    reference = _stack(config)
    actual_stack = _stack(_full_config(config, method, 3))
    actual_stack.load_state_dict(reference.state_dict())
    x = torch.randn(9, 2, config.hidden_size, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    actual = actual_stack(x, None)
    expected = reference(reference_x, None)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    _assert_gradient(x.grad, reference_x.grad)
    for parameter, ref_parameter in zip(actual_stack.parameters(), reference.parameters()):
        _assert_gradient(parameter.grad, ref_parameter.grad)


@pytest.mark.usefixtures("cpu_checkpoint_rng")
def test_full_recompute_frozen_hidden_with_live_pipeline_state(monkeypatch):
    """Live side inputs suffice for checkpointing even when received hidden is frozen."""
    torch.manual_seed(481)
    _record_losses(monkeypatch)
    config = _config(coefficient=0)
    reference = _stack(config)
    stacks, _ = _stacks(reference, _full_config(config, "uniform", 3), (7,), "sbhd")
    reference_stacks, _ = _stacks(reference, config, (7,), "sbhd")
    calls = _observe_layers(stacks[1:])
    x = torch.randn(9, 2, config.hidden_size, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    outputs = []
    for chunk_stacks, inputs in ((stacks, x), (reference_stacks, reference_x)):
        payload = chunk_stacks[0](inputs, None)
        payload = replace(payload, tensors=(payload.tensors[0].detach(), *payload.tensors[1:]))
        chunk_stacks[1].set_input_tensor(payload)
        outputs.append(chunk_stacks[1](None, None))
    torch.testing.assert_close(*outputs, atol=0, rtol=0)
    assert all(not enabled for _, enabled in calls)
    for output in outputs:
        output.square().sum().backward()
    assert len(calls) == 2 * len(stacks[1].layers)
    _assert_gradient(x.grad, reference_x.grad)
    for stack, ref_stack in zip(stacks, reference_stacks):
        for parameter, ref_parameter in zip(stack.parameters(), ref_stack.parameters()):
            _assert_gradient(parameter.grad, ref_parameter.grad)


@pytest.mark.usefixtures("cpu_checkpoint_rng")
def test_recompute_restore_rebuilds_differentiable_fused_k_views(monkeypatch):
    """Replay flat buffers must remain connected to explicit checkpoint state inputs."""
    _record_losses(monkeypatch)
    config = _config()
    stacks, _ = _stacks(_stack(config), config, (7,), "sbhd")
    payload = stacks[0](torch.randn(9, 2, config.hidden_size, requires_grad=True), None)
    hidden, context, _ = payload.restore()
    state = context.cross_layer_state
    state.global_kv_flat = state.indexer_k_flat = None
    with torch.no_grad():
        state.prepare_fused_kv()
    assert not state.global_kv_flat.requires_grad and not state.indexer_k_flat.requires_grad
    # Only declare fused views here; the CPU forward above used native attention.
    config.dsa_kernel_backend = "cudnn"
    region = stacks[1].forward_adapter.checkpoint_region(0, 2, hidden, context)
    tensors = region.codec.export(context, region.schema.inputs)
    positional = tuple(
        tensor.detach().requires_grad_(tensor.requires_grad) if tensor is not None else None
        for tensor in tensors
    )
    replay = region.codec.restore(region.schema.inputs, positional, region.input_metadata)
    replay_state = replay.cross_layer_state
    objective = (
        replay.mhc_state.pre_mix.square().sum()
        + replay_state.global_kv_flat.square().sum()
        + replay_state.indexer_k_flat.square().sum()
    )
    differentiable = tuple(
        tensor for tensor in positional if tensor is not None and tensor.requires_grad
    )
    gradients = torch.autograd.grad(objective, differentiable)
    for tensor, gradient in zip(differentiable, gradients):
        torch.testing.assert_close(gradient, 2 * tensor)


def test_canonical_codec_restores_fresh_packed_state_for_outstanding_microbatches():
    """Changing integer prefixes/selection values cannot leak through a saved payload."""
    import weakref

    from megatron.core.transformer.experimental_attention_variant.csa2 import (
        CSA2State,
        CSA2StateCodec,
    )
    from megatron.core.transformer.experimental_attention_variant.csa_utils.thd_utils import (
        build_csa2_thd_layout,
    )
    from megatron.core.transformer.state_boundary import validate_metadata

    snapshots = []
    for lengths in ([1, 3], [2, 2]):
        params, _, valid = _packed(lengths, [2, 4])
        state = CSA2State(
            global_kv=torch.randn(3, 1, 4, requires_grad=True),
            global_indices=torch.full((valid.numel(), 1), lengths[0], dtype=torch.int32),
            kv_source_layer=2,
            index_source_layer=4,
            last_layer=4,
            sequence_length=valid.numel(),
            batch_size=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
            thd_layout=build_csa2_thd_layout(params, valid.numel()),
        )
        state.compressed_layout = state.thd_layout.for_compression(2)
        codec = CSA2StateCodec()
        fields, metadata = codec.fields(state), codec.metadata(state)
        validate_metadata(metadata)
        tensors = codec.export(state, fields)
        reference = weakref.ref(state)
        del state
        assert reference() is None
        snapshots.append((tensors, codec, fields, metadata))
    for tensors, codec, fields, metadata in reversed(snapshots):
        replay = codec.restore(fields, tensors, metadata)
        second = codec.restore(fields, tensors, metadata)
        assert replay is not second and replay.thd_layout is not second.thd_layout
        assert replay.compressed_layout is not second.compressed_layout
        assert torch.equal(
            replay.global_indices[:, 0],
            replay.thd_layout.cu_seqlens[1].expand(replay.sequence_length),
        )
        replay.global_kv.sum().backward()
        torch.testing.assert_close(tensors[0].grad, torch.ones_like(tensors[0]))
        with pytest.raises(ValueError, match="namespace"):
            CSA2StateCodec(namespace="csa2.mtp").restore(fields, tensors, metadata)
        wrong = (replace(fields[0], key="csa2.decoder/global_kv:L0"), *fields[1:])
        with pytest.raises(ValueError, match="source version"):
            codec.restore(wrong, tensors, metadata)
