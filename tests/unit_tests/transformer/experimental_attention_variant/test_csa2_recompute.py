# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Single-pass mHC replay with shared CSA2 state and outstanding microbatches."""

from argparse import ArgumentParser
from dataclasses import replace

import pytest
import torch

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
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_pipeline import (
    build_csa2_pipeline_plan,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.training.arguments import _add_network_size_args
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2 import (
    _CPUFrequencyTable,
    _Linear,
    _packed,
    _record_losses,
    _RMSNorm,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2_pipeline import (
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
def test_recompute_when_only_shared_output_receives_gradient(monkeypatch, checkpoints, field):
    """A side-output-only objective never reaches the hidden-state recompute hook."""
    _record_losses(monkeypatch)
    torch.manual_seed(182)
    config = _config()
    reference = _stack(config)
    stacks, _ = _stacks(reference, _recompute_config(config, None), (7,), "sbhd")
    ref_stacks, _ = _stacks(reference, config, (7,), "sbhd")
    x = torch.randn(9, 2, config.hidden_size, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    actual = stacks[0](x, None)
    expected = ref_stacks[0](ref_x, None)
    index = actual.boundary.field_names.index(field)
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


@pytest.mark.parametrize("module", ["core_attn", "layernorm", "mla_up_proj", "moe", "mlp"])
def test_unsupported_recompute_is_rejected(module):
    with pytest.raises(ValueError, match="activation recomputation"):
        replace(_config(), recompute_granularity="selective", recompute_modules=[module])


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["sbhd", "thd"])
@pytest.mark.parametrize("group_size", [None, 1, 3])
def test_mhc_with_mla_up_proj_recompute(
    monkeypatch, checkpoints, native_attention, dtype, layout, group_size
):
    records = _record_losses(monkeypatch)
    torch.manual_seed(542)
    config = replace(_config(dtype), multi_latent_attention=True)
    reference = _stack(config, attention=native_attention)
    recompute_config = replace(
        _recompute_config(config, group_size), recompute_modules=["mhc", "mla_up_proj", "layernorm"]
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
