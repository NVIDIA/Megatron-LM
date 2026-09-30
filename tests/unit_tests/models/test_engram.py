# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Engram numerics, V4.1 parity, Hybrid residual integration, and EP lookup.

The reference formulas follow DeepSeek-V4.1-Flash inference/engram.py and
inference/model.py. Their backward is the derivative of the native formulas,
not an inference quantization kernel. EP tests also run on CPU with Gloo:
torchrun --standalone --nproc-per-node=2 -m pytest ... -k ep_lookup
"""

from __future__ import annotations

import argparse
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from megatron.core.models.engram import Engram, EngramConfig, apply_engram_to_hybrid_stack_spec
from megatron.core.models.engram.config import (
    TOKENIZER_MAP_FORMAT,
    TOKENIZER_MAP_VERSION,
    allocate_table_sizes,
)
from megatron.core.models.engram.distributed_embedding import EPShardedMultiTableEmbedding
from megatron.core.models.engram.hashing import build_ngram_hashes
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import _add_engram_args
from tools.engram.generate_tokenizer_map import build_layer_multipliers


def write_tokenizer_map(
    path: Path,
    *,
    vocab_size: int = 32,
    layer_ids: tuple[int, ...] = (1,),
    max_ngram_order: int = 3,
    hash_seed: int = 0,
    pad_token_id: int = 0,
    remap: list[int] | None = None,
    multipliers: tuple[int, ...] = (13, 17, 19),
    hash_layer_ids: tuple[int, ...] | None = None,
) -> Path:
    """Write a small valid tokenizer map without tokenizer dependencies."""
    remap = list(range(vocab_size)) if remap is None else remap
    compressed_vocab_size = max(remap) + 1
    artifact = {
        "format": TOKENIZER_MAP_FORMAT,
        "version": TOKENIZER_MAP_VERSION,
        "source_vocab_size": len(remap),
        "compressed_vocab_size": compressed_vocab_size,
        "pad_token_id": pad_token_id,
        "compressed_pad_token_id": remap[pad_token_id],
        "max_ngram_order": max_ngram_order,
        "hash_seed": hash_seed,
        "layer_ids": list(layer_ids),
        "hash_layer_ids": list(layer_ids if hash_layer_ids is None else hash_layer_ids),
        "layer_multipliers": {
            str(layer_id): list(multipliers[:max_ngram_order]) for layer_id in layer_ids
        },
        "remap": remap,
    }
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


def make_module_config(
    *,
    num_streams: int = 1,
    sequence_parallel: bool = False,
    deterministic_mode: bool = False,
    dtype=torch.float64,
):
    """Return the minimal TransformerConfig interface consumed by Engram."""
    return SimpleNamespace(
        use_cpu_initialization=True,
        params_dtype=dtype,
        perform_initialization=True,
        init_method=lambda tensor: torch.nn.init.normal_(tensor, mean=0.0, std=0.1),
        hidden_size=8,
        enable_hyper_connections=num_streams > 1,
        num_residual_streams=num_streams,
        layernorm_epsilon=1e-5,
        sequence_parallel=sequence_parallel,
        deterministic_mode=deterministic_mode,
    )


def make_pg_collection(ep=None, tp=None, expt_dp=None) -> ProcessGroupCollection:
    """Build only the process groups used by the Engram module."""
    return ProcessGroupCollection(ep=ep, tp=tp, expt_dp=expt_dp)


def _official_reference(module: Engram, hidden_states: torch.Tensor, input_ids: torch.Tensor):
    hashes = build_ngram_hashes(
        input_ids,
        module.tokenizer_remap,
        module.hash_multipliers,
        module.table_sizes,
        module.engram_config.max_ngram_order,
        module.engram_config.num_hash_heads,
        module.engram_config.hash_boundary_token_id,
        reset_at_boundary=module.engram_config.variant_spec.resets_windows_at_boundary_token,
    )
    embeddings = torch.cat(
        [
            F.embedding(hashes[..., table_id], table.weight)
            for table_id, table in enumerate(module.embedding.tables)
        ],
        dim=-1,
    )
    streams = hidden_states.transpose(0, 1).view(
        input_ids.shape[0], input_ids.shape[1], module.num_streams, module.hidden_size
    )

    def group_rms_norm(norm, tensor):
        # Independent per-stream RMSNorm expressed over the fused width, incl. the optional
        # zero-centered gamma - the reference math the module's EngramGroupRMSNorm must match.
        grouped = tensor.view(*tensor.shape[:-1], module.num_streams, module.hidden_size)
        normed = grouped * torch.rsqrt(grouped.pow(2).mean(-1, keepdim=True) + norm.eps)
        gamma = norm.weight.view(module.num_streams, module.hidden_size)
        if norm.zero_centered:
            gamma = 1.0 + gamma
        return (normed * gamma).flatten(-2)

    key = group_rms_norm(module.key_norm, module.key_projection(embeddings)).view(
        input_ids.shape[0], input_ids.shape[1], module.num_streams, module.hidden_size
    )
    query = group_rms_norm(module.query_norm, streams.flatten(-2)).view_as(streams)
    score = (key * query).sum(dim=-1) / math.sqrt(module.hidden_size)
    score = score.abs().clamp_min(1e-6).sqrt() * score.sign()
    gate = score.sigmoid().unsqueeze(-1)
    value = gate * module.value_projection(embeddings).unsqueeze(2)
    normed = group_rms_norm(module.conv_norm, value.flatten(-2))
    # Official causal convolution: zero history of (kernel_size - 1) * dilation positions,
    # expressed as a symmetric-pad conv truncated to the sequence length.
    convolved = F.conv1d(
        normed.transpose(1, 2),
        module.short_conv.weight,
        padding=module.conv_history_length,
        dilation=module.short_conv.dilation,
        groups=module.short_conv.groups,
    )[..., : input_ids.shape[1]]
    output = value + F.silu(convolved).transpose(1, 2).view_as(value)
    return output.view(input_ids.shape[0], input_ids.shape[1], -1).transpose(0, 1)


@pytest.mark.parametrize("num_streams", [1, 4])
@pytest.mark.parametrize("variant", ["deepseek", "qwen"])
def test_official_forward_and_backward_math(tmp_path, num_streams, variant):
    torch.manual_seed(123)
    artifact = write_tokenizer_map(tmp_path / "map.json", vocab_size=32, layer_ids=(1,))
    config = EngramConfig(
        global_vocab_sizes=(17, 17),
        layer_ids=(1,),
        max_ngram_order=3,
        num_hash_heads=2,
        memory_dim=8,
        kernel_size=4,
        hash_seed=0,
        boundary_token_id=0,
        tokenizer_map_path=str(artifact) if variant == "deepseek" else "",
        variant=variant,
        unigram_vocab_size=32 if variant == "qwen" else None,
    )
    module = Engram(
        config=make_module_config(num_streams=num_streams),
        engram_config=config,
        layer_number=1,
        pg_collection=make_pg_collection(),
    )
    assert torch.count_nonzero(module.short_conv.weight) == 0
    with torch.no_grad():
        module.short_conv.weight.normal_(mean=0.0, std=0.03)

    input_ids = torch.tensor([[1, 4, 0, 7], [3, 5, 9, 2]], dtype=torch.int64)
    hidden = torch.randn(4, 2, 8 * num_streams, dtype=torch.float64, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    actual = module(hidden, input_ids)
    expected = _official_reference(module, reference_hidden, input_ids)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    parameters = [hidden, *module.parameters()]
    reference_parameters = [reference_hidden, *module.parameters()]
    projection = torch.randn_like(actual)
    actual_grads = torch.autograd.grad((actual * projection).sum(), parameters, retain_graph=True)
    expected_grads = torch.autograd.grad(
        (expected * projection).sum(), reference_parameters, retain_graph=True
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-9, atol=1e-9)
    assert module.embedding.tables[0].weight.is_engram_embedding
    assert not module.embedding.tables[0].weight.allreduce


def _config(tmp_path, *, layer_ids=(3,), hash_layer_ids=(1,), **kwargs):
    artifact_path = write_tokenizer_map(
        tmp_path / "v41-map.json",
        layer_ids=layer_ids,
        hash_layer_ids=hash_layer_ids,
        max_ngram_order=4,
        multipliers=(13, 17, 19, 23),
        pad_token_id=2,
    )
    artifact = json.loads(artifact_path.read_text())
    artifact["layer_multipliers"] = build_layer_multipliers(
        32, list(layer_ids), 4, 0, list(hash_layer_ids)
    )
    artifact_path.write_text(json.dumps(artifact))
    settings = dict(
        global_vocab_sizes=(17, 17, 17),
        layer_ids=layer_ids,
        hash_layer_ids=hash_layer_ids,
        max_ngram_order=4,
        num_hash_heads=2,
        memory_dim=8,
        kernel_size=0,
        hash_seed=0,
        boundary_token_id=2,
        tokenizer_map_path=str(artifact_path),
        compressed_vocab_size=32,
        variant="deepseek_v41",
    )
    settings.update(kwargs)
    return EngramConfig(**settings)


def _hash_reference(tokens, config, layer_number):
    """Scalar addressing oracle; never calls production shifting/hash helpers."""
    result = []
    multipliers = config.multipliers(layer_number)
    for row in tokens.tolist():
        compressed = [int(config.tokenizer_remap[token]) for token in row]
        hashes = []
        for position in range(len(row)):
            suffix = [
                compressed[position - shift] if position >= shift else config.hash_boundary_token_id
                for shift in range(config.max_ngram_order)
            ]
            columns = []
            rolling = suffix[0] * multipliers[0]
            for order in range(2, config.max_ngram_order + 1):
                rolling ^= suffix[order - 1] * multipliers[order - 1]
                start = (order - 2) * config.num_hash_heads
                for prime in config.table_sizes(layer_number)[
                    start : start + config.num_hash_heads
                ]:
                    columns.append(rolling % prime)
            hashes.append(columns)
        result.append(hashes)
    return torch.tensor(result, dtype=torch.int64, device=tokens.device)


def _reference(hidden, tokens, module, weights):
    """Published V4.1 residual update using independently supplied weights."""
    hashes = _hash_reference(tokens, module.engram_config, module.layer_number)
    memory = (
        torch.cat(
            [
                F.embedding(hashes[..., index], weights[f"embedding.tables.{index}.weight"])
                for index in range(hashes.shape[-1])
            ],
            dim=-1,
        )
        .transpose(0, 1)
        .contiguous()
    )
    key, value = F.linear(memory, weights["wkv.weight"]).split(
        (module.num_streams * module.hidden_size, module.hidden_size), dim=-1
    )
    key = key.float().unflatten(-1, (module.num_streams, module.hidden_size))
    h = hidden.float().unflatten(-1, (module.num_streams, module.hidden_size))
    q_weight = weights["query_norm.weight"].float().view(module.num_streams, -1)
    k_weight = weights["key_norm.weight"].float().view(module.num_streams, -1)
    rstd = torch.rsqrt(h.square().mean(-1) + module.query_norm.eps) * torch.rsqrt(
        key.square().mean(-1) + module.key_norm.eps
    )
    dot = (h * (q_weight * k_weight) * key).sum(-1) * rstd * module.hidden_size**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).flatten(-2).to(hidden.dtype)


def test_v41_hash_identities_and_official_table_layout(tmp_path):
    config = _config(tmp_path, layer_ids=(3, 29), hash_layer_ids=(1, 14))
    half_bound = (np.iinfo(np.int64).max // 32) // 2
    for placement, reference_id in zip(config.layer_ids, (1, 14)):
        expected = (
            np.random.default_rng(10007 * reference_id).integers(
                0, half_bound, size=4, dtype=np.int64
            )
            * 2
            + 1
        )
        assert config.multipliers(placement) == tuple(expected)
        tokens = torch.tensor([[2, 7, 8, 2, 31], [11, 11, 11, 1, 0]])
        actual = build_ngram_hashes(
            tokens,
            config.tokenizer_remap,
            torch.tensor(config.multipliers(placement)),
            torch.tensor(config.table_sizes(placement)),
            4,
            2,
            config.hash_boundary_token_id,
        )
        torch.testing.assert_close(
            actual, _hash_reference(tokens, config, placement), rtol=0, atol=0
        )
    official = allocate_table_sizes((16000000,) * 3, (3, 29), 8)
    assert tuple(map(sum, official.values())) == (384006168, 384016682)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("streams", [1, 4])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_v41_native_forward_and_backward(tmp_path, dtype, streams, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU Engram parity")
    torch.manual_seed(29)
    module_config = make_module_config(num_streams=streams, dtype=dtype)
    module_config.use_cpu_initialization = device == "cpu"
    module_config.layernorm_epsilon = 1e-20
    module = Engram(module_config, _config(tmp_path), 3, make_pg_collection())
    assert module.short_conv is None and module.conv_norm is None
    assert module.wkv.bias is None
    assert module.query_norm.weight.ndim == module.key_norm.weight.ndim == 1
    assert not any("conv" in name or name.endswith("bias") for name, _ in module.named_parameters())
    tokens = torch.tensor([[2, 1, 8, 9, 3], [11, 11, 11, 1, 0]], device=device)
    hidden = torch.randn(5, 2, streams * 8, dtype=dtype, device=device, requires_grad=True)
    ref_hidden = hidden.detach().clone().requires_grad_(True)
    weights = {
        name: param.detach().clone().requires_grad_(True)
        for name, param in module.named_parameters()
    }
    actual = module.add_to_residual(hidden, tokens)
    expected = _reference(ref_hidden, tokens, module, weights)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    probe = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (hidden, *module.parameters()), probe)
    expected_grads = torch.autograd.grad(expected, (ref_hidden, *weights.values()), probe)
    for grad, reference_grad in zip(actual_grads, expected_grads):
        assert torch.isfinite(grad).all()
        torch.testing.assert_close(grad, reference_grad, rtol=0, atol=0)
    if dtype == torch.bfloat16:
        # Make the increment just larger than half an ULP at 1.0. Rounding it
        # before addition lands exactly on the tie and loses the update on
        # both CPU and CUDA, independently of their random-number streams.
        with torch.no_grad():
            for table in module.embedding.tables:
                table.weight.fill_(1.0)
            module.wkv.weight.zero_()
            module.wkv.weight[streams * module.hidden_size :, 0] = 2**-7
            witness = torch.ones_like(hidden)
            actual = module.add_to_residual(witness, tokens)
            rounded_increment = module(witness, tokens).to(dtype)
        torch.testing.assert_close(actual, torch.full_like(actual, 1.0 + 2**-7), rtol=0, atol=0)
        torch.testing.assert_close(witness + rounded_increment, witness, rtol=0, atol=0)


def test_v41_zero_score_uses_copysign(tmp_path):
    module = Engram(
        make_module_config(dtype=torch.float32), _config(tmp_path), 3, make_pg_collection()
    )
    with torch.no_grad():
        for table in module.embedding.tables:
            table.weight.fill_(0.25)
        module.wkv.weight.zero_()
        module.wkv.weight[module.hidden_size :].fill_(0.125)
    tokens = torch.tensor([[2, 3]])
    hidden = torch.zeros(2, 1, module.hidden_size)
    actual = module.add_to_residual(hidden, tokens)
    expected = (
        torch.sigmoid(torch.tensor(1e-3)) * 0.125 * 0.25 * module.engram_config.total_memory_dim
    )
    torch.testing.assert_close(actual, torch.full_like(actual, expected), rtol=0, atol=0)


def test_v41_artifact_rejects_wrong_hash_identity_or_compressed_vocab(tmp_path):
    config = _config(tmp_path)
    artifact_path = tmp_path / "v41-map.json"
    artifact = json.loads(artifact_path.read_text())
    artifact["hash_layer_ids"] = [3]
    artifact_path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="hash_layer_ids mismatch"):
        config._load_tokenizer_map()
    artifact["hash_layer_ids"] = [1]
    artifact_path.write_text(json.dumps(artifact))
    config.compressed_vocab_size = 99092
    with pytest.raises(ValueError, match="compressed_vocab_size mismatch"):
        config._load_tokenizer_map()


def test_v41_cli_defaults(tmp_path):
    config = _config(tmp_path)
    args = _add_engram_args(argparse.ArgumentParser()).parse_args(
        [
            "--engram-variant",
            "deepseek_v41",
            "--engram-vocab-sizes",
            "17",
            "17",
            "17",
            "--engram-layer-ids",
            "3",
            "--engram-hash-layer-ids",
            "1",
            "--engram-tokenizer-map",
            config.tokenizer_map_path,
        ]
    )
    args.pipeline_model_parallel_size = 1
    transformer = TransformerConfig(num_layers=4, hidden_size=8, num_attention_heads=2)
    parsed = EngramConfig.from_args(args, transformer)
    assert (parsed.max_ngram_order, parsed.memory_dim, parsed.head_dim) == (4, 2048, 256)
    assert parsed.kernel_size == 0 and parsed.boundary_token_id == 2
    assert transformer.engram_enabled


@pytest.mark.parametrize(
    "mhc_mode", ["none", "original", "single_pass", "original_fallback", "single_pass_fallback"]
)
def test_hybrid_injects_before_mhc_exactly_once(tmp_path, mhc_mode, monkeypatch):
    # Reuse the small native attention/FFN fixtures, while exercising the real
    # TransformerLayer, HybridStack and HyperConnectionHybridLayer executors.
    from tests.unit_tests.ssm.test_hybrid_state_adapter import _config as hybrid_config
    from tests.unit_tests.ssm.test_hybrid_state_adapter import _submodules

    if not torch.cuda.is_available():
        # Original mHC uses a CUDA NVTX context even for native CPU math.
        monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *_args, **_kwargs: nullcontext())

    config = hybrid_config(
        num_layers=4,
        enable_hyper_connections=mhc_mode != "none",
        mhc_single_pass=mhc_mode.startswith("single_pass"),
    )
    memory_config = _config(tmp_path)
    spec = apply_engram_to_hybrid_stack_spec(
        ModuleSpec(HybridStack, submodules=_submodules(config)), memory_config, "*-*-", config
    )
    groups = ProcessGroupCollection()
    groups.tp = groups.cp = groups.pp = torch.distributed.ProcessGroup(0, 1)
    groups.ep = groups.expt_dp = None

    def make_stack():
        return HybridStack(
            config,
            spec.submodules,
            layer_type_list=list("*-*-"),
            pp_layer_offset=0,
            pre_process=True,
            post_process=False,
            post_layer_norm=False,
            pg_collection=groups,
        )

    actual_stack, reference_stack = make_stack(), make_stack()
    if mhc_mode.endswith("_fallback"):
        # Exercise the wrapper's general inner-layer call as well as the
        # attention/MLP fast paths; both must inject on the multi-stream residual once.
        for stack in (actual_stack, reference_stack):
            for layer in stack.layers:
                monkeypatch.setattr(
                    layer,
                    "_call_inner_transformer_layer_without_local_bda",
                    lambda *args, **kwargs: None,
                )
    reference_stack.load_state_dict(actual_stack.state_dict())
    actual_params = dict(actual_stack.named_parameters())
    reference_params = dict(reference_stack.named_parameters())
    target = reference_stack.layers[2]
    inner = getattr(target, "inner_layer", target)
    memory = inner.engram
    inner.engram = None
    tokens = torch.tensor([[2, 3, 5, 5, 2]])

    def inject(_module, args, kwargs):
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = _reference(
            kwargs["hidden_states"], tokens, memory, dict(memory.named_parameters())
        )
        return args, kwargs

    target.register_forward_pre_hook(inject, with_kwargs=True)
    hidden = torch.randn(5, 1, 8, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    actual = actual_stack(hidden, attention_mask=None, input_ids=tokens)
    expected = reference_stack(reference_hidden, attention_mask=None, input_ids=tokens)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    probe = torch.randn_like(actual)
    actual.backward(probe)
    expected.backward(probe)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=0, atol=0)
    for name, parameter in actual_params.items():
        torch.testing.assert_close(parameter.grad, reference_params[name].grad, rtol=0, atol=0)
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for name, parameter in actual_params.items()
        if ".engram.embedding." in name
    )


def test_hybrid_rejects_misplaced_v41_layer(tmp_path):
    from tests.unit_tests.ssm.test_hybrid_state_adapter import _config as hybrid_config
    from tests.unit_tests.ssm.test_hybrid_state_adapter import _submodules

    config = hybrid_config(num_layers=4)
    engram = _config(tmp_path, layer_ids=(2,), hash_layer_ids=(1,))
    with pytest.raises(ValueError, match="before the attention sublayer"):
        apply_engram_to_hybrid_stack_spec(
            ModuleSpec(HybridStack, submodules=_submodules(config)), engram, "*-*-", config
        )


@pytest.fixture(scope="module")
def ep_group():
    distributed = torch.distributed
    owns_world = not distributed.is_initialized()
    if owns_world:
        distributed.init_process_group("gloo")
    group = distributed.new_group(backend="gloo")
    try:
        yield group
    finally:
        distributed.barrier(group=group)
        distributed.destroy_process_group(group)
        if owns_world:
            distributed.destroy_process_group()


@pytest.mark.skipif(
    int(os.getenv("WORLD_SIZE", "1")) < 2, reason="requires torchrun with >=2 ranks"
)
@pytest.mark.parametrize("empty_owner", [False, True])
def test_ep_lookup_gradients_and_adam_step(empty_owner, ep_group):
    distributed = torch.distributed
    group = ep_group
    rank, size = distributed.get_rank(group), distributed.get_world_size(group)
    config = make_module_config(dtype=torch.float32, deterministic_mode=True)
    embedding = EPShardedMultiTableEmbedding(
        config, (17, 19), 4, config.init_method, ep_group=group
    )
    full_weights = [
        (torch.arange(rows * 4).reshape(rows, 4).float() / 100).requires_grad_()
        for rows in (17, 19)
    ]
    with torch.no_grad():
        for table, full in zip(embedding.tables, full_weights):
            table.weight.copy_(full[table.row_start : table.row_end])

    def ids(peer):
        far = 0 if empty_owner else 16
        return torch.tensor([[[0, 0], [far, far], [peer % 3, peer % 3], [0, 0]]])

    actual = embedding(ids(rank))
    expected = torch.stack(
        [F.embedding(ids(rank)[..., index], weight) for index, weight in enumerate(full_weights)],
        dim=-2,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    (actual.square().sum() * (rank + 1)).backward()
    for peer in range(size):
        for index, weight in enumerate(full_weights):
            (F.embedding(ids(peer)[..., index], weight).square().sum() * (peer + 1)).backward()
    for table, full in zip(embedding.tables, full_weights):
        torch.testing.assert_close(table.weight.grad, full.grad[table.row_start : table.row_end])
    torch.optim.Adam(embedding.parameters(), lr=0.01).step()
    torch.optim.Adam(full_weights, lr=0.01).step()
    for table, full in zip(embedding.tables, full_weights):
        torch.testing.assert_close(table.weight, full[table.row_start : table.row_end])
