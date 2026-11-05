# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.hybrid.hybrid_layer_allocation import (
    get_hybrid_layer_counts,
    get_hybrid_total_layer_count,
    parse_hybrid_pattern,
)
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.hybrid.recurrence import (
    LoopDistribution,
    recurrent_layer_indices,
    validate_recurrence_config,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize(
    "pattern,flat,ranges",
    [
        ("-:*--:-", "-*---", ((1, 4),)),
        (":*-:", "*-", ((0, 2),)),
        ("*-", "*-", ()),
        ("-:*:-*:-", "-*-*-", ((1, 2), (2, 4))),
        ("-:*:--:G-:-", "-*--G--", ((1, 2), (2, 4), (4, 6))),
        (":*:-G-:", "*-G-", ((0, 1), (1, 4))),
    ],
)
def test_parse_recurrent_pattern(pattern, flat, ranges):
    parsed = parse_hybrid_pattern(pattern)
    assert parsed.main_pattern == flat
    assert parsed.recurrent_layer_ranges == ranges
    assert get_hybrid_total_layer_count(pattern) == len(flat)
    assert get_hybrid_layer_counts(pattern) == get_hybrid_layer_counts(flat)


@pytest.mark.parametrize(
    "pattern,ranges", [("-:*-:-/-/-", ((1, 3),)), ("-:*:-:-/-/-", ((1, 2), (2, 3)))]
)
def test_recurrent_pattern_with_mtp(pattern, ranges):
    parsed = parse_hybrid_pattern(pattern)
    assert parsed.main_pattern == "-*--"
    assert parsed.recurrent_layer_ranges == ranges
    assert parsed.mtp_pattern == "-"
    assert parsed.mtp_num_depths == 2
    assert get_hybrid_total_layer_count(pattern) == 4
    assert get_hybrid_layer_counts(pattern)["-"] == 5


@pytest.mark.parametrize(
    "pattern", ["-:*", "-::", "-:*::-", "-::*:-", "::", "-:*:|-", "*-/:*-:", "-:x:-"]
)
def test_invalid_recurrent_patterns(pattern):
    with pytest.raises(ValueError):
        parse_hybrid_pattern(pattern)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "1",
        "0:1",
        "-1:1",
        "1.5:1",
        "1:0",
        "1:-1",
        "1:nan",
        "1:inf",
        "1:0.4",
        "1:1,1:1",
        "1:1,",
        "|1:1|",
        "|1:1|2:1|",
    ],
)
def test_invalid_loop_distribution(value):
    with pytest.raises(ValueError):
        LoopDistribution.from_string(value)


@pytest.mark.parametrize("pattern", [":*-:", ":*:-*-:"])
def test_cli_policy_reaches_native_builder(monkeypatch, pattern):
    from megatron.training.argument_utils import hybrid_config_from_args
    from megatron.training.arguments import parse_args, validate_args
    from megatron.training.models.hybrid import HybridModelBuilder

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pretrain_hybrid.py",
            "--hybrid-layer-pattern",
            pattern,
            "--n-loops-distribution",
            "1:.5,4:.5",
            "--hidden-size",
            "128",
            "--num-attention-heads",
            "4",
            "--seq-length",
            "4",
            "--max-position-embeddings",
            "4",
            "--micro-batch-size",
            "1",
        ],
    )
    args = validate_args(parse_args())
    assert args.num_layers == len(pattern.replace(":", ""))
    model_config = hybrid_config_from_args(args)
    assert model_config.n_loops_distribution == "1:.5,4:.5"
    model_config.vocab_size = 32
    monkeypatch.setattr("megatron.training.models.hybrid.HybridModel", lambda **kwargs: kwargs)
    builder = HybridModelBuilder(model_config)
    kwargs = builder.build_model(SimpleNamespace(), pre_process=True, post_process=True)
    assert kwargs["n_loops_distribution"] == "1:.5,4:.5"
    assert kwargs["hybrid_layer_pattern"] == pattern
    with pytest.raises(ValueError, match="FSDP"):
        builder.build_distributed_models(SimpleNamespace(), use_megatron_fsdp=True)


def test_recurrence_boundaries_do_not_shift_dsv4_layer_metadata():
    from megatron.training.models.deepseek_v4 import normalize_dsv4_hybrid_csa_compress_ratios

    args = SimpleNamespace(experimental_attention_variant="dsv4_hybrid", csa_compress_ratios=None)
    config_kwargs = {}
    normalize_dsv4_hybrid_csa_compress_ratios(args, config_kwargs, "C:W-:H")
    assert config_kwargs["csa_compress_ratios"] == [4, 0, 0, 128]


def test_distribution_and_execution_order():
    assert LoopDistribution.from_string(None) == LoopDistribution.from_string("1:1.0")
    policy = LoopDistribution.from_string("1:0.25,2:0.25,4:0.25,8:0.25")
    assert policy.counts == (1, 2, 4, 8)
    assert list(recurrent_layer_indices(4, ((1, 3),), 3)) == [0, 1, 2, 1, 2, 1, 2, 3]
    # Each region finishes all its loops before the next region starts.
    assert list(recurrent_layer_indices(5, ((0, 1), (2, 4)), 2)) == [0, 0, 1, 2, 3, 2, 3, 4]


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_loop_override(count):
    with pytest.raises(ValueError, match="positive integer"):
        list(recurrent_layer_indices(2, ((0, 2),), count))


def test_sampling_replays_checkpointed_rng_and_eval_does_not_sample():
    policy = LoopDistribution.from_string("1:0.5,4:0.5")
    group = Mock()
    group.size.return_value = 1
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        state = torch.get_rng_state()
        first = [policy.sample(True, torch.device("cpu"), group) for _ in range(32)]
        torch.set_rng_state(state)
        assert first == [policy.sample(True, torch.device("cpu"), group) for _ in range(32)]
        assert set(first) == {1, 4}
        state = torch.get_rng_state()
        assert policy.sample(False, torch.device("cpu"), group) == 4
        assert torch.equal(state, torch.get_rng_state())
        assert LoopDistribution.from_string("2:1").sample(True, torch.device("cpu"), None) == 2
        assert torch.equal(state, torch.get_rng_state())
    with pytest.raises(ValueError, match="tp_dp_cp"):
        policy.sample(True, torch.device("cpu"), None)


def test_sampling_uses_explicit_group_and_source(monkeypatch):
    group = Mock()
    group.size.return_value = 2
    monkeypatch.setattr(torch.distributed, "get_backend", lambda pg: "gloo")
    monkeypatch.setattr(torch.distributed, "get_global_rank", lambda pg, rank: 17)
    broadcasts = []

    def broadcast(tensor, src, group):
        broadcasts.append((src, group, tensor.device.type))
        tensor.fill_(4)

    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    assert LoopDistribution.from_string("1:.5,4:.5").sample(True, torch.device("cpu"), group) == 4
    assert broadcasts == [(17, group, "cpu")]


@pytest.mark.parametrize(
    "field,value",
    [
        ("pipeline_model_parallel_size", 2),
        ("virtual_pipeline_model_parallel_size", 2),
        ("context_parallel_size", 2),
        ("recompute_granularity", "full"),
        ("recompute_granularity", "selective"),
        ("cuda_graph_impl", "local"),
        ("fp8", "e4m3"),
        ("fp4", "e2m1"),
        ("moe_shortcut_connection", True),
        ("overlap_moe_expert_parallel_comm", True),
        ("fine_grained_activation_offloading", True),
    ],
)
def test_unsupported_execution_modes(field, value):
    config = TransformerConfig(num_layers=4, hidden_size=128, num_attention_heads=4)
    setattr(config, field, value)
    with pytest.raises(ValueError, match="Hybrid recurrence does not yet support"):
        validate_recurrence_config(config)


@pytest.fixture
def model_factory():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    torch.manual_seed(123)

    def build(
        pattern="-:*-:-", distribution="3:1", dtype=torch.float32, grad_fusion=False, **kwargs
    ):
        config = TransformerConfig(
            num_layers=get_hybrid_total_layer_count(pattern),
            hidden_size=128,
            num_attention_heads=4,
            params_dtype=dtype,
            bf16=dtype == torch.bfloat16,
            use_cpu_initialization=True,
            hidden_dropout=0,
            attention_dropout=0,
            gradient_accumulation_fusion=grad_fusion,
            gdn_kernel_backend="torch",
            deterministic_mode="G" in pattern,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
        )
        return HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=32,
            max_sequence_length=4,
            hybrid_layer_pattern=pattern,
            n_loops_distribution=distribution,
            **kwargs,
        ).cuda()

    yield build
    Utils.destroy_model_parallel()


def _inputs():
    tokens = torch.tensor([[1, 2, 3, 4]], device="cuda")
    positions = torch.arange(4, device="cuda").unsqueeze(0)
    return tokens, positions


@pytest.mark.parametrize(
    "pattern,flat,execution_order",
    [
        ("-:*-:-", "-*--", [0, 1, 2, 1, 2, 1, 2, 3]),
        ("-:G-:-", "-G--", [0, 1, 2, 1, 2, 1, 2, 3]),
        ("-:*:-*:-", "-*-*-", [0, 1, 1, 1, 2, 3, 2, 3, 2, 3, 4]),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_repeated_forward_and_gradients_match_manual_unroll(
    model_factory, pattern, flat, execution_order, dtype
):
    model = model_factory(pattern=pattern, dtype=dtype)
    reference = model_factory(pattern=flat, distribution=None, dtype=dtype)
    reference.load_state_dict(model.state_dict())
    tokens, positions = _inputs()
    actual = model(tokens, positions, None)
    hidden = reference.embedding(tokens, positions)
    for index in execution_order:
        hidden, _ = reference.decoder.layers[index](hidden, attention_mask=None)
    hidden = reference.decoder.final_norm(hidden)
    expected, _ = reference.output_layer(hidden)
    expected = expected.transpose(0, 1).contiguous()
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for (name, parameter), (ref_name, ref_parameter) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == ref_name
        assert parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, ref_parameter.grad, msg=name)


@pytest.mark.parametrize("pattern", ["-:*-:-", "-:*:-*:-"])
def test_one_loop_and_policy_changes_preserve_checkpoint_structure(model_factory, pattern):
    baseline = model_factory(pattern=pattern.replace(":", ""), distribution=None)
    recurrent = model_factory(pattern=pattern, distribution="1:.5,3:.5")
    recurrent.load_state_dict(baseline.state_dict(), strict=True)
    baseline.eval()
    recurrent.eval()
    tokens, positions = _inputs()
    torch.testing.assert_close(
        recurrent(tokens, positions, None, n_loops=1), baseline(tokens, positions, None)
    )
    assert len(recurrent.decoder.layers) == len(baseline.decoder.layers)
    assert sum(p.numel() for p in recurrent.parameters()) == sum(
        p.numel() for p in baseline.parameters()
    )
    assert recurrent.state_dict().keys() == baseline.state_dict().keys()
    baseline_shards = baseline.sharded_state_dict()
    recurrent_shards = recurrent.sharded_state_dict()
    assert recurrent_shards.keys() == baseline_shards.keys()
    for key, shard in recurrent_shards.items():
        if isinstance(shard, ShardedTensor):
            expected = baseline_shards[key]
            assert (shard.key, shard.global_shape, shard.global_offset) == (
                expected.key,
                expected.global_shape,
                expected.global_offset,
            )
    restarted = model_factory(pattern=pattern, distribution="2:1")
    restarted.load_state_dict(recurrent.state_dict(), strict=True)
    assert restarted.decoder.loop_distribution.counts == (2,)


@pytest.mark.parametrize("pattern", ["-:*-:-", "-:*:-*:-"])
def test_distributed_checkpoint_policy_restart(model_factory, tmp_path_dist_ckpt, pattern):
    baseline = model_factory(pattern=pattern.replace(":", ""), distribution=None)
    path = tmp_path_dist_ckpt / f"hybrid_recurrence_{len(baseline.decoder.layers)}"
    path.mkdir(parents=True, exist_ok=True)
    dist_checkpointing.save({"model": baseline.sharded_state_dict()}, path)
    recurrent = model_factory(pattern=pattern, distribution="2:1")
    loaded = dist_checkpointing.load({"model": recurrent.sharded_state_dict()}, path)
    recurrent.load_state_dict(loaded["model"], strict=True)
    baseline.eval()
    recurrent.eval()
    tokens, positions = _inputs()
    torch.testing.assert_close(
        recurrent(tokens, positions, None, n_loops=1), baseline(tokens, positions, None)
    )
    assert recurrent.decoder.loop_distribution.counts == (2,)


@pytest.mark.parametrize("grad_fusion", [False, True])
@pytest.mark.parametrize("pattern", ["-:*-:-", "-:*:-*:-"])
def test_ddp_accumulates_shared_weight_gradients(model_factory, grad_fusion, pattern):
    model = model_factory(pattern=pattern, grad_fusion=grad_fusion)
    reference = model_factory(pattern=pattern)
    reference.load_state_dict(model.state_dict())
    ddp = DistributedDataParallel(
        config=model.config,
        ddp_config=DistributedDataParallelConfig(overlap_grad_reduce=True),
        module=model,
        pg_collection=model.pg_collection,
    )
    ddp.zero_grad_buffer()
    tokens, positions = _inputs()
    with ddp.no_sync():
        ddp(tokens, positions, None, n_loops=1).square().mean().backward()
    ddp(tokens, positions, None, n_loops=3).square().mean().backward()
    ddp.finish_grad_sync()
    reference(tokens, positions, None, n_loops=1).square().mean().backward()
    reference(tokens, positions, None, n_loops=3).square().mean().backward()
    for parameter, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter.main_grad, expected.grad)
    assert any(p.main_grad.abs().sum() > 0 for p in model.decoder.layers[1].parameters())


@pytest.mark.parametrize(
    "pattern,execution_order",
    [("-:*-:-", [0, 1, 2, 1, 2, 1, 2, 3]), ("-:*:-*:-", [0, 1, 1, 1, 2, 3, 2, 3, 2, 3, 4])],
)
def test_override_and_eval_execution_order(model_factory, pattern, execution_order):
    model = model_factory(pattern=pattern, distribution="1:.5,3:.5")
    calls = []
    handles = [
        layer.register_forward_hook(lambda module, inputs, output, i=i: calls.append(i))
        for i, layer in enumerate(model.decoder.layers)
    ]
    norm_calls = []
    handles.append(
        model.decoder.final_norm.register_forward_hook(lambda *args: norm_calls.append(1))
    )
    tokens, positions = _inputs()
    try:
        model.eval()
        state = torch.get_rng_state()
        model(tokens, positions, None)
        assert torch.equal(state, torch.get_rng_state())
        assert calls == execution_order
        assert norm_calls == [1]
        calls.clear()
        model(tokens, positions, None, n_loops=1)
        assert calls == list(range(len(model.decoder.layers)))
        assert model.decoder.loop_distribution.counts == (1, 3)
        with pytest.raises(ValueError, match="cached inference"):
            model(tokens, positions, None, inference_context=SimpleNamespace())
        calls.clear()
        with pytest.raises(ValueError, match="positive integer"):
            model(tokens, positions, None, n_loops=0)
        assert calls == []
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize(
    "pattern,execution_order",
    [("-:*-:-", [0, 1, 2, 1, 2, 1, 2, 3]), ("-:*:-*:-", [0, 1, 1, 1, 2, 3, 2, 3, 2, 3, 4])],
)
def test_training_resamples_once_per_forward(model_factory, monkeypatch, pattern, execution_order):
    model = model_factory(pattern=pattern, distribution="1:.5,3:.5")
    draws = iter([torch.tensor(0.1), torch.tensor(0.9)])
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: next(draws))
    calls = []
    handles = [
        layer.register_forward_hook(lambda module, inputs, output, i=i: calls.append(i))
        for i, layer in enumerate(model.decoder.layers)
    ]
    tokens, positions = _inputs()
    try:
        model(tokens, positions, None)
        assert calls == list(range(len(model.decoder.layers)))
        calls.clear()
        model(tokens, positions, None)
        assert calls == execution_order
    finally:
        for handle in handles:
            handle.remove()


def test_later_recompute_setting_cannot_bypass_recurrence(model_factory):
    model = model_factory()
    model.decoder.config.recompute_granularity = "full"
    tokens, positions = _inputs()
    with pytest.raises(ValueError, match="activation recomputation"):
        model(tokens, positions, None)


def test_distribution_requires_boundaries(model_factory):
    with pytest.raises(ValueError, match="requires ':' boundaries"):
        model_factory(pattern="-*--")
    with pytest.raises(ValueError, match="complete non-pipelined"):
        model_factory(pre_process=False)
