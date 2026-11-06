# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import random
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

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
from megatron.training.state import TrainState
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
@pytest.mark.parametrize("inference_n_loops", [None, 2])
def test_cli_policy_reaches_native_builder(monkeypatch, pattern, inference_n_loops):
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
        ]
        + (
            ["--inference-n-loops", str(inference_n_loops)] if inference_n_loops is not None else []
        ),
    )
    args = validate_args(parse_args())
    assert args.num_layers == len(pattern.replace(":", ""))
    model_config = hybrid_config_from_args(args)
    assert model_config.n_loops_distribution == "1:.5,4:.5"
    assert model_config.inference_n_loops == inference_n_loops
    model_config.vocab_size = 32
    monkeypatch.setattr("megatron.training.models.hybrid.HybridModel", lambda **kwargs: kwargs)
    builder = HybridModelBuilder(model_config)
    kwargs = builder.build_model(SimpleNamespace(), pre_process=True, post_process=True)
    assert kwargs["n_loops_distribution"] == "1:.5,4:.5"
    assert kwargs["inference_n_loops"] == inference_n_loops
    assert kwargs["hybrid_layer_pattern"] == pattern
    with pytest.raises(ValueError, match="FSDP"):
        builder.build_distributed_models(SimpleNamespace(), use_megatron_fsdp=True)


@pytest.mark.parametrize("value", [0, -1])
def test_cli_rejects_non_positive_inference_loop_counts(monkeypatch, value):
    from megatron.training.arguments import parse_args, validate_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pretrain_hybrid.py",
            "--hybrid-layer-pattern",
            ":*-:",
            "--inference-n-loops",
            str(value),
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
    with pytest.raises(ValueError, match="inference_n_loops must be a positive integer"):
        validate_args(parse_args())


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


def test_sampling_is_stateless_and_does_not_communicate(monkeypatch):
    policy = LoopDistribution.from_string("1:0.5,3:0.5")

    def forbidden(*args, **kwargs):
        pytest.fail("Loop sampling must not use global RNG draws, reseeding, or collectives")

    monkeypatch.setattr(torch, "rand", forbidden)
    monkeypatch.setattr(random, "random", forbidden)
    monkeypatch.setattr(random, "seed", forbidden)
    for name in ("broadcast", "all_reduce", "all_gather"):
        monkeypatch.setattr(torch.distributed, name, forbidden)
    python_state = random.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
            state = torch.get_rng_state()
            first = [policy.sample(True, step=step) for step in range(64)]
            assert torch.equal(state, torch.get_rng_state())
            assert python_state == random.getstate()
            # Pin the mapping so future changes cannot silently change restart behavior.
            assert first[:8] == [3, 1, 3, 1, 1, 3, 3, 1]
            for rank_seed in (13, 29):
                torch.manual_seed(rank_seed)
                random.setstate(random.Random(rank_seed).getstate())
                rank_python_state = random.getstate()
                rank_torch_state = torch.get_rng_state()
                restored_policy = LoopDistribution.from_string("1:0.5,3:0.5")
                assert first == [restored_policy.sample(True, step=step) for step in range(64)]
                assert rank_python_state == random.getstate()
                assert torch.equal(rank_torch_state, torch.get_rng_state())
            assert policy.sample(False) == 3
            assert LoopDistribution.from_string("2:1").sample(True) == 2
    finally:
        random.setstate(python_state)


@pytest.mark.parametrize("step", [None, -1, True, 1.5, "1"])
def test_stochastic_sampling_requires_a_logical_step(step):
    with pytest.raises(
        ValueError, match="^Stochastic hybrid recurrence requires a non-negative integer step$"
    ):
        LoopDistribution.from_string("1:.5,4:.5").sample(True, step=step)


def test_step_sampling_respects_distribution():
    policy = LoopDistribution.from_string("1:.25,2:.25,4:.25,8:.25")
    frequencies = {count: 0 for count in policy.counts}
    for step in range(4096):
        frequencies[policy.sample(True, step=step)] += 1
    for count in policy.counts:
        assert frequencies[count] / 4096 == pytest.approx(0.25, abs=0.03)


@pytest.mark.parametrize("pattern", ["-*--", "-:*-:-", "-:*:-*:-"])
def test_pretrain_supplies_minibatch_step_to_all_microbatches(monkeypatch, pattern):
    import pretrain_hybrid

    train_state = TrainState(iteration=11)
    model = Mock(pg_collection=object(), vp_stage=None)
    monkeypatch.setattr(
        pretrain_hybrid, "get_args", lambda: SimpleNamespace(hybrid_layer_pattern=pattern)
    )
    monkeypatch.setattr(pretrain_hybrid, "get_train_state", lambda: train_state)
    monkeypatch.setattr(pretrain_hybrid, "get_timers", lambda: Mock())
    batch = SimpleNamespace(get_batch=lambda: {}, get_packed_seq_params=lambda: None)
    monkeypatch.setattr(pretrain_hybrid, "get_batch", lambda *args, **kwargs: batch)
    monkeypatch.setattr(
        "megatron.training.distillation.logits_saver.get_logits_saver", lambda: None
    )
    stimer = MagicMock()
    stimer.return_value = stimer
    monkeypatch.setattr(pretrain_hybrid, "stimer", stimer)
    pretrain_hybrid.forward_step(None, model)
    pretrain_hybrid.forward_step(None, model)
    train_state.iteration += 1
    pretrain_hybrid.forward_step(None, model)
    if ":" in pattern:
        assert [call.kwargs["recurrence_step"] for call in model.call_args_list] == [11, 11, 12]
    else:
        assert all("recurrence_step" not in call.kwargs for call in model.call_args_list)


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


@pytest.mark.parametrize(
    "pattern,bodies", [("-:*-:-", ((1, 2),)), ("-:G-:-", ((1, 2),)), ("-:*:-*:-", ((1,), (2, 3)))]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_recurrence_matches_independent_layer_instances(model_factory, pattern, bodies, dtype):
    """Compare reused modules with fresh, untied modules for every logical execution."""
    model = model_factory(pattern=pattern, dtype=dtype)
    flat = pattern.replace(":", "")
    tokens, positions = _inputs()
    # Reuse the recurrent model across changing depths and shapes to expose stale state.
    for n_loops, seq_len in [(3, 4), (1, 3), (2, 2), (3, 4)]:
        execution_order = [0]
        for body in bodies:
            for _ in range(n_loops):
                execution_order.extend(body)
        execution_order.append(len(flat) - 1)
        reference = model_factory(
            pattern="".join(flat[index] for index in execution_order),
            distribution=None,
            dtype=dtype,
        )
        reference.embedding.load_state_dict(model.embedding.state_dict())
        reference.decoder.final_norm.load_state_dict(model.decoder.final_norm.state_dict())
        reference.output_layer.load_state_dict(model.output_layer.state_dict())
        for copy, index in zip(reference.decoder.layers, execution_order, strict=True):
            copy.load_state_dict(model.decoder.layers[index].state_dict())
        assert len({id(layer) for layer in reference.decoder.layers}) == len(execution_order)
        copy_parameters = [
            parameter.data_ptr()
            for layer in reference.decoder.layers
            for parameter in layer.parameters()
        ]
        assert len(copy_parameters) == len(set(copy_parameters))
        source_parameters = {parameter.data_ptr() for parameter in model.parameters()}
        assert source_parameters.isdisjoint(
            parameter.data_ptr() for parameter in reference.parameters()
        )

        model.zero_grad(set_to_none=True)
        decoder_input = torch.randn(
            seq_len, 1, model.config.hidden_size, device="cuda", dtype=dtype, requires_grad=True
        )
        reference_input = decoder_input.detach().clone().requires_grad_()
        actual = model(tokens, positions, None, decoder_input=decoder_input, n_loops=n_loops)
        expected = reference(tokens, positions, None, decoder_input=reference_input)
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        expected.square().sum().backward()
        assert decoder_input.grad is not None and reference_input.grad is not None
        torch.testing.assert_close(decoder_input.grad, reference_input.grad)
        for module, copy in (
            (model.decoder.final_norm, reference.decoder.final_norm),
            (model.output_layer, reference.output_layer),
        ):
            for parameter, other in zip(module.parameters(), copy.parameters(), strict=True):
                assert parameter.grad is not None and other.grad is not None
                torch.testing.assert_close(parameter.grad, other.grad)
        copies = [dict(layer.named_parameters()) for layer in reference.decoder.layers]
        for index, layer in enumerate(model.decoder.layers):
            occurrences = [i for i, source in enumerate(execution_order) if source == index]
            for name, parameter in layer.named_parameters():
                assert parameter.grad is not None, name
                # Accumulate in backward execution order, in the parameter's gradient dtype.
                accumulated = torch.zeros_like(parameter.grad)
                for occurrence in reversed(occurrences):
                    gradient = copies[occurrence][name].grad
                    assert gradient is not None, name
                    accumulated.add_(gradient)
                torch.testing.assert_close(
                    parameter.grad, accumulated, msg=f"layer {index}: {name}"
                )


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


@pytest.mark.parametrize("pattern", ["-:*-:-", "-:*:-*:-"])
def test_non_recurrent_checkpoint_resumes_optimizer_and_loop_training(
    model_factory, tmp_path, pattern
):
    baseline = model_factory(pattern=pattern.replace(":", ""), distribution=None)
    optimizer = torch.optim.AdamW(baseline.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    train_state = TrainState()
    tokens, positions = _inputs()

    def update(model, opt, lr_scheduler, **kwargs):
        opt.zero_grad(set_to_none=True)
        loss = model(tokens, positions, None, **kwargs).square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        for parameter in model.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
        opt.step()
        lr_scheduler.step()

    update(baseline, optimizer, scheduler)
    train_state.iteration += 1
    train_state.consumed_train_samples = 1
    path = tmp_path / f"non_recurrent_rank_{Utils.rank}.pt"
    torch.save(
        {
            "model": baseline.state_dict(),
            "hybrid_layer_pattern": baseline.hybrid_layer_pattern,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_state": train_state.state_dict(),
        },
        path,
    )
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert ":" not in loaded["hybrid_layer_pattern"]
    recurrent = model_factory(pattern=pattern, distribution="2:.5,4:.5")
    recurrent.load_state_dict(loaded["model"], strict=True)
    resumed_optimizer = torch.optim.AdamW(recurrent.parameters(), lr=1e-3)
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(resumed_optimizer, step_size=1, gamma=0.9)
    resumed_optimizer.load_state_dict(loaded["optimizer"])
    resumed_scheduler.load_state_dict(loaded["scheduler"])
    resumed_state = TrainState()
    resumed_state.load_state_dict(loaded["train_state"])
    assert resumed_state.iteration == train_state.iteration
    assert resumed_state.consumed_train_samples == train_state.consumed_train_samples
    assert resumed_scheduler.state_dict() == scheduler.state_dict()
    for parameter, restored in zip(baseline.parameters(), recurrent.parameters(), strict=True):
        for name, value in optimizer.state[parameter].items():
            torch.testing.assert_close(value, resumed_optimizer.state[restored][name])

    # One-loop continuation must reproduce both the output and the next optimizer update.
    with torch.no_grad():
        torch.testing.assert_close(
            recurrent(tokens, positions, None, n_loops=1), baseline(tokens, positions, None)
        )
    update(baseline, optimizer, scheduler)
    update(recurrent, resumed_optimizer, resumed_scheduler, n_loops=1)
    resumed_state.iteration += 1
    for parameter, restored in zip(baseline.parameters(), recurrent.parameters(), strict=True):
        torch.testing.assert_close(parameter, restored)
    assert resumed_scheduler.get_last_lr() == scheduler.get_last_lr()

    # The restored optimizer can then train the same weights at a greater recurrent depth.
    calls = []
    handle = recurrent.decoder.layers[1].register_forward_hook(lambda *args: calls.append(1))
    previous_weight = recurrent.output_layer.weight.detach().clone()
    try:
        update(
            recurrent, resumed_optimizer, resumed_scheduler, recurrence_step=resumed_state.iteration
        )
        assert calls == [1, 1, 1, 1]
        assert not torch.equal(previous_weight, recurrent.output_layer.weight)
    finally:
        handle.remove()


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
def test_training_keeps_one_count_per_minibatch(model_factory, pattern, execution_order):
    model = model_factory(pattern=pattern, distribution="1:.5,3:.5")
    calls = []
    handles = [
        layer.register_forward_hook(lambda module, inputs, output, i=i: calls.append(i))
        for i, layer in enumerate(model.decoder.layers)
    ]
    tokens, positions = _inputs()
    try:
        # Step 0 uses three loops; step 1 uses one. Returning to step 0 tests retry/replay.
        for step in (0, 1, 0):
            for _ in range(3):
                calls.clear()
                model(tokens, positions, None, recurrence_step=step)
                expected = list(range(len(model.decoder.layers))) if step == 1 else execution_order
                assert calls == expected
        calls.clear()
        with pytest.raises(ValueError, match="non-negative integer step"):
            model(tokens, positions, None)
        assert calls == []
    finally:
        for handle in handles:
            handle.remove()


def test_later_recompute_setting_cannot_bypass_recurrence(model_factory):
    model = model_factory()
    model.decoder.config.recompute_granularity = "full"
    tokens, positions = _inputs()
    with pytest.raises(ValueError, match="activation recomputation"):
        model(tokens, positions, None)


@pytest.mark.parametrize("inference_n_loops", [0, -1, True, 1.5])
def test_invalid_inference_loop_counts(model_factory, inference_n_loops):
    with pytest.raises(ValueError, match="inference_n_loops must be a positive integer"):
        model_factory(inference_n_loops=inference_n_loops)


@pytest.mark.parametrize("pattern", ["-:*-:-", "-:*:-*:-"])
@pytest.mark.parametrize(
    "inference_n_loops,override,expected",
    [(None, None, 3), (2, None, 2), (5, None, 5), (2, 1, 1), (None, 2, 2)],
)
def test_inference_loop_count_precedence(
    model_factory, pattern, inference_n_loops, override, expected
):
    model = model_factory(
        pattern=pattern, distribution="1:.5,3:.5", inference_n_loops=inference_n_loops
    )
    calls = []
    handle = model.decoder.layers[1].register_forward_hook(lambda *args: calls.append(1))
    tokens, positions = _inputs()
    try:
        with torch.no_grad():
            model.eval()
            model(tokens, positions, None, n_loops=override)
            assert len(calls) == expected
            calls.clear()
            model.train()
            model(tokens, positions, None, recurrence_step=0)
            assert len(calls) == 3  # The inference setting must not change training sampling.
            calls.clear()
            model.eval()
            model(tokens, positions, None)
            assert len(calls) == (3 if inference_n_loops is None else inference_n_loops)
            with pytest.raises(ValueError, match="cached inference"):
                model(tokens, positions, None, inference_context=SimpleNamespace())
    finally:
        handle.remove()


def test_distribution_requires_boundaries(model_factory):
    with pytest.raises(ValueError, match="requires ':' boundaries"):
        model_factory(pattern="-*--")
    with pytest.raises(ValueError, match="requires ':' boundaries"):
        model_factory(pattern="-*--", distribution=None, inference_n_loops=2)
    with pytest.raises(ValueError, match="complete non-pipelined"):
        model_factory(pre_process=False)
