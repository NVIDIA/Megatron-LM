# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Checkpoint GLU layout contracts, including optimizer-state ownership."""

from argparse import Namespace
from copy import deepcopy

import pytest
import torch

from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory
from megatron.training.glu_checkpointing import (
    prepare_glu_checkpoint_for_load,
    prepare_glu_checkpoint_for_save,
    validate_glu_checkpoint_backend,
    validate_glu_optimizer_layout,
)

ROUTED = "decoder.layers.0.mlp.experts.linear_fc1."
SHARED = "decoder.layers.0.mlp.shared_experts.linear_fc1."


def _args(routed=None, shared=None, tp=1, etp=1):
    return Namespace(
        moe_mlp_glu_interleave_size=routed,
        moe_shared_expert_glu_interleave_size=shared,
        use_grouped_gemm_for_shared_expert=shared is not None,
        tensor_model_parallel_size=tp,
        expert_tensor_parallel_size=etp,
        ckpt_format="torch_dist",
    )


def _interleaved(tensor, size, axis=0):
    """Reference layout: alternate complete gate and up channel blocks."""
    gate, up = tensor.chunk(2, dim=axis)
    blocks = [
        block
        for pair in zip(gate.split(size, dim=axis), up.split(size, dim=axis))
        for block in pair
    ]
    return torch.cat(blocks, dim=axis)


def _canonical(shape):
    return torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)


def _saved_layout(routed=None, shared=None, tp=1, etp=1):
    return {
        "model": "contiguous",
        "optimizer": {"routed": routed, "shared": shared},
        "tensor_model_parallel_size": tp,
        "expert_tensor_parallel_size": etp,
    }


@pytest.mark.parametrize("size", [2, 32])
def test_loaded_fc1_preserves_complete_swiglu_mlp_output(size):
    """The checkpoint conversion must preserve FC1 activation and FC2 semantics."""
    hidden = 3
    intermediate = 2 * size
    fc1 = torch.linspace(-1, 1, 2 * intermediate * hidden).reshape(-1, hidden)
    fc2 = torch.linspace(-0.5, 0.8, hidden * intermediate).reshape(hidden, intermediate)
    inputs = torch.linspace(-0.7, 0.9, 5 * hidden).reshape(-1, hidden)
    loaded = prepare_glu_checkpoint_for_load(
        {"model": {ROUTED + "weight0": fc1}}, _args(routed=size)
    )

    gate, up = torch.nn.functional.linear(inputs, fc1).chunk(2, dim=-1)
    expected = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, fc2)
    projected = torch.nn.functional.linear(inputs, loaded["model"][ROUTED + "weight0"])
    blocks = projected.reshape(inputs.shape[0], intermediate // size, 2, size)
    actual_gate = blocks[:, :, 0, :].flatten(1)
    actual_up = blocks[:, :, 1, :].flatten(1)
    actual = torch.nn.functional.linear(torch.nn.functional.silu(actual_gate) * actual_up, fc2)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "suffix,shape,axis",
    [("weight0", (8, 3), 0), ("bias0", (8,), 0), ("weight", (3, 8, 3), 1), ("bias", (3, 8), 1)],
)
def test_indexed_and_single_weight_and_bias_use_channel_axis(suffix, shape, axis):
    original = _canonical(shape)
    checkpoint = {"model": {ROUTED + suffix: original}}
    loaded = prepare_glu_checkpoint_for_load(checkpoint, _args(routed=2))

    torch.testing.assert_close(
        loaded["model"][ROUTED + suffix], _interleaved(original, 2, axis), rtol=0, atol=0
    )
    saved = prepare_glu_checkpoint_for_save(loaded, _args(routed=2))
    torch.testing.assert_close(saved["model"][ROUTED + suffix], original, rtol=0, atol=0)
    assert checkpoint["model"][ROUTED + suffix] is original
    torch.testing.assert_close(original, _canonical(shape), rtol=0, atol=0)


def test_save_restores_canonical_model_without_mutating_master_or_moments():
    canonical = _canonical((8, 3))
    runtime = _interleaved(canonical, 2)
    optimizer = {
        "param_state": {
            "fp32_master": runtime.clone(),
            "exp_avg": runtime.clone() + 100,
            "exp_avg_sq": runtime.clone() + 200,
        }
    }
    snapshots = {key: value.clone() for key, value in optimizer["param_state"].items()}
    checkpoint = {"model": {ROUTED + "weight0": runtime}, "optimizer": optimizer}

    saved = prepare_glu_checkpoint_for_save(checkpoint, _args(routed=2))

    assert saved is not checkpoint
    assert saved["model"] is not checkpoint["model"]
    assert saved["optimizer"] is optimizer
    assert checkpoint["model"][ROUTED + "weight0"] is runtime
    torch.testing.assert_close(runtime, _interleaved(canonical, 2), rtol=0, atol=0)
    torch.testing.assert_close(saved["model"][ROUTED + "weight0"], canonical, rtol=0, atol=0)
    assert saved["glu_checkpoint_layout"] == _saved_layout(routed=2)
    assert "glu_checkpoint_layout" not in checkpoint
    for key, original in snapshots.items():
        torch.testing.assert_close(optimizer["param_state"][key], original, rtol=0, atol=0)


@pytest.mark.parametrize("as_factory", [False, True])
def test_sharded_save_transforms_data_without_changing_sharding_or_optimizer(as_factory):
    canonical = _canonical((8, 3))
    runtime = _interleaved(canonical, 2)
    shard_key = "decoder.layers.mlp.experts.linear_fc1.weight"

    def build(key, tensor, replica_id, flattened_range):
        assert flattened_range is None
        gate, up = tensor.chunk(2, dim=0)
        return [
            ShardedTensor.from_rank_offsets(key, gate, (0, 0, 2), replica_id=replica_id),
            ShardedTensor.from_rank_offsets(key, up, (0, 1, 2), replica_id=replica_id),
        ]

    if as_factory:
        original = ShardedTensorFactory(
            shard_key, runtime, build, lambda parts: torch.cat(parts, dim=0), replica_id=7
        )
    else:
        original = ShardedTensor.from_rank_offsets(shard_key, runtime, (0, 0, 1), replica_id=7)
    optimizer = {"model_factory_or_shard": original}
    checkpoint = {"model": {ROUTED + "weight0": original}, "optimizer": optimizer}

    saved = prepare_glu_checkpoint_for_save(checkpoint, _args(routed=2))
    transformed = saved["model"][ROUTED + "weight0"]

    assert transformed is not original
    assert type(transformed) is type(original)
    assert transformed.key == original.key
    assert transformed.replica_id == original.replica_id
    assert original.data is runtime
    assert saved["optimizer"] is optimizer
    assert optimizer["model_factory_or_shard"] is original
    torch.testing.assert_close(original.data, _interleaved(canonical, 2), rtol=0, atol=0)
    if as_factory:
        assert transformed.build_fn is original.build_fn
        assert transformed.merge_fn is original.merge_fn
        built = transformed.build()
        torch.testing.assert_close(built[0].data, canonical[:4], rtol=0, atol=0)
        torch.testing.assert_close(built[1].data, canonical[4:], rtol=0, atol=0)
    else:
        assert transformed.global_shape == original.global_shape
        assert transformed.global_offset == original.global_offset
        torch.testing.assert_close(transformed.data, canonical, rtol=0, atol=0)


def test_shared_experts_have_independent_layout_and_other_projections_are_untouched():
    tensors = {
        ROUTED + "weight0": _canonical((16, 3)),
        ROUTED + "bias0": _canonical((16,)),
        SHARED + "weight": _canonical((16, 3)),
        SHARED + "bias": _canonical((16,)),
        "decoder.layers.0.mlp.linear_fc1.weight": _canonical((16, 3)),
        "decoder.layers.0.mlp.experts.linear_fc2.weight0": _canonical((3, 8)),
        "decoder.layers.0.mixer.in_proj.weight": _canonical((16, 3)),
    }
    checkpoint = {"model0": tensors, "model1": {}}
    loaded = prepare_glu_checkpoint_for_load(checkpoint, _args(routed=2, shared=4))

    for key, tensor in tensors.items():
        if key.startswith(ROUTED):
            torch.testing.assert_close(loaded["model0"][key], _interleaved(tensor, 2))
        elif key.startswith(SHARED):
            torch.testing.assert_close(loaded["model0"][key], _interleaved(tensor, 4))
        else:
            assert loaded["model0"][key] is tensor
    assert loaded["model1"] == {}


def test_non_grouped_shared_expert_ignores_inactive_interleave_setting():
    args = _args(shared=2)
    args.use_grouped_gemm_for_shared_expert = False
    weight = _canonical((8, 3))
    checkpoint = {"model": {SHARED + "weight": weight}}

    loaded = prepare_glu_checkpoint_for_load(checkpoint, args)
    assert loaded["model"][SHARED + "weight"] is weight
    saved = prepare_glu_checkpoint_for_save(loaded, args)
    assert saved["glu_checkpoint_layout"]["optimizer"]["shared"] is None


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_model_conversion_keeps_checkpoint_precision_for_master_initialization(dtype):
    weight = _canonical((8, 3)).to(dtype)
    checkpoint = {"model": {ROUTED + "weight0": weight}}
    loaded = prepare_glu_checkpoint_for_load(checkpoint, _args(routed=2))

    converted = loaded["model"][ROUTED + "weight0"]
    assert converted.dtype == dtype
    assert converted.data_ptr() != weight.data_ptr()
    torch.testing.assert_close(converted.float(), _interleaved(weight.float(), 2), rtol=0, atol=0)


def test_two_load_save_round_trips_do_not_double_interleave():
    canonical = _canonical((16, 3))
    args = _args(routed=2)
    checkpoint = {"model": {ROUTED + "weight0": canonical}, "args": _args()}
    for _ in range(2):
        loaded = prepare_glu_checkpoint_for_load(checkpoint, args)
        torch.testing.assert_close(
            loaded["model"][ROUTED + "weight0"], _interleaved(canonical, 2), rtol=0, atol=0
        )
        # Real saves write the active run's args, which must not override the marker.
        loaded["args"] = args
        checkpoint = prepare_glu_checkpoint_for_save(loaded, args)
        torch.testing.assert_close(checkpoint["model"][ROUTED + "weight0"], canonical)


@pytest.mark.parametrize("target_size", [2, 4, None])
def test_legacy_native_layout_is_interpreted_from_source_args(target_size):
    canonical = _canonical((16, 3))
    source_runtime = _interleaved(canonical, 2)
    checkpoint = {"model": {ROUTED + "weight0": source_runtime}, "args": _args(routed=2)}

    loaded = prepare_glu_checkpoint_for_load(checkpoint, _args(routed=target_size))

    expected = canonical if target_size is None else _interleaved(canonical, target_size)
    torch.testing.assert_close(loaded["model"][ROUTED + "weight0"], expected, rtol=0, atol=0)
    torch.testing.assert_close(source_runtime, _interleaved(canonical, 2), rtol=0, atol=0)


@pytest.mark.parametrize("source_size,target_size", [(None, 2), (2, None), (2, 4)])
@pytest.mark.parametrize("component", ["routed", "shared"])
def test_optimizer_layout_switch_requires_weights_only_load(source_size, target_size, component):
    source = {component: source_size}
    target = {component: target_size}
    checkpoint = {"glu_checkpoint_layout": _saved_layout(**source), "optimizer": {}}

    with pytest.raises(ValueError):
        validate_glu_optimizer_layout(checkpoint, _args(**target), loading_optimizer=True)
    validate_glu_optimizer_layout(checkpoint, _args(**target), loading_optimizer=False)


def test_optimizer_resume_accepts_matching_runtime_layout():
    checkpoint = {"glu_checkpoint_layout": _saved_layout(routed=2, shared=4), "optimizer": {}}
    validate_glu_optimizer_layout(checkpoint, _args(routed=2, shared=4), loading_optimizer=True)


@pytest.mark.parametrize(
    "source,target",
    [
        (_saved_layout(shared=2), _args(shared=2, tp=2)),
        (_saved_layout(routed=2), _args(routed=2, etp=2)),
    ],
)
def test_interleaved_optimizer_resharding_rejects_tp_or_etp_change(source, target):
    checkpoint = {"glu_checkpoint_layout": source, "optimizer": {}}

    with pytest.raises(ValueError):
        validate_glu_optimizer_layout(checkpoint, target, loading_optimizer=True)
    validate_glu_optimizer_layout(checkpoint, target, loading_optimizer=False)


def test_contiguous_optimizer_does_not_add_new_tp_resharding_restriction():
    checkpoint = {"glu_checkpoint_layout": _saved_layout(), "optimizer": {}}
    validate_glu_optimizer_layout(checkpoint, _args(tp=2, etp=2), loading_optimizer=True)


def test_legacy_interleaved_model_cannot_be_resharded_even_without_optimizer():
    checkpoint = {"args": _args(routed=2)}
    with pytest.raises(ValueError):
        validate_glu_optimizer_layout(checkpoint, _args(routed=2, etp=2), loading_optimizer=False)


@pytest.mark.parametrize("single_weight", [False, True])
def test_adam_update_and_resume_match_canonical_and_uninterrupted_training(single_weight):
    """Real FP32 parameters and Adam moments keep their channel ownership on resume."""
    size, experts, hidden, intermediate = 2, 3, 3, 4
    canonical = torch.linspace(-0.6, 0.8, experts * 2 * intermediate * hidden).reshape(
        experts, 2 * intermediate, hidden
    )
    inputs = torch.linspace(-0.7, 0.9, experts * 5 * hidden).reshape(experts, 5, hidden)
    fc2 = torch.linspace(-0.5, 0.8, hidden * intermediate).reshape(hidden, intermediate)
    keys = [ROUTED + "weight"] if single_weight else [ROUTED + f"weight{i}" for i in range(experts)]
    axis = 1 if single_weight else 0
    source_tensors = [canonical] if single_weight else list(canonical.unbind(0))
    reference = [torch.nn.Parameter(t.clone()) for t in source_tensors]
    runtime = [torch.nn.Parameter(_interleaved(t, size, axis)) for t in source_tensors]
    reference_optimizer = torch.optim.Adam(reference, lr=0.002)
    runtime_optimizer = torch.optim.Adam(runtime, lr=0.002)

    def step(parameters, optimizer, interleaved):
        optimizer.zero_grad(set_to_none=True)
        weights = parameters[0].unbind(0) if single_weight else parameters
        losses = []
        for expert, weight in enumerate(weights):
            projected = torch.nn.functional.linear(inputs[expert], weight)
            if interleaved:
                blocks = projected.reshape(5, intermediate // size, 2, size)
                gate, up = blocks[:, :, 0, :].flatten(1), blocks[:, :, 1, :].flatten(1)
            else:
                gate, up = projected.chunk(2, dim=-1)
            output = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, fc2)
            losses.append(output.square().mean())
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        return loss.detach()

    torch.testing.assert_close(
        step(reference, reference_optimizer, False), step(runtime, runtime_optimizer, True)
    )
    args = _args(routed=size)
    # Copying models an on-disk snapshot; optimizer.state_dict() itself contains aliases.
    checkpoint = deepcopy(
        prepare_glu_checkpoint_for_save(
            {
                "model": {key: parameter.detach() for key, parameter in zip(keys, runtime)},
                "optimizer": runtime_optimizer.state_dict(),
            },
            args,
        )
    )
    for key, parameter in zip(keys, reference):
        torch.testing.assert_close(checkpoint["model"][key], parameter)
    validate_glu_optimizer_layout(checkpoint, args, loading_optimizer=True)
    loaded = prepare_glu_checkpoint_for_load(checkpoint, args)
    resumed = [torch.nn.Parameter(loaded["model"][key].clone()) for key in keys]
    resumed_optimizer = torch.optim.Adam(resumed, lr=0.002)
    resumed_optimizer.load_state_dict(loaded["optimizer"])

    expected_loss = step(reference, reference_optimizer, False)
    torch.testing.assert_close(step(runtime, runtime_optimizer, True), expected_loss)
    torch.testing.assert_close(step(resumed, resumed_optimizer, True), expected_loss)
    resumed_model = prepare_glu_checkpoint_for_save({"model": dict(zip(keys, resumed))}, args)[
        "model"
    ]
    for key, expected, uninterrupted, restored in zip(keys, reference, runtime, resumed):
        torch.testing.assert_close(resumed_model[key], expected)
        torch.testing.assert_close(restored, uninterrupted, rtol=0, atol=0)
        for moment in ("exp_avg", "exp_avg_sq"):
            expected_moment = _interleaved(reference_optimizer.state[expected][moment], size, axis)
            torch.testing.assert_close(resumed_optimizer.state[restored][moment], expected_moment)
            torch.testing.assert_close(
                resumed_optimizer.state[restored][moment],
                runtime_optimizer.state[uninterrupted][moment],
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize("ckpt_format", ["torch_dcp", "fsdp_dtensor"])
def test_backend_guard_leaves_unrelated_layouts_unchanged(ckpt_format):
    validate_glu_checkpoint_backend(
        {}, _args(), ckpt_format=ckpt_format, skip_load_to_model_and_opt=True
    )
    with pytest.raises(NotImplementedError):
        validate_glu_checkpoint_backend({}, _args(routed=2), ckpt_format=ckpt_format)


def test_interleaving_rejects_inplace_load_even_for_supported_checkpoint_format():
    with pytest.raises(NotImplementedError):
        validate_glu_checkpoint_backend(
            {}, _args(routed=2), ckpt_format="torch_dist", skip_load_to_model_and_opt=True
        )


@pytest.mark.parametrize("size", [0, -2, 1.5])
def test_invalid_interleave_size_is_rejected(size):
    with pytest.raises(ValueError):
        prepare_glu_checkpoint_for_load({}, _args(routed=size))
