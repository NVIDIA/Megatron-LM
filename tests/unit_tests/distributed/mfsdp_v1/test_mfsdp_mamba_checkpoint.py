# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Native Mamba FSDP checkpoint layout, Adam-state and resumed-update regression."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.tensor import DTensor

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_mixer import MambaMixer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data():
    """This synthetic checkpoint test needs no downloaded test assets."""


def _build(tp_size, strategy, seed, *, lr):
    model_parallel_cuda_manual_seed(seed)
    config = TransformerConfig(
        hidden_size=256,
        num_layers=1,
        num_attention_heads=8,
        tensor_model_parallel_size=tp_size,
        sequence_parallel=tp_size > 1,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
    )
    mixer = MambaMixer(
        config,
        hybrid_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
        config.hidden_size,
        layer_number=1,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"]),
    ).cuda()
    wrapped = FullyShardedDataParallel(
        config=config,
        ddp_config=DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            data_parallel_sharding_strategy=strategy,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=False,
        ),
        module=Float16Module(config, mixer),
        fsdp_unit_modules=[MambaMixer],
    )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam", lr=lr, bf16=True, use_distributed_optimizer=True, clip_grad=0.0
        ),
        [wrapped],
    )
    return wrapped, optimizer


def _raw_state(model, optimizer, *, loading=False):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.sharded_state_dict(
            {}, is_loading=loading, metadata={"distrib_optim_sharding_type": "fsdp_dtensor"}
        ),
    }


def _preprocess(model, state, *, checkpoint_metadata=None):
    from megatron.training.checkpointing import preprocess_fsdp_dtensor_state_dict

    return preprocess_fsdp_dtensor_state_dict(
        SimpleNamespace(swiglu=False, num_experts=None),
        state,
        model,
        checkpoint_metadata=checkpoint_metadata,
    )


def _snapshot(tree):
    if isinstance(tree, DTensor):
        return (tuple(tree.shape), tree.to_local().detach().cpu().clone())
    if isinstance(tree, torch.Tensor):
        return tree.detach().cpu().clone()
    if isinstance(tree, dict):
        return {key: _snapshot(value) for key, value in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(_snapshot(value) for value in tree)
    return tree


def _assert_exact_local(left, right, path="state"):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0, msg=lambda msg: f"{path}: {msg}")
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_exact_local(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_exact_local(a, b, f"{path}[{index}]")
    else:
        assert left == right, path


def _assert_exact(left, right):
    # A rank-local assertion must not leave peers entering the next collective.
    error = None
    try:
        _assert_exact_local(left, right)
    except AssertionError as exc:
        error = str(exc)
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    assert not any(errors), errors


def _step(model, optimizer):
    optimizer.zero_grad()
    model.zero_grad_buffer()
    # Same sequence on every DP rank; TP ranks own consecutive sequence slices.
    torch.manual_seed(987)
    x = torch.randn(32, 1, 256, device="cuda", dtype=torch.bfloat16)
    tp = model.config.tensor_model_parallel_size
    x = x.chunk(tp, dim=0)[dist.get_rank() % tp].contiguous()
    output, _ = model(x)
    loss = output.float().square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    # Include TP synchronization of sequence-parallel layernorm gradients.
    finalize_model_grads([model])
    success, _, _ = optimizer.step()
    assert success
    return loss.detach().clone()


@pytest.mark.internal
@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("strategy", ["optim_grads", "optim_grads_params"])
@pytest.mark.parametrize("legacy_layout", [False, True])
def test_mamba_fsdp_roundtrip(tp_size, strategy, legacy_layout, tmp_path_dist_ckpt, monkeypatch):
    """Check disk keys/shapes, exact weights/moments and the next Adam update.

    Disabling the new handler is a negative control: a component-layout checkpoint
    cannot load into the old fused-key template with strict DCP loading.
    """
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp_size)
    try:
        source, source_optimizer = _build(tp_size, strategy, 123, lr=1e-3)
        # The loading template initializes Adam with one dummy step. Use more
        # steps here so equality also proves the saved step counter is restored.
        for _ in range(3):
            _step(source, source_optimizer)
        before = _snapshot(_raw_state(source, source_optimizer))
        state = _preprocess(source, _raw_state(source, source_optimizer))
        projection = next(key for key in state["model"] if key.endswith("in_proj.weight.z"))
        prefix = projection.removesuffix("in_proj.weight.z")
        # These names/sizes are independent of the preprocessing implementation.
        mixer = source.module.module.module
        expected = {
            "in_proj.weight": dict(
                z=mixer.d_inner,
                x=mixer.d_inner,
                B=mixer.ngroups * mixer.d_state,
                C=mixer.ngroups * mixer.d_state,
                dt=mixer.nheads,
            ),
            "conv1d.weight": dict(
                x=mixer.d_inner, B=mixer.ngroups * mixer.d_state, C=mixer.ngroups * mixer.d_state
            ),
            "conv1d.bias": dict(
                x=mixer.d_inner, B=mixer.ngroups * mixer.d_state, C=mixer.ngroups * mixer.d_state
            ),
        }
        for fused, components in expected.items():
            assert prefix + fused not in state["model"]
            for component, rows in components.items():
                key = f"{prefix}{fused}.{component}"
                assert state["model"][key].shape[0] == rows
                for moment in ("exp_avg", "exp_avg_sq"):
                    matched = [
                        value[moment]
                        for name, value in state["optimizer"]["state"].items()
                        if name.endswith(f"{fused}.{component}")
                    ]
                    assert len(matched) == 1
                    assert matched[0].shape == state["model"][key].shape
        import megatron.training.checkpointing as checkpointing

        if legacy_layout:
            # Reproduce the old save path independently of metadata detection.
            with monkeypatch.context() as patch:
                patch.setattr(
                    checkpointing,
                    "handle_mamba_in_state_dict",
                    lambda model, msd, osd, **kwargs: (msd, osd),
                )
                state = _preprocess(source, _raw_state(source, source_optimizer))
        path = tmp_path_dist_ckpt / f"mamba-{strategy}-tp{tp_size}-legacy{legacy_layout}"
        dcp.save(state, checkpoint_id=path)

        destination, destination_optimizer = _build(tp_size, strategy, 456, lr=2e-3)
        raw = _raw_state(destination, destination_optimizer, loading=True)
        # Restore raw aliases after loading, as the training checkpoint path does.
        metadata = dcp.FileSystemReader(path).read_metadata().state_dict_metadata
        target = _preprocess(destination, raw, checkpoint_metadata=metadata)
        dcp.load(target, checkpoint_id=path)
        destination.load_state_dict(raw["model"], strict=False)
        destination_optimizer.load_state_dict(raw["optimizer"])
        _assert_exact(before, _snapshot(_raw_state(destination, destination_optimizer)))
        _assert_exact(_step(source, source_optimizer), _step(destination, destination_optimizer))
        _assert_exact(
            _snapshot(_raw_state(source, source_optimizer)),
            _snapshot(_raw_state(destination, destination_optimizer)),
        )

        if not legacy_layout:
            with monkeypatch.context() as patch:
                patch.setattr(
                    checkpointing,
                    "handle_mamba_in_state_dict",
                    lambda model, msd, osd, **kwargs: (msd, osd),
                )
                broken = _preprocess(destination, _raw_state(destination, destination_optimizer))
                with pytest.raises(dcp.CheckpointException, match="Missing key"):
                    dcp.load(broken, checkpoint_id=path)
    finally:
        Utils.destroy_model_parallel()
