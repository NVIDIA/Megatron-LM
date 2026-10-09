# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Import canonical checkpoints into real DDP MoE models with active or inactive GLU layouts."""

import sys
from copy import deepcopy

import pytest
import torch

from megatron.core import dist_checkpointing
from megatron.core.distributed import DistributedDataParallel
from megatron.core.enums import ModelType
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.training import checkpointing
from megatron.training.global_vars import get_args, set_args
from megatron.training.training import force_param_sync, setup_model_and_optimizer
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe import test_moe_single_grouped_weight_numerics as numerics

# The GB200 CI selector scans this file's text before pytest resolves inherited marks.
pytestmark = [pytest.mark.launch_on_gb200, *numerics.pytestmark]


@pytest.fixture
def moe_case(monkeypatch):
    # Reuse the existing DDP/DistOpt model and batch setup, without inheriting its tests.
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    # Set this before setup_method snapshots the environment, so monkeypatch restores
    # the caller's value after teardown_method finishes.
    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    case = numerics.TestMoESingleGroupedWeightNumerics()
    case.setup_method(None)
    try:
        Utils.initialize_distributed()
        yield case
    finally:
        case.teardown_method(None)


def _enable_native_glu_fusion(monkeypatch):
    """Enable native TE fusion locally even if conftest imported TE before this fixture."""
    from transformer_engine.pytorch.ops.fused import grouped_mlp
    from transformer_engine.pytorch.ops.fuser import OperationFuser

    fused_cls = grouped_mlp.GroupedMLP_CuTeGEMMGLU
    # Older TE caches False and omits registration when the import-time env is unset.
    # Re-run the original capability checks without changing that shared cache.
    uncached_check = getattr(fused_cls.is_supported, "__wrapped__", None)
    if uncached_check is not None:
        monkeypatch.setattr(fused_cls, "is_supported", classmethod(uncached_check))
    native_fusion = getattr(grouped_mlp, "fuse_glu_ops", None)
    if native_fusion is None:
        native_fusion = grouped_mlp.fuse_ops
    callbacks = OperationFuser.forward_backward_fusion_functions
    if native_fusion not in callbacks:
        # Replace rather than mutate the shared list; pytest restores it after this case.
        monkeypatch.setattr(
            OperationFuser, "forward_backward_fusion_functions", [native_fusion, *callbacks]
        )


def _setup_model(
    case,
    directory,
    *,
    single_weight,
    single_bias,
    interleave,
    use_op_fuser,
    precision,
    load,
    gated=True,
    restore_optimizer=False,
):
    args = case.create_test_args(
        precision=precision,
        primary_param_gather=False,
        single_weight=single_weight,
        gradient_accumulation_fusion=False,
        use_transformer_engine_op_fuser=use_op_fuser,
    )
    args.add_bias_linear = True
    args.add_qkv_bias = True
    args.swiglu = gated
    args.bias_gelu_fusion = False
    args.bias_swiglu_fusion = False
    args.bias_dropout_fusion = False
    args.moe_single_grouped_bias = single_bias
    args.moe_mlp_glu_interleave_size = interleave
    args.save = directory
    args.load = directory if load else None
    args.ckpt_format = "torch_dist"
    args.use_dist_ckpt = True
    args.auto_detect_ckpt_format = False
    args.async_save = False
    args.ckpt_assume_constant_structure = False
    args.ckpt_load_validate_sharding_integrity = True
    args.dist_ckpt_strictness = "assume_ok_unexpected"
    args.no_save_optim = not restore_optimizer
    args.no_load_optim = not restore_optimizer
    args.no_save_rng = True
    args.no_load_rng = True
    args.load_main_params_from_ckpt = True
    set_args(args)
    torch.manual_seed(1234)
    Utils.initialize_model_parallel()
    model, optimizer, scheduler = setup_model_and_optimizer(
        model_type=ModelType.encoder_or_decoder, model_provider_func=case.model_provider
    )
    assert isinstance(model[0], DistributedDataParallel)
    experts = [module for module in model[0].modules() if isinstance(module, TEGroupedMLP)]
    assert len(experts) == 1
    experts = experts[0]
    for linear in (experts.linear_fc1, experts.linear_fc2):
        assert linear.use_bias
        assert linear.single_grouped_weight == single_weight
        assert linear.single_grouped_bias == single_bias
    assert experts._with_fused_impl == use_op_fuser
    assert experts.config.gated_linear_unit == gated
    assert not experts.config.bias_activation_fusion
    if load and not restore_optimizer:
        # setup_model_and_optimizer calls the real loader once, including master initialization.
        assert args.iteration == 1
        # Rebuild model storage from the loaded FP32 masters before checking exact rows.
        optimizer.quantize_and_sync_model_params_from_main_params()
    return model, optimizer, scheduler, experts


def _snapshot_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _snapshot_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_snapshot_state(item) for item in value)
    return deepcopy(value)


def _assert_state_equal(actual, expected, path="state"):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(
            actual, expected, rtol=0, atol=0, msg=lambda message: f"{path}: {message}"
        )
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            _assert_state_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_state_equal(actual_item, expected_item, f"{path}[{index}]")
    else:
        assert actual == expected, path


def _snapshot_training_state(model, optimizer, scheduler):
    args = get_args()
    optimizer_states = []
    for part in getattr(optimizer, "chained_optimizers", [optimizer]):
        assert isinstance(part, DistributedOptimizer)
        assert part.config.optimizer == "adam"
        assert not part.config.use_precision_aware_optimizer
        assert torch.distributed.get_world_size(part.data_parallel_group) > 1
        masters = [param for group in part.optimizer.param_groups for param in group["params"]]
        assert masters and all(param.dtype == torch.float32 for param in masters)
        assert sum(param.numel() for param in masters) < sum(
            param.numel() for group in part.model_float16_groups for param in group
        )
        states = [part.optimizer.state[param] for param in masters]
        for name in ("exp_avg", "exp_avg_sq"):
            assert all(name in state for state in states)
            assert any(torch.count_nonzero(state[name]).item() for state in states)
        # The common state includes Adam step counts and group LR/WD; parameter
        # state includes each local FP32 master shard and both Adam moments.
        common = part.state_dict()
        assert all(group["step"] == args.iteration for group in common["optimizer"]["param_groups"])
        optimizer_states.append({"common": common, "masters": masters, "state": states})
    assert scheduler.num_steps == args.consumed_train_samples
    return _snapshot_state(
        {
            "model": dict(model[0].named_parameters()),
            "optimizer": optimizer_states,
            "scheduler": scheduler.state_dict(),
            "iteration": args.iteration,
            "consumed_train_samples": args.consumed_train_samples,
        }
    )


def _train_step(case, model, optimizer, scheduler):
    model[0].train()
    model[0].zero_grad_buffer()
    optimizer.zero_grad()
    model[0].set_is_first_microbatch()
    batch = case.get_batch()
    loss = model[0](
        input_ids=batch[0],
        labels=batch[1],
        position_ids=batch[2],
        attention_mask=batch[3],
        loss_mask=batch[4],
    ).mean()
    assert torch.isfinite(loss)
    loss.backward()
    model[0].finish_grad_sync()
    update_successful, _, _ = optimizer.step()
    assert update_successful
    args = get_args()
    scheduler.step(increment=args.global_batch_size)
    args.iteration += 1
    args.consumed_train_samples += args.global_batch_size
    return loss.detach().float().cpu()


def _canonical_parameters(*, gated=True):
    """Independent, nonzero values; no production layout helper builds this oracle."""
    generator = torch.Generator().manual_seed(1729)
    fc1_rows = 512 if gated else 256
    return {
        "linear_fc1.weight": (0.08 * torch.randn(2, fc1_rows, 256, generator=generator)).bfloat16(),
        "linear_fc1.bias": torch.linspace(-0.4, 0.6, 2 * fc1_rows).reshape(2, fc1_rows).bfloat16(),
        "linear_fc2.weight": (0.05 * torch.randn(2, 256, 256, generator=generator)).bfloat16(),
        "linear_fc2.bias": torch.linspace(-0.2, 0.3, 2 * 256).reshape(2, 256).bfloat16(),
    }


def _read_parameter(experts, name):
    layer, parameter = name.split(".")
    linear = getattr(experts, layer)
    if getattr(linear, f"single_grouped_{parameter}"):
        value = getattr(linear, parameter)
        # GroupedTensor owns packed storage; Tensor views need not expose that storage.
        return (
            value.rowwise_data.detach().cpu().reshape(linear.num_gemms, -1, linear.in_features)
            if parameter == "weight"
            else value.rowwise_data.detach().cpu().reshape(linear.num_gemms, -1)
        )
    return torch.stack(
        [getattr(linear, f"{parameter}{idx}").detach().cpu() for idx in range(linear.num_gemms)]
    )


def _expected_runtime(canonical, interleave):
    expected = {}
    for name, tensor in canonical.items():
        if name.startswith("linear_fc1.") and interleave is not None:
            # Index channels independently from the production reshape/transpose implementation.
            half = tensor.shape[1] // 2
            rows = [
                row
                for start in range(0, half, interleave)
                for offset in (0, half)
                for row in range(start + offset, start + offset + interleave)
            ]
            tensor = tensor[:, rows]
        expected[name] = tensor
    return expected


def _assert_parameters(experts, expected):
    for name, tensor in expected.items():
        torch.testing.assert_close(_read_parameter(experts, name), tensor, rtol=0, atol=0)


def _assert_checkpoint_parameters(directory, iteration, canonical):
    # Inspect actual disk tensors; save/load must not cancel a layout bug.
    checkpoint_dir = checkpointing.get_checkpoint_name(directory, iteration, return_base_dir=True)
    disk = dist_checkpointing.load_plain_tensors(checkpoint_dir)
    for name, expected in canonical.items():
        matching = [value for key, value in disk.items() if key.endswith(f"experts.{name}")]
        assert len(matching) == 1, (name, list(disk))
        torch.testing.assert_close(matching[0].cpu().reshape_as(expected), expected, rtol=0, atol=0)


def _save_canonical_checkpoint(case, directory, canonical, *, gated=True):
    model, optimizer, scheduler, experts = _setup_model(
        case,
        directory,
        single_weight=False,
        single_bias=False,
        interleave=None,
        use_op_fuser=False,
        precision="bf16",
        load=False,
        gated=gated,
    )
    with torch.no_grad():
        for name, tensor in canonical.items():
            layer, parameter = name.split(".")
            for idx, value in enumerate(tensor):
                getattr(getattr(experts, layer), f"{parameter}{idx}").copy_(value)
    optimizer.reload_model_params()
    force_param_sync(model, optimizer=optimizer)
    _assert_parameters(experts, canonical)
    checkpointing.save_checkpoint(1, model, optimizer, scheduler, 0)
    torch.distributed.barrier()
    _assert_checkpoint_parameters(directory, 1, canonical)


def _forward(case, model, experts):
    expert_outputs = []

    def capture(_module, _inputs, output):
        expert_outputs.append(output[0].detach().float().clone())

    hook = experts.register_forward_hook(capture)
    try:
        model[0].eval()
        model[0].set_is_first_microbatch()
        batch = case.get_batch()
        with torch.no_grad():
            losses = model[0](
                input_ids=batch[0],
                labels=batch[1],
                position_ids=batch[2],
                attention_mask=batch[3],
                loss_mask=batch[4],
            ).float()
        assert len(expert_outputs) == 1
        assert torch.isfinite(losses).all()
        assert torch.isfinite(expert_outputs[0]).all()
        return expert_outputs[0], losses
    finally:
        hook.remove()


def _forward_with_fp32_intermediates(case, model, experts, monkeypatch):
    """Keep the unfused MXFP8 reference's intermediates at the fused kernel's precision.

    This is forward-only: parameters and MXFP8 quantizers are unchanged, while FC1,
    SwiGLU and FC2 keep FP32 intermediates until the final BF16 output. Casting an
    already rounded BF16 result to FP32 would not recover the lost precision.
    """
    from transformer_engine.pytorch.ops import GroupedLinear, ScaledSwiGLU

    original_linear = GroupedLinear._fuser_forward_grouped_tensor
    original_activation = ScaledSwiGLU._scaled_glu_forward
    calls = []

    def reference_ops():
        # TEGroupedMLP builds these lazily on the first forward.
        return tuple(experts._fused_ops[0].children()) if experts._fused_ops else ()

    def linear_fp32(op, **kwargs):
        ops = reference_ops()
        if not any(op is reference_op for reference_op in ops):
            return original_linear(op, **kwargs)
        assert not torch.is_grad_enabled() and not torch.is_autocast_enabled()
        assert kwargs["dtype"] == torch.bfloat16
        assert kwargs["with_quantized_compute"] and kwargs["out_buffer"] is None
        is_fc2 = op is ops[2]
        assert kwargs["input_"].dtype == (torch.float32 if is_fc2 else torch.bfloat16)
        # Override after TE checks grouped-path eligibility using the BF16 parameters.
        # FC2 must also keep FP32, or it would round the activation before quantizing it.
        kwargs["dtype"] = torch.float32
        output, saved = original_linear(op, **kwargs)
        assert output.dtype == torch.float32
        calls.append("fc2" if is_fc2 else "fc1")
        return (output.bfloat16() if is_fc2 else output), saved

    def activation_fp32(op, input_, scales):
        ops = reference_ops()
        if not any(op is reference_op for reference_op in ops):
            return original_activation(op, input_, scales)
        assert not torch.is_grad_enabled() and not torch.is_autocast_enabled()
        assert input_.dtype == torch.float32
        # The fused MXFP8 activation rounds router probabilities to BF16 too.
        output = original_activation(op, input_, scales.bfloat16().float())
        assert output.dtype == torch.float32
        calls.append("activation")
        return output

    with monkeypatch.context() as reference:
        reference.setattr(GroupedLinear, "_fuser_forward_grouped_tensor", linear_fp32)
        reference.setattr(ScaledSwiGLU, "_scaled_glu_forward", activation_fp32)
        output, losses = _forward(case, model, experts)
    assert calls == ["fc1", "activation", "fc2"]
    return output, losses


def _assert_execution_plan(experts, *, fused):
    """An op-fuser configuration flag alone does not prove the fused kernel ran."""
    (sequence,) = experts._fused_ops
    selected = [
        (type(op).__name__, tuple(indices))
        for group in sequence._module_groups
        for op, indices in group._forward_ops
    ]
    expected = (
        [("GroupedMLP_CuTeGEMMGLU", (0, 1, 2))]
        if fused
        else [("GroupedLinear", (0,)), ("ScaledSwiGLU", (1,)), ("GroupedLinear", (2,))]
    )
    assert selected == expected, (
        f"Expected {expected}, got {selected}. Fused MXFP8 requires a GB200 container "
        "with native TE/cuDNN grouped-MLP fusion support."
    )


@pytest.mark.parametrize("single_weight", [False, True], ids=["indexed-weight", "single-weight"])
@pytest.mark.parametrize("single_bias", [False, True], ids=["indexed-bias", "single-bias"])
@pytest.mark.parametrize("use_op_fuser", [False, True], ids=["module", "op-fuser"])
@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
def test_canonical_checkpoint_matches_interleaved_ddp(
    moe_case, tmp_path_dist_ckpt, monkeypatch, single_weight, single_bias, use_op_fuser, precision
):
    """Catch missing load swizzling, even if a same-layout save/load round-trip would pass.

    Both execution layouts load the same independently checked canonical checkpoint.
    Nonzero FC1/FC2 biases participate in all four weight/bias storage combinations.
    MXFP8 op-fuser cases require actual joint fusion and an FP32-intermediate reference.
    The negative control bypasses only the load conversion and must break expert output.
    Optimizer moments are deliberately not restored when changing the execution layout.
    """
    numerics._skip_if_unsupported(precision)
    if precision == "mxfp8" and use_op_fuser:
        _enable_native_glu_fusion(monkeypatch)
    canonical = _canonical_parameters()
    with TempNamedDir(tmp_path_dist_ckpt / "canonical_glu_checkpoint", sync=True) as directory:
        _save_canonical_checkpoint(moe_case, directory, canonical)

        model, optimizer, scheduler, experts = _setup_model(
            moe_case,
            directory,
            single_weight=single_weight,
            single_bias=single_bias,
            interleave=None,
            use_op_fuser=use_op_fuser,
            precision=precision,
            load=True,
        )
        _assert_parameters(experts, canonical)
        if precision == "mxfp8" and use_op_fuser:
            reference_output, reference_losses = _forward_with_fp32_intermediates(
                moe_case, model, experts, monkeypatch
            )
        else:
            reference_output, reference_losses = _forward(moe_case, model, experts)
        if use_op_fuser:
            _assert_execution_plan(experts, fused=False)
        del experts, model, optimizer, scheduler

        model, optimizer, scheduler, experts = _setup_model(
            moe_case,
            directory,
            single_weight=single_weight,
            single_bias=single_bias,
            interleave=32,
            use_op_fuser=use_op_fuser,
            precision=precision,
            load=True,
        )
        expected_runtime = _expected_runtime(canonical, 32)
        _assert_parameters(experts, expected_runtime)
        actual_output, actual_losses = _forward(moe_case, model, experts)
        if use_op_fuser:
            _assert_execution_plan(experts, fused=precision == "mxfp8")
        tolerance = 5e-3 if precision == "bf16" else 5e-2
        torch.testing.assert_close(actual_output, reference_output, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(actual_losses, reference_losses, rtol=tolerance, atol=tolerance)

        # Reproduce the pre-fix behavior using the same real checkpoint loader and model.
        with monkeypatch.context() as missing_conversion:
            missing_conversion.setattr(
                checkpointing, "prepare_glu_checkpoint_for_load", lambda state, args: state
            )
            checkpointing.load_checkpoint(model, optimizer, scheduler, strict=True)
        optimizer.quantize_and_sync_model_params_from_main_params()
        _assert_parameters(experts, canonical)
        with pytest.raises(AssertionError):
            _assert_parameters(experts, expected_runtime)
        wrong_output, _ = _forward(moe_case, model, experts)
        assert not torch.allclose(
            wrong_output, reference_output, rtol=tolerance, atol=tolerance
        ), "The negative control must detect the original missing GLU checkpoint conversion."


@pytest.mark.parametrize("single_weight", [False, True], ids=["indexed-weight", "single-weight"])
@pytest.mark.parametrize("single_bias", [False, True], ids=["indexed-bias", "single-bias"])
def test_non_glu_checkpoint_ignores_interleave_size(
    moe_case, tmp_path_dist_ckpt, single_weight, single_bias
):
    """An inactive GLU setting must not reorder ordinary GELU FC1 weights or biases.

    Real DDP models load the same independent checkpoint with interleave unset or 32.
    Check exact parameters and forward/loss parity, then inspect the saved disk tensors.
    FC1 has 256 rows: divisible by 2 * 32, so the missing GLU guard silently permutes it.
    """
    canonical = _canonical_parameters(gated=False)
    with TempNamedDir(tmp_path_dist_ckpt / "non_glu_checkpoint", sync=True) as directory:
        _save_canonical_checkpoint(moe_case, directory, canonical, gated=False)
        model, optimizer, scheduler, experts = _setup_model(
            moe_case,
            directory,
            single_weight=single_weight,
            single_bias=single_bias,
            interleave=None,
            use_op_fuser=False,
            precision="bf16",
            load=True,
            gated=False,
        )
        _assert_parameters(experts, canonical)
        assert experts.config.activation_func is torch.nn.functional.gelu
        reference_output, reference_losses = _forward(moe_case, model, experts)
        del experts, model, optimizer, scheduler

        model, optimizer, scheduler, experts = _setup_model(
            moe_case,
            directory,
            single_weight=single_weight,
            single_bias=single_bias,
            interleave=32,
            use_op_fuser=False,
            precision="bf16",
            load=True,
            gated=False,
        )
        _assert_parameters(experts, canonical)
        actual_output, actual_losses = _forward(moe_case, model, experts)
        torch.testing.assert_close(actual_output, reference_output, rtol=0, atol=0)
        torch.testing.assert_close(actual_losses, reference_losses, rtol=0, atol=0)

        checkpointing.save_checkpoint(2, model, optimizer, scheduler, 0)
        torch.distributed.barrier()
        _assert_checkpoint_parameters(directory, 2, canonical)
        _assert_parameters(experts, canonical)


@pytest.mark.parametrize("interleave", [None, 32], ids=["canonical", "interleaved"])
def test_bf16_full_optimizer_resume_matches_uninterrupted_training(
    moe_case, tmp_path_dist_ckpt, interleave, monkeypatch
):
    """Restore real DP shards and Adam state, then compare two further parameter updates.

    Saving after two updates exercises nonzero moments and an advanced LR schedule.
    Compare every model parameter, FP32 master, Adam moment/step, and scheduler state
    exactly, both immediately after restore and after each subsequent update. Dropout
    is disabled and each step uses the same deterministic batch, so RNG loading is
    deliberately unnecessary. Full optimizer loading must not rebuild model weights
    from masters before the immediate model comparison.
    """
    numerics._skip_if_unsupported("bf16")
    # Determinism tests can set this globally during collection. Keep exercising
    # GroupedTensor + bias, whose TE dbias kernel requires atomic adds.
    monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")
    setup_kwargs = dict(
        single_weight=False,
        single_bias=False,
        interleave=interleave,
        use_op_fuser=False,
        precision="bf16",
        restore_optimizer=True,
    )
    with TempNamedDir(tmp_path_dist_ckpt / "bf16_optimizer_resume", sync=True) as directory:
        model, optimizer, scheduler, experts = _setup_model(
            moe_case, directory, load=False, **setup_kwargs
        )
        reference_losses, reference_states = [], []
        for _ in range(4):
            reference_losses.append(_train_step(moe_case, model, optimizer, scheduler))
            reference_states.append(_snapshot_training_state(model, optimizer, scheduler))
        del experts, model, optimizer, scheduler

        model, optimizer, scheduler, experts = _setup_model(
            moe_case, directory, load=False, **setup_kwargs
        )
        for step in range(2):
            loss = _train_step(moe_case, model, optimizer, scheduler)
            _assert_state_equal(loss, reference_losses[step])
            _assert_state_equal(
                _snapshot_training_state(model, optimizer, scheduler), reference_states[step]
            )

        canonical = {name: _read_parameter(experts, name) for name in _canonical_parameters()}
        assert all(tensor.dtype == torch.bfloat16 for tensor in canonical.values())
        if interleave is not None:
            for name, tensor in canonical.items():
                if name.startswith("linear_fc1."):
                    # Independent oracle: collect alternating gate/up blocks, without
                    # using the production reshape/transpose conversion.
                    blocks = tensor.split(interleave, dim=1)
                    canonical[name] = torch.cat((*blocks[::2], *blocks[1::2]), dim=1)

        args = get_args()
        assert args.no_save_optim is False and args.no_load_optim is False
        assert (
            checkpointing._build_sharded_state_dict_metadata(args)["distrib_optim_sharding_type"]
            == "dp_reshardable"
        )
        force_param_sync(model, optimizer=optimizer)
        checkpointing.save_checkpoint(2, model, optimizer, scheduler, 0)
        torch.distributed.barrier()
        _assert_state_equal(
            _snapshot_training_state(model, optimizer, scheduler), reference_states[1]
        )
        _assert_checkpoint_parameters(directory, 2, canonical)
        del experts, model, optimizer, scheduler

        model, optimizer, scheduler, experts = _setup_model(
            moe_case, directory, load=True, **setup_kwargs
        )
        # No extra master -> model sync here: independently check both loaded copies.
        _assert_state_equal(
            _snapshot_training_state(model, optimizer, scheduler), reference_states[1]
        )
        for step in range(2, 4):
            loss = _train_step(moe_case, model, optimizer, scheduler)
            _assert_state_equal(loss, reference_losses[step])
            _assert_state_equal(
                _snapshot_training_state(model, optimizer, scheduler), reference_states[step]
            )
