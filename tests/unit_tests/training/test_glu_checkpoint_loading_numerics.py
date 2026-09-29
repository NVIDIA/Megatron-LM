# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Import one canonical checkpoint into real DDP MoE models with either GLU layout."""

import sys

import pytest
import torch

from megatron.core import dist_checkpointing
from megatron.core.distributed import DistributedDataParallel
from megatron.core.enums import ModelType
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.training import checkpointing
from megatron.training.global_vars import set_args
from megatron.training.training import force_param_sync, setup_model_and_optimizer
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe import test_moe_single_grouped_weight_numerics as numerics

pytestmark = numerics.pytestmark


@pytest.fixture
def moe_case(monkeypatch):
    # Reuse the existing DDP/DistOpt model and batch setup, without inheriting its tests.
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    case = numerics.TestMoESingleGroupedWeightNumerics()
    case.setup_method(None)
    try:
        Utils.initialize_distributed()
        yield case
    finally:
        case.teardown_method(None)


def _setup_model(
    case, directory, *, single_weight, single_bias, interleave, use_op_fuser, precision, load
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
    args.no_save_optim = True
    args.no_load_optim = True
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
    assert not experts.config.bias_activation_fusion
    if load:
        # setup_model_and_optimizer calls the real loader once, including master initialization.
        assert args.iteration == 1
        # Rebuild model storage from the loaded FP32 masters before checking exact rows.
        optimizer.quantize_and_sync_model_params_from_main_params()
    return model, optimizer, scheduler, experts


def _canonical_parameters():
    """Independent, nonzero values; no production layout helper builds this oracle."""
    generator = torch.Generator().manual_seed(1729)
    return {
        "linear_fc1.weight": (0.08 * torch.randn(2, 512, 256, generator=generator)).bfloat16(),
        "linear_fc1.bias": torch.linspace(-0.4, 0.6, 2 * 512).reshape(2, 512).bfloat16(),
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
    The negative control bypasses only the load conversion and must break expert output.
    Optimizer moments are deliberately not restored when changing the execution layout.
    """
    numerics._skip_if_unsupported(precision)
    canonical = _canonical_parameters()
    with TempNamedDir(tmp_path_dist_ckpt / "canonical_glu_checkpoint", sync=True) as directory:
        model, optimizer, scheduler, experts = _setup_model(
            moe_case,
            directory,
            single_weight=False,
            single_bias=False,
            interleave=None,
            use_op_fuser=False,
            precision="bf16",
            load=False,
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

        # Inspect actual disk tensors before testing the loader. Save/load cannot cancel a bug.
        checkpoint_dir = checkpointing.get_checkpoint_name(directory, 1, return_base_dir=True)
        disk = dist_checkpointing.load_plain_tensors(checkpoint_dir)
        for name, expected in canonical.items():
            matching = [value for key, value in disk.items() if key.endswith(f"experts.{name}")]
            assert len(matching) == 1, (name, list(disk))
            torch.testing.assert_close(
                matching[0].cpu().reshape_as(expected), expected, rtol=0, atol=0
            )
        del disk, experts, model, optimizer, scheduler

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
        reference_output, reference_losses = _forward(moe_case, model, experts)
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
        tolerance = 5e-3 if precision == "bf16" else 5e-2
        torch.testing.assert_close(actual_output, reference_output, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(actual_losses, reference_losses, rtol=tolerance, atol=tolerance)

        # Reproduce the pre-fix behavior using the same real checkpoint loader and model.
        with monkeypatch.context() as missing_conversion:
            missing_conversion.setattr(
                checkpointing,
                "prepare_glu_checkpoint_for_load",
                lambda state, args, **kwargs: state,
            )
            # Indexed routed tensors now convert inside the Core factory. Disable
            # that conversion as well to retain the original negative control.
            missing_conversion.setattr(experts.config, "moe_mlp_glu_interleave_size", None)
            checkpointing.load_checkpoint(model, optimizer, scheduler, strict=True)
        optimizer.quantize_and_sync_model_params_from_main_params()
        _assert_parameters(experts, canonical)
        with pytest.raises(AssertionError):
            _assert_parameters(experts, expected_runtime)
        wrong_output, _ = _forward(moe_case, model, experts)
        assert not torch.allclose(
            wrong_output, reference_output, rtol=tolerance, atol=tolerance
        ), "The negative control must detect the original missing GLU checkpoint conversion."
