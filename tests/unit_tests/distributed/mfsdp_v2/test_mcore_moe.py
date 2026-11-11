# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MoE training and gradient clipping tests for the MFSDP v2 adapter."""

import logging
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
from megatron.core.models.hybrid.hybrid_layer_specs import (
    hybrid_stack_spec,
    wide_residual_hybrid_stack_spec,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.fully_sharded_optimizer import FullyShardedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.moe.experts import SequentialMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig, WideResidualConfig
from tests.unit_tests.distributed.mfsdp_v2.mcore_training_utils import (
    run_forward_backward_on_microbatches,
)
from tests.unit_tests.distributed.mfsdp_v2.mcore_utils import (
    destroy_model_parallel,
    require_canonical_grouped_glu,
)
from tests.unit_tests.test_utilities import Utils

logger = logging.getLogger(__name__)


@pytest.fixture
def process_group_collections(distributed_setup):
    """Yield EP=2 process groups and singleton reference groups, then clean them up."""
    world_size = distributed_setup.world_size
    if world_size < 2 or world_size % 2:
        pytest.skip("MFSDP v2 EP adapter test requires an even world size of at least two.")
    Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=2)
    process_group_collection = ProcessGroupCollection.use_mpu_process_groups()
    assert process_group_collection.ep.size() == 2
    assert process_group_collection.expt_dp.size() == world_size // 2
    reference_group = torch.distributed.new_group(
        [torch.distributed.get_rank()], use_local_synchronization=True
    )
    reference_process_group_collection = ProcessGroupCollection(
        tp=reference_group,
        expt_tp=reference_group,
        cp=reference_group,
        pp=reference_group,
        tp_cp=reference_group,
        tp_dp_cp=reference_group,
        ep=reference_group,
        tp_ep=reference_group,
        expt_dp=reference_group,
        dp=reference_group,
        dp_cp=reference_group,
        embd=None,
        pos_embd=None,
    )
    model_parallel_cuda_manual_seed(1234)
    yield process_group_collection, reference_process_group_collection
    destroy_model_parallel()


@pytest.mark.parametrize(
    "grouped_gemm", [True, False, "canonical"], ids=["grouped", "sequential", "canonical"]
)
def test_standard_moe_training_and_clipping(
    process_group_collections, distributed_setup, grouped_gemm, monkeypatch
):
    """Compare ordinary MoE training and clipping against an EP=1 reference."""
    process_group_collection, reference_process_group_collection = process_group_collections
    canonical = grouped_gemm == "canonical"
    if canonical:
        require_canonical_grouped_glu("bf16")
        if distributed_setup.world_size < 4:
            pytest.skip("Canonical EP=2 experts require EDP>=2 to exercise parameter sharding.")
        monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
        grouped_gemm = True
    config = TransformerConfig(
        num_layers=2,
        hidden_size=256 if canonical else 64,
        num_attention_heads=4,
        num_moe_experts=4,
        expert_model_parallel_size=2,
        moe_layer_freq=[0, 1],
        moe_token_dispatcher_type="alltoall",
        moe_router_topk=2,
        moe_grouped_gemm=grouped_gemm,
        moe_ffn_hidden_size=256 if canonical else 128,
        moe_shared_expert_intermediate_size=128,
        moe_use_grouped_tensor=canonical,
        use_transformer_engine_op_fuser=canonical,
        moe_mlp_glu_interleave_size=None,
        gated_linear_unit=canonical,
        activation_func=torch.nn.functional.silu if canonical else torch.nn.functional.gelu,
        add_bias_linear=False,
        use_cpu_initialization=True,
        bf16=canonical,
        params_dtype=torch.bfloat16 if canonical else torch.float32,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        gradient_accumulation_fusion=False,
        attention_backend=AttnBackend.unfused,
    )
    stack_spec = deepcopy(hybrid_stack_spec)
    stack_spec.submodules.moe_layer.submodules.mlp = get_moe_module_spec(
        use_te=True, num_experts=config.num_moe_experts, moe_grouped_gemm=grouped_gemm
    )
    torch.manual_seed(123)
    reference_model = HybridModel(
        config=replace(config, expert_model_parallel_size=1),
        hybrid_stack_spec=stack_spec,
        vocab_size=128,
        max_sequence_length=8,
        hybrid_layer_pattern="*E",
        pg_collection=reference_process_group_collection,
    ).cuda()
    model = HybridModel(
        config=config,
        hybrid_stack_spec=stack_spec,
        vocab_size=128,
        max_sequence_length=8,
        hybrid_layer_pattern="*E",
        pg_collection=process_group_collection,
    ).cuda()
    _sync_model_parameters(reference_model, model)
    reference_model.ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
    model = FullyShardedDataParallel(
        config=model.config,
        ddp_config=DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            megatron_fsdp_version=2,
            use_distributed_optimizer=False,
            data_parallel_sharding_strategy="optim_grads_params",
            fsdp_all_gather_in_start_param_sync=False,
        ),
        module=model,
        pg_collection=process_group_collection,
    )
    assert isinstance(model.module, FsdpModule)

    layer = model.module.decoder.layers[1]
    assert isinstance(layer, FsdpModule)
    assert not isinstance(layer.mlp, FsdpModule)
    assert not isinstance(layer.mlp.router, FsdpModule)
    assert not isinstance(layer.mlp.shared_experts, FsdpModule)
    assert isinstance(layer.mlp.experts, FsdpModule)

    optimizer_config = OptimizerConfig(
        lr=1.0e-3,
        weight_decay=0.0,
        use_distributed_optimizer=False,
        clip_grad=1.0e-4,
        bf16=canonical,
        params_dtype=config.params_dtype,
    )
    reference_optimizer = get_megatron_optimizer(optimizer_config, [reference_model])
    optimizer = get_megatron_optimizer(optimizer_config, [model])
    assert isinstance(optimizer, FullyShardedOptimizer)
    optimizer.reload_model_params()

    world_size = distributed_setup.world_size
    local_batch_size = 2
    torch.manual_seed(4321)
    input_ids = torch.randint(0, 128, (world_size * local_batch_size, 8), device="cuda")
    position_ids = torch.arange(8, device="cuda").repeat(world_size * local_batch_size, 1)
    targets = torch.randn(world_size * local_batch_size, 8, 128, device="cuda")
    input_slice = slice(
        torch.distributed.get_rank() * local_batch_size,
        (torch.distributed.get_rank() + 1) * local_batch_size,
    )

    reference_losses = _train(
        reference_model, reference_optimizer, input_ids, position_ids, targets
    )
    losses = _train(
        model, optimizer, input_ids[input_slice], position_ids[input_slice], targets[input_slice]
    )
    torch.distributed.all_reduce(losses, op=torch.distributed.ReduceOp.AVG)
    if torch.distributed.get_rank() == 0:
        logger.info("MFSDP v2 EP loss curve: %s", losses.tolist())
        logger.info("MFSDP v2 EP reference loss curve: %s", reference_losses.tolist())
    assert torch.isfinite(losses).all()
    assert torch.isfinite(reference_losses).all()
    assert losses[-1] < losses[0]
    if canonical:
        import transformer_engine.pytorch as te

        for experts in (layer.mlp.experts, reference_model.decoder.layers[1].mlp.experts):
            fused = experts._fused_ops[0]._module_groups[0]._forward_ops[0][0]
            assert isinstance(fused, te.ops.fused.GroupedMLP_CuTeGEMMGLUBf16)
            assert fused._gate_then_up
        assert all(
            group.mesh.size() == distributed_setup.world_size // 2
            for group in layer.mlp.experts.parameter_groups
        )
        # BF16 gradient reductions use different accumulation orders in the EP=1
        # full-batch reference and the EP=2/EDP=2 training path.
        torch.testing.assert_close(losses, reference_losses, rtol=2e-3, atol=2e-4)
    else:
        torch.testing.assert_close(losses, reference_losses)


@pytest.mark.parametrize("overlap", [False, True], ids=["no-overlap", "overlap"])
def test_shortcut_moe_training_and_clipping(process_group_collections, distributed_setup, overlap):
    """Keep shortcut parameters together while matching reference training with wide residuals."""
    process_group_collection, reference_process_group_collection = process_group_collections
    config = TransformerConfig(
        num_layers=3,
        hidden_size=64,
        num_attention_heads=4,
        num_moe_experts=4,
        expert_model_parallel_size=2,
        moe_layer_freq=[0, 1, 0],
        moe_token_dispatcher_type="alltoall",
        moe_router_topk=2,
        moe_grouped_gemm=True,
        moe_ffn_hidden_size=128,
        moe_shortcut_connection=True,
        moe_shortcut_parallel=overlap,
        moe_shortcut_post_norm=True,
        moe_shared_expert_intermediate_size=128,
        add_bias_linear=False,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        gradient_accumulation_fusion=False,
        attention_backend=AttnBackend.unfused,
        wide_residual=WideResidualConfig(num_streams=3, learned_retention=True),
        recompute_granularity="selective",
        recompute_modules=["moe_act"],
    )
    torch.manual_seed(123)
    reference_model = HybridModel(
        config=replace(config, expert_model_parallel_size=1),
        hybrid_stack_spec=wide_residual_hybrid_stack_spec,
        vocab_size=128,
        max_sequence_length=8,
        hybrid_layer_pattern="*E*",
        pg_collection=reference_process_group_collection,
    ).cuda()
    model = HybridModel(
        config=config,
        hybrid_stack_spec=wide_residual_hybrid_stack_spec,
        vocab_size=128,
        max_sequence_length=8,
        hybrid_layer_pattern="*E*",
        pg_collection=process_group_collection,
    ).cuda()
    _sync_model_parameters(reference_model, model)
    reference_model.ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
    model = FullyShardedDataParallel(
        config=model.config,
        ddp_config=DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            megatron_fsdp_version=2,
            use_distributed_optimizer=False,
            data_parallel_sharding_strategy="optim_grads_params",
            fsdp_all_gather_in_start_param_sync=False,
        ),
        module=model,
        pg_collection=process_group_collection,
    )
    assert isinstance(model.module, FsdpModule)

    block = model.module.decoder.layers[0]
    assert isinstance(block, FsdpModule)
    assert not isinstance(block.compute_layer, FsdpModule)
    assert not isinstance(block.moe_layer, FsdpModule)
    assert not isinstance(block.moe_layer.mlp, FsdpModule)
    assert isinstance(block.moe_layer.mlp.experts, FsdpModule)
    assert isinstance(model.module.decoder.layers[-1], FsdpModule)

    optimizer_config = OptimizerConfig(
        lr=1.0e-3, weight_decay=0.0, use_distributed_optimizer=False, clip_grad=1.0e-4
    )
    reference_optimizer = get_megatron_optimizer(optimizer_config, [reference_model])
    optimizer = get_megatron_optimizer(optimizer_config, [model])
    assert isinstance(optimizer, FullyShardedOptimizer)
    optimizer.reload_model_params()

    world_size = distributed_setup.world_size
    local_batch_size = 2
    torch.manual_seed(4321)
    input_ids = torch.randint(0, 128, (world_size * local_batch_size, 8), device="cuda")
    position_ids = torch.arange(8, device="cuda").repeat(world_size * local_batch_size, 1)
    targets = torch.randn(world_size * local_batch_size, 8, 128, device="cuda")
    input_slice = slice(
        torch.distributed.get_rank() * local_batch_size,
        (torch.distributed.get_rank() + 1) * local_batch_size,
    )

    reference_losses = _train(
        reference_model, reference_optimizer, input_ids, position_ids, targets
    )
    losses = _train(
        model, optimizer, input_ids[input_slice], position_ids[input_slice], targets[input_slice]
    )
    torch.distributed.all_reduce(losses, op=torch.distributed.ReduceOp.AVG)
    if torch.distributed.get_rank() == 0:
        logger.info("MFSDP v2 EP loss curve: %s", losses.tolist())
        logger.info("MFSDP v2 EP reference loss curve: %s", reference_losses.tolist())
    assert torch.isfinite(losses).all()
    assert torch.isfinite(reference_losses).all()
    assert losses[-1] < losses[0]
    torch.testing.assert_close(losses, reference_losses)


def _sync_model_parameters(reference_model, model):
    """Copy reference parameters, mapping global expert indices to local experts."""
    # CPU initialization gives each rank the same reference weights. Remap the
    # EP=1 reference's global expert indices to the EP=2 model's local indices.
    model.load_state_dict(reference_model.state_dict(), strict=False)
    for name, moe in model.named_modules():
        if not isinstance(moe, MoELayer):
            continue
        reference_moe = reference_model.get_submodule(name)
        if isinstance(moe.experts, SequentialMLP):
            for local, global_ in enumerate(moe.local_expert_indices):
                moe.experts.local_experts[local].load_state_dict(
                    reference_moe.experts.local_experts[global_].state_dict()
                )
            continue
        for fc in ("linear_fc1", "linear_fc2"):
            model_fc = getattr(moe.experts, fc)
            reference_fc = getattr(reference_moe.experts, fc)
            for local, global_ in enumerate(moe.local_expert_indices):
                for parameter_name in ("weight", "bias"):
                    model_parameter = getattr(model_fc, f"{parameter_name}{local}", None)
                    reference_parameter = getattr(reference_fc, f"{parameter_name}{global_}", None)
                    if model_parameter is not None:
                        model_parameter.data.copy_(reference_parameter.data)


def _train(model, optimizer, input_ids, position_ids, targets):
    """Train one model for five steps and return its stacked losses."""

    def forward():
        output = model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
        return torch.nn.functional.mse_loss(output.float(), targets)

    losses = []
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        (loss,) = run_forward_backward_on_microbatches(model, forward)
        success, pre_clip_norm, _ = optimizer.step()
        assert success
        assert (
            pre_clip_norm > optimizer.config.clip_grad
        ), "Gradients must exceed the clipping threshold to exercise clipping."
        losses.append(loss)
    return torch.stack(losses)


@pytest.fixture
def canonical_expert_process_groups(distributed_setup):
    """Use DP-only expert sharding so both two- and four-rank runs exercise ZeRO-3."""
    if distributed_setup.world_size < 2:
        pytest.skip("Canonical grouped experts require at least two FSDP ranks.")
    Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    yield ProcessGroupCollection.use_mpu_process_groups()
    destroy_model_parallel()


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize(
    ("precision", "native_parameters"),
    [("bf16", False), ("mxfp8", False), ("nvfp4", False), ("mxfp8", True)],
    ids=["bf16", "mxfp8", "nvfp4", "mxfp8_native_gather"],
)
def test_canonical_grouped_experts_reshard_and_update(
    canonical_expert_process_groups, distributed_setup, monkeypatch, precision, native_parameters
):
    """Canonical cuDNN experts survive real gather/backward/Adam/reshard cycles.

    BF16 resident weights and native MXFP8 gathers both keep the logical checkpoint and
    optimizer layout independent of compute precision. The unsharded reference uses the
    same fused arithmetic, isolating
    the distributed parameter/gradient lifecycle from quantization error. Rank-local batches
    differ, and an empty expert becomes active on the next step.
    """
    import functools

    cudnn = require_canonical_grouped_glu(precision)
    import transformer_engine.pytorch as te
    from torch.distributed.tensor import Replicate
    from transformer_engine.common.recipe import Format, MXFP8BlockScaling, NVFP4BlockScaling
    from transformer_engine.pytorch.optimizers import FusedAdam

    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TERowParallelGroupedLinear,
    )
    from megatron.core.fp8_utils import get_fp8_context
    from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, TEGroupedMLP

    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    fused_type = (
        te.ops.fused.GroupedMLP_CuTeGEMMGLUBf16
        if precision == "bf16"
        else te.ops.fused.GroupedMLP_CuTeGEMMGLU
    )
    calls = []
    for direction, name in (
        ("forward", "grouped_gemm_glu_wrapper_sm100"),
        ("backward", "grouped_gemm_dglu_wrapper_sm100"),
    ):
        original = getattr(cudnn, name)

        def trace(*args, _original=original, _direction=direction, **kwargs):
            calls.append((_direction, kwargs.get("glu_layout"), kwargs.get("b_tensor") is None))
            return _original(*args, **kwargs)

        monkeypatch.setattr(cudnn, name, functools.wraps(original)(trace))
    fused_type.grouped_gemm_activation_kernel.cache_clear()
    fused_type.grouped_gemm_dactivation_kernel.cache_clear()

    recipe = {
        "bf16": None,
        "mxfp8": MXFP8BlockScaling(fp8_format=Format.E4M3),
        "nvfp4": NVFP4BlockScaling(disable_stochastic_rounding=True),
    }[precision]
    config = TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=4,
        num_moe_experts=3,
        moe_ffn_hidden_size=256,
        moe_grouped_gemm=True,
        moe_use_grouped_tensor=True,
        moe_single_grouped_weight=False,
        use_transformer_engine_op_fuser=True,
        moe_mlp_glu_interleave_size=None,
        activation_func=torch.nn.functional.silu,
        gated_linear_unit=True,
        add_bias_linear=False,
        bias_activation_fusion=False,
        gradient_accumulation_fusion=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        fp8="e4m3" if precision == "mxfp8" else None,
        fp8_recipe="mxfp8",
        fp8_param=native_parameters,
        fp4="e2m1" if precision == "nvfp4" else None,
        fp4_recipe="nvfp4",
    )
    pg = canonical_expert_process_groups
    submodules = GroupedMLPSubmodules(
        linear_fc1=TEColumnParallelGroupedLinear, linear_fc2=TERowParallelGroupedLinear
    )
    torch.manual_seed(125)
    with get_fp8_context(config, is_init=True):
        reference = TEGroupedMLP(3, config, submodules, pg).cuda()
        module = TEGroupedMLP(3, config, submodules, pg).cuda()
    reference_weights = dict(reference.named_parameters())
    if native_parameters:
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

        assert all(isinstance(parameter, MXFP8Tensor) for parameter in module.parameters())
        # FSDP initializes its FP32 masters from preserved values, not the quantized bytes.
        with torch.no_grad():
            for name, parameter in module.named_parameters():
                parameter.get_high_precision_init_val().copy_(
                    reference_weights[name].get_high_precision_init_val()
                )
    else:
        module.load_state_dict(reference.state_dict())
    masters = {
        name: torch.nn.Parameter(
            (parameter.get_high_precision_init_val() if native_parameters else parameter)
            .detach()
            .to(device=parameter.device, dtype=torch.float32)
            .clone()
        )
        for name, parameter in reference_weights.items()
    }
    reference_optimizer = FusedAdam(list(masters.values()), lr=1.0e-3, weight_decay=0.0)
    model = FullyShardedDataParallel(
        config=config,
        ddp_config=DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            megatron_fsdp_version=2,
            fp8_param_gather=native_parameters,
            data_parallel_sharding_strategy="optim_grads_params",
            use_distributed_optimizer=False,
            megatron_fsdp_main_params_dtype=torch.float32,
            megatron_fsdp_main_grads_dtype=torch.float32,
            megatron_fsdp_grad_comm_dtype=torch.float32,
            fsdp_all_gather_in_start_param_sync=False,
        ),
        module=module,
        pg_collection=pg,
    )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            lr=1.0e-3,
            weight_decay=0.0,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=False,
            clip_grad=0.0,
        ),
        [model],
        use_gloo_process_groups=False,
    )
    assert isinstance(optimizer, FullyShardedOptimizer)
    assert isinstance(model.module, FsdpModule)
    assert all(
        group.mesh.size() == distributed_setup.world_size for group in module.parameter_groups
    )
    gathered_phases = []
    if native_parameters:

        def check_forward_planes(current, _inputs):
            for group in current.parameter_groups:
                buffer = group._unsharded_model_weight
                assert buffer.is_rowwise_allocated
                # REGISTRATION is the adapter's default execution order: even the
                # first forward may prefetch this single unit's own backward.
            gathered_phases.append("forward")

        def check_backward_planes(current, _grad_outputs):
            for group in current.parameter_groups:
                buffer = group._unsharded_model_weight
                assert buffer.is_columnwise_allocated
            gathered_phases.append("backward")

        module.register_forward_pre_hook(check_forward_planes)
        module.register_full_backward_pre_hook(check_backward_planes)
    for step, sizes in enumerate(((256, 0, 512), (128, 256, 384))):
        optimizer.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        model.zero_grad_buffer()
        torch.manual_seed(780 + step + 10 * distributed_setup.rank)
        source = torch.randn(sum(sizes), 256, device="cuda", dtype=torch.bfloat16)
        probabilities = torch.rand(sum(sizes), device="cuda", dtype=torch.float32)
        upstream = torch.randn_like(source) / 128
        counts = torch.tensor(sizes, device="cuda", dtype=torch.int64)
        results = []
        for current in (reference, model):
            x = source.clone().requires_grad_()
            probs = probabilities.clone().requires_grad_()
            with te.autocast(enabled=recipe is not None, recipe=recipe):
                output, bias = current(x, counts, probs)
            assert bias is None
            output.backward(upstream)
            results.append((output.detach(), x.grad, probs.grad))
            ops = reference._fused_ops[0] if current is reference else module._fused_ops[0]
            assert isinstance(ops._module_groups[0]._forward_ops[0][0], fused_type)
        model.finish_grad_sync()
        for index, (expected, actual) in enumerate(zip(*results)):
            # dprob uses a floating-point reduction; GEMM outputs and dgrad are exact.
            tolerance = dict(rtol=2e-5, atol=2e-7) if index == 2 else dict(rtol=0, atol=0)
            torch.testing.assert_close(actual, expected, **tolerance)
        for group in module.parameter_groups:
            assert not group.is_rowwise_unsharded
            if native_parameters:
                assert not group._unsharded_model_weight.is_columnwise_allocated
            assert group.main_grad is not None
            full_grad = group.main_grad.redistribute((Replicate(),))
            for index, parameter in enumerate(group.fsdp_parameters):
                name = parameter.fqns[0]
                expected_grad = reference_weights[name].grad.float()
                torch.distributed.all_reduce(expected_grad, op=torch.distributed.ReduceOp.AVG)
                masters[name].grad = expected_grad
                actual_grad = parameter.sharded.grad
                assert (
                    actual_grad.to_local().data_ptr()
                    == group.main_grad.get_tensor_view(index).data_ptr()
                )
                torch.testing.assert_close(
                    full_grad.get_tensor_view(index), expected_grad, rtol=2e-5, atol=2e-7
                )
        success, _, _ = optimizer.step()
        assert success
        reference_optimizer.step()
        for group in module.parameter_groups:
            full_master = group.main_weight.redistribute((Replicate(),))
            for index, parameter in enumerate(group.fsdp_parameters):
                name = parameter.fqns[0]
                torch.testing.assert_close(
                    full_master.get_tensor_view(index), masters[name], rtol=1e-5, atol=2e-7
                )
                with torch.no_grad():
                    if native_parameters:
                        reference_weights[name]._get_quantizer().set_usage(
                            rowwise=True, columnwise=True
                        )
                        reference_weights[name].quantize_(masters[name])
                    else:
                        reference_weights[name].copy_(masters[name])
    if native_parameters:
        assert gathered_phases == ["forward", "backward"] * 2
    assert {direction for direction, _, _ in calls} == {"forward", "backward"}
    assert all(layout == "gate_then_up" and discrete for _, layout, discrete in calls)
    fused_type.grouped_gemm_activation_kernel.cache_clear()
    fused_type.grouped_gemm_dactivation_kernel.cache_clear()


@pytest.mark.parametrize(
    "unsupported", ["single_weight", "single_bias", "fp4_storage", "fp4_gather", "fp4_tensor"]
)
def test_canonical_grouped_experts_reject_unsupported_storage(
    canonical_expert_process_groups, unsupported
):
    """Storage modes without an MFSDP v2 contract fail before allocating shards."""
    config = TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=4,
        num_moe_experts=3,
        moe_ffn_hidden_size=256,
        moe_grouped_gemm=True,
        moe_use_grouped_tensor=True,
        moe_single_grouped_weight=unsupported == "single_weight",
        moe_single_grouped_bias=unsupported == "single_bias",
        add_bias_linear=unsupported == "single_bias",
        fp4="e2m1" if unsupported in ("fp4_storage", "fp4_gather") else None,
        fp4_param=unsupported == "fp4_storage",
        gradient_accumulation_fusion=False,
    )
    ddp_config = DistributedDataParallelConfig(
        use_megatron_fsdp=True,
        megatron_fsdp_version=2,
        fp4_param_gather=unsupported == "fp4_gather",
    )
    module = torch.nn.Linear(256, 256, device="cuda")
    if unsupported == "fp4_tensor":
        import transformer_engine.pytorch as te
        from transformer_engine.common import recipe

        if torch.cuda.get_device_capability()[0] < 10 or not hasattr(recipe, "NVFP4BlockScaling"):
            pytest.skip("Native NVFP4 parameter construction requires Blackwell and NVFP4 TE.")
        # A per-module/outer TE initialization override can create native FP4 while
        # config.fp4_param and ddp_config.fp4_param_gather both remain False.
        with te.quantized_model_init(recipe=recipe.NVFP4BlockScaling()):
            module = te.Linear(256, 256, bias=False, params_dtype=torch.bfloat16, device="cuda")
        assert not config.fp4_param and not ddp_config.fp4_param_gather
    match = "per-expert weights and biases" if unsupported.startswith("single") else "FP4 parameter"
    if unsupported == "fp4_tensor":
        match += " 'weight'"
    with pytest.raises(ValueError, match=match):
        FullyShardedDataParallel(
            config=config,
            ddp_config=ddp_config,
            module=module,
            pg_collection=canonical_expert_process_groups,
        )
