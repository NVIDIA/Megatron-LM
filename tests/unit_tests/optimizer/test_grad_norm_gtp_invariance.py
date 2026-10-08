# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Gradient-norm invariance across the GTP weight-rematerialization axis.

``get_grad_norm_fp32`` sums each rank's local squared gradient over a process group that
spans the whole world once GTP is enabled, so every gradient element must be contributed
by exactly one rank. GTP-sharded weights are counted on every GTP rank because each rank
owns distinct rows; parameters replicated along the GTP axis are counted only on GTP rank
0. ``param_is_not_gtp_duplicate`` is what draws that line. When it draws it wrong the
reported norm scales with ``gtp_remat_size`` instead of staying fixed, which silently
changes how gradient clipping behaves while leaving every other logged quantity alone.

These tests pin the contract: for a fixed logical model carrying fixed logical gradients,
the reduced norm equals the single-process norm at every ``gtp_remat_size``. Data
parallelism is held at 1 by pairing each GTP size with a pipeline size, so the only thing
that varies between parameterizations is how the same parameters are spread over ranks.

Scope: this covers the reduction and deduplication contract in the optimizer, with
parameters tagged by hand. It does not verify that a real model tags its parameters
correctly; that needs a check against the model's own parameter count at runtime.
"""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.clip_grads import get_grad_norm_fp32
from megatron.core.optimizer.optimizer import MegatronOptimizer
from tests.unit_tests.test_utilities import Utils

# Row count of every logical weight. Must stay divisible by each GTP size under test.
ROWS_PER_WEIGHT = 8
COLUMNS_PER_WEIGHT = 4
GRADIENT_SEED = 20260904


class _ParametersOnlyOptimizer(MegatronOptimizer):
    """Concrete optimizer that only exposes a parameter list to the gradient-norm helpers.

    ``MegatronOptimizer`` is abstract, and every abstract method below is unrelated to the
    norm path; they are stubbed so the real filtering code in ``_filter_grads_for_norm``
    can be exercised without building a full distributed optimizer.
    """

    def __init__(self, parameters, grad_stats_parallel_group, tensor_parallel_group):
        super().__init__(torch.optim.SGD(parameters, lr=1.0), OptimizerConfig())
        self.grad_stats_parallel_group = grad_stats_parallel_group
        self.tp_group = tensor_parallel_group
        self.expert_tp_group = tensor_parallel_group

    def prepare_grads(self) -> bool:
        return False

    def step_with_ready_grads(self) -> bool:
        return True

    def zero_grad(self, set_to_none: bool = True):
        pass

    def get_loss_scale(self) -> torch.Tensor:
        return torch.ones(1, device='cuda')

    def reload_model_params(self, state_dict=None):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def step(self):
        return True, None, None

    def sharded_state_dict(self, model_sharded_state_dict, is_loading=False, metadata=None):
        return {}


def _logical_layer_grads(layer_index):
    """Full weight and bias gradients of one logical layer, identical on every rank."""
    generator = torch.Generator().manual_seed(GRADIENT_SEED + layer_index)
    weight_grad = torch.randn(
        ROWS_PER_WEIGHT, COLUMNS_PER_WEIGHT, generator=generator, dtype=torch.float32
    )
    bias_grad = torch.randn(COLUMNS_PER_WEIGHT, generator=generator, dtype=torch.float32)
    return weight_grad, bias_grad


def _single_process_norm(num_layers):
    """L2 norm of the whole logical model's gradients, computed without any sharding."""
    squared_sum = 0.0
    for layer_index in range(num_layers):
        weight_grad, bias_grad = _logical_layer_grads(layer_index)
        squared_sum += weight_grad.double().pow(2).sum().item()
        squared_sum += bias_grad.double().pow(2).sum().item()
    return squared_sum**0.5


def _build_local_parameters(num_layers, gtp_remat_size):
    """Materialize this rank's share of the logical model.

    The pipeline rank selects which layers live here, and the GTP rank selects which rows
    of each weight live here. Biases are replicated along the GTP axis, mirroring the
    layernorms and biases that GTP leaves unsharded in a real model.
    """
    pipeline_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()
    gtp_rank = parallel_state.get_gtp_weight_remat_rank()

    layers_per_stage = num_layers // pipeline_size
    rows_per_shard = ROWS_PER_WEIGHT // gtp_remat_size

    parameters = []
    for offset in range(layers_per_stage):
        layer_index = pipeline_rank * layers_per_stage + offset
        weight_grad, bias_grad = _logical_layer_grads(layer_index)

        weight_shard = weight_grad[gtp_rank * rows_per_shard : (gtp_rank + 1) * rows_per_shard]
        weight = torch.nn.Parameter(torch.zeros_like(weight_shard).cuda())
        weight.grad = weight_shard.detach().clone().cuda()
        if gtp_remat_size > 1:
            # Distinct rows per GTP rank, so every rank contributes this shard.
            weight.is_gtp_weight_remat = True
        parameters.append(weight)

        bias = torch.nn.Parameter(torch.zeros_like(bias_grad).cuda())
        bias.grad = bias_grad.detach().clone().cuda()
        parameters.append(bias)

    return parameters


def _setup_optimizer(gtp_remat_size):
    """Initialize a data-parallel-free grid at this GTP size and return the stub optimizer."""
    world_size = Utils.world_size
    if world_size % gtp_remat_size != 0:
        pytest.skip(f"world size {world_size} is not divisible by gtp_remat_size {gtp_remat_size}")

    pipeline_size = world_size // gtp_remat_size
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pipeline_size,
        gtp_remat_size=gtp_remat_size,
    )

    # One layer per rank keeps the layer count divisible by every pipeline size under test.
    num_layers = world_size
    parameters = _build_local_parameters(num_layers, gtp_remat_size)
    optimizer = _ParametersOnlyOptimizer(
        parameters,
        parallel_state.get_intra_distributed_optimizer_instance_group(),
        parallel_state.get_tensor_model_parallel_group(),
    )
    return optimizer, num_layers


@pytest.mark.parametrize("gtp_remat_size", [1, 2, 4, 8])
class TestGradNormGTPInvariance:
    """The reduced gradient norm must not depend on how many ways GTP splits the weights."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_grad_norm_matches_single_process_norm(self, gtp_remat_size):
        optimizer, num_layers = _setup_optimizer(gtp_remat_size)

        grads_for_norm = optimizer.get_grads_for_grad_norm()
        grad_norm = float(
            get_grad_norm_fp32(
                grads_for_norm, grad_stats_parallel_group=optimizer.get_grad_stats_parallel_group()
            )
        )

        assert grad_norm == pytest.approx(_single_process_norm(num_layers), rel=1e-5)

    def test_every_gradient_element_is_counted_once(self, gtp_remat_size):
        """Element-count form of the same contract, which localizes a failure.

        A norm that comes out high says only that something was counted twice; the counted
        element total says by how much, and comparing it against the logical parameter
        count is the check that can be run inside a real job.
        """
        optimizer, num_layers = _setup_optimizer(gtp_remat_size)

        grads_for_norm = optimizer.get_grads_for_grad_norm()
        counted_elements = torch.tensor(
            [sum(grad.numel() for grad in grads_for_norm)], dtype=torch.int64, device='cuda'
        )
        torch.distributed.all_reduce(
            counted_elements, group=optimizer.get_grad_stats_parallel_group()
        )

        elements_per_layer = ROWS_PER_WEIGHT * COLUMNS_PER_WEIGHT + COLUMNS_PER_WEIGHT
        assert counted_elements.item() == num_layers * elements_per_layer


class TestGradNormWithModuleLocalGTP:
    """MIMO-shaped case: the GTP axis lives in a module's own grid, not in MPU globals.

    ``examples/mimo/pretrain_mimo.py`` passes ``skip_model_parallel_init=True`` and
    ``examples/mimo/training/distributed.py`` asserts the MPU globals are unset, so a MIMO run
    never creates the global GTP group; each module's axis lives in its own HyperCommGrid and
    reaches the optimizer through ``pg_collection.gtp_remat``. Reading the MPU rank there
    yields 0 on every rank, which keeps each GTP-replicated parameter on every peer and
    inflates both the gradient norm and the zero count by the peer count.

    The optimizer therefore passes its own group to ``param_is_not_gtp_duplicate``, the same
    way it already passes ``tp_group`` to ``param_is_not_tensor_parallel_duplicate``.
    """

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_module_local_gtp_group_deduplicates_replicated_params(self):
        world_size = Utils.world_size
        if world_size % 2 != 0:
            pytest.skip(f"world size {world_size} must be even to form GTP pairs")

        # Mirror a MIMO run: MPU knows nothing about GTP, the module's own grid does.
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=1
        )

        module_gtp_size = 2
        num_stages = world_size // module_gtp_size
        rank = torch.distributed.get_rank()
        stage = rank // module_gtp_size
        gtp_rank = rank % module_gtp_size

        # The module-local GTP group a fixed param_is_not_gtp_duplicate would consult.
        module_gtp_group = None
        for stage_index in range(num_stages):
            peers = [
                stage_index * module_gtp_size + peer_index for peer_index in range(module_gtp_size)
            ]
            group = torch.distributed.new_group(ranks=peers)
            if rank in peers:
                module_gtp_group = group

        rows_per_shard = ROWS_PER_WEIGHT // module_gtp_size
        weight_grad, bias_grad = _logical_layer_grads(stage)

        weight_shard = weight_grad[gtp_rank * rows_per_shard : (gtp_rank + 1) * rows_per_shard]
        weight = torch.nn.Parameter(torch.zeros_like(weight_shard).cuda())
        weight.grad = weight_shard.detach().clone().cuda()
        weight.is_gtp_weight_remat = True

        # Replicated along the module's GTP axis: both peers hold this identical gradient.
        bias = torch.nn.Parameter(torch.zeros_like(bias_grad).cuda())
        bias.grad = bias_grad.detach().clone().cuda()

        optimizer = _ParametersOnlyOptimizer(
            [weight, bias],
            parallel_state.get_intra_distributed_optimizer_instance_group(),
            parallel_state.get_tensor_model_parallel_group(),
        )
        # What get_megatron_optimizer sets from the module's own collection.
        optimizer.gtp_group = module_gtp_group

        grads_for_norm = optimizer.get_grads_for_grad_norm()
        grad_norm = float(
            get_grad_norm_fp32(
                grads_for_norm, grad_stats_parallel_group=optimizer.get_grad_stats_parallel_group()
            )
        )

        expected_norm = _single_process_norm(num_stages)
        inflation = grad_norm / expected_norm
        assert grad_norm == pytest.approx(expected_norm, rel=1e-5), (
            f"gradient norm inflated by {inflation:.4f}x: the bias of every stage was counted "
            f"once per GTP peer ({module_gtp_size} peers)"
        )

    def test_mpu_fallback_still_serves_optimizers_without_a_group(self):
        """An optimizer that sets no GTP group keeps the MPU behavior it had before.

        Non-MIMO callers reach the same code path with ``gtp_group`` unset, so the fallback has
        to stay correct for them: MPU carries the axis, and at gtp_remat_size 1 every rank is
        rank 0 and every parameter is kept.
        """
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=1
        )

        weight_grad, _ = _logical_layer_grads(0)
        weight = torch.nn.Parameter(torch.zeros_like(weight_grad).cuda())
        weight.grad = weight_grad.detach().clone().cuda()

        optimizer = _ParametersOnlyOptimizer(
            [weight],
            parallel_state.get_intra_distributed_optimizer_instance_group(),
            parallel_state.get_tensor_model_parallel_group(),
        )

        assert len(optimizer.get_grads_for_grad_norm()) == 1
