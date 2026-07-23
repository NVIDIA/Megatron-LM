# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Numerical correctness for destination CP in the non-colocated MIMO bridge."""

from functools import partial

import pytest
import torch
import torch.distributed as dist
from packaging import version

import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.optimizer import get_mimo_optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import MultiModuleProcessGroupCollection
from tests.unit_tests.models.mimo.test_mimo_1f1b_schedule import (
    create_all_embedding_groups,
    create_hypercomm_grid,
    destroy_all_grids,
    get_mimo_model,
    is_rank_in_grid,
)
from tests.unit_tests.models.mimo.test_mimo_colocated_correctness import (
    _BatchIterator,
    _copy_ref_params_to_dist,
    _generate_and_broadcast_global_batches,
    _set_deterministic_env,
    _slice_batch,
    _snapshot_first_layer_encoder_grads,
    _wire_training_hooks,
)
from tests.unit_tests.test_utilities import Utils


def _clone_batch_value(value):
    """Clone a nested batch so sequential reference and target runs cannot share mutations."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_batch_value(item) for key, item in value.items()}
    return value


def _build_noncolocated_role_batches(global_batches, encoder_grid, llm_grid):
    """Build encoder-DP slices or full LLM batches for this disjoint-grid rank."""
    if is_rank_in_grid(encoder_grid):
        encoder_dp = encoder_grid.get_pg("dp")
        return [
            _slice_batch(batch, encoder_dp.size(), encoder_dp.rank()) for batch in global_batches
        ]
    if is_rank_in_grid(llm_grid):
        return [_clone_batch_value(batch) for batch in global_batches]
    raise AssertionError(f"Rank {dist.get_rank()} belongs to neither MIMO grid")


def _forward_step(data_iterator, model):
    """Forward step using the per-token loss contract expected by the MIMO grad hooks."""

    def loss_func(loss_mask, output_tensor):
        def result(loss, num_tokens, reduced):
            return loss, num_tokens, {'loss_reduced': reduced}

        zero = torch.tensor(0.0, device='cuda', requires_grad=True)
        one = torch.tensor(1, device='cuda', dtype=torch.int)
        if output_tensor is None:
            return result(zero, one, 0.0)

        if isinstance(output_tensor, dict):
            output = output_tensor.get(
                MIMO_LANGUAGE_MODULE_KEY, next(iter(output_tensor.values()), None)
            )
        else:
            output = output_tensor

        if output is None:
            return result(zero, one, 0.0)

        loss = output.float().sum()
        num_tokens = loss_mask.sum().to(torch.int).clamp(min=1) if loss_mask is not None else one
        return result(loss, num_tokens, loss)

    batch = next(data_iterator) if data_iterator is not None else {'input_ids': None}
    output_tensor, loss_mask = model(**batch)
    return output_tensor, partial(loss_func, loss_mask)


def _run_1f1b_configuration(
    mimo_model,
    module_to_grid_map,
    topology,
    language_pg,
    vision_pg,
    encoder_grid,
    llm_grid,
    encoder_name,
    batches,
    optimizer,
    micro_batch_size,
    seq_length,
    num_microbatches,
):
    """Run one deterministic non-colocated 1F1B step and snapshot encoder gradients."""
    _wire_training_hooks(mimo_model, module_to_grid_map, language_pg, vision_pg)
    communicator = MultiModulePipelineCommunicator(
        module_to_grid_map,
        topology,
        mimo_model.config,
        dim_mapping={'s': 0, 'h': 2, 'b': 1},
        module_output_ndim={encoder_name: 2},
    )

    module_pgs = {}
    language_model_module_name = None
    if is_rank_in_grid(encoder_grid):
        module_pgs[encoder_name] = vision_pg
    if is_rank_in_grid(llm_grid):
        module_pgs[MIMO_LANGUAGE_MODULE_KEY] = language_pg
        language_model_module_name = MIMO_LANGUAGE_MODULE_KEY
    pg_collection = MultiModuleProcessGroupCollection(
        module_pgs=module_pgs, language_model_module_name=language_model_module_name
    )

    needs_data = (
        is_rank_in_grid(encoder_grid) and is_pp_first_stage(encoder_grid.get_pg("pp"))
    ) or (
        is_rank_in_grid(llm_grid)
        and (is_pp_first_stage(llm_grid.get_pg("pp")) or is_pp_last_stage(llm_grid.get_pg("pp")))
    )
    data_iterator = _BatchIterator(batches) if needs_data else None

    optimizer.zero_grad()
    losses = schedule.forward_backward_pipelining_without_interleaving(
        forward_step_func=_forward_step,
        data_iterator=data_iterator,
        model=[mimo_model],
        num_microbatches=num_microbatches,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        forward_only=False,
        p2p_communicator=communicator,
        pg_collection=pg_collection,
    )

    encoder_grads = (
        _snapshot_first_layer_encoder_grads(mimo_model, encoder_name)
        if is_rank_in_grid(encoder_grid)
        else {}
    )
    success, grad_norm, _ = optimizer.step()
    assert success, "Optimizer step failed"
    assert grad_norm is not None and grad_norm > 0, f"Expected positive grad norm, got {grad_norm}"
    assert torch.isfinite(
        torch.as_tensor(grad_norm)
    ).all(), f"Expected finite grad norm, got {grad_norm}"

    if is_rank_in_grid(llm_grid) and is_pp_last_stage(llm_grid.get_pg("pp")):
        assert losses, "Expected losses on last LLM stage"
        for loss_dict in losses:
            loss = torch.as_tensor(loss_dict['loss_reduced'])
            assert torch.isfinite(loss).all(), f"Expected finite loss, got {loss}"
    return encoder_grads


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse('2.3.0'),
    reason="Device mesh requires PyTorch 2.3+",
)
class TestNonColocatedCPCorrectness:
    """Compare destination CP=2 encoder gradients with a deterministic CP=1 baseline."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_distributed()
        cls.world_size = dist.get_world_size()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def teardown_method(self):
        destroy_all_grids()

    def test_cp2_matches_cp1_encoder_gradients(self):
        """CP2 reconstructs the same encoder gradient as a deterministic CP1 baseline."""
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")

        _set_deterministic_env()
        previous_deterministic_algorithms = torch.are_deterministic_algorithms_enabled()
        previous_cudnn_deterministic = torch.backends.cudnn.deterministic
        previous_cudnn_benchmark = torch.backends.cudnn.benchmark
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        encoder_name = "images"
        hidden_size = 128
        seq_length = 64
        vocab_size = 1000
        micro_batch_size = 4
        num_microbatches = 4
        ref_mimo = None
        target_mimo = None

        try:
            ref_encoder_grid = create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=4)
            ref_llm_grid = create_hypercomm_grid(offset=4, tp=2, cp=1, pp=2, dp=1)
            target_encoder_grid = create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=4)
            target_llm_grid = create_hypercomm_grid(offset=4, tp=1, cp=2, pp=2, dp=1)
            create_all_embedding_groups(
                [ref_encoder_grid, ref_llm_grid, target_encoder_grid, target_llm_grid]
            )

            ddp_config = DistributedDataParallelConfig(
                overlap_grad_reduce=True, bucket_size=10000, use_distributed_optimizer=False
            )
            torch.manual_seed(12345)
            (ref_mimo, ref_module_to_grid_map, ref_topology, ref_language_pg, ref_vision_pg) = (
                get_mimo_model(
                    encoder_name=encoder_name,
                    encoder_grid=ref_encoder_grid,
                    llm_grid=ref_llm_grid,
                    hidden_size=hidden_size,
                    num_layers=2,
                    vocab_size=vocab_size,
                    seq_len=seq_length,
                    ddp_config=ddp_config,
                    bf16=True,
                    bias=False,
                    dropout=False,
                    per_token_loss=True,
                )
            )

            torch.manual_seed(12345)
            (
                target_mimo,
                target_module_to_grid_map,
                target_topology,
                target_language_pg,
                target_vision_pg,
            ) = get_mimo_model(
                encoder_name=encoder_name,
                encoder_grid=target_encoder_grid,
                llm_grid=target_llm_grid,
                hidden_size=hidden_size,
                num_layers=2,
                vocab_size=vocab_size,
                seq_len=seq_length,
                ddp_config=ddp_config,
                bf16=True,
                bias=False,
                dropout=False,
                per_token_loss=True,
            )

            if is_rank_in_grid(ref_encoder_grid):
                _copy_ref_params_to_dist(
                    ref_mimo.modality_submodules[encoder_name].module,
                    target_mimo.modality_submodules[encoder_name].module,
                    ref_encoder_grid.get_pg("tp"),
                    target_encoder_grid.get_pg("tp"),
                )
            else:
                assert is_rank_in_grid(ref_llm_grid)
                _copy_ref_params_to_dist(
                    ref_mimo.language_model.module,
                    target_mimo.language_model.module,
                    ref_llm_grid.get_pg("tp"),
                    target_llm_grid.get_pg("tp"),
                )
            dist.barrier()

            optimizer_config = OptimizerConfig(
                optimizer='adam',
                lr=1e-4,
                weight_decay=0.01,
                clip_grad=1.0,
                bf16=True,
                use_distributed_optimizer=False,
            )
            ref_optimizer = get_mimo_optimizer(ref_mimo, optimizer_config)

            torch.manual_seed(99999)
            global_batches = _generate_and_broadcast_global_batches(
                global_mbs=micro_batch_size,
                seq_length=seq_length,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                encoder_name=encoder_name,
                num_batches=num_microbatches,
                modality_dtype=torch.bfloat16,
            )
            ref_batches = _build_noncolocated_role_batches(
                global_batches, ref_encoder_grid, ref_llm_grid
            )
            target_batches = _build_noncolocated_role_batches(
                global_batches, target_encoder_grid, target_llm_grid
            )

            ref_grads = _run_1f1b_configuration(
                ref_mimo,
                ref_module_to_grid_map,
                ref_topology,
                ref_language_pg,
                ref_vision_pg,
                ref_encoder_grid,
                ref_llm_grid,
                encoder_name,
                ref_batches,
                ref_optimizer,
                micro_batch_size,
                seq_length,
                num_microbatches,
            )
            dist.barrier()

            target_optimizer = get_mimo_optimizer(target_mimo, optimizer_config)
            target_grads = _run_1f1b_configuration(
                target_mimo,
                target_module_to_grid_map,
                target_topology,
                target_language_pg,
                target_vision_pg,
                target_encoder_grid,
                target_llm_grid,
                encoder_name,
                target_batches,
                target_optimizer,
                micro_batch_size,
                seq_length,
                num_microbatches,
            )

            local_error = None
            if is_rank_in_grid(ref_encoder_grid):
                if set(ref_grads) != set(target_grads):
                    local_error = (
                        f"First-layer gradient names differ: reference={set(ref_grads)}, "
                        f"target={set(target_grads)}"
                    )
                else:
                    for name in sorted(ref_grads):
                        try:
                            torch.testing.assert_close(
                                target_grads[name], ref_grads[name], rtol=1e-3, atol=1e-3
                            )
                        except AssertionError as error:
                            local_error = f"First-layer gradient {name!r} differs: {error}"
                            break

            gradients_match = torch.tensor(
                0 if local_error else 1, device='cuda', dtype=torch.int32
            )
            dist.all_reduce(gradients_match, op=dist.ReduceOp.MIN)
            assert gradients_match.item() == 1, (
                local_error or "A remote encoder rank reported a CP1-vs-CP2 gradient mismatch"
            )
        finally:
            if target_mimo is not None:
                target_mimo.destroy()
            if ref_mimo is not None:
                ref_mimo.destroy()
            torch.use_deterministic_algorithms(previous_deterministic_algorithms)
            torch.backends.cudnn.deterministic = previous_cudnn_deterministic
            torch.backends.cudnn.benchmark = previous_cudnn_benchmark
