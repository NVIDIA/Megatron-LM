# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import copy

import pytest
import torch
from torch.testing import assert_close

import megatron.core.parallel_state as mpu
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import HAVE_TE_MXFP8TENSOR
from megatron.core.utils import is_torch_min_version
from tests.unit_tests.distributed.megatron_fsdp.utils import (
    make_gpt_mock_data_iterator,
    make_moe_args_model_and_optimizer,
    pretrain_forward_backward,
    set_manual_seed,
)
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(scope="class")
def ref_cache():
    """
    Shared read/write cache for an class.
    Keys: arbitrary strings, values: anything (tensors, dicts, etc.).
    """
    return {}


class TestMegatronFSDPE2E:

    @staticmethod
    def _training_loop(seed=42, **kwargs):
        """
        Run a small deterministic (optional) training loop using a mocked MoE/GPT model and optimizer.
        This helper initializes model-parallel state, creates a model and optimizer via
        make_moe_args_model_and_optimizer, constructs a mock GPT data iterator, and runs
        NUM_TRAINING_STEPS iterations of forward/backward/optimization. Losses from each
        training step are collected and returned.
        Args:
            seed (int, optional): RNG seed for reproducibility. Default: 42.
            **kwargs: Configuration overrides (all optional). Recognized keys:
                - vocab_size (int): Vocabulary size for the mock model. Default: 100.
                - seq_length (int): Sequence length used for the mock data. Default: 128.
                - micro_batch_size (int): Per-microbatch size. Default: 2.
                - global_batch_size (int): Global batch size across data-parallel ranks. Default: 32.
                - train_iters (int): Number of training iterations to run. Default: 20.
                - tensor_model_parallel_size (int): Tensor model parallel world size. Default: 1.
                - pipeline_model_parallel_size (int): Pipeline model parallel world size. Default: 1.
                - num_layers_per_virtual_pipeline_stage (int or None): Virtual pipeline configuration.
                - expert_model_parallel_size (int): Expert model parallel size for MoE. Default: 1.
                - expert_tensor_parallel_size (int): Expert tensor parallel size for MoE. Default: 1.
                - num_distributed_optimizer_instances (int): Number of distributed optimizer instances. Default: 1.
        Returns:
            list: A list of length train_iters containing the per-step language-model loss values
            (the value appended from output[-1] each iteration). Loss objects are returned as produced
            by the training utilities (typically tensors or scalars).
        Side effects:
            - Calls Utils.initialize_model_parallel(...) and Utils.destroy_model_parallel().
            - Sets global RNG state via set_manual_seed(seed).
            - Constructs models/optimizers via make_moe_args_model_and_optimizer and a data iterator
              via make_gpt_mock_data_iterator.
            - Runs optimizer.zero_grad(), pretrain_forward_backward(...), and optim.step() repeatedly.
            - Calculates the number of micro-batches per step as:
                global_batch_size // micro_batch_size // data_parallel_world_size.
              This requires that global_batch_size be divisible by micro_batch_size * data_parallel_world_size.
        Raises:
            ValueError: If batch-size arithmetic or other setup assumptions (e.g., divisibility) are violated.
        """
        # Configuration parameters with defaults
        VOCAB_SIZE = kwargs.get("vocab_size", 100)
        MAX_SEQ_LEN = kwargs.get("seq_length", 128)
        MICRO_BATCH_SIZE = kwargs.get("micro_batch_size", 2)
        GLOBAL_BATCH_SIZE = kwargs.get("global_batch_size", 32)
        NUM_TRAINING_STEPS = kwargs.get("train_iters", 20)
        TP = kwargs.get("tensor_model_parallel_size", 1)
        PP = kwargs.get("pipeline_model_parallel_size", 1)
        VPP = kwargs.get("num_layers_per_virtual_pipeline_stage", None)
        EP = kwargs.get("expert_model_parallel_size", 1)
        ETP = kwargs.get("expert_tensor_parallel_size", 1)
        OUTER_DP = kwargs.get("num_distributed_optimizer_instances", 1)

        # Initialize model parallel groups
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=TP,
            pipeline_model_parallel_size=PP,
            expert_model_parallel_size=EP,
            expert_tensor_parallel_size=ETP,
            num_distributed_optimizer_instances=OUTER_DP,
        )
        DP_GROUP = mpu.get_data_parallel_group()

        # Set manual seed for reproducibility
        set_manual_seed(seed)

        # Create model and optimizer
        model_chunks, optim = make_moe_args_model_and_optimizer(
            ut_filename="test_mcore_fully_sharded_data_parallel.py",
            micro_batch_size=MICRO_BATCH_SIZE,
            global_batch_size=GLOBAL_BATCH_SIZE,
            vocab_size=VOCAB_SIZE,
            padded_vocab_size=VOCAB_SIZE,
            seq_length=MAX_SEQ_LEN,
            sequence_parallel=TP > 1,
            tensor_model_parallel_size=TP,
            pipeline_model_parallel_size=PP,
            num_layers_per_virtual_pipeline_stage=VPP,
            train_iters=NUM_TRAINING_STEPS,
            **kwargs,
        )
        if kwargs.get("use_megatron_fsdp", False) and kwargs.get(
            "use_precision_aware_optimizer", False
        ):
            assert (
                not optim.optimizer.master_weights
            ), "Megatron-FSDP should not use FusedAdam master weights."

        # Prepare data iterator
        data_iterator = make_gpt_mock_data_iterator(
            dp_group=DP_GROUP,
            vocab_size=VOCAB_SIZE,
            sequence_length=MAX_SEQ_LEN,
            batch_size=MICRO_BATCH_SIZE,
            num_samples=GLOBAL_BATCH_SIZE * NUM_TRAINING_STEPS,
        )

        outputs = []

        # Training loop
        for _ in range(NUM_TRAINING_STEPS):
            optim.zero_grad()
            output = pretrain_forward_backward(
                model=model_chunks,
                data_iterator=data_iterator,
                sequence_length=MAX_SEQ_LEN,
                micro_batch_size=MICRO_BATCH_SIZE,
                num_micro_batches=GLOBAL_BATCH_SIZE // MICRO_BATCH_SIZE // DP_GROUP.size(),
            )
            optim.step()

            # Collect loss
            outputs.append(output[-1])

        Utils.destroy_model_parallel()

        return outputs

    @pytest.mark.skipif(
        not is_torch_min_version("2.4.0"), reason="Test needs to be updated for torch >= 2.4.0"
    )
    @pytest.mark.parametrize("nd_topology", [pytest.param({"EP": 2}, id="EP2")])
    @pytest.mark.parametrize(
        "spec_configs",
        [
            pytest.param(
                dict(
                    data_parallel_sharding_strategy="optim_grads_params",
                    fsdp_double_buffer=True,
                    recompute_granularity="full",
                    recompute_method="uniform",
                    recompute_num_layers=1,
                    use_fully_shard_api=True,
                ),
                id="optim_grads_params_double_buffer",
            )
        ],
    )
    def test_compatible_with_nd_parallel(self, ref_cache, nd_topology, spec_configs):
        if spec_configs.get("fp8_recipe") == "mxfp8" and (
            torch.cuda.get_device_capability()[0] < 10 or not HAVE_TE_MXFP8TENSOR
        ):
            pytest.skip("Requires PyTorch & CUDA device with TE MXFP8Tensor support")

        nd_topology_str = "_".join([f"{k}{v}" for k, v in nd_topology.items()])
        if nd_topology_str not in ref_cache:
            distopt_spec_configs = copy.deepcopy(spec_configs)
            distopt_spec_configs["fp8_param_gather"] = False
            ref_cache[nd_topology_str] = TestMegatronFSDPE2E._training_loop(
                use_distributed_optimizer=True, **distopt_spec_configs
            )

        outputs = TestMegatronFSDPE2E._training_loop(
            use_megatron_fsdp=True,
            init_model_with_meta_device=True,
            ckpt_format="fsdp_dtensor",
            gradient_accumulation_fusion=False,
            **spec_configs,
        )
        reference_outputs = ref_cache[nd_topology_str]

        if torch.distributed.get_rank() == 0:
            for step, (output, ref_output) in enumerate(zip(outputs, reference_outputs)):
                loss = output["lm loss"]
                ref_loss = ref_output["lm loss"]
                assert_close(
                    loss,
                    ref_loss,
                    atol=0,
                    rtol=0.05,
                    msg=(
                        f"Loss mismatch at step {step}, FSDP Loss = {loss.item()}, "
                        f"Reference Loss = {ref_loss.item()}"
                        f", Compare = {compare_losses(loss.item(), ref_loss.item())}"
                        f", outputs = {outputs}, reference_outputs = {reference_outputs}"
                    ),
                )


def compare_losses(loss_a: float, loss_b: float, reference: str = "b"):
    """
    Compare two loss values with absolute and relative differences.

    Parameters
    ----------
    loss_a : float
        First loss value (e.g., baseline model).
    loss_b : float
        Second loss value (e.g., new model).
    reference : {"a", "b"}, default "b"
        Which loss to treat as the reference when computing the
        relative difference. If "b", relative diff is vs loss_b;
        if "a", vs loss_a.

    Returns
    -------
    dict with keys:
        "abs_diff" : float
            |loss_a - loss_b|
        "rel_diff" : float
            |loss_a - loss_b| / reference_loss
        "better" : str
            "a" if loss_a < loss_b, "b" if loss_b < loss_a, "equal" otherwise.
    """
    abs_diff = abs(loss_a - loss_b)

    if reference == "a":
        ref = loss_a
    else:
        ref = loss_b

    if ref == 0:
        rel_diff = float("inf")  # or None, depending on your preference
    else:
        rel_diff = abs_diff / ref

    if loss_a < loss_b:
        better = "a"
    elif loss_b < loss_a:
        better = "b"
    else:
        better = "equal"

    return {"abs_diff": abs_diff, "rel_diff": rel_diff, "better": better}
