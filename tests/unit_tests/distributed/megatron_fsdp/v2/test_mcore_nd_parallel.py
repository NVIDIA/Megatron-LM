# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
import copy

import pytest
import torch
from torch.testing import assert_close

import megatron.core.parallel_state as mpu
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import HAVE_TE_MXFP8TENSOR
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.mixed_precision import (
    HAVE_TE_NVFP4,
    HAVE_TE_NVFP4_RECIPE,
)
from megatron.core.utils import is_torch_min_version
from tests.unit_tests.distributed.megatron_fsdp.utils import (
    make_gpt_mock_data_iterator,
    make_moe_args_model_and_optimizer,
    pretrain_forward_backward,
    set_manual_seed,
)
from tests.unit_tests.test_utilities import Utils

STRICT_LOSS_ATOL = 5e-3
STRICT_PARAM_ATOL = 5e-3
STRICT_PARAM_RTOL = 1e-3


@pytest.fixture(scope="class")
def ref_cache():
    """
    Shared read/write cache for an class.
    Keys: arbitrary strings, values: anything (tensors, dicts, etc.).
    """
    return {}


class TestMegatronFSDPE2E:

    @staticmethod
    def _normalize_param_name(name):
        while name.startswith("module."):
            name = name[len("module.") :]
        return name

    @staticmethod
    def _materialize_param_tensor(param):
        from torch.distributed.tensor import DTensor

        from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
            uneven_dtensor_to_full_tensor,
        )
        from megatron.core.fp8_utils import dequantize_fp8_tensor, is_float8tensor

        tensor = param.detach()
        if isinstance(tensor, DTensor):
            tensor = uneven_dtensor_to_full_tensor(tensor)
        elif is_float8tensor(tensor):
            tensor = dequantize_fp8_tensor(tensor)
        return tensor.detach().float().cpu()

    @staticmethod
    def _capture_named_params(model_chunks):
        # All ranks must enter DTensor gather collectives, but only rank 0
        # keeps CPU copies for comparison.
        snapshots = {}
        for chunk_idx, model_chunk in enumerate(model_chunks):
            for name, param in model_chunk.named_parameters():
                tensor = TestMegatronFSDPE2E._materialize_param_tensor(param)
                if torch.distributed.get_rank() == 0:
                    key = f"{chunk_idx}.{TestMegatronFSDPE2E._normalize_param_name(name)}"
                    snapshots[key] = tensor
        return snapshots

    @staticmethod
    def _assert_replicated_weight_buffers_match(model_chunks):
        from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.fsdp_module import FSDPModule

        for model_chunk in model_chunks:
            for _, module in model_chunk.named_modules():
                if not isinstance(module, FSDPModule):
                    continue
                for param_group in module._fsdp_param_groups:
                    if (
                        param_group.model_weight_buffer is None
                        or param_group.model_weight_buffer.is_distributed
                    ):
                        continue
                    param_group.unshard(bwd_pass=False)
                    if param_group.transpose_weight_buffer is not None:
                        param_group.unshard(bwd_pass=True)

                    for buffer_name, buffer in (
                        ("model_weight_buffer", param_group.model_weight_buffer),
                        ("transpose_weight_buffer", param_group.transpose_weight_buffer),
                    ):
                        if buffer is None or buffer.is_distributed:
                            continue
                        gathered = [
                            torch.empty_like(buffer.data)
                            for _ in range(torch.distributed.get_world_size(param_group.dp_group))
                        ]
                        torch.distributed.all_gather(gathered, buffer.data, group=param_group.dp_group)
                        for group_rank, replica in enumerate(gathered):
                            assert torch.equal(buffer.data, replica), (
                                f"Replicated {buffer_name} mismatch for "
                                f"param_group={param_group.param_group_id}, "
                                f"group_rank={group_rank}"
                            )

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
        VOCAB_SIZE = kwargs.pop("vocab_size", 100)
        MAX_SEQ_LEN = kwargs.pop("seq_length", 128)
        MICRO_BATCH_SIZE = kwargs.pop("micro_batch_size", 2)
        GLOBAL_BATCH_SIZE = kwargs.pop("global_batch_size", 32)
        NUM_TRAINING_STEPS = kwargs.pop("train_iters", 20)
        TP = kwargs.pop("TP", 1)
        PP = kwargs.pop("PP", 1)
        VPP = kwargs.pop("VPP", None)
        EP = kwargs.pop("EP", 1)
        ETP = kwargs.pop("ETP", 1)
        OUTER_DP = kwargs.pop("OUTER_DP", 1)
        capture_param_snapshots = kwargs.pop("capture_param_snapshots", False)
        verify_replicated_weight_buffers = kwargs.pop(
            "verify_replicated_weight_buffers", False
        )
        return_dict = kwargs.pop("return_dict", capture_param_snapshots)

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
        param_snapshots = []

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
            if verify_replicated_weight_buffers:
                TestMegatronFSDPE2E._assert_replicated_weight_buffers_match(model_chunks)

            # Collect loss
            outputs.append(output[-1])
            if capture_param_snapshots:
                param_snapshots.append(
                    TestMegatronFSDPE2E._capture_named_params(model_chunks)
                )

        Utils.destroy_model_parallel()

        if return_dict:
            result = {"outputs": outputs}
            if capture_param_snapshots:
                result["param_snapshots"] = param_snapshots
            return result
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
                    recompute_granularity="full",
                    recompute_method="uniform",
                    recompute_num_layers=1,
                    overlap_param_gather=True,
                    overlap_grad_reduce=True,
                    use_megatron_fsdp_v2=True,
                    gradient_accumulation_fusion=True,
                    fsdp_trace_pool=True,
                ),
                id="optim_grads_params_double_buffer",
            ),
            pytest.param(
                dict(
                    bf16=True,
                    data_parallel_sharding_strategy="optim_grads_params",
                    fp8="e4m3",
                    fp8_param_gather=True,
                    fp8_recipe="mxfp8",
                    moe_grouped_gemm=True,
                    overlap_param_gather=True,
                    overlap_grad_reduce=True,
                    use_megatron_fsdp_v2=True,
                ),
                id="optim_grads_params_mxfp8_param_gather",
            ),
            pytest.param(
                dict(
                    bf16=True,
                    data_parallel_sharding_strategy="optim_grads_params",
                    fp4="e2m1",
                    fp4_recipe="nvfp4",
                    fp4_param_gather=True,
                    main_grads_dtype="fp32",
                    main_params_dtype="fp32",
                    overlap_param_gather=True,
                    overlap_grad_reduce=True,
                    use_megatron_fsdp_v2=True,
                ),
                id="optim_grads_params_nvfp4_param_gather",
            ),
        ],
    )
    def test_compatible_with_nd_parallel(self, ref_cache, nd_topology, spec_configs):
        if spec_configs.get("fp8_recipe") == "mxfp8" and (
            not torch.cuda.is_available()
            or torch.cuda.get_device_capability()[0] < 10
            or not HAVE_TE_MXFP8TENSOR
        ):
            pytest.skip("Requires PyTorch & CUDA device with TE MXFP8Tensor support")

        if spec_configs.get("fp4_param_gather"):
            if not torch.cuda.is_available():
                pytest.skip("CUDA is required for NVFP4")
            if not (HAVE_TE_NVFP4 and HAVE_TE_NVFP4_RECIPE):
                pytest.skip("NVFP4 requires Transformer Engine >= 2.7.0.dev0")
            try:
                from transformer_engine.pytorch.fp8 import check_nvfp4_support

                is_nvfp4_available, reason = check_nvfp4_support()
                if not is_nvfp4_available:
                    pytest.skip("NVFP4 not available: " + reason)
            except ImportError:
                pytest.skip("NVFP4 support check requires Transformer Engine >= 2.7.0.dev0")

        reference_kind = "distopt"
        ref_cache_key = (
            reference_kind,
            tuple(sorted(nd_topology.items())),
            tuple(sorted((key, repr(value)) for key, value in spec_configs.items())),
        )
        if ref_cache_key not in ref_cache:
            reference_spec_configs = copy.deepcopy(spec_configs)
            reference_spec_configs["use_megatron_fsdp_v2"] = False
            reference_spec_configs["gradient_accumulation_fusion"] = False
            reference_spec_configs["fp8_param_gather"] = False
            ref_cache[ref_cache_key] = TestMegatronFSDPE2E._training_loop(
                use_distributed_optimizer=True, **nd_topology, **reference_spec_configs
            )

        fsdp_spec_configs = copy.deepcopy(spec_configs)
        fsdp_spec_configs.setdefault("gradient_accumulation_fusion", False)
        outputs = TestMegatronFSDPE2E._training_loop(
            use_megatron_fsdp=True,
            init_model_with_meta_device=True,
            ckpt_format="fsdp_dtensor",
            **nd_topology,
            **fsdp_spec_configs,
        )
        reference_outputs = ref_cache[ref_cache_key]

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
                        f"Loss mismatch at step {step}, FSDP Loss = {loss.detach().item()}, "
                        f"Reference Loss = {ref_loss.detach().item()}"
                        f", Compare = {compare_losses(loss.detach().item(), ref_loss.detach().item())}"
                        f", outputs = {outputs}, reference_outputs = {reference_outputs}"
                    ),
                )

    @pytest.mark.skipif(
        not is_torch_min_version("2.4.0"), reason="Test needs to be updated for torch >= 2.4.0"
    )
    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(
                dict(
                    strategy="optim",
                    precision_configs=dict(bf16=True),
                    reference_kind="distopt",
                    capture_param_snapshots=True,
                ),
                id="bf16-optim",
            ),
            pytest.param(
                dict(
                    strategy="optim_grads",
                    precision_configs=dict(bf16=True),
                    reference_kind="distopt",
                    capture_param_snapshots=True,
                ),
                id="bf16-optim_grads",
            ),
            pytest.param(
                dict(
                    strategy="optim_grads_params",
                    precision_configs=dict(bf16=True),
                    reference_kind="distopt",
                    capture_param_snapshots=True,
                ),
                id="bf16-optim_grads_params",
            ),
            pytest.param(
                dict(
                    strategy="optim_grads_params",
                    precision_configs=dict(
                        bf16=True,
                        fp8="e4m3",
                        fp8_param_gather=True,
                        fp8_recipe="mxfp8",
                        main_grads_dtype="fp32",
                        main_params_dtype="fp32",
                        exp_avg_dtype="bf16",
                        exp_avg_sq_dtype="bf16",
                        moe_grouped_gemm=True,
                        use_precision_aware_optimizer=True,
                    ),
                    reference_kind="fsdp_v1",
                    capture_param_snapshots=False,
                ),
                id="mxfp8_param_gather-optim_grads_params",
            ),
        ],
    )
    def test_strict_iter_equivalence_zero_strategies(self, ref_cache, case):
        strategy = case["strategy"]
        precision_configs = case["precision_configs"]
        if precision_configs.get("fp8_recipe") == "mxfp8" and (
            not torch.cuda.is_available()
            or torch.cuda.get_device_capability()[0] < 10
            or not HAVE_TE_MXFP8TENSOR
        ):
            pytest.skip("Requires PyTorch & CUDA device with TE MXFP8Tensor support")
        if Utils.world_size < 2:
            pytest.skip("Requires at least 2 distributed ranks for ZeRO sharding")

        common_configs = dict(
            data_parallel_sharding_strategy=strategy,
            train_iters=3,
            seq_length=64,
            micro_batch_size=1,
            global_batch_size=8,
            # Keep strict iter-equivalence on the ordinary model-init path.  The
            # FSDP v1/v2 meta-init paths materialize nested FSDP units in a
            # different order, so they can legitimately start from different
            # random initial weights even with the same seed.
            init_model_with_meta_device=False,
            gradient_accumulation_fusion=False,
            overlap_param_gather=False,
            overlap_grad_reduce=False,
            verify_replicated_weight_buffers=strategy in ("optim", "optim_grads"),
            **precision_configs,
        )
        reference_kind = case["reference_kind"]
        capture_param_snapshots = case["capture_param_snapshots"]
        ref_cache_key = (
            "strict_iter_equivalence",
            reference_kind,
            strategy,
            capture_param_snapshots,
            tuple(sorted((key, repr(value)) for key, value in common_configs.items())),
        )

        if ref_cache_key not in ref_cache:
            reference_configs = copy.deepcopy(common_configs)
            if reference_kind == "fsdp_v1":
                reference_configs["use_megatron_fsdp_v2"] = False
                ref_cache[ref_cache_key] = TestMegatronFSDPE2E._training_loop(
                    use_megatron_fsdp=True,
                    ckpt_format="fsdp_dtensor",
                    capture_param_snapshots=capture_param_snapshots,
                    return_dict=True,
                    **reference_configs,
                )
            else:
                ref_cache[ref_cache_key] = TestMegatronFSDPE2E._training_loop(
                    use_distributed_optimizer=True,
                    capture_param_snapshots=capture_param_snapshots,
                    return_dict=True,
                    **reference_configs,
                )

        fsdp_configs = copy.deepcopy(common_configs)
        fsdp_configs["use_megatron_fsdp_v2"] = True
        actual = TestMegatronFSDPE2E._training_loop(
            use_megatron_fsdp=True,
            ckpt_format="fsdp_dtensor",
            capture_param_snapshots=capture_param_snapshots,
            return_dict=True,
            **fsdp_configs,
        )
        reference = ref_cache[ref_cache_key]

        if torch.distributed.get_rank() != 0:
            return

        assert len(actual["outputs"]) == len(reference["outputs"])
        for step, (output, ref_output) in enumerate(
            zip(actual["outputs"], reference["outputs"])
        ):
            loss = output["lm loss"]
            ref_loss = ref_output["lm loss"]
            assert_close(
                loss,
                ref_loss,
                atol=STRICT_LOSS_ATOL,
                rtol=0,
                msg=(
                    f"Loss mismatch at step {step}, strategy={strategy}, "
                    f"actual={loss.detach().item()}, reference={ref_loss.detach().item()}, "
                    f"compare={compare_losses(loss.detach().item(), ref_loss.detach().item())}"
                ),
            )

        if not capture_param_snapshots:
            return

        assert len(actual["param_snapshots"]) == len(reference["param_snapshots"])
        for step, (params, ref_params) in enumerate(
            zip(actual["param_snapshots"], reference["param_snapshots"])
        ):
            missing = sorted(set(ref_params) ^ set(params))
            assert not missing, (
                f"Parameter key mismatch at step {step}, strategy={strategy}: {missing[:20]}"
            )
            for name in sorted(ref_params):
                assert_close(
                    params[name],
                    ref_params[name],
                    atol=STRICT_PARAM_ATOL,
                    rtol=STRICT_PARAM_RTOL,
                    msg=(
                        f"Parameter mismatch at step {step}, strategy={strategy}, "
                        f"name={name}, actual_shape={tuple(params[name].shape)}, "
                        f"reference_shape={tuple(ref_params[name].shape)}"
                    ),
                )

    @pytest.mark.skipif(
        not is_torch_min_version("2.4.0"), reason="Test needs to be updated for torch >= 2.4.0"
    )
    @pytest.mark.parametrize(
        "strategy,precision_configs",
        [
            pytest.param(
                strategy,
                dict(
                    bf16=True,
                    fp8="e4m3",
                    fp8_param_gather=True,
                    fp8_recipe="mxfp8",
                    main_grads_dtype="fp32",
                    main_params_dtype="fp32",
                    exp_avg_dtype="bf16",
                    exp_avg_sq_dtype="bf16",
                    moe_grouped_gemm=True,
                    use_precision_aware_optimizer=True,
                ),
                id=f"mxfp8_param_gather-{strategy}",
            )
            for strategy in ("optim", "optim_grads")
        ],
    )
    def test_zero_strategy_non_equivalent_precision_paths_run(
        self, strategy, precision_configs
    ):
        """Exercise valid ZeRO paths that intentionally lack a strict reference.

        MXFP8 ZeRO-1/2 refreshes replicated quantized compute buffers after
        sharded optimizer updates; v1 and v2 do not provide a strict multi-step
        golden comparison for that replicated-weight quantization path.
        """
        if precision_configs.get("fp8_recipe") == "mxfp8" and (
            not torch.cuda.is_available()
            or torch.cuda.get_device_capability()[0] < 10
            or not HAVE_TE_MXFP8TENSOR
        ):
            pytest.skip("Requires PyTorch & CUDA device with TE MXFP8Tensor support")
        if Utils.world_size < 2:
            pytest.skip("Requires at least 2 distributed ranks for ZeRO sharding")

        outputs = TestMegatronFSDPE2E._training_loop(
            use_megatron_fsdp=True,
            use_megatron_fsdp_v2=True,
            ckpt_format="fsdp_dtensor",
            data_parallel_sharding_strategy=strategy,
            train_iters=3,
            seq_length=64,
            micro_batch_size=1,
            global_batch_size=8,
            init_model_with_meta_device=False,
            gradient_accumulation_fusion=False,
            overlap_param_gather=False,
            overlap_grad_reduce=False,
            **precision_configs,
        )

        if torch.distributed.get_rank() != 0:
            return

        assert len(outputs) == 3
        for step, output in enumerate(outputs):
            loss = output["lm loss"]
            assert torch.isfinite(loss), (
                f"Non-finite loss at step {step}, strategy={strategy}, "
                f"precision={precision_configs}"
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
