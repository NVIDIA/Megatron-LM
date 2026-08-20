# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""End-to-end EP 1F1B-overlap parity for M-FSDP v2.

Train the same small GPT model under M-FSDP v2 and the distributed-optimizer
reference, then compare per-step losses and parameter snapshots.
"""

import pytest
import torch
from torch.distributed.tensor import DTensor
from torch.testing import assert_close

import megatron.core.parallel_state as mpu
from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    gather_and_compute_chunk_metadata,
    uneven_dtensor_to_full_tensor,
)
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.utils import is_te_min_version, is_torch_min_version
from megatron.training.global_vars import destroy_global_vars
from tests.unit_tests.distributed.mfsdp_v1.utils import (
    make_gpt_mock_data_iterator,
    make_moe_args_model_and_optimizer,
    pretrain_forward_backward,
    set_manual_seed,
)
from tests.unit_tests.test_utilities import Utils


class TestMfsdpV2OverlapParity:
    NUM_STEPS = 20
    SEQUENCE_LENGTH = 64
    MICRO_BATCH_SIZE = 1
    GLOBAL_BATCH_SIZE = 8
    VOCAB_SIZE = 100

    def teardown_method(self):
        destroy_global_vars()
        destroy_num_microbatches_calculator()

    @staticmethod
    def _normalize_parameter_name(name):
        while name.startswith("module."):
            name = name[len("module.") :]
        return name

    @classmethod
    def _capture_parameters(cls, model):
        parameters = {}
        for chunk_index, model_chunk in enumerate(model):
            for name, parameter in model_chunk.named_parameters():
                tensor = parameter.detach()
                if isinstance(tensor, DTensor):
                    tensor = uneven_dtensor_to_full_tensor(tensor)
                name = cls._normalize_parameter_name(name)
                parameters[f"{chunk_index}.{name}"] = tensor.float().cpu()
        return parameters

    @classmethod
    def _load_parameters(cls, model, reference_parameters):
        with torch.no_grad():
            for chunk_index, model_chunk in enumerate(model):
                for name, parameter in model_chunk.named_parameters():
                    name = cls._normalize_parameter_name(name)
                    reference = reference_parameters[f"{chunk_index}.{name}"].to(
                        device=parameter.device, dtype=parameter.dtype
                    )
                    tensor = parameter.detach()
                    if isinstance(tensor, DTensor):
                        chunk = gather_and_compute_chunk_metadata(tensor)
                        local_slice = tuple(
                            slice(offset, offset + size)
                            for offset, size in zip(chunk.offsets, chunk.sizes)
                        )
                        tensor._local_tensor.copy_(reference[local_slice])
                    else:
                        tensor.copy_(reference)

    @classmethod
    def _run_training(cls, use_mfsdp_v2, case, initial_parameters=None):
        model_parallel_config = case["model_parallel_config"]
        Utils.initialize_model_parallel(**model_parallel_config)
        data_parallel_group = mpu.get_data_parallel_group()
        set_manual_seed(42)

        fsdp_args = {}
        if use_mfsdp_v2:
            fsdp_args = {
                "use_megatron_fsdp": True,
                "megatron_fsdp_version": 2,
                "ckpt_format": "fsdp_dtensor",
            }

        try:
            model, optimizer = make_moe_args_model_and_optimizer(
                ut_filename=__file__,
                model_family=case["model_family"],
                micro_batch_size=cls.MICRO_BATCH_SIZE,
                global_batch_size=cls.GLOBAL_BATCH_SIZE,
                vocab_size=cls.VOCAB_SIZE,
                padded_vocab_size=cls.VOCAB_SIZE,
                seq_length=cls.SEQUENCE_LENGTH,
                train_iters=cls.NUM_STEPS,
                gradient_accumulation_fusion=False,
                use_distributed_optimizer=not use_mfsdp_v2,
                **model_parallel_config,
                **case["model_config"],
                **fsdp_args,
            )
            if initial_parameters is not None:
                cls._load_parameters(model, initial_parameters)
                for sub_optimizer in getattr(optimizer, "chained_optimizers", [optimizer]):
                    sub_optimizer._copy_main_params_to_model_params()

            captured_initial_parameters = cls._capture_parameters(model)
            data_iterator = make_gpt_mock_data_iterator(
                dp_group=data_parallel_group,
                num_samples=cls.GLOBAL_BATCH_SIZE * cls.NUM_STEPS,
                vocab_size=cls.VOCAB_SIZE,
                sequence_length=cls.SEQUENCE_LENGTH,
                batch_size=cls.MICRO_BATCH_SIZE,
                seed=42,
            )
            losses = []
            parameter_snapshots = []
            run_name = "MFSDP v2" if use_mfsdp_v2 else "Reference"

            for step in range(cls.NUM_STEPS):
                optimizer.zero_grad()
                output = pretrain_forward_backward(
                    model=model,
                    data_iterator=data_iterator,
                    sequence_length=cls.SEQUENCE_LENGTH,
                    micro_batch_size=cls.MICRO_BATCH_SIZE,
                    num_micro_batches=cls.GLOBAL_BATCH_SIZE
                    // cls.MICRO_BATCH_SIZE
                    // data_parallel_group.size(),
                )
                loss = output[-1]["lm loss"].detach().cpu()
                update_successful, grad_norm, _ = optimizer.step()
                losses.append(loss)
                if torch.distributed.get_rank() == 0:
                    grad_norm_text = "None" if grad_norm is None else f"{float(grad_norm):.8f}"
                    print(
                        f"[{case['name']}][{run_name}] "
                        f"step={step + 1}/{cls.NUM_STEPS} "
                        f"loss={loss.item():.8f} grad_norm={grad_norm_text} "
                        f"update_successful={update_successful}",
                        flush=True,
                    )
                parameter_snapshots.append(cls._capture_parameters(model))

            return {
                "initial_parameters": captured_initial_parameters,
                "losses": losses,
                "parameters": parameter_snapshots,
            }
        finally:
            Utils.destroy_model_parallel()

    @pytest.mark.skipif(
        not is_torch_min_version("2.4.0"), reason="Requires PyTorch 2.4.0 or newer."
    )
    def test_compatible_with_nd_parallel(self):
        """MFSDP v2 EP overlap matches a distributed-optimizer reference."""
        if not is_te_min_version("2.3.0"):
            pytest.skip("1F1B expert-parallel overlap requires Transformer Engine 2.3.0 or newer.")

        case = {
            "name": "EP2 optim_grads_params 1F1B overlap",
            "model_family": "gpt",
            "model_parallel_config": {"expert_model_parallel_size": 2},
            "model_config": {
                "bf16": True,
                "data_parallel_sharding_strategy": "optim_grads_params",
                "clip_grad": 0.0,
                "megatron_fsdp_main_grads_dtype": torch.float32,
                "moe_grouped_gemm": True,
                "moe_token_dispatcher_type": "alltoall",
                "overlap_moe_expert_parallel_comm": True,
                "delay_wgrad_compute": True,
            },
            "loss_tolerance": {"atol": 0, "rtol": 0.05},
            "parameter_tolerance": {"atol": 5e-3, "rtol": 1e-3},
        }

        reference = self._run_training(use_mfsdp_v2=False, case=case)
        if torch.distributed.get_rank() == 0:
            print(f"[{case['name']}] Reference run completed successfully.", flush=True)

        actual = self._run_training(
            use_mfsdp_v2=True, case=case, initial_parameters=reference["initial_parameters"]
        )
        if torch.distributed.get_rank() == 0:
            print(f"[{case['name']}] MFSDP v2 run completed successfully.", flush=True)

        assert len(actual["losses"]) == len(reference["losses"])
        for step, (loss, reference_loss) in enumerate(zip(actual["losses"], reference["losses"])):
            assert_close(
                loss,
                reference_loss,
                **case["loss_tolerance"],
                msg=lambda msg: f"Loss mismatch at step {step}: {msg}",
            )

        assert len(actual["parameters"]) == len(reference["parameters"])
        for step, (parameters, reference_parameters) in enumerate(
            zip(actual["parameters"], reference["parameters"])
        ):
            assert parameters.keys() == reference_parameters.keys()
            for name in parameters:
                assert_close(
                    parameters[name],
                    reference_parameters[name],
                    **case["parameter_tolerance"],
                    msg=lambda msg: f"Parameter {name!r} mismatch at step {step}: {msg}",
                )
