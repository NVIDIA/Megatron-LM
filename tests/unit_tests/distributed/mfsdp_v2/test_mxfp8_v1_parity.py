# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""End-to-end parity of M-FSDP v2 MXFP8 model weights against Megatron-FSDP v1.

Both sides train the same small GPT-MoE model built with MXFP8 primary
weights (``--fp8-recipe mxfp8 --fp8-param-gather`` with ``--fp8`` mode): TE
layers create ``MXFP8Tensor`` parameters under ``fp8_model_init``, which
``FsdpModule`` detects and routes to ``Fp8ParameterGroup`` on the v2 side
and which v1's fp8 param-gather machinery handles natively on the reference
side. Per-step losses and parameter snapshots are compared with the same
tolerances as the ND-parallel parity test.

Requires a Blackwell (device capability >= 10) GPU with Transformer Engine
MXFP8Tensor support. Run with torchrun:
    torchrun --nproc-per-node 4 -m pytest -s -x \
      tests/unit_tests/distributed/mfsdp_v2/test_mxfp8_v1_parity.py
"""

import pytest
import torch
from torch.distributed.tensor import DTensor
from torch.testing import assert_close

import megatron.core.parallel_state as mpu
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
    Fp8ParameterGroup,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import (
    HAVE_TE_MXFP8TENSOR,
    is_float8tensor,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    gather_and_compute_chunk_metadata,
    uneven_dtensor_to_full_tensor,
)
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.training.global_vars import destroy_global_vars
from tests.unit_tests.distributed.mfsdp_v1.utils import (
    make_gpt_mock_data_iterator,
    make_moe_args_model_and_optimizer,
    pretrain_forward_backward,
    set_manual_seed,
)
from tests.unit_tests.test_utilities import Utils


class TestMegatronFSDPE2EMxfp8:
    NUM_STEPS = 10
    SEQUENCE_LENGTH = 64
    MICRO_BATCH_SIZE = 1
    GLOBAL_BATCH_SIZE = 8
    # MXFP8 block quantization requires every fp8 weight dim to be divisible
    # by 32 (hidden 128 is; the base harness vocab of 100 is not).
    VOCAB_SIZE = 128

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
        """Capture the non-fp8 parameters as float32 tensors.

        The fp8 primary weights are excluded: on the v2 side they rest as
        sharded DTensors of the fp32 main weights, while on the v1 side
        ``named_parameters`` returns the quantized tensors, so there is no
        representation both sides expose at rest. Their parity is covered by
        the per-step loss and grad-norm comparisons.
        """
        fp8_names = set()
        for chunk_index, model_chunk in enumerate(model):
            for name, parameter in model_chunk.named_parameters():
                if is_float8tensor(parameter):
                    fp8_names.add(f"{chunk_index}.{cls._normalize_parameter_name(name)}")
        parameters = {}
        for chunk_index, model_chunk in enumerate(model):
            for name, parameter in model_chunk.named_parameters():
                tensor = parameter.detach()
                if isinstance(tensor, DTensor):
                    tensor = uneven_dtensor_to_full_tensor(tensor)
                name = cls._normalize_parameter_name(name)
                parameters[f"{chunk_index}.{name}"] = tensor.float().cpu()
            for group in getattr(model_chunk, "_parameter_groups", []):
                if not isinstance(group, Fp8ParameterGroup):
                    continue
                for fsdp_parameter in group.fsdp_parameters:
                    for fqn in fsdp_parameter.fqns:
                        fp8_names.add(f"{chunk_index}.{cls._normalize_parameter_name(fqn)}")
        return parameters, fp8_names

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

        # MXFP8 primary weights via fp8 param gather. fp8 mode (--fp8) is
        # required by the config validation (fp8_param requires fp8); the v2
        # adapter now allows the mxfp8 fp8-param-gather combination.
        fp8_args = {"fp8": "e4m3", "fp8_recipe": "mxfp8", "fp8_param_gather": True, "bf16": True}
        fsdp_args = {}
        if use_mfsdp_v2:
            fsdp_args = {
                "use_megatron_fsdp": True,
                "megatron_fsdp_version": 2,
                # MXFP8 primary weights under fp8_model_init are materialized
                # eagerly; meta-device init with fp8 weights needs validation.
                "init_model_with_meta_device": False,
                "ckpt_format": "fsdp_dtensor",
            }
        else:
            # Reference: Megatron-FSDP v1, the established fp8 param-gather
            # path (v1 requires the distributed optimizer).
            fsdp_args = {
                "use_megatron_fsdp": True,
                "megatron_fsdp_version": 1,
                "init_model_with_meta_device": True,
                "ckpt_format": "fsdp_dtensor",
            }

        try:
            model, optimizer = make_moe_args_model_and_optimizer(
                ut_filename=__file__,
                model_family="gpt",
                micro_batch_size=cls.MICRO_BATCH_SIZE,
                global_batch_size=cls.GLOBAL_BATCH_SIZE,
                vocab_size=cls.VOCAB_SIZE,
                padded_vocab_size=cls.VOCAB_SIZE,
                seq_length=cls.SEQUENCE_LENGTH,
                train_iters=cls.NUM_STEPS,
                gradient_accumulation_fusion=False,
                use_distributed_optimizer=not use_mfsdp_v2,
                **model_parallel_config,
                **fp8_args,
                **case["model_config"],
                **fsdp_args,
            )
            if initial_parameters is not None:
                cls._load_parameters(model, initial_parameters)
                for sub_optimizer in getattr(optimizer, "chained_optimizers", [optimizer]):
                    sub_optimizer._copy_main_params_to_model_params()

            captured_initial_parameters, fp8_names = cls._capture_parameters(model)
            data_iterators = [
                make_gpt_mock_data_iterator(
                    dp_group=data_parallel_group,
                    num_samples=cls.GLOBAL_BATCH_SIZE * cls.NUM_STEPS,
                    vocab_size=cls.VOCAB_SIZE,
                    sequence_length=cls.SEQUENCE_LENGTH,
                    batch_size=cls.MICRO_BATCH_SIZE,
                    seed=42,
                )
                for _ in model
            ]
            data_iterator = data_iterators if len(model) > 1 else data_iterators[0]
            losses = []
            grad_norms = []
            parameter_snapshots = []
            run_name = "MFSDP v2" if use_mfsdp_v2 else "MFSDP v1 reference"

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
                pipeline_parallel = model_parallel_config.get("pipeline_model_parallel_size", 1) > 1
                if pipeline_parallel:
                    if mpu.is_pipeline_last_stage():
                        loss = output[-1]["lm loss"].detach().float().clone()
                    else:
                        loss = torch.zeros((), device="cuda")
                else:
                    loss = output[-1]["lm loss"].detach().cpu()
                update_successful, grad_norm, _ = optimizer.step()
                grad_norms.append(grad_norm)
                if pipeline_parallel:
                    torch.distributed.broadcast(
                        loss,
                        src=mpu.get_pipeline_model_parallel_last_rank(),
                        group=mpu.get_pipeline_model_parallel_group(),
                    )
                    loss = loss.cpu()
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
                snapshot, _ = cls._capture_parameters(model)
                parameter_snapshots.append(snapshot)

            return {
                "initial_parameters": captured_initial_parameters,
                "losses": losses,
                "grad_norms": grad_norms,
                "parameters": parameter_snapshots,
                "fp8_names": fp8_names,
            }
        finally:
            Utils.destroy_model_parallel()

    @pytest.mark.skipif(
        not HAVE_TE_MXFP8TENSOR or torch.cuda.get_device_capability()[0] < 10,
        reason="Requires a Blackwell GPU with Transformer Engine MXFP8Tensor support.",
    )
    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(
                {
                    "name": "EP2 optim_grads_params MXFP8",
                    "model_parallel_config": {"expert_model_parallel_size": 2},
                    "model_config": {
                        "data_parallel_sharding_strategy": "optim_grads_params",
                        "megatron_fsdp_main_grads_dtype": torch.float32,
                        # Keep the hybrid Mamba projection dimensions divisible by
                        # the MXFP8 block size (32).
                        "mamba_num_heads": 32,
                        "moe_token_dispatcher_type": "alltoall",
                    },
                    "loss_tolerance": {"atol": 0, "rtol": 0.05},
                    "parameter_tolerance": {"atol": 5e-3, "rtol": 1e-3},
                },
                id="ep2-optim_grads_params-mxfp8",
            )
        ],
    )
    def test_parity_with_mfsdp_v1(self, case):
        """MFSDP v2 MXFP8 matches the Megatron-FSDP v1 fp8 param-gather path.

        Note: pipeline parallelism with EP is not covered — the MFSDP
        adapters' ``_get_dp_tp_mesh`` assumes the world is dp_cp x ep x tp
        (no PP ranks), a pre-existing gap unrelated to fp8.
        """
        reference = self._run_training(use_mfsdp_v2=False, case=case)
        if torch.distributed.get_rank() == 0:
            print(f"[{case['name']}] MFSDP v1 reference completed successfully.", flush=True)

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

        assert len(actual["grad_norms"]) == len(reference["grad_norms"])
        for step, (grad_norm, reference_grad_norm) in enumerate(
            zip(actual["grad_norms"], reference["grad_norms"])
        ):
            if grad_norm is not None and reference_grad_norm is not None:
                assert_close(
                    torch.as_tensor(grad_norm),
                    torch.as_tensor(reference_grad_norm),
                    atol=0,
                    rtol=0.05,
                    msg=lambda msg: f"Grad norm mismatch at step {step}: {msg}",
                )

        assert actual["fp8_names"] == reference["fp8_names"]
        assert len(actual["parameters"]) == len(reference["parameters"])
        for step, (parameters, reference_parameters) in enumerate(
            zip(actual["parameters"], reference["parameters"])
        ):
            assert parameters.keys() == reference_parameters.keys()
            for name in parameters:
                if name in actual["fp8_names"]:
                    continue
                assert_close(
                    parameters[name],
                    reference_parameters[name],
                    **case["parameter_tolerance"],
                    msg=lambda msg: f"Parameter {name!r} mismatch at step {step}: {msg}",
                )
