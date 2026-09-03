# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""End-to-end EP 1F1B-overlap parity for MFSDP v2.

Train the same small GPT model under MFSDP v2 and the distributed-optimizer
reference, then compare per-step losses and parameter snapshots.  The helpers
are local to this test because the MFSDP v1 test utilities now build only
HybridModel.
"""

import os
import sys
from functools import partial

import pytest
import torch
from torch.distributed.tensor import DTensor
from torch.testing import assert_close

import megatron.core.parallel_state as mpu
from gpt_builders import gpt_builder
from megatron.core.distributed import finalize_model_grads
from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    gather_and_compute_chunk_metadata,
    uneven_dtensor_to_full_tensor,
)
from megatron.core.enums import ModelType
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.utils import is_te_min_version, is_torch_min_version
from megatron.training.arguments import parse_args, validate_args
from megatron.training.global_vars import destroy_global_vars, set_global_variables
from megatron.training.training import setup_model_and_optimizer
from model_provider import model_provider
from tests.unit_tests.test_utilities import Utils


def _set_manual_seed(seed: int) -> None:
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)


def _make_mock_data_iterator(
    *,
    dp_group: torch.distributed.ProcessGroup,
    num_batches: int,
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    seed: int,
):
    """Yield deterministic, rank-local GPT batches for both training runs."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1009 * dp_group.rank())
    position_ids = torch.arange(sequence_length, dtype=torch.int64).unsqueeze(0)
    attention_mask = torch.triu(
        torch.ones((1, sequence_length, sequence_length), dtype=torch.bool), diagonal=1
    )

    for _ in range(num_batches):
        tokens = torch.randint(
            vocab_size, (batch_size, sequence_length), dtype=torch.int64, generator=generator
        )
        yield {
            "tokens": tokens,
            "labels": (tokens + 1) % vocab_size,
            "loss_mask": torch.ones((batch_size, sequence_length), dtype=torch.float32),
            "attention_mask": attention_mask.expand(batch_size, -1, -1, -1),
            "position_ids": position_ids.expand(batch_size, -1),
        }


def _forward_step_func(data_iterator, model, return_schedule_plan: bool = False):
    """Run eager forward or build the plan required by combined 1F1B."""
    data = next(data_iterator)
    tokens = data["tokens"].cuda(non_blocking=True)
    labels = data["labels"].cuda(non_blocking=True)
    loss_mask = data["loss_mask"].cuda(non_blocking=True)
    attention_mask = data["attention_mask"].cuda(non_blocking=True)
    position_ids = data["position_ids"].cuda(non_blocking=True)

    def loss_func(output_tensor):
        losses = output_tensor.float().view(-1)
        flat_loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses * flat_loss_mask) / flat_loss_mask.sum()
        # The overlap schedule releases the loss node after backward.  Retain a
        # standalone value for the end-to-end comparison.
        return loss, {"lm loss": loss.detach().clone()}

    if return_schedule_plan:
        schedule_plan = model.build_schedule_plan(
            tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
        )
        return schedule_plan, loss_func

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
    return output_tensor, loss_func


def _pretrain_forward_backward(
    *, model, data_iterator, sequence_length: int, micro_batch_size: int, num_micro_batches: int
):
    forward_backward_func = get_forward_backward_func()
    return forward_backward_func(
        forward_step_func=_forward_step_func,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=num_micro_batches,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        forward_only=False,
    )


def _make_model_and_optimizer(*, use_mfsdp_v2: bool, overrides: dict):
    """Build a small MoE GPT through the current pretrain configuration API."""
    base_args = {
        "num_layers": 4,
        "hidden_size": 128,
        "num_attention_heads": 2,
        "max_position_embeddings": 128,
        "bf16": False,
        "add_bias_linear": False,
        "swiglu": True,
        "position_embedding_type": "rope",
        "rotary_percent": 1.0,
        "hidden_dropout": 0.0,
        "attention_dropout": 0.0,
        "num_experts": 4,
        "moe_shared_expert_intermediate_size": 256,
        "moe_layer_freq": [0, 0, 1, 1],
        "moe_permute_fusion": True,
        "moe_router_fusion": True,
        "moe_router_topk": 2,
        "moe_router_dtype": "fp32",
        "create_attention_mask_in_dataloader": True,
        "lr": 3e-5,
        "min_lr": 3e-5,
        "use_distributed_optimizer": not use_mfsdp_v2,
        "use_megatron_fsdp": use_mfsdp_v2,
        "megatron_fsdp_version": 2,
        "ckpt_format": "fsdp_dtensor" if use_mfsdp_v2 else "torch_dist",
        "gradient_accumulation_fusion": False,
        "finalize_model_grads_func": finalize_model_grads,
        "spec": None,
        "mtp_num_layers": None,
    }
    base_args.update(overrides)

    original_argv = sys.argv
    try:
        sys.argv = [__file__]
        args = parse_args()
    finally:
        sys.argv = original_argv
    for key, value in base_args.items():
        setattr(args, key, value)
    validate_args(args)

    destroy_global_vars()
    destroy_num_microbatches_calculator()
    set_global_variables(args, build_tokenizer=False)

    cfg_container = Utils.pretrain_config_from_global_args(args, "gpt")
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    model, optimizer, _ = setup_model_and_optimizer(
        model_type=ModelType.encoder_or_decoder,
        model_provider_func=partial(model_provider, gpt_builder),
        cfg_container=cfg_container,
        pg_collection=pg_collection,
    )
    return model, optimizer


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
    def _normalize_parameter_name(name: str) -> str:
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
    def _run_training(cls, *, use_mfsdp_v2: bool, initial_parameters=None):
        Utils.initialize_model_parallel(expert_model_parallel_size=2)
        data_parallel_group = mpu.get_data_parallel_group()
        _set_manual_seed(42)

        try:
            model, optimizer = _make_model_and_optimizer(
                use_mfsdp_v2=use_mfsdp_v2,
                overrides={
                    "micro_batch_size": cls.MICRO_BATCH_SIZE,
                    "global_batch_size": cls.GLOBAL_BATCH_SIZE,
                    "vocab_size": cls.VOCAB_SIZE,
                    "padded_vocab_size": cls.VOCAB_SIZE,
                    "untie_embeddings_and_output_weights": True,
                    "seq_length": cls.SEQUENCE_LENGTH,
                    "train_iters": cls.NUM_STEPS,
                    "expert_model_parallel_size": 2,
                    "bf16": True,
                    "data_parallel_sharding_strategy": "optim_grads_params",
                    "clip_grad": 0.0,
                    "megatron_fsdp_main_grads_dtype": torch.float32,
                    "moe_grouped_gemm": True,
                    "moe_token_dispatcher_type": "alltoall",
                    "overlap_moe_expert_parallel_comm": True,
                    "delay_wgrad_compute": True,
                },
            )
            if use_mfsdp_v2 and os.environ.get("MFSDP_PARITY_DEBUG"):
                for model_chunk in model:
                    adapter = model_chunk
                    while not hasattr(adapter, "post_backward"):
                        adapter = adapter.module
                    root = adapter.module
                    source_model = root
                    while not hasattr(source_model, "decoder"):
                        source_model = source_model.module
                    final_layernorm = source_model.decoder.final_layernorm
                    final_layernorm.register_forward_hook(
                        lambda _module, _inputs, _output: print(
                            f"[MFSDP parity debug][rank={torch.distributed.get_rank()}] "
                            "final_layernorm forward",
                            flush=True,
                        )
                    )
                    for group in root.parameter_groups:
                        for fsdp_parameter in group.fsdp_parameters:
                            if not any(
                                fqn.endswith("decoder.final_layernorm.weight")
                                for fqn in fsdp_parameter.fqns
                            ):
                                continue
                            fsdp_parameter.unsharded.register_post_accumulate_grad_hook(
                                lambda parameter: print(
                                    f"[MFSDP parity debug][rank={torch.distributed.get_rank()}] "
                                    "final_layernorm grad ready "
                                    f"none={parameter.grad is None}",
                                    flush=True,
                                )
                            )

                    original_reduce_gradient_groups = root._reduce_gradient_groups

                    def debug_reduce_gradient_groups(
                        *, root=root, original=original_reduce_gradient_groups
                    ):
                        missing = [
                            fsdp_parameter.fqns
                            for group in root.parameter_groups
                            for fsdp_parameter in group.fsdp_parameters
                            if group.requires_grad and fsdp_parameter.unsharded.grad is None
                        ]
                        countdown = root._trainable_parameter_countdown
                        print(
                            f"[MFSDP parity debug][rank={torch.distributed.get_rank()}] "
                            f"root reduce phase={root.phase} "
                            f"countdown={countdown._value}/{countdown.initial_value} "
                            f"missing={missing}",
                            flush=True,
                        )
                        return original()

                    root._reduce_gradient_groups = debug_reduce_gradient_groups
            if initial_parameters is not None:
                cls._load_parameters(model, initial_parameters)
                for sub_optimizer in getattr(optimizer, "chained_optimizers", [optimizer]):
                    sub_optimizer._copy_main_params_to_model_params()

            captured_initial_parameters = cls._capture_parameters(model)
            num_micro_batches = (
                cls.GLOBAL_BATCH_SIZE // cls.MICRO_BATCH_SIZE // data_parallel_group.size()
            )
            data_iterator = _make_mock_data_iterator(
                dp_group=data_parallel_group,
                num_batches=cls.NUM_STEPS * num_micro_batches,
                batch_size=cls.MICRO_BATCH_SIZE,
                sequence_length=cls.SEQUENCE_LENGTH,
                vocab_size=cls.VOCAB_SIZE,
                seed=42,
            )
            losses = []
            parameter_snapshots = []
            run_name = "MFSDP v2" if use_mfsdp_v2 else "DistOpt"

            for step in range(cls.NUM_STEPS):
                for model_chunk in model:
                    model_chunk.zero_grad_buffer()
                optimizer.zero_grad()
                output = _pretrain_forward_backward(
                    model=model,
                    data_iterator=data_iterator,
                    sequence_length=cls.SEQUENCE_LENGTH,
                    micro_batch_size=cls.MICRO_BATCH_SIZE,
                    num_micro_batches=num_micro_batches,
                )
                loss = (
                    torch.stack(
                        [microbatch_output["lm loss"].detach() for microbatch_output in output]
                    )
                    .mean()
                    .cpu()
                )
                update_successful, grad_norm, _ = optimizer.step()
                losses.append(loss)
                if torch.distributed.get_rank() == 0:
                    grad_norm_text = "None" if grad_norm is None else f"{float(grad_norm):.8f}"
                    print(
                        f"[{run_name}] step={step + 1}/{cls.NUM_STEPS} "
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
        not is_torch_min_version("2.6.0"),
        reason="Combined expert-parallel overlap requires PyTorch 2.6.0 or newer.",
    )
    @pytest.mark.skipif(
        not is_te_min_version("2.7.0"),
        reason="Delayed wgrad without gradient-accumulation fusion requires TE 2.7 or newer.",
    )
    def test_compatible_with_nd_parallel(self, distributed_setup):
        """MFSDP v2 EP delayed-wgrad overlap matches DistOpt without static prefetch."""
        if (
            distributed_setup.world_size < 2
            or distributed_setup.world_size % 2
            or distributed_setup.world_size > self.GLOBAL_BATCH_SIZE
        ):
            pytest.skip("Requires an even world size between two and the global batch size.")

        reference = self._run_training(use_mfsdp_v2=False)
        if torch.distributed.get_rank() == 0:
            print("DistOpt reference run completed successfully.", flush=True)

        actual = self._run_training(
            use_mfsdp_v2=True, initial_parameters=reference["initial_parameters"]
        )
        if torch.distributed.get_rank() == 0:
            print("MFSDP v2 run completed successfully.", flush=True)

        for run_name, run in (("DistOpt", reference), ("MFSDP v2", actual)):
            changed_parameters = [
                name
                for name, initial_parameter in run["initial_parameters"].items()
                if not torch.equal(initial_parameter, run["parameters"][-1][name])
            ]
            assert changed_parameters, f"{run_name} did not update any parameters."

        assert len(actual["losses"]) == len(reference["losses"])
        for step, (loss, reference_loss) in enumerate(zip(actual["losses"], reference["losses"])):
            assert_close(
                loss,
                reference_loss,
                atol=0,
                rtol=0.05,
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
                    atol=5e-3,
                    rtol=1e-3,
                    msg=lambda msg: f"Parameter {name!r} mismatch at step {step}: {msg}",
                )
