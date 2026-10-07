# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Train a complete GDN/attention language model with native parallel schedules.

This is a reduced-size integration test, not qualification of a pretrained Qwen
checkpoint. Each topology compares offloading with a baseline at the same topology.
"""

from collections.abc import Iterator
from dataclasses import replace
from functools import partial
from typing import Any

import pytest
import torch

from megatron.core.dist_checkpointing.dict_utils import dict_list_map_outplace, diff
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_layer_with_experimental_attention_variant_spec,
)
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer.optimizer import MegatronOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import PipelineOffloadManager
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import HAVE_FLA
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import ModuleSpec, TransformerConfig
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import get_batch_on_this_cp_rank
from tests.unit_tests.test_utilities import Utils


def _training_config(
    tp: int, sp: bool, cp: int, pp: int, vp: int | None, fraction: float, recompute_norm: bool
) -> TransformerConfig:
    """Configure the reduced-size complete decoder and its parallel topology."""
    return TransformerConfig(
        num_layers=8,
        hidden_size=256,
        ffn_hidden_size=896,
        num_attention_heads=8,
        num_query_groups=2,
        kv_channels=64,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        layernorm_zero_centered_gamma=True,
        qk_layernorm=True,
        attention_output_gate=True,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        add_bias_linear=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        bf16=True,
        pipeline_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        gradient_accumulation_fusion=False,
        tensor_model_parallel_size=tp,
        sequence_parallel=sp,
        context_parallel_size=cp,
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=vp,
        # PP=2/VPP requires the native ordering for both directions to the same peer.
        batch_p2p_comm=False,
        attention_backend=AttnBackend.fused if cp > 1 else AttnBackend.unfused,
        experimental_attention_variant="gdn",
        linear_attention_freq=4,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        recompute_granularity="selective" if recompute_norm else None,
        recompute_modules=["gdn_norm_out"] if recompute_norm else [],
        fine_grained_activation_offloading=True,
        offload_modules=["gdn_core_attn"],
        activation_offload_fraction=fraction,
        min_offloaded_tensor_size=1024,
    )


def _build_training_model(
    config: TransformerConfig, groups: ProcessGroupCollection
) -> tuple[list[DistributedDataParallel], MegatronOptimizer]:
    """Construct full decoder blocks, DDP gradient buffers and the BF16 Adam optimizer."""
    # Both layer recipes include their MLP: eight complete decoder blocks,
    # with three GDN blocks for every gated GQA block.
    layer_specs = get_transformer_layer_with_experimental_attention_variant_spec(config)
    stack_spec = ModuleSpec(
        module=HybridStack,
        submodules=HybridStackSubmodules(gdn_layer=layer_specs[0], attention_layer=layer_specs[3]),
    )
    torch.manual_seed(31)
    model_parallel_cuda_manual_seed(31)
    models = []
    vp = config.virtual_pipeline_model_parallel_size
    pattern = "GGG*GGG*"
    chunk_size = len(pattern) // (config.pipeline_model_parallel_size * (vp or 1))
    pattern = "|".join(
        pattern[start : start + chunk_size] for start in range(0, len(pattern), chunk_size)
    )
    for stage in range(vp or 1):
        vp_stage = stage if vp is not None else None
        model = HybridModel(
            config=config,
            hybrid_stack_spec=stack_spec,
            hybrid_layer_pattern=pattern,
            vocab_size=512,
            max_sequence_length=128,
            pre_process=is_pp_first_stage(groups.pp) and is_vp_first_stage(vp_stage, vp),
            post_process=is_pp_last_stage(groups.pp) and is_vp_last_stage(vp_stage, vp),
            share_embeddings_and_output_weights=True,
            position_embedding_type="rope",
            rotary_percent=0.25,
            rotary_base=10_000_000,
            pg_collection=groups,
            vp_stage=vp_stage,
        ).cuda()
        models.append(
            DistributedDataParallel(
                config,
                DistributedDataParallelConfig(grad_reduce_in_fp32=True),
                Float16Module(config, model),
                pg_collection=groups,
            )
        )
        models[-1].broadcast_params()
    optimizer = get_megatron_optimizer(
        OptimizerConfig(bf16=True, lr=1e-3, clip_grad=1.0),
        models,
        use_gloo_process_groups=False,
        pg_collection=groups,
    )
    config.grad_scale_func = optimizer.scale_loss
    config.finalize_model_grads_func = partial(finalize_model_grads, pg_collection=groups)
    return models, optimizer


def _forward_step(
    batches: Iterator[dict[str, torch.Tensor]], model: torch.nn.Module
) -> tuple[torch.Tensor, Any]:
    batch = next(batches)
    output = model(batch["tokens"], batch["position_ids"], None, labels=batch["labels"])

    def loss_func(losses: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss = losses.float().mean()
        return loss, {"loss": loss.detach().clone()}

    return output, loss_func


def _snapshot_leaf(value: Any) -> Any:
    """Freeze a tensor value before the next optimizer step overwrites its storage."""
    return value.detach().clone() if isinstance(value, torch.Tensor) else value


@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("recompute_norm", [False, True])
@pytest.mark.parametrize(
    "tp,sp,cp,pp,vp",
    [
        (1, False, 1, 1, None),
        (2, False, 1, 1, None),
        (2, True, 1, 1, None),
        (1, False, 2, 1, None),
        (1, False, 1, 2, None),
        (1, False, 1, 2, 2),
        (2, True, 2, 1, None),
        (2, True, 1, 2, None),
    ],
    ids=["dp", "tp", "tp_sp", "cp", "pp", "pp_vpp", "tp_sp_cp", "tp_sp_pp"],
)
def test_gdn_offload_training(
    tp: int,
    sp: bool,
    cp: int,
    pp: int,
    vp: int | None,
    fraction: float,
    recompute_norm: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Match losses, accumulated gradients, updated weights and Adam states on every rank."""
    if Utils.world_size % (tp * cp * pp):
        pytest.skip("This topology requires a world size divisible by TP * CP * PP.")
    # Avoid attention-backward nondeterminism obscuring the offload comparison.
    # GDN still uses FLA: config.deterministic_mode would select its reference path.
    monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    Utils.initialize_model_parallel(tp, pp, vp, context_parallel_size=cp)
    off_interface.reset_instance()
    try:
        groups = ProcessGroupCollection.use_mpu_process_groups()
        config = _training_config(tp, sp, cp, pp, vp, fraction, recompute_norm)
        baseline_config = replace(
            config, fine_grained_activation_offloading=False, offload_modules=[]
        )
        baseline, baseline_optimizer = _build_training_model(baseline_config, groups)
        offloaded, offload_optimizer = _build_training_model(config, groups)
        for expected, actual in zip(baseline, offloaded, strict=True):
            actual.load_state_dict(expected.state_dict())
        # The optimizer has already created FP32 master weights; reload them
        # after aligning model weights, before any gradient accumulation.
        offload_optimizer.reload_model_params()
        schedule = get_forward_backward_func(pp_size=pp, vp_size=vp)
        generator = torch.Generator().manual_seed(19 + groups.dp.rank())
        manager = PipelineOffloadManager.get_instance()
        for _ in range(3):
            batches = []
            for _ in range(2):
                tokens = torch.randint(0, 512, (1, 129), generator=generator, device="cpu").cuda()
                batch = {
                    "tokens": tokens[:, :-1].contiguous(),
                    "labels": tokens[:, 1:].contiguous(),
                    "position_ids": torch.arange(128, device="cuda").unsqueeze(0),
                }
                batches.append(get_batch_on_this_cp_rank(batch, False, cp_group=groups.cp))
            # Fixed weights and zero dropout must reproduce each repeated example.
            # This also catches VPP payload mixups in the disabled baseline.
            batches = batches * 2
            snapshots = []
            for models, optimizer in (
                (baseline, baseline_optimizer),
                (offloaded, offload_optimizer),
            ):
                optimizer.zero_grad()
                for model in models:
                    model.zero_grad_buffer()
                losses = schedule(
                    forward_step_func=_forward_step,
                    data_iterator=[iter(batches) for _ in models],
                    model=models,
                    num_microbatches=4,
                    seq_length=128,
                    micro_batch_size=1,
                    forward_only=False,
                    p2p_communicator=P2PCommunicator(groups.pp, models[0].config),
                    pg_collection=groups,
                )
                # Reject a bad baseline on every rank before the next collective.
                losses_valid = torch.tensor(
                    len(losses) == (4 if is_pp_last_stage(groups.pp) else 0)
                    and not any(diff(losses[:2], losses[2:])),
                    dtype=torch.int32,
                    device="cuda",
                )
                torch.distributed.all_reduce(losses_valid, op=torch.distributed.ReduceOp.MIN)
                assert losses_valid.item(), "Loss records must preserve repeated microbatch inputs."
                grads = {
                    f"{stage}.{name}": parameter.main_grad.detach().clone()
                    for stage, model in enumerate(models)
                    for name, parameter in model.named_parameters()
                }
                success, grad_norm, _ = optimizer.step()
                assert success and grad_norm > 0
                snapshots.append(
                    {
                        "losses": losses,
                        "grads": grads,
                        "grad_norm": grad_norm,
                        "weights": {
                            f"{stage}.{name}": parameter.detach().clone()
                            for stage, model in enumerate(models)
                            for name, parameter in model.named_parameters()
                        },
                        "optimizer": dict_list_map_outplace(_snapshot_leaf, optimizer.state_dict()),
                    }
                )
            torch.cuda.synchronize()
            assert not any(diff(snapshots[0], snapshots[1]))
            assert manager.cpu_tensor_pool.get_pool_status()["global_stats"]["current_in_use"] == 0
            assert not manager._is_warmup
            if fraction == 0:
                assert manager.offload_summary_total_bytes == 0
            else:
                assert manager.offload_summary_total_bytes > 0
            del snapshots
    finally:
        torch.cuda.synchronize()
        off_interface.reset_instance()
        Utils.destroy_model_parallel()
