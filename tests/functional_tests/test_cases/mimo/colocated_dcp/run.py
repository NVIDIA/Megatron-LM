# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Four-GPU MIMO correctness/health exercise using native schedules and optimizer.

Run with ``torchrun --nproc-per-node=4 .../run.py --layout dcp --steps 50
--output /path/run.json --save /path/checkpoint``. Resume with ``--load`` and
the same layout/seed; ``--steps`` is the final source-step index. CP1, static_cp2
and dcp use identical source batches and initialization. This is a small real
CLIP/GPT/MoE/MTP model qualification, not a full Qwen or performance benchmark.
"""

import argparse
import hashlib
import io
import json
import math
import os
from copy import copy
from dataclasses import fields, is_dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from examples.mimo.training.packed_step import PackedMimoStep, forward_step
from megatron.core import dist_checkpointing, parallel_state
from megatron.core.datasets.data_schedule import wrap_data_iterator
from megatron.core.dist_checkpointing.mapping import (
    LocalNonpersistentObject,
    ShardedBase,
    ShardedObject,
)
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo import MimoModel, MimoModelConfig
from megatron.core.models.mimo.planning import MediaTask, assign_media_to_producers
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

CHECKPOINT_METADATA = {"distrib_optim_sharding_type": "dp_reshardable", "checkpoint_version": 3.1}


def build_model(config, pg):
    """Real one-layer ViT and two-layer MoE decoder, plus a third MoE layer for MTP."""
    layer = get_gpt_layer_with_transformer_engine_spec(num_experts=4, moe_grouped_gemm=True)
    language = ModuleSpec(
        GPTModel,
        params=dict(
            config=config,
            transformer_layer_spec=layer,
            mtp_block_spec=get_gpt_mtp_block_spec(config, layer, True, pp_rank=0),
            vocab_size=256,
            max_sequence_length=512,
            position_embedding_type="rope",
            scatter_embedding_sequence_parallel=False,
            share_embeddings_and_output_weights=False,
            pg_collection=pg,
        ),
    )
    encoder_pg = copy(pg)
    for name in ("tp", "cp", "tp_cp", "ep", "expt_tp", "tp_ep", "tp_ep_pp"):
        setattr(encoder_pg, name, pg.tp)
    for name in ("dp", "dp_cp", "dp_gtp_remat", "dp_cp_gtp_remat", "expt_dp", "expt_dp_gtp_remat"):
        setattr(encoder_pg, name, pg.dp_cp)
    vision_config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=128,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        hidden_dropout=0,
        attention_dropout=0,
    )
    vision = ModuleSpec(
        VisionModalitySubmodules,
        params={"pg_collection": encoder_pg},
        submodules={
            "encoders": {
                "clip": ModuleSpec(
                    CLIPViTModel,
                    params=dict(
                        transformer_config=vision_config,
                        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
                        img_h=16,
                        img_w=16,
                        patch_dim=8,
                        pg_collection=encoder_pg,
                    ),
                )
            }
        },
    )
    model = MimoModel(
        MimoModelConfig(
            language_model_spec=language,
            modality_submodules_spec={"images": vision},
            special_token_ids={"images": 255},
            kv_format="thd",
        ),
        cp_group=pg.cp,
        tp_group=pg.tp,
        external_feature_transport=True,
    ).cuda()
    model.pg_collection = pg
    model.model_type = ModelType.encoder_or_decoder
    return model


def source_batch(step, seed, domain):
    """Deterministic multi-image and text-only samples, independent of rank/packing."""
    samples, tasks = {}, []
    for sid, length in enumerate((192, 128, 96, 64, 64, 32, 32, 32, 32)):
        logical = length - 3
        tokens = (torch.arange(length) + 11 * sid + step % 3) % 127 + 1
        labels = (tokens % 127) + 1
        image_count = (3, 2, 1, 1, 0, 1, 0, 0, 0)[sid] if step % 7 != 6 else 0
        media_ids = [3 * sid + image for image in range(image_count)]
        tokens[2 : 2 + 5 * image_count] = 255
        tokens[logical:] = labels[logical:] = 0
        mask = torch.zeros(length)
        mask[2 + 5 * image_count : logical] = 1
        samples[sid] = dict(
            tokens=tokens,
            labels=labels,
            loss_mask=mask,
            position_ids=torch.arange(logical),
            original_seq_len=logical,
            padded_seq_len=length,
            media_ids=media_ids,
        )
        tasks.extend(MediaTask(media_id, 5) for media_id in media_ids)
    assignments, slices = assign_media_to_producers(tasks, domain)
    images = [
        torch.randn((3, 16, 16), generator=torch.Generator().manual_seed(seed + task.media_id))
        for task in assignments[dist.get_rank()]
    ]
    inputs = {"images": {"clip": {"x": torch.stack(images).cuda().bfloat16()}}} if images else {}
    return samples, slices, inputs


def fingerprint(value, *, skip_optimizer_padding=False):
    """Exact local tensor/scalar fingerprint for checkpoint restoration, not CP equivalence."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, ShardedBase):
            visit(item.data)
        elif isinstance(item, LocalNonpersistentObject):
            visit(item.unwrap())
        elif isinstance(item, torch.Tensor):
            digest.update(str((item.shape, item.dtype)).encode())
            digest.update(
                item.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes()
            )
        elif isinstance(item, torch.Generator):
            visit(item.get_state())
        elif isinstance(item, io.BytesIO):
            visit(item.getvalue())
        elif is_dataclass(item):
            for field in fields(item):
                visit(field.name)
                visit(getattr(item, field.name))
        elif isinstance(item, dict):
            padding = item.get("padding")
            if (
                skip_optimizer_padding
                and isinstance(padding, LocalNonpersistentObject)
                and padding.unwrap() is True
            ):
                # dp_reshardable creates uninitialized padding that optimizer load discards.
                return
            for key in sorted(item, key=str):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        elif (
            isinstance(item, (str, int, float, bool, bytes, torch.dtype, torch.device))
            or item is None
        ):
            digest.update(str(item).encode())
        else:
            raise TypeError(f"Unsupported fingerprint value: {type(item)}")

    visit(value)
    return digest.hexdigest()


def check_branch_gradients(model, has_images, group):
    """Require finite native main_grads and actual vision/MTP participation."""
    flags = torch.zeros(4, device="cuda", dtype=torch.int)
    for index, module in enumerate((model.modality_submodules, model.language_model.mtp)):
        assert module is not None and any(
            parameter.requires_grad for parameter in module.parameters()
        )
        for parameter in module.parameters():
            if parameter.requires_grad:
                grad = parameter.main_grad
                flags[index] |= (~torch.isfinite(grad).all()).int()
                flags[index + 2] |= grad.ne(0).any().int()
    dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=group)
    assert not flags[:2].any(), "Non-finite encoder or MTP gradient"
    assert flags[3] and (flags[2] or not has_images), "Missing encoder or MTP gradient"
    return {"vision_nonzero": bool(flags[2]), "mtp_nonzero": bool(flags[3])}


def checkpoint_state(model, optimizer, step, args, loading=False):
    model_state = model.sharded_state_dict()
    rng = dict(
        cpu=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state(),
        tracker=get_cuda_rng_tracker().get_states(),
    )
    return dict(
        model=model_state,
        optimizer=optimizer.sharded_state_dict(
            model_state, is_loading=loading, metadata=CHECKPOINT_METADATA
        ),
        step=step,
        layout=args.layout,
        seed=args.seed,
        rng=ShardedObject("rng", rng, (dist.get_world_size(),), (dist.get_rank(),)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=("cp1", "static_cp2", "dcp"), required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--load", type=Path)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 4, "This E2E protocol requires four CUDA ranks"
    cp_size, dynamic = (1 if args.layout == "cp1" else 2), args.layout == "dcp"
    parallel_state.initialize_model_parallel(
        context_parallel_size=cp_size,
        expert_model_parallel_size=4,
        expert_tensor_parallel_size=1,
        dynamic_context_parallel=dynamic,
        create_gloo_process_groups=False,
    )
    torch.manual_seed(args.seed)
    model_parallel_cuda_manual_seed(args.seed)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=128,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        hidden_dropout=0,
        attention_dropout=0,
        context_parallel_size=cp_size,
        expert_model_parallel_size=4,
        num_moe_experts=4,
        moe_grouped_gemm=True,
        moe_router_topk=2,
        moe_token_dispatcher_type="alltoall",
        moe_router_load_balancing_type="global_aux_loss",
        moe_aux_loss_coeff=0.01,
        mtp_num_layers=1,
        mtp_loss_scaling_factor=0.1,
        calculate_per_token_loss=True,
        dynamic_context_parallel=dynamic,
        sequence_packing_scheduler="default_dynamic_cp" if dynamic else "dp_balanced",
        max_seqlen_per_dp_cp_rank={"cp1": 256, "static_cp2": 128, "dcp": 64}[args.layout],
        pad_packed_seq_alignment="max",
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
    )
    model = build_model(config, pg)
    ddp = DistributedDataParallel(
        config,
        DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            use_distributed_optimizer=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            average_in_collective=False,
        ),
        Float16Module(config, model),
        pg_collection=pg,
    )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            lr=0.001,
            min_lr=0.001,
            weight_decay=0,
            clip_grad=1.0,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=True,
        ),
        [ddp],
        config_overrides={},
        pg_collection=pg,
        use_gloo_process_groups=False,
    )
    adapter = PackedMimoStep(model, pg, lambda source: source, modality_name="images")
    adapter.install()
    start = 0
    report = dict(
        layout=args.layout,
        seed=args.seed,
        steps=[],
        initial_model_hash=fingerprint(model.state_dict()),
    )
    if args.load:
        restored = dist_checkpointing.load(
            checkpoint_state(model, optimizer, 0, args, True), str(args.load), strict="raise_all"
        )
        assert (restored["layout"], restored["seed"]) == (args.layout, args.seed)
        model.load_state_dict(restored["model"], strict=True)
        optimizer.load_state_dict(restored["optimizer"])
        torch.set_rng_state(restored["rng"]["cpu"])
        torch.cuda.set_rng_state(restored["rng"]["cuda"])
        get_cuda_rng_tracker().set_states(restored["rng"]["tracker"])
        start = int(restored["step"])
        restored_state = checkpoint_state(model, optimizer, start, args)
        saved = json.loads((args.load / f"rank-{dist.get_rank()}-fingerprints.json").read_text())
        actual = {
            name: fingerprint(restored_state[name], skip_optimizer_padding=name == "optimizer")
            for name in ("model", "optimizer", "rng")
        }
        assert (
            actual == saved
        ), "Checkpoint did not restore all local model/Adam/RNG tensors exactly"
        report["restore_verified"] = actual
    if args.steps <= start:
        raise ValueError("--steps must exceed the restored step (or zero for a fresh run)")
    schedule = get_forward_backward_func(pp_size=1)
    for step in range(start, args.steps):
        ddp.zero_grad_buffer()
        optimizer.zero_grad()
        source = source_batch(step, args.seed, dist.get_process_group_ranks(pg.dp_cp))
        iterator, rounds, _, _ = wrap_data_iterator(iter([source]), config, 1, pg)
        sizes = [item.cp_group.size() for item in adapter.plan.rounds]
        losses = schedule(
            forward_step_func=forward_step,
            data_iterator=iterator,
            model=ddp,
            num_microbatches=rounds,
            seq_length=512,
            micro_batch_size=1,
            pg_collection=pg,
        )
        assert not adapter.runtime.active
        if step == start:
            report["branch_gradients"] = check_branch_gradients(
                model, any(sample["media_ids"] for sample in source[0].values()), pg.dp_cp
            )
        updated, grad_norm, _ = optimizer.step()
        assert updated and grad_norm is not None and math.isfinite(float(grad_norm))
        lm = torch.stack([item["lm loss"] for item in losses]).sum(0)
        dist.all_reduce(lm, group=pg.dp_cp)
        MTPLossLoggingHelper.reduce_metrics_in_tracker()
        mtp = MTPLossLoggingHelper.tracker["loss_values"].mean().item()
        MTPLossLoggingHelper.clean_metrics_in_tracker()
        auxiliary = {}
        get_moe_metrics_tracker().report(
            1 / rounds,
            step + 1,
            num_layers=2,
            mtp_num_layers=1,
            total_loss_dict=auxiliary,
            pg_collection=pg,
        )
        row = dict(
            step=step + 1,
            source_hash=fingerprint(source),
            lm_loss=(lm[0] / lm[1]).item(),
            supervised_tokens=int(lm[1]),
            mtp_loss=mtp,
            grad_norm=float(grad_norm),
            local_cp_sizes=sizes,
            aux={name: float(value) for name, value in auxiliary.items()},
        )
        assert all(math.isfinite(value) for value in (row["lm_loss"], mtp, *row["aux"].values()))
        assert row["supervised_tokens"] > 0 and row["grad_norm"] > 0
        assert mtp > 0 and auxiliary, "The MTP and native MoE auxiliary objectives must participate"
        report["steps"].append(row)
        if dist.get_rank() == 0:
            print(json.dumps(row), flush=True)
    state = checkpoint_state(model, optimizer, args.steps, args)
    report["final_fingerprints"] = {
        name: fingerprint(state[name], skip_optimizer_padding=name == "optimizer")
        for name in ("model", "optimizer", "rng")
    }
    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
        dist_checkpointing.save(state, str(args.save), content_metadata=CHECKPOINT_METADATA)
        (args.save / f"rank-{dist.get_rank()}-fingerprints.json").write_text(
            json.dumps(report["final_fingerprints"])
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_name(f"{args.output.stem}-rank{dist.get_rank()}.json").write_text(
        json.dumps(report, indent=2)
    )
    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
