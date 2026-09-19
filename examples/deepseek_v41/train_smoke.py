# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Reduced-width V4.1 training/performance harness with the released full-depth schedule.

Run with torch.distributed.run. All file output lives under --output-dir.
This exercises the MCore pipeline schedule, DDP, optimizer, EP, all architecture
components, gradient accumulation, and distributed checkpoint round trips.
"""

import argparse
import json
import logging
import os
import statistics
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch

from megatron.core import config as core_config
from megatron.core import dist_checkpointing, parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.models.deepseek_v41.config import DeepSeekV41Config
from megatron.core.models.deepseek_v41.engram_hash import EngramLayout
from megatron.core.models.deepseek_v41.image_processing import ImageInput
from megatron.core.models.deepseek_v41.model import DeepSeekV41Model
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import Float16Module

logger = logging.getLogger(__name__)


def build_config(args):
    """Scale widths and table capacities, retaining all roles and component depth."""
    hf = json.loads(Path(__file__).with_name("config_flash.json").read_text())
    text = hf["text_config"]
    if args.depth == "small":
        text.update(
            num_hidden_layers=6,
            compress_ratios=[0, 2, 2, 1, 1, 1],
            kv_source_layer_ids=[1, 3],
            index_source_layer_ids=[1, 3, 4],
            candidate_source_layer_id=3,
            engram_layer_ids=[1, 4],
            dspark_target_layer_ids=[3, 4, 5],
        )
    text.update(
        hidden_size=args.hidden_size,
        num_attention_heads=8,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=128,
        o_lora_rank=64,
        o_groups=4,
        moe_intermediate_size=args.hidden_size // 2,
        n_routed_experts=8,
        num_experts_per_tok=2,
        index_n_heads=4,
        index_head_dim=128,
        index_topk=min(32, args.seq_length),
        candidate_topk_blocks=4,
        candidate_block_size=8,
        engram_vocab_size=101,
        engram_n_heads=2,
        engram_head_dim=16,
        engram_compressed_vocab_size=1024,
        dspark_n_routed_experts=4,
        dspark_num_experts_per_tok=2,
        dspark_noise_token_id=1023,
        dspark_markov_rank=32,
        vocab_size=1024,
    )
    layout = EngramLayout.from_args(
        SimpleNamespace(
            **{
                k.replace("engram_pad_token_id", "engram_pad_id"): v
                for k, v in text.items()
                if k.startswith("engram_")
            }
        )
    )
    text["engram_num_embeddings"] = [sum(sum(row) for row in layer) for layer in layout.primes]
    vision = hf["vision_config"]
    vision.update(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        patch_size=2,
        min_pixels=36,
        num_hidden_layers=2 if args.depth == "small" else vision["num_hidden_layers"],
    )
    return DeepSeekV41Config.from_hf(
        hf,
        params_dtype=torch.bfloat16,
        bf16=True,
        pipeline_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        expert_model_parallel_size=args.ep,
        moe_token_dispatcher_type="alltoall",
        dsa_indexer_loss_coeff=0.01,
        dsa_indexer_use_sparse_loss=True,
        dsa_kernel_backend="cudnn" if args.fused else "none",
        use_fused_mhc=args.fused,
        moe_grouped_gemm=args.fused,
        moe_permute_fusion=args.fused,
        apply_rope_fusion=args.fused,
        bias_activation_fusion=args.fused,
        gradient_accumulation_fusion=args.fused,
    )


def main():
    """Train, validate gradients/updates, and emit machine-readable performance evidence."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", choices=("small", "full"), default="small")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--seq-length", type=int, default=64)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--microbatches", type=int, default=2)
    parser.add_argument("--ep", type=int, default=2)
    parser.add_argument("--fused", action="store_true")
    parser.add_argument("--distributed-optimizer", action="store_true")
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.seq_length < 16:
        parser.error("sequence length must leave room for an image and a DSpark block")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    core_config.ENABLE_EXPERIMENTAL = True
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl")
    parallel_state.initialize_model_parallel(expert_model_parallel_size=args.ep)
    groups = ProcessGroupCollection.use_mpu_process_groups()
    model_parallel_cuda_manual_seed(1234)
    torch.manual_seed(1234)
    config = build_config(args)
    raw = DeepSeekV41Model(
        config, 1024, args.seq_length, pg_collection=groups, token_map=torch.arange(1024)
    ).cuda()
    model = Float16Module(config, raw)
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,
        use_distributed_optimizer=args.distributed_optimizer,
        check_for_nan_in_grad=True,
    )
    model = DistributedDataParallel(config, ddp_config, model, pg_collection=groups)
    optim_config = OptimizerConfig(
        lr=1e-4,
        min_lr=1e-5,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=args.distributed_optimizer,
        clip_grad=1.0,
        log_num_zeros_in_grad=True,
    )
    optimizer = get_megatron_optimizer(
        optim_config, [model], pg_collection=groups, use_gloo_process_groups=False
    )
    config.grad_scale_func = optimizer.scale_loss
    config.finalize_model_grads_func = partial(finalize_model_grads, pg_collection=groups)
    config.no_sync_func = model.no_sync
    generator = torch.Generator(device="cuda").manual_seed(5000 + torch.distributed.get_rank())
    tokens = torch.randint(0, 1000, (1, args.seq_length + 1), device="cuda", generator=generator)
    positions = torch.arange(args.seq_length, device="cuda").unsqueeze(0)
    image = ImageInput(
        1,
        torch.randn(9, 3, 2, 2, device="cuda", generator=generator),
        3,
        3,
        torch.tensor([0, 1, 2, 3], device="cuda"),
    )
    anchors = torch.tensor([[args.seq_length // 2]], device="cuda")

    def forward_step(iterator, wrapped):
        result = wrapped(
            tokens[:, :-1],
            positions,
            labels=tokens[:, 1:],
            images=[[image]],
            draft_anchor_positions=anchors,
        )
        loss = result.backbone.float().mean() + result.draft_loss

        def reduce_loss(value):
            # The pipeline schedule divides the returned scalar in place.
            return value, {"loss": value.detach().clone()}

        return loss, reduce_loss

    schedule = get_forward_backward_func()
    rows = []
    probes = {
        "backbone": raw.decoder.layers[0].attention.linear_q_down_proj.weight,
        "engram": raw.decoder.layers[config.engram_config.layer_ids[0]]
        .engram.embed.tables[0]
        .weight,
        "vision": raw.vision.patch_embed.proj.weight,
        "aligner": raw.aligner.w1.weight,
        "dspark": raw.dspark.main_proj.weight,
    }
    before = {name: parameter.detach().clone() for name, parameter in probes.items()}
    for step in range(args.steps):
        optimizer.zero_grad()
        model.zero_grad_buffer()
        torch.cuda.synchronize()
        start = time.perf_counter()
        losses = schedule(
            forward_step_func=forward_step,
            data_iterator=None,
            model=model,
            num_microbatches=args.microbatches,
            seq_length=args.seq_length,
            micro_batch_size=1,
            forward_only=False,
        )
        success, grad_norm, zeros = optimizer.step()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        loss = sum(item["loss"].item() for item in losses) / len(losses)
        if not success or not torch.isfinite(torch.tensor([loss, grad_norm])).all():
            raise RuntimeError(f"Non-finite or skipped step {step}: {loss=}, {grad_norm=}")
        row = dict(
            step=step + 1,
            loss=loss,
            grad_norm=float(grad_norm),
            milliseconds=elapsed,
            peak_memory_gb=torch.cuda.max_memory_allocated() / 2**30,
        )
        rows.append(row)
        if torch.distributed.get_rank() == 0:
            logger.info("V41_TRAIN_STEP %s", json.dumps(row))
    updates = {name: not torch.equal(before[name], parameter) for name, parameter in probes.items()}
    if not all(updates.values()):
        raise RuntimeError(f"Some components did not update: {updates}")
    if args.checkpoint:
        checkpoint_dir = args.output_dir / "checkpoint"
        if torch.distributed.get_rank() == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()
        before_load = {
            name: parameter.detach().clone() for name, parameter in raw.named_parameters()
        }
        snapshot = raw.sharded_state_dict(metadata={"dp_cp_group": groups.dp_cp})
        dist_checkpointing.save(snapshot, checkpoint_dir)
        with torch.no_grad():
            for parameter in raw.parameters():
                parameter.zero_()
        restored = dist_checkpointing.load(
            raw.sharded_state_dict(metadata={"dp_cp_group": groups.dp_cp}), checkpoint_dir
        )
        raw.load_state_dict(restored, strict=True)
        for name, parameter in raw.named_parameters():
            if not torch.equal(before_load[name], parameter):
                raise RuntimeError(f"Checkpoint changed parameter {name}")
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        report = dict(
            arguments=vars(args) | {"output_dir": str(args.output_dir)},
            rows=rows,
            median_ms=statistics.median(r["milliseconds"] for r in rows[2:] or rows),
            torch=torch.__version__,
            gpu=torch.cuda.get_device_name(),
            logical_depth=len(raw.decoder.layers),
            parameter_updates=updates,
            components=["CSA2", "single-pass mHC", "Engram", "vision", "DSpark"],
        )
        (args.output_dir / "results.json").write_text(json.dumps(report, indent=2))
        logger.info("DSV41_TRAINING_OK")
    torch.distributed.barrier()
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
