# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Train the QwenImage diffusion transformer with experimental Megatron-FSDP.

This example intentionally focuses on the large diffusion transformer. It uses
synthetic packed latents and text embeddings so users can validate distributed
model and optimizer sharding before connecting a VAE, text encoder, and dataset.
"""

import argparse
import logging
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
    fully_shard_optimizer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="QwenImage transformer training with experimental Megatron-FSDP."
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen-Image",
        help="Hugging Face model ID or local QwenImage checkpoint directory.",
    )
    parser.add_argument("--revision", default=None, help="Optional Hugging Face model revision.")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load only from the local Hugging Face cache or a local model path.",
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--latent-height",
        type=int,
        default=32,
        help="Synthetic VAE latent height; must be divisible by two.",
    )
    parser.add_argument(
        "--latent-width",
        type=int,
        default=32,
        help="Synthetic VAE latent width; must be divisible by two.",
    )
    parser.add_argument("--text-sequence-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable Diffusers gradient checkpointing before applying FSDP.",
    )
    parser.add_argument(
        "--use-symm-mem",
        action="store_true",
        help="Use the experimental NCCL symmetric-memory communication path.",
    )
    return parser.parse_args()


def flat_dp_placements() -> Placements:
    """Return one-dimensional ZeRO-3-style placements."""
    return Placements(dp_axes=["dp"], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])


def fully_shard_qwen_image_transformer(
    transformer: nn.Module,
    *,
    mesh: DeviceMesh,
    placements: Placements,
    mixed_precision_policy: MixedPrecisionPolicy | None = None,
    use_symm_mem: bool = False,
) -> None:
    """Apply experimental Megatron-FSDP to a QwenImage transformer.

    QwenImage stores its repeated DiT layers in ``transformer_blocks``. Sharding
    those blocks bottom-up creates independently prefetchable FSDP units. Sharding
    the transformer last makes them share one root context and assigns the input,
    output, timestep, and normalization parameters to the root unit.

    Args:
        transformer: QwenImage transformer or a compatible module with a
            ``transformer_blocks`` sequence.
        mesh: One-dimensional data-parallel device mesh.
        placements: Parameter, gradient, and optimizer placements.
        mixed_precision_policy: Optional main-weight and main-gradient precision.
        use_symm_mem: Whether communication staging uses NCCL symmetric memory.
    """
    transformer_blocks = getattr(transformer, "transformer_blocks", None)
    if not isinstance(transformer_blocks, (nn.ModuleList, nn.Sequential)):
        raise TypeError(
            "Expected a QwenImage-compatible transformer_blocks ModuleList or Sequential."
        )
    if not transformer_blocks:
        raise ValueError("QwenImage transformer_blocks must not be empty.")

    policy = mixed_precision_policy or MixedPrecisionPolicy()
    for block in transformer_blocks:
        fully_shard(
            block,
            mesh=mesh,
            placements=placements,
            mixed_precision_policy=policy,
            use_symm_mem=use_symm_mem,
        )
    fully_shard(
        transformer,
        mesh=mesh,
        placements=placements,
        mixed_precision_policy=policy,
        use_symm_mem=use_symm_mem,
    )


def _load_transformer(args: argparse.Namespace, device: torch.device) -> nn.Module:
    """Load the official Diffusers QwenImage transformer on one rank."""
    try:
        from diffusers import QwenImageTransformer2DModel
    except ImportError as error:
        raise RuntimeError(
            "This example requires a current Diffusers installation. "
            "See examples/megatron_fsdp/README.md for the uv command."
        ) from error

    transformer = QwenImageTransformer2DModel.from_pretrained(
        args.model_id,
        subfolder="transformer",
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    return transformer.to(device=device, dtype=torch.bfloat16)


def _load_transformer_one_local_rank_at_a_time(
    args: argparse.Namespace, device: torch.device
) -> nn.Module:
    """Avoid every rank on one node reading checkpoint shards at the same time."""
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", dist.get_world_size()))
    transformer = None
    for loader_local_rank in range(local_world_size):
        load_error = None
        if local_rank == loader_local_rank:
            try:
                transformer = _load_transformer(args, device)
            except Exception as error:  # pylint: disable=broad-exception-caught
                # Keep every rank in the failure handshake.
                load_error = error

        failed = torch.tensor(int(load_error is not None), device=device, dtype=torch.int32)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            message = f"QwenImage checkpoint loading failed on local rank {loader_local_rank}."
            if load_error is not None:
                raise RuntimeError(message) from load_error
            raise RuntimeError(message)

    assert transformer is not None
    return transformer


def _synthetic_qwen_image_batch(
    transformer: nn.Module,
    *,
    batch_size: int,
    latent_height: int,
    latent_width: int,
    text_sequence_length: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[dict[str, object], torch.Tensor]:
    """Build one packed flow-matching batch matching QwenImage's forward API."""
    if latent_height % 2 or latent_width % 2:
        raise ValueError("QwenImage latent height and width must be divisible by two.")

    grid_height = latent_height // 2
    grid_width = latent_width // 2
    image_sequence_length = grid_height * grid_width
    config = transformer.config
    dtype = next(transformer.parameters()).dtype
    clean = torch.randn(
        batch_size,
        image_sequence_length,
        config.in_channels,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    noise = torch.randn(clean.shape, device=device, dtype=dtype, generator=generator)
    sigma = torch.rand(batch_size, 1, 1, device=device, dtype=torch.float32, generator=generator)
    noisy = (1.0 - sigma.to(dtype)) * clean + sigma.to(dtype) * noise
    prompt_embeds = torch.randn(
        batch_size,
        text_sequence_length,
        config.joint_attention_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    prompt_mask = torch.ones(batch_size, text_sequence_length, device=device, dtype=torch.bool)
    model_inputs: dict[str, object] = {
        "hidden_states": noisy,
        "encoder_hidden_states": prompt_embeds,
        "encoder_hidden_states_mask": prompt_mask,
        "timestep": sigma.flatten(),
        "img_shapes": [(1, grid_height, grid_width)] * batch_size,
        "return_dict": False,
    }
    return model_inputs, noise - clean


def _validate_args(args: argparse.Namespace) -> None:
    """Validate arguments before allocating the model."""
    for name in ("steps", "batch_size", "gradient_accumulation_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least one.")
    if args.text_sequence_length < 1:
        raise ValueError("--text-sequence-length must be at least one.")
    if args.latent_height < 2 or args.latent_width < 2:
        raise ValueError("--latent-height and --latent-width must be at least two.")


def _initialize_distributed() -> tuple[torch.device, DeviceMesh]:
    """Initialize one NCCL process per local CUDA device."""
    required_environment = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    missing = [name for name in required_environment if name not in os.environ]
    if missing:
        raise RuntimeError(
            "Launch this example with torchrun; missing environment variables: "
            + ", ".join(missing)
        )
    if not torch.cuda.is_available():
        raise RuntimeError("QwenImage experimental FSDP training requires CUDA.")

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl")
    mesh = init_device_mesh(device.type, (dist.get_world_size(),), mesh_dim_names=("dp",))
    return device, mesh


def _mean_across_ranks(value: torch.Tensor) -> torch.Tensor:
    """Return a detached scalar averaged across data-parallel ranks."""
    mean = value.detach().float()
    dist.all_reduce(mean, op=dist.ReduceOp.SUM)
    return mean / dist.get_world_size()


def train(args: argparse.Namespace) -> None:
    """Run synthetic QwenImage flow-matching training."""
    _validate_args(args)
    device, mesh = _initialize_distributed()
    rank = dist.get_rank()
    try:
        torch.manual_seed(args.seed)
        transformer = _load_transformer_one_local_rank_at_a_time(args, device)
        transformer.train()
        fully_shard_qwen_image_transformer(
            transformer,
            mesh=mesh,
            placements=flat_dp_placements(),
            mixed_precision_policy=MixedPrecisionPolicy(
                main_params_dtype=torch.float32, main_grads_dtype=torch.bfloat16
            ),
            use_symm_mem=args.use_symm_mem,
        )

        optimizer = torch.optim.AdamW(
            transformer.parameters(), lr=args.learning_rate, foreach=False
        )
        fully_shard_optimizer(optimizer)
        generator = torch.Generator(device=device).manual_seed(args.seed + rank)

        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            step_loss = torch.zeros((), device=device, dtype=torch.float32)
            for _ in range(args.gradient_accumulation_steps):
                model_inputs, target = _synthetic_qwen_image_batch(
                    transformer,
                    batch_size=args.batch_size,
                    latent_height=args.latent_height,
                    latent_width=args.latent_width,
                    text_sequence_length=args.text_sequence_length,
                    device=device,
                    generator=generator,
                )
                prediction = transformer(**model_inputs)[0]
                if prediction.shape != target.shape:
                    raise RuntimeError(
                        f"QwenImage prediction shape {prediction.shape} "
                        f"does not match flow target {target.shape}."
                    )
                loss = torch.nn.functional.mse_loss(prediction.float(), target.float())
                (loss / args.gradient_accumulation_steps).backward()
                step_loss.add_(loss.detach())

            optimizer.step()
            mean_loss = _mean_across_ranks(step_loss / args.gradient_accumulation_steps)
            if rank == 0:
                logger.info("step=%d loss=%.6f", step + 1, mean_loss.item())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    """Run the QwenImage experimental FSDP example."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(parse_args())


if __name__ == "__main__":
    main()
