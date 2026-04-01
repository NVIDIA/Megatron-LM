import argparse
import os
from pathlib import Path
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.distributed as dist

import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor


# -----------------------
# Model definitions
# -----------------------

class ToyBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


class ToyModel(nn.Module):
    def __init__(self, dim: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(ToyBlock(dim) for _ in range(n_layers))
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out(x)


# -----------------------
# Distributed init / mesh
# -----------------------

def init_distributed() -> torch.distributed.device_mesh.DeviceMesh:
    """Initialize process group and device mesh."""
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    mesh = init_device_mesh("cuda", mesh_shape=(world_size,))
    return mesh


def build_fsdp_model(
    dim: int,
    n_layers: int,
    use_megatron_fsdp: bool,
) -> Tuple["FSDPModule", torch.distributed.device_mesh.DeviceMesh]:
    if use_megatron_fsdp:
        from megatron.core.distributed.fsdp.src.megatron_fsdp import fully_shard_v2 as fully_shard, FSDPModule
    else:
        from torch.distributed.fsdp import fully_shard, FSDPModule

    mesh = init_distributed()
    model = ToyModel(dim=dim, n_layers=n_layers).to("cuda")

    # Example: per-layer sharding
    for layer in model.layers:
        fully_shard(layer, mesh=mesh)

    # Optionally shard the root as well
    fully_shard(model, mesh=mesh)

    assert isinstance(model, ToyModel)
    assert isinstance(model, FSDPModule)

    p = next(model.parameters())
    assert isinstance(p, DTensor)
    return model, mesh


# -----------------------
# Checkpoint helpers
# -----------------------

class AppState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.

    Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
    and optimizer.
    """

    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict
        }

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"]
        )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    ckpt_dir: str,
) -> None:
    rank = dist.get_rank()
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    ckpt_step_dir = os.path.join(ckpt_dir, f"step_{step:06d}")

    state = {"app": AppState(model, optimizer), "step": step}
    dcp.save(state_dict=state, checkpoint_id=ckpt_step_dir)

    if rank == 0:
        print(f"[rank0] Saved checkpoint to {ckpt_dir}")


def load_checkpoint_if_available(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ckpt_dir: str,
) -> int:
    """
    Load the latest checkpoint if present.
    Returns starting step (step+1) for training.
    """
    if not os.path.exists(ckpt_dir):
        return 0

    all_ckpts = sorted(
        [f for f in os.listdir(ckpt_dir) if f.startswith("step_")]
    )
    last_ckpt = os.path.join(ckpt_dir, all_ckpts[-1]) if all_ckpts else None

    if last_ckpt is None:
        return 0

    step = torch.zeros([1])
    state = {"app": AppState(model, optimizer), "step": step}
    dcp.load(state_dict=state, checkpoint_id=last_ckpt)

    return int(step.item()) + 1


# -----------------------
# Training loop
# -----------------------

def train(
    args: argparse.Namespace,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    start_step: int = 0,
) -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    model.train()
    step = start_step

    for epoch in range(args.epochs):
        for _ in range(args.steps_per_epoch):
            # Dummy data
            x = torch.randn(args.batch_size, args.model_dim, device="cuda")
            y = model(x)
            loss = y.sum() / (world_size * args.batch_size)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % args.log_interval == 0 and rank == 0:
                print(f"[rank0] epoch={epoch} step={step} loss={loss.item():.4f}")

            if args.ckpt_dir and step % args.ckpt_interval == 0 and step > 0:
                save_checkpoint(model, optimizer, step, args.ckpt_dir)

            step += 1

    # Final checkpoint
    if args.ckpt_dir:
        save_checkpoint(model, optimizer, step, args.ckpt_dir)


# -----------------------
# __main__ entry
# -----------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy FSDP2 training example")
    parser.add_argument("--model-dim", type=int, default=1024)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--steps-per-epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--ckpt-interval", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--use-megatron-fsdp", action="store_true", help="Use Megatron-FSDP instead of PyTorch FSDP2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model, _ = build_fsdp_model(
        dim=args.model_dim,
        n_layers=args.n_layers,
        use_megatron_fsdp=args.use_megatron_fsdp,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_step = 0
    if args.ckpt_dir:
        start_step = load_checkpoint_if_available(model, optimizer, args.ckpt_dir)

    train(args, model, optimizer, start_step=start_step)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
