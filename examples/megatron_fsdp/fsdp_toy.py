# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

# -----------------------
# Model definitions
# -----------------------

class ToyBlock(nn.Module):
    """MLP-style block with expand→project, similar to transformer FFN."""

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden_dim = dim * expansion
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self._use_activation_checkpointing = False

    def forward(self, x):
        if self._use_activation_checkpointing:
            return torch.utils.checkpoint.checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)

    def _forward_impl(self, x):
        return self.dropout(self.down(torch.nn.functional.gelu(self.gate(x)) * self.up(x)))


class ToyModel(nn.Module):
    def __init__(self, dim: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(ToyBlock(dim) for _ in range(n_layers))
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out(x)

    def enable_activation_checkpointing(self):
        for layer in self.layers:
            layer._use_activation_checkpointing = True


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
    use_activation_checkpointing: bool = False,
    enable_cuda_graph: bool = True,
    enable_trace_pool: bool = False,
) -> Tuple["FSDPModule", torch.distributed.device_mesh.DeviceMesh]:
    if use_megatron_fsdp:
        try:
            from megatron_fsdp.v2 import FSDPModule, fully_shard
        except (ImportError, ModuleNotFoundError) as err:
            from megatron.core.distributed.fsdp.src.megatron_fsdp.v2 import FSDPModule, fully_shard
    else:
        from torch.distributed.fsdp import FSDPModule, fully_shard

    mesh = init_distributed()
    model = ToyModel(dim=dim, n_layers=n_layers).to(device="cuda", dtype=torch.bfloat16)

    if use_activation_checkpointing:
        model.enable_activation_checkpointing()

    if use_megatron_fsdp:
        sublayer_kwargs = dict(
            enable_cuda_graph=enable_cuda_graph,
            enable_trace_pool=enable_trace_pool,
        )
        kwargs = dict(
            enable_trace_pool=enable_trace_pool,
        )
    else:
        sublayer_kwargs = {}
        kwargs = {}

    for layer in model.layers:
        fully_shard(layer, mesh=mesh, **sublayer_kwargs)
    fully_shard(model, mesh=mesh, **kwargs)

    assert isinstance(model, ToyModel)
    assert isinstance(model, FSDPModule)

    if use_megatron_fsdp:
        model._log_parameter_groups()

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
        try:
            from megatron_fsdp.uneven_dtensor import get_state_dict
        except ImportError:
            from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
                get_state_dict,
            )

        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict,
        }

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    ckpt_dir: str,
) -> None:
    rank = dist.get_rank()
    if rank == 0:
        print(f"[rank0] Saving checkpoint step={step} ...")
    t0 = time.time()

    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    ckpt_step_dir = os.path.join(ckpt_dir, f"step_{step:06d}")

    state = {"app": AppState(model, optimizer), "step": step}
    dcp.save(state_dict=state, checkpoint_id=ckpt_step_dir)

    if rank == 0:
        print(f"[rank0] Saved checkpoint to {ckpt_dir} ({time.time() - t0:.1f}s)")


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

    rank = dist.get_rank()
    if rank == 0:
        print(f"[rank0] Loading checkpoint from {last_ckpt} ...")
    t0 = time.time()

    step = torch.zeros([1])
    state = {"app": AppState(model, optimizer), "step": step}
    dcp.load(state_dict=state, checkpoint_id=last_ckpt)

    if rank == 0:
        print(f"[rank0] Loaded checkpoint ({time.time() - t0:.1f}s)")

    return int(step.item()) + 1


# -----------------------
# Training loop
# -----------------------

def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def train(
    args: argparse.Namespace,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    start_step: int = 0,
) -> None:
    rank = dist.get_rank()

    model.train()
    step = start_step
    t_start = time.time()

    for epoch in range(args.epochs):
        for _ in range(args.steps_per_epoch):
            # Dummy data
            x = torch.randn(args.batch_size, args.seq_len, args.model_dim,
                            device="cuda", dtype=torch.bfloat16)
            y = model(x)
            loss = y.sum() / (args.batch_size * args.seq_len)
            loss.backward()
            if args.use_megatron_fsdp and args.release_memory_pool:
                model.release_memory_pool()
            optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()

            if step % args.log_interval == 0 and rank == 0:
                t_now = time.time()
                elapsed = t_now - t_start
                it_s = elapsed / max(step - start_step + 1, 1)
                alloc = _fmt_bytes(torch.cuda.memory_allocated())
                max_reserved = _fmt_bytes(torch.cuda.max_memory_reserved())
                print(
                    f"[rank0] epoch={epoch} step={step} loss={loss.item():.4f} "
                    f"alloc={alloc} max_reserved={max_reserved} "
                    f"it={elapsed:.1f}s ({it_s * 1000:.0f}ms/it)"
                )

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
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Sequence length (larger = more compute vs communication)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--steps-per-epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ckpt-dir", type=str, default=None,
                        help="Directory for DCP checkpoints (omit to skip)")
    parser.add_argument("--ckpt-interval", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--use-megatron-fsdp", action="store_true", help="Use Megatron-FSDP instead of PyTorch FSDP2")
    parser.add_argument("--activation-checkpoint", action="store_true", help="Enable activation checkpointing on transformer layers")
    parser.add_argument("--no-cuda-graph", action="store_false", dest="cuda_graph",
                        default=True, help="Disable CUDA graph capture")
    parser.add_argument("--use-trace-pool", action="store_true", default=False,
                        help="Use TracePoolAllocator for stable buffer addresses")
    parser.add_argument("--release-memory-pool", action="store_true", default=False,
                        help="Call FSDPModule.release_memory_pool() after each backward "
                        "to release allocator slot tensors and CUDA graphs. "
                        "Only takes effect with --use-megatron-fsdp.")
    parser.add_argument("--record-memory-history", type=str, default=None, metavar="DIR",
                        help="Enable CUDA memory recording, dump snapshot to this directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(
            max_entries=100000,
            stacks="all",
        )

    model, _ = build_fsdp_model(
        dim=args.model_dim,
        n_layers=args.n_layers,
        use_megatron_fsdp=args.use_megatron_fsdp,
        use_activation_checkpointing=args.activation_checkpoint,
        enable_cuda_graph=args.cuda_graph,
        enable_trace_pool=args.use_trace_pool,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_step = 0
    if args.ckpt_dir:
        start_step = load_checkpoint_if_available(model, optimizer, args.ckpt_dir)

    train(args, model, optimizer, start_step=start_step)

    if args.record_memory_history:
        rank = dist.get_rank()
        out_dir = args.record_memory_history
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        snapshot_path = os.path.join(out_dir, f"memory_snapshot_rank{rank}.pickle")
        torch.cuda.memory._dump_snapshot(snapshot_path)
        if rank == 0:
            print(f"[rank0] Memory snapshot dumped to {snapshot_path}")
        torch.cuda.memory._record_memory_history(enabled=None)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
