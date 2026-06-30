"""
Single-file end-to-end training test for QwenImageTransformer2DModel
with Megatron FSDP v1/v2 and PyTorch FSDP1.

Self-contained — requires only stock packages (`torch`, `diffusers`,
`megatron-fsdp`).  The transformer body is the upstream diffusers model.

Features:
  - FSDP1 / Megatron FSDP v1 / v2 backends
  - Per-block torch.compile
  - CUDA graph capture (v2 with --cuda-graph)
  - TracePoolAllocator (v2 with --trace-pool)
  - Gradient checkpointing (--gradient_checkpointing)
  - FA2 / FA3 / native attention backends
  - Memory history OOM dump (--record-memory-history)
  - Per-step GPU memory tracking
  - NVML real GPU memory logging
  - Numerical verification probe (--verify)
  - Nsight profiling support (--cuda_profiler_capture)

Usage (per node):
  torchrun --nnodes=$NNODES --node_rank=$NODE_RANK \
           --nproc_per_node=8 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
           test_qwenimage.py \
           --backend {fsdp1|mfsdp|mfsdpv2} \
           --sharding {full|hybrid} \
           --batch_size 4 --height 512 --width 512 \
           --bench_steps 20 --warmup_steps 3 \
           --compile --cuda-graph --trace-pool

Reports per-step latency, peak GPU memory, and per-step memory summary.
`--verify` probes global loss + grad-norm for numerical agreement across
backends (adds sync overhead — only relative deltas are meaningful).

Note: mfsdpv2 uses a 1D device mesh (no HSDP).  `--sharding hybrid` with
mfsdpv2 falls back to full sharding on the 1D mesh.
"""
import argparse
import atexit
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import time
import traceback

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from diffusers.models.transformers.transformer_qwenimage import (
    QwenImageTransformer2DModel,
    QwenImageTransformerBlock,
)
from diffusers.models.attention_dispatch import attention_backend

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)



# ---------- FA3 training shim ------------------------------------------------
# diffusers 0.37.x registers `_flash_3` via a custom-op that only has
# forward + fake — there is no autograd registration (see the TODO in
# diffusers/models/attention_dispatch.py pointing at the unmerged
# Dao-AILab/flash-attention#1590), so training trips
#   "Trying to backward through _diffusers_flash_attn_3._flash_attn_forward.default
#    but no autograd formula was registered."
# `flash_attn_interface.flash_attn_func` from the FA3 wheel itself IS
# autograd-aware. Patching the registry entry to call it directly is the
# smallest change that makes `attention_backend("_flash_3")` usable in training,
# matching the FA3 path our internal fork (model/qwen_image/modules/attention_utils.py)
# already relies on for H800 training.
def _enable_fa3_training_patch():
    import flash_attn_interface
    from diffusers.models.attention_dispatch import (
        _AttentionBackendRegistry, AttentionBackendName,
    )

    def _flash_attention_3_training(
        query, key, value, attn_mask=None, scale=None,
        is_causal=False, return_lse=False, _parallel_config=None,
    ):
        if attn_mask is not None:
            raise ValueError("`attn_mask` is not supported for flash-attn 3.")
        out = flash_attn_interface.flash_attn_func(
            q=query, k=key, v=value,
            softmax_scale=scale, causal=is_causal,
            return_attn_probs=return_lse,
        )
        if return_lse:
            o, lse, *_ = out
            return o, lse.permute(0, 2, 1)
        return out[0] if isinstance(out, tuple) else out

    _AttentionBackendRegistry._backends[AttentionBackendName._FLASH_3] = (
        _flash_attention_3_training
    )
    _AttentionBackendRegistry._supported_arg_names[AttentionBackendName._FLASH_3] = (
        set(_flash_attention_3_training.__code__.co_varnames[
            :_flash_attention_3_training.__code__.co_argcount])
    )


# ----------------------------- nvtx -----------------------------
def nvtx_range_start(name):
    if hasattr(torch.cuda.nvtx, "range_start"):
        return torch.cuda.nvtx.range_start(name)
    torch.cuda.nvtx.range_push(name)
    return None


def nvtx_range_end(handle):
    if handle is None:
        torch.cuda.nvtx.range_pop()
    else:
        torch.cuda.nvtx.range_end(handle)


@contextmanager
def nvtx_range(name):
    handle = nvtx_range_start(name)
    try:
        yield
    finally:
        nvtx_range_end(handle)


_fwd_nvtx_handles = []
_bwd_nvtx_handles = []


class _NvtxFwdStartBwdEnd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        _fwd_nvtx_handles.append(nvtx_range_start("fwd"))
        return x

    @staticmethod
    def backward(ctx, grad):
        if _bwd_nvtx_handles:
            nvtx_range_end(_bwd_nvtx_handles.pop())
        return grad


class _NvtxFwdEndBwdStart(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if _fwd_nvtx_handles:
            nvtx_range_end(_fwd_nvtx_handles.pop())
        return x

    @staticmethod
    def backward(ctx, grad):
        _bwd_nvtx_handles.append(nvtx_range_start("bwd"))
        return grad


def _looks_like_vit_module(name, module):
    lowered_name = name.lower()
    lowered_class = module.__class__.__name__.lower()
    compact_name = lowered_name.replace(".", "_")
    compact_class = lowered_class.replace("_", "")
    tokens = set(compact_name.split("_"))
    return (
        "vit" in tokens
        or "visiontransformer" in compact_class
        or "visiontransformer" in compact_name.replace("_", "")
        or "vision_model" in lowered_name
        or "vision_tower" in lowered_name
        or "image_encoder" in lowered_name
        or lowered_name.endswith("visual")
    )


def install_vit_nvtx_tag(model):
    for name, module in model.named_modules():
        if not name or not _looks_like_vit_module(name, module):
            continue
        original_forward = module.forward

        def tagged_forward(*args, _original_forward=original_forward, **kwargs):
            with nvtx_range("vit_fwd"):
                return _original_forward(*args, **kwargs)

        module.forward = tagged_forward
        return name
    return None


# ------------------------ memory history / OOM dump ------------------------
class MemoryHistoryManager:
    """Records CUDA memory history and dumps snapshots on OOM and/or exit."""

    def __init__(self, out_dir: str, oom_only: bool = False):
        self._out_dir = out_dir
        self._oom_only = oom_only
        self._dumped = False

    def start(self):
        Path(self._out_dir).mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._record_memory_history(
            max_entries=200000, stacks="all",
        )
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:
            print(f"[rank0] Memory history recording enabled, "
                  f"dump dir={self._out_dir} oom_only={self._oom_only}")

    def dump(self, tag: str = ""):
        if self._dumped:
            return
        self._dumped = True
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        suffix = f"_{tag}" if tag else ""
        path = os.path.join(self._out_dir, f"memory_snapshot_rank{rank}{suffix}.pickle")
        try:
            torch.cuda.memory._dump_snapshot(path)
            if rank == 0:
                print(f"[rank0] Memory snapshot dumped: {path}")
        except Exception as e:
            if rank == 0:
                print(f"[rank0] Memory snapshot dump failed: {e}")

    def stop(self):
        torch.cuda.memory._record_memory_history(enabled=None)

    def dump_on_normal_exit(self):
        if self._oom_only or self._dumped:
            return
        self.dump()

    def dump_on_oom(self):
        self.dump(tag="OOM")


# ------------------------------------------------------------------
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True, choices=["fsdp1", "mfsdp", "mfsdpv2"])
    p.add_argument("--sharding", default="hybrid", choices=["full", "hybrid"])
    p.add_argument("--pretrained_model_name_or_path", required=True,
                   help="HF repo or local dir containing the QwenImage transformer.")
    p.add_argument("--subfolder", default="transformer")
    p.add_argument("--attention", default="_flash_3",
                   help="diffusers attention backend name (e.g. _flash_3, flash, native).")
    p.add_argument("--mixed_precision", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--instruction_seq_len", type=int, default=64)
    p.add_argument("--vl_patch_size", type=int, default=14)
    p.add_argument("--vl_merge_size", type=int, default=2)
    p.add_argument("--bench_steps", type=int, default=20)
    p.add_argument("--warmup_steps", type=int, default=3)
    p.add_argument("--num_gpus_per_node", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--compile", action="store_true",
                   help="Per-block torch.compile on transformer_blocks.")
    p.add_argument("--trace-pool", action="store_true",
                   help="Use TracePoolAllocator for stable buffer addresses (mfsdpv2 only).")
    p.add_argument("--cuda-graph", action="store_true",
                   help="Enable CUDA graph capture on leaf FSDP modules (mfsdpv2 only).")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--verify", action="store_true",
                   help="Cross-rank gloss + global grad-norm probe; timing INVALID for this run.")
    p.add_argument("--cuda_profiler_capture", action="store_true",
                   help="Bracket bench steps with cudaProfilerStart/Stop for Nsight capture ranges.")
    p.add_argument("--record-memory-history", type=str, default=None, metavar="DIR",
                   help="Enable CUDA memory recording (max_entries=200000). "
                        "Dumps a snapshot to DIR on normal exit AND on OOM. "
                        "Files: memory_snapshot_rank{N}.pickle. "
                        "Loadable at https://pytorch.org/memory_viz.")
    p.add_argument("--record-memory-history-oom-only", action="store_true",
                   help="When set with --record-memory-history, only dump the "
                        "snapshot on OOM (skip normal exit dump).")
    return p.parse_args()


# ------------------------ FSDP wrap helpers ------------------------
def wrap_fsdp1(model, world_size, num_gpus_per_node, dtype, sharding, local_rank):
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy, MixedPrecision, BackwardPrefetch,
    )
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    use_hsdp = sharding == "hybrid" and (world_size // num_gpus_per_node) > 1
    if use_hsdp:
        num_nodes = world_size // num_gpus_per_node
        mesh = init_device_mesh("cuda", (num_nodes, num_gpus_per_node),
                                mesh_dim_names=("replicate", "shard"))
        strat = ShardingStrategy.HYBRID_SHARD
    else:
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("shard",))
        strat = ShardingStrategy.FULL_SHARD

    mp = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
    return FSDP(
        model, device_mesh=mesh, sharding_strategy=strat, mixed_precision=mp,
        auto_wrap_policy=ModuleWrapPolicy({QwenImageTransformerBlock}),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE, forward_prefetch=True,
        use_orig_params=True, device_id=local_rank, limit_all_gathers=True,
    )


def wrap_mfsdp(model, world_size, num_gpus_per_node, dtype, sharding,
               sync_each_microbatch):
    from megatron_fsdp import fully_shard_model, MixedPrecisionPolicy

    use_hsdp = sharding == "hybrid" and (world_size // num_gpus_per_node) > 1
    if use_hsdp:
        num_nodes = world_size // num_gpus_per_node
        mesh = init_device_mesh("cuda", (num_nodes, num_gpus_per_node),
                                mesh_dim_names=("dp_outer", "dp_shard"))
        sub = mesh[("dp_outer", "dp_shard")]
        (sub._flatten if hasattr(sub, "_flatten") else sub.flatten)("hsdp")
        hsdp_kwargs = dict(dp_outer_dim="dp_outer",
                           hybrid_fsdp_group=mesh["hsdp"].get_group(),
                           outer_dp_sharding_strategy=0)   # REPLICATE outer
    else:
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))
        hsdp_kwargs = {}

    mp = MixedPrecisionPolicy(main_params_dtype=dtype, main_grads_dtype=dtype,
                              grad_comm_dtype=dtype)
    return fully_shard_model(
        module=model,
        fsdp_unit_modules=[QwenImageTransformerBlock],
        device_mesh=mesh, dp_shard_dim="dp_shard",
        zero_dp_strategy=3,
        sync_model_each_microbatch=sync_each_microbatch,
        overlap_grad_reduce=True, overlap_param_gather=True,
        mixed_precision_policy=mp, **hsdp_kwargs,
    )


def wrap_mfsdpv2(model, world_size, num_gpus_per_node, dtype, sharding,
                 enable_trace_pool=False, enable_cuda_graph=False):
    from megatron.core.distributed.fsdp.src.megatron_fsdp.v2 import (
        fully_shard, MixedPrecisionPolicy,
    )

    use_hsdp = sharding == "hybrid" and (world_size // num_gpus_per_node) > 1
    if use_hsdp:
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))
        shard_strategy = "optim_grads_params"
        if dist.get_rank() == 0:
            print("[mfsdpv2] HSDP not supported; falling back to full sharding on 1D mesh")
    else:
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))
        shard_strategy = "optim_grads_params"

    mp = MixedPrecisionPolicy(main_params_dtype=dtype, main_grads_dtype=dtype,
                              grad_comm_dtype=dtype)

    for blk in model.transformer_blocks:
        fully_shard(blk, mesh=mesh, mp_policy=mp,
                    sharding_strategy=shard_strategy,
                    enable_unshard_prefetch=True,
                    enable_async_reduce_grad=True,
                    enable_trace_pool=enable_trace_pool,
                    enable_cuda_graph=enable_cuda_graph)

    fully_shard(model, mesh=mesh, mp_policy=mp,
                sharding_strategy=shard_strategy,
                enable_unshard_prefetch=True,
                enable_async_reduce_grad=True,
                enable_trace_pool=enable_trace_pool)

    return model


# ------------------------ verify probe ------------------------
def _local_grad(p):
    """Read this param's local-shard gradient across both backends.

    fsdp1 (use_orig_params): p.grad is the local flat shard.
    mfsdp: grad lives in an internal param_and_grad_buffer; p.grad is None.
           Try .get_main_grad() -> .main_grad -> .decoupled_grad.
    """
    g = p.grad
    if g is None:
        getter = getattr(p, "get_main_grad", None)
        if getter is not None:
            try: g = getter()
            except Exception: g = None
    if g is None: g = getattr(p, "main_grad", None)
    if g is None: g = getattr(p, "decoupled_grad", None)
    if g is None: return None
    if hasattr(g, "to_local"): g = g.to_local()
    return g


@torch.no_grad()
def verify_stats(model, loss, device):
    """Global gloss (mean over ranks) + global grad-norm (over all shards/ranks)."""
    torch.cuda.synchronize()  # mfsdp reduce-scatters on a side stream
    sumsq = torch.zeros((), device=device, dtype=torch.float32)
    n = 0
    for p in model.parameters():
        if not p.requires_grad: continue
        g = _local_grad(p)
        if g is None: continue
        sumsq += g.detach().float().pow(2).sum()
        n += 1
    dist.all_reduce(sumsq, op=dist.ReduceOp.SUM)
    gloss = loss.detach().float().clone()
    dist.all_reduce(gloss, op=dist.ReduceOp.SUM)
    return (gloss / dist.get_world_size()).item(), sumsq.sqrt().item(), n


# ------------------------ synthetic batch ------------------------
def qwen25vl_vision_tokens(h, w, patch=14, merge=2):
    f = patch * merge
    return max(1, round(h/f)) * max(1, round(w/f))


def make_batch(args, device, dtype, gen):
    B = args.batch_size
    H, W = args.height // 16, args.width // 16   # 16x packing (VAE 8x + 2x2 patchify)
    seq = H * W
    txt_len = args.instruction_seq_len + qwen25vl_vision_tokens(
        args.height, args.width, args.vl_patch_size, args.vl_merge_size)
    hidden = torch.randn(B, seq, 64, device=device, dtype=dtype, generator=gen)
    timestep = torch.rand(B, device=device, dtype=dtype, generator=gen) * 1000
    prompt = torch.randn(B, txt_len, 3584, device=device, dtype=dtype, generator=gen)
    img_shapes = [[(1, H, W)]] * B
    # No padding in synthetic batch → omit attention mask so FA3 path is reachable.
    return hidden, timestep, prompt, img_shapes, [txt_len] * B


# ----------------------------- main -----------------------------
def main():
    args = parse()
    if args.attention == "_flash_3":
        _enable_fa3_training_patch()
    rank, local_rank, world = (int(os.environ[k]) for k in
                               ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32

    if rank == 0:
        txt_len = args.instruction_seq_len + qwen25vl_vision_tokens(
            args.height, args.width, args.vl_patch_size, args.vl_merge_size)
        print(f"[{args.backend}] world={world} dtype={dtype} bs={args.batch_size} "
              f"img={args.height}x{args.width} txt={txt_len} "
              f"sharding={args.sharding} compile={args.compile} gc={args.gradient_checkpointing}")

    model = QwenImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder=args.subfolder, torch_dtype=dtype,
    ).to(dtype)
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    model.train()

    if args.backend == "fsdp1":
        model = wrap_fsdp1(model, world, args.num_gpus_per_node, dtype,
                           args.sharding, local_rank)
    elif args.backend == "mfsdpv2":
        model = wrap_mfsdpv2(model, world, args.num_gpus_per_node, dtype,
                             args.sharding, enable_trace_pool=args.trace_pool,
                             enable_cuda_graph=args.cuda_graph)
    else:
        # verify needs grads finished before optimizer.step(); benchmark path keeps overlap
        model = wrap_mfsdp(model, world, args.num_gpus_per_node, dtype,
                           args.sharding, sync_each_microbatch=True)

    if args.compile:
        inner = model.module if hasattr(model, "module") else model
        for blk in inner.transformer_blocks:
            blk.compile()

    tagged_vit = install_vit_nvtx_tag(model)
    if rank == 0:
        if tagged_vit is not None:
            print(f"[nvtx] vit_fwd tagged module: {tagged_vit}")
        else:
            print("[nvtx] no ViT-like module found; vit_fwd tag skipped")

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=args.lr, fused=True)
    if args.backend == "mfsdp":
        from megatron_fsdp import fully_shard_optimizer
        fully_shard_optimizer(optim)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # ---- memory history recording ----
    mem_mgr = None
    if args.record_memory_history:
        mem_mgr = MemoryHistoryManager(
            out_dir=args.record_memory_history,
            oom_only=args.record_memory_history_oom_only,
        )
        mem_mgr.start()
        # Dump on normal exit via atexit (unless oom_only)
        if not args.record_memory_history_oom_only:
            atexit.register(mem_mgr.dump_on_normal_exit)

    _mem_log("after_model_init", rank=rank)

    gen = torch.Generator(device=device).manual_seed(args.seed + rank)
    step_times = []
    cuda_profiler_active = False
    oom_occurred = False
    with attention_backend(args.attention):
        try:
            for step in range(args.warmup_steps + args.bench_steps):
                if args.cuda_profiler_capture and step == args.warmup_steps:
                    torch.cuda.synchronize()
                    dist.barrier()
                    torch.cuda.profiler.start()
                    cuda_profiler_active = True
                    if rank == 0:
                        print("[nsys] cudaProfilerStart")
                hs, ts, pe, img_shapes, txt_lens = make_batch(args, device, dtype, gen)
                hs.requires_grad_(True)
                torch.cuda.synchronize(); dist.barrier()
                t0 = time.perf_counter()

                hs = _NvtxFwdStartBwdEnd.apply(hs)
                t_fwd = time.perf_counter()
                with torch.amp.autocast("cuda", dtype=dtype):
                    out = model(hidden_states=hs, timestep=ts / 1000,
                                encoder_hidden_states=pe,
                                img_shapes=img_shapes, txt_seq_lens=txt_lens, return_dict=False)
                    pred = (out[0] if isinstance(out, tuple) else out)[:, :hs.size(1)]
                    pred = _NvtxFwdEndBwdStart.apply(pred)
                loss = pred.float().pow(2).mean()
                loss.backward()
                t_bwd = time.perf_counter()

                if step < args.warmup_steps:
                    _mem_log(f"step={step} fwd_bwd fwd_ms={((t_bwd - t_fwd) * 1000):.1f} "
                             f"bwd_ms={((time.perf_counter() - t_bwd) * 1000):.1f}", rank=rank)

                if args.verify:
                    gloss, gnorm, n = verify_stats(model, loss, device)

                with nvtx_range("optimizer"):
                    if args.backend == "mfsdp":
                        optim.step(sync_grad_before_optimizer_step=True,
                                   install_optimized_model_weights=True)
                        optim.zero_grad(set_to_none=True, zero_grad_buffer=True)
                    elif args.backend == "mfsdpv2":
                        optim.step()
                        optim.zero_grad(set_to_none=True)
                    else:
                        optim.step(); optim.zero_grad(set_to_none=True)

                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                if step >= args.warmup_steps:
                    step_times.append(dt)
                if rank == 0:
                    tag = "warmup" if step < args.warmup_steps else "bench "
                    if args.verify:
                        print(f"[{args.backend}] {tag} step {step:3d} | VERIFY (timing invalid) | "
                              f"gloss={gloss:.6f} | gnorm={gnorm:.4f} | n_grad={n}")
                    else:
                        print(f"[{args.backend}] {tag} step {step:3d} | {dt*1000:8.2f} ms | "
                              f"loss={loss.item():.4f}")

        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            oom_occurred = True
            if rank == 0:
                print(f"\n[rank0] OOM / CUDA error at step {step}: {exc}")
                traceback.print_exc()
            if mem_mgr is not None:
                mem_mgr.dump_on_oom()
            raise

    if cuda_profiler_active:
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.profiler.stop()
        if rank == 0:
            print("[nsys] cudaProfilerStop")

    peak = torch.tensor([torch.cuda.max_memory_allocated() / 1e9], device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    if rank == 0:
        avg = sum(step_times) / max(1, len(step_times)) * 1000
        print(f"\n[{args.backend}] avg step (n={len(step_times)}): {avg:.2f} ms | "
              f"peak mem: {peak.item():.2f} GB")

    _mem_log("final", rank=rank)

    if mem_mgr is not None:
        mem_mgr.dump_on_normal_exit()
        mem_mgr.stop()

    dist.destroy_process_group()


def _fmt_bytes(n: int) -> str:
    for power, suffix in [(4, "TB"), (3, "GB"), (2, "MB"), (1, "KB"), (0, "B")]:
        unit = 1024 ** power
        if n >= unit:
            return f"{n / unit:.2f} {suffix}"
    return f"{n} B"


def _mem_log(tag="", rank=None):
    """Log CUDA memory stats for the current rank."""
    if rank is None:
        rank = torch.distributed.get_rank()
    alloc = torch.cuda.memory_allocated()
    max_alloc = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_reserved = torch.cuda.max_memory_reserved()
    prefix = f"[rank{rank}] {tag}" if tag else f"[rank{rank}]"
    print(f"{prefix} alloc={_fmt_bytes(alloc)} max_alloc={_fmt_bytes(max_alloc)} "
          f"reserved={_fmt_bytes(reserved)} max_reserved={_fmt_bytes(max_reserved)}")


if __name__ == "__main__":
    main()
