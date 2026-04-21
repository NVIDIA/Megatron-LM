"""
evaluate_profile.py — drop-in replacement for evaluate.py that adds a memory
and parameter profiling report before running evaluation.

After the model is loaded and flextron hooks are applied, it:
  1. Counts actual parameters (numel + bytes) per layer type via model.named_parameters(),
     correctly accounting for TP sharding and EP expert distribution so that the reported
     count equals the TOTAL model parameters across all ranks.
  2. Allocates KV and SSM caches and measures their actual dtype and size in bytes.
     SSM cache only walks MambaMixer (not MambaLayer, which is a passthrough wrapper)
     to avoid double-counting.
  3. Runs get_num_parameters() and get_memory_footprint() with the same config.
  4. Prints a side-by-side comparison table on rank 0.

Only rank 0 prints; all other ranks are silent for the profiling section.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))

import torch
import torch.distributed as dist

from megatron.training import get_args
from megatron.elastification.arguments import add_flextron_args
from megatron.training.initialize import initialize_megatron
from megatron.training import print_rank_0
from megatron.training.checkpointing import load_checkpoint
from megatron.core.utils import get_model_config
from megatron.core import parallel_state
from megatron.training import get_model
from megatron.training.training import (
    evaluate_and_print_results,
    update_train_iters,
    build_train_valid_test_data_iterators,
)
from megatron.elastification.router.flex_budget_utils import (
    get_num_parameters,
    get_memory_footprint,
    get_kv_cache_size,
    get_mamba_ssm_cache_size,
    get_max_buffer_size,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def bytes_to_gb(b):
    return b / (1024 ** 3)

def dtype_name(dt):
    return str(dt).replace("torch.", "")


def count_total_params(model_wrapper):
    """
    Count total model parameters correctly in a distributed setting.

    Megatron distributes parameters across ranks in two ways:
      - Tensor Parallel (TP): weight matrices are column/row-sliced.
        ColumnParallelLinear / RowParallelLinear mark their weight with
        `param.tensor_model_parallel = True`.  Each TP rank holds 1/tp_size
        of these weights, so we multiply their numel by tp_size.
      - Expert Parallel (EP): routed expert weights are sharded across EP ranks.
        Each EP rank holds num_experts/ep_size experts.
        We identify these by parameter path: name contains '.mlp.experts'
        but NOT 'shared_expert' (the shared expert is replicated, not sharded).

    Pipeline Parallel (PP) stages each hold a disjoint set of layers, so we
    all_reduce across the PP group to sum contributions from all stages.

    Data Parallel (DP) ranks are identical — we count only once.
    """
    try:
        m = model_wrapper.module
    except AttributeError:
        m = model_wrapper

    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    ep_size = parallel_state.get_expert_model_parallel_world_size()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()

    buckets = {k: {"numel": 0, "bytes": 0} for k in
               ["attn", "mamba", "moe_expert", "moe_shared", "moe_router",
                "emb", "out", "norm", "other"]}

    for name, p in m.named_parameters():
        numel  = p.numel()
        nbytes = numel * p.element_size()

        # ── TP scaling: params marked tensor_model_parallel are sharded ──────
        tp_factor = tp_size if getattr(p, 'tensor_model_parallel', False) else 1
        numel  *= tp_factor
        nbytes *= tp_factor

        # ── EP scaling: routed expert params are sharded across EP ranks ─────
        # Path contains '.mlp.experts' but not 'shared_expert'
        if '.mlp.experts' in name and 'shared_expert' not in name:
            numel  *= ep_size
            nbytes *= ep_size
            bucket = "moe_expert"
        # ── classify by parameter path ────────────────────────────────────────
        elif 'output_layer' in name:
            bucket = "out"
        elif 'embedding' in name:
            bucket = "emb"
        elif 'self_attention' in name or 'linear_qkv' in name or 'linear_proj' in name:
            bucket = "attn"
        elif 'mixer' in name or '.mamba' in name.lower():
            bucket = "mamba"
        elif 'shared_expert' in name:
            bucket = "moe_shared"
        elif 'mlp.router' in name or 'mlp.expert_bias' in name:
            bucket = "moe_router"
        elif 'norm' in name.lower() or 'layernorm' in name.lower():
            bucket = "norm"
        else:
            bucket = "other"

        buckets[bucket]["numel"]  += numel
        buckets[bucket]["bytes"] += nbytes

    # ── sum across PP stages (each holds a disjoint set of layers) ────────────
    if pp_size > 1:
        numel_tensor = torch.tensor(
            [b["numel"] for b in buckets.values()],
            dtype=torch.long, device=torch.cuda.current_device()
        )
        bytes_tensor = torch.tensor(
            [b["bytes"] for b in buckets.values()],
            dtype=torch.long, device=torch.cuda.current_device()
        )
        dist.all_reduce(numel_tensor, op=dist.ReduceOp.SUM,
                        group=parallel_state.get_pipeline_model_parallel_group())
        dist.all_reduce(bytes_tensor, op=dist.ReduceOp.SUM,
                        group=parallel_state.get_pipeline_model_parallel_group())
        for i, key in enumerate(buckets):
            buckets[key]["numel"]  = numel_tensor[i].item()
            buckets[key]["bytes"] = bytes_tensor[i].item()

    total_numel  = sum(b["numel"]  for b in buckets.values())
    total_nbytes = sum(b["bytes"] for b in buckets.values())
    return buckets, total_numel, total_nbytes, tp_size, ep_size, pp_size


def profile_model(model_wrapper, args, config):
    """
    Full profiling report: param counts + cache sizes vs utility estimates.
    Only prints on rank 0.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0

    # All ranks participate in the all_reduce inside count_total_params
    buckets, total_numel, total_nbytes, tp_size, ep_size, pp_size = \
        count_total_params(model_wrapper)

    if rank != 0:
        return

    try:
        m = model_wrapper.module
    except AttributeError:
        m = model_wrapper

    print("\n" + "=" * 72)
    print("PARAMETER PROFILING")
    print(f"  TP={tp_size}  EP={ep_size}  PP={pp_size}")
    print("=" * 72)

    print(f"\n{'Layer type':<16} {'Params (total)':>18} {'Bytes':>16}")
    print("-" * 54)
    for key, b in buckets.items():
        if b["numel"] == 0:
            continue
        print(f"  {key:<14} {b['numel']:>18,} {b['bytes']:>16,}")
    print("-" * 54)
    print(f"  {'TOTAL':<14} {total_numel:>18,} {total_nbytes:>16,}  "
          f"({bytes_to_gb(total_nbytes):.3f} GB)")

    # ── utility function estimates ─────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("UTILITY FUNCTION ESTIMATES  (get_num_parameters)")
    print("=" * 72)

    est_all, est_active = get_num_parameters(
        hybrid_pattern=m.hybrid_layer_pattern,
        mamba_num_heads=config.mamba_num_heads,
        mamba_d_head=config.mamba_head_dim,
        mamba_d_state=config.mamba_state_dim,
        num_attention_heads=config.num_attention_heads,
        num_query_groups=config.num_query_groups,
        ffn_hidden_size=config.ffn_hidden_size,
        hidden_size=config.hidden_size,
        kv_channels=config.kv_channels,
        vocab_size=m.vocab_size,
        tied_vocab=m.share_embeddings_and_output_weights,
        num_experts=config.num_moe_experts,
        shared_expert_intermediate_size=config.moe_shared_expert_intermediate_size,
        moe_router_topk=config.moe_router_topk,
    )

    diff_all    = total_numel - est_all
    diff_active = (buckets["attn"]["numel"] + buckets["mamba"]["numel"] +
                   buckets["emb"]["numel"]  + buckets["out"]["numel"] +
                   buckets["moe_shared"]["numel"] +
                   buckets["moe_expert"]["numel"] // (config.num_moe_experts // config.moe_router_topk)
                   ) - est_active  # rough active estimate

    print(f"\n  {'':30} {'All params':>15} {'Active params':>15}")
    print(f"  {'-'*62}")
    print(f"  {'Utility estimate':<30} {est_all:>15,} {est_active:>15,}")
    print(f"  {'Actual (measured)':<30} {total_numel:>15,}")
    print(f"  {'Difference':<30} {diff_all:>+15,}  ({100*diff_all/max(est_all,1):.2f}%)")

    # ── KV and SSM cache ───────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("INFERENCE CACHE PROFILING")
    print("=" * 72)

    mem_bs  = getattr(args, 'mem_batch_size', 1)
    mem_seq = getattr(args, 'mem_infer_seq_len', args.seq_length)
    print(f"\n  mem_batch_size={mem_bs},  mem_infer_seq_len={mem_seq}")

    kv_total_bytes  = 0
    kv_layers       = 0
    kv_dtypes       = set()
    ssm_total_bytes = 0
    ssm_layers      = 0
    ssm_dtypes      = set()

    for name, mod in m.named_modules():
        cls = type(mod).__name__

        # KV cache: probe attention modules that expose _allocate_memory
        if hasattr(mod, '_allocate_memory') and hasattr(mod, 'key_hidden_size'):
            try:
                k = mod._allocate_memory(mem_seq, mem_bs, mod.key_hidden_size,
                                         torch.bfloat16)
                v = mod._allocate_memory(mem_seq, mem_bs, mod.val_hidden_size,
                                         torch.bfloat16)
                kv_total_bytes += k.numel() * k.element_size() \
                                + v.numel() * v.element_size()
                kv_dtypes.add(k.dtype)
                kv_layers += 1
                del k, v
            except Exception:
                pass

        # SSM cache: ONLY MambaMixer — MambaLayer.allocate_inference_cache is a
        # passthrough to self.mixer.allocate_inference_cache, so walking both
        # would double-count.  We identify MambaMixer by class name.
        if cls == 'MambaMixer' and hasattr(mod, 'allocate_inference_cache'):
            try:
                conv_s, ssm_s = mod.allocate_inference_cache(mem_bs, mem_seq)
                ssm_total_bytes += conv_s.numel() * conv_s.element_size() \
                                 + ssm_s.numel()  * ssm_s.element_size()
                ssm_dtypes.add(ssm_s.dtype)
                ssm_layers += 1
                del conv_s, ssm_s
            except Exception:
                pass

    print(f"\n  KV cache:  {kv_layers} attention layers")
    if kv_layers:
        print(f"    dtype        : {', '.join(dtype_name(d) for d in kv_dtypes)}")
        print(f"    total bytes  : {kv_total_bytes:,}  ({bytes_to_gb(kv_total_bytes):.4f} GB)")

    print(f"\n  SSM cache: {ssm_layers} MambaMixer layers  "
          f"(MambaLayer passthrough excluded)")
    if ssm_layers:
        print(f"    dtype        : {', '.join(dtype_name(d) for d in ssm_dtypes)}")
        print(f"    total bytes  : {ssm_total_bytes:,}  ({bytes_to_gb(ssm_total_bytes):.4f} GB)")
        # break down conv_state vs ssm_state contribution
        print(f"    Note: includes both conv_state and ssm_state per layer")

    # ── utility estimates for caches ──────────────────────────────────────────
    print("\n" + "=" * 72)
    print("UTILITY ESTIMATES  (get_kv_cache_size / get_mamba_ssm_cache_size)")
    print("=" * 72)

    kv_est_elems  = get_kv_cache_size(
        hybrid_pattern=m.hybrid_layer_pattern,
        num_attention_heads=config.num_attention_heads,
        num_query_groups=config.num_query_groups,
        kv_channels=config.kv_channels,
        mem_infer_seq_len=mem_seq,
        mem_batch_size=mem_bs,
    )
    ssm_est_elems = get_mamba_ssm_cache_size(
        hybrid_pattern=m.hybrid_layer_pattern,
        mamba_num_heads=config.mamba_num_heads,
        mamba_d_head=config.mamba_head_dim,
        mamba_d_state=config.mamba_state_dim,
        mem_batch_size=mem_bs,
    )

    kv_est_bf16  = int(kv_est_elems)  * 2
    ssm_est_bf16 = int(ssm_est_elems) * 2
    ssm_est_fp32 = int(ssm_est_elems) * 4

    print(f"\n  KV cache:")
    print(f"    util elements       : {int(kv_est_elems):,}")
    print(f"    util bytes (BF16×2) : {kv_est_bf16:,}  ({bytes_to_gb(kv_est_bf16):.4f} GB)")
    print(f"    actual bytes        : {kv_total_bytes:,}  ({bytes_to_gb(kv_total_bytes):.4f} GB)")
    print(f"    match               : {'YES ✓' if kv_est_bf16 == kv_total_bytes else 'NO ✗'}")

    # Compute what the correct SSM utility should be (ssm_state + conv_state)
    n_mamba = m.hybrid_layer_pattern.count('M')
    d_inner  = config.mamba_num_heads * config.mamba_head_dim
    ngroups  = config.mamba_num_groups if hasattr(config, 'mamba_num_groups') else 8
    cdim     = d_inner + 2 * ngroups * config.mamba_state_dim
    d_conv   = 4
    conv_est_elems = mem_bs * cdim * d_conv * n_mamba
    ssm_full_bf16  = ssm_est_bf16 + conv_est_elems * 2

    print(f"\n  SSM cache (util only counts ssm_state, not conv_state):")
    print(f"    util elements (ssm_state only)  : {int(ssm_est_elems):,}")
    print(f"    util bytes (BF16×2, ssm only)   : {ssm_est_bf16:,}  "
          f"({bytes_to_gb(ssm_est_bf16):.4f} GB)")
    print(f"    util bytes (FP32×4, code value) : {ssm_est_fp32:,}  "
          f"({bytes_to_gb(ssm_est_fp32):.4f} GB)  ← code uses this")
    print(f"    est conv_state (BF16×2)         : {conv_est_elems*2:,}  "
          f"({bytes_to_gb(conv_est_elems*2):.4f} GB)")
    print(f"    util bytes (BF16×2, ssm+conv)   : {ssm_full_bf16:,}  "
          f"({bytes_to_gb(ssm_full_bf16):.4f} GB)")
    print(f"    actual bytes                    : {ssm_total_bytes:,}  "
          f"({bytes_to_gb(ssm_total_bytes):.4f} GB)")
    print(f"    match (ssm+conv, BF16)          : "
          f"{'YES ✓' if ssm_full_bf16 == ssm_total_bytes else 'NO ✗'}")

    # ── full memory footprint ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("FULL MEMORY FOOTPRINT  (get_memory_footprint vs measured)")
    print("=" * 72)

    mem_est_gb = get_memory_footprint(
        hybrid_pattern=m.hybrid_layer_pattern,
        mamba_num_heads=config.mamba_num_heads,
        mamba_d_head=config.mamba_head_dim,
        mamba_d_state=config.mamba_state_dim,
        num_attention_heads=config.num_attention_heads,
        num_query_groups=config.num_query_groups,
        ffn_hidden_size=config.ffn_hidden_size,
        hidden_size=config.hidden_size,
        kv_channels=config.kv_channels,
        vocab_size=m.vocab_size,
        tied_vocab=m.share_embeddings_and_output_weights,
        mem_infer_seq_len=mem_seq,
        mem_batch_size=mem_bs,
        prefill_chunk_size=getattr(config, 'prefill_chunk_size', 16384),
        moe_num_experts=config.num_moe_experts,
        shared_expert_intermediate_size=config.moe_shared_expert_intermediate_size,
        moe_router_topk=config.moe_router_topk,
    )
    if isinstance(mem_est_gb, torch.Tensor):
        mem_est_gb = mem_est_gb.item()

    measured_total_gb = bytes_to_gb(total_nbytes + kv_total_bytes + ssm_total_bytes)

    print(f"\n  {'Component':<34} {'Utility (GB)':>12} {'Measured (GB)':>14}")
    print(f"  {'-'*62}")
    print(f"  {'Params (BF16×2)':<34} {bytes_to_gb(est_all*2):>12.4f} "
          f"{bytes_to_gb(total_nbytes):>14.4f}")
    print(f"  {'KV cache (BF16×2)':<34} {bytes_to_gb(kv_est_bf16):>12.4f} "
          f"{bytes_to_gb(kv_total_bytes):>14.4f}")
    print(f"  {'SSM (code: FP32×4, ssm only)':<34} {bytes_to_gb(ssm_est_fp32):>12.4f} "
          f"{bytes_to_gb(ssm_total_bytes):>14.4f}")
    print(f"  {'SSM (fixed: BF16×2, ssm+conv)':<34} {bytes_to_gb(ssm_full_bf16):>12.4f} "
          f"{bytes_to_gb(ssm_total_bytes):>14.4f}")
    print(f"  {'-'*62}")
    print(f"  {'Total (get_memory_footprint)':<34} {mem_est_gb:>12.4f}")
    print(f"  {'Total (measured params+kv+ssm)':<34} {measured_total_gb:>12.4f}")
    print(f"\n  Note: max buffer is transient (not persistently allocated); "
          f"not measured here.")
    print("=" * 72 + "\n")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    assert args.hybrid_layer_pattern is not None, "Hybrid override pattern is required"

    from functools import partial
    from pretrain_mamba import (
        train_valid_test_datasets_provider,
        model_provider,
        forward_step
    )
    from mamba_builders import mamba_builder
    model_provider = partial(model_provider, mamba_builder)

    def get_model_provider():
        return model_provider

    model = get_model(get_model_provider(), wrap_with_ddp=False)
    config = get_model_config(model[0])

    if args.hybrid_layer_pattern is not None and getattr(config, 'flextron', False):
        from megatron.elastification.flextron_utils import (
            setup_flextron_model, inject_flextron_forward_logic
        )
        setup_flextron_model(model[0].module)
        inject_flextron_forward_logic(model[0].module)

    if args.load is not None:
        iteration, num_floating_point_operations_so_far = load_checkpoint(
            model, None, None, strict=True
        )

    # ── profiling ─────────────────────────────────────────────────────────────
    profile_model(model[0], args, config)

    # ── evaluation ────────────────────────────────────────────────────────────
    args.iteration = 0
    args.curr_iteration = 0
    update_train_iters(args)
    train_valid_test_datasets_provider.is_distributed = True
    train_iterator, _, _ = build_train_valid_test_data_iterators(
        train_valid_test_datasets_provider
    )

    prefix = f'iteration {args.iteration}'
    evaluate_and_print_results(
        prefix=prefix,
        forward_step_func=forward_step,
        data_iterator=train_iterator,
        model=model,
        iteration=args.iteration,
        process_non_loss_data_func=None,
        config=config,
        verbose=True,
        write_to_tensorboard=False,
    )
    print_rank_0('done :-)')


if __name__ == '__main__':
    initialize_megatron(extra_args_provider=add_flextron_args)
    main()
