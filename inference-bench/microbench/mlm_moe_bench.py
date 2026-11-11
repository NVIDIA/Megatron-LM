"""Routed MoE as Megatron runs it under the NVLS dispatcher: worst-case buffers, valid prefix.

Simulates one MoE layer of one decode step on one GPU for a given deployment, e.g. Ultra v3 at
TP8 x EP64 with C concurrent requests:
  DP groups = EP / TP; tokens per group = C / DP; local tokens per GPU (sequence parallel) = that / TP
  gathered valid rows T = local x EP; worst-case buffer rows R = max_tokens / TP x EP (65,536)
Rows past T hold stale data from earlier steps. Both paths write an fp32 combine (RSV) buffer
[R, hidden] and copy the valid prefix in each step (standing in for the AGV, same for both).

  vllm_triton     megatron today: vllm_fused_moe on the R-row buffers, device valid_tokens,
                  num_tokens_hint = local x EP (megatron's host estimate), out = RSV
  trtllm_branch   nvcsathe/Megatron-LM@trtllm-fused-moe: fill routing with -1 over all R rows (the
                  branch also zeroes probs, which is unnecessary), copy the valid prefix, pack ids|bf16 probs over R rows,
                  trtllm_bf16_routed_moe(do_finalize=False, tune_max_num_tokens=R),
                  padding_aware_finalize into RSV (device valid_tokens)
  trtllm_bounded  same, but every pass works on the first B = local x EP rows (the bound a per-step
                  EP consensus makes host-known; all ranks equally loaded here), and the finalize
                  loops only over valid tokens (finalize_valid_loop.py)
Times per layer in a CUDA graph (48 layers per graph), with the trtllm path's parts timed too, and
checks both against an fp32 reference on the valid rows.

    python inference-bench/microbench/mlm_moe_bench.py --concurrency 128
    python inference-bench/microbench/mlm_moe_bench.py --concurrency 1 8 32 128 512 1024
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csathe_trtllm_finalize import padding_aware_finalize  # noqa: E402
from finalize_valid_loop import padding_aware_finalize_valid_loop  # noqa: E402
from grouped_gemm_bench import E_GLOBAL, K, N, TOPK, reference, time_graph  # noqa: E402


def trtllm_weights(fc1, fc2):
    """BlockMajorK weights exactly as the branch builds them (_build_trtllm_shuffled_weights)."""
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        convert_to_block_layout,
        get_w2_permute_indices_with_cache,
    )

    cache, w1s, w2s = {}, [], []
    for e in range(fc1.shape[0]):
        a = fc1[e].contiguous().view(torch.uint8)
        b = fc2[e].contiguous().view(torch.uint8)
        ia = _maybe_get_cached_w3_w1_permute_indices(cache, a, 128, is_gated_act_gemm=False)
        ib = get_w2_permute_indices_with_cache(cache, b, 128)
        w1s.append(convert_to_block_layout(a[ia].contiguous(), 128).view(torch.bfloat16))
        w2s.append(convert_to_block_layout(b[ib].contiguous(), 128).view(torch.bfloat16))
    return torch.stack(w1s), torch.stack(w2s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[128])
    ap.add_argument("--tp", type=int, nargs="+", default=[8])
    ap.add_argument("--ep", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--bound-mode", choices=["tight", "fixed"], default="tight",
                    help="trtllm_bounded rows: tight = local x EP per step (needs EP consensus); "
                         "fixed = graph_max_tokens x EP / TP (static decode-graph ceiling)")
    ap.add_argument("--graph-max-tokens", type=int, default=512,
                    help="--inference-cuda-graph-max-tokens, for --bound-mode fixed")
    args = ap.parse_args()
    for tp in args.tp:
        run(args, tp)


def run(args, tp):

    import flashinfer.fused_moe as fm
    from flashinfer.fused_moe import WeightLayout
    from flashinfer.fused_moe.core import ActivationType as FiAct

    from megatron.core.inference.moe import ActivationType, vllm_fused_moe

    ep = args.ep
    e_local = E_GLOBAL // ep
    dp = ep // tp
    rows = args.max_tokens // tp * ep
    torch.manual_seed(0)
    fc1 = (torch.randn(e_local, N, K, device="cuda") * K ** -0.5).bfloat16()
    fc2 = (torch.randn(e_local, K, N, device="cuda") * N ** -0.5).bfloat16()
    w1_bmk, w2_bmk = trtllm_weights(fc1, fc2)

    # Worst-case NVLS buffers, filled with stale garbage (as after earlier, larger steps).
    hidden_buf = (torch.randn(rows, K, device="cuda") * 0.5).bfloat16()
    routing_buf = torch.randint(0, E_GLOBAL, (rows, TOPK), device="cuda", dtype=torch.int32)
    probs_buf = torch.rand(rows, TOPK, device="cuda")
    rsv = torch.zeros(rows, K, device="cuda", dtype=torch.float32)
    valid = torch.zeros(1, dtype=torch.int32, device="cuda")

    print(f"GPU {torch.cuda.get_device_name()}; TP{tp} x EP{ep} ({dp} DP groups), {e_local} local experts "
          f"of {E_GLOBAL}, top-{TOPK}, K={K}, N={N}; worst-case buffer {rows} rows; fp32 RSV")

    for conc in args.concurrency:
        per_group = -(-conc // dp)
        local = -(-per_group // tp)
        t = local * ep  # gathered valid rows (all ranks equally loaded)
        g = torch.Generator(device="cuda").manual_seed(conc)
        v_hidden = (torch.randn(t, K, generator=g, device="cuda") * 0.5).bfloat16()
        v_routing = torch.rand(t, E_GLOBAL, generator=g, device="cuda").topk(TOPK, dim=1).indices.int()
        v_probs = torch.rand(t, TOPK, generator=g, device="cuda")
        v_probs = v_probs / v_probs.sum(1, keepdim=True)
        hidden_buf[:t].copy_(v_hidden)
        valid.fill_(t)
        hint = local * ep

        def agv():  # the all-gather's write of the valid prefix (same cost for both paths)
            routing_buf[:t].copy_(v_routing)
            probs_buf[:t].copy_(v_probs)

        def vllm_step():
            agv()
            vllm_fused_moe(hidden_buf, probs_buf, fc1, fc2, ActivationType.SQUARED_RELU, e_local, 0,
                           valid, routing_buf, out=rsv, num_tokens_hint=hint)

        state = {}

        def trt_prep():
            # Branch: update_metadata pre-masks every row with -1. (The branch also zeroes probs;
            # not needed: a -1 id stays -1 after packing, and -1 rows are never weighted.)
            routing_buf.fill_(-1)
            agv()
            state["packed"] = (routing_buf << 16) | probs_buf.to(torch.bfloat16).view(torch.int16).to(torch.int32)

        def trt_kernel():
            state["out"] = fm.trtllm_bf16_routed_moe(
                topk_ids=state["packed"], hidden_states=hidden_buf, gemm1_weights=w1_bmk,
                gemm2_weights=w2_bmk, num_experts=E_GLOBAL, top_k=TOPK, n_group=None,
                topk_group=None, intermediate_size=N, local_expert_offset=0,
                local_num_experts=e_local, routed_scaling_factor=None, routing_method_type=0,
                use_shuffled_weight=True, weight_layout=WeightLayout.BlockMajorK, do_finalize=False,
                tune_max_num_tokens=rows, activation_type=FiAct.Relu2.value)

        def trt_finalize():
            o = state["out"]
            padding_aware_finalize(gemm2_output=o[0], expert_weights=o[1], inverse_map=o[2],
                                   output=rsv, valid_tokens=valid, top_k=TOPK)

        def trt_step():
            trt_prep()
            trt_kernel()
            trt_finalize()

        # Rows the bounded kernel sees: tight = local tokens x EP (per-step, needs EP consensus);
        # fixed = the decode graph ceiling graph_max_tokens / TP per rank, x EP (static).
        bound = hint if args.bound_mode == "tight" else args.graph_max_tokens * ep // tp
        if bound < t:
            print(f"\nconcurrency {conc}: skipped, valid rows {t} exceed the {args.bound_mode} bound {bound}")
            continue

        def bnd_prep():
            routing_buf[:bound].fill_(-1)
            agv()
            state["bpacked"] = (routing_buf[:bound] << 16) | probs_buf[:bound].to(torch.bfloat16).view(
                torch.int16).to(torch.int32)

        def bnd_kernel():
            state["bout"] = fm.trtllm_bf16_routed_moe(
                topk_ids=state["bpacked"], hidden_states=hidden_buf[:bound], gemm1_weights=w1_bmk,
                gemm2_weights=w2_bmk, num_experts=E_GLOBAL, top_k=TOPK, n_group=None,
                topk_group=None, intermediate_size=N, local_expert_offset=0,
                local_num_experts=e_local, routed_scaling_factor=None, routing_method_type=0,
                use_shuffled_weight=True, weight_layout=WeightLayout.BlockMajorK, do_finalize=False,
                tune_max_num_tokens=bound, activation_type=FiAct.Relu2.value)

        def bnd_finalize():
            o = state["bout"]
            padding_aware_finalize_valid_loop(gemm2_output=o[0], expert_weights=o[1], inverse_map=o[2],
                                              output=rsv[:bound], valid_tokens=valid, top_k=TOPK)

        def bnd_step():
            bnd_prep()
            bnd_kernel()
            bnd_finalize()

        with torch.inference_mode():
            ref = reference(v_hidden, v_routing.long(), v_probs, fc1, fc2)
            errs = {}
            for name, step in (("vllm_triton", vllm_step), ("trtllm_branch", trt_step),
                               ("trtllm_bounded", bnd_step)):
                rsv.fill_(float("nan"))
                step()
                torch.cuda.synchronize()
                errs[name] = ((rsv[:t] - ref).norm() / ref.norm().clamp_min(1e-6)).item()
            times = {
                "vllm_triton": time_graph(vllm_step),
                "trtllm_branch": time_graph(trt_step),
                "trtllm_bounded": time_graph(bnd_step),
            }
            trt_prep()
            parts = {"prep (fill+pack)": time_graph(trt_prep)}
            parts["trtllm kernel"] = time_graph(trt_kernel)
            trt_kernel()
            parts["finalize"] = time_graph(trt_finalize)
            bparts = {"prep (fill+pack)": time_graph(bnd_prep)}
            bparts["trtllm kernel"] = time_graph(bnd_kernel)
            bnd_kernel()
            bparts["finalize (valid loop)"] = time_graph(bnd_finalize)
            agv_us = time_graph(agv)

        print(f"\nconcurrency {conc}: {per_group}/DP group, {local} local tokens/GPU, valid rows T={t}, "
              f"hint={hint}, buffer R={rows}")
        print(f"  correctness (rel err vs fp32 ref on valid rows): "
              + ", ".join(f"{k} {v:.2e}" for k, v in errs.items()))
        print("  per layer: " + " | ".join(f"{k} {v:.1f} us" for k, v in times.items())
              + f"  (all include the AGV prefix copy, {agv_us:.1f} us)")
        print("  trtllm_branch  parts: " + ", ".join(f"{k} {v:.1f} us" for k, v in parts.items()))
        print(f"  trtllm_bounded parts ({args.bound_mode} bound B={bound} rows): "
              + ", ".join(f"{k} {v:.1f} us" for k, v in bparts.items()))
        for name in ("trtllm_branch", "trtllm_bounded"):
            delta = (times[name] - times["vllm_triton"]) * 48 / 1000
            print(f"  x48 layers: {delta:+.2f} ms/step for {name} vs vllm_triton")


if __name__ == "__main__":
    main()
