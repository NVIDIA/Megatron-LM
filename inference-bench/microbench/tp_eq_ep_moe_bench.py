"""Routed MoE layer per GPU with TP = EP: vLLM Triton vs FlashInfer TRT-LLM, across EP and concurrency.

With TP = EP (ETP = 1) the EP group is one data-parallel group, so every rank holds the same batch
and the gathered row count is host-known: ceil(C / TP) * EP for C decode tokens (rows past C are
CUDA-graph padding, expert id -1). No worst-case buffer is needed. One MoE layer on one GPU of
Nemotron 3 Ultra's shape: 512 experts, 512 / EP local, top-22, K=2048, N=5120, squared ReLU, bf16
combine (RSV) buffer:

  vllm_triton  megatron vllm_fused_moe(valid_tokens=C, num_tokens_hint=rows, out=RSV)
  trtllm       -1 fill + id<<16|bf16-prob packing over the rows, trtllm_bf16_routed_moe
               (do_finalize=False), padding-aware finalize into RSV (loop to valid_tokens)

Answers (1) which kernel is faster per (EP, C) and (2) how the expert-compute time per GPU changes
with EP at fixed C (wide EP: fewer local experts, fewer weight bytes per GPU). Dispatch / combine
communication, which grows with EP, is not included.

    python inference-bench/microbench/tp_eq_ep_moe_bench.py [--ep 4 8 16 32 64] [--concurrency 1 2 ... 2048]
"""

import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finalize_valid_loop import padding_aware_finalize_valid_loop  # noqa: E402
from grouped_gemm_bench import E_GLOBAL, K, N, TOPK, time_graph  # noqa: E402
from mlm_moe_bench import trtllm_weights  # noqa: E402


def reference(hidden, routing, probs, fc1, fc2, e_local):
    out = torch.zeros(hidden.shape[0], K, device="cuda")
    for e in range(e_local):
        tok, slot = (routing == e).nonzero(as_tuple=True)
        if tok.numel():
            h = torch.relu(hidden[tok].float() @ fc1[e].float().t()) ** 2
            out.index_add_(0, tok, (h.bfloat16().float() @ fc2[e].float().t()) * probs[tok, slot][:, None])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ep", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    ap.add_argument("--concurrency", type=int, nargs="+", default=[2 ** i for i in range(12)])
    ap.add_argument("--csv", default="inference-bench/microbench/tp_eq_ep_moe_bench.csv")
    args = ap.parse_args()

    import flashinfer.fused_moe as fm
    from flashinfer.fused_moe import WeightLayout
    from flashinfer.fused_moe.core import ActivationType as FiAct

    from megatron.core.inference.moe import ActivationType, vllm_fused_moe

    print(f"GPU {torch.cuda.get_device_name()}; Ultra shape, TP = EP, bf16 combine buffer; "
          "per MoE layer per GPU in us (48 layers per CUDA graph)")
    rows_out = []
    for ep in args.ep:
        e_local = E_GLOBAL // ep
        torch.manual_seed(ep)
        fc1 = (torch.randn(e_local, N, K, device="cuda") * K ** -0.5).bfloat16()
        fc2 = (torch.randn(e_local, K, N, device="cuda") * N ** -0.5).bfloat16()
        w1, w2 = trtllm_weights(fc1, fc2)
        max_rows = -(-max(args.concurrency) // ep) * ep
        hidden_buf = torch.zeros(max_rows, K, device="cuda", dtype=torch.bfloat16)
        routing_buf = torch.full((max_rows, TOPK), -1, device="cuda", dtype=torch.int32)
        probs_buf = torch.zeros(max_rows, TOPK, device="cuda")
        rsv = torch.zeros(max_rows, K, device="cuda", dtype=torch.bfloat16)
        valid = torch.zeros(1, dtype=torch.int32, device="cuda")
        print(f"\n== EP{ep} (TP{ep}): {e_local} local experts, "
              f"{e_local * N * K * 2 * 2 / 1e9:.2f} GB expert weights per GPU")
        print(f"{'conc':>5} {'rows':>5} {'pairs':>5} {'hit':>4} | {'vllm_triton':>11} {'trtllm':>8} | "
              f"{'faster':>11} {'ratio':>6} | {'best TB/s':>9}")

        for c in args.concurrency:
            rows = -(-c // ep) * ep  # sequence-parallel padding to a multiple of TP (= EP)
            g = torch.Generator(device="cuda").manual_seed(c)
            v_hidden = (torch.randn(c, K, generator=g, device="cuda") * 0.5).bfloat16()
            v_routing = torch.rand(c, E_GLOBAL, generator=g, device="cuda").topk(TOPK, dim=1).indices.int()
            v_probs = torch.rand(c, TOPK, generator=g, device="cuda")
            v_probs = v_probs / v_probs.sum(1, keepdim=True)
            hidden_buf[:c].copy_(v_hidden)
            hidden_buf[c:rows].zero_()
            valid.fill_(c)
            h, r, p, out = hidden_buf[:rows], routing_buf[:rows], probs_buf[:rows], rsv[:rows]
            local = v_routing < e_local
            pairs, hit = int(local.sum()), int(torch.unique(v_routing[local]).numel())
            st = {}

            def write_prefix():  # dispatch output for the valid rows (same for both paths)
                r[:c].copy_(v_routing)
                p[:c].copy_(v_probs)

            def vllm_layer():
                write_prefix()
                vllm_fused_moe(h, p, fc1, fc2, ActivationType.SQUARED_RELU, e_local, 0, valid, r,
                               out=out, num_tokens_hint=rows)

            def trt_layer():
                r.fill_(-1)
                write_prefix()
                packed = (r << 16) | p.to(torch.bfloat16).view(torch.int16).to(torch.int32)
                o = fm.trtllm_bf16_routed_moe(
                    topk_ids=packed, hidden_states=h, gemm1_weights=w1, gemm2_weights=w2,
                    num_experts=E_GLOBAL, top_k=TOPK, n_group=None, topk_group=None,
                    intermediate_size=N, local_expert_offset=0, local_num_experts=e_local,
                    routed_scaling_factor=None, routing_method_type=0, use_shuffled_weight=True,
                    weight_layout=WeightLayout.BlockMajorK, do_finalize=False,
                    tune_max_num_tokens=rows, activation_type=FiAct.Relu2.value)
                padding_aware_finalize_valid_loop(gemm2_output=o[0], expert_weights=o[1],
                                                  inverse_map=o[2], output=out, valid_tokens=valid,
                                                  top_k=TOPK)

            with torch.inference_mode():
                ref = reference(v_hidden, v_routing.long(), v_probs, fc1, fc2, e_local)
                errs, times = {}, {}
                for name, fn in (("vllm_triton", vllm_layer), ("trtllm", trt_layer)):
                    out.fill_(float("nan"))
                    fn()
                    torch.cuda.synchronize()
                    errs[name] = ((out[:c].float() - ref).norm() / ref.norm().clamp_min(1e-6)).item()
                    try:
                        times[name] = time_graph(fn)
                    except Exception as e:  # noqa: BLE001
                        times[name] = float("nan")
                        print(f"  {name} EP{ep} C={c}: {type(e).__name__}: {str(e)[:200]}")
            bad = [n for n, e in errs.items() if not e < 2e-2]
            faster = min(times, key=times.get)
            ratio = times["vllm_triton"] / times["trtllm"]
            best_tbs = hit * N * K * 2 * 2 / (times[faster] * 1e-6) / 1e12
            flag = f"  WRONG: {bad} {[f'{errs[b]:.1e}' for b in bad]}" if bad else ""
            print(f"{c:5d} {rows:5d} {pairs:5d} {hit:4d} | {times['vllm_triton']:11.1f} {times['trtllm']:8.1f} | "
                  f"{faster:>11} {ratio:6.2f} | {best_tbs:9.2f}{flag}")
            rows_out.append({"ep": ep, "local_experts": e_local, "concurrency": c, "rows": rows,
                             "pairs": pairs, "experts_hit": hit,
                             "vllm_triton_us": round(times["vllm_triton"], 2),
                             "trtllm_us": round(times["trtllm"], 2),
                             "err_vllm": errs["vllm_triton"], "err_trtllm": errs["trtllm"]})
        del fc1, fc2, w1, w2, hidden_buf, routing_buf, probs_buf, rsv
        torch.cuda.empty_cache()

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)

    print("\nPer-GPU MoE layer time vs EP at fixed concurrency (best kernel), us:")
    eps = args.ep
    print(f"{'conc':>5} | " + " | ".join(f"{'EP' + str(e):>8}" for e in eps))
    for c in args.concurrency:
        cells = []
        for e in eps:
            m = [x for x in rows_out if x["ep"] == e and x["concurrency"] == c]
            cells.append(f"{min(m[0]['vllm_triton_us'], m[0]['trtllm_us']):8.1f}" if m else f"{'-':>8}")
        print(f"{c:5d} | " + " | ".join(cells))
    print(f"\ncsv -> {args.csv}")


if __name__ == "__main__":
    main()
