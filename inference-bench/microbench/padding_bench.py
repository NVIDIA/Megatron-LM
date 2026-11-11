"""How routed-MoE kernels pay for padding rows in Megatron's worst-case token buffer.

Megatron's NVLS dispatcher all-gathers into a buffer sized for the worst case,
round_up(max_tokens) / TP * EP rows (65,536 for Ultra at TP8 x EP64), with the T valid rows packed
at the front. Kernels that size their work from the tensor shape pay for every row; kernels that
take a device-side valid count do not.

For valid tokens T and total rows R (padding rows: zero hidden state, expert id -1), times per call
in a CUDA graph (48 calls):
  trtllm_fin     flashinfer trtllm_bf16_routed_moe, do_finalize=True  (writes [R, 2048] output)
  trtllm_nofin   same, do_finalize=False (unfinalized; our own combine over valid rows not included)
  vllm_triton    megatron vllm_fused_moe with valid_tokens=T, num_tokens_hint=T (control)
and checks that the first T rows of the padded trtllm_fin call match the exact-size call.

    python inference-bench/microbench/padding_bench.py
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grouped_gemm_bench import (  # noqa: E402
    E_GLOBAL, E_LOCAL, EP, K, N, TOPK, backend_fi_trtllm_bf16, make_inputs, time_graph,
)


def padded(t, rows, seed):
    hidden, routing, probs = make_inputs(t, seed=seed)
    h = torch.zeros(rows, K, dtype=hidden.dtype, device="cuda")
    r = torch.full((rows, TOPK), -1, dtype=routing.dtype, device="cuda")
    p = torch.zeros(rows, TOPK, dtype=probs.dtype, device="cuda")
    h[:t], r[:t], p[:t] = hidden, routing, probs
    return h, r, p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--valid", type=int, nargs="+", default=[1, 8, 32, 128, 512, 1024])
    ap.add_argument("--rows", type=int, nargs="+", default=[512, 4096, 16384, 65536])
    args = ap.parse_args()

    import flashinfer.fused_moe as fm
    from flashinfer.fused_moe import WeightLayout
    from flashinfer.fused_moe.core import ActivationType as FiAct

    from megatron.core.inference.moe import ActivationType, vllm_fused_moe
    from megatron.core.inference.moe.flashinfer_mxfp8 import pack_routed_mxfp8_routing

    torch.manual_seed(0)
    fc1 = (torch.randn(E_LOCAL, N, K, device="cuda") * K ** -0.5).bfloat16()
    fc2 = (torch.randn(E_LOCAL, K, N, device="cuda") * N ** -0.5).bfloat16()
    setup, _ = backend_fi_trtllm_bf16()
    st = setup(fc1, fc2)

    def trtllm(h, packed, finalize):
        out = fm.trtllm_bf16_routed_moe(
            topk_ids=packed, hidden_states=h, gemm1_weights=st["fc1"], gemm2_weights=st["fc2"],
            num_experts=E_GLOBAL, top_k=TOPK, n_group=None, topk_group=None, intermediate_size=N,
            local_expert_offset=0, local_num_experts=E_LOCAL, routed_scaling_factor=None,
            routing_method_type=1, use_shuffled_weight=True, weight_layout=WeightLayout.BlockMajorK,
            do_finalize=finalize, activation_type=FiAct.Relu2.value)
        return out[0] if isinstance(out, (list, tuple)) else out

    print(f"GPU {torch.cuda.get_device_name()}; Ultra EP{EP} shape, per MoE call in us (48 calls/graph)")
    header = f"{'valid T':>7} {'variant':>12} | {'R=T':>8} | " + " | ".join(f"{'R=' + str(r):>8}" for r in args.rows)
    print(header)
    print("-" * len(header))
    for t in args.valid:
        rows_list = [t] + list(args.rows)
        results = {"trtllm_fin": [], "trtllm_nofin": [], "vllm_triton": []}
        exact_out = None
        for rows in rows_list:
            if rows < t or (rows == t and rows_list.index(rows) > 0):
                for name in results:
                    results[name].append(f"{'-':>8}")  # buffer smaller than (or same as) T
                continue
            h, r, p = padded(t, rows, seed=t)
            packed = pack_routed_mxfp8_routing(r, p)
            valid = torch.tensor([t], dtype=torch.int32, device="cuda")
            with torch.inference_mode():
                out = trtllm(h, packed, True)
                if rows == t:
                    exact_out = out[:t].clone()
                elif exact_out is not None:
                    err = ((out[:t].float() - exact_out.float()).norm()
                           / exact_out.float().norm().clamp_min(1e-6)).item()
                    if err > 1e-2:
                        print(f"  WARNING T={t} R={rows}: padded output differs from exact ({err:.2e})")
                for name, fn in (
                    ("trtllm_fin", lambda: trtllm(h, packed, True)),
                    ("trtllm_nofin", lambda: trtllm(h, packed, False)),
                    ("vllm_triton", lambda: vllm_fused_moe(h, p, fc1, fc2, ActivationType.SQUARED_RELU,
                                                           E_LOCAL, 0, valid, r, num_tokens_hint=t)),
                ):
                    try:
                        results[name].append(f"{time_graph(fn):8.1f}")
                    except Exception as e:  # noqa: BLE001
                        results[name].append(f"{'err':>8}")
                        print(f"  {name} T={t} R={rows}: {type(e).__name__}: {str(e)[:200]}")
            del h, r, p, packed
            torch.cuda.empty_cache()
        for name, cells in results.items():
            print(f"{t:7d} {name:>12} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
