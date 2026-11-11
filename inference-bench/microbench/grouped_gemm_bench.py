"""Routed-expert MoE (grouped GEMM) backends in isolation, at Nemotron 3 Ultra's per-GPU shape.

Ultra at EP64: 512 experts, 8 local, top-22, hidden (latent) K=2048, expert FFN N=5120, non-gated
squared ReLU. With the NVLS dispatcher every GPU sees all T gathered tokens and runs the MoE for the
(token, expert) pairs that hit its 8 experts. One call = fc1 + activation + fc2 + weighted sum.

Backends (each skipped with the reason if unavailable or wrong):
  vllm_triton      megatron vllm_fused_moe (Triton, bf16) - the current default
  fi_cutlass_bf16  flashinfer.fused_moe.cutlass_fused_moe (bf16) - megatron's flashinfer backend
  fi_trtllm_bf16   flashinfer.fused_moe.trtllm_bf16_routed_moe (bf16) - what vLLM uses for Ultra
  fi_mxfp8         megatron flashinfer_routed_mxfp8_moe (TRT-LLM MXFP8); opt-in via --backends

Correctness vs an fp32 PyTorch reference (relative error), then per-call time in a CUDA graph
(48 calls = one per MoE layer) and achieved bandwidth over the weights of the local experts hit.

    python inference-bench/microbench/grouped_gemm_bench.py [--tokens 1 8 128 1024]
"""

import argparse
import statistics
import traceback

import torch

E_GLOBAL, EP, TOPK = 512, 64, 22
E_LOCAL = E_GLOBAL // EP
K, N = 2048, 5120
CALLS = 48
PEAK_TBS = 8.0


def make_inputs(t, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    hidden = (torch.randn(t, K, generator=g, device="cuda") * 0.5).bfloat16()
    scores = torch.rand(t, E_GLOBAL, generator=g, device="cuda")
    routing = scores.topk(TOPK, dim=1).indices  # global expert ids
    probs = torch.rand(t, TOPK, generator=g, device="cuda")
    probs = (probs / probs.sum(1, keepdim=True)).float()
    return hidden, routing, probs


def reference(hidden, routing, probs, fc1, fc2):
    out = torch.zeros(hidden.shape[0], K, device="cuda")
    for e in range(E_LOCAL):
        tok, slot = (routing == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        h = hidden[tok].float() @ fc1[e].float().t()
        h = torch.relu(h) ** 2
        y = h.bfloat16().float() @ fc2[e].float().t()
        out.index_add_(0, tok, y * probs[tok, slot][:, None])
    return out


def rel_err(out, ref):
    return ((out.float() - ref).norm() / ref.norm().clamp_min(1e-6)).item()


# ── Backends: each returns (setup_fn(weights) -> state, run_fn(state, hidden, routing, probs)) ──


def backend_vllm_triton():
    from megatron.core.inference.moe import ActivationType, vllm_fused_moe

    def setup(fc1, fc2):
        return {"fc1": fc1, "fc2": fc2, "valid": {}}

    def run(st, hidden, routing, probs):
        # valid_tokens must exist before CUDA-graph capture (no H2D copies while capturing).
        v = st["valid"].get(hidden.shape[0])
        if v is None:
            v = st["valid"][hidden.shape[0]] = torch.tensor(
                [hidden.shape[0]], dtype=torch.int32, device="cuda"
            )
        return vllm_fused_moe(hidden, probs, st["fc1"], st["fc2"], ActivationType.SQUARED_RELU,
                              E_LOCAL, 0, v, routing, num_tokens_hint=hidden.shape[0])

    return setup, run


def backend_fi_cutlass_bf16():
    import flashinfer.fused_moe as fm
    from flashinfer.fused_moe.core import ActivationType

    def setup(fc1, fc2):
        return {"fc1": fc1, "fc2": fc2}

    def run(st, hidden, routing, probs):
        return fm.cutlass_fused_moe(hidden, routing.int(), probs, st["fc1"], st["fc2"], torch.bfloat16,
                                    quant_scales=None, activation_type=ActivationType.Relu2,
                                    ep_size=EP, ep_rank=0)[0]

    return setup, run


def backend_fi_trtllm_bf16():
    import flashinfer.fused_moe as fm
    from flashinfer.fused_moe import WeightLayout
    from flashinfer.fused_moe.core import ActivationType, convert_to_block_layout

    from megatron.core.inference.moe.flashinfer_mxfp8 import pack_routed_mxfp8_routing

    try:
        from flashinfer.fused_moe.core import get_w2_permute_indices_with_cache
    except ImportError:
        from flashinfer.fused_moe.core import _maybe_get_cached_w2_permute_indices as get_w2_permute_indices_with_cache

    def prep(w):  # [E, rows, cols] -> shuffled BlockMajorK [E, cols/bk, rows, bk]
        cache, outs = {}, []
        for e in range(w.shape[0]):
            perm = get_w2_permute_indices_with_cache(cache, w[e].view(torch.uint8), 128)
            shuffled = w[e].view(torch.uint8)[perm.to(w.device)].contiguous().view(torch.bfloat16)
            outs.append(convert_to_block_layout(shuffled.view(torch.uint8), 128).view(torch.bfloat16))
        return torch.stack(outs).contiguous()

    def setup(fc1, fc2):
        return {"fc1": prep(fc1), "fc2": prep(fc2), "packed": {}}

    def run(st, hidden, routing, probs):
        # This FlashInfer takes packed int32 routing: expert_id << 16 | bf16 probability bits.
        key = (routing.data_ptr(), tuple(routing.shape), probs.data_ptr())
        packed = st["packed"].get(key)
        if packed is None:
            packed = st["packed"][key] = pack_routed_mxfp8_routing(routing, probs)
        out = fm.trtllm_bf16_routed_moe(
            topk_ids=packed, hidden_states=hidden,
            gemm1_weights=st["fc1"], gemm2_weights=st["fc2"], num_experts=E_GLOBAL, top_k=TOPK,
            n_group=None, topk_group=None, intermediate_size=N, local_expert_offset=0,
            local_num_experts=E_LOCAL, routed_scaling_factor=None, routing_method_type=1,
            use_shuffled_weight=True, weight_layout=WeightLayout.BlockMajorK, do_finalize=True,
            activation_type=ActivationType.Relu2.value)
        return out[0] if isinstance(out, (list, tuple)) else out

    return setup, run


def backend_fi_mxfp8():
    from flashinfer import mxfp8_quantize, shuffle_matrix_a
    from flashinfer.fused_moe.core import ActivationType

    from megatron.core.inference.moe import flashinfer_mxfp8 as fim

    def prep(w):  # [E, rows, cols] bf16 -> TRT-LLM Major-K MXFP8 stack
        e, rows, cols = w.shape
        datas, scales = [], []
        for i in range(e):
            q, s = mxfp8_quantize(w[i].contiguous(), False)
            datas.append(shuffle_matrix_a(q.view(torch.uint8), 128).view(q.dtype))
            logical = s.view(torch.uint8).reshape(rows, cols // 32)
            scales.append(fim._shuffle_routed_scale(logical))
        return fim.FlashInferRoutedMXFP8Weight(
            data=torch.stack(datas).contiguous(), scale=torch.stack(scales).contiguous(),
            logical_rows=rows, logical_cols=cols, padded_rows=rows, padded_cols=cols)

    def setup(fc1, fc2):
        return {"fc1": prep(fc1), "fc2": prep(fc2)}

    def run(st, hidden, routing, probs):
        return fim.flashinfer_routed_mxfp8_moe(hidden, routing, probs, st["fc1"], st["fc2"],
                                               num_experts=E_GLOBAL, local_expert_offset=0,
                                               activation_type=ActivationType.Relu2.value)

    return setup, run


BACKENDS = {
    "vllm_triton": (backend_vllm_triton, 2),
    "fi_cutlass_bf16": (backend_fi_cutlass_bf16, 2),
    "fi_trtllm_bf16": (backend_fi_trtllm_bf16, 2),
    "fi_mxfp8": (backend_fi_mxfp8, 1),  # bytes per weight
}


def time_graph(fn, reps=30):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(CALLS):
            fn()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / CALLS)
    return statistics.median(ts)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", type=int, nargs="+", default=[2 ** i for i in range(14)])
    p.add_argument("--backends", nargs="+", default=["vllm_triton", "fi_cutlass_bf16", "fi_trtllm_bf16"],
                   help=f"any of {list(BACKENDS)} (fi_mxfp8 is opt-in)")
    args = p.parse_args()

    torch.manual_seed(0)
    fc1 = (torch.randn(E_LOCAL, N, K, device="cuda") * K ** -0.5).bfloat16()
    fc2 = (torch.randn(E_LOCAL, K, N, device="cuda") * N ** -0.5).bfloat16()
    expert_mb = (N * K * 2 * 2) / 1e6
    print(f"GPU {torch.cuda.get_device_name()}; Ultra EP64 shape: {E_LOCAL} local of {E_GLOBAL} experts, "
          f"top-{TOPK}, K={K}, N={N}; {expert_mb:.0f} MB bf16 per expert (fc1+fc2); {CALLS} calls/graph")

    states, runs = {}, {}
    for name in args.backends:
        make, _ = BACKENDS[name]
        try:
            setup, run = make()
            states[name], runs[name] = setup(fc1, fc2), run
        except Exception as e:  # noqa: BLE001
            print(f"  skip {name}: {type(e).__name__}: {e}")

    # Correctness at a mid size.
    hidden, routing, probs = make_inputs(256, seed=1)
    ref = reference(hidden, routing, probs, fc1, fc2)
    for name in list(runs):
        try:
            with torch.inference_mode():
                out = runs[name](states[name], hidden, routing, probs)
            err = rel_err(out[: hidden.shape[0]], ref)
            print(f"  {name:16s} rel err vs fp32 reference: {err:.3e}")
            if err > (0.05 if name == "fi_mxfp8" else 0.02):
                print(f"  drop {name}: output does not match the reference")
                runs.pop(name)
        except Exception as e:  # noqa: BLE001
            print(f"  drop {name}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
            runs.pop(name)

    names = list(runs)
    print(f"\nper MoE call, us (achieved TB/s over hit local experts; % of {PEAK_TBS} TB/s)")
    print(f"{'tokens':>6} {'pairs':>6} {'hit':>3} | " + " | ".join(f"{n:>22}" for n in names))
    for t in args.tokens:
        hidden, routing, probs = make_inputs(t, seed=t)
        local = routing < E_LOCAL
        pairs, hit = int(local.sum()), int(torch.unique(routing[local]).numel())
        cells = []
        for name in names:
            bpw = BACKENDS[name][1]
            try:
                with torch.inference_mode():
                    us = time_graph(lambda: runs[name](states[name], hidden, routing, probs))
                tbs = hit * N * K * 2 * bpw / (us * 1e-6) / 1e12
                cells.append(f"{us:8.1f} ({tbs:4.1f}, {100 * tbs / PEAK_TBS:3.0f}%)")
            except Exception as e:  # noqa: BLE001
                cells.append(f"{'err ' + type(e).__name__:>22}")
                if t == args.tokens[0]:
                    print(f"  {name} timing error: {type(e).__name__}: {str(e)[:300]}")
        print(f"{t:6d} {pairs:6d} {hit:3d} | " + " | ".join(f"{c:>22}" for c in cells))


if __name__ == "__main__":
    main()
