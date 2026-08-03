#!/usr/bin/env python3
"""Decode-attention bake-off at the Qwen3-30B-A3B shapes: FA2 vs flashinfer trtllm-gen.

The matched per-category budget (GAP-S18) puts mcore 0.409 ms/step behind vLLM on
attention with the same launch count, which points at kernel generation rather than at
anything Megatron does around the call: FA2's kvcache kernel is SM80-era, while vLLM runs
flashinfer's trtllm-gen `fmhaSm100f` on Blackwell. Both packages are installed here, so
the question is answerable without touching `attention.py`.

Three things this measures that an e2e run cannot separate:

  1. **Is the trtllm-gen kernel actually faster at *our* shapes** (B=256, 32 q heads,
     4 kv heads, D=128, and a KV length that grows to ~1 k during the OSL=1024 run),
     or only at the small-batch shapes decode benchmarks usually quote.
  2. **Does it accept mcore's 256-token pages.** Megatron defaults to
     `block_size_tokens=256`; trtllm-gen is usually exercised at 16-64, so a rejection
     here would mean the integration also needs a page-size change -- much larger scope.
  3. **Do the two agree numerically**, checked before either timing is believed.

Device time is measured under graph replay, not in a launch loop: one Python-side launch
of either kernel is tens of microseconds, the same order as the kernel, so a plain loop
reports host time and both arms look identical (this is exactly how the first fp8 GEMM
sweep concluded "no change" -- see FP8-WEIGHTS-S18).

Usage: python harness_attn.py [--seqlens 512,1024] [--pages 256,128,64]
"""

import argparse
import logging
import os

import torch

B = 256  # requests in flight at the benchmarked regime
HQ = 32  # q heads at TP=1
HKV = 4  # num_query_groups
D = 128  # kv_channels
DTYPE = torch.bfloat16


logging.getLogger("flashinfer").setLevel(logging.WARNING)


def graph_time(fn, iters=50):
    """Device time per call, measured under graph replay. Returns microseconds."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / iters


def make_cache(seqlen, page):
    """Paged KV in the NHD layout Megatron already uses: [pages, page, HKV, D]."""
    pages_per_seq = (seqlen + page - 1) // page
    num_pages = B * pages_per_seq
    k = torch.randn(num_pages, page, HKV, D, dtype=DTYPE, device="cuda")
    v = torch.randn(num_pages, page, HKV, D, dtype=DTYPE, device="cuda")
    block_table = (
        torch.arange(num_pages, dtype=torch.int32, device="cuda").reshape(B, pages_per_seq)
    )
    seqlens = torch.full((B,), seqlen, dtype=torch.int32, device="cuda")
    return k, v, block_table, seqlens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlens", default="512,1024,2048")
    ap.add_argument("--pages", default="256,128,64")
    args = ap.parse_args()

    from flash_attn import flash_attn_with_kvcache

    try:
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache

        import flashinfer

        print(f"flashinfer {getattr(flashinfer, '__version__', '?')}")
    except Exception as exc:  # noqa: BLE001
        print(f"flashinfer unavailable: {exc}")
        trtllm_batch_decode_with_kv_cache = None

    scale = D**-0.5
    # trtllm-gen wants a zeroed scratch buffer; size it once, generously, outside any
    # timed region. 128 MiB is what vLLM allocates for this path.
    ws = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    print(f"\n{'seqlen':>7} {'page':>5} {'FA2 us':>9} {'trtllm us':>11} {'delta':>8}  note")
    for seqlen in [int(x) for x in args.seqlens.split(",")]:
        for page in [int(x) for x in args.pages.split(",")]:
            k, v, bt, sl = make_cache(seqlen, page)
            q4 = torch.randn(B, 1, HQ, D, dtype=DTYPE, device="cuda")  # FA2: (B,S,H,D)
            q3 = q4.reshape(B, HQ, D)  # flashinfer: (tokens,H,D)

            def fa2():
                return flash_attn_with_kvcache(
                    q=q4,
                    k_cache=k,
                    v_cache=v,
                    cache_seqlens=sl,
                    block_table=bt,
                    softmax_scale=scale,
                    causal=True,
                    num_splits=0,
                )

            # FA2's paged path requires page % 256 == 0, so smaller pages are
            # trtllm-only arms: the reference then comes from the 256-page run.
            if page % 256 == 0:
                t_fa2 = graph_time(fa2)
                ref = fa2().reshape(B, HQ, D).float()
            else:
                t_fa2, ref = float("nan"), None

            t_fi, note = float("nan"), ""
            if trtllm_batch_decode_with_kv_cache is not None:
                def fi():
                    return trtllm_batch_decode_with_kv_cache(
                        query=q3,
                        kv_cache=(k, v),
                        workspace_buffer=ws,
                        block_tables=bt,
                        seq_lens=sl,
                        max_seq_len=seqlen,
                        bmm1_scale=scale,
                        bmm2_scale=1.0,
                        kv_layout="NHD",
                    )

                try:
                    got = fi().float()
                    if ref is not None:
                        md = (got - ref).abs().max().item()
                        rel = md / ref.abs().max().item()
                        note = f"max|d|={md:.4f} rel={rel:.2e}"
                        if rel > 2e-2:
                            note += " MISMATCH"
                    else:
                        note = "trtllm-only (FA2 needs page%256==0)"
                    t_fi = graph_time(fi)
                except Exception as exc:  # noqa: BLE001
                    note = f"rejected: {str(exc).splitlines()[0][:70]}"

            ok = t_fi == t_fi and t_fa2 == t_fa2
            delta = f"{100*(t_fi-t_fa2)/t_fa2:+.1f}%" if ok else "-"
            print(f"{seqlen:>7} {page:>5} {t_fa2:>9.1f} {t_fi:>11.1f} {delta:>8}  {note}")
            del k, v, bt, sl, q4, q3
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
