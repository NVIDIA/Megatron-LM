# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Does a CUDA event pair recorded *inside* a graph capture yield usable per-replay
timings? Everything in the in-situ attribution plan rests on yes, and nothing in the
decode workload can answer it cheaply, so answer it here in isolation first.

Four things have to hold:

1. ``event.record()`` during capture must not raise, and must become a graph node
   rather than a no-op.
2. After a replay plus a sync, ``start.elapsed_time(end)`` must return the duration
   of *that replay's* enclosed work, not the capture-time value and not garbage.
3. The number must track a known ground truth. Two matmuls with a ~4x size ratio
   give a ratio to check against, and each pair is also compared to the same op
   timed eagerly.
4. Repeated replays must agree, so a single readback is a usable sample.

Run: ``python dev/moe_fused/insitu_probe.py``
"""

import sys

import torch


def eager_ms(fn, iters=50):
    """Median-ish eager duration, warmup excluded."""
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters


def main():
    if not torch.cuda.is_available():
        print("FAIL: no CUDA device")
        return 1
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    print(f"device: {torch.cuda.get_device_name(dev)}  torch {torch.__version__}")

    # Two very different sizes, so a wrong reading is obvious rather than plausible.
    big_n, small_n = 4096, 2048
    a_b = torch.randn(big_n, big_n, device=dev, dtype=torch.bfloat16)
    b_b = torch.randn(big_n, big_n, device=dev, dtype=torch.bfloat16)
    a_s = torch.randn(small_n, small_n, device=dev, dtype=torch.bfloat16)
    b_s = torch.randn(small_n, small_n, device=dev, dtype=torch.bfloat16)
    out_b = torch.empty(big_n, big_n, device=dev, dtype=torch.bfloat16)
    out_s = torch.empty(small_n, small_n, device=dev, dtype=torch.bfloat16)

    big = lambda: torch.matmul(a_b, b_b, out=out_b)
    small = lambda: torch.matmul(a_s, b_s, out=out_s)

    eb, es = eager_ms(big), eager_ms(small)
    print(f"eager:  big {eb:.4f} ms   small {es:.4f} ms   ratio {eb / es:.2f}x")

    # Events must exist before capture; only the *record* belongs to the graph.
    e = {k: torch.cuda.Event(enable_timing=True) for k in ("bs", "be", "ss", "se")}

    # Warm the workspace/autotune caches outside capture, or the first replay pays
    # for allocations that the steady state does not.
    for _ in range(5):
        big()
        small()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(s):
            with torch.cuda.graph(g):
                e["bs"].record()
                big()
                e["be"].record()
                e["ss"].record()
                small()
                e["se"].record()
    except Exception as ex:
        print(f"FAIL(1): recording an event during capture raised: {type(ex).__name__}: {ex}")
        return 1
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    print("PASS(1): event.record() inside capture did not raise")

    # A pre-replay read would prove nothing either way, so go straight to replays.
    rows = []
    for i in range(12):
        g.replay()
        torch.cuda.synchronize()
        try:
            bms = e["bs"].elapsed_time(e["be"])
            sms = e["ss"].elapsed_time(e["se"])
        except Exception as ex:
            print(f"FAIL(2): elapsed_time after replay raised: {type(ex).__name__}: {ex}")
            return 1
        rows.append((bms, sms))
        if i < 4:
            print(f"  replay {i}: big {bms:.4f} ms   small {sms:.4f} ms")

    bs = [r[0] for r in rows[2:]]
    ss = [r[1] for r in rows[2:]]
    bmed = sorted(bs)[len(bs) // 2]
    smed = sorted(ss)[len(ss) // 2]
    bspread = (max(bs) - min(bs)) / bmed * 100
    sspread = (max(ss) - min(ss)) / smed * 100
    print(f"PASS(2): elapsed_time works after replay")
    print(f"in-graph: big {bmed:.4f} ms   small {smed:.4f} ms   ratio {bmed / smed:.2f}x")
    print(f"spread over 10 replays: big {bspread:.1f}%   small {sspread:.1f}%")

    ok = True
    # Absolute agreement with eager, which is the claim that matters: these numbers
    # are going to be quoted as component costs.
    for name, ing, eag in (("big", bmed, eb), ("small", smed, es)):
        err = abs(ing - eag) / eag * 100
        verdict = "PASS" if err < 15 else "FAIL"
        if err >= 15:
            ok = False
        print(f"{verdict}(3,{name}): in-graph {ing:.4f} vs eager {eag:.4f} ms -> {err:.1f}% error")

    if bspread > 10 or sspread > 10:
        print(f"FAIL(4): replay-to-replay spread too wide to sample once")
        ok = False
    else:
        print(f"PASS(4): replays agree within 10%, a single readback is a usable sample")

    # The whole point is per-site attribution, so the sum of the parts must also
    # reconstruct a whole-graph measurement.
    wall_s, wall_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall_s.record()
    for _ in range(20):
        g.replay()
    wall_e.record()
    torch.cuda.synchronize()
    whole = wall_s.elapsed_time(wall_e) / 20
    parts = bmed + smed
    err = abs(whole - parts) / whole * 100
    verdict = "PASS" if err < 20 else "FAIL"
    if err >= 20:
        ok = False
    print(f"{verdict}(5): whole graph {whole:.4f} ms vs summed sites {parts:.4f} ms -> {err:.1f}%")

    print("\nVERDICT: " + ("USABLE" if ok else "NOT USABLE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
