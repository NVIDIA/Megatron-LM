# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Can an *external* event record inside a graph be timed? (probe 2 said plain ones cannot.)

``insitu_probe.py`` established that ``torch.cuda.Event.record()`` during capture
becomes a graph node without raising, but that ``elapsed_time`` on it afterwards
fails with ``cudaErrorInvalidValue``. That is CUDA behaving as documented: an
ordinary event-record node exists only for intra-graph ordering, and carries no
timestamp the host can read.

CUDA's answer is ``cudaEventRecordWithFlags(event, stream, cudaEventRecordExternal)``.
An external event-record node updates the event on every replay, so the host can
synchronize on it and time against it -- which is precisely what per-component
attribution inside the decode graph needs. PyTorch's ``Event.record()`` takes no
flags, so the call is made through ctypes on the runtime, using the raw handles
torch already exposes (``Event.cuda_event``, ``Stream.cuda_stream``).

Two details this probe exists to pin down, both of which would silently produce
wrong numbers rather than errors:

1. torch creates the underlying ``cudaEvent_t`` lazily on first record, so
   ``cuda_event`` is null until the event has been recorded once *outside* capture.
2. The external flag must be passed on the capturing stream, not the default one.

Same acceptance criteria as probe 2: no raise, readable after replay, agreement
with eager timing, stability across replays, and parts that reconstruct the whole.

Run: ``python dev/moe_fused/insitu_probe3.py``
"""

import ctypes
import sys

import torch

CUDA_EVENT_RECORD_EXTERNAL = 1


def load_runtime():
    for name in ("libcudart.so", "libcudart.so.13", "libcudart.so.12", "libcudart.so.11.0"):
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        fn = lib.cudaEventRecordWithFlags
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
        fn.restype = ctypes.c_int
        return lib, fn, name
    return None, None, None


def eager_ms(fn, iters=50):
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

    lib, record_ext, libname = load_runtime()
    if record_ext is None:
        print("FAIL(0): could not load libcudart / cudaEventRecordWithFlags")
        return 1
    print(f"PASS(0): loaded {libname}, cudaEventRecordWithFlags resolved")

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

    e = {k: torch.cuda.Event(enable_timing=True) for k in ("bs", "be", "ss", "se")}
    # Force the lazy cudaEvent_t into existence *before* capture, or cuda_event is
    # null and the ctypes call would record onto nothing.
    for ev in e.values():
        ev.record()
    torch.cuda.synchronize()
    handles = {}
    for k, ev in e.items():
        h = ev.cuda_event
        if not h:
            print(f"FAIL(0b): event {k} still has a null handle after an eager record")
            return 1
        handles[k] = ctypes.c_void_p(h)
    print("PASS(0b): all four events have live handles")

    for _ in range(5):
        big()
        small()
    torch.cuda.synchronize()

    def rec(k, stream_ptr):
        rc = record_ext(handles[k], stream_ptr, CUDA_EVENT_RECORD_EXTERNAL)
        if rc != 0:
            raise RuntimeError(f"cudaEventRecordWithFlags({k}) returned {rc}")

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(s):
            with torch.cuda.graph(g, stream=s):
                # Must be the *capturing* stream: an external record on any other
                # stream is cudaErrorIllegalState (401). torch.cuda.graph() uses its
                # own side stream unless one is passed, so read it back from inside
                # rather than assuming, which is how the first attempt went wrong.
                sp = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
                rec("bs", sp)
                big()
                rec("be", sp)
                rec("ss", sp)
                small()
                rec("se", sp)
    except Exception as ex:
        print(f"FAIL(1): external record during capture failed: {type(ex).__name__}: {ex}")
        return 1
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    print("PASS(1): external event records captured into the graph")

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
    print("PASS(2): elapsed_time works after replay")
    print(f"in-graph: big {bmed:.4f} ms   small {smed:.4f} ms   ratio {bmed / smed:.2f}x")
    print(f"spread over 10 replays: big {bspread:.1f}%   small {sspread:.1f}%")

    ok = True
    for name, ing, eag in (("big", bmed, eb), ("small", smed, es)):
        err = abs(ing - eag) / eag * 100
        v = "PASS" if err < 15 else "FAIL"
        ok = ok and err < 15
        print(f"{v}(3,{name}): in-graph {ing:.4f} vs eager {eag:.4f} ms -> {err:.1f}% error")

    if bspread > 10 or sspread > 10:
        print("FAIL(4): replay-to-replay spread too wide to sample once")
        ok = False
    else:
        print("PASS(4): replays agree within 10%")

    ws, we = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ws.record()
    for _ in range(20):
        g.replay()
    we.record()
    torch.cuda.synchronize()
    whole = ws.elapsed_time(we) / 20
    parts = bmed + smed
    err = abs(whole - parts) / whole * 100
    v = "PASS" if err < 20 else "FAIL"
    ok = ok and err < 20
    print(f"{v}(5): whole graph {whole:.4f} ms vs summed sites {parts:.4f} ms -> {err:.1f}%")

    print("\nVERDICT: " + ("USABLE" if ok else "NOT USABLE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
