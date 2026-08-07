"""Measure the achievable HBM streaming ceiling on this GPU.

Every memory-bound decision gate in this campaign divides a byte count by a
bandwidth constant. The constant in use (6.081 TB/s) predates this session and is
the denominator behind the MoE grouped-GEMM floor of 49.66 us/layer/rank that
vLLM appears to beat -- so it needs to be a measured machine constant, not a
datasheet number or an inherited one.

Three probes, in increasing fidelity to the thing being modelled:
  1. pure read   -- reduction over one large buffer
  2. read+write  -- copy between two buffers
  3. MoE-shaped  -- read the actual per-layer expert weight footprint as the
                    grouped GEMM sees it: `local_experts` separate tensors of
                    [ffn, hidden] and [ffn, hidden] gate/up plus [hidden, ffn] down
"""

import torch

GiB = 1024**3


def timed(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters / 1e3  # seconds


def report(label, seconds, nbytes):
    tbps = nbytes / seconds / 1e12
    print(f"  {label:44s} {seconds*1e6:9.2f} us  {nbytes/1e6:9.2f} MB  {tbps:6.3f} TB/s")
    return tbps


def main():
    dev = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"  SMs={props.multi_processor_count}  total_mem={props.total_memory/GiB:.1f} GiB")
    print()

    results = {}

    print("1. pure read (reduction over one buffer)")
    for gib in (1, 2, 4):
        n = int(gib * GiB // 2)
        x = torch.empty(n, dtype=torch.bfloat16, device=dev)
        x.normal_()
        t = timed(lambda: x.sum())
        results[f"read_{gib}GiB"] = report(f"read {gib} GiB bf16", t, n * 2)
        del x
        torch.cuda.empty_cache()
    print()

    print("2. read+write (copy between buffers)")
    for gib in (1, 2):
        n = int(gib * GiB // 2)
        a = torch.empty(n, dtype=torch.bfloat16, device=dev)
        a.normal_()
        b = torch.empty_like(a)
        t = timed(lambda: b.copy_(a))
        results[f"copy_{gib}GiB"] = report(f"copy {gib} GiB bf16 (r+w)", t, n * 2 * 2)
        del a, b
        torch.cuda.empty_cache()
    print()

    # 3. The shape the MoE gate actually models: Qwen3-30B-A3B, EP4.
    hidden, ffn, n_local = 2048, 768, 32
    print(f"3. MoE per-layer weight footprint (hidden={hidden} ffn={ffn} "
          f"local_experts={n_local})")
    # gate+up fused = [2*ffn, hidden]; down = [hidden, ffn]
    gate_up = [torch.empty(2 * ffn, hidden, dtype=torch.bfloat16, device=dev) for _ in range(n_local)]
    down = [torch.empty(hidden, ffn, dtype=torch.bfloat16, device=dev) for _ in range(n_local)]
    for t_ in gate_up + down:
        t_.normal_()
    nbytes = sum(t_.numel() * 2 for t_ in gate_up + down)

    def read_all():
        s = 0
        for t_ in gate_up:
            s = s + t_.sum()
        for t_ in down:
            s = s + t_.sum()
        return s

    t = timed(read_all)
    tbps = report("read all local expert weights (per layer)", t, nbytes)

    # single contiguous buffer of the same size, as an upper bound on how fast
    # that footprint could possibly be streamed
    flat = torch.empty(nbytes // 2, dtype=torch.bfloat16, device=dev)
    flat.normal_()
    t2 = timed(lambda: flat.sum())
    tbps2 = report("same bytes, one contiguous buffer", t2, nbytes)

    print()
    print("=== summary ===")
    peak = max(results.values())
    print(f"  best pure-read bandwidth        : {peak:6.3f} TB/s")
    print(f"  MoE-shaped read (32 tensors x2) : {tbps:6.3f} TB/s")
    print(f"  MoE-shaped, contiguous          : {tbps2:6.3f} TB/s")
    print(f"  per-layer weight bytes          : {nbytes/1e6:.2f} MB")
    for name, bw in (("best pure read", peak), ("MoE-shaped", tbps),
                     ("MoE contiguous", tbps2)):
        print(f"  floor at {name:16s}: {nbytes/(bw*1e12)*1e6:7.2f} us/layer/rank")


if __name__ == "__main__":
    main()
