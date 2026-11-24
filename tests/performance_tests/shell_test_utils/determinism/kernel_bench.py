# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-kernel determinism perf dashboard for the cuDNN Frontend kernels.

Kernels: flash attention (SDPA) and linear attention (GDN, KDA, GDN-2, GDP), forward and
backward. The kernels come from the ``cudnn`` package installed in the container
(nvidia-cudnn-frontend >= 1.29.1 with ``cudnn.linear_attention`` and ``cudnn.sdpa``);
this file needs no cuDNN Frontend checkout.

Every kernel is timed in two modes. "det" is what ``--deterministic-mode`` sets in
``megatron/training/determinism.py``: ``torch.use_deterministic_algorithms(True)`` with
``fill_uninitialized_memory`` off. "default" is ``torch.use_deterministic_algorithms(False)``.
A kernel that is deterministic by construction shows no difference, and the dashboard
proves that by measurement.

Every measurement is one subprocess, so a kernel that crashes loses one row, not the run.
The mode that starts alternates between repetitions, so warmup and drift favor neither
side. The timer is ``torch.cuda.synchronize`` + ``perf_counter`` around the forward call
and around ``torch.autograd.grad``. Each timed call has a fixed cost of about 0.25 ms, so
a row below about 0.5 ms shows mostly that cost, not the kernel.

Usage (one GPU):
    MEGATRON_REV=$(git log --oneline -1) \\
        python tests/performance_tests/shell_test_utils/determinism/kernel_bench.py OUT [--smoke]
Writes OUT.json (with every timed sample) and OUT.md. ``MEGATRON_REV`` is optional: it
replaces the ``git log`` call when the container has no usable git.
"""

import argparse
import functools
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

SEED = 17
SEQLENS, SMOKE_SEQLENS = (2048, 8192, 32768), (4096,)
REPS, SMOKE_REPS = 5, 1  # processes per mode
WARMUP, RUNS = 5, 50  # calls per process
TIMEOUT_S = 900
MODES = ("det", "default")
PHASES = ("fwd", "bwd")
SHAPES = {
    "flash": "1 sequence of S tokens, 16 query heads, 8 key/value heads, dim 128, causal, bf16",
    "linear": "2 sequences of S tokens, 8 heads, key and value dim 128, bf16; gdp: 2 steps/token",
}


# --------------------------------------------------------------------------- worker


def _sdpa(seqlen):
    import cudnn.sdpa.fwd.torch_op as sdpa_op
    import torch

    gen = torch.Generator(device="cuda").manual_seed(SEED)

    def rand(heads):
        t = torch.randn(1, heads, seqlen, 128, device="cuda", generator=gen)
        return t.to(torch.bfloat16).requires_grad_()

    inputs = [rand(16), rand(8), rand(8)]  # q, k, v in BHSD
    return inputs, lambda: sdpa_op.sdpa(*inputs, is_causal=True, return_lse=True)[0]


def _linear(kernel, seqlen):
    import cudnn.linear_attention as la
    import torch
    import torch.nn.functional as F

    gen = torch.Generator(device="cuda").manual_seed(SEED)
    heads, dim, n = 8, 128, 2 if kernel == "gdp" else 1  # n = Householder steps per token
    total = 2 * seqlen
    cu_seqlens = torch.tensor([0, seqlen, total], dtype=torch.int32, device="cuda")

    def rand(*shape):
        return torch.randn(*shape, device="cuda", generator=gen)

    def leaf(t, dtype=torch.bfloat16):
        return t.to(dtype).requires_grad_()

    def gate(*shape):  # a value in (0, 1)
        return torch.sigmoid(rand(*shape))

    def decay(*shape):  # a log decay in (-0.1, 0)
        return leaf(-0.1 * gate(*shape), torch.float32)

    qkv = [
        leaf(F.normalize(rand(total, heads, dim), dim=-1)),
        leaf(F.normalize(rand(total * n, heads, dim), dim=-1)),
        leaf(rand(total * n, heads, dim)),
    ]
    if kernel == "gdn":  # scalar decay and scalar beta per token
        inputs = [*qkv, decay(total, heads), leaf(gate(total, heads), torch.float32)]
        return inputs, lambda: la.gated_delta_net(*inputs, cu_seqlens)[0]
    if kernel == "gdp":  # as GDN, with beta per Householder step
        inputs = [*qkv, decay(total, heads), leaf(gate(total * n, heads), torch.float32)]
        return inputs, lambda: la.gated_delta_product(*inputs, cu_seqlens, n)[0]
    if kernel == "kda":  # per-key-channel decay, scalar beta
        inputs = [*qkv, decay(total, heads, dim), leaf(gate(total, heads), torch.float32)]
        return inputs, lambda: la.kimi_delta_attention(*inputs, cu_seqlens)[0]
    assert kernel == "gdn2"  # per-key-channel decay and erase gate, per-value write gate
    gates = [leaf(gate(total, heads, dim)), leaf(gate(total, heads, dim))]
    inputs = [*qkv, decay(total, heads, dim), *gates]
    return inputs, lambda: la.gated_delta_net_v2(*inputs, cu_seqlens)[0]


# name -> (family, builder); builder(seqlen) returns (inputs, forward)
KERNELS = {
    "sdpa": ("flash", _sdpa),
    **{k: ("linear", functools.partial(_linear, k)) for k in ("gdn", "kda", "gdn2", "gdp")},
}


def worker(kernel, seqlen, mode):
    """Time one kernel in one mode. Prints one RESULT line."""
    import cudnn
    import torch

    torch.manual_seed(SEED)  # grad_out below: the same in every process
    torch.use_deterministic_algorithms(mode == "det")
    # As megatron/training/determinism.py: the call above turns the fill on, it costs a
    # kernel launch per empty allocation, and reproducibility does not need it.
    torch.utils.deterministic.fill_uninitialized_memory = False
    inputs, forward = KERNELS[kernel][1](seqlen)
    grad_out = torch.randn_like(forward())

    def timed(fn):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1e3, result

    def raw_bytes(t):
        return t.contiguous().view(torch.uint8)

    def step():
        """One forward and one backward. Returns {phase: (ms, result tensors)}."""
        fwd_ms, out = timed(forward)
        bwd_ms, grads = timed(lambda: torch.autograd.grad(out, inputs, grad_out))
        results = {"fwd": (fwd_ms, [out.detach()]), "bwd": (bwd_ms, list(grads))}
        # Did the work happen? Every output and gradient is finite and non-zero.
        for _, tensors in results.values():
            assert all(torch.isfinite(t).all() and t.ne(0).any() for t in tensors)
        return results

    # The reference comes from an untimed call, so its clones do not disturb a timed call.
    first = {phase: [t.clone() for t in tensors] for phase, (_, tensors) in step().items()}
    result = {phase: {"samples_ms": [], "replay": True} for phase in PHASES}
    for i in range(WARMUP + RUNS):
        for phase, (ms, tensors) in step().items():
            same = all(
                torch.equal(raw_bytes(a), raw_bytes(b)) for a, b in zip(first[phase], tensors)
            )
            result[phase]["replay"] &= same  # the same bytes as the reference?
            if i >= WARMUP:
                result[phase]["samples_ms"].append(ms)
        del tensors  # keep one set of results alive during the next call, not two
    for phase, tensors in first.items():  # bytes, shape and dtype, to compare processes
        sha = hashlib.sha256()
        for t in tensors:
            sha.update(f"{t.dtype}{tuple(t.shape)}".encode() + raw_bytes(t).cpu().numpy().tobytes())
        result[phase]["digest"] = sha.hexdigest()
    result["env"] = {
        "gpu": torch.cuda.get_device_name(),
        "cudnn": f"{cudnn.__version__} (backend {cudnn.backend_version()})",
        "torch": torch.__version__,
    }
    print("RESULT " + json.dumps(result))


# --------------------------------------------------------------------------- driver


def run_one(kernel, seqlen, mode):
    """Run one measurement process. Returns the worker's RESULT dict or an error."""
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", kernel, str(seqlen), mode]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {TIMEOUT_S} s"}
    if proc.returncode == 0:
        for line in reversed(proc.stdout.strip().splitlines()):
            if line.startswith("RESULT "):
                return json.loads(line[len("RESULT ") :])
    tail = (proc.stderr.strip().splitlines() or ["no RESULT line"])[-3:]
    return {"error": f"exit {proc.returncode}: " + " | ".join(tail)[-600:]}


def summarize(results, phase):
    """One mode and phase of one row, over the processes that succeeded. None if none did."""
    if not results:
        return None
    medians = [statistics.median(r[phase]["samples_ms"]) for r in results]
    return {
        "median_ms": statistics.median(medians),
        "min_ms": min(medians),
        "max_ms": max(medians),
        "processes": len(results),
        "replay": all(r[phase]["replay"] for r in results),
        # None = not tested: one process cannot differ from itself.
        "same_across_processes": (
            len({r[phase]["digest"] for r in results}) == 1 if len(results) > 1 else None
        ),
    }


def verdict(det, default):
    """A gap inside the process-to-process range is not a difference."""
    if min(det["processes"], default["processes"]) < 3:
        return "too few runs"
    if det["min_ms"] > default["max_ms"]:
        return "det slower"
    if det["max_ms"] < default["min_ms"]:
        return "det faster"
    return "no measurable difference"


def measure(kernel, seqlen, reps):
    """One dashboard row. Every row has the same keys; a missing value is None."""
    row = {"family": KERNELS[kernel][0], "kernel": kernel, "scale": f"S={seqlen}"}
    row.update({"errors": [], "env": None, "processes": []})
    for rep in range(reps):
        for mode in MODES if rep % 2 == 0 else MODES[::-1]:  # balanced starting order
            result = run_one(kernel, seqlen, mode)
            if "error" in result:
                row["errors"].append(f"{mode}: {result['error']}")
                continue
            row["env"] = result.pop("env")
            row["processes"].append({"mode": mode, **result})  # in run order
    for phase in PHASES:
        stats = {
            mode: summarize([p for p in row["processes"] if p["mode"] == mode], phase)
            for mode in MODES
        }
        both = all(stats.values())
        row[phase] = {
            **stats,
            "ratio": stats["det"]["median_ms"] / stats["default"]["median_ms"] if both else None,
            "verdict": verdict(stats["det"], stats["default"]) if both else None,
        }
    return row


def render(env, rows):
    """Markdown dashboard: one row per kernel and scale."""
    out = ["# Determinism kernel dashboard", ""]
    out += [f"- {k}: {v}" for k, v in env.items()]
    out += ["", "Times are ms: median (min-max) over processes. Ratio = det / default."]
    out += ["det = `torch.use_deterministic_algorithms(True)` with `fill_uninitialized_memory`"]
    out += ["off, as `--deterministic-mode` sets it. default = `False`."]
    out += ["Replay = `in-process, cross-process`: the mode gives the same bytes on every call"]
    out += ["in a process, and the same bytes in every process."]
    out += ["A row below about 0.5 ms shows mostly the fixed cost of one call.", ""]
    head = ["family", "kernel", "scale"]
    head += [f"{p} {x}" for p in PHASES for x in ("det", "default", "ratio", "verdict", "replay")]
    out += ["| " + " | ".join([*head, "status"]) + " |", "|" + "---|" * (len(head) + 1)]

    def yes_no(value):
        return {True: "yes", False: "NO", None: "untested"}[value]

    for r in rows:
        cells = [r["family"], r["kernel"], r["scale"]]
        for phase in PHASES:
            stats = r[phase]
            for s in (stats[mode] for mode in MODES):
                text = f"{s['median_ms']:.3f} ({s['min_ms']:.3f}-{s['max_ms']:.3f})" if s else "-"
                cells.append(text)
            cells.append(f"{stats['ratio']:.2f}x" if stats["ratio"] else "-")
            cells.append(stats["verdict"] or "-")
            replay = [
                f"{mode} {yes_no(s['replay'])}, {yes_no(s['same_across_processes'])}"
                for mode, s in ((mode, stats[mode]) for mode in MODES)
                if s
            ]
            cells.append(" / ".join(replay) or "-")
        cells.append("FAILED: " + "; ".join(r["errors"])[:160] if r["errors"] else "ok")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def megatron_rev():
    """``git log`` of this checkout, or "unknown": a missing git must not lose the run."""
    cmd = ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "log", "--oneline", "-1"]
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", nargs="?", help="output prefix: writes OUT.json and OUT.md")
    parser.add_argument("--smoke", action="store_true", help="one scale, one process per mode")
    parser.add_argument("--worker", nargs=3, metavar=("KERNEL", "SEQLEN", "MODE"))
    args = parser.parse_args()
    if args.worker:
        kernel, seqlen, mode = args.worker
        return worker(kernel, int(seqlen), mode)
    if not args.out:
        parser.error("the output prefix is required")
    reps = SMOKE_REPS if args.smoke else REPS
    rows = []
    for kernel in KERNELS:
        for seqlen in SMOKE_SEQLENS if args.smoke else SEQLENS:
            rows.append(measure(kernel, seqlen, reps))
            print(json.dumps({k: v for k, v in rows[-1].items() if k != "processes"}), flush=True)

    env = {
        **next((r["env"] for r in rows if r["env"]), {}),
        "megatron": os.environ.get("MEGATRON_REV") or megatron_rev(),
        "processes per mode": reps,
        "calls per process": f"{WARMUP} warmup + {RUNS} timed",
        **{f"{family} input": shape for family, shape in SHAPES.items()},
    }
    page = render(env, rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump({"env": env, "rows": rows}, f, indent=2)
    with open(args.out + ".md", "w") as f:
        f.write(page)
    print(page)
    failed = [r for r in rows if r["errors"]]
    sys.exit(f"{len(failed)} of {len(rows)} rows had errors" if failed else 0)


if __name__ == "__main__":
    main()
