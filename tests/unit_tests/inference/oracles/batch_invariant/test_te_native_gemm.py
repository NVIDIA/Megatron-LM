# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for the `te_native` batch-invariant backend: cuBLASLt workspace starvation.

Kernel: `megatron/core/transformer/custom_layers/batch_invariant_kernels.py`
(`_enable_te_native_workspace_starvation` / `_disable_te_native_workspace_starvation`, applied by
`enable_batch_invariant_mode(backend="te_native")`, the default `batch_invariant_backend`).
It replaces no GEMM: it pins `CUBLASLT_WORKSPACE_SIZE=0` and patches TE's cuBLAS workspace size
to 1 KiB, so that cuBLAS(Lt) cannot select split-K reductions and picks one K-reduction recipe
for every M. Inference reaches it through every dense GEMM of a batch-invariant model on the
default backend: `torch.matmul(x, w.t())` in the inference-optimized linears (aten mm),
`F.linear` with a bias (aten addmm), TE linears, and the MoE router's fp32-output TE GEMM.

Contract (written from the backend's documentation in `enable_batch_invariant_mode` and
`_enable_te_native_workspace_starvation`):
- Enabled before the first GEMM (as `megatron/training/initialize.py` does, before distributed
  initialization), every dense bf16 GEMM above keeps its native kernels and is batch invariant:
  each output row's bits depend only on that row of the input and on the weight, not on the
  number of rows M or the row's position. Values are those of a GEMM with fp32 accumulation.
- While enabled: `CUBLASLT_WORKSPACE_SIZE` is "0" whatever it was before (pinned, not
  defaulted), TE's workspace size is 1 KiB, and neither `aten::mm`/`aten::addmm` nor TE's
  `general_gemm` / RMSNorm are replaced. Disabling restores the previous environment value and
  TE's workspace function.

Reference: fp64 `x @ w^T (+ bias)` for the probe rows. Trusted implementation: the same call in
the test process outside batch-invariant mode (cuBLAS(Lt) with its default workspace). Tolerance
class: bit-exact for batch invariance and the pinned state; floating point per row (default
factor 2) for bf16 values; exact on integer inputs for every entry (the fp32-output router GEMM
is checked only this way: its summation order is what the starvation changes).

Structure (read after the contract): the environment pin acts on cuBLAS(Lt) handles created
after it, so batch invariance is checked in a fresh process (`_CHILD`) that enables the backend
before any GEMM, exactly as production does, then runs each GEMM entry point on batches of
M = 1 ... 8192 rows (B-1/B/B+1 of 16, 32, 64, 128, 256 and production sizes) with four probe
rows at the start, an odd middle offset and the end. Shapes: the attention projection
(4096 x 4096), the FFN down projection (14336 -> 4096; large K, where split-K is most
attractive to a heuristic), and the router (7168 -> 256, 4096 -> 128, fp32
output).

Main vs open PRs: every case here passes on main. A bias-free aten mm (`torch.matmul(x, w.t())`,
the inference-optimized linears' GEMM) is not batch invariant under te_native on H100 on main
(finding O8a-M8, pdx 7986825: 26 of 16384 elements differ at M = 17 vs 4, K = N = 4096), so it
is not an entry here. #8708 (n15) runs native aten GEMMs on cuBLASLt with a starved workspace,
with `test_native_aten_gemms_batch_invariant` in `transformer/test_te_layers_batch_invariant.py`.

Hardware: H100 (1 GPU), with Transformer Engine.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik
from tests.unit_tests.inference.oracles import oracle_utils as ou

pytestmark = [ou.requires_sm(ou.HOPPER)]

_DEVICE = "cuda"
_REPO = Path(__file__).resolve().parents[5]

PROBE_ROWS = 4
M_SIZES = sorted(
    {1, 2, 3, PROBE_ROWS}
    | set(ou.boundary_sizes(16, 32, 64, 128, 256, multiples=(), production=(1000, 4096, 8192)))
    - {0}
)
TE_NATIVE_WORKSPACE_BYTES = 1024

# entry -> GEMM shapes (k, n): the weight is [n, k]
CASES = {
    "aten_addmm": [(4096, 4096), (14336, 4096)],
    "te_linear": [(4096, 4096), (14336, 4096)],
    "te_router_fp32": [(7168, 256), (4096, 128)],
}


def _starts(m: int):
    if m < PROBE_ROWS:
        return [0]
    return sorted({0, min((m - PROBE_ROWS) // 2 | 1, m - PROBE_ROWS), m - PROBE_ROWS})


def _tensor(shape, seed, scale=1.0):
    """CPU-generated bf16 values (identical in the test and the child process)."""
    g = ou.make_generator(seed)
    return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)


def _case_tensors(k, n, seed):
    w = _tensor((n, k), seed, k**-0.5)
    bias = _tensor((n,), seed + 1)
    probe = _tensor((PROBE_ROWS, k), seed + 2)
    return w, bias, probe


def _integers(shape, seed):
    """Integers in [-7, 7] (identical in the child): every product and every sum over K <= 14336
    is an integer below 2**20, exact in fp32 in any summation order."""
    g = ou.make_generator(seed)
    return torch.randint(-7, 8, shape, generator=g).to(torch.bfloat16)


# The child process: enables te_native before any GEMM, then runs one entry point. It receives
# the entry, the device, the output path and the cases as arguments, and regenerates the inputs
# from their seeds with the same CPU generator as `_tensor`.
_CHILD = r"""
import json, os, sys
import torch
from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik

entry, device, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
spec = json.loads(sys.argv[4])
status = {"cuda_initialized_before_enable": torch.cuda.is_initialized()}
bik.enable_batch_invariant_mode(backend="te_native")
torch.cuda.set_device(device)
import transformer_engine.pytorch as te
import transformer_engine.pytorch.cpp_extensions.gemm as te_gemm_mod
from megatron.core.extensions.transformer_engine import te_general_gemm

status["env"] = os.environ.get("CUBLASLT_WORKSPACE_SIZE")
status["te_workspace"] = int(te_gemm_mod.get_cublas_workspace_size_bytes())


def tensor(shape, seed, scale=1.0):
    g = torch.Generator()
    g.manual_seed(seed)
    return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16).cuda()


def integers(shape, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randint(-7, 8, shape, generator=g).to(torch.bfloat16).cuda()


def starts(m):
    p = spec["probe_rows"]
    return [0] if m < p else sorted({0, min((m - p) // 2 | 1, m - p), m - p})


results = {}
with torch.no_grad():
    for case in spec["cases"]:
        k, n, seed = case["k"], case["n"], case["seed"]
        w = tensor((n, k), seed, k ** -0.5)
        bias = tensor((n,), seed + 1)
        probe = tensor((spec["probe_rows"], k), seed + 2)
        filler = tensor((max(spec["m"]), k), seed + 3)
        if entry == "aten_addmm":
            run = lambda x: torch.nn.functional.linear(x, w, bias)
        elif entry == "te_linear":
            module = te.Linear(k, n, bias=False, params_dtype=torch.bfloat16, device="cuda")
            module.weight.copy_(w)
            module.eval()
            run = module
        else:
            run = lambda x: te_general_gemm(w, x, torch.float32, layout="TN")[0]
        rows = {}
        for m in spec["m"]:
            for start in starts(m):
                x = filler[:m].clone()
                p = min(spec["probe_rows"], m)
                x[start : start + p] = probe[:p]
                rows[f"{m}:{start}"] = run(x)[start : start + p].cpu()
        results[case["name"]] = rows
        # Exact values: small integers, whose products and sums are exact in fp32 in any order.
        w = integers((n, k), seed + 4)
        bias = integers((n,), seed + 5)
        if entry == "te_linear":
            module.weight.copy_(w)
        results[case["name"] + ":exact"] = run(integers((spec["probe_rows"], k), seed + 6)).cpu()
bik.disable_batch_invariant_mode()
torch.save({"status": status, "results": results}, out_path)
"""


def _run_child(entry, out_path):
    spec = {
        "probe_rows": PROBE_ROWS,
        "m": M_SIZES,
        "cases": [
            {"name": f"k{k}-n{n}", "k": k, "n": n, "seed": 100 + i}
            for i, (k, n) in enumerate(CASES[entry])
        ],
    }
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_REPO), env.get("PYTHONPATH", "")])
    env.pop("CUBLASLT_WORKSPACE_SIZE", None)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _CHILD,
            entry,
            str(torch.cuda.current_device()),
            str(out_path),
            json.dumps(spec),
        ],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=1500,
    )
    assert proc.returncode == 0, f"child exited {proc.returncode}:\n{proc.stderr[-4000:]}"
    return spec, torch.load(out_path)


def gemm_reference(x, w, bias=None):
    out = x @ w.t()
    return out if bias is None else out + bias


def _trusted(entry, probe, w, bias):
    """The same entry point in this process, outside batch-invariant mode."""
    with torch.no_grad():
        if entry == "aten_addmm":
            return F.linear(probe, w, bias)
        if entry == "te_linear":
            import transformer_engine.pytorch as te

            module = te.Linear(
                w.shape[1], w.shape[0], bias=False, params_dtype=torch.bfloat16, device=_DEVICE
            ).eval()
            module.weight.copy_(w)
            return module(probe)
        from megatron.core.extensions.transformer_engine import te_general_gemm

        return te_general_gemm(w, probe, torch.float32, layout="TN")[0]


@pytest.mark.parametrize("entry", list(CASES))
def test_gemms_are_batch_invariant(entry, tmp_path):
    """In a process that enabled te_native before its first GEMM: four probe rows give the same
    bits at every M and position, for each shape; the workspace is pinned; the probe rows'
    values are right against fp64 (bf16 outputs bounded per row by the same call outside the
    mode), and on integer inputs every output is exact (fp32 sums of integers are exact in any
    order; rounded once to bf16). (A bias-free aten mm is not an entry: finding O8a-M8, #8708.)"""
    spec, got = _run_child(entry, tmp_path / f"{entry}.pt")
    status = got["status"]
    assert status["env"] == "0", f"CUBLASLT_WORKSPACE_SIZE is {status['env']!r} in the child"
    assert status["te_workspace"] == TE_NATIVE_WORKSPACE_BYTES, f"TE workspace {status}"
    for case in spec["cases"]:
        rows = got["results"][case["name"]]
        base = rows[f"{PROBE_ROWS}:0"]
        for key, out in rows.items():
            where = "M={} start={}".format(*key.split(":"))
            ou.assert_exact(out, base[: out.shape[0]], what=f"{entry} {case['name']} {where}")
        k, n, seed = case["k"], case["n"], case["seed"]
        w, bias, probe = (t.to(_DEVICE) for t in _case_tensors(k, n, seed))
        use_bias = bias if entry == "aten_addmm" else None
        if base.dtype != torch.float32:
            # A bf16 output is dominated by its one rounding. An fp32 output over K = 7168 is
            # dominated by the summation order, which the starved workspace changes (no split-K):
            # it measured 6.7x the full-workspace cuBLASLt error on a correct GEMM (pdx 7986825),
            # so fp32 values are checked only exactly, on the integer inputs below.
            ou.assert_close_to_reference(
                base.to(_DEVICE),
                ou.fp64_reference(gemm_reference, probe, w, use_bias),
                _trusted(entry, probe, w, use_bias),
                dim=-1,
                what=f"{entry} {case['name']} values",
            )
        w_int, x_int = _integers((n, k), seed + 4).to(_DEVICE), _integers((PROBE_ROWS, k), seed + 6)
        bias_int = _integers((n,), seed + 5).to(_DEVICE) if entry == "aten_addmm" else None
        exact = got["results"][case["name"] + ":exact"]
        ref = ou.fp64_reference(gemm_reference, x_int.to(_DEVICE), w_int, bias_int)
        ou.assert_exact(
            exact.to(_DEVICE), ref.to(exact.dtype), what=f"{entry} {case['name']} exact"
        )


def test_enable_pins_and_disable_restores(monkeypatch):
    """te_native pins the environment (over a preset value) and TE's workspace size, replaces no
    GEMM or norm, and disabling restores the preset value and TE's function."""
    import transformer_engine.pytorch as te
    import transformer_engine.pytorch.cpp_extensions as te_cpp
    import transformer_engine.pytorch.cpp_extensions.gemm as te_gemm_mod

    monkeypatch.setenv("CUBLASLT_WORKSPACE_SIZE", "4096")
    size_fn = te_gemm_mod.get_cublas_workspace_size_bytes
    general_gemm = te_cpp.general_gemm
    rmsnorm_forward = te.RMSNorm.forward
    calls = []
    monkeypatch.setattr(bik, "matmul_persistent", lambda *a, **k: calls.append("triton"))
    a = torch.randn(8, 64, device=_DEVICE).to(torch.bfloat16)
    with bik.set_batch_invariant_mode(True, backend="te_native"):
        assert os.environ["CUBLASLT_WORKSPACE_SIZE"] == "0", "the preset value was not pinned"
        assert te_gemm_mod.get_cublas_workspace_size_bytes() == TE_NATIVE_WORKSPACE_BYTES
        assert te_cpp.general_gemm is general_gemm, "te_native replaced TE's general_gemm"
        assert te.RMSNorm.forward is rmsnorm_forward, "te_native replaced TE's RMSNorm"
        torch.mm(a, a.t())
    assert calls == [], "te_native replaced aten::mm"
    assert os.environ["CUBLASLT_WORKSPACE_SIZE"] == "4096", "the preset value was not restored"
    assert te_gemm_mod.get_cublas_workspace_size_bytes is size_fn, "TE's size fn not restored"
