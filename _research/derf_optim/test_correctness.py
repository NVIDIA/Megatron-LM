"""Forward + backward correctness gate for Derf/DyT optimization options.

Run as ``python -m _research.derf_optim.test_correctness --option 1``.
Asserts each candidate fused implementation matches a pure-Python (Derf|DyT) +
nn.Linear reference within 1e-3 relative on forward output, parameter grads,
and input grad. Run before launching a throughput sbatch.
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F


def _eager_derf_linear(x, weight, w_norm, b_norm, alpha, s, lin_bias):
    dtype = x.dtype
    pre = w_norm.to(dtype) * torch.erf(alpha.to(dtype) * x + s.to(dtype)) + b_norm.to(dtype)
    return F.linear(pre, weight, lin_bias)


def _eager_dyt_linear(x, weight, w_norm, b_norm, alpha, lin_bias):
    dtype = x.dtype
    pre = w_norm.to(dtype) * torch.tanh(alpha.to(dtype) * x) + b_norm.to(dtype)
    return F.linear(pre, weight, lin_bias)


def _check(name, ref, got, rtol=1e-3, atol=1e-4):
    diff = (ref - got).abs()
    rel = diff / (ref.abs() + atol)
    max_rel = rel.max().item()
    max_abs = diff.max().item()
    ok = max_rel < rtol or max_abs < atol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:24s}  max_abs={max_abs:.2e}  max_rel={max_rel:.2e}")
    return ok


def _make_params(seed, hidden, output, dtype, has_s):
    torch.manual_seed(seed)
    weight = torch.randn(output, hidden, dtype=dtype) * 0.02
    w_norm = torch.ones(hidden, dtype=dtype)
    b_norm = torch.zeros(hidden, dtype=dtype)
    alpha = torch.tensor(0.5, dtype=dtype)
    s = torch.tensor(0.0, dtype=dtype) if has_s else None
    return weight, w_norm, b_norm, alpha, s


def _run_pair(eager_fn, compiled_fn, has_s, label, seq, batch, hidden, output, dtype):
    weight, w_norm, b_norm, alpha, s = _make_params(0, hidden, output, dtype, has_s)
    torch.manual_seed(1)
    x_init = torch.randn(seq, batch, hidden, dtype=dtype)

    def _forward(fn):
        params = {
            "weight":  weight.detach().clone().requires_grad_(True),
            "w_norm":  w_norm.detach().clone().requires_grad_(True),
            "b_norm":  b_norm.detach().clone().requires_grad_(True),
            "alpha":   alpha.detach().clone().requires_grad_(True),
        }
        if has_s:
            params["s"] = s.detach().clone().requires_grad_(True)
        x = x_init.detach().clone().requires_grad_(True)
        if has_s:
            out = fn(x, params["weight"], params["w_norm"], params["b_norm"], params["alpha"], params["s"], None)
        else:
            out = fn(x, params["weight"], params["w_norm"], params["b_norm"], params["alpha"], None)
        out.sum().backward()
        return out, x, params

    eager_out, eager_x, eager_p = _forward(eager_fn)
    cmp_out,   cmp_x,   cmp_p   = _forward(compiled_fn)

    all_pass = True
    all_pass &= _check(f"{label} forward",       eager_out,            cmp_out)
    all_pass &= _check(f"{label} dx",            eager_x.grad,         cmp_x.grad)
    all_pass &= _check(f"{label} dw_norm",       eager_p["w_norm"].grad, cmp_p["w_norm"].grad)
    all_pass &= _check(f"{label} db_norm",       eager_p["b_norm"].grad, cmp_p["b_norm"].grad)
    all_pass &= _check(f"{label} d_alpha",       eager_p["alpha"].grad,  cmp_p["alpha"].grad)
    if has_s:
        all_pass &= _check(f"{label} d_s",       eager_p["s"].grad,      cmp_p["s"].grad)
    all_pass &= _check(f"{label} d_weight_lin",  eager_p["weight"].grad, cmp_p["weight"].grad)
    return all_pass


def test_option1(seq=64, batch=2, hidden=128, output=192, dtype=torch.float32):
    print(f"[option1] dtype={dtype} seq={seq} batch={batch} hidden={hidden} output={output}")
    from _research.derf_optim.option1_compile import _compiled_derf_linear, _compiled_dyt_linear

    ok = True
    ok &= _run_pair(_eager_derf_linear, _compiled_derf_linear, True,  "Derf", seq, batch, hidden, output, dtype)
    ok &= _run_pair(_eager_dyt_linear,  _compiled_dyt_linear,  False, "DyT",  seq, batch, hidden, output, dtype)
    print(f"[option1] {'OK' if ok else 'FAIL'}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--option", choices=["1", "2", "3"], default="1")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    args = p.parse_args()

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    if args.option == "1":
        ok = test_option1(dtype=dtype)
    else:
        raise NotImplementedError(f"option {args.option} test not yet wired")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
