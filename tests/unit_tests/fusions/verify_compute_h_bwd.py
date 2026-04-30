"""Standalone verification of compute_h backward formulas.

Pure PyTorch — no cuTile needed. Compares hand-derived gradients against autograd.
"""

import math

import torch

torch.manual_seed(42)
DEVICE = "cpu"
DTYPE = torch.float64  # float64 for precise gradient checking


def forward_proj_rms_compute_h(x, weight, alpha_pre, alpha_post, alpha_res, bias, n, eps):
    """Forward: proj_rms + compute_h. All differentiable."""
    proj = x @ weight.t()  # [M, N]
    norm = x.norm(dim=-1, keepdim=True)  # [M, 1]
    K = x.shape[-1]
    N = proj.shape[-1]
    r = norm / math.sqrt(K)  # [M, 1]

    # Build alpha vector from 3 scalars
    alpha = torch.cat([
        alpha_pre.expand(n), alpha_post.expand(n), alpha_res.expand(N - 2 * n),
    ])

    # compute_h
    h = proj * alpha.unsqueeze(0) / (r + eps) + bias.unsqueeze(0)
    h_pre = h[..., :n].sigmoid()
    h_post = h[..., n:2 * n].sigmoid() * 2
    h_res = h[..., 2 * n:]
    y = torch.cat([h_pre, h_post, h_res], dim=-1)
    return y, r, proj, norm


def manual_backward_compute_h(
    grad_y, grad_r_ext, y_activated, proj, r,
    alpha_pre, alpha_post, alpha_res, n, eps,
):
    """Hand-derived backward through compute_h + r conversion.

    Returns: grad_proj, grad_r_for_bwd, grad_alpha_pre, grad_alpha_post, grad_alpha_res, grad_bias
    """
    M, N = proj.shape

    # Build alpha vector
    alpha = torch.cat([
        alpha_pre.expand(n), alpha_post.expand(n), alpha_res.expand(N - 2 * n),
    ])

    # 1. Activation backward -> grad_h
    y_pre = y_activated[:, :n]
    y_post = y_activated[:, n:2 * n]
    gy_pre = grad_y[:, :n]
    gy_post = grad_y[:, n:2 * n]
    gy_res = grad_y[:, 2 * n:]

    gh_pre = gy_pre * y_pre * (1.0 - y_pre)
    gh_post = gy_post * (y_post / 2.0) * (1.0 - y_post / 2.0) * 2.0
    gh_res = gy_res

    grad_h = torch.cat([gh_pre, gh_post, gh_res], dim=-1)

    # 2. Linear backward: h = proj * alpha / (r + eps) + bias
    r_eps = r + eps
    alpha_row = alpha.unsqueeze(0)

    grad_proj = grad_h * alpha_row / r_eps
    grad_r_from_h = -(grad_h * proj * alpha_row / (r_eps * r_eps)).sum(dim=-1, keepdim=True)

    # Per-column grad_alpha, then reduce to 3 scalars
    ga_all = (grad_h * proj / r_eps).sum(dim=0)  # [N]
    grad_alpha_pre = ga_all[:n].sum()
    grad_alpha_post = ga_all[n:2 * n].sum()
    grad_alpha_res = ga_all[2 * n:].sum()
    grad_bias = grad_h.sum(dim=0)

    # 3. Total grad_r = from compute_h + external
    grad_r_total = grad_r_from_h
    if grad_r_ext is not None:
        grad_r_total = grad_r_total + grad_r_ext

    # 4. Convert grad_r from r_new=norm/sqrt(K) to r_old=1/(r_new+eps)
    grad_r_for_bwd = grad_r_total * (-r_eps * r_eps)

    return grad_proj, grad_r_for_bwd, grad_alpha_pre, grad_alpha_post, grad_alpha_res, grad_bias


def manual_backward_proj_rms(grad_proj, grad_r_for_bwd, x, weight, norm, eps):
    """Hand-derived backward through proj_rms (reference, no kernel).

    grad_r_for_bwd is w.r.t. r_old = 1/(norm/sqrt(K) + eps).
    """
    K = x.shape[-1]
    inv_sqrt_k = 1.0 / math.sqrt(K)

    # grad through proj = x @ weight.T
    grad_x_from_proj = grad_proj @ weight  # [M, K]
    grad_weight = grad_proj.t() @ x  # [N, K]

    # grad through r_old = 1/(norm/sqrt(K) + eps)
    # d(r_old)/d(norm) = -1/(norm/sqrt(K)+eps)^2 * (1/sqrt(K))
    # d(norm)/d(x_i) = x_i / norm
    u = norm * inv_sqrt_k + eps
    coeff = -1.0 / (u * u) * inv_sqrt_k  # [M, 1]
    inv_norm = torch.where(norm > 0, 1.0 / norm, torch.zeros_like(norm))
    grad_x_from_r = grad_r_for_bwd * coeff * x * inv_norm  # [M, K]

    grad_x = grad_x_from_proj + grad_x_from_r
    return grad_x, grad_weight


def test_backward(M, n, K, eps=1e-6):
    """Compare manual backward against autograd."""
    N = n * n + 2 * n

    x = torch.randn(M, K, dtype=DTYPE, device=DEVICE, requires_grad=True)
    weight = torch.randn(N, K, dtype=DTYPE, device=DEVICE, requires_grad=True)
    alpha_pre = torch.randn(1, dtype=DTYPE, device=DEVICE, requires_grad=True)
    alpha_post = torch.randn(1, dtype=DTYPE, device=DEVICE, requires_grad=True)
    alpha_res = torch.randn(1, dtype=DTYPE, device=DEVICE, requires_grad=True)
    bias = torch.randn(N, dtype=DTYPE, device=DEVICE, requires_grad=True)

    grad_y = torch.randn(M, N, dtype=DTYPE, device=DEVICE)
    grad_r_ext = torch.randn(M, 1, dtype=DTYPE, device=DEVICE)

    # ---- Autograd path ----
    y_auto, r_auto, _, _ = forward_proj_rms_compute_h(
        x, weight, alpha_pre, alpha_post, alpha_res, bias, n, eps,
    )
    loss = (y_auto * grad_y).sum() + (r_auto * grad_r_ext).sum()
    loss.backward()

    auto_grad_x = x.grad.clone()
    auto_grad_w = weight.grad.clone()
    auto_grad_ap = alpha_pre.grad.clone()
    auto_grad_apo = alpha_post.grad.clone()
    auto_grad_ar = alpha_res.grad.clone()
    auto_grad_b = bias.grad.clone()

    # ---- Manual path ----
    x2 = x.detach().clone()
    w2 = weight.detach().clone()
    ap2 = alpha_pre.detach().clone()
    apo2 = alpha_post.detach().clone()
    ar2 = alpha_res.detach().clone()
    b2 = bias.detach().clone()

    y_man, r_man, proj_man, norm_man = forward_proj_rms_compute_h(
        x2, w2, ap2, apo2, ar2, b2, n, eps,
    )

    # Step 1: manual compute_h backward
    (
        grad_proj, grad_r_for_bwd,
        grad_alpha_pre, grad_alpha_post, grad_alpha_res,
        grad_bias,
    ) = manual_backward_compute_h(
        grad_y, grad_r_ext, y_man, proj_man, r_man, ap2, apo2, ar2, n, eps,
    )

    # Step 2: manual proj_rms backward
    grad_x_man, grad_w_man = manual_backward_proj_rms(
        grad_proj, grad_r_for_bwd, x2, w2, norm_man, eps,
    )

    # ---- Compare ----
    def check(name, manual, auto, atol=1e-10, rtol=1e-7):
        diff = (manual - auto).abs().max().item()
        rel = diff / (auto.abs().max().item() + 1e-15)
        ok = diff < atol or rel < rtol
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {name:14s}  max_abs_diff={diff:.3e}  rel={rel:.3e}")
        return ok

    print(f"\n--- M={M}, n={n}, K={K}, N={N} ---")
    all_ok = True
    all_ok &= check("grad_x", grad_x_man, auto_grad_x)
    all_ok &= check("grad_weight", grad_w_man, auto_grad_w)
    all_ok &= check("grad_alpha_pre", grad_alpha_pre, auto_grad_ap)
    all_ok &= check("grad_alpha_post", grad_alpha_post, auto_grad_apo)
    all_ok &= check("grad_alpha_res", grad_alpha_res, auto_grad_ar)
    all_ok &= check("grad_bias", grad_bias, auto_grad_b)
    return all_ok


if __name__ == "__main__":
    print("Verifying manual backward formulas against autograd (float64)...")
    results = []
    for M, n, K in [(32, 2, 64), (64, 4, 256), (16, 3, 128), (128, 4, 512)]:
        results.append(test_backward(M, n, K))

    print("\n" + "=" * 50)
    if all(results):
        print("ALL PASSED")
    else:
        print("SOME FAILED")
