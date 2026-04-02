"""Diff cross-GPU tensor dumps to localize where B300 vs H100 first disagree.

Usage:
    python diff_dumps.py dump_b300.pt dump_h100.pt
"""
import sys
import torch


def main():
    if len(sys.argv) != 3:
        print("usage: python diff_dumps.py <a.pt> <b.pt>", file=sys.stderr)
        sys.exit(1)

    a = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    b = torch.load(sys.argv[2], map_location="cpu", weights_only=False)

    common = sorted(set(a.keys()) & set(b.keys()))
    only_a = sorted(set(a.keys()) - set(b.keys()))
    only_b = sorted(set(b.keys()) - set(a.keys()))

    if only_a:
        print(f"only in {sys.argv[1]}: {only_a}")
    if only_b:
        print(f"only in {sys.argv[2]}: {only_b}")

    print(f"\n{'key':<40s} {'shape':<25s} {'bitwise':>10s} {'max_abs':>14s} {'mean_abs':>14s} {'nonzero%':>10s}")
    print("-" * 115)
    for k in common:
        ta, tb = a[k], b[k]
        if not isinstance(ta, torch.Tensor) or not isinstance(tb, torch.Tensor):
            continue
        if ta.shape != tb.shape:
            print(f"{k:<40s} shape mismatch: {tuple(ta.shape)} vs {tuple(tb.shape)}")
            continue
        # Exact bitwise check: torch.equal compares dtype + every element exactly.
        bitwise = "✅" if torch.equal(ta, tb) else "❌"
        d = (ta.float() - tb.float()).abs()
        print(f"{k:<40s} {str(tuple(ta.shape)):<25s} {bitwise:>10s} "
              f"{d.max().item():>14.6e} {d.mean().item():>14.6e} "
              f"{100.0 * (d > 0).float().mean().item():>9.4f}%")


if __name__ == "__main__":
    main()
