"""Compare two tp_fwd.py dumps (CPU-only) for cross-TP parity.

Reports per-layer hidden-state max-abs/max-rel diff, final-logit max-abs/max-rel
diff, and greedy-token match. Use to gate V1-V5 (DUT vs TP=1 oracle).

    python3 compare_logits.py --ref logits_tp1.pt --dut logits_tp2sp.pt
"""
import argparse

import torch


def _stats(a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    max_abs = d.max().item()
    denom = b.abs().clamp_min(1e-6)
    max_rel = (d / denom).max().item()
    mean_abs = d.mean().item()
    return max_abs, mean_abs, max_rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="oracle dump (TP=1)")
    ap.add_argument("--dut", required=True, help="DUT dump (TP>1 +/- SP)")
    ap.add_argument("--logits-band", type=float, default=3.0,
                    help="max-abs logits tolerance (bf16 band; TP1-vs-HF was 1.7)")
    ap.add_argument("--layer-mean-band", type=float, default=0.15,
                    help="per-layer mean-abs tolerance (HYBRID worst was 0.11)")
    ap.add_argument("--real-len", type=int, default=0,
                    help="if >0, evaluate greedy match on only the first N (real, non-pad) "
                         "positions; SP needs seq%%TP==0 so inputs are padded, and pad "
                         "positions are meaningless for greedy parity.")
    args = ap.parse_args()

    ref = torch.load(args.ref, map_location="cpu", weights_only=False)
    dut = torch.load(args.dut, map_location="cpu", weights_only=False)

    print(f"REF tp={ref.get('tp')} sp={ref.get('sp')} te={ref.get('te')}")
    print(f"DUT tp={dut.get('tp')} sp={dut.get('sp')} te={dut.get('te')}")

    # Per-layer hidden states (if both captured). The DECISIVE metric is mean_abs
    # (max_rel blows up on near-zero entries via the 1e-6 denom clamp — not meaningful;
    # max_abs is O(1) bf16-reduction-order band). A bug shows as mean_abs that GROWS with
    # TP degree or a shape mismatch / sudden jump at one layer.
    worst_layer_mean = 0.0
    layer_shape_ok = True
    if "layers" in ref and "layers" in dut:
        print("\nPer-layer hidden-state diff (idx: max_abs | mean_abs | max_rel):")
        for k in sorted(ref["layers"]):
            if k not in dut["layers"]:
                continue
            ra, rb = ref["layers"][k], dut["layers"][k]
            if ra.shape != rb.shape:
                print(f"  L{k}: SHAPE MISMATCH ref={tuple(ra.shape)} dut={tuple(rb.shape)}")
                layer_shape_ok = False
                continue
            ma, mn, mr = _stats(ra, rb)
            worst_layer_mean = max(worst_layer_mean, mn)
            flag = "  <-- mean_abs over band" if mn > args.layer_mean_band else ""
            print(f"  L{k:2d}: {ma:.3e} | {mn:.3e} | {mr:.3e}{flag}")
        print(f"\nworst per-layer mean_abs = {worst_layer_mean:.3e} (band {args.layer_mean_band})")

    # Final logits.
    logits_ok = True
    rl, dl = ref["logits"], dut["logits"]
    if rl.shape == dl.shape:
        ma, mn, mr = _stats(rl, dl)
        print(f"\nFinal logits: max_abs={ma:.3e} mean_abs={mn:.3e} max_rel={mr:.3e} "
              f"(band {args.logits_band})")
        logits_ok = ma <= args.logits_band
    else:
        print(f"\nFinal logits SHAPE MISMATCH ref={tuple(rl.shape)} dut={tuple(dl.shape)}")
        logits_ok = False
        ma = float("inf")

    # Greedy. Restrict to real (non-pad) positions when --real-len is given.
    rg = rl.argmax(dim=-1)
    dg = dl.argmax(dim=-1)
    if args.real_len:
        rg = rg[:, : args.real_len]
        dg = dg[:, : args.real_len]
    exact = torch.equal(rg, dg)
    print(f"\nGreedy ref = {rg.tolist()}")
    print(f"Greedy dut = {dg.tolist()}")
    print(f"GREEDY EXACT = {exact}  (real-len={args.real_len or 'all'})")

    # Verdict: greedy-exact (decisive) + logits within band + per-layer mean within band.
    layer_ok = layer_shape_ok and worst_layer_mean <= args.layer_mean_band
    verdict = exact and logits_ok and layer_ok
    print(f"\n=== PARITY VERDICT: {'PASS' if verdict else 'FAIL'} ===")
    print(f"  greedy_exact={exact}  logits_within_band={logits_ok} (max_abs {ma:.3e})  "
          f"per_layer_within_band={layer_ok} (worst_mean {worst_layer_mean:.3e})")


if __name__ == "__main__":
    main()
