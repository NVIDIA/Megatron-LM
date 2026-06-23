"""Diagnose per-position greedy flips between an oracle and DUT logits dump.

For each real position, prints the oracle's top-2 (token, logit) and the gap, plus
the DUT's logit for the oracle-argmax vs the DUT-argmax. A near-tie (gap within the
bf16 band) that flips is benign reduction-order noise; a large gap that flips is a bug.

    python3 diag_pos.py --ref logits_tp1_pad.pt --dut logits_tp4sp_pad.pt --real-len 10
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--dut", required=True)
    ap.add_argument("--real-len", type=int, default=10)
    args = ap.parse_args()

    ref = torch.load(args.ref, map_location="cpu", weights_only=False)["logits"].float()[0]
    dut = torch.load(args.dut, map_location="cpu", weights_only=False)["logits"].float()[0]
    S = min(args.real_len, ref.shape[0], dut.shape[0])

    print(f"pos | ref top1 (tok:logit) | ref top2 | ref gap | dut@ref_top1 | dut top1 | flip?")
    n_flip = 0
    for p in range(S):
        rv, ri = ref[p].topk(2)
        dtop_v, dtop_i = dut[p].max(0)
        ref_top1_tok = ri[0].item()
        dut_at_ref = dut[p, ref_top1_tok].item()
        gap = (rv[0] - rv[1]).item()
        flip = ref_top1_tok != dtop_i.item()
        if flip:
            n_flip += 1
        print(f"{p:3d} | {ref_top1_tok}:{rv[0]:.4f} | {ri[1].item()}:{rv[1]:.4f} | "
              f"{gap:.4f} | {dut_at_ref:.4f} | {dtop_i.item()}:{dtop_v:.4f} | "
              f"{'FLIP' if flip else ''}")
    print(f"\nreal-position greedy flips: {n_flip}/{S}")
    if n_flip:
        print("Interpretation: a flip with ref_gap within the bf16 band (~O(0.1-2)) and a "
              "near-equal dut logit for both tokens is benign reduction-order noise; a flip "
              "with a large ref_gap is a real divergence.")


if __name__ == "__main__":
    main()
