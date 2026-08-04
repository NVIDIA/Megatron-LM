"""Check the fused padding sentinel against the standalone mask kernel.

The fused path writes -1 from inside the selection kernel; the reference runs the
selection and then the separate mask launch. Agreement has to hold for a full row
set, a partially padded row set, and the degenerate all-padding case, because a
silent disagreement routes CUDA-graph padding tokens to real experts.
"""

import os

import torch

os.environ.setdefault("MCORE_ROUTER_FUSED_TOPK", "1")
os.environ["MCORE_FUSED_ROUTE_MASK"] = "1"

from megatron.core.inference.moe import router_topk as rt
from megatron.core.transformer.moe.inference_routing_mask_kernel import mask_routing_padding

TOKENS, EXPERTS, TOPK = 256, 128, 8


def reference(logits, real_cnt):
    """Unfused answer: selection, then the standalone mask launch."""
    probs, idx = rt.fused_softmax_topk(logits, TOPK, mask_padding=False)
    cnt = torch.tensor([real_cnt], dtype=torch.int32, device=logits.device)
    mask_routing_padding(idx, cnt, 0)
    return probs, idx


def main():
    torch.manual_seed(0)
    dev = "cuda"
    logits = torch.randn(TOKENS, EXPERTS, dtype=torch.float32, device=dev)

    print(f"{'real_cnt':>9} {'idx exact':>10} {'probs max|d|':>13} {'pad rows all -1':>16}")
    ok = True
    for real_cnt in (TOKENS, 200, 137, 1, 0):
        cnt = torch.tensor([real_cnt], dtype=torch.int32, device=dev)
        rt.publish_graph_padding(cnt)

        p_ref, i_ref = reference(logits, real_cnt)
        p_fus, i_fus = rt.fused_softmax_topk(logits, TOPK, mask_padding=True)
        torch.cuda.synchronize()

        idx_exact = bool(torch.equal(i_ref, i_fus))
        pmax = (p_ref.float() - p_fus.float()).abs().max().item()
        pad_ok = bool((i_fus[real_cnt:] == -1).all()) if real_cnt < TOKENS else True
        real_ok = bool((i_fus[:real_cnt] >= 0).all())
        tagged = getattr(i_fus, "_mcore_padding_masked", False)

        ok &= idx_exact and pad_ok and real_ok and pmax == 0.0 and tagged
        print(f"{real_cnt:>9} {str(idx_exact):>10} {pmax:>13.3e} {str(pad_ok):>16}")

    print("\nPASS" if ok else "\nFAIL")


if __name__ == "__main__":
    main()
