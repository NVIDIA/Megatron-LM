# A2 bounded MoE return: training result

The branch uses the ordinary training `MoEAlltoAllTokenDispatcher` with BF16, NCCL all-to-all, EP2 or EP4, and TP/ETP/CP/PP=1. It replaces the full return receive and sender inverse-sort buffers with bounded communication staging and a compact peer-order index. The baseline and candidate use the same frozen Megatron `main` commit `c711dc0a7f4f7460045cae18ec2f9ff5f4b92b5e`, model inputs, dtype, and expert implementation. Tests ran on 8×RTX 5090 hardware with the existing Python/PyTorch/Transformer Engine environment; no packages or drivers were changed.

Each time is the median of the slower EP2 rank over 20 complete optimizer steps after five warmups. The two independent launch pairs reversed arm order. Speedup is native time divided by candidate time, so values below 1 mean a slowdown. Peak memory is the maximum allocated across the two ranks and all measured steps. These formal v5 runs used GPUs 6–7.

| 2-layer GPT, BF16, E16, top-k4 | Native ms, launches 0 / 1 | v5 ms, launches 0 / 1 | Speedup, launches 0 / 1 | Native peak MiB, launches 0 / 1 | v5 peak MiB, launches 0 / 1 |
|---|---:|---:|---:|---:|---:|
| H1024, 4096 tokens, q=2048 | 66.51 / 48.64 | 67.02 / 73.33 | 0.992× / 0.663× | 2296.78 / 2296.67 | 2296.57 / 2296.44 |
| H512, 8192 tokens, q=512 | 63.86 / 64.04 | 97.63 / 92.83 | 0.654× / 0.690× | 1410.16 / 1410.16 | 1410.42 / 1410.42 |

The large configuration has substantial launch-to-launch time variation, so its two pairs do not support a speedup claim. The long configuration consistently slowed down. Forward-only peaks did fall: 2282.92 / 2282.83 to 2270.08 / 2270.08 MiB for the large case, and 1227.28 / 1227.24 to 1203.26 / 1203.28 MiB for the long case. Finer steady-state probes later corrected the peak location: it occurs after the second layer's expert backward, between its attention-output backward boundary and the first layer's MoE postprocess backward boundary. Sequential-expert and activation-recompute probes did not move the complete-model peak into the return path. The current branch therefore does **not** meet the full-step peak-memory objective and is not ready as a memory-optimization PR.

A follow-up chunk-size sweep on the H512 / 8192-token shape found q=4096 much faster than q=512. Two new reversed-order, 5-warmup / 20-measured-step pairs on GPUs 6–7 gave:

| H512, 8192 tokens, BF16, EP2 | Native, launches 0 / 1 | q=4096, launches 0 / 1 | Speedup, launches 0 / 1 | Native / candidate peak allocated MiB | Native / candidate peak reserved MiB |
|---|---:|---:|---:|---:|---:|
| Complete optimizer step | 64.36 / 63.02 ms | 67.69 / 65.55 ms | 0.951× / 0.961× | 1410.16 / 1410.42 | 1822 / 1774 |

Forward allocated peak fell from 1227.12 / 1227.05 to 1217.21 / 1217.22 MiB. The larger chunk cuts the earlier q=512 slowdown to 4–5%, but does not produce a complete-step allocated-peak saving or a speedup over native. q=2048, 6144, and 8192 pilots were slower than q=4096. A corresponding H1024 / 4096-token screening pair was speed-neutral with worse reserved memory; its second pair ran out of GPU memory and is excluded. Further round-major packing variants reduced GPU operation count but were slower in EP1/EP2 pilots; an EP4 fixed-route 20-step multi-round variant showed BF16 accumulation-order drift (first-gradient relative L2 about 0.0059), so neither variant replaced this branch. The experimental failure does not change the exact EP4 q=512 comparison above.

An asynchronous double-buffer prototype passed isolated EP1/EP2 forward/backward tests and had one encouraging short pilot. Three reversed-order pairs of 30 measured optimizer steps after five warmups on GPUs 5–6 rejected it:

| H512, 8192 tokens, BF16, EP2, q=2048 | Native, launches 0 / 1 / 2 | Double buffer, launches 0 / 1 / 2 | Speedup, launches 0 / 1 / 2 | Native / candidate peak allocated MiB | Native / candidate peak reserved MiB |
|---|---:|---:|---:|---:|---:|
| Complete optimizer step | 56.95 / 58.40 / 57.98 ms | 68.74 / 67.95 / 67.82 ms | 0.828× / 0.860× / 0.855× | 1410.16 / 1410.42 | 1822 / 1774 |

Its forward allocated peak fell by about 10 MiB, but the full-step allocated peak did not. The short pilot was not representative, and the double-buffer code was not integrated.

A follow-up tried the existing Transformer Engine expert weight-gradient accumulation path on both arms, leaving the dense layers unchanged. On the H1024 / 4096-token EP2 pilot, native and q=2048 complete-step peaks were 2304.78 and 2304.71 MiB, with 59.04 and 60.82 ms steps; it did not shift the peak into the return stage. Enabling global gradient fusion was unavailable in the existing environment because the required Apex CUDA extension is absent; no packages were installed. A GroupedTensor/single-weight EP1 screen also raised peak allocation rather than reducing it. Native expert-wgrad overlap raised the H1024 / 4096-token EP2 peak to 3036.99 MiB in its pilot. These are diagnostic screens, not an A2 speed or memory claim.

The requested expert-backward extension was screened on H512 / 8192 tokens, BF16, EP2, GPUs 5–6 (three warmups and ten complete optimizer steps). Speedup is the native median of the slower rank divided by the candidate median of the slower rank; these are single-launch diagnostics, not a performance claim.

| Expert-backward mode | Native step | Candidate step | Speedup | Native / candidate peak allocated |
|---|---:|---:|---:|---:|
| Delay expert wgrad until after dispatch backward, on the current stream | 63.70 ms | 67.82 ms | 0.939× | 1410.16 / 1426.16 MiB |
| Accumulate expert wgrad into main gradients | 63.70 ms | 64.11 ms | 0.994× | 1410.16 / 1412.16 MiB |

Both variants failed to lower the complete-step peak. A separate three-warmup/one-step two-rank probe put the 1410.2 / 1385.3 MiB peaks between the second layer's attention-output backward event (1213.3 / 1235.4 MiB peak so far) and the first layer's MoE postprocess backward event. The peak was already reached before the first layer's expert FC2 backward began on both ranks. The earlier attribution to grouped-expert backward was too broad; the data only localize the peak to the attention-side interval. A subsequent probe on GPUs 5–6 was invalidated when another workload filled GPU 6, so it is excluded. No process was terminated.

Additional attention submodule hooks narrowed that interval: on both ranks the high-water mark was still 1213.3 / 1235.5 MiB at the second layer's attention-core output backward boundary, then reached 1410.2 / 1385.4 MiB by its QKV output backward boundary. The peak is therefore inside the attention-core backward interval, before QKV linear backward. These hooks are diagnostic and were disabled in timing runs.

The v9 implementation creates the peer-order index with one contiguous range and one concatenation instead of launching a small range fill for every expert/peer span. It preserves the same communication plan and BF16 accumulation order. On GPUs 2 and 4 (GPU 4 also hosted a separate memory profiler), two reversed-order EP2 comparisons used H512, 8192 tokens, q=4096, five warmups and 20 complete optimizer steps. Speedup below is v5 time divided by v9 time; both arms had the same 1410.42 MiB complete-step allocated peak.

| Return index construction | v5 step ms | v9 step ms | v9 speedup over v5 | v5 / v9 full-step peak |
|---|---:|---:|---:|---:|
| Launch pair 0, v5 then v9 | 71.59 | 68.66 | 1.043× | 1410.42 / 1410.42 MiB |
| Launch pair 1, v9 then v5 | 69.23 | 64.43 | 1.074× | 1410.42 / 1410.42 MiB |

Against native on the same GPUs, v9 measured 64.91 vs 64.47 ms (0.993×) in one launch pair and 71.56 vs 63.87 ms (0.893×) in the reversed pair. The variable shared-node load prevents a claim that v9 has reached native speed. Native peak allocated was 1410.16 MiB and v9 was 1410.42 MiB in both pairs; forward peak fell by about 10 MiB. A one-layer H512 / 8192-token screen similarly reduced forward peak by 11.74 MiB but left the complete-step 971.78 MiB peak unchanged, while step time increased from 37.97 to 40.78 ms. A direct forward concatenation prototype (v10) passed the isolated EP2 checks but was slower than v9, 69.39 vs 66.88 ms, so it was rejected.

The v9 isolated EP2 and EP4 operators passed all 36 FP64/FP32/BF16 forward/backward cases per rank, including tails and empty peers. A deterministic fixed-route BF16 EP2 comparison over 20 complete optimizer steps found v9 and v5 identical on both ranks: initial weights, routes, every loss, first gradients and final parameters all matched exactly. Native versus v9 at q=512 had first-gradient relative L2 about 0.0055 and final-parameter relative L2 about 1.7e-5. The same multi-round difference was present in v5; the earlier exact native/v5 EP4 comparison used a one-round return. Temporary snapshot tensors were deleted after the comparisons.

The isolated return operator passed 36 forward/backward cases per rank on both EP2 and EP4, including FP64, FP32, BF16, skewed/empty splits, and expert spans crossing chunk boundaries. A 20-step EP4 fixed-route full-model comparison matched native initial parameter hashes, routes, every loss, first-step gradients, and final parameters exactly on all four ranks. The BF16 accumulator matches the native unpermute path. No pretrained weights were downloaded, and temporary parameter/gradient snapshots were deleted after comparison. Eight-rank model training remains untested because other users occupied part of the node; their processes were not interrupted.
