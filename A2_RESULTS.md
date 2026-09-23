# A2 bounded MoE return: training result

The branch uses the ordinary training `MoEAlltoAllTokenDispatcher` with BF16, NCCL all-to-all, EP2 or EP4, and TP/ETP/CP/PP=1. It replaces the full return receive and sender inverse-sort buffers with bounded communication staging and a compact peer-order index. The baseline and candidate use the same frozen Megatron `main` commit `c711dc0a7f4f7460045cae18ec2f9ff5f4b92b5e`, model inputs, dtype, and expert implementation. Tests ran on 8×RTX 5090 hardware with the existing Python/PyTorch/Transformer Engine environment; no packages or drivers were changed.

Each time is the median of the slower EP2 rank over 20 complete optimizer steps after five warmups. The two independent launch pairs reversed arm order. Speedup is native time divided by candidate time, so values below 1 mean a slowdown. Peak memory is the maximum allocated across the two ranks and all measured steps. These formal v5 runs used GPUs 6–7.

| 2-layer GPT, BF16, E16, top-k4 | Native ms, launches 0 / 1 | v5 ms, launches 0 / 1 | Speedup, launches 0 / 1 | Native peak MiB, launches 0 / 1 | v5 peak MiB, launches 0 / 1 |
|---|---:|---:|---:|---:|---:|
| H1024, 4096 tokens, q=2048 | 66.51 / 48.64 | 67.02 / 73.33 | 0.992× / 0.663× | 2296.78 / 2296.67 | 2296.57 / 2296.44 |
| H512, 8192 tokens, q=512 | 63.86 / 64.04 | 97.63 / 92.83 | 0.654× / 0.690× | 1410.16 / 1410.16 | 1410.42 / 1410.42 |

The large configuration has substantial launch-to-launch time variation, so its two pairs do not support a speedup claim. The long configuration consistently slowed down. Forward-only peaks did fall: 2282.92 / 2282.83 to 2270.08 / 2270.08 MiB for the large case, and 1227.28 / 1227.24 to 1203.26 / 1203.28 MiB for the long case. The complete training peak occurs later, inside grouped-expert backward, after return backward has finished. Sequential-expert and activation-recompute probes did not move the complete-model peak into the return path. The current branch therefore does **not** meet the full-step peak-memory objective and is not ready as a memory-optimization PR.

The isolated return operator passed 36 forward/backward cases per rank on both EP2 and EP4, including FP64, FP32, BF16, skewed/empty splits, and expert spans crossing chunk boundaries. A 20-step EP4 fixed-route full-model comparison matched native initial parameter hashes, routes, every loss, first-step gradients, and final parameters exactly on all four ranks. The BF16 accumulator matches the native unpermute path. No pretrained weights were downloaded, and temporary parameter/gradient snapshots were deleted after comparison. Eight-rank model training remains untested because other users occupied part of the node; their processes were not interrupted.
