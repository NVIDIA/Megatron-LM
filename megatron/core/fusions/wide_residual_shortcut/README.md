# Fused wide-residual shortcut seams

For wide-residual hybrid models with shortcut-connected MoE (`--moe-shortcut-wide-residual-norm-fusion`;
needs BF16 parameters, RMSNorm, 2-6 wide streams, a shared expert, no dropout/bias, no CP/PP, and a GB300
for training; the residual stream is FP32 with `--fp32-residual-connection`, otherwise BF16, and the
connection logits may be FP32 or BF16 — all arithmetic is FP32 either way), `ShortcutMoEBlock` replaces
each residual seam with one Gluon kernel that mixes the wide-residual streams with the branch update,
writes the new residual, and emits the RMSNorm readers that consume it — directly as MXFP8 for an
exclusive TE Linear consumer under an MXFP8 recipe, otherwise BF16. `wide_residual_write_read_rmsnorm_quant`
writes the Mamba/GDN/attention output and produces the shared expert's normalized input; `shortcut_add_wide_residual_write_two_reader_rmsnorm_quant`
adds the routed and shared MoE outputs (the shortcut add, passed between blocks as a `PendingShortcutAdd`) and
produces the next block's compute-layer and shortcut-routing inputs. The shortcut add is optional: a
`PendingShortcutAdd` without a shared update is a single write read twice, which a lone wide-residual Mamba
layer directly before a fused block uses (M → [* E]) so that block's two-reader kernel writes its update and
emits both inputs. The matching backward is one fused kernel per seam; with
`--residual-stream-recompute-num-layers N` the residuals are rebuilt by bit-exact checkpoint replay instead
of being saved. Residual writes are bit-identical to replay and across kernel
schedules, reader outputs differ by at most one BF16 ulp between schedules, and the tests check every
output and gradient against independent FP64/autograd references.

| boundary | updates | readers | kernel |
|---|--:|--:|---|
| lone layer → lone layer, compute → MoE inside a block, block → lone layer | 1 | 1 | `wide_residual_write_read_rmsnorm_quant` |
| block → block (shortcut add) | 2 (routed + shared) | 2 | `shortcut_add_..._two_reader_rmsnorm_quant` |
| lone layer → block (M → [* E]) | 1 | 2 | `shortcut_add_..._two_reader_rmsnorm_quant`, no shared update |

## Performance

GB300, Triton 3.7.1, 3 wide streams; kernel time per call under CUDA graphs (20 launches per graph over
rotating input copies, median of 10 replays). **Unfused** is what Megatron runs without the fusion:
`streamwise_sigmoid_writeback`, `streamwise_sigmoid_read` and TE `RMSNorm` per reader (two-reader: plus the
routed + shared add), and in MXFP8 runs TE MXFP8 quantization of reader zero (rowwise, plus columnwise in
training as a TE Linear does for its saved input); unfused backward is autograd through those ops.
**Fused** is the forward/backward kernel; MXFP8 backward also emits reader zero's columnwise operand (it is
produced by checkpoint replay instead when replay is on). %SOL = compulsory bytes / fused time / 7.926 TB/s
(forward: FP32 residual read + write, BF16 updates, reader outputs and scales, rstd; backward: residual,
carry and residual-gradient FP32 traffic, saved update, reader and update gradients, rstd, MXFP8 columns).
Below 32 tokens the forward is the inference kernel (no backward state) and MXFP8/backward do not apply
(rows must be multiples of 32; model training additionally needs multiples of 1024 tokens).

### Forward: `wide_residual_write_read_rmsnorm_quant` (one reader)

| model (hidden) | tokens | BF16 unfused us | BF16 fused us | BF16 speedup | BF16 %SOL | MXFP8 unfused us | MXFP8 fused us | MXFP8 speedup | MXFP8 %SOL |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Nano (3072) | 1 | 39.0 | 1.7 | 23.20x | 1% | – | – | – | – |
| Nano (3072) | 4 | 43.0 | 1.8 | 24.26x | 2% | – | – | – | – |
| Nano (3072) | 16 | 45.0 | 1.9 | 23.46x | 9% | – | – | – | – |
| Nano (3072) | 64 | 46.8 | 2.7 | 17.43x | 26% | 50.3 | 3.8 | 13.17x | 18% |
| Nano (3072) | 256 | 13.4 | 3.9 | 3.48x | 72% | 17.0 | 4.2 | 4.05x | 64% |
| Nano (3072) | 1K | 33.2 | 12.9 | 2.57x | 86% | 37.6 | 13.3 | 2.83x | 81% |
| Nano (3072) | 4K | 115.1 | 51.5 | 2.24x | 86% | 123.8 | 50.8 | 2.44x | 85% |
| Nano (3072) | 8K | 224.2 | 99.8 | 2.25x | 89% | 238.6 | 98.1 | 2.43x | 88% |
| Super (4096) | 1 | 38.0 | 1.7 | 21.84x | 1% | – | – | – | – |
| Super (4096) | 4 | 38.8 | 1.8 | 22.03x | 3% | – | – | – | – |
| Super (4096) | 16 | 38.7 | 1.9 | 20.20x | 12% | – | – | – | – |
| Super (4096) | 64 | 40.6 | 2.9 | 13.87x | 32% | 44.0 | 4.0 | 11.00x | 22% |
| Super (4096) | 256 | 13.5 | 5.6 | 2.39x | 66% | 17.2 | 6.0 | 2.88x | 60% |
| Super (4096) | 1K | 34.4 | 18.2 | 1.89x | 81% | 38.7 | 18.5 | 2.09x | 77% |
| Super (4096) | 4K | 128.0 | 68.3 | 1.88x | 87% | 138.6 | 66.4 | 2.09x | 86% |
| Super (4096) | 8K | 240.6 | 133.6 | 1.80x | 89% | 261.5 | 129.0 | 2.03x | 89% |
| Ultra (10240) | 1 | 43.4 | 2.0 | 21.65x | 2% | – | – | – | – |
| Ultra (10240) | 4 | 47.0 | 2.0 | 23.21x | 7% | – | – | – | – |
| Ultra (10240) | 16 | 47.8 | 2.1 | 22.47x | 27% | – | – | – | – |
| Ultra (10240) | 64 | 60.5 | 6.9 | 8.82x | 34% | 63.5 | 6.4 | 9.92x | 35% |
| Ultra (10240) | 256 | 35.6 | 10.5 | 3.40x | 88% | 39.5 | 11.5 | 3.45x | 78% |
| Ultra (10240) | 1K | 111.7 | 45.7 | 2.44x | 81% | 119.0 | 45.9 | 2.59x | 78% |
| Ultra (10240) | 4K | 414.3 | 166.6 | 2.49x | 89% | 441.9 | 162.6 | 2.72x | 88% |
| Ultra (10240) | 8K | 812.3 | 330.2 | 2.46x | 90% | 864.5 | 319.1 | 2.71x | 90% |

### Forward: `shortcut_add_wide_residual_write_two_reader_rmsnorm_quant` (two readers)

| model (hidden) | tokens | BF16 unfused us | BF16 fused us | BF16 speedup | BF16 %SOL | MXFP8 unfused us | MXFP8 fused us | MXFP8 speedup | MXFP8 %SOL |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Nano (3072) | 1 | 51.6 | 1.8 | 28.21x | 1% | – | – | – | – |
| Nano (3072) | 4 | 56.9 | 1.9 | 29.72x | 3% | – | – | – | – |
| Nano (3072) | 16 | 59.1 | 1.9 | 30.97x | 11% | – | – | – | – |
| Nano (3072) | 64 | 61.8 | 3.3 | 19.00x | 26% | 65.5 | 4.3 | 15.14x | 19% |
| Nano (3072) | 256 | 22.7 | 5.5 | 4.14x | 61% | 26.4 | 5.8 | 4.57x | 57% |
| Nano (3072) | 1K | 51.6 | 16.3 | 3.17x | 83% | 56.1 | 16.3 | 3.44x | 80% |
| Nano (3072) | 4K | 186.7 | 62.9 | 2.97x | 86% | 195.0 | 62.0 | 3.15x | 85% |
| Nano (3072) | 8K | 365.5 | 121.9 | 3.00x | 89% | 380.3 | 119.6 | 3.18x | 88% |
| Super (4096) | 1 | 47.2 | 1.7 | 27.13x | 1% | – | – | – | – |
| Super (4096) | 4 | 50.4 | 1.9 | 27.07x | 4% | – | – | – | – |
| Super (4096) | 16 | 50.4 | 1.9 | 26.43x | 15% | – | – | – | – |
| Super (4096) | 64 | 53.2 | 3.6 | 14.85x | 31% | 56.8 | 4.5 | 12.61x | 24% |
| Super (4096) | 256 | 22.0 | 7.2 | 3.04x | 62% | 25.7 | 7.2 | 3.57x | 61% |
| Super (4096) | 1K | 49.6 | 23.2 | 2.14x | 77% | 54.4 | 25.6 | 2.12x | 68% |
| Super (4096) | 4K | 199.3 | 83.1 | 2.40x | 87% | 210.2 | 81.3 | 2.58x | 86% |
| Super (4096) | 8K | 373.0 | 162.3 | 2.30x | 89% | 394.7 | 158.2 | 2.50x | 88% |
| Ultra (10240) | 1 | 62.7 | 2.0 | 31.31x | 2% | – | – | – | – |
| Ultra (10240) | 4 | 62.8 | 2.0 | 30.77x | 9% | – | – | – | – |
| Ultra (10240) | 16 | 67.9 | 2.6 | 26.13x | 27% | – | – | – | – |
| Ultra (10240) | 64 | 84.8 | 7.6 | 11.12x | 37% | 89.1 | 9.5 | 9.43x | 29% |
| Ultra (10240) | 256 | 59.7 | 15.5 | 3.86x | 73% | 64.3 | 16.3 | 3.94x | 67% |
| Ultra (10240) | 1K | 183.1 | 64.7 | 2.83x | 70% | 191.3 | 67.3 | 2.84x | 65% |
| Ultra (10240) | 4K | 695.6 | 204.8 | 3.40x | 88% | 722.8 | 199.9 | 3.62x | 87% |
| Ultra (10240) | 8K | 1359.2 | 403.2 | 3.37x | 89% | 1411.0 | 392.4 | 3.60x | 89% |

### Backward: `wide_residual_write_read_rmsnorm_quant` (one reader)

| model (hidden) | tokens | BF16 unfused us | BF16 fused us | BF16 speedup | BF16 %SOL | MXFP8 unfused us | MXFP8 fused us | MXFP8 speedup | MXFP8 %SOL |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Nano (3072) | 64 | 251.2 | 12.7 | 19.85x | 8% | 250.4 | 15.0 | 16.74x | 7% |
| Nano (3072) | 256 | 37.2 | 15.3 | 2.42x | 27% | 37.4 | 17.9 | 2.09x | 24% |
| Nano (3072) | 1K | 86.6 | 32.3 | 2.68x | 52% | 87.7 | 37.0 | 2.37x | 46% |
| Nano (3072) | 4K | 261.4 | 88.9 | 2.94x | 75% | 260.8 | 119.9 | 2.18x | 57% |
| Nano (3072) | 8K | 501.7 | 163.5 | 3.07x | 82% | 501.7 | 221.4 | 2.27x | 62% |
| Super (4096) | 64 | 229.4 | 11.5 | 19.92x | 12% | 229.3 | 13.5 | 17.02x | 11% |
| Super (4096) | 256 | 34.2 | 15.5 | 2.20x | 36% | 34.8 | 18.5 | 1.88x | 31% |
| Super (4096) | 1K | 89.6 | 41.7 | 2.15x | 53% | 89.8 | 50.0 | 1.80x | 46% |
| Super (4096) | 4K | 302.3 | 113.7 | 2.66x | 78% | 302.0 | 153.5 | 1.97x | 59% |
| Super (4096) | 8K | 584.3 | 213.9 | 2.73x | 83% | 585.1 | 287.8 | 2.03x | 63% |
| Ultra (10240) | 64 | 287.4 | 15.0 | 19.10x | 23% | 287.3 | 17.5 | 16.37x | 20% |
| Ultra (10240) | 256 | 79.9 | 29.8 | 2.68x | 47% | 79.7 | 34.7 | 2.30x | 41% |
| Ultra (10240) | 1K | 235.3 | 78.1 | 3.01x | 71% | 234.6 | 104.8 | 2.24x | 54% |
| Ultra (10240) | 4K | 859.4 | 266.3 | 3.23x | 83% | 859.3 | 358.8 | 2.39x | 63% |
| Ultra (10240) | 8K | 1683.6 | 517.4 | 3.25x | 86% | 1683.5 | 697.8 | 2.41x | 65% |

### Backward: `shortcut_add_wide_residual_write_two_reader_rmsnorm_quant` (two readers)

| model (hidden) | tokens | BF16 unfused us | BF16 fused us | BF16 speedup | BF16 %SOL | MXFP8 unfused us | MXFP8 fused us | MXFP8 speedup | MXFP8 %SOL |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Nano (3072) | 64 | 340.7 | 13.1 | 26.07x | 8% | 341.1 | 15.5 | 22.02x | 7% |
| Nano (3072) | 256 | 59.0 | 16.0 | 3.69x | 27% | 59.0 | 18.4 | 3.21x | 24% |
| Nano (3072) | 1K | 139.9 | 34.3 | 4.08x | 51% | 139.8 | 39.6 | 3.53x | 45% |
| Nano (3072) | 4K | 430.4 | 95.7 | 4.50x | 73% | 429.4 | 126.2 | 3.40x | 57% |
| Nano (3072) | 8K | 821.0 | 175.0 | 4.69x | 80% | 821.8 | 232.6 | 3.53x | 61% |
| Super (4096) | 64 | 311.9 | 12.6 | 24.70x | 12% | 312.7 | 14.3 | 21.84x | 10% |
| Super (4096) | 256 | 53.8 | 16.7 | 3.22x | 35% | 53.7 | 19.3 | 2.78x | 31% |
| Super (4096) | 1K | 139.7 | 44.7 | 3.13x | 52% | 139.5 | 53.1 | 2.63x | 45% |
| Super (4096) | 4K | 480.5 | 122.8 | 3.91x | 76% | 479.1 | 162.5 | 2.95x | 59% |
| Super (4096) | 8K | 926.1 | 228.8 | 4.05x | 81% | 925.1 | 302.4 | 3.06x | 63% |
| Ultra (10240) | 64 | 392.3 | 16.4 | 23.92x | 22% | 393.1 | 19.0 | 20.74x | 20% |
| Ultra (10240) | 256 | 130.2 | 34.2 | 3.81x | 43% | 129.7 | 39.2 | 3.31x | 38% |
| Ultra (10240) | 1K | 390.2 | 88.0 | 4.43x | 66% | 391.0 | 114.4 | 3.42x | 52% |
| Ultra (10240) | 4K | 1421.1 | 297.1 | 4.78x | 78% | 1418.2 | 388.3 | 3.65x | 61% |
| Ultra (10240) | 8K | 2789.8 | 616.6 | 4.52x | 76% | 2786.4 | 792.7 | 3.51x | 60% |

A single write read twice (M → [* E]) moves one BF16 update fewer than the shortcut add and saves no
update sum. Its forward uses the two-reader configs (training within 1.4% of the best of 32 recipes) and
a few dedicated inference leaves; its backward is the two-reader kernel with one update gradient.
Unfused here is `streamwise_sigmoid_writeback` of the one update, then the two reads and RMSNorms.

### Forward: `shortcut_add_wide_residual_write_two_reader_rmsnorm_quant` without a shared update (single write, two readers)

| model (hidden) | tokens | BF16 unfused us | BF16 fused us | BF16 speedup | BF16 %SOL | MXFP8 unfused us | MXFP8 fused us | MXFP8 speedup | MXFP8 %SOL |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Nano (3072) | 1 | 49.3 | 1.9 | 26.49x | 1% | – | – | – | – |
| Nano (3072) | 4 | 54.5 | 1.9 | 28.11x | 2% | – | – | – | – |
| Nano (3072) | 16 | 57.6 | 2.0 | 29.05x | 9% | – | – | – | – |
| Nano (3072) | 64 | 59.9 | 3.1 | 19.57x | 24% | 63.7 | 4.1 | 15.55x | 18% |
| Nano (3072) | 256 | 22.5 | 5.0 | 4.50x | 60% | 26.5 | 5.3 | 4.99x | 54% |
| Nano (3072) | 1K | 49.7 | 14.5 | 3.42x | 82% | 52.4 | 14.8 | 3.54x | 78% |
| Nano (3072) | 4K | 174.8 | 55.6 | 3.14x | 86% | 184.8 | 55.6 | 3.33x | 83% |
| Nano (3072) | 8K | 343.3 | 107.5 | 3.19x | 89% | 358.1 | 106.9 | 3.35x | 86% |
| Super (4096) | 1 | 52.4 | 1.9 | 28.00x | 1% | – | – | – | – |
| Super (4096) | 4 | 54.2 | 2.0 | 27.21x | 3% | – | – | – | – |
| Super (4096) | 16 | 54.5 | 2.0 | 27.19x | 12% | – | – | – | – |
| Super (4096) | 64 | 57.5 | 3.5 | 16.31x | 28% | 61.2 | 4.5 | 13.74x | 22% |
| Super (4096) | 256 | 21.7 | 6.9 | 3.14x | 57% | 25.8 | 7.2 | 3.57x | 53% |
| Super (4096) | 1K | 47.2 | 21.3 | 2.22x | 75% | 50.2 | 22.3 | 2.25x | 69% |
| Super (4096) | 4K | 184.5 | 73.8 | 2.50x | 86% | 195.4 | 72.5 | 2.70x | 85% |
| Super (4096) | 8K | 343.4 | 143.6 | 2.39x | 88% | 365.2 | 139.7 | 2.61x | 88% |
| Ultra (10240) | 1 | 65.4 | 2.0 | 32.16x | 2% | – | – | – | – |
| Ultra (10240) | 4 | 70.9 | 2.1 | 33.51x | 7% | – | – | – | – |
| Ultra (10240) | 16 | 72.9 | 2.6 | 28.17x | 24% | – | – | – | – |
| Ultra (10240) | 64 | 88.8 | 6.5 | 13.72x | 38% | 92.1 | 8.5 | 10.84x | 28% |
| Ultra (10240) | 256 | 57.7 | 16.2 | 3.56x | 61% | 63.4 | 15.2 | 4.17x | 63% |
| Ultra (10240) | 1K | 175.5 | 65.9 | 2.66x | 60% | 182.0 | 60.9 | 2.99x | 63% |
| Ultra (10240) | 4K | 660.0 | 180.2 | 3.66x | 88% | 687.4 | 175.4 | 3.92x | 88% |
| Ultra (10240) | 8K | 1289.3 | 356.0 | 3.62x | 89% | 1341.1 | 344.1 | 3.90x | 89% |

### Backward: `shortcut_add_wide_residual_write_two_reader_rmsnorm_quant` without a shared update (single write, two readers)

| model (hidden) | tokens | BF16 unfused us | BF16 fused us | BF16 speedup | BF16 %SOL | MXFP8 unfused us | MXFP8 fused us | MXFP8 speedup | MXFP8 %SOL |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Nano (3072) | 64 | 341.6 | 12.7 | 26.83x | 9% | 341.6 | 14.9 | 22.93x | 7% |
| Nano (3072) | 256 | 65.9 | 17.9 | 3.68x | 24% | 65.5 | 20.7 | 3.17x | 22% |
| Nano (3072) | 1K | 146.7 | 36.6 | 4.01x | 48% | 140.6 | 39.5 | 3.56x | 45% |
| Nano (3072) | 4K | 429.7 | 94.6 | 4.54x | 74% | 429.3 | 126.3 | 3.40x | 57% |
| Nano (3072) | 8K | 821.7 | 175.1 | 4.69x | 80% | 819.4 | 232.2 | 3.53x | 62% |
| Super (4096) | 64 | 366.6 | 13.3 | 27.56x | 11% | 362.7 | 15.5 | 23.37x | 10% |
| Super (4096) | 256 | 60.9 | 18.5 | 3.29x | 31% | 59.9 | 21.8 | 2.74x | 27% |
| Super (4096) | 1K | 146.5 | 47.4 | 3.09x | 49% | 140.7 | 53.3 | 2.64x | 45% |
| Super (4096) | 4K | 478.8 | 122.3 | 3.92x | 76% | 479.8 | 161.7 | 2.97x | 59% |
| Super (4096) | 8K | 925.2 | 227.8 | 4.06x | 82% | 924.0 | 302.6 | 3.05x | 63% |
| Ultra (10240) | 64 | 457.6 | 17.9 | 25.60x | 20% | 456.8 | 21.0 | 21.73x | 18% |
| Ultra (10240) | 256 | 138.8 | 36.0 | 3.86x | 40% | 138.8 | 41.2 | 3.37x | 36% |
| Ultra (10240) | 1K | 396.9 | 90.0 | 4.41x | 65% | 391.3 | 114.2 | 3.43x | 52% |
| Ultra (10240) | 4K | 1418.8 | 298.3 | 4.76x | 78% | 1421.8 | 386.4 | 3.68x | 62% |
| Ultra (10240) | 8K | 2784.5 | 618.7 | 4.50x | 75% | 2784.7 | 792.7 | 3.51x | 60% |

### BF16 residual streams

Without `--fp32-residual-connection` the stream is BF16: half the residual bytes, the same FP32 arithmetic.
The forward then needs more rows in flight and uses its own per-shape configs (`_FORWARD_CONFIGS_BF16`,
eight BF16 columns per thread where that helps): 82-89% of peak DRAM at 8K-16K tokens for the one-reader and
shortcut-add seams, and 82-88% for the single write read twice with BF16 readers (76-81% with the MXFP8
handoff). The backward runs the same FP32 arithmetic on half the bytes and becomes issue-bound (about 45-63% of
peak DRAM at 8K, 1.1x faster than with an FP32 stream).
