# Fused wide-residual shortcut seams

For wide-residual hybrid models with shortcut-connected MoE (`--moe-shortcut-wide-residual-norm-fusion`;
needs `--fp32-residual-connection`, BF16 parameters, RMSNorm, 2-6 wide streams, a shared expert, no
dropout/bias, no CP/PP, and a GB300 for training), `ShortcutMoEBlock` replaces each residual seam with one
Gluon kernel that mixes the FP32 wide-residual streams with the branch update, writes the new residual, and
emits the RMSNorm readers that consume it — directly as MXFP8 for an exclusive TE Linear consumer under an
MXFP8 recipe, otherwise BF16. `wide_residual_write_read_rmsnorm_quant` writes the Mamba/GDN/attention
output and produces the shared expert's normalized input; `shortcut_add_wide_residual_write_two_reader_rmsnorm_quant`
adds the routed and shared MoE outputs (the shortcut add, passed between blocks as a `PendingShortcutAdd`) and
produces the next block's compute-layer and shortcut-routing inputs. The matching backward is one fused
kernel per seam; with `--residual-stream-recompute-num-layers N` the residuals are rebuilt by bit-exact
checkpoint replay instead of being saved. Residual writes are bit-identical to replay and across kernel
schedules, reader outputs differ by at most one BF16 ulp between schedules, and the tests check every
output and gradient against independent FP64/autograd references.

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
