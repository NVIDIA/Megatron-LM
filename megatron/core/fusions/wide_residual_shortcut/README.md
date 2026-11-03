# Fused wide-residual shortcut seams

For wide-residual hybrid models with shortcut-connected MoE (`--moe-shortcut-wide-residual-norm-fusion`;
needs BF16 parameters, RMSNorm, 2-6 wide streams, a shared expert, no dropout/bias, no CP/PP, no
layernorm recompute, at most 65536 local tokens per call (16384 above hidden 4096), and a GB300
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
| compute → MoE inside a block | 1 | 1 | `wide_residual_write_read_rmsnorm_quant` |
| lone layer → lone layer (M → -, - → M) | 1 | 1 | `shortcut_add_wide_residual_write_read_rmsnorm_quant`, no shared update |
| block → lone layer ([* E] → M) | 2 (routed + shared) | 1 | `shortcut_add_wide_residual_write_read_rmsnorm_quant` |
| block → block (shortcut add) | 2 (routed + shared) | 2 | `shortcut_add_..._two_reader_rmsnorm_quant` |
| lone layer → block (M → [* E]) | 1 | 2 | `shortcut_add_..._two_reader_rmsnorm_quant`, no shared update |

Lone wide-residual Mamba layers (Mamba or GDP mixers) and MLP-only layers expose the read/RMSNorm
they apply (`pending_write_reader`) and the write they produce (`pending_write_writer`). When the
next layer applies a layer's write in its seam kernel, `group_layers_into_shortcut_blocks` sets
`defers_residual_write` and the layer returns its write as a `PendingShortcutAdd`, which the next
layer receives as `hidden_states`. In a stack of such layers every seam after the first layer's
read is one kernel, including under checkpoint replay intervals. MLP-only layers use a standalone
pre-MLP RMSNorm in the fusion stack specs; lone attention/GDN layers are not chained.

## Performance

GB300, Triton 3.7.1, 3 wide streams; fused kernel time per call under CUDA graphs (20 launches per graph
over rotating input copies with every output kept alive, median of 10 replays). Two modes: an FP32
residual stream with the MXFP8 handoff of reader zero (reader one, when present, stays BF16), and a BF16
residual stream (BF16 connection logits) with BF16 readers. %DRAM = compulsory bytes / fused time /
7.926 TB/s. Forward bytes: residual read + write, BF16 updates, reader outputs and MXFP8 scales, and in
training the shortcut add's BF16 update sum and rstd. Backward bytes: previous residual, incoming carry and
residual gradient, saved update, reader and update gradients, rstd, and MXFP8 columns of reader zero
(materialized in the backward call; with checkpoint replay they come from replay instead). Below 32 tokens
the forward is the inference kernel (no backward state) and MXFP8 and the backward do not apply (rows must
be multiples of 32; model training additionally needs multiples of 1024 tokens). The backward time includes
its scratch resets and final gradient reductions.

### Forward: `wide_residual_write_read_rmsnorm_quant` (one update, one reader)

| model (hidden) | tokens | FP32 stream + MXFP8 us | %DRAM | BF16 stream us | %DRAM |
|---|--:|--:|--:|--:|--:|
| Nano (3072) | 1 | – | – | 1.7 | 0% |
| Nano (3072) | 4 | – | – | 1.7 | 1% |
| Nano (3072) | 32 | 3.4 | 10% | 2.5 | 8% |
| Nano (3072) | 64 | 3.9 | 17% | 2.6 | 15% |
| Nano (3072) | 256 | 5.6 | 48% | 4.7 | 34% |
| Nano (3072) | 1K | 15.1 | 71% | 12.6 | 50% |
| Nano (3072) | 4K | 51.3 | 84% | 32.1 | 79% |
| Nano (3072) | 8K | 98.7 | 87% | 60.0 | 85% |
| Super (4096) | 1 | – | – | 1.7 | 0% |
| Super (4096) | 4 | – | – | 1.8 | 2% |
| Super (4096) | 32 | 3.9 | 11% | 2.8 | 9% |
| Super (4096) | 64 | 4.5 | 20% | 3.1 | 17% |
| Super (4096) | 256 | 6.9 | 52% | 5.4 | 39% |
| Super (4096) | 1K | 19.3 | 74% | 17.0 | 50% |
| Super (4096) | 4K | 66.3 | 86% | 41.5 | 82% |
| Super (4096) | 8K | 129.2 | 89% | 79.0 | 86% |
| Ultra (10240) | 1 | – | – | 2.0 | 1% |
| Ultra (10240) | 4 | – | – | 2.0 | 4% |
| Ultra (10240) | 32 | 8.6 | 13% | 6.7 | 10% |
| Ultra (10240) | 64 | 9.4 | 24% | 8.2 | 16% |
| Ultra (10240) | 256 | 14.5 | 62% | 12.7 | 42% |
| Ultra (10240) | 1K | 46.8 | 76% | 45.4 | 47% |
| Ultra (10240) | 4K | 165.5 | 86% | 101.8 | 83% |
| Ultra (10240) | 8K | 321.7 | 89% | 195.9 | 86% |

### Forward: `shortcut_add_wide_residual_write_two_reader_rmsnorm_quant` (shortcut add: two updates, two readers)

| model (hidden) | tokens | FP32 stream + MXFP8 us | %DRAM | BF16 stream us | %DRAM |
|---|--:|--:|--:|--:|--:|
| Nano (3072) | 1 | – | – | 1.8 | 0% |
| Nano (3072) | 4 | – | – | 1.9 | 2% |
| Nano (3072) | 32 | 3.9 | 11% | 3.3 | 8% |
| Nano (3072) | 64 | 4.5 | 18% | 3.9 | 14% |
| Nano (3072) | 256 | 6.7 | 49% | 6.0 | 36% |
| Nano (3072) | 1K | 18.2 | 72% | 17.2 | 51% |
| Nano (3072) | 4K | 62.0 | 85% | 42.6 | 82% |
| Nano (3072) | 8K | 120.1 | 87% | 81.0 | 86% |
| Super (4096) | 1 | – | – | 1.8 | 1% |
| Super (4096) | 4 | – | – | 1.9 | 2% |
| Super (4096) | 32 | 4.5 | 12% | 3.5 | 10% |
| Super (4096) | 64 | 5.2 | 21% | 4.8 | 15% |
| Super (4096) | 256 | 8.3 | 53% | 7.3 | 40% |
| Super (4096) | 1K | 23.7 | 74% | 24.9 | 47% |
| Super (4096) | 4K | 81.5 | 86% | 57.1 | 82% |
| Super (4096) | 8K | 158.5 | 88% | 109.0 | 85% |
| Ultra (10240) | 1 | – | – | 2.0 | 1% |
| Ultra (10240) | 4 | – | – | 2.0 | 5% |
| Ultra (10240) | 32 | 10.3 | 13% | 9.7 | 9% |
| Ultra (10240) | 64 | 11.5 | 24% | 10.2 | 18% |
| Ultra (10240) | 256 | 19.1 | 57% | 16.6 | 44% |
| Ultra (10240) | 1K | 67.3 | 65% | 59.6 | 49% |
| Ultra (10240) | 4K | 199.8 | 87% | 134.0 | 87% |
| Ultra (10240) | 8K | 392.2 | 89% | 263.3 | 88% |

### Forward: `shortcut_add_wide_residual_write_two_reader_rmsnorm_quant` without a shared update (one update, two readers)

| model (hidden) | tokens | FP32 stream + MXFP8 us | %DRAM | BF16 stream us | %DRAM |
|---|--:|--:|--:|--:|--:|
| Nano (3072) | 1 | – | – | 1.7 | 0% |
| Nano (3072) | 4 | – | – | 1.9 | 2% |
| Nano (3072) | 32 | 3.7 | 10% | 3.1 | 7% |
| Nano (3072) | 64 | 4.5 | 16% | 3.1 | 14% |
| Nano (3072) | 256 | 6.3 | 46% | 5.2 | 34% |
| Nano (3072) | 1K | 17.8 | 65% | 15.1 | 47% |
| Nano (3072) | 4K | 55.7 | 83% | 36.8 | 78% |
| Nano (3072) | 8K | 107.0 | 86% | 69.5 | 82% |
| Super (4096) | 1 | – | – | 1.8 | 1% |
| Super (4096) | 4 | – | – | 1.9 | 2% |
| Super (4096) | 32 | 4.3 | 11% | 3.4 | 9% |
| Super (4096) | 64 | 5.0 | 19% | 3.9 | 15% |
| Super (4096) | 256 | 7.6 | 50% | 6.2 | 38% |
| Super (4096) | 1K | 21.7 | 71% | 20.7 | 46% |
| Super (4096) | 4K | 71.9 | 85% | 46.9 | 81% |
| Super (4096) | 8K | 139.5 | 88% | 88.9 | 86% |
| Ultra (10240) | 1 | – | – | 2.0 | 1% |
| Ultra (10240) | 4 | – | – | 2.0 | 5% |
| Ultra (10240) | 32 | 8.4 | 14% | 8.1 | 9% |
| Ultra (10240) | 64 | 9.2 | 26% | 8.7 | 17% |
| Ultra (10240) | 256 | 17.4 | 55% | 14.3 | 42% |
| Ultra (10240) | 1K | 60.3 | 64% | 50.8 | 47% |
| Ultra (10240) | 4K | 176.4 | 87% | 115.9 | 82% |
| Ultra (10240) | 8K | 344.7 | 89% | 224.5 | 85% |

### Backward: `wide_residual_write_read_rmsnorm_quant` (one update, one reader)

| model (hidden) | tokens | FP32 stream + MXFP8 us | %DRAM | BF16 stream us | %DRAM |
|---|--:|--:|--:|--:|--:|
| Nano (3072) | 32 | 13.2 | 4% | 11.6 | 3% |
| Nano (3072) | 64 | 15.0 | 7% | 12.0 | 5% |
| Nano (3072) | 256 | 18.6 | 23% | 14.9 | 16% |
| Nano (3072) | 1K | 38.5 | 44% | 26.8 | 36% |
| Nano (3072) | 4K | 120.2 | 57% | 71.4 | 53% |
| Nano (3072) | 8K | 221.5 | 62% | 130.1 | 59% |
| Super (4096) | 32 | 14.2 | 5% | 11.5 | 3% |
| Super (4096) | 64 | 14.9 | 10% | 13.0 | 6% |
| Super (4096) | 256 | 19.3 | 30% | 15.2 | 21% |
| Super (4096) | 1K | 50.9 | 45% | 34.1 | 37% |
| Super (4096) | 4K | 154.2 | 59% | 91.4 | 56% |
| Super (4096) | 8K | 288.1 | 63% | 171.6 | 59% |
| Ultra (10240) | 32 | 16.0 | 11% | 13.8 | 7% |
| Ultra (10240) | 64 | 18.5 | 19% | 14.9 | 13% |
| Ultra (10240) | 256 | 35.4 | 40% | 26.5 | 30% |
| Ultra (10240) | 1K | 108.5 | 52% | 63.3 | 50% |
| Ultra (10240) | 4K | 358.3 | 64% | 216.1 | 59% |
| Ultra (10240) | 8K | 695.2 | 66% | 428.2 | 59% |

### Backward: `shortcut_add_wide_residual_write_two_reader_rmsnorm_quant` (shortcut add: two updates, two readers)

| model (hidden) | tokens | FP32 stream + MXFP8 us | %DRAM | BF16 stream us | %DRAM |
|---|--:|--:|--:|--:|--:|
| Nano (3072) | 32 | 14.1 | 4% | 11.9 | 3% |
| Nano (3072) | 64 | 15.1 | 7% | 12.5 | 5% |
| Nano (3072) | 256 | 19.0 | 24% | 15.3 | 17% |
| Nano (3072) | 1K | 40.7 | 44% | 30.0 | 34% |
| Nano (3072) | 4K | 126.8 | 56% | 78.7 | 52% |
| Nano (3072) | 8K | 232.2 | 62% | 150.3 | 55% |
| Super (4096) | 32 | 14.9 | 5% | 12.1 | 4% |
| Super (4096) | 64 | 15.7 | 9% | 13.3 | 6% |
| Super (4096) | 256 | 20.1 | 30% | 16.0 | 21% |
| Super (4096) | 1K | 55.5 | 43% | 38.2 | 36% |
| Super (4096) | 4K | 162.6 | 59% | 103.5 | 53% |
| Super (4096) | 8K | 302.6 | 63% | 198.3 | 56% |
| Ultra (10240) | 32 | 17.2 | 11% | 14.8 | 7% |
| Ultra (10240) | 64 | 19.6 | 19% | 16.1 | 13% |
| Ultra (10240) | 256 | 39.4 | 38% | 30.5 | 28% |
| Ultra (10240) | 1K | 114.8 | 52% | 73.8 | 47% |
| Ultra (10240) | 4K | 388.9 | 61% | 263.0 | 52% |
| Ultra (10240) | 8K | 793.2 | 60% | 489.8 | 56% |

### Backward: `shortcut_add_wide_residual_write_two_reader_rmsnorm_quant` without a shared update (one update, two readers)

| model (hidden) | tokens | FP32 stream + MXFP8 us | %DRAM | BF16 stream us | %DRAM |
|---|--:|--:|--:|--:|--:|
| Nano (3072) | 32 | 14.3 | 4% | 11.9 | 3% |
| Nano (3072) | 64 | 15.1 | 7% | 12.6 | 5% |
| Nano (3072) | 256 | 18.9 | 24% | 15.3 | 17% |
| Nano (3072) | 1K | 41.9 | 43% | 30.1 | 34% |
| Nano (3072) | 4K | 126.8 | 56% | 78.6 | 53% |
| Nano (3072) | 8K | 232.1 | 62% | 150.1 | 55% |
| Super (4096) | 32 | 15.1 | 5% | 12.0 | 4% |
| Super (4096) | 64 | 15.6 | 10% | 13.7 | 6% |
| Super (4096) | 256 | 20.1 | 30% | 16.1 | 21% |
| Super (4096) | 1K | 54.0 | 44% | 38.1 | 36% |
| Super (4096) | 4K | 162.3 | 59% | 103.6 | 53% |
| Super (4096) | 8K | 302.9 | 63% | 198.5 | 55% |
| Ultra (10240) | 32 | 17.1 | 11% | 14.8 | 7% |
| Ultra (10240) | 64 | 19.6 | 19% | 16.1 | 13% |
| Ultra (10240) | 256 | 39.5 | 38% | 30.6 | 28% |
| Ultra (10240) | 1K | 114.5 | 52% | 73.9 | 47% |
| Ultra (10240) | 4K | 388.5 | 61% | 262.5 | 52% |
| Ultra (10240) | 8K | 793.1 | 60% | 488.8 | 56% |
