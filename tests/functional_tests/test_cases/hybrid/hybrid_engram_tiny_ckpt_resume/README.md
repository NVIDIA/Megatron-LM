# Hybrid Engram checkpoint resume

This short native fixture exercises row-sharded Engram memory with eight data-parallel
ranks, a dense block, three MoE blocks and one attention/MoE MTP depth. Engram belongs
only to the second main-stack attention layer. Each update consumes two microbatches
per rank. Dense/expert parameters use the distributed Adam optimizer; Engram tables
use RowSparseAdam through the ordinary optimizer builder.

The runner trains through update 100, saves at update 50, then restores that checkpoint
and repeats updates 51–100. It compares the main and MTP losses and gradient zero counts
using the existing functional-test thresholds. BF16 execution uses the runner's native
nondeterministic comparison mode. This scalar comparison complements the unit tests
for complete sparse optimizer state; it does not establish arbitrary-topology restore
or bitwise-identical floating-point updates.

The fixture uses the existing Common Pile CI corpus and its GPT-2 vocabulary/merges.
Its preparation step creates a local Hugging Face tokenizer from those files without
changing token IDs, retokenizing the corpus or downloading Hub metadata. No private
data path, Engram collision metric, W&B credential or long-training checkpoint is needed.

Register this case through `engram-functional-tests.yaml` on the dev/H100 L1 suite.
The first canonical run must produce the real `golden_values_dev_dgx_h100.json` artifact;
download it using the repository's golden-value workflow and review it before committing.
Missing golden values are an outstanding CI gate, not a passing result. Do not substitute
values from another configuration or GPU platform, skip pytest, or change shared
comparison thresholds to bootstrap this case.
