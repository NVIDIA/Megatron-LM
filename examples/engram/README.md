# Hybrid MoE and Engram on FineWeb

These examples train fresh Hybrid models using an existing local GPT-2 tokenizer
and pre-tokenized FineWeb IndexedDataset files. Both arms use hidden size 512,
eight logical blocks, dense FFN 1344, expert/shared FFN 256, routed top-8, one shared
expert, 12 attention heads, KV width 128, one query group, attention output gating,
RMSNorm, SwiGLU, RoPE, BF16, and MTP1.

The MoE baseline has 256 routed experts. MoE+Engram has 208 and inserts memory ID 1
before the second attention. Engram uses 2/3-grams, eight heads per order, embedding
width 320 per order, minimum table sizes 205700, kernel 4, and dilation 3. Tables
use native Adam with learning-rate multiplier 5 and weight decay 0; the backbone
uses Muon with Adam betas (0.9, 0.95), weight decay 0.1, and clipping 1.0.

Both arms retain 36,754 training updates, sequence length 4096, global batch 64,
training seed 2026, Engram seed 0, and LR 8e-4 to 8e-5. Native WSD has 1,000 warmup
updates and 3,675 cosine-decay updates. MTP weight is 0.3 until decay starts, then
0.15. `pretrain_hybrid.py` uses the native training-step schedule, deriving the weight
from the restored completed-update count. The LR schedule is unchanged. Set
`MTP_DECAY_START=30` only for a separate short boundary test.

## Run

Copy `local_data.sh.example` to `local_data.sh` and fill in paths to the local tokenizer,
per-split IndexedDataset configuration, writable cache, and output directory. Both
arms must share the same data manifest and sampling schedule. Data preparation and
experiment-specific audit/report tooling belong outside the submitted repository.

```bash
# Keep the complete data/LR horizon; only stop execution after update 50.
EXIT_INTERVAL=50 SAVE_INTERVAL=25 bash examples/engram/pretrain_moe_baseline.sh
EXIT_INTERVAL=50 SAVE_INTERVAL=25 bash examples/engram/pretrain_moe_engram.sh
```

Default topology is eight GPUs, TP1/PP1/CP1/EP8/ETP1, micro batch 8, global batch 64.
The dispatcher defaults to flex/HybridEP; `MOE_TOKEN_DISPATCHER=alltoall` selects the
native alternative. Use the same choice in both arms. Model and batch changes must
be recorded explicitly rather than silently substituted during acceptance.

Each arm loads from its own checkpoint directory by default. For a controlled
restart, point `LOAD_PATH` at the chosen checkpoint root and use a separate `RUN_ROOT`.
The new optimizer format restores native Adam state; no older optimizer migration
is performed. TensorBoard and optional W&B record the native training metrics. W&B defaults to offline; set `WANDB_MODE=online`
explicitly when online logging is wanted.

## Complete holdout evaluation

Provide `full_valid.json` and `full_test.json` beside the training split configuration,
with the respective complete holdout selected as the native validation split. Then run:

```bash
LOAD_PATH=/path/to/checkpoints FULL_EVAL_SPLIT=valid bash examples/engram/pretrain_moe_baseline.sh
LOAD_PATH=/path/to/checkpoints FULL_EVAL_SPLIT=test bash examples/engram/pretrain_moe_engram.sh
```

Native full validation keeps partial holdout tails and pads only with zero-loss
samples, preventing rank-length differences from repeating examples. Native
validation metrics contain token-weighted CE; valid and test use separate
`-full-valid` and `-full-test` output directories.
Run both splits for both arms when reporting final model quality.
