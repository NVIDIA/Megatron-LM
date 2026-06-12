# MTP Loss Split Across Pipeline Stages

This feature splits the **output projection + loss computation** of a Multi-Token-Prediction
(MTP) model — both the main-model loss and each MTP-layer loss — across **multiple pipeline
stages**, instead of forcing every loss to be computed on the single last pipeline stage.
This finer-grained distribution can help to balance the compute load more evenly across
PP ranks and reduces pipeline bubbles.

---

## 1 Usage & Configuration

### 1.1 What it does

In standalone MTP, the pipeline carries a **dim-0 concatenation of hidden-state chunks**
`[main, mtp_0, mtp_1, ...]` (`N = 1 + mtp_num_layers` chunks). Normally all `N` losses are
computed on the last stage. With loss split, the layout places several loss slots (`L`) on
**different stages**; each loss stage computes the losses for the chunks it owns and forwards the
remaining (lower-index) prefix to the next loss stage.

### 1.2 How to enable

Set a `pipeline_model_parallel_layout` whose loss slots are distributed across more than one
stage. The layout tokens are: `E` = embedding, `t` = transformer/decoder layer, `m` = MTP layer,
`L` = output+loss slot. Stages are separated by `|`; `(...)*n` repeats a group.

```yaml
ARGS:
  pipeline_model_parallel_size: 8
  pipeline_model_parallel_layout: "E|(t|)*10m|m|m|LL|LL"
  num_layers: 10
  mtp_num_layers: 3
  mtp_loss_scaling_factor: 0.1
```

With `num_layers=10` the layout expands to 16 virtual stages, pp size is 8, **VPP=2** (each PP rank hosts
two virtual stages). The assignment is:

| PP rank | VPP 0 | VPP 1 |
|---------|-------|-------|
| 0 | embedding (`E`) | decoder layer 7 (`t`) |
| 1 | decoder layer 0 (`t`) | decoder layer 8 (`t`) |
| 2 | decoder layer 1 (`t`) | decoder layer 9 (`t`) |
| 3 | decoder layer 2 (`t`) | MTP layer 0 (`m`) |
| 4 | decoder layer 3 (`t`) | MTP layer 1 (`m`) |
| 5 | decoder layer 4 (`t`) | MTP layer 2 (`m`) |
| 6 | decoder layer 5 (`t`) | output+loss for MTP layer 1 & 2, chunks `[2,3]` (`LL`) |
| 7 | decoder layer 6 (`t`) | output+loss for MTP layer 0 + main, chunks `[0,1]` (`LL`) |

### 1.3 Key configuration arguments

| Arg | Meaning |
|-----|---------|
| `pipeline_model_parallel_layout` | Layout string. Loss split is detected automatically when more than one stage contains an `L`. |
| `pipeline_model_parallel_size` | Number of PP stages; must match the number of `\|`-separated stages (times VPP). |
| `mtp_num_layers` | Number of MTP layers. The total number of hidden-state chunks is `N = 1 + mtp_num_layers`. |
| `mtp_loss_scaling_factor` | MTP loss scale (applied via `MTPLossAutoScaler`); unchanged by the split. |
| `num_layers` / `moe_layer_freq` | Main-model decoder layers; must match the count of `t` in the layout. |
| `untie_embeddings_and_output_weights` | The split assumes an **independent** `output_layer` weight (such as DeepSeek-V3). The replicated `output_layer` on the non-final loss stage is kept in sync via a dedicated process group. |

### 1.4 Validation rules (enforced in `validate_layer_layout`)

- First token must be `E`, last token must be `L`; exactly one `E`.
- `count(t) == num_layers`, `count(m) == mtp_num_layers`; all decoder (`t`) before any MTP (`m`).
- MTP may not live on the first PP rank.
- **When loss is split (more than one `L` stage): `sum(L slots) == 1 + mtp_num_layers`.** A single
  loss stage (legacy) is exempt and owns all chunks internally.

### 1.5 Chunk-to-stage assignment

Chunk index `j` corresponds to: `j=0` → main model (labels rolled 0×), `j=k+1` → MTP layer `k`
(labels rolled `k+1`×). Loss stages are walked in pipeline order and the **highest-index chunks
go to the earliest loss stage, peeling downward**; the final loss stage owns chunk `0` (the main
model). For the reference recipe: virtual stage 14 → `[2,3]` (non-final), virtual stage 15 → `[0,1]` (final).

---

## 2 Design

### 2.1 Data flow (chained slice-down)

```
virtual stage  0  (E)          ──► hidden_state
virtual stage  1–10 (t×10)     ──► hidden_state        [decoder layers 0–9, pass-through]
virtual stage 11  (m, MTP 0)   ──► cat([main, mtp0])
virtual stage 12  (m, MTP 1)   ──► cat([main, mtp0, mtp1])
virtual stage 13  (m, MTP 2)   ──► cat([main, mtp0, mtp1, mtp2])
                                                   │
                                                   ▼
                                      virtual stage 14  (LL)
                                      │  compute chunks [2,3] loss  (mtp1, mtp2)
                                      │  MTPLossAutoScaler attaches loss to the forwarded tensor
                                      └─► cat([main, mtp0])
                                                   │
                                                   ▼
                                      virtual stage 15  (LL)
                                         compute chunk [1] loss  (mtp0)
                                         compute chunk [0] loss  (main)
```

Each loss stage computes the losses for its owned (high-index) chunks, attaches the scaled loss
to the tensor it keeps forwarding (so gradients flow back along the P2P chain to the right MTP
layers), and sends only the remaining low-index **prefix** downstream. The final loss stage
consumes chunk 0 and forwards nothing.

This is "scheme A" (chained slice-down) — chosen over broadcasting the full concat to every loss
stage, which would break the single P2P chain and need extra send targets + a backward reduce.

### 2.2 Loss-stage process group

A new `_LOSS_GROUP` (built symmetrically inside the PP construction loop, like `EMBEDDING_GROUP`,
so it cannot deadlock) ties together the ranks that compute loss. It is used to:

- **Initialize**: all-reduce the `output_layer` weights from the final loss stage to every loss
  stage so the replicas start identical (`_sync_output_layer_across_loss_group`).
- **Gradients**: SUM all-reduce the `output_layer` gradients across loss stages
  (`_allreduce_output_layer_grads_across_loss_group`) so the two replicas behave as one logical
  weight (combined grad = G6 + G7 = the baseline total).

### 2.3 Output-layer replication & grad-norm dedup

Because the `output_layer` is replicated on every loss stage, the non-final replica's weight is
flagged `weight.shared = True` so `param_is_not_shared` counts it **once** for grad-norm /
param-norm (otherwise the gradient would be counted on both ranks and shrink the clip coefficient,
biasing training).

### 2.4 Distributed-optimizer bucketing & checkpoints

- The replicated `output_layer` is given its **own grad bucket** (flag `param.shared_output_layer`,
  generalizing the embedding's `_does_param_require_new_bucket` mechanism) so the two replicas'
  bucket layout aligns and distributed-optimizer state can be de-duplicated for dist-ckpt.
- For distributed checkpoint, the non-final loss stage's `output_layer.weight` is written with
  `replica_id=(1,0,dp)` (mirroring the tied-embedding convention) to avoid a replica_id collision
  between the two PP replicas.
