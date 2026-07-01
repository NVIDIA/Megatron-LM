# MTP Layer Split Across Pipeline Stages

This feature distributes **Multi-Token-Prediction (MTP) layers** across
**multiple consecutive pipeline stages** (mtp_split mode), instead of
forcing all MTP layers to reside on a single PP rank.  Spreading MTP
layers across ranks reduces the per-rank compute and memory burden,
enabling larger MTP configurations without bottlenecking one stage.

---

## 1 Usage & Configuration

### 1.1 What it does

In standalone MTP (`mtp_standalone`), all `mtp_num_layers` MTP layers live
on a single PP rank's last virtual stage.  In **mtp_split**, the `m` tokens
in the `pipeline_model_parallel_layout` string may be spread across multiple
consecutive PP ranks — uniformly or non-uniformly — as long as the total
count equals `mtp_num_layers`.

Each MTP split rank builds only the subset of layers assigned to it, receives
the hidden state from the previous rank, pre-rolls `input_ids`/`position_ids`
by its global layer offset, and forwards the grown hidden-state concatenation
to the next stage.

### 1.2 How to enable

Set a `pipeline_model_parallel_layout` whose `m` tokens span more than one
pipeline stage.  The layout tokens are: `E` = embedding, `t` = transformer/
decoder layer, `m` = MTP layer, `L` = output+loss slot.  Stages are separated
by `|`; `(...)*n` repeats a group.

```yaml
ARGS:
  pipeline_model_parallel_size: 8
  pipeline_model_parallel_layout: "E|(t|)*11m|m|m|L"
  num_layers: 11
  mtp_num_layers: 3
  mtp_loss_scaling_factor: 0.1
```

With `num_layers=11` the layout expands to 16 virtual stages, pp size is 8,
**VPP=2** (each PP rank hosts two virtual stages).  The assignment is:

| PP rank | VPP 0 | VPP 1 |
|---------|-------|-------|
| 0 | embedding (`E`) | decoder layer 7 (`t`) |
| 1 | decoder layer 0 (`t`) | decoder layer 8 (`t`) |
| 2 | decoder layer 1 (`t`) | decoder layer 9 (`t`) |
| 3 | decoder layer 2 (`t`) | decoder layer 10 (`t`) |
| 4 | decoder layer 3 (`t`) | MTP layer 0 (`m`) |
| 5 | decoder layer 4 (`t`) | MTP layer 1 (`m`) |
| 6 | decoder layer 5 (`t`) | MTP layer 2 (`m`) |
| 7 | decoder layer 6 (`t`) | output+loss (`L`) |

Each of PP4, PP5, PP6 builds exactly one MTP layer.  PP4's layer offset is 0,
PP5's is 1, PP6's is 2.

Non-uniform distributions are also valid.  For example `E|(t|)*12mm|m|L`
places 2 MTP layers on PP5 (offset=0) and 1 on PP6 (offset=2), totalling 3.

### 1.3 Key configuration arguments

| Arg | Meaning |
|-----|---------|
| `pipeline_model_parallel_layout` | Layout string.  mtp_split is detected automatically when `m` tokens span more than one PP rank. |
| `pipeline_model_parallel_size` | Number of PP stages; must match the number of `\|`-separated stages (times VPP). |
| `mtp_num_layers` | Total number of MTP layers across all ranks.  The sum of `m` counts in the layout must equal this value. |
| `mtp_loss_scaling_factor` | MTP loss scale (applied via `MTPLossAutoScaler`); unchanged by the split. |
| `num_layers` / `moe_layer_freq` | Main-model decoder layers; must match the count of `t` in the layout. |

### 1.4 Validation rules (enforced in `validate_layer_layout`)

- First token must be `E`, last token must be `L`; exactly one `E`.
- `count(t) == num_layers`, `count(m) == mtp_num_layers`; all decoder (`t`) before any MTP (`m`).
- MTP may not live on the first PP rank.
- MTP must only appear in the **last VPP stage** of each rank that holds MTP.
- **mtp_split**: `m` tokens may span any number of consecutive PP ranks; total must equal `mtp_num_layers` but per-rank counts can be non-uniform.
- **mtp_standalone** (single MTP rank): the one MTP rank must hold all `mtp_num_layers` layers on its last VPP stage.

### 1.5 Layer offset

Each MTP split rank queries `get_layer_offset(LayerType.mtp, vp_stage, pp_rank)`
to determine its **global layer offset** — the number of MTP layers that
precede it in the pipeline.  This offset is used to:

1. **Pre-roll `input_ids`/`position_ids`** by `offset` steps before the
   forward loop, so that local layer `k` receives the correct token embedding
   for prediction target `offset + k + 1`.
2. **Set `layer_number`** on each `MultiTokenPredictionLayer` so logging,
   attention bias, and checkpoint keys are globally consistent.
3. **Slice the incoming hidden-state concatenation** to drop the prefix chunks
   already computed by earlier MTP ranks.

---

## 2 Design

### 2.1 Data flow

```
virtual stage  0  (E)          ──► hidden_state                        [s, b, h]
virtual stage  1–7 (t×7)       ──► hidden_state
virtual stage  8–11 (t×4)      ──► hidden_state                        [decoder layers]
virtual stage 12  (m, MTP 0)   ──► cat([main, mtp0])                   [2s, b, h]
virtual stage 13  (m, MTP 1)   ──► cat([main, mtp0, mtp1])             [3s, b, h]
virtual stage 14  (m, MTP 2)   ──► cat([main, mtp0, mtp1, mtp2])       [4s, b, h]
virtual stage 15  (L)          ──► output projection + loss for all 4 chunks
```

Each MTP split rank receives the concatenated hidden states built by prior
MTP ranks, slices out the chunk it needs (position `offset`), runs its MTP
layer(s), and appends the new chunk(s) to the tensor it forwards downstream.
The final loss stage (`L`) receives the full `[1 + mtp_num_layers]`-chunk
concatenation and computes all losses in one place.

### 2.2 Pre-rolling input IDs for split ranks

A rank at offset `k > 0` does not start from `input_ids` rolled 0 times.
Before the forward loop, `MultiTokenPredictionBlock.forward` pre-rolls
`input_ids`, `position_ids`, and `padding_mask` by `offset` steps using
the existing `roll_tensor` primitive (CP-aware):

```python
if offset > 0:
    for _ in range(offset):
        input_ids, _ = roll_tensor(input_ids, shifts=-1, ...)
        position_ids, _ = roll_tensor(position_ids, shifts=-1, ...)
        if padding_mask is not None:
            padding_mask, _ = roll_tensor(padding_mask, shifts=-1, ...)
```

After this pre-roll, the standard per-layer loop that increments rolls by 1
each iteration produces the correct targets for global layers
`[offset, offset+1, ..., offset+local_num_layers-1]`.

### 2.3 Changes to existing assertions

Three assertions that previously enforced "all MTP on one rank" were relaxed:

| Location | Old assertion | New assertion |
|----------|--------------|---------------|
| `pipeline_parallel_layer_layout.py` | `count(m on rank) == mtp_num_layers` | allowed for multi-rank; standalone path still checks equality |
| `multi_token_prediction.py` `get_mtp_num_layers_to_build` | `num_to_build == mtp_num_layers or 0` | `0 <= num_to_build <= mtp_num_layers` |
| `gpt_layer_specs.py` `get_gpt_mtp_block_spec_for_backend` | `len(mtp_layer_specs) == mtp_num_layers` | `0 < len(mtp_layer_specs) <= mtp_num_layers` |

The forward loop was also changed from `range(config.mtp_num_layers)` to
`range(len(self.layers))` so that a split rank that holds fewer than
`mtp_num_layers` layers iterates over only its local layers.

### 2.4 Checkpoint compatibility

Layer numbers are set globally (via the offset), so checkpoint keys are
identical whether MTP layers are standalone or split.  No migration is
required when switching between the two modes, provided `mtp_num_layers`
and the total number of decoder layers stay the same.
