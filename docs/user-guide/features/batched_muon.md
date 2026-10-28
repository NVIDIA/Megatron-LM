# Batched Muon orthogonalization

Duplicated Muon can combine compatible expert matrices into batched
Newton–Schulz calls to reduce kernel launches:

```bash
--optimizer muon --muon-tp-mode duplicated --muon-ns-batch-size 32
```

The batch size is an upper bound per optimizer group, matrix shape, dtype, and
device. Smaller buckets use smaller batches; a single remaining matrix uses the
existing per-parameter path. The default batch size of one preserves the inherited
Emerging Optimizers step exactly.

Duplicated batching applies to two-dimensional expert matrices that need no TP
or GTP reconstruction and no QKV splitting. Other parameters retain their existing
orthogonalization path. Layer-sharded Muon continues to use the same option to
batch complete matrices at their Newton–Schulz homes.

Values above one require Emerging Optimizers 0.3.0 or later. Duplicated batching
requires the duplicated mode and does not support `--muon-use-syrk` or
AdaptiveMuon. Layer-sharded batching retains its existing SYRK requirements.

Each matrix is normalized and orthogonalized independently. Momentum, weight
decay, update scaling, and per-parameter optimizer state are retained. Batched
GEMMs can round differently from separate matrix operations, so enabling batching
can change floating-point results. The optimizer checkpoint format is unchanged,
and the batch size can be changed when loading optimizer state. Transient memory
grows with the number of matrices stacked in a batch; lower the limit if needed.
