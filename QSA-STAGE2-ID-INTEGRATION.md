# Isolated QSA Stage-2 selected-ID integration

Base: MCore `e05299fa5`. The compact Stage-2 KL helper was ported from the
separately reviewed prototype `7b516830a`. This branch changes MCore only;
Bridge and Relax argument parsing are untouched.

`TransformerConfig.qsa_indexer_loss_coeff` defaults to zero. Positive values
activate KL only during training with gradients enabled and with
`MCORE_QSA_SPARSE_BACKEND=id_sparse`. Indexer Q and compressed K come from
`hidden.detach()` and remain differentiable; hard TopK stays under `no_grad`.
The teacher recomputes selected-token attention under `no_grad`, sums all TP
heads before block MaxPool, and passes the KL to `DSAIndexerLossAutoScaler`.
For CP, each rank computes KL on its own zigzag query rows against global keys
and the same full-sequence selection.

Supported prototype boundary: unpadded SBHD with at most 4096 total query
tokens per batch, `attention_dropout=0`, and an explicit TP process group when
TP>1. Packed/padded input and longer batches fail before projection or core
attention. `coeff=0` keeps the original forward path and selection format.

Validation gates before production use:

- TP2 test checks explicit-group execution and equal replicated-indexer
  gradients across ranks; it does not compare the teacher to TP1 numerically.
- CP2 test checks the CP-averaged local-query indexer gradient against the CP1
  full-query gradient for mean-loss semantics. Per-token-loss reduction is not
  covered.
- Single-GPU tests check unchanged main output and main QKV/hidden gradients,
  nonzero indexer query/key projection gradients, reentrant and nonreentrant
  checkpoint gradients, and early rejection of packed and long inputs.
- The core retains the most recent selection and its indexer graph for
  selective recompute. Multi-microbatch peak memory and selective-core
  checkpoint lifetime need measurement.
- The Python teacher gathers up to `K_B * R + R - 1` keys per query chunk.
  Large `R` or `K_B` can exceed memory despite bounded asymptotic size.
  Packed key padding and mixed-document pooling require a separate design.
- No Bridge/Relax coefficient transport, optimizer membership, checkpoint
  save/resume, full-model training step, or 256K execution is established.
