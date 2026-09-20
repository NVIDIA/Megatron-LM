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

Supported prototype boundary: SBHD or THD with at most 4096 total physical
query tokens per batch, `attention_dropout=0`, and an explicit TP process group
when TP>1. THD Stage-2 requires producer-owned immutable CPU metadata
`packed_seq_params.qsa_stage2_layout_cpu=(physical_cu, valid_lengths)`, with
both entries tuples of Python integers. `physical_cu` starts at zero and ends
at the full physical token count; `valid_lengths` has one true token length per
physical segment, with `0` for a trailing padding-only segment. The producer
must construct this metadata from the same source lengths as the device cu
fields; MCore cannot compare device cu values to CPU values without a GPU sync.
MCore rejects a replaced or mutated device cu on a cached packed object, but
does not check the device values against CPU values on the first use. The older
`qsa_stage2_valid_lengths` device tensor remains available for producer
compatibility; this Stage-2 loss consumes only the CPU tuple as its source of
true lengths.
The original THD cu fields remain physical, preserving the main attention/CP
route. The KL excludes invalid query rows and all
selected/tail teacher keys for those rows. Real causal queries cannot select
padding keys because each selected block and tail ends no later than that
query. Missing or structurally invalid CPU lengths fail before indexer projection;
physical zero-length segments are rejected. Longer batches fail before
projection or core attention. `coeff=0` keeps the original forward path and
selection format and does not require the new metadata.

Validation gates before production use:

- TP2 tests check explicit-group execution and equal replicated-indexer
  gradients across ranks, including padded THD; they do not compare the
  teacher to TP1 numerically.
- CP2 tests check the CP-averaged local-query gradient against CP1 for SBHD
  mean loss, and the summed local-query gradient against CP1 for THD per-token
  loss with padding under standard RoPE and mRoPE. The test applies the same
  global token divisor to each result; a real RL loss/optimizer step remains
  unverified.
- Single-GPU tests check unchanged main output and main QKV/hidden gradients,
  nonzero indexer query/key projection gradients, packed reentrant and
  nonreentrant checkpoint gradients, padding independence, a zero-valid tail
  segment, and early rejection of missing THD metadata and long inputs.
- The pipeline schedule now registers the DSA indexer loss scale hook for
  `gdn` only when the QSA coefficient is positive. Coefficient zero leaves
  the prior schedule hook choice intact.
- The core retains the most recent selection and its indexer graph for
  selective recompute. Multi-microbatch peak memory and selective-core
  checkpoint lifetime need measurement.
- The Python teacher gathers up to `K_B * R + R - 1` keys per query chunk.
  Large `R` or `K_B` can exceed memory despite bounded asymptotic size.
  The short-geometry THD route keeps the rectangular pooled-key table. It
  cannot be merged directly with the separate mixed-document compact router:
  that route needs differentiable pooled keys and a per-document block prefix
  carried into KL. The CPU layout is validated once per packed object and its
  device tensors are cached; the Stage-2 length check and THD max-document
  calculation no longer read GPU scalars. Other pre-existing syncs remain:
  `_pool_keys` reads `doc_ids.max()`, `_select_blocks` reads `all_selected`,
  and `qsa_stage2_sparse_kl` reads several dynamic validity predicates.
  Throughput still needs measurement and further sync removal.
- No Bridge/Relax coefficient transport, optimizer membership, checkpoint
  save/resume, full-model training step, or 256K execution is established.
  The current Relax Stage-2 producer supplies only a device true-length tensor;
  its separate CPU layout proposal must be integrated before using this MCore
  commit. Dynamic CP,
  MTP, and mixed-document compact routing have no Stage-2 acceptance yet.
