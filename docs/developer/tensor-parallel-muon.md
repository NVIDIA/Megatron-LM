# Tensor-parallel Muon parameter metadata

Muon uses a parameter's logical matrix shape when orthogonalizing its gradient.
In `duplicated` TP mode, a parameter with `partition_dim=0` or `1` is gathered
across the tensor-parallel group before Newton-Schulz; an unsharded parameter
uses `partition_dim=-1`. The `tensor_model_parallel` flag alone does not select
this optimizer path.

A duplicated TE projection owns a complete weight on every TP rank. TE receives
`parallel_mode=None` for this projection, but its default parameter metadata
can retain `partition_dim=0`. The MCore wrappers therefore explicitly restore
all three replicated-parameter attributes after TE initializes the parameters:

- `tensor_model_parallel=False`
- `partition_dim=-1`
- `partition_stride=1`

This applies to both `TELinear` in duplicated mode and
`TERMSNormDuplicatedLinear`. Expert/data-parallel reduction attributes retain
their existing meanings. Without this normalization, duplicated Muon can
concatenate repeated copies of an already complete matrix and compute an update
for the wrong matrix shape.

`test_duplicated_telinear_muon_update_matches_unsharded_reference` in
`tests/unit_tests/test_emerging_optimizers.py` exercises a real TE parameter
and real Newton-Schulz updates at TP2 against a complete-matrix TP1 reference,
including both tall and wide weight matrices.
`test_duplicated_linear_has_replicated_tensor_parallel_metadata` in
`tests/unit_tests/transformer/moe/test_latent_moe_layer.py` covers both wrappers.
