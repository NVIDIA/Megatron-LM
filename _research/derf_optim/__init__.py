"""Throughput optimisation for the DyT/Derf normalisation path.

When ``--normalization`` is `DyT` or `Derf`, the spec must unfuse from TE's
``LayerNormColumnParallelLinear`` (TE's fused norm-linear kernel only knows
LayerNorm and RMSNorm), costing roughly 27% throughput vs the RMSNorm
baseline at our 350M / bf16 / TP=1 shape.

``option1_compile.py`` is the recommended recovery path: a TP=1 composite
module that wraps the norm and ``F.linear`` in a
``torch.compile(fullgraph=True)`` region. Inductor fuses the elementwise
norm into the matmul prologue, recovering most of the lost throughput.
Wired via ``APERTUS_DERF_OPTIM=compile``; see
``megatron/core/models/gpt/gpt_layer_specs.py`` for the dispatch.
"""
