"""Throughput optimisation for the DyT/Derf normalisation path.

When ``--normalization`` is `DyT` or `Derf`, the spec unfuses the TE
LayerNormColumnParallelLinear (TE's fused norm-linear kernel only knows
LayerNorm and RMSNorm), costing roughly 27% throughput vs the RMSNorm
baseline at our 350M / bf16 / TP=1 shape.

This package provides one Python-side recovery path:

* ``option1_compile.py`` -- a TP=1 composite module that wraps the norm and
  ``F.linear`` in a ``torch.compile(fullgraph=True)`` region. Inductor
  fuses the elementwise norm into the matmul prologue, recovering most of
  the throughput. Wired via ``APERTUS_DERF_OPTIM=compile``.

See ``RESULTS.md`` for the empirical comparison against five other
approaches (Triton-fused matmul, hand-rolled CUDA, TE-style two-kernel
pipeline, etc.) and why this is the recommended ship.
"""
