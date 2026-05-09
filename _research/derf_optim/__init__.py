"""Derf throughput optimization experiments.

Three approaches to recover the ~27% throughput lost from unfusing TE's
LayerNormColumnParallelLinear when --normalization Derf is selected:

* option1_compile.py  -- torch.compile of (Derf + nn.Linear) composite
* option2_triton.py   -- custom Triton fused norm-linear kernel
* option3_te_patch/   -- vendored TE fork with Derf in the CUDA template

Each option exposes a `make_qkv_class()` and `make_fc1_class()` factory that
returns a class matching Megatron's expected `linear_qkv` / `linear_fc1`
submodule interface (forward returns `(output, bias)` tuple), with the
norm fused in. The spec wiring in `megatron/core/models/gpt/gpt_layer_specs.py`
checks `APERTUS_DERF_OPTIM` and swaps these in when set.
"""
