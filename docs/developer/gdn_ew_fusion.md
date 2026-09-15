# GDN Elementwise Fusion

`MCORE_GDN_FUSION=1` opts into fused Gated DeltaNet preparation and output
gating. Preparation combines causal convolution, SiLU, Q/K L2 normalization,
head expansion, layouts, and decay/write gates. Output gating combines GDN's
output RMSNorm with SiLU gating. The attention core and GQA layers are unchanged.

Preparation supports CUDA BF16 projections with batch size 1, 5,152 local
features, four key heads, sixteen value heads, head dimension 128, convolution
width 4, and context-parallel size 1. It requires FLA's convolution backward
kernel. Unsupported shapes and deterministic mode retain the existing path.

`MCORE_GDN_COMMON_OPT=1` independently enables fixed-launch unfused convolution.
Both options reuse validated packed-sequence metadata while tensor identities
and mutation versions remain unchanged. Both default to disabled.

The kernels preserve intermediate BF16 rounding and additive Q/K L2 epsilon
of `1e-6`. They support first-order autograd only. Floating-point operation
ordering can differ from the unfused path; correctness tests do not establish
training convergence or bitwise equivalence to that path.
