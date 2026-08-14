# Internal GDR CuTe DSL backend

Set `TransformerConfig.gdn_gdr_backend="internal"` to use the in-tree Gated
Delta Rule backend. It runs a local SM100 CuTe DSL forward kernel for the
general supported contract. For the narrower fused-backward contract, it uses
FLA forward/state preparation and the fused CuTe DSL backward from
`mcore_gdn_opt@ca349c26549c0a6bf2f69dc2b29936013550a44f`, matching that
revision's verified configuration. Unsupported stages fall back to FLA
automatically in `auto` mode.

The kernel sources live under
`megatron/core/ssm/gated_delta_net/internal_gdn_backend/kernels`; no Git
submodule or separate kernel package installation is required. CuTe DSL and
FLA are provided by the Megatron Core development/CI environment. Build and
run in that environment rather than installing dependencies on the host.

Runtime selection is controlled by `MCORE_GDN_INTERNAL_BACKEND`:

- `auto` (default): use CuTe DSL when the input is supported, otherwise FLA.
- `cute`: require the CuTe DSL path and raise if the input is unsupported.
- `fla`: always use FLA.

The CuTe path currently requires CUDA 13+, an SM100 GPU, contiguous bf16/fp16
BTHD tensors, equal Q/K/V head counts, head dimension 128, and sequence
lengths divisible by 64. Initial/final state, context parallelism, grouped
value attention, and in-kernel QK/beta transforms currently use the FLA
fallback.

The fused backward has a narrower verified contract: bf16, H=64, D=128, and
exactly two 64-token-aligned sequences. Dense `B=2` inputs are packed as
`[0,T,2T]`; packed inputs must provide three contiguous int32 offsets. The
adapter promotes gate and beta tensors to FP32 and supplies a zero final-state
gradient when the model does not request one.
