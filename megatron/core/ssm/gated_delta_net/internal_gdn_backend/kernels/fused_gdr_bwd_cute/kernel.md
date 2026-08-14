# SM100 fused GDR backward

This package is vendored byte-for-byte from
`mcore_gdn_opt@ca349c26549c0a6bf2f69dc2b29936013550a44f`.
The production `kernel.py` source SHA256 is
`29dec2291ee7f06792a8aec4f12b5377af71057420d32ca0eb65447a110ca271`.

The kernel consumes two packed, 64-token-aligned sequences with 64 heads and
head dimension 128. Dedicated warp roles stage inputs with TMA, execute the
tcgen05 recurrent backward pipeline, reduce gate and beta gradients, and store
`dq`, `dk`, `dv`, `dg`, `db`, and `dh0`. Ownership is communicated through the
source-defined mbarrier pipeline; the vendored implementation intentionally
preserves that verified schedule without Megatron-specific kernel edits.

The Megatron adapter reshapes dense `B=2` inputs to packed
`cu_seqlens=[0,T,2T]`, promotes gate and beta tensors to FP32, supplies a zero
final-state gradient when absent, and restores output shapes and dtypes.
