# In-tree CuTe DSL GDR kernels

This directory contains the SM100 Gated Delta Rule forward and fused backward
kernels used by the Megatron Core internal backend. The implementation was
migrated from the former out-of-tree integration and no longer requires a Git
submodule or separately installed extension package.

`fused_gdr_bwd_cute` is vendored from
`mcore_gdn_opt@ca349c26549c0a6bf2f69dc2b29936013550a44f`; its local
`kernel.md` records the exact source hash and supported shape contract.

The prefill wrapper and kernel are derived from FlashInfer commit
`e8d31317bedb4efd52559a2234f4cb9e83428cb9`. Source files retain their original
copyright and SPDX license notices. See `gdn_blackwell_chunked_kernel.md` for
the kernel design and supported shape contract.
