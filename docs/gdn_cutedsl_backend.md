# Internal GDR CuTe DSL backend

Set `TransformerConfig.gdn_gdr_backend="internal"` to route the Gated Delta
Rule through the FLA-compatible wrapper in `mcore_gdn_opt`. The wrapper uses
the optimized SM100 CuTe DSL/CUTLASS stages for supported shapes and defaults
to per-stage FLA fallback for unsupported shapes.

Initialize the pinned kernel source from the Megatron-LM checkout:

```bash
git submodule update --init --recursive third_party/mcore_gdn_opt
```

Build and install it inside the Megatron-LM CI/GPU development container, not
on the host. Follow `third_party/mcore_gdn_opt/README.md`; all kernel packages
must be installed in the same environment that runs Megatron-LM.

Runtime policy is controlled by `MCORE_GDN_OPT_BACKEND=auto|cuda|fla` and the
per-stage `MCORE_GDN_OPT_ENABLE_*` flags documented by `mcore_gdn_opt`. Use
`auto` (the default) for production fallback, and `cuda` when validating that
the requested shape takes only optimized kernels.
