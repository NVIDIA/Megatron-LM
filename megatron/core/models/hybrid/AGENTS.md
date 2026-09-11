For new or modified code, keep files directly under this directory generic and
independent of any particular layer implementation. Put layer-specific behavior in
`layers/` or with the owning layer's code. Top-level files should contain shared
`HybridModel` orchestration and only the minimal registration or dispatch needed to
compose layers. Keep parallelization logic in this directory similarly minimal; place
it in the appropriate parallelism package and retain only the integration needed by
`HybridModel`.
