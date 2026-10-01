For new or modified code, keep files directly under this directory generic and
independent of any particular layer implementation. Put layer-specific behavior in
`layers/` or with the owning layer's code. Top-level files should contain shared
`HybridModel` orchestration and only the minimal registration or dispatch needed to
compose layers. Keep parallelization logic in this directory similarly minimal; place
it in the appropriate parallelism package and retain only the integration needed by
`HybridModel`.

## Hybrid layer specs

Keep the module specfication-related contents of `hybrid_layer_specs.py` comptime-available. Do not switch behavior at runtime depending on CLI arguments, configuration values, environment variables, etc. Instead, create new, unambiguously named `ModuleSpec`s for each behavior you want to support. The correct specs should then be composed at model construction time. We want `hybrid_layer_specs.py` to only define one precise behavior per module spec. We should be able to understand the exact theory that is executed at runtime just from reading the module specs at comptime. For example:

Avoid the following:
```python
def get_hybrid_stack_spec(add_input_norms: bool = False) -> ModuleSpec:
    input_norm = TENorm if add_input_norms else IdentityOp
    return ModuleSpec(
        module=HybridStack,
        submodules=HybridStackSubmodules(
            mamba_layer=ModuleSpec(
                module=MambaLayer,
                submodules=MambaLayerSubmodules(
                    norm=input_norm,
                    mixer=ModuleSpec(
                        module=MambaMixer,
                        submodules=MambaMixerSubmodules(
                            in_proj=TELayerNormColumnParallelLinear, out_proj=TERowParallelLinear
                        ),
                    ),
                    mamba_bda=get_bias_dropout_add,
                ),
            ),
[...]
```

Instead, do this:
```python
input_norm = TENorm
hybrid_stack_spec = ModuleSpec(
    module=HybridStack,
    submodules=HybridStackSubmodules(
        mamba_layer=ModuleSpec(
            module=MambaLayer,
            submodules=MambaLayerSubmodules(
                mixer=ModuleSpec(
                    norm=input_norm,
                    module=MambaMixer,
                    submodules=MambaMixerSubmodules(
                        in_proj=TELayerNormColumnParallelLinear, out_proj=TERowParallelLinear
                    ),
                ),
                mamba_bda=get_bias_dropout_add,
            ),
        ),
[...]
```

The `input_norm` can then be configured at comptime (e.g., to be `IdentityOp`), but must not be configured at runtime. Users can pass in custom-built `hybrid_stack_spec`s through the `--spec` argument.

For other examples, check out how optional support for fused MLA down-projections is handled, or how switching to GDN2 is handled.
