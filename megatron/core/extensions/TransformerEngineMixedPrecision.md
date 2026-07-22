# About

Megatron training exposes the argument "--te-precision-config-file"
to allow experimentation with fine-grained control over the precision
of modules within a megatron network.

## Design Goals

The design aims to support configuration of the precision of linear
and grouped linear modules via the selection of a transformer engine
quantization recipe.

The fp8_autocast abstraction is already used to enable and disable a
single quantization recipe when evaluating the forward pass of a network.
This same mechanism is extended to execute targeted layers with the
desired quantization recipe, permitting mixed precision recipes.

The configurations function by optionally overriding the precision a module
would execute in. Not every module must have a configured override. Modules
are checked by module name against a sequence of patterns to determine if
an override recipe is applicable. By default, if the non-overridden precision
of a layer is non-quantized, as the primary desired use case is to customize
modules that are already quantized, and it is useful to respect other arguments
like `--first-last-layers-bf16`.

## Limitations

Relying on the module name to match against a configuration means the match is
executed post-initialization, and initialization customization for a recipe
override such as `fp4-param` and `fp8-param` are not in scope.

The validation precision configurations rely on self.training. They have not
yet been verified compatible with cuda-graphs and/or activation recompute.

There are some decisions in megatron that are made using the TransformerConfig's
settings for fp4 and fp8, possibly including layer number rather than using the
quantization autocast context. The configured overrides do not inform these
decisions with the current implementation.

## Validation precision

It is supported to configure a different precision when evaluating against the
validation set (when module.training is False). When evaluating a quantization
recipe, having a consistent forward pass for evaluation versus a baseline isolates
the quality of learning from the ability to infer with the quantization.

## Recipe configuration

Recipe configurations are named entries in a "configs" dictionary.

These examples show an mxfp8 recipe, a bf16 recipe, an mxfp8 recipe that
evaluates in bf16, and an nvfp4 recipe that evaluates in bf16.
```
configs:
  mxfp8:
    transformer_engine_config_type: "TEQuantizationParams"
    training_recipe:
      fp8_quantization_recipe: "mxfp8"
  bf16:
    transformer_engine_config_type: "TEQuantizationParams"
    training_recipe: {}
  mxfp8_evaluate_bf16:
    transformer_engine_config_type: "TEQuantizationParams"
    training_recipe:
      fp8_quantization_recipe: "mxfp8"
    evaluation_recipe: {}
  nvfp4_evaluate_bf16:
    transformer_engine_config_type: "TEQuantizationParams"
    training_recipe:
      fp4_quantization_recipe: "nvfp4"
    evaluation_recipe: {}
```

Recipes are selected by matchers. Currently implemented are glob style
expressions.

Matchers are ordered, and the first enabled matcher to match against
a module name chooses the config from the configs list.

In this example, assuming a default quantization recipe is enabled,
attention linear modules `linear_qkv` and `linear_proj` are selected
for the "bf16" recipe override and mamba mixer linear layers `out_proj`
and `in_proj` are selected for the "mxfp8" recipe override.

```
matchers:
  attn_qkv_bf16:
    config: "bf16"
    type: "glob"
    pattern: "*.linear_qkv"
    enabled: true
  attn_proj_bf16:
    config: "bf16"
    type: "glob"
    pattern: "*.linear_proj"
    enabled: true
  mamba_outproj_mxfp8:
    config: "mxfp8"
    type: "glob"
    pattern: "*mixer.out_proj"
    enabled: true
  mamba_inproj_mxfp8:
    config: "mxfp8"
    type: "glob"
    pattern: "*mixer.in_proj"
    enabled: true
```

Matches or modules that do not match to a configuration, and execute with their
default precision, will be logged so that quantization configurations can be
observed. Make sure to set `--logging-level` (to 20) in order to emit to logs.
