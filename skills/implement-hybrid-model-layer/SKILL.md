---
name: implement-hybrid-model-layer
description: Implement a new layer for the `HybridModel`. Use when the user asks you to implement a new layer. In case it's unclear which model the user refers to, ask them to clarify which model they want the layer implemented for before activating the skill.
metadata:
  author: Jan Ebert <janpabloe@nvidia.com>
---

<!-- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

To implement a new layer for the [`HybridModel`](../../megatron/core/models/hybrid/hybrid_model.py) in [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), you can generally follow these steps:

1. Check out [`megatron/core/models/hybrid/AGENTS.md`](../../megatron/core/models/hybrid/AGENTS.md) for general instructions on working with `HybridModel`.
2. Find a good place for the new layer. Generally, methods based on SSMs (state-space models) belong in [`megatron/core/ssm`](../../megatron/core/ssm). Methods based on Attention and/or Transformers belong in [`megatron/core/transformer`](../../megatron/core/transformer). `HybridModel`-specific layers belong in [`megatron/core/models/hybrid/layers`](../../megatron/core/models/hybrid/layers).
3. Create a config class for the new layer. The new config class should subclass [`TransformerConfig`](../../megatron/core/transformer/transformer_config.py). New fields should only be added to the new config class. They should _not_ be added to `TransformerConfig` unless absolutely required.
4. Implement the layer itself. Check out the [`parallelism-guide.md`](../../docs/user-guide/parallelism-guide.md) for settings to consider for highly parallel workloads. Consult documentation in the [`docs/user-guide/features`](../../docs/user-guide/features) directory for more in-depth feature documentation.
5. Integrate the new layer into the [`HybridStackSubmodules`](../../megatron/core/models/hybrid/hybrid_block.py) class and various [`ModuleSpec` setups that are provided for `HybridModel`](../../megatron/core/models/hybrid/hybrid_layer_specs.py).
6. Integrate the new layer into the [`HybridStack.__init__`](../../megatron/core/models/hybrid/hybrid_block.py) method.
7. Add a new symbol and mapping for the layer to the [`Symbols`](../../megatron/core/models/hybrid/layers/utils.py) class.
8. Write tests for the new functionality. Include at least one integration test that builds and executes an actual model. Be sure to check the various permutations of parallelism settings that the layer supports.

Deviate as necessary; this recipe isn't set in stone, but it should provide guidance and resources for successful integration of a new layer.

Before you start the implementation, make sure you have all the necessary details. Ask the user to provide more information in case the request is not clear enough. Present questions important to resolve before starting implementation work to the user for discussion.

When done, summarize your work to the user so they can review the new features and provide feedback. Make it easy for the user to understand trade-offs in the implementation.
