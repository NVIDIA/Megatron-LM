# HyperConnection Block-Level Recomputation Design

## Core Design

Extend `CheckpointWithoutOutput` with an optional `ckpt_manager` parameter. When provided, checkpoints auto-register to the manager, and `discard_output_and_register_recompute` only discards output without registering individual hooks (manager handles unified hook).

**Key Insight**: `CheckpointWithoutOutput` detaches input and saves it to `ctx.saved_tensors` during forward. When checkpoint[i]'s output is checkpoint[i+1]'s input, they share the same tensor storage. By recomputing in forward order, checkpoint[i]'s `_recompute` writes results back to the original storage, and checkpoint[i+1]'s `saved_tensors` automatically gets the correct data.

**Critical**: The last layer's final BDA output in the block must NOT be checkpointed. It serves as the `hook_tensor` for registering the unified recompute hook.

```
Forward Pass:
┌─────────────────────────────────────────────────────────────────────┐
│  TransformerBlock.forward()                                         │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ manager = MHCBlockRecomputeManager()                          │  │
│  │                                                               │  │
│  │ for i, layer in enumerate(self.layers):                       │  │
│  │     is_last = (i == len(self.layers) - 1)                     │  │
│  │     hidden = layer(..., mhc_recompute_manager=manager,        │  │
│  │                        is_last_layer_in_block=is_last)        │  │
│  │                                                               │  │
│  │ ┌─────────────────────────────────────────────────────────┐   │  │
│  │ │ Layer 0..N-2 (NOT last layer):                          │   │  │
│  │ │   - All HyperConnection ops: CHECKPOINTED               │   │  │
│  │ │   - All BDAs: CHECKPOINTED                              │   │  │
│  │ │   - All outputs: DISCARDED                              │   │  │
│  │ └─────────────────────────────────────────────────────────┘   │  │
│  │                                                               │  │
│  │ ┌─────────────────────────────────────────────────────────┐   │  │
│  │ │ Layer N-1 (LAST layer, is_last_layer_in_block=True):    │   │  │
│  │ │   - HyperConnection ops: CHECKPOINTED                   │   │  │
│  │ │   - Self-Attn BDA: CHECKPOINTED                         │   │  │
│  │ │   - MLP BDA: NOT CHECKPOINTED ← hook_tensor             │   │  │
│  │ └─────────────────────────────────────────────────────────┘   │  │
│  │                                                               │  │
│  │ manager.discard_all_outputs_and_register_unified_recompute(   │  │
│  │     hook_tensor=hidden_states  # last layer's MLP BDA output  │  │
│  │ )                                                             │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

Backward Pass:
┌─────────────────────────────────────────────────────────────────────┐
│  grad(last_mlp_bda_output) triggers unified recompute hook          │
│      │                                                              │
│      ▼                                                              │
│  for each checkpoint in forward_order:                              │
│      checkpoint._recompute(None)  # restores output storage         │
│      │                                                              │
│      ▼                                                              │
│  Continue backward with recomputed activations                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Implementation

### CheckpointWithoutOutput (Modified)

Location: `megatron/core/tensor_parallel/random.py`

```python
class CheckpointWithoutOutput(object):
    """
    Checkpoint a model or part of the model and release the output.
    
    When ckpt_manager is provided:
    - checkpoint() auto-registers this object to the manager
    - discard_output_and_register_recompute() becomes a no-op
    """

    def __init__(self, fp8=False, ckpt_manager=None):
        self.fp8 = fp8 is not None
        self.ckpt_manager = ckpt_manager
        self.run_function = None
        self.fwd_cpu_rng_state = None
        self.fwd_cuda_rng_state = None
        self.fwd_cuda_rng_state_tracker = None
        self.ctx = None
        self.outputs = None

    def checkpoint(self, run_function, *args):
        """Checkpoint function. Auto-registers to ckpt_manager if provided."""
        self.run_function = run_function
        self.rng_states = _get_all_rng_states()

        outputs = CheckpointWithoutOutputFunction.apply(run_function, self, *args)
        self.outputs = outputs
        if isinstance(self.outputs, torch.Tensor):
            self.outputs = (self.outputs,)
        
        # Auto-register to manager if provided
        if self.ckpt_manager is not None:
            self.ckpt_manager.add_checkpoint(self)
        
        return outputs

    def discard_output_and_register_recompute(self, hook_tensor):
        """
        Release the output tensor storages and register the recompute hook.
        
        If ckpt_manager is provided, this is a NO-OP. Manager handles everything
        via discard_all_outputs_and_register_unified_recompute().
        """
        if self.ckpt_manager is not None:
            return  # No-op when manager is set
        
        # Discard output storage
        for output in self.outputs:
            output.untyped_storage().resize_(0)

        # Register individual recompute hook
        if hook_tensor.requires_grad:
            hook_tensor.register_hook(self._recompute)

    def _recompute(self, _):
        """Used as a hook to recompute the output."""
        # ... (unchanged)
```

### MHCBlockRecomputeManager

Location: `megatron/core/tensor_parallel/random.py`

```python
class MHCBlockRecomputeManager:
    """
    Manages multiple CheckpointWithoutOutput objects within a TransformerBlock
    for HyperConnection computations, enabling unified recomputation during backward pass.
    """

    def __init__(self):
        self.checkpoints = []

    def add_checkpoint(self, ckpt: CheckpointWithoutOutput):
        self.checkpoints.append(ckpt)

    def discard_all_outputs_and_register_unified_recompute(self, hook_tensor: Tensor):
        # Discard all checkpoint outputs to save memory
        for ckpt in self.checkpoints:
            for output in ckpt.outputs:
                output.untyped_storage().resize_(0)

        # Register unified recompute hook on hook_tensor
        # hook_tensor should be the last layer's final BDA output (NOT checkpointed)
        if hook_tensor.requires_grad:
            hook_tensor.register_hook(self._unified_recompute_hook)

    def _unified_recompute_hook(self, grad_output):
        for ckpt in self.checkpoints:
            ckpt._recompute(None)
```

### HyperConnectionModule

```python
class HyperConnectionModule(MegatronModule):
    
    def forward(
        self,
        hidden_states: Tensor,
        residual: Tensor, 
        training: bool = True,
        mhc_recompute_manager: Optional[MHCBlockRecomputeManager] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if mhc_recompute_manager is not None:
            return self._forward_with_checkpoint(
                hidden_states, residual, mhc_recompute_manager
            )
        else:
            return self._forward_normal(hidden_states, residual)
    
    def _forward_normal(
        self, hidden_states: Tensor, residual: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        h_pre, h_post, h_res = self.compute_mappings(hidden_states)
        aggregated = self.aggregate(hidden_states, h_pre)
        mixed = self.apply_h_res(h_res, residual)
        return aggregated, mixed, h_post
    
    def _forward_with_checkpoint(
        self,
        hidden_states: Tensor,
        residual: Tensor,
        manager: MHCBlockRecomputeManager,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        from megatron.core.tensor_parallel.random import CheckpointWithoutOutput
        
        # Checkpoints auto-register to manager via ckpt_manager parameter
        h_pre, h_post, h_res = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
            self.compute_mappings, hidden_states
        )
        
        aggregated = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
            self.aggregate, hidden_states, h_pre
        )
        
        mixed = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
            self.apply_h_res, h_res, residual
        )
        
        return aggregated, mixed, h_post
    
    def apply_h_post_with_checkpoint(
        self,
        x_with_bias: Tuple[Tensor, Optional[Tensor]],
        h_post: Tensor,
        manager: Optional[MHCBlockRecomputeManager] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if manager is not None:
            from megatron.core.tensor_parallel.random import CheckpointWithoutOutput
            return CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
                self.apply_h_post, x_with_bias, h_post
            )
        else:
            return self.apply_h_post(x_with_bias, h_post)
```

### TransformerLayer

```python
class TransformerLayer(GraphableMegatronModule, BaseTransformerLayer):
    
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        ...,
        mhc_recompute_manager: Optional[MHCBlockRecomputeManager] = None,
        is_last_layer_in_block: bool = False,
    ):
        hidden_states, context = self._forward_attention(
            hidden_states,
            attention_mask,
            ...,
            mhc_recompute_manager=mhc_recompute_manager,
        )
        
        hidden_states = self._forward_mlp(
            hidden_states,
            ...,
            mhc_recompute_manager=mhc_recompute_manager,
            is_last_layer_in_block=is_last_layer_in_block,
        )
        
        return hidden_states, context
    
    def _forward_attention(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        ...,
        mhc_recompute_manager: Optional[MHCBlockRecomputeManager] = None,
    ):
        residual = hidden_states
        
        if self.config.enable_hyper_connections and self.do_self_attention_hyper_connection:
            hidden_states, residual, self_attn_hc_h_post = self.self_attention_hyper_connection(
                hidden_states,
                residual,
                mhc_recompute_manager=mhc_recompute_manager,
            )
        
        input_layernorm_output = self.input_layernorm(hidden_states)
        
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            ...
        )
        
        if self.config.enable_hyper_connections and self.do_self_attention_hyper_connection:
            attention_output_with_bias = self.self_attention_hyper_connection.apply_h_post_with_checkpoint(
                attention_output_with_bias,
                self_attn_hc_h_post,
                manager=mhc_recompute_manager,
            )
        
        # Self-Attention BDA: always checkpoint when manager is set
        self.self_attn_bda_checkpoint = tensor_parallel.CheckpointWithoutOutput(
            ckpt_manager=mhc_recompute_manager
        )
        hidden_states = self.self_attn_bda_checkpoint.checkpoint(
            self.self_attn_bda(...),
            attention_output_with_bias, residual, self.hidden_dropout
        )
        self.self_attn_bda_checkpoint.discard_output_and_register_recompute(hidden_states)
        
        return hidden_states, context
    
    def _forward_mlp(
        self,
        hidden_states: Tensor,
        ...,
        mhc_recompute_manager: Optional[MHCBlockRecomputeManager] = None,
        is_last_layer_in_block: bool = False,
    ):
        residual = hidden_states
        
        if self.config.enable_hyper_connections and self.do_mlp_hyper_connection:
            hidden_states, residual, mlp_hc_h_post = self.mlp_hyper_connection(
                hidden_states,
                residual,
                mhc_recompute_manager=mhc_recompute_manager,
            )
        
        # LayerNorm checkpoint
        if self.recompute_pre_mlp_layernorm:
            self.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput(
                ckpt_manager=mhc_recompute_manager
            )
            pre_mlp_layernorm_output = self.pre_mlp_norm_checkpoint.checkpoint(
                self.pre_mlp_layernorm, hidden_states
            )
        else:
            pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)
        
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output, ...)
        
        if self.recompute_pre_mlp_layernorm:
            self.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(
                mlp_output_with_bias[0]
            )
        
        if self.config.enable_hyper_connections and self.do_mlp_hyper_connection:
            mlp_output_with_bias = self.mlp_hyper_connection.apply_h_post_with_checkpoint(
                mlp_output_with_bias,
                mlp_hc_h_post,
                manager=mhc_recompute_manager,
            )
        
        return self._forward_post_mlp(
            mlp_output_with_bias, residual, mhc_recompute_manager, is_last_layer_in_block
        )
    
    def _forward_post_mlp(
        self,
        mlp_output_with_bias: Tuple[Tensor, Optional[Tensor]],
        residual: Tensor,
        mhc_recompute_manager: Optional[MHCBlockRecomputeManager] = None,
        is_last_layer_in_block: bool = False,
    ):
        # MLP BDA: checkpoint only if NOT the last layer in block
        # Last layer's MLP BDA output serves as hook_tensor for unified recompute
        if mhc_recompute_manager is not None and not is_last_layer_in_block:
            self.mlp_bda_checkpoint = tensor_parallel.CheckpointWithoutOutput(
                ckpt_manager=mhc_recompute_manager
            )
            hidden_states = self.mlp_bda_checkpoint.checkpoint(
                self.mlp_bda(...),
                mlp_output_with_bias, residual, self.hidden_dropout
            )
            self.mlp_bda_checkpoint.discard_output_and_register_recompute(hidden_states)
        else:
            # Last layer OR no manager: normal BDA without checkpoint
            hidden_states = self.mlp_bda(...)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )
        
        return hidden_states
```

### TransformerBlock

No need for separate forward functions. Simply pass `is_last_layer_in_block` to each layer.

```python
class TransformerBlock(GraphableMegatronModule, MegatronModule):
    
    def forward(self, hidden_states, ...):
        # Determine if MHC recompute should be used
        use_mhc_recompute = (
            self.training and
            self.config.enable_hyper_connections and
            self.config.recompute_hyper_connections
        )
        
        manager = MHCBlockRecomputeManager() if use_mhc_recompute else None
        
        # Expand for hyper connections if needed
        if self.config.enable_hyper_connections and self.pre_process:
            hidden_states = HyperConnectionModule.input_expand(
                hidden_states, self.num_residual_streams
            )
        
        # Forward through layers
        num_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            is_last = (i == num_layers - 1)
            
            hidden_states, context = layer(
                hidden_states=hidden_states,
                ...,
                mhc_recompute_manager=manager,
                is_last_layer_in_block=is_last,
            )
        
        # Contract if needed
        if self.config.enable_hyper_connections and self.has_final_layernorm_in_this_stage():
            hidden_states = HyperConnectionModule.output_contract(
                hidden_states, self.num_residual_streams
            )
        
        # Final layernorm
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
        
        # Register unified recompute hook on last layer's output
        # (which is NOT checkpointed, so its storage is valid)
        if manager is not None:
            manager.discard_all_outputs_and_register_unified_recompute(hidden_states)
        
        return hidden_states
```

## Configuration

```python
@dataclass
class TransformerConfig:
    recompute_hyper_connections: bool = False
    """
    Enable recomputation for HyperConnection intermediate activations.
    
    Requirements:
    - Only effective when enable_hyper_connections=True and training=True
    - Must use recompute_granularity='selective'
    - Cannot be used together with recompute_mlp=True
    """

# Example usage
config = TransformerConfig(
    hidden_size=4096,
    num_layers=32,
    enable_hyper_connections=True,
    num_residual_streams=4,
    recompute_hyper_connections=True,
    recompute_granularity='selective',
    recompute_pre_mlp_layernorm=True,
    recompute_mlp=False,  # Must be disabled
)
```

## Constraint

When MHC recompute is enabled, `recompute_pre_mlp_layernorm` and `recompute_mlp` **cannot be enabled simultaneously** because they use different checkpoint mechanisms that conflict with each other.

---

**Version**: v2.4  
**Date**: 2026-01-20  
**Status**: Draft
