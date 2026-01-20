# HyperConnection Block-Level Recomputation 设计文档

## 1. 概述

### 1.1 背景

在启用了HyperConnection（mHC）的GPTModel中，每个TransformerLayer会产生多个中间激活值，包括：
- `h_pre`, `h_post`, `h_res` 映射计算结果
- `apply_h_res` 后的residual中间结果
- 各个LayerNorm的输出
- attention/mlp的输出

现有的selective recomputation机制（`CheckpointWithoutOutput`）仅针对单个submodule（如layernorm）进行checkpoint，无法有效减少HyperConnection引入的额外内存开销。

### 1.2 目标

设计并实现一个**TransformerBlock级别**的recomputation机制，专门针对启用HyperConnection的场景：

1. **内存优化**：不存储每个HyperConnection的计算结果（h_pre, h_post, h_res, 以及apply_h_res后的residual）
2. **仅存储Block输入**：整个TransformerBlock只保存input tensor
3. **延迟Recompute**：在block output进行backward时，触发整个block内部的recomputation
4. **兼容性**：与现有的`CheckpointWithoutOutput`机制协同工作

### 1.3 核心思路

#### 简化设计原则

本设计的核心是**复用现有的 `CheckpointWithoutOutput` 机制**，不对其进行任何修改。通过引入一个轻量级的 `MHCBlockRecomputeManager` 来统一管理多个 checkpoint：

```
┌────────────────────────────────────────────────────────────────────────────┐
│                   简化设计：MHC Block Recompute Manager                     │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  核心原则：                                                                │
│  1. 不修改 CheckpointWithoutOutput                                         │
│  2. Manager 仅负责收集和统一调度                                           │
│  3. 利用 tensor storage 共享机制实现数据传递                               │
│  4. Manager 传入 HyperConnection.forward()，使 TransformerLayer 无感知    │
│                                                                            │
│  工作流程：                                                                │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │ Forward Pass                                                       │   │
│  │   1. TransformerBlock 创建 MHCBlockRecomputeManager                │   │
│  │   2. Manager 传入每个 HyperConnectionModule.forward()              │   │
│  │   3. HyperConnection 内部使用 CheckpointWithoutOutput 并注册到 mgr │   │
│  │   4. TransformerBlock 在 output 上调用 discard_and_register_...    │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                              │                                             │
│                              ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │ Backward Pass                                                      │   │
│  │   1. grad 到达 hook_tensor，触发 unified recompute hook            │   │
│  │   2. 按 forward 顺序依次调用每个 checkpoint 的 _recompute          │   │
│  │   3. 每个 _recompute 将 output 写回原 tensor storage               │   │
│  │   4. 下一个 checkpoint 的 input 自动获得正确数据                    │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

**关键洞察**：`CheckpointWithoutOutput` 在 forward 时会 detach input 并保存到 `ctx.saved_tensors`。当 checkpoint[i] 的 output 是 checkpoint[i+1] 的 input 时，它们共享同一个 tensor storage。因此，当我们按顺序 recompute 时，checkpoint[i] 的 `_recompute` 会将结果写回原 storage，checkpoint[i+1] 的 `saved_tensors` 自然就能获取到正确的数据。

#### Block 级别的数据流

```
Forward Pass:
┌─────────────────────────────────────────────────────────────────┐
│  TransformerBlock                                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ manager = MHCBlockRecomputeManager()                      │  │
│  │                                                           │  │
│  │ Input ───────────────────────────────────────────────────│  │
│  │    │                                                      │  │
│  │    ▼                                                      │  │
│  │ [HyperConnection Computations - All Checkpointed]        │  │
│  │    │ HyperConnection.forward(hidden, residual, manager)  │  │
│  │    │ ┌─ compute_mappings ─────────────────────────────┐  │  │
│  │    │ │   CheckpointWithoutOutput → manager.add()       │  │  │
│  │    │ │   output: (h_pre, h_post, h_res) ─ DISCARDED ──│──│──│─► freed
│  │    │ └─────────────────────────────────────────────────┘  │  │
│  │    │                                                      │  │
│  │    │ ┌─ aggregate ────────────────────────────────────┐  │  │
│  │    │ │   CheckpointWithoutOutput → manager.add()       │  │  │
│  │    │ │   output: aggregated ─ DISCARDED ──────────────│──│──│─► freed
│  │    │ └─────────────────────────────────────────────────┘  │  │
│  │    │                                                      │  │
│  │    │ ┌─ apply_h_res ──────────────────────────────────┐  │  │
│  │    │ │   CheckpointWithoutOutput → manager.add()       │  │  │
│  │    │ │   output: residual_mixed ─ DISCARDED ──────────│──│──│─► freed
│  │    │ └─────────────────────────────────────────────────┘  │  │
│  │    │                                                      │  │
│  │    ├─ LayerNorm                                          │  │
│  │    ├─ Attention                                          │  │
│  │    │                                                      │  │
│  │    │ ┌─ apply_h_post ─────────────────────────────────┐  │  │
│  │    │ │   CheckpointWithoutOutput → manager.add()       │  │  │
│  │    │ │   output ─ DISCARDED ──────────────────────────│──│──│─► freed
│  │    │ └─────────────────────────────────────────────────┘  │  │
│  │    │                                                      │  │
│  │    └─ BDA ───────────────────────────────────────────────│  │
│  │                                                           │  │
│  │ ... (repeat for MLP submodule)                           │  │
│  │                                                           │  │
│  │ Block Output ◄──────────────────────────────────────────│  │
│  │                                                           │  │
│  │ manager.discard_all_outputs_and_register_unified_recompute│  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

Backward Pass:
┌─────────────────────────────────────────────────────────────────┐
│  grad(hook_tensor) triggers unified recompute hook              │
│      │                                                          │
│      ▼                                                          │
│  for each checkpoint in forward_order:                          │
│      checkpoint._recompute(None)  # restores output storage     │
│      │                                                          │
│      ▼                                                          │
│  Continue backward with recomputed activations                  │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 详细设计

### 2.1 核心组件

#### 2.1.1 `MHCBlockRecomputeManager` 类

**位置**：`megatron/core/tensor_parallel/random.py`

**目的**：管理一个TransformerBlock内所有需要checkpoint的HyperConnection计算，统一调度recompute。

```python
class MHCBlockRecomputeManager:
    """
    MHC (Manifold-Constrained Hyper-Connections) Block-Level Recompute Manager.

    Manages multiple CheckpointWithoutOutput objects within a TransformerBlock for
    HyperConnection computations, enabling unified recomputation during backward pass.

    Design Philosophy:
    - This manager is passed into HyperConnectionModule.forward() so that the checkpoint
      logic is encapsulated within HyperConnection, making TransformerLayer unaware of
      the detailed checkpoint process.
    - When manager is None, HyperConnection operates normally without checkpointing.
    - When manager is provided, HyperConnection wraps its computations with
      CheckpointWithoutOutput and registers them to the manager.

    Usage:
        # In TransformerBlock:
        manager = MHCBlockRecomputeManager()

        # Pass manager to each layer's HyperConnection (via TransformerLayer)
        for layer in self.layers:
            hidden_states = layer.forward(..., mhc_recompute_manager=manager)

        # After all layers, register unified recompute on final output
        final_output = hidden_states.sum()  # or loss
        manager.discard_all_outputs_and_register_unified_recompute(final_output)
    """

    def __init__(self):
        """Initialize the MHCBlockRecomputeManager."""
        self.checkpoints = []

    def add_checkpoint(self, ckpt: CheckpointWithoutOutput):
        """Add a CheckpointWithoutOutput object to the manager."""
        self.checkpoints.append(ckpt)

    def discard_all_outputs_and_register_unified_recompute(self, hook_tensor: Tensor):
        """
        Discard all checkpoint outputs and register a unified recompute hook.
        """
        # Discard all checkpoint outputs to save memory
        for ckpt in self.checkpoints:
            for output in ckpt.outputs:
                output.untyped_storage().resize_(0)

        # Register unified recompute hook
        if hook_tensor.requires_grad:
            hook_tensor.register_hook(self._unified_recompute_hook)

    def _unified_recompute_hook(self, grad_output):
        """Unified recompute hook that recomputes all checkpoints in forward order."""
        for ckpt in self.checkpoints:
            ckpt._recompute(None)
```

**设计要点**：

1. **简单轻量**：Manager 只维护一个 checkpoint 列表，无需复杂的依赖图
2. **复用现有机制**：直接调用 `CheckpointWithoutOutput._recompute()`，无需重新实现
3. **顺序保证**：按添加顺序（即 forward 顺序）执行 recompute，自动满足数据依赖
4. **封装性**：Manager 传入 HyperConnection，TransformerLayer 无需感知 checkpoint 细节

### 2.2 HyperConnectionModule 修改

在 `HyperConnectionModule` 中增加对 `MHCBlockRecomputeManager` 的支持：

```python
class HyperConnectionModule(MegatronModule):
    """
    Unified mHC (Manifold-Constrained Hyper-Connections) module.
    """
    
    def forward(
        self,
        hidden_states: Tensor,
        residual: Tensor, 
        training: bool = True,
        mhc_recompute_manager: Optional[MHCBlockRecomputeManager] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Full mHC forward pass.
        
        Args:
            hidden_states: [s, b, n*C] - n-stream hidden states
            residual: [s, b, n*C] - n-stream hidden states (x_l)
            training: Whether in training mode
            mhc_recompute_manager: Optional manager for block-level recomputation.
                                   If provided, all HyperConnection computations will
                                   be wrapped with CheckpointWithoutOutput.
        
        Returns:
            aggregated: [s, b, C] - aggregated input for layer computation
            mixed: [s, b, n*C] - mixed output (H_res @ x_l)
            h_post: [s, b, n] - expansion weights
        """
        if mhc_recompute_manager is not None:
            # === Checkpoint mode: wrap each computation ===
            return self._forward_with_checkpoint(
                hidden_states, residual, mhc_recompute_manager
            )
        else:
            # === Normal mode: no checkpointing ===
            return self._forward_normal(hidden_states, residual)
    
    def _forward_normal(
        self, hidden_states: Tensor, residual: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Normal forward without checkpointing."""
        # Compute mappings
        h_pre, h_post, h_res = self.compute_mappings(hidden_states)
        
        # Aggregate for layer input
        aggregated = self.aggregate(hidden_states, h_pre)
        
        # Apply h_res to residual
        mixed = self.apply_h_res(h_res, residual)
        
        return aggregated, mixed, h_post
    
    def _forward_with_checkpoint(
        self,
        hidden_states: Tensor,
        residual: Tensor,
        manager: MHCBlockRecomputeManager,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward with checkpointing for block-level recomputation."""
        from megatron.core.tensor_parallel.random import CheckpointWithoutOutput
        
        # 1. Checkpoint: compute_mappings
        ckpt_compute = CheckpointWithoutOutput()
        h_pre, h_post, h_res = ckpt_compute.checkpoint(
            self.compute_mappings, hidden_states
        )
        manager.add_checkpoint(ckpt_compute)
        
        # 2. Checkpoint: aggregate
        ckpt_aggregate = CheckpointWithoutOutput()
        aggregated = ckpt_aggregate.checkpoint(
            self.aggregate, hidden_states, h_pre
        )
        manager.add_checkpoint(ckpt_aggregate)
        
        # 3. Checkpoint: apply_h_res
        ckpt_apply_h_res = CheckpointWithoutOutput()
        mixed = ckpt_apply_h_res.checkpoint(
            self.apply_h_res, h_res, residual
        )
        manager.add_checkpoint(ckpt_apply_h_res)
        
        return aggregated, mixed, h_post
    
    def apply_h_post_with_checkpoint(
        self,
        x_with_bias: Tuple[Tensor, Optional[Tensor]],
        h_post: Tensor,
        manager: Optional[MHCBlockRecomputeManager] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Apply H_post with optional checkpointing.
        
        This is called separately after the main layer computation (attention/mlp).
        """
        if manager is not None:
            from megatron.core.tensor_parallel.random import CheckpointWithoutOutput
            
            ckpt_apply_h_post = CheckpointWithoutOutput()
            result = ckpt_apply_h_post.checkpoint(
                self.apply_h_post, x_with_bias, h_post
            )
            manager.add_checkpoint(ckpt_apply_h_post)
            return result
        else:
            return self.apply_h_post(x_with_bias, h_post)
```

### 2.3 TransformerLayer 修改

**设计目标**：TransformerLayer 只需要将 `mhc_recompute_manager` 传递给 HyperConnection，无需感知具体的 checkpoint 过程。

#### 2.3.1 当前代码分析

查看当前 `_forward_mlp` 的实现（简化版）：

```python
def _forward_mlp(self, hidden_states, ...):
    # Residual connection
    residual = hidden_states

    # HyperConnection 前处理
    if self.config.enable_hyper_connections and self.do_mlp_hyper_connection:
        hidden_states, residual, mlp_hc_h_post = self.mlp_hyper_connection(
            hidden_states, residual
        )

    # Pre-MLP LayerNorm (可选 checkpoint)
    if self.recompute_pre_mlp_layernorm:
        self.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
        pre_mlp_layernorm_output = self.pre_mlp_norm_checkpoint.checkpoint(
            self.pre_mlp_layernorm, hidden_states
        )
    else:
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

    # MLP (可选 checkpoint)
    if self.recompute_mlp:
        mlp_output_with_bias = tensor_parallel.checkpoint(...)
    else:
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output, ...)

    # 注册 pre_mlp layernorm 的 recompute hook
    if self.recompute_pre_mlp_layernorm:
        self.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(
            mlp_output_with_bias[0]
        )

    # HyperConnection 后处理
    if self.config.enable_hyper_connections and self.do_mlp_hyper_connection:
        mlp_output_with_bias = self.mlp_hyper_connection.apply_h_post(
            mlp_output_with_bias, mlp_hc_h_post
        )

    return self._forward_post_mlp(mlp_output_with_bias, residual)
```

#### 2.3.2 修改后的代码

```python
class TransformerLayer(GraphableMegatronModule, BaseTransformerLayer):
    
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        ...,
        mhc_recompute_manager: Optional[MHCBlockRecomputeManager] = None,
    ):
        """
        TransformerLayer forward pass.
        
        Args:
            ...
            mhc_recompute_manager: Optional manager for MHC block-level recomputation.
                                   When provided, HyperConnection computations will be
                                   checkpointed and registered to this manager.
        """
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
        )
        
        return hidden_states, context
    
    def _forward_attention(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        ...,
        mhc_recompute_manager: Optional[MHCBlockRecomputeManager] = None,
    ):
        """Forward pass for self-attention with optional MHC recomputation."""
        
        # Residual connection
        residual = hidden_states
        
        # === Self-Attention HyperConnection 前处理 ===
        if self.config.enable_hyper_connections and self.do_self_attention_hyper_connection:
            # 将 manager 传入 HyperConnection，checkpoint 逻辑在其内部处理
            hidden_states, residual, self_attn_hc_h_post = self.self_attention_hyper_connection(
                hidden_states,
                residual,
                mhc_recompute_manager=mhc_recompute_manager,  # 新增参数
            )
        
        # LayerNorm (保持现有逻辑)
        input_layernorm_output = self.input_layernorm(hidden_states)
        
        # Self-Attention
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            ...
        )
        
        # === Self-Attention HyperConnection 后处理 ===
        if self.config.enable_hyper_connections and self.do_self_attention_hyper_connection:
            # 使用新方法，支持可选的 checkpoint
            attention_output_with_bias = self.self_attention_hyper_connection.apply_h_post_with_checkpoint(
                attention_output_with_bias,
                self_attn_hc_h_post,
                manager=mhc_recompute_manager,  # 新增参数
            )
        
        # BDA
        hidden_states = self.self_attn_bda(...)(
            attention_output_with_bias, residual, self.hidden_dropout
        )
        
        return hidden_states, context
    
    def _forward_mlp(
        self,
        hidden_states: Tensor,
        ...,
        mhc_recompute_manager: Optional[MHCBlockRecomputeManager] = None,
    ):
        """Forward pass for MLP with optional MHC recomputation."""
        
        # Residual connection
        residual = hidden_states
        
        # === MLP HyperConnection 前处理 ===
        if self.config.enable_hyper_connections and self.do_mlp_hyper_connection:
            # 将 manager 传入 HyperConnection，checkpoint 逻辑在其内部处理
            hidden_states, residual, mlp_hc_h_post = self.mlp_hyper_connection(
                hidden_states,
                residual,
                mhc_recompute_manager=mhc_recompute_manager,  # 新增参数
            )
        
        # === Pre-MLP LayerNorm ===
        # 注意：当启用 MHC recompute 时，pre_mlp_layernorm 的 recompute 也会被统一管理
        if self.recompute_pre_mlp_layernorm:
            self.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            pre_mlp_layernorm_output = self.pre_mlp_norm_checkpoint.checkpoint(
                self.pre_mlp_layernorm, hidden_states
            )
            # 如果启用了 MHC recompute，将 layernorm checkpoint 也加入 manager
            if mhc_recompute_manager is not None:
                mhc_recompute_manager.add_checkpoint(self.pre_mlp_norm_checkpoint)
        else:
            pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)
        
        # === MLP ===
        # 注意：当前实现中，recompute_mlp 和 MHC recompute 互斥（见下文说明）
        if self.recompute_mlp:
            mlp_output_with_bias = tensor_parallel.checkpoint(...)
        else:
            mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output, ...)
        
        # === 注册 pre_mlp layernorm 的 recompute hook ===
        # 关键：当启用 MHC recompute 时，不再单独注册 hook
        # 因为 manager 会统一处理所有 checkpoint 的 recompute
        if self.recompute_pre_mlp_layernorm and mhc_recompute_manager is None:
            self.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(
                mlp_output_with_bias[0]
            )
        
        # === MLP HyperConnection 后处理 ===
        if self.config.enable_hyper_connections and self.do_mlp_hyper_connection:
            mlp_output_with_bias = self.mlp_hyper_connection.apply_h_post_with_checkpoint(
                mlp_output_with_bias,
                mlp_hc_h_post,
                manager=mhc_recompute_manager,  # 新增参数
            )
        
        return self._forward_post_mlp(mlp_output_with_bias, residual)
```

### 2.4 TransformerBlock 修改

```python
class TransformerBlock(GraphableMegatronModule, MegatronModule):
    
    def forward(self, hidden_states, ...):
        if self._should_use_mhc_recompute():
            return self._forward_with_mhc_recompute(hidden_states, ...)
        else:
            return self._forward_normal(hidden_states, ...)
    
    def _should_use_mhc_recompute(self) -> bool:
        """Determine if MHC block-level recomputation should be used."""
        return (
            self.training and
            self.config.enable_hyper_connections and
            self.config.recompute_hyper_connections
        )
    
    def _forward_with_mhc_recompute(self, hidden_states, ...):
        """Forward pass with MHC block-level recomputation."""
        # Create recompute manager for this block
        manager = MHCBlockRecomputeManager()
        
        # Expand for hyper connections if needed
        if self.config.enable_hyper_connections and self.pre_process:
            hidden_states = HyperConnectionModule.input_expand(
                hidden_states, self.num_residual_streams
            )
        
        # Forward through layers, passing manager to each
        for layer in self.layers:
            hidden_states, context = layer(
                hidden_states=hidden_states,
                ...,
                mhc_recompute_manager=manager,  # 传入 manager
            )
        
        # Contract if needed
        if self.config.enable_hyper_connections and self.has_final_layernorm_in_this_stage():
            hidden_states = HyperConnectionModule.output_contract(
                hidden_states, self.num_residual_streams
            )
        
        # Final layernorm
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
        
        # Compute hook tensor (typically used with loss)
        # 注意：实际使用时，hook_tensor 应该是 loss 或其他依赖于所有 checkpoint 的 tensor
        hook_tensor = hidden_states.sum()
        
        # Discard all MHC intermediates and register unified recompute
        manager.discard_all_outputs_and_register_unified_recompute(hook_tensor)
        
        return hidden_states
```

## 3. 重要限制与约束

### 3.1 `recompute_pre_mlp_layernorm` 与 `recompute_mlp` 的互斥性

**当前限制**：在启用 MHC recompute 的场景下，`recompute_pre_mlp_layernorm` 和 `recompute_mlp` **不能同时启用**。

#### 原因分析

查看当前 `_forward_mlp` 的逻辑：

```python
# Pre-MLP LayerNorm checkpoint
if self.recompute_pre_mlp_layernorm:
    self.pre_mlp_norm_checkpoint = CheckpointWithoutOutput()
    pre_mlp_layernorm_output = self.pre_mlp_norm_checkpoint.checkpoint(
        self.pre_mlp_layernorm, hidden_states
    )

# MLP (可能也有 checkpoint)
if self.recompute_mlp:
    mlp_output_with_bias = tensor_parallel.checkpoint(...)  # 使用不同的 checkpoint 机制
else:
    mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output, ...)

# 关键：pre_mlp layernorm 的 recompute hook 注册在 mlp_output_with_bias[0] 上
if self.recompute_pre_mlp_layernorm:
    self.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(
        mlp_output_with_bias[0]
    )
```

**问题**：
1. `recompute_mlp` 使用的是 `tensor_parallel.checkpoint()`（PyTorch 风格的完整 checkpoint）
2. `recompute_pre_mlp_layernorm` 使用的是 `CheckpointWithoutOutput`（selective recompute）
3. 当两者同时启用时，`pre_mlp_norm_checkpoint.discard_output_and_register_recompute()` 的 hook 注册位置会与 MLP checkpoint 的重算逻辑冲突

#### 配置约束

启用 MHC recompute 时，必须满足以下配置：

```python
# 有效配置
config = TransformerConfig(
    enable_hyper_connections=True,
    recompute_hyper_connections=True,
    recompute_granularity='selective',  # 必须是 selective
    # 以下二选一：
    recompute_pre_mlp_layernorm=True,   # OK
    recompute_mlp=False,                 # 必须关闭
)

# 无效配置 - 会导致问题
config = TransformerConfig(
    enable_hyper_connections=True,
    recompute_hyper_connections=True,
    recompute_pre_mlp_layernorm=True,
    recompute_mlp=True,  # ❌ 不允许
)
```

#### 后续优化方向

使用 `MHCBlockRecomputeManager` 后，理论上可以统一管理所有的 checkpoint，包括 pre_mlp_layernorm 和 MLP。后续可以考虑：

1. 将 MLP 的 checkpoint 也改用 `CheckpointWithoutOutput`
2. 统一注册到 `MHCBlockRecomputeManager`
3. 移除 `recompute_pre_mlp_layernorm` 和 `recompute_mlp` 的互斥限制

这需要对 MLP checkpoint 机制进行重构，确保与 MHC recompute 兼容。

### 3.2 配置参数

```python
@dataclass
class TransformerConfig:
    # ... existing fields ...
    
    recompute_hyper_connections: bool = False
    """
    Whether to enable recomputation for HyperConnection intermediate activations.
    When enabled, all HyperConnection computations (compute_mappings, aggregate,
    apply_h_res, apply_h_post) will be checkpointed and their outputs discarded
    after forward pass. The outputs are recomputed during backward pass.
    
    Requirements:
    - Only effective when enable_hyper_connections=True and training=True
    - Must use recompute_granularity='selective'
    - Cannot be used together with recompute_mlp=True (see section 3.1)
    
    Recommended configuration:
        enable_hyper_connections=True
        recompute_hyper_connections=True
        recompute_granularity='selective'
        recompute_pre_mlp_layernorm=True
        recompute_mlp=False  # Important!
    """
```

## 4. 数据传递机制

### 4.1 Storage 共享机制详解

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     数据传递：Storage 共享机制                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Forward:                                                                   │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │ ckpt1.checkpoint│     │ ckpt2.checkpoint│     │ ckpt3.checkpoint│       │
│  │ compute_mappings│────►│   aggregate     │────►│  apply_h_res    │       │
│  │ out: h_pre,...  │     │ out: aggregated │     │ out: mixed      │       │
│  └────────┬────────┘     └────────┬────────┘     └────────┬────────┘       │
│           │                       │                       │                 │
│           │ detach & save         │ detach & save         │                 │
│           │ to ctx.saved_tensors  │ to ctx.saved_tensors  │                 │
│           ▼                       ▼                       ▼                 │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │ ckpt1.ctx       │     │ ckpt2.ctx       │     │ ckpt3.ctx       │       │
│  │ saved: [x]      │     │ saved: [x,h_pre]│     │ saved: [h_res,r]│       │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘       │
│                                                                             │
│  * h_pre, aggregated, mixed 等 tensor 的 detached 版本共享 storage          │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  After discard_all_outputs:                                                 │
│  所有 checkpoint output 的 storage 全部 resize_(0)，内存释放                │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Backward (unified recompute):                                              │
│                                                                             │
│  Step 1: ckpt1._recompute(None)                                             │
│    - inputs = ckpt1.ctx.saved_tensors  # [x]                                │
│    - h_pre_new, h_post_new, h_res_new = compute_mappings(x)                 │
│    - h_pre.storage = h_pre_new.storage  # 写回原 storage                    │
│                                                                             │
│  Step 2: ckpt2._recompute(None)                                             │
│    - inputs = ckpt2.ctx.saved_tensors  # [x, h_pre*]                        │
│    - h_pre* 现在指向 h_pre_new 的数据（storage 已被 refill）                │
│    - aggregated_new = aggregate(x, h_pre*)                                  │
│    - aggregated.storage = aggregated_new.storage                            │
│                                                                             │
│  ... 以此类推 ...                                                           │
│                                                                             │
│  关键：按 forward 顺序 recompute 确保数据依赖被正确满足                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 5. 内存分析

### 5.1 现有方案内存占用

对于一个启用HyperConnection的TransformerLayer（n个residual streams）：

| 组件 | Tensor数量 | 形状 | 内存估算 |
|------|-----------|------|---------|
| h_pre | 1 | [s, b, n] | s×b×n×dtype |
| h_post | 1 | [s, b, n] | s×b×n×dtype |
| h_res | 1 | [s, b, n, n] | s×b×n²×dtype |
| aggregated | 1 | [s, b, C] | s×b×C×dtype |
| residual_mixed | 1 | [s, b, n×C] | s×b×n×C×dtype |

总计每层约：`s×b×(2n + n² + C + nC)×dtype`

### 5.2 新方案内存占用

| 组件 | Tensor数量 | 形状 | 内存估算 |
|------|-----------|------|---------|
| block_input (saved by first ckpt) | 1 | [s, b, n×C] | s×b×n×C×dtype |
| checkpoint metadata | - | - | negligible |

总计整个block：`~s×b×n×C×dtype` + 少量 metadata

### 5.3 内存节省

假设 n = 4, C = 4096, L = 32 layers：

每层节省：`s×b×(2n + n² + C) ≈ s×b×(8 + 16 + 4096) ≈ s×b×4120`

对于32层的block，总体节省约为 **~4-5x**。

## 6. 实现计划

### Phase 1: 核心基础设施 ✅
1. 实现 `MHCBlockRecomputeManager` 类
2. 添加单元测试验证基本功能

### Phase 2: HyperConnectionModule 集成
1. 修改 `HyperConnectionModule.forward()` 支持 manager 参数
2. 实现 `_forward_with_checkpoint()` 方法
3. 实现 `apply_h_post_with_checkpoint()` 方法

### Phase 3: TransformerLayer 集成
1. 修改 `TransformerLayer.forward()` 支持 manager 参数
2. 修改 `_forward_attention()` 传递 manager
3. 修改 `_forward_mlp()` 传递 manager 并处理 layernorm checkpoint

### Phase 4: TransformerBlock 集成
1. 修改 `TransformerBlock.forward()` 添加 MHC recompute 路径
2. 添加配置参数验证
3. 实现 manager 的生命周期管理

### Phase 5: 测试与优化
1. 单元测试：验证数值正确性
2. 集成测试：与现有 checkpoint 机制兼容性
3. 性能测试：内存和速度 benchmark
4. 边界条件处理：PP/TP/CP 并行场景

## 7. 风险与缓解

### 7.1 数值精度
**风险**：重算可能因RNG状态不一致导致数值差异

**缓解**：
- `CheckpointWithoutOutput` 已有完善的 RNG 状态保存/恢复机制
- 添加数值精度验证测试

### 7.2 性能开销
**风险**：重算引入额外计算开销

**缓解**：
- HyperConnection 计算相对轻量，重算开销可控
- 仅在训练时启用
- 提供配置开关，允许用户权衡内存和计算

### 7.3 兼容性
**风险**：与现有 selective recompute、CPU offloading 等特性冲突

**缓解**：
- 复用现有 `CheckpointWithoutOutput` 机制，继承其兼容性
- 明确 `recompute_pre_mlp_layernorm` 和 `recompute_mlp` 的互斥关系
- 在 config 验证中添加检查

## 8. API 示例

```python
# 配置示例
config = TransformerConfig(
    hidden_size=4096,
    num_layers=32,
    enable_hyper_connections=True,
    num_residual_streams=4,
    # 启用 MHC recomputation
    recompute_hyper_connections=True,
    recompute_granularity='selective',
    recompute_pre_mlp_layernorm=True,
    recompute_mlp=False,  # 重要：必须关闭
)

# 模型使用
model = GPTModel(config=config, ...)

# 训练时自动启用 recomputation
for batch in dataloader:
    loss = model(batch)
    loss.backward()  # 触发 unified recompute hook
```

## 9. 待讨论问题

1. **与 TE checkpoint 的交互**：FP8 场景下如何与 TransformerEngine 的 checkpoint 机制协作？
2. **Pipeline Parallel 边界**：跨 PP stage 的 block 如何处理？
3. **统一 MLP checkpoint**：后续是否将 `recompute_mlp` 也改用 `MHCBlockRecomputeManager` 管理？

## 10. 参考

- 现有 `CheckpointWithoutOutput` 实现：`megatron/core/tensor_parallel/random.py`
- 新增 `MHCBlockRecomputeManager` 实现：`megatron/core/tensor_parallel/random.py`
- HyperConnection 实现：`megatron/core/transformer/hyper_connection.py`
- TransformerLayer 实现：`megatron/core/transformer/transformer_layer.py`
- mHC 论文：Manifold-Constrained Hyper-Connections

---

**文档版本**: v2.1  
**作者**: [待填写]  
**日期**: 2026-01-19  
**状态**: Draft - 待 Review

### 变更历史

| 版本 | 日期 | 变更内容 |
|------|------|----------|
| v1.0 | 2026-01-19 | 初始版本 |
| v1.1 | 2026-01-19 | 优化设计：引入 persistent/refillable input 分类，显式依赖声明 |
| v2.0 | 2026-01-19 | **简化设计**：移除 input 分类和 Wrapper，直接复用 CheckpointWithoutOutput |
| v2.1 | 2026-01-19 | **详细设计**：(1) 重命名为 MHCBlockRecomputeManager (2) Manager 传入 HyperConnection.forward() (3) 明确 recompute_pre_mlp_layernorm 和 recompute_mlp 互斥限制 (4) 详细描述 TransformerLayer 修改 |
