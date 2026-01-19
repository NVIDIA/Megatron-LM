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

#### 1.3.1 checkpoint_without_output 输入分类优化

本设计对 `checkpoint_without_output` 进行了关键优化，将输入参数显式分为两类：

```
┌────────────────────────────────────────────────────────────────────────────┐
│                  checkpoint_without_output 输入参数分类                     │
├────────────────────────────────────┬───────────────────────────────────────┤
│        Persistent Inputs           │         Refillable Inputs             │
│      (与前序 checkpoint 无关)       │      (由前序 checkpoint output 填充)  │
├────────────────────────────────────┼───────────────────────────────────────┤
│  • 在 ctx.saved_tensors 中保存     │  • 不保存在 ctx 中                    │
│  • Block 原始输入、常量参数等        │  • 来自上游 checkpoint 的输出         │
│  • 内存占用固定                     │  • Recompute 时通过 fill 注入         │
├────────────────────────────────────┼───────────────────────────────────────┤
│  Example: block_input, dropout_p   │  Example: h_pre, h_res, aggregated    │
└────────────────────────────────────┴───────────────────────────────────────┘
```

**优化优势**：
1. **内存效率**：refillable inputs 不占用存储空间
2. **语义清晰**：`fill_refillable_inputs` 只处理声明的 refillable 参数
3. **依赖显式化**：通过 `input_deps` 声明数据来源，便于调试和验证
4. **向后兼容**：不指定 indices 时默认保存所有 inputs

#### 1.3.2 Block 级别的数据流

```
Forward Pass:
┌─────────────────────────────────────────────────────────────────┐
│  TransformerBlock                                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Input (saved) ────────────────────────────────────────── │  │
│  │    │                                                      │  │
│  │    ▼                                                      │  │
│  │ [Layer 1: Attention Submodule]                           │  │
│  │    │ ┌─ HyperConnection (h_pre, h_post, h_res) ──────┐   │  │
│  │    │ │   • aggregate(hidden, h_pre)                   │   │  │
│  │    │ │   • apply_h_res(h_res, residual) ─ FREE ──────┼───│──│─► discarded
│  │    │ └────────────────────────────────────────────────┘   │  │
│  │    │                                                      │  │
│  │    ├─ LayerNorm (CheckpointWithoutOutput)                │  │
│  │    ├─ Attention                                          │  │
│  │    ├─ apply_h_post                                       │  │
│  │    └─ BDA ──────────────────────────────────────────────│──│─► output (kept temporarily)
│  │         │                                                 │  │
│  │    ▼                                                      │  │
│  │ [Layer 2: MLP Submodule]                                 │  │
│  │    │ ┌─ HyperConnection ─────────────────────────────┐   │  │
│  │    │ │   • aggregate, apply_h_res ─ FREE ────────────┼───│──│─► discarded
│  │    │ └───────────────────────────────────────────────┘   │  │
│  │    │                                                      │  │
│  │    ├─ LayerNorm (CheckpointWithoutOutput)                │  │
│  │    ├─ MLP                                                │  │
│  │    ├─ apply_h_post                                       │  │
│  │    └─ BDA ──────────────────────────────────────────────│──│─► output
│  │                                                           │  │
│  │ ... (repeat for all layers in block)                     │  │
│  │                                                           │  │
│  │ Block Output ◄────────────────────────────────────────── │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  After forward: discard all intermediate activations,          │
│                 register recompute hook on block output        │
└─────────────────────────────────────────────────────────────────┘

Backward Pass:
┌─────────────────────────────────────────────────────────────────┐
│  grad(Block Output) triggers recompute hook                     │
│      │                                                          │
│      ▼                                                          │
│  Recompute entire block from saved input                        │
│      │                                                          │
│      ▼                                                          │
│  Continue backward with recomputed activations                  │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 详细设计

### 2.1 核心组件

#### 2.1.1 `checkpoint_without_output` 优化设计

**核心思想**：将 checkpoint 的输入参数显式分为两类，使内存管理和 recompute 逻辑更加清晰。

##### 输入参数分类

```
┌────────────────────────────────────────────────────────────────────────┐
│                     checkpoint_without_output 输入参数                  │
├─────────────────────────────────┬──────────────────────────────────────┤
│      persistent_inputs          │         refillable_inputs            │
│   (与前序 checkpoint 无关)       │    (由前序 checkpoint output 提供)    │
├─────────────────────────────────┼──────────────────────────────────────┤
│ • Block 原始输入                 │ • 前序 submodule 的输出              │
│ • Config/超参数                  │ • h_pre, h_post, h_res              │
│ • 非 tensor 常量                 │ • aggregated 结果                   │
│ • 训练相关参数                   │ • apply_h_res 的输出                │
├─────────────────────────────────┼──────────────────────────────────────┤
│         ✓ 保存在 ctx            │          ✗ 不保存在 ctx             │
│     forward 时 clone 保存       │    recompute 时由 manager refill    │
└─────────────────────────────────┴──────────────────────────────────────┘
```

##### 优化后的 API 设计

```python
def checkpoint_without_output(
    func: Callable,
    *args,
    persistent_input_indices: Tuple[int, ...] = None,  # 需要保存的参数索引
    refillable_input_indices: Tuple[int, ...] = None,  # 可被 refill 的参数索引
    **kwargs
):
    """
    执行 func 并进行 selective checkpoint，但区分 persistent 和 refillable inputs。
    
    Args:
        func: 要执行的函数
        *args: 函数参数
        persistent_input_indices: 需要在 ctx 中保存的参数索引 (default: all indices)
        refillable_input_indices: 不保存、后续由 BlockLevelCheckpointManager refill 的参数索引
        **kwargs: 传递给 func 的关键字参数
    
    设计原则：
        - persistent_input_indices 和 refillable_input_indices 应互斥且覆盖所有 tensor args
        - 如果都不指定，默认所有 args 都是 persistent（兼容现有行为）
        - refillable inputs 在 ctx 中用 placeholder 占位，不占用实际内存
    """
    pass
```

##### `CheckpointWithoutOutputWrapper` 类

```python
class CheckpointWithoutOutputWrapper:
    """
    Wrapper that wraps a CheckpointWithoutOutput instance and provides
    the ability to inject refillable tensors from BlockLevelCheckpointManager.
    
    设计原理：
        在 forward pass 中，refillable_inputs 被标记但不保存实际数据。
        在 backward pass 触发 recompute 前，manager 调用 fill_refillable_inputs
        将前序 checkpoint 的输出注入到正确的位置。
    """
    
    def __init__(
        self,
        checkpoint_ctx: CheckpointContext,  # checkpoint_without_output 返回的 ctx
        refillable_indices: Tuple[int, ...],
        persistent_indices: Tuple[int, ...],
    ):
        self.checkpoint_ctx = checkpoint_ctx
        self.refillable_indices = refillable_indices
        self.persistent_indices = persistent_indices
        # Refillable inputs 的 placeholder，recompute 前会被填充
        self._refillable_placeholders: Dict[int, Tensor] = {}
    
    def fill_refillable_inputs(self, **inputs_by_index: Dict[int, Tensor]):
        """
        从外部（BlockLevelCheckpointManager）填充 refillable inputs。
        
        Args:
            inputs_by_index: {arg_index: tensor} 映射，只包含 refillable indices
        
        Example:
            # 假设 arg[0] 是 persistent，arg[1], arg[2] 是 refillable
            wrapper.fill_refillable_inputs({
                1: prev_checkpoint_output_1,  # 来自前序 checkpoint
                2: prev_checkpoint_output_2,
            })
        
        语义：
            这个方法只填充 refillable inputs，persistent inputs 已经在 ctx 中保存。
            调用后，checkpoint 拥有完整的输入信息用于 recompute。
        """
        for idx, tensor in inputs_by_index.items():
            if idx not in self.refillable_indices:
                raise ValueError(
                    f"Index {idx} is not a refillable input. "
                    f"Refillable indices: {self.refillable_indices}"
                )
            self._refillable_placeholders[idx] = tensor
    
    def get_all_inputs_for_recompute(self) -> Tuple[Tensor, ...]:
        """
        获取 recompute 所需的完整输入。
        
        Returns:
            按原始顺序排列的所有输入 tensors (persistent + refillable)
        
        内部逻辑：
            1. 从 ctx.saved_tensors 获取 persistent inputs
            2. 从 _refillable_placeholders 获取 refillable inputs
            3. 按原始 arg 顺序合并返回
        """
        all_inputs = {}
        
        # 从 ctx 获取 persistent inputs
        persistent_tensors = self.checkpoint_ctx.saved_tensors
        for i, idx in enumerate(self.persistent_indices):
            all_inputs[idx] = persistent_tensors[i]
        
        # 从 placeholders 获取 refillable inputs
        for idx, tensor in self._refillable_placeholders.items():
            all_inputs[idx] = tensor
        
        # 按顺序返回
        max_idx = max(all_inputs.keys())
        return tuple(all_inputs.get(i) for i in range(max_idx + 1))
    
    def get_recompute_hook(self):
        """Return the recompute hook function for backward."""
        return self.checkpoint_ctx._recompute
```

##### 具体示例：HyperConnection apply_h_res

```python
# 场景：apply_h_res(h_res, residual) 
# - h_res: 来自 compute_mappings 的输出（refillable）
# - residual: 可能来自 block input 或前一层的输出（refillable）

# Forward pass
wrapper = checkpoint_without_output(
    hyper_connection.apply_h_res,
    h_res,           # arg[0]: refillable - 来自 compute_mappings checkpoint
    residual,        # arg[1]: refillable - 来自前序 layer
    persistent_input_indices=(),     # 没有需要单独保存的
    refillable_input_indices=(0, 1), # 全部由前序 checkpoint 提供
)

# 注册到 manager
manager.register_submodule_checkpoint(
    name="layer_0_attn_hc_apply_h_res",
    wrapper=wrapper,
    output=residual_mixed,
    submodule_type=SubmoduleType.HYPER_CONNECTION_APPLY_H_RES,
    input_deps={
        0: "layer_0_attn_hc_compute.h_res",   # 指定依赖
        1: "layer_0_input_residual",
    }
)

# Backward pass (在 manager._unified_recompute_hook 中)
# Manager 根据 input_deps 自动 fill
wrapper.fill_refillable_inputs({
    0: recomputed_h_res,      # 从 compute_mappings 重算得到
    1: recomputed_residual,   # 从前序 layer 重算得到
})
```

**关键设计优势**：

1. **内存效率**：refillable inputs 不保存，只保留轻量级索引信息
2. **语义清晰**：`fill_refillable_inputs` 只处理需要填充的部分，职责单一
3. **依赖显式化**：通过 `input_deps` 明确每个 checkpoint 的数据来源
4. **向后兼容**：不指定 indices 时默认保存所有 inputs，兼容现有用法

#### 2.1.2 `BlockLevelCheckpointManager` 类

**目的**：管理一个TransformerBlock内所有submodule的checkpoint，协调依赖关系并执行 refill + recompute。

```python
class BlockLevelCheckpointManager:
    """
    Manages block-level checkpointing for TransformerBlock with HyperConnection.
    
    核心职责：
    1. 维护 submodule checkpoint 之间的依赖关系图
    2. 在 backward 时按拓扑顺序 recompute 并 refill 下游 checkpoint
    3. 协调内存释放和 recompute timing
    
    关键设计：
    - 每个 checkpoint 的 refillable_inputs 通过 input_deps 显式声明依赖
    - recompute 时按依赖顺序执行，确保 refill 数据可用
    """
    
    def __init__(self):
        self.submodule_checkpoints: Dict[str, SubmoduleCheckpoint] = {}  # name -> checkpoint
        self.checkpoint_order: List[str] = []  # 按 forward 顺序记录
        self.block_input: Optional[Tensor] = None
        self.rng_states: Optional[Tuple] = None
        # 依赖图：checkpoint_name -> {input_idx: source_checkpoint_name.output_key}
        self.dependency_graph: Dict[str, Dict[int, str]] = {}
        
    def save_block_input(self, hidden_states: Tensor):
        """
        Save the block input for later recomputation.
        Called at the beginning of TransformerBlock.forward().
        """
        self.block_input = hidden_states.detach().clone()
        self.rng_states = _get_all_rng_states()
        # 将 block_input 注册为特殊的 "源"
        self._register_source("block_input", hidden_states)
    
    def _register_source(self, name: str, tensor: Tensor):
        """注册一个数据源（用于依赖解析）"""
        pass
    
    def register_submodule_checkpoint(
        self,
        name: str,
        wrapper: CheckpointWithoutOutputWrapper,
        output: Union[Tensor, Tuple[Tensor, ...]],
        submodule_type: SubmoduleType,
        input_deps: Optional[Dict[int, str]] = None,
        output_keys: Optional[List[str]] = None,
    ):
        """
        注册一个 submodule 的 checkpoint（延迟释放 output）。
        
        Args:
            name: Checkpoint 唯一标识 (e.g., "layer_0_attn_hc_compute")
            wrapper: CheckpointWithoutOutputWrapper 实例
            output: Submodule 输出（单个 tensor 或 tuple）
            submodule_type: Submodule 类型
            input_deps: {arg_index: "source_name.output_key"} 映射
                        指定 refillable inputs 的数据来源
                        Example: {0: "layer_0_attn_hc_compute.h_res", 
                                  1: "block_input"}
            output_keys: 输出的命名（用于被下游引用）
                        Example: ["h_pre", "h_post", "h_res"]
        
        设计说明：
            - input_deps 中的 index 必须与 wrapper 的 refillable_indices 对应
            - output_keys 使得其他 checkpoint 可以通过 "name.output_key" 引用
        """
        checkpoint_info = SubmoduleCheckpoint(
            name=name,
            wrapper=wrapper,
            output=output,
            submodule_type=submodule_type,
            input_deps=input_deps or {},
            output_keys=output_keys or [],
        )
        self.submodule_checkpoints[name] = checkpoint_info
        self.checkpoint_order.append(name)
        
        # 更新依赖图
        if input_deps:
            self.dependency_graph[name] = input_deps
    
    def finalize_block(self, block_output: Tensor):
        """
        Called at the end of TransformerBlock.forward().
        
        执行步骤：
        1. 验证依赖图完整性
        2. 释放所有 intermediate outputs 的内存
        3. 在 block_output 上注册 unified recompute hook
        """
        # Step 1: 验证依赖图
        self._validate_dependency_graph()
        
        # Step 2: 释放所有 intermediate outputs (free memory)
        for name in self.checkpoint_order:
            ckpt = self.submodule_checkpoints[name]
            self._free_tensor_storage(ckpt.output)
        
        # Step 3: Register unified recompute hook
        if block_output.requires_grad:
            block_output.register_hook(self._unified_recompute_hook)
    
    def _validate_dependency_graph(self):
        """验证所有 input_deps 引用的源都存在"""
        all_sources = set(self.submodule_checkpoints.keys())
        all_sources.add("block_input")
        
        for name, deps in self.dependency_graph.items():
            for idx, source_ref in deps.items():
                source_name = source_ref.split(".")[0]
                if source_name not in all_sources:
                    raise ValueError(
                        f"Checkpoint '{name}' references unknown source '{source_name}'"
                    )
    
    def _free_tensor_storage(self, tensor: Union[Tensor, Tuple]):
        """释放 tensor 的底层存储"""
        if isinstance(tensor, tuple):
            for t in tensor:
                if t is not None:
                    t.untyped_storage().resize_(0)
        elif tensor is not None:
            tensor.untyped_storage().resize_(0)
    
    def _unified_recompute_hook(self, grad_output: Tensor):
        """
        Unified recompute hook triggered during backward.
        
        按拓扑顺序执行 recompute，并在每步之后 refill 下游 checkpoint。
        """
        with _fork_rng():
            _set_all_rng_states(*self.rng_states)
            
            # 存储已 recompute 的输出（用于 refill 下游）
            recomputed_outputs: Dict[str, Union[Tensor, Tuple]] = {
                "block_input": self.block_input.detach().requires_grad_(True)
            }
            
            # 按 forward 顺序 recompute
            with torch.enable_grad():
                for name in self.checkpoint_order:
                    ckpt = self.submodule_checkpoints[name]
                    
                    # Step 1: 为当前 checkpoint 填充 refillable inputs
                    self._fill_refillable_inputs(ckpt, recomputed_outputs)
                    
                    # Step 2: 执行 recompute
                    inputs = ckpt.wrapper.get_all_inputs_for_recompute()
                    output = ckpt.wrapper.checkpoint_ctx.func(*inputs)
                    
                    # Step 3: 存储 recomputed output（供下游使用）
                    recomputed_outputs[name] = output
                    
                    # Step 4: 填充 checkpoint 的 output storage
                    self._refill_output_storage(ckpt.output, output)
    
    def _fill_refillable_inputs(
        self, 
        ckpt: SubmoduleCheckpoint, 
        recomputed_outputs: Dict[str, Union[Tensor, Tuple]]
    ):
        """
        为 checkpoint 的 refillable inputs 填充数据。
        
        从 recomputed_outputs 中根据 input_deps 查找对应数据。
        """
        if not ckpt.input_deps:
            return
        
        refill_data = {}
        for arg_idx, source_ref in ckpt.input_deps.items():
            # 解析 source_ref: "source_name" 或 "source_name.output_key"
            parts = source_ref.split(".", 1)
            source_name = parts[0]
            output_key = parts[1] if len(parts) > 1 else None
            
            source_output = recomputed_outputs[source_name]
            
            if output_key is not None:
                # 从 tuple output 中按 key 获取
                source_ckpt = self.submodule_checkpoints.get(source_name)
                if source_ckpt and source_ckpt.output_keys:
                    key_idx = source_ckpt.output_keys.index(output_key)
                    refill_data[arg_idx] = source_output[key_idx]
                else:
                    raise ValueError(f"Cannot resolve output_key '{output_key}' for '{source_name}'")
            else:
                refill_data[arg_idx] = source_output
        
        ckpt.wrapper.fill_refillable_inputs(refill_data)
    
    def _refill_output_storage(self, original: Tensor, recomputed: Tensor):
        """将 recomputed tensor 的数据写入 original 的 storage"""
        original.untyped_storage().resize_(recomputed.untyped_storage().size())
        original.copy_(recomputed)
```

#### 2.1.3 `SubmoduleCheckpoint` 数据类

```python
@dataclass
class SubmoduleCheckpoint:
    """
    Stores checkpoint information for a single submodule.
    
    设计说明：
        - wrapper: 包含 checkpoint context 和 input 分类信息
        - input_deps: 声明 refillable inputs 的数据来源
        - output_keys: 为 output 的各部分命名（支持被下游引用）
    """
    name: str
    wrapper: CheckpointWithoutOutputWrapper
    output: Union[Tensor, Tuple[Tensor, ...]]
    submodule_type: SubmoduleType
    
    # 输入依赖映射：{arg_index: "source_checkpoint.output_key"}
    # Example: {0: "layer_0_hc_compute.h_res", 1: "block_input"}
    input_deps: Dict[int, str] = field(default_factory=dict)
    
    # 输出命名（用于被下游依赖引用）
    # Example: ["h_pre", "h_post", "h_res"] for compute_mappings
    output_keys: List[str] = field(default_factory=list)
    
    def get_output_by_key(self, key: str) -> Tensor:
        """根据 key 获取 output 中的特定 tensor"""
        if not self.output_keys:
            raise ValueError(f"Checkpoint '{self.name}' has no output_keys defined")
        idx = self.output_keys.index(key)
        if isinstance(self.output, tuple):
            return self.output[idx]
        elif idx == 0:
            return self.output
        else:
            raise IndexError(f"Output key index {idx} out of range for non-tuple output")


class SubmoduleType(Enum):
    """Submodule 类型枚举，用于分类和调试"""
    LAYERNORM = "layernorm"
    HYPER_CONNECTION_COMPUTE = "hc_compute"       # compute_mappings -> (h_pre, h_post, h_res)
    HYPER_CONNECTION_AGGREGATE = "hc_aggregate"   # aggregate(hidden, h_pre)
    HYPER_CONNECTION_APPLY_H_RES = "hc_apply_h_res"   # apply_h_res(h_res, residual)
    HYPER_CONNECTION_APPLY_H_POST = "hc_apply_h_post" # apply_h_post(output, h_post)
    ATTENTION = "attention"
    MLP = "mlp"
    BDA = "bda"
```

##### 依赖关系示意图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Block-Level Dependency Graph                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  block_input ─────────────────────────────────────────────────┐             │
│       │                                                        │             │
│       ▼                                                        │             │
│  ┌─────────────────────┐                                       │             │
│  │ hc_compute          │ persistent: [hidden_states]           │             │
│  │ output_keys:        │ refillable: []                        │             │
│  │   h_pre, h_post,    │                                       │             │
│  │   h_res             │                                       │             │
│  └─────────┬───────────┘                                       │             │
│            │                                                    │             │
│            ├──────────────────────┐                             │             │
│            │                      │                             │             │
│            ▼                      ▼                             ▼             │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │ hc_aggregate        │  │ hc_apply_h_res      │  │                     │  │
│  │ persistent: []      │  │ persistent: []      │  │                     │  │
│  │ refillable:         │  │ refillable:         │  │                     │  │
│  │   {0: block_input,  │  │   {0: hc_compute.   │  │                     │  │
│  │    1: hc_compute.   │  │       h_res,        │  │                     │  │
│  │       h_pre}        │  │    1: block_input}  │  │                     │  │
│  └─────────┬───────────┘  └─────────┬───────────┘  │                     │  │
│            │                        │               │                     │  │
│            ▼                        │               │                     │  │
│  ┌─────────────────────┐            │               │                     │  │
│  │ layernorm           │            │               │                     │  │
│  │ persistent: []      │            │               │                     │  │
│  │ refillable:         │            │               │                     │  │
│  │   {0: hc_aggregate} │            │               │                     │  │
│  └─────────┬───────────┘            │               │                     │  │
│            │                        │               │                     │  │
│            ▼                        │               │                     │  │
│  ┌─────────────────────┐            │               │                     │  │
│  │ attention           │            │               │                     │  │
│  │ (not checkpointed)  │            │               │                     │  │
│  └─────────┬───────────┘            │               │                     │  │
│            │                        │               │                     │  │
│            ▼                        │               │                     │  │
│  ┌─────────────────────┐            │               │                     │  │
│  │ hc_apply_h_post     │            │               │                     │  │
│  │ persistent: []      │            │               │                     │  │
│  │ refillable:         │            │               │                     │  │
│  │   {0: attention,    │◄───────────┼───────────────┘                     │  │
│  │    1: hc_compute.   │            │                                     │  │
│  │       h_post}       │            │                                     │  │
│  └─────────┬───────────┘            │                                     │  │
│            │                        │                                     │  │
│            ▼                        ▼                                     │  │
│  ┌─────────────────────────────────────────────────┐                      │  │
│  │ bda                                              │                      │  │
│  │ persistent: [dropout_prob, training_flag, ...]  │                      │  │
│  │ refillable:                                      │                      │  │
│  │   {0: hc_apply_h_post,                          │                      │  │
│  │    1: hc_apply_h_res}                           │                      │  │
│  └─────────────────────────────────────────────────┘                      │  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 TransformerLayer 修改

在`TransformerLayer`中，需要增加对block-level checkpoint的支持，使用优化后的 API 显式声明 persistent/refillable inputs。

```python
class TransformerLayer(GraphableMegatronModule, BaseTransformerLayer):
    
    def __init__(self, ...):
        # ... existing init ...
        
        # Block-level checkpoint support
        self.block_checkpoint_manager: Optional[BlockLevelCheckpointManager] = None
        self.enable_block_level_recompute = (
            config.enable_hyper_connections and 
            config.hyper_connection_recompute_granularity == "block"
        )
    
    def set_block_checkpoint_manager(self, manager: BlockLevelCheckpointManager):
        """Called by TransformerBlock to set the shared checkpoint manager."""
        self.block_checkpoint_manager = manager
    
    def _forward_attention_with_block_recompute(self, hidden_states, residual, ...):
        """
        Forward pass for attention submodule with block-level recomputation.
        
        使用优化后的 checkpoint API：
        - persistent_input_indices: 需要保存在 ctx 的参数
        - refillable_input_indices: 由前序 checkpoint 填充的参数
        - input_deps: 声明 refillable 参数的数据来源
        """
        manager = self.block_checkpoint_manager
        layer_prefix = f"layer_{self.layer_number}_attn"
        
        # === 1. HyperConnection: compute mappings ===
        # hidden_states 来自 block_input（第一层）或前一层输出
        # 这里作为 persistent input 保存，因为它是 layer 的入口
        if self.do_self_attention_hyper_connection:
            hc_compute_wrapper = checkpoint_without_output(
                self.self_attention_hyper_connection.compute_mappings,
                hidden_states,  # arg[0]: 需要保存
                persistent_input_indices=(0,),
                refillable_input_indices=(),  # 无 refillable
            )
            h_pre, h_post, h_res = hc_compute_wrapper.checkpoint_ctx.output
            
            manager.register_submodule_checkpoint(
                name=f"{layer_prefix}_hc_compute",
                wrapper=hc_compute_wrapper,
                output=(h_pre, h_post, h_res),
                submodule_type=SubmoduleType.HYPER_CONNECTION_COMPUTE,
                input_deps={},  # 无外部依赖，persistent 已保存
                output_keys=["h_pre", "h_post", "h_res"],  # 定义输出命名
            )
            
            # === 2. Aggregate: 不单独 checkpoint，inline 计算 ===
            # aggregate 是简单计算，不值得单独 checkpoint
            aggregated = self.self_attention_hyper_connection.aggregate(
                hidden_states, h_pre
            )
            # 将 aggregated 注册为中间结果供下游引用
            manager._register_source(f"{layer_prefix}_aggregated", aggregated)
            
            # === 3. Apply h_res to residual ===
            # h_res 来自 hc_compute，residual 来自 block_input 或前层
            hc_res_wrapper = checkpoint_without_output(
                self.self_attention_hyper_connection.apply_h_res,
                h_res,      # arg[0]: refillable from hc_compute.h_res
                residual,   # arg[1]: refillable from previous layer
                persistent_input_indices=(),  # 全部 refillable，不保存
                refillable_input_indices=(0, 1),
            )
            residual_mixed = hc_res_wrapper.checkpoint_ctx.output
            
            manager.register_submodule_checkpoint(
                name=f"{layer_prefix}_hc_apply_h_res",
                wrapper=hc_res_wrapper,
                output=residual_mixed,
                submodule_type=SubmoduleType.HYPER_CONNECTION_APPLY_H_RES,
                input_deps={
                    0: f"{layer_prefix}_hc_compute.h_res",  # 来自 hc_compute
                    1: self._get_residual_source(),        # block_input 或 prev_layer
                },
            )
        else:
            aggregated = hidden_states
            residual_mixed = residual
        
        # === 4. LayerNorm ===
        if self.recompute_input_layernorm:
            ln_wrapper = checkpoint_without_output(
                self.input_layernorm,
                aggregated,  # arg[0]: refillable from aggregate
                persistent_input_indices=(),
                refillable_input_indices=(0,),
            )
            ln_output = ln_wrapper.checkpoint_ctx.output
            
            manager.register_submodule_checkpoint(
                name=f"{layer_prefix}_ln",
                wrapper=ln_wrapper,
                output=ln_output,
                submodule_type=SubmoduleType.LAYERNORM,
                input_deps={0: f"{layer_prefix}_aggregated"},
            )
        else:
            ln_output = self.input_layernorm(aggregated)
            manager._register_source(f"{layer_prefix}_ln_output", ln_output)
        
        # === 5. Attention (通常不 checkpoint，计算密集但内存友好) ===
        attention_output_with_bias = self.self_attention(ln_output, ...)
        manager._register_source(f"{layer_prefix}_attn_output", attention_output_with_bias)
        
        # === 6. Apply h_post ===
        if self.do_self_attention_hyper_connection:
            hc_post_wrapper = checkpoint_without_output(
                self.self_attention_hyper_connection.apply_h_post,
                attention_output_with_bias,  # arg[0]: refillable from attention
                h_post,                       # arg[1]: refillable from hc_compute
                persistent_input_indices=(),
                refillable_input_indices=(0, 1),
            )
            post_output = hc_post_wrapper.checkpoint_ctx.output
            
            manager.register_submodule_checkpoint(
                name=f"{layer_prefix}_hc_apply_h_post",
                wrapper=hc_post_wrapper,
                output=post_output,
                submodule_type=SubmoduleType.HYPER_CONNECTION_APPLY_H_POST,
                input_deps={
                    0: f"{layer_prefix}_attn_output",
                    1: f"{layer_prefix}_hc_compute.h_post",
                },
            )
            attention_output_with_bias = post_output
        
        # === 7. BDA (Bias-Dropout-Add) ===
        # dropout_prob 等标量参数作为 persistent
        bda_wrapper = checkpoint_without_output(
            self.self_attn_bda(self.training, self.config.bias_dropout_fusion),
            attention_output_with_bias,  # arg[0]: refillable
            residual_mixed,              # arg[1]: refillable
            self.hidden_dropout,         # arg[2]: persistent (scalar)
            persistent_input_indices=(2,),  # 只保存 dropout prob
            refillable_input_indices=(0, 1),
        )
        hidden_states = bda_wrapper.checkpoint_ctx.output
        
        manager.register_submodule_checkpoint(
            name=f"{layer_prefix}_bda",
            wrapper=bda_wrapper,
            output=hidden_states,
            submodule_type=SubmoduleType.BDA,
            input_deps={
                0: f"{layer_prefix}_hc_apply_h_post" if self.do_self_attention_hyper_connection 
                   else f"{layer_prefix}_attn_output",
                1: f"{layer_prefix}_hc_apply_h_res" if self.do_self_attention_hyper_connection
                   else self._get_residual_source(),
            },
        )
        
        return hidden_states
    
    def _get_residual_source(self) -> str:
        """获取 residual 的来源名称"""
        if self.layer_number == 1:
            return "block_input"
        else:
            return f"layer_{self.layer_number - 1}_mlp_bda"
```

**优化后的关键变化**：

1. **显式 input 分类**：每个 checkpoint 明确声明哪些 inputs 是 persistent（需保存），哪些是 refillable（不保存）
2. **依赖链显式化**：通过 `input_deps` 清晰表达数据流向
3. **内存节省**：refillable inputs 不占用 ctx 存储空间
4. **fill 语义简化**：`fill_refillable_inputs` 只需要填充声明的 refillable 参数

### 2.3 TransformerBlock 修改

```python
class TransformerBlock(GraphableMegatronModule, MegatronModule):
    
    def forward(self, hidden_states, ...):
        # ... existing code ...
        
        if self.config.enable_hyper_connections and self._should_use_block_level_recompute():
            return self._forward_with_block_recompute(hidden_states, ...)
        else:
            return self._forward_normal(hidden_states, ...)
    
    def _should_use_block_level_recompute(self) -> bool:
        """Determine if block-level recomputation should be used."""
        return (
            self.training and
            self.config.hyper_connection_recompute_granularity == "block"
        )
    
    def _forward_with_block_recompute(self, hidden_states, ...):
        """
        Forward pass with block-level recomputation for HyperConnection.
        """
        # Create checkpoint manager for this block
        manager = BlockLevelCheckpointManager()
        
        # Save block input
        manager.save_block_input(hidden_states)
        
        # Expand for hyper connections if needed
        if self.config.enable_hyper_connections and self.pre_process:
            hidden_states = HyperConnectionModule.input_expand(
                hidden_states, self.num_residual_streams
            )
        
        # Forward through layers with checkpoint manager
        for layer in self.layers:
            layer.set_block_checkpoint_manager(manager)
            hidden_states, context = layer(
                hidden_states=hidden_states,
                ...
            )
            layer.set_block_checkpoint_manager(None)  # Clear reference
        
        # Contract if needed
        if self.config.enable_hyper_connections and self.has_final_layernorm_in_this_stage():
            hidden_states = HyperConnectionModule.output_contract(
                hidden_states, self.num_residual_streams
            )
        
        # Final layernorm
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
        
        # Finalize block - discard intermediates and register recompute hook
        manager.finalize_block(hidden_states)
        
        return hidden_states
```

### 2.4 重算逻辑详解

#### 2.4.1 核心流程：Refill + Recompute Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Backward 时的 Refill + Recompute 流程                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  grad_output arrives at block_output                                         │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ _unified_recompute_hook 触发                                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 恢复 RNG 状态                                                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ for each checkpoint in forward_order:                               │    │
│  │                                                                      │    │
│  │   ┌──────────────────────────────────────────────────────────────┐  │    │
│  │   │ Step 1: Fill refillable inputs                               │  │    │
│  │   │                                                               │  │    │
│  │   │ • 查询 input_deps 中声明的依赖                                │  │    │
│  │   │ • 从 recomputed_outputs 字典获取对应 tensor                  │  │    │
│  │   │ • 调用 wrapper.fill_refillable_inputs({idx: tensor, ...})    │  │    │
│  │   │                                                               │  │    │
│  │   │ Note: persistent inputs 已在 ctx.saved_tensors 中，无需 fill  │  │    │
│  │   └──────────────────────────────────────────────────────────────┘  │    │
│  │                          │                                          │    │
│  │                          ▼                                          │    │
│  │   ┌──────────────────────────────────────────────────────────────┐  │    │
│  │   │ Step 2: Get all inputs and recompute                         │  │    │
│  │   │                                                               │  │    │
│  │   │ inputs = wrapper.get_all_inputs_for_recompute()              │  │    │
│  │   │   • persistent inputs ← ctx.saved_tensors                    │  │    │
│  │   │   • refillable inputs ← _refillable_placeholders             │  │    │
│  │   │   • merge by original arg index                              │  │    │
│  │   │                                                               │  │    │
│  │   │ output = checkpoint_ctx.func(*inputs)                        │  │    │
│  │   └──────────────────────────────────────────────────────────────┘  │    │
│  │                          │                                          │    │
│  │                          ▼                                          │    │
│  │   ┌──────────────────────────────────────────────────────────────┐  │    │
│  │   │ Step 3: Store output for downstream                          │  │    │
│  │   │                                                               │  │    │
│  │   │ recomputed_outputs[checkpoint_name] = output                 │  │    │
│  │   │ (供后续 checkpoint 的 refillable inputs 使用)                 │  │    │
│  │   └──────────────────────────────────────────────────────────────┘  │    │
│  │                          │                                          │    │
│  │                          ▼                                          │    │
│  │   ┌──────────────────────────────────────────────────────────────┐  │    │
│  │   │ Step 4: Refill original output storage                       │  │    │
│  │   │                                                               │  │    │
│  │   │ • 将 recomputed output 复制到原 output tensor 的 storage     │  │    │
│  │   │ • 使 autograd 能正确回溯梯度                                 │  │    │
│  │   └──────────────────────────────────────────────────────────────┘  │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       ▼                                                                      │
│  Autograd 继续 backward，每个 checkpoint 的 output 已被正确填充              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 2.4.2 `_unified_recompute_hook` 实现

```python
def _unified_recompute_hook(self, grad_output: Tensor):
    """
    按 forward 顺序重算整个 block，利用依赖图 refill inputs。
    
    关键优化点：
    1. 只恢复 persistent inputs（从 ctx.saved_tensors）
    2. refillable inputs 通过 fill_refillable_inputs 从上游 recomputed outputs 获取
    3. 避免了重复存储前序 checkpoint 的输出
    """
    with _fork_rng():
        _set_all_rng_states(*self.rng_states)
        
        # 初始化 recomputed outputs 字典
        # block_input 是唯一的 "根" 数据源
        recomputed_outputs: Dict[str, Union[Tensor, Tuple[Tensor, ...]]] = {
            "block_input": self.block_input.detach().requires_grad_(True)
        }
        
        with torch.enable_grad():
            # 按 forward 顺序遍历所有 checkpoint
            for name in self.checkpoint_order:
                ckpt = self.submodule_checkpoints[name]
                
                # ========== Step 1: Fill refillable inputs ==========
                # 只填充 refillable 部分，persistent 已在 ctx 中
                if ckpt.input_deps:
                    refill_data = {}
                    for arg_idx, source_ref in ckpt.input_deps.items():
                        # 解析引用: "source_name" 或 "source_name.output_key"
                        tensor = self._resolve_source_reference(
                            source_ref, recomputed_outputs
                        )
                        refill_data[arg_idx] = tensor
                    
                    # 填充 refillable inputs
                    ckpt.wrapper.fill_refillable_inputs(refill_data)
                
                # ========== Step 2: Get complete inputs and recompute ==========
                # 合并 persistent (from ctx) + refillable (just filled)
                all_inputs = ckpt.wrapper.get_all_inputs_for_recompute()
                
                # 执行 recompute
                output = ckpt.wrapper.checkpoint_ctx.func(*all_inputs)
                
                # ========== Step 3: Store for downstream ==========
                recomputed_outputs[name] = output
                
                # ========== Step 4: Refill original output storage ==========
                # 将 recomputed 数据复制回原 output tensor，使 autograd 正确工作
                self._refill_output_storage(ckpt.output, output)
    
    def _resolve_source_reference(
        self, 
        source_ref: str, 
        recomputed_outputs: Dict[str, Union[Tensor, Tuple]]
    ) -> Tensor:
        """
        解析依赖引用，从 recomputed_outputs 获取对应 tensor。
        
        支持的格式:
        - "checkpoint_name": 获取整个 output（单个 tensor）
        - "checkpoint_name.output_key": 获取 output tuple 中的特定元素
        """
        parts = source_ref.split(".", 1)
        source_name = parts[0]
        
        if source_name not in recomputed_outputs:
            raise RuntimeError(
                f"Source '{source_name}' not yet recomputed. "
                f"Check checkpoint order and dependencies."
            )
        
        source_output = recomputed_outputs[source_name]
        
        if len(parts) == 1:
            # 直接返回整个 output
            return source_output
        else:
            # 按 output_key 索引
            output_key = parts[1]
            source_ckpt = self.submodule_checkpoints.get(source_name)
            
            if source_ckpt is None or not source_ckpt.output_keys:
                raise ValueError(
                    f"Cannot resolve '{output_key}' for '{source_name}': "
                    f"no output_keys defined"
                )
            
            key_idx = source_ckpt.output_keys.index(output_key)
            return source_output[key_idx]
```

#### 2.4.3 设计优势总结

| 方面 | 优化前 | 优化后 |
|------|--------|--------|
| **ctx 存储** | 保存所有 inputs | 只保存 persistent inputs |
| **refill 语义** | 不清晰，需猜测哪些需要 fill | 显式声明 refillable indices |
| **依赖关系** | 隐式，难以追踪 | 通过 input_deps 显式声明 |
| **内存占用** | 大量重复存储 | 最小化：只存必要的根数据 |
| **调试友好度** | 低 | 高：依赖图可视化/验证 |

### 2.5 配置参数

在`TransformerConfig`中添加新的配置项：

```python
@dataclass
class TransformerConfig:
    # ... existing fields ...
    
    hyper_connection_recompute_granularity: Optional[str] = None
    """
    Recomputation granularity for HyperConnection.
    Options:
    - None: No block-level recomputation (use existing selective recompute)
    - "block": Block-level recomputation for all HyperConnection intermediates
    - "layer": Layer-level recomputation (finer granularity than block)
    """
    
    hyper_connection_recompute_include_layernorm: bool = True
    """
    Whether to include LayerNorm in block-level recomputation.
    If False, LayerNorm outputs will be saved separately.
    """
    
    hyper_connection_recompute_include_bda: bool = True
    """
    Whether to include BDA operations in block-level recomputation.
    """
```

## 3. 内存分析

### 3.1 现有方案内存占用

对于一个启用HyperConnection的TransformerLayer（n个residual streams）：

| 组件 | Tensor数量 | 形状 | 内存估算 |
|------|-----------|------|---------|
| h_pre | 1 | [s, b, n] | s×b×n×dtype |
| h_post | 1 | [s, b, n] | s×b×n×dtype |
| h_res | 1 | [s, b, n, n] | s×b×n²×dtype |
| residual_mixed | 1 | [s, b, n×C] | s×b×n×C×dtype |
| ln_output | 2 | [s, b, C] | 2×s×b×C×dtype |
| attention_output | 1 | [s, b, n×C] | s×b×n×C×dtype |
| mlp_output | 1 | [s, b, n×C] | s×b×n×C×dtype |

总计每层约：`s×b×(3n + n² + 4nC + 2C)×dtype`

### 3.2 新方案内存占用

| 组件 | Tensor数量 | 形状 | 内存估算 |
|------|-----------|------|---------|
| block_input | 1 | [s, b, C] | s×b×C×dtype |
| block_output | 1 | [s, b, C] | s×b×C×dtype |

总计整个block：`2×s×b×C×dtype`

### 3.3 内存节省

假设：
- n = 4 (residual streams)
- C = 4096 (hidden size)
- L = 32 (layers per block)

内存节省比例约为：
```
节省 = L × (3n + n² + 4nC + 2C) / 2C
     ≈ L × (12 + 16 + 65536 + 8192) / 8192
     ≈ L × 9 (approximately 9x per layer)
```

对于32层的block，总体节省约为 **~280x**。

## 4. 实现计划

### Phase 1: 核心基础设施
1. 实现 `CheckpointWithOutputWrapper`
2. 实现 `SubmoduleCheckpoint` 数据类
3. 实现 `BlockLevelCheckpointManager` 基本框架

### Phase 2: TransformerLayer 集成
1. 修改 `TransformerLayer` 支持 block checkpoint manager
2. 实现 attention submodule 的 block-level checkpoint
3. 实现 MLP submodule 的 block-level checkpoint

### Phase 3: TransformerBlock 集成
1. 修改 `TransformerBlock.forward` 添加 block-level recompute 路径
2. 实现 `finalize_block` 和 `_unified_recompute_hook`
3. 添加配置参数验证

### Phase 4: 测试与优化
1. 单元测试：验证数值正确性
2. 集成测试：与现有 checkpoint 机制兼容性
3. 性能测试：内存和速度 benchmark
4. 边界条件处理：PP/TP/CP 并行场景

## 5. 风险与缓解

### 5.1 数值精度
**风险**：重算可能因RNG状态不一致导致数值差异

**缓解**：
- 严格保存和恢复所有RNG状态（CPU、CUDA、tracker）
- 添加数值精度验证测试

### 5.2 性能开销
**风险**：重算引入额外计算开销

**缓解**：
- 仅在训练时启用
- 提供细粒度配置，允许用户权衡内存和计算

### 5.3 兼容性
**风险**：与现有 selective recompute、CPU offloading 等特性冲突

**缓解**：
- 明确互斥配置
- 在 config 验证中添加检查
- 编写兼容性文档

## 6. API 示例

```python
# 配置示例
config = TransformerConfig(
    hidden_size=4096,
    num_layers=32,
    enable_hyper_connections=True,
    num_residual_streams=4,
    # 启用 block-level recomputation
    hyper_connection_recompute_granularity="block",
    hyper_connection_recompute_include_layernorm=True,
    hyper_connection_recompute_include_bda=True,
)

# 模型使用
model = GPTModel(config=config, ...)

# 训练时自动启用 block-level recomputation
for batch in dataloader:
    loss = model(batch)
    loss.backward()  # 触发 recompute hook
```

## 7. 待讨论问题

1. **Recompute 粒度选择**：是否需要支持比 block 更细的粒度（如 layer-level）？
2. **与 TE checkpoint 的交互**：FP8 场景下如何与 TransformerEngine 的 checkpoint 机制协作？
3. **Pipeline Parallel 边界**：跨 PP stage 的 block 如何处理 input tensor 的保存？
4. **CUDA Graph 兼容性**：是否需要支持 CUDA Graph 场景？

## 8. 参考

- 现有 `CheckpointWithoutOutput` 实现：`megatron/core/tensor_parallel/random.py`
- HyperConnection 实现：`megatron/core/transformer/hyper_connection.py`
- TransformerLayer 实现：`megatron/core/transformer/transformer_layer.py`
- mHC 论文：Manifold-Constrained Hyper-Connections

---

**文档版本**: v1.1  
**作者**: [待填写]  
**日期**: 2026-01-19  
**状态**: Draft - 待 Review

### 变更历史

| 版本 | 日期 | 变更内容 |
|------|------|----------|
| v1.0 | 2026-01-19 | 初始版本 |
| v1.1 | 2026-01-19 | 优化 checkpoint_without_output 设计：引入 persistent/refillable input 分类，显式依赖声明 |
