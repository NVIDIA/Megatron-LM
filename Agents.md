# Agents.md — Megatron-LM

> AI agent working guide. Encodes architectural knowledge, coding conventions, and key implementation details for the Megatron-LM project.

---

## 1. Project Overview

Megatron-LM contains two components:
- **Megatron Core** (`megatron/core/`) — GPU-optimized composable training library (TP/PP/DP/EP/CP parallelism)
- **Megatron-LM** (root scripts + `megatron/training/`) — Reference training scripts and training infrastructure

**Key Distinction**: `megatron/core/` uses `TransformerConfig` dataclass configuration; `megatron/training/` uses `args = get_args()` global namespace. Never mix these.

---

## 2. Code Organization

```
Megatron-LM/
├── pretrain_gpt.py               # GPT entry point (calls pretrain())
├── pretrain_t5.py                # T5 entry point
├── pretrain_mamba.py             # Mamba/Hybrid entry point
├── megatron/
│   ├── core/                     # ★ Megatron Core library ★
│   │   ├── models/               #   Model implementations (GPT, BERT, T5, VLM)
│   │   │   └── gpt/
│   │   │       ├── gpt_model.py
│   │   │       └── gpt_layer_specs.py  # ModuleSpec definitions
│   │   ├── transformer/          #   Transformer building blocks
│   │   │   ├── transformer_config.py   # TransformerConfig dataclass (150+ params)
│   │   │   ├── transformer_block.py    # TransformerBlock (layer container)
│   │   │   ├── transformer_layer.py    # TransformerLayer (single layer)
│   │   │   ├── attention.py            # Attention base class
│   │   │   ├── dot_product_attention.py  # Standard DotProduct Attention
│   │   │   ├── multi_latent_attention.py # DeepSeek-style MLA
│   │   │   ├── mlp.py                  # Standard MLP / SwiGLU
│   │   │   ├── moe/                    # Mixture of Experts
│   │   │   │   ├── moe_layer.py
│   │   │   │   ├── router.py           # TopK / Expert Choice routing
│   │   │   │   └── moe_utils.py
│   │   │   ├── experimental_attention_variant/  # DSA, Gated Delta Net, etc.
│   │   │   └── spec_utils.py           # ModuleSpec mechanism
│   │   ├── tensor_parallel/      #   Tensor parallelism
│   │   ├── pipeline_parallel/    #   Pipeline parallelism
│   │   ├── distributed/          #   DDP, FSDP
│   │   ├── optimizer/            #   Distributed optimizer
│   │   ├── datasets/             #   Dataset loading
│   │   └── inference/            #   Inference engines
│   ├── training/                 #   Training infrastructure
│   │   ├── training.py           #   ★ Main training loop + FLOPs calculation ★
│   │   ├── arguments.py          #   CLI argument definitions (165KB, 5000+ lines)
│   │   ├── checkpointing.py      #   Checkpoint save/load
│   │   └── initialize.py         #   Initialization (parallel groups, random seeds, etc.)
│   ├── legacy/                   #   ⚠️ Deprecated code, don't modify
│   └── post_training/            #   Quantization, distillation, pruning
├── tests/
│   ├── unit_tests/               #   Unit tests (mirrors source structure)
│   └── functional_tests/         #   End-to-end integration tests
└── examples/                     #   Training example scripts
```

---

## 3. Coding Conventions

### Pre-commit hooks (only for `megatron/core/`)

```yaml
# .pre-commit-config.yaml
- Black:    --skip-magic-trailing-comma --skip-string-normalization
- isort:    standard configuration
- pylint:   megatron/core/ only
```

> **Important**: Code in `megatron/training/` is not constrained by Black/isort, but should maintain consistent style.

### Conventions
- Use `print_rank_0()` for logging (only outputs on rank 0)
- Distributed process groups accessed via `mpu` module or `ProcessGroupCollection`
- In `megatron/core/` use config objects for params; in `megatron/training/` use `args = get_args()`
- Modules inherit from `MegatronModule` (not `torch.nn.Module`)
- New core/ features must include tests in `tests/unit_tests/`

---

## 4. Key Architectural Concepts

### 4.1 Parallelism Strategies

| Abbr | Full Name | What it Parallelizes | Parameter |
|------|-----------|----------------------|-----------|
| TP | Tensor Parallel | Splits tensors within layers | `--tensor-model-parallel-size` |
| PP | Pipeline Parallel | Splits layers across GPUs | `--pipeline-model-parallel-size` |
| DP | Data Parallel | Data sharding | Automatic (remaining GPUs) |
| CP | Context Parallel | Sequence length splitting | `--context-parallel-size` |
| EP | Expert Parallel | MoE expert splitting | `--expert-model-parallel-size` |

### 4.2 ModuleSpec Pattern

Model architectures are defined via `ModuleSpec` (not hard-coded):

```python
# megatron/core/models/gpt/gpt_layer_specs.py
layer_spec = ModuleSpec(
    module=TransformerLayer,
    submodules=TransformerLayerSubmodules(
        self_attention=ModuleSpec(module=SelfAttention, ...),
        mlp=ModuleSpec(module=MLP, ...),
    )
)
```

### 4.3 TransformerConfig

`megatron/core/transformer/transformer_config.py` is the core configuration dataclass. All model architecture parameters are defined here. When adding features involving attention variants, you typically need to add configuration fields here.

### 4.4 Attention Type Hierarchy

```
Standard Attention
├── MHA (Multi-Head Attention) — num_query_groups == num_attention_heads
├── GQA (Grouped Query Attention) — group_query_attention=True
├── MLA (Multi-Latent Attention) — multi_latent_attention=True
│   └── Uses q_lora_rank, kv_lora_rank, qk_head_dim, v_head_dim, qk_pos_emb_head_dim
└── Experimental variants (experimental_attention_variant)
    ├── gated_delta_net — linear attention
    └── DSA (Dynamic Sparse Attention)

Attention patterns (orthogonal to above types):
├── Full Causal (default) — standard causal mask
├── Sliding Window — window_size parameter (like Gemma 3)
└── Chunked Attention — chunk_size parameter (like Llama 4)
```

---

## 5. FLOPs Calculation System (Important)

### 5.1 Location

FLOPs calculation is in the `num_floating_point_operations(args, batch_size)` function in `megatron/training/training.py`.

### 5.2 Function Structure

```python
num_floating_point_operations(args, batch_size)
├── calculate_layer_counts()    # Count layer types for hybrid models
├── mlp_layer_flops()          # MLP layer FLOPs
├── moe_layer_flops()          # MoE layer FLOPs
├── attn_layer_flops()         # Attention layer FLOPs
├── mamba_layer_flops()        # Mamba layer FLOPs
├── hybrid_flops()             # Hybrid model total FLOPs
└── transformer_flops()        # ★ Standard Transformer total FLOPs ★
```

### 5.3 Formula Conventions

- **3x multiplier**: Each GEMM needs 3 executions (forward + backward wgrad + backward dgrad) → `forward_backward_expansion_factor = 3`
- **2x FMA**: m×n matrix times n×k matrix = 2mnk floating point operations → `fma_expansion_factor = 2`
- **Causal mask**: Attention FLOPs divided by 2 (`seq_length / 2`) because causal mask is only half non-zero
- **SwiGLU**: FFN expansion factor is 3 (vs standard FFN's 2) → `ffn_expansion_factor = 3 if args.swiglu else 2`

### 5.4 Attention FLOPs Formulas

**MHA/GQA** (in `transformer_flops()`):
```
standard_self_attn_term = 3 * 2 * (
    hidden_size * (Q_proj_size + K_proj_size + V_proj_size + gate_proj_size)  # QKV projection
    + Q_proj_size * seq_length / 2 * 2   # ★ core attention: QK^T and (QK^T)V ★
    + Q_proj_size * hidden_size          # output projection
)
```

**★ Key Issue (Issue #1725) ★**:
`seq_length / 2` assumes FULL causal attention. For **Sliding Window Attention**, actual FLOPs should be based on `min(seq_length, window_size) / 2` not `seq_length / 2`. For **Chunked Attention**, should be based on `chunk_size`. Current code doesn't distinguish these attention patterns, leading to FLOPs overestimation.

**MLA** (DeepSeek style):
```
core attn FLOPs = seq_length / 2 * num_heads * (qk_head_dim + qk_pos_emb_head_dim)  # QK^T
                + seq_length / 2 * num_heads * v_head_dim                             # attn*V
```

### 5.5 Parameter Mapping

FLOPs calculation uses `args` (from `get_args()`), not `TransformerConfig`. Key parameters:

| args field | Meaning |
|------------|---------|
| `args.seq_length` | Sequence length |
| `args.hidden_size` | Hidden layer size |
| `args.num_attention_heads` | Number of attention heads |
| `args.num_query_groups` | GQA groups |
| `args.kv_channels` | Dimension per attention head |
| `args.ffn_hidden_size` | FFN intermediate size |
| `args.swiglu` | Whether to use SwiGLU |
| `args.group_query_attention` | Whether to use GQA |
| `args.multi_latent_attention` | Whether to use MLA |
| `args.num_experts` | MoE number of experts |
| `args.moe_router_topk` | TopK routing |
| `args.experimental_attention_variant` | Experimental attention variant name |
| `args.linear_attention_freq` | Linear attention frequency |

**⚠️ Parameters that don't currently exist but may need to be added**:
- `args.sliding_window_size` — Sliding window size
- `args.chunk_attention_size` — Chunked attention size
- `args.attention_pattern_type` — "full_causal" / "sliding_window" / "chunked"

These need to be added in the appropriate `_add_*_args()` function in `megatron/training/arguments.py`.

---

## 6. Common Task Guides

### 6.1 Modifying FLOPs Calculation

1. **File**: `megatron/training/training.py` → `num_floating_point_operations()` function
2. **Locate**: Inside the `transformer_flops()` inner function, find `standard_self_attn_term`
3. **Core attention FLOPs part**: `query_projection_size * args.seq_length / 2 * 2`
4. **How to modify**:
   - Check attention pattern (e.g., `args.attention_pattern_type`)
   - Replace `args.seq_length` with effective sequence length
   - For sliding window: `effective_seq_len = min(args.seq_length, args.sliding_window_size)`
   - For chunked: `effective_seq_len = args.chunk_attention_size`
5. **Add new parameters**: In `megatron/training/arguments.py` in `_add_network_size_args()`
6. **Test**: Add tests for FLOPs calculation in `tests/unit_tests/`
7. **Don't forget**: MLA branch also has the same `seq_length / 2` issue

### 6.2 Adding New Attention Variant

1. Implementation: Create file in `megatron/core/transformer/`
2. Register ModuleSpec: `megatron/core/models/gpt/gpt_layer_specs.py`
3. Add config: `megatron/core/transformer/transformer_config.py`
4. Add CLI arguments: `megatron/training/arguments.py`
5. Update FLOPs: `megatron/training/training.py`
6. Add tests: `tests/unit_tests/transformer/`

### 6.3 Adding New MoE Functionality

1. Core implementation: `megatron/core/transformer/moe/`
2. Routing logic: `router.py`
3. Load balancing: aux loss in `moe_utils.py`
4. Parallelism strategy: Handle EP (Expert Parallel) process groups

---

## 7. Testing

### Structure
```
tests/unit_tests/          # Mirrors megatron/core/ structure
tests/functional_tests/    # End-to-end tests
```

### Running
```bash
# All unit tests
pytest tests/unit_tests/ -v

# Specific module
pytest tests/unit_tests/transformer/ -v

# With coverage
pytest --cov=megatron tests/unit_tests/
```

### Pattern
```python
# Test file mirrors source path:
# megatron/core/transformer/attention.py → tests/unit_tests/transformer/test_attention.py

import pytest
from megatron.core.transformer.transformer_config import TransformerConfig

class TestMyFeature:
    def setup_method(self):
        self.config = TransformerConfig(num_layers=2, hidden_size=64, ...)

    def test_basic(self):
        assert ...

    @pytest.mark.parametrize("param", [1, 2, 4])
    def test_parametrized(self, param):
        assert ...
```

---

## 8. Common Pitfalls

1. **Don't modify `megatron/legacy/`** — Deprecated, kept only for backward compatibility
2. **args vs config** — `core/` uses TransformerConfig, `training/` uses get_args(), don't mix
3. **Distributed consistency** — Code runs on multiple GPUs, ensure all ranks execute same code paths
4. **FP16/BF16 wrapping** — Model wrapped by `Float16Module`, be careful with dtype handling
5. **Pre-commit scope** — black/isort/pylint only check `megatron/core/`
6. **Backward compatibility** — Don't break existing argument parsing or checkpoint loading format
7. **FLOPs formula modifications** — Ensure updating both `transformer_flops()` and `hybrid_flops()` code paths
8. **MoE layer frequency** — `moe_layer_freq` can be int or list, handle both cases

---

## 9. Decision Trees

### Which directory should I modify?

```
Affects model architecture/parallelism/optimizer?
├─ Yes → megatron/core/
└─ No → Affects training loop/arguments/checkpointing?
    ├─ Yes → megatron/training/
    └─ No → Model-specific entry point?
        └─ Yes → pretrain_*.py
```

### FLOPs Calculation Modification Path

```
Need to modify FLOPs calculation?
├─ Standard Transformer → transformer_flops() inner function
├─ Hybrid (Mamba+Transformer) → hybrid_flops() inner function
├─ New parameters → arguments.py _add_network_size_args()
└─ Both → Ensure both paths are updated
```

---

## 10. Quick Reference

| What | Where |
|------|-------|
| **FLOPs calculation** | `megatron/training/training.py` → `num_floating_point_operations()` |
| **Add CLI argument** | `megatron/training/arguments.py` → relevant `_add_*_args()` function |
| **Model implementation** | `megatron/core/models/` — copy existing model (e.g., `gpt/`) |
| **Attention mechanism** | `megatron/core/transformer/attention.py` or add new spec |
| **Add dataset** | `megatron/core/datasets/` — inherit from `MegatronDataset` |
| **Parallelism** | `megatron/core/parallel_state.py` and `megatron/core/{tensor,pipeline,distributed}_parallel/` |
| **Training loop** | `megatron/training/training.py` — `pretrain()`, `train_step()`, `evaluate()` |
| **Checkpointing** | `megatron/core/dist_checkpointing/` — implement `sharded_state_dict()` |

---

