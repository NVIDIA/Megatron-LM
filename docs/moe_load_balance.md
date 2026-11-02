# MoE Scheduler (experimental)

MoE Scheduler reduces load imbalance in dropless mixture-of-experts models.
It runs after routing and before token dispatch, placing copies of busy experts
in extra physical slots and redirecting tokens to those copies. EPLB can also
periodically move the home experts, which hold the persistent model and optimizer
state, to balance work across expert-parallel (EP) ranks.

## Main features

- Three load planners: `echo`, `eplb`, and `moon_ep`.
- Configurable replica slots, with weight transfer and gradient reduction through
  Peer-TMA or NCCL P2P.
- Periodic EPLB home-expert migration, including optimizer state, after successful
  optimizer updates. This also supports execution without replicas.
- BF16 and native-parameter MXFP8 execution, shared or per-layer replica storage,
  and checkpoint save/resume with scheduler state.

## Usage

Enable the scheduler in an existing MoE training configuration. For example,
these settings use eight logical experts, four EP ranks, and four replica slots
across the EP group:

```yaml
num_moe_experts: 8
expert_model_parallel_size: 4
moe_enable_scheduler: true
moe_scheduler_planner_type: echo
moe_scheduler_expert_dispatcher_type: replica_peer_tma
moe_scheduler_num_idle_experts: 4

moe_token_dispatcher_type: flex
moe_flex_dispatcher_backend: hybridep
moe_grouped_gemm: true
use_transformer_engine_op_fuser: true
gradient_accumulation_fusion: true
moe_router_dtype: fp32
bf16: true
add_bias_linear: false
gated_linear_unit: true
```

These are configuration fields, not a complete training recipe. For the training
CLI, use the corresponding hyphenated options, such as `--moe-enable-scheduler`,
`--moe-scheduler-planner-type echo`, and `--moe-scheduler-num-idle-experts 4`.

### Choose a planner and transport

| Setting | Behavior |
| --- | --- |
| `moe_scheduler_planner_type: echo` | Redistributes overloaded expert work to replica slots. |
| `moe_scheduler_planner_type: eplb` | Uses global expert loads to choose replication and placement; optionally migrates home experts. |
| `moe_scheduler_planner_type: moon_ep` | Uses MoonEP placement and token rerouting; requires as many replica slots as logical experts. |
| `moe_scheduler_expert_dispatcher_type: replica_peer_tma` | Uses Peer-TMA within one NVLink domain; requires native NCCL symmetric-memory support. |
| `moe_scheduler_expert_dispatcher_type: replica_nccl` | Uses packed NCCL P2P over the EP group; supports home migration. |

`moe_scheduler_num_idle_experts` counts replica slots across the entire EP group.
Both this count and `num_moe_experts` must be divisible by the EP size.
For `moon_ep`, set it equal to `num_moe_experts` and provide equal physical token
counts on all EP ranks before scheduling.

### Enable home-expert migration

Use EPLB with NCCL and a positive update interval:

```yaml
moe_scheduler_planner_type: eplb
moe_scheduler_expert_dispatcher_type: replica_nccl
moe_scheduler_num_idle_experts: 4
moe_scheduler_home_update_interval: 50
```

The interval counts successful optimizer updates. Setting it to `0` keeps home
placements fixed. Set `moe_scheduler_num_idle_experts: 0` with a positive interval
to balance only through home migration, retaining native expert computation and
allocating no replica weights or gradients.

### Configure memory and precision

| Option | Values / default | Purpose |
| --- | --- | --- |
| `moe_scheduler_replica_storage` | `shared` (default), `layerwise` | Share replica storage across layers or allocate it per layer. |
| `moe_scheduler_replica_mxfp8_storage` | `reuse` (default), `separate` | Reuse or separate MXFP8 directional weight storage. |
| `moe_scheduler_replica_exchange_buffer_mb` | `256` | Bound the NCCL replica communication workspace in MiB. |
| `moe_scheduler_home_exchange_buffer_mb` | `256` | Bound the home-migration communication workspace in MiB. |

For MXFP8, additionally set `fp8: e4m3`, `fp8_recipe: mxfp8`, and
`fp8_param: true`. Other FP8 recipes and FP4 are unsupported.

## Usage requirements

- Use EP size greater than one, HybridEP flex token dispatch, TE grouped GEMM
  with the operation fuser, and fused gradient accumulation. HybridEP must be
  installed even when NCCL handles expert-weight transfer.
- Use BF16 execution, FP32 router scores, expert tensor parallelism of one,
  bias-free experts, and discrete expert weights. Use a supported fused activation
  such as SwiGLU, with expert input and FFN dimensions divisible by 128.
- Disable expert capacity limits, capacity padding, and token dropping.
- Do not combine `moe_enable_scheduler` with `moe_virtual_expert_load_balance`.
- Full activation recomputation and MoE-layer recomputation are unsupported.
- Attention-only CUDA graphs are supported. Capturing MoE requires Peer-TMA,
  an explicit `moe_expert_rank_capacity_factor`, and shared replica storage or
  MXFP8 layerwise storage with `reuse`. Separate `moe_router` and
  `moe_preprocess` capture scopes are unsupported.
