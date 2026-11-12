# Full Qwen3-30B-A3B H100 classic-P2P calibration

This is the measurement record of the calibration that `"hardware": "h100"`
selects under the `dynamic_usp` scheduler type. It describes one recipe and
how it was measured; four microbatches per step, an expert-parallel size of 8
and this case matrix are not requirements for calibrations of other models.

The coefficient file is
`megatron/core/datasets/dcp_runtime_scheduler/models/qwen3_h100_p2p.json`. Its
loader is `qwen3_h100_p2p.py` in the same directory. The settings that
start-up checks, the sample and batch limits and the accuracy summary are in
the
[runtime scheduler README](../../../megatron/core/datasets/dcp_runtime_scheduler/README.md#measured-h100-calibration);
the launch arguments are in the
[configuration reference](../../mcore-configure-dcp-cost-model/references/model-options.md#measured-h100-calibration).

## Why it is a separate cost profile

DynamicUSP's original profile models U8 all-to-all plus ring for groups of 8,
16 or 32 ranks, and its calibration procedure asks for a profile of your own
when the structure differs. Classic P2P context parallelism issues no
all-to-all and forms ring groups of 1, 2, 4 or 8 ranks. With a fixed
4096-token buffer on every rank, the time of a whole layer is close to a
constant of the CP size: on average 7.4, 8.8, 10.3 and 13.2 ms forward and 4.2,
5.1, 6.3 and 8.0 ms backward for CP 1, 2, 4 and 8 in the fitting cases. The
upstream table has entries for rates, all-to-all and ring curves and per-tier
multipliers, but none for a per-layer constant of each CP size, and only its
CP 8 tier overlaps the measured sizes.

The measured profile therefore has its own terms, per CP size and direction:

- Attention, in milliseconds per layer for the whole self-attention module
  including projections and context-parallel communication: non-negative
  coefficients for a bias, the largest valid causal work of a rank under
  zigzag ownership, the padded sum of squared lengths per rank and the number
  of real sequences. Lengths are in units of 4096 tokens.
- Expert and remaining work per rank: a bias plus a slope on valid tokens per
  rank, averaged over each expert-parallel block in the timeline.
- One overhead per microbatch and one tail per iteration.

It is planned by the same DynamicUSP allocation and count search as the
original profile.

## Recipe

| Setting | Measured value |
| --- | --- |
| Model | Qwen3-30B-A3B, all 48 layers, BF16 |
| Devices | Four nodes, eight H100 per node, 32 ranks |
| Placement | Size-aligned CP groups inside one eight-GPU node |
| Parallelism | TP1, PP1, EP8; CP 1, 2, 4 and 8 |
| Attention | Fused attention, `cp_comm_type=p2p`, no CUDA graphs, no activation recomputation |
| MoE | 128 experts, top-8, flex/HybridEP, variable-token padding, forced-balanced routing |
| Buffers | 4096 physical tokens per rank, dataset alignment 64, `extend_last`, zigzag ownership |
| Samples | Logical lengths 1 through 16384 within the legal CP capacity; at most 64 sequences per group |
| Global batch | 256 samples, microbatch size 1 |

The measurement ran with:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_NVLS_ENABLE=0
export NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1 NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export NVTE_NORM_FWD_USE_CUDNN=1 NVTE_NORM_BWD_USE_CUDNN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Routing was forced-balanced in every measurement. The timings do not describe
learned routing, and start-up rejects a configuration without
`moe_router_force_load_balancing`. HybridEP dispatch and combine are separate
from attention P2P.

## Module measurements

The fit uses the loader's own feature and timeline code: valid causal work
under physical zigzag ownership, padded lengths and the real sequence count for
attention, valid tokens for the expert remainder. Reuse that code when
refitting rather than a logical-length approximation, and do not charge the
expert-parallel contribution twice.

The sampling design has 60 fitting cases and 14 held-out cases over CP 1, 2, 4
and 8, declared before collection: per CP size an equal-load group, a low-load
group and sequence-count cases up to 64 sequences, with logical lengths from 1
to 16384. Every case has 20 measured steps. The first case of a run has 10
warm-up steps, later cases five. Paired runs with and without timers were
recorded for CP 1 and CP 8.

Timers cover all 48 layers on the two interior microbatches of four per step,
with production DDP enabled, so first-forward and last-backward DDP behavior is
not counted as per-layer cost. Module cases start from a CP8/DP4 input layout
and use a learning rate of zero. Step wall time is collected separately from
the module observations.

## Step terms

A module-only fit has no per-microbatch or per-iteration term. Three real
global batches of 256 samples were replayed at every feasible microbatch count
(four to seven), with two warm-up and five measured trials each, giving 11
rows. The module prediction for those rows was 3.1 % to 5.9 % below the
measured step time. A per-batch intercept and one shared per-microbatch slope
were fitted to the residual: 1.13 ms per microbatch and a 118.4 ms iteration
tail (the median batch intercept).

## Fit and comparison results

| Quantity | Cases | Forward, median / largest | Backward, median / largest |
| --- | --- | --- | --- |
| Attention, fitting cases | 60 | 0.6 % / 2.9 % | 1.3 % / 7.7 % |
| Attention, held-out cases | 14 | 0.7 % / 1.8 % | 3.1 % / 14.6 % |
| Attention plus remainder against the whole block, held-out cases | 14 | 1.3 % / 2.7 % | 12.0 % / 19.6 % |

The backward remainder is one line for all CP sizes. Against the measured
whole block on the held-out cases, backward is predicted about 17 % low at CP 1
and 2, within 1 % at CP 4 and about 13 % high at CP 8.

Refitting after dropping two more steps per case moves predictions by at most
0.06 % forward and 0.29 % backward. The step terms reproduce the 11 rows they
were fitted on within 0.73 %. There is no step-level comparison on other
batches, no module case with mixed CP sizes in one step, and no comparison of
step time against an unscheduled run. All results come from one H100 system.

## Use

```json
{
  "type": "dynamic_usp",
  "hardware": "h100",
  "max_cp_size": 8
}
```

- `max_cp_size` is required: the calibration ends at CP 8, below DPxCP = 32.
  It is 8, or a smaller size the run can form.
- Start-up compares the training configuration with the recipe and names the
  first setting that differs.
- Every global batch that training or evaluation schedules holds 256 samples;
  start-up checks the configured batch sizes.
- A sample is at most 16384 tokens long. A longer one stops every rank at the
  step that plans it, with the sample and the limit.
- A layout of the default heuristic that uses CP 16 or 32 is discarded as the
  starting point of the search.

## Content of the coefficient file

A refit of this recipe has the structure of the packaged file:

| Field | Content |
| --- | --- |
| `schema_version`, `name`, `unit` | `1`, `"qwen3_h100_p2p"`, `"ms_per_layer"` |
| `domain` | The recipe, with fixed values: `hardware` `"h100"`, `nodes` 4, `ranks_per_node` 8, `world_size` 32, `ep_size` 8, `tokens_per_rank` 4096, `dataset_alignment` 64, `cp_sizes` `[1, 2, 4, 8]`, `partition` `"zigzag"` |
| `domain.min_logical_length`, `domain.max_logical_length` | Measured sample-length range in tokens |
| `domain.max_group_seqs` | Largest number of sequences in a group, at most 64 |
| `domain.tail_padding` | Tail policy of the measurement, `"extend_last"` or `"append_dummy_seq"` |
| `domain.step_overhead_global_batch_size` | Samples per global batch of the step fit |
| `domain.config` | The training settings of the measurement, under the same setting names as in the packaged file. Start-up compares every entry with the training configuration. The entries for the model and MoE shape, forced-balanced routing, zigzag partitioning, fused attention, the parallel sizes and the precision keep their packaged values. |
| `features.names`, `features.token_unit` | The attention features in coefficient order, from `bias`, `logical_s1_per_rank`, `logical_s2_per_rank`, `padded_s1_per_rank`, `padded_s2_per_rank`, `sequence_count` and `zigzag_max_causal_work`; the token unit of the length features |
| `profiles.classic_p2p.attention` | For each of `"1"`, `"2"`, `"4"` and `"8"`: `forward` and `backward` lists of non-negative coefficients, one per feature |
| `profiles.classic_p2p.remainder_ms` | `forward` and `backward`: a bias and a slope on the valid tokens per rank, in units of `tokens_per_rank` |
| `profiles.classic_p2p.microbatch_overhead_ms` | `forward` and `backward` overhead per microbatch |
| `profiles.classic_p2p.iteration_tail_ms` | Iteration tail |
| `profiles.classic_p2p.source`, `profiles.classic_p2p.validation` | How the coefficients were measured, and the fit and comparison errors. Both are required; they appear in `model_metadata` of the trace. |

Name the refit in the runtime file:

```json
{
  "type": "dynamic_usp",
  "hardware": "h100",
  "max_cp_size": 8,
  "calibration": "calibration/qwen3_h100_p2p.json"
}
```

A relative path is resolved beside the runtime file. The file must not change
during a run. The
[calibration skill](../SKILL.md#validate-and-use-the-table) says how to
confirm the file in use from the trace. Check new predictions through the
loader and the planner against predictions written independently by the fit,
and keep fit inputs, case lists and reports outside the runtime package.
Report the measured configuration, the fit and held-out errors and the limits
above with the result.
