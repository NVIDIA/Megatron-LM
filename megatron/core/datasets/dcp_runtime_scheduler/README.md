# CPU runtime DCP scheduling

With dynamic context parallelism the default scheduler groups the DPxCP ranks
(data-parallel size times context-parallel size) with a workload heuristic. One
argument replaces it with a CPU scheduler that plans each global batch from a
cost model; `type` in its file selects the cost model and its search algorithm:

```text
--dynamic-cp-runtime-config /path/to/runtime.json
```

| `type` | Cost model | Search algorithm |
| --- | --- | --- |
| `dcp_benchmark` | Packaged attention predictions plus the user's non-attention times per layer, for [ordinary GQA](#ordinary-gqa) or [full Qwen3.5-35B-A3B](#full-qwen35-35b-a3b); the training configuration decides which | Lightweight planner |
| `dynamic_usp` | A DynamicUSP cost profile: the [original profile](#original-profile) or, with `"hardware": "h100"`, the [calibration measured for classic P2P](#measured-h100-calibration) | DynamicUSP allocation and count search; [Stage 2](#stage-2) is an option of the original profile |

The scheduler runs in one CPU process on each DPxCP leader, without access to
the GPUs. It decides which samples share a CP group and which ranks form each
group, not how context parallelism is executed. A runtime file and launch
arguments for every cost model are in the [configuration
reference](../../../../skills/mcore-configure-dcp-cost-model/references/model-options.md).

## Runtime file and start-up checks

| Key | `type` | Meaning |
| --- | --- | --- |
| `type` | both | Required. `dcp_benchmark` or `dynamic_usp`. |
| `hardware` | `dcp_benchmark` | Required. `h100` or `gb200`: selects the attention parameters and the [NVLink topology](#nvlink-topology). |
| `non_attention` | `dcp_benchmark` | Required. Object with exactly `forward_ms_per_layer` and `backward_ms_per_layer`, the user's [non-attention times](#dcp_benchmark-cost-models). |
| `hardware` | `dynamic_usp` | Optional. `h100` selects the measured calibration; without it the original profile is used. |
| `calibration` | `dynamic_usp` | Optional. Coefficient file that replaces the packaged coefficients of the selected profile. A relative path is resolved beside the runtime file. |
| `max_cp_size` | both | Largest CP group the scheduler may form; see [CP group sizes](#cp-group-sizes). |
| `async` | both | Optional, default `true`: plan the next global batch while the current one trains; see [Planning during training](#planning-during-training). |
| `trace_dir` | both | Optional. Directory for planning records; see [Trace records](#trace-records). `~` is expanded; a relative path is relative to the working directory. |

These are all the keys: another key, a key given twice, a missing required key
or a value of the wrong type stops training while it starts. Every rank reads
and checks the file then, and not again: the schema when the arguments are
parsed and, once the model is built, the file against the model and parallel
configuration (each cost model's conditions are below). A failed check stops
training with a message that names what must change; the [configuration
reference](../../../../skills/mcore-configure-dcp-cost-model/references/model-options.md#when-start-up-rejects-the-configuration)
lists the messages. When all checks pass, rank 0 prints one line that starts
with `Runtime DCP scheduler:` and names what the file selected: the cost model
(`dcp_benchmark` for ordinary GQA, `qwen35_full`, `usp_calib_v1` for the
original profile or `qwen3_h100_p2p`), the NVLink domain and CP group sizes.

Both types require `--dynamic-context-parallel` (hence
`--calculate-per-token-loss`), a `--min-dynamic-context-parallel-size` that is
a power of two dividing DPxCP, sequence parallelism off or tensor-parallel size
1, and one pipeline stage without virtual stages or the conditions of [more
than one pipeline stage](#more-than-one-pipeline-stage). The runtime file, and
a `calibration` file for the whole run, must be byte-identical on every rank.

## CP group sizes

The scheduler forms size-aligned CP groups whose sizes are powers of two that
divide DPxCP, from `--min-dynamic-context-parallel-size` up to the largest size
the cost model covers: 64 for ordinary GQA, 32 for full Qwen3.5, 8 for the H100
calibration; the original DynamicUSP profile covers 8, 16 and 32. A sample fits
a group of `cp` ranks when its length, rounded up to a multiple of 2 x DPxCP,
is at most `cp * max_seqlen_per_dp_cp_rank`. A group holds at most 64
sequences, or a smaller `--thd-max-packed-sequences`, of which a dummy sequence
appended by the tail policy takes one entry (64 gives 63).

`max_cp_size` caps the group size and must be exactly one of the sizes the run
can form. Start-up requires it and names the value when the largest such size
is not the full DPxCP group (DPxCP is not a power of two, or is larger than the
cost model covers), to make the limit on the sample length explicit.

## NVLink topology

The NVLink topology is not configured. It follows from `hardware`, which makes
the cost model read DPxCP rank `r` as a GPU of NVLink domain `r // N`:

| `hardware` | NVLink domain | `N` |
| --- | --- | --- |
| `h100` | One node: 8 consecutive ranks | 8 |
| `gb200` | One domain of 72 GPUs: every rank of the run | 72 |

Start-up cannot see on which node a rank runs, so the launch has to match the
table: on `h100` every node of the job contributes all 8 of its GPUs. Ranks are
numbered node by node: `torch.distributed.run`, started once per node with
`--nnodes`, `--nproc-per-node` and `--node-rank` as in the [SLURM launch
skill](../../../../skills/mcore-run-on-slurm/SKILL.md), gives the workers of
the node with `--node-rank n` the ranks from `n * nproc_per_node` on.

## Planning during training

With `"async": true` a training step plans the next global batch while the
current one trains, when it is not the last step, the dataloader type is not
`external`, and the next batch can be read ahead: `--num-workers` is above 0 or
the varlen dataset (`--use-varlen-dataset`) is used. In every other case, and
in evaluation, a batch is planned at the start of its step. With
`--num-workers 0`, reading ahead makes a dataset transform that draws random
numbers in the training process draw them before the current step trains.

The layout of the default dynamic-CP heuristic is the starting point of every
search; one that violates `max_cp_size`, the CP sizes of the cost model or the
sequence cap is discarded, with the reason in the trace.

When a batch cannot be scheduled, for example because a sample fits no allowed
CP size, every DPxCP rank raises the same `RuntimeScheduleError` at that step;
the message names the sample and the limits. Any other planner failure is
raised on every rank as `RuntimeError: CPU DCP planner failed: ...`, and so is
a planner that does not answer within 300 s; `--distributed-timeout-minutes`
(default 10) must stay above those 300 s for this error to be the one reported.
With tensor-parallel size above 1 only the ranks of tensor-parallel rank 0
raise, which alone read data; the others wait until that time-out.

## Trace records

With `trace_dir` every rank that reads data (tensor-parallel rank 0) appends
one JSON object per line to `rank<NNNNN>.jsonl` (`NNNNN` is the global rank) in
that directory, which every rank creates and write-tests at start-up.

| Field | Meaning |
| --- | --- |
| `event` | `consume`: a planned batch is handed to training. `submit`: a batch was sent to the planner one step ahead; such a record has `iterator` and `sequence_number` and no plan. |
| `iterator`, `sequence_number` | Data iterator of the record (0 is the training iterator in a normal run, evaluation iterators follow) and the batch counter of that iterator. |
| `lookahead`, `asynchronous` | Whether the batch was planned one step ahead and used as planned; the configured `async`. |
| `global_lengths`, `padded_lengths` | Sample lengths of the global batch, as read and as stored. |
| `rows`, `predicted_ms` | The plan (per microbatch and per DPxCP rank, the sample indices of the rank's CP group; neighbouring ranks with the same list form one group) and its predicted time. |
| `statistics` | Search statistics. `incumbent_rejection` and `incumbent_rejection_detail` say why the default layout was discarded. For `dynamic_usp`: `candidates` (every microbatch count that was tried, with its predicted time or the reason it has no layout), `selected_incumbent`, `capacity_fallback` and `incumbent_unpriced` (a plan that is not a result of the [search](#dynamic_usp-cost-profiles), and why a default layout was not scored), `stage1_weight` (the sample weight in use: the object `{"A", "B", "C", "default"}` or the string `"cost_model_sample_work"`) and, with more than one pipeline stage, `pipeline` (the stages, the virtual stages, the `schedule`, the count multiple and the step terms in use). |
| `statistics.stage2` | Only with [Stage 2](#stage-2) switched on: `skipped` (why Stage 2 did no work for the plan, else null), `counts` (the outcome of every microbatch; `ADOPTED`: re-packed), `planned` (the layout that the plan is: `stage2`, `stage1` or `default`), `stage1_predicted_ms` and `stage2_predicted_ms`, `microbatches` (a record for each microbatch that was re-packed, failed or is marked `runtime_capacity`, at most 64; `microbatches_omitted` counts the rest), `first_error` and `rescore_error` (a contained error) and `guard_threshold` (the threshold of the ring placement guard). |
| `model_metadata` | What the cost model loaded: its name, inputs and file digests; `nvlink_domain_size`, the `N` of the [NVLink topology](#nvlink-topology) or null (original profile with a `calibration` table). For `dynamic_usp`, `calibration_path`, `calibration_sha256` and `solver`, the [solver coefficients](#original-profile) in use. For full Qwen3.5, `source_tail_padding`. |
| `pipeline`, `pipeline_terms_are_default`, `pipeline_agreement_key` | Only with [more than one pipeline stage](#more-than-one-pipeline-stage): the stages, the virtual stages and the count multiple (both 1 for a run without virtual stages); the `solver.pipeline` terms left at their defaults; the key of the lengths and the plan that the stages compared at this step. |

## `dcp_benchmark` cost models

The cost model predicts the attention of every CP group. The rest of each layer
is measured by the user: `forward_ms_per_layer` and `backward_ms_per_layer` are
the time one rank spends in that rest of one layer during the forward or the
backward pass of one microbatch, constants between 0 and 100000 ms. Nothing is
inferred, fitted or defaulted. With full recomputation (ordinary GQA) the cost
model itself replays each layer's forward before its backward, so the backward
value excludes the replay. How to measure the values is in the [configuration
reference](../../../../skills/mcore-configure-dcp-cost-model/references/model-options.md#measuring-non-attention-times).

### Ordinary GQA

Used for every model without GDN layers. The cost model predicts the core
attention (the attention kernel with its context-parallel communication,
without the projections); the user's value is everything else in the layer. The
predictor (`models/attention.py`; its docstring has the formula) is an example
of a cost model, not a calibrated product. Start-up admits:

| Setting | Covered |
| --- | --- |
| `hardware` | `h100` or `gb200` |
| `num_attention_heads` / `num_query_groups` / `kv_channels` | 32/4/128, 64/4/128, 16/2/256, 32/2/256, 32/8/128 or 64/8/128 |
| DPxCP | Up to 64 |
| CP group sizes | 1 to 64 |
| `max_seqlen_per_dp_cp_rank` | Up to 1,048,576 tokens. Only an even number trains: an odd one that start-up admits stops the first step with an error about the packed sequence alignment, whatever `max_cp_size` |
| Parallelism | `tensor_model_parallel_size` 1, `pipeline_model_parallel_size` 1, no virtual pipeline stages |
| Attention | `attention_backend` `fused`, `cp_comm_type` `p2p`; no multi-latent attention, experimental attention variant or hybrid model |
| Precision and mode | BF16 without FP8/FP4, `cuda_graph_impl` `none` |
| Recomputation | `recompute_granularity` unset, or `full` with `recompute_method` `uniform` |
| Per-rank buffer | `pad_packed_seq_alignment` `max` or equal to `max_seqlen_per_dp_cp_rank`, `thd_tail_padding_policy` `extend_last` |

The coefficients come from two calibrations of NVIDIA's DCP attention
benchmark; its source marks that of the causal block as not independently
validated. The predictor was compared with whole core-attention times of the
packs it was developed on, not with a blind set. Mean error, then known limits:

| `hardware` | Measured | Forward | Backward |
| --- | --- | --- | --- |
| `gb200` | CP 1: 108 packs measured twice | 3.4 % | 2.5 % |
| `gb200` | CP 2, 3 and 4: 108 packs | 11.1 % | 6.7 % |
| `gb200` | CP 2, 3, 5 and 8 on 8 GPUs: 8 packs | 16.4 % | 11.5 % |
| `h100` | CP 1: 108 packs measured twice | 3.1 % | 3.4 % |
| `h100` | CP 1, other runs: 2088 measurements of 1296 packs | 3.8 % | 5.5 % |
| `h100` | CP 2, 4, 9 and 16: 144 packs | 8.7 % | 3.8 % |
| `h100` | CP 2, 3 and 4 inside one node: 108 packs | 5.7 % | 2.3 % |
| `h100` | CP 17 to 128: 251 packs on 533 placements | 16.8 % | 5.2 % |

- Forward plus backward is not monotonic in the valid tokens of a pack: in 176
  of 589 steps that only added valid tokens it fell, by up to 43 %.
- Under-prediction grows with the CP group, favouring large ones; most at 4096
  tokens per rank: forward 30 % low at `h100` CP 17 to 64 (forward plus
  backward 14 %), 25 % at `gb200` CP 5 and 8 (4 packs; none measured above).
- Measured at 4096, 8192 and 16384 tokens per rank; other buffers, over 64
  sequences at CP 1 and exchanges outside 8 to 128 MiB are extrapolated.
- `gb200`, 32/2/256: the backward price of the causal block jumps between the
  measured lengths (CP 1, one sequence: 4.9 ms at 9472 tokens, 8.2 ms at 9600).
- No pack of more than 3 sequences was measured above CP 32; up to CP 32
  (`h100`, backward) packs of 33 to 64 sequences are 8 % to 11 % low.

### Full Qwen3.5-35B-A3B

Used when the model has GDN layers (`experimental_attention_variant` `gdn`).
The cost model covers the whole model, 40 decoder layers and one MTP layer, and
predicts each whole attention module (30 GDN, 11 full attention) with
projections and layout conversion. The user's value is everything outside it:
one value for all layers or one per layer type, `gdn`, `gqa` (full attention)
and optionally `mtp` for the MTP layer, which otherwise uses the `gqa` value.
Start-up admits one configuration and names the first setting that differs:

| Setting | Required value |
| --- | --- |
| Layers | `num_layers` 40, `mtp_num_layers` 1 without `mtp_use_repeated_layer`, `linear_attention_freq` 4 (three GDN layers, then one full-attention layer) |
| Attention | `hidden_size` 2048, `num_attention_heads` 16, `num_query_groups` 2, `kv_channels` 256, `qk_layernorm`, `attention_output_gate`, `attention_dropout` 0 |
| GDN | `linear_conv_kernel_dim` 4, `linear_num_key_heads` 16, `linear_num_value_heads` 32, `linear_key_head_dim` and `linear_value_head_dim` 128, `gdn_kernel_backend` `fla`, `linear_cp_mode` `chunkwise` |
| Hardware | `"hardware": "gb200"` in the file |
| Parallelism | DPxCP 32, `expert_model_parallel_size` 32, `tensor_model_parallel_size` 1, `pipeline_model_parallel_size` 1, no virtual pipeline stages |
| Precision and mode | BF16 without FP8/FP4, `cuda_graph_impl` `none`, no recomputation |
| Context parallelism | `cp_comm_type` `p2p`, `cp_partition_mode` `contiguous`, `attention_backend` `auto` or `fused` |
| Per-rank buffer | `max_seqlen_per_dp_cp_rank` 16384, `pad_packed_seq_alignment` `max` or 16384 |

One coefficient vector per direction prices a module at every CP size, fitted
on the calibration packs of one measurement on 32 GB200 ranks (full attention:
8 packs at each CP size; GDN: 512 points per direction). Error on its held-out
packs, six pack shapes at each CP size, then the known limits:

| Module | Held out | Forward, mean / worst | Backward, mean / worst |
| --- | --- | --- | --- |
| Full attention | 192 packs | 0.6 % / 2.9 % | 1.4 % / 8.2 % |
| GDN | 256 points each | 2.2 % / 11.4 % | 1.8 % / 6.7 % |

- Measured with the native CP transport, ring-routed layout conversion and
  `extend_last` tail padding; applied here to classic P2P with power-of-two CP
  sizes and the `append_dummy_seq` tail policy that GDN requires.
- The weak spot is the full-attention backward price at CP 2 to 5 (2.4 % to
  4.3 %, about 4 % low at CP 2). The measured packs hold 1 to 32 sequences
  (GDN: up to 64) of at least 1,024 tokens and are 25 % to 98 % full; prices
  outside that range are extrapolated.
- Against the limits of the source experiment (3 % weighted error, 5 % at the
  90th percentile, 10 % for one pack) the full-attention module passes and the
  GDN forward pass does not (`fit` in `models/full35b/parameters.json`).

## `dynamic_usp` cost profiles

Both profiles are planned by a port of DynamicUSP's `solve_gbs` (commit
`1ff97a4ae0beb5e088fadb9e112265bdd194b24d`). For one number of microbatches,
Stage 1 chooses the CP group sizes of every microbatch and places the samples
so that the weight per rank is balanced. The count search lays out up to four
microbatch counts from a lower bound on and plans the one whose predicted step
time is lowest. `ORIGIN.json` lists every ported part, the differences, what is
not included and what was compared with upstream. In short:

- A CP group holds at most 64 sequences and stores every sequence padded to a
  multiple of 2 x DPxCP; upstream has no sequence limit and counts unpadded
  tokens. The lower bound on the count also counts sequences, and when capacity
  rejects every count of the window the search goes on past it, over at most 16
  counts, where upstream raises an error (`runtime/hot_switch.py`).
- The default layout is scored with the same step time and kept unless the
  search result is predicted faster; when the search yields no layout and none
  was priced, a capacity-only layout is used. The trace statistics mark both.
- The time of one layout grows about fourfold when the number of samples
  doubles. Measured without a default layout on one host: below 1 s at 256
  samples, up to about 40 s at 2048 and about 100 s at 4096.
- Plans are defined on CPython 3.12 or later, the project's interpreter. An
  older one rounds the built-in `sum` of floats differently, and plans were
  seen to differ there: the H100 sample weight, Stage 1 of the original profile
  on batches of equal lengths, the Stage 2 keep rule at an exact tie.

### Original profile

`{"type": "dynamic_usp"}` selects DynamicUSP's original cost profile: a fixed
synthetic model of 44 layers with 24 query heads of dimension 160, U8
all-to-all plus ring communication and an iteration tail. It reads neither the
training model configuration nor the tensor- and expert-parallel sizes, so its
predictions do not follow the model that is trained. It needs DPxCP and
`--min-dynamic-context-parallel-size` to be 8, 16 or 32, its CP group sizes.
Its packaged table was measured on nodes of 8 GPUs (all-to-all inside a node,
ring between nodes); the start-up line shows that NVLink domain, and none for a
`calibration` table.

`calibration` names a coefficient table in the format of the packaged
`models/usp_calib_v1.json`, with `"version": "v1.1"`; it replaces the
coefficients, not the formulas. The packaged table is the upstream example
table and was not fitted to a Megatron model; the [calibration
skill](../../../../skills/mcore-calibrate-dcp-runtime/SKILL.md) describes how
to measure one. A table may carry a top-level `"solver"` object with the
coefficients of the search; the packaged table has none, so the defaults apply:

| Key | Meaning | Default |
| --- | --- | --- |
| `solver.sample_weight` | `{"A": a, "B": b, "C": c}`. The Stage 1 weight of a sample of L tokens is `max(1, A*L^2/1e10 + B*L/1e5 + C)`. | A = 7706.03421944, B = 6528.86747592, C = 351.14154099: upstream's published measured fit of the forward cost of one sample in a group of eight ranks, not derived from the packaged table. The start-up line has a notice when it applies to a `calibration` table. |
| `solver.tier_time_factors` | `{"8": 1.0, "16": f16, "32": f32}`: predicted time per unit of per-rank weight at each CP size, relative to the smallest CP size the run forms. With it Stage 1 balances predicted time instead of weight. Values are at least 1; keys are CP sizes of the profile as plain decimal numbers, and a key above the largest size the run forms is ignored. The factors are used as written and are not rescaled: a run at DPxCP 32 with `--min-dynamic-context-parallel-size 16` needs a table derived for the sizes 16 and 32. A table of ones is off. | Absent: weight balance |
| `solver.pipeline` | `per_mb_overhead_ms`: overhead per microbatch, added to the predicted step time once for every microbatch. `compute_fwd_share` and `comm_fwd_share`: forward shares of the compute and of the communication time of a group, numbers from 0 to 1; with one pipeline stage they change no prediction, as upstream, and with [more than one](#more-than-one-pipeline-stage) they split every group time into its forward and its backward part. | `per_mb_overhead_ms` 0, `compute_fwd_share` 1/3.9, `comm_fwd_share` 0.5 |
| `solver.stage2` | `{"enabled": true}` switches [Stage 2](#stage-2) on. Optional `"keep_threshold"`, a positive number: a sample with more than that many times the mean work per rank of its microbatch is placed first, in an empty group above 8 ranks. | Off; 8.4 |

### More than one pipeline stage

- Only the original profile plans for a pipeline-parallel size P above 1. With
  virtual pipeline stages `microbatch_group_size_per_vp_stage` is unset or
  equal to P and the number of microbatches of every plan is a multiple of P,
  so a global batch needs at least P x DPxCP / (largest CP size of the run)
  samples. Without them any number is planned, also below P.
- The step time replays a 1F1B order on the forward and backward time of every
  CP group (`runtime/pipeline.py`): upstream's interleaved order with virtual
  stages, Megatron's order of a training step without. The three
  `solver.pipeline` terms change it; the start-up line says when they are
  defaults. The stages count as equal shares of the model, which start-up does
  not inspect.
- Every stage plans the same batch in its own CPU process. All ranks compare
  their settings and Python version (major.minor) at start-up and their
  gathered lengths and plans at every step; a difference stops training.
- Not verified: P above 1 was not run on GPUs with NCCL process groups, and no
  step time of it was compared with a measured one.

### Stage 2

DynamicUSP's Stage 2 (`runtime/stage2.py`; its docstring lists the differences
from upstream) re-packs the samples of each microbatch on its fixed CP groups
when that makes its slowest group faster. It is off by default and offered by
the original profile only: `solver.stage2` in a `calibration` table switches it
on. Start-up then prints `Stage 2 on (keep_threshold=...)`, or `Stage 2
switched on but idle: ...` when the run forms no CP group above 8 ranks.

- As upstream, the layout after Stage 2 is the result of the search; the port
  scores it again with the step time, which can rise, and compares it with the
  default layout. An error inside Stage 2 keeps the Stage 1 layout of the
  microbatch (`PROFILE_EXECUTION_ERROR`); rank 0 warns once per process.
- Groups above 8 ranks are placed under upstream's ring placement guard, which
  requires two more table fields. A re-pack needs such a group next to another
  group in a microbatch (DPxCP 32) and in each such group a sample above the
  guard's floor: 33368 tokens at CP 16 for the packaged table ([calibration
  skill](../../../../skills/mcore-calibrate-dcp-runtime/SKILL.md#stage-2)).
- With Stage 2 on, plans were compared offline with unmodified upstream and are
  equal for lengths that the runtime does not pad and groups below its sequence
  limit. Training speed was not measured.

### Measured H100 calibration

`{"type": "dynamic_usp", "hardware": "h100", "max_cp_size": 8}` selects a
second cost profile, produced with DynamicUSP's calibration procedure for
classic P2P context parallelism (ring groups of 1, 2, 4 or 8 ranks without
all-to-all) and planned by the same search. It prices whole self-attention per
layer at each CP size, an expert remainder that grows with the valid tokens, an
overhead per microbatch and one iteration tail. Start-up names the first
setting that differs from the measured recipe:

| Setting | Required value |
| --- | --- |
| Model | Full Qwen3-30B-A3B: `num_layers` 48, `hidden_size` 2048, `ffn_hidden_size` 6144, `num_attention_heads` 32, `num_query_groups` 4, `kv_channels` 128, `qk_layernorm`, `normalization` `RMSNorm`, `layernorm_epsilon` 1e-6, `gated_linear_unit` without `add_bias_linear`, `hidden_dropout` 0, `attention_dropout` 0 |
| Features the recipe does not use | No MTP layer (`mtp_num_layers` unset or 0), no `attention_output_gate`, no experimental attention variant, no multi-latent attention, not a hybrid model |
| MoE | `num_moe_experts` 128, `moe_router_topk` 8, `moe_ffn_hidden_size` 768, in every layer (`moe_layer_freq` 1), no shared expert (`moe_shared_expert_intermediate_size` unset), `moe_grouped_gemm`, `moe_permute_fusion`, `moe_router_fusion`, no capacity factor (`moe_expert_capacity_factor` unset, `moe_pad_expert_input_to_capacity` off) |
| Dispatch | `moe_token_dispatcher_type` `flex` with `moe_flex_dispatcher_backend` `hybridep` and `moe_hybridep_pad_variable_tokens` (which also sets `moe_hybridep_pad_uneven_dispatch_inputs`) |
| Router | `moe_router_force_load_balancing`, `moe_router_dtype` `fp32`, `moe_router_load_balancing_type` `aux_loss`, `moe_router_pre_softmax` off, no router groups (`moe_router_num_groups` and `moe_router_group_topk` unset) |
| Hardware | `"hardware": "h100"` in the file: 32 H100 in four eight-GPU nodes, eight consecutive ranks per node (see [NVLink topology](#nvlink-topology)). The configuration cannot show this; the file states it. |
| Parallelism | DPxCP 32, `expert_model_parallel_size` 8, `expert_tensor_parallel_size` 1, `tensor_model_parallel_size` 1, `pipeline_model_parallel_size` 1, no virtual pipeline stages, no sequence parallelism |
| Precision and mode | BF16 without FP16/FP8/FP4, `attention_backend` `fused`, `cuda_graph_impl` `none`, no recomputation (`recompute_granularity`, `recompute_method` and `recompute_num_layers` unset, `recompute_modules` at its default), `cross_entropy_loss_fusion` with `cross_entropy_fusion_impl` `native` |
| Context parallelism | `cp_comm_type` `p2p`, `cp_partition_mode` `zigzag`; CP groups of 1, 2, 4 or 8 ranks, so `max_cp_size` is required and is 8 or a smaller size the run can form |
| Per-rank buffer | `max_seqlen_per_dp_cp_rank` 4096, `pad_packed_seq_alignment` `max` or 4096, `thd_tail_padding_policy` `extend_last`, `thd_max_packed_sequences` 64 |
| Samples | At most 16384 tokens each, the measured range. A longer sample stops every rank at the step that plans it. |
| Global batch | 256 samples, in training (`--global-batch-size`, also with a batch-size schedule) and in evaluation (`--eval-global-batch-size`, by default the training value). Both are checked while training starts; `--eval-iters 0` without `--full-validation` switches the evaluation check off. A batch of another size that still reaches the planner is rejected at its step. |

The attention terms were fitted on 60 fixed-layout cases with 20 measured steps
each (largest error 2.9 % forward, 7.7 % backward). Errors, then known limits:

| Compared on 14 further cases | Forward, median / largest | Backward, median / largest |
| --- | --- | --- |
| Attention terms | 0.7 % / 1.8 % | 3.1 % / 14.6 % |
| Attention plus expert remainder against the measured whole block | 1.3 % / 2.7 % | 12.0 % / 19.6 % |

- The backward block is predicted about 17 % low at CP 1 and 2, within 1 % at
  CP 4 and about 13 % high at CP 8. Routing was forced-balanced throughout.
- The overhead per microbatch and the iteration tail were fitted on three
  recorded global batches, with no step-level comparison on other batches.
- The Stage 1 sample weight (A, B, C) was not fitted, and the weight used
  instead jumps where a sample needs the next CP size; the step model
  synchronises more ranks than DynamicUSP's objective ([measurement
  record](../../../../skills/mcore-calibrate-dcp-runtime/references/qwen3-h100-p2p.md#what-was-not-fitted)).

`calibration` names a refit of the same recipe in the format of the packaged
`models/qwen3_h100_p2p.json`; another tensor- or expert-parallel size needs its
own measured profile. The [calibration
skill](../../../../skills/mcore-calibrate-dcp-runtime/SKILL.md) describes the
measurement, the file and how to add a profile for other hardware.

## Provenance

`ORIGIN.json`, the manifest of `models/full35b` and the digests recorded by the
cost models identify the packaged parameters and their sources. DynamicUSP
attribution and its license are in `DYNAMIC_USP_NOTICE.txt`. The attention
predictor was written for this package from two time models of the benchmark.
