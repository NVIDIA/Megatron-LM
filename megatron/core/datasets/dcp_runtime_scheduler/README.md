# CPU runtime DCP scheduling

With dynamic context parallelism the default scheduler groups the DPxCP ranks
(data-parallel size times context-parallel size) of every global batch with a
workload heuristic. One argument replaces it with a CPU scheduler that plans
each global batch from a cost model:

```text
--dynamic-cp-runtime-config /path/to/runtime.json
```

`type` in that file selects a complete scheduler, a cost model bound to a
search algorithm:

| `type` | Cost model | Search algorithm |
| --- | --- | --- |
| `dcp_benchmark` | Packaged attention predictions plus the user's non-attention times per layer, for [ordinary GQA](#ordinary-gqa) or [full Qwen3.5-35B-A3B](#full-qwen35-35b-a3b); the training configuration decides which | Lightweight planner |
| `dynamic_usp` | A DynamicUSP cost profile: the [original profile](#original-profile) or, with `"hardware": "h100"`, the [calibration measured for classic P2P](#measured-h100-calibration) | DynamicUSP allocation and count search |

The scheduler runs in one CPU process on the DPxCP leader and has no access to
the GPUs. It decides which samples share a CP group and which ranks form each
group; it does not change how context parallelism is executed. Model structure
and parallel sizes come from the training configuration.

The [configuration reference](../../../../skills/mcore-configure-dcp-cost-model/references/model-options.md)
has a runtime file and launch arguments for every cost model.

## Runtime file

| Key | `type` | Meaning |
| --- | --- | --- |
| `type` | both | Required. `dcp_benchmark` or `dynamic_usp`. |
| `hardware` | `dcp_benchmark` | Required. `h100` or `gb200`: selects the attention parameters and the [NVLink topology](#nvlink-topology). |
| `non_attention` | `dcp_benchmark` | Required. Object with exactly `forward_ms_per_layer` and `backward_ms_per_layer`, the user's [non-attention times](#non-attention-times). |
| `hardware` | `dynamic_usp` | Optional. `h100` selects the measured calibration; without it the original profile is used. |
| `calibration` | `dynamic_usp` | Optional. Coefficient file that replaces the packaged coefficients of the selected profile. A relative path is resolved beside the runtime file. |
| `max_cp_size` | both | Largest CP group the scheduler may form; see [CP group sizes](#cp-group-sizes). |
| `async` | both | Optional, default `true`: plan the next global batch while the current one trains; see [Lookahead](#lookahead). |
| `trace_dir` | both | Optional. Directory for planning records; see [Trace records](#trace-records). `~` is expanded; a relative path is relative to the working directory. |

These are all the keys. Another key, a key given twice, a missing required key
or a value of the wrong type stops training while it starts; an unknown key is
reported with the keys the selected `type` accepts. No key describes the
network: the NVLink topology follows from `hardware`.

## Start-up validation

The file is read and checked while training starts, on every rank, and is not
read again during training:

1. When the arguments are parsed: the keys and value types.
2. After the model is built and before data is loaded: the file against the
   model and parallel configuration (the conditions of the selected cost model
   are listed below), and that the cost model has parameters for every CP group
   size the scheduler may form. A size without parameters is reported together
   with the `max_cp_size` that avoids it, when one exists. `trace_dir` is
   created and write-tested.

A failed check stops training with a message that names what must change. A
message with "does not match the training configuration" names a launch
setting. The
[configuration reference](../../../../skills/mcore-configure-dcp-cost-model/references/model-options.md#when-start-up-rejects-the-configuration)
lists the messages and what to change for each.

When all checks pass, rank 0 prints one line with what the file selected:

```text
Runtime DCP scheduler: type=dcp_benchmark, cost model=dcp_benchmark, hardware=h100 (NVLink domain: 8 consecutive ranks), CP group sizes=[1, 2, 4, 8, 16, 32], at most 64 sequences per CP group, async=on, runtime file=/path/to/runtime.json
```

`cost model` is `dcp_benchmark` (ordinary GQA), `qwen35_full`, `usp_calib_v1`
(the original DynamicUSP profile) or `qwen3_h100_p2p` (the H100 calibration).
`NVLink domain` is the number of consecutive ranks the cost model takes to
share one [NVLink domain](#nvlink-topology).

Requirements for both types:

- `--dynamic-context-parallel`, which itself requires
  `--calculate-per-token-loss`.
- Pipeline-parallel size 1 without virtual pipeline stages; sequence
  parallelism off or tensor-parallel size 1.
- `--min-dynamic-context-parallel-size` is a power of two that divides DPxCP.
- The runtime file is byte-identical on every rank. Otherwise every rank stops
  with the number of ranks that read a different file.
- A `calibration` file does not change during a run. Its digest is recorded at
  start-up; a planner process that later loads other bytes stops training.

## CP group sizes

The scheduler forms size-aligned CP groups whose sizes are powers of two that
divide DPxCP, from `--min-dynamic-context-parallel-size` up to the largest size
the cost model covers: 64 for ordinary GQA, 32 for full Qwen3.5, 8 for the H100
calibration. The original DynamicUSP profile covers the sizes 8, 16 and 32.

`max_cp_size` caps the group size. It must be exactly one of the sizes the run
can form; another value is rejected with the list of those sizes, and nothing
is rounded. A runtime file shared by runs with different DPxCP needs a value
that is valid for each of them.

When the largest size the run can form is not the full DPxCP group (DPxCP is
not a power of two, or is larger than the cost model covers), `max_cp_size`
must be set: samples that the training configuration admits could then not be
scheduled, and start-up asks for that limit to be explicit and names the value.

A sample fits a CP group of `cp` ranks when its length, rounded up to a
multiple of 2 x DPxCP, is at most `cp * max_seqlen_per_dp_cp_rank`.

## NVLink topology

The NVLink topology is not configured. It follows from `hardware`, and the
cost model reads DPxCP rank `r` as a GPU of NVLink domain `r // N`:

| `hardware` | NVLink domain | `N` |
| --- | --- | --- |
| `h100` | One node: 8 consecutive ranks | 8 |
| `gb200` | One domain of 72 GPUs: every rank of the run | 72 |

The scheduler forms size-aligned groups. On `h100` a group of up to 8 ranks
therefore lies inside one node and a larger group covers whole nodes; on
`gb200` every group lies inside the one domain:

| Cost model | `hardware` | CP sizes inside one domain | CP sizes that cross domains |
| --- | --- | --- | --- |
| Ordinary GQA | `h100` | 1, 2, 4, 8 | 16, 32, 64: two, four and eight nodes |
| Ordinary GQA | `gb200` | 1 to 64 | None |
| Full Qwen3.5 | `gb200` | 1 to 32 | None |
| H100 calibration | `h100` | 1, 2, 4, 8 | None; larger groups are not formed |
| Original DynamicUSP profile | Not named | 8 | 16, 32: a ring over two and four nodes |

A group has the same predicted time on whichever nodes it lies. For `h100`
the ordinary GQA parameters cover a group that lies inside one node or starts
on a node boundary and have no value for any other group; the H100
calibration has none for a group that leaves a node. The scheduler forms
neither.

The original DynamicUSP profile takes no `hardware` and reads no rank
positions. Its packaged table was measured on nodes of 8 GPUs, with the
all-to-all of every 8 ranks inside a node and the ring between nodes; the
start-up line shows that domain. A table given with `calibration` does not
say where it was measured, and the line shows no domain.

Start-up cannot see on which node a rank runs, so the launch has to match
the table:

- `h100`: ranks `8k` to `8k+7` run on one node, which means that every node
  of the job contributes all 8 of its GPUs.
- `gb200`: all ranks of the run belong to one 72-GPU NVLink domain.
- Ranks are numbered node by node. `torch.distributed.run`, started once per
  node with `--nnodes`, `--nproc-per-node` and `--node-rank` as in the
  [SLURM launch skill](../../../../skills/mcore-run-on-slurm/SKILL.md),
  gives the workers of the node with `--node-rank n` the ranks from
  `n * nproc_per_node` on; on `h100` nodes, `--nproc-per-node 8` gives the
  placement above.

The cost models that take `hardware` require tensor- and pipeline-parallel
size 1, so that a DPxCP rank is the global rank of the launch.

## Sequences per CP group

A CP group holds at most 64 sequences, the range the cost models cover. A
smaller `--thd-max-packed-sequences` applies instead. When the tail policy
appends a dummy sequence (`append_dummy_seq` with `--pad-packed-seq-alignment`)
one of the `--thd-max-packed-sequences` entries is reserved for it, so 64 gives
63. Ordinary GQA on DPxCP = 64 holds at most 32 sequences in a group of 64
ranks. The start-up line shows the cap of the run.

## Planning during training

### Lookahead

With `"async": true` the scheduler plans the next global batch while the
current one trains. A training step does this when it is not the last step,
the dataloader type is not `external`, and the next batch can be read ahead:
`--num-workers` is above 0 or the varlen dataset (`--use-varlen-dataset`) is
used. In every other case, with `"async": false` and in evaluation, a batch is
planned at the start of its step. The trace field `lookahead` says which
happened for each batch.

Lookahead reads the next batch one step early. With `--num-workers 0` a
dataset transform that draws random numbers in the training process then
draws them before the current step trains.

### Default layout

The default dynamic-CP heuristic still proposes a layout for every batch. It
is only the starting point of the search. When it violates a constraint of
the scheduler (a group larger than `max_cp_size`, a CP size the cost model
does not cover, too many sequences in a group) it is discarded, the reason is
recorded in the trace, and planning continues.

### Errors

When a batch cannot be scheduled, for example because a sample fits no allowed
CP size, every DPxCP rank raises the same `RuntimeScheduleError` at that step;
the message names the sample and the limits. Any other planner failure is
raised on every rank as `RuntimeError: CPU DCP planner failed: ...`. A planner
that does not answer within 600 s stops every rank in the same way.

## Trace records

With `trace_dir` every rank appends one JSON object per line to
`rank<NNNNN>.jsonl` in that directory (`NNNNN` is the global rank). The
directory is created and write-tested on every rank while training starts.

| Field | Meaning |
| --- | --- |
| `event` | `consume`: a planned batch is handed to training. `submit`: a batch was sent to the planner one step ahead. |
| `iterator` | Data iterator of the record, numbered in the order the iterators are first scheduled in the process: 0 is the training iterator in a normal run, evaluation iterators follow. |
| `sequence_number` | Batch counter of that iterator. |
| `lookahead` | Whether the batch was planned one step ahead and used as planned. |
| `asynchronous` | The configured `async`. |
| `global_lengths`, `padded_lengths` | Sample lengths of the global batch, as read and as stored. |
| `rows` | The plan: per microbatch and per DPxCP rank, the sample indices of the rank's CP group. Neighbouring ranks with the same list form one group. |
| `predicted_ms` | Predicted time of the planned global batch. |
| `statistics` | Search statistics. `incumbent_rejection` and `incumbent_rejection_detail` say why the default layout was discarded. `model_sequence_limits` lists the CP sizes for which the cost model holds fewer sequences than the cap, for example `{"64": 32}`. |
| `model_metadata` | What the cost model loaded: its name, inputs and file digests. `nvlink_domain_size` is the `N` of the [NVLink topology](#nvlink-topology), or null when the cost model does not know it. For `dynamic_usp`, `calibration_path` and `calibration_sha256` identify the coefficient file in use. For full Qwen3.5, `source_tail_padding` and `executed_tail_padding`. |

`submit` records have `event`, `iterator` and `sequence_number` and no plan.

## `dcp_benchmark` cost models

### Ordinary GQA

Used for every model without GDN layers. The cost model predicts the core
attention of every CP group (the attention kernel with its context-parallel
communication, without the projections); the rest of each layer is the user's
[non-attention time](#non-attention-times).

```json
{
  "type": "dcp_benchmark",
  "hardware": "h100",
  "non_attention": {
    "forward_ms_per_layer": null,
    "backward_ms_per_layer": null
  }
}
```

Both nulls must be replaced by measured numbers; the file is rejected until
they are.

The packaged attention parameters cover:

| | |
| --- | --- |
| `hardware` | `h100` or `gb200` |
| `num_attention_heads` / `num_query_groups` / `kv_channels` | 32/4/128, 64/4/128, 16/2/256, 32/2/256, 32/8/128 or 64/8/128 |
| DPxCP | Up to 64 |

and these CP group sizes, depending on `max_seqlen_per_dp_cp_rank`:

| `max_seqlen_per_dp_cp_rank` | `h100` | `gb200` |
| --- | --- | --- |
| 4096 | 1 to 64 | 1 to 64 with `kv_channels` 128; 1 only with `kv_channels` 256 |
| 8192 | 1 only | 1 only |
| 16384 | 1 only | None |
| Another value | None | None |

Where only size 1 is covered, the file needs `"max_cp_size": 1` and every
sample has to fit one rank. Start-up reports every size without parameters.

The training configuration must have:

| Setting | Required value |
| --- | --- |
| Parallelism | `tensor_model_parallel_size` 1, `pipeline_model_parallel_size` 1, no virtual pipeline stages |
| Attention | `attention_backend` `fused`, `cp_comm_type` `p2p`; no multi-latent attention, experimental attention variant or hybrid model |
| Precision and mode | BF16 without FP8/FP4, `cuda_graph_impl` `none` |
| Recomputation | `recompute_granularity` unset, or `full` with `recompute_method` `uniform` |
| Per-rank buffer | `pad_packed_seq_alignment` `max` or equal to `max_seqlen_per_dp_cp_rank`, `thd_tail_padding_policy` `extend_last` |

### Full Qwen3.5-35B-A3B

Used when the model has GDN layers (`experimental_attention_variant` `gdn`).
The cost model covers the complete model, 40 decoder layers and one MTP layer
in their order: 30 GDN and 11 full-attention modules. It predicts each whole
attention module, including projections and layout conversion; the rest of
each layer is the user's [non-attention time](#non-attention-times), one value
for all layers or one per layer type: `gdn`, `gqa` (full attention) and
optionally `mtp` for the MTP layer, which otherwise uses the `gqa` value.

```json
{
  "type": "dcp_benchmark",
  "hardware": "gb200",
  "non_attention": {
    "forward_ms_per_layer": {"gdn": null, "gqa": null, "mtp": null},
    "backward_ms_per_layer": {"gdn": null, "gqa": null, "mtp": null}
  }
}
```

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

CP groups have the power-of-two sizes from `--min-dynamic-context-parallel-size`
up to 32. GDN with a fixed per-rank buffer uses the `append_dummy_seq` tail
policy, so `--thd-max-packed-sequences 64` gives at most 63 sequences per
group.

The FA/GDN module parameters were measured on 32 GB200 ranks with the native CP
transport, ring-routed layout conversion and `extend_last` tail padding; here
they are applied to classic P2P with power-of-two CP sizes and the
`append_dummy_seq` tail policy that GDN requires.

In the source experiment the full-attention module predictions met the
accuracy target; the GDN module predictions and the attention modules taken
together did not meet it on held-out cases. These results are recorded as
flags (`source_status.accuracy` in `models/full35b/parameters.json`); no
accuracy figures are packaged.

### Non-attention times

`forward_ms_per_layer` and `backward_ms_per_layer` are measured by the user:
the time one rank spends in one layer during the forward or the backward pass
of one microbatch, without the part the cost model predicts. That part differs
between the two models:

| Model | Predicted by the cost model | The user's value contains |
| --- | --- | --- |
| Ordinary GQA | Core attention: the attention kernel with its context-parallel communication | Everything else in the layer: QKV and output projections, normalisation, MLP or MoE with expert-parallel communication |
| Full Qwen3.5 | The whole attention module (full attention or GDN) with projections and layout conversion | Everything outside the attention module: normalisation, MoE with expert-parallel communication |

A value is a constant in milliseconds per layer, per microbatch and per rank,
between 0 and 100000. Nothing is inferred, fitted or defaulted. With full
recomputation (ordinary GQA) the cost model itself replays each layer's forward
before its backward, so the backward value excludes the replay. The
[configuration reference](../../../../skills/mcore-configure-dcp-cost-model/references/model-options.md#measuring-non-attention-times)
describes how to measure the values.

## `dynamic_usp` cost profiles

### Original profile

```json
{
  "type": "dynamic_usp"
}
```

DynamicUSP's original cost profile: a fixed synthetic model of 44 layers with
24 query heads of dimension 160, U8 all-to-all plus ring communication and an
iteration tail. It describes that structure and does not read the training
model configuration, so its predictions do not follow the layers, heads or
communication of the model that is trained. It needs DPxCP and
`--min-dynamic-context-parallel-size` to be 8, 16 or 32, the CP group sizes of
the profile, and takes no `non_attention` input.

`calibration` names a coefficient table in the format of the packaged
`models/usp_calib_v1.json`; it replaces the coefficients, not the formulas.
The packaged table is the upstream example table and was not fitted to a
Megatron model. The
[calibration skill](../../../../skills/mcore-calibrate-dcp-runtime/SKILL.md)
describes how to measure a table.

### Measured H100 calibration

```json
{
  "type": "dynamic_usp",
  "hardware": "h100",
  "max_cp_size": 8
}
```

A second cost profile of this type, produced with DynamicUSP's calibration
procedure for classic P2P context parallelism and planned by the same search.
The original profile describes U8 all-to-all plus ring groups of 8, 16 or 32
ranks; classic P2P forms ring groups of 1, 2, 4 or 8 ranks without all-to-all.
The calibration therefore has its own terms for each CP size: whole
self-attention forward and backward per layer (projections and context-parallel
communication included), an expert remainder that grows with the valid tokens
and is averaged over each expert-parallel block, an overhead per microbatch and
one iteration tail. It takes no `non_attention` input.

The measured recipe is the domain. Start-up names the first setting that
differs from it:

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
| Global batch | 256 samples, in training and in evaluation |

Besides the model shape, two required values differ from the argument
defaults and are easy to miss: `--norm-epsilon 1e-6` and
`--moe-router-dtype fp32`.

The global batch size is checked while training starts, for training
(`--global-batch-size`, also with a batch-size schedule) and for evaluation
(`--eval-global-batch-size`, by default the training value). Evaluation is
checked when it is enabled; `--eval-iters 0` without `--full-validation`
switches that check off. A batch of another size that still reaches the
planner is rejected at its step.

The attention terms were fitted on 60 fixed-layout cases with 20 measured steps
each (largest error 2.9 % forward, 7.7 % backward) and compared with 14 further
cases: at most 1.8 % forward and 14.6 % backward, medians 0.7 % and 3.1 %.
Attention plus expert remainder against the measured whole block on those 14
cases: at most 2.7 % forward and 19.6 % backward, medians 1.3 % and 12.0 %; the
backward block is predicted about 17 % low at CP 1 and 2, within 1 % at CP 4 and
about 13 % high at CP 8. The overhead per microbatch and the iteration tail
were fitted on three recorded global batches at every feasible microbatch count
(11 rows, residual at most 0.7 %); there is no step-level comparison on other
batches. All measurements used forced-balanced routing.

`calibration` names a refit of the same recipe in the format of the packaged
`models/qwen3_h100_p2p.json`. The
[calibration skill](../../../../skills/mcore-calibrate-dcp-runtime/SKILL.md)
describes the measurement, the file and how to add a profile for other
hardware.

## Provenance

`ORIGIN.json`, the manifests under `models/` and the digests recorded by the
cost models identify the packaged parameters and their sources. DynamicUSP
attribution and its license are in `DYNAMIC_USP_NOTICE.txt`. Other
architectures need their own attention predictors and layer programs.
