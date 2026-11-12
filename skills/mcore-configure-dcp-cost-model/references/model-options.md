# Runtime file and launch arguments

One file is passed alongside the normal model, data and training arguments:

```text
--dynamic-cp-runtime-config /path/to/runtime.json
```

The [runtime scheduler README](../../../megatron/core/datasets/dcp_runtime_scheduler/README.md)
is the reference for the keys of the file, the start-up checks, the CP group
sizes and what each cost model covers. This page has, for each cost model, the
file to write and launch arguments that pass the start-up checks, and the way
to measure the non-attention times.

The launch blocks hold the arguments the scheduler checks. Add the remaining
model arguments (embeddings, position encoding, vocabulary) and the data,
optimizer, checkpoint and logging arguments of the run.

The launch also has to place the ranks as the cost model assumes; start-up
cannot check that. `hardware` fixes the
[NVLink topology](../../../megatron/core/datasets/dcp_runtime_scheduler/README.md#nvlink-topology):
on `h100` every node runs 8 consecutive ranks (all 8 GPUs of the node), on
`gb200` all ranks belong to one 72-GPU NVLink domain. The README says which
launch numbers the ranks node by node.

## Ordinary GQA

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

Replace both nulls with [measured values](#measuring-non-attention-times); a
file with a null is rejected. Each value is one number for all layers. The
per-layer-type form has the single type `gqa` here, for example
`{"gqa": 0.5}`. `hardware` is `h100` or `gb200`.

Launch arguments on 32 ranks:

```bash
--bf16 --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
--attention-backend fused --cuda-graph-impl none \
--cp-comm-type p2p --context-parallel-size 1 \
--dynamic-context-parallel --min-dynamic-context-parallel-size 1 \
--sequence-packing-scheduler default_dynamic_cp --calculate-per-token-loss \
--max-seqlen-per-dp-cp-rank 4096 --pad-packed-seq-alignment 4096 \
--thd-tail-padding-policy extend_last --thd-max-packed-sequences 64 \
--dynamic-cp-runtime-config /path/to/runtime.json
```

The model arguments must give one of the
[covered attention geometries](../../../megatron/core/datasets/dcp_runtime_scheduler/README.md#ordinary-gqa),
for example `--num-attention-heads 32 --group-query-attention
--num-query-groups 4 --kv-channels 128` for Qwen3-30B-A3B. Any number of
layers is accepted. With full recomputation add
`--recompute-granularity full --recompute-method uniform` and
`--recompute-num-layers`.

## Full Qwen3.5-35B-A3B

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

Replace every null with a [measured value](#measuring-non-attention-times).
One number instead of a mapping applies to all 41 layers. In a mapping `gdn`
and `gqa` (the full-attention layers) are required and `mtp` is optional;
without it the MTP layer uses the `gqa` value. Another name is rejected with
the names this model expects.

Launch arguments on 32 GB200 ranks, one for every
[checked setting](../../../megatron/core/datasets/dcp_runtime_scheduler/README.md#full-qwen35-35b-a3b):

```bash
--num-layers 40 --mtp-num-layers 1 --hidden-size 2048 \
--num-attention-heads 16 --group-query-attention --num-query-groups 2 \
--kv-channels 256 --qk-layernorm --attention-output-gate \
--attention-dropout 0.0 \
--experimental-attention-variant gdn --linear-attention-freq 4 \
--linear-conv-kernel-dim 4 --linear-num-key-heads 16 \
--linear-num-value-heads 32 --linear-key-head-dim 128 \
--linear-value-head-dim 128 --gdn-kernel-backend fla \
--bf16 --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
--expert-model-parallel-size 32 --cuda-graph-impl none \
--cp-comm-type p2p --linear-cp-mode chunkwise \
--cp-partition-mode contiguous --context-parallel-size 32 \
--dynamic-context-parallel --min-dynamic-context-parallel-size 1 \
--sequence-packing-scheduler default_dynamic_cp --calculate-per-token-loss \
--max-seqlen-per-dp-cp-rank 16384 --pad-packed-seq-alignment max \
--thd-tail-padding-policy append_dummy_seq --thd-max-packed-sequences 64 \
--dynamic-cp-runtime-config /path/to/runtime.json
```

## Measuring non-attention times

`dcp_benchmark` needs two measured constants per layer type, in milliseconds
per layer, per microbatch and per rank: the forward and the backward time of
a layer without the part the cost model predicts. The tree has no measuring
script; forward and backward hooks are enough.

1. Launch the training configuration that will use the scheduler, without
   `--dynamic-cp-runtime-config`: the same model, precision, parallel sizes,
   `--max-seqlen-per-dp-cp-rank`, padding and tail policy. With the fixed
   per-rank buffer of that configuration every rank processes
   `max_seqlen_per_dp_cp_rank` tokens in every microbatch; this is the load
   the values describe.
2. Register hooks on every transformer layer (`decoder.layers[i]` of the GPT
   model, `mtp.layers[0].mtp_model_layer` for the MTP layer) and on the
   submodule the cost model predicts:

   | Model | Predicted submodule of a layer |
   | --- | --- |
   | Ordinary GQA | `self_attention.core_attention` |
   | Full Qwen3.5 | `self_attention` |

   Forward: `register_forward_pre_hook` and `register_forward_hook`.
   Backward: `register_full_backward_pre_hook` and
   `register_full_backward_hook`.
3. In every hook call `torch.cuda.synchronize()` and read
   `time.perf_counter()`. The time between the pre-hook and the hook of a
   module is its forward or backward time.
4. For every layer and microbatch subtract the submodule's time from the
   layer's time, separately for forward and backward.
5. Discard warm-up steps. Average over the layers of one type, the
   microbatches, the ranks and the measured steps, and convert to
   milliseconds.

Use the hooks for the measurement run only; the synchronisation slows
training. The values are constants: work that depends on the number of valid
tokens enters only through this average, so measure on batches that represent
the data.

With full recomputation (ordinary GQA) every layer's forward runs a second
time during the backward pass, and the forward hooks fire twice. The cost
model adds that replay itself, from its attention prediction and the forward
value: give the forward time of one pass and the backward time without the
replay.

Values are finite, not negative and at most 100000.

## Original DynamicUSP profile

```json
{
  "type": "dynamic_usp"
}
```

Launch arguments on 8, 16 or 32 ranks:

```bash
--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
--context-parallel-size 1 \
--dynamic-context-parallel --min-dynamic-context-parallel-size 8 \
--sequence-packing-scheduler default_dynamic_cp --calculate-per-token-loss \
--max-seqlen-per-dp-cp-rank 4096 \
--dynamic-cp-runtime-config /path/to/runtime.json
```

The profile does not read the model configuration. Its packaged table was
measured on nodes of 8 GPUs with 8 consecutive ranks per node. Optional
`"calibration": "calibration/calib_table_v1.json"` loads a measured table; the
[calibration skill](../../mcore-calibrate-dcp-runtime/SKILL.md) describes it.

## Measured H100 calibration

```json
{
  "type": "dynamic_usp",
  "hardware": "h100",
  "max_cp_size": 8
}
```

Launch arguments on 32 H100 ranks in four eight-GPU nodes, one for every
[checked setting](../../../megatron/core/datasets/dcp_runtime_scheduler/README.md#measured-h100-calibration):

```bash
--num-layers 48 --hidden-size 2048 --ffn-hidden-size 6144 \
--num-attention-heads 32 --group-query-attention --num-query-groups 4 \
--kv-channels 128 --qk-layernorm --normalization RMSNorm \
--norm-epsilon 1e-6 --swiglu --disable-bias-linear \
--attention-dropout 0.0 --hidden-dropout 0.0 \
--num-experts 128 --moe-router-topk 8 --moe-ffn-hidden-size 768 \
--moe-layer-freq 1 --moe-router-dtype fp32 \
--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep \
--moe-hybridep-pad-variable-tokens --moe-router-force-load-balancing \
--moe-grouped-gemm --moe-permute-fusion --moe-router-fusion \
--bf16 --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
--expert-model-parallel-size 8 \
--attention-backend fused --cuda-graph-impl none \
--cross-entropy-loss-fusion --cross-entropy-fusion-impl native \
--micro-batch-size 1 --global-batch-size 256 \
--cp-comm-type p2p --context-parallel-size 1 \
--dynamic-context-parallel --min-dynamic-context-parallel-size 1 \
--sequence-packing-scheduler default_dynamic_cp --calculate-per-token-loss \
--max-seqlen-per-dp-cp-rank 4096 --pad-packed-seq-alignment 4096 \
--thd-tail-padding-policy extend_last --thd-max-packed-sequences 64 \
--dynamic-cp-runtime-config /path/to/runtime.json
```

Evaluation uses the training batch size unless `--eval-global-batch-size` is
given; it must be 256 as well, or evaluation is switched off with
`--eval-iters 0`. Samples are at most 16384 tokens long. Optional
`"calibration": "calibration/qwen3_h100_p2p.json"` loads a refit of the same
recipe; the [calibration skill](../../mcore-calibrate-dcp-runtime/SKILL.md)
describes it.

## When start-up rejects the configuration

Training stops before the first step with one of these messages:

| Message | What to change |
| --- | --- |
| `Runtime configuration '<file>': ...` with an unknown or missing key, a wrong type or a value out of range | The file. An unknown key is listed with the allowed keys. |
| `... max_cp_size=<n> is not a CP group size this run can form ...` | Set `max_cp_size` to one of the sizes the message lists. |
| `The runtime DCP scheduler forms CP groups of [...] ... Set "max_cp_size": <n> ...` | Add that value to the file, or use a power-of-two DPxCP that the cost model covers. |
| `The '<type>' cost model cannot price CP groups of [...] ranks ...` | Set the `max_cp_size` the message names. When it names none, the cost model has no parameters for this hardware, attention geometry and `max_seqlen_per_dp_cp_rank`. |
| `... does not match the training configuration: ...` | The launch setting the message names, to the value it requires. A configuration the cost model does not cover cannot use that scheduler type. |
| `... was measured with 256 samples per global batch ...` | `--global-batch-size` and `--eval-global-batch-size` to 256, or `--eval-iters 0`. |
| `<k> of <N> DPxCP ranks read a runtime configuration that differs ...` | Give every node the same file. |
| `trace_dir '<path>' ... cannot be created or written on this rank` | Use a directory every rank can write. |
