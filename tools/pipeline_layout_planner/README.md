# Pipeline Layout Planner

Pick a `--pipeline-model-parallel-layout` for a model whose pipeline stages are uneven, for
example a MoE model with MTP layers and a large vocabulary, from two short profiling runs.

With pipeline parallelism, every step waits for the slowest stage. Without a custom layout,
Megatron splits decoder layers evenly and puts the embedding on the first stage and the MTP
layers and the loss (which includes the output layer) on the last stage. The output layer
multiplies every token by a `hidden x vocabulary` matrix and each MTP layer adds a transformer
block plus another pass through the output layer, so the last stage can carry several times the
work of the others.

The planner:

1. fits forward and backward time per micro-batch for each unit (`E`, `t`, `m`, `L`) and a
   per-stage constant from the per-rank `forward-compute` and `backward-compute` timers of two
   profiling runs;
2. enumerates every layout that `PipelineParallelLayerLayout.validate_layer_layout` accepts;
3. simulates the non-interleaved 1F1B schedule for each and adds the per-step overhead;
4. drops layouts whose estimated peak memory exceeds `--hbm-gb`;
5. prints the fastest layouts next to Megatron's default split and the count-based split
   (`[E, t*N, m*K, L]` cut into equal unit counts).

## Workflow

Run all commands from the repository root. Put the model flags of your training command in a
file (for example `args.txt`); the planner reads the shape flags it needs and ignores the rest.

```bash
# 1. Two layouts whose timings identify every unit cost.
python -m tools.pipeline_layout_planner profile-layouts --megatron-args args.txt --pp-size 4
# -> ["Et*4|t*4|t*4|t*4mL", "Et*7|t*5|t*4m|L"]

# 2. Train each for about 20 iterations with level-2 timers and save the logs.
#    <training command> --pipeline-model-parallel-layout "Et*4|t*4|t*4|t*4mL" \
#      --timing-log-level 2 --timing-log-option all --log-interval 1 --exit-interval 20

# 3. Fit unit costs. --ranks-per-stage is TP x CP x DP (ranks per pipeline stage).
python -m tools.pipeline_layout_planner fit --megatron-args args.txt --data-parallel-size 2 \
  --ranks-per-stage 2 --run "Et*4|t*4|t*4|t*4mL=prof1.log" --run "Et*7|t*5|t*4m|L=prof2.log" \
  -o costs.json

# 4. Optional: re-estimate the per-step overhead from runs without level-2 timers, for example
#    the default split and the count-based split. Level-2 timers add a few percent of step time.
python -m tools.pipeline_layout_planner calibrate --megatron-args args.txt \
  --data-parallel-size 2 --pp-size 4 --costs costs.json \
  --run "default=default.log" --run "Et*4|t*5|t*5|t*2mL=count.log" -o costs.json

# 5. Rank layouts.
python -m tools.pipeline_layout_planner plan --megatron-args args.txt --data-parallel-size 2 \
  --pp-size 4 --costs costs.json --hbm-gb 130

# 6. Compare a measured run with the prediction.
python -m tools.pipeline_layout_planner score --megatron-args args.txt --data-parallel-size 2 \
  --pp-size 4 --costs costs.json --layout "Et*5|t*6|t*5|mL" --measured-ms 3800
```

`size` reports parameters, FLOPs and memory per unit (and per stage with `--layout`).
`baselines` prints the default and count-based layouts. `plan` and `score` also accept
`--tflops` instead of `--costs` for a FLOP-only estimate, which is less accurate: on the
measurements below, FLOPs overestimated the MTP layer by about 2.8x and underestimated the loss
stage.

Instead of `--megatron-args`, `--shape` takes a JSON file of `ModelShape` fields (see
`model.py`). Pass `--vocab-size` for the padded vocabulary size when using `--megatron-args`.

## Measured Results

16-layer MoE (hidden 2048, top-8, vocabulary 128256, one MTP layer, full recompute), PP4,
timer-free step times on the same node:

| Setup | Planner layout | Default | Count-based | Planner | vs default | Prediction error |
| --- | --- | --- | --- | --- | --- | --- |
| 8x H200, 64 experts, EP2, DP2, MBS 4, GBS 128 | `Et*5\|t*6\|t*5\|mL` | 5.28 s | 4.27 s | 3.80 s | 1.39x | 2.7% |
| 4x A100 80GB, 32 experts, EP1, MBS 1, GBS 32 | `Et*5\|t*6\|t*5\|mL` | 6.54 s | 5.25 s | 4.26 s | 1.54x | 1.0% |

## Limitations

- Only layouts without virtual pipelining are searched and simulated. Layouts with more than one
  virtual stage per rank are validated but the interleaved schedule is not simulated.
- `--ranks-per-stage` assumes pipeline parallelism is the outermost rank dimension (the default
  `tp-cp-ep-dp-pp` order).
- Layouts with MTP on a stage without the loss ran about 12 percent slower than predicted in the
  measurements above; `plan` skips them unless `--include-standalone-mtp` is given.
- Layouts with two equally slow stages ran about 6 percent slower than predicted (stage jitter is
  not modeled); ties are broken in favor of a single bottleneck stage.
- The memory estimate counts weights, gradients, optimizer state and logits. It was calibrated at
  micro-batch 1 and is conservative for the loss stage at larger micro-batches.
- Expert-parallel routing imbalance is not modeled beyond what the profiling runs average in.
