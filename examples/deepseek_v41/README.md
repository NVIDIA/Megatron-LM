# DeepSeek-V4.1 training smoke and performance runs

`config_flash.json` is the released architecture configuration from
[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json).
`train_smoke.py` reduces widths, experts, vocabulary and Engram capacity for an
accessible training test. `--depth full` retains all 40 logical backbone blocks,
32 vision blocks, both Engram insertion points and three DSpark blocks.
It does **not** instantiate the released 763B-parameter checkpoint.

Use two GPUs initially. The test uses MCore DDP, its forward/backward schedule,
gradient accumulation, expert dispatch and AdamW. All five components must update;
NaN or skipped iterations fail the run. The checkpoint option zeros and restores
all model parameters and requires exact equality on every rank.

```bash
uv run python -m torch.distributed.run --standalone --nproc-per-node=2 \
  examples/deepseek_v41/train_smoke.py --depth small --ep 2 --steps 3 \
  --distributed-optimizer --checkpoint --output-dir /shared/v41-smoke

uv run python -m torch.distributed.run --standalone --nproc-per-node=2 \
  examples/deepseek_v41/train_smoke.py --depth small --ep 2 --steps 3 \
  --fused --distributed-optimizer --checkpoint --output-dir /shared/v41-fused-smoke

uv run python -m torch.distributed.run --standalone --nproc-per-node=2 \
  examples/deepseek_v41/train_smoke.py --depth full --hidden-size 1024 \
  --seq-length 512 --steps 8 --ep 2 --fused --distributed-optimizer \
  --output-dir /shared/v41-full-depth
```

Both native and fused variants use identical model dimensions. `--fused` enables
the cuDNN/FlashMLA attention path, fused RoPE, supported mHC operations, grouped
MoE GEMMs, permutation and activation fusion. It does not enable CUDA graphs or
weight quantization. FlashMLA, cuDNN DSA and a matching Transformer Engine build
must be available for the fused path; native mode uses torch attention math.

Every successful run prints `DSV41_TRAINING_OK` after the cross-rank validation
barrier and writes `results.json` under the requested output directory. The report
contains per-step losses, gradient norms, timings, peak memory and component-update
checks. Median time excludes the first two iterations when possible. Keep jobs
isolated when interpreting performance measurements.

The test uses synthetic data and random weights. It validates execution and
training behavior, not convergence or equivalence to the released quantized
checkpoint. Full support details are in `docs/models/deepseek_v41.md`.
