# Ling-V3 Tiny pretraining example

This example runs the full Ling-V3 Tiny hybrid pattern from fresh initialization with
`MockGPTDataset`.

Run it on one 8-GPU node:

```bash
bash examples/ling_v3_tiny/train_ling_v3_tiny_8gpu.sh
```

The default is a two-step BF16 training smoke run without validation or test iterations.
`TRAIN_ITERS`, `SEQ_LENGTH`, and `MASTER_PORT` may be overridden. No checkpoint is written unless
`SAVE_DIR` is explicitly set.
