## Performance Results

### H100

Experiment Setup:
- 256 H100 GPUS
- NVLink 4th Generation
- InfiniBand 8x50 GB/s


| Model | Dispatcher | System | Precision | #GPUs | SEQ LEN | TP | CP | EP | PP | EDP | ETP | VPP | DP | FSDP | MBS | GBS | GA | recompute | Step time (sec) | Per GPU TF | MFU | Mem | Notes | Code |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Cross-node EP strategy | | | | | | | | | | | | | | | | | | | | | | | | |
| 235B | DeepEP | H100 | BF16 | 256 | 4096 | 2 | 1 | 32 | 8 | 1 | 1 | 4 | 16 | / | 1 | 2048 | 128 | norm, act | 19.7 | 245 |  |  |  | moe_dev(f3e6c5) |
| 235B | DeepEP | H100 | BF16 | 256 | 4096 | 2 | 1 | 32 | 8 | 1 | 1 | 4 | 16 | / | 1 | 2048 | 128 | norm, act | 17.6 | 276 |  |  | +1f1b overlap | moe_dev(f3e6c5) |
| Intra-node EP strategy |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 235B | DeepEP | H100 | BF16 | 256 | 4096 | 2 | 1 | 8 | 1 | 32 | 1 | 1 | 128 | On | 4 | 2048 | 4 | full | 17.6 | 276 |  |  | disable log para norm | main(1e057005) |
| 235B | A2A | H100 | BF16 | 256 | 4096 | 2 | 1 | 8 | 32 | 1 | 1 | 3 | 4 | / | 1 | 2048 | 512 | norm, act | 24.2 | 200 |  |  |  | main(1e057005) |
| 235B | DeepEP | H100 | BF16 | 256 | 4096 | 4 | 1 | 8 | 8 | 4 | 1 | 4 | 8 | / | 1 | 2048 | 256 |  | 29.0 | 167 |  |  |  | main(1e057005) |
| Next-80B-A3B | DeepEP | H100 | BF16 | 128 | 4096 | 1 | 1 | 32 | 4 | 1 | 1 | 3 | 32 | / | 1 | 256 | 8 | norm, act, shared_expert | 2.2 | 89 |  |  |  | dev(bbc762d5) |

*Note: The performance of Qwen3-Next-80B-A3B is not well tuned. Acceleration is coming soon!*

#### Long Context

Experiment Setup:
- NVLink 4th Generation
- InfiniBand 8x50 GB/s
- Model: Qwen3-235B-A22
- Dispatcher: DeepEP
- Precision: bf16

| GPU | SEQ_LEN | NNODES | TP | CP | GBS | TFLOPS |
| --- | ------- | ------ | -- | -- | --- | ------ |
| H100| 16384   | 32     | 4  | 1  | 256 | 295 (forced balance) / 237 (dropless) |
| H100| 32768   | 32     | 4  | 2  | 256 | 335 (forced balance) / 277 (dropless) |
| H100| 65536   | 32     | 4  | 4  | 256 | 368 (forced balance) / 320 (dropless) |
| H100| 131072  | 32     | 4  | 8  | 256 | 389 (forced balance) / 357 (dropless) |

Command:

```bash
NNODES=${NNODES} OPTIMIZER_OFFLOAD=0 A2A_OVERLAP=0 MODEL=Qwen3-235B-A22B PP=8 VPP=4 TP=${TP} EP=8 CP=${CP} GBS=${GBS} SEQ_LEN=${SEQ_LEN} PR=bf16 bash sbatch_benchmarking.sh --recompute-granularity selective --recompute-modules layernorm moe
```

Notes:

* Selective recompute requires PR [!2125](https://github.com/NVIDIA/TransformerEngine/pull/2125) of TransformerEngine to fix a memory issue of sequence parallel. If you run OOM without this PR, please use `--recompute-granularity full --recompute-method uniform --recompute-num-layers 1` to enable full recompute.
* Add `--moe-router-force-load-balancing` to test forced load balancing.

---

### GB200

Experiment Setup:
- Model: Qwen3-235B-A22B
- Dispatcher: HybridEP
- NVLink Domain Size: 72

| GPU | SEQ_LEN | NNODES | Precision | TP | PP | VPP | CP | EP | MBS | GBS | TFLOPS | Notes |
| --- | ------- | ------ | --------- | -- | -- | --- | -- | -- | --- | --- | ------ | ----- |
| GB200 | 4096   | 64 | BF16  | 1 | 8 | 1  | 1 | 8  | 1 | 8192 | 750  | |
| GB200 | 4096   | 64 | MXFP8 | 1 | 4 | 6  | 1 | 64 | 3 | 3072 | 919  | activation offloading required |
| GB200 | 131072 | 32 | MXFP8 | 4 | 4 | 12 | 4 | 32 | 1 | 1024 | 1150 | long context |

Command (BF16):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_GRAPH_REGISTER=0 DISPATCHER=hybridep SEGMENT=2 PR=bf16 A2A_OVERLAP=0 TP=1 PP=8 VPP=1 EP=8 NNODES=64 MBS=1 GBS=8192 bash ./sbatch_benchmarking.sh --moe-router-force-load-balancing --cuda-graph-impl transformer_engine --cuda-graph-scope attn moe_router moe_preprocess
```

Command (MXFP8):

```bash
PR=mxfp8 MBS=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_GRAPH_REGISTER=0 DISPATCHER=hybridep A2A_OVERLAP=1 TP=1 PP=4 VPP=6 EP=64 SEGMENT=16 NNODES=64 GBS=3072 bash ./sbatch_benchmarking.sh --recompute-granularity selective --recompute-modules moe_act layernorm --moe-router-force-load-balancing --cuda-graph-impl transformer_engine --cuda-graph-scope attn moe_router moe_preprocess --fine-grained-activation-offloading --offload-modules expert_fc1 moe_act --delay-offload-until-cuda-graph --use-separate-send-recv-groups
```

Command (MXFP8, 128k long context):

```bash
PR=mxfp8 SEQ_LEN=131072 CP=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_GRAPH_REGISTER=0 DISPATCHER=hybridep SEGMENT=8 A2A_OVERLAP=0 TP=4 PP=4 VPP=12 EP=32 NNODES=32 MBS=1 GBS=1024 bash ./sbatch_benchmarking.sh --recompute-granularity selective --recompute-modules moe_act layernorm --moe-router-force-load-balancing
```

Notes:
* MXFP8 config requires the activation offloading feature. Please check the PR status before using it.

---

### GB300

Experiment Setup:
- Model: Qwen3-235B-A22B
- Dispatcher: HybridEP
- NVLink Domain Size: 72

| GPU | SEQ_LEN | NNODES | Precision | TP | PP | VPP | CP | EP | MBS | GBS | TFLOPS | Notes |
| --- | ------- | ------ | --------- | -- | -- | --- | -- | -- | --- | --- | ------ | ----- |
| GB300 | 4096 | 16 | MXFP8 | 1 | 1 | 1 | 1 | 64 | 1 | 8192 | 1055 | |

Command:

```bash
PR=mxfp8 MBS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_GRAPH_REGISTER=0 DISPATCHER=hybridep SEGMENT=16 A2A_OVERLAP=1 TP=1 PP=1 VPP=1 EP=64 NNODES=16 GBS=8192 bash ./sbatch_benchmarking.sh --moe-router-force-load-balancing --cuda-graph-impl transformer_engine --cuda-graph-scope attn moe_router moe_preprocess
```
