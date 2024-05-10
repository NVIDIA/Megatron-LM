# Experiment 3: E2E Throughput Speedup Scalibility

| Model | Description | Scripts |
| ----------- | ----------- | ----------- |
| 6.7B | Baseline without quantization | 6_7B_Baseline |
| 6.7B | Grad and WeightDiff Quantization | 6_7B_QWG |
| 13B | Baseline without quantization | 13B_Baseline |
| 13B | Grad and WeightDiff Quantization | 13B_QWG |
| 18B | Baseline without quantization | 18B_Baseline |
| 18B | Grad and WeightDiff Quantization | 18B_QWG |

**Run on Bytedance cluster with Infiniband(400Gbps)** \
*Env: up to 16 nodes each equipped with 8xA100 GPUs* \
*Note: For Speed Test, Sequence Length=1024, Accumulation Step=1, using Mock Data*

Each Script will run on [4, 6, 8, 12, 16] 8xA100 nodes respectivelly.