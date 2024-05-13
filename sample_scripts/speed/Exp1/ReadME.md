# Experiment 1: E2E Throughput Speedup Test with Different Model Size

## Experiments

| Model | Description | Scripts |
| ----------- | ----------- | ----------- |
| 1.3B | Baseline without quantization | 1_3B_Baseline |
| 1.3B | Grad and WeightDiff Quantization | 1_3B_QWG |
| 2.7B | Baseline without quantization | 2_7B_Baseline |
| 2.7B | Grad and WeightDiff Quantization | 2_7B_QWG |
| 6.7B | Baseline without quantization | 6_7B_Baseline |
| 6.7B | Grad and WeightDiff Quantization | 6_7B_QWG |
| 18B | Baseline without quantization | 18B_Baseline |
| 18B | Grad and WeightDiff Quantization | 18B_QWG |

**Run on Bytedance cluster with Infiniband(400Gbps)** \
*Env: up to 8 nodes each equipped with 8xA100 GPUs* \
*Note: For Speed Test, Sequence Length=1024, Accumulation Step=1, using Mock Data*

Each Script will run on [4, 6, 8, 12, 16] 8xA100 nodes respectivelly.
