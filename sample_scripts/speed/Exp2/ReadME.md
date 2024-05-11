# Experiment 2: E2E Throughput Speedup Test with Quantization Breakdown

| Model | Description | Scripts |
| ----------- | ----------- | ----------- |
| 2.7B | Baseline without quantization | 2_7B_Baseline |
| 2.7B | Only Quantize Gradient | 2_7B_QGrad |
| 2.7B | Only Quantize Weight Diff | 2_7B_QWeightdiff |
| 2.7B | Quantize Grad and WeightDiff | 2_7B_QWG |


**Run on Bytedance cluster with Infiniband(400Gbps)** \
*Env: up to 8 nodes each equipped with 8xA100 GPUs* \
*Note: For Speed Test, Sequence Length=1024, Accumulation Step=1, using Mock Data*

Each Script will run on [4, 8] 8xA100 nodes respectivelly.