# pipeline_parallel package

This package contains implementations for two different pipeline parallelism
schedules (one without interleaving and one with interleaving, see [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
for details), and a default no-pipelining schedule. It also contains methods
for the point-to-point communication that is needed between pipeline stages.

