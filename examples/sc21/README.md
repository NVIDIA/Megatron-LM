# Reproducing Figures in SC21 Paper


This directory contains some of the scripts that were used to produce the
results in the [Megatron paper](https://arxiv.org/pdf/2104.04473.pdf) that is
to appear at [SuperComputing 2021](https://sc21.supercomputing.org/). These
scripts use [Slurm](https://slurm.schedmd.com/documentation.html) with the
[pyxis plugin](https://github.com/NVIDIA/pyxis), but can be modified for other
schedulers as well.


## Setup

All the cluster-dependent variables are in [`CONFIG.sh`](./CONFIG.sh). Please
update the unspecified values (in angle brackets `<...>`) before launching any
scripts.



## Scripts

Below is a list of scripts that can be used to reproduce various figures in our
[paper](https://arxiv.org/pdf/2104.04473.pdf):

* [run_table_1.sh](./run_table_1.sh): Table 1 showing weak-scaling throughput
for GPT models ranging from 1 billion to 1 trillion parameters.
* [run_figure_11.sh](./run_figure_11.sh): Figure 11 showing the weak-scaling
performance of pipeline parallelism.
* [run_figure_12.sh](./run_figure_12.sh): Figure 12 showing the effect of
the interleaved schedule on a 175B GPT model.
* [run_figure_13.sh](./run_figure_13.sh): Figure 13 showing the effect of
different degrees of pipeline and tensor model parallelism on a model with
162.2 billion parameters.
* [run_figure_14.sh](./run_figure_14.sh): Figure 14 showing the effect of
different degrees of data and pipeline model parallelism on a model with
5.9 billion parameters.
* [run_figure_15.sh](./run_figure_15.sh): Figure 15 showing the effect of
different degrees of data and tensor model parallelism on a model with
5.9 billion parameters.
* [run_figure_16.sh](./run_figure_16.sh): Figure 16 showing the effect of
microbatch size.
* [run_figure_17.sh](./run_figure_17.sh): Figure 17 showing the effect of
activation recomputation.
* [run_figure_18.sh](./run_figure_18.sh): Figure 18 showing the effect of
the scatter-gather communication optimization.
