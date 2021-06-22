#!/bin/bash

srun -p batch_short,batch -A gpu_adlr_nlp -t 2:00:00 --nodes=1 --ntasks-per-node=16 --gres=gpu:16,gpfs:circe --job-name=interact --container-mounts=/gpfs/fs1/projects/gpu_adlr/datasets:/gpfs/fs1/projects/gpu_adlr/datasets,/home/zihanl:/home/zihanl --container-image=gitlab-master.nvidia.com/adlr/megatron-lm/pytorch-nlp-retriever-faiss:20.12-py3-devel --exclusive --pty bash
