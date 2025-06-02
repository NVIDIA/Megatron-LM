#!/bin/bash


ulimit -n 65000
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=WARN
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ACCEL_MEMORY_GB=24
export NLTK_DATA=/home/ubuntu/efs/data/nltk_data

rm -rf /tmp/pytest-of-ubuntu/
pslist=$(ps -ef | grep pytest | grep -v 'grep' | awk '{print $2}')

if [ -z $plist ]
then
	torchrun --nproc_per_node=8 -m pytest -x -v -s tests/unit_tests/*.py 
	torchrun --nproc_per_node=8 -m pytest -x -v -s tests/unit_tests/data 
	torchrun --nproc_per_node=8 -m pytest -x -v -s tests/unit_tests/fusions
	torchrun --nproc_per_node=8 -m pytest -x -v -s tests/unit_tests/tensor_parallel 
	torchrun --nproc_per_node=8 -m pytest -x -v -s tests/unit_tests/pipeline_parallel 
	torchrun --nproc_per_node=8 -m pytest -x -v -s tests/unit_tests/models 
	torchrun --nproc_per_node=8 -m pytest -x -v -s tests/unit_tests/inference 
	torchrun --nproc_per_node=8 -m pytest -x -v -s tests/unit_tests/distributed 
	torchrun --nproc_per_node=8 -m pytest -x -v -s tests/unit_tests/dist_checkpointing 
	torchrun --nproc_per_node=8 -m pytest -x -v -s tests/unit_tests/transformer
else
	echo "cleanup running pytests"
fi
