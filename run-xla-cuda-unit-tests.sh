#!/bin/bash

ulimit -n 65000
export PJRT_DEVICE=CUDA
export NLTK_DATA=/data/nltk_data
rm -rf /tmp/pytest*

pslist=$(ps -ef | grep pytest | grep -v 'grep' | awk '{print $2}')


if [ -z $plist ]
then
	torchrun --nproc_per_node=8 -m pytest -x -v -s tests/unit_tests/*.py
	torchrun --nproc_per_node=8 -m pytest -x -v -s tests/unit_tests/data 
	torchrun --nproc_per_node=8 -m pytest -x -v -s tests/unit_tests/fusions 
	torchrun --nproc_per_node=8 -m pytest -x -v -s tests/unit_tests/tensor_parallel 
	torchrun --nproc_per_node=8 -m pytest -x -v -s tests/unit_tests/pipeline_parallel 
	torchrun --standalone --nproc_per_node=8 -m pytest -x -v -s tests/unit_tests/models 
	torchrun --standalone --nproc_per_node=8 -m pytest -x -v -s tests/unit_tests/inference 
	torchrun  --nproc_per_node=8 -m pytest -x -v -s tests/unit_tests/distributed 
	torchrun --nproc_per_node=8 -m pytest -x -v -s tests/unit_tests/dist_checkpointing 
	torchrun --nproc_per_node=8 -m pytest -x -v -s tests/unit_tests/transformer
else
	echo "cleanup running pytests"
fi
