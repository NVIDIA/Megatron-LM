#!/bin/bash


ulimit -n 65000
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
echo "CUDA available: $CUDA_AVAILABLE"

if [[ "$CUDA_AVAILABLE" == "True" ]]
then
	export NCCL_SOCKET_IFNAME=lo
	export NCCL_DEBUG=WARN
	export CUDA_DEVICE_MAX_CONNECTIONS=1
	export NVTE_DEBUG=1 
	export NVTE_DEBUG_LEVEL=2
fi
export NLTK_DATA=/workspace/data/nltk_data

rm -rf /tmp/pytest-of-ubuntu/
pslist=$(ps -ef | grep pytest | grep -v 'grep' | awk '{print $2}')

if [ -z $plist ]
then
	py_files=$(find tests/unit_tests -maxdepth 1 -type f -name "*.py" 2>/dev/null)
	for py_file in $py_files
	do
		echo "Testing: $py_file"
		torchrun --standalone  --nproc_per_node=8 -m pytest -x -v -s $py_file
	done
	
	folders=("data" "fusions" "tensor_parallel" "pipeline_parallel" "models" \
		"inference" "distributed" "dist_checkpointing" "transformer")
	for folder in ${folders[@]}
	do
		# Find all directories under tests/unit-tests/$folder
		for dir in $(find "tests/unit_tests/$folder" -type d 2>/dev/null)
		do
			# Skip if no Python files in this directory
			py_files=$(find $dir -maxdepth 1 -type f -name "*.py" 2>/dev/null)
			
			if [ -n "$py_files" ]; then
				echo "Running tests in directory: $dir"
				torchrun --standalone  --nproc_per_node=8 -m pytest -x -v -s $dir

				for py_file in $py_files
				do
					echo "Testing: $py_file"
					torchrun --nproc_per_node=8 -m pytest -x -v -s $py_file
				done
			fi
		done
	done
else
	echo "cleanup running pytests"
fi
