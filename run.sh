export PYTHONPATH=${PWD}:${PYTHONPATH}
mkdir -p ../logs
torchrun --nproc_per_node=6 tests/unit_tests/models/heterogenous_parallel/train.py 2>&1 | tee ../logs/train.log
