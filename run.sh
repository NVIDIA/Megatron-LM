export PYTHONPATH=${PWD}:${PYTHONPATH}
torchrun --nproc_per_node=8 tests/unit_tests/models/heterogenous_parallel/train.py
