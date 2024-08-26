export CUDA_DEVICE_MAX_CONNECTIONS=1
PYTHONPATH=$PYTHON_PATH:./megatron torchrun --nproc-per-node 2 --master-port 29800 examples/run_simple_mcore_train_loop.py
