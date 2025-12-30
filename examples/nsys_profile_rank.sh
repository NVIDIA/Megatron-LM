#!/bin/bash

set -e

# nsys profile -t cuda,nvtx,osrt -s none --cpuctxsw none --python-sampling true --python-sampling-frequency 1000 $@ || true

# nsys profile -t cuda,nvtx,osrt --force-overwrite true \
#     --capture-range=cudaProfilerApi --capture-range-end=stop --gpu-metrics-device=0 \
#     --python-sampling-frequency 1000 --python-sampling true \
#     $@ || true

nsys profile -w true -t cublas,cuda,nvtx,osrt -s cpu -c cudaProfilerApi -o "$NSYS_DIR/datetime_${DATETIME}_gpt_sft_hetero_cp_iter2_4_flash_global_8192_rank${OMPI_COMM_WORLD_RANK}" $@ || true

# PROFILE_RANKS=(0 1 2 3 4 5 6 7 8)

# if [[ " ${PROFILE_RANKS[*]} " =~ " $OMPI_COMM_WORLD_RANK " ]]; then
#     nsys profile -w true -t cublas,cuda,nvtx,osrt -s cpu -c cudaProfilerApi -o "datetime_${DATETIME}_gpt_sft_hetero_cp_iter2_4_flash_global_8192_rank${OMPI_COMM_WORLD_RANK}" $@ || true
# else
#     $@ || true
# fi
