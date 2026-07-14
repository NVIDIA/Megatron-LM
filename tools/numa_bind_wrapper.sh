#!/bin/bash
# Maps each LOCAL_RANK to its GPU's NUMA node based on `nvidia-smi topo -m`.
# For GB200 NVL72 topology: GPU0,1 -> NUMA 0; GPU2,3 -> NUMA 1.
case $LOCAL_RANK in
    0|1) NODE=0 ;;
    2|3) NODE=1 ;;
    *)   echo "no NUMA mapping for LOCAL_RANK=$LOCAL_RANK" >&2; NODE= ;;
esac
if [ -n "$NODE" ]; then
    exec numactl --cpunodebind=$NODE --membind=$NODE python "$@"
else
    exec python "$@"
fi