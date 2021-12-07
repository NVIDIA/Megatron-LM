#!/bin/bash

CHECKPOINT_PATH=checkpoints/gpt2_345m
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
b=8
mp=1
experts=128
nodes=4
gpus=8
procs=$(($nodes * $gpus))

use_tutel=""
#use_tutel="--use-tutel"

ds_inference=""
ds_inference="--ds-inference"

#launch_cmd="mpirun --tag-output --allow-run-as-root -np $procs --map-by ppr:8:node -hostfile /job/hostfile  -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca  coll_hcoll_enable 0  -x PATH -x LD_LIBRARY_PATH=/opt/hpcx/nccl_rdma_sharp_plugin/lib/:/mellanox/sharp/lib:/opt/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=WARN -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt/msft/topo.xml -x NCCL_DEBUG_SUBSYS=INIT,ENV  -x SCCL_XML_FILES=/home/amawa/a2a-32.xml -x SNCCL_NET_SHARED_BUFFERS=0 -x NCCL_ALGO=RING,TREE  -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python"

launch_cmd="deepspeed --num_gpus=$gpus --num_nodes=$nodes"

program_cmd="tools/generate_samples_gpt.py \
       --tensor-model-parallel-size $mp \
       --num-layers 24 \
       --hidden-size 2048 \
       --load $CHECKPOINT_PATH \
       --num-attention-heads 16 \
       --max-position-embeddings 1024 \
       --tokenizer-type GPT2BPETokenizer \
       --fp16 \
       --num-experts $experts \
       --micro-batch-size $b \
       --seq-length 101 \
       --out-seq-length 101 \
       --temperature 1.0 \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --genfile unconditional_samples.json \
       --top_p 0.9 \
       --log-interval 1 \
       --num-samples $((20*$b)) 
       --deepspeed \
       $use_tutel $ds_inference"

echo $launch_cmd $program_cmd
$launch_cmd $program_cmd
#       --recompute
