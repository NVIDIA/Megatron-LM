#!/bin/bash

CHECKPOINT_PATH=checkpoints/gpt2_345m
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
mp=1
experts=128
nodes=$1
gpus=8
procs=$(($nodes * $gpus))

##### --- setup the program args to make things runnable for all combos ---- ######

args=()
fnames=()

for bs in 8 128
do 
   # 1b
   layers=20
   dim=2048
   args+=("--num-layers $layers --hidden-size $dim --micro-batch-size $bs --num-samples $((20*$bs))")
   fnames+=("gpus-$procs-layers-$layers-dim-$dim-bs-$bs")

   # 2b, 4b
   for layers in 18 36
   do
      dim=3072
      args+=("--num-layers $layers --hidden-size $dim --micro-batch-size $bs --num-samples $((20*$bs))")
      fnames+=("gpus-$procs-layers-$layers-dim-$dim-bs-$bs")
   done
      
   #8b
   layers=40
   dim=4096
   args+=("--num-layers $layers --hidden-size $dim --micro-batch-size $bs --num-samples $((20*$bs))")
   fnames+=("gpus-$procs-layers-$layers-dim-$dim-bs-$bs")
done

######## -------------------------------------------------------------- #######

use_tutel=""
#use_tutel="--use-tutel"

ds_inference=""
ds_inference=" --ds-inference"

#numa_bind=""
numa_bind="--bind-to numa"

launch_cmd="mpirun $numa_bind --tag-output --allow-run-as-root -np $procs -npernode 8 -hostfile /job/hostfile  -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca  coll_hcoll_enable 0 -x LD_PRELOAD=/home/amawa/saemal/msccl/build/lib/libnccl.so -x PATH -x LD_LIBRARY_PATH=/opt/hpcx/nccl_rdma_sharp_plugin/lib/:/mellanox/sharp/lib:/opt/msft/nccl-rdma-sharp-plugins-2.8.3/lib:/home/amawa/saemal/msccl/build/lib:$LD_LIBRARY_PATH -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=WARN -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt/msft/topo.xml -x NCCL_DEBUG_SUBSYS=ALL"

sccl_cmd="-x NCCL_ALGO=SCCL,RING,TREE -x SCCL_XML_FILES=/home/amawa/a2a-32.xml -x NCCL_NET_SHARED_BUFFERS=0 -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python" 
nccl_cmd="-x NCCL_ALGO=RING,TREE -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python"

#launch_cmd="deepspeed --num_gpus=$gpus --num_nodes=$nodes"

program_cmd="tools/generate_samples_gpt.py \
       --tensor-model-parallel-size $mp \
       --load $CHECKPOINT_PATH \
       --num-attention-heads 32 \
       --max-position-embeddings 1024 \
       --tokenizer-type GPT2BPETokenizer \
       --fp16 \
       --num-experts $experts \
       --seq-length 30 \
       --out-seq-length 30 \
       --temperature 1.0 \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --genfile unconditional_samples.json \
       --top_p 0.9 \
       --log-interval 1 --deepspeed"

program_cmd+=$use_tutel 
program_cmd+=$ds_inference

i=0
for fname in "${fnames[@]}"
do
echo $i
echo ${args[$i]}
cmd="$launch_cmd $nccl_cmd $program_cmd ${args[$i]}"
redir="$PWD/output_dir/$fname"
echo $cmd "&>" $redir
$cmd &> $redir
echo "------------------------------------------"
((i=i+1))
done


#$launch_cmd $nccl_cmd $program_cmd

