#!/bin/bash
export TORCH_CUDA_ARCH_LIST=8.6+PTX
CHECKPOINT_PATH=checkpoints/gpt2_345m
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
b=8
mp=16
experts=128
nodes=1
gpus=8

use_tutel=""
#use_tutel="--use-tutel"

ds_inference=""
#ds_inference="--ds-inference"

numa_bind=""
#numa_bind="--bind-to numa"

NUM_LAYERS=(46)
#launch_cmd="LD_LIBRARY_PATH=/home/amawa/saemal/msccl/build/lib:/opt/hpcx/nccl_rdma_sharp_plugin/lib/:/mellanox/sharp/lib:/opt/msft/nccl-rdma-sharp-plugins-2.8.3/lib:/home/amawa/saemal/msccl/build/lib: 

sccl_cmd="-x LD_PRELOAD=/home/amawa/saemal/msccl/build/lib/libnccl.so -x PATH -x LD_LIBRARY_PATH=/opt/hpcx/nccl_rdma_sharp_plugin/lib/:/mellanox/sharp/lib:/opt/msft/nccl-rdma-sharp-plugins-2.8.3/lib:/home/amawa/saemal/msccl/build/lib:$LD_LIBRARY_PATH  -x NCCL_ALGO=SCCL,RING,TREE -x SCCL_XML_FILES=/home/amawa/a2a-32.xml -x NCCL_NET_SHARED_BUFFERS=0 -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python" 
nccl_cmd="-x PATH -x LD_LIBRARY_PATH=/opt/hpcx/nccl_rdma_sharp_plugin/lib/:/mellanox/sharp/lib:/opt/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH -x NCCL_ALGO=RING,TREE -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python"


#experts=("64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 128 128")


#experts="64 64 64 64 64 64 64 64 128 128"
#NUM_LAYERS=(24)
#HIDDEN=(2048)
#HEADS=(16)
#NODES=(1 2 4 8 16)
#for ns in ${!NODES[@]};
#do
#for mp in 1
#do
#for k in ${!NUM_LAYERS[@]};
#do
#
#nodes=${NODES[$ns]}
#procs=$(($nodes * $gpus))
#launch_cmd="mpirun $numa_bind -np $procs -npernode 8 -hostfile /job/hostfile  -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca  coll_hcoll_enable 0 -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=WARN -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt/msft/topo.xml -x NCCL_DEBUG_SUBSYS=ALL"
#
#L=${NUM_LAYERS[$k]}
#H=${HIDDEN[$k]}
#A=${HEADS[$k]}
##experts1=${experts[$k]}
#program_cmd="tools/generate_samples_gpt.py \
#       --tensor-model-parallel-size $mp \
#       --num-layers $L \
#       --hidden-size $H \
#       --load $CHECKPOINT_PATH \
#       --num-attention-heads $A \
#       --max-position-embeddings 1024 \
#       --tokenizer-type GPT2BPETokenizer \
#       --fp16 \
#       --mlp-type standard \
#       --num-experts ${experts} \
#       --micro-batch-size $b \
#       --seq-length 10 \
#       --out-seq-length 10 \
#       --temperature 1.0 \
#       --vocab-file $VOCAB_FILE \
#       --merge-file $MERGE_FILE \
#       --genfile unconditional_samples.json \
#       --top_p 0.9 \
#       --log-interval 1 \
#       --num-samples $((100*$b))
#       --deepspeed \
#       $use_tutel $ds_inference"
#
#echo $launch_cmd $nccl_cmd $program_cmd
#
#$launch_cmd $nccl_cmd $program_cmd &> base512_alltoall_batch_nodes_${nodes}_${b}_mp_${mp}_layer_${L}.log
##       --recompute
#done
#done
#done

#experts="64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 128 128"

#NUM_LAYERS=(20)
#HIDDEN=(2048)
#HEADS=(16)
#NODES=(16)
#for ns in ${!NODES[@]};
#do
#for mp in 1
#do
#for k in ${!NUM_LAYERS[@]};
#do
#
#nodes=${NODES[$ns]}
#procs=$(($nodes * $gpus))
#launch_cmd="mpirun $numa_bind -np $procs -npernode 8 -hostfile /job/hostfile  -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca  coll_hcoll_enable 0 -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=WARN -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt/msft/topo.xml -x NCCL_DEBUG_SUBSYS=ALL"
#
#L=${NUM_LAYERS[$k]}
#H=${HIDDEN[$k]}
#A=${HEADS[$k]}
##experts1=${experts[$k]}
#program_cmd="tools/generate_samples_gpt.py \
#       --tensor-model-parallel-size $mp \
#       --num-layers $L \
#       --hidden-size $H \
#       --load $CHECKPOINT_PATH \
#       --num-attention-heads $A \
#       --max-position-embeddings 1024 \
#       --tokenizer-type GPT2BPETokenizer \
#       --fp16 \
#       --mlp-type standard \
#       --num-experts ${experts} \
#       --micro-batch-size $b \
#       --seq-length 10 \
#       --out-seq-length 10 \
#       --temperature 1.0 \
#       --vocab-file $VOCAB_FILE \
#       --merge-file $MERGE_FILE \
#       --genfile unconditional_samples.json \
#       --top_p 0.9 \
#       --log-interval 1 \
#       --num-samples $((100*$b))
#       --deepspeed \
#       $use_tutel $ds_inference"
#
#echo $launch_cmd $nccl_cmd $program_cmd
#
#$launch_cmd $nccl_cmd $program_cmd &> base512_alltoall_batch_nodes_${nodes}_${b}_mp_${mp}_layer_${L}.log
##       --recompute
#done
#done
#done

#NUM_LAYERS=(24)
#HIDDEN=(2048)
#HEADS=(16)
#NODES=(16)
#for ns in ${!NODES[@]};
#do
#for mp in 1
#do
#for k in ${!NUM_LAYERS[@]};
#do
#
#nodes=${NODES[$ns]}
#procs=$(($nodes * $gpus))
#launch_cmd="mpirun $numa_bind -np $mp -npernode 8 -hostfile /job/hostfile  -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca  coll_hcoll_enable 0 -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=WARN -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt/msft/topo.xml -x NCCL_DEBUG_SUBSYS=ALL"
#
#L=${NUM_LAYERS[$k]}
#H=${HIDDEN[$k]}
#A=${HEADS[$k]}
##experts1=${experts[$k]}
#program_cmd="tools/generate_samples_gpt.py \
#       --tensor-model-parallel-size $mp \
#       --num-layers $L \
#       --hidden-size $H \
#       --load $CHECKPOINT_PATH \
#       --num-attention-heads $A \
#       --max-position-embeddings 1024 \
#       --tokenizer-type GPT2BPETokenizer \
#       --fp16 \
#       --mlp-type standard \
#       --num-experts ${experts} \
#       --micro-batch-size $b \
#       --seq-length 10 \
#       --out-seq-length 10 \
#       --temperature 1.0 \
#       --vocab-file $VOCAB_FILE \
#       --merge-file $MERGE_FILE \
#       --genfile unconditional_samples.json \
#       --top_p 0.9 \
#       --log-interval 1 \
#       --num-samples $((100*$b))
#       --deepspeed \
#       $use_tutel $ds_inference"
#
#echo $launch_cmd $nccl_cmd $program_cmd
#
#$launch_cmd $nccl_cmd $program_cmd &> opt_split_alltoall_batch_nodes_${nodes}_${b}_mp_${mp}_layer_${L}.log
##       --recompute
#done
#done
#done
ds_inference="--ds-inference"

numa_bind="--bind-to numa"
#experts="64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 128 128"
##experts="64 64 64 64 64 64 64 64 64 64 128 128"
#
#NUM_LAYERS=(40)
#HIDDEN=(4096)
#HEADS=(32)
#NODES=(16)
#for ns in ${!NODES[@]};
#do
#for mp in 8
#do
#for k in ${!NUM_LAYERS[@]};
#do
#
#nodes=${NODES[$ns]}
#procs=$(($nodes * $gpus))
#launch_cmd="mpirun $numa_bind -np $procs -npernode 8 -hostfile /job/hostfile  -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca  coll_hcoll_enable 0 -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=WARN -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt/msft/topo.xml -x NCCL_DEBUG_SUBSYS=ALL"
#
#L=${NUM_LAYERS[$k]}
#H=${HIDDEN[$k]}
#A=${HEADS[$k]}
##experts1=${experts[$k]}
#program_cmd="tools/generate_samples_gpt.py \
#       --tensor-model-parallel-size $mp \
#       --num-layers $L \
#       --hidden-size $H \
#       --load $CHECKPOINT_PATH \
#       --num-attention-heads $A \
#       --max-position-embeddings 1024 \
#       --tokenizer-type GPT2BPETokenizer \
#       --fp16 \
#       --mlp-type residual \
#       --num-experts ${experts} \
#       --micro-batch-size $b \
#       --seq-length 10 \
#       --out-seq-length 10 \
#       --temperature 1.0 \
#       --vocab-file $VOCAB_FILE \
#       --merge-file $MERGE_FILE \
#       --genfile unconditional_samples.json \
#       --top_p 0.9 \
#       --log-interval 1 \
#       --num-samples $((100*$b))
#       --deepspeed \
#       $use_tutel $ds_inference"
#
#echo $launch_cmd $nccl_cmd $program_cmd
#
#$launch_cmd $nccl_cmd $program_cmd  &> finale_${nodes}_${b}_mp_${mp}_layer_${L}.log
##       --recompute
#done
#done
#done


#experts="64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 128 128"
#experts="64 64 64 128 128"
#experts="64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 128 128"
#experts=("64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 128 128")
experts=128
NUM_LAYERS=(29 44 58 64 78)
HIDDEN=(8192 8192 8192 8192)
HEADS=(64 64 64 64)
NODES=(32)
for ns in ${!NODES[@]};
do
for mp in 8
do
for k in ${!NUM_LAYERS[@]};
do

nodes=${NODES[$ns]}
procs=$(($nodes * $gpus))
launch_cmd="mpirun $numa_bind -np $procs -npernode 8 -hostfile /job/hostfile  -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca  coll_hcoll_enable 0 -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=WARN -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt/msft/topo.xml -x NCCL_DEBUG_SUBSYS=ALL"

L=${NUM_LAYERS[$k]}
H=${HIDDEN[$k]}
A=${HEADS[$k]}
#experts1=${experts[$k]}
program_cmd="tools/generate_samples_gpt.py \
       --tensor-model-parallel-size $mp \
       --num-layers $L \
       --hidden-size $H \
       --load $CHECKPOINT_PATH \
       --num-attention-heads $A \
       --max-position-embeddings 1024 \
       --tokenizer-type GPT2BPETokenizer \
       --fp16 \
       --num-experts ${experts} \
       --mlp-type standard \
       --micro-batch-size $b \
       --seq-length 10 \
       --out-seq-length 10 \
       --temperature 1.0 \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --genfile unconditional_samples.json \
       --top_p 0.9 \
       --log-interval 1 \
       --num-samples $((100*$b))
       --deepspeed \
       $use_tutel $ds_inference"

echo $launch_cmd $nccl_cmd $program_cmd

$launch_cmd $nccl_cmd $program_cmd &> testing_noar_${nodes}_${L}.log
#       --recompute
done
done
done
#NUM_LAYERS=(30)
#HIDDEN=(8192)
#HEADS=(64)
#NODES=(16)
#for ns in ${!NODES[@]};
#do
#for mp in 32
#do
#for k in ${!NUM_LAYERS[@]};
#do
#
#nodes=${NODES[$ns]}
#procs=$(($nodes * $gpus))
#launch_cmd="mpirun $numa_bind -np $procs -npernode 8 -hostfile /job/hostfile  -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca  coll_hcoll_enable 0 -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=WARN -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt/msft/topo.xml -x NCCL_DEBUG_SUBSYS=ALL"
#
#L=${NUM_LAYERS[$k]}
#H=${HIDDEN[$k]}
#A=${HEADS[$k]}
##experts1=${experts[$k]}
#program_cmd="tools/generate_samples_gpt.py \
#       --tensor-model-parallel-size $mp \
#       --num-layers $L \
#       --hidden-size $H \
#       --load $CHECKPOINT_PATH \
#       --num-attention-heads $A \
#       --max-position-embeddings 1024 \
#       --tokenizer-type GPT2BPETokenizer \
#       --fp16 \
#       --mlp-type standard \
#       --num-experts ${experts} \
#       --micro-batch-size $b \
#       --seq-length 10 \
#       --out-seq-length 10 \
#       --temperature 1.0 \
#       --vocab-file $VOCAB_FILE \
#       --merge-file $MERGE_FILE \
#       --genfile unconditional_samples.json \
#       --top_p 0.9 \
#       --log-interval 1 \
#       --num-samples $((100*$b))
#       --deepspeed \
#       $use_tutel $ds_inference"
#
#echo $launch_cmd $nccl_cmd $program_cmd
#
#$launch_cmd $nccl_cmd $program_cmd &> base512_alltoall_batch_nodes_${nodes}_${b}_mp_${mp}_layer_${L}.log
##       --recompute
#done
#done
#done
#
#
#
#NUM_LAYERS=(44)
#HIDDEN=(8192)
#HEADS=(64)
#NODES=(16)
#for ns in ${!NODES[@]};
#do
#for mp in 64
#do
#for k in ${!NUM_LAYERS[@]};
#do
#
#nodes=${NODES[$ns]}
#procs=$(($nodes * $gpus))
#launch_cmd="mpirun $numa_bind -np $procs -npernode 8 -hostfile /job/hostfile  -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca  coll_hcoll_enable 0 -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=WARN -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt/msft/topo.xml -x NCCL_DEBUG_SUBSYS=ALL"
#
#L=${NUM_LAYERS[$k]}
#H=${HIDDEN[$k]}
#A=${HEADS[$k]}
##experts1=${experts[$k]}
#program_cmd="tools/generate_samples_gpt.py \
#       --tensor-model-parallel-size $mp \
#       --num-layers $L \
#       --hidden-size $H \
#       --load $CHECKPOINT_PATH \
#       --num-attention-heads $A \
#       --max-position-embeddings 1024 \
#       --tokenizer-type GPT2BPETokenizer \
#       --fp16 \
#       --mlp-type standard \
#       --num-experts ${experts} \
#       --micro-batch-size $b \
#       --seq-length 10 \
#       --out-seq-length 10 \
#       --temperature 1.0 \
#       --vocab-file $VOCAB_FILE \
#       --merge-file $MERGE_FILE \
#       --genfile unconditional_samples.json \
#       --top_p 0.9 \
#       --log-interval 1 \
#       --num-samples $((100*$b))
#       --deepspeed \
#       $use_tutel $ds_inference"
#
#echo $launch_cmd $nccl_cmd $program_cmd
#
#$launch_cmd $nccl_cmd $program_cmd &> base512_alltoall_batch_nodes_${nodes}_${b}_mp_${mp}_layer_${L}.log
##       --recompute
#done
#done
#done


#experts=1
#
#
#NUM_LAYERS=(50)
#HIDDEN=(8192)
#HEADS=(64)
#NODES=(1)
#for ns in ${!NODES[@]};
#do
#for mp in 16
#do
#for k in ${!NUM_LAYERS[@]};
#do
#
#nodes=${NODES[$ns]}
#procs=$(($nodes * 1))
#launch_cmd="mpirun $numa_bind -np $procs -npernode 8 -hostfile /job/hostfile  -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca  coll_hcoll_enable 0 -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=WARN -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt/msft/topo.xml -x NCCL_DEBUG_SUBSYS=ALL"
#
#L=${NUM_LAYERS[$k]}
#H=${HIDDEN[$k]}
#A=${HEADS[$k]}
##experts1=${experts[$k]}
#program_cmd="tools/generate_samples_gpt.py \
#       --tensor-model-parallel-size $mp \
#       --num-layers $L \
#       --hidden-size $H \
#       --load $CHECKPOINT_PATH \
#       --num-attention-heads $A \
#       --max-position-embeddings 1024 \
#       --tokenizer-type GPT2BPETokenizer \
#       --fp16 \
#       --mlp-type standard \
#       --num-experts ${experts} \
#       --micro-batch-size $b \
#       --seq-length 10 \
#       --out-seq-length 10 \
#       --temperature 1.0 \
#       --vocab-file $VOCAB_FILE \
#       --merge-file $MERGE_FILE \
#       --genfile unconditional_samples.json \
#       --top_p 0.9 \
#       --log-interval 1 \
#       --num-samples $((100*$b))
#       --deepspeed \
#       $use_tutel $ds_inference"
#
#echo $launch_cmd $nccl_cmd $program_cmd
#
#$launch_cmd $nccl_cmd $program_cmd &> dense_batch_nodes_${nodes}_${b}_mp_${mp}_layer_${L}.log
##       --recompute
#done
#done
#done
#
#
#NUM_LAYERS=(70 96)
#HIDDEN=(12288 12288 12288)
#HEADS=(96 96 96)
#NODES=(1)
#for ns in ${!NODES[@]};
#do
#for mp in 64
#do
#for k in ${!NUM_LAYERS[@]};
#do
#
#nodes=${NODES[$ns]}
#procs=$(($nodes * 1))
#launch_cmd="mpirun $numa_bind -np $mp -npernode 8 -hostfile /job/hostfile  -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca  coll_hcoll_enable 0 -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=WARN -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt/msft/topo.xml -x NCCL_DEBUG_SUBSYS=ALL"
#
#L=${NUM_LAYERS[$k]}
#H=${HIDDEN[$k]}
#A=${HEADS[$k]}
##experts1=${experts[$k]}
#program_cmd="tools/generate_samples_gpt.py \
#       --tensor-model-parallel-size $mp \
#       --num-layers $L \
#       --hidden-size $H \
#       --load $CHECKPOINT_PATH \
#       --num-attention-heads $A \
#       --max-position-embeddings 1024 \
#       --tokenizer-type GPT2BPETokenizer \
#       --fp16 \
#       --mlp-type standard \
#       --num-experts ${experts} \
#       --micro-batch-size $b \
#       --seq-length 10 \
#       --out-seq-length 10 \
#       --temperature 1.0 \
#       --vocab-file $VOCAB_FILE \
#       --merge-file $MERGE_FILE \
#       --genfile unconditional_samples.json \
#       --top_p 0.9 \
#       --log-interval 1 \
#       --num-samples $((100*$b))
#       --deepspeed \
#       $use_tutel $ds_inference"
#
#echo $launch_cmd $nccl_cmd $program_cmd
#
#$launch_cmd $nccl_cmd $program_cmd &> dense_batch_nodes_${nodes}_${b}_mp_${mp}_layer_${L}.log
##       --recompute
#done
#done
#done
#NUM_LAYERS=(40)
#HIDDEN=(4096)
#HEADS=(32)
#NODES=(4 8 16)
#for ns in ${!NODES[@]};
#do
#for mp in 8
#do
#for k in ${!NUM_LAYERS[@]};
#do
#
#nodes=${NODES[$ns]}
#procs=$(($nodes * $gpus))
#launch_cmd="mpirun $numa_bind -np $procs -npernode 8 -hostfile /job/hostfile  -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca  coll_hcoll_enable 0 -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=WARN -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt/msft/topo.xml -x NCCL_DEBUG_SUBSYS=ALL"
#
#L=${NUM_LAYERS[$k]}
#H=${HIDDEN[$k]}
#A=${HEADS[$k]}
##experts1=${experts[$k]}
#program_cmd="tools/generate_samples_gpt.py \
#       --tensor-model-parallel-size $mp \
#       --num-layers $L \
#       --hidden-size $H \
#       --load $CHECKPOINT_PATH \
#       --num-attention-heads $A \
#       --max-position-embeddings 1024 \
#       --tokenizer-type GPT2BPETokenizer \
#       --fp16 \
#       --mlp-type standard \
#       --num-experts ${experts} \
#       --micro-batch-size $b \
#       --seq-length 10 \
#       --out-seq-length 10 \
#       --temperature 1.0 \
#       --vocab-file $VOCAB_FILE \
#       --merge-file $MERGE_FILE \
#       --genfile unconditional_samples.json \
#       --top_p 0.9 \
#       --log-interval 1 \
#       --num-samples $((100*$b))
#       --deepspeed \
#       $use_tutel $ds_inference"
#
#echo $launch_cmd $nccl_cmd $program_cmd
#
#$launch_cmd $nccl_cmd $program_cmd &> 512_alltoall_batch_nodes_${nodes}_${b}_mp_${mp}_layer_${L}.log
##       --recompute
#done
#done
#done
##
#
#NUM_LAYERS=(32 46)
#HIDDEN=(8192 8192)
#HEADS=(64 64)
#NODES=(16)
#for ns in ${!NODES[@]};
#do
#for mp in 8
#do
#for k in ${!NUM_LAYERS[@]};
#do
#
#nodes=${NODES[$ns]}
#procs=$(($nodes * $gpus))
#launch_cmd="mpirun $numa_bind -np $procs -npernode 8 -hostfile /job/hostfile  -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca  coll_hcoll_enable 0 -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=WARN -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt/msft/topo.xml -x NCCL_DEBUG_SUBSYS=ALL"
#
#L=${NUM_LAYERS[$k]}
#H=${HIDDEN[$k]}
#A=${HEADS[$k]}
##experts1=${experts[$k]}
#program_cmd="tools/generate_samples_gpt.py \
#       --tensor-model-parallel-size $mp \
#       --num-layers $L \
#       --hidden-size $H \
#       --load $CHECKPOINT_PATH \
#       --num-attention-heads $A \
#       --max-position-embeddings 1024 \
#       --tokenizer-type GPT2BPETokenizer \
#       --fp16 \
#       --mlp-type standard \
#       --num-experts ${experts} \
#       --micro-batch-size $b \
#       --seq-length 10 \
#       --out-seq-length 10 \
#       --temperature 1.0 \
#       --vocab-file $VOCAB_FILE \
#       --merge-file $MERGE_FILE \
#       --genfile unconditional_samples.json \
#       --top_p 0.9 \
#       --log-interval 1 \
#       --num-samples $((100*$b))
#       --deepspeed \
#       $use_tutel $ds_inference"
#
#echo $launch_cmd $nccl_cmd $program_cmd
#
#$launch_cmd $nccl_cmd $program_cmd &> 512_alltoall_batch_nodes_${nodes}_${b}_mp_${mp}_layer_${L}.log
##       --recompute
#done
#done
#done
#
#
#NUM_LAYERS=(58)
#HIDDEN=(8192)
#HEADS=(64)
#NODES=(16)
#for ns in ${!NODES[@]};
#do
#for mp in 16
#do
#for k in ${!NUM_LAYERS[@]};
#do
#
#nodes=${NODES[$ns]}
#procs=$(($nodes * $gpus))
#launch_cmd="mpirun $numa_bind -np $procs -npernode 8 -hostfile /job/hostfile  -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca  coll_hcoll_enable 0 -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=WARN -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt/msft/topo.xml -x NCCL_DEBUG_SUBSYS=ALL"
#
#L=${NUM_LAYERS[$k]}
#H=${HIDDEN[$k]}
#A=${HEADS[$k]}
##experts1=${experts[$k]}
#program_cmd="tools/generate_samples_gpt.py \
#       --tensor-model-parallel-size $mp \
#       --num-layers $L \
#       --hidden-size $H \
#       --load $CHECKPOINT_PATH \
#       --num-attention-heads $A \
#       --max-position-embeddings 1024 \
#       --tokenizer-type GPT2BPETokenizer \
#       --fp16 \
#       --mlp-type standard \
#       --num-experts ${experts} \
#       --micro-batch-size $b \
#       --seq-length 10 \
#       --out-seq-length 10 \
#       --temperature 1.0 \
#       --vocab-file $VOCAB_FILE \
#       --merge-file $MERGE_FILE \
#       --genfile unconditional_samples.json \
#       --top_p 0.9 \
#       --log-interval 1 \
#       --num-samples $((100*$b))
#       --deepspeed \
#       $use_tutel $ds_inference"
#
#echo $launch_cmd $nccl_cmd $program_cmd
#
#$launch_cmd $nccl_cmd $program_cmd &> 512_alltoall_batch_nodes_${nodes}_${b}_mp_${mp}_layer_${L}.log
##       --recompute
#done
#done
#done

#

#NUM_LAYERS=(20 40)
##launch_cmd="deepspeed --num_gpus=$gpus --num_nodes=$nodes"
#for mp in 8
#do
#for b in 128
#do
#for k in ${!NUM_LAYERS[@]};
#do
#L=${NUM_LAYERS[$k]}
#program_cmd="tools/generate_samples_gpt.py \
#       --tensor-model-parallel-size $mp \
#       --num-layers $L \
#       --hidden-size 8192 \
#       --load $CHECKPOINT_PATH \
#       --num-attention-heads 64 \
#       --max-position-embeddings 1024 \
#       --tokenizer-type GPT2BPETokenizer \
#       --fp16 \
#       --mlp-type standard \
#       --num-experts ${experts} \
#       --micro-batch-size $b \
#       --seq-length 30 \
#       --out-seq-length 30 \
#       --temperature 1.0 \
#       --vocab-file $VOCAB_FILE \
#       --merge-file $MERGE_FILE \
#       --genfile unconditional_samples.json \
#       --top_p 0.9 \
#       --log-interval 1 \
#       --num-samples $((100*$b))
#       --deepspeed \
#       $use_tutel $ds_inference"
#
#echo $launch_cmd $nccl_cmd $program_cmd
#
#$launch_cmd $nccl_cmd $program_cmd &> batch_${b}_mp_${mp}_layer_${L}.log
#done
#done
#done
##       --recompute
#
#
##launch_cmd="deepspeed --num_gpus=$gpus --num_nodes=$nodes"
#for mp in 16
#do
#for b in 128
#do
#program_cmd="tools/generate_samples_gpt.py \
#       --tensor-model-parallel-size $mp \
#       --num-layers 52 \
#       --hidden-size 8192 \
#       --load $CHECKPOINT_PATH \
#       --num-attention-heads 64 \
#       --max-position-embeddings 1024 \
#       --tokenizer-type GPT2BPETokenizer \
#       --fp16 \
#       --mlp-type standard \
#       --num-experts ${experts} \
#       --micro-batch-size $b \
#       --seq-length 30 \
#       --out-seq-length 30 \
#       --temperature 1.0 \
#       --vocab-file $VOCAB_FILE \
#       --merge-file $MERGE_FILE \
#       --genfile unconditional_samples.json \
#       --top_p 0.9 \
#       --log-interval 1 \
#       --num-samples $((100*$b))
#       --deepspeed \
#       $use_tutel $ds_inference"
#
#echo $launch_cmd $nccl_cmd $program_cmd
#
#$launch_cmd $nccl_cmd $program_cmd &> batch_${b}_mp_${mp}_layer_${L}.log
#done
#done
#done
