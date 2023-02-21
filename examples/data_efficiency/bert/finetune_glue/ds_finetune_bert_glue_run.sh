hostname_and_rank=$1
master_port=$2
pretrained_checkpoint=$3

# hostname_and_rank="worker-0:0,1,2,3"
# master_port=12345
# pretrained_checkpoint="/blob/users/conglli/project/bert_with_pile/checkpoint/bert-pile-0.336B-iters-2M-lr-1e-4-min-1e-5-wmup-10000-dcy-2M-sty-linear-gbs-1024-mbs-16-gpu-64-zero-0-mp-1-pp-1-nopp"

tasks=(
    RTE
    MRPC
    STS-B
    CoLA
    SST-2
    QNLI
    QQP
    MNLI
)

seeds=(
    1234
    1235
    1236
    1237
    1238
)

lrs=(
    2e-5
    3e-5
    4e-5
    5e-5
)

for ((i=0;i<${#tasks[@]};++i)); do
    task=${tasks[i]}
    for ((j=0;j<${#seeds[@]};++j)); do
        seed=${seeds[j]}
        for ((k=0;k<${#lrs[@]};++k)); do
            lr=${lrs[k]}
            bash ds_finetune_bert_glue.sh ${hostname_and_rank} ${master_port} ${seed} ${task} ${lr} ${pretrained_checkpoint}
        done
    done
done