#!/bin/bash

PORT=50088
SSH_PORT=22
NAME="Megatron-LM"
PASSWORD="1234"

sudo docker run --gpus all -td --rm \
    -p $PORT:$SSH_PORT \
    -v /home/kwangryeol/python/Megatron-LM:/workspace/megatron \
    -v /home/kwangryeol/Data1/datasets:/workspace/dataset \
    -v /home/kwangryeol/Data2/ckpt/Megatron-LM:/workspace/checkpoints \
    --name $NAME \
    nvcr.io/nvidia/pytorch:23.08-py3 

sudo docker exec $NAME /bin/bash -c "echo root:$PASSWORD | chpasswd"
sudo docker exec $NAME apt update
sudo docker exec $NAME apt install -yqq net-tools vim openssh-server screen
sudo docker exec $NAME /bin/bash -c "echo PermitRootLogin yes >> /etc/ssh/sshd_config"
sudo docker exec $NAME service ssh start
sudo docker exec $NAME pip3 install nltk


echo "Setting done"