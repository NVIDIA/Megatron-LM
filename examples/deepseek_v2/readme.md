# Deepseek-v2 Megatron 

### How to run DeepseekV2 using Megatron LM

#### Download datasets and pretrained checkpoints
<pre>
cd /data
mkdir deepseek-ckpts
cd deepseek-ckpts
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-ckpts/DeepSeek-V2-Lite.tgz
tar -zxf DeepSeek-V2-Lite.tgz

mkdir deepseek-datasets
cd deepseek-datasets
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/SlimPajama.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/alpaca_zh-train.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/alpaca_zh-valid.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/mmap_deepseekv2_datasets_text_document.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/mmap_deepseekv2_datasets_text_document.idx
</pre>

Create docker image and run docker container interactively 

<pre>
docker build -t "YOUR_IMAGE_TAG" .
docker run -it \
  --network=host\
  --device /dev/dri \
  --device=/dev/kfd \
  --ipc=host --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size=64G \
  -v /data/deepseek-ckpts/deepseek-datasets:/workspace/data/deepseekv2-train-datasets \
  -v /data/deepseek-ckpts/DeepSeek-V2-Lite:/workspace/data/deepseek-ckpts/DeepSeek-V2-Lite \
  "YOUR_IMAGE_TAG"

</pre>

Inside the container, you can run deepseek_v2 pre-training with the following command.
<pre>
bash examples/deepseek_v2/train_deepseekv2.sh
</pre>
